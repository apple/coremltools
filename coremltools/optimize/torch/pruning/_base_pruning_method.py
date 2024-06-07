#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
import types as _types
from typing import Any as _Any
from typing import Dict as _Dict
from typing import NamedTuple as _NamedTuple
from typing import Optional as _Optional
from typing import cast as _cast

import numpy as _np
import torch as _torch
import torch.nn.utils.prune as _prune
import torch.utils.hooks as _hooks

from coremltools.optimize.torch._typing import ParamsDict as _ParamsDict
from coremltools.optimize.torch._utils.state_dict_utils import (
    LoadStateDictPostHook as _LoadStateDictPostHook,
)
from coremltools.optimize.torch.pruning._utils import block2_sparsity as _block2_sparsity
from coremltools.optimize.torch.pruning._utils import structured_sparsity as _structured_sparsity
from coremltools.optimize.torch.pruning._utils import (
    unstructured_sparsity as _unstructured_sparsity,
)

_logger = _logging.getLogger(__name__)


class BaseDynamicPruningMethod(_prune.BasePruningMethod):
    """
    Extension of PyTorch's native pruning infra for seamless
    model export and progressive sparsity schedules

    This class works by registering itself as a forward pre-hook
    into each prune-able `nn.Module` to apply the pruning mask
    """

    _tensor_name: str
    scheduled: bool

    def update_mask(self, module: _torch.nn.Module, scheduled_value: float) -> None:
        raise NotImplementedError()

    def bind_module(self, module: _torch.nn.Module) -> None:
        module.pruning_method = self  # type: ignore

        orig_get_state = getattr(module, "__getstate__", None)

        # Override state method of module instance to exclude the non-leaf tensor
        # which is neither a parameter nor a buffer
        # See: https://discuss.pytorch.org/t/using-nn-utils-prune-causes-torch-tensor-deepcopy-to-fail/107470
        def __getstate__(self: _torch.nn.Module) -> _Dict[str, _Any]:
            if orig_get_state is not None:
                state: _Dict[str, _Any] = orig_get_state()
            else:
                state = dict(self.__dict__)

            if hasattr(self, "pruning_method"):
                pruner = _cast(BaseDynamicPruningMethod, self.pruning_method)
                if pruner._tensor_name in state:
                    state[pruner._tensor_name] = None
            return state

        module.__getstate__ = _types.MethodType(__getstate__, module)  # type: ignore[assignment]

    @classmethod
    def from_module_and_params(
        cls, module: _torch.nn.Module, param_name: str = "weight", **params: _ParamsDict
    ) -> "BaseDynamicPruningMethod":
        """
        Factory method of this class that is tied to a particular nn.Module
        """
        pruning_method: BaseDynamicPruningMethod
        pruning_method = super(BaseDynamicPruningMethod, cls).apply(
            module, name=param_name, **params
        )
        pruning_method.bind_module(module)
        return pruning_method

    def _remove_impl(self, module: _torch.nn.Module, fuse_pruning_mask: bool) -> None:
        assert self._tensor_name is not None

        # Restore the (pruned) tensor under its original name
        orig = module._parameters[self._tensor_name + "_orig"]
        assert orig is not None

        if fuse_pruning_mask:
            pruned_orig = None
            if self.scheduled:
                current_mask = module._buffers[self._tensor_name + "_mask"]
                assert current_mask is not None
                current_amount = self.infer_sparsity_amount_from_external_mask(
                    current_mask
                )  # may have been loaded from ckpt and current_amount != self.amount:  # self.amount may be
                # out-of-sync with the ckpt

                if hasattr(self, "amount") and not _np.isclose(
                    current_amount, self.amount, rtol=1 / orig.numel()
                ):
                    _logger.warning(
                        f"Pruning method {self.__class__}'s sparsity schedule state ({self.amount}) is inconsistent "
                        f"with pruning mask's current state ({current_amount}). This is probably harmless "
                        f"if you are exporting a pruned model"
                    )
                    # We have detected an inconsistent state so we correct for this by updating the
                    # pruning method's schedule. This correction will ensure the following `self._apply_mask_impl`
                    # call to use the correct self.amount
                    self.update_mask(module, current_amount)
                    pruned_orig = current_mask.to(orig.dtype) * orig

            if pruned_orig is None:
                pruned_orig = self._apply_mask_impl(module)

            orig.data = pruned_orig.data

        setattr(module, self._tensor_name, orig)
        del module._parameters[self._tensor_name + "_orig"]
        del module._buffers[self._tensor_name + "_mask"]

    def remove(self, module: _torch.nn.Module, fuse_pruning_mask: bool = True) -> _torch.nn.Module:
        """Removes pruning masks and forward_pre_hooks from the module

        If `fuse_pruning_mask` is True, then weights are fused with the pruning
        mask before re-registering the weights under the original name
        """
        name = self._tensor_name
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, BaseDynamicPruningMethod) and hook._tensor_name == name:
                self._remove_impl(module, fuse_pruning_mask)
                del module._forward_pre_hooks[k]
                if hasattr(module, "pruning_method"):
                    delattr(module, "pruning_method")
                return module

        raise ValueError(
            f"Parameter '{name}' of module {module} has to be pruned "
            f"before pruning can be removed."
        )

    def _apply_mask_impl(self, module: _torch.nn.Module) -> _torch.Tensor:
        # Identical to prune.BasePruningMethod.apply_mask as the default method for fusing weights and masks
        # Exposed to allow overriding by complex pruning algorithms
        assert self._tensor_name is not None, "Module {} has to be pruned".format(module)
        mask = getattr(module, self._tensor_name + "_mask")
        orig = getattr(module, self._tensor_name + "_orig")
        pruned_tensor: _torch.Tensor = mask.to(dtype=orig.dtype) * orig
        return pruned_tensor

    def apply_mask(self, module: _torch.nn.Module) -> _torch.Tensor:
        return self._apply_mask_impl(module)

    def infer_sparsity_amount_from_external_mask(self, external_mask: _torch.Tensor) -> float:
        """
        Infer the sparsity amount from a given binary mask based on the granularity
        configuration of the pruning method
        """
        if hasattr(self, "granularity"):
            # rank 2: torch.Linear, rank 3: torch.Conv1d, rank 4: torch.Conv2d, rank 5: torch.Conv3d
            rank = len(external_mask.shape)

            if self.granularity == "per_scalar" or rank == 2:
                return external_mask.eq(0).float().mean().item()
            elif rank in [3, 4, 5]:
                if self.granularity == "per_kernel":
                    start_dim = 2
                elif self.granularity == "per_channel":
                    start_dim = 1
                else:
                    raise ValueError(
                        f"Can not infer sparsity amount for granularity: {self.granularity}"
                    )
                return external_mask.flatten(start_dim).eq(0).all(-1).float().mean().item()
            else:
                raise ValueError(f"weights tensor rank must be in [2, 3, 4, 5], got {rank}")

    def get_sparsity_summary(self, module: _torch.nn.Module) -> _Dict[str, _torch.tensor]:
        """
        Returns summary of the current state of pruning of module, indexed with name.
        """
        assert self._tensor_name is not None, "Module {} has not been pruned".format(module)
        weight: _torch.Tensor = getattr(module, self._tensor_name).detach()
        if hasattr(module, "weight_fake_quant") and hasattr(module.weight_fake_quant, "palettize"):
            weight = module.weight_fake_quant.palettize(weight)

        summary = {
            "structured_weight_sparsity": _structured_sparsity(weight),
            "unstructured_weight_sparsity": _unstructured_sparsity(weight),
        }

        if weight.size(0) % 2 == 0:
            summary["block2_weight_sparsity"] = _block2_sparsity(weight)
        else:
            summary["block2_weight_sparsity"] = -1  # Not applicable
        return summary


class _SyncScheduledValueLoadStateDictPostHook(_LoadStateDictPostHook):
    def __init__(self, scheduled_value_name: str):
        super().__init__()
        self._scheduled_value_name = scheduled_value_name

    def __call__(self, module: _torch.nn.Module, incompatible_keys: _NamedTuple) -> None:
        if hasattr(module, "pruning_method"):
            pruning_method: ScheduledBaseDynamicPruningMethod = module.pruning_method
            assert hasattr(pruning_method, "_tensor_name"), (
                f"state_dict cannot be loaded. Attribute _tensor_name "
                f"missing from pruning forward hook installed on the "
                f"module: {module}"
            )
            assert hasattr(pruning_method, self._scheduled_value_name), (
                f"state_dict cannot be loaded. Attribute {self._scheduled_value_name} "
                f"missing from pruning forward hook installed on the module {module}"
            )
            scheduled_value_buffer_name = (
                f"{pruning_method._tensor_name}_{self._scheduled_value_name}"
            )
            assert hasattr(module, scheduled_value_buffer_name), (
                f"state_dict cannot be loaded. Buffer {scheduled_value_buffer_name} "
                f"missing from module: {module}"
            )
            scheduled_value = getattr(module, scheduled_value_buffer_name)
            # set pruning method amount to be the same as the value from state dict
            if isinstance(scheduled_value, _torch.Tensor):
                scheduled_value = scheduled_value.data.item()
            setattr(pruning_method, self._scheduled_value_name, scheduled_value)


class ScheduledBaseDynamicPruningMethod(BaseDynamicPruningMethod):
    """
    An extension of BaseDynamicPruningMethod for scheduled pruners
    where the pruning amount is changed externally over the
    course of the training.
    """

    def __init__(self, scheduled_value: _Any, scheduled_value_name: str, **kwargs: _ParamsDict):
        super().__init__()
        self.scheduled_value_name = scheduled_value_name
        setattr(self, scheduled_value_name, scheduled_value)
        self.sync_scheduled_value_post_hook_handle: _Optional[_hooks.RemovableHandle] = None

    def bind_module(self, module: _torch.nn.Module) -> None:
        super().bind_module(module)
        param_tensor = getattr(module, self._tensor_name + "_orig")
        scheduled_value = getattr(self, self.scheduled_value_name)
        scheduled_value_tensor = _torch.tensor(scheduled_value, device=param_tensor.device)
        module.register_buffer(
            self._tensor_name + "_" + self.scheduled_value_name,
            scheduled_value_tensor,
        )
        self.sync_scheduled_value_post_hook_handle = module.register_load_state_dict_post_hook(
            _SyncScheduledValueLoadStateDictPostHook(self.scheduled_value_name)
        )

    def update_mask(self, module: _torch.nn.Module, scheduled_value: float) -> None:
        assert self._tensor_name is not None
        assert self.scheduled

        # Get the original non-pruned parameter tensor
        orig = getattr(module, self._tensor_name + "_orig")

        assert (
            orig is not None
        ), "Must have called apply() to initialize pruning before calling update_mask()"

        # Update scheduled value
        setattr(self, self.scheduled_value_name, scheduled_value)
        # keep scheduled value buffer in sync
        scheduled_value_tensor: _torch.Tensor = getattr(
            module, self._tensor_name + "_" + self.scheduled_value_name
        )
        scheduled_value_tensor.fill_(scheduled_value)

        # Update the mask with the new amount
        module.register_buffer(
            self._tensor_name + "_mask",
            self.compute_mask(orig, default_mask=None),
        )

    def _remove_impl(self, module: _torch.nn.Module, fuse_pruning_mask: bool) -> None:
        super()._remove_impl(module, fuse_pruning_mask)
        del module._buffers[self._tensor_name + "_" + self.scheduled_value_name]
        if self.sync_scheduled_value_post_hook_handle is not None:
            self.sync_scheduled_value_post_hook_handle.remove()
            self.sync_scheduled_value_post_hook_handle = None
