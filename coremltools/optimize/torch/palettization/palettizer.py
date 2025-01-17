#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
from typing import Dict as _Dict
from typing import Optional as _Optional

import torch as _torch
import torch.nn as _nn
from torch.ao.quantization import FakeQuantize as _FakeQuantize

from coremltools.optimize.torch._typing import ParamsDict as _ParamsDict
from coremltools.optimize.torch._utils.joint_compression_utils import (
    is_pruner_prepared as _is_pruner_prepared,
)
from coremltools.optimize.torch._utils.math_utils import rmse_error as _rmse_error
from coremltools.optimize.torch._utils.metadata_utils import (
    register_metadata_version as _register_metadata_version,
)
from coremltools.optimize.torch._utils.torch_utils import get_eval_model as _get_eval_model
from coremltools.optimize.torch._utils.torch_utils import (
    normalize_fsdp_module_name as _normalize_fsdp_module_name,
)
from coremltools.optimize.torch._utils.validation_utils import (
    validate_param_config as _validate_param_config,
)
from coremltools.optimize.torch.base_model_optimizer import (
    BaseTrainingTimeModelOptimizer as _BaseTrainingTimeModelOptimizer,
)
from coremltools.optimize.torch.base_model_optimizer import _Report
from coremltools.optimize.torch.palettization._custom_conversion import (
    get_conversion_custom_config_dict as _get_conversion_custom_config_dict,
)
from coremltools.optimize.torch.palettization._supported_modules import (
    DKMPalettizerModulesRegistry as _DKMPalettizerModulesRegistry,
)
from coremltools.optimize.torch.palettization.fake_palettize import FakePalettize as _FakePalettize
from coremltools.optimize.torch.palettization.palettization_config import (
    DEFAULT_PALETTIZATION_ADVANCED_OPTIONS as _DEFAULT_PALETTIZATION_ADVANCED_OPTIONS,
)
from coremltools.optimize.torch.palettization.palettization_config import (
    DEFAULT_PALETTIZATION_SCHEME as _DEFAULT_PALETTIZATION_SCHEME,
)
from coremltools.optimize.torch.palettization.palettization_config import (
    DKMPalettizerConfig as _DKMPalettizerConfig,
)
from coremltools.optimize.torch.palettization.palettization_config import (
    ModuleDKMPalettizerConfig as _ModuleDKMPalettizerConfig,
)

_logger = _logging.getLogger(__name__)


class Palettizer(_BaseTrainingTimeModelOptimizer):
    pass


class _PalettizerLUTObserver(_torch.quantization.MovingAveragePerChannelMinMaxObserver):
    """
    Subclassing _torch.quantization.MovingAveragePerChannelMinMaxObserver to add reset_parameters method
    """
    def reset_parameters(self) -> None:
        """
        FSDP expects reset_parameters method to initialize parameters/buffers in submodules
        Initializing buffers used by the MovingAveragePerChannelMinMaxObserver
        """
        self.min_val = _torch.empty(0)
        self.max_val = _torch.empty(0)
        self.eps = _torch.tensor([_torch.finfo(_torch.float32).eps])


class DKMPalettizer(Palettizer):
    """
    A palettization algorithm based on `"DKM: Differentiable K-Means Clustering Layer for Neural Network
    Compression" <https://arxiv.org/pdf/2108.12659.pdf>`_. It clusters the weights
    using a differentiable version of ``k-means``, allowing the lookup table (LUT)
    and indices of palettized weights to be learnt using a gradient-based optimization algorithm such as SGD.

    Example:

        .. code-block:: python

            import torch
            from coremltools.optimize.torch.palettization import (
                DKMPalettizer,
                DKMPalettizerConfig,
                ModuleDKMPalettizerConfig,
            )

            # code that defines the pytorch model, loss and optimizer.
            model, loss_fn, optimizer = create_model_loss_and_optimizer()

            # initialize the palettizer
            config = DKMPalettizerConfig(global_config=ModuleDKMPalettizerConfig(n_bits=4))

            palettizer = DKMPalettizer(model, config)

            # prepare the model to insert FakePalettize layers for palettization
            model = palettizer.prepare(inplace=True)

            # use palettizer in your PyTorch training loop
            for inputs, labels in data:
                output = model(inputs)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                palettizer.step()

            # fold LUT and indices into weights
            model = palettizer.finalize(inplace=True)

    Args:
        model (:py:class:`torch.nn.Module`): Model on which the palettizer will act.
        config (:py:class:`DKMPalettizerConfig`): Config which specifies how
            different submodules in the model will be configured for palettization.
            Default config is used when passed as ``None``.
    """
    def __init__(self, model: _nn.Module, config: _Optional[_DKMPalettizerConfig] = None):
        config = _DKMPalettizerConfig() if config is None else config
        super().__init__(model, config)
        self._milestones = {}
        self._palettization_module_mapping = _DKMPalettizerModulesRegistry.REGISTRY
        self._supported_modules = _DKMPalettizerModulesRegistry.get_supported_modules()

    def _palettize_supported_modules(self):
        """
        Method to palettize supported modules.
        """
        for name, submodule in self._model.named_modules(remove_duplicate=True):
            config = self._config.get_module_config(name, submodule)
            if isinstance(submodule, self._supported_modules):
                palettizer_module = _DKMPalettizerModulesRegistry.get_palettizer_module(submodule)
                if config is not None and palettizer_module is not None:
                    submod_configs = config if isinstance(config, list) else [config]
                    for submod_config in submod_configs:
                        if all(
                            param.numel() > submod_config.weight_threshold
                            for param, _ in palettizer_module.get_palettizable_parameters(submodule)
                        ):
                            module_level_advanced_options = self._get_module_level_advanced_options(
                                submodule, submod_config
                            )
                            default_config = _DEFAULT_PALETTIZATION_SCHEME[_nn.Linear]
                            n_bits = (
                                submod_config.n_bits
                                if submod_config.n_bits is not None
                                else _DEFAULT_PALETTIZATION_SCHEME.get(
                                    type(submodule), default_config
                                )["n_bits"]
                            )
                            cluster_dim = (
                                submod_config.cluster_dim
                                if submod_config.cluster_dim is not None
                                else _DEFAULT_PALETTIZATION_SCHEME.get(
                                    type(submodule), default_config
                                )["cluster_dim"]
                            )
                            enable_per_channel_scale = (
                                submod_config.enable_per_channel_scale
                                if submod_config.enable_per_channel_scale is not None
                                else _DEFAULT_PALETTIZATION_SCHEME.get(
                                    type(submodule), default_config
                                )["enable_per_channel_scale"]
                            )
                            updated_config = None
                            for (
                                param,
                                param_name,
                            ) in palettizer_module.get_palettizable_parameters(submodule):
                                updated_config = _validate_param_config(
                                    name + "." + param_name,
                                    param,
                                    submodule,
                                    submod_config,
                                    [
                                        "palettization_cluster_dim",
                                        "palettization_group_size",
                                    ],
                                    module_level_advanced_options,
                                )
                                if not updated_config:
                                    break
                            if not updated_config:
                                continue

                            self._attach_qconfig(
                                submodule,
                                n_bits,
                                cluster_dim,
                                enable_per_channel_scale,
                                updated_config.group_size,
                                updated_config.quant_min,
                                updated_config.quant_max,
                                updated_config.lut_dtype,
                                updated_config.dtype,
                                updated_config.quantize_activations,
                                module_level_advanced_options,
                            )
                            self._milestones[name] = updated_config.milestone

    @staticmethod
    def _attach_qconfig(
        module: _nn.Module,
        n_bits: int,
        cluster_dim: int,
        enable_per_channel_scale: bool,
        group_size: _Optional[int],
        quant_min: int,
        quant_max: int,
        lut_dtype: str,
        dtype: _torch.dtype,
        quantize_activations: bool,
        advanced_options: _Dict,
    ):
        """
        Attach :py:class:`QConfig` to a module so that it can be palettized.
        """
        fq_activation = _nn.Identity
        fq_weight = _FakePalettize.with_args(
            observer=_PalettizerLUTObserver.with_args(
                quant_min=quant_min, quant_max=quant_max, dtype=dtype
            ),
            n_bits=n_bits,
            cluster_dim=cluster_dim,
            enable_per_channel_scale=enable_per_channel_scale,
            group_size=group_size,
            quant_min=quant_min,
            quant_max=quant_max,
            lut_dtype=lut_dtype,
            advanced_options=advanced_options,
        )
        if quantize_activations:
            fq_activation = _FakeQuantize.with_args(
                observer=_PalettizerLUTObserver.with_args(
                    quant_min=quant_min, quant_max=quant_max, dtype=dtype
                ),
                quant_min=quant_min,
                quant_max=quant_max,
            )

        module.qconfig = _torch.quantization.QConfig(activation=fq_activation, weight=fq_weight)

    @staticmethod
    def _get_module_level_advanced_options(
        module: _nn.Module, module_level_config: _ModuleDKMPalettizerConfig
    ) -> _ParamsDict:
        """
        Returns advanced_options for a module. First checks whether the user specified something for
        those options in the palettization_config. If not, uses the options from the
        DEFAULT_PALETTIZATION_SCHEME of that module type.
        """
        module_level_advanced_options = {}
        for key in _DEFAULT_PALETTIZATION_ADVANCED_OPTIONS.keys():
            if (
                key == "cluster_permute"
                and module_level_config.cluster_dtype == "oc_last"
                and hasattr(module, "weight")
            ):
                cluster_permute = list(range(module.weight.dim()))
                cluster_permute = cluster_permute[1:] + cluster_permute[:1]
                module_level_advanced_options[key] = cluster_permute
            else:
                module_level_advanced_options[key] = getattr(module_level_config, key)
        return module_level_advanced_options

    def prepare(self, inplace: bool = False) -> _nn.Module:
        """
        Prepares a model for palettization aware training by inserting :py:class:`FakePalettize` layers in appropriate
        places as specified by the config.

        Args:
            inplace (:obj:`bool`): If ``True``, model transformations are carried out in-place and
                the original module is mutated, otherwise a copy of the model is mutated and returned.
        """
        if _is_pruner_prepared(self):
            raise RuntimeError(
                "Model has been prepared for pruning. When running joint compression, "
                "first prepare quantizer/palettizer and then pruner, to ensure insertion of "
                "fake quantization layers does not remove pruning forward hooks."
            )
        self._model = self._get_model_for_compression(inplace)
        self._model.train()
        self._palettize_supported_modules()
        self._model = _torch.quantization.prepare_qat(
            self._model, mapping=self._palettization_module_mapping, inplace=True
        )
        self._default_enable_fake_palett()
        return self._model

    def _default_enable_fake_palett(self):
        for name, module in self._model.named_modules():
            if name in self._milestones and self._milestones[name] == 0:
                self._enable_fake_palett_impl(module, True)

    def finalize(self, model: _Optional[_nn.Module] = None, inplace: bool = False) -> _nn.Module:
        """
        Removes :py:class:`FakePalettize` layers from a model and creates new model weights from the ``LUT`` and
        ``indices`` buffers.

        This function is called to prepare a palettized model for export using
        `coremltools <https://coremltools.readme.io/docs>`_.

        Args:
            model (:obj:`nn.Module`): model to finalize.
            inplace (:obj:`bool`): If ``True``, model transformations are carried out in-place and
                the original module is mutated; otherwise, a copy of the model is mutated and returned.
        """
        if model is None:
            model = self._model
        model.eval()

        finalized_model = _torch.quantization.convert(
            model,
            convert_custom_config_dict=_get_conversion_custom_config_dict(),
            inplace=inplace,
        )

        if model is None:
            self._model = finalized_model
        _register_metadata_version(finalized_model)
        return finalized_model

    def step(self):
        """
        Step through the palettizer. When the number of times ``step``
        is called is equal to ``milestone``, palettization is enabled.
        """
        for name, module in self._model.named_modules():
            # FSDP flattens model parameters leading to change in module names
            # _normalize_fsdp_module_name function call to get original module name
            normalized_name = _normalize_fsdp_module_name(name)
            if normalized_name in self._milestones:
                if self._step_count == self._milestones[normalized_name]:
                    self._enable_fake_palett_impl(module, True)
                    self._init_prune_threshold_and_module_wise_target_sparsity(module)
                if self._step_count > self._milestones[normalized_name]:
                    self._update_prune_threshold(module)
        self._step_count += 1

    @staticmethod
    def _init_prune_threshold_and_module_wise_target_sparsity(module: _torch.nn.Module):
        if hasattr(module, "weight_fake_quant") and hasattr(module, "weight_mask"):
            non_zero_weights = module.weight_mask.count_nonzero().item()
            total_weights = _torch.numel(module.weight_mask)
            target_module_level_sparsity = 1 - non_zero_weights / total_weights
            inverse_mask = (module.weight_mask + 1) % 2
            n_bits = module.weight_fake_quant.n_bits
            cluster_dim = module.weight_fake_quant.cluster_dim
            add_extra_centroid = module.weight_fake_quant.add_extra_centroid
            n_clusters = 2 ** int(n_bits) + int(add_extra_centroid)
            prune_threshold_init = _torch.abs(inverse_mask * module.weight_orig).max() / (
                total_weights / cluster_dim / n_clusters
            )

            module.weight_fake_quant.prune_threshold = prune_threshold_init
            module.weight_fake_quant._target_module_level_sparsity = target_module_level_sparsity

    @staticmethod
    def _update_prune_threshold(module: _torch.nn.Module):
        if hasattr(module, "weight_fake_quant") and hasattr(module, "weight_mask"):
            weight_detached = module.weight.detach()
            qweight = module.weight_fake_quant.palettize(weight_detached)

            sparsity = 1 - qweight.count_nonzero() / qweight.numel()
            prune_ratio = float(module.weight_fake_quant._target_module_level_sparsity) / (
                sparsity + 1e-7
            )
            if prune_ratio > 0 and abs(prune_ratio - 1) > 0.01:
                prune_multiplier = max(min(prune_ratio, 1.25), 0.9)
                module.weight_fake_quant.prune_threshold *= prune_multiplier

    def enable_fake_palett(self, flag: bool):
        _logging.info(
            f"[{type(self).__name__}] " + ("enable" if flag else "disable") + " fake_palett"
        )
        for name, module in self._model.named_modules():
            self._enable_fake_palett_impl(module, flag)

    @staticmethod
    def _enable_fake_palett_impl(module: _torch.nn.Module, flag: bool):
        def enable_fn(mod):
            if hasattr(mod, "enable_fake_palett"):
                mod.enable_fake_palett(flag)

        module.apply(enable_fn)

    def report(self) -> _Report:
        """
        Returns a dictionary with important statistics related to current state of palettization.
        Each key in the dictionary corresponds to a module name, and the
        value is a dictionary containing the statistics, such as number of clusters and
        cluster dimension, number of parameters, and so on.
        """
        report = _Report()
        with _get_eval_model(self._model) as model:
            with _torch.no_grad():
                for name, module in model.named_modules():
                    module_summary = dict()
                    if hasattr(module, "weight_fake_quant"):
                        module_summary["device"] = module.weight.device
                        qweight = module.weight_fake_quant.forward(module.weight.detach())
                        lut_dtype = module.weight_fake_quant.lut_dtype
                        cluster_permute = module.weight_fake_quant.cluster_permute
                        module_summary["error"] = _rmse_error(
                            module.weight.detach(), qweight
                        ).item()
                        n_clusters = module.weight_fake_quant.n_clusters
                        module_summary["#params"] = int(_torch.numel(qweight))
                        cluster_dim = module.weight_fake_quant.cluster_dim
                        module_summary["#dtype"] = (
                            f":num_clusters: {n_clusters} <{lut_dtype, cluster_permute}> "
                            f"dim={cluster_dim}"
                        )
                        report[name] = module_summary
        return report
