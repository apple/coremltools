#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy as _copy
import logging as _logging
import tempfile as _tempfile
from collections import OrderedDict as _OrderedDict
from contextlib import contextmanager as _contextmanager
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import Iterable as _Iterable
from typing import List as _List
from typing import NewType as _NewType
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

import cattrs as _cattrs
import torch as _torch
import torch.multiprocessing as _mp
from attr import define as _define
from attr import field as _field
from attrs import validators as _validators
from torch.distributed.fsdp import FullStateDictConfig as _FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
from torch.distributed.fsdp import ShardingStrategy as _ShardingStrategy
from torch.distributed.fsdp import StateDictType as _StateDictType

from coremltools.optimize.torch._utils.dist_utils import ddp_setup as _ddp_setup
from coremltools.optimize.torch._utils.dist_utils import is_leader as _is_leader
from coremltools.optimize.torch._utils.fsdp_utils import FSDPAutoWrapPolicy as _FSDPAutoWrapPolicy
from coremltools.optimize.torch._utils.k_means import KMeansConfig as _KMeansConfig
from coremltools.optimize.torch._utils.k_means import (
    KMeansSupportedModulesRegistry as _KMeansSupportedModulesRegistry,
)
from coremltools.optimize.torch._utils.k_means import ParallelKMeans as _ParallelKMeans
from coremltools.optimize.torch._utils.k_means import SequentialKMeans as _SequentialKMeans
from coremltools.optimize.torch._utils.report_utils import (
    compute_post_training_report as _compute_post_training_report,
)
from coremltools.optimize.torch._utils.torch_utils import (
    maybe_convert_str_to_dtype as _maybe_convert_str_to_dtype,
)
from coremltools.optimize.torch._utils.torch_utils import (
    maybe_convert_str_to_mod_type as _maybe_convert_str_to_mod_type,
)
from coremltools.optimize.torch._utils.validation_utils import (
    validate_param_config as _validate_param_config,
)
from coremltools.optimize.torch.base_model_optimizer import (
    BaseDataCalibratedModelOptimizer as _BaseDataCalibratedModelOptimizer,
)
from coremltools.optimize.torch.base_model_optimizer import _Report
from coremltools.optimize.torch.optimization_config import (
    ModuleOptimizationConfig as _ModuleOptimizationConfig,
)
from coremltools.optimize.torch.optimization_config import OptimizationConfig as _OptimizationConfig
from coremltools.optimize.torch.optimization_config import (
    PalettizationGranularity,
    _structure_from_dict_hook_factory,
)

_logger = _logging.getLogger(__name__)


@_define
class ModuleSKMPalettizerConfig(_ModuleOptimizationConfig):
    """
    Configuration class for specifying global and module-level palettization options for
    :py:class:`SKMPalettizer` algorithm.

    Args:
        n_bits (:obj:`int`): Number of bits to use for palettizing the weights. Defaults to ``4``.
        lut_dtype (:py:class:`torch.dtype`): The dtype to use for representing each element in lookup tables.
            When value is ``None``, no quantization is performed. Supported values are :py:class:`torch.int8` and
            :py:class:`torch.uint8`. Defaults to ``None``.
        granularity (:py:class:`PalettizationGranularity`) – Granularity for palettization.
            One of ``per_tensor`` or ``per_grouped_channel``. Defaults to ``per_tensor``.
        group_size (:obj:`int`): Specify the number of channels in a group.
            Only effective when granularity is ``per_grouped_channel``.
        channel_axis (:obj:`int`): Specify the channel axis to form a group of channels.
            Only effective when granularity is ``per_grouped_channel``. Defaults to output channel axis.
        cluster_dim (:obj:`int`): The dimension of centroids for each lookup table.
            The centroid is a scalar by default. When ``cluster_dim > 1``, it indicates 2-D clustering,
            and each ``cluster_dim`` length of weight vectors along the output channel are palettized
            using the same 2-D centroid. The length of each entry in the lookup tables is equal to ``cluster_dim``.
        enable_per_channel_scale (:obj:`bool`): When set to ``True``, weights are normalized along the output channels
            using per-channel scales before being palettized. This is not supported with ``cluster_dim > 1``.

    This class supports two different configurations to structure the palettization:

    1. **Per-tensor palettization**:  This is the default configuration where the whole tensor shares a single lookup
    table. The ``granularity`` is set to ``per_tensor``, and ``group_size`` is ``None``.

    2. **Per-grouped-channel palettization**: In this configuration, the number of channels ``group_size`` along
    ``channel_axis`` share the same lookup table. For example, for a weight matrix of shape ``(16, 25)``, if we provide
    ``group_size = 8``, the shape of the lookup table would be ``(2, 2^n_bits)``.

    .. note::
        Grouping is currently only supported along either the input or output channel axis.
    """

    n_bits: int = _field(default=4, validator=_validators.instance_of(int))
    lut_dtype: _torch.dtype = _field(
        default=None,
        converter=lambda val: _maybe_convert_str_to_dtype(val) if val else val,
        validator=_validators.optional(
            [
                _validators.instance_of(_torch.dtype),
                _validators.in_([_torch.int8, _torch.uint8]),
            ]
        ),
    )
    granularity: PalettizationGranularity = _field(
        default="per_tensor",
        converter=PalettizationGranularity,
        validator=_validators.in_(PalettizationGranularity),
    )
    group_size: _Optional[int] = _field(
        default=None, validator=_validators.optional(_validators.instance_of(int))
    )
    channel_axis: int = _field(
        default=0,
        validator=_validators.optional([_validators.instance_of(int), _validators.in_([0, 1])]),
    )
    cluster_dim: _Optional[int] = _field(
        default=None, validator=_validators.optional(_validators.instance_of(int))
    )
    enable_per_channel_scale: bool = _field(
        default=False, validator=_validators.optional(_validators.instance_of(bool))
    )

    @group_size.validator
    def per_grouped_channel_granularity(self, attribute, value):
        if self.granularity == PalettizationGranularity.per_grouped_channel:
            assert (
                value is not None
            ), "group_size has to be specified along with per_grouped_channel granularity."
            assert value > 0, "group_size should be greater than zero"
        else:
            assert value is None, "group_size can't be specified along with per_tensor granularity."

    @cluster_dim.validator
    def no_per_channel_scale(self, attribute, value):
        if value and value > 1:
            assert (
                self.enable_per_channel_scale == False
            ), f"Enabling per_channel_scale is not supported for cluster_dim={value} larger than 1"


_ModuleTypeConfigType = _NewType(
    "ModuleTypeConfigType",
    _Dict[_Union[_Callable, str], _Optional[ModuleSKMPalettizerConfig]],
)


@_define
class SKMPalettizerConfig(_OptimizationConfig):
    """
    Configuration class for specifying how different submodules of a model are
    palettized by :py:class:`SKMPalettizer`.

    Args:
        global_config (:py:class:`ModuleSKMPalettizerConfig`): Config to be applied globally
            to all supported modules. Missing values are chosen from the default config.
        module_type_configs (:obj:`dict` of :obj:`str` to :py:class:`ModuleSKMPalettizerConfig`):
            Module type configs applied to a specific module class, such as :py:class:`torch.nn.Linear`.
            The keys can be either strings or module classes.
        module_name_configs (:obj:`dict` of :obj:`str` to :py:class:`ModuleSKMPalettizerConfig`):
            Module-level configs applied to specific modules. The name of the module must either be
            a regex or a fully qualified name that can be used to fetch it from the top level module
            using the ``module.get_submodule(target)`` method.
        calibration_nsamples (:obj:`int`): Number of samples to be used for calibration.
    """

    global_config: _Optional[ModuleSKMPalettizerConfig] = _field(
        default=None,
        validator=_validators.optional(_validators.instance_of(ModuleSKMPalettizerConfig)),
    )
    module_type_configs: _ModuleTypeConfigType = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.instance_of((str, _Callable)),
            value_validator=_validators.optional(
                _validators.instance_of(ModuleSKMPalettizerConfig)
            ),
            mapping_validator=_validators.instance_of(dict),
        ),
    )
    module_name_configs: _Dict[str, _Optional[ModuleSKMPalettizerConfig]] = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.instance_of(str),
            value_validator=_validators.optional(
                _validators.instance_of(ModuleSKMPalettizerConfig)
            ),
            mapping_validator=_validators.instance_of(dict),
        ),
    )
    calibration_nsamples: int = _field(default=128, validator=_validators.instance_of(int))

    def __attrs_post_init__(self):
        if (
            self.global_config is None
            and len(self.module_type_configs) == 0
            and len(self.module_name_configs) == 0
        ):
            self.global_config = ModuleSKMPalettizerConfig()
        self.module_type_configs = {
            _maybe_convert_str_to_mod_type(key): val
            for key, val in self.module_type_configs.items()
        }

    @classmethod
    def from_dict(cls, config_dict: _Dict[str, _Any]) -> "SKMPalettizerConfig":
        super().from_dict(config_dict)
        converter = _cattrs.Converter(forbid_extra_keys=True)
        converter.register_structure_hook(
            _ModuleTypeConfigType,
            _structure_from_dict_hook_factory(ModuleSKMPalettizerConfig),
        )
        return converter.structure_attrs_fromdict(config_dict, cls)


class SKMPalettizer(_BaseDataCalibratedModelOptimizer):
    """
    Perform post-training palettization of weights by running a weighted k-means
    on the model weights. The weight values used for weighing different elements of
    a model's weight matrix are computed using the Fisher information matrix, which
    is an approximation of the Hessian. These weight values indicate how sensitive
    a given weight element is: the more sensitive an element, the larger the impact perturbing
    or palettizing it has on the model’s loss function. This means that weighted k-means
    moves the clusters closer to the sensitive weight values, allowing them to be
    represented more exactly. This leads to a lower degradation in model performance
    after palettization. The Fisher information matrix is computed using a few
    samples of calibration data.

    This algorithm implements `SqueezeLLM: Dense-and-Sparse Quantization <https://arxiv.org/pdf/2306.07629.pdf>`_.

    Example:

            .. code-block:: python

                import torch.nn as nn
                from coremltools.optimize.torch.palettization import (
                    SKMPalettizer,
                    SKMPalettizerConfig,
                )

                model = nn.Sequential(
                    OrderedDict(
                        {
                            "conv": nn.Conv2d(1, 20, (3, 3)),
                            "relu1": nn.ReLU(),
                            "conv2": nn.Conv2d(20, 20, (3, 3)),
                            "relu2": nn.ReLU(),
                        }
                    )
                )

                dataloader = load_calibration_data()

                # define callable for loss function
                def loss_fn(model, data):
                    inp, target = data
                    out = model(inp)
                    return nn.functional.mse_loss(out, target)


                # initialize the palettizer
                config = SKMPalettizerConfig.from_dict(
                    {
                        "global_config": {
                            "n_bits": 4,
                        },
                        "calibration_nsamples": 16,
                    }
                )

                compressor = SKMPalettizer(model, config)
                compressed_model = compressor.compress(dataloader=dataloader, loss_fn=loss_fn)

    Args:
        model (:obj:`torch.nn.Module`): Module to be compressed.
        config (:py:class:`LayerwiseCompressorConfig`): Config that specifies how
            different submodules in the model will be compressed.
    """

    _supported_modules: _Tuple = _KMeansSupportedModulesRegistry.get_supported_modules()
    _SENSITIVITY_CLIP_THR: int = 1e-12

    def __init__(self, model: _torch.nn.Module, config: _Optional[SKMPalettizerConfig] = None):
        config = SKMPalettizerConfig() if config is None else config
        super().__init__(model, config)
        self._tempdir = _tempfile.TemporaryDirectory()
        self._sensitivity_path = self._tempdir.name + "/sensitivity.pt"
        self._model_checkpoint_path = self._tempdir.name + "/model.pt"

    def _compute_sensitivity_impl_single_worker(
        self, dataset: _List, loss_fn: _Callable, sensitivity_path: _Optional[str]
    ):
        """
        Computes sensitivity for the model weights using a single process.
        """
        if _torch.cuda.is_available():
            self._model.cuda()

        self._model.zero_grad()

        with self._register_grad_square_hooks(self._model):
            for didx, data in enumerate(dataset):
                _logger.info(f"Computing sensitivity using sample {didx}")
                loss = loss_fn(self._model, data)
                loss.backward()

            sensitivity_dict = dict()
            for name, param in self._model.named_parameters(remove_duplicate=True):
                if param.requires_grad:
                    sensitivity_dict[name] = -param.grad.cpu()

            _torch.save(sensitivity_dict, self._get_sensitivity_path(sensitivity_path))

    def _compute_sensitivity_impl_multiple_workers(
        self,
        rank: int,
        num_workers: int,
        dataset: _List,
        loss_fn: _Callable,
        sensitivity_path: _Optional[str] = None,
        fsdp_auto_wrap_policy: _Optional[_FSDPAutoWrapPolicy] = None,
    ):
        """
        Computes sensitivity for the model weights using multiple processes.
        This mode is useful for large models for which computing gradients on a single
        process is infeasible because the model does not fit on a single GPU. The model is
        sharded on multiple GPUs using :py:class:`FullyShardedDataParallel`, which enables
        distributed computation of gradients.

        If ``sensitivity_path`` is passed as ``None``, the sensitivity matrices are
        stored temporarily and deleted after model compression. Otherwise, they are
        saved at the location specified by ``sensitivity_path``.

        Args:
            rank (:obj:`int`):  Rank of the worker process on which this function is executed
            num_workers (:obj:`int`): Number of workers used for computing sensitivity
            dataset (:py:class:`Iterable`): An iterable where each element
                is an input to the model to be compressed. Used for computing gradients of model weights.
            loss_fn (:obj:`Callable`): A callable which takes the model and data as input and performs
                a forward pass on the model and computes the training loss
            sensitivity_path (:obj:`str` or ``None``): An optional path for saving the sensitivity
                of weights. Defaults to ``None``.
            fsdp_auto_wrap_policy (:py:class:`_FSDPAutoWrapPolicy` or ``None``): Policy to apply
                :py:class:`FullyShardedDataParallel` to submodules of ``model``. Defaults to ``None``.
        """
        _ddp_setup(rank, num_workers)
        auto_wrap_policy = (
            fsdp_auto_wrap_policy.get_policy() if fsdp_auto_wrap_policy is not None else None
        )
        model = _FSDP(
            module=self._model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=_ShardingStrategy.FULL_SHARD,
            use_orig_params=False,
            device_id=_torch.cuda.current_device(),
            sync_module_states=True,
        )

        # We want to compute squares of gradients of the un-sharded parameters
        # to use later for k-means. However, parameters are sharded and gradients
        # are also computed in the sharded state. And there is no efficient way
        # to un-shard them, hence we use an optimizer to add the sharded gradients
        # to the parameters, which can later be un-sharded when we save the state dict.
        optim = _torch.optim.SGD(
            [param for param in model.parameters() if param.requires_grad], lr=1.0
        )
        optim.zero_grad()

        with self._register_grad_square_hooks(model):
            for didx, data in enumerate(dataset):
                if _is_leader():
                    _logger.info(f"Computing sensitivity using sample {didx}")
                loss = loss_fn(model, data)
                loss.backward()

            # we set the parameters to zero so that when we call optim.step,
            # the parameter values are equal to the square of the gradient
            with _torch.no_grad():
                for param in model.parameters():
                    param.data.zero_()

            optim.step()

            cfg = _FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with _FSDP.state_dict_type(model, _StateDictType.FULL_STATE_DICT, cfg):
                sensitivity_dict = model.state_dict()

            if _is_leader():
                _torch.save(sensitivity_dict, self._get_sensitivity_path(sensitivity_path))

    def _get_dataset(self, rank: int, num_workers: int, dataloader: _Iterable) -> _List[_Any]:
        """
        Create a subset of dataloader for worker with given rank.
        """
        dataset = []
        num_samples = self._config.calibration_nsamples // num_workers
        sampled = 0
        for idx, data in enumerate(dataloader):
            if idx % num_workers == rank:
                dataset.append(_copy.deepcopy(data))
                sampled += 1
            if sampled == num_samples:
                break
        return dataset

    @staticmethod
    @_contextmanager
    def _register_grad_square_hooks(model: _torch.nn.Module):
        """
        Context manager for registering gradient squaring hooks within the context
        and unregistering them on exit.
        """
        hook_handles = []
        for param in model.parameters():
            if param.requires_grad:
                hook_handles.append(param.register_hook(lambda grad: _torch.square(grad)))
        try:
            yield model
        finally:
            for handle in hook_handles:
                handle.remove()

    def _get_sensitivity_path(self, sensitivity_path: _Optional[str]) -> str:
        """
        Return sensitivity_path if it's not None else a temporary path
        """
        return sensitivity_path if sensitivity_path is not None else self._sensitivity_path

    def compute_sensitivity(
        self,
        dataloader: _Iterable,
        loss_fn: _Callable,
        sensitivity_path: _Optional[str] = None,
        num_sensitivity_workers: int = 1,
        fsdp_auto_wrap_policy: _Optional[_FSDPAutoWrapPolicy] = None,
    ) -> _Dict[str, _Any]:
        """
        Compute sensitivities of model's weights. A weight element's sensitivity indicates
        how much effect perturbing it has on the model's loss function. The sensitivities
        are computed as Fisher information of the model's weights.

        If ``sensitivity_path`` is passed as a non ``None`` value, the sensitivity matrices
        saved at the location specified by ``sensitivity_path``.

        When computing sensitivity of large models, it is recommended to use ``num_sensitivity_workers``
        equal to the number of GPUs available. This is because computing gradients using a single
        process maybe infeasible for a large model as it may not fit on a single GPU.
        When ``num_sensitivity_workers > 1``, the model is sharded on multiple GPUs using
        :py:class:`FullyShardedDataParallel`, which enables distributed computation of gradients.

        Args:
            dataloader (:py:class:`Iterable`): An iterable where each element
                is an input to the model to be compressed. Used for computing gradients of model weights.
            loss_fn (:obj:`Callable`): A callable which takes the model and data as input and performs
                a forward pass on the model and computes the training loss
            sensitivity_path (:obj:`str` or ``None``): An optional path for saving the sensitivity
                of weights. Defaults to ``None``.
            num_sensitivity_workers (:obj:`int`): Number of worker processes used for computing sensitivity.
                Defaults to ``1``.
            fsdp_auto_wrap_policy (:py:class:`_FSDPAutoWrapPolicy` or ``None``): Policy which specifies
                how different submodules of ``model`` are wrapped with individual
                :py:class:`FullyShardedDataParallel` wrappers. This argument is only used when
                ``num_sensitivity_workers > 1`` and it is only necessary when the model cannot be fit on a single GPU.
                 Please refer to documentation of :py:class:`_FSDPAutoWrapPolicy` for more details.
                 Defaults to ``None`.
        """
        if num_sensitivity_workers > 1 and not _torch.cuda.is_available():
            _logger.warning(
                "num_sensitivity_workers > 1 is only supported on GPUs with CUDA. Setting "
                "num_sensitivity_workers to 1, since a CUDA compatible PyTorch installation"
                "couldn't be found."
            )
            num_sensitivity_workers = 1

        # We save the model's state dict so that we can restore it later
        # We need to do this because _compute_sensitivity_impl_multiple_workers
        # sets the parameters' value to squares of their gradients and
        # _compute_sensitivity_impl_single_worker can modify layers such as batch norm
        # during forward pass
        _torch.save(self._model.state_dict(), self._model_checkpoint_path)
        if num_sensitivity_workers == 1:
            self._compute_sensitivity_impl_single_worker(
                self._get_dataset(0, 1, dataloader),
                loss_fn,
                sensitivity_path,
            )
        else:
            if fsdp_auto_wrap_policy is None:
                _logger.warning(
                    "num_sensitivity_workers > 1 and fsdp_auto_wrap_policy is None. For a large model, this might "
                    "lead to OOM issue on GPUs. Consider setting fsdp_auto_wrap_policy to indicate how different "
                    "submodules of the model should be wrapped with FSDP wrappers to prevent all gather for all "
                    "parameters on all GPUs."
                )

            ctx = _mp.get_context("spawn")

            worker_processes = [
                ctx.Process(
                    target=self._compute_sensitivity_impl_multiple_workers,
                    args=(
                        rank,
                        num_sensitivity_workers,
                        self._get_dataset(rank, num_sensitivity_workers, dataloader),
                        loss_fn,
                        sensitivity_path,
                        fsdp_auto_wrap_policy,
                    ),
                    name=f"Process-{rank}",
                )
                for rank in range(num_sensitivity_workers)
            ]
            for worker_process in worker_processes:
                worker_process.start()
                _logger.info(f"Started {worker_process.name} for computing sensitivity.")

            for worker_process in worker_processes:
                worker_process.join()
                _logger.info(f"Finished {worker_process.name}.")

        # restore the original state of the model
        self._model.cpu()
        old_state_dict = _torch.load(self._model_checkpoint_path)
        self._model.load_state_dict(old_state_dict)

        return self._process_sensitivity(sensitivity_path)

    def _process_sensitivity(self, sensitivity_path: _Optional[str] = None) -> _Dict[str, _Any]:
        """
        Post process the sensitivity values to normalize them.
        """
        raw_sensitivity_dict = _torch.load(self._get_sensitivity_path(sensitivity_path))
        sensitivity_dict = dict()
        for key, val in raw_sensitivity_dict.items():
            # Since optimizer sets param value as: p <= p - learning_rate * (grad**2),
            # we need to negate the values to get grad**2
            val = 100 * -val
            if len(val.nonzero()) == 0:
                val[val == 0] = 1.0

            # normalize sensitivity between 0 and 1
            val = val / _torch.max(val)

            # Clipping very small or zero sensitivity values stabilizes k-means,
            # they can lead to divergence otherwise
            val[val == 0] = _torch.min(val[val != 0])
            val[val < self._SENSITIVITY_CLIP_THR] = self._SENSITIVITY_CLIP_THR

            sensitivity_dict[key] = val

        # If user wants to save sensitivity values at the specified path
        # we save them in the processed state
        if sensitivity_path is not None:
            _torch.save(sensitivity_dict, sensitivity_path)
        return sensitivity_dict

    def _compute_outlier_mask(self, sensitivity: _torch.Tensor, outliers: float) -> _torch.Tensor:
        """
        Compute outlier masks using the sensitivity values.
        """
        sensitivity_flat = sensitivity.flatten()
        numel = sensitivity_flat.numel()
        num_outliers = int(numel * (outliers / 100.0))
        mask = _torch.ones_like(sensitivity_flat, dtype=_torch.bool)
        mask[_torch.argsort(sensitivity_flat, descending=True)[:num_outliers]] = False
        return mask.reshape(sensitivity.shape)

    def _get_submodules_to_compress(self) -> _Iterable[_Tuple[str, _torch.nn.Module]]:
        """
        Return an iterator over the names and submodules to be compressed.
        """
        for name, submodule in self._model.named_modules():
            yield name, submodule

    def compress(
        self,
        dataloader: _Optional[_Iterable] = None,
        loss_fn: _Optional[_Callable] = None,
        sensitivity_path: _Optional[str] = None,
        num_kmeans_workers: int = 1,
        num_sensitivity_workers: int = 1,
        inplace: bool = False,
        fsdp_auto_wrap_policy: _Optional[_FSDPAutoWrapPolicy] = None,
    ) -> _torch.nn.Module:
        """
        Compresses a model's weights using Fisher information sensitivity based weighted k-means
        palettization.

        Args:
            dataloader (:py:class:`Iterable`): An iterable where each element
                is an input to the model to be compressed. Used for computing gradients of model weights.
                This argument is not needed if ``sensitivity_path`` is specified and will be ignored.
                It is required then ``sensitivity_path`` is ``None``.  Defaults to ``None``.
            loss_fn (:obj:`Callable`): A callable which takes the model and data as input and performs
                a forward pass on the model and computes the training loss. This argument is not needed if
                ``sensitivity_path`` is specified and will be ignored. It is required when ``sensitivity_path``
                is ``None``. Defaults to ``None``.
            sensitivity_path (:obj:`str` or ``None``): An optional path from which the sensitivity values
                are loaded. If ``sensitivity_path`` is not ``None``, sensitivity values are loaded from the
                path specified, otherwise, sensitivity values are computed using the ``dataloader`` and
                ``loss_fn``. The sensitivity values stored at ``sensitivity_path`` should be a dictionary
                from strings indicating fully qualified parameter names to tensors with the same shape as the
                parameters, with each element of the tensor indicating how important that element is. This is
                usally the output of the :py:meth:`compute_sensitivity` method. Defaults to ``None``.
            num_kmeans_workers (:obj:`int`): Number of worker processes to use for performing k-means.
                It is recommended to use more than one worker process to parallelize the clustering,
                especially when multiple CPUs are available. Defaults to ``1``.
            num_sensitivity_workers (:obj:`int`): Number of worker processes to use for computing
                sensitivity. For large models, it is recommended to set this value to the number
                of GPUs available. Defaults to ``1``.
            inplace (:obj:`bool`): If ``True``, model transformations are carried out in-place and
                the original module is mutated, otherwise a copy of the model is mutated and returned.
                Defaults to ``False``.
            fsdp_auto_wrap_policy (:py:class:`_FSDPAutoWrapPolicy` or ``None``): Policy which specifies
                how different submodules of ``model`` are wrapped with individual
                :py:class:`FullyShardedDataParallel` wrappers. This argument is only used when
                ``num_sensitivity_workers > 1`` and it is only necessary when the model cannot be fit on a single GPU.
                 Please refer to documentation of :py:class:`_FSDPAutoWrapPolicy` for more details.
                 Defaults to ``None`.
        """
        self._model = super().compress(dataloader=dataloader, inplace=inplace)
        if sensitivity_path is None:
            sensitivity_dict = self.compute_sensitivity(
                dataloader,
                loss_fn,
                sensitivity_path,
                num_sensitivity_workers,
                fsdp_auto_wrap_policy=fsdp_auto_wrap_policy,
            )
        else:
            _logger.info(f"Loading sensitivity values from {sensitivity_path}.")
            sensitivity_dict = _torch.load(sensitivity_path)

        kmeans_config_dict = dict()
        for name, submodule in self._get_submodules_to_compress():
            submod_config = self._config.get_module_config(name, submodule)
            if submod_config is None:
                continue

            k_means_module_cls = _KMeansSupportedModulesRegistry.get_kmeans_module(submodule)
            if k_means_module_cls is None:
                continue

            for param_name in k_means_module_cls.parameter_names:
                # Validate configuration for parameter
                param = submodule.get_parameter(param_name)
                updated_config = _validate_param_config(
                    name + "." + param_name,
                    param,
                    submodule,
                    submod_config,
                    ["palettization_group_size", "palettization_cluster_dim"],
                )
                if not updated_config:
                    continue

                sensitivity_key = f"{name}.{param_name}" if len(name) > 0 else param_name
                sensitivity = sensitivity_dict[sensitivity_key]

                if name not in kmeans_config_dict:
                    kmeans_config_dict[name] = {}

                kmeans_config_dict[name][param_name] = _KMeansConfig(
                    n_bits=updated_config.n_bits,
                    axis=updated_config.channel_axis,
                    lut_dtype=updated_config.lut_dtype,
                    block_size=updated_config.group_size,
                    importance=sensitivity,
                    cluster_dim=updated_config.cluster_dim,
                    enable_per_channel_scale=updated_config.enable_per_channel_scale,
                )

        if num_kmeans_workers > 1:
            return _ParallelKMeans.cluster_weights(
                self._model, kmeans_config_dict, num_workers=num_kmeans_workers
            )
        else:
            return _SequentialKMeans.cluster_weights(self._model, kmeans_config_dict)

    def report(self) -> _Report:
        return _compute_post_training_report(
            self._uncompressed_model,
            self._model,
            supported_modules=self._supported_modules,
        )
