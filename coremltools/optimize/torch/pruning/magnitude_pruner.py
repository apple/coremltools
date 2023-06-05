#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy as _copy
import logging as _logging
from collections import OrderedDict as _OrderedDict
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import NewType as _NewType
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

import attrs as _attrs
import cattrs as _cattrs
import torch as _torch
from attr import define as _define
from attr import field as _field
from attrs import validators as _validators

from coremltools.optimize.torch._typing import ParamsDict as _ParamsDict
from coremltools.optimize.torch.optimization_config import (
    ModuleOptimizationConfig as _ModuleOptimizationConfig,
)
from coremltools.optimize.torch.optimization_config import OptimizationConfig as _OptimizationConfig
from coremltools.optimize.torch.optimization_config import (
    _structure_from_dict_hook_factory,
    _validate_module_type_keys_factory,
)
from coremltools.optimize.torch.pruning._base_pruner import (
    BasePrunerWithPruningMethod as _BasePrunerWithPruningMethod,
)
from coremltools.optimize.torch.pruning._base_pruner import _allowed_granularity_values
from coremltools.optimize.torch.pruning._base_pruning_method import (
    ScheduledBaseDynamicPruningMethod as _ScheduledBaseDynamicPruningMethod,
)
from coremltools.optimize.torch.pruning._utils import (
    magnitude_ranked_mask as _magnitude_ranked_mask,
)
from coremltools.optimize.torch.pruning._utils import n_m_mask as _n_m_mask
from coremltools.optimize.torch.pruning.pruning_scheduler import (
    ConstantSparsityScheduler as _ConstantSparsityScheduler,
)
from coremltools.optimize.torch.pruning.pruning_scheduler import (
    PruningScheduler as _PruningScheduler,
)
from coremltools.optimize.torch.pruning.pruning_scheduler import _PruningSchedulerType

_logger = _logging.getLogger(__name__)


_SUPPORTED_MODULES = (_torch.nn.Linear, _torch.nn.Conv1d, _torch.nn.Conv2d, _torch.nn.Conv3d)


@_define
class ModuleMagnitudePrunerConfig(_ModuleOptimizationConfig):
    """
    Module level configuration for :py:class:`MagnitudePruner`.

    Args:
        scheduler (:py:class:`PruningScheduler`): A pruning scheduler which specifies how the
            sparsity should be changed over the course of the training.
        initial_sparsity (:obj:`float`): Desired fraction of zeroes at the beginning of the
            training process.
        target_sparsity (:obj:`float`): Desired fraction of zeroes at the end of the
            training process.
        granularity (:obj:`str`): Specifies the granularity at which the pruning mask will be
            computed. Can be one of ``per_layer``, ``per_channel``, ``per_kernel``, ``per_scalar``.
        block_size (:obj:`int`): Block size for inducing block sparsity within the mask. This
            is applied on the output channel dimension of the parameter (the ``0`` -th dimension). Having the zeros aligned
            in the parameter helps gain latency/memory performance on-device. ``block_size`` must be greater than ``1``
            to enable block sparsity, and must be at most half the number of output channels, and must be divisible by
            the number of output channels.
        n_m_ratio (:obj:`tuple` of :obj:`int`): A tuple of two integers which specify how ``n:m`` pruning should be
            applied. In ``n:m`` pruning, out of every ``m`` elements,
            ``n`` with lowest magnitude are set to zero. When ``n_m_ratio`` is not ``None``, ``block_size``,
            ``granularity``, and ``initial_sparsity`` should be ``1``, ``per_scalar``, and ``0.0`` respectively.
            The value of ``target_sparsity`` is ignored and the actualy target sparsity is determined by the
            ``n:m`` ratio. For more information, see `Learning N:M Fine-Grained Structured Sparse Neural Networks From Scratch <https://arxiv.org/abs/2102.04010>`_.
        dim (:obj:`int`): Dimension along which blocks of ``m`` elements are chosen when applying ``n:m`` sparsity. This
            parameter is only used when ``n_m_ratio`` is not ``None``.
        param_name (:obj:`str`): The name of the parameter to be pruned. Defaults to ``weight``.
    """

    scheduler: _PruningSchedulerType = _field(
        default=_ConstantSparsityScheduler(begin_step=0),
        validator=_validators.instance_of(_PruningScheduler),
    )
    initial_sparsity: float = _field(default=0.0, validator=_validators.instance_of(float))
    target_sparsity: float = _field(default=0.5, validator=_validators.instance_of(float))
    granularity: str = _field(
        default="per_scalar",
        validator=[_validators.instance_of(str), _validators.in_(_allowed_granularity_values)],
    )
    block_size: int = _field(default=1, validator=_validators.instance_of(int))
    n_m_ratio: _Optional[_Tuple[int, int]] = _field(
        default=None,
        validator=_attrs.validators.optional(
            _validators.deep_iterable(
                member_validator=_validators.instance_of(int),
                iterable_validator=_validators.instance_of((tuple, list)),
            )
        ),
    )
    dim: int = _field(default=1, validator=_validators.instance_of(int))
    param_name: str = _field(default="weight", validator=_validators.instance_of(str))

    def __attrs_post_init__(self):
        if self.n_m_ratio is not None:
            assert (
                len(self.n_m_ratio) == 2
            ), f"n_m_ratio must be a tuple of 2 integers, received: {self.n_m_ratio}"
            n, m = self.n_m_ratio
            assert m > 0, f"Received n_m_ratio (n, m): {self.n_m_ratio}. m must be greater than 0."
            assert n <= m, (
                f"Received n_m_ratio (n, m): {self.n_m_ratio}. The number of zero in a block (n) "
                f"must be less than or equal to the block size (m)."
            )
            if self.block_size is not None and self.block_size > 1:
                raise ValueError(
                    f"Received block_size = {self.block_size} and n_m_ratio = {self.n_m_ratio}. "
                    f"These two modes are mutually exclusive. When n_m_ratio != None, "
                    f"the only allowed value of block_size is 1. "
                    f"n_m_ratio should be equal to None for block_size > 1."
                )
            if self.granularity is not None and self.granularity != "per_scalar":
                raise ValueError(
                    f"Received granularity = {self.granularity} and n_m_ratio = {self.n_m_ratio}. "
                    f"When n_m_ratio != None, the only allowed value of granularity is "
                    f"per_scalar."
                )
            if self.initial_sparsity is not None and self.initial_sparsity > 0.0:
                raise ValueError(
                    f"Received initial_sparsity = {self.initial_sparsity} and "
                    f"n_m_ratio = {self.nm_ratio}. When n_m_ratio != None, the only allowed "
                    f"value of initial_sparsity is 0."
                )


_ModuleTypeConfigType = _NewType(
    "ModuleTypeConfigType",
    _Dict[_Union[_Callable, str], _Optional[ModuleMagnitudePrunerConfig]],
)


@_define
class MagnitudePrunerConfig(_OptimizationConfig):
    """
    Configuration for :py:class:`MagnitudePruner`.

    Args:
        global_config (:py:class:`ModuleMagnitudePrunerConfig`): Config to be applied globally
            to all supported modules. Missing values are chosen from the default config.
        module_type_configs (:obj:`dict` of :obj:`str` to :py:class:`ModuleMagnitudePrunerConfig`): Module
            type level configs applied to a specific module class, such as :py:class:`torch.nn.Linear`.
            The keys can be either strings or module classes.
        module_name_configs (:obj:`dict` of :obj:`str` to :py:class:`ModuleMagnitudePrunerConfig`): Module level
            configs applied to specific modules. The name of the module must be a fully qualified name that can
            be used to fetch it from the top level module using the ``module.get_submodule(target)`` method.
    """

    global_config: _Optional[ModuleMagnitudePrunerConfig] = _field(
        default=None,
        validator=_validators.optional(_validators.instance_of(ModuleMagnitudePrunerConfig)),
    )
    module_type_configs: _ModuleTypeConfigType = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.and_(
                _validators.instance_of((str, _Callable)),
                _validate_module_type_keys_factory(_SUPPORTED_MODULES),
            ),
            value_validator=_validators.optional(
                _validators.instance_of(ModuleMagnitudePrunerConfig)
            ),
            mapping_validator=_validators.instance_of(dict),
        ),
    )
    module_name_configs: _Dict[str, _Optional[ModuleMagnitudePrunerConfig]] = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.instance_of(str),
            value_validator=_validators.optional(
                _validators.instance_of(ModuleMagnitudePrunerConfig)
            ),
            mapping_validator=_validators.instance_of(dict),
        ),
    )

    def __attrs_post_init__(self):
        if (
            self.global_config is None
            and len(self.module_type_configs) == 0
            and len(self.module_name_configs) == 0
        ):
            self.global_config = ModuleMagnitudePrunerConfig()

    @classmethod
    def from_dict(cls, config_dict: _Dict[str, _Any]) -> "MagnitudePrunerConfig":
        super().from_dict(config_dict)
        converter = _cattrs.Converter(forbid_extra_keys=True)
        converter.register_structure_hook(
            _ModuleTypeConfigType,
            _structure_from_dict_hook_factory(ModuleMagnitudePrunerConfig),
        )
        return converter.structure_attrs_fromdict(config_dict, cls)


class _MagnitudePruningMethod(_ScheduledBaseDynamicPruningMethod):
    """
    Magnitude-based static mask pruning method as described in
    `To Prune or Not to Prune <https://arxiv.org/pdf/1710.01878>`_.
    """
    _tensor_name: str
    scheduled: bool = True
    amount: float

    def __init__(
        self,
        amount: float,
        block_size: int,
        granularity: str,
        n_m_ratio: _Optional[_Tuple[int, int]] = None,
        dim: _Optional[int] = None,
        **kwargs: _ParamsDict,
    ):
        super().__init__(scheduled_value=amount, scheduled_value_name="amount")
        self.block_size = block_size
        self.granularity = granularity
        self.n_m_ratio = n_m_ratio
        self.dim = dim

    def compute_mask(self, t: _torch.Tensor, default_mask: _torch.Tensor) -> _torch.Tensor:
        if self.n_m_ratio is not None:
            _, block_size = self.n_m_ratio
            num_zeros = int(self.amount * block_size)
            if num_zeros == 0:
                # when number of zeros is < 0, we increase sparsity gradually
                return _magnitude_ranked_mask(t, self.amount, 1, self.granularity).float()
            else:
                return _n_m_mask(t, (num_zeros, block_size), self.dim).float()
        else:
            return _magnitude_ranked_mask(t, self.amount, self.block_size, self.granularity).float()


@_define
class _MagnitudePrunerInfo:
    config: ModuleMagnitudePrunerConfig
    module: _torch.nn.Module
    sparsity_level: float


class MagnitudePruner(_BasePrunerWithPruningMethod):
    """
    This pruning algorithm was inspired by the paper `"To prune or not to prune"
    <https://arxiv.org/pdf/1710.01878.pdf>`_.

    In order to achieve the desired sparsity, the algorithm sorts a module's weight matrix
    by the magnitude of its elements, and sets all elements less than a threshold to zero.
    Magnitude is computed using L1 norm when granularity is ``per_scalar``, otherwise, L2
    norm is used.

    The pruner can be configured at different granularities such as per scalar, per kernel,
    per channel (output channel), or per layer, to induce varying sparsity structures in the
    weight matrix.

    When the ``block_size`` parameter is provided in the config, zeros are induced in the same locations
    (across all other axes) in ``block_size`` number of consecutive output channels.

    When the ``n_m_ratio`` parameter is provided in the config, out of every ``m`` elements, the smallest ``n``
    are set to zero. The ``m`` element blocks are chosen along the dimension specified by the ``dim`` parameter.

    Example:
            .. code-block:: python

                import torch
                from collections import OrderedDict
                from coremltools.optimize.torch.pruning import MagnitudePruner, MagnitudePrunerConfig

                # define model and loss function
                model = torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("conv1", torch.nn.Conv2d(3, 32, 3, padding="same")),
                            ("conv2", torch.nn.Conv2d(32, 32, 3, padding="same")),
                        ]
                    )
                )
                loss_fn = define_loss()  # define the loss function

                # initialize pruner and configure it
                # we only prune the fisrt conv layer
                config = MagnitudePrunerConfig.from_dict(
                    {
                        "module_name_configs": {
                            "conv1": {
                                "scheduler": {"update_steps": [3, 5, 7]},
                                "target_sparsity": 0.75,
                                "granularity": "per_channel",
                            },
                        }
                    }
                )

                pruner = MagnitudePruner(model, config)

                # insert pruning layers in the model
                model = pruner.prepare()

                for inputs, labels in data:
                    output = model(inputs)
                    loss = loss_fn(output, labels)
                    loss.backward()
                    optimizer.step()
                    pruner.step()

                # commit pruning masks to model parameters
                pruner.finalize(inplace=True)

    Args:
        model (:py:class:`torch.nn.Module`): Model on which the pruner will act.
        config (:py:class:`MagnitudePrunerConfig`): Config which specifies how
            different submodules in the model will be configured for pruning.
            Default config is used when passed as ``None``.
    """
    _supported_modules: _Tuple = _SUPPORTED_MODULES

    def __init__(self, model: _torch.nn.Module, config: _Optional[MagnitudePrunerConfig] = None):
        config = MagnitudePrunerConfig() if config is None else config
        super().__init__(model, config)

    def prepare(self, inplace: bool = False) -> _torch.nn.Module:
        if self._is_prepared:
            _logger.warning(
                "Model has already been prepared for pruning. This API call will be a no-op."
            )
            return self._model
        self._model = super().prepare(inplace=inplace)
        for name, submodule in self._model.named_modules(remove_duplicate=True):
            submod_config = self._config.get_module_config(name, submodule)
            if isinstance(submodule, self._supported_modules) and submod_config is not None:
                submod_config = _copy.deepcopy(submod_config)
                if submod_config.n_m_ratio is not None:
                    num_zeros, block_size = submod_config.n_m_ratio
                    # Add target sparsity to make scheduler work
                    submod_config.target_sparsity = float(num_zeros) / float(block_size)
                _MagnitudePruningMethod.from_module_and_params(
                    submodule,
                    param_name=submod_config.param_name,
                    amount=submod_config.initial_sparsity,
                    block_size=submod_config.block_size,
                    granularity=submod_config.granularity,
                    n_m_ratio=submod_config.n_m_ratio,
                    dim=submod_config.dim,
                )
                self._pruner_info[name] = _MagnitudePrunerInfo(
                    config=submod_config,
                    module=submodule,
                    sparsity_level=submod_config.initial_sparsity,
                )
        return self._model

    def step(self):
        if not self._is_prepared:
            _logger.warning(
                "Model has not been prepared for pruning. This API call "
                "will be a no-op. prepare method must be called before "
                "a call to the step method."
            )
            return
        self._step_count += 1
        for name, pruner_info in self._pruner_info.items():
            if hasattr(pruner_info.module, "pruning_method"):
                sparsity_level = pruner_info.config.scheduler.compute_sparsity(
                    self._step_count,
                    prev_sparsity=pruner_info.sparsity_level,
                    config=pruner_info.config,
                )
                if sparsity_level != pruner_info.sparsity_level:
                    pruner_info.module.pruning_method.update_mask(
                        pruner_info.module, sparsity_level
                    )
                pruner_info.sparsity_level = sparsity_level
