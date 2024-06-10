#  Copyright (c) 2024, Apple Inc. All rights reserved.
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
    Configuration class for specifying global and module level pruning options for magnitude pruning
    algorithm implemented in :py:class:`MagnitudePruner`.

    This class supports four different modes of sparsity:

    1. **Unstructured sparsity**: This is the default sparsity mode used by :py:class:`MagnitudePruner`.
    It is activated when ``block_size = 1``, ``n_m_ratio = None`` and ``granularity = per_scalar``.
    In this mode, the ``n`` weights with the lowest absolute values are set to 0,
    where ``n = floor(size_of_weight_tensor * target_sparsity)``.
    For example, given the following:

       * ``weight = [0.3, -0.2, -0.01, 0.05]``
       * ``target_sparsity = 0.75``

    The pruned weight would be ``[0.3, 0, 0, 0]``

    2.  **Block structured sparsity**: This mode is activated when ``block_size > 1`` and ``n_m_ratio = None``.
    In this mode, the weight matrix is first reshaped to a rank 2 matrix by folding all dimensions ``>= 1``
    into a single dimension. Then, blocks of size ``block_size`` along the ``0-th`` dimension,
    which have the lowest ``L2`` norm, are set to 0. The number of blocks which are zeroed out is
    determined by the ``target_sparsity`` parameter. The blocks are chosen in a non-overlapping fashion.

    For example:

        .. code-block:: python

            # Given a 4 x 2 weight with the following value, and block_size = 2.
            [
                [1, 3],
                [-6, -7],
                [0, 3],
                [-9, 2],
            ]

            # L2 norm  is computed along the 0-th dimension for blocks of size 2:
            [
                [6.08, 7.62],
                [9.00, 3.61],
            ]

            # Then the smallest values are picked to prune. So if target_sparsity = 0.5,
            # then the blocks that will be pruned will be with ones with L2 norm values
            # of 6.08 and 3.61. And hence, the elements in the first and third
            # block are pruned. The final pruned tensor is:
            [
                [0, 3],
                [0, -7],
                [0, 0],
                [-9, 0],
            ]

    3. **n:m structured sparsity**: This mode is activated when ``n_m_ratio != None``. Similar to
    block structured sparsity, in this mode, the weight matrix is reshaped to a rank 2 matrix.
    Then, out of non-overlapping blocks of size ``m`` along the ``0-th`` or ``1-st`` dimension, the ``n``
    elements with the smallest absolute value are set to 0. The dimension along which the blocks
    are chosen is controlled by the ``dim`` parameter and it defaults to ``1``. For linear layers,
    ``dim = 1`` and ratios where ``m`` is a factor of 16 (e.g. ``3:4``, ``7:8`` etc.) are recommended
    to get latency gains for models executing specifically on the CPU.

    For example:

        .. code-block:: python

            # Given a 4 x 4 weight of
            [
                [3, 4, 7, 6],
                [1, 8, -3, -8],
                [-2, -3, -4, 0],
                [5, 4, -3, -2],
            ]

            # For n_m_ratio = (1, 2) with dim = 1 (default), the resulting pruned weight is
            [
                [0, 4, 7, 0],
                [0, 8, 0, -8],
                [0, -3, -4, 0],
                [5, 0, -3, 0],
            ]

    4. **General structured sparsity**: This mode is activated when ``granularity`` is set to
    one of ``per_channel`` or ``per_kernel``. It only applies to weights of ``rank >= 3``.
    For example, a rank 4 weight matrix of shape ``[C_o x C_i x H x W]`` can be thought
    of as ``C_o`` matrices of shape ``[C_i x H X W]`` or ``C_o*C_i`` matrices of size ``[H x W]``.
    ``per_channel`` granularity sets some of the ``[C_i x H X W]`` matrices to 0 whereas
    ``per_kernel`` granularity sets some of the ``[H x W]`` matrices to 0.

    When granularity is ``per_channel``, the weight matrix is reshaped to a rank 2 matrix,
    where all dimensions ``>= 1`` are folded into a single dimension. Then ``L2`` norm is
    computed for all rows and the weights corresponding to ``n`` smallest ``L2`` norm rows
    are set to 0 to achieve ``target_sparsity``.

    For example:

    .. code-block:: python

            # Given a 2 x 2 x 1 x 2 weight, granularity = per_channel,
            [
                [
                    [[2, -1]],
                    [[-3, 2]],
                ],
                [
                    [[5, -2]],
                    [[-1, -3]],
                ],
            ]

            # It is first reshaped to shape 2 x 4, i.e.:
            [
                [2, -1, -3, 2],
                [5, -2, -1, -3],
            ]

            # Then L2 norm is computed for each row of the matrix:
            [4.2426, 6.2450]

            # Finally, to achieve target sparsity = 0.5, since the first element is
            # smaller, the corresponding row is set to 0, resulting in the pruned weight:
            [
                [
                    [[0, 0]],
                    [[0, 0]],
                ],
                [
                    [[5, -2]],
                    [[-1, -3]],
                ],
            ]

    When granularity is ``per_kernel``, the weight matrix is reshaped to a rank 3 matrix,
    where all dimensions ``>= 2`` are folded into a single dimension. Then ``L2`` norm is
    computed for all vectors along the last dimension, ``dim = 2`` and the weights corresponding
    to the ``n`` smallest ``L2`` norm vectors are set to 0 to achieve ``target_sparsity``.

    For the same example as before, setting granularity ``per_kernel`` will achieve:

    .. code-block:: python

            # The original 2 x 2 x 1 x 2 weight matrix is reshaped into shape 2 x 2 x 2, i.e.:
            [
                [[2, -1], [-3, 2]],
                [[5, -2], [-1, -3]],
            ]

            # Then L2 norm is computed for each of the 4 vectors of size 2, [2, -1], [-3, 2], etc.:
            [
                [2.2361, 3.6056],
                [5.3852, 3.1623],
            ]

            # Finally, to achieve target sparsity = 0.5, since the first and last elements are
            # smallest, the corresponding row in the weights is set to 0,
            # resulting in the pruned weight:
            [
                [
                    [[0, 0]],
                    [[-3, 2]],
                ],
                [
                    [[5, -2]],
                    [[0, 0]],
                ],
            ]


    Args:
        scheduler (:py:class:`PruningScheduler`): A pruning scheduler which specifies how the
            sparsity should be changed over the course of the training. Defaults to constant
            sparsity scheduler which sets the  sparsity to ``target_sparsity`` at step ``0``.
        initial_sparsity (:obj:`float`): Desired fraction of zeroes at the beginning of the
            training process. Defaults to ``0.0``.
        target_sparsity (:obj:`float`): Desired fraction of zeroes at the end of the
            training process. Defaults to ``0.5``.
        granularity (:obj:`str`): Specifies the granularity at which the pruning mask will be
            computed. Can be one of ``per_channel``, ``per_kernel`` or ``per_scalar``.
            Defaults to ``per_scalar``.
        block_size (:obj:`int`): Block size for inducing block sparsity within the mask. This
            is applied on the output channel dimension of the parameter (the ``0`` -th dimension).
            Having larger block size may be beneficial for latency compared to smaller block sizes,
            for models running on certain compute units such as the neural engine.
            ``block_size`` must be greater than ``1`` to enable block sparsity, and must be at most half
            the number of output channels. When the number of output channels is not divisible by the block size,
            the weight matrix is padded with zeros to compute the pruning mask and then un-padded to the original size.
            Defaults to ``1``.
        n_m_ratio (:obj:`tuple` of :obj:`int`): A tuple of two integers which specify how ``n:m`` pruning should be
            applied. In ``n:m`` pruning, out of every ``m`` elements,
            ``n`` with lowest magnitude are set to zero. When ``n_m_ratio`` is not ``None``, ``block_size``,
            ``granularity``, and ``initial_sparsity`` should be ``1``, ``per_scalar``, and ``0.0`` respectively.
            The value of ``target_sparsity`` is ignored and the actual target sparsity is determined by the
            ``n:m`` ratio. For more information, see `Learning N:M Fine-Grained Structured Sparse Neural Networks From Scratch
            <https://arxiv.org/abs/2102.04010>`_. Defaults to ``None``, which means ``n:m`` sparsity is not used.
        dim (:obj:`int`): Dimension along which blocks of ``m`` elements are chosen when applying ``n:m`` sparsity. This
            parameter is only used when ``n_m_ratio`` is not ``None``. Defaults to ``1``.
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


_ModuleTypeConfigType = _NewType(
    "ModuleTypeConfigType",
    _Dict[_Union[_Callable, str], _Optional[ModuleMagnitudePrunerConfig]],
)


@_define
class MagnitudePrunerConfig(_OptimizationConfig):
    """
    Configuration class for specifying how different submodules in a model are pruned by :py:class:`MagnitudePruner`.

    Args:
        global_config (:py:class:`ModuleMagnitudePrunerConfig`): Config to be applied globally
            to all supported modules. Missing values are chosen from the default config.
        module_type_configs (:obj:`dict` of :obj:`str` to :py:class:`ModuleMagnitudePrunerConfig`): Module
            type level configs applied to a specific module class, such as :py:class:`torch.nn.Linear`.
            The keys can be either strings or module classes. If ``module_type_config`` is set to ``None``
            for a module type, it wouldn't get pruned.
        module_name_configs (:obj:`dict` of :obj:`str` to :py:class:`ModuleMagnitudePrunerConfig`): Module level
            configs applied to specific modules. The name of the module must be a fully qualified name that can
            be used to fetch it from the top level module using the ``module.get_submodule(target)`` method. If
            ``module_name_config`` is set to ``None`` for a module, it wouldn't get pruned.
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
    `To prune, or not to prune: exploring the efficacy of  pruning for model
    compression <https://arxiv.org/pdf/1710.01878.pdf>`_
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
    A pruning algorithm based on `To prune, or not to prune: exploring the efficacy of
    pruning for model compression <https://arxiv.org/pdf/1710.01878.pdf>`_. It extends the idea in the paper
    to different kinds of structured sparsity modes, in addition to unstructured sparsity. In order to
    achieve the desired sparsity, this algorithm sorts a module's weight matrix by the magnitude of
    its elements, and sets all elements less than a threshold to zero.

    Four different modes of sparsity are supported, encompassing both structured and unstructured
    sparsity. For details on how to select these different sparsity modes, please see
    :py:class:`ModuleMagnitudePrunerConfig`.

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
                "Model has already been prepared for pruning. This API call "
                "will be a no-op."
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
