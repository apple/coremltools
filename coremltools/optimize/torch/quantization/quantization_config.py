#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
from collections import OrderedDict as _OrderedDict
from enum import Enum as _Enum
from enum import unique as _unique
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import List as _List
from typing import NewType as _NewType
from typing import Optional as _Optional
from typing import Union as _Union

import cattrs as _cattrs
import torch as _torch
import torch.ao.quantization as _aoquant
from attr import define as _define
from attr import field as _field
from attrs import validators as _validators

from coremltools.optimize.torch._utils.torch_utils import (
    maybe_convert_str_to_dtype as _maybe_convert_str_to_dtype,
)
from coremltools.optimize.torch._utils.torch_utils import (
    maybe_convert_str_to_mod_type as _maybe_convert_str_to_mod_type,
)
from coremltools.optimize.torch.optimization_config import (
    ModuleOptimizationConfig as _ModuleOptimizationConfig,
)
from coremltools.optimize.torch.optimization_config import OptimizationConfig as _OptimizationConfig
from coremltools.optimize.torch.optimization_config import _structure_from_dict_hook_factory

_logger = _logging.getLogger(__name__)


@_unique
class ObserverType(_Enum):
    """
    An enum indicating the type of observer. Allowed options are moving_average_min_max and mix_max.
    """
    moving_average_min_max = "moving_average_min_max"
    mix_max = "min_max"

    @staticmethod
    def get_observer(observer_type: "ObserverType", is_per_channel: bool) -> _Any:
        _str_to_observer_map = {
            "moving_average_min_max": _aoquant.MovingAverageMinMaxObserver,
            "min_max": _aoquant.MinMaxObserver,
            "moving_average_min_max_per_channel": _aoquant.MovingAveragePerChannelMinMaxObserver,
            "min_max_per_channel": _aoquant.PerChannelMinMaxObserver,
        }
        observer_name = observer_type.value
        if is_per_channel:
            observer_name = f"{observer_name}_per_channel"
        return _str_to_observer_map[observer_name]


@_unique
class QuantizationScheme(_Enum):
    """
    An enum indicating the type of quantization to be performed. Allowed options are symmetric
    and affine.
    """

    symmetric = "symmetric"
    affine = "affine"

    @staticmethod
    def get_qscheme(
        quantizaton_scheme: "QuantizationScheme", is_per_channel: bool
    ) -> _torch.qscheme:
        _str_to_qscheme_map = {
            "symmetric": _torch.per_tensor_symmetric,
            "affine": _torch.per_tensor_affine,
            "symmetric_per_channel": _torch.per_channel_symmetric,
            "affine_per_channel": _torch.per_channel_affine,
        }
        quantization_scheme_name = quantizaton_scheme.value
        if is_per_channel:
            quantization_scheme_name = f"{quantization_scheme_name}_per_channel"
        return _str_to_qscheme_map[quantization_scheme_name]


_default_quantization_options = {
    "weight_dtype": _torch.qint8,
    "weight_per_channel": True,
    "activation_dtype": _torch.quint8,
    "observer": ObserverType.moving_average_min_max,
    "quantization_scheme": QuantizationScheme.symmetric,
}


@_define
class ModuleLinearQuantizerConfig(_ModuleOptimizationConfig):
    """
    Module level configuration for :py:class:`LinearQuantizer`.

    Args:
        weight_dtype (:py:class:`torch.dtype`): The dtype to use for quantizing the weights. Defaults to
            :py:class:`torch.qint8`.
        weight_observer (:py:class:`ObserverType`): Type of observer to use for quantizing weights. Defaults
            to ``moving_average_min_max``.
        weight_per_channel (:obj:`bool`): When ``True``, weights are quantized per channel; otherwise, per tensor.
        activation_dtype (:py:class:`torch.dtype`): The dtype to use for quantizing the activations. Defaults to
            :py:class:`torch.qint8`.
        activation_observer (:py:class:`ObserverType`): Type of observer to use for quantizing activations. Defaults
            to ``moving_average_min_max``.
        quantization_scheme: (:py:class:`QuantizationScheme`): Type of quantization configuration to use. When
            this parameter is set to :py:class:`QuantizationScheme.symmetric`, all weights are
            quantized with zero point as zero, and all activations are quantized with zero point as zero for
            non-negative activations and 128 for all other activations. When it is set to
            :py:class:`QuantizationScheme.affine`, zero point can be set anywhere in the range of values allowed
            for the quantized weight/activation.
        milestones (:obj:`list` of :obj:`int`): A list of four integers indicating milestones to use during
            quantization. The first milestone corresponds to enabling observers, the second to enabling fake
            quantization simulation, the third to disabling observers, and the last
            to freezing batch norm statistics.
    """

    weight_dtype: _torch.dtype = _field(
        default=_default_quantization_options["weight_dtype"],
        converter=_maybe_convert_str_to_dtype,
        validator=[
            _validators.instance_of(_torch.dtype),
            _validators.in_([_torch.qint8, _torch.quint8, _torch.float32]),
        ],
    )
    weight_observer: ObserverType = _field(
        default=_default_quantization_options["observer"],
        converter=ObserverType,
        validator=_validators.in_(ObserverType),
    )
    weight_per_channel: bool = _field(
        default=_default_quantization_options["weight_per_channel"],
        validator=_validators.instance_of(bool),
    )
    activation_dtype: _torch.dtype = _field(
        default=_default_quantization_options["activation_dtype"],
        converter=_maybe_convert_str_to_dtype,
        validator=[
            _validators.instance_of(_torch.dtype),
            _validators.in_([_torch.quint8, _torch.float32]),
        ],
    )
    activation_observer: ObserverType = _field(
        default=_default_quantization_options["observer"],
        converter=ObserverType,
        validator=_validators.in_(ObserverType),
    )
    quantization_scheme: QuantizationScheme = _field(
        default=_default_quantization_options["quantization_scheme"],
        converter=QuantizationScheme,
        validator=_validators.in_(QuantizationScheme),
    )
    milestones: _Optional[_List[int]] = _field(
        default=None,
        validator=_validators.optional(
            _validators.deep_iterable(
                member_validator=_validators.instance_of(int),
                iterable_validator=_validators.instance_of(list),
            )
        ),
    )

    def __attrs_post_init__(self):
        if self.weight_dtype == _torch.float32 and self.activation_dtype != _torch.float32:
            raise ValueError(
                f"Unsupported configuration: weight_dtype = {self.weight_dtype}, "
                f"activation_dtype = {self.activation_dtype}. When weights are not quantized,"
                f"activations cannot be quantized."
            )

    @milestones.validator
    def _check_milestones(self, attribute, value):
        if value is not None:
            assert len(value) == 4, (
                f"Received milestones = {value}. "
                f"Milestones should be of length 4. "
                f"Refer to docs for more information."
            )


_ModuleTypeConfigType = _NewType(
    "ModuleTypeConfigType",
    _Dict[_Union[_Callable, str], _Optional[ModuleLinearQuantizerConfig]],
)


@_define
class LinearQuantizerConfig(_OptimizationConfig):
    """
    Configuration for :py:class:`LinearQuantizer`.

    Args:
        global_config (:py:class:`ModuleLinearQuantizerConfig`): Config to be applied globally
            to all supported modules. Missing values are chosen from the default config.
        module_type_configs (:obj:`dict` of :obj:`str` to :py:class:`ModuleLinearQuantizerConfig`):
            Module type level configs applied to a specific
            module class, such as :py:class:`torch.nn.Linear`. The keys can be either strings
            or module classes.
        module_name_configs (:obj:`dict` of :obj:`str` to :py:class:`ModuleLinearQuantizerConfig`):
            Module level configs applied to specific modules.
            The name of the module must be a fully qualified name that can be used to fetch it
            from the top level module using the ``module.get_submodule(target)`` method.
        non_traceable_module_names (:obj:`list` of :obj:`str`):
            Names of modules which cannot be traced using ``torch.fx``.

    .. note::
        The ``quantization_scheme`` parameter must be the same across all configs.

    """

    global_config: _Optional[ModuleLinearQuantizerConfig] = _field(
        default=None,
        validator=_validators.optional(_validators.instance_of(ModuleLinearQuantizerConfig)),
    )
    module_type_configs: _ModuleTypeConfigType = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.instance_of((str, _Callable)),
            value_validator=_validators.optional(
                _validators.instance_of(ModuleLinearQuantizerConfig)
            ),
            mapping_validator=_validators.instance_of(dict),
        ),
    )
    module_name_configs: _Dict[str, _Optional[ModuleLinearQuantizerConfig]] = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.instance_of(str),
            value_validator=_validators.optional(
                _validators.instance_of(ModuleLinearQuantizerConfig)
            ),
            mapping_validator=_validators.instance_of(dict),
        ),
    )
    non_traceable_module_names: _List[str] = _field(
        default=list(),
        validator=_validators.deep_iterable(
            member_validator=_validators.instance_of(str),
        ),
    )

    def __attrs_post_init__(self):
        if (
            self.global_config is None
            and len(self.module_type_configs) == 0
            and len(self.module_name_configs) == 0
        ):
            self.global_config = ModuleLinearQuantizerConfig()
        self.module_type_configs = {
            _maybe_convert_str_to_mod_type(key): val
            for key, val in self.module_type_configs.items()
        }
        self._validate_same_params(["quantization_scheme"])

    @classmethod
    def from_dict(cls, config_dict: _Dict[str, _Any]) -> "LinearQuantizerConfig":
        super().from_dict(config_dict)
        converter = _cattrs.Converter(forbid_extra_keys=True)
        converter.register_structure_hook(
            _ModuleTypeConfigType,
            _structure_from_dict_hook_factory(ModuleLinearQuantizerConfig),
        )
        return converter.structure_attrs_fromdict(config_dict, cls)
