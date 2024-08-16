#  Copyright (c) 2024, Apple Inc. All rights reserved.
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
    get_n_bits_from_dtype as _get_n_bits_from_dtype,
)
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
    An enum indicating the type of observer.
    Allowed options are moving_average_min_max, min_max, ema_min_max, ema_percentile, mse, ema_mse, lsq and lsq_plus.
    """

    moving_average_min_max = "moving_average_min_max"
    min_max = "min_max"

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
    "weight_ch_axis": 0,
    "activation_dtype": _torch.quint8,
    "observer": ObserverType.moving_average_min_max,
    "quantization_scheme": QuantizationScheme.symmetric,
}


# Backends only support 4 and 8 bit quantization
_SUPPORTED_N_BITS = [4, 8, 32]


@_define
class ModuleLinearQuantizerConfig(_ModuleOptimizationConfig):
    """
    Configuration class for specifying global and module level quantization options for linear quantization
    algorithm implemented in :py:class:`LinearQuantizer`.

    Linear quantization algorithm simulates the effects of quantization during training, by quantizing
    and dequantizing the weights and/or activations during the model's forward pass. The forward and
    backward pass computations are conducted in ``float`` dtype, however, these ``float`` values follow
    the constraints imposed by ``int8`` and ``quint8`` dtypes. For more details, please refer to
    `Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
    <https://arxiv.org/pdf/1712.05877.pdf>`_.

    For most applications, the only parameters that need to be set are ``quantization_scheme`` and
    ``milestones``.

    By default, ``quantization_scheme`` is set to :py:class:`QuantizationScheme.symmetric`, which means
    all weights are quantized with zero point as zero, and activations are quantized with zero point as zero for
    non-negative activations and 128 for all other activations. The weights are quantized using :py:class:`torch.qint8`
    and activations are quantized using :py:class:`torch.quint8`.

    Linear quantization algorithm inserts ``observers`` for each weight/activation tensor.
    These observers collect statistics of these tensors' values, for example, the minimum and maximum values they can
    take. These statistics are then used to compute the scale and zero point, which are in turn used for quantizing
    the weights/activations. By default, ``moving_average_min_max`` observer is used. For more details, please
    check `MinMaxObserver <https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MinMaxObserver.html#torch.ao.quantization.observer.MinMaxObserver>`_.

    The ``milestones`` parameter controls the flow of the quantization algorithm.  The example below illustrates its
    usage in more detail:

    .. code-block:: python

            model = define_model()

            config = LinearQuantizerConfig(
                global_config=ModuleLinearQuantizerConfig(
                    quantization_scheme="symmetric",
                    milestones=[0, 100, 300, 200],
                )
            )

            quantizer = LinearQuantizer(model, config)

            # prepare the model to insert FakeQuantize layers for QAT
            model = quantizer.prepare()

            # use quantizer in your PyTorch training loop
            for inputs, labels in data:
                output = model(inputs)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                quantizer.step()

            # In this example, from step 0 onwards, observers will collect statistics
            # of the values of weights/activations. However, between steps 0 and 100,
            # effects of quantization will not be simulated. At step 100, quantization
            # simulation will begin and at step 300, observer statistics collection will
            # stop. A batch norm layer computes mean and variance of input batch for normalizing
            # it during training, and collects running estimates of its computed mean and variance,
            # which are then used for normalization during evaluation. At step 200, batch norm
            # statistics collection is frozen, and the batch norm layers switch to evaluation
            # mode, thus more closely simulating the inference numerics during training time.

    Args:
        weight_dtype (:py:class:`torch.dtype`): The dtype to use for quantizing the weights. The number of bits used
            for quantization is inferred from the dtype. When dtype is set to :py:class:`torch.float32`, the weights
            corresponding to that layer are not quantized.  Defaults to :py:class:`torch.int8` which corresponds to
            8-bit quantization.
        weight_observer (:py:class:`ObserverType`): Type of observer to use for quantizing weights.
            Defaults to ``moving_average_min_max``.
        weight_per_channel (:obj:`bool`): When ``True``, weights are quantized per channel; otherwise, per tensor.
        activation_dtype (:py:class:`torch.dtype`): The dtype to use for quantizing the activations. When dtype
            is set to :py:class:`torch.float32`, the activations corresponding to that layer are not quantized.
            Defaults to :py:class:`torch.quint8`.
        activation_observer (:py:class:`ObserverType`): Type of observer to use for quantizing activations.
            Defaults to ``moving_average_min_max``.
        quantization_scheme: (:py:class:`QuantizationScheme`): Type of quantization configuration to use. When
            this parameter is set to :py:class:`QuantizationScheme.symmetric`, all weights are
            quantized with zero point as zero, and activations are quantized with zero point as zero for
            non-negative activations and 128 for all other activations. When it is set to
            :py:class:`QuantizationScheme.affine`, zero point can be set anywhere in the range of values allowed
            for the quantized weight/activation. Defaults to :py:class:`QuantizationScheme.symmetric`.
        milestones (:obj:`list` of :obj:`int`): A list of four integers indicating milestones to use during
            quantization. The first milestone corresponds to enabling observers, the second to enabling fake
            quantization simulation, the third to disabling observers, and the last to freezing batch norm statistics.
            Defaults to ``None``, which means the ``step`` method of :py:class:`LinearQuantizer` will be a no-op and
            all observers and quantization simulation will be turned on from the first step, batch norm layers always
            operate in training mode, and mean and variance statistics collection is not frozen.
    """

    weight_dtype: _Union[str, _torch.dtype] = _field(
        default=_default_quantization_options["weight_dtype"],
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
        self.weight_n_bits = _get_n_bits_from_dtype(self.weight_dtype)
        self.weight_dtype = _maybe_convert_str_to_dtype(self.weight_dtype)
        if self.weight_dtype not in [_torch.qint8, _torch.quint8, _torch.float32]:
            raise ValueError(
                f"weight_dtype must be one of (_torch.qint8, _torch.quint8, _torch.float32) not {self.weight_dtype}"
            )

    @milestones.validator
    def _check_milestones(self, attribute, value):
        if value is not None:
            assert len(value) == 4, (
                f"Received milestones = {value}. "
                f"Milestones should be of length 4. "
                f"Refer to docs for more information."
            )

    @classmethod
    def from_dict(cls, config_dict):
        converter = _cattrs.Converter(forbid_extra_keys=True)
        converter.register_structure_hook(
            _Union[str, _torch.dtype],
            lambda obj, type: obj,
        )
        return converter.structure_attrs_fromdict(config_dict, cls)


_ModuleTypeConfigType = _NewType(
    "ModuleTypeConfigType",
    _Dict[_Union[_Callable, str], _Optional[ModuleLinearQuantizerConfig]],
)


@_define
class LinearQuantizerConfig(_OptimizationConfig):
    """
    Configuration class for specifying how different submodules of a model are
    quantized by :py:class:`LinearQuantizer`.

    In order to disable quantizing a layer or an operation, ``module_type_config`` or
    ``module_name_config`` corresponding to that operation can be set to ``None``.

    For example:

    .. code-block:: python

            # The following config will enable weight only quantization for all layers:
            config = LinearQuantizerConfig.from_dict(
                {
                    "global_config": {
                        "activation_dtype": "float32",
                    }
                }
            )

            # The following config will disable quantization for all linear layers and
            # set quantization mode to weight only quantization for convolution layers:
            config = LinearQuantizerConfig.from_dict(
                {
                    "module_type_configs": {
                        "Linear": None,
                        "Conv2d": {
                            "activation_dtype": "float32",
                        },
                    }
                }
            )

            # The following config will disable quantization for layers named conv1 and conv2:
            config = LinearQuantizerConfig.from_dict(
                {
                    "module_name_configs": {
                        "conv1": None,
                        "conv2": None,
                    }
                }
            )

            # If model has some methods and attributes which are not used in the forward
            # pass, but are needed to be preserved after quantization is added, they can
            # be preserved on the quantized model by passing them in preserved_attributes
            # parameter

            model = MyModel()
            model.key_1 = value_1
            model.key_2 = value_2

            config = LinearQuantizerConfig.from_dict({"preserved_attributes": ["key_1", "key_2"]})

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
        preserved_attributes (:obj:`list` of :obj:`str`): Names of attributes of the model
            which should be preserved on the prepared and finalized models, even if they are not
            used in the model's forward pass.

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
    preserved_attributes: _List[str] = _field(
        factory=list,
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
            _Union[str, _torch.dtype],
            lambda obj, type: obj,
        )
        converter.register_structure_hook(
            _ModuleTypeConfigType,
            _structure_from_dict_hook_factory(ModuleLinearQuantizerConfig),
        )
        return converter.structure_attrs_fromdict(config_dict, cls)
