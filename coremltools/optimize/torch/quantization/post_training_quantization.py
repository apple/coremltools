#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
from collections import OrderedDict as _OrderedDict
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import List as _List
from typing import NewType as _NewType
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Type as _Type
from typing import Union as _Union

import cattrs as _cattrs
import numpy as _np
import torch as _torch
import torch.nn as _nn
from attr import define as _define
from attr import field as _field
from attrs import validators as _validators

from coremltools.converters.mil.mil.ops.defs.iOS18 import constexpr_blockwise_shift_scale
from coremltools.optimize.coreml._utils import compute_qparams as _ct_compute_qparams
from coremltools.optimize.torch._utils.metadata_utils import (
    CompressionMetadata as _CompressionMetadata,
)
from coremltools.optimize.torch._utils.report_utils import (
    compute_post_training_report as _compute_post_training_report,
)
from coremltools.optimize.torch._utils.torch_utils import get_atomic_layers as _get_atomic_layers
from coremltools.optimize.torch._utils.torch_utils import (
    get_n_bits_from_dtype as _get_n_bits_from_dtype,
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
    BasePostTrainingModelOptimizer as _BasePostTrainingModelOptimizer,
)
from coremltools.optimize.torch.base_model_optimizer import _Report
from coremltools.optimize.torch.optimization_config import (
    ModuleOptimizationConfig as _ModuleOptimizationConfig,
)
from coremltools.optimize.torch.optimization_config import OptimizationConfig as _OptimizationConfig
from coremltools.optimize.torch.optimization_config import (
    QuantizationGranularity,
    _structure_from_dict_hook_factory,
)
from coremltools.optimize.torch.quantization import QuantizationScheme as _QuantizationScheme

_default_ptq_options = {
    "weight_dtype": "int8",
    "granularity": "per_channel",
    "quantization_scheme": _QuantizationScheme.symmetric,
    "block_size": None,
}

_logger = _logging.getLogger(__name__)


@_define
class ModulePostTrainingQuantizerConfig(_ModuleOptimizationConfig):
    """
    Configuration class for specifying global and module level quantizer options for
    :py:class:`PostTrainingQuantizer` algorithm.

    Args:
        weight_dtype (:py:class:`torch.dtype`): The dtype to use for quantizing the weights. The number of bits used
            for quantization is inferred from the dtype. When dtype is set to :py:class:`torch.float32`, the weights
            corresponding to that layer are not quantized. Defaults to :py:class:`torch.int8` which corresponds to
            8-bit quantization.
        granularity (:py:class:`QuantizationGranularity`): Specifies the granularity at which quantization parameters
            will be computed. Can be one of ``per_channel``, ``per_tensor`` or ``per_block``. When using ``per_block``,
            ``block_size`` argument must be specified. Defaults to ``per_channel``.
        quantization_scheme (:py:class:`~.coremltools.optimize.torch.quantization.quantization_config.QuantizationScheme`): Type of
            quantization configuration to use. When this parameter is set to ``QuantizationScheme.symmetric``,
            all weights are quantized with zero point as zero. When it is set to ``QuantizationScheme.affine``,
            zero point can be set anywhere in the range of values allowed for the quantized weight.
            Defaults to ``QuantizationScheme.symmetric``.
        block_size (:obj:`tuple` of :obj:`int` or :obj:`int`): When ``block_size`` is specified, ``block_size``
            number of values will share the same quantization parameters of scale (and zero point if applicable) across
            the input channel axis. A tuple of integers can be provided for arbitrary sized blockwise quantization.
            See below for more details on different possible configurations. Defaults to ``None``.

    This class supports three different configurations to structure the quantization:

    1. **Per-channel quantization**: This is the default configuration where ``granularity`` is ``per_channel`` and
    ``block_size`` is ``None``. In this configuration, quantization parameters are computed for each output channel.

    2. **Per-tensor quantization**: In this configuration, quantization parameters are computed for the tensor as a whole. That is,
    all values in the tensor will share a single scale (and a single zero point if applicable). The ``granularity`` argument is set
    to ``per_tensor``.

    3. **Per-block quantization**: This configuration is used to structure the tensor for blockwise quantization. The ``granularity`` 
    is set to ``per_block`` and the ``block_size`` argument has to be specified. The ``block_size`` argument can either be of type
    ``int`` or ``tuple``:
        * int: In this case, each row along the output-channel axis will have its own quantization parameters (similar to ``per_channel``).
               Additionally, ``block_size`` number of values will share the same quantization parameters, along the input channel axis.
               For example, for a weight matrix of shape ``(10, 10)``, if we provide ``block_size = 2``, the shape of the quantization
               parameters would be ``(10, 5)``.
        * tuple: For more advanced configuration, users can provide an arbitrary n-dimensional block to share the quantization parameters.
                 This is specified in the form of a tuple where each value corresponds to the block size for the respective axis of the
                 weight matrix. The length of the provided tuple should be at most the number of dimensions of the weight matrix.

    .. note:
        When performing 4-bit quantization, ``weight_dtype`` is set to :py:class:`torch.int8` for ``int4`` or
        :py:class:`torch.uint8` for ``uint4``. This is because PyTorch currently doesn't provide support for 4-bit
        data types. However, the quantization range is set according to 4-bit quantization and based on
        whether the ``weight_dtype`` is signed or unsigned.
    """
    weight_dtype: _Union[str, _torch.dtype] = _field(
        default=_default_ptq_options["weight_dtype"],
        converter=_maybe_convert_str_to_dtype,
        validator=[
            _validators.instance_of(_torch.dtype),
            _validators.in_([_torch.int8, _torch.uint8, _torch.float32]),
        ],
    )
    granularity: QuantizationGranularity = _field(
        default=_default_ptq_options["granularity"],
        converter=QuantizationGranularity,
        validator=_validators.in_(QuantizationGranularity),
    )
    quantization_scheme: _QuantizationScheme = _field(
        default=_default_ptq_options["quantization_scheme"],
        converter=_QuantizationScheme,
        validator=_validators.in_(_QuantizationScheme),
    )
    block_size: _Optional[_Union[int, _Tuple[int]]] = _field(
        default=_default_ptq_options["block_size"],
        converter=lambda val: (val,) if type(val) is int else val,
        validator=_validators.optional(
            _validators.deep_iterable(
                member_validator=_validators.instance_of(int),
                iterable_validator=_validators.instance_of(tuple),
            )
        ),
    )

    def __attrs_post_init__(self):
        self.weight_n_bits = _get_n_bits_from_dtype(self.weight_dtype)

    @block_size.validator
    def per_block_granularity(self, attribute, value):
        if self.granularity == QuantizationGranularity.per_block:
            assert (
                value is not None
            ), "block_size has to be specified along with per_block granularity."
        else:
            assert (
                value is None
            ), "block_size can't be specified along with per_tensor or per_channel granularity."

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
    _Dict[_Union[_Callable, str], _Optional[ModulePostTrainingQuantizerConfig]],
)


@_define
class PostTrainingQuantizerConfig(_OptimizationConfig):
    """
    Configuration class for specifying how different submodules of a model
    should be post-training quantized by :py:class:`PostTrainingQuantizer`.

    Args:
        global_config (:py:class:`ModulePostTrainingQuantizerConfig`): Config to be applied globally
            to all supported modules.
        module_type_configs (:obj:`dict` of :obj:`str` to :py:class:`ModulePostTrainingQuantizerConfig`):
            Module type configs applied to a specific module class, such as :py:class:`torch.nn.Linear`.
            The keys can be either strings or module classes.
        module_name_configs (:obj:`dict` of :obj:`str` to :py:class:`ModulePostTrainingQuantizerConfig`):
            Module name configs applied to specific modules. This can be a dictionary with module names pointing to their
            corresponding :py:class:`ModulePostTrainingQuantizerConfig`.
    """

    global_config: _Optional[ModulePostTrainingQuantizerConfig] = _field(
        default=None,
        validator=_validators.optional(_validators.instance_of(ModulePostTrainingQuantizerConfig)),
    )
    module_type_configs: _ModuleTypeConfigType = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.instance_of((str, _Callable)),
            value_validator=_validators.optional(
                _validators.instance_of(ModulePostTrainingQuantizerConfig)
            ),
            mapping_validator=_validators.instance_of(dict),
        ),
    )
    module_name_configs: _Dict[str, _Optional[ModulePostTrainingQuantizerConfig]] = _field(
        factory=_OrderedDict,
        validator=_validators.deep_mapping(
            key_validator=_validators.instance_of(str),
            value_validator=_validators.optional(
                _validators.instance_of(ModulePostTrainingQuantizerConfig)
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
            self.global_config = ModulePostTrainingQuantizerConfig()
        self.module_type_configs = {
            _maybe_convert_str_to_mod_type(key): val
            for key, val in self.module_type_configs.items()
        }
        self._validate_same_params(["quantization_scheme"])

    @classmethod
    def from_dict(cls, config_dict: _Dict[str, _Any]) -> "PostTrainingQuantizerConfig":
        super().from_dict(config_dict)
        converter = _cattrs.Converter(forbid_extra_keys=True)
        converter.register_structure_hook(
            _Union[str, _torch.dtype],
            lambda obj, type: obj,
        )
        converter.register_structure_hook(
            _ModuleTypeConfigType,
            _structure_from_dict_hook_factory(ModulePostTrainingQuantizerConfig),
        )
        return converter.structure_attrs_fromdict(config_dict, cls)


class PostTrainingQuantizer(_BasePostTrainingModelOptimizer):
    """
    Perform post-training quantization on a torch model. After quantization, weights of all
    submodules selected for quantization contain full precision values obtained by quantizing
    and dequantizing the original weights which captures the error induced by quantization.

    .. note::
        After quantization, the weight values stored will still remain in full precision, therefore
        the PyTorch model size will not be reduced. To see the reduction in model size, please convert
        the model using ``coremltools.convert(...)``, which will produce a model intermediate language 
        (MIL) model containing the compressed weights.

        Example:

            .. code-block:: python

                import torch.nn as nn
                from coremltools.optimize.torch.quantization import (
                    PostTrainingQuantizerConfig,
                    PostTrainingQuantizer,
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

                # initialize the quantizer
                config = PostTrainingQuantizerConfig.from_dict(
                    {
                        "global_config": {
                            "weight_dtype": "int8",
                        },
                    }
                )

                ptq = PostTrainingQuantizer(model, config)
                quantized_model = ptq.compress()

    Args:
        model (:obj:`torch.nn.Module`): Module to be compressed.
        config (:py:class:`PostTrainingQuantizerConfig`): Config that specifies how
            different submodules in the model will be quantized.
    """

    _supported_modules: _Tuple[_Type[_torch.nn.Module]] = (
        _nn.Conv1d,
        _nn.Conv2d,
        _nn.Conv3d,
        _nn.ConvTranspose1d,
        _nn.ConvTranspose2d,
        _nn.ConvTranspose3d,
        _nn.Linear,
        _nn.MultiheadAttention,
    )

    def _get_quantization_mode(
        self, weight_dtype: _torch.dtype, quantization_scheme: _QuantizationScheme
    ):
        """
        Returns quantization mode as string
        """
        if quantization_scheme not in [
            _QuantizationScheme.affine,
            _QuantizationScheme.symmetric,
        ]:
            raise ValueError(
                f" Linear quantization scheme must be one of (affine, "
                f"symmetric) not {quantization_scheme}"
            )
        quantization_mode = (
            "LINEAR_SYMMETRIC" if quantization_scheme == _QuantizationScheme.symmetric else "LINEAR"
        )
        return quantization_mode

    def __init__(self, model: _torch.nn.Module, config: PostTrainingQuantizerConfig = None):
        config = PostTrainingQuantizerConfig() if config is None else config
        super().__init__(model, config)

    def _compute_quantization_params(
        self,
        weight: _np.ndarray,
        nbits: int,
        dtype: _np.dtype,
        block_sizes: _List[int],
        quantization_mode: _Optional[str] = None,
        signed: bool = True,
    ) -> _Optional[_Tuple[_np.ndarray, _np.ndarray, _Optional[_np.ndarray]]]:
        """
        Compute quantization parameters
        """

        ret = _ct_compute_qparams(
            weight=weight,
            nbits=nbits,
            quantization_mode=quantization_mode,
            dtype=dtype,
            block_sizes=block_sizes,
            signed=signed,  # Always used signed dtype range
        )

        return ret

    def _dequantize_weight(
        self,
        quantized_weight: _np.ndarray,
        scale: _np.ndarray,
        zero_point: _Optional[_np.ndarray],
        quantization_mode: _Optional[str] = None,
    ):
        """
        De-quantize weights
        """

        dequantized_weight = constexpr_blockwise_shift_scale.decompress(
            quantized_weight, scale, zero_point
        )

        return dequantized_weight

    @_torch.no_grad()
    def _quantize_weight(
        self,
        submod_name: str,
        submodule: _torch.nn.Module,
        submod_config: ModulePostTrainingQuantizerConfig,
        param_name: str,
    ) -> _Optional[_Tuple[_torch.Tensor, _torch.Tensor, _Optional[_torch.Tensor]]]:
        """
        Helper function to perform the quantization on a PyTorch submodule's parameter

        Args:
            submod_name (:obj:`str`): Name of the submodule
            submodule (:obj:`torch.nn.Module`) Submodule which is being quantized
            submod_config (:py:class:`ModulePostTrainingQuantizerConfig`): Config for the submodule
            param_name (:obj:`str`): Name of the parameter within the submodule to quantize

        .. note::
            This function extracts the numpy array out of the torch weight value and
            uses that for performing the quantization
        """

        torch_weight = submodule.get_parameter(param_name)
        weight = torch_weight.numpy()

        block_sizes = [0] * weight.ndim
        assert len(block_sizes) >= 2, "Weight matrix has to be at least 2D or greater"

        if submod_config.granularity == QuantizationGranularity.per_channel:
            blocking_axis = (
                1
                if isinstance(
                    submodule,
                    (
                        _nn.ConvTranspose1d,
                        _nn.ConvTranspose2d,
                        _nn.ConvTranspose3d,
                    ),
                )
                else 0
            )
            block_sizes[blocking_axis] = 1

        elif submod_config.granularity == QuantizationGranularity.per_block:
            updated_config = _validate_param_config(
                submod_name + "." + param_name,
                torch_weight,
                submodule,
                submod_config,
                ["quantization_block_size"],
            )
            if not updated_config:
                _logger.warning(f"Unable to quantize layer {submod_name} - skipping it.")
                return
            block_size_config = list(updated_config.block_size)
            if isinstance(
                submodule,
                (
                    _nn.ConvTranspose1d,
                    _nn.ConvTranspose2d,
                    _nn.ConvTranspose3d,
                ),
            ):
                block_sizes[: len(block_size_config)] = block_size_config[::-1]
            else:
                block_sizes[: len(block_size_config)] = block_size_config

        quantization_mode = self._get_quantization_mode(
            submod_config.weight_dtype, submod_config.quantization_scheme
        )

        ret = self._compute_quantization_params(
            weight=weight,
            nbits=submod_config.weight_n_bits,
            quantization_mode=quantization_mode,
            dtype=weight.dtype,
            block_sizes=block_sizes,
            signed=True,
        )  # Always used signed dtype range

        if ret is None:
            _logger.warning(f"Unable to quantize layer {submod_name} - skipping it.")
            return

        quant_weight, scale, zp = ret

        dequant_weight = self._dequantize_weight(quant_weight, scale, zp, quantization_mode)

        # Convert back to torch tensors
        dequant_weight = _torch.from_numpy(dequant_weight)
        scale = _torch.from_numpy(scale)
        if zp is not None:
            zp = _torch.from_numpy(zp)

        # Replace the parameter's value
        submodule.get_parameter(param_name).data.copy_(dequant_weight)

        # Register compression metadata
        metadata = self._get_compression_metadata(param_name, submod_config, scale, zp)
        metadata.register(submodule)

    def _get_compression_metadata(self, param_name, submod_config, scale, zero_point):
        metadata = _CompressionMetadata(param_name)

        metadata.compression_type = ["quantization"]
        metadata.quantization_n_bits = submod_config.weight_n_bits
        metadata.quantization_scale = scale
        if submod_config.quantization_scheme == _QuantizationScheme.affine:
            assert zero_point is not None
            metadata.zero_point = zero_point

        return metadata

    def compress(self, inplace: bool = False) -> _torch.nn.Module:
        """
        Compress the supported layers in the module by quantizing each weight value of the layer.

        Args:
            inplace (:obj:`bool`): If ``True``, model transformations are carried out in-place and
                the original module is mutated, otherwise a copy of the model is mutated and returned.
                Defaults to ``False``.
        """
        self._model = super().compress(inplace=inplace)
        for submod_name, submodule in _get_atomic_layers(
            self._model, layer_types=list(self._supported_modules)
        ).items():
            submod_config = self._config.get_module_config(submod_name, submodule)
            if submod_config is None:
                continue

            # TODO: Replace this with supported modules abstraction
            # --- Conv, ConvTranspose & Linear layers ---
            if isinstance(submodule, self._supported_modules) and not isinstance(
                submodule, _nn.MultiheadAttention
            ):
                assert hasattr(
                    submodule, "weight"
                ), f"No parameter named weight in submodule {submod_name}"
                self._quantize_weight(submod_name, submodule, submod_config, "weight")

            # --- MultiheadAttention layer ---
            elif isinstance(submodule, _nn.MultiheadAttention):
                param_names = [
                    "in_proj_weight",
                    "q_proj_weight",
                    "k_proj_weight",
                    "v_proj_weight",
                ]
                for param_name in param_names:
                    if not hasattr(submodule, param_name):
                        continue
                    if getattr(submodule, param_name) is None:
                        continue
                    self._quantize_weight(submod_name, submodule, submod_config, param_name)

                if hasattr(submodule, "out_proj") and submodule.out_proj.weight is not None:
                    self._quantize_weight(
                        f"{submod_name}.out_proj",
                        submodule.out_proj,
                        submod_config,
                        "weight",
                    )
        return self._model

    def report(self) -> _Report:
        return _compute_post_training_report(
            self._uncompressed_model,
            self._model,
            supported_modules=self._supported_modules,
        )
