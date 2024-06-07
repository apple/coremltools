#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import cattrs
import numpy as np
from attrs import define, field, validators

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.types.type_mapping import is_builtin, numpy_type_to_builtin_type

from .._config import OpCompressorConfig, _check_weight_threshold, _normalize_dtype

"""
Activation Linear Quantization configuration
"""

# TODO: This should be refactored to reuse OpLinearQuantizerConfig (rdar://129257210).
@define
class OpActivationLinearQuantizerConfig(OpCompressorConfig):
    """
    Parameters
    ----------
    mode: str
        Mode for linear quantization:

        * ``"linear_symmetric"`` (default): Input data are quantized in the range
          ``[-R, R]``, where :math:`R = max(abs(w_r))`.

    dtype: str or np.generic or mil.type
        Determines the quantized data type.

        * The allowed values are:
            * ``np.int8`` (the default)
            * ``coremltools.converters.mil.mil.types.int8``

    weight_threshold: int
        If the operation has weight, above which activation are compressed.

        Set the same ``weight_threshold`` for activation as for weight linear quantization can guarantee
        valid operations get both weight and activation quantization to improve efficiency.
        * If not provided, it will be set to ``2048``, in which operations with weights bigger than ``2048``
          elements are compressed.
    """

    # TODO: enable more modes/dtypes (rdar://129257210).
    mode: str = field(default="linear_symmetric", validator=validators.instance_of(str))
    dtype: Union[str, type] = field(default=types.int8, converter=_normalize_dtype)

    # Set the same ``weight_threshold`` for activation linear quantization as for weight linear quantization can guarantee
    # valid operations get both the weight (if weight exists) and activation linear quantized to improve efficiency.
    weight_threshold: Optional[int] = field(
        default=2048,
        validator=validators.optional([validators.instance_of(int), _check_weight_threshold]),
    )

    _ACTIVATION_AFFINE_QUANTIZATION_MODES = ("LINEAR_SYMMETRIC",)

    @mode.validator
    def check_mode(self, attr, mode):
        if not mode.upper() in self._ACTIVATION_AFFINE_QUANTIZATION_MODES:
            raise ValueError(
                f'Only mode {self._ACTIVATION_AFFINE_QUANTIZATION_MODES} supported for activation affine quantization. Got mode: "{mode}".'
            )

    @dtype.validator
    def check_dtype(self, attr, dtype):
        if not types.is_builtin(dtype):
            raise ValueError(f"Invalid dtype. Should be builtin dtype, but got {type(dtype)}")
        if not (types.is_int(dtype) and dtype.get_bitwidth() in {8} and not dtype.is_unsigned()):
            raise ValueError(
                f"Invalid dtype. Should be int8, but got {types.builtin_to_string(dtype)}"
            )

    def __attrs_post_init__(self):
        self.mode = self.mode.upper()
        if not is_builtin(self.dtype):
            self.dtype = numpy_type_to_builtin_type(self.dtype)

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "OpActivationLinearQuantizerConfig":
        def _structure_type(value, dtype):
            if isinstance(value, type):
                return value
            else:
                if not isinstance(value, str) or value not in ("int8",):
                    raise ValueError(f'"dtype" must be type of type or str ["int8"]. Got {value}')
                return getattr(np, value)

        converter = cattrs.Converter(forbid_extra_keys=True)
        converter.register_structure_hook(type, _structure_type)
        return converter.structure(config_dict, cls)
