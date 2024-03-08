#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Optional as _Optional

import torch as _torch
import torch.ao.quantization as _aoquant
from attr import define as _define
from torch.ao.quantization.quantizer.quantizer import (
    QuantizationSpec as _TorchQuantizationSpec,
)

from coremltools.optimize.torch.quantization.quantization_config import (
    ModuleLinearQuantizerConfig as _ModuleLinearQuantizerConfig,
)
from coremltools.optimize.torch.quantization.quantization_config import ObserverType as _ObserverType
from coremltools.optimize.torch.quantization.quantization_config import (
    QuantizationScheme as _QuantizationScheme,
)


@_define
class AnnotationConfig:
    """
    Module/Operator level configuration class for :py:class:`CoreMLQuantizer`.

    For each module/operator, defines the dtype, quantization scheme and observer type
    for input(s), output and weights (if any).
    """

    input_activation: _Optional[_TorchQuantizationSpec] = None
    output_activation: _Optional[_TorchQuantizationSpec] = None
    weight: _Optional[_TorchQuantizationSpec] = None

    @staticmethod
    def _normalize_dtype(dtype: _torch.dtype) -> _torch.dtype:
        """
        PyTorch export quantizer only supports uint8 and int8 data types,
        so we map the quantized dtypes to the corresponding supported dtype.
        """
        dtype_map = {
            _torch.quint8: _torch.uint8,
            _torch.qint8: _torch.int8,
        }
        return dtype_map.get(dtype, dtype)

    @classmethod
    def from_quantization_config(
        cls,
        quantization_config: _Optional[_ModuleLinearQuantizerConfig],
    ) -> _Optional["AnnotationConfig"]:
        """
        Creates a :py:class:`AnnotationConfig` from ``ModuleLinearQuantizerConfig``
        """
        if (
            quantization_config is None
            or quantization_config.weight_dtype == _torch.float32
        ):
            return None

        # Activation QSpec
        if quantization_config.activation_dtype == _torch.float32:
            output_activation_qspec = None
        else:
            activation_qscheme = _QuantizationScheme.get_qscheme(
                quantization_config.quantization_scheme,
                is_per_channel=False,
            )
            activation_dtype = cls._normalize_dtype(
                quantization_config.activation_dtype
            )
            output_activation_qspec = _TorchQuantizationSpec(
                observer_or_fake_quant_ctr=_aoquant.FakeQuantize.with_args(
                    observer=_ObserverType.get_observer(
                        quantization_config.activation_observer,
                        is_per_channel=False,
                    ),
                    dtype=activation_dtype,
                    qscheme=activation_qscheme,
                ),
                dtype=activation_dtype,
                qscheme=activation_qscheme,
            )

        # Weight QSpec
        weight_qscheme = _QuantizationScheme.get_qscheme(
            quantization_config.quantization_scheme,
            is_per_channel=quantization_config.weight_per_channel,
        )
        weight_dtype = cls._normalize_dtype(quantization_config.weight_dtype)
        weight_qspec = _TorchQuantizationSpec(
            observer_or_fake_quant_ctr=_aoquant.FakeQuantize.with_args(
                observer=_ObserverType.get_observer(
                    quantization_config.weight_observer,
                    is_per_channel=quantization_config.weight_per_channel,
                ),
                dtype=weight_dtype,
                qscheme=weight_qscheme,
            ),
            dtype=weight_dtype,
            qscheme=weight_qscheme,
        )
        return AnnotationConfig(
            input_activation=output_activation_qspec,
            output_activation=output_activation_qspec,
            weight=weight_qspec,
        )
