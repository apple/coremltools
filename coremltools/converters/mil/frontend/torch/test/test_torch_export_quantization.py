#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
from typing import Tuple

import pytest

from coremltools._deps import _HAS_EXECUTORCH, _HAS_TORCH_EXPORT_API

if not _HAS_TORCH_EXPORT_API:
    pytest.skip(allow_module_level=True, reason="torch.export is required")

USE_EDGE_DIALECT = [False]
if _HAS_EXECUTORCH:
    USE_EDGE_DIALECT.append(True)

import torch

_TORCH_VERSION = torch.__version__
_EXPECTED_TORCH_VERSION = "2.2.0"
if _TORCH_VERSION < _EXPECTED_TORCH_VERSION:
    pytest.skip(
        allow_module_level=True, reason=f"PyTorch {_EXPECTED_TORCH_VERSION} or higher is required"
    )

from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

import coremltools as ct
import coremltools.converters.mil.mil.types as types
from coremltools.converters.mil.testing_utils import get_op_types_in_program
from coremltools.optimize.torch.quantization._coreml_quantizer import CoreMLQuantizer
from coremltools.optimize.torch.quantization.quantization_config import (
    LinearQuantizerConfig,
    QuantizationScheme,
)

from .testing_utils import TorchBaseTest, TorchFrontend


class TestTorchExportQuantization(TorchBaseTest):
    @staticmethod
    def make_torch_quantized_graph(
        model,
        example_inputs: Tuple[torch.Tensor],
        quantizer_name: str,
        quantization_type: str,
        is_per_channel: bool,
        nbit: int,
    ) -> torch.fx.GraphModule:
        assert quantizer_name in ("XNNPack", "CoreML")
        assert quantization_type in ("PTQ", "QAT")
        assert nbit in (4, 8)

        if quantizer_name == "CoreML" and nbit == 4:
            pytest.skip("4-bit Core ML quantizer is under development")

        if torch.__version__ <= _EXPECTED_TORCH_VERSION:
            if (quantizer_name, is_per_channel, nbit) != ("CoreML", False, 8):
                pytest.xfail("Need at least torch 2.3.0 to run this test.")

        pre_autograd_aten_dialect = capture_pre_autograd_graph(model, example_inputs)

        if quantizer_name == "XNNPack":
            # As of iOS 18, Core ML does not have 4-bit activation quantization,
            # so we only test 4-bit weight
            if nbit == 4:
                weight_qmin = -8
                weight_qmax = 7
            else:
                weight_qmin = -128
                weight_qmax = 127
            quantization_config = get_symmetric_quantization_config(
                is_per_channel=is_per_channel,
                is_qat=(quantization_type == "QAT"),
                is_dynamic=False,
                weight_qmin=weight_qmin,
                weight_qmax=weight_qmax,
            )
            quantizer = XNNPACKQuantizer().set_global(quantization_config)
        elif quantizer_name == "CoreML":
            quantization_config = LinearQuantizerConfig.from_dict(
                {
                    "global_config": {
                        "quantization_scheme": QuantizationScheme.symmetric,
                        "activation_dtype": torch.quint8,
                        "weight_dtype": torch.qint8,
                        "weight_per_channel": is_per_channel,
                    }
                }
            )
            quantizer = CoreMLQuantizer(quantization_config)

        if quantization_type == "PTQ":
            prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
        elif quantization_type == "QAT":
            prepared_graph = prepare_qat_pt2e(pre_autograd_aten_dialect, quantizer)

        prepared_graph(*example_inputs)
        converted_graph = convert_pt2e(prepared_graph)
        return converted_graph

    @pytest.mark.parametrize(
        "quantizer_name, quantization_type, is_per_channel, nbit, use_edge_dialect",
        itertools.product(
            ("XNNPack", "CoreML"),
            ("PTQ", "QAT"),
            (True, False),
            (4, 8),
            USE_EDGE_DIALECT,
        ),
    )
    def test_conv_relu(
        self, quantizer_name, quantization_type, is_per_channel, nbit, use_edge_dialect
    ):
        SHAPE = (1, 3, 256, 256)

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, padding=1
                )
                self.relu = torch.nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a = self.conv(x)
                return self.relu(a)

        model = Model()

        example_inputs = (torch.randn(SHAPE),)
        converted_graph = self.make_torch_quantized_graph(
            model,
            example_inputs,
            quantizer_name,
            quantization_type,
            is_per_channel,
            nbit,
        )

        minimum_deployment_target = ct.target.iOS17
        if nbit == 4:
            minimum_deployment_target = max(minimum_deployment_target, ct.target.iOS18)
        _, mlmodel, _, _, _, _ = self.run_compare_torch(
            SHAPE,
            converted_graph,
            frontend=TorchFrontend.EXIR,
            use_edge_dialect=use_edge_dialect,
            backend=("mlprogram", "fp16"),
            minimum_deployment_target=minimum_deployment_target,
        )

        op_types_in_program = get_op_types_in_program(mlmodel._mil_program)
        if nbit == 4:
            assert "constexpr_blockwise_shift_scale" in op_types_in_program
            constexpr_blockwise_shift_scale_op = mlmodel._mil_program.find_ops(
                op_type="constexpr_blockwise_shift_scale", exactly_one=True
            )[0]
            assert constexpr_blockwise_shift_scale_op.data.dtype in (types.int4, types.uint4)
        else:
            assert "constexpr_affine_dequantize" in op_types_in_program
            constexpr_affine_dequantize_op = mlmodel._mil_program.find_ops(
                op_type="constexpr_affine_dequantize", exactly_one=True
            )[0]
            assert constexpr_affine_dequantize_op.quantized_data.dtype in (types.int8, types.uint8)

    @pytest.mark.parametrize(
        "quantizer_name, quantization_type, is_per_channel, nbit, use_edge_dialect",
        itertools.product(
            ("XNNPack", "CoreML"),
            ("PTQ", "QAT"),
            (True, False),
            (4, 8),
            USE_EDGE_DIALECT,
        ),
    )
    def test_linear(
        self, quantizer_name, quantization_type, is_per_channel, nbit, use_edge_dialect
    ):
        SHAPE = (1, 5)

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        model = Model()

        example_inputs = (torch.randn(SHAPE),)
        converted_graph = self.make_torch_quantized_graph(
            model,
            example_inputs,
            quantizer_name,
            quantization_type,
            is_per_channel,
            nbit,
        )

        minimum_deployment_target = ct.target.iOS17
        if nbit == 4:
            minimum_deployment_target = max(minimum_deployment_target, ct.target.iOS18)
        _, mlmodel, _, _, _, _ = self.run_compare_torch(
            SHAPE,
            converted_graph,
            frontend=TorchFrontend.EXIR,
            use_edge_dialect=use_edge_dialect,
            backend=("mlprogram", "fp16"),
            minimum_deployment_target=minimum_deployment_target,
        )

        op_types_in_program = get_op_types_in_program(mlmodel._mil_program)
        if nbit == 4:
            assert "constexpr_blockwise_shift_scale" in op_types_in_program
            constexpr_blockwise_shift_scale_op = mlmodel._mil_program.find_ops(
                op_type="constexpr_blockwise_shift_scale", exactly_one=True
            )[0]
            assert constexpr_blockwise_shift_scale_op.data.dtype in (types.int4, types.uint4)
        else:
            assert "constexpr_affine_dequantize" in op_types_in_program
            constexpr_affine_dequantize_op = mlmodel._mil_program.find_ops(
                op_type="constexpr_affine_dequantize", exactly_one=True
            )[0]
            assert constexpr_affine_dequantize_op.quantized_data.dtype in (types.int8, types.uint8)
