#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import pytest
from typing import Tuple

from coremltools._deps import _HAS_EXECUTORCH

if not _HAS_EXECUTORCH:
    pytest.skip(allow_module_level=True, reason="executorch is required")

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

_TORCH_VERSION = torch.__version__
_EXPECTED_TORCH_VERSION = "2.2.0"
if _TORCH_VERSION < _EXPECTED_TORCH_VERSION:
    pytest.skip(allow_module_level=True, reason=f"PyTorch {_EXPECTED_TORCH_VERSION} or higher is required")

import coremltools as ct
from coremltools.optimize.torch.quantization.quantization_config import (
    LinearQuantizerConfig,
    QuantizationScheme,
)
from coremltools.optimize.torch.quantization._coreml_quantizer import CoreMLQuantizer

from .testing_utils import TorchBaseTest, TorchFrontend


class TestExecutorchQuantization(TorchBaseTest):
    @staticmethod
    def make_torch_quantized_graph(
        model,
        example_inputs: Tuple[torch.Tensor],
        quantizer_name: str,
        quantization_type: str,
    ) -> torch.fx.GraphModule:
        assert quantizer_name in {"XNNPack", "CoreML"}
        assert quantization_type in {"PTQ", "QAT"}

        pre_autograd_aten_dialect = capture_pre_autograd_graph(model, example_inputs)

        if quantizer_name == "XNNPack":
            quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        elif quantizer_name == "CoreML":
            quantization_config = LinearQuantizerConfig.from_dict(
                {
                    "global_config": {
                        "quantization_scheme": QuantizationScheme.symmetric,
                        "milestones": [0, 0, 10, 10],
                        "activation_dtype": torch.quint8,
                        "weight_dtype": torch.qint8,
                        "weight_per_channel": True,
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
        "quantizer_name, quantization_type",
        itertools.product(
            ("XNNPack", "CoreML"),
            ("PTQ", "QAT")
        )
    )
    def test_conv_relu(self, quantizer_name, quantization_type):
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
        )

        self.run_compare_torch(
            SHAPE,
            converted_graph,
            frontend=TorchFrontend.EXIR,
            backend=("mlprogram", "fp16"),
            minimum_deployment_target=ct.target.iOS17,
        )

    @pytest.mark.parametrize(
        "quantizer_name, quantization_type",
        itertools.product(
            ("XNNPack", "CoreML"),
            ("PTQ", "QAT")
        )
    )
    def test_linear(self, quantizer_name, quantization_type):
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
        )

        self.run_compare_torch(
            SHAPE,
            converted_graph,
            frontend=TorchFrontend.EXIR,
            backend=("mlprogram", "fp16"),
            minimum_deployment_target=ct.target.iOS17,
        )
