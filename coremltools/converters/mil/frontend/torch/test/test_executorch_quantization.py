#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

import coremltools as ct
from coremltools._deps import _HAS_EXECUTORCH

if not _HAS_EXECUTORCH:
    pytest.skip(allow_module_level=True, reason="executorch is required")

from .testing_utils import TorchBaseTest, TorchFrontend


class TestExecutorchQuantization(TorchBaseTest):
    def test_conv_relu(self):
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

        example_args = (torch.randn(SHAPE),)
        pre_autograd_aten_dialect = capture_pre_autograd_graph(model, example_args)

        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
        converted_graph = convert_pt2e(prepared_graph)

        self.run_compare_torch(
            SHAPE,
            converted_graph,
            frontend=TorchFrontend.EXIR,
            backend=("mlprogram", "fp16"),
            minimum_deployment_target=ct.target.iOS17,
        )
