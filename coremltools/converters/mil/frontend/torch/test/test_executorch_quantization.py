#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest

from coremltools._deps import _HAS_EXECUTORCH

if not _HAS_EXECUTORCH:
    pytest.skip(allow_module_level=True, reason="executorch is required")

import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

import executorch.exir

import coremltools as ct


def quantize_model():
    SHAPE = (1, 3, 256, 256)
    class SimpleConv(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1
            )
            self.relu = torch.nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            a = self.conv(x)
            return self.relu(a)
        
    model = SimpleConv()
    
    example_args = (torch.randn(SHAPE),)
    pre_autograd_aten_dialect = capture_pre_autograd_graph(model, example_args)
    with open("pre-autograd.log", 'w') as f:
        print(pre_autograd_aten_dialect, file=f)

    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
    converted_graph = convert_pt2e(prepared_graph)
    with open("quantized.log", 'w') as f:
        print(converted_graph, file=f)

    aten_dialect = torch.export.export(converted_graph, example_args)
    with open("aten-dialect.log", 'w') as f:
        print(aten_dialect, file=f)

    edge_dialect = executorch.exir.to_edge(aten_dialect).exported_program()
    with open("edge-dialect.log", 'w') as f:
        print(edge_dialect, file=f)

    coreml_model = ct.convert(
        edge_dialect,
        minimum_deployment_target=ct.target.iOS17,
    )

    x = torch.randn(SHAPE)
    y_original = model(x).detach().numpy()
    y_coreml = coreml_model.predict({"arg2_1": x})["dequantize_2"]
    np.testing.assert_allclose(y_coreml, y_original)


if __name__ == "__main__":
    quantize_model()
