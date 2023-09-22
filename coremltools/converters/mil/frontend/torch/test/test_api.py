# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os

import pytest
import torch

import coremltools as ct
from coremltools._deps import (
    _HAS_EXECUTORCH,
    _HAS_TORCH,
    MSG_EXECUTORCH_NOT_FOUND,
    MSG_TORCH_NOT_FOUND,
)
from coremltools.converters.mil.testing_reqs import backends

if _HAS_TORCH:
    import torch

if _HAS_EXECUTORCH:
    from executorch import exir

    _CAPTURE_CONFIG = exir.CaptureConfig(enable_aot=True)
    _EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
        _check_ir_validity=False,
    )


@pytest.fixture
def torch_model():
    class TestModule(torch.nn.Module):
        def __init__(self):
            super(TestModule, self).__init__()
            self.linear = torch.nn.Linear(10, 20)

        def forward(self, x):
            return self.linear(x)

    model = TestModule()
    model.eval()
    return model

@pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
class TestPyTorchConverter:
    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_no_inputs(torch_model, backend):

        traced_torch_model = torch.jit.trace(torch_model, torch.rand(1, 10))
        with pytest.raises(ValueError) as e:
            ct.convert(traced_torch_model, convert_to=backend[0])
        e.match(r'Expected argument for pytorch "inputs" not provided')


    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_pth_extension(torch_model, tmpdir, backend):
        # test for issue: https://github.com/apple/coremltools/issues/917

        shape = (1, 10)
        traced_torch_model = torch.jit.trace(torch_model, torch.rand(*shape))

        model_path = os.path.join(str(tmpdir), "torch_model.pth")
        traced_torch_model.save(model_path)

        ct.convert(
            model_path,
            source='pytorch',
            inputs=[
                ct.TensorType(
                    shape=shape,
                )
            ],
            convert_to=backend[0],
        )


@pytest.mark.skipif(not _HAS_EXECUTORCH, reason=MSG_EXECUTORCH_NOT_FOUND)
class TestEdgeIRConverter:
    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_inputs(
        torch_model, backend
    ):  # TODO: rdar://115845792 ([Executorch] Handle user provided inputs/outputs in the convert API)

        shape = (1, 10)
        exir_program = exir.capture(torch_model, (torch.rand(*shape),), _CAPTURE_CONFIG).to_edge(
            _EDGE_COMPILE_CONFIG
        )

        with pytest.raises(AssertionError) as e:
            ct.convert(
                exir_program,
                convert_to=backend[0],
                inputs=[ct.TensorType(shape=shape)],
            )
        e.match(r"'inputs' argument should be None for ExportedProgram")

    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_outputs(
        torch_model, backend
    ):  # TODO: rdar://115845792 ([Executorch] Handle user provided inputs/outputs in the convert API)

        shape = (1, 10)
        exir_program = exir.capture(torch_model, (torch.rand(*shape),), _CAPTURE_CONFIG).to_edge(
            _EDGE_COMPILE_CONFIG
        )

        with pytest.raises(AssertionError) as e:
            ct.convert(exir_program, convert_to=backend[0], outputs=[ct.TensorType(name="result")])
        e.match(r"'outputs' argument should be None for ExportedProgram")
