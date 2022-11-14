# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os

import pytest
import torch
import torchvision

import coremltools as ct
from coremltools._deps import _HAS_TORCH, MSG_TORCH_NOT_FOUND

if _HAS_TORCH:
    import torch
    import torchvision


@pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
class TestPyTorchConverter:
    @staticmethod
    def test_no_inputs():
        model = torchvision.models.mobilenet_v2()
        model.eval()

        example_input = torch.rand(1, 3, 256, 256)

        traced_model = torch.jit.trace(model, example_input)

        with pytest.raises(ValueError) as e:
            ct.convert(traced_model)
        e.match(r'Expected argument for pytorch "inputs" not provided')


    @staticmethod
    def test_pth_extension(tmpdir):
        # test for issue: https://github.com/apple/coremltools/issues/917
        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.linear = torch.nn.Linear(10, 20)

            def forward(self, x):
                return self.linear(x)

        model = TestModule()
        model.eval()
        example_input = torch.rand(1, 10)
        traced_model = torch.jit.trace(model, example_input)
        model_path = os.path.join(str(tmpdir), "torch_model.pth")
        traced_model.save(model_path)

        ct.convert(
            model_path,
            source='pytorch',
            inputs=[
                ct.TensorType(
                    shape=example_input.shape,
                )
            ],
        )
