# Copyright (c) 2022, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest

import coremltools
from coremltools._deps import _HAS_TORCH, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil.testing_reqs import backends

if _HAS_TORCH:
    import torch
    import torch.nn.functional as F
    from torch import nn


@pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
class TestModelScripting:
    @staticmethod
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test(backend):
        # Example code from https://coremltools.readme.io/docs/model-scripting

        class _LoopBody(nn.Module):
            def __init__(self, channels):
                super(_LoopBody, self).__init__()
                conv = nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    padding=1,
                )
                self.conv = conv

            def forward(self, x):
                x = self.conv(x)
                x = F.relu(x)
                return x


        class ControlFlowNet(nn.Module):
            def __init__(self, num_channels: int):
                super(ControlFlowNet, self).__init__()
                self.loop_body = _LoopBody(num_channels)

            def forward(self, x):
                avg = torch.mean(x)
                if avg.item() < 0:
                    loop_count = 2
                else:
                    loop_count = 1
                for _ in range(loop_count):
                    x = self.loop_body(x)
                return x

        model = ControlFlowNet(num_channels=3)
        scripted_model = torch.jit.script(model)

        mlmodel = coremltools.converters.convert(
            scripted_model,
            inputs=[coremltools.TensorType(shape=(1, 3, 64, 64))],
            convert_to=backend[0],
        )
