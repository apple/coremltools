# Copyright (c) 2026, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""Regression test for the stateful-output aliasing guard.

Background: the Core ML runtime proxy crashes with a segmentation fault
(no Python traceback) when loading an mlprogram whose function output Var
is the same Var that feeds a ``coreml_update_state`` op. The pattern is
trivial to write when porting a torch decoder transformer: assign to a KV
cache via ``self.cache[:] = merged`` then ``return merged``.

The converter now rejects this case with a clear ``ValueError`` that names
both the offending output and the affected state and points at the
workaround. This test pins both behaviours.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

import coremltools as ct


EMBED = 8
MAX_SEQ = 16


class _AliasingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, MAX_SEQ, EMBED))

    def forward(self, x):
        # The merged tensor is BOTH written to the cache AND returned —
        # the runtime-crashing pattern.
        merged = self.cache + x
        self.cache[:] = merged
        return merged


class _NonAliasingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, MAX_SEQ, EMBED))

    def forward(self, x):
        # The returned tensor is reduced — it does not feed the state-write
        # chain, so the converter accepts the program.
        merged = self.cache + x
        self.cache[:] = merged
        return merged.sum(dim=-1, keepdim=True)


def _convert(model):
    model.eval()
    model.cache.zero_()
    traced = torch.jit.trace(model, (torch.randn(1, MAX_SEQ, EMBED),))
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, MAX_SEQ, EMBED), dtype=np.float16)],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, MAX_SEQ, EMBED), dtype=np.float16
                ),
                name="cache",
            )
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
    )


class TestStatefulOutputAliasGuard:
    def test_aliasing_pattern_raises_clear_error(self):
        with pytest.raises(ValueError) as excinfo:
            _convert(_AliasingNet())
        message = str(excinfo.value)
        # Error must name the offending output, the affected state, and
        # point at the workaround.
        assert "merged" in message
        assert "cache" in message
        assert "Workaround" in message or "workaround" in message

    def test_non_aliasing_pattern_converts(self):
        mlmodel = _convert(_NonAliasingNet())
        assert isinstance(mlmodel, ct.models.MLModel)
