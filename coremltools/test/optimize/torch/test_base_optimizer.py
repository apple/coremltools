#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch

from coremltools.optimize.torch.base_model_optimizer import (
    BaseDataCalibratedModelOptimizer,
    BasePostTrainingModelOptimizer,
)
from coremltools.optimize.torch.palettization import DKMPalettizer
from coremltools.optimize.torch.pruning import MagnitudePruner
from coremltools.optimize.torch.quantization import LinearQuantizer


@pytest.mark.parametrize("optimizer", [MagnitudePruner, LinearQuantizer, DKMPalettizer])
@pytest.mark.parametrize("inplace", [True, False])
def test_report_model_train_state(optimizer, inplace):
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 31, 2, 1), torch.nn.Conv2d(31, 21, 2, 1))

    opt = optimizer(model)
    if optimizer == LinearQuantizer:
        p_model = opt.prepare(inplace=inplace, example_inputs=(torch.randn(1),))
    else:
        p_model = opt.prepare(inplace=inplace)

    p_model.train()
    opt.report()
    assert p_model.training

    p_model.eval()
    opt.report()
    assert not p_model.training


@pytest.mark.parametrize(
    "optimizer", [BasePostTrainingModelOptimizer, BaseDataCalibratedModelOptimizer]
)
@pytest.mark.parametrize("inplace", [True, False])
def test_inplace_behavior_for_optimizers(optimizer, inplace):
    def create_model():
        return torch.nn.Sequential(torch.nn.Conv2d(1, 31, 2, 1), torch.nn.Conv2d(31, 21, 2, 1))

    class DummyOptimizer(optimizer):
        def report(self):
            return None

        @torch.no_grad()
        def compress(self, *args, inplace, **kwargs):
            super().compress(*args, inplace=inplace, **kwargs)
            self._model[0].weight.data = torch.ones_like(self._model[0].weight.data)
            return self._model

    model = create_model()
    opt = DummyOptimizer(model)
    opt.compress(dataloader=None, inplace=inplace)

    if inplace:
        assert id(opt._model) == id(model)
        assert id(opt._uncompressed_model) != id(model)
    else:
        assert id(opt._model) != id(model)
        assert id(opt._uncompressed_model) == id(model)

    assert torch.all(opt._model[0].weight == 1)
    assert not torch.all(opt._uncompressed_model[0].weight == 1)
