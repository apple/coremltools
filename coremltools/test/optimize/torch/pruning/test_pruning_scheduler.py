#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import sys

import pytest
import torch

from coremltools.optimize.torch.pruning import (
    ConstantSparsityScheduler,
    MagnitudePruner,
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig,
    PolynomialDecayScheduler,
)


@pytest.fixture
def simple_module():
    return torch.nn.Conv2d(3, 3, (3, 3), bias=False, groups=1)


@pytest.mark.skipif(sys.platform == "darwin", reason="temporarily disabled.")
@pytest.mark.parametrize('steps_and_expected', [[[4, 7, 9], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.875, 0.875, 1.0, 1.0]],
                                                [[3], [0.0, 0.0, 1.0, 1.0]]])
def test_polynomial_decay_correctness(simple_module, steps_and_expected):
    """
    Tests correctness of polynomial decay schedule.

    Note: Schedule can be stepped beyond the maximum step specified in update_steps.
    Beyond the max update step, the sparsity stays at the target sparsity.
    For example, in the first test case, we step 10 times, whereas max step is 9. At the 10th call
    to schedule.step, sparsity remains at 1.0.
    """

    update_steps, expected_sparsitys = steps_and_expected
    config = MagnitudePrunerConfig().set_global(
        ModuleMagnitudePrunerConfig(
            scheduler=PolynomialDecayScheduler(update_steps=update_steps),
            initial_sparsity=0.0, target_sparsity=1.0,
        )
    )
    pruner = MagnitudePruner(simple_module, config)
    pruner.prepare(inplace=True)

    for expected in expected_sparsitys:
        pruner.step()
        assert pruner._pruner_info[''].sparsity_level == expected


@pytest.mark.parametrize('steps', [[2.5, 6.5, 3.3],
                                   [[2, 3], [3, 5]],
                                   [-2, 0, 2]])
def test_polynomial_decay_initialization_failure(steps):
    with pytest.raises(Exception):
        PolynomialDecayScheduler(update_steps=steps)
    with pytest.raises(Exception):
        PolynomialDecayScheduler(update_steps=torch.tensor(steps))


@pytest.mark.skipif(sys.platform == "darwin", reason="temporarily disabled.")
@pytest.mark.parametrize('step_and_target', [(4, 0.5), (0, 0.8)])
def test_constant_sparsity_correctness(simple_module, step_and_target):
    """
    Tests correctness of spline schedule.

    Note: Schedule can be stepped beyond the maximum step specified in update_steps.
    Beyond the max update step, the sparsity stays at the target sparsity.
    For example, in the first test case, we step 10 times, whereas max step is 9. At the 10th call
    to schedule.step, sparsity remains at 1.0.
    """
    begin_step, target_sparsity = step_and_target
    initial_sparsity = target_sparsity if begin_step == 0 else 0.0
    config = MagnitudePrunerConfig().set_global(
        ModuleMagnitudePrunerConfig(
            scheduler=ConstantSparsityScheduler(begin_step=begin_step),
            initial_sparsity=initial_sparsity, target_sparsity=target_sparsity,
        )
    )
    pruner = MagnitudePruner(simple_module, config)
    pruner.prepare(inplace=True)
    for _ in range(begin_step):
        assert pruner._pruner_info[''].sparsity_level == initial_sparsity
        pruner.step()
    assert pruner._pruner_info[''].sparsity_level == target_sparsity
