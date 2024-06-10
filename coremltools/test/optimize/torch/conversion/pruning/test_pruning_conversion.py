#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest

ct = pytest.importorskip("coremltools")
import coremltools.test.optimize.torch.conversion.conversion_utils as util
from coremltools.optimize.torch.pruning import MagnitudePruner, MagnitudePrunerConfig


# region MagnitudePruner
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {
                "global_config": {
                    "initial_sparsity": 0.5,
                    "target_sparsity": 0.5,
                }
            },
            id="unstructured_sparsity",
        ),
        pytest.param(
            {
                "global_config": {
                    "initial_sparsity": 0.5,
                    "target_sparsity": 0.5,
                    "block_size": 2,
                }
            },
            id="block_structured_sparsity",
        ),
        pytest.param(
            {
                "global_config": {
                    "initial_sparsity": 0.5,
                    "target_sparsity": 0.5,
                    "n_m_ratio": (1, 2),
                }
            },
            id="n_m_structured_sparsity",
        ),
        pytest.param(
            {
                "global_config": {
                    "initial_sparsity": 0.5,
                    "target_sparsity": 0.5,
                    "granularity": "per_channel",
                }
            },
            id="general_structured_sparsity",
        ),
    ],
)
@pytest.mark.skipif(ct.utils._macos_version() < (15, 0), reason="Only supported on macOS 15+")
def test_magnitude_pruner(config, mnist_model, mnist_example_input):
    pruner_config = MagnitudePrunerConfig.from_dict(config)
    pruner = MagnitudePruner(mnist_model, pruner_config)
    pruned_model = get_pruned_model(pruner)

    util.convert_and_verify(
        pruned_model,
        mnist_example_input,
        pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
        expected_ops=["constexpr_sparse_to_dense"],
    )

# endregion

# region GlobalUnstructuredPruner

# endregion

# region STRPruner

# endregion


# region HelperMethods
def get_pruned_model(pruner):
    pruner.prepare(inplace=True)
    pruner.step()
    return pruner.finalize()

# endregion
