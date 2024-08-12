#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import pytest

ct = pytest.importorskip("coremltools")
import coremltools.test.optimize.torch.conversion.conversion_utils as util
from coremltools.optimize.torch.layerwise_compression import (
    LayerwiseCompressor,
    LayerwiseCompressorConfig,
)
from coremltools.optimize.torch.palettization import DKMPalettizer, DKMPalettizerConfig
from coremltools.optimize.torch.pruning import MagnitudePruner, MagnitudePrunerConfig
from coremltools.optimize.torch.quantization import LinearQuantizer, LinearQuantizerConfig


@pytest.mark.skipif(ct.utils._macos_version() < (15, 0), reason="Only supported on macOS 15+")
def test_joint_pruning_quantization(mnist_model, mnist_example_input):
    example_input = mnist_example_input
    quant_config = LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "milestones": [0, 0, 10, 10],
            }
        }
    )
    prune_config = MagnitudePrunerConfig.from_dict({"global_config": {"target_sparsity": 0.5}})

    quantizer = LinearQuantizer(mnist_model, quant_config)
    quant_model = quantizer.prepare(example_inputs=(example_input,))

    pruner = MagnitudePruner(quant_model, prune_config)
    pruned_quant_model = pruner.prepare(inplace=True)

    quantizer.step()
    pruner.step()

    # Do a forward pass for pruner mask to be applied
    # Alternatively can set initial sparsity to target sparsity
    pruned_quant_model(example_input)

    quant_finalized_model = quantizer.finalize(inplace=True)
    finalized_model = pruner.finalize(quant_finalized_model)

    util.convert_and_verify(
        finalized_model,
        example_input,
        minimum_deployment_target=ct.target.iOS18,
        expected_ops=[
            "constexpr_sparse_to_dense",
            "constexpr_sparse_blockwise_shift_scale",
        ],
    )



@pytest.mark.skipif(ct.utils._macos_version() < (15, 0), reason="Only supported on macOS 15+")
@pytest.mark.parametrize(
    "config, expected_ops",
    [
        pytest.param(
            {"global_config": {"algorithm": "sparse_gpt"}},
            ["constexpr_sparse_to_dense"],
            id="pruning",
        ),
        pytest.param(
            {"global_config": {"algorithm": "sparse_gpt", "weight_dtype": "uint4"}},
            ["constexpr_sparse_to_dense", "constexpr_sparse_blockwise_shift_scale"],
            id="joint_pruning_quantization",
        ),
        pytest.param(
            {
                "global_config": {
                    "algorithm": "sparse_gpt",
                    "weight_dtype": "uint4",
                    "enable_normal_float": True,
                }
            },
            ["constexpr_sparse_to_dense", "constexpr_lut_to_sparse"],
            id="joint_pruning_palettization",
        ),
    ],
)
def test_sparsegpt(config, mnist_model, mnist_example_input, expected_ops):
    compressor_config = LayerwiseCompressorConfig.from_dict(config)
    compressor = LayerwiseCompressor(mnist_model, compressor_config)

    def calibration_loader():
        yield mnist_example_input

    compressed_model = compressor.compress(calibration_loader(), device="cpu")

    util.convert_and_verify(
        compressed_model,
        mnist_example_input,
        expected_ops=expected_ops,
    )
