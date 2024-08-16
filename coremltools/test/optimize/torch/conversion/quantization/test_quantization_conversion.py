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
from coremltools.optimize.torch.quantization import LinearQuantizer, LinearQuantizerConfig


# region LinearQuantizer
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {"global_config": {"quantization_scheme": "symmetric"}},
            id="symmetric_per_tensor",
        ),
        pytest.param({"global_config": {"quantization_scheme": "affine"}}, id="affine_per_tensor"),
        pytest.param(
            {
                "global_config": {
                    "weight_dtype": "qint4",
                    "quantization_scheme": "symmetric",
                }
            },
            id="4bit_symmetric_per_tensor",
        ),
        pytest.param(
            {
                "global_config": {
                    "weight_dtype": "qint4",
                    "quantization_scheme": "affine",
                }
            },
            id="4bit_affine_per_tensor",
        ),
    ],
)
@pytest.mark.skipif(ct.utils._macos_version() < (15, 0), reason="Only supported on macOS 15+")
@pytest.mark.parametrize("model", ["mnist_model", "mnist_model_conv_transpose"])
def test_linear_quantizer(config, model, mnist_example_input, request):
    quantizer_config = LinearQuantizerConfig.from_dict(config)
    quantizer = LinearQuantizer(request.getfixturevalue(model), quantizer_config)
    quantized_model = get_quantized_model(quantizer, mnist_example_input)

    util.convert_and_verify(
        quantized_model,
        mnist_example_input,
        expected_ops=["constexpr_blockwise_shift_scale"],
    )


# endregion


# region GPTQ
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {"global_config": {"algorithm": "gptq", "weight_dtype": "uint4"}},
            id="4bit",
        ),
        pytest.param(
            {
                "global_config": {
                    "algorithm": "gptq",
                    "weight_dtype": "uint8",
                    "block_size": 32,
                    "granularity": "per_block",
                }
            },
            id="blockwise",
        ),
        pytest.param(
            {
                "global_config": {
                    "algorithm": "gptq",
                    "weight_dtype": "uint4",
                    "block_size": 32,
                    "granularity": "per_block",
                }
            },
            id="4bit_blockwise",
        ),
    ],
)
@pytest.mark.skipif(ct.utils._macos_version() < (15, 0), reason="Only supported on macOS 15+")
def test_gptq(config, mnist_model, mnist_example_input):
    compressor_config = LayerwiseCompressorConfig.from_dict(config)
    compressor = LayerwiseCompressor(mnist_model, compressor_config)

    def calibration_loader():
        yield mnist_example_input

    compressed_model = compressor.compress(calibration_loader(), device="cpu")

    util.convert_and_verify(
        compressed_model,
        mnist_example_input,
        expected_ops=["constexpr_blockwise_shift_scale"],
    )
# endregion


# region PTQ
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            {"global_config": {"weight_dtype": "int4", "granularity": "per_tensor"}},
            id="4bit_per_tensor",
        ),
        pytest.param(
            {"global_config": {"weight_dtype": "int4", "granularity": "per_channel"}},
            id="4bit_per_channel",
        ),
        pytest.param(
            {
                "global_config": {
                    "weight_dtype": "int4",
                    "granularity": "per_block",
                    "block_size": 16,
                }
            },
            id="4bit_per_block",
        ),
    ],
)
@pytest.mark.skipif(ct.utils._macos_version() < (15, 0), reason="Only supported on macOS 15+")
def test_ptq(mnist_model, mnist_example_input, config):
    pytest.importorskip("coremltools.optimize.coreml._utils")
    from coremltools.optimize.torch.quantization.post_training_quantization import (
        PostTrainingQuantizer,
        PostTrainingQuantizerConfig,
    )

    model = mnist_model
    ptq_config = PostTrainingQuantizerConfig.from_dict(config)
    ptq = PostTrainingQuantizer(model, ptq_config)
    compressed_model = ptq.compress()

    util.convert_and_verify(
        compressed_model,
        mnist_example_input,
        expected_ops=["constexpr_blockwise_shift_scale"],
    )


# endregion

# region HelperMethods

def get_quantized_model(quantizer, example_input):
    quantizer.prepare(example_inputs=(example_input,), inplace=True)
    quantizer.step()
    model = quantizer.finalize()
    return model


# endregion
