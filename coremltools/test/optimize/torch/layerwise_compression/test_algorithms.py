#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from contextlib import nullcontext as does_not_raise

import pytest
import torch
import torch.nn as nn
from attr import define, field, validators

from coremltools.optimize.torch._utils.metadata_utils import (
    METADATA_VERSION,
    METADATA_VERSION_BUFFER,
    CompressionMetadata,
    CompressionType,
)
from coremltools.optimize.torch.layerwise_compression import (
    LayerwiseCompressor,
    LayerwiseCompressorConfig,
)
from coremltools.optimize.torch.layerwise_compression._quant import Quantizer
from coremltools.optimize.torch.layerwise_compression.algorithms import (
    GPTQ,
    LayerwiseCompressionAlgorithmConfig,
    ModuleGPTQConfig,
    ModuleSparseGPTConfig,
)


@pytest.mark.parametrize(
    "global_config_and_class",
    [
        ({"algorithm": "gptq", "weight_dtype": "uint4"}, ModuleGPTQConfig),
        (
            {
                "algorithm": "sparse_gpt",
                "weight_dtype": "uint8",
                "target_sparsity": 0.25,
            },
            ModuleSparseGPTConfig,
        ),
    ],
)
def test_obs_compression_algorithm_config(global_config_and_class):
    """
    Test the registry-based configuration of the :py:class:`LayerwiseCompressionAlgorithmConfig`
    using :py:class:`LayerwiseCompressorConfig`
    """

    global_config, class_type = global_config_and_class
    # compress
    config = LayerwiseCompressorConfig.from_dict(
        {
            "global_config": global_config,
            "input_cacher": "default",
            "calibration_nsamples": 128,
        }
    )
    algo = global_config.get("algorithm")
    algo_class = LayerwiseCompressionAlgorithmConfig.get_class(algo)
    assert algo_class == class_type
    assert isinstance(config.global_config, class_type)


def test_custom_obs_compression_algorithm_config():
    @LayerwiseCompressionAlgorithmConfig.register("foo")
    @define
    class FooConfig(LayerwiseCompressionAlgorithmConfig):
        bar: str = field(default=None, validator=validators.instance_of(str))
        algorithm: str = field(default="foo", validator=validators.instance_of(str))

    config = LayerwiseCompressorConfig.from_dict(
        {"global_config": {"algorithm": "foo", "bar": "baz"}}
    )

    assert isinstance(config.global_config, FooConfig)
    assert config.global_config.bar == "baz"


@pytest.mark.parametrize(
    "input_size, expectation",
    [
        (512, does_not_raise()),
        (1024, does_not_raise()),
        (480, pytest.raises(ValueError)),
        (960, pytest.raises(ValueError)),
    ],
)
def test_block_size_validation_gptq(input_size, expectation):
    """
    Test handling of block_size configuration for GPTQ algorithm
    """
    config = ModuleGPTQConfig.from_dict(
        {
            "algorithm": "gptq",
            "weight_dtype": "uint8",
            "block_size": 128,
            "granularity": "per_block",
        }
    )

    _model = nn.Transformer(d_model=input_size, nhead=8)
    layer = _model.encoder.layers.get_submodule("0.linear1")

    with expectation:
        gptq = GPTQ(layer, config)
        assert gptq is not None


@pytest.mark.parametrize("block_size", [32, 1024])
def test_blockwise_compression_gptq(block_size):
    model = nn.Sequential(nn.Linear(256, 100))
    config = ModuleGPTQConfig.from_dict(
        {
            "algorithm": "gptq",
            "weight_dtype": "uint8",
            "block_size": block_size,
            "granularity": "per_block",
        }
    )
    compressor_config = LayerwiseCompressorConfig().set_global(global_config=config)
    compressor = LayerwiseCompressor(model, compressor_config)

    def dataloader():
        yield torch.rand(10, 256)

    compressed_model = compressor.compress(dataloader=dataloader(), device="cpu")

    if model[0].weight.shape[1] % block_size != 0:
        # No compression; layer skipped
        assert torch.equal(compressed_model[0].weight, model[0].weight)
    else:
        # Compression should have occurred
        assert not torch.equal(compressed_model[0].weight, model[0].weight)


@pytest.mark.parametrize(
    "config",
    [
        {"global_config": {"algorithm": "gptq", "weight_dtype": "uint4"}},
        {
            "global_config": {
                "algorithm": "gptq",
                "weight_dtype": "uint8",
                "block_size": 16,
                "granularity": "per_block",
            }
        },
        {
            "global_config": {
                "algorithm": "gptq",
                "weight_dtype": "uint4",
                "enable_normal_float": True,
            }
        },
        {
            "global_config": {
                "algorithm": "gptq",
                "weight_dtype": "uint3",
                "enable_normal_float": True,
            }
        },
    ],
)
@pytest.mark.parametrize(
    "model, input_shape",
    [
        (nn.Sequential(nn.Linear(4096, 1024)), (1, 4096)),
        (nn.Sequential(nn.Conv2d(32, 64, 3)), (1, 32, 224, 224)),
    ],
)
def test_gptq_metadata(config, model, input_shape):
    """
    Test registration of metadata buffers for GPTQ algorithm
    """
    # Setup to get compressed model
    compressor_config = LayerwiseCompressorConfig.from_dict(config)
    compressor = LayerwiseCompressor(model, compressor_config)

    def calibration_loader():
        yield torch.rand(input_shape)

    compressed_model = compressor.compress(calibration_loader(), device="cpu")

    # Extract registered metadata from state_dict
    state_dict = compressed_model[0].state_dict()
    metadata_dict = CompressionMetadata.from_state_dict(state_dict)
    assert len(metadata_dict) == 1
    assert "weight" in metadata_dict

    # Verification
    metadata = metadata_dict["weight"]
    if compressor_config.global_config.enable_normal_float:
        assert metadata.compression_type == [CompressionType.palettization.value]
        assert metadata.lut.shape == (1,) * state_dict["weight"].dim() + (
            2**compressor_config.global_config.weight_n_bits,
            1,
        )
        assert metadata.palettization_scale.shape == (state_dict["weight"].shape[0], 1)
    else:
        assert metadata.compression_type == [CompressionType.quantization.value]
        assert metadata.quantization_n_bits == compressor_config.global_config.weight_n_bits
        assert metadata.zero_point.shape == metadata.quantization_scale.shape
        assert metadata.quantization_scale.shape[0] == state_dict["weight"].shape[0]
        block_size = compressor_config.global_config.block_size
        if block_size is None:
            assert metadata.quantization_scale.shape[1] == 1
        else:
            assert (
                metadata.quantization_scale.shape[1] == state_dict["weight"].shape[1] / block_size
            )

    assert METADATA_VERSION_BUFFER in compressed_model.state_dict()
    assert torch.equal(compressed_model.state_dict()[METADATA_VERSION_BUFFER], METADATA_VERSION)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"global_config": {"algorithm": "sparse_gpt"}}, id="pruning"),
        pytest.param(
            {"global_config": {"algorithm": "sparse_gpt", "weight_dtype": "float16"}},
            id="pruning_half_precision",
        ),
        pytest.param(
            {"global_config": {"algorithm": "sparse_gpt", "weight_dtype": "uint8"}},
            id="pruning_quantization",
        ),
        pytest.param(
            {
                "global_config": {
                    "algorithm": "sparse_gpt",
                    "weight_dtype": "uint4",
                    "enable_normal_float": True,
                }
            },
            id="pruning_palettization",
        ),
    ],
)
def test_sparse_gpt_metadata(config):
    """
    Test registration of metadata buffers for GPTQ algorithm
    """
    # Setup to get compressed model
    model = nn.Sequential(nn.Linear(4096, 1024))
    compressor_config = LayerwiseCompressorConfig.from_dict(config)
    compressor = LayerwiseCompressor(model, compressor_config)

    def calibration_loader():
        yield torch.rand(1, 4096)

    compressed_model = compressor.compress(calibration_loader(), device="cpu")

    # Extract registered metadata from state_dict
    state_dict = compressed_model[0].state_dict()
    metadata_dict = CompressionMetadata.from_state_dict(state_dict)
    assert len(metadata_dict) == 1
    assert "weight" in metadata_dict

    # Verification
    metadata = metadata_dict["weight"]
    if compressor_config.global_config.enable_normal_float:
        assert metadata.compression_type == [
            CompressionType.pruning.value,
            CompressionType.palettization.value,
        ]
        assert metadata.lut.shape == (
            1,
            1,
            2**compressor_config.global_config.weight_n_bits,
            1,
        )
        assert metadata.palettization_scale.shape == (state_dict["weight"].shape[0], 1)
    elif (
        compressor_config.global_config.weight_n_bits is not None
        and compressor_config.global_config.weight_n_bits < 16
    ):
        assert metadata.compression_type == [
            CompressionType.pruning.value,
            CompressionType.quantization.value,
        ]
        assert metadata.quantization_n_bits == compressor_config.global_config.weight_n_bits
        assert metadata.zero_point.shape == metadata.quantization_scale.shape

    assert METADATA_VERSION_BUFFER in compressed_model.state_dict()
    assert torch.equal(compressed_model.state_dict()[METADATA_VERSION_BUFFER], METADATA_VERSION)


@pytest.mark.parametrize(
    "config",
    [
        {
            "global_config": {
                "algorithm": "gptq",
                "weight_dtype": "uint8",
                "block_size": 16,
                "granularity": "per_block",
            }
        },
        {
            "global_config": {
                "algorithm": "gptq",
                "weight_dtype": "uint8",
                "block_size": None,
                "granularity": "per_block",
            }
        },
    ],
)
def test_gptq_block_size_configs(config):
    model = nn.Sequential(nn.Linear(4096, 1024))
    compressor_config = LayerwiseCompressorConfig.from_dict(config)
    compressor = LayerwiseCompressor(model, compressor_config)

    def calibration_loader():
        yield torch.rand(1, 4096)

    compressed_model = compressor.compress(calibration_loader(), device="cpu")


def test_gptq_static_blocking():
    model = nn.Sequential(nn.Linear(6, 8))

    compressor_config = LayerwiseCompressorConfig.from_dict(
        {
            "global_config": {
                "algorithm": "gptq",
                "weight_dtype": "uint8",
                "block_size": 2,
                "granularity": "per_block",
                "use_activation_order_heuristic": True,
            }
        }
    )
    compressor = LayerwiseCompressor(model, compressor_config)

    def calibration_loader():
        yield torch.randn(1, 6)

    compressed_model = compressor.compress(calibration_loader(), device="cpu")
    block_size = compressor_config.global_config.block_size

    quantizer = Quantizer(
        n_bits=8,
        per_channel=True,
        symmetric=True,
        enable_normal_float=False,
    )

    expected_scale = compressed_model[0]._buffers["_COREML_/weight/quantization_scale"]

    with torch.no_grad():
        for block_idx in range(3):
            start_idx = block_size * block_idx
            block = model[0].weight[:, start_idx : start_idx + block_size]
            quantizer.find_params(block, weight=True)
            assert torch.all(quantizer.scale.flatten() == expected_scale[:, block_idx])
