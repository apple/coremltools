#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from contextlib import nullcontext as does_not_raise

import pytest
import torch

from coremltools.optimize.torch._utils.metadata_utils import (
    METADATA_VERSION,
    METADATA_VERSION_BUFFER,
    CompressionMetadata,
    CompressionType,
    register_metadata_version,
)


@pytest.mark.parametrize(
    "metadata_dict, expectation",
    [
        (
            {
                "param_name": "weight",
                "quantization_scale": torch.rand(3, 1),
                "quantization_n_bits": 4,
                "compression_type": ["pruning", "quantization"],
            },
            does_not_raise(),
        ),
        (
            {
                "param_name": "weight",
                "quantization_scale": torch.rand(3, 1),
                "quantization_n_bits": 4,
                "compression_type": ["pruning", "quantizatoin"],  # mis-spelled
            },
            pytest.raises(KeyError),
        ),
    ],
)
def test_metadata_from_dict(metadata_dict, expectation):
    with expectation:
        metadata = CompressionMetadata.from_dict(metadata_dict)
        assert torch.equal(metadata.quantization_scale, metadata_dict["quantization_scale"])
        assert metadata.quantization_n_bits == metadata_dict["quantization_n_bits"]
        assert metadata.compression_type == [
            CompressionType[x].value for x in metadata_dict["compression_type"]
        ]

        for key, value in metadata.as_dict().items():
            if key not in metadata_dict:
                assert value is None


@pytest.mark.parametrize(
    "state_dict",
    [
        {
            "_COREML_/weight/quantization_scale": torch.rand(3, 1),
            "_COREML_/weight/quantization_n_bits": torch.tensor(4),
            "_COREML_/weight/compression_type": torch.tensor([1, 2]),
            "_COREML_/bias/quantization_scale": torch.rand(3, 1),
            "_COREML_/bias/quantization_n_bits": torch.tensor(8),
            "_COREML_/bias/compression_type": torch.tensor([1, 3]),
        }
    ],
)
def test_metadata_from_state_dict(state_dict):
    metadata_dict = CompressionMetadata.from_state_dict(state_dict)
    print(metadata_dict)
    assert len(metadata_dict) == 2
    assert "weight" in metadata_dict
    assert "bias" in metadata_dict
    for param in ["weight", "bias"]:
        metadata = metadata_dict[param]
        assert metadata.param_name == param
        assert torch.equal(
            metadata.quantization_scale,
            state_dict[f"_COREML_/{param}/quantization_scale"],
        )
        assert (
            metadata.quantization_n_bits
            == state_dict[f"_COREML_/{param}/quantization_n_bits"].item()
        )
        assert (
            metadata.compression_type == state_dict[f"_COREML_/{param}/compression_type"].tolist()
        )

        non_none_keys = [
            "quantization_n_bits",
            "quantization_scale",
            "param_name",
            "compression_type",
        ]
        for key, value in metadata.as_dict().items():
            if key not in non_none_keys:
                assert value is None


@pytest.mark.parametrize(
    "metadata_dict",
    [
        {
            "param_name": "weight",
            "zero_point": torch.rand(3, 1),
            "compression_type": ["pruning", "quantization"],
        },
    ],
)
def test_register(metadata_dict):
    module = torch.nn.Conv2d(3, 32, 3)
    metadata = CompressionMetadata.from_dict(metadata_dict)

    state_dict = module.state_dict()
    for key in metadata_dict:
        assert metadata._get_metadata_buffer_name(key) not in state_dict

    metadata.register(module)

    state_dict = module.state_dict()
    for key, value in metadata_dict.items():
        if key != "param_name":
            metadata_key = metadata._get_metadata_buffer_name(key)
            if key == "compression_type":
                metadata_value = torch.tensor([CompressionType[x].value for x in value])
            else:
                metadata_value = torch.tensor(value)
            assert metadata_key in state_dict
            assert torch.equal(state_dict[metadata_key], metadata_value)


def test_chaining_compression_type():
    module = torch.nn.Conv2d(3, 32, 3)
    metadata = CompressionMetadata(param_name="weight")
    metadata.compression_type = ["pruning"]

    metadata.register(module)

    buffer_name = metadata._get_metadata_buffer_name("compression_type")
    assert buffer_name in module.state_dict()
    assert torch.equal(module.state_dict()[buffer_name], torch.tensor([1]))

    metadata2 = CompressionMetadata(param_name="weight")
    metadata2.compression_type = ["palettization"]

    metadata2.register(module)
    assert buffer_name in module.state_dict()
    assert torch.equal(module.state_dict()[buffer_name], torch.tensor([1, 2]))


def test_register_metadata_version():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3), torch.nn.ReLU())
    assert METADATA_VERSION_BUFFER not in model.state_dict()
    register_metadata_version(model)
    assert METADATA_VERSION_BUFFER in model.state_dict()
    assert torch.equal(model.state_dict()[METADATA_VERSION_BUFFER], METADATA_VERSION)
