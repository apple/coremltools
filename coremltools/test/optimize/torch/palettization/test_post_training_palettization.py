#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy

import pytest
import torch
import torch.functional as F
import torch.nn as nn

from coremltools.optimize.torch._utils.metadata_utils import CompressionMetadata
from coremltools.optimize.torch.palettization import (
    PostTrainingPalettizer,
    PostTrainingPalettizerConfig,
    SKMPalettizer,
    SKMPalettizerConfig,
)


@pytest.fixture
def simple_model():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    return Net()


def test_no_config(simple_model):
    # Would do a 4-bit kmeans for all supported modules after giving a warning
    ptpalettizer = PostTrainingPalettizer(simple_model)
    palettized_model = ptpalettizer.compress()
    assert palettized_model.conv1.weight.unique().size()[0] == 16
    assert palettized_model.conv2.weight.unique().size()[0] == 16
    assert palettized_model.fc1.weight.unique().size()[0] == 16
    assert palettized_model.fc2.weight.unique().size()[0] == 16
    assert palettized_model.fc3.weight.unique().size()[0] == 16


@pytest.mark.parametrize(
    "config_dict,expected_output",
    [
        (
            {"global_config": {"n_bits": 4}},
            ["==16", "==16", "==16", "==16", "==16"],
        ),
        (
            {
                "module_name_configs": {
                    "conv1": {"n_bits": 4},
                    "fc1": {"n_bits": 2},
                },
            },
            ["==16", ">16", "==4", ">4", ">4"],
        ),
        (
            {
                "module_type_configs": {
                    nn.Conv2d: {"n_bits": 4},
                    nn.Linear: {"n_bits": 2},
                },
            },
            ["==16", "==16", "==4", "==4", "==4"],
        ),
        (
            {
                "module_type_configs": {
                    # Invalid cluster_dim gets ignored.
                    # Conv2d should be skipped
                    nn.Conv2d: {"n_bits": 4, "cluster_dim": 5},
                    nn.Linear: {"n_bits": 2},
                },
            },
            [">16", ">16", "==4", "==4", "==4"],
        ),
    ],
)
def test_post_training_palettization_dict_config(simple_model, config_dict, expected_output):
    dict_config = PostTrainingPalettizerConfig.from_dict(config_dict)
    ptpalettizer = PostTrainingPalettizer(simple_model, dict_config)
    palettized_model = ptpalettizer.compress()
    i = 0
    for name, mod in palettized_model.named_modules():
        if hasattr(mod, "weight"):
            assert eval(f"mod.weight.unique().size()[0] {expected_output[i]}")
            i += 1


@pytest.mark.parametrize(
    "config_dict,expected_output",
    [
        (
            {
                "module_name_configs": {
                    "conv1": {
                        "n_bits": 4,
                        "granularity": "per_tensor",
                        "cluster_dim": 3,
                    },
                    "conv2": {
                        "n_bits": 4,
                        "granularity": "per_tensor",
                        "cluster_dim": 3,
                    },
                    "fc3": {
                        "n_bits": 2,
                        "granularity": "per_tensor",
                        "cluster_dim": 2,
                    },
                },
            },
            ["==16", ">16", "==4"],
        ),
    ],
)
def test_post_training_vector_palettization_dict_config(simple_model, config_dict, expected_output):
    dict_config = PostTrainingPalettizerConfig.from_dict(config_dict)
    ptpalettizer = PostTrainingPalettizer(simple_model, dict_config)
    palettized_model = ptpalettizer.compress()
    i = 0
    for name, mod in palettized_model.named_modules():
        # Only validate the layers that get palettized.
        if name in config_dict["module_name_configs"] and hasattr(mod, "weight"):
            _cluster_dim = config_dict["module_name_configs"][name]["cluster_dim"]
            weight_reshaped = mod.weight.flatten(1).transpose(0, 1).reshape(-1, _cluster_dim)
            unique_vector = torch.unique(weight_reshaped, dim=0)
            assert eval(f"len(unique_vector) {expected_output[i]}")
            i += 1


@pytest.mark.parametrize(
    "config_dict",
    [
        {
            "n_bits": 4,
            "granularity": "per_tensor",
        },
        {
            "n_bits": 4,
            "granularity": "per_grouped_channel",
            "group_size": 4,
        },
        {
            "n_bits": 4,
            "cluster_dim": 4,
        },
        {
            "n_bits": 4,
            "granularity": "per_grouped_channel",
            "group_size": 4,
            "enable_per_channel_scale": True,
        },
    ],
)
@pytest.mark.parametrize(
    "lut_dtype",
    [torch.int8, torch.uint8],
)
@pytest.mark.parametrize(
    "layer",
    ["conv2", "fc2"],
)
def test_ptp_int_lut(simple_model, config_dict, lut_dtype, layer):
    config_dict["lut_dtype"] = lut_dtype
    module_config = {"module_name_configs": {layer: config_dict}}
    config = PostTrainingPalettizerConfig.from_dict(module_config)
    ptpalettizer = PostTrainingPalettizer(simple_model, config)
    palettized_model = ptpalettizer.compress()

    submodule = palettized_model.get_submodule(layer)
    metadata_dict = CompressionMetadata.from_state_dict(submodule.state_dict())
    metadata = metadata_dict["weight"]
    assert metadata.quantization_n_bits == 8
    scale = metadata.quantization_scale
    zp = metadata.zero_point
    lut = metadata.lut

    if lut_dtype == torch.int8:
        assert zp is None
        lut_quant = lut / scale
        assert torch.min(lut_quant).int() >= -127
        assert torch.max(lut_quant).int() <= 128
    else:
        assert zp is not None
        lut_quant = lut / scale + zp
        assert torch.min(lut_quant).int() >= 0
        assert torch.max(lut_quant).int() <= 254


def loss_fn(model, input):
    out = model(input)
    return nn.functional.mse_loss(out, torch.rand(1, 10))


def test_compute_sensitivity_single_worker_mutability(mnist_model, mnist_example_input):
    config = {"global_config": {"n_bits": 4}}
    skm_config = SKMPalettizerConfig.from_dict(config)
    palettizer = SKMPalettizer(mnist_model, skm_config)

    state_dict_before = copy.deepcopy(palettizer._model.state_dict())

    def calibration_loader():
        yield mnist_example_input

    palettizer.compute_sensitivity(
        dataloader=calibration_loader(), loss_fn=loss_fn, num_sensitivity_workers=1
    )

    state_dict_after = palettizer._model.state_dict()
    assert len(state_dict_before) == len(state_dict_after)
    for key in state_dict_before:
        assert torch.equal(state_dict_before[key], state_dict_after[key])
