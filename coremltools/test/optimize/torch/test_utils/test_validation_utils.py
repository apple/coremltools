#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch

from coremltools.optimize.torch._utils.validation_utils import (
    ConfigValidator,
    validate_param_config,
)
from coremltools.optimize.torch.palettization import (
    ModulePostTrainingPalettizerConfig,
    ModuleSKMPalettizerConfig,
)
from coremltools.optimize.torch.quantization import ModulePostTrainingQuantizerConfig


@pytest.mark.parametrize(
    "config, expectation",
    [
        (
            ModulePostTrainingPalettizerConfig(
                n_bits=4, granularity="per_grouped_channel", group_size=4
            ),
            True,
        ),
        (
            ModulePostTrainingPalettizerConfig(n_bits=4, granularity="per_tensor", cluster_dim=3),
            False,
        ),
        (
            ModulePostTrainingPalettizerConfig(n_bits=4, granularity="per_tensor", cluster_dim=4),
            True,
        ),
    ],
)
def test_validate_param_config(config, expectation):
    module = torch.nn.Conv2d(16, 32, 5)
    result = validate_param_config(
        "weight",
        module.weight,
        module,
        config,
        ["palettization_group_size", "palettization_cluster_dim"],
    )
    if expectation:
        assert result is not None
    else:
        assert result is None


def test_validate_no_check():
    module = torch.nn.Conv2d(3, 16, 5)
    config = ModuleSKMPalettizerConfig()
    validator = ConfigValidator("weight", module.weight, module, config)
    with pytest.raises(AssertionError):
        validator.validate(["invalid_check"])

@pytest.mark.parametrize(
    "group_size, channel_axis, expectation",
    [
        pytest.param(4, None, True, id="default_axis"),
        pytest.param(4, 0, True, id="axis_0"),
        pytest.param(4, 1, True, id="axis_1"),
        pytest.param(5, None, False, id="default_indivisible_group_size"),
        pytest.param(5, 0, False, id="axis_0_indivisible_group_size"),
        pytest.param(5, 1, False, id="axis_1_indivisible_group_size"),
    ],
)
def test_validate_palettization_group_size(group_size, channel_axis, expectation):
    module = torch.nn.Conv2d(16, 32, 5)
    if channel_axis:
        config = ModuleSKMPalettizerConfig(
            n_bits=4,
            granularity="per_grouped_channel",
            group_size=group_size,
            channel_axis=channel_axis,
        )
    else:
        config = ModuleSKMPalettizerConfig(
            n_bits=4,
            granularity="per_grouped_channel",
            group_size=group_size,
        )
    validator = ConfigValidator("weight", module.weight, module, config)
    assert validator.validate(["palettization_group_size"]) == expectation


@pytest.mark.parametrize(
    "block_size, sanitized_block_size, expectation",
    [
        pytest.param(4, (1, 4), True, id="default_axis_int_block_size"),
        pytest.param((1, 4), (1, 4), True, id="tuple_with_per_channel"),
        pytest.param((4, 16), (4, 16), True, id="tuple_block_size"),
        pytest.param((4, 16, 5, 5), (4, 16), True, id="tuple_block_size_greater_than_ndim"),
        pytest.param((0, 16), -1, False, id="per_block_without_per_channel"),
        pytest.param((0, 0), -1, False, id="no_blocking_tuple"),
        pytest.param(0, -1, False, id="no_blocking_int"),
        pytest.param(5, -1, False, id="non_divisible_block_size_int"),
        pytest.param((5, 5), -1, False, id="non_divisible_block_size_tuple"),
        pytest.param((5, 16), -1, False, id="non_divisible_block_size_tuple_axis_0"),
        pytest.param((4, 5), -1, False, id="non_divisible_block_size_tuple_axis_1"),
    ],
)
def test_validate_quantization_block_size(block_size, sanitized_block_size, expectation):
    module = torch.nn.Conv2d(16, 32, 5)
    config = ModulePostTrainingQuantizerConfig(
        weight_dtype="int4", granularity="per_block", block_size=block_size
    )
    validator = ConfigValidator("weight", module.weight, module, config)
    assert validator.validate(["quantization_block_size"]) == expectation

    if expectation is True:
        assert validator.config.block_size == sanitized_block_size


@pytest.mark.parametrize(
    "cluster_dim, expectation",
    [
        pytest.param(None, True, id="cluster_dim_unspecified"),
        pytest.param(1, True, id="cluster_dim_scalar"),
        pytest.param(4, True, id="cluster_dim_valid_1"),
        pytest.param(8, True, id="cluster_dim_valid_2"),
        pytest.param(3, False, id="cluster_dim_invalid_1"),
        pytest.param(5, False, id="cluster_dim_invalid_1"),
    ],
)
def test_validate_palettization_cluster_dim(cluster_dim, expectation):
    module = torch.nn.Conv2d(3, 16, 5)
    config = ModulePostTrainingPalettizerConfig(
        n_bits=4, granularity="per_tensor", cluster_dim=cluster_dim
    )
    validator = ConfigValidator("weight", module.weight, module, config)
    assert validator.validate(["palettization_cluster_dim"]) == expectation
