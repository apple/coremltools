#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch as _torch

from coremltools.optimize.torch._utils.torch_utils import (
    normalize_fsdp_module_name as _normalize_fsdp_module_name,
)
from coremltools.optimize.torch._utils.torch_utils import (
    transform_to_ch_axis as _transform_to_ch_axis,
)


@pytest.mark.parametrize(
    "input_shape, ch_axis, expected_shape",
    [
        ((1, 2), 0, (1, 2)),
        ((1, 2), 1, (2, 1)),
        ((3, 2, 1), 0, (3, 2)),
        ((2, 3, 1), 1, (3, 2)),
        ((3, 1, 2), 2, (2, 3)),
        ((1, 2, 3, 4), 0, (1, 24)),
        ((4, 3, 2, 1), 1, (3, 8)),
        ((3, 4, 2, 1), 2, (2, 12)),
        ((2, 3, 1, 4), 3, (4, 6)),
    ],
)
def test_torch_utils_transform_to_ch_axis(input_shape, ch_axis, expected_shape):
    input_tensor = _torch.ones(input_shape)

    output_tensor = _transform_to_ch_axis(input_tensor, ch_axis)

    assert output_tensor.shape == expected_shape


@pytest.mark.parametrize(
    "module_names",
    [
        ("conv1", "conv1"),
        ("_fsdp_wrapped_module.conv1", "conv1"),
        (
            "model._fsdp_module._fsdp_wrapped_module.layers.layer_0.feed_forward._fsdp_wrapped_module.hidden_transform.linear_0",
            "model.layers.layer_0.feed_forward.hidden_transform.linear_0",
        ),
        ("model._fsd_module.conv1", "model._fsd_module.conv1"),
    ],
)
def test_torch_utils_normalize_fsdp_module_name(module_names):

    name, normalized_name = module_names
    assert _normalize_fsdp_module_name(name) == normalized_name
