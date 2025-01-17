#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest

from coremltools.optimize.torch._utils.torch_utils import (
    normalize_fsdp_module_name as _normalize_fsdp_module_name,
)


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
def test_normalize_fsdp_module_name(module_names):

    name, normalized_name = module_names
    assert _normalize_fsdp_module_name(name) == normalized_name
