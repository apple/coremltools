#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch

from coremltools.optimize.torch._utils.fsdp_utils import (
    ModuleWrapPolicy,
    SizeBasedWrapPolicy,
    sync_tensor,
)


@pytest.mark.parametrize(
    "reduce_op",
    [
        torch.distributed.ReduceOp.AVG,
        torch.distributed.ReduceOp.SUM,
        torch.distributed.ReduceOp.PRODUCT,
        torch.distributed.ReduceOp.MIN,
        torch.distributed.ReduceOp.MAX,
    ],
)
def test_fsdp_utils_sync_tensor(reduce_op):
    # Does not actually test the op when on non-CPU device and torch.distributed is initialized.
    # Just tests that "passthrough" works

    input_tensor = torch.ones((1, 2, 3, 4))

    output_tensor = sync_tensor(input_tensor, reduce_op)

    assert output_tensor.shape == input_tensor.shape


def test_fsdp_utils_module_wrap_policy():
    """
    Test constructor for underlying FSDP policy is called with correct arguments
    """
    module_classes = [torch.nn.Linear, torch.nn.Conv2d]
    policy = ModuleWrapPolicy(module_classes=module_classes)
    policy = policy.get_policy()
    assert policy._module_classes == set(module_classes)


def test_fsdp_utils_size_based_policy():
    """
    Test constructor for underlying FSDP policy is called with correct arguments
    """
    min_num_params = 100
    policy = SizeBasedWrapPolicy(min_num_params=min_num_params)
    policy = policy.get_policy()
    assert policy.keywords["min_num_params"] == min_num_params
