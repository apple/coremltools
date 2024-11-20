#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy

import pytest
import torch

from coremltools.optimize.torch.palettization._utils import devectorize, vectorize


@pytest.mark.parametrize(
    "cluster_dim_reshape_expected_shape_expected_first_row",
    [
        (2, (2, 3, 4), (12, 2), torch.tensor([0, 12])),
        (2, (4, 3, 2), (12, 2), torch.tensor([0, 6])),
        (3, (4, 3, 2), (8, 3), torch.tensor([0, 6, 12])),
    ],
)
def test_vectorize_cluster_dim_gt_1(
    cluster_dim_reshape_expected_shape_expected_first_row,
):
    (
        cluster_dim,
        reshape,
        expected_shape,
        expected_first_row,
    ) = cluster_dim_reshape_expected_shape_expected_first_row
    partition_weight_tensor = torch.arange(24).reshape(reshape)
    vectorized_weight_tensor, _ = vectorize(partition_weight_tensor, cluster_dim)
    assert tuple(vectorized_weight_tensor.shape), expected_shape
    assert torch.equal(vectorized_weight_tensor[0], expected_first_row)


@pytest.mark.parametrize(
    "cluster_dim_reshape",
    [
        (2, (2, 3, 4)),
        (2, (4, 3, 2)),
        (3, (2, 3, 4)),
        (3, (4, 3, 2)),
    ],
)
def test_devectorize_cluster_dim_gt_1(cluster_dim_reshape):
    cluster_dim, reshape = cluster_dim_reshape
    partition_weight_tensor = torch.arange(24).reshape(reshape)
    pwt_copy = copy.deepcopy(partition_weight_tensor)
    vectorized_weight_tensor, _ = vectorize(partition_weight_tensor, cluster_dim)
    devectorized_partition_weight_tensor = devectorize(
        vectorized_weight_tensor, None, torch.Size(reshape), cluster_dim
    )
    assert torch.equal(pwt_copy, devectorized_partition_weight_tensor)
