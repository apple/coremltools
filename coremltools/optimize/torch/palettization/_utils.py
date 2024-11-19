#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Tuple as _Tuple

import torch as _torch
import torch.distributed as _dist


def vectorize(
    current_tensor, cluster_dim, vector_ch_axis=0
) -> _Tuple[_torch.Tensor, _torch.Tensor]:
    """
    Function to vectorize a tensor till the point where its numel is divisible by cluster_dim. The remaining parameters
    are returned as a pad.

    vector_ch_axis 0
    tensor with shape [x_1, x_2, x_3, ..., x_N] --> [x_N, x_2, x_3, ..., x_(N-1), x_1] --> [x_N*x_2*x_3* ... *x_(N-1)*x_1/cluster_dim, cluster_dim]

    vector_ch_axis 1
    tensor with shape [x_1, x_2, x_3, ..., x_N] --> [x_1, x_N, x_3, ..., x_(N-1), x_2] --> [x_1*x_N*x_3* ... *x_(N-1)*x_2/cluster_dim, cluster_dim]
    """
    num_misalignment = _torch.numel(current_tensor) % cluster_dim

    if cluster_dim > 1:
        current_tensor = current_tensor.transpose(vector_ch_axis, -1)

    pad = None
    if num_misalignment:
        current_tensor = current_tensor.flatten()
        pad = current_tensor[-num_misalignment:]
        current_tensor = current_tensor[:-num_misalignment]

    return current_tensor.reshape(-1, cluster_dim), pad


def devectorize(current_tensor, pad, target_size, cluster_dim, vector_ch_axis=0) -> _torch.Tensor:
    """
    Function to devectorize by tracing back the vectorize operation in the method above.

    vector_ch_axis 0
    target_size [x_1, x_2, ..., x_N]
    tensor with shape [x_1*x_2* ... *x_N/cluster_dim, cluster_dim] --> [x_N, x_2, x_3, ..., x_(N-1), x_1] --> [x_1, x_2, x_3, ..., x_N]

    vector_ch_axis 1
    target_size [x_1, x_2, ..., x_N]
    tensor with shape [x_1*x_2* ... *x_N/cluster_dim, cluster_dim] --> [x_1, x_N, x_3, ..., x_(N-1), x_2] --> [x_1, x_2, x_3, ..., x_N]
    """
    if pad is not None:
        current_tensor = _torch.cat([current_tensor.flatten(), pad])

    if cluster_dim > 1:
        if vector_ch_axis == 0:
            current_tensor = current_tensor.reshape(
                target_size[-1:] + target_size[1:-1] + target_size[0:1]
            ).transpose(0, -1)
        else:
            current_tensor = current_tensor.reshape(
                target_size[0:1] + target_size[-1:] + target_size[2:-1] + target_size[1:2]
            ).transpose(1, -1)

        return current_tensor

    return current_tensor.reshape(target_size)


def get_shard_list(length) -> list:
    """
    Function to generate shard_list for different partitions.
    """

    distributed_world_size = (
        _dist.get_world_size() if _dist.is_available() and _dist.is_initialized() else 1
    )
    shard_size = max(1, length // distributed_world_size)
    shard_list = list(range(0, length, shard_size))
    if len(shard_list) > distributed_world_size:
        shard_list = shard_list[:distributed_world_size] + [length]
    else:
        shard_list += [length] * (distributed_world_size + 1 - len(shard_list))

    return shard_list
