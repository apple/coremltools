#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Tuple as _Tuple

import torch as _torch
import torch.distributed as _dist

def vectorize(current_tensor, cluster_dim) -> _Tuple[_torch.Tensor, _torch.Tensor]:
    """
    Function to vectorize a tensor till the point where its numel is divisible by cluster_dim. The remaining parameters
    are returned as a pad.
    """
    num_misalignment = _torch.numel(current_tensor) % cluster_dim

    if cluster_dim > 1:
        current_tensor = current_tensor.transpose(0, -1)

    pad = None
    if num_misalignment:
        current_tensor = current_tensor.flatten()
        pad = current_tensor[-num_misalignment:]
        current_tensor = current_tensor[:-num_misalignment]

    return current_tensor.reshape(-1, cluster_dim), pad


def devectorize(current_tensor, pad, target_size, cluster_dim) -> _torch.Tensor:
    """
    Function to devectorize by tracing back the vectorize operation in the method above.
    """
    if pad is not None:
        current_tensor = _torch.cat([current_tensor.flatten(), pad])

    if cluster_dim > 1:
        current_tensor = current_tensor.reshape(_torch.Size(tuple(target_size)[::-1]))
        current_tensor = current_tensor.transpose(0, -1)
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
