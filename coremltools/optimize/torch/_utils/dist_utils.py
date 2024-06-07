#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os as _os

import torch as _torch
import torch.distributed as _dist


def ddp_setup(rank: int, world_size: int):
    """
    Set environment variables which are used for initializing distributed
    process group for :py:class:`DistributedDataParallel`.

    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    _os.environ["MASTER_ADDR"] = "localhost"
    _os.environ["MASTER_PORT"] = "12355"
    _os.environ["WORLD_SIZE"] = f"{world_size}"
    _os.environ["RANK"] = f"{rank}"
    _os.environ["LOCAL_RANK"] = f"{rank}"
    _torch.cuda.set_device(f"cuda:{rank}")
    _dist.init_process_group("nccl", rank=rank, world_size=world_size)


def is_leader():
    """
    Returns ``True`` if the rank of the current process is 0.
    """
    if _dist.is_initialized():
        return _dist.get_rank() == 0
    return True
