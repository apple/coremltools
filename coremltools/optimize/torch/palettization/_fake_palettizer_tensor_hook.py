#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import gc
from typing import Callable as _Callable
from typing import Tuple as _Tuple

import torch as _torch
import torch.distributed as _dist

from ._utils import get_shard_list as _get_shard_list

MAX_RECURSION_DEPTH = 10


class _FakePalettizerTensorHook:
    """
    _FakePalettizerTensorHook is the custom hook that implements many of the tensor packing and unpacking
    techniques illustrated in the paper `eDKM: An Efficient and Accurate Train-time Weight Clustering for Large
    Language Models <https://arxiv.org/pdf/2309.00964.pdf>`_
    """

    SOFTMAX_BACKWARD = "SoftmaxBackward"
    CLAMP_BACKWARD = "ClampBackward"
    DIST_BACKWARD = "EuclideanDistBackward"
    TRANS_BACKWARD = "TransposeBackward"
    STACK_BACKWARD = "StackBackward"
    INDEX_BACKWARD = "IndexBackward"
    DIV_BACKWARD = "DivBackward"
    SLICE_BACKWARD = "SliceBackward"
    VIEW_BACKWARD = "ViewBackward"
    EXPAND_BACKWARD = "ExpandBackward"
    RESHAPE_BACKWARD = "ReshapeAliasBackward"
    TOCOPY_BACKWARD = "ToCopyBackward"

    gc_trigger = None
    last_report = {}

    def __init__(
        self,
        zero_threshold,
        device,
        min_size=0,
        max_mem=1.0,
        use_unique=False,
        use_shard=False,
    ):
        self.min_size = max(min_size, 64)
        self.max_mem = max_mem
        self.tensor_dict = {}
        self.tensor_counter = {}
        self.total_requested = 0
        self.total_allocated = 0
        self.use_unique = use_unique
        self.use_shard = use_shard
        self.pack_counter = -1
        self.device = device
        self.zero_threshold = zero_threshold

        t = _torch.cuda.get_device_properties(device).total_memory
        a = _torch.cuda.memory_allocated(device)

        self.use_cpu = (a / t) > abs(self.max_mem) and hasattr(_torch.autograd, "graph")
        if self.use_cpu:
            if self.__class__.gc_trigger is None:
                self.__class__.gc_trigger = True

        if self.__class__.gc_trigger:
            gc.collect()

    def _copy_to_device(self, x) -> _torch.Tensor:
        if self.use_cpu:
            packed = _torch.empty(x.size(), dtype=x.dtype, layout=x.layout, pin_memory=True)
            packed.copy_(x, non_blocking=True)
            return packed

        return x

    def _unique_tensor(self, x) -> _Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        if x.size(1) <= 1 or x.size(0) <= 1024:
            return x

        y, y_i = x.float().unique(return_inverse=True, dim=0)
        y_base = 0

        y = y.to(x.dtype)
        y = self._copy_to_device(y)

        max_y_size = y.size(0)

        if max_y_size >= _torch.iinfo(_torch.int16).max:
            y_base = max_y_size // 2
            y_i -= y_base
            max_y_size = y_base + 1

        y_i = _lower_int(y_i, 0, max_y_size)
        y_i = self._copy_to_device(y_i)

        return y, y_i, y_base

    def _compress_tensor(self, x, dtype) -> list:
        if x.numel() <= self.min_size:
            return x

        if x.dim() > 1:
            x = x.flatten(end_dim=-2)

        world_size = _dist.get_world_size()
        rank = _dist.get_rank()

        if len(x) < world_size or not self.use_shard:
            x = x.to(dtype)
            if self.use_unique:
                x = self._unique_tensor(x)
            return x

        shard_list = _get_shard_list(len(x))

        tensor_list = [None] * world_size
        shard = x[shard_list[rank] : shard_list[rank + 1]].to(dtype)

        if self.use_unique:
            tensor_list[rank] = self._unique_tensor(shard)
        else:
            tensor_list[rank] = self._copy_to_device(shard)

        for i in range(world_size):
            shard = x[shard_list[i] : shard_list[i + 1]]
            if i != rank:
                tensor_list[i] = {"size": shard.size(), "dtype": dtype}

        return tensor_list

    def pack(self, x) -> _Tuple[str, _Callable, _torch.device, _torch.Tensor]:
        """
        Function that will be called every time an operation saves a tensor for backward.
        """
        key = None
        op = lambda z: z.view(size)
        if x.numel() <= self.min_size:
            return x

        x_clone = x.clone() if self.max_mem <= 0 else None
        device = x.device
        size = x.size()

        if x.dtype.is_floating_point:
            grad_fn_list = []
            full_grad_fn_list = []
            c_grad_fn = x.grad_fn

            while len(grad_fn_list) < 2:
                if c_grad_fn:
                    str_grad_fn = str(type(c_grad_fn))

                    full_grad_fn_list.append(str_grad_fn)

                    if (
                        self.__class__.RESHAPE_BACKWARD in str_grad_fn
                        or self.__class__.TOCOPY_BACKWARD in str_grad_fn
                        or self.__class__.EXPAND_BACKWARD in str_grad_fn
                    ):
                        pass
                    else:
                        grad_fn_list.append(str_grad_fn)

                    c_grad_fn = c_grad_fn.next_functions[0][0] if c_grad_fn.next_functions else None
                else:
                    break

            if key is None:
                for _ in range(len(grad_fn_list), 2):
                    grad_fn_list.append("None")

                if (
                    self.__class__.SOFTMAX_BACKWARD in grad_fn_list[0]
                    and self.__class__.DIV_BACKWARD in grad_fn_list[1]
                ):
                    key = "softmax" + f".{self.pack_counter}"
                elif (
                    self.__class__.CLAMP_BACKWARD in grad_fn_list[0]
                    and self.__class__.SOFTMAX_BACKWARD in grad_fn_list[1]
                ):
                    key = "softmax" + f".{self.pack_counter}"
                    op = lambda z: z.view(size).clamp(min=self.zero_threshold)
                elif self.__class__.DIST_BACKWARD in grad_fn_list[0]:
                    self.pack_counter += 1
                    key = "x_c_dist" + f".{self.pack_counter}"
                elif (
                    self.__class__.VIEW_BACKWARD in grad_fn_list[0]
                    and self.__class__.DIST_BACKWARD in grad_fn_list[1]
                ):
                    key = "x_c_dist" + f".{self.pack_counter}"
                elif (
                    (
                        self.__class__.VIEW_BACKWARD in grad_fn_list[0]
                        and self.__class__.STACK_BACKWARD in grad_fn_list[1]
                    )
                    or (
                        self.__class__.STACK_BACKWARD in grad_fn_list[0]
                        and self.__class__.INDEX_BACKWARD in grad_fn_list[1]
                    )
                    or (
                        self.__class__.STACK_BACKWARD in grad_fn_list[0]
                        and self.__class__.SLICE_BACKWARD in grad_fn_list[1]
                    )
                ):
                    key = "X.b" + f".{-1}"
                elif (
                    self.__class__.TRANS_BACKWARD in grad_fn_list[0]
                    and self.__class__.STACK_BACKWARD in grad_fn_list[1]
                ):
                    key = "X.b" + f".{-1}"
                    if key in self.tensor_dict:
                        size = x.mT.size()
                        op = lambda z: z.reshape(size).mT
                    else:
                        key = None

            if key is None:
                key = self._compress_tensor(x, x.dtype)
            elif key not in self.tensor_dict:
                w = self._compress_tensor(x, x.dtype)
                self.tensor_dict[key] = w
        else:
            key = self._compress_tensor(x, _torch.uint8)
            op = lambda z: z.to(device, _torch.int32)

        return key, op, device, x_clone

    def unpack(self, x) -> _torch.Tensor:
        """
        Function that will be called to return a
         value to compute a new tensor, which is the one actually used during the backward pass.
        """
        if isinstance(x, tuple):
            key, op, device, y = x

            look_up = isinstance(key, str)
            if look_up:
                v = self.tensor_dict[key]
            else:
                v = key

            v = _decompress_tensor(v, device)

            if look_up:
                self.tensor_dict[key] = v

            x = op(v)

        return x


def _lower_int(x, x_min=None, x_max=None) -> _torch.Tensor:
    if x_min is None:
        x_min, x_max = x.min(), x.max()
    for t in [_torch.uint8, _torch.int8, _torch.int16, _torch.int32]:
        if _torch.iinfo(t).bits >= _torch.iinfo(x.dtype).bits:
            break
        if _torch.iinfo(t).min <= x_min and x_max <= _torch.iinfo(t).max:
            x = x.to(t)
            break
    return x


def _deunique_tensor(x, device) -> _torch.Tensor:
    y, y_i, y_base = x
    y = y.to(device, non_blocking=True)
    y_i = y_i.to(_torch.int32)
    if y_base > 0:
        y_i += y_base
    return y[y_i]


def _decompress_tensor(x, device) -> _torch.Tensor:
    if not isinstance(x, list):
        if isinstance(x, tuple):
            x = _deunique_tensor(x, device=device)
        return x

    distributed_world_size = _dist.get_world_size()
    distributed_rank = _dist.get_rank()
    for i in range(distributed_world_size):
        if isinstance(x[i], dict):
            x[i] = _torch.empty(**x[i], device=device)
        else:
            if isinstance(x[i], tuple):
                x[i] = _deunique_tensor(x[i], device=device)
            else:
                x[i] = x[i].to(device, non_blocking=True)

    _dist.all_gather(x[:distributed_world_size], x[distributed_rank])
    return _torch.concat(x, dim=0)
