#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import torch as _torch
import torch.nn.functional as _F


class _FakePalettizationTensorHook:
    """
    _FakePalettizationTensorHook is the class to assist in using CPU when we only want to utilize a certain percentage
    of the GPU memory.
    """
    gc_trigger = None

    def __init__(
        self, size_list, use_cpu: bool = False, name: str = None, palett_tau: float = 0.0001
    ):
        self.name = name
        self.size_list = size_list
        self.tensor_list = [None] * len(size_list)
        self.device_list = [None] * len(size_list)
        self.use_cpu = use_cpu
        self.palett_tau = palett_tau

    def init_pack(self, x: _torch.Tensor):
        """
        Method that initialises packing and saving values to CPU.
        """
        if x.size() in self.size_list:
            idx = self.size_list.index(x.size())

            if self.tensor_list[idx] is None:
                self.device_list[idx] = x.device

                if self.use_cpu:
                    self.tensor_list[idx] = _torch.empty(
                        x.size(), dtype=x.dtype, layout=x.layout, pin_memory=True
                    )
                    self.tensor_list[idx].copy_(x)
                else:
                    self.tensor_list[idx] = x

            elif _torch.equal(self.tensor_list[idx][0].to(self.device_list[idx]), x[0]):
                pass
            else:
                assert False

            return idx

        return x

    def init_unpack(self, x: _torch.Tensor):
        """
        Method that initialises un-packing and retrieving values from CPU.
        """
        if isinstance(x, int):
            idx = x

            assert self.tensor_list[idx] is not None
            self.tensor_list[idx] = self.tensor_list[idx].to(
                self.device_list[idx], non_blocking=True
            )
            return self.tensor_list[idx]

        return x

    def reuse_pack(self, x: _torch.Tensor):
        """
        Method to pack reused variables on to CPU.
        """
        if x.layout != _torch.sparse_coo and x.size() in self.size_list:
            idx = self.size_list.index(x.size())

            assert self.size_list[idx] is not None

            header = self.tensor_list[idx][0].to(self.device_list[idx])

            if _torch.equal(x[0], -header * header / self.palett_tau):
                return idx, "x_c_dist"
            elif _torch.equal(x[0], _F.softmax(-header * header / self.palett_tau)):
                return idx, "softmax"
            else:
                return x.to_sparse(), "sparse"

        return x

    def reuse_unpack(self, x: _torch.Tensor):
        """
        Method to unpack reused variables from CPU.
        """
        if isinstance(x, tuple):
            obj, op = x
            if isinstance(obj, int):
                idx = obj
                assert self.tensor_list[idx] is not None
                self.tensor_list[idx] = self.tensor_list[idx].to(self.device_list[idx])

                if op == "softmax":
                    val = self.tensor_list[idx] * self.tensor_list[idx] / self.palett_tau
                    return _F.softmax(-val, dim=1)
                elif op == "x_c_dist":
                    return -self.tensor_list[idx] * self.tensor_list[idx] / self.palett_tau
                elif op == "transpose":
                    return self.tensor_list[idx].T
                else:
                    assert False
            elif op == "sparse":
                return obj.to_dense()
        return x

    def debug_hook(self, x: _torch.Tensor):
        return x
