#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# Original implementation from https://github.com/IST-DASLab/sparsegpt
# Copyright 2023 IST Austria Distributed Algorithms and Systems Lab. All Rights Reserved.

import torch as _torch

_normal_float_palette = {
    # The 4 bit numbers are copied from QLoRA paper: https://arxiv.org/abs/2305.14314
    4: _torch.tensor(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ]
    ),
    # The 3 bit numbers are obtained from bitsandbytes: https://github.com/TimDettmers/bitsandbytes/blob/18e827d666fa2b70a12d539ccedc17aa51b2c97c/bitsandbytes/functional.py#L236
    3: _torch.tensor([-1.0, -0.4786292, -0.21714179, 0.0, 0.1609302, 0.33791524, 0.562617, 1.0]),
}


def quantize(
    x: _torch.Tensor,
    scale: _torch.Tensor,
    zero: _torch.Tensor,
    max_q: _torch.Tensor,
    enable_normal_float: bool,
):
    """
    Quantize ``x`` by rounding and clamping the value using specified
    quantization parameters.
    """
    n_bits = _torch.log2(max_q + 1).item()
    if enable_normal_float:
        if n_bits not in _normal_float_palette:
            raise ValueError(f"Normal float format is not supported for {n_bits}.")
        nf_palette = _normal_float_palette[n_bits]
        nf_palette = nf_palette.to(x.device)
        distances = _torch.cdist((x / scale).view(-1, 1), nf_palette.unsqueeze(0).T)
        indices = _torch.min(distances, dim=1).indices
        return scale * nf_palette[indices].view(x.shape)
    else:
        q = _torch.clamp(_torch.round(x / scale) + zero, 0, max_q)
        return scale * (q - zero)


class Quantizer(_torch.nn.Module):
    """
    A module for quantizing tensors by scaling, shifting, rounding and clamping them such that the values
    are represented in ``n_bits`` precision.
    """

    def __init__(
        self,
        n_bits: int,
        per_channel: bool = True,
        symmetric: bool = False,
        enable_normal_float: bool = False,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        max_shrink: float = 0.8,
        group_rows: int = 1,
    ):
        super().__init__()
        self._per_channel = per_channel
        self._symmetric = symmetric
        self._enable_normal_float = enable_normal_float
        self._mse = mse
        self._norm = norm
        self._grid = grid
        self._max_shrink = max_shrink
        self._group_rows = group_rows
        self.register_buffer("max_q", _torch.tensor(2**n_bits - 1))
        self.register_buffer("scale", _torch.zeros(1))
        self.register_buffer("zero", _torch.zeros(1))

    def find_params(self, x, weight=False):
        """
        Compute quantization parameters.
        """
        device = x.device
        self.max_q = self.max_q.to(device)

        shape = x.shape
        if self._per_channel:
            if weight:
                x = x.flatten(1)
                if self._group_rows > 1:
                    x = x.reshape((x.shape[0] // self._group_rows, -1))
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = _torch.zeros(x.shape[0], device=device)
        x_min = _torch.minimum(x.min(1)[0], tmp)
        x_max = _torch.maximum(x.max(1)[0], tmp)

        if self._symmetric:
            x_max = _torch.maximum(_torch.abs(x_min), x_max)
            tmp = x_min < 0
            if _torch.any(tmp):
                x_min[tmp] = -x_max[tmp]
        tmp = (x_min == 0) & (x_max == 0)
        x_min[tmp] = -1
        x_max[tmp] = +1

        if self._enable_normal_float:
            self.scale = _torch.maximum(x_max, abs(x_min))
        else:
            self.scale = (x_max - x_min) / self.max_q

        if self._symmetric:
            self.zero_point = _torch.full_like(self.scale, (self.max_q + 1) / 2)
        else:
            self.zero_point = _torch.round(-x_min / self.scale)

        if self._mse:
            best = _torch.full([x.shape[0]], float("inf"), device=device)
            for i in range(int(self._max_shrink * self._grid)):
                p = 1 - i / self._grid
                x_min1 = p * x_min
                x_max1 = p * x_max
                scale1 = (x_max1 - x_min1) / self.max_q
                zero_point1 = (
                    _torch.round(-x_min1 / scale1) if not self._symmetric else self.zero_point
                )
                q = quantize(
                    x,
                    scale1.unsqueeze(1),
                    zero_point1.unsqueeze(1),
                    self.max_q,
                    self._enable_normal_float,
                )
                q -= x
                q.abs_()
                q.pow_(self._norm)
                err = _torch.sum(q, 1)
                tmp = err < best
                if _torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero_point[tmp] = zero_point1[tmp]
        if not self._per_channel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero_point = self.zero_point.repeat(tmp)

        if weight:
            if self._group_rows > 1:
                self.scale = self.scale.unsqueeze(1).repeat(1, self._group_rows)
                self.zero_point = self.zero_point.unsqueeze(1).repeat(1, self._group_rows)
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero_point = self.zero_point.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero_point = self.zero_point.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero_point = self.zero_point.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero_point = self.zero_point.unsqueeze(0)

    def quantize(self, x):
        """
        Quantize ``x`` using pre-computed quantization parameters.
        """
        if self.ready():
            return quantize(x, self.scale, self.zero_point, self.max_q, self._enable_normal_float)
        return x

    def enabled(self):
        """
        Returns ``True`` if quantization is enabled.
        """
        return self.max_q > 0

    def ready(self):
        """
        Returns ``True`` if quantization parameters have been computed.
        """
        return _torch.all(self.scale != 0)
