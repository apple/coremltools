#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# Implementation for reset_min_max_vals, _save_to_state_dict and _load_from_state_dict
# has been adapted from https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/observer.py

# Implementation for EMAMinMaxObserver has been adapted from
# https://github.com/ModelTC/MQBench/blob/main/mqbench/observer.py

import distutils as _distutils
from typing import Any as _Any
from typing import Dict as _Dict
from typing import Optional as _Optional

import torch as _torch
import torch.ao.quantization as _aoquant
from torch.ao.quantization.observer import (
    UniformQuantizationObserverBase as _UniformQuantizationObserverBase,
)

from coremltools.optimize.torch._utils.fsdp_utils import sync_tensor as _sync_tensor
from coremltools.optimize.torch._utils.torch_utils import get_torch_version as _get_torch_version
from coremltools.optimize.torch._utils.torch_utils import (
    transform_to_ch_axis as _transform_to_ch_axis,
)
from coremltools.optimize.torch.quantization._utils import is_symmetric_quant as _is_symmetric_quant


class NoopObserver(_aoquant.NoopObserver):
    """
    Extends aoquant.NoopObserver to add support for accepting factory_kwargs which are
    passed to it during qconfig.weight() creation in QAT Conv/Linear modules.
    """

    def __init__(
        self,
        dtype: _torch.dtype = _torch.float16,
        custom_op_name: str = "",
        factory_kwargs: _Dict[str, _Any] = None,
    ):
        super().__init__(dtype, custom_op_name)


class CustomObserverBase(_UniformQuantizationObserverBase):
    """
    The CustomObserverBase class is a base class for observers that perform
    calibration for quantization by computing representative
    minimum and maximum values on a per-tensor or per-channel basis.

    Args:
        dtype (:py:class:`torch.dtype`): The dtype to use for quantizing the weights.
            Must be either :py:class:`torch.quint8` or :py:class:`torch.qint8`.
        qscheme (:obj:`torch.qscheme`): Type of quantization configuration to use.
        reduce_range (:obj:`bool`): When ``True``, quant_min and quant_max are adjusted to
            use one less bit of quantized precision.
        quant_min (:obj:`Optional[int]`): Minimum integer quantization value. Defaults to ``None``.
        quant_max (:obj:`Optional[int]`): Maximum integer quantization value. Defaults to ``None``.
        ch_axis (:obj:`int`): Channel axis to use for per-channel quantization. Use ``-1`` for
            per-tensor quantization and non-negative values for per-channel quantization. Defaults to ``-1``.
    """

    min_val: _torch.Tensor
    max_val: _torch.Tensor

    def __init__(
        self,
        dtype: _torch.dtype = _torch.quint8,
        qscheme: _torch.qscheme = _torch.per_tensor_affine,
        reduce_range: bool = False,
        quant_min: _Optional[int] = None,
        quant_max: _Optional[int] = None,
        ch_axis: int = -1,
        factory_kwargs: _Any = None,
        is_dynamic: bool = False,
    ):
        kwargs = {}
        if _get_torch_version(_torch.__version__) >= _distutils.version.StrictVersion("2.2"):
            kwargs["is_dynamic"] = is_dynamic

        super(CustomObserverBase, self).__init__(
            dtype, qscheme, reduce_range, quant_min, quant_max, **kwargs
        )

        self.ch_axis = ch_axis

        self.register_buffer("min_val", _torch.tensor(float("inf")))
        self.register_buffer("max_val", _torch.tensor(float("-inf")))

        self.device = None

        self.synchronize = True

    def calculate_qparams(self):
        """
        Compute the quantization parameters using the min_val and max_val buffers.
        Return the scale and zero-point quantization parameters.
        """

        if self.synchronize:
            self.min_val.data = _sync_tensor(self.min_val, _torch.distributed.ReduceOp.MIN).data
            self.max_val.data = _sync_tensor(self.max_val, _torch.distributed.ReduceOp.MAX).data

        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)

        if self.synchronize:
            scale.data = _sync_tensor(scale).data
            zero_point.data = _sync_tensor(zero_point).data

        # scale and zero-point are required to be 1-dimensional for export
        if self.ch_axis != -1:
            scale = _torch.squeeze(scale)
            zero_point = _torch.squeeze(zero_point)

        # See https://pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MinMaxObserver.html
        # for information on how scale and zp are computed in PyTorch
        if _is_symmetric_quant(self.qscheme):
            if self.dtype == _torch.quint8 and not self.has_customized_qrange:
                zero_point.data.fill_(128.0)
            elif self.dtype == _torch.quint8:
                assert self.has_customized_qrange
                zero_point.data.fill_(float((self.quant_min + self.quant_max) // 2))
            else:
                assert self.dtype == _torch.qint8
                zero_point.data.zero_()
        else:
            zero_point.data.round_().clamp_(self.quant_min, self.quant_max)
            zero_point.data = zero_point.float().data

        zero_point = zero_point.int()

        return scale, zero_point

    @_torch.jit.export
    def reset_min_max_vals(self):
        """
        Resets the min/max values.
        """

        self.min_val.copy_(_torch.tensor(float("inf")))
        self.max_val.copy_(_torch.tensor(float("-inf")))

    def _save_to_state_dict(self, destination: _Any, prefix: _Any, keep_vars: _Any):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "min_val"] = self.min_val
        destination[prefix + "max_val"] = self.max_val

    def _load_from_state_dict(
        self,
        state_dict: _Any,
        prefix: _Any,
        local_metadata: _Any,
        strict: _Any,
        missing_keys: _Any,
        unexpected_keys: _Any,
        error_msgs: _Any,
    ):
        min_val_name, max_val_name = prefix + "min_val", prefix + "max_val"

        if min_val_name in state_dict:
            if state_dict[min_val_name].shape == _torch.Size([0]):
                state_dict[min_val_name] = _torch.tensor(float("inf"))
        if max_val_name in state_dict:
            if state_dict[max_val_name].shape == _torch.Size([0]):
                state_dict[max_val_name] = _torch.tensor(float("-inf"))

        local_state = ["min_val", "max_val"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EMAMinMaxObserver(CustomObserverBase):
    """
    The EMAMinMaxObserver observer class performs calibration for quantization
    by computing representative minimum and maximum values on a per-tensor or per-channel basis.
    Each forward call results in a computed minimum and maximum, which is then averaged using
    the exponential moving average (EMA) formula to get the aggregated minimum and maximum.

    Args:
        dtype (:py:class:`torch.dtype`): The dtype to use for quantizing the weights.
            Must be either :py:class:`torch.quint8` or :py:class:`torch.qint8`.
        qscheme (:obj:`torch.qscheme`): Type of quantization configuration to use.
        reduce_range (:obj:`bool`): When ``True``, quant_min and quant_max are adjusted to
            use one less bit of quantized precision.
        quant_min (:obj:`Optional[int]`): Minimum integer quantization value. Defaults to ``None``.
        quant_max (:obj:`Optional[int]`): Maximum integer quantization value. Defaults to ``None``.
        ch_axis (:obj:`int`): Channel axis to use for per-channel quantization. Use ``-1`` for
            per-tensor quantization and non-negative values for per-channel quantization. Defaults to ``-1``.
        ema_ratio (:obj:`float`): Constant to be used for EMA computation.
            Aggregated min/max val is ema_ratio * prev_val + (1 - ema_ratio) * curr_val. Defaults to ``0.9``.
    """

    def __init__(
        self,
        dtype: _torch.dtype = _torch.quint8,
        qscheme: _torch.qscheme = _torch.per_tensor_affine,
        reduce_range: bool = False,
        quant_min: _Optional[int] = None,
        quant_max: _Optional[int] = None,
        ch_axis: int = -1,
        ema_ratio: float = 0.9,
        factory_kwargs: _Any = None,
        is_dynamic: bool = False,
    ):
        super(EMAMinMaxObserver, self).__init__(
            dtype,
            qscheme,
            reduce_range,
            quant_min,
            quant_max,
            ch_axis,
            is_dynamic=is_dynamic,
        )

        self.ema_ratio = ema_ratio

    def forward(self, x: _torch.Tensor):
        if not isinstance(x, _torch.Tensor) or x.numel() == 0:
            return x

        if self.device is None:
            self.device = x.device
            self.min_val = self.min_val.to(self.device)
            self.max_val = self.max_val.to(self.device)

        x_detach = x.detach()
        x_detach = x_detach.to(self.min_val.dtype)

        if self.ch_axis == -1:
            min_val_cur, max_val_cur = _torch.aminmax(x_detach)
        else:
            y = _transform_to_ch_axis(x_detach, self.ch_axis)
            min_val_cur, max_val_cur = _torch.aminmax(y, dim=1)

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.ema_ratio + min_val_cur * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + max_val_cur * (1.0 - self.ema_ratio)

        return x

    @_torch.jit.export
    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}, ch_axis={self.ch_axis}, ema_ratio={self.ema_ratio}"
