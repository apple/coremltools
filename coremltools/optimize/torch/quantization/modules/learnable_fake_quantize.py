#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# Implementation for LearnableFakeQuantize class adapted from:
# - https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/_learnable_fake_quantize.py
# - https://github.com/ModelTC/MQBench/blob/main/mqbench/fake_quantize/lsq.py
# - https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fake_quantize.py

from typing import Any as _Any
from typing import List as _List

import torch as _torch
from torch.ao.quantization import FakeQuantizeBase as _FakeQuantizeBase
from torch.ao.quantization.observer import ObserverBase as _ObserverBase
from torch.ao.quantization.observer import _with_args

from coremltools.optimize.torch._utils.fsdp_utils import sync_tensor as _sync_tensor
from coremltools.optimize.torch.quantization._utils import (
    is_pytorch_defined_observer as _is_pytorch_defined_observer,
)
from coremltools.optimize.torch.quantization._utils import is_symmetric_quant as _is_symmetric_quant


def grad_scale(x: _torch.Tensor, scale: _torch.Tensor):
    x_scale = x * scale
    return (x - x_scale).detach() + x_scale


def round_pass(x: _torch.Tensor):
    return (x.round() - x).detach() + x


class LearnableFakeQuantize(_FakeQuantizeBase):
    """
    Class that handles the initialization and learning of quantization parameters
    as part of the Learnable Quantization algorithm.

    Please see the papers below for more details on the algorithm:
    - `Learned Step Size Quantization <https://arxiv.org/pdf/1902.08153>`_.
    - `LSQ+: Improving low-bit quantization through learnable offsets and better initialization <https://arxiv.org/pdf/2004.09576>`_.

    Args:
        observer (:obj:`torch.ao.quantization.observer.ObserverBase`): Observer to be used to initialize the
            quantization parameters.
        dtype (:py:class:`torch.dtype`): The dtype to use for quantizing the weights.
            Must be either :py:class:`torch.quint8` or :py:class:`torch.qint8`.
        qscheme (:obj:`torch.qscheme`): Type of quantization configuration to use.
        reduce_range (:obj:`bool`): When ``True``, quant_min and quant_max are adjusted to
            use one less bit of quantized precision.
    """

    scale: _torch.nn.Parameter
    zero_point: _torch.nn.Parameter

    def __init__(
        self,
        observer: _ObserverBase,
        dtype: _torch.dtype,
        qscheme: _torch.qscheme,
        reduce_range: bool = False,
        **observer_kwargs,
    ):
        super().__init__()

        assert dtype == _torch.quint8 or dtype == _torch.qint8
        self.num_bits_quantize = 8

        self.signed = dtype == _torch.qint8
        self.reduce_range = reduce_range

        num_bits = self.num_bits_quantize - (1 if self.reduce_range else 0)

        self.has_customized_qrange = self.reduce_range

        if "quant_min" in observer_kwargs and observer_kwargs["quant_min"] is not None:
            self.quant_min = observer_kwargs["quant_min"]
            self.has_customized_qrange = True
        else:
            if self.signed:
                self.quant_min = -(2 ** (num_bits - 1))
            else:
                self.quant_min = 0

        if "quant_max" in observer_kwargs and observer_kwargs["quant_max"] is not None:
            self.quant_max = observer_kwargs["quant_max"]
            self.has_customized_qrange = True
        else:
            if self.signed:
                self.quant_max = (2 ** (num_bits - 1)) - 1
            else:
                self.quant_max = (2**num_bits) - 1

        # Otherwise, standard quant_min and quant_max are used as appropriate for the datatype
        if self.has_customized_qrange:
            observer_kwargs["quant_min"] = self.quant_min
            observer_kwargs["quant_max"] = self.quant_max

        self.dtype = dtype
        self.qscheme = qscheme

        # ch_axis should be 0 for weights and 1 for 1D/2D/3D spatial activations
        # unless there are additional batch dimensions
        # ch_axis should be -1 if not per-channel quantization
        if "ch_axis" in observer_kwargs:
            self.ch_axis = observer_kwargs["ch_axis"]
        else:
            self.ch_axis = -1

        # If self.ch_axis != -1, then this needs to be updated later
        self.channel_count = 1

        self.activation_post_process = observer(
            dtype=self.dtype, qscheme=self.qscheme, **observer_kwargs
        )

        assert self.quant_min is not None
        assert self.quant_max is not None
        assert self.activation_post_process.quant_min is not None
        assert self.activation_post_process.quant_max is not None
        assert _torch.iinfo(self.activation_post_process.dtype).min is not None
        assert _torch.iinfo(self.activation_post_process.dtype).max is not None

        assert _torch.iinfo(self.activation_post_process.dtype).min <= self.quant_min
        assert self.quant_max <= _torch.iinfo(self.activation_post_process.dtype).max

        self.scale = _torch.nn.Parameter(_torch.ones(1))
        self.zero_point = _torch.nn.Parameter(
            _torch.zeros(1), requires_grad=not _is_symmetric_quant(self.qscheme)
        )

        self.register_buffer("eps", _torch.tensor([_torch.finfo(_torch.float32).eps]))

        self.device = None

        self.shape = None

        self.mask_shape = None

    def set_device(self, device: _torch.device):
        """
        Method that sets the device of the parameters.

        Args:
            device (:py:class:`torch.device`): Device where to hold learnable parameters.
        """

        self.to(device)
        self.device = device

    def set_ch_axis(self, ch_axis: int):
        """
        Method that sets the channel axis. This method should only be called before starting training.

        Args:
            ch_axis (:obj:`int`): Non-negative channel axis index.
        """

        assert ch_axis >= 0

        self.ch_axis = ch_axis
        if hasattr(self.activation_post_process, "ch_axis"):
            self.activation_post_process.ch_axis = self.ch_axis

    def set_param_shape(self, shape: _List[int]):
        """
        Method that sets the parameter shape. This method should only be called before starting training
        and after calling the set_ch_axis method.
        Should only be used for per-channel quantization.

        Args:
            shape (:obj:`List[int]`): Extent of the learnable quantization parameters.
        """

        assert self.ch_axis != -1

        self.shape = shape
        self.mask_shape = [1] * len(self.shape)
        self.mask_shape[self.ch_axis] = self.shape[self.ch_axis]
        self.channel_count = self.mask_shape[self.ch_axis]

        if self.device is None:
            self.scale = _torch.nn.Parameter(_torch.ones(self.mask_shape))
            self.zero_point = _torch.nn.Parameter(
                _torch.zeros(self.mask_shape),
                requires_grad=not _is_symmetric_quant(self.qscheme),
            )
        else:
            self.scale = _torch.nn.Parameter(_torch.ones(self.mask_shape, device=self.device))
            self.zero_point = _torch.nn.Parameter(
                _torch.zeros(self.mask_shape, device=self.device),
                requires_grad=not _is_symmetric_quant(self.qscheme),
            )

    @_torch.jit.export
    def calculate_qparams(self):
        """
        Method that computes, processes and returns the learned quantization parameters.
        """

        # Need to do this here because backward pass can update scale after a training step
        self.scale.data.clamp_(min=self.eps.item())

        # Need to do this here because backward pass can update zero_point after a training step
        if _is_symmetric_quant(self.qscheme) and self.signed:
            self.zero_point.data.zero_()
        elif _is_symmetric_quant(self.qscheme):
            assert not self.signed
            if not self.has_customized_qrange:
                self.zero_point.data.fill_(128.0)
            else:
                self.zero_point.data.fill_(float((self.quant_min + self.quant_max) // 2))
        else:
            self.zero_point.data.round_().clamp_(self.quant_min, self.quant_max)
            self.zero_point.data = self.zero_point.float().data

        scale = self.scale.detach()
        zero_point = self.zero_point.detach()

        zero_point = zero_point.int()

        # scale and zero-point are required to be 1-dimensional for export
        if self.ch_axis != -1:
            scale = _torch.squeeze(scale)
            zero_point = _torch.squeeze(zero_point)

        return scale, zero_point

    def forward(self, x: _torch.Tensor):
        if not isinstance(x, _torch.Tensor):
            return x

        if x.device is None:
            return x

        if self.device is None:
            self.set_device(x.device)

        if self.observer_enabled[0] == 1:
            self.activation_post_process(x.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()

            # PyTorch defined observers need to be force-synchronized before backward pass
            # for observer phase
            if _is_pytorch_defined_observer(self.activation_post_process):
                _scale = _sync_tensor(_scale)
                _zero_point = _sync_tensor(_zero_point)

            if self.ch_axis != -1:
                _scale = _scale.reshape(self.mask_shape)
                _zero_point = _zero_point.reshape(self.mask_shape)

            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point.float())

        if self.fake_quant_enabled[0] == 1:
            # Make sure scale and zero_point values are appropriate before executing fake_quant ops
            self.scale.data.clamp_(min=self.eps.item())

            if _is_symmetric_quant(self.qscheme) and self.signed:
                self.zero_point.data.zero_()
            elif _is_symmetric_quant(self.qscheme):
                assert not self.signed
                if not self.has_customized_qrange:
                    self.zero_point.data.fill_(128.0)
                else:
                    self.zero_point.data.fill_(float((self.quant_min + self.quant_max) // 2))
            else:
                self.zero_point.data.round_().clamp_(self.quant_min, self.quant_max)
                self.zero_point.data = self.zero_point.float().data

            computed_grad_scale = 1.0
            if self.ch_axis != -1:
                computed_grad_scale = 1.0 / (
                    (x.numel() / x.shape[self.ch_axis] * self.quant_max) ** 0.5
                )
            else:
                computed_grad_scale = 1.0 / ((x.numel() * self.quant_max) ** 0.5)
            mod_scale = grad_scale(self.scale, computed_grad_scale)

            # x_q = clamp(round(x / scale + zero_point), quant_min, quant_max)
            # x_r = (x_q - zero_point) * scale

            zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
            # There should be no scaling of zero_point for symmetric quantization
            mod_zero_point = (
                grad_scale(zero_point, computed_grad_scale)
                if not _is_symmetric_quant(self.qscheme)
                else zero_point
            )

            # Fake the zero_point being used to avoid DDP performance penalty
            # 1.0 * 128.0 or 1.0 * float((self.quant_min + self.quant_max) // 2)
            # for symmetric quant and not self.signed case
            include_zero_point = 0.0 if (_is_symmetric_quant(self.qscheme) and self.signed) else 1.0

            x = x / mod_scale + include_zero_point * mod_zero_point
            x = round_pass(x)
            x = _torch.clamp(x, self.quant_min, self.quant_max)
            x = (x - include_zero_point * mod_zero_point) * mod_scale

        else:
            # Fake the scale and zero_point being used to avoid DDP performance penalty
            x = x + 0.0 * self.scale + 0.0 * self.zero_point

        return x

    @_torch.jit.export
    def extra_repr(self):
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, observer_enabled={self.observer_enabled}, "
            f"quant_min={self.activation_post_process.quant_min}, quant_max={self.activation_post_process.quant_max}, "
            f"dtype={self.dtype}, qscheme={self.qscheme}, ch_axis={self.ch_axis}, "
            f"scale={self.scale}, zero_point={self.zero_point}"
        )

    def _save_to_state_dict(self, destination: _Any, prefix: _Any, keep_vars: _Any):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "scale"] = self.scale
        destination[prefix + "zero_point"] = self.zero_point

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
        # Removing this function throws an error that the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ["scale", "zero_point"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. New
                # buffers/parameters are created here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == "scale":
                    self.scale = _torch.nn.Parameter(_torch.zeros_like(val))
                else:
                    assert name == "zero_point"
                    self.zero_point = _torch.nn.Parameter(_torch.zeros_like(val))
                # For torchscript module we need to update the attributes here since we do not
                # call the "_load_from_state_dict" function defined module.py
                if _torch.jit.is_scripting():
                    if name == "scale":
                        self.scale.copy_(val)
                    else:
                        assert name == "zero_point"
                        self.zero_point.copy_(val)
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

    @classmethod
    def with_args(cls, **kwargs):
        fake_quant_constructor = _with_args(cls, **kwargs)
        # need to assign the correct module to fake_quantize
        # constructors to satisfy public v private requirements
        fake_quant_constructor.__module__ = (
            "coremltools.optimize.torch.quantization.learnable_fake_quantize"
        )
        return fake_quant_constructor
