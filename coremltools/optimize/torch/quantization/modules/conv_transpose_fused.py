#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# Original implementation from https://github.com/pytorch/pytorch/blob/main/torch/ao/nn/intrinsic/qat/modules/conv_fused.py
# Copyright (c) 2016 Facebook, Inc (Adam Paszke)

from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import TypeVar as _TypeVar

import torch
import torch.ao.nn.intrinsic as nni
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _single, _triple
from torch.nn.utils import fuse_conv_bn_weights

from coremltools.optimize.torch.quantization.modules.fused_modules import (
    ConvTransposeBn1d as iConvTransposeBn1d,
)
from coremltools.optimize.torch.quantization.modules.fused_modules import (
    ConvTransposeBn2d as iConvTransposeBn2d,
)
from coremltools.optimize.torch.quantization.modules.fused_modules import (
    ConvTransposeBn3d as iConvTransposeBn3d,
)

_BN_CLASS_MAP = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}

__all__ = ["ConvTransposeBn1d", "ConvTransposeBn2d", "ConvTransposeBn3d"]
MOD = _TypeVar("MOD", bound=nn.modules.conv._ConvTransposeNd)


class _ConvTransposeBnNd(nni.qat.modules.conv_fused._ConvBnNd):

    _FLOAT_MODULE = MOD

    def __init__(
        self,
        # ConvNd args
        in_channels: int,
        out_channels: int,
        kernel_size: _Tuple[int, ...],
        stride: _Tuple[int, ...],
        padding: _Tuple[int, ...],
        dilation: _Tuple[int, ...],
        transposed: bool,
        output_padding: _Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        # BatchNormNd args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn: bool = False,
        qconfig=None,
        dim: int = 2,
    ):
        nni.qat.modules.conv_fused._ConvBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
            dim,
        )

    def forward(self, input, output_size: _Optional[_List[int]] = None):
        assert isinstance(self.padding, _Tuple)
        return self._forward(input, output_size)

    def _forward(self, input, output_size):
        # if self._enable_slow_path_for_better_numerical_stability:
        #     return self._forward_slow(input)
        return self._forward_approximate(input, output_size)

    def _forward_approximate(self, input, output_size):
        """
        Taken from nni.qat.modules.conv_fused._ConvBnNd
        Changes made for weight_shape and bias_shape

        Approximated method to fuse conv and bn. It requires only one forward pass.
        conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std
        """
        assert self.bn.running_var is not None
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[1] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        # using zero bias here since the bias for original conv
        # will be added later
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias, dtype=input.dtype)
        else:
            zero_bias = torch.zeros(
                self.out_channels, device=scaled_weight.device, dtype=input.dtype
            )

        conv = self._conv_forward(input, scaled_weight, zero_bias, output_size)
        conv_orig = conv / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            conv_orig = conv_orig + self.bias.reshape(bias_shape)
        conv = self.bn(conv_orig)
        return conv

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
        # The ignore is because _FLOAT_MODULE is a TypeVar here where the bound
        # has no __name__ (code is fine though)
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat." + cls.__name__ + ".from_float only works for " + cls._FLOAT_MODULE.__name__
        )  # type: ignore[attr-defined]
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        conv, bn = mod[0], mod[1]

        qat_convbn = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            output_padding=conv.output_padding,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            eps=bn.eps,
            momentum=bn.momentum,
            freeze_bn=False,
            qconfig=qconfig,
        )
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.bn.weight = bn.weight
        qat_convbn.bn.bias = bn.bias
        qat_convbn.bn.running_mean = bn.running_mean
        qat_convbn.bn.running_var = bn.running_var
        # mypy error: Cannot determine type of 'num_batches_tracked'
        qat_convbn.bn.num_batches_tracked = bn.num_batches_tracked  # type: ignore[has-type]
        return qat_convbn

    def to_float(self):
        """
        transpose applied to weights

        taken from torch.ao.nn.intrinsic.qat.modules.conv_fused._ConvBnNd
        """

        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined]
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.bias is not None,
            self.dilation,
            self.padding_mode,
        )
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())

        if cls._FLOAT_BN_MODULE:  # type: ignore[attr-defined]
            # fuse bn into conv
            assert self.bn.running_var is not None and self.bn.running_mean is not None
            conv.weight.data = conv.weight.data.transpose(1, 0)
            conv.weight, conv.bias = fuse_conv_bn_weights(
                conv.weight,
                conv.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias,
            )
            conv.weight.data = conv.weight.data.transpose(1, 0)

        if cls._FLOAT_RELU_MODULE:  # type: ignore[attr-defined]
            modules = []
            modules.append(conv)
            relu = cls._FLOAT_RELU_MODULE()  # type: ignore[attr-defined]
            modules.append(relu)
            conv_relu = cls._FUSED_FLOAT_MODULE(*modules)  # type: ignore[attr-defined]
            conv_relu.train(self.training)
            return conv_relu
        else:
            conv.train(self.training)
            return conv


class ConvTransposeBn1d(_ConvTransposeBnNd, nn.ConvTranspose1d):
    r"""
    A ConvTransposeBn1d module is a module fused from ConvTranspose1d and BatchNorm1d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.ConvTranspose1d` and
    :class:`torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.ConvTranspose1d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """

    _FLOAT_BN_MODULE = nn.BatchNorm1d
    _FLOAT_RELU_MODULE: None = None
    _FLOAT_MODULE = iConvTransposeBn1d
    _FLOAT_CONV_MODULE = nn.ConvTranspose1d

    def __init__(
        self,
        # ConvTranspose1d args
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        # BatchNorm1d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding_ = _single(output_padding)

        _ConvTransposeBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding_,
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
            dim=1,
        )

    def _conv_forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: _Optional[Tensor],
        output_size: _Optional[_List[int]] = None,
    ) -> Tensor:

        num_spatial_dims = 1
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,
        )  # type: ignore[arg-type]
        return F.conv_transpose1d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class ConvTransposeBn2d(_ConvTransposeBnNd, nn.ConvTranspose2d):
    r"""
    A ConvTransposeBn2d module is a module fused from ConvTranspose2d and BatchNorm2d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.ConvTranspose2d` and
    :class:`torch.nn.BatchNorm2d`.

    Similar to :class:`torch.nn.ConvTranspose2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """

    _FLOAT_BN_MODULE = nn.BatchNorm2d
    _FLOAT_RELU_MODULE: None = None
    _FLOAT_MODULE = iConvTransposeBn2d
    _FLOAT_CONV_MODULE = nn.ConvTranspose2d

    def __init__(
        self,
        # ConvTranspose2d args
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        # BatchNorm2d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding_ = _pair(output_padding)

        _ConvTransposeBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding_,
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
            dim=2,
        )

    def _conv_forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: _Optional[Tensor],
        output_size: _Optional[_List[int]] = None,
    ) -> Tensor:

        num_spatial_dims = 2
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,
        )  # type: ignore[arg-type]
        return F.conv_transpose2d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class ConvTransposeBn3d(_ConvTransposeBnNd, nn.ConvTranspose3d):
    r"""
    A ConvTransposeBn3d module is a module fused from ConvTranspose2d and BatchNorm3d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.ConvTranspose3d` and
    :class:`torch.nn.BatchNorm3d`.

    Similar to :class:`torch.nn.ConvTranspose3d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """

    _FLOAT_BN_MODULE = nn.BatchNorm3d
    _FLOAT_RELU_MODULE: None = None
    _FLOAT_MODULE = iConvTransposeBn3d
    _FLOAT_CONV_MODULE = nn.ConvTranspose3d

    def __init__(
        self,
        # ConvTranspose3d args
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        output_padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        # BatchNorm3d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding_ = _triple(output_padding)

        _ConvTransposeBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding_,
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
            dim=3,
        )

    def _conv_forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: _Optional[Tensor],
        output_size: _Optional[_List[int]] = None,
    ) -> Tensor:

        num_spatial_dims = 3
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,
        )  # type: ignore[arg-type]
        return F.conv_transpose3d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
