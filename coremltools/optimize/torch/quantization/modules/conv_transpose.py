#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# Original implementation from https://github.com/pytorch/pytorch/blob/main/torch/ao/nn/qat/modules/conv.py

from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import TypeVar as _TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as _Tensor
from torch.ao.nn.intrinsic import _FusedModule
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _single, _triple

__all__ = ["ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"]
MOD = _TypeVar("MOD", bound=nn.modules.conv._ConvTransposeNd)


class _ConvTransposeNd(torch.ao.nn.qat.modules.conv._ConvNd):

    _FLOAT_MODULE = MOD

    def __init__(
        self,
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
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
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
            qconfig,
            **factory_kwargs
        )
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    @staticmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module

        Args:
           `mod`: a float module, either produced by torch.ao.quantization utilities
           or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__  # type: ignore[attr-defined]
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        if issubclass(type(mod), _FusedModule):
            mod = mod[0]  # type: ignore[index]
        qconfig = mod.qconfig

        qat_conv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            output_padding=mod.output_padding,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            qconfig=qconfig,
        )
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv

    def to_float(self):
        """This works for both single qat conv, and the qat conv - relu modules
        to convert the qat module to a floating point module
        """
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined, operator]
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,  # type: ignore[arg-type]
            stride=self.stride,  # type: ignore[arg-type]
            padding=self.padding,  # type: ignore[arg-type]
            dilation=self.dilation,  # type: ignore[arg-type]
            output_padding=self.output_padding,
            groups=self.groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
        )
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())
        # conv relu
        if issubclass(cls, _FusedModule):
            modules = [conv]
            assert hasattr(cls, "_FLOAT_RELU_MODULE")
            relu = cls._FLOAT_RELU_MODULE()  # type: ignore[attr-defined]
            modules.append(relu)
            fused = cls._FLOAT_MODULE(*modules)  # type: ignore[arg-type, attr-defined, operator]
            fused.train(self.training)
            return fused
        else:
            return conv


class ConvTranspose1d(_ConvTransposeNd, nn.ConvTranspose1d):
    r"""
    A ConvTranspose1d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as`torch.nn.ConvTranspose1d`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
    for documentation.

    Similar to :class:`~torch.nn.ConvTranspose1d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE = nn.ConvTranspose1d
    _FLOAT_CONV_MODULE = nn.ConvTranspose1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = _single(padding)
        dilation_ = _single(dilation)
        output_padding_ = _single(output_padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            transposed=True,
            output_padding=output_padding_,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: _Tensor, output_size: _Optional[_List[int]] = None) -> _Tensor:
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose1d")

        assert isinstance(self.padding, _Tuple)
        # One cannot replace _List by _Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `_Tuple[T, ...]`.
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
            self.weight_fake_quant(self.weight),
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )

    @classmethod
    def from_float(cls, mod):
        return super().from_float(cls, mod)


class ConvTranspose2d(_ConvTransposeNd, nn.ConvTranspose2d):
    r"""
    A ConvTranspose2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.ConvTranspose2d`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    for documentation.

    Similar to `torch.nn.ConvTranspose2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE = nn.ConvTranspose2d
    _FLOAT_CONV_MODULE = nn.ConvTranspose2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        output_padding_ = _pair(output_padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            transposed=True,
            output_padding=output_padding_,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: _Tensor, output_size: _Optional[_List[int]] = None) -> _Tensor:
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose1d")

        assert isinstance(self.padding, _Tuple)
        # One cannot replace _List by _Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `_Tuple[T, ...]`.
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
            self.weight_fake_quant(self.weight),
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )

    @classmethod
    def from_float(cls, mod):
        return super().from_float(cls, mod)


class ConvTranspose3d(_ConvTransposeNd, nn.ConvTranspose3d):
    r"""
    A ConvTranspose3d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.ConvTranspose3d`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html
    for documentation.

    Similar to `torch.nn.ConvTranspose3d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """

    _FLOAT_MODULE = nn.ConvTranspose3d
    _FLOAT_CONV_MODULE = nn.ConvTranspose3d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        output_padding: _size_3_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_3_t = 1,
        padding_mode: str = "zeros",
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = _triple(padding)
        dilation_ = _triple(dilation)
        output_padding_ = _triple(output_padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            transposed=True,
            output_padding=output_padding_,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: _Tensor, output_size: _Optional[_List[int]] = None) -> _Tensor:
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose1d")

        assert isinstance(self.padding, _Tuple)
        # One cannot replace _List by _Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `_Tuple[T, ...]`.
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
            self.weight_fake_quant(self.weight),
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )

    @classmethod
    def from_float(cls, mod):
        return super().from_float(cls, mod)
