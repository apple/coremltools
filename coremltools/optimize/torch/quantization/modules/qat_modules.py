#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict as _OrderedDict
from typing import Type as _Type
from typing import Union as _Union

import torch as _torch
import torch.ao.nn.intrinsic as _nni
import torch.ao.nn.qat as _nnqat
import torch.ao.quantization as _aoquant
import torch.nn as _nn
import torch.nn.intrinsic.qat as _nniqat

import coremltools.optimize.torch.quantization.modules.conv_transpose as _qconv_transpose
import coremltools.optimize.torch.quantization.modules.conv_transpose_fused as _qconv_transpose_fused
import coremltools.optimize.torch.quantization.modules.fused_modules as _fuse


class _ConvAct(_torch.nn.Sequential):
    root_mod: _Type[_nn.Module]
    qat_mod: _Union[
        _nnqat.Conv1d,
        _nnqat.Conv2d,
        _nnqat.Conv3d,
        _qconv_transpose.ConvTranspose1d,
        _qconv_transpose.ConvTranspose2d,
        _qconv_transpose.ConvTranspose3d,
    ]
    fused_mod: _Union[
        _fuse.ConvAct1d,
        _fuse.ConvAct2d,
        _fuse.ConvAct3d,
        _fuse.ConvTransposeAct1d,
        _fuse.ConvTransposeAct2d,
        _fuse.ConvTransposeAct3d,
    ]

    def __init__(self, conv: _nn.Module, act: _nn.Module, qconfig: _aoquant.QConfig):
        super().__init__(_OrderedDict([("conv", conv), ("act", act)]))
        self.qconfig = qconfig

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return self.act(self.conv(x))

    @property
    def weight(self):
        return self.conv.weight

    @property
    def weight_fake_quant(self):
        return self.conv.weight_fake_quant

    @classmethod
    def from_float(cls, mod: _nn.Module):
        if isinstance(mod.conv, cls.qat_mod):
            conv = mod.conv

        else:
            assert isinstance(mod.conv, cls.root_mod), (
                f"Failed to convert module for QAT. "
                f"Expected module type {cls.root_mod}, "
                f"received type {type(mod.conv)}."
            )
            conv = cls.qat_mod.from_float(mod.conv)

        conv.activation_post_process = None
        return cls(conv, mod.act, mod.qconfig)

    def to_float(self) -> _nn.Module:
        return self.fused_mod(
            conv=self.conv.to_float(),
            act=self.act,
        )


class _ConvBnAct(_ConvAct):
    intr_mod: _Type[_nn.Module]
    qat_mod: _Union[
        _nniqat.ConvBn1d,
        _nniqat.ConvBn2d,
        _nniqat.ConvBn3d,
        _qconv_transpose_fused.ConvTransposeBn1d,
        _qconv_transpose_fused.ConvTransposeBn2d,
        _qconv_transpose_fused.ConvTransposeBn3d,
    ]
    fused_mod: _Union[
        _fuse.ConvAct1d,
        _fuse.ConvAct2d,
        _fuse.ConvAct3d,
        _fuse.ConvTransposeAct1d,
        _fuse.ConvTransposeAct2d,
        _fuse.ConvTransposeAct3d,
    ]

    @classmethod
    def from_float(cls, mod: _nn.Module):
        if isinstance(mod.conv, cls.intr_mod):
            conv = cls.qat_mod.from_float(mod.conv)
        else:
            conv = mod.conv
            assert isinstance(conv, cls.qat_mod), (
                f"Failed to convert module for QAT. "
                f"Expected module type {cls.qat_mod}, "
                f"received type {type(conv)}."
            )
        conv.activation_post_process = None
        return cls(conv, mod.act, mod.qconfig)


class ConvAct1d(_ConvAct):
    root_mod = _nn.Conv1d
    qat_mod = _nnqat.Conv1d
    fused_mod = _fuse.ConvAct1d
    pass


class ConvAct2d(_ConvAct):
    root_mod = _nn.Conv2d
    qat_mod = _nnqat.Conv2d
    fused_mod = _fuse.ConvAct2d
    pass


class ConvAct3d(_ConvAct):
    root_mod = _nn.Conv3d
    qat_mod = _nnqat.Conv3d
    fused_mod = _fuse.ConvAct3d
    pass


class ConvTransposeAct1d(_ConvAct):
    root_mod = _nn.ConvTranspose1d
    qat_mod = _qconv_transpose.ConvTranspose1d
    fused_mod = _fuse.ConvTransposeAct1d
    pass


class ConvTransposeAct2d(_ConvAct):
    root_mod = _nn.ConvTranspose2d
    qat_mod = _qconv_transpose.ConvTranspose2d
    fused_mod = _fuse.ConvTransposeAct2d
    pass


class ConvTransposeAct3d(_ConvAct):
    root_mod = _nn.ConvTranspose3d
    qat_mod = _qconv_transpose.ConvTranspose3d
    fused_mod = _fuse.ConvTransposeAct3d
    pass


class ConvBnAct1d(_ConvBnAct):
    intr_mod = _nni.ConvBn1d
    qat_mod = _nniqat.ConvBn1d
    fused_mod = _fuse.ConvAct1d
    pass


class ConvBnAct2d(_ConvBnAct):
    intr_mod = _nni.ConvBn2d
    qat_mod = _nniqat.ConvBn2d
    fused_mod = _fuse.ConvAct2d
    pass


class ConvBnAct3d(_ConvBnAct):
    intr_mod = _nni.ConvBn3d
    qat_mod = _nniqat.ConvBn3d
    fused_mod = _fuse.ConvAct3d
    pass


class ConvTransposeBnAct1d(_ConvBnAct):
    intr_mod = _fuse.ConvTransposeBn1d
    qat_mod = _qconv_transpose_fused.ConvTransposeBn1d
    fused_mod = _fuse.ConvTransposeAct1d
    pass


class ConvTransposeBnAct2d(_ConvBnAct):
    intr_mod = _fuse.ConvTransposeBn2d
    qat_mod = _qconv_transpose_fused.ConvTransposeBn2d
    fused_mod = _fuse.ConvTransposeAct2d
    pass


class ConvTransposeBnAct3d(_ConvBnAct):
    intr_mod = _fuse.ConvTransposeBn3d
    qat_mod = _qconv_transpose_fused.ConvTransposeBn3d
    fused_mod = _fuse.ConvTransposeAct3d
    pass


class LinearAct(_torch.nn.Sequential):
    def __init__(self, linear: _nnqat.Linear, act: _nn.Module, qconfig: _aoquant.QConfig):
        super().__init__(_OrderedDict([("linear", linear), ("act", act)]))
        self.qconfig = qconfig

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return self.act(self.linear(x))

    @property
    def weight(self):
        return self.linear.weight

    @property
    def weight_fake_quant(self):
        return self.linear.weight_fake_quant

    @classmethod
    def from_float(cls, mod: _fuse.LinearAct):
        if isinstance(mod.linear, _nnqat.Linear):
            linear = mod.linear

        else:
            assert isinstance(mod.linear, _nn.Linear), (
                f"Failed to convert module for QAT. "
                f"Expected module type {_nn.Linear}, "
                f"received type {type(mod.linear)}."
            )
            linear = _nnqat.Linear.from_float(mod.linear)

        linear.activation_post_process = None
        return cls(linear, mod.act, mod.qconfig)

    def to_float(self) -> _fuse.LinearAct:
        return _fuse.LinearAct(
            linear=self.linear.to_float(),
            act=self.act,
        )


def update_bn_stats(mod):
    """
    update_bn_stats methods updates BatchNorm statistics

    This method was originally defined in torch.nn.intrinsic.qat.modules.conv_fused. However, it limited the type of
    quantized fused modules for which BatchNorm statistics can be updated
    """
    if hasattr(mod, "update_bn_stats"):
        mod.update_bn_stats()


def freeze_bn_stats(mod):
    """
    freeze_bn_stats method freezes BatchNorm statistics

    This method was originally defined in torch.nn.intrinsic.qat.modules.conv_fused. However, it limited the type of
    quantized fused modules for which BatchNorm statistics can be frozen
    """
    if hasattr(mod, "freeze_bn_stats"):
        mod.freeze_bn_stats()
