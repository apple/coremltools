#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict as _OrderedDict
from typing import Type as _Type

import torch.ao.nn.quantized.reference as _reference
import torch.nn as _nn


class _QuantizedConvAct(_nn.Sequential):
    ref_quant_mod: _Type[_nn.Module]

    def __init__(self, conv: _nn.Module, act: _nn.Module):
        super().__init__(_OrderedDict([("conv", conv), ("act", act)]))

    @classmethod
    def from_float(cls, float_conv_act, weight_qparams):
        conv = cls.ref_quant_mod.from_float(float_conv_act.conv, weight_qparams)
        return cls(conv, float_conv_act.act)


class QuantizedConvAct1d(_QuantizedConvAct):
    ref_quant_mod = _reference.Conv1d
    pass


class QuantizedConvAct2d(_QuantizedConvAct):
    ref_quant_mod = _reference.Conv2d
    pass


class QuantizedConvAct3d(_QuantizedConvAct):
    ref_quant_mod = _reference.Conv3d
    pass


class QuantizedConvTransposeAct1d(_QuantizedConvAct):
    ref_quant_mod = _reference.ConvTranspose1d
    pass


class QuantizedConvTransposeAct2d(_QuantizedConvAct):
    ref_quant_mod = _reference.ConvTranspose2d
    pass


class QuantizedConvTransposeAct3d(_QuantizedConvAct):
    ref_quant_mod = _reference.ConvTranspose3d
    pass


class QuantizedLinearAct(_nn.Sequential):
    def __init__(self, linear: _reference.Linear, act: _nn.Module):
        super().__init__(_OrderedDict([("linear", linear), ("act", act)]))

    @classmethod
    def from_float(cls, float_linear_act, weight_qparams):
        linear = _reference.Linear.from_float(float_linear_act.linear, weight_qparams)
        return cls(linear, float_linear_act.act)
