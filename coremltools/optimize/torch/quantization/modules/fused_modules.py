#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict as _OrderedDict
from typing import Union as _Union

import torch as _torch
import torch.nn as _nn
import torch.nn.intrinsic as _nni


class _ConvAct(_torch.nn.Sequential):
    def __init__(self, conv: _nn.Module, act: _nn.Module):
        super().__init__(_OrderedDict([("conv", conv), ("act", act)]))

    @property
    def weight(self):
        return self.conv.weight


class _ConvBnAct(_torch.nn.Sequential):
    intr_mod: _Union[_nni.ConvBn1d, _nni.ConvBn2d, _nni.ConvBn3d]

    def __init__(self, conv: _nn.Module, bn: _nn.Module, act: _nn.Module):
        super().__init__(_OrderedDict([("conv", self.intr_mod(conv, bn)), ("act", act)]))

    @property
    def weight(self):
        return self.conv.weight


class ConvAct1d(_ConvAct):
    pass

class ConvAct2d(_ConvAct):
    pass


class ConvAct3d(_ConvAct):
    pass


class ConvBnAct1d(_ConvBnAct):
    intr_mod = _nni.ConvBn1d
    pass


class ConvBnAct2d(_ConvBnAct):
    intr_mod = _nni.ConvBn2d
    pass


class ConvBnAct3d(_ConvBnAct):
    intr_mod = _nni.ConvBn3d
    pass


class LinearAct(_torch.nn.Sequential):
    def __init__(self, linear: _nn.Linear, act: _nn.Module):
        super().__init__(_OrderedDict([("linear", linear), ("act", act)]))

    @property
    def weight(self):
        return self.linear.weight
