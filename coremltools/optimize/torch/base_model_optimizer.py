#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from collections import UserDict as _UserDict
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import torch as _torch

from coremltools.optimize.torch._utils.python_utils import get_str as _get_str
from coremltools.optimize.torch.optimization_config import OptimizationConfig as _OptimizationConfig

_logger = _logging.getLogger(__name__)


class _Report(_UserDict):
    def __repr__(self):
        if len(self.data) < 1:
            return ""
        column_names = list(self.data.values())[0].keys()
        column_names = ["name"] + list(column_names)
        print_list = [column_names]
        print_list += [
            [f"{key}"] + [_get_str(val[cn]) for cn in column_names[1:]]
            for key, val in self.data.items()
        ]
        col_size = [max(map(len, col)) for col in zip(*print_list)]
        ret_str = [
            " | ".join(
                f"{' ' * (col_size[idx] - len(val))}{val}" for idx, val in enumerate(print_list[0])
            )
        ]
        ret_str += [" | ".join(f"{'-' * cs}" for cs in col_size)]
        for pl in print_list[1:]:
            ret_str.append(
                " | ".join(f"{' ' * (col_size[idx] - len(val))}{val}" for idx, val in enumerate(pl))
            )
        return "\n".join(ret_str)


class BaseModelOptimizer(_ABC):
    _supported_modules: _Tuple

    def __init__(self, model: _torch.nn.Module, config: _Optional[_OptimizationConfig] = None):
        self._model = model
        self._config = config
        self._step_count = 0

    @_abstractmethod
    def prepare(self, *args, **kwargs) -> _torch.nn.Module:
        raise NotImplementedError()

    @_abstractmethod
    def step(self):
        raise NotImplementedError()

    @_abstractmethod
    def finalize(
        self, model: _Optional[_torch.nn.Module] = None, inplace: bool = False
    ) -> _torch.nn.Module:
        raise NotImplementedError()

    @_abstractmethod
    def report(self) -> _Report:
        raise NotImplementedError()

    @property
    def supported_modules(self):
        return self._supported_modules
