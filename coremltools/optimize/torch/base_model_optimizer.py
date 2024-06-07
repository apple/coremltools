#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy as _copy
import logging as _logging
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from collections import UserDict as _UserDict
from typing import Iterable as _Iterable
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Type as _Type

import torch as _torch

from coremltools.optimize.torch._utils.python_utils import get_str as _get_str
from coremltools.optimize.torch.optimization_config import OptimizationConfig as _OptimizationConfig

_logger = _logging.getLogger(__name__)


class _Report(_UserDict):
    """
    A dictionary with pretty printing.
    """
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
    """
    An abstract base class for implementing optimizers.
    """
    _supported_modules: _Tuple[_Type[_torch.nn.Module]]

    def __init__(self, model: _torch.nn.Module, config: _Optional[_OptimizationConfig] = None):
        self._model = model
        self._config = config

    @_abstractmethod
    def report(self) -> _Report:
        raise NotImplementedError()

    @property
    def supported_modules(self) -> _Tuple[_Type[_torch.nn.Module]]:
        return self._supported_modules

    def _get_model_for_compression(self, inplace: bool):
        return self._model if inplace else _copy.deepcopy(self._model)


class BaseTrainingTimeModelOptimizer(BaseModelOptimizer):
    """
    An abstract base class for implementing optimization algorithms which
    are integrated in model training pipelines. These optimizers simulate
    model compression and learn compression parameters during model training.
    """

    def __init__(self, model: _torch.nn.Module, config: _Optional[_OptimizationConfig] = None):
        super().__init__(model, config)
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


class BasePostTrainingModelOptimizer(BaseModelOptimizer):
    """
    An abstract base class for implementing optimization algorithms which
    perform zero-shot compression, after a model has been trained. These
    optimizers do no need any data to perform compression.
    """

    def __init__(self, model: _torch.nn.Module, config: _Optional[_OptimizationConfig] = None):
        super().__init__(model, config)
        self._uncompressed_model = None

    def compress(self, *args, inplace: bool = False, **kwargs) -> _torch.nn.Module:
        # if inplace is True:
        #   self._uncompressed_model -> deep copy of model passed by user
        #   self._model -> model passed by user
        # if inplace is False:
        #   self._uncompressed_model -> model passed by user
        #   self._model -> deep copy of model passed by user
        self._uncompressed_model = self._get_model_for_compression(inplace=not inplace)
        self._model = self._get_model_for_compression(inplace=inplace)
        return self._model


class BaseDataCalibratedModelOptimizer(BaseModelOptimizer):
    """
    An abstract base class for optimization algorithms which use calibration data
    to compress models.
    """

    def __init__(self, model: _torch.nn.Module, config: _Optional[_OptimizationConfig] = None):
        super().__init__(model, config)
        self._uncompressed_model = None

    def compress(
        self, dataloader: _Iterable, *args, inplace: bool = False, **kwargs
    ) -> _torch.nn.Module:
        # if inplace is True:
        #   self._uncompressed_model -> deep copy of model passed by user
        #   self._model -> model passed by user
        # if inplace is False:
        #   self._uncompressed_model -> model passed by user
        #   self._model -> deep copy of model passed by user
        self._uncompressed_model = self._get_model_for_compression(inplace=not inplace)
        self._model = self._get_model_for_compression(inplace=inplace)
        return self._model
