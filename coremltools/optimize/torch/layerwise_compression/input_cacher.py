#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# Original implementation from https://github.com/IST-DASLab/sparsegpt
# Copyright 2023 IST Austria Distributed Algorithms and Systems Lab. All Rights Reserved.

import logging as _logging
import re as _re
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from typing import Dict as _Dict
from typing import Iterable as _Iterable
from typing import List as _List
from typing import Tuple as _Tuple
from typing import Union as _Union

import torch as _torch
import torch.nn as _nn

from coremltools.optimize.torch._utils.python_utils import ClassRegistryMixin as _ClassRegistryMixin

_logger = _logging.getLogger(__name__)


class StopExecution(ValueError):
    pass


class FirstLayerInputCacher(_ABC, _ClassRegistryMixin):
    """
    A template class for getting the inputs to feed to the first layer of the model
    which is set up for compression.
    """

    def __init__(self, model: _nn.Module, layers: str):
        self._model = model
        self._layers = layers

    @_abstractmethod
    def cache(
        self, dataloader: _Iterable, nsamples: int, device: str
    ) -> _Tuple[_List[_torch.Tensor], _Dict[str, _torch.Tensor]]:
        """
        Cache inputs and keyword arguments to be fed to first layer of the model
        which is set up for compression.

        Args:
            dataloader (:py:class:`Iterable`): An iterable where each element
                is an input to the model to be compressed.
            nsamples (:obj:`int`): Number of samples to cache.
            device (:obj:`str`): Device string for device to run compression on.
        """
        raise NotImplementedError("Method not implemented in base class.")


@FirstLayerInputCacher.register("gpt")
class GPTFirstLayerInputCacher(FirstLayerInputCacher):
    """
    An implementation of :py:class:`FirstLayerInputCacher` for GPT style models.
    Computes inputs to feed to the first layer of the model which is set up for compression.

    Args:
        model (:obj:`torch.nn.Module`): Module to be compressed.
        layers (:obj:`str`): Regex string for the decoder layers of the model.
    """

    def __init__(
        self,
        model: _nn.Module,
        layers: _Union[str, _List],
    ):
        super().__init__(model, layers)
        self._pre_layers = []
        self._first_layer = None
        for layer_name, layer in model.named_modules(remove_duplicate=True):
            if self._first_layer_match(layer_name, layer):
                self._pre_layers.append(layer)
                self._first_layer = layer
                # break the first time there's a match
                break
            elif len(list(layer.children())) == 0:
                self._pre_layers.append(layer)
        if self._first_layer is None:
            _logger.warning(
                "Could not find first decoder layer based on",
                f"decoder layer path {layers} regex",
            )

    def _first_layer_match(self, layer_name: str, layer: _torch.nn.Module) -> bool:
        if isinstance(self._layers, str):
            return _re.fullmatch(self._layers, layer_name)
        elif isinstance(self._layers, list):
            if isinstance(self._layers[0], str):
                return _re.fullmatch(self._layers[0], layer_name)
            else:
                return layer == self._layers[0]

    def _feed_data(self, dataloader: _Iterable, nsamples: int, device: str):
        """
        Feed data to the model so that the inputs to the first layer can be cached.
        """
        num_sampled = 0
        for batch in dataloader:
            try:
                self._model(batch.to(device))
            except StopExecution:
                pass
            num_sampled += 1
            if num_sampled >= nsamples:
                break

    @staticmethod
    def _get_input_cacher_pre_hook(inputs, kwarg_inputs):
        """
        Returns forward_pre_hook for caching inputs and keyword arguments
        to the first decoder layer of a GPT model.
        """

        def input_cacher_pre_hook(module, args, kwargs):
            inputs.append(args)
            for key, val in kwargs.items():
                kwarg_inputs[key] = val
            raise StopExecution()

        return input_cacher_pre_hook

    def cache(
        self, dataloader: _Iterable, nsamples: int, device: str
    ) -> _Tuple[_List[_torch.Tensor], _Dict[str, _torch.Tensor]]:
        """
        Cache inputs and keyword arguments to be fed to the first decoder layer
        of a GPT style model.

        Args:
            dataloader (:py:class:`Iterable`): An iterable where each element
                is an input to the model to be compressed.
            nsamples (:obj:`int`): Number of samples to cache.
            device (:obj:`str`): Device string for device to run compression on.
        """
        for layer in self._pre_layers:
            layer.to(device)

        inputs, kwarg_inputs = [], {}
        input_cacher_handle = self._first_layer.register_forward_pre_hook(
            self._get_input_cacher_pre_hook(inputs, kwarg_inputs), with_kwargs=True
        )
        self._feed_data(dataloader, nsamples, device)
        input_cacher_handle.remove()

        for layer in self._pre_layers:
            layer.cpu()

        for key, val in kwarg_inputs.items():
            if isinstance(val, _torch.Tensor):
                kwarg_inputs[key] = val.to(device)

        return inputs, kwarg_inputs


@FirstLayerInputCacher.register("default")
class DefaultInputCacher(FirstLayerInputCacher):
    def cache(
        self, dataloader: _Iterable, nsamples: int, device: str
    ) -> _Tuple[_List[_torch.Tensor], _Dict[str, _torch.Tensor]]:
        """
        Cache inputs and keyword arguments to be fed to first layer of the model
        which is set up for compression.

        Args:
            dataloader (:py:class:`Iterable`): An iterable where each element
                is an input to the model to be compressed.
            nsamples (:obj:`int`): Number of samples to cache.
            device (:obj:`str`): Device string for device to run compression on.
        """
        inputs = []
        sampled = 0
        for batch in dataloader:
            inputs.append(batch.to(device))
            sampled += 1
            if sampled == nsamples:
                break
        return inputs, {}
