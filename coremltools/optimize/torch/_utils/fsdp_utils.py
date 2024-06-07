#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from functools import partial as _partial
from typing import Iterable as _Iterable
from typing import Type as _Type

import torch as _torch
from attr import define as _define
from torch.distributed.fsdp.wrap import ModuleWrapPolicy as _TorchModuleWrapPolicy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy as _size_based_auto_wrap_policy


class FSDPAutoWrapPolicy(_ABC):
    """
    An abstract base class for implementing an `FSDP <https://pytorch.org/docs/stable/fsdp.html>`_ auto wrap policy.

    Wrapping a model with ``FSDP`` wrapper, ``FSDP(model)``, results in a single FSDP unit for the entire model.
    Thus, during the model's execution, the ``all-gather`` operation collects all the parameters of the model on all
    GPUs and hence, parameter sharding doesn't save any CUDA memory.

    To avoid this, one can specify a :py:class:`FSDPAutoWrapPolicy`, which automatically creates multiple FSDP units
    nested within the top level FSDP unit, based on certain criteria such as a minimum size limit for each FSDP
    unit or based on the class structure of the model. This way, only one FSDP unit needs to collect full
    parameters at a time, and one can compute gradients for a much larger model, which wouldn't be possible otherwise.

    For more details, please refer to `FSDP documentation <https://pytorch.org/docs/stable/fsdp.html>`_
    """
    @_abstractmethod
    def get_policy(self):
        """
        Return a policy for wrapping different submodules of a model with FSDP wrapper.
        """


@_define
class ModuleWrapPolicy(FSDPAutoWrapPolicy):
    """
    An auto wrap policy which wraps instances of modules with classes specified by ``module_classes`` into separate
    FSDP units.

    This policy is useful for transformer like models which can be naturally split into distinct submodules.

    For example, for a GPT style decoder model, with ``Attention`` and ``FeedForward`` as the two
    types of layers in it, one can specify ``module_classes = [Attention, FeedForward]``. This would lead to
    each instance of ``Attention`` and ``FeedForward`` layer in the model to be wrapped
    into an individual FSDP unit.
    """

    module_classes: _Iterable[_Type[_torch.nn.Module]]

    def get_policy(self):
        return _TorchModuleWrapPolicy(self.module_classes)


@_define
class SizeBasedWrapPolicy:
    """
    An auto wrap policy which creates a new FSDP instances when the number of parameters in the the current FSDP
    unit exceeds ``min_num_params``.
    """

    min_num_params: int

    def get_policy(self):
        return _partial(_size_based_auto_wrap_policy, min_num_params=self.min_num_params)
