#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from typing import Union as _Union

import attr as _attr
import torch as _torch
from attr import define as _define
from attr import field as _field
from attrs import validators as _validators

from coremltools.optimize.torch._utils.torch_utils import (
    list_or_str_to_tensor as _list_or_str_to_tensor,
)
from coremltools.optimize.torch.optimization_config import (
    ModuleOptimizationConfig as _ModuleOptimizationConfig,
)
from coremltools.optimize.torch.pruning._utils import spline as _spline


@_define
class PruningScheduler(_ABC):
    """
    An abstraction for implementing schedules to be used for
    changing the sparsity of pruning masks applied by various types of
    pruning algorithms to module parameters over the course of the training.
    """

    @_abstractmethod
    def compute_sparsity(
        self, step_count: int, prev_sparsity: float, config: _ModuleOptimizationConfig
    ) -> float:
        """
        Compute the sparsity at the next step given the previous sparsity
        and the module optimization config.

        Args:
            step_count (:obj:`int`): Current step count.
            prev_sparsity (:obj:`float`): Sparsity at previous step.
            config (:py:class:`ModuleOptimizationConfig`): Optimization
                config for the module which contains information such as
                target sparsity and initial sparsity.
        """
        raise NotImplementedError()


@_define
class PolynomialDecayScheduler(PruningScheduler):
    r"""
    A pruning scheduler inspired by the paper `"To prune or not to prune" <https://arxiv.org/pdf/1710.01878.pdf>`_.

    It sets the sparsity at step :math:`t` using the formula:

    .. math::

            sparsity_t = target\_sparsity + (initial\_sparsity - target\_sparsity)
                       * (1 - \frac{update\_index}{total\_number\_of\_updates}) ^ {power}

    If :math:`t` is in :math:`update\_steps`, else it keeps the sparsity at its previous value.

    Here, :math:`update\_index` is the index of :math:`t` in the :math:`update\_steps` array and
    :math:`total\_number\_of\_updates` is the length of :math:`update\_steps` array.

    Args:
        update_steps (:obj:`list` of :obj:`int` or :obj:`str`): The indices of
            optimization steps at which pruning should be performed. This can
            be passed in as a string representing the range, such as
            ``range(start_index, end_index, step_size)``.
        power (:obj:`int`, optional): Exponent to be used in the
            sparsity function. Defaults to ``3``.
    """

    update_steps: _torch.tensor = _field(
        converter=_list_or_str_to_tensor, eq=_attr.cmp_using(eq=_torch.equal)
    )
    power: int = _field(default=3, validator=_validators.instance_of(int))

    @update_steps.validator
    def _check_update_steps(self, attribute, value):
        assert (
            len(value.size()) == 1
        ), f"update_steps: {value} must be a 1-D tensor or list of ints."
        for elem in value:
            if elem.int() != elem:
                raise ValueError(f"Each element of update_steps {value} must be an integer.")
            assert (
                elem >= 0
            ), f"All elements of update_steps must be non-negative. Received: {value}."

    def compute_sparsity(
        self, step_count: int, prev_sparsity: float, config: _ModuleOptimizationConfig
    ) -> float:
        cur_step_update_steps_mask = step_count == self.update_steps
        if _torch.any(cur_step_update_steps_mask):
            update_number = _torch.nonzero(cur_step_update_steps_mask, as_tuple=True)[0].item()
            update_step_shape = self.update_steps.shape[0]
            if update_step_shape == 1:
                t = 1.0
            else:
                t = update_number / (update_step_shape - 1)
            initial_sparsity = (
                config.initial_sparsity if hasattr(config, "initial_sparsity") else 0.0
            )
            assert hasattr(config, "target_sparsity"), (
                f"Attribute target_sparsity not found in config {config}. "
                f"{self.__class__} only works with configs "
                f"which have this attribute."
            )
            return _spline(initial_sparsity, config.target_sparsity, t, self.power)
        return prev_sparsity


@_define
class ConstantSparsityScheduler(PruningScheduler):
    """
    A pruning schedule with constant sparsity throughout training.

    Sparsity is set to zero initially and to ``target_sparsity`` at
    step ``begin_step``.

    Args:
        begin_step (:obj:`int`): step at which to begin pruning.
    """

    begin_step: int = _field(validator=_validators.instance_of(int))

    def compute_sparsity(
        self, step_count: int, prev_sparsity: float, config: _ModuleOptimizationConfig
    ) -> float:
        if step_count >= self.begin_step:
            assert hasattr(config, "target_sparsity"), (
                f"Attribute target_sparsity not found in config {config}. "
                f"{self.__class__} only works with configs "
                f"which have this attribute."
            )
            return config.target_sparsity
        return prev_sparsity


_PruningSchedulerType = _Union[PolynomialDecayScheduler, ConstantSparsityScheduler]
