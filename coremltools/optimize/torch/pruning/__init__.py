#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
.. _coremltools_optimize_torch_pruning:

.. include:: pruning_desc.rst

_`MagnitudePruner`
==================

.. autoclass:: coremltools.optimize.torch.pruning.ModuleMagnitudePrunerConfig
    :members: from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.pruning.MagnitudePrunerConfig
    :members: set_global, set_module_type, set_module_name, from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.pruning.MagnitudePruner
    :members: prepare, step, report, finalize

Pruning scheduler
=================

:obj:`coremltools.optimize.torch.pruning.pruning_scheduler` submodule contains classes
that implement pruning schedules, which can be used for changing the
sparsity of pruning masks applied by various types of pruning algorithms
to prune neural network parameters.


Base class
----------

.. autoclass:: coremltools.optimize.torch.pruning.pruning_scheduler.PruningScheduler
    :show-inheritance:
    :no-members:


PolynomialDecayScheduler
------------------------

.. autoclass:: coremltools.optimize.torch.pruning.pruning_scheduler.PolynomialDecayScheduler
    :show-inheritance:
    :members: compute_sparsity


ConstantSparsityScheduler
-------------------------

.. autoclass:: coremltools.optimize.torch.pruning.pruning_scheduler.ConstantSparsityScheduler
    :show-inheritance:
    :members: compute_sparsity
"""


from .magnitude_pruner import MagnitudePruner, MagnitudePrunerConfig, ModuleMagnitudePrunerConfig
from .pruning_scheduler import ConstantSparsityScheduler, PolynomialDecayScheduler
