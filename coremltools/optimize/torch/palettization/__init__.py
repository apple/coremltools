#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
.. _coremltools_optimize_torch_palettization:

.. include:: palettization_desc.rst
     :end-line: 7

_`DKMPalettizer`
================

Top level APIs
--------------

.. autoclass:: coremltools.optimize.torch.palettization.ModuleDKMPalettizerConfig
    :members: from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.palettization.DKMPalettizerConfig
    :members: set_global, set_module_type, set_module_name, from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.palettization.DKMPalettizer
    :members: prepare, step, report, finalize


_`Palettization layers for DKM`
-------------------------------

.. autoclass:: coremltools.optimize.torch.palettization.FakePalettize
    :no-members:

_`SensitiveKMeans`
==================

.. autoclass:: coremltools.optimize.torch.palettization.ModuleSKMPalettizerConfig
    :members: from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.palettization.SKMPalettizerConfig
    :members: set_global, set_module_type, set_module_name, from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.palettization.SKMPalettizer
    :members: compute_sensitivity, compress

_`PostTrainingPalettization`
============================

.. autoclass:: coremltools.optimize.torch.palettization.ModulePostTrainingPalettizerConfig
    :members: from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.palettization.PostTrainingPalettizerConfig
    :members: set_global, set_module_type, set_module_name, from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.palettization.PostTrainingPalettizer
    :members: compress
"""

from .fake_palettize import FakePalettize
from .palettization_config import DKMPalettizerConfig, ModuleDKMPalettizerConfig
from .palettizer import DKMPalettizer
from .post_training_palettization import (
    ModulePostTrainingPalettizerConfig,
    PostTrainingPalettizer,
    PostTrainingPalettizerConfig,
)
from .sensitive_k_means import ModuleSKMPalettizerConfig, SKMPalettizer, SKMPalettizerConfig
