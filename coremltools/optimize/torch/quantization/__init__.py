#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
.. _coremltools_optimize_torch_qat:

Quantization refers to techniques for performing neural network computations in lower precision than
floating point. Quantization can reduce a model’s size and also improve a model’s inference latency and
memory bandwidth requirement, because many hardware platforms offer high-performance implementations of quantized
operations.

_`LinearQuantizer`
==================

.. autoclass::  coremltools.optimize.torch.quantization.ModuleLinearQuantizerConfig
    :members: from_dict, as_dict, from_yaml

.. autoclass::  coremltools.optimize.torch.quantization.LinearQuantizerConfig
    :members: set_global, set_module_type, set_module_name, from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.quantization.LinearQuantizer
    :members: prepare, step, report, finalize

.. autoclass::  coremltools.optimize.torch.quantization.ObserverType

.. autoclass::  coremltools.optimize.torch.quantization.QuantizationScheme

_`PostTrainingQuantization`
============================

.. autoclass:: coremltools.optimize.torch.quantization.ModulePostTrainingQuantizerConfig
    :members: from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.quantization.PostTrainingQuantizerConfig
    :members: set_global, set_module_type, set_module_name, from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.quantization.PostTrainingQuantizer
    :members: compress

"""

from .quantization_config import (
    LinearQuantizerConfig,
    ModuleLinearQuantizerConfig,
    ObserverType,
    QuantizationScheme,
)
from .quantizer import LinearQuantizer

from .post_training_quantization import (
    ModulePostTrainingQuantizerConfig,
    PostTrainingQuantizer,
    PostTrainingQuantizerConfig,
)
