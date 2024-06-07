#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
.. _coremltools_optimize_torch_layerwise_compression:

_`LayerwiseCompressor`
==================================

.. autoclass:: coremltools.optimize.torch.layerwise_compression.LayerwiseCompressorConfig
    :members: from_dict, as_dict, from_yaml, get_layers

.. autoclass:: coremltools.optimize.torch.layerwise_compression.LayerwiseCompressor
    :members: compress

Algorithms
==========

:obj:`coremltools.optimize.torch.layerwise_compression.algorithms` submodule contains classes
that implement the algorithms to be used with :py:class:`LayerwiseCompressor`,
which can be used to compress LLM-based models

GPTQ
----

.. autoclass:: coremltools.optimize.torch.layerwise_compression.algorithms.ModuleGPTQConfig
    :show-inheritance:

.. autoclass:: coremltools.optimize.torch.layerwise_compression.algorithms.GPTQ
    :show-inheritance:

SparseGPT
---------

.. autoclass:: coremltools.optimize.torch.layerwise_compression.algorithms.ModuleSparseGPTConfig
    :show-inheritance:

.. autoclass:: coremltools.optimize.torch.layerwise_compression.algorithms.SparseGPT
    :show-inheritance:


Base class for layerwise compression algorithms config
------------------------------------------------------

.. autoclass:: coremltools.optimize.torch.layerwise_compression.LayerwiseCompressionAlgorithmConfig
    :show-inheritance:
    :no-members:

Base class for layerwise compression algorithms
-----------------------------------------------

.. autoclass:: coremltools.optimize.torch.layerwise_compression.LayerwiseCompressionAlgorithm
    :show-inheritance:
    :members: add_batch, cleanup, compress

Input Cacher
============

:obj:`coremltools.optimize.torch.layerwise_compression.input_cacher` submodule contains classes
which provide a way of capturing the model's inputs up till the first module set up
to be compressed.

FirstLayerInputCacher
---------------------

.. autoclass:: coremltools.optimize.torch.layerwise_compression.FirstLayerInputCacher
    :show-inheritance:
    :members: cache

DefaultInputCacher
------------------

.. autoclass:: coremltools.optimize.torch.layerwise_compression.DefaultInputCacher
    :show-inheritance:
    :members: cache

GPTFirstLayerInputCacher
------------------------

.. autoclass:: coremltools.optimize.torch.layerwise_compression.GPTFirstLayerInputCacher
    :show-inheritance:
    :members: cache

"""


from .algorithms import (
    GPTQ,
    LayerwiseCompressionAlgorithm,
    LayerwiseCompressionAlgorithmConfig,
    ModuleGPTQConfig,
    ModuleSparseGPTConfig,
    SparseGPT,
)
from .input_cacher import DefaultInputCacher, FirstLayerInputCacher, GPTFirstLayerInputCacher
from .layerwise_compressor import LayerwiseCompressor, LayerwiseCompressorConfig
