Quantization
============

Quantization refers to techniques for performing neural network computations in lower precision than
floating point. Quantization can reduce a model’s size and also improve a model’s inference latency and
memory bandwidth requirement, because many hardware platforms offer high-performance implementations of quantized
operations.

.. autoclass::  coremltools.optimize.torch.quantization.ModuleLinearQuantizerConfig
    :members: from_dict, as_dict, from_yaml

.. autoclass::  coremltools.optimize.torch.quantization.LinearQuantizerConfig
    :members: set_global, set_module_type, set_module_name, from_dict, as_dict, from_yaml

.. autoclass:: coremltools.optimize.torch.quantization.LinearQuantizer
    :members: prepare, step, report, finalize

.. autoclass::  coremltools.optimize.torch.quantization.ObserverType

.. autoclass::  coremltools.optimize.torch.quantization.QuantizationScheme

.. autoclass::  coremltools.optimize.torch.quantization.ModulePostTrainingQuantizerConfig

.. autoclass::  coremltools.optimize.torch.quantization.PostTrainingQuantizer

.. autoclass::  coremltools.optimize.torch.quantization.PostTrainingQuantizerConfig

.. autoclass:: coremltools.optimize.torch.layerwise_compression.LayerwiseCompressorConfig
    :members: from_dict, as_dict, from_yaml, get_layers

.. autoclass:: coremltools.optimize.torch.layerwise_compression.LayerwiseCompressor
    :members: compress

GPTQ
----

.. autoclass:: coremltools.optimize.torch.layerwise_compression.algorithms.ModuleGPTQConfig
    :show-inheritance:

.. autoclass:: coremltools.optimize.torch.layerwise_compression.algorithms.GPTQ
    :show-inheritance: