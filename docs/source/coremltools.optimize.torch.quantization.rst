Training-Time Quantization
==========================

Quantization refers to techniques for performing neural network computations in lower precision than
floating point. Quantization can reduce a model’s size and also improve a model’s inference latency and
memory bandwidth requirement, because many hardware platforms offer high-performance implementations of quantized
operations.

.. automodule:: coremltools.optimize.torch.quantization

    .. autoclass:: ModuleLinearQuantizerConfig
        :members: from_dict, as_dict, from_yaml

    .. autoclass:: LinearQuantizerConfig
        :members: set_global, set_module_type, set_module_name, from_dict, as_dict, from_yaml

    .. autoclass:: LinearQuantizer
        :members: prepare, step, report, finalize

    .. autoclass:: ObserverType

    .. autoclass:: QuantizationScheme

