# Migration Workflow

If you used `coremltools` 3 for neural network model conversion from TensorFlow or ONNX/PyTorch to Core ML, update your workflow as follows when you upgrade to `coremltools` 4 and newer:

| Conversion from | coremltools 3                                                                                                                                                                                                                                                           | coremltools 4 and newer                                                                          |
| :-------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------- |
| TensorFlow      | Install [coremltools 3.4](https://pypi.org/project/coremltools/3.4/) and [tfcoreml 1.1](https://pypi.org/project/tfcoreml/1.1/) and use the `tfcoreml.convert` API.                                                                                                     | Use the new `coremltools.convert` API. See [Unified Conversion API](unified-conversion-api). |
| PyTorch         | First export the PyTorch model to the [ONNX](https://github.com/onnx/onnx) format and then install [coremltools 3.4](https://pypi.org/project/coremltools/3.4/) and [onnx-coreml 1.3](https://pypi.org/project/onnx-coreml/1.3/) and use the `onnx_coreml.convert` API. | Use the new `coremltools.convert` API. See [Unified Conversion API](unified-conversion-api). |

## Convert from TensorFlow

With `coremltools` 4 and newer versions, you do not need to install the [tfcoreml](https://github.com/tf-coreml/tf-coreml) package to convert TensorFlow models. The TensorFlow converter is fully integrated in coremltools and available in the [Unified Conversion API](unified-conversion-api).

```{admonition} For older deployment targets

To deploy the Core ML model to a target that is iOS 12, macOS 10.13, watchOS 5, tvOS 12, or an older version, use [coremltools 3 and tfcoreml 1](https://github.com/apple/coremltools/blob/3.4/examples/NeuralNetworkGuide.md#tensorflow-conversion).
```

## Convert from PyTorch

You can directly convert from PyTorch using the newest version of coremltools, which includes a PyTorch converter available through the [Unified Conversion API](unified-conversion-api). You no longer need to use the two-step process for converting PyTorch models using the [ONNX](https://github.com/onnx/onnx) format. 

```{admonition} For older deployment targets

To deploy the Core ML model to a target that is iOS 12, macOS 10.13, watchOS 5, tvOS 12, or an older version, use [coremltools 3 and onnx-coreml 1](https://github.com/apple/coremltools/blob/3.4/examples/NeuralNetworkGuide.md#onnx-converter)
```

## Deprecated Methods and Support

In coremltools 4 and newer, the the following class and methods available in previous versions are deprecated:

- `convert_neural_network_weights_to_fp16()`, `convert_neural_network_spec_weights_to_fp16()`, and `quantize_spec_weights()`. Use the `quantize_weights()` method instead. For instructions, see [Quantization](quantization-overview).
- The NeuralNetworkShaper class. 
- `get_allowed_shape_ranges()`.
- `can_allow_multiple_input_shapes()`.
- `visualize_spec()` method of the MLModel class. You can use the [netron](https://github.com/lutzroeder/netron) open source viewer to visualize Core ML models.
- `get_custom_layer_names()`, `replace_custom_layer_name()`, and  `has_custom_layer()`: These were moved to internal methods.
- Caffe converter
- Keras.io and ONNX converters will be deprecated in coremltools 6. Users are recommended to transition to the TensorFlow/PyTorch conversion using the [Unified Conversion API](unified-conversion-api). 

Support for Python 2 has been deprecated since [coremltools 4.1](https://github.com/apple/coremltools/releases/tag/4.1). The current version of coremltools includes wheels for Python 3.5, 3.6, 3.7, and 3.8.




