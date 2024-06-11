# New Features

The following sections describe new features and improvements in the most recent versions of Core ML Tools.

## New in Core ML Tools 8

- Support for [Stateful](stateful-models) Core ML models 
- Support for [Multifunction](multifunction-models) Core ML models
- Several new features in model compression, to see a full list check out the 
  [whats new](opt-whats-new) page in the optimization section.


## Previous Versions

The [coremltools 7](https://github.com/apple/coremltools/releases/tag/7.2) package 
includes APIs for [optimizing the models](opt-overview) to use less storage space, 
reduce power consumption, and reduce latency during inference. 
Key optimization techniques include pruning, quantization, and palettization.

The [coremltools 6](https://github.com/apple/coremltools/releases/tag/6.3) package 
offers the following features to optimize the model conversion process:

- Model compression utilities; see [Compressing Neural Network Weights](quantization-neural-network).
- Float 16 input/output types including image. See [Image Input and Output](image-inputs).

For a full list of changes from `coremltools` 5.2, see [Release Notes](#release-notes).


## Release Notes

Learn about changes to the `coremltools` package from the following release notes:

- [Release Notes (newest)](https://github.com/apple/coremltools/releases/)

For information about previous releases, see the following:

- [Release Notes for coremltools 7.2](https://github.com/apple/coremltools/releases/tag/7.2)
- [Release Notes for coremltools 6.3](https://github.com/apple/coremltools/releases/tag/6.3)
- [Release Notes for coremltools 5.2](https://github.com/apple/coremltools/releases/tag/5.2)
- [Release Notes for coremltools 5.0](https://github.com/apple/coremltools/releases/tag/5.0)
- [Release Notes for coremltools 4.1](https://github.com/apple/coremltools/releases/tag/4.1)
- [Release Notes for coremltools 4.0](https://github.com/apple/coremltools/releases/tag/4.0)
- [Release Notes for coremltools 3.4](https://github.com/apple/coremltools/releases/tag/3.4)
- [All release notes](https://github.com/apple/coremltools/releases)


## Migration Workflow (Core ML Tools 3 &rarr; 4)

If you used `coremltools` 3 for neural network model conversion from TensorFlow or ONNX/PyTorch to Core ML, update your workflow as follows when you upgrade to `coremltools` 4 and newer:

| Conversion from | coremltools 3                                                                                                                                                                                                                                                           | coremltools 4 and newer                                                                          |
| :-------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------- |
| TensorFlow      | Install [coremltools 3.4](https://pypi.org/project/coremltools/3.4/) and [tfcoreml 1.1](https://pypi.org/project/tfcoreml/1.1/) and use the `tfcoreml.convert` API.                                                                                                     | Use the new `coremltools.convert` API. See [Unified Conversion API](unified-conversion-api). |
| PyTorch         | First export the PyTorch model to the [ONNX](https://github.com/onnx/onnx) format and then install [coremltools 3.4](https://pypi.org/project/coremltools/3.4/) and [onnx-coreml 1.3](https://pypi.org/project/onnx-coreml/1.3/) and use the `onnx_coreml.convert` API. | Use the new `coremltools.convert` API. See [Unified Conversion API](unified-conversion-api). |

### Convert from TensorFlow

With `coremltools` 4 and newer versions, you do not need to install the [tfcoreml](https://github.com/tf-coreml/tf-coreml) package to convert TensorFlow models. The TensorFlow converter is fully integrated in coremltools and available in the [Unified Conversion API](unified-conversion-api).

```{admonition} For older deployment targets

To deploy the Core ML model to a target that is iOS 12, macOS 10.13, watchOS 5, tvOS 12, or an older version, use [coremltools 3 and tfcoreml 1](https://github.com/apple/coremltools/blob/3.4/examples/NeuralNetworkGuide.md#tensorflow-conversion).
```

### Convert from PyTorch

You can directly convert from PyTorch using the newest version of coremltools, which includes a PyTorch converter available through the [Unified Conversion API](unified-conversion-api). You no longer need to use the two-step process for converting PyTorch models using the [ONNX](https://github.com/onnx/onnx) format. 

```{admonition} For older deployment targets

To deploy the Core ML model to a target that is iOS 12, macOS 10.13, watchOS 5, tvOS 12, or an older version, use [coremltools 3 and onnx-coreml 1](https://github.com/apple/coremltools/blob/3.4/examples/NeuralNetworkGuide.md#onnx-converter)
```

### Deprecated Methods and Support

In coremltools 4 and newer, the following class and methods available in previous versions are deprecated:

- `convert_neural_network_weights_to_fp16()`, `convert_neural_network_spec_weights_to_fp16()`, and `quantize_spec_weights()`. Use the `quantize_weights()` method instead. For instructions, see [Quantization](quantization-neural-network).
- The NeuralNetworkShaper class. 
- `get_allowed_shape_ranges()`.
- `can_allow_multiple_input_shapes()`.
- `visualize_spec()` method of the MLModel class. You can use the [netron](https://github.com/lutzroeder/netron) open source viewer to visualize Core ML models.
- `get_custom_layer_names()`, `replace_custom_layer_name()`, and  `has_custom_layer()`: These were moved to internal methods.
- Caffe converter
- Keras.io and ONNX converters will be deprecated in coremltools 6. Users are recommended to transition to the TensorFlow/PyTorch conversion using the [Unified Conversion API](unified-conversion-api). 

The current version of coremltools ([version 7.1](https://github.com/apple/coremltools)) includes wheels for Python 3.7, 3.8, 3.9, 3.10, and 3.11.




