# Core ML Tools FAQs

This page offers frequently asked questions (FAQs):


- [What are some major changes between versions of Core ML Tools?](#core-ml-tools-versions)
- [How do I convert models from PyTorch?](#pytorch-conversion)
- [How do I convert models from Keras?](#keras-conversion)
- [How do I fix a Core ML model prediction that has a high numerical error compared to the source model?](#fixing-high-numerical-error)
- [How do I handle image preprocessing parameters when converting torchvision models?](#image-preprocessing-for-converting-torchvision)
- [How do I handle an "Error in declaring network" or an "Error computing NN outputs"?](#error-in-declaring-network-or-computing-nn-outputs)
- [Are TensorFlow or PyTorch the only starting points to make a deep learning Core ML model?](#starting-a-deep-learning-core-ml-model)
- [How do I handle the unsupported op error "convert function for op not implemented"?](#handling-an-unsupported-op)
- [Can I choose custom names for the input and outputs of the model during conversion?](#choosing-custom-names-for-input-and-outputs)
- [If I change my fixed-shape model to use flexible inputs, will it still run on the  Neural Engine?](#neural-engine-with-flexible-input-shapes)
- [Why use `ct.optimize.torch` rather than PyTorch's default quantization?](#why-optimizetorch-is-better-than-pytorchs-default-quantization)
- [My model's initialization in Python takes a long time. How can I speed it up?](#use-a-compiled-model-for-faster-initialization)

***

## Core ML Tools Versions

### coremltools 7

For an overview, see [New Features](new-features). This release includes more APIs for optimizing the models to use less storage space, reduce power consumption, and reduce latency during inference. Key optimization techniques include pruning, quantization, and palettization. For details, see [Optimizing Models](opt-overview).

For details about the release, see [Release Notes](https://github.com/apple/coremltools/releases/).


### Previous releases

The following are highlights of previous releases:

#### coremltools 6

- Added model compression utilities to compress the weights of a Core ML model, thereby reducing the space occupied by the model.
- Enabled Float 16 input/output types including grayscale images and float 16 Multiarrays. 

For details, see [Release Notes for coremltools 6.3](https://github.com/apple/coremltools/releases/tag/6.3).


#### coremltools 5

- Core ML models now also use a directory format, called [`.mlpackage`](https://developer.apple.com/documentation/coreml/core_ml_api/updating_a_model_file_to_a_model_package), rather than just a protobuf file.
- Added a new backend: [ML program](convert-to-ml-program), which offers [typed execution](typed-execution) and a new GPU runtime backed by MPSGraph.

For details, see the following:

- [Release Notes for coremltools 5.2](https://github.com/apple/coremltools/releases/tag/5.2)
- [Release Notes for coremltools 5.0](https://github.com/apple/coremltools/releases/tag/5.0)


### coremltools 4

Major upgrade. Highlights:

- Introduced the [Unified Conversion API](unified-conversion-api) with the `convert()` method for converting models from TensorFlow 1, TensorFlow 2 (tf.keras), and PyTorch.
- Introduced the [Model Intermediate Language](model-intermediate-language) (MIL) as an internal intermediate representation (IR) for unifying the conversion pipeline, and added graph passes to this common IR. Passes that improve performance continue to be added, so we recommend that you always use the latest version of coremltools to convert your models.

For details, see the following:

- [Release Notes for coremltools 4.1](https://github.com/apple/coremltools/releases/tag/4.1)
- [Release Notes for coremltools 4.0](https://github.com/apple/coremltools/releases/tag/4.0)
- [Release Notes for coremltools 3.4](https://github.com/apple/coremltools/releases/tag/3.4)
- [All release notes](https://github.com/apple/coremltools/releases)


## PyTorch Conversion

- Prior to Core ML Tools 4: Use [`onnx-coreml`](https://github.com/onnx/onnx-coreml), which internally calls into `coremltools`.
- Core ML Tools 4 and newer: Use the [Unified Conversion API](unified-conversion-api). (The `onnx-coreml` converter is frozen and no longer updated or maintained.)

## Keras Conversion

As of the Core ML Tools 4 release, the `coremltools.keras.convert` converter is no longer maintained, and is officially deprecated in Core ML Tools 5 . The [Unified Conversion API](unified-conversion-api) supports conversion of `tf.keras` models, using a TensorFlow 2 (TF2) backend.

If you have an older [Keras.io](http://keras.io/) model that uses TensorFlow 1 (TF1), we recommend exporting it as a TF1 frozen graph def (`.pb`) file. You can then convert this file using the [Unified Conversion API](unified-conversion-api). For an example of how to export the old keras model to `.pb`, see method `_save_h5_as_frozen_pb` in the [Troubleshooting section of the coremltools 3 Neural Network Guide](https://github.com/apple/coremltools/blob/3.4/examples/NeuralNetworkGuide.md#troubleshooting).

## Fixing High Numerical Error

For a neural network, set the compute unit to CPU as described in [Set the Compute Units](load-and-convert-model.md#set-the-compute-units). For example:

```python
# neural networks
model = ct.convert(source_model,
                   compute_units=ct.ComputeUnit.CPU_ONLY)
# or when loading the model
model = ct.models.MLModel("model.mlmodel", compute_units=ct.ComputeUnit.CPU_ONLY)

# now when prediction is called on this model, it will use the 
# higher precision Float32 CPU path for execution. 

# to check the compute unit of an already loaded model,
# simply check the property 
model.compute_unit
```

For an ML program, set `compute_precision` to Float 32 as described in [Set the ML Program Precision](convert-to-ml-program.md#set-the-ml-program-precision). For example:

```python
# ml programs

# provide a higher compute precision during conversion
model = ct.convert(source_model, compute_precision=ct.precision.FLOAT32)
```

For more information, see [Typed Execution](typed-execution).

## Image Preprocessing for Converting torchvision

Preprocessing parameters differ between [torchvision](https://pytorch.org/vision/stable/index.html) and Core ML Tools but can be easily translated, as described in [Add Image Preprocessing Options](image-inputs.md#add-image-preprocessing-options). For example, you can set the scale and bias for an [`ImageType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#coremltools.converters.mil.input_types.ImageType), which corresponds to the torchvision parameters: 

```python
scale = 1/(0.226*255.0)
bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]
image_input = ct.ImageType(shape=example_input.shape, 
                           scale=scale, 
                           bias=bias)
```

## Error in Declaring Network or Computing NN Outputs

File an issue at the [`coremltools` Github repository](https://github.com/apple/coremltools) by following the instructions in [Issues and Queries](how-to-contribute.md#issues-and-queries). As a workaround, try using `CPUOnly` compute units during conversion, as described in [Set the Compute Units](load-and-convert-model.md#set-the-compute-units). For example:

```python
import coremltools as ct
model = ct.convert(tf_model, compute_units=ct.ComputeUnit.CPU_ONLY)

# or if loading a pre converted model
model = ct.models.MLModel("model.mlmodel", compute_units=ct.ComputeUnit.CPU_ONLY)
```

## Starting a Deep Learning Core ML Model

You can define a Core ML model directly by building it with the MIL builder API. This API is similar to the `torch.nn` or the `tf.keras` API for model construction. For an example, see [Create a MIL Program](model-intermediate-language.md#create-a-mil-program).

## Handling an Unsupported Op

Be sure that you are using the newest version of Core ML Tools. If you still get this error, please file an issue in the [`coremltools` Github repo](https://github.com/apple/coremltools) by following the instructions in [Issues and Queries](how-to-contribute.md#issues-and-queries).

As a workaround, you may want to write a translation function from the missing op to the existing MIL ops. For examples, see [Composite Operators](composite-operators).

## Choosing Custom Names for Input and Outputs

When using [`ct.convert()`](https://apple.github.io/coremltools/source/coremltools.converters.mil.html#coremltools.converters._converters_entry.convert), the input names and output names are automatically picked up by the converter from the source model. After conversion you can see these names by doing one of the following:

- Saving the model and using it with Xcode, as described in [Save and Load the Model](introductory-quickstart.md#save-and-load-the-model) and [Use the Model with Xcode](introductory-quickstart.md#use-the-model-with-xcode).
- Getting the model spec object and printing the names, as shown in the following example:

```python
model = ct.models.MLModel('MyModel.mlmodel')
spec = model.get_spec()

# get input names
input_names = [inp.name for inp in spec.description.input]

# get output names
output_names = [out.name for out in spec.description.output]
```

You can update these names by using the [`rename_feature`](mlmodel-utilities.md#rename-a-feature) API.

## Neural Engine With Flexible Input Shapes

When converting a fixed-shape model that already runs on the Neural Engine (NE) to use flexible inputs, you should specify a flexible input shape with a set of predetermined shapes using [`EnumeratedShapes`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#enumeratedshapes). The converted model will run on the NE, unless the conversion introduces dynamic layers not supported on the NE, such as converting a static reshape to a fully dynamic reshape.

With `EnumeratedShapes` the model can be optimized for the finite set of input shapes on the device during compilation. You can provide up to 128 different shapes. If you need more flexibility for inputs, consider setting the range for each dimension.

For details and examples of using flexible input shapes, see [Flexible Input Shapes](flexible-inputs).

## Why `optimize.torch` is better than PyTorch's default quantization

You can use PyTorch's quantization APIs directly, and then convert the model to Core ML. 
However, the converted model performance may not be optimal. 
The PyTorch API default settings 
(symmetric asymmetric quantization modes and which ops are quantized) 
are not optimal for the Core ML stack and Apple hardware. 
If you use the Core ML Tools `coremltools.optimize.torch` APIs 
the correct default settings are applied automatically.

## Use a compiled model for faster initialization

If your model initialization in Python takes a long time, use a *compiled* Core ML model ([CompiledMLModel](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.CompiledMLModel)) rather than  [MLModel](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.model.MLModel) for making predictions. For large models, using a compiled model can save considerable time in initializing the model. For details, see [Using Compiled Python Models for Prediction](model-prediction.md#using-compiled-python-models-for-prediction).

