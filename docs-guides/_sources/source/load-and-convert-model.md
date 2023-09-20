```{eval-rst}
.. index::
    single: convert to; workflow
```



# Load and Convert Model Workflow
 
The typical conversion process with the [Unified Conversion API](convert-learning-models) is to load the model to infer its type, and then use the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method to convert it to the Core ML format. Follow these steps:

1. Import `coremltools` (as `ct` for the following code snippets), and load a TensorFlow or PyTorch model.
    
	```python TensorFlow
	import coremltools as ct

	# Load TensorFlow model
	import tensorflow as tf # Tf 2.2.0

	tf_model = tf.keras.applications.MobileNet()
	```
    
	```python PyTorch
	import coremltools as ct

	# Load PyTorch model (and perform tracing)
	torch_model = torchvision.models.mobilenet_v2()
	torch_model.eval() 

	example_input = torch.rand(1, 3, 256, 256)
	traced_model = torch.jit.trace(torch_model, example_input)
	```

2. Convert the TensorFlow or PyTorch model using `convert()`:
    
	```python TensorFlow
	# Convert using the same API
	model_from_tf = ct.convert(tf_model)
	```
    
	```python PyTorch
	# Convert using the same API. Note that we need to provide "inputs" for pytorch conversion.
	model_from_torch = ct.convert(traced_model,
								  inputs=[ct.TensorType(name="input", 
														shape=example_input.shape)])
	```

The conversion produces an `MLModel` object which you can use to make predictions, change metadata, or save to the Core ML format for use in Xcode. 

By default, older versions of the Unified Conversion API create a neural network, but you can use the `convert_to` parameter to specify the `mlprogram` model type for an [ML program](convert-to-ml-program) model:

```python TensorFlow
# Convert using the same API
model_from_tf = ct.convert(tf_model, convert_to="mlprogram")
```

```python PyTorch
# Convert using the same API. Note that we need to provide "inputs" for pytorch conversion.
model_from_torch = ct.convert(traced_model,
							  convert_to="mlprogram",
                              inputs=[ct.TensorType(name="input", 
                                                    shape=example_input.shape)])
```

Since the `neuralnetwork` format is widely available, it is still the default format produced by versions of the [Unified Conversion API](unified-conversion-api) older than 7.0. However, in 7.0 and newer versions, the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method produces an `mlprogram` by default with the `iOS15`/`macOS12` deployment target. You can override this behavior by providing a `minimum_deployment_target` or `convert_to` value.

For more information, see the [MLModel Overview](mlmodel).

```{admonition} Conversion Options

The [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method tries to infer as much as possible from the source network, but some information may not be present, such as input names, types, shapes, and classifier options. For more information see [Conversion Options](conversion-options).
```

```{eval-rst}
.. index:: 
    single: TensorFlow; convert from
```


## Convert From TensorFlow 2

TensorFlow 2 models are typically exported as `tf.Model` objects in the SavedModel or HDF5 file formats. For additional TensorFlow formats you can convert, see [TensorFlow 2 Workflow](tensorflow-2).

The following example demonstrates how to use the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method to convert an Xception model from `tf.keras.applications`:

```python
import coremltools as ct 
import tensorflow as tf

# Load from .h5 file
tf_model = tf.keras.applications.Xception(weights="imagenet", 
                                          input_shape=(299, 299, 3))

# Convert to Core ML
model = ct.convert(tf_model)
```

## Convert From TensorFlow 1

The conversion API can also convert models from TensorFlow 1. These models are generally exported with the extension `.pb`, in the frozen protobuf file format, using TensorFlow 1's freeze graph utility. You can pass this model directly into the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method. For details, see [TensorFlow 1 Workflow](tensorflow-1-workflow).

The following example demonstrates how to convert a pre-trained MobileNet model in the frozen protobuf format to Core ML. 

```{admonition} Download for the Following Example

To run the following example, first [download this pre-trained model](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz "mobilenet_v1_1.0_224_frozen.tg").
```

```python
import coremltools as ct

# Convert a frozen graph from TensorFlow 1 to Core ML
mlmodel = ct.convert("mobilenet_v1_1.0_224/frozen_graph.pb")
```

The MobileNet model in the previous example already has a defined input shape, so you do not need to provide it. However, in some cases the TensorFlow model does not contain a fully defined input shape. You can pass an input shape that is compatible with the model into the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method in order to provide the shape information, as shown in the following example.

```{admonition} Download for the Following Example

To run the following example, first [download this pre-trained model](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz "mobilenet_v2_1.0_224.tgz").

```

```python
import coremltools as ct

# Needs additional shape information
mlmodel = ct.convert("mobilenet_v2_1.0_224_frozen.pb",
                    inputs=[ct.TensorType(shape=(1, 224, 224, 3))])
```

```{eval-rst}
.. index:: 
    single: PyTorch; convert from
```

## Convert from PyTorch

You can convert PyTorch models that are either traced or in already the TorchScript format. For example, you can convert a model obtained using [PyTorch's save and load APIs](https://pytorch.org/tutorials/beginner/saving_loading_models.html) to Core ML using the same Unified Conversion API as the previous example:

```python
import coremltools as ct
import torch
import torchvision

# Get a pytorch model and save it as a *.pt file
model = torchvision.models.mobilenet_v2()
model.eval()
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("torchvision_mobilenet_v2.pt")

# Convert the saved PyTorch model to Core ML
mlmodel = ct.convert("torchvision_mobilenet_v2.pt",
                    inputs=[ct.TensorType(shape=(1, 3, 224, 224))])
```

For more details on tracing and scripting to produce PyTorch models for conversion, see [Converting from PyTorch](convert-pytorch).

```{eval-rst}
.. index:: 
    single: compute units
    single: ML program; compute units
```


## Set the Compute Units

Normally you convert a model by using [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) without using the `compute_units` parameter. In most cases you don’t need it, because the converter picks the default optimized path for fast execution while loading the model. The default setting (`ComputeUnit.ALL`) uses all compute units available, including the Apple Neural Engine (ANE), the CPU, and the graphics processing unit (GPU). Whether you are using [ML programs](convert-to-ml-program) or [neural networks](convert-to-neural-network), the defaults for conversion and prediction are picked to execute the model in the most performant way, as described in [Typed Execution](typed-execution).

However, you may find it useful, especially for debugging, to specify the actual compute units when converting or loading a model by using the `compute_units` parameter. The parameter is based on the [MLComputeUnits](https://developer.apple.com/documentation/coreml/mlcomputeunits) enumeration in the Swift developer language — compute units are employed when [loading a Core ML model](https://developer.apple.com/documentation/coreml/mlmodel/3600218-load), taking in MLmodelConfiguration which includes compute units. Therefore, both the [MLModel](https://apple.github.io/coremltools/source/coremltools.models.html#module-coremltools.models.model) class and  [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) provide the `compute_units` parameter. 

The `compute_units` parameter can have the following values:

- `coremltools.ComputeUnit.CPU_ONLY`: Limit the model to use only the CPU.
- `coremltools.ComputeUnit.CPU_AND_GPU`: Use both the CPU and GPU, but not the ANE.
- `coremltools.ComputeUnit.CPU_AND_NE`: Use both the CPU and ANE, but not the GPU.
- `coremltools.ComputeUnit.ALL`: The default setting uses all compute units available, including the ANE, CPU, and GPU.

For example, the following converts the model and sets the `compute_units` to CPU only:

```python
model = ct.convert(tf_model, compute_units=ct.ComputeUnit.CPU_ONLY)
```

For details on how to use this parameter with neural networks, see [Neural Network Untyped Tensors](typed-execution.md#neural-network-untyped-tensors). For details on using it with ML programs, see [ML Program Typed Tensors](typed-execution.md#ml-program-typed-tensors).


