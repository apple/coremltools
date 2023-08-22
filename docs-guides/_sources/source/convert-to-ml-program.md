# Convert Models to ML Programs

You can convert a [TensorFlow](https://www.tensorflow.org "TensorFlow") or [PyTorch](https://pytorch.org "PyTorch") model, or a model created directly in the [Model Intermediate Language (MIL)](model-intermediate-language), to a Core ML model that is either  an ML program or a neural network. The [Unified Conversion API](unified-conversion-api) can produce either type of model with the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#coremltools.converters._converters_entry.convert) method.

## Convert to an ML Program

In Core ML Tools 7.0b2 and newer versions, the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method produces an `mlprogram` by default:

```python
# Convert to an ML Program
import coremltools as ct  # Core ML Tools version 7.0b2
model = ct.convert(source_model)
```

The above example produces an `mlprogram` with an `iOS15`/`macOS12` deployment target (or newer). You can override this behavior by providing a `minimum_deployment_target` value, such as `minimum_deployment_target=target.iOS14` or older.

## Convert to a Neural Network

With versions of Core ML Tools older than 7.0b2, if you didn't specify the model type, or your `minimum_deployment_target` was a version older than iOS15, macOS12, watchOS8, or tvOS15, the model was converted by default to a neural network.

To convert to a neural network using Core ML Tools version 7.0b2 or newer, specify the model type with the `convert_to` parameter, as shown in the following example:

```python
import coremltools as ct  # Core ML Tools version 7.0b2
# provide the "convert_to" argument to convert to a neural network
model = ct.convert(source_model, convert_to="neuralnetwork")
```

Alternatively, you can use the `minimum_deployment_target` parameter to specify a target such as `minimum_deployment_target=target.iOS14` or older, as shown in the following example: 

```python
import coremltools as ct  # Core ML Tools version 7.0b2
# provide the "minimum_deployment_target" argument to convert to a neural network
model = ct.convert(source_model, 
                   minimum_deployment_target=ct.target.iOS14)
```

## Set the ML Program Precision

You can optionally set the precision type (float 16 or float 32) of the weights and the intermediate tensors in the ML program during conversion. The ML program type offers an additional `compute_precision` parameter as shown in the following example:

```python
# produce a Float 16 typed model
# this is also the default if compute_precision argument is skipped
model = ct.convert(source_model, 
                   convert_to="mlprogram", 
                   compute_precision=ct.precision.FLOAT16)
                    
# produce a Float 32 typed model,
# useful if the model needs higher precision, and float 16 is not sufficient 
model = ct.convert(source_model, 
                   convert_to="mlprogram", 
                   compute_precision=ct.precision.FLOAT32)
```

For details on ML program precision, see [Typed Execution](typed-execution).

```{admonition} Float 16 Default

For ML programs, Core ML Tools version 5.0b3 and newer produces a model with float 16 precision by default (previous beta versions produced float 32 by default). You can override the default precision by using the `compute_precision` parameter of [`coremltools.convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#coremltools.converters._converters_entry.convert).
```

## Availability of ML Programs

The ML program model type is available as summarized in the following table: 

|   | Neural Network | ML Program |
| ----------- | ----------- | ----------- |
| Minimum deployment target | macOS 10.13, iOS 11, watchOS 4, tvOS 11 | macOS 12, iOS 15, watchOS 8, tvOS 15 |
| Supported file formats | `.mlmodel` or `.mlpackage` | `.mlpackage` |


