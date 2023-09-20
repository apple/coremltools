```{eval-rst}
.. index::
    pair: ML program; convert to
```


# Convert Models to ML Programs

This section describes the `ML program` model type. It is an evolution of the neural network model type that has been available since the first version of Core ML. 

The ML program model type is a foundation for future improvements. ML programs are available for the iOS15, macOS12, watchOS8, and tvOS15 deployment targets. For details, see [Availability of ML Programs](#availability-of-ml-programs).

```{note}
To convert a model to the older neural network model type, see [Convert Models to Neural Networks](convert-to-neural-network). For a comparison of the ML program and neural network model types, see [Comparing ML Programs and Neural Networks](comparing-ml-programs-and-neural-networks).
```

You can convert a [TensorFlow](https://www.tensorflow.org "TensorFlow") or [PyTorch](https://pytorch.org "PyTorch") model, or a model created directly in the [Model Intermediate Language (MIL)](model-intermediate-language), to a Core ML model that is either  an ML program or a neural network. The [Unified Conversion API](unified-conversion-api) can produce either type of model with the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#coremltools.converters._converters_entry.convert) method.

In Core ML Tools 7.0 and newer versions, the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method produces an `mlprogram` by default:

```python
# Convert to an ML Program
import coremltools as ct  # Core ML Tools version 7.0
model = ct.convert(source_model)
```

The above example produces an `mlprogram` with an `iOS15`/`macOS12` deployment target (or newer). You can override this behavior by providing a `minimum_deployment_target` value, such as `minimum_deployment_target=target.iOS14` or older.

```{eval-rst}
.. index:: 
   single: precision type
   single: ML program; precision type
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

```{eval-rst}
.. index:: 
   single: model package
   single: ML program; save as model package
   single: save a model package
```

## Save ML Programs as Model Packages

The ML program type uses the [Core ML model package](https://developer.apple.com/documentation/coreml/updating_a_model_file_to_a_model_package) container format that separates the model into components and offers more flexible metadata editing. Since an ML program decouples the weights from the program architecture, it cannot be saved as an `.mlmodel` file.

Use the `save()` method to save a file with the `.mlpackage` extension, as shown in the following example:

```python
model.save("my_model.mlpackage")
```

```{warning} Requires Xcode 13 and Newer

The model package format is supported on Xcode 13

```

## Find the Model Type in a Model Package

If you need to determine whether an `mlpackage` file contains a neural network or an ML program, you can open it in Xcode 13, and look at the model type.

On a Linux system you can use Core ML Tools 5 or newer to inspect this property: 

```python
# load MLModel object
model = ct.models.MLModel("model.mlpackage")

# get the spec object
spec = model.get_spec()
print("model type: {}".format(spec.WhichOneof('Type')))
```

## Availability of ML Programs

The ML program model type is available as summarized in the following table: 

|   | Neural Network | ML Program |
| ----------- | ----------- | ----------- |
| Minimum deployment target | macOS 10.13, iOS 11, watchOS 4, tvOS 11 | macOS 12, iOS 15, watchOS 8, tvOS 15 |
| Supported file formats | `.mlmodel` or `.mlpackage` | `.mlpackage` |



