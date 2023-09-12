```{eval-rst}
.. index:: conversion options
```

# New Conversion Options

You can use the [Unified Conversion API](unified-conversion-api) to convert a TensorFlow or PyTorch model to the Core ML model format as either a neural network or an [ML program](convert-to-ml-program). The following are the newest conversion options.

```{eval-rst}
.. index:: 
    single: convert_to parameter
    single: ML program; minimum_deployment_target
    single: neural network; minimum_deployment_target
```

## Convert to ML Program or Neural Network

To set the type of the model representation produced by the converter. use either the `minimum_deployment_target` parameter or the `convert_to` parameter with [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry). 

The converter produces an ML program if the target is `>=` `iOS15`, `macOS12`, `watchOS8`, or `tvOS15`, or if `convert_to` is set to `‘mlprogram’`; otherwise it produces a neural network. 

If neither the `minimum_deployment_target` nor the `convert_to` parameter is specified, the converter produces a neural network with as minimum of a deployment target as possible.

To learn about the differences between neural networks and ML programs, see [ML Programs](convert-to-ml-program).

```{eval-rst}
.. index:: 
    single: ML program; compute precision
    single: compute precision
```

## Set the Compute Precision for an ML Program

For ML programs, coremltools produces a model with float 16 precision by default. You can override the default precision by using the `compute_precision` parameter. For details, see [Set the ML Program Precision](convert-to-ml-program.md#set-the-ml-program-precision). 

## Pick the Compute Units for Execution

The converter picks the default optimized path for fast execution while loading the model. The default setting (`ComputeUnit.ALL`) uses all compute units available, including the Apple Neural Engine (ANE), the CPU, and the graphics processing unit (GPU). 

However, you may find it useful, especially for debugging, to specify the actual compute units when converting or loading a model by using the `compute_units` parameter. For details, see [Set the compute units](load-and-convert-model.md#set-the-compute-units).

```{eval-rst}
.. index:: 
    single: input type options
    single: output type options
```

## Input and Output Type Options

Starting in iOS 16 and macOS 13, you can use float 16 [`MLMultiarrays`](https://developer.apple.com/documentation/coreml/mlmultiarray) for model inputs and outputs, and if you are using grayscale image types, you can now specify a new grayscale float 16 type. You can also specify an [`ImageType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#coremltools.converters.mil.input_types.ImageType) for input _and_ for output with `convert()`. The new float 16 types help eliminate extra casts at inputs and outputs for models that execute in float 16 precision. 

For details, see [Model Input and Output types](model-input-and-output-types). For image-specific options see [Image Input and Output](image-inputs).

