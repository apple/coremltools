```{eval-rst}
.. index:: 
    single: input type options
    single: output type options
    single: MLMultiArray
    single: ImageType
```

# Model Input and Output Types

When using the Core ML Tools [Unified Conversion API](unified-conversion-api), you can specify various properties for the model inputs and outputs using the `inputs` and `outputs` parameters for [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry). The following summarizes the key options.

## Use the Default Behavior

The `convert()` method generates by default a [Core ML](https://developer.apple.com/documentation/coreml) model with a multidimensional array ([`MLMultiArray`](https://developer.apple.com/documentation/coreml/mlmultiarray)) as the type for both input and output. The data types, names, and shapes are picked up automatically from a TensorFlow source model. For a PyTorch model you must provide the input shape.

In Core ML Tools 7.0 and newer versions, the default input/output dtype for models converted to the `mlprogram` type are float 16 for the `iOS16`/`macOS13` and newer deployment targets. Also, the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method produces an `mlprogram` (ML program) by default with an `iOS15`/`macOS12` or newer deployment target. For more details, see [Convert Models to ML Programs](convert-to-ml-program).


## Use Images

To produce a Core ML model with images for input and output, use the [`ImageType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#imagetype) class to specify the `inputs` and `outputs` parameters for `convert()`. For details and examples, see [Image Input and Output](image-inputs). 

## Provide the Shape of the Input

To convert PyTorch models, you must provide the input shape using the `shape` parameter with  [`TensorType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#tensortype) or [`ImageType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#imagetype), since the PyTorch traced program does not contain the shape information.

For TensorFlow models, the shape is automatically picked up from the model. However, it is good practice to provide at least a static shape, which enables the converter to apply graph optimizations and produce a more efficient model. For variable input shapes use [`EnumeratedShapes`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html?highlight=enumeratedshapes#enumeratedshapes). For details and an example, see [Select from Predetermined Shapes](https://coremltools.readme.io/docs/flexible-inputs#select-from-predetermined-shapes). 


```{eval-rst}
.. index:: dtype
```

## Set the dtype

Use the `dtype` parameter with [`TensorType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#tensortype) to override data types (such as float 32, float 16, and integer). The `dtype` parameter can take either a [NumPy](https://numpy.org/) `dtype` (such as `np.float32` and `np.int32`) or an MIL type with `TensorType` (such as `coremltools.converters.mil.mil.types.fp32`).

Starting in coremltools version 6, you can use the `np.float16` type with ML programs, which can reduce the overhead of input and output type conversions for `float16` typed models (which is the default precision for ML programs). For more information, see [ML Programs](convert-to-ml-program).

For example, the following code snippet converts the `source_model` to a Core ML model with float 16 multiarray (`TensorType`) input and output:

```python
# to produce a model with float 16 input and output of type multiarray
mlmodel = ct.convert(
    source_model,
    inputs=[ct.TensorType(shape=input.shape, dtype=np.float16)],
    outputs=[ct.TensorType(dtype=np.float16)],
    minimum_deployment_target=ct.target.iOS16,
)
```

```{eval-rst}
.. index:: 
    single: PyTorch; set names
```

## Set Names for PyTorch Conversion

For PyTorch model conversion, use `name` with [`TensorType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#tensortype) to set the input and output names of the converted model. For example, the following code snippet will produce a Core ML model with `inputs` named `my_input_name` and `outputs` named `my_output_name`:

```python
mlmodel = ct.convert(
    source_torch_model,
    inputs=[ct.TensorType(shape=input.shape, name="my_input_name")],
    outputs=[ct.TensorType(name="my_output_name")],
    minimum_deployment_target=ct.target.iOS16,
)
```

For TensorFlow conversions, the names are picked up automatically from the TF graph. Unlike PyTorch models in which the inputs and outputs are ordered, with TensorFlow models you can’t provide your own names, because in the TF graph the input and output tensors are referred to by the names. After converting the model, you can change the names of the inputs and outputs using the [`rename_feature()`](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.utils.rename_feature) method. For an example, see [Rename a Feature](mlmodel-utilities.md#rename-a-feature).
