```{eval-rst}
.. index:: 
    single: MLModel; utilities
    single: rename a feature
```


# MLModel Utilities

The following are useful utilities for processing `MLModel` objects. To learn more about MLModel, see [MLModel Overview](mlmodel). For the full list of utilities, see the [API Reference](https://apple.github.io/coremltools/source/coremltools.models.html#module-coremltools.models.utils).

## Rename a Feature

A _feature_ in this case refers to a model input or a model output. You can rename a feature in the specification using the [`rename_feature()`](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.utils.rename_feature) method. For example:

```python
import coremltools as ct

# Get the protobuf spec
model = ct.models.MLModel('MyModel.mlpackage')
spec = model.get_spec()
                        
# Edit the spec
ct.utils.rename_feature(spec, 'old_feature_name', 'new_feature_name')

# reload the model with the updated spec and re-save
model = ct.models.MLModel(spec)
# or 
model = ct.models.MLModel(spec, weights_dir=model.weights_dir) # if model is an mlprogram
model.save("MyModel.mlpackage")
```

To get the names of the inputs and outputs of the model: 

```python
import coremltools as ct

# Get the protobuf spec
model = ct.models.MLModel('MyModel.mlmodel')
spec = model.get_spec()

# get input names
input_names = [inp.name for inp in spec.description.input]

# get output names
output_names = [out.name for out in spec.description.output]
```

## Convert All Double Multi-array Feature Descriptions to Float

You can convert all `double` multi-array feature descriptions (input, output, and training input) to float multi-arrays using the [`convert_double_to_float_multiarray_type()`](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.utils.convert_double_to_float_multiarray_type) method. For example:

```python
import coremltools as ct

# Get the protobuf spec
model = ct.models.MLModel('MyModel.mlmodel')
spec = model.get_spec()
                        
# In-place convert multi-array type of spec
ct.utils.convert_double_to_float_multiarray_type(spec)
```

## Evaluate Classifier, Regressor, and Transformer Models

To test the correctness of a conversion, you can use use evaluation methods to compare predictions to the original framework. Use this type of evaluation for models that don’t deal with probabilities.

For example, you can evaluate a Core ML classifier model and compare it against predictions from the original framework using the [`evaluate_classifier()`](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.utils.evaluate_classifier) method, as shown in the following code snippet:

```python
import coremltools as ct

# Evaluate a classifier specification for testing.
metrics =  ct.utils.evaluate_classifier(spec, 
                   'data_and_predictions.csv', 'target')

# Print metrics
print(metrics)
{"samples": 10, num_errors: 0}
```

To evaluate a regressor model, use the [`evaluate_regressor()`](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.utils.evaluate_regressor) method:

```python
// Evaluate a CoreML regression model and compare against predictions from the original framework (for testing correctness of conversion)

metrics = coremltools.utils.evaluate_regressor(spec, 'data_and_predictions.csv', 'target')
print(metrics)
{"samples": 10, "rmse": 0.0, max_error: 0.0}
```

To evaluate a transformer model, use the [`evaluate_transformer()`](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.utils.evaluate_transformer) method:

```python
// Evaluate a transformer specification for testing.

input_data = [{'input_1': 1, 'input_2': 2}, {'input_1': 3, 'input_2': 3}]
expected_output = [{'input_1': 2.5, 'input_2': 2.0}, {'input_1': 1.3, 'input_2': 2.3}]
metrics = coremltools.utils.evaluate_transformer(scaler_spec, input_data, expected_output)
```


```{eval-rst}
.. index:: 
    single: weight metadata
```


## Get Weights Metadata

The  [`get_weights_metadata()`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.get_weights_metadata) utility provides a convenient way to inspect the properties of the weights in the model. You may want to use it to identify weights that are larger than a certain size, or have a sparsity greater than a certain percentage, or have a certain number of unique elements. 

For example, if you want to compare the weights of a model showing unexpected results with the weights of a model with predictable results, you can use `get_weights_metadata()` to get a list of all the weights with their metadata. Use the `weight_threshold` parameter to set which weights are returned. A weight is included in the resulting dictionary only if its total number of elements are greater than `weight_threshold`. 

The metadata returned by the utility also offers information about the child ops the weight feeds into. The data returned by the API can then be used to customize the optimization of the model via the `ct.optimize.coreml` API. 

### Using the Metadata 

The  [`get_weights_metadata()`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.get_weights_metadata) utility returns the weights metadata as a  [CoreMLWeightMetaData](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.CoreMLWeightMetaData) dictionary mapping each weight’s name to its metadata. These results are useful when constructing the [`coremltools.optimize.coreml.OptimizationConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig) API.

For example, with the [OptimizationConfig](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig) class you have fine-grain control over applying different optimization configurations to different weights by directly setting `op_type_configs` and `op_name_configs` or using [`set_op_name`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig.set_op_name) and [`set_op_type`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig.set_op_type). When using [`set_op_name`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig.set_op_name), you need to know the name for the `const` op that produces the weight. The  `get_weights_metadata()` utility provides the weight name and the corresponding weight numpy data, along with meta information such as the sparsity. 

To represent weights with sparse representation whose elements are greater than 3000 and the percentage of 0s are greater than 75%, you can call `get_weights_metadata()` and iterate through all weights to filter out only those weights. The results might be two weights (such as `["weight_1", "weight_4"])`, and you can then use  [`set_op_name`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig.set_op_name) or directly set  `op_name_configs` accordingly.

The utility is also useful for the [`set_op_type()`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig.set_op_type) API, which lets you set the optimization config for all weights that feed into the same type of op, such as `conv`. The  `get_weights_metadata()` utility provides information about what op the weight feeds into (the `child_ops` attribute of the CoreMLWeightMetaData), so that you can utilize the information to validate the intent of optimizing those weights.


### Using a Weight Threshold

The following example includes weights with sizes greater than 2048. The code loads the `MobileNetV2.mlpackage` converted in [Getting Started](introductory-quickstart) (from the previously trained [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2) model, which is based on the [tensorflow.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)). It uses [`get_weights_metadata()`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.get_weights_metadata), which specifies the size threshold (`2048`) in the `weight_threshold` parameter. A weight tensor is included in the resulting dictionary only if its total number of elements are greater than `weight_threshold`:

```python
import coremltools as ct

mlmodel = ct.models.MLModel("MobileNetV2.mlpackage")
weight_metadata_dict = ct.optimize.coreml.get_weights_metadata(
    mlmodel, weight_threshold=2048
)
```

You can then include weights in the dictionary based on their metadata:

```python
# get the weight names with size > 25600
large_weights = []
for k, v in weight_metadata_dict.items():
    if v.val.size >= 25600:
        large_weights.append(k)

# get the weight names with sparsity >= 50%
sparse_weights = []
for k, v in weight_metadata_dict.items():
    if v.sparsity >= 0.5:
        sparse_weights.append(k)

# get the weight names with unique elements <= 16
palettized_weights = []
for k, v in weight_metadata_dict.items():
    if v.unique_values <= 16:
        palettized_weights.append(k)
```

```python
# Access to each weight metadata (type of CoreMLWeightMetaData).
weight_meta_data = weight_metadata_dict["mobilenetv2_1_00_224_block_1_project_BN_FusedBatchNormV3_nchw_weight_0_to_fp16"]
print("---")
print("numpy array data for the weight: ")
print(weight_meta_data.val) # This is the numpy array data for the weight
print("---")
print("Sparsity of the weight: ", weight_meta_data.sparsity) # The sparsity of the weight
print("---")
print("Number of unique values in the weight: ", weight_meta_data.unique_values) # Number of unique values in the weight
print("---")

# Access to the child ops this weight feeds into.
# In this example, the weight feeds into only one op (conv_5),
# as the result, weight_meta_data.child_ops is a list of length 1.
child_op = weight_meta_data.child_ops[0]
print("name of the child op: ", child_op.name) # name of the child op
print("---")
print("op type of the child op: ", child_op.op_type) # op type of the child op
print("---")
print("dictionary: ", child_op.params_name_mapping) # this dictionary shows the input parameters of the child op and their corresponding ops' names
print("---")
```


The output from the above example would be:

```
---
numpy array data for the weight: 
[[[[ 0.4062  ]]

  [[-0.04984 ]]

  [[-0.4714  ]]

  ...

  [[-0.4846  ]]

  [[ 0.4563  ]]

  [[ 0.001579]]]


 [[[ 0.746   ]]

  [[-0.208   ]]

  [[ 0.3416  ]]

  ...

  [[ 0.1913  ]]

  [[-0.0241  ]]

  [[-0.02843 ]]]


 [[[ 0.094   ]]

  [[ 0.778   ]]

  [[ 0.315   ]]

  ...

  [[-0.5137  ]]

  [[-0.3682  ]]

  [[-1.2     ]]]


 ...


 [[[-0.1445  ]]

  [[ 0.6494  ]]

  [[ 0.703   ]]

  ...

  [[ 0.1827  ]]

  [[ 0.2012  ]]

  [[-0.391   ]]]


 [[[-0.04352 ]]

  [[ 1.12    ]]

  [[ 0.5605  ]]

  ...

  [[ 0.4165  ]]

  [[-0.4824  ]]

  [[ 0.5176  ]]]


 [[[-0.2922  ]]

  [[ 0.3481  ]]

  [[-0.2795  ]]

  ...

  [[-0.3572  ]]

  [[ 0.2546  ]]

  [[ 0.0194  ]]]]
---
Sparsity of the weight:  0.0
---
Number of unique values in the weight:  2077
---
name of the child op:  mobilenetv2_1_00_224_block_1_project_BN_FusedBatchNormV3_nchw_cast_fp16
---
op type of the child op:  conv
---
dictionary:  OrderedDict([('weight', 'mobilenetv2_1_00_224_block_1_project_BN_FusedBatchNormV3_nchw_weight_0_to_fp16')])
---
```


