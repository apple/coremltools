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

The  [`get_weights_metadata()`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.get_weights_metadata) utility provides a convenient way to inspect the properties of the weights in the model. You may want to use it for debugging, or for identifying weights that are larger than a certain size, have a sparsity greater than a certain percentage, or have a certain number of unique elements. 

For example, if you want to compare the weights of a model showing unexpected results with the weights of a model with predictable results, you can use `get_weights_metadata()` to get a list of all the weights with their metadata. Use the `weight_threshold` parameter to set which weights are returned. A weight is included in the resulting dictionary only if its total number of elements are greater than `weight_threshold`. 

The metadata returned by the utility also offers 
information about the child ops the weight feeds into. 
The data returned by the API can then be used to customize the 
optimization of the model via the `ct.optimize.coreml` API. 


### Using the Metadata 

The  [`get_weights_metadata()`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.get_weights_metadata) utility returns the weights metadata as an ordered dictionary that maps to strings in [CoreMLWeightMetaData](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.CoreMLWeightMetaData) and preserves the sequential order of the weights. The results are useful when constructing [`cto.coreml.OptimizationConfig`](https://apple.github.io/coremltools/docs-guides/source/optimizecoreml-api-overview.html#customizing-ops-to-compress).

For example, with the [OptimizationConfig](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig) class you have fine-grain control over applying different optimization configurations to different weights by directly setting `op_type_configs` and `op_name_configs` or using [`set_op_name`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig.set_op_name) and [`set_op_type`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig.set_op_type). When using [`set_op_name`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig.set_op_name), you need to know the name for the `const` op that produces the weight. The  `get_weights_metadata()` utility provides the weight name and the corresponding weight numpy data, along with metadata information. 


### Example

The following code loads the `SegmentationModel_with_metadata.mlpackage` saved in [Converting a PyTorch Segmentation Model](pytorch-conversion-examples.md#open-the-model-in-xcode). It uses [`get_weights_metadata()`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.get_weights_metadata), which specifies the size threshold (`2048`) in the `weight_threshold` parameter. A weight tensor is included in the resulting dictionary only if its total number of elements are greater than `weight_threshold`.

The example also shows how to get the name of the last weight in the model. The code palettizes all ops except the last weight, which is a common practical scenario when the last layer is more sensitive and should be skipped from quantization:

```python
import coremltools.optimize as cto

from coremltools.models import MLModel
from coremltools.optimize.coreml import get_weights_metadata

mlmodel = MLModel("SegmentationModel_with_metadata.mlpackage")
weight_metadata_dict = get_weights_metadata(mlmodel, weight_threshold=2048)

# iterate over all the weights returned 

large_weights = [] # get the weight names with size > 25600
sparse_weights = [] # get the weight names with sparsity >= 50% 
palettized_weights = [] # get the weight names with unique elements <= 16 

for weight_name, weight_metadata in weight_metadata_dict.items():
     # weight_metadata.val: numpy array of the weight data
     if weight_metadata.val.size >= 25600: large_weights.append(weight_name)

     # weight_metadata.sparsity: ratio of 0s in the weight, between [0,1]
     if weight_metadata.sparsity >= 0.5: sparse_weights.append(weight_name)

     # weight_metadata.unique_values : number of unique values in the weight
     if weight_metadata.unique_values <= 16: palettized_weights.append(weight_name)

     # Access to the child ops this weight feeds into.  
     child_op = weight_metadata.child_ops[0]
     # child_op.name: name of the child op
     # child_op.op_type:  op type of the child op
     # child_op.params_name_mapping:  this dictionary shows the input parameters of the child op and their corresponding ops' names

# Palettize all weights except for the last weight
last_weight_name = list(weight_metadata_dict.keys())[-1]
global_config = cto.coreml.OpPalettizerConfig(nbits=6, mode="kmeans")
config = cto.coreml.OptimizationConfig(
    global_config=global_config,
    op_name_configs={last_weight_name: None},
)
compressed_mlmodel = cto.coreml.palettize_weights(mlmodel, config)

```

## Bisect Model

In certain scenarios, you may want to break a large Core ML model into two smaller models. For instance, if you are deploying a model to run on neural engine on an iPhone, it cannot be larger than 1 GB. If you are working with, say, [Stable Diffusion](https://github.com/apple/ml-stable-diffusion) 1.5 model which is 1.72 GB large (Float 16 precision), then it needs to be broken up into two chunks, each less than 1 GB. The utility `ct.models.utils.bisect_model` will allow you to do exactly that. When using this API, you can also opt-in to package the two chunks of the model into a pipeline model, so that its still a single mlpackage file, with the two models arranged in a sequential manner.

The example below shows how to bisect a model, test the accuracy, and save them on disk.

```python

import coremltools as ct

model_path = "my_model.mlpackage"
output_dir = "./output/"

# The following code will produce two smaller models:
# `./output/my_model_chunk1.mlpackage` and `./output/my_model_chunk2.mlpackage`
# It also compares the output numerical of the original Core ML model with the chunked models.
ct.models.utils.bisect_model(
    model_path,
    output_dir,
)

# The following code will produce a single pipeline model `./output/my_model_chunked_pipeline.mlpackage`
ct.models.utils.bisect_model(
    model_path,
    output_dir,
    merge_chunks_to_pipeline=True,
)

# You can also pass the MLModel object directly
mlmodel = ct.models.MLModel(model_path)
ct.models.utils.bisect_model(
    mlmodel,
    output_dir,
    merge_chunks_to_pipeline=True,
)
```
