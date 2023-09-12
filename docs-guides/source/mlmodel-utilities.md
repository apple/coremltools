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
