
```{eval-rst}
.. index:: 
   single: MLModel; overview and spec
   single: prediction; with MLModel
```

# MLModel Overview

An MLModel encapsulates a [Core ML](https://developer.apple.com/documentation/coreml) model's prediction methods, configuration, and model description. You can use the `coremltools` package to convert trained models from a variety of training tools into Core ML models. For the full list of model types, see [Core ML Model](https://apple.github.io/coremltools/mlmodel/Format/Model.html).

The converters in `coremltools` return a converted model as an `MLModel` object. You can then save the `MLModel` as an `.mlmodel` file, use it to make predictions, modify the input and output descriptions, and set the model's metadata. The [MLModel class](https://apple.github.io/coremltools/source/coremltools.models.html#module-coremltools.models.model) defines the minimal interface to a Core ML model in Python. 

## Core ML Specification

A key component of Core ML is the public specification for representing machine learning models. This specification is defined in the [Model.proto](https://github.com/apple/coremltools/blob/master/mlmodel/format/Model.proto "Model.proto on GitHub") protobuf (for a more readable version, see [Core ML Model](https://apple.github.io/coremltools/mlmodel/Format/Model.html)), which can be created using any language supported by protobuf (such as Python, C++, Java, C#, and Perl).

At a high level, the protobuf specification consists of the following:

- _Model description:_ The model name and and the types of input and output.
- _Model parameters:_ The set of parameters required to represent a specific instance of the model.
- _Metadata:_ Information about the origin, license, and author of the model.

## Examples

The following code snippets demonstrate how to use an MLModel. To use these code snippets, first import `coremltools` as follows:

```python
import coremltools as ct
```

### Load and Save the MLModel

```python
# Load the MLModel
mlmodel = ct.models.MLModel('path/to/the/model.mlmodel')

# Save the MLModel
mlmodel.save('path/to/the/saved/model.mlmodel')
```

### Use the MLModel for Prediction

```python
# Use the MLModel for prediction
mlmodel.predict(...)
```

### Work With the Spec

The `spec` object is the parsed protobuf object of the Core ML model. The following code snippets show how you can get the spec from the MLModel, print its description, get the type of MLModel, and save the MLModel directly from the `spec`. It also shows how to convert the `spec` to MLModel and compile the model in one step:

```python
# Get the spec from the MLModel
spec = mlmodel.get_spec()

# Print the input/output description for the MLModel
print(spec.description)

# Get the type of MLModel (NeuralNetwork, SupportVectorRegressor, Pipeline etc)
print(spec.WhichOneof('Type'))

# Save out the MLModel directly from the spec
ct.models.utils.save_spec(spec, 'path/to/the/saved/model.mlmodel')

# Convert spec to MLModel. This step also compiles the model.
mlmodel = ct.models.MLModel(spec)

# Load the spec from the saved .mlmodel file directly
spec = ct.models.utils.load_spec('path/to/the/model.mlmodel')
```

### Update the Metadata and Input/Output Descriptions

The following example converts a model from [Scikit-learn](https://scikit-learn.org/stable/). The model predicts the price of a house based on three features (bedroom, bath, size). 

The example shows how you can use the MLModel object returned by the conversion to update the metadata and input/output descriptions that are displayed in the Xcode preview. (For details about the Xcode preview, see [Xcode Model Preview Types](xcode-model-preview-types).

```python
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load data
data = pd.read_csv('houses.csv')

# Train a model
model = LinearRegression()
model.fit(data[["bedroom", "bath", "size"]], data["price"])

# Convert and save the scikit-learn model
import coremltools as ct

model = ct.converters.sklearn.convert(model, ["bedroom", "bath", "size"], "price")

# Set model metadata
model.author = 'John Smith'
model.license = 'BSD'
model.short_description = 'Predicts the price of a house in the Seattle area.'
model.version = '1'

# Set feature descriptions manually
model.input_description['bedroom'] = 'Number of bedrooms'
model.input_description['bathrooms'] = 'Number of bathrooms'
model.input_description['size'] = 'Size (in square feet)'

# Set the output descriptions
model.output_description['price'] = 'Price of the house'

# Save the model
model.save('HousePricer.mlmodel')
```

