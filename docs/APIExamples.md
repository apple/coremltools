# API Code snippets

## Converting between MLModel and Spec
 
```python
import coremltools

# Load MLModel
mlmodel = coremltools.models.MLModel('path/to/the/model.mlmodel')

# use model for prediction
mlmodel.predict(...)

# save the model
mlmodel.save('path/to/the/saved/model.mlmodel')

# Get spec from the model
spec = mlmodel.get_spec()

# print input/output description for the model
print(spec.description)

# get the type of Model (NeuralNetwork, SupportVectorRegressor, Pipeline etc)
print(spec.WhichOneof('Type'))

# save out the model directly from the spec
coremltools.models.utils.save_spec(spec,'path/to/the/saved/model.mlmodel')

# convert spec to MLModel, this step compiles the model as well
mlmodel = coremltools.models.MLModel(spec)

# Load the spec from the saved .mlmodel file directly
spec = coremltools.models.utils.load_spec('path/to/the/model.mlmodel')
```

## Visualizing Neural Network CoreML models
```python
import coremltools

nn_mlmodel = coremltools.models.MLModel('path/to/the/model.mlmodel')
nn_mlmodel.visualize_spec()

# To print a succinct description of the neural network
spec = nn_mlmodel.get_spec()
from  coremltools.models.neural_network.printer import print_network_spec
print_network_spec(spec)
```

Another useful tool for visualizing CoreML models and models from other frameworks: [Netron](https://github.com/lutzroeder/netron)

## Printing the pre-processing parameters 

This is useful for image based neural network models

```python
import coremltools

spec = coremltools.models.utils.load_spec('path/to/the/saved/model.mlmodel')

# Get neural network portion of the spec
if spec.WhichOneof('Type') == 'neuralNetworkClassifier':
  nn = spec.neuralNetworkClassifier
if spec.WhichOneof('Type') == 'neuralNetwork':
  nn = spec.neuralNetwork
elif spec.WhichOneof('Type') == 'neuralNetworkRegressor':
  nn = spec.neuralNetworkRegressor
else:
    raise ValueError('MLModel must have a neural network')
    
print(nn.preprocessing)

```

## Changing MLMultiArray input/output datatypes

[Here](https://github.com/apple/coremltools/blob/d07421460f9f0ad1a2e9cf8b5248670358a24a1a/mlmodel/format/FeatureTypes.proto#L106 ) is the list of supported datatypes.
For instance, change the datatype from 'double' to 'float32': 

```python
import coremltools
from coremltools.proto import FeatureTypes_pb2 as ft

model = coremltools.models.MLModel('path/to/the/saved/model.mlmodel')
spec = model.get_spec()

def _set_type_as_float32(feature):
  if feature.type.HasField('multiArrayType'):
    feature.type.multiArrayType.dataType = ft.ArrayFeatureType.FLOAT32

# iterate over the inputs
for input_ in spec.description.input:
    _set_type_as_float32(input_)
    
# iterate over the outputs
for output_ in spec.description.output:
    _set_type_as_float32(output_)

model = coremltools.models.MLModel(spec)
model.save('path/to/the/saved/model.mlmodel')

```