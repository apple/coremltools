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
coremltools.models.utils.save_spec(spec, 'path/to/the/saved/model.mlmodel')

# convert spec to MLModel, this step compiles the model as well
mlmodel = coremltools.models.MLModel(spec)

# Load the spec from the saved .mlmodel file directly
spec = coremltools.models.utils.load_spec('path/to/the/model.mlmodel')
```

## Visualizing Neural Network Core ML models

```python
import coremltools

mlmodel = coremltools.models.MLModel('path/to/the/model.mlmodel')
mlmodel.visualize_spec()

# To print a succinct description of the neural network
spec = mlmodel.get_spec()
from coremltools.models.neural_network.printer import print_network_spec

print_network_spec(spec, style='coding')
# or
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

## Prediction with an image input

An mlmodel that takes an input of type image requires a PIL image during the prediction call.

```python
import coremltools
import numpy as np
import PIL.Image

# load a model whose input type is "Image"
model = coremltools.models.MLModel('path/to/the/saved/model.mlmodel')

Height = 20  # use the correct input image height
Width = 60  # use the correct input image width


# Scenario 1: load an image from disk
def load_image(path, resize_to=None):
    # resize_to: (Width, Height)
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32)
    return img_np, img


# load the image and resize using PIL utilities
_, img = load_image('/path/to/image.jpg', resize_to=(Width, Height))
out_dict = model.predict({'image': img})

# Scenario 2: load an image from a numpy array
shape = (Height, Width, 3)  # height x width x RGB
data = np.zeros(shape, dtype=np.uint8)
# manipulate numpy data
pil_img = PIL.Image.fromarray(data)
out_dict = model.predict({'image': pil_img})
```

Now, let us say the Core ML model has an input type of MultiArray, but it really represents an image.
How can a jpeg image be used to call predict on such a model? For this, the loaded image should first
be converted to a numpy array. Here is one way to do it:

```python

Height = 20  # use the correct input image height
Width = 60  # use the correct input image width


# assumption: the mlmodel's input is of type MultiArray and of shape (1, 3, Height, Width)
model_expected_input_shape = (1, 3, Height, Width) # depending on the model description, this could be (3, Height, Width)

# load the model
model = coremltools.models.MLModel('path/to/the/saved/model.mlmodel')

def load_image_as_numpy_array(path, resize_to=None):
    # resize_to: (Width, Height)
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32) # shape of this numpy array is (Height, Width, 3)
    return img_np

# load the image and resize using PIL utilities
img_as_np_array = load_image_as_numpy_array('/path/to/image.jpg', resize_to=(Width, Height)) # shape (Height, Width, 3)

# note that PIL returns an image in the format in which the channel dimension is in the end,
# which is different than CoreML's input format, so that needs to be modified
img_as_np_array = np.transpose(img_as_np_array, (2,0,1)) # shape (3, Height, Width)

# add the batch dimension if the model description has it
img_as_np_array = np.reshape(img_as_np_array, model_expected_input_shape)

# now call predict
out_dict = model.predict({'image': img_as_np_array})
```  


## Building an mlmodel from scratch using Neural Network Builder

The neural network [builder class](https://github.com/apple/coremltools/blob/master/coremltools/models/neural_network/builder.py) can be used to programmatically construct a CoreML model.
Lets look at an example of
making a tiny 2 layer model with a convolution layer (with random weights) and an activation.

To find the list of all the neural network layer types supported see [this](https://github.com/aseemw/coremltools/blob/f95f9b230f6a1bd8b0d9ee298b78d7786e3e7cfd/mlmodel/format/NeuralNetwork.proto#L472)
portion of the NeuralNetwork.proto

```python
import coremltools
import coremltools.models.datatypes as datatypes
from coremltools.models import neural_network as neural_network
import numpy as np

input_features = [('data', datatypes.Array(*(1, 3, 10, 10)))]
output_features = [('output', None)]

builder = neural_network.NeuralNetworkBuilder(input_features, output_features,
                                              disable_rank5_shape_mapping=True)

builder.add_convolution(name='conv',
                        kernel_channels=3,
                        output_channels=3,
                        height=1,
                        width=1,
                        stride_height=1,
                        stride_width=1,
                        border_mode='valid',
                        groups=1,
                        W=np.random.rand(1, 1, 3, 3),
                        b=np.random.rand(3),
                        has_bias=True,
                        input_name='data',
                        output_name='conv')

builder.add_activation(name='prelu',
                       non_linearity='PRELU',
                       input_name='conv',
                       output_name='output',
                       params=np.array([1.0, 2.0, 3.0]))

spec = builder.spec
model = coremltools.models.MLModel(spec)
model.save('conv_prelu.mlmodel')

output_dict = model.predict({'data': np.ones((3, 10, 10))}, useCPUOnly=False)
print(output_dict['output'].shape)
print(output_dict['output'].flatten()[:3])
```

## Print out layer attributes for debugging

Sometimes we want to print out weights of a particular layer for debugging purposes.
Following is an example showing how we can utilize the `protobuf` APIs to access any
attributes include weight parameters. This code snippet uses the model we created in
the previous example.

```python
import coremltools
import numpy as np

model = coremltools.models.MLModel('conv_prelu.mlmodel')

spec = model.get_spec()
print(spec)

layer = spec.neuralNetwork.layers[0]
weight_params = layer.convolution.weights

print('Weights of {} layer: {}.'.format(layer.WhichOneof('layer'), layer.name))
print(np.reshape(np.asarray(weight_params.floatValue), (1, 1, 3, 3)))
```

## Quantizing a neural network mlmodel

```python
from coremltools.models.neural_network.quantization_utils import quantize_weights

model = coremltools.models.MLModel('model.mlmodel')
# Example 1: 8-bit linear
quantized_model = quantize_weights(model, nbits=8, quantization_mode="linear")

# Example 2: Quantize to FP-16 weights
quantized_model = quantize_weights(model, nbits=16)

# Example 3: 4-bit k-means generated look-up table
quantized_model = quantize_weights(model, nbits=4, quantization_mode="kmeans")

# Example 4: 8-bit symmetric linear quantization skipping bias,
# batchnorm, depthwise-convolution, and convolution layers
# with less than 4 channels or 4096 elements
from coremltools.models.neural_network.quantization_utils import AdvancedQuantizedLayerSelector

selector = AdvancedQuantizedLayerSelector(
    skip_layer_types=['batchnorm', 'bias', 'depthwiseConv'],
    minimum_conv_kernel_channels=4,
    minimum_conv_weight_count=4096)
quantized_model = quantize_weights(model, 8, quantization_mode='linear_symmetric',
                                   selector=selector)

# Example 5: 8-bit linear quantization skipping the layer with name 'dense_2'
from coremltools.models.neural_network.quantization_utils import QuantizedLayerSelector


class MyLayerSelector(QuantizedLayerSelector):

    def __init__(self):
        super(MyLayerSelector, self).__init__()

    def do_quantize(self, layer, **kwargs):
        ret = super(MyLayerSelector, self).do_quantize(layer)
        if not ret or layer.name == 'dense_2':
            return True


selector = MyLayerSelector()
quantized_model = quantize_weights(
    model, 8, quantization_mode='linear', selector=selector)
```
