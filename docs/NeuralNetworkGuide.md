# Neural Network Guide

This document describes how to get neural network models into the Core ML format, either via automatic conversion or by building them
from scratch pragmatically. We also discuss various utilities available to edit the `mlmodel` such as quantization, making the input shape
flexible, changing the input/output names, types, inspecting mlmodels, printing a text description of the model etc.

What are the layers supported by Core ML? For the latest list along with all the parameterizations, check out the
[neuralnetwork.proto](https://github.com/apple/coremltools/blob/master/mlmodel/format/NeuralNetwork.proto) file, which is a [protobuf](https://developers.google.com/protocol-buffers/docs/pythontutorial)
description of a neural network model.
Since its a big file, its easier to navigate either by starting from the [top level proto message](https://github.com/apple/coremltools/blob/875abd9707dbe65eb92a31dbb54a68d6581e68ad/mlmodel/format/NeuralNetwork.proto#L130)
or by directly looking at the [layer types](https://github.com/apple/coremltools/blob/875abd9707dbe65eb92a31dbb54a68d6581e68ad/mlmodel/format/NeuralNetwork.proto#L472).
An auto-generated documentation, built from the comments in the proto file, can be found [here](https://apple.github.io/coremltools/coremlspecification/sections/NeuralNetwork.html).


Please make sure that you have installed the latest `coremltools`, `tfcoreml` (if using tensorflow converter) and `onnx-coreml` (if using ONNX converter) python packages.  

```bash
pip install --upgrade coremltools
pip install --upgrade tfcoreml
pip install --upgrade onnx-coreml
```

[Jupyter notebook examples for converters](../examples/neural_network_inference/) 

# Table of Contents

* [Keras.io converter (TF 1.x backend)](#kerasio-converter-tf-1x-backend)
* [TensorFlow conversion](#TensorFlow-conversion)
    * [TensorFlow 1 converter](#tensorFlow-1-converter)
    * [TensorFlow 2 converter (tf.keras)](#tensorflow-2-converter-tfkeras)
* [ONNX converter (PyTorch conversion)](#ONNX-converter)
* [Building an mlmodel using the Builder API](#Building-an-mlmodel-using-the-Builder-API)
* [Model quantization](#Model-quantization)
* [Model prediction](#Model-predictions)
* [Model inspection and editing](#Model-inspection-and-editing)
  * [Printing description](#Printing-description)
  * [Flexible input/output shapes](#Flexible-inputoutput-shapes)
  * [Modifying input/output names and types](#Modifying-inputoutput-names-and-types)
  * [Inspecting model for debugging](#Inspecting-model-for-debugging)
  * [Miscellaneous Examples](#Miscellaneous-examples)

## Keras.io Converter (TF 1.x backend)

Models created via the [Keras.io API](https://keras.io), with Tensorflow 1.x backend,
and saved in the `.h5` format can be converted to Core ML.

The coremltools Keras converter supports Keras versions 2.2+.
(Versions below 2.2, up to 1.2.2 are supported, however they are no longer
maintained, i.e. no bug fixes will be made for versions below 2.2)

```python
# convert by providing path to a .h5 file
mlmodel = coremltools.converters.keras.convert('keras_model.h5')
mlmodel.save('coreml_model.mlmodel')

# convert by providing a Keras model object
from keras.models import load_model
keras_model = load_model("keras_model.h5")
mlmodel = coremltools.converters.keras.convert(keras_model)
```

The convert function can take several additional arguments, such as:

* `input_names`, `output_names`: to provide custom names for inputs and outputs
* `image_input_names`: to get an mlmodel such that its input is of type image
* `is_bgr`, `red_bias`, `green_bias`, `blue_bias`, `image_scale`: to provide parameters for image pre-processing
  when the input is of type image (i.e. if `image_input_names` is being used).
* `class_labels`, `predicted_feature_name`: to produce an mlmodel of type neural network classifier
* `model_precision`: to produce a quantized model. Equivalently, the mlmodel can be quantized post conversion as well, see the section
  on [Model quantization](#Model-quantization)
* `respect_trainable`: to produce an updatable mlmodel. See examples [here](https://github.com/apple/coremltools/tree/master/examples/updatable_models)
* `add_custom_layers`, `custom_conversion_functions`: to add a custom layer in the generated mlmodel.
  This is useful when Keras has a layer (native or lambda) that Core ML does not support.
  For a description of CoreML custom layers see this nice [overview](http://machinethink.net/blog/coreml-custom-layers/).

For a complete list of arguments that can be passed to the convert method, see [here](https://apple.github.io/coremltools/generated/coremltools.converters.keras.convert.html)
or at the [function signature in code](https://github.com/apple/coremltools/blob/875abd9707dbe65eb92a31dbb54a68d6581e68ad/coremltools/converters/keras/_keras_converter.py#L344)

#### Troubleshooting

* conversion of models defined via the `tf.keras` API is not supported via the coremltools keras converter.
  However, when `tf.keras` is used in TensorFlow 2.x and the model exported to `.h5` format, it can be converted via the 
  [TensorFlow converter](#tensorflow-converter-tf-1x-tf-2-tfkeras-with-tf-2). 

* models with Keras lambda layers: use `custom_conversion_functions`, so that Keras lambda layers can be mapped to Core ML custom layers

* What if the converter errors out due to an unsupported layer, or an unsupported parameter in a layer?
  The coremltools Keras converter targets the Core ML specification version 3, the one released during the macOS 10.14, iOS 12 release cycle.
  Majority of the native layers of the `keras.io` API can be mapped to the iOS 12 Core ML layers.
  However if a conversion error due to an unsupported layer comes up, the recommended route is one of the following:

  - Upgrade to TensorFlow 2.x, and then use the newer `tf.keras` API and convert to Core ML via the 
  [TensorFlow converter](#tensorflow-converter-tf-1x-tf-2-tfkeras-with-tf-2)
  - With TensorFlow 1.x, save the Keras model as a frozen graph def file in `.pb` format, instead of `.h5`. 
  Then use the [TensorFlow converter](#tensorflow-converter-tf-1x-tf-2-tfkeras-with-tf-2).   
  Example: 
  
```python
from keras.models import Sequential
from keras.layers import Dense, ReLU

h5_path = '/tmp/keras_model.h5'
pb_path = '/tmp/keras_model.pb'
mlmodel_path = '/tmp/keras_model.mlmodel'

model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(ReLU())
model.save(h5_path)

input_tensor_shapes, output_node_names = _save_h5_as_frozen_pb(h5_path, pb_path) # defined below

# convert the .pb file to .mlmodel via tfcoreml converter
import tfcoreml
mlmodel = tfcoreml.convert(
        tf_model_path = pb_path,
        mlmodel_path = mlmodel_path,
        output_feature_names = output_node_names,
        input_name_shape_dict = input_tensor_shapes,
        minimum_ios_deployment_target='13')
```

Function to convert .h5 to .pb in Tensorflow 1.x:

```python
def _save_h5_as_frozen_pb(h5_path, frozen_model_path, has_variables=True):
    from keras.models import load_model
    from keras import backend as K
    import tensorflow as tf
    import shutil, tempfile, os
    from tensorflow.python.tools.freeze_graph import freeze_graph

    K.set_learning_phase(0)
    model = load_model(h5_path)
    model_dir = tempfile.mkdtemp()
    graph_def_file = os.path.join(model_dir, 'tf_graph.pb')
    checkpoint_file = os.path.join(model_dir, 'tf_model.ckpt')

    output_node_names = []
    if isinstance(model.output, list):
        for idx in range(len(model.output)):
            output_node_names.append(model.output[idx].name[:-2])
    else:
        output_node_names.append(model.output.name[:-2])

    tf_graph = K.get_session().graph
    tf.reset_default_graph()
    if has_variables:
        with tf_graph.as_default() as g:
            saver = tf.train.Saver()

    with tf.Session(graph=tf_graph) as sess:
        sess.run(tf.global_variables_initializer())
        # save graph definition somewhere
        tf.train.write_graph(sess.graph, model_dir, graph_def_file, as_text=False)
        # save the weights
        if has_variables:
            saver.save(sess, checkpoint_file)

    K.clear_session()

    # freeze the graph
    if has_variables:
        freeze_graph(input_graph=graph_def_file,
                     input_saver="",
                     input_binary=True,
                     input_checkpoint=checkpoint_file,
                     output_node_names=",".join(output_node_names),
                     restore_op_name="save/restore_all",
                     filename_tensor_name="save/Const:0",
                     output_graph=frozen_model_path,
                     clear_devices=True,
                     initializer_nodes="")

    if os.path.exists(model_dir):
      shutil.rmtree(model_dir)

    input_tensor_shapes = {}
    if isinstance(model.input, list):
        for idx in range(len(model.input)):
            input_shape = [i for i in model.input_shape[idx]]
            for i, d in enumerate(input_shape):
                if d is None:
                    input_shape[i] = 1

            input_tensor_shapes[model.input[idx].name[:-2]] = input_shape
    else:
        input_shape = [i for i in model.input_shape]
        for i, d in enumerate(input_shape):
            if d is None:
                input_shape[i] = 1
        input_tensor_shapes[model.input.name[:-2]] = input_shape

    return input_tensor_shapes, output_node_names

```
  


Note: an alternative route that can be used in all of the cases above
 is to first convert the [Keras model to the `.onnx` format](https://github.com/onnx/keras-onnx) and then use the [ONNX converter](#ONNX-converter) described below.
 The ONNX converter has been updated to target all Core ML specification versions from 1 to 4 (iOS 11 to iOS 13).

## TensorFlow conversion

TensorFlow models can be converted to Core ML by using the `tfcoreml` converter
([link](https://github.com/tf-coreml/tf-coreml) to the GitHub repo), which
depends on the coremltools package.

```bash
pip install --upgrade tfcoreml
```

## TensorFlow 1 converter

To convert models trained/saved via TensorFlow 1, first export them into the frozen graph def format, which is a protobuf file
format with `.pb` as the extension. Frozen `.pb` files can be obtained by using TensorFlow's
`tensorflow.python.tools.freeze_graph` utility.

[This](../examples/neural_network_inference/tensorflow_converter/Tensorflow_1/linear_mnist_example.ipynb) Jupyter notebook shows how to freeze a graph to produce a `.pb` file.

There are several other Jupyter notebook examples for conversion 
[here](../examples/neural_network_inference/tensorflow_converter/Tensorflow_1).

```python
import tfcoreml

tfcoreml.convert(tf_model_path='my_model.pb',
                 mlmodel_path='my_model.mlmodel',
                 output_feature_names=['softmax:0'],  # name of the output tensor (appended by ":0")
                 input_name_shape_dict={'input:0': [1, 227, 227, 3]},  # map from input tensor name (placeholder op in the graph) to shape
                 minimum_ios_deployment_target='12')

# if the above invocation fails with an error, then update the
# minimum_ios_deployment_target to invoke the newer converter path:

tfcoreml.convert(tf_model_path='my_model.pb',
                 mlmodel_path='my_model.mlmodel',
                 output_feature_names=['softmax'],  # name of the output op
                 input_name_shape_dict={'input': [1, 227, 227, 3]},  # map from the placeholder op in the graph to shape (can have -1s)
                 minimum_ios_deployment_target='13')
```

The argument `minimum_ios_deployment_target` controls the set of Core ML layers that are used by the converter.
When its value is set to `'12'`, only the set of layers that were shipped in Core ML during the iOS 12, macOS 14 release cycle are used.
It is recommended to first use this setting, since if successful, it produces a Core ML model that can be deployed to iOS 12 and higher.
In case, it results in an error due to an unsupported op or parameter, then the target should be set to `'13'`, so that the
converter can utilize all the layers (including control flow, recurrent layers etc) that were shipped in Core ML in iOS 13.

## TensorFlow 2 converter (tf.keras)

There are 3 ways to export an inference graph in TensorFlow 2: 

1. Use the `tf.keras` APIs (Sequential, Functional, or Subclassing) and export to the `.h5` HDF5 file or `SavedModel` directory format.
2. Use TensorFlow's low-level APIs (along with `tf.keras`) and export to a `SavedModel` directory format.
3. Use TensorFlow's low-level APIs and export as `concrete functions` format.

For all 3 cases `tfcoreml`'s `convert()` function can be used to convert your model into Core ML model format. The argument `minimum_ios_deployment_target` must be set to `'13'`.

**Converting a `tf.keras` HDF5 model**:

```python
from tensorflow.keras.applications import ResNet50
import tfcoreml

keras_model = ResNet50(weights=None, input_shape=(224, 224, 3))
keras_model.save('./model.h5')

# print input name, output name, input shape
print(keras_model.input.name)
print(keras_model.input_shape)
print(keras_model.output.name)


model = tfcoreml.convert('./model.h5',
                         input_name_shape_dict={'input_1': (1, 224, 224, 3)},
                         output_feature_names=['Identity'],
                         minimum_ios_deployment_target='13')

model.save('./model.mlmodel')
```

```python
import tensorflow as tf
import tfcoreml

keras_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

keras_model.save('/tmp/keras_model.h5')

# print input name, output name, input shape
print(keras_model.input.name)
print(keras_model.input_shape)
print(keras_model.output.name)

model = tfcoreml.convert(tf_model_path='/tmp/keras_model.h5',
                         input_name_shape_dict={'flatten_input': (1, 28, 28)},
                         output_feature_names=['Identity'],
                         minimum_ios_deployment_target='13')
model.save('/tmp/keras_model.mlmodel')
```

**Converting a SavedModel:**

```python
from tensorflow.keras.applications import MobileNet
import tfcoreml

keras_model = MobileNet(weights=None, input_shape=(224, 224, 3))
keras_model.save('./savedmodel', save_format='tf')
# tf.saved_model.save(keras_model, './savedmodel')

model = tfcoreml.convert('./savedmodel',
                         mlmodel_path='./model.mlmodel',
                         input_name_shape_dict={'input_1': (1, 224, 224, 3)},
                         output_feature_names=['Identity'],
                         minimum_ios_deployment_target='13')
```

See notebooks in [here](../examples/neural_network_inference/tensorflow_converter/Tensorflow_2) 
or the [unit test cases](https://github.com/apple/coremltools/blob/master/coremltools/converters/tensorflow/test/test_tf_2x.py) for more examples, on how to save to `.h5` or `SavedModel` or `concrete functions`.

Note: When the value of `minimum_ios_deployment_target` is set to `'13'`, `tfcoreml` directly calls coremltools to convert the TensorFlow models, as can be seen [here](https://github.com/tf-coreml/tf-coreml/blob/674c30572867cd9d00dc930c0ee625f5b27de757/tfcoreml/_tf_coreml_converter.py#L672).
The conversion code for `minimum_ios_deployment_target` less than or equal to `'12'` is entirely present in the `tfcoreml` GitHub repo, whereas for `minimum_ios_deployment_target` is equal to `'13'` (or greater than in the future) the code is entirely present in the coremltools GitHub repo.

### Known Issues / Troubleshooting

- Although majority of Core ML 3 (iOS 13, macOs 15) layers have been updated in the converter, there might be a few missing layers, or cases not handled.
  Please [file a GitHub issue](https://github.com/apple/coremltools/issues/new/choose) if you encounter a bug while using the argument `minimum_ios_deployment_target='13'`.
- The `tf.keras` conversion is only supported when TensorFlow 2.x is used.
- TensorFlow 2.x model conversion is not supported with Python 2.
- Currently there are issues while exporting `tf.keras` graphs, that contain recurrent layers, to the `.h5` format

## ONNX Converter

PyTorch and MXNet models can be first exported to the ONNX format and then converted to Core ML via the
`onnx-coreml`(https://github.com/onnx/onnx-coreml) converter.

```python
from onnx_coreml import convert
ml_model = convert(model='my_model.onnx', target_ios='13')
ml_model.save('my_model.mlmodel')
```

See more examples [here](../examples/neural_network_inference/onnx_converter)  
Additional converter arguments are explained [here](https://github.com/onnx/onnx-coreml#parameters)

### Converting PyTorch model 

Converting PyTorch model to CoreML model is a two step process:
1. Convert PyTorch model to ONNX model
  - PyTorch model can be converted into ONNX model using `torch.onnx.export`
  - Reference: https://pytorch.org/docs/stable/onnx.html#id2
  - Tools required: [PyTorch](https://pytorch.org/get-started/locally/)
2. Convert ONNX model to CoreML 
  - Take the `.onnx` model and pass it to the function `onnx_coreml.convert()`
  - Tools required: [onnx-coreml](https://pypi.org/project/onnx-coreml/)
  


**PyTorch to ONNX conversion:**

  - Create a pyTorch model
    ```
      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      # Step 0 - (a) Define ML Model
      class small_model(nn.Module):
          def __init__(self):
              super(small_model, self).__init__()
              self.fc1 = nn.Linear(768, 256)
              self.fc2 = nn.Linear(256, 10)

          def forward(self, x):
              y = F.relu(self.fc1(x))
              y = F.softmax(self.fc2(y))
              return y
    ```
  - Load model
    ```
      # Step 0 - (b) Create model or Load from dist
      model = small_model()
      dummy_input = torch.randn(768)
    ```
  - Convert From PyTorch to ONNX
    ```
      # Step 1 - PyTorch to ONNX model
      torch.onnx.export(model, dummy_input, './small_model.onnx')
    ```
**ONNX to CoreML:**   
```python
      # Step 2 - ONNX to CoreML model
      from onnx_coreml import convert
      mlmodel = convert(model='./small_model.onnx', target_ios='13')
      # Save converted CoreML model
      mlmodel.save('small_model.mlmodel')
```

#### What about frameworks other than PyTorch?
Step 1 can be replaced by respective framework to ONNX converter.


#### Known Issues / Troubleshooting

- If the onnx opset version is greater than 9 and there are issues during conversion, please try exporting to onnx
  opset version 9 and then converting to Core ML
- onnx models with weight quantization and control flow layers (loop, branch) will give a conversion error since
  support for those has not been added yet to the converter

## Building an mlmodel Using the Builder API

[Code snippet](https://github.com/apple/coremltools/blob/master/docs/APIExamples.md#building-an-mlmodel-from-scratch-using-neural-network-builder)

## Model Quantization

`coremltools` provides utilities for performing post-training quantization for the weight parameters,
to reduce the size of the `.mlmodel` file. By default the converters produce mlmodel that have weights
in FP32 precision. These can be quantized to either FP16 or to 8 bits, 7 bits, up to all the way to 1 bit.
The lower the number of bits, more the chances of degrading the model accuracy. The loss in accuracy varies with
the model.

[Here](https://github.com/apple/coremltools/blob/master/docs/APIExamples.md#quantizing-a-neural-network-mlmodel)
is a code snippet on using the quantization utilities.

## Model Predictions

Neural network models can take as inputs two datatypes: either multi-arrays or image types.
When using coremltools to call predict on a model, in the case of multi-arrays, a numpy array must be fed.
In the case of an image, a PIL image python object should be used.

Multi-array prediction:

```python
import coremltools
import numpy as np

model = coremltools.models.MLModel('path/to/the/saved/model.mlmodel')

# print input description to get input shape
print(model.description.input)

input_shape = (...) # insert correct shape of the input

# call predict
output_dict = model.predict({'input_name': np.random.rand(*input_shape)}, useCPUOnly=True)
```

Image prediction:

```python
import coremltools
import numpy as np
import PIL.Image

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

## Model Inspection and Editing

[Code snippet](https://github.com/apple/coremltools/blob/master/docs/APIExamples.md#converting-between-mlmodel-and-spec)
for loading mlmodel and converting it to spec and vice versa.

[Netron](https://github.com/lutzroeder/netron) is a nice tool to visualize Core ML neural network models.

### Printing Description

To print a text description of the model:

```python
import coremltools

nn_mlmodel = coremltools.models.MLModel('path/to/the/model.mlmodel')

# To print a succinct description of the neural network
spec = nn_mlmodel.get_spec()
from coremltools.models.neural_network.printer import print_network_spec

print_network_spec(spec, style='coding')
# or
print_network_spec(spec)
```

To print information about the pre-processing parameters of the model (only applicable if the input is of type image)

```python
import coremltools

def _get_nn_sepc(spec):
    if spec.WhichOneof('Type') == 'neuralNetworkClassifier':
        nn_spec = spec.neuralNetworkClassifier
    if spec.WhichOneof('Type') == 'neuralNetwork':
        nn_spec = spec.neuralNetwork
    elif spec.WhichOneof('Type') == 'neuralNetworkRegressor':
        nn_spec = spec.neuralNetworkRegressor
    else:
        raise ValueError('Spec does not have a neural network')

spec = coremltools.models.utils.load_spec('path/to/the/saved/model.mlmodel')

# Get neural network portion of the spec
nn_spec = _get_nn_sepc()

# print pre-processing parameter
print(nn_spec.preprocessing)
```

### Flexible input/output shapes

There are several [utilities](https://apple.github.io/coremltools/generated/coremltools.models.neural_network.flexible_shape_utils.html#module-coremltools.models.neural_network.flexible_shape_utils)
 to mark inputs with `flexible` shapes.

### Modifying Input/Output Names and Types

```python
import coremltools

model = coremltools.models.MLModel('path/to/the/saved/model.mlmodel')
spec = model.get_spec()

# lets say the name of the input feature is "input" that we want to rename to "input_tensor"

coremltools.utils.rename_feature(spec, current_name='input', new_name='input_tensor')
model = coremltools.models.MLModel(spec)
model.save('path/to/the/saved/model.mlmodel')
```

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

### Inspecting Model for Debugging

Sometimes we want to print out weights of a particular layer for debugging purposes.
Following is an example showing how we can utilize the `protobuf` APIs to access any
attributes including the weight parameters. This code snippet uses the model we created in
the [this](https://github.com/apple/coremltools/blob/master/docs/APIExamples.md#building-an-mlmodel-from-scratch-using-neural-network-builder)
example.

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

### Miscellaneous Examples

- [Control flow Core ML model via the builder library](../examples/neural_network_inference/Image_preprocessing_per_channel_scale.ipynb)
- [Per channel scale pre-processing](../examples/neural_network_inference/Neural_network_control_flow_power_iteration.ipynb)
- [Image type as output, for a style transfer network](https://github.com/tf-coreml/tf-coreml/blob/master/examples/style_transfer_example.ipynb)
- [Setting image pre-processing correctly](https://github.com/tf-coreml/tf-coreml/blob/master/examples/inception_v1_preprocessing_steps.ipynb)
