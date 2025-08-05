```{eval-rst}
.. index:: 
    single: TensorFlow 1; convert workflow
    single: TensorFlow 1; convert from
```


# TensorFlow 1 Workflow

Use the frozen graph format for conversion from TensorFlow 1. After training, always export the model for inference to this format using the `tensorflow.python.tools.freeze_graph` method. TensorFlow 1 pre-trained models are also generally available in the frozen `.pb` file format.

## Export as a Frozen Graph and Convert

The following example demonstrates how to export a model to the frozen graph format and convert it to a [Core ML program](convert-to-ml-program) model. Follow these steps:

1. Define a simple model with random weights:

```python
import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
  x = tf.placeholder(tf.float32, shape=[None, 20], name="input")
  W = tf.Variable(tf.truncated_normal([20, 10], stddev=0.1))
  b = tf.Variable(tf.ones([10]))
  y = tf.matmul(x, W) + b
  output_names = [y.op.name]
```

2. Export the graph as a frozen graph:

```python
import tempfile
import os 
from tensorflow.python.tools.freeze_graph import freeze_graph

model_dir = tempfile.mkdtemp()
graph_def_file = os.path.join(model_dir, 'tf_graph.pb')
checkpoint_file = os.path.join(model_dir, 'tf_model.ckpt')
frozen_graph_file = os.path.join(model_dir, 'tf_frozen.pb')

with tf.Session(graph=graph) as sess:
  # initialize variables
  sess.run(tf.global_variables_initializer())
  # save graph definition somewhere
  tf.train.write_graph(sess.graph, model_dir, graph_def_file, as_text=False)
  # save the weights
  saver = tf.train.Saver()
  saver.save(sess, checkpoint_file)

  # take the graph definition and weights 
  # and freeze into a single .pb frozen graph file
  freeze_graph(input_graph=graph_def_file,
               input_saver="",
               input_binary=True,
               input_checkpoint=checkpoint_file,
               output_node_names=",".join(output_names),
               restore_op_name="save/restore_all",
               filename_tensor_name="save/Const:0",
               output_graph=frozen_graph_file,
               clear_devices=True,
               initializer_nodes="")
  
print("TensorFlow frozen graph saved at {}".format(frozen_graph_file))
```

3. Convert the model to an ML program and save it as a [Core ML model package](convert-to-ml-program.md#save-ml-programs-as-model-packages):

```python
import coremltools as ct

mlmodel = ct.convert(frozen_graph_file, convert_to="mlprogram")
mlmodel.save(frozen_graph_file.replace("pb","mlpackage")))
```

```{eval-rst}
.. index:: 
    single: TensorFlow 1; convert pre-trained model
```

## Convert a Pre-trained Model

The following example demonstrates how to use a downloaded pre-trained model, load it as a `tf.Graph` object, and then convert and compare predictions with Core ML. This example also compares predictions after conversion to verify the numerical accuracy. 

1. Download the [`float_v2_1.0_224` MobileNet V2 model](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) from [TensorFlow models](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet).

2. Load the model in TensorFlow as a `tf.graph` object.

```python
import tensorflow as tf

path = "~/Downloads/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224_frozen.pb"

# Load the protobuf file from the disk and parse it to retrieve the
# graph_def
with tf.io.gfile.GFile(path, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Import the graph_def into a new Graph
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")
```

3. Run the TensorFlow graph to get a prediction on a random input. 

```{attention}
We recommend running the original TensorFlow model once to verify the model, before invoking the Core ML converter.
```

To run this TensorFlow graph, you need to know its input and output names. Since there is no simple TensorFlow API that can directly provide this information, inspect the operations in the graph:

```python
ops = graph.get_operations()
N = len(ops)
# print all the placeholder ops, these would be the inputs
print("Inputs ops: ")
for op in ops:
    if op.type == "Placeholder":
        print("op name: {}, output shape : {}".
              format(op.name, op.outputs[0].get_shape()))


# print all the tensors that are the first output of an op
# and do not feed into any other op
# these are prospective outputs
print("\nProspective output tensor(s): ", )
sink_ops = []
input_tensors = set()
for op in ops:
    for x in op.inputs:
        if x.name not in input_tensors:
            input_tensors.add(x.name)
for op in ops:
    if len(op.outputs) > 0:
        x = op.outputs[0]
        if x.name not in input_tensors:
            print("tensor name: {}, tensor shape : {}, parent op type: {}"
                  .format(x.name, x.get_shape(), op.type))
```

The following **Out** snippet shows what results are printed. Use this information to run the TensorFlow graph.

```text Out
Inputs ops: 
op name: input, output shape : <unknown>

Prospective output tensor(s): 
tensor name: MobilenetV2/Predictions/Reshape_1:0, tensor shape : (?, 1001), parent op type: Reshape
```

In the above example, the input shape is missing from the TensorFlow graph. This means it can take a variety of shapes. For this example, use `(1, 224, 224, 3)`, since this is typically used with MobileNet models. 

```python
import numpy as np

x = np.random.rand(1, 224, 224, 3)

with tf.Session(graph = graph) as sess:
    tf_out = sess.run('MobilenetV2/Predictions/Reshape_1:0',
                      feed_dict={'input:0': x})
```

4. Convert the model to a Core ML neural network, and compare predictions. Since the TensorFlow graph lacks shape information, provide it to the converter using the `inputs` argument.

The following example provides a fixed input shape, but you can use a flexible input shape to enable the generated Core ML model to work using different input shapes. For more information, see [Flexible Input Shapes](flexible-inputs). 

```python
import coremltools as ct
mlmodel = ct.convert(graph,
                     inputs=[ct.TensorType(shape=x.shape)])

# Core ML model prediction
coreml_out_dict = mlmodel.predict({"input" : x}, useCPUOnly=True)
coreml_out = list(coreml_out_dict.values())[0]
np.testing.assert_allclose(tf_out, coreml_out, rtol=1e-3, atol=1e-2)
```

```{admonition} Use Image Inputs
The coremltools converter generates by default a model with a  [`MLMultiArray`](https://developer.apple.com/documentation/coreml/mlmultiarray) as the input, which works for checking conversion and numerical accuracy. For the final export, the best practice is to use the image type. For details, see [Image Inputs](image-inputs).
```

## More Examples

The following examples demonstrate some of the capabilities of the Core ML Tools converter for converting [TensorFlow 1](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf) models to [Core ML](https://developer.apple.com/documentation/coreml):

- [Converting a TensorFlow 1 Image Classifier](convert-a-tensorflow-1-image-classifier): Demonstrates the importance of setting the image preprocessing parameters correctly during conversion to get the right results.
- [Converting a TensorFlow 1 DeepSpeech Model](convert-a-tensorflow-1-deepspeech-model): Demonstrates automatic handling of flexible shapes using automatic speech recognition.
