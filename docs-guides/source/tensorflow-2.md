```{eval-rst}
.. index:: 
    single: TensorFlow 2; convert workflow
    single: TensorFlow 2; convert from
```


# TensorFlow 2 Workflow

To convert a [TensorFlow 2](https://www.tensorflow.org/api_docs) model, provide one of following formats to the converter:

- [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras "Module: tf.keras")
- [HDF5 file path](https://keras.io/api/models/model_saving_apis/ "Model saving & serialization APIs") (`.h5`)
- [`SavedModel`](https://www.tensorflow.org/guide/saved_model "Using the SavedModel format") directory path
- A [concrete function] \(<https://www.tensorflow.org/guide/function> "Better performance with tf.function")

```{admonition} Recommended Format

The most convenient way to convert from TensorFlow 2 is to use an object of the  [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras "Module: tf.keras") class. If you download a pre-trained model (`SavedModel` or `HDF5`), first check that you can load it as a `tf.keras.Model` and run the `predict()` method on it. Then pass the model into the Core ML Tools converter.
```

This page demonstrates the following typical workflows:

- [Convert a pre-trained model](#convert-a-pre-trained-model): Downloading a pre-trained model in the `SavedModel` or `.h5` file format, loading it as a `tf.keras.Model`, and then converting the model.
- [Convert a user-defined model](#convert-a-user-defined-model): Defining a model from scratch, training it, and then converting the model.


```{eval-rst}
.. index:: 
    single: TensorFlow 2; convert pre-trained model
```

## Convert a Pre-trained Model

The following example demonstrates how to convert an [Xception model](https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception "Module: tf.keras.applications.xception") in `HDF5` format (a `.h5` file) from [`tf.keras.applications`](https://www.tensorflow.org/api_docs/python/tf/keras/applications "Module: tf.keras.applications"):

```python
import coremltools as ct 
import tensorflow as tf

# Load from .h5 file
tf_model = tf.keras.applications.Xception(weights="imagenet", 
                                          input_shape=(299, 299, 3))

# Convert to Core ML
model = ct.convert(tf_model)
```

The following is another example of converting a pre-trained model. This model is downloaded from [TensorFlow Hub](https://tfhub.dev). Follow these steps:

1. Download the MobileNet `SavedModel` directory from [imagenet in TensorFlow Hub](https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/classification/4 "imagenet/mobilenet_v2_050_192/classification").

```python
# Tested with TensorFlow 2.6.2
import tensorflow as tf
import tensorflow_hub as tf_hub
import numpy as np

model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(192, 192, 3)),
        tf_hub.KerasLayer(
          "https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/classification/4"
        )
])

model.build([1, 192, 192, 3])  # Batch input shape.
```

2. Load the model as a Keras model, and ensure that it is loaded correctly by applying a prediction call. 

```python
# random input data to check that predict works
x = np.random.rand(1, 192, 192, 3)
tf_out = model.predict([x])
```

3. Convert the model to an [ML program](convert-to-ml-program) without specifying the input type, in order to generate a multidimensional array ([`MLMultiArray`](https://developer.apple.com/documentation/coreml/mlmultiarray)) input for convenience in checking predictions: 

```python
import coremltools as ct

# convert to Core ML and check predictions
mlmodel = ct.convert(model, convert_to="mlprogram")
```

4. Since the model operates on images, convert with the image input type before saving the model as a [Core ML model package](convert-to-ml-program.md#save-ml-programs-as-model-packages):

```python
coreml_out_dict = mlmodel.predict({"image":x})
coreml_out = list(coreml_out_dict.values())[0]
np.testing.assert_allclose(tf_out, coreml_out, rtol=1e-2, atol=1e-1)

# convert to an image input Core ML model
# mobilenet model expects images to be normalized in the interval [-1,1]
# hence bias of -1 and scale of 1/127
mlmodel = ct.convert(model, convert_to="mlprogram",
                    inputs=[ct.ImageType(bias=[-1,-1,-1], scale=1/127)])

mlmodel.save("mobilenet.mlpackage")
```

```{eval-rst}
.. index:: 
    single: TensorFlow 2; convert user-defined model
```

## Convert a User-defined Model

The most convenient way to define a model is to use the  `tf.keras` APIs. You can define your model using [sequential](https://www.tensorflow.org/guide/keras/sequential_model "The Sequential model"), [functional](https://www.tensorflow.org/guide/keras/functional "The Functional API") or [subclassing](https://www.tensorflow.org/guide/keras/custom_layers_and_models "Making new Layers and Models via subclassing") APIs, and then convert directly to Core ML. 

Alternatively, you can first save the Keras model to the `HDF5` (`.h5`) or the `SavedModel` file format, and then provide the file path with the `convert()` method. For details about saving the model, see [Save and load Keras models](https://www.tensorflow.org/guide/keras/save_and_serialize "Save and load Keras models").

```{eval-rst}
.. index:: 
    single: TensorFlow 2; convert sequential model
```

### Convert a Sequential Model

The following example defines and converts a Sequential `tf.keras` model to an [ML program](convert-to-ml-program):

```python convert_tf_keras_model
# Tested with TensorFlow 2.6.2
import tensorflow as tf
import coremltools as ct

tf_keras_model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

# Pass in `tf.keras.Model` to the Unified Conversion API
mlmodel = ct.convert(tf_keras_model, convert_to="mlprogram")

# or save the keras model in SavedModel directory format and then convert
tf_keras_model.save('tf_keras_model')
mlmodel = ct.convert('tf_keras_model', convert_to="mlprogram")

# or load the model from a SavedModel and then convert
tf_keras_model = tf.keras.models.load_model('tf_keras_model')
mlmodel = ct.convert(tf_keras_model, convert_to="mlprogram")

# or save the keras model in HDF5 format and then convert
tf_keras_model.save('tf_keras_model.h5')
mlmodel = ct.convert('tf_keras_model.h5', convert_to="mlprogram")
```

```{eval-rst}
.. index:: 
    single: TensorFlow 2; convert a Keras model
```

### Convert a Keras Model With Subclassing

The following example defines and converts a Keras model with subclassing and a custom Keras layer, using a low-level TensorFlow API. The [custom layer example](https://www.tensorflow.org/guide/keras/functional#extend_the_api_using_custom_layers) of the functional Keras API can be converted to an ML program or a neural network by passing the final model object to the converter. The following example converts the model to an ML program:

```python
# Tested with TensorFlow 2.6.2
import coremltools as ct
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)
model = keras.Model(inputs, outputs)

mlmodel = ct.convert(model, convert_to="mlprogram")
```

```{eval-rst}
.. index:: 
    single: TensorFlow 2; convert concrete function
```

### Convert a TensorFlow Concrete Function

The following example converts a TensorFlow concrete function to an ML program:

```python
# Tested with TensorFlow 2.6.2
import coremltools as ct
import tensorflow as tf
import numpy as np

# define a concrete TF function for approximate version of GeLU activation
@tf.function(input_signature=[tf.TensorSpec(shape=(6,), dtype=tf.float32)])
def gelu_tanh_activation(x):
	a = (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))
	y = 0.5 * (1.0 + tf.tanh(a))
	return x * y

conc_func = gelu_tanh_activation.get_concrete_function()

# provide the concrete fucntion as a list
mlmodel = ct.convert([conc_func], convert_to="mlprogram")
```

```{admonition} Converting a BERT Transformer Model

To learn how to convert an object of the [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras "Module: tf.keras") class, and a [`SavedModel`](https://www.tensorflow.org/hub/tf2_saved_model "SavedModels from TF Hub in TensorFlow 2") in the TensorFlow 2 format, see [Converting TensorFlow 2 BERT Transformer Models](convert-tensorflow-2-bert-transformer-models).
```
