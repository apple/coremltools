
```{eval-rst}
.. index:: 
   single: prediction; input and output
   single: prediction; specify compute units
   single: prediction; image
```

# Model Prediction

After converting a source model to a Core ML model, you can evaluate the Core ML model by verifying that the predictions made by the Core ML model match the predictions made by the source model. 

The following example makes predictions for the `HousePricer.mlmodel` using the [`predict()`](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.model.MLModel.predict) method. 

```python
import coremltools as ct

# Load the model
model = ct.models.MLModel('HousePricer.mlmodel')

# Make predictions
predictions = model.predict({'bedroom': 1.0, 'bath': 1.0, 'size': 1240})
```

```{admonition} macOS Required for Model Prediction

For the prediction API, coremltools interacts with the Core ML framework which is available on macOS only. The prediction API is not available on Linux. 

However, Core ML models can be [imported and executed with TVM](https://tvm.apache.org/docs/tutorials/frontend/from_coreml.html), which may provide a way to test Core ML models on non-macOS systems.
```

## Types of Inputs and Outputs

Core ML supports several [feature types](https://apple.github.io/coremltools/mlmodel/Format/FeatureTypes.html) for inputs and outputs. The following are two feature types that are commonly used with neural network models: 

- `ArrayFeatureType`, which maps to the MLMultiArray Feature Value in Swift 
- `ImageFeatureType`, which maps to the Image Feature Value in Swift

When using the Core ML model in your Xcode app, use an [MLFeatureValue](https://developer.apple.com/documentation/coreml/mlfeaturevalue), which wraps an underlying value and bundles it with that value’s type, represented by [MLFeatureType](https://developer.apple.com/documentation/coreml/mlfeaturetype).

To evaluate a Core ML model in python using the [`predict()`](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.model.MLModel.predict) method, use one of the following inputs:

- For a multi-array, use a [NumPy](https://numpy.org) array.
- For an image, use a [PIL](https://en.wikipedia.org/wiki/Python_Imaging_Library) image python object.

```{admonition} Learn More About Image Input and Output

To learn how to work with images and achieve better performance and more convenience, see [Image Input and Output](image-inputs).
```

## Specifying Compute Units

If you don't specify compute units when converting or loading a model, all compute units available on the device are used for execution including the Apple Neural Engine (ANE), the CPU, and the graphics processing unit (GPU). 

You can control which compute unit the model runs on by setting the `compute_units` argument when converting a model (with [`coremltools.convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#coremltools.converters._converters_entry.convert)) or loading a model (with [`coremltools.models.MLModel`](https://apple.github.io/coremltools/source/coremltools.models.html#module-coremltools.models.model)). Calling [`predict()`](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.model.MLModel.predict) on the converted or loaded model restricts the model to use only the specific compute units for execution. 

For example, the following sets the compute units to CPU only when loading the model:

```python
model = ct.model.MLModel('path/to/the/saved/model.mlmodel', compute_units=ct.ComputeUnit.CPU_ONLY)
```

```{admonition} Deprecated Flag

In previous versions of coremltools, you would restrict execution to the CPU by specifying the `useCPUOnly=True` flag. This flag is now deprecated. Instead, use the `compute_units` parameter .
```

For more information and values for this parameter, see [Set the Compute Units](load-and-convert-model.md#set-the-compute-units).

## Multi-array Prediction

A model that takes a `MultiArray` input requires a NumPy array as an input with the `predict()` call. For example:

```python
import coremltools as ct
import numpy as np

model = ct.models.MLModel('path/to/the/saved/model.mlmodel')

# Print input description to get input shape.
print(model.get_spec().description.input)

input_shape = (...) # insert correct shape of the input

# Call predict.
output_dict = model.predict({'input_name': np.random.rand(*input_shape)})
```

## Image Prediction

A model that takes an image input requires a PIL image as an input with the `predict()` call. For example:

```python
import coremltools as ct
import numpy as np
import PIL.Image

# Load a model whose input type is "Image".
model = ct.models.MLModel('path/to/the/saved/model.mlmodel')

Height = 20  # use the correct input image height
Width = 60  # use the correct input image width


# Scenario 1: load an image from disk.
def load_image(path, resize_to=None):
    # resize_to: (Width, Height)
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32)
    return img_np, img


# Load the image and resize using PIL utilities.
_, img = load_image('/path/to/image.jpg', resize_to=(Width, Height))
out_dict = model.predict({'image': img})

# Scenario 2: load an image from a NumPy array.
shape = (Height, Width, 3)  # height x width x RGB
data = np.zeros(shape, dtype=np.uint8)
# manipulate NumPy data
pil_img = PIL.Image.fromarray(data)
out_dict = model.predict({'image': pil_img})
```

## Image Prediction for a Multi-array Model

If the Core ML model has a `MultiArray` input type that actually represents a JPEG image, you can still use the JPEG image for the prediction if you first convert the loaded image to a NumPy array, as shown in this example:

```python
Height = 20  # use the correct input image height
Width = 60  # use the correct input image width

# Assumption: the mlmodel's input is of type MultiArray and of shape (1, 3, Height, Width).
model_expected_input_shape = (1, 3, Height, Width) # depending on the model description, this could be (3, Height, Width)

# Load the model.
model = coremltools.models.MLModel('path/to/the/saved/model.mlmodel')

def load_image_as_numpy_array(path, resize_to=None):
    # resize_to: (Width, Height)
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32) # shape of this numpy array is (Height, Width, 3)
    return img_np

# Load the image and resize using PIL utilities.
img_as_np_array = load_image_as_numpy_array('/path/to/image.jpg', resize_to=(Width, Height)) # shape (Height, Width, 3)

# PIL returns an image in the format in which the channel dimension is in the end,
# which is different than Core ML's input format, so that needs to be modified.
img_as_np_array = np.transpose(img_as_np_array, (2,0,1)) # shape (3, Height, Width)

# Add the batch dimension if the model description has it.
img_as_np_array = np.reshape(img_as_np_array, model_expected_input_shape)

# Now call predict.
out_dict = model.predict({'image': img_as_np_array})
```

