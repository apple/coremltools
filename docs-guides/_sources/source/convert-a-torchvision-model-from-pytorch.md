```{eval-rst}
.. index:: 
    single: PyTorch; convert torchvision model
    single: torchvision model
```

# Converting a torchvision Model from PyTorch

The following example shows how to convert into Core ML a [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) model trained using PyTorch. MobileNet is a type of convolutional neural network designed for mobile and embedded vision applications. 

The example is similar to the one provided in [Getting Started](introductory-quickstart), in which you convert the TensorFlow version of the model. 

In this example you do the following:

1. Load a pre-trained model from [torchvision](https://pytorch.org/vision/stable/index.html#torchvision), a package of datasets, model architectures, and common image transformations.
2. Trace the model to generate TorchScript using the `torch.jit.trace` command.
3. Download the class labels.
4. Preprocess the image input for torchvision models.
5. Convert the traced model to a Core ML neural network using the  [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method.
6. Load an image to use for testing.
7. Make a prediction with the converted model.
8. Make a prediction with the original torch-traced model for comparison.

Once you have converted the model, you can follow the steps in [Save and Load the Model](introductory-quickstart.md#save-and-load-the-model) and [Use the Model with Xcode](introductory-quickstart.md#use-the-model-with-xcode).

## Load the MobileNetV2 Model

The example uses a pre-trained version of the [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) model from [torchvision](https://pytorch.org/vision/stable/index.html). Follow these steps:

1. Load the pre-trained version of MobileNetV2:
	
	```python
	import torch
	import torchvision

	# Load a pre-trained version of MobileNetV2 model.
	torch_model = torchvision.models.mobilenet_v2(pretrained=True)
	```

2. Set the model to evaluation mode:
	
	```python
	# Set the model in evaluation mode.
	torch_model.eval()
	```

```{admonition} Set the Model to Evaluation Mode

To ensure that operations such as dropout are disabled, it's important to set the model to evaluation mode (not training mode) before tracing. This setting also results in a more optimized version of the model for conversion.
```

## Trace the Model

The process of tracing takes an example input and traces its flow through the model. To understand the reasons for tracing and how to trace a PyTorch model, see [Model Tracing](model-tracing).

You can trace the model by creating an example image input, as shown in the following code using random data. The rank and shape of the tensor depends on your model's use case. If your model expects a fixed-size input, use that size for the example image. In all cases, the rank of the tensor must be fixed.

```python
# Trace the model with random data.
example_input = torch.rand(1, 3, 224, 224) 
traced_model = torch.jit.trace(torch_model, example_input)
out = traced_model(example_input)
```

## Download the Class Labels

MobileNetV2 is pre-trained on the [ImageNet](https://en.wikipedia.org/wiki/ImageNet) benchmark dataset. Download the class labels from the labels text file, and remove the first class (which is the background):

```python
# Download class labels in ImageNetLabel.txt.
import urllib
label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
class_labels = urllib.request.urlopen(label_url).read().decode("utf-8").splitlines()
class_labels = class_labels[1:] # remove the first class which is background
assert len(class_labels) == 1000
```

```{eval-rst}
.. index:: 
    single: PyTorch; preprocess image input
    single: preprocessing for images
```


## Preprocess the Image Input for torchvision Models

Image-based models typically require the input image to be preprocessed before using it with the converted model. For the details of how to preprocess image input for torchvision models, see [Preprocessing for Torch](image-inputs.md#preprocessing-for-torch).

The Core ML Tools [`ImageType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#coremltools.converters.mil.input_types.ImageType) input type lets you specify the `scale` and `bias` parameters. The scale is applied to the image first, and then the bias is added. Import coremltools, and before converting, specify the `ImageType` input type as follows:

```python
# Set the image scale and bias for input image preprocessing.
import coremltools as ct
scale = 1/(0.226*255.0)
bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

image_input = ct.ImageType(name="input_1",
                           shape=example_input.shape,
                           scale=scale, bias=bias)
```

```{admonition} Images for Input and Output

By default, the Core ML Tools converter generates a Core ML model with inputs of type [`MLMultiArray`](https://developer.apple.com/documentation/coreml/mlmultiarray). By providing an additional inputs argument, as shown in the next section, you can use either [`TensorType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#tensortype) or [`ImageType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#coremltools.converters.mil.input_types.ImageType). This example uses `ImageType`. To learn how to work with images for input and output, see [Image Input and Output](image-inputs).
```

## Convert to a Neural Network

Convert the model to a Core ML neural network using the Core ML Tools   [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method. Specify the `inputs` parameter with the preprocessed `image_input` from the previous section:

```python
# Using image_input in the inputs parameter:
# Convert to Core ML using the Unified Conversion API.
model = ct.convert(
    traced_model,
    inputs=[image_input],
    classifier_config = ct.ClassifierConfig(class_labels),
    compute_units=ct.ComputeUnit.CPU_ONLY,
)
```

With the converted model in memory, you can save it using the `.mlmodel` extension. It may also be helpful to display a confirmation message:

```python
# Save the converted model.
model.save("mobilenet.mlmodel")
# Print a confirmation message.
print('model converted and saved')
```

You can now incorporate this model into an application in Xcode, as described in [Use the Model With Xcode](introductory-quickstart.md#use-the-model-with-xcode).

The above example also sets the `class_labels` for classifying the image, and the `compute_units` to restrict execution to the CPU. For more information about `compute_units`, see [Set the Compute Units](load-and-convert-model.md#set-the-compute-units).

## Load the Test Image

The next step is to load an image using [PIL](https://en.wikipedia.org/wiki/Python_Imaging_Library), to use as input for testing the original PyTorch model and the converted model. Resize the input image for consistency so that it is 224 x 224 pixels, and specify `ANTIALIAS` for the algorithm to use for resampling pixels from one size to another:

```python
# Load the test image and resize to 224, 224.
img_path = "daisy.jpg"
img = PIL.Image.open(img_path)
img = img.resize([224, 224], PIL.Image.ANTIALIAS)
```

Right-click the following image and save it as `daisy.jpg` in the same folder as your Python project.

```{figure} images/daisy.jpg
:alt: Daisy image
:align: center
:class: imgnoborder

Right-click this image and save it as `daisy.jpg` in the same folder as your Python project.
```

```{eval-rst}
.. index:: 
    single: protobuf spec
```

## Get the protobuf spec

To get the fields and types used in the model, get the protobuf spec with [get_spec()](https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.model.MLModel.get_spec), and select the `dictionaryType` output to use for displaying the results:

```python
# Get the protobuf spec of the model.
spec = model.get_spec()
for out in spec.description.output:
    if out.type.WhichOneof('Type') == "dictionaryType":
        coreml_dict_name = out.name
        break
```

```{eval-rst}
.. index:: 
    single: PyTorch; make prediction
```

## Make a Core ML Prediction

You can now make a prediction with the converted model, using the test image. To learn more about making predictions, see [Model Prediction](model-prediction). The code for `coreml_out_dict["classLabel"]` returns the top-level class label.

```python
# Make a prediction with the Core ML version of the model.
coreml_out_dict = model.predict({"input_1" : img})
print("coreml predictions: ")
print("top class label: ", coreml_out_dict["classLabel"])

coreml_prob_dict = coreml_out_dict[coreml_dict_name]

values_vector = np.array(list(coreml_prob_dict.values()))
keys_vector = list(coreml_prob_dict.keys())
top_3_indices_coreml = np.argsort(-values_vector)[:3]
for i in range(3):
    idx = top_3_indices_coreml[i]
    score_value = values_vector[idx]
    class_id = keys_vector[idx]
    print("class name: {}, raw score value: {}".format(class_id, score_value))
```

When you run this example, the output should be something like the following, using the image of a daisy as the input:

```text Output
coreml predictions: 
top class label:  daisy
class name: daisy, raw score value: 15.690682411193848
class name: vase, raw score value: 8.516773223876953
class name: ant, raw score value: 8.169312477111816
```


## Make a PyTorch Prediction and Compare

To test the accuracy of the converted model with respect to the traced (TorchScript) model, make a prediction with the test image using the original PyTorch model. 

### Convert the Image to a Tensor

Convert the image to a tensor for input into the PyTorch model:

1. Convert the PIL image to a numPy array, and add a dimension. The result is `(1, 224, 224, 3)`.
2. The PyTorch model expects as input a torch tensor of shape `(1, 3, 224, 224)`, so you need to reshape the numPy array from the previous step by transposing it.
3. The PyTorch model was trained assuming that the input is normalized to the pixel range of `[0,1]`. However, this example tests the model with a [PIL image](https://en.wikipedia.org/wiki/Python_Imaging_Library) as input, which is in the range of `[0,255]`. Therefore, divide the array by 255.
4. Convert the array to a tensor for input to the PyTorch model.

```python
# Make a prediction with the Torch version of the model:
# prepare the input numpy array.
img_np = np.asarray(img).astype(np.float32) # (224, 224, 3)
img_np = img_np[np.newaxis, :, :, :] # (1, 224, 224, 3)
img_np = np.transpose(img_np, [0, 3, 1, 2]) # (1, 3, 224, 224)
img_np = img_np / 255.0
torch_tensor_input = torch.from_numpy(img_np)
```

### Make a Prediction with Torch and Print Outputs

The torchvision [`transforms.Normalize`](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Normalize) class normalizes a tensor image with the mean and standard deviation. To [script the transformation](https://pytorch.org/vision/stable/transforms.html#scriptable-transforms), use  `torch.nn.Sequential`:

```python
# Preprocess model for Torch.
transform_model = torch.nn.Sequential(
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
)
```

Make a prediction using the traced PyTorch model and the normalized image, and print the output, including the top three indices and the score value for each one:

```python
# Invoke prediction and print outputs.
torch_out = traced_model(transform_model(torch_tensor_input))

torch_out_np = torch_out.detach().numpy().squeeze()
top_3_indices = np.argsort(-torch_out_np)[:3]
print('torch top 3 predictions: ')
for i in range(3):
    idx = top_3_indices[i]
    score_value = torch_out_np[idx]
    class_id = class_labels[idx]
    print("class name: {}, raw score value: {}".format(class_id, score_value))
```

When you run this example, the output should be something like the following, using the image of a daisy as the input:

```text
torch top 3 predictions: 
class name: daisy, raw score value: 15.65333366394043
class name: vase, raw score value: 8.527873992919922
class name: ant, raw score value: 8.256473541259766
```

As you can see from the results, the converted model performs very closely to the original model — the raw score values are very similar.

