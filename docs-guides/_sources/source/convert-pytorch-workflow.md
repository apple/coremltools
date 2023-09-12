```{eval-rst}
.. index:: 
    single: PyTorch; convert workflow
    single: Torchscript
```


# PyTorch Conversion Workflow

```{admonition} Minimum Deployment Target

The Core ML Tools [Unified Conversion API](unified-conversion-api) produces Core ML models for iOS 13, macOS 10.15, watchOS 6, tvOS 13 or newer deployment targets. If your primary deployment target is iOS 12 or earlier, you can find limited conversion support for PyTorch models via the [onnx-coreml](https://github.com/onnx/onnx) package.
```

## Generate a TorchScript Version

[TorchScript](https://pytorch.org/docs/stable/jit.html) is an intermediate representation of a PyTorch model. To generate a TorchScript representation from PyTorch code, use PyTorch's JIT tracer ([`torch.jit.trace`](https://pytorch.org/docs/stable/generated/torch.jit.trace.html)) to _trace_ the model, as shown in the following example:

```python
import torch
import torchvision

# Load a pre-trained version of MobileNetV2
torch_model = torchvision.models.mobilenet_v2(pretrained=True)
# Set the model in evaluation mode.
torch_model.eval()

# Trace the model with random data.
example_input = torch.rand(1, 3, 224, 224) 
traced_model = torch.jit.trace(torch_model, example_input)
out = traced_model(example_input)
```

The process of tracing takes an example input and traces its flow through the model. You can trace the model by creating an example image input, as shown in the above code using random data. To understand the reasons for tracing and how to trace a PyTorch model, see [Model Tracing](model-tracing).

```{admonition} Set the Model to Evaluation Mode

To ensure that operations such as dropout are disabled, it's important to set the model to evaluation mode (_not_ training mode) before tracing. This setting also results in a more optimized version of the model for conversion.
```

If your model uses a data-dependent control flow, such as a loop or conditional, the traced model won't generalize to other inputs. In such cases you can experiment with applying PyTorch's JIT script ([`torch.jit.script`](https://pytorch.org/docs/stable/generated/torch.jit.script.html)) to your model as described in [Model Scripting](model-scripting). You can also use a combination of tracing and scripting.

## Convert to Core ML

Convert the traced or scripted model to Core ML using the Unified Conversion API [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method. In the `inputs` parameter, you can use either [`TensorType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#tensortype) or [`ImageType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#coremltools.converters.mil.input_types.ImageType) as the input type.

The following example uses `TensorType` and converts the PyTorch traced model to a [Core ML program](convert-to-ml-program) model. For more about image input conversions, see [Image Inputs](image-inputs).

```python
import coremltools as ct

# Using image_input in the inputs parameter:
# Convert to Core ML program using the Unified Conversion API.
model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)]
 )
```

With the converted ML model in memory, you can save it as a [Core ML model package](convert-to-ml-program.md#save-ml-programs-as-model-packages):

```python
# Save the converted model.
model.save("newmodel.mlpackage")
```

As an alternative, you can convert the model to a neural network by eliminating the `convert_to` parameter:

```python
# Using image_input in the inputs parameter:
# Convert to Core ML neural network using the Unified Conversion API.
model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)]
 )
```

With the converted neural network in memory, you can save it as an `mlmodel` file:

```python
# Save the converted model.
model.save("newmodel.mlmodel")
```

```{admonition} For More Information

- To learn how TorchScript works, see the [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html).
- To learn how to get better performance and more convenience when using images, see [Image Input and Output](image-inputs).
- For the details of how to preprocess image input for models in PyTorch's  [torchvision](https://pytorch.org/vision/stable/index.html) library, see [Preprocessing for Torch](image-inputs.md#preprocessing-for-torch).
- For examples of converting PyTorch models, see the following:
    - [Converting a Natural Language Processing Model](convert-nlp-model)
    - [Converting a torchvision Model from PyTorch](convert-a-torchvision-model-from-pytorch)
    - [Converting a PyTorch Segmentation Model](pytorch-conversion-examples)
```

