```{eval-rst}
.. index:: 
    single: PyTorch; convert workflow
    single: Torchscript
```


# PyTorch Conversion Workflow

```{admonition} Minimum Deployment Target

The Core ML Tools [Unified Conversion API](unified-conversion-api) produces Core ML models for iOS 13, macOS 10.15, watchOS 6, tvOS 13 or newer deployment targets. If your primary deployment target is iOS 12 or earlier, you can find limited conversion support for PyTorch models via the [onnx-coreml](https://github.com/onnx/onnx) package.
```

To export a model from PyTorch to Core ML, there are 2 steps:
1. Capture the PyTorch model graph from the original torch.nn.Module, via [`torch.jit.trace`](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) or [`torch.export.export`](https://pytorch.org/docs/stable/export.html#torch.export.export).
2. Convert the PyTorch model graph to Core ML, via the Core ML Tools [Unified Conversion API](unified-conversion-api).

The conversion from a graph captured via `torch.jit.trace` has been supported for many versions of Core ML Tools, hence it is the stable and the more performant path. For now, this is the recommended way for converting PyTorch models to Core ML.

The conversion from `torch.export` graph has been newly added to Core ML Tools 8.0.
It is currently in beta state, in line with the export API status in PyTorch.
As of Core ML Tools 8.0, representative models such as MobileBert, ResNet, ViT, [MobileNet](convert-a-torchvision-model-from-pytorch), [DeepLab](convert-a-pytorch-segmentation-model), [OpenELM](convert-openelm) can be converted, and the total PyTorch op translation test coverage is roughly ~70%. You can start trying the torch.export path on your models that are working with torch.jit.trace already, so as to gradually move them to the export path as PyTorch also [moves](https://github.com/pytorch/pytorch/issues/103841#issuecomment-1605017153) its support and development to that path over a period of time. In case you hit issues (e.g. models converted via export path are slower than the ones converted from jit.trace path), please report them on Github.

Now let us take a closer look at how to convert from PyTorch to Core ML through an example.

## Obtain the Original PyTorch Model

As an illustration, here we define a toy PyTorch model

```python
import torch
import torchvision

# Load a pre-trained version of MobileNetV2
torch_model = torchvision.models.mobilenet_v2(pretrained=True)
# Set the model in evaluation mode.
torch_model.eval()
```

```{admonition} Set the Model to Evaluation Mode

To ensure that training time operations such as dropout, batch norm with moving average etc. are disabled, it is important to set the model to evaluation mode before tracing / exporting. This setting also results in a more optimized version of the model for deployment.
```

## Capture the PyTorch Graph

### TorchScript

[TorchScript](https://pytorch.org/docs/stable/jit.html) is an intermediate representation of a PyTorch model. To generate a TorchScript representation from PyTorch code, use PyTorch's JIT tracer ([`torch.jit.trace`](https://pytorch.org/docs/stable/generated/torch.jit.trace.html)) to _trace_ the model, as shown in the following example:

```python
# Trace the model with random data.
example_input = torch.rand(1, 3, 224, 224) 
traced_model = torch.jit.trace(torch_model, example_input)
```

The process of tracing takes an example input and traces its flow through the model. You can trace the model by creating an example image input, as shown in the above code using random data. To understand the reasons for tracing and how to trace a PyTorch model, see [Model Tracing](model-tracing).

If your model uses a data-dependent control flow, such as a loop or conditional, the traced model won't generalize to other inputs. In such cases you can use JIT script ([`torch.jit.script`](https://pytorch.org/docs/stable/generated/torch.jit.script.html)) to capture the graph accurately. Core ML Tools offers limited support for models that are obtained from `torch.jit.script` API, so it may or may not work for your model. See [Model Scripting](model-scripting) to learn more.

### ExportedProgram

[ExportedProgram](https://pytorch.org/docs/stable/export.html) is a new intermediate representation of a PyTorch model introduced in PyTorch 2. To generate an ExportedProgram representation from PyTorch code, use PyTorch's exporter ([`torch.export.export`](https://pytorch.org/docs/stable/export.html#torch.export.export)) to _export_ the model, as shown in the following example:

```python
# Export the model with random data.
example_inputs = (torch.rand(1, 3, 224, 224),)
exported_program = torch.export.export(torch_model, example_inputs)
```

The process of exporting takes an example input and traces its flow through the model. You can trace the model by creating an example image input, as shown in the above code using random data.

## Convert to Core ML

Convert the traced / exported model to Core ML using the Unified Conversion API [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method. In the `inputs` parameter, you can use either [`TensorType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#tensortype) or [`ImageType`](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#coremltools.converters.mil.input_types.ImageType) as the input type for the traced model.

The following example uses `TensorType` and converts the PyTorch traced / exported model to a [Core ML program](convert-to-ml-program) model. For more about image input conversions, see [Image Inputs](image-inputs).

```python
import coremltools as ct

# Convert to Core ML program using the Unified Conversion API.
model_from_trace = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
)
model_from_export = ct.convert(exported_program)
```

With the converted ML model in memory, you can save it as a [Core ML model package](convert-to-ml-program.md#save-ml-programs-as-model-packages):

```python
# Save the converted model.
model_from_trace.save("newmodel_from_trace.mlpackage")
model_from_export.save("newmodel_from_export.mlpackage")
```

```{admonition} For More Information

- To learn how TorchScript works, see the [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html).
- To learn how ExportedProgram works, see the [torch.export tutorial](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html)
- To learn how to get better performance and more convenience when using images, see [Image Input and Output](image-inputs).
- For the details of how to preprocess image input for models in PyTorch's  [torchvision](https://pytorch.org/vision/stable/index.html) library, see [Preprocessing for Torch](image-inputs.md#preprocessing-for-torch).
- For examples of converting PyTorch models, see the following:
    - [Converting a torchvision Model from PyTorch](convert-a-torchvision-model-from-pytorch)
    - [Converting a PyTorch Segmentation Model](convert-a-pytorch-segmentation-model)
    - [Converting an Open Efficient Language Model](convert-openelm)
```
