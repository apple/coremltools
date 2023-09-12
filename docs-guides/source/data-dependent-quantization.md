```{eval-rst}
.. index:: 
    single: quantization; training-time
    single: PyTorch; quantization APIs
    single: quantization; PyTorch APIs
    single: LinearQuantizer
```

# Training-Time Quantization

The [`LinearQuantizer`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer) class implements training-time quantization, also known as quantization-aware training (QAT) as described in the paper [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/pdf/1712.05877.pdf). [`LinearQuantizer`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer) quantizes both weights and activations, whereas [Post-Training Quantization](data-free-quantization) quantizes only the weights.

```{admonition} PyTorch quantization APIs
You can use PyTorch's quantization APIs directly, and then convert the model to Core ML. However, the converted model performance may not be optimal. The PyTorch API default settings (symmetric asymmetric quantization modes and which ops are quantized) are not optimal for the Core ML stack and Apple hardware. If you use the Core ML Tools `coremltools.optimize.torch` APIs, as described in this section, the correct default settings are applied automatically.
```

## Use `LinearQuantizer`

Follow these key steps: 

- Define the [`LinearQuantizerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizerConfig) config to specify the quantization parameters.
- Initialize the [`LinearQuantizer`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer) object.
- Call the [`prepare`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer.prepare) API to insert fake quantization layers in the PyTorch model.
- Run the usual training loop, with the addition of the [`quantizer.step`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer.step) call.
- Once the model has converged, use the [`finalize`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer.finalize) API to prepare the model for conversion to Core ML.

The following code sample shows how you can use [`LinearQuantizer`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer) to perform training-time quantization on your PyTorch model.  

```python
from collections import OrderedDict

import torch
import torch.nn as nn

import coremltools as ct
from coremltools.optimize.torch.quantization import LinearQuantizer, LinearQuantizerConfig

model = nn.Sequential(
    OrderedDict(
        {
            "conv": nn.Conv2d(1, 20, (3, 3)),
            "relu1": nn.ReLU(),
            "conv2": nn.Conv2d(20, 20, (3, 3)),
            "relu2": nn.ReLU(),
        }
    )
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
data = get_data()

# Initialize the quantizer
config = LinearQuantizerConfig.from_dict(
    {
        "global_config": {
            "quantization_scheme": "symmetric",
            "milestones": [0, 100, 400, 200],
        }
    }
)
quantizer = LinearQuantizer(model, config)

# Prepare the model to insert FakeQuantize layers for QAT
example_input = torch.rand(1, 1, 20, 20)
model = quantizer.prepare(example_inputs=example_input, inplace=True)

# Use quantizer in your PyTorch training loop
for inputs, labels in data:
    output = model(inputs)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
    quantizer.step()

# Convert operations to their quanitzed counterparts using parameters learnt via QAT
model = quantizer.finalize(inplace=True)

# Convert the PyTorch models to CoreML format
traced_model = torch.jit.trace(model, example_input)
coreml_model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)],
    minimum_deployment_target=ct.target.iOS17,
)
coreml_model.save("~/quantized_model.mlpackage")
```

The two key parameters in [`ModuleLinearQuantizerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.ModuleLinearQuantizerConfig) that need to be set for training-time quantization are `quantization_scheme` and `milestones`. The allowed values for `quantization_scheme` are `symmetric` and `affine`. In `symmetric` mode, `zero_point` is always set to zero, whereas `affine` mode is able to use any zero point in the `quint8` or `int8` range, depending on the dtype used. 

The `milestones` parameter controls the flow of the quantization algorithm, and calling the [`step`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer.step) API on the quantizer object steps through these milestones. The `milestones` parameter is an array of size 4, and each element is an integer indicating the training step at which the stage corresponding to that element comes into effect. A detailed explanation of these various stages can be found in the API Reference for [`ModuleLinearQuantizerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.ModuleLinearQuantizerConfig).

## How It Works

The [`LinearQuantizer`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer) class simulates the effects of quantization during training by quantizing and de-quantizing the weights and activations during the model’s forward pass. The forward and backward pass computations are conducted in `float32` dtype. However, these `float32` values follow the constraints imposed by `int8` and `quint8` dtypes, for weights and activations respectively. This allows the model weights to adjust and reduce the error introduced by quantization.  [Straight-Through Estimation](https://arxiv.org/pdf/1308.3432.pdf) is used for computing gradients of non-differentiable operations introduced by simulated quantization.

The [LinearQuantizer](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer) algorithm is implemented as an extension of [FX Graph Mode Quantization](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html) in PyTorch. It first traces the PyTorch model symbolically to obtain a [torch.fx](https://pytorch.org/docs/stable/fx.html) graph capturing all the operations in the model. It then analyzes this graph, and inserts [FakeQuantize](https://pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html) layers in the graph. [FakeQuantize](https://pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)  layer insertion locations are chosen such that model inference on hardware is optimized and only weights and activations which benefit from quantization are quantized. 

Since the `prepare` method uses [prepare_qat_fx](https://pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_fx.prepare_qat_fx.html) to insert quantization layers, the model returned from the method is a [torch.fx.GraphModule](https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule), and as a result custom methods defined on the original model class may not be available on the returned model. Some models, like those with dynamic control flow, may not be traceable into a `torch.fx.GraphModule`. We recommend following the instructions in [Limitations of Symbolic Tracing](https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing) and [FX Graph Mode Quantization User Guide](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html) to update your model first, before using [LinearQuantizer](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer) algorithm.

## Example

[Linear Quantization Tutorial](https://apple.github.io/coremltools/_examples/linear_quantization.html): Learn how to train a simple convolutional neural network using `LinearQuantizer`. This algorithm simulates the effects of quantization during training, by quantizing and dequantizing the weights and/or activations during the model’s forward pass. You can download a Jupyter Notebook version and the source code from the tutorial.
