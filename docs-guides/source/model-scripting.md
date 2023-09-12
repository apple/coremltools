# Model Scripting

```
.. index:: 
    single: PyTorch; model scripting
    single: model scripting
    single: PyTorch; JIT script
    single: TorchScript
```

Model tracing is not appropriate in all cases. If your model includes a data-dependent control flow, such as a loop or conditional, you can experiment with PyTorch's [JIT script](https://pytorch.org/docs/stable/generated/torch.jit.script.html) to _script_ the model by analyzing the code to generate TorchScript.

```{warning}
If you convert a scripted model, a warning appears explaining that support for scripted models is experimental.
```

## Why Tracing May Not be Accurate

Assume a model that runs a convolutional layer in a loop, with the loop terminating when some function of the output of that layer is true.

If you trace the model with some input `x`, and the loop runs five times for input `x`, the traced model encodes computing the convolution five times, since that's exactly what happens during the trace. 

However, if you trace the model with a different input `y`, and the loop runs only four times, the traced model encodes computing the convolution only _four_ times. 

Thus, the same model can produce two different traces. This is probably not the intent of the model, so tracing won't produce the correct TorchScript version of the model.

## Use JIT Script

Use PyTorch's [JIT script](https://pytorch.org/docs/stable/generated/torch.jit.script.html) to script the model and convert it to TorchScript. To demonstrate, follow this code fragment:

- Define a single convolution plus activation block.
- If the input tensor to the network has mean value less than zero, run the block twice.
- Otherwise, run the block once.
- Return the output of the last pass through the block as the model's output.

To construct this model in code:

```python
class _LoopBody(nn.Module):
    def __init__(self, channels):
        super(_LoopBody, self).__init__()
        conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )
        self.conv = conv

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x

class ControlFlowNet(nn.Module):
    def __init__(self, num_channels: int):
        super(ControlFlowNet, self).__init__()
        self.loop_body = _LoopBody(num_channels)

    def forward(self, x):
        avg = torch.mean(x)
        if avg.item() < 0:
            loop_count = 2
        else:
            loop_count = 1
        for _ in range(loop_count):
            x = self.loop_body(x)
        return x
```

Since the model uses both conditionals and loops, you can't just trace it. Instead, directly invoke the JIT script to convert it to TorchScript, and then convert it to a Core ML neural network:

```python
model = ControlFlowNet(num_channels=3)
scripted_model = torch.jit.script(model)

import coremltools
mlmodel = coremltools.converters.convert(
  scripted_model,
  inputs=[coremltools.TensorType(shape=(1, 3, 64, 64))],
)
```

The result is a Core ML neural network that correctly includes the original model's control flow.

## Mix Tracing and Scripting

Scripting captures _the entire_ structure of a model, so it makes sense to apply scripting only to small models, or to parts of a model. Apply a mix of scripting and tracing to optimize which parts of the model you actually want to capture, and which parts you don't need. For example, use JIT script only on control flow sections, and trace all other sections of the graph. You should keep control flow sections as small as possible.

On the other hand, if you have a very complicated computation graph inside a control flow, you may want to apply tracing to the graph inside the control flow, and then apply scripting to the model as a whole. 

For example, in experiments to determine which activation function produces the best result, you may have an example model with a parameter that enables you to select from several different activation functions. After you've trained the model, you probably don't want to change which activation you're using. However, if you were to script the model it would still have that selection logic. To separate the control flow body, make a small change to the top-level module as follows, so that you are tracing the loop body module inside the main model: 

```python
class ControlFlowNet2(nn.Module):
    def __init__(self, num_channels: int):
        super(ControlFlowNet2, self).__init__()
        self.loop_body = _LoopBody(num_channels)
        self.loop_body = torch.jit.trace(self.loop_body, torch.randn(1,3,64,64))

    def forward(self, x):
        avg = torch.mean(x)
        if avg.item() < 0:
            loop_count = 2
        else:
            loop_count = 1
        for _ in range(loop_count):
            x = self.loop_body(x)
        return x
```

When the JIT script encounters `loop_body`, it sees that it has already been converted into TorchScript and will skip over it.

At this point you can instantiate the model, script it, and convert it just like before:

```python
model = ControlFlowNet2(num_channels=3)
scripted_model = torch.jit.script(model)

import coremltools
mlmodel = coremltools.converters.convert(
  scripted_model,
  inputs=[coremltools.TensorType(shape=(1, 3, 64, 64))],
)
```

For more examples of converting models to TorchScript, see the [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html).



