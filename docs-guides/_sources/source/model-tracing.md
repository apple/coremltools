```
.. index:: 
    single: PyTorch; model tracing
    single: model tracing
    single: TorchScript
```

# Model Tracing

The easiest way to generate TorchScript for your model is to use PyTorch's [JIT tracer](https://pytorch.org/docs/stable/generated/torch.jit.trace.html). Tracing runs an example input tensor through your model, and captures the operations that are invoked as that input makes its way through the model's layers.

```{admonition} Tracing Limitations

Tracing the model captures only the operations that are performed for a specific input. If your model uses a data-dependent control flow, such as a loop or conditional, the traced model won't generalize to other inputs. In such cases you can experiment with applying PyTorch's JIT scripter to your model as described in [Model Scripting](model-scripting). Use the JIT scripter only on control flow sections, and trace all other sections of the graph. You should keep the control flow section as small as possible.
```

The following example builds a simple model from scratch and traces it to generate the TorchScript object needed by the converter. Follow these steps:

1. Define a simple layer module to reuse:
	
	```python
	import torch
	import torch.nn as nn
	import torch.nn.functional as F

	# Define a simple layer module we'll reuse in our network.
	class Layer(nn.Module):
		def __init__(self, dims):
			super(Layer, self).__init__()
			self.conv1 = nn.Conv2d(*dims)

		def forward(self, x):
			x = F.relu(self.conv1(x))
			x = F.max_pool2d(x, (2, 2))
			return x
	```

2. Define a simple network consisting of several base layers:
	
	```python
	# A simple network consisting of several base layers.
	class SimpleNet(nn.Module):
		def __init__(self):
			super(SimpleNet, self).__init__()
			self.layer1 = Layer((3, 6, 3))
			self.layer2 = Layer((6, 16, 1))

		def forward(self, x):
			x = self.layer1(x)
			x = self.layer2(x)
			return x
	```

3. Instantiate the network:
	
	```python
	model = SimpleNet()  # Instantiate the network.
	```

4. Define the input, which is needed by jit tracer, and trace the model.
	
	Run `torch.jit.trace` on your model with an example input, and save the resulting traced object. For an example input, you can use one sample of training or validation data, or even use randomly-generated data as shown in the following code snippet:
	
	```python
	example = torch.rand(1, 3, 224, 224)  # Example input, needed by jit tracer.
	traced = torch.jit.trace(model, example)  # Generate TorchScript by tracing.
	```

```{note}

Don’t worry if your model is fully convolutional or otherwise has variable-sized inputs. You can fully describe your model's input shape when you convert the TorchScript model to Core ML.
```

5. Optionally pass the traced model directly to the converter:
	
	```
	traced.save(“model.pt”) # Optional, can pass traced model directly to converter.
	
	```



