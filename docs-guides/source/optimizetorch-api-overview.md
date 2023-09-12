```{eval-rst}
.. index:: 
    single: optimize.torch
```

# optimize.torch API Overview

Use [`coremltools.optimize.torch`](https://apple.github.io/coremltools/source/coremltools.optimize.html#training-time-compression) (training-time compression) to train your model in a compression-aware fashion, or start from a pre-trained `float` precision model  and fine-tune it with training data. 

## API Format and Interface

All model optimizers use a similar API format:

```python
import coremltools.optimize.torch as cto

pruner = cto.pruning.MagnitudePruner(model, config)

quantizer = cto.quantization.LinearQuantizer(model, config)

palettizer = cto.palettization.DKMPalettizer(model, config)
```

- `model` is the `torch.nn.Module` instance you want to optimize.
- `config` specifies how the model will be configured for optimization. These configuration objects share the same API among different optimization techniques.

All model optimizers also provide the same interface for optimizing the models:

- `prepare`: Insert model optimization layers in the  model.
- `step`: Step through the optimization schedule.
- `report`: Create a report with information about current state of the optimization such as the current sparsity of a layer. 
- `finalize`: Create model weights from learned optimization parameters, and make the model ready for export using [`coremltools.convert`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#coremltools.converters._converters_entry.convert). 

The following set of examples show how these optimizers can be created and integrated in PyTorch code.

## Creating Configurations

To initialize a model optimizer such as [`MagnitudePruner`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.MagnitudePruner), create it using a [YAML](https://en.wikipedia.org/wiki/YAML) file or programmatically.

### Create a YAML File

Create a YAML configuration file (`config.yaml`) describing how different modules are to be configured. The file can contain up to three sections:

- `global_config`: Used to specify a configuration that is applied globally, on all supported module types (`torch.nn.Conv2d`, `torch.nn.Linear`, `torch.nn.ConvTranspose2d`, and so on).
- `module_type_configs`: Used to specify a common configuration for all modules of the same type (e.g. `torch.nn.Conv2d`).
- `module_name_configs`: In a `torch.nn.Module`, all submodules have a unique name. Using this option you can set a particular submodule's configuration.

**Example 1: Configure Globally**

The following sample `config.yaml` file configures parameters of the pruner globally. These parameters are used to configure all supported modules:

```yaml
global_config:
  scheduler:
  update_steps: [100, 200, 300, 500]
  target_sparsity: 0.8
```

**Example 2: Configure More Granularly**

The following sample `config.yaml` file  configures parameters of the pruner on a more granular level, using different sparsity types (`block` and `n:m`) for convolution and linear layers, respectively. The configuration also sets `module2.linear` to `null` in order to leave it alone and not prune it.

```yaml
module_type_configs:
  Linear:
    scheduler:
      update_steps: [100, 200, 300, 500]
    n_m_ratio: [3, 4]
  Conv2d:
    scheduler:
      update_steps: [100, 200, 300, 500]
    target_sparsity: 0.5
    block_size: 2

module_name_configs:
  module2.conv1:
    scheduler:
      update_steps: [100, 200, 300, 500]
    target_sparsity: 0.75
  module2.linear: null
```

**Using the Configuration**

The following example shows how to use the `config.yaml` configuration file with a `MagnitudePruner`:

```python
import torch
import coremltools as ct
from coremltools.optimize.torch.pruning import MagnitudePruner, MagnitudePrunerConfig

model, loss_fn, optimizer = create_model_and_optimizer()
data = create_data()

# Initialize pruner and configure it
config = MagnitudePrunerConfig.from_yaml("config.yaml")
pruner = MagnitudePruner(model, config)

# Insert pruning layers in the model
model = pruner.prepare()

for inputs, labels in data:
    output = model(inputs)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
    pruner.step()

# Commit pruning masks to model parameters
pruner.finalize(inplace=True)

# Export
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
    minimum_deployment_target=ct.target.iOS16,
)
coreml_model.save("~/pruned_model.mlpackage")
```

### Initialize Programmatically

You may prefer to configure a model optimizer programmatically, which is useful especially if you have complex conditions for configuring the optimizer that are hard to express in a YAML file.

**Programmatic Example 1**

```python
import torch

import coremltools as ct
from coremltools.optimize.torch.palettization import (
    DKMPalettizer,
    DKMPalettizerConfig,
    ModuleDKMPalettizerConfig,
)

# code that defines the pytorch model, and optimizer
model, loss_fn, optimizer = create_model_and_optimizer()
data = create_data()

# Initialize the palettizer
config = DKMPalettizerConfig(
    global_config=ModuleDKMPalettizerConfig(n_bits=4, cluster_dim=4)
)

palettizer = DKMPalettizer(model, config)

# Prepare the model to insert FakePalettize layers for palettization
model = palettizer.prepare(inplace=True)

# Use palettizer in the PyTorch training loop
for inputs, labels in data:
    output = model(inputs)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
    palettizer.step()

# Fold LUT and indices into weights
model = palettizer.finalize(inplace=True)

# Export
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    pass_pipeline=ct.PassPipeline.DEFAULT_PALETTIZATION,
    minimum_deployment_target=ct.target.iOS16,
)
coreml_model.save("~/palettized_model.mlpackage")
```

**Programmatic Example 2**

If you want to quantize _only_ the convolution modules that have a kernel size of 1, you can do so:

```python
import torch

import coremltools as ct
from coremltools.optimize.torch.quantization import (
    LinearQuantizer,
    LinearQuantizerConfig,
    ModuleLinearQuantizerConfig,
    ObserverType,
    QuantizationScheme,
)

model, loss_fn, optimizer = create_model_and_optimizer()
data = create_data()

# Initialize the quantizer
global_config = ModuleLinearQuantizerConfig(
    quantization_scheme=QuantizationScheme.symmetric
)

config = LinearQuantizerConfig().set_global(global_config)

# We only want to quantize convolution layers which have a kernel size of 1 or all linear layers.
for name, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        if m.kernel_size == (1, 1):
            config = config.set_module_name(
                name,
                ModuleLinearQuantizerConfig(
                    weight_observer=ObserverType.mix_max, weight_per_channel=True
                ),
            )
        else:
            config = config.set_module_name(name, None)

quantizer = LinearQuantizer(model, config)

# Prepare the model to insert FakeQuantize layers for QAT
example_input = torch.rand(1, 3, 224, 224)
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
traced_model = torch.jit.trace(model, example_input)

coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    minimum_deployment_target=ct.target.iOS17,
)
coreml_model.save("~/quantized_model.mlpackage")
```

## Tutorials

- [Magnitude Pruning Tutorial](https://apple.github.io/coremltools/_examples/magnitude_pruning.html): Learn how to train a simple convolutional neural network using `MagnitudePruner`. 
- [Palettization Using Differentiable K-Means Tutorial](https://apple.github.io/coremltools/_examples/dkm_palettization.html): Learn how to palettize a neural network using `DKMPalettizer`, which clusters the weights using a differentiable version of `k-means`, allowing the lookup table (LUT) and indices of palettized weights to be learned using a gradient-based optimization algorithm.
- [Linear Quantization Tutorial](https://apple.github.io/coremltools/_examples/linear_quantization.html): Learn how to train a simple convolutional neural network using `LinearQuantizer`. This algorithm simulates the effects of quantization during training, by quantizing and dequantizing the weights and/or activations during the model’s forward pass. 

From each tutorial you can download a Jupyter Notebook version and the source code.
