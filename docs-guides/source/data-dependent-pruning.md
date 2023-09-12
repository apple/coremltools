```{eval-rst}
.. index:: 
    single: pruning; training-time
    single: MagnitudePruner
```

# Training-Time Pruning

The [Training-Time Pruning](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#training-time-pruning) API in `coremltools.optimize.torch` builds on top of the [BasePruningMethod](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.BasePruningMethod.html) API in PyTorch and extends it to:

- Make it easy to configure different submodules in a model to use different pruning configurations.
- Provide different pruning modes such as `block` sparsity and `n:m` sparsity.
- Provide a mechanism to update the pruning mask during training, following a pruning schedule.

## Use `MagnitudePruner`

Follow these key steps: 

- Define the [`MagnitudePrunerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.MagnitudePrunerConfig) config to specify the parameters of pruning.
- Initialize the pruner object using [`MagnitudePruner`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.MagnitudePruner) .
- Call the [`prepare`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.MagnitudePruner.prepare) API to update the torch model.
- Run the usual training loop, with the addition of the `pruner.step` call.
- Once the model has converged, use the [`finalize`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.MagnitudePruner.finalize) API to prepare the model for conversion to Core ML.

The following code sample shows how you can use `MagnitudePruner` to perform training-time pruning on your PyTorch model.

```python
from collections import OrderedDict

import torch

import coremltools as ct
from coremltools.optimize.torch.pruning import MagnitudePruner, MagnitudePrunerConfig

model = torch.nn.Sequential(
    OrderedDict(
        [
            ("conv1", torch.nn.Conv2d(3, 32, 3, padding="same")),
            ("conv2", torch.nn.Conv2d(32, 32, 3, padding="same")),
        ]
    )
)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
data = get_data()

# initialize pruner and configure it
# we will configure the pruner for all conv2d layers
config = MagnitudePrunerConfig.from_dict(
    {
        "module_type_configs": {
            "Conv2d": {
                "scheduler": {"update_steps": [3, 5, 7]},
                "target_sparsity": 0.75,
                "granularity": "per_scalar",
            },
        }
    }
)

pruner = MagnitudePruner(model, config)

# insert pruning layers in the model
model = pruner.prepare()

for inputs, labels in data:
    output = model(inputs)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
    pruner.step()

# commit pruning masks to model parameters
pruner.finalize(inplace=True)

# trace and convert the model
example_input = torch.rand(1, 3, 224, 224)  # shape of input for the model
traced_model = torch.jit.trace(model, example_input)
coreml_model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)],
    pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
    minimum_deployment_target=ct.target.iOS17,
)
coreml_model.save("~/pruned_model.mlpackage")
```

The [MagnitudePruner](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.MagnitudePruner) class implements a weight norm guided pruning algorithm based on the paper [To prune or not to prune](https://arxiv.org/pdf/1710.01878.pdf).  It extends the algorithm of the paper by also providing options to perform structured `block` and `n:m` sparsity. These options can be set via the [MagnitudePrunerConfig](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.MagnitudePrunerConfig) and [ModuleMagnitudePrunerConfig](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.ModuleMagnitudePrunerConfig) objects. 

The key idea of the algorithm is that in each training step, the values of the weight tensors are sorted according to their magnitudes and smallest values are set to zero, while the non-zero values are updated during the training process. A [polynomial scheduler](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.pruning_scheduler.PolynomialDecayScheduler) is used to gradually increase the amount of sparsity, controlled via the `update_steps` parameter, until the desired level of sparsity (as specified by `target_sparsity` parameter) is achieved. 

For options on how to set different pruning configs for different weights in the same network, see  [`optimize.torch` API Overview](optimizetorch-api-overview).

## How It Works

[MagnitudePruner](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.MagnitudePruner) works by inserting forward pre-hooks on the submodules that are set up for pruning, as specified using [MagnitudePrunerConfig](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.MagnitudePrunerConfig) and [ModuleMagnitudePrunerConfig](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.ModuleMagnitudePrunerConfig). It also registers buffers containing the pruning masks on these submodules. 

During the model's forward pass, its original weights are multiplied with these pruning masks, thus simulating the effects of pruning during training. This enables the model to learn to predict well using pruned weights. 

The original weights of the model are updated using back propagation during backward pass. The pruning masks are updated gradually over the course of the training using a [Pruning Scheduler](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#pruning-scheduler) to increase the amount of sparsity introduced.

Calling the [`step`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.MagnitudePruner.step) method on the pruner object advances the pruning scheduler by one step. Pruning scheduler computes the amount of sparsity induced as a function of the training step. The updated amount of sparsity is then used by the pruner object to compute new pruning masks. Once the target sparsity has been achieved, the pruner will maintain that level of sparsity for the rest of the training.

The pruning API copies the original weight parameter onto a new parameter called `weight_orig`. The `weight` parameter will be the pruned weights (used for the forward pass) while the `weight_orig` will be the un-pruned weights (used for the backward pass). Therefore, if you plan on having other module hooks that use the `weight` parameter, the hooks will return the pruned weights after using the pruning API.

```{admonition} Use torch.nn Modules

Since `coremltools.optimize.torch` APIs are built on top of PyTorch, we recommend using [`torch.nn`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) modules in your model. For example, a custom written `Conv2d` layer in model wouldn't get pruned. We recommend using `torch.nn.Conv2d` instead. The following layers are supported for pruning: `torch.nn.Linear`, `torch.nn.Conv1d`, `torch.nn.Conv2d` and `torch.nn.Conv3d`.
```

## Example

[Magnitude Pruning Tutorial](https://apple.github.io/coremltools/_examples/magnitude_pruning.html): Learn how to train a simple convolutional neural network using `MagnitudePruner`. You can download a Jupyter Notebook version and the source code from the tutorial.

