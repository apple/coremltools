```{eval-rst}
.. index:: 
    single: palettization; training-time
    single: DKMPalettizer
```

# Training-Time Palettization

The [DKMPalettizer](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer) class implements a palettization algorithm based on the [DKM (Differentiable K-means)](https://machinelearning.apple.com/research/differentiable-k-means) paper.  The hyper-parameters of the algorithm can be set via the [DKMPalettizerConfig](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizerConfig) object. 

The key idea of the algorithm is that in each training step, a soft k-means cluster assignment of weight tensors is performed such that each operation in the process is differentiable. This allows for gradient updates to take place for the weights while a [lookup table (LUT)](https://en.wikipedia.org/wiki/Lookup_table) of centroids and indices are learned.

## Use the `DKMPalettizer`

Follow these key steps: 

1. Define a [`DKMPalettizerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizerConfig) config to specify the palettization parameters.
2. Initialize the palettizer object using [`DKMPalettizer`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer).
3. Call the [`prepare`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer.prepare) API to update the PyTorch model with palettization-friendly modules.
4. Run the usual training loop, with the addition of the [`palettizer.step`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer.step) call.
5. Once the model has converged, use the [`finalize`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer.finalize) API to prepare the model for conversion to Core ML.

The following sample code shows how you can use [`DKMPalettizer`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer) to perform training-time palettization on your PyTorch model.

```python
import torch
import torch.nn as nn

import coremltools as ct
from coremltools.optimize.torch.palettization import DKMPalettizer, DKMPalettizerConfig

model = nn.Sequential(nn.Linear(4, 500), nn.Sigmoid(), nn.Linear(500, 4), nn.Sigmoid())
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
data = create_data()

# Prepare model for palettization
module_config = {nn.Linear: {"n_bits": 2, "weight_threshold": 1000, "milestone": 2}}
config = DKMPalettizerConfig.from_dict({"module_type_configs": module_config})
palettizer = DKMPalettizer(model, config)

prepared_model = palettizer.prepare()

# Fine-tune the model for a few epochs after this.
for inputs, labels in data:
    output = model(inputs)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
    palettizer.step()

# prepare for conversion
finalized_model = palettizer.finalize()

# trace and convert
example_input = torch.rand(1, 4) # shape of input for the model
traced_model = torch.jit.trace(finalized_model, example_input)

coreml_model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)],
    pass_pipeline=ct.PassPipeline.DEFAULT_PALETTIZATION,
    minimum_deployment_target=ct.target.iOS16,
)
coreml_model.save("~/compressed_model.mlpackage")
```

Set the following key parameters:

- `n_bits` : This controls the number of clusters, which are `2^n_bits` .
- `weight_threshold`: Weight tensors that are smaller than this size are not palettized. Defaults to 2048. 
- `milestone` : The number of times the [`palettizer.step`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer.step) API has to be called before palettization is enabled. This number can be a training step number if the `palettizer.step` API is called once every training step, or it can be an epoch number if the `palettizer.step` API is called once every epoch. Defaults to zero, in which case palettization is enabled from the start of the training loop. 

For options on how to set different palettization configs for different weights in the same network, see  [optimize.torch API Overview](optimizetorch-api-overview).

## How It Works

The training-time palettization algorithm works by inserting palettization submodules inside a model. These submodules simulate palettization during training using a differentiable version of the k-means algorithm, thus helping the model learn to predict well with palettized weights. 

Palettization is implemented as an extension of [Eager Mode Quantization](https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization) in PyTorch. In particular, in place of the [FakeQuantize](https://pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html) layers inserted during quantization, palettization inserts [`FakePalettize`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.FakePalettize) layers. These `FakePalettize` layers encode the the lookup table (LUT) and index parameters, which are used for palletizing the weights. 

The DKM algorithm, which forms the basis of palettization implemented in `coremltools.optimize.torch`, uses an attention based mechanism to compute centroids in a differentiable way, following these steps:

1. The algorithm first performs a normal k-means operation to get `C=2^n_bits` clusters and then calculates a distance matrix between all the weights in a layer(`W`) and all the clusters.
2. This distance matrix is used to calculate an attention matrix, which stores information of closeness of individual weights to the `C` clusters.
3. New clusters are computed from a matrix multiplication of attention matrix and the weights.
4. Steps 2 and 3 are repeated a certain number of times or until the newly computed clusters are closer to the old clusters.

Since the algorithm involves computation of the distance and the attention matrices, the space complexity of the algorithm is roughly `O(2^(n_bits)*W)`, which in practice can take substantial memory. Therefore, we recommend using `2, 4-bit` options for training-time palettization. Also, as described in [Accuracy and Performance](performance-impact.md#weight-palettization), for higher bit palettization, post-training palettization provides a good compression-accuracy tradeoff.

```{admonition} Use torch.nn Modules

Since `coremltools.optimize.torch` APIs are built on top of PyTorch, we recommend using [`torch.nn`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) modules in your model. For example, a custom written `Conv2d` layer in model wouldn't get palettized. We recommend using `torch.nn.Conv2d` instead. The following modules are supported for palettization: `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, `torch.nn.Linear`, `torch.nn.LayerNorm`, `torch.nn.Embedding`, and `torch.nn.MultiheadAttention`.
```

## Example

[Palettization Using Differentiable K-Means Tutorial](https://apple.github.io/coremltools/_examples/dkm_palettization.html): Learn how to palettize a neural network using `DKMPalettizer`, which clusters the weights using a differentiable version of `k-means`, allowing the lookup table (LUT) and indices of palettized weights to be learned using a gradient-based optimization algorithm. You can download a Jupyter Notebook version and the source code from the tutorial.
