# API Overview

## Pruning APIs for Core ML model
- [`OpMagnitudePrunerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpMagnitudePrunerConfig): Prune the weights with a constant sparsity percentile.
- [`OpThresholdPrunerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpThresholdPrunerConfig): Set all weight values below a certain value.

### Data free Pruning
Here is a simple example showing the usage of [`OpThresholdPrunerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpThresholdPrunerConfig):
```python
from coremltools.optimize.coreml import (
    OpThresholdPrunerConfig,
    OptimizationConfig,
    prune_weights,
)

config = OptimizationConfig(global_config=OpThresholdPrunerConfig(
  threshold=0.03
))
model_compressed = prune_weights(model, config=config)
```
- All weight values below a certain value, as specified by `threshold`, are set to zero.

Another way to perform data-free pruning would be using the [`OpMagnitudePrunerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpMagnitudePrunerConfig). Below, we see how to configure it with different config parameters based on the op type and op name:

```python
from coremltools.optimize.coreml import (
    OpMagnitudePrunerConfig,
    OptimizationConfig,
    prune_weights,
)

global_config = OpMagnitudePrunerConfig(
    target_sparsity=0.5,
    weight_threshold=1024,
)
linear_config = OpMagnitudePrunerConfig(target_sparsity=0.75)
config = OptimizationConfig(
  global_config=op_config,
  op_type_configs={"linear": linear_config},
  op_name_configs={"fc": None}
)
model_compressed = prune_weights(model, config=config)
```
- `target_sparsity`: Lowest magnitude values up to `target_sparsity` are set to zero.
- `weight_threshold`: Weight tensors only of size (# of elements) greater than `weight_threshold` are pruned.
- Structured sparsity such as block-structured or `n:m` structured can be applied using `block_size` and `n_m_ratio` respectively.
- `op_type_configs` and `op_name_configs`: Configure the modules at a more fine-grained level. Here we configure all `linear` layers with 75% sparsity, skip pruning the `fc` layer, and the remaining layers are pruned to 50% sparsity.
- [`get_weights_metadata()`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.get_weights_metadata): Utility that provides detailed information about all the weights in the Core ML model, which can be used to find the names of the ops to customize.


## Pruning APIs for Torch model
- `SparseGPT` using [`LayerwiseCompressor`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#sparsegpt): A post-training calibration data-based compression algorithm based on the paper [SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://arxiv.org/pdf/2301.00774.pdf)
- `MagnitudePruner`:  A weight norm guided pruning algorithm based on the paper [To prune or not to prune](https://arxiv.org/pdf/1710.01878.pdf)

### Calibration data based Pruning (SparseGPT)
The following example shows how to compress a model using `SparseGPT` and [`LayerwiseCompressor`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#sparsegpt). Here we provide the pruning config using a `yaml` file. Across all APIs, the configs can be provided either in code via a dictionary structure or via `yaml` files.


`sparse_gpt_config.yaml`:
```yaml
algorithm: "sparsegpt"
layers:
  - 'model.layer\d+'
global_config:
  target_sparsity: 0.5
calibration_nsamples: 125
```

```python
from coremltools.optimize.torch import (
    LayerwiseCompressor,
    LayerwiseCompressorConfig,
)

config = LayerwiseCompressorConfig.from_yaml("sparse_gpt_config.yaml")

compressor = LayerwiseCompressor(model, config)
model = compressor.compress(dataloader=dataloader, device=torch.device("cuda"))
```
- `algorithm` is set to `"sparsegpt"` in the [`LayerwiseCompressor`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#sparsegpt) algorithm.
- `target_sparsity`: Refers to the amount of sparsity to apply for each layer’s weight tensor.
- `layers`: Layers to be pruned. This is a list of either fully-qualified layer (module) name(s) or a regex for the layer name(s).
- `weight_dtype`, `quantization_granularity` and `quantization_scheme` can be configured to quantize the non-zero weights for further compression.
-  `n:m` structured sparsity can be set through the `n_m_ratio` option.
- The `compress` method takes in a dataloader for the calibration dataset as well as the device for performing computation. The dataloader is an iterable of the inputs that need to be fed into the model. 


### Data-free Pruning
As mentioned in the previous [Pruning Algorithms](opt-pruning-algos) section, the [`MagnitudePruner`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#magnitude-pruning) can be used to perform a data-free pruning to experiment with different pruning structures. In the example below, `n:m` structured (with a ratio `6:8`) pruning is applied to the model.

```python
from coremltools.optimize.torch import (
  MagnitudePrunerConfig,
  MagniutdePruner,
)

config = MagniutdePrunerConfig.from_dict({
  "global_config": {
    "n_m_ratio": [6, 8]
  }
})
pruner = MagniutdePruner(model, config)

pruner.prepare()
model = pruner.finalize()
```
Here the `ConstantSparsityScheduler` is being used by default to prune the model in a data-free manner.


### Training time Pruning
The [`MagnitudePruner`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#magnitude-pruning) can be used to introduce sparsity while fine-tuning the model to adapt to the loss of accuracy due to the sparsification of the model. In the example below, 75% sparsity is applied to all convolution layers of the model in a gradual incremental manner.
```python
from coremltools.optimize.torch import (
  MagnitudePrunerConfig,
  MagniutdePruner,
)

config = MagnitudePrunerConfig.from_dict({
    "module_type_configs": {
      "Conv2d": {
        "scheduler": {"update_steps": "range(0, 100, 5)"},
        "target_sparsity": 0.75,
        "granularity": "per_scalar",
      },
    }
})

pruner = MagnitudePruner(model, config)

model = pruner.prepare()


for epoch in range(num_epochs):
  for inp, label in train_dataloader:
    train_step(inp, label)
    pruner.step()

pruner.finalize(inplace=True)
```
- `target_sparsity`: Refers to the amount of sparsity that the model will finally have.
- `granularity`: One of `per_scalar`, `per_kernel` or `per_channel` allows for different ways of structuring the sparsity in the weight tensor.
- The `scheduler` (above uses the `PolynomialDecayScheduler`) incrementally adds sparsity through the course of training to make sure the weights can adapt to the introduction of sparsity. The `update_steps` parameter refers to the training steps upon which the sparsity has to be introduced. In this example, the sparsity is applied every five steps starting from zero all the way up to 100. 
- `MagnitudePruner.prepare` helps to insert the pruning layers and hooks on to the model.
- `MagnitudePruner.step` incremenetally adds sparsity based on the sparsity schedule described by the `scheduler`.
- `MagnitudePruner.finalize` commits all the changes on the model by replacing the pruned weights with zeros.

### Converting Torch models to Core ML
If the Torch model already contains weights that have been zeroed out but are still in a dense representation, the `ct.optimize.coreml` APIs mentioned above can be used to generate a sparse representation Core ML model. If the Torch model was pruned using the `ct.optimize.torch` APIs mentioned above, then simply calling `ct.convert` should be sufficient to generate the sparse Core ML model.

For more details, refer to [PyTorch Conversion Workflow](convert-pytorch-workflow).
