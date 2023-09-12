```{eval-rst}
.. index:: 
    single: pruning; post-training
    single: OpThresholdPrunerConfig, OpMagnitudePrunerConfig
```

# Post-Training Pruning

To sparsify the weights of a Core ML model, you can use one of two configurations:

- [`OpThresholdPrunerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpThresholdPrunerConfig): Sets all weight values below a certain value.
- [`OpMagnitudePrunerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpMagnitudePrunerConfig): Prune the weights with a constant sparsity percentile.

The following example shows how to use them. 

## `OpThresholdPrunerConfig`

```python
from coremltools.optimize.coreml import (
    OpThresholdPrunerConfig,
    OptimizationConfig,
    prune_weights,
)

op_config = OpThresholdPrunerConfig(
    threshold=0.03,
    minimum_sparsity_percentile=0.55,
    weight_threshold=1024,
)
config = OptimizationConfig(global_config=op_config)
model_compressed = prune_weights(model, config=config)
```

[`OpThresholdPrunerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpThresholdPrunerConfig) works by setting all weight values below a certain value, as specified by `threshold`, to zero. In the resulting weight tensor, sparse representation is used only if the proportion of values that are zero is greater than a level, as specified by  `minimum_sparsity_percentile`. Otherwise dense format will be used. 

The `weight_threshold` parameter specifies the minimum number of elements that the weight tensor must have for this operation to take effect. In the previous code sample, since `weight_threshold=1024` was specified, all the weight tensors that have less than `1024` elements will be left untouched, while the tensors of size greater than `1024` will be sparsified according to the `threshold` and `minimum_sparsity_percentile` settings. 

## `OpMagnitudePrunerConfig`

```python
from coremltools.optimize.coreml import (
    OpMagnitudePrunerConfig,
    OptimizationConfig,
    prune_weights,
)

op_config = OpMagnitudePrunerConfig(
    target_sparsity=0.6,
    weight_threshold=1024,
)
config = OptimizationConfig(global_config=op_config)
model_compressed = prune_weights(model, config=config)
```

When using [`OpMagnitudePrunerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpMagnitudePrunerConfig), the lowest magnitude weight values up to the percentile as specified by `target_sparsity` are set to zero, and the weight is stored in the sparse format. You can also specify to apply  structured `block` or `n:m` sparsity using the parameters `block_size` and `n_m_ratio` respectively  in `OpMagnitudePrunerConfig`. 

For options on how to set different pruning configs for different weights in the same network, see [Customizing Ops to Compress](optimizecoreml-api-overview.md#customizing-ops-to-compress). 

For more details on the parameters available in the config, see the API Reference: 

- [`OpThresholdPrunerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpThresholdPrunerConfig) 
- [`OpMagnitudePrunerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpMagnitudePrunerConfig) 
- [`OptimizationConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig)
- [`prune_weights`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.prune_weights)

```{admonition} Use Post-Training Pruning APIs for Benchmarking

Unless the weights in your Core ML model are known to have a lot of zeros, using data-free pruning is typically going to lead to a large accuracy loss for any meaningful level of sparsity. Therefore, use the [`prune_weights`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.prune_weights) method to experiment with different patterns and levels of sparsity to see the impact on size reduction and performance, and then use the results as a guiding factor to find a config that then you prune for using fine tuning.
```

