```{eval-rst}
.. index:: 
    single: quantization; post-training
    single: linear_quantize_weights, OpLinearQuantizerConfig
```

# Post-Training Quantization

You can linearly quantize the weights of your Core ML model by using the [`linear_quantize_weights`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.linear_quantize_weights) method as follows: 

```python
import coremltools.optimize.coreml as cto

op_config = cto.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)
config = cto.OptimizationConfig(global_config=op_config)

compressed_8_bit_model = cto.linear_quantize_weights(model, config=config)
```

```{admonition} Quantize Activations Plus Weights

To quantize the activations in addition to the weights, use [Training-Time Quantization](data-dependent-quantization).
```

The [`linear_quantize_weights`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.linear_quantize_weights) method iterates over the weights of the model. Those weights whose sizes are above the specified `weight_threshold` are quantized to the 8-bit range according to the `mode` specified in [`OpLinearQuantizerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpLinearQuantizerConfig). The method defaults to `linear_symmetric`,  which uses only `per-channel` `scales` and no `zero-points`. You can also choose a `linear` mode which uses a `zero-point` as well, which may help to get slightly better accuracy.

For options on how to set different quantization configs for different weights in the same network, see  [Customizing Ops to Compress](optimizecoreml-api-overview.md#customizing-ops-to-compress).

For more details on the parameters available in the config, see the following in the API Reference:

- [OpLinearQuantizerConfig](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpLinearQuantizerConfig)
- [OptimizationConfig](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig)
- [linear_quantize_weights](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.linear_quantize_weights)

If your model's accuracy drops considerably after quantizing the weights of the model, or your model is fully resident on the Neural Engine and you want to see if you can get more latency gains, then consider quantizing both the weights and activation using [Training-Time Quantization](data-dependent-quantization).

