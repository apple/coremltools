```{eval-rst}
.. index:: 
    single: optimize.coreml
```

# optimize.coreml API Overview

Use [`coremltools.optimize.coreml`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#module-coremltools.optimize.coreml) (post-training compression) to compress weights in the model. Weight compression reduces the space occupied by the model. However, the precision of the intermediate tensors and the compute precision of the ops are not altered — at load time or prediction time, weights are decompressed into float precision, and all computations use float precision.

## Steps to Compress a Model

The steps to compress an [ML Program](convert-to-ml-program) (`mlprogram`) model are as follows: 

- Load the model from disk into memory, unless it is already in memory (the object returned by [`ct.convert`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#coremltools.converters._converters_entry.convert) ).
- Define an [op-specific configuration](#op-specific-configurations) for the compression technique.
- Initialize [`ct.optimize.coreml.OptimizationConfig` ](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig).
- Call the corresponding compress weights method (such as [`cto.palettize_weights`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.palettize_weights) in the following example) to get the updated compressed model.

The following is an example of 6-bit palettization: 

```python
import coremltools as ct
import coremltools.optimize.coreml as cto

# load model
mlmodel = ct.models.MLModel(uncompressed_model_path)

# define op config 
op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=6)

# define optimization config by applying the op config globally to all ops 
config = cto.OptimizationConfig(global_config=op_config)

# palettize weights
compressed_mlmodel = cto.palettize_weights(mlmodel, config)
```

## Op-Specific Configurations

For sparsity, the op-specific configs can be defined using [`cto.OpThresholdPrunerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpThresholdPrunerConfig) or [`cto.OpMagnitudePrunerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpMagnitudePrunerConfig), and the method to prune is [`cto.prune_weights`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.prune_weights). 

For 8-bit linear quantization, the op-specific configs are defined using [`cto.OpLinearQuantizerConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OpLinearQuantizerConfig) and the method to quantize is [`cto.linear_quantize_weights`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.linear_quantize_weights) . 

The [`OptimizationConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig) can also be initialized from a [YAML](https://en.wikipedia.org/wiki/YAML) file. Example:

```
# linear_config.yaml file

config_type: "OpLinearQuantizerConfig"
global_config:
	mode: "linear_symmetric"
	dtype: "int8"
```

```python
import coremltools.optimize.coreml as cto

config = cto.OptimizationConfig.from_yaml("linear_config.yaml")
compressed_mlmodel = cto.linear_quantize_weights(mlmodel, config)
```

For details of key parameters to be set in each of the configs, see [Post-Training Pruning](pruning-a-core-ml-model), [Post-Training Palettization](post-training-palettization) and [Post-Training Quantization](data-free-quantization).

## Customizing Ops to Compress

Using the `global_config` flag in the [`OptimizationConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig) class applies the same config to all the ops with weights in the model. More granular control can be achieved by using the `op_type_configs` and `op_name_configs` flags of `OptimizationConfig`. 

The following example shows 6-bit palettization applied to all ops, with the exception that all the `linear` ops are set to 8 bits, and two of the `conv` ops called `conv_1` and `conv_4` are omitted from palettization.  

```python
import coremltools.optimize.coreml as cto

global_config = cto.OpPalettizerConfig(nbits=6, mode="kmeans")
linear_config = cto.OpPalettizerConfig(nbits=8, mode="kmeans")
config = cto.OptimizationConfig(
    global_config=global_config,
    op_type_configs={"linear": linear_config},
    op_name_configs={"conv1": None, "conv3": None},
)
compressed_mlmodel = cto.palettize_weights(mlmodel, config)
```

Using such customizations, different weights in a single model can be compressed with different techniques and configurations.  

## Requirements for Using optimize.coreml

Documented in the [Post-Training Compression](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#module-coremltools.optimize.coreml) section of the API Reference, the `optimize` submodule is available in Core ML Tools 7.0 and newer versions. The methods in this submodule are  available only for the [ML Program](convert-to-ml-program) (`mlprogram`) model type, which is the [recommended](target-conversion-formats) Core ML model format. 

For the APIs to compress the weights of neural networks, see [Compressing Neural Network Weights](quantization-neural-network).

```{admonition} API Compatibility

Note that coremltools 6.0 provided model compression APIs under the `coremltools.compression_utils.*` submodule. Those functions are now available under `coremltools.optimize.coreml.*`
```

