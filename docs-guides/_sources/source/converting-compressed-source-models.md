```{eval-rst}
.. index:: 
    single: convert compressed models
```

# Converting Compressed Source Models

This section shows how you can indicate to the converter to use the sparse or palettized representations to store the weights. This is required when you bring in a source model whose weights are compressed but still represented in dense `float`format.  This is the case for PyTorch models that are updated and fine-tuned using the [`ct.optimize.torch`](optimizetorch-api-overview) APIs.  

## Convert models with sparse weights

If your source model weights have lots of zeros, then specify the `pass_pipeline` argument as follows: 

```python
import coremltools as ct

model_with_sparse_weights = ct.convert(
    mlmodel,
    pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
    minimum_deployment_target=ct.target.iOS17,
)
```

During the conversion process, an additional [graph pass](graph-passes-intro) will be run, which will convert the weight values below a certain low threshold (`default=1e-3`) to exact zeros, and then use the a sparse representation  to store the weights, thereby saving space. 

Remember to use this option to convert any PyTorch model that has been pruned using the [`ct.optimize.torch.pruning.MagnitudePruner`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.MagnitudePruner) method. 

## Convert models with palettized weights

If your source model weights are palettized; that is, clustered and can only take on a few discrete values, then you can save space by invoking the following pass during conversion: 

```python
import coremltools as ct

model_with_lut_weights = ct.convert(
    mlmodel,
    pass_pipeline=ct.PassPipeline.DEFAULT_PALETTIZATION,
    minimum_deployment_target=ct.target.macOS13,
)
```

This will invoke an additional [graph pass](graph-passes-intro) that will automatically detect whether the weights have repeated values. If they do, and if the overall weight tensor has at most 256 or less unique values (which means it can be represented with an 8-bit or less [lookup table](https://en.wikipedia.org/wiki/Lookup_table)), it will use a more compact representation using lookup tables (LUTs) to store them. 

 Remember to use this option to convert any PyTorch model that has been compressed using the [`ct.optimize.torch.palettization.DKMPalettizer`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.palettization.html#coremltools.optimize.torch.palettization.DKMPalettizer) method.

## Convert PyTorch models with quantized weights and activations

For PyTorch models with quantized weights and/or activations, no additional `pass_pipeline` flag is required since PyTorch stores quantized weights or activations using `qint` or `quint` data types, and additional quantization ops are used. This is picked up automatically by the conversion process, which then automatically uses linear quantized storage format to store weights.

