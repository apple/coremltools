Conversion
==========

Given a compressed torch model whose weights are compressed but still
represented in dense `float` format,
we want the converted Core ML model to have the corresponding
[compressed MIL](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#module-coremltools.converters.mil.mil.ops.defs.iOS16.constexpr_ops)
ops. This will ensure that the compression benefits at runtime are realized.

The converted model with the compressed MIL ops can be generated in two ways:

- By using the compression info stored explicitly in the torch model: if you are using
  coremltools 8 and `ct.optimize.torch.*` APIs, the APIs
  will automatically insert compression info in the torch model, which will be picked
  by `ct.convert`. To learn more details on how this is implemented,
  see the section [Use compression info embedded in torch models](#use-compression-info-embedded-in-torch-models).
- By specifying a graph pass during conversion: this is required for some compression modes if
  you are using Core ML Tools 7. To learn more about this, see the section [Specify pass pipeline](#specify-pass-pipeline).
  Note that this does not cover all the available compression modes. The first option above 
  is the recommended workflow.

## Specify pass pipeline

If you are using Core ML Tools 7, you need to specify the graph pass for converting palettized or
sparse torch models.

### Convert models with sparse weights

If your source model weights have lots of zeros, then specify the `pass_pipeline` argument as
follows:

```python
import coremltools as ct

model_with_sparse_weights = ct.convert(
    mlmodel,
    pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
    minimum_deployment_target=ct.target.iOS17,
)
```

During the conversion process, an additional [graph pass](graph-passes-intro) will be run, which
will convert the weight values below a certain low threshold (`default=1e-3`) to exact zeros, and
then use a sparse representation to store the weights, thereby saving space.

### Convert models with palettized weights

If your source model weights are palettized; that is, clustered and can only take on a few discrete
values, then you can save space by invoking the following pass during conversion:

```python
import coremltools as ct

model_with_lut_weights = ct.convert(
    mlmodel,
    pass_pipeline=ct.PassPipeline.DEFAULT_PALETTIZATION,
    minimum_deployment_target=ct.target.macOS13,
)
```

This will invoke an additional [graph pass](graph-passes-intro) that will automatically detect
whether the weights have repeated values. If they do, and if the overall weight tensor has at most
256 or less unique values (which means it can be represented with an 8-bit or
less [lookup table](https://en.wikipedia.org/wiki/Lookup_table)), it will use a more compact
representation using lookup tables (LUTs) to store them.

## Use compression info embedded in torch models

The previous method with pass pipeline has the following limitations:

- With more compression granularity supported in Core ML Tools 8, it’s hard to infer the
  compression info accurately. For example, if the model is compressed by grouped channel-wise
  palettization, we need group size and channel axis to correctly divide weights into groups. In addition,
  the fp16 numerical instability could lead to incorrect inferred n-bit values.
- You always need to specify the `pass_pipeline` parameter. If you forget, the conversion
  will still succeed, but the converted Core ML model will be a float model without compressed MIL ops.

In Core ML tools 8, the `ct.optimize.torch.*` APIs have been updated so that they
now embed the info about the compression that was performed on the torch model.
The info is then read by the `ct.convert` API automatically, so as a user you do not have to do
anything special: just use the APIs to compress your torch models and convert them. There is no need 
to specify any pass pipelines.

This approach is more general and allows for handling compression modes, such as per-grouped-channel
palettization, per-block 4-bit quantization, etc., which are not natively expressible in torch models.

Here is the protocol of the compression information embedded into torch models:

- The compression information is stored by torch’s registered buffers.
- The registered buffer names should be in the format of `_COREML_/<parameter_name>/<field_name>`.
  For example, `'dense1._COREML_/weight/n_bits'`, `'dense1._COREML_/weight/quantization_scale'`, etc.
- Available `field_name` options are `compression_type`, `quantization_n_bits`, `quantization_scale`,
  `zero_point`, `lut`, and `palettization_scale`.
  Details about the meaning of each field can be found
  in [Compression info protocol](#compression-info-protocol). The following subsections
  also go over a few examples.

### Convert models with sparse weights

If the model is produced by `ct.optimize.torch.pruning.*` APIs, you can convert it directly.

```python
import coremltools as ct

model_with_sparse_weights = ct.convert(
    pruned_torch_model,
    minimum_deployment_target=ct.target.iOS18,
)
```

If the model is produced by other tools, you can update the torch model, prior to tracing and
conversion, by embedding the compression information via `register_buffer` for each module that has been
compressed, as shown below:

```python
# Assume the model has two conv layers and both of them got pruned.
pruned_torch_model.register_buffer("_COREML_/metadata_version", torch.tensor(1))
pruned_torch_model.conv_1.register_buffer("_COREML_/weight/compression_type", torch.tensor([1]))
pruned_torch_model.conv_2.register_buffer("_COREML_/weight/compression_type", torch.tensor([1]))

model_with_sparse_weights = ct.convert(
    pruned_torch_model,
    minimum_deployment_target=ct.target.iOS18,
)
```

### Convert models with palettized weights

If the model is produced by `ct.optimize.torch`, you can convert it directly.

```python
import coremltools as ct

model_with_lut_weights = ct.convert(
    palettized_torch_model,
    minimum_deployment_target=ct.target.iOS18,
)
```

If the model is produced by other tools, you can embed compression information by registering buffers
for each layer, as shown below.

```python
# Assume the model has two conv layers, and both of them got palettized, while the first layer has per-channel-scale.
palettized_torch_model.register_buffer("_COREML_/metadata_version", torch.tensor(1))
palettized_torch_model.conv_1.register_buffer("_COREML_/weight/compression_type", torch.tensor([2]))
palettized_torch_model.conv_1.register_buffer("_COREML_/weight/lut", torch.tensor(lut1))
palettized_torch_model.conv_1.register_buffer(
    "_COREML_/weight/palettization_scale", torch.from_numpy(scale_1)
)
palettized_torch_model.conv_2.register_buffer("_COREML_/weight/compression_type", torch.tensor([2]))
palettized_torch_model.conv_2.register_buffer("_COREML_/weight/lut", torch.tensor(lut2))

model_with_lut_weights = ct.convert(
    palettized_torch_model,
    minimum_deployment_target=ct.target.iOS18,
)
```

### Convert PyTorch models with quantized weights and activations

If you use PyTorch’s built-in quantization tool, the produced compressed models store quantized
weights or activations using `qint` or `quint` data types, and additional quantization ops are used.
This is picked up automatically by the conversion process, which then automatically uses linear
quantized storage format to store weights.

However, for now, the PyTorch built-in quantization tool only supports per-tensor and per-channel
quantization. For more compression schemas / granularity, you need to use `ct.optimize.torch.*` APIs
or other third-party tools.

If you use `ct.optimize.torch.*` APIs to quantize your torch model, the compression information will be
automatically stored in the compressed torch model, which can be converted seamlessly.

If you use other tools, you can inject compression information by registering buffers for each layer, as
shown below.

```python
# Assume the model has two linear layers and only the second linear layer's weight and bias got quantized.
quantized_torch_model.register_buffer("_COREML_/metadata_version", torch.tensor(1))
quantized_torch_model.linear_2.register_buffer("_COREML_/weight/compression_type",
                                               torch.tensor([3]))
quantized_torch_model.linear_2.register_buffer("_COREML_/weight/quantization_n_bits",
                                               torch.tensor(n_bits))
quantized_torch_model.linear_2.register_buffer(
    "_COREML_/weight/quantization_scale", torch.from_numpy(weight_scale)
)
quantized_torch_model.linear_2.register_buffer("_COREML_/bias/compression_type", torch.tensor([3]))
quantized_torch_model.linear_2.register_buffer("_COREML_/bias/quantization_n_bits",
                                               torch.tensor(n_bits))
quantized_torch_model.linear_2.register_buffer(
    "_COREML_/bias/quantization_scale", torch.from_numpy(bias_scale)
)

model_with_quantized_weights = ct.convert(
    quantized_torch_model,
    minimum_deployment_target=ct.target.iOS18,
)
```

### Convert models with jointly compressed weights

Starting from Core ML Tools 8, the `ct.optimize.torch.*` APIs allows for joint compression schemes,
such as `pruning + quantization` and `pruning + palettization`.
The produced models can also be converted seamlessly.

Similarly, if you use other tools which quantized the non-zero entries of the sparse weight, you can
inject compression information by registering buffers for each layer, as shown below.

```python
# Assume the model has two convs layers and both got jointly compressed by pruning + quantization.
joint_compressed_model.register_buffer("_COREML_/metadata_version", torch.tensor(1))
joint_compressed_model.conv_1.register_buffer("_COREML_/weight/compression_type",
                                              torch.tensor([1, 3]))
joint_compressed_model.conv_1.register_buffer("_COREML_/weight/quantization_n_bits",
                                              torch.tensor(n_bits))
joint_compressed_model.conv_1.register_buffer(
    "_COREML_/weight/quantization_scale", torch.from_numpy(scale_1)
)
joint_compressed_model.conv_1.register_buffer("_COREML_/weight/zero_point",
                                              torch.from_numpy(zero_point_1))
joint_compressed_model.conv_2.register_buffer("_COREML_/weight/compression_type",
                                              torch.tensor([1, 3]))
joint_compressed_model.conv_2.register_buffer("_COREML_/weight/quantization_n_bits",
                                              torch.tensor(n_bits))
joint_compressed_model.conv_2.register_buffer(
    "_COREML_/weight/quantization_scale", torch.from_numpy(scale_2)
)
joint_compressed_model.conv_2.register_buffer("_COREML_/weight/zero_point",
                                              torch.from_numpy(zero_point_2))

joint_compressed_mlmodel = ct.convert(
    joint_compressed_model,
    minimum_deployment_target=ct.target.iOS18,
)
```

For joint pruning + palettization, it’s similar:

```python
# Assume the model has two convs layers and both got jointly compressed by pruning + palettization.
joint_compressed_model.register_buffer("_COREML_/metadata_version", torch.tensor(1))
joint_compressed_model.conv_1.register_buffer("_COREML_/weight/compression_type",
                                              torch.tensor([1, 2]))
joint_compressed_model.conv_1.register_buffer("_COREML_/weight/lut", torch.tensor(lut_1_params.lut))
joint_compressed_model.conv_2.register_buffer("_COREML_/weight/compression_type",
                                              torch.tensor([1, 2]))
joint_compressed_model.conv_2.register_buffer("_COREML_/weight/lut", torch.tensor(lut_2_params.lut))

joint_compressed_mlmodel = ct.convert(
    joint_compressed_model,
    minimum_deployment_target=ct.target.iOS18,
)
```

### Compression info protocol

As described in [Use compression info embedded in torch models](#use-compression-info-embedded-in-torch-models),
you can register buffers in the compressed torch model to specify compression related information.
All the information stored in this newly introduced protocol will be fully honored by Core ML Tools
when constructing the compressed MIL ops.

This section records the detailed meaning of each field that you could specify, as well as the
versioning change. The version is an integer that can be specified by `_COREML_/metadata_version`
at the model level.

#### Version 1

**Quantization Related Fields**

| Key	                 | Original Type	 | Optional	 | Sample Value / Shape	    | Additional Notes	                                                                                         |
|----------------------|----------------|-----------|--------------------------|-----------------------------------------------------------------------------------------------------------|
| quantization_n_bits	 | int	           | No	       | `torch.tensor(4)`	       | 	                                                                                                         |
| quantization_scale	  | tensor	        | No	       | `torch.Size([3072, 1])`	 | Rank of the scale need to match weight’s rank. The block_size on each dim will be inferred by the shape.	 |
| zero_point	          | tensor	        | Yes	      | `torch.Size([3072, 1])`	 | 	 Same shape as `quantization_scale`.                                                                     |

Details about the shape of the scale and zero-point can be found in the `iOS18` `constexpr_blockwise_shift_scale` op.

**Palettization Related Fields**

| Key	                 | Original Type	 | Optional	 | Sample Value / Shape	        | Additional Notes	                                                                                                                                       |
|----------------------|----------------|-----------|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| lut	                 | tensor	        | No	       | `torch.Size([2, 2, 16, 1])`	 | Rank of the lut should be weight’s rank + 2. The shape can be used to infer palettization configurations like group size, axis, cluster dim and n-bit.	 |
| palettization_scale	 | tensor	        | Yes	      | `torch.Size([3072, 1])`	     | 	 Rank of the scale need to match weight’s rank.                                                                                                        |

Details about the shape of the lookup table can be found in the `iOS18` `constexpr_lut_to_dense` op.

**Compression Type Field**

| Key	              | Original Type	 | Optional	 | Sample Value / Shape	   | Additional Notes	                                                                                                                                               |
|-------------------|----------------|-----------|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| compression_type	 | list of int	   | No	       | `torch.tensor([1, 3])`	 | List of compression types applied to the parameter in the order in which they were applied. `1` means pruning, `2` means palettization, `3` means quantization	 |

All these fields can be combined to cover all Core ML Tools 8 new compression schemas, as described
below.

**Quantization**

| Feature	                 | Required Fields	                                                                                       |
|--------------------------|--------------------------------------------------------------------------------------------------------|
|
 blockwise quantization	 | n_bits<br>quantization_scale<br>zero_point (for affine & unsigned symmetric)<br>compression_type = [3]	 |
|
 4-bit quantization	      | n_bits<br>quantization_scale<br>zero_point (for affine & unsigned symmetric)<br>compression_type = [3]	 |

**Palettization**

| Feature	                       | Required Fields	                                                                                          |
|--------------------------------|-----------------------------------------------------------------------------------------------------------|
|
 LUT with per-channel scales	   | palettization_scale<br>compression_type = [2]	                                                            |
|
 vector palettization	          | cluster_dim<br>compression_type = [2]	                                                                    |
|
 grouped-channel palettization	 | group_size<br>group_axis<br>compression_type = [2]	                                                       |
|
 LUT with 8-bit values	         | n_bits<br>quantization_scale<br>zero_point (for affine & unsigned symmetric)<br>compression_type = [2, 3]	 |

**Joint compression**

| Feature	                                                                                                        | Required Fields	                                                                                          |
|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
|
 pruning + quantization (sparse weight where non-zero values are 8-bit quantized)	                                | compression_type = [1, 3]<br>n_bits<br>quantization_scale<br>zero_point (for affine and unsigned symmetric)	 |
|
 pruning + palettization (sparse weight where non-zero values are n-bit palettized)	                              | compression_type = [1, 2]<br>Same as required keys for palettization	                                     |
|
 pruning + palettization + quantization (sparse weight where non-zero values are n-bit palettized with int8 LUT)	 | compression_type = [1, 2, 3]<br>Same as required keys for LUT with 8-bit values	                           |
