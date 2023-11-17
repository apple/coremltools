```{eval-rst}
.. index:: 
    single: optimization; workflow
```

# Optimization Workflow

Core ML Tools offers two ways to incorporate model compression into your workflow:

- [_Post-training data-free compression_](optimization-workflow.md#post-training-compression). Use this faster method with a Core ML model, either created or converted from another model. You can quickly try different techniques with different configurations.
- [_Training-time compression_](optimization-workflow.md#training-time-compression). Use this method with a PyTorch model while in training. It lets you fine-tune with data for higher accuracy.

Since model compression is a lossy operation, in both cases you should evaluate the model on the validation data set and compare it with the uncompressed model to ascertain the loss in accuracy and see if that is acceptable. To inspect the properties of the weights in the model, see [_Getting Weight Metadata_](optimization-workflow.md#getting-weight-metadata) in this section.

```{eval-rst}
.. index:: 
    single: post-training compression
    single: compression; post-training
```

## Post-Training Compression

Start with a Core ML model that uses `float` precision weights. It can be a model you created and trained yourself, or a pre-trained model, in either case converted by the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method. This workflow is also commonly referred to as  _post-training quantization_.

```{figure} images/compress_coreml_models.jpg
:alt: Data-Free Model compression workflow
:align: center
:width: 300px
:class: imgnoborder

Post-training (data-free) model compression workflow.
```

### Compression Steps

To directly compress the Core ML model, follow these steps: 

1. Load the `.mlpackage` model in memory using the [`models.MLModel()`](https://apple.github.io/coremltools/source/coremltools.models.html#module-coremltools.models.model) API in Core ML Tools.
2. Use one of the methods available in `optimize.coreml.*` that takes the loaded model, iterates over its weights one-by-one, compresses them, and then returns an updated model with these compressed weights. For available options, see [optimize.coreml API overview](optimizecoreml-api-overview).
3. Save the model to disk.

### Benefits and Drawbacks

Benefits of this post-training approach include:

- It is data free; that is, you do not need to have access to the training data.
- It is fast, as no fine-tuning is involved. Compressing very large models still takes time, as an optimization algorithm runs when compressing the weights. However, compared to fine-tuning with data, this method is much faster.

These benefits make this approach more flexible. You can try different techniques with different configurations relatively quickly, to compare accuracy and performance impacts (such as size and latency). The results can guide you to decide whether to deploy the compressed model resulting from this process, or to explore training-time compression to improve accuracy, which involves fine-tuning with data.

A drawback of this approach is that you may observe a steep decline in accuracy as compared to the amount of compression, depending on the model and the task. That is, for lower amounts of compression, you may observe the accuracy to be close to that of the uncompressed model, but as you choose configurations to increase the compression, the accuracy may decline very sharply.

```{eval-rst}
.. index:: 
    single: training-time compression
    single: compression; training-time
```

## Training-Time Compression

Train your model in a compression-aware fashion, or start from a pre-trained `float` precision model  and fine-tune it with training data. The resulting model typically has higher accuracy because compression is introduced gradually and in a differentiable manner to allow the weights to readjust to the new constraints imposed on them. 

```{figure} images/in-training-optimize-workflow.png
:alt: Training-time optimization workflow
:align: center
:width: 400px
:class: imgnoborder

Training-time optimization workflow.
```


### Incorporate Compression Into Training Workflow

To incorporate optimizations compatible with Core ML into your training workflow, follow these steps: 

1. Before fine-tuning, make sure that the PyTorch model can be exported to Core ML using the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method. This will ensure that you are starting with a model that is compatible with Core ML. 
2. Use one of the compression methods from the `optimize.torch.*` module to update your PyTorch model (see [Training-Time Compression](https://apple.github.io/coremltools/source/coremltools.optimize.html#training-time-compression)).
3. Fine-tune the model using the original training code for a few iterations. Then export the model to Core ML and measure performance to decide whether or not you are satisfied with it. 
4. Fine-tune the model fully, using the data and the original PyTorch training code, until the desired level of accuracy is achieved. The weights will get adjustments to allow for compression.
5. Convert the model to Core ML using the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) method. You may need to pass an additional flag during conversion to indicate to the converter which compression scheme was employed during fine-tuning, so that it can pick the appropriate weight representation. For details, see [optimize.torch API overview](optimizetorch-api-overview).

### Benefits and Drawbacks

The benefit of this approach, compared to post-training compression, is that you can get a more favorable trade-off in accuracy vs. compression amount. The drawbacks include requiring training data and spending more time in fine-tuning, the costs of which may be very high, especially for a relatively larger model. 



```{figure} images/trade-off_curve.jpg
:alt: Accuracy trade-off with compression
:align: center
:width: 400px
:class: imgnoborder

A hypothetical "accuracy-compression amount" trade-off curve. The dotted curve corresponds to data free compression, while the solid curve represents training time compression.
```


### How the APIs Work

The `ct.optimize.torch` APIs provide model optimization algorithms for [Pruning](pruning), [Palettization](palettization), and [Quantization Aware Training (QAT)](quantization-aware-training). The APIs accept a PyTorch module as input and return a transformed PyTorch module as output. Optimization is incorporated seamlessly into the training process so that you can use the transformed module in your training loop.

The transformations are accomplished either by wrapping the submodules with wrapper layers (as in the case of quantization-aware training), or by installing [hooks](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks) on the submodules (for example, when mutating masks for parameter pruning). You can also combine these techniques.

For more information about using the APIs, see [optimize.torch API overview](optimizetorch-api-overview).


```{eval-rst}
.. index:: 
    single: weight metadata
```


## Getting Weight Metadata

The  [`get_weights_metadata()`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.get_weights_metadata) utility provides a convenient way to inspect the properties of the weights in the model. You may want to use it to identify weights that are larger than a certain size, or have a sparsity greater than a certain percentage, or have a certain number of unique elements. 

For example, if you want to compare the weights of a model showing unexpected results with the weights of a model with predictable results, you can use `get_weights_metadata()` to get a list of all the weights with their metadata. Use the `weight_threshold` parameter to set which weights are returned. A weight is included in the resulting dictionary only if its total number of elements are greater than `weight_threshold`. 

The metadata returned by the utility also offers information about the child ops the weight feeds into. The data returned by the API can then be used to customize the optimization of the model via the `ct.optimize.coreml` API. 

### Using the Metadata 

The  [`get_weights_metadata()`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.get_weights_metadata) utility returns the weights metadata as a  [CoreMLWeightMetaData](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.CoreMLWeightMetaData) dictionary mapping each weight’s name to its metadata. These results are useful when constructing the [`coremltools.optimize.coreml.OptimizationConfig`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig) API.

For example, with the [OptimizationConfig](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig) class you have fine-grain control over applying different optimization configurations to different weights by directly setting `op_type_configs` and `op_name_configs` or using [`set_op_name`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig.set_op_name) and [`set_op_type`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig.set_op_type). When using [`set_op_name`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig.set_op_name), you need to know the name for the `const` op that produces the weight. The  `get_weights_metadata()` utility provides the weight name and the corresponding weight numpy data, along with meta information such as the sparsity. 

To represent weights with sparse representation whose elements are greater than 3000 and the percentage of 0s are greater than 75%, you can call `get_weights_metadata()` and iterate through all weights to filter out only those weights. The results might be two weights (such as `["weight_1", "weight_4"])`, and you can then use  [`set_op_name`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig.set_op_name) or directly set  `op_name_configs` accordingly.

The utility is also useful for the [`set_op_type()`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.config.html#coremltools.optimize.coreml.OptimizationConfig.set_op_type) API, which lets you set the optimization config for all weights that feed into the same type of op, such as `conv`. The  `get_weights_metadata()` utility provides information about what op the weight feeds into (the `child_ops` attribute of the CoreMLWeightMetaData), so that you can utilize the information to validate the intent of optimizing those weights.


### Using a Weight Threshold

The following example includes weights with sizes greater than 2048. The code loads the `MobileNetV2.mlpackage` converted in [Getting Started](introductory-quickstart) (from the previously trained [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2) model, which is based on the [tensorflow.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)). It uses [`get_weights_metadata()`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.get_weights_metadata), which specifies the size threshold (`2048`) in the `weight_threshold` parameter. A weight tensor is included in the resulting dictionary only if its total number of elements are greater than `weight_threshold`:

```python
import coremltools as ct

mlmodel = ct.models.MLModel("MobileNetV2.mlpackage")
weight_metadata_dict = ct.optimize.coreml.get_weights_metadata(
    mlmodel, weight_threshold=2048
)
```

You can then include weights in the dictionary based on their metadata:

```python
# get the weight names with size > 25600
large_weights = []
for k, v in weight_metadata_dict.items():
    if v.val.size >= 25600:
        large_weights.append(k)

# get the weight names with sparsity >= 50%
sparse_weights = []
for k, v in weight_metadata_dict.items():
    if v.sparsity >= 0.5:
        sparse_weights.append(k)

# get the weight names with unique elements <= 16
palettized_weights = []
for k, v in weight_metadata_dict.items():
    if v.unique_values <= 16:
        palettized_weights.append(k)
```

```python
# Access to each weight metadata (type of CoreMLWeightMetaData).
weight_meta_data = weight_metadata_dict["mobilenetv2_1_00_224_block_1_project_BN_FusedBatchNormV3_nchw_weight_0_to_fp16"]
print("---")
print("numpy array data for the weight: ")
print(weight_meta_data.val) # This is the numpy array data for the weight
print("---")
print("Sparsity of the weight: ", weight_meta_data.sparsity) # The sparsity of the weight
print("---")
print("Number of unique values in the weight: ", weight_meta_data.unique_values) # Number of unique values in the weight
print("---")

# Access to the child ops this weight feeds into.
# In this example, the weight feeds into only one op (conv_5),
# as the result, weight_meta_data.child_ops is a list of length 1.
child_op = weight_meta_data.child_ops[0]
print("name of the child op: ", child_op.name) # name of the child op
print("---")
print("op type of the child op: ", child_op.op_type) # op type of the child op
print("---")
print("dictionary: ", child_op.params_name_mapping) # this dictionary shows the input parameters of the child op and their corresponding ops' names
print("---")
```


The output from the above example would be:

```
---
numpy array data for the weight: 
[[[[ 0.4062  ]]

  [[-0.04984 ]]

  [[-0.4714  ]]

  ...

  [[-0.4846  ]]

  [[ 0.4563  ]]

  [[ 0.001579]]]


 [[[ 0.746   ]]

  [[-0.208   ]]

  [[ 0.3416  ]]

  ...

  [[ 0.1913  ]]

  [[-0.0241  ]]

  [[-0.02843 ]]]


 [[[ 0.094   ]]

  [[ 0.778   ]]

  [[ 0.315   ]]

  ...

  [[-0.5137  ]]

  [[-0.3682  ]]

  [[-1.2     ]]]


 ...


 [[[-0.1445  ]]

  [[ 0.6494  ]]

  [[ 0.703   ]]

  ...

  [[ 0.1827  ]]

  [[ 0.2012  ]]

  [[-0.391   ]]]


 [[[-0.04352 ]]

  [[ 1.12    ]]

  [[ 0.5605  ]]

  ...

  [[ 0.4165  ]]

  [[-0.4824  ]]

  [[ 0.5176  ]]]


 [[[-0.2922  ]]

  [[ 0.3481  ]]

  [[-0.2795  ]]

  ...

  [[-0.3572  ]]

  [[ 0.2546  ]]

  [[ 0.0194  ]]]]
---
Sparsity of the weight:  0.0
---
Number of unique values in the weight:  2077
---
name of the child op:  mobilenetv2_1_00_224_block_1_project_BN_FusedBatchNormV3_nchw_cast_fp16
---
op type of the child op:  conv
---
dictionary:  OrderedDict([('weight', 'mobilenetv2_1_00_224_block_1_project_BN_FusedBatchNormV3_nchw_weight_0_to_fp16')])
---
```






