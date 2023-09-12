```{eval-rst}
.. index:: 
    single: optimization; workflow
```

# Optimization Workflow

Core ML Tools offers two ways to incorporate model compression into your workflow:

- [_Post-training data-free compression_](optimization-workflow.md#post-training-compression). Use this faster method with a Core ML model, either created or converted from another model. You can quickly try different techniques with different configurations.
- [_Training-time compression_](optimization-workflow.md#training-time-compression). Use this method with a PyTorch model while in training. It lets you fine-tune with data for higher accuracy.

Since model compression is a lossy operation, in both cases you should evaluate the model on the validation data set and compare it with the uncompressed model to ascertain the loss in accuracy and see if that is acceptable. 

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



