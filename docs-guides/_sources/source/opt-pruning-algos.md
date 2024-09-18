Pruning Algorithms
==================

Core ML Tools offers a variety of algorithms for pruning model weights. 

## Post-Training Pruning

In this algorithm, a subset of the elements in a weight matrix are zeroed out, in a data free or **zero-shot** manner.
This can be done either by setting all elements less than a given threshold to zero or by sorting
the elements and setting the lowest values, up to a certain percentile (known as *target sparsity*), to zero.

Unless the weights in your Core ML model are known to have a lot of zeros, 
using data-free pruning is typically going to lead to a large accuracy loss for 
any meaningful level of sparsity. Therefore, this algorithm is typically used
to experiment with different patterns (like `n:m` sparsity, block structured sparsity, etc.) 
and levels of sparsity. This experimentation lets you see the impact on size reduction and latency, 
and then use the results as a guiding factor to find a config that you can then use to 
prune either using calibration data or fine-tuning. 

Supported API(s):

- [`coremltools.optimize.coreml.prune_weights`](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.prune_weights)
- [`coreltools.optimize.torch.pruning.MagnitudePruner`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.MagnitudePruner) with [`ConstantSparsityScheduler`](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.pruning_scheduler.ConstantSparsityScheduler) where `begin_step` is set to `0`


## SparseGPT

This algorithm implements [SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://arxiv.org/pdf/2301.00774),
which prunes the model weights layer by layer, in a **one-shot** manner, by minimizing the L2 norm 
of the difference between the original activations and activations obtained on pruning the weights 
of the layer. This is knows as the L2 reconstruction loss. The activations are computed using a 
few samples of training data. Once a layer is compressed, the output activations produced by the 
pruned layer are used as inputs to the next layer which is set up for pruning. The minimization problem 
can be solved exactly using the Hessian of the L2 reconstruction loss. 

It also supports jointly pruning and quantizing the model weights. 

Typically, 128 samples are sufficient for applying this algorithm.
In practice, it works well, much better than data-free pruning, for large transformer-based architectures.

Supported API:

- [LayerwiseCompressor](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#sparsegpt)

## MagnitudePruner

This algorithm is based on [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/pdf/1710.01878). 
It extends the idea to different kinds of structured sparsity modes, in addition to unstructured sparsity. In order to achieve the 
desired sparsity, it sorts a module’s weight matrix by the magnitude of its elements, and sets all elements less than 
a threshold to zero. It maintains the full weight matrix and a mask of `0`s and `1`s. The mask multiplied with the
full weight matrix gets used during forward pass. Since this multiplication is a differentiable operation, weights
still get trained during gradient descent, which allows the model to adapt to presence of `0`s. 

This algorithm typically works best for higher levels of sparsity and structured sparsity modes. 
Depending on the model and the sparsity chosen, more fine-tuning may be required.

Supported API:

- [MagnitudePruner](https://apple.github.io/coremltools/source/coremltools.optimize.torch.pruning.html#coremltools.optimize.torch.pruning.MagnitudePruner)

## Choosing an algorithm

Use data free post-training pruning to quickly generate models with different levles of sparsity and 
different modes of structured sparsity and study the impact on model size and latency. Once a 
config meets the requirements, use `SparseGPT` and a few samples of training data to prune the model with
the chosen config. If the pruned model achieves the required accuracy, you are done. Otherwise, use
`MagnitudePruner` to prune the model in a compression aware manner. 

## Accuracy Benchmarks

### Methodology
    
In the tables below, we provide accuracy benchmarks on several models, pruned using `coremltools.optimize` APIs.

See [Palettization Performance](opt-palettization-perf.md) page to learn more about how the benchmarked models were generated.

All evaluations were performed on the final compressed (or uncompressed) CoreML models, using the validation subset of the dataset linked in [Model Info](opt-palettization-perf.md#model-info).
The training time pruned models were trained for three trials, starting from the same pre-trained weights, 
and using a different ordering of data during training for each trial. For these models, 
we report the mean accuracy across the three trials, along with the standard deviation.

### Results

| Model Name                                                                                                                                                                      | Config                                                                                                                                                      | Optimization Algorithm | Compression Ratio | Accuracy                  |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|-------------------|---------------------------|
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileNetV2Alpha1.mlpackage.zip)                                                             | Float16                                                                                                                                                     | n/a                    | 1.0               | 71.86                     |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV2Alpha1UnstructuredSparsity50.mlpackage.zip)                    | Unstructured Sparsity 50%                                                                                                                                   | MagnitudePruner        | 1.37              | 71.83 ± 0.01              |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV2Alpha1UnstructuredSparsity75.mlpackage.zip)                    | Unstructured Sparsity 75%                                                                                                                                   | MagnitudePruner        | 1.73              | 69.47 ± 0.07              |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileNetV3Small.mlpackage.zip)                                                            | Float16                                                                                                                                                     | n/a                    | 1.0               | 67.58                     |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV3SmallUnstructuredSparsity50.mlpackage.zip)                   | [Unstructured Sparsity 50%](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV3SmallUnstructuredSparsity50.yaml)| MagnitudePruner        | 1.73              | 66.55 ± 0.03              |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV3SmallUnstructuredSparsity75.mlpackage.zip)                   | [Unstructured Sparsity 75%](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV3SmallUnstructuredSparsity75.yaml)| MagnitudePruner        | 3.06              | 60.52 ± 0.06              |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/ResNet50.mlpackage.zip)                                                                             | Float16                                                                                                                                                     | n/a                    | 1.0               | 76.14                     |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/ResNet50UnstructuredSparsity50.mlpackage.zip)                                    | [Unstructured Sparsity 50%](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/ResNet50UnstructuredSparsity50.yaml)        | MagnitudePruner        | 1.77              | 73.64 ± 0.04              |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/ResNet50UnstructuredSparsity75.mlpackage.zip)                                    | [Unstructured Sparsity 75%](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/ResNet50UnstructuredSparsity75.yaml)        | MagnitudePruner        | 3.17              | 73.40 ± 0.08              |


**Note**: The trained and compressed models and the `coremltools.optimize.torch` config files used for compression can be downloaded by clicking the respective links embedded in the model and config names.
