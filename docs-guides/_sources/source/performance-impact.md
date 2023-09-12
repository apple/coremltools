```{eval-rst}
.. index:: 
    single: optimization; accuracy and performance
```

# Accuracy and Performance

This section provides a few concrete [examples](performance-impact.md#examples) of the tradeoff between accuracy and performance, as described in [Optimization Workflow](optimization-workflow). The accuracy of the compressed model depends not only on the type of model and the task for which it is trained, but also on the compression ratio. This section helps you understand the impact of compressing models, not just on model size, but also on latency and runtime memory consumption.

```{note}
Results for only a handful of models are presented in this section. These can vary greatly based on the tasks and model types.
```

## Compression Ratios

For measuring the compression ratio, we use a `float16` precision model as the baseline, since this is the default precision setting for [mlprograms](typed-execution). For each compression technique, the expected compressions ratios are as follows:

### Weight Pruning

The main parameter that controls the compression factor is the amount of sparsity. A 50% sparse model implies that half of the weight values have been set to 0. Using sparse representation, only the non-zero elements will be stored, thereby saving half the memory. 

However, the locations of the non-zero values also need to be stored in order to fully reconstruct the weight tensor. Therefore, such a model will have a compression ratio slightly less than 2. For a 75% sparse model, the ratio will be slightly less than 4, and so on. In practice, the compression ratio will typically be less, as certain weight tensors (such as ones that are smaller in size) may be skipped from compression to preserve more model accuracy. 

### Weight Palettization

The `n_bits` parameter controls model size with weight Palettization. `n_bits=8,4,2` will correspond to the maximum possible compression ratios of `2,4,8` respectively. Since additional memory needs to be allocated to store the look-up table, the compression ratio will be less than the maximum possible. Similar to pruning, if certain layers that are sensitive to precision are skipped from being palletized the overall compression ratio may be lower. 

### Linear 8-Bit Quantization

Since linear 8-bit quantization always uses 8 bits, the compression ratio is close to 2 with this scheme. In practice, it would be slightly less than 2, as a bit of additional memory is required to store `per-channel` scale parameters. 

## Effect on Latency and Runtime Memory

While the primary advantage of compressing weights is model size reduction, improvements in latency and runtime memory usage _may_ occur for compressed models as compared to their uncompressed versions, as seen in the  [examples](performance-impact.md#examples). This is due to runtime changes and enhancements made in Core ML starting from iOS 17, macOS 14, watchOS 10, and tvOS 17.

The runtime performance is dependent on a variety of factors, including model architecture, hardware generation, the dominant compute unit the ops of the model are dispatched to, parametric configurations of the ops, and so on. Since performance characteristics are expected to change and generally improve with each software update, we recommend that you perform measurements on the device(s) of interest using the [Xcode Performance tab and/or Core ML Instruments](https://developer.apple.com/videos/play/wwdc2022/10027/) for your particular model deployment and app code. 

To understand the impact of compressing models on latency and runtime memory consumption, consider the case when the activations are in `float` precision, and only the weights are compressed (which is the most common case, unless both activations and weights are compressed to 8-bit linear). In this scenario, during the time of computation, the weights need to be _decompressed_ so that they can be brought back in the same dense `float` representation as the activations, and multiplied with them. Compressing the weights does not, technically,  reduce the amount of computation to be done at runtime and hence is not expected to reduce the time taken for a prediction. In fact, this decompression was explicitly done during the model _load_ time in the `iOS16/macOS13/watchOS9/tvOS16` version of the Core ML framework. Therefore in this case, the prediction would happen exactly as it would for the corresponding uncompressed model, keeping the runtime memory usage and latency unchanged. 

With changes made from `iOS17/macOS13` onwards, the weight _decompression_ may instead happen at the _prediction_ time. The decision of _when_ to decompress the weights depends on several factors including the op types, the hardware generation, the backend hardware unit (CPU, GPU or Neural Engine) that the ops run on, the backend compiler optimizations etc. 

For a given weight, if a decision is made to decompress at _prediction_ time instead of _load_ time, weights will be kept in their reduced size form until later — until they need to be loaded for the compute. This means that fewer number of bytes will be required to be moved from the main memory to the compute engine, thereby reducing the memory movement time. However, it also means that they need to be decompressed on the fly while doing the compute, thereby increasing the prediction time. Depending on how these two effects add up, the overall prediction time may decrease or may even increase (while runtime memory usage is lower).

Specifically for models that are "weight memory bound" rather than "compute bound" or "activation memory bound" (meaning that the bottleneck is in the reading of weight data), decompressing weights "just in time" can be very beneficial if the decompression operation can be done efficiently. Since the decision to decompress weights at load or prediction time is now dynamic and influenced by several factors, it is mainly guided by the principle of providing the best tradeoff between runtime memory and prediction speed, as possible on a given hardware unit. Currently, in most cases Neural Engine is the compute backend that opts into just in time weight decompression, hence offering the possibility of lower runtime memory and latencies. However, there are variations to be expected based on the specific type of the compression scheme, for this and impact on latency for activation quantized models, learn more in the sections [Pruning Overview](pruning-overview), [Palettization Overview](palettization-overview) and [Quantization Overview](quantization-overview). 

## Examples

In the tables below, we provide benchmarks on several models, compressed using various techniques in `coremltools.optmize`, to illustrate the trade-off between accuracy, model size and runtime latency. 

The training time compressed models were obtained by fine-tuning the `float32` PyTorch models with weights initialized from the checkpoints linked in the [Model Info](performance-impact.md#model-info) table, and using methods from `coremltools.optimize.torch` to perform compression. The datasets used for fine-tuning the models are also linked in the same table, along with the accuracy metric being reported. We used fine-tuning recipes which are commonly used in literature for the task at hand, and standard data augmentations. 

Similarly, the post training compressed models were obtained by compressing the converted `float16` Core ML models, with pre-trained weights, using methods from `coremltools.optimize.coreml` module. Models were palettized using the `kmeans` mode and quantized using the `linear_symmetric` mode. 

All evaluations were performed on the final compressed (or uncompressed) CoreML models, using the validation subset of the dataset linked in [Model Info](performance-impact.md#model-info). The training time compressed models were trained for three trials, starting from the same pre-trained weights, and using a different ordering of data during training for each trial. For these models, we report the **mean** accuracy across the three trials, along with the **standard deviation**. 

 The trained and compressed models and the `coremltools.optimize.torch` config files used for compression can be downloaded by clicking the respective links embedded in the model and config names.

The latency numbers were captured using the Xcode **Performance** tab, using the `median` statistic. Compute unit selection is `all` unless otherwise noted. The latency numbers are sensitive to the device state, and may vary depending on the device state and build versions. 

- Device: iPhone 14 Pro
- iOS build: iOS17 Developer Beta 1 
- Xcode : Xcode 15 Beta 1 

### Model Info

| Model Name | Task | Pre-trained Weights | Dataset | Accuracy Metric |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| MobileNetv2-1.0 | Image Classification | [Torchvision](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth) | [ImageNet](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html) | Top-1 Accuracy (%) |
| MobileNetv3-small | Image Classification | [Torchvision](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth) | [ImageNet](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html) | Top-1 Accuracy (%) |
| ResNet50 | Image Classification | [Torchvision](https://download.pytorch.org/models/resnet50-0676ba61.pth) | [ImageNet](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html) | Top-1 Accuracy (%) |
| MobileViTv2-1.0 | Image Classification | cvnets | [ImageNet](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html) | Top-1 Accuracy (%) |
| CenterNet (ResNet34 backbone) | Object Detection | Torchvision [backbone](https://download.pytorch.org/models/resnet34-b627a593.pth) | [MS-COCO](https://pytorch.org/vision/main/generated/torchvision.datasets.CocoDetection.html#torchvision.datasets.CocoDetection) | mAP |


### Pruning

| Model Name | Config | Optimization Workflow | Compression Ratio | Accuracy | Latency in ms (per batch) |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileNetV2Alpha1.mlpackage.zip) | Float16 | n/a | 1.0 | 71.86 | 0.52 |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV2Alpha1UnstructuredSparsity50.mlpackage.zip) | Unstructured Sparsity 50% | Training Time | 1.37 | 71.83 ± 0.01 | 0.45 |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV2Alpha1UnstructuredSparsity75.mlpackage.zip) | Unstructured Sparsity 75% | Training Time | 1.73 | 69.47 ± 0.07 | 0.45 |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileNetV3Small.mlpackage.zip) | Float16 | n/a | 1.0 | 67.58 | 0.20 |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV3SmallUnstructuredSparsity50.mlpackage.zip) | [Unstructured Sparsity 50%](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV3SmallUnstructuredSparsity50.yaml) | Training Time | 1.73 | 66.55 ± 0.03 | 0.18 |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV3SmallUnstructuredSparsity75.mlpackage.zip) | [Unstructured Sparsity 75%](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV3SmallUnstructuredSparsity75.yaml) | Training Time | 3.06 | 60.52 ± 0.06 | 0.18 |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/ResNet50.mlpackage.zip) | Float16 | n/a | 1.0 | 76.14 | 1.42 |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/ResNet50UnstructuredSparsity50.mlpackage.zip) | [Unstructured Sparsity 50%](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/ResNet50UnstructuredSparsity50.yaml) | Training Time | 1.77 | 73.64 ± 0.04 | 1.39 |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/ResNet50UnstructuredSparsity75.mlpackage.zip) | [Unstructured Sparsity 75%](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/ResNet50UnstructuredSparsity75.yaml) | Training Time | 3.17 | 73.40 ± 0.08 | 1.19 |

### Palettization

| Model Name | Config | Optimization Workflow | Compression Ratio | Accuracy | Latency in ms (per batch) |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileNetV2Alpha1.mlpackage.zip) | Float16 | n/a | 1.0 | 71.86 | 0.52 |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/MobileNetV2Alpha1ScalarPalettization2Bit.mlpackage.zip) | [2 bit](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/MobileNetV2Alpha1ScalarPalettization2Bit.yaml) | Training Time | 5.92 | 68.81 ± 0.04 | 0.49 |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/MobileNetV2Alpha1ScalarPalettization4Bit.mlpackage.zip) | [4 bit](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/MobileNetV2Alpha1ScalarPalettization4Bit.yaml) | Training Time | 3.38 | 70.60 ± 0.08 | 0.49 |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/palettized/MobileNetV2Alpha1ScalarPalettization6Bit.mlpackage.zip) | 6 bit | Post Training | 2.54 | 70.89 | 0.47 |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/palettized/MobileNetV2Alpha1ScalarPalettization8Bit.mlpackage.zip) | 8 bit | Post Training | 1.97 | 71.80 | 0.48 |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileNetV3Small.mlpackage.zip) | Float16 | n/a | 1.0 | 67.58 | 0.20 |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/MobileNetV3SmallScalarPalettization2Bit.mlpackage.zip) | [2 bit](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/MobileNetV3SmallScalarPalettization2Bit.yaml) | Training Time | 5.82 | 59.82 ± 0.98 | 0.22 |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/MobileNetV3SmallScalarPalettization4Bit.mlpackage.zip) | [4 bit](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/MobileNetV3SmallScalarPalettization4Bit.yaml) | Training Time | 3.47 | 67.23 ± 0.04 | 0.2 |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/palettized/MobileNetV3SmallScalarPalettization6Bit.mlpackage.zip) | 6 bit | Post Training | 2.6 | 65.46 | 0.22 |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/palettized/MobileNetV3SmallScalarPalettization8Bit.mlpackage.zip) | 8 bit | Post Training | 1.93 | 67.44 | 0.21 |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/ResNet50.mlpackage.zip) | Float16 | n/a | 1.0 | 76.14 | 1.42 |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/ResNet50ScalarPalettization2Bit.mlpackage.zip) | [2 bit](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/ResNet50ScalarPalettization2Bit.yaml) | Training Time | 7.63 | 75.47 ± 0.05 | 1.39 |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/ResNet50ScalarPalettization4Bit.mlpackage.zip) | [4 bit](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/ResNet50ScalarPalettization4Bit.yaml) | Training Time | 3.9 | 76.63 ± 0.01 | 1.37 |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/palettized/ResNet50ScalarPalettization6Bit.mlpackage.zip) | 6 bit | Post Training | 2.65 | 75.68 | 1.31 |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/palettized/ResNet50ScalarPalettization8Bit.mlpackage.zip) | 8 bit | Post Training | 1.99 | 76.05 | 1.34 |
| [CenterNet (ResNet34 backbone)](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/CenterNetResNet34.mlpackage.zip) | Float16 | n/a | 1.0 | 29.0 | 7.48 |
| [CenterNet (ResNet34 backbone)](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/CenterNetResNet34ScalarPalettization2Bit.mlpackage.zip) | [2 bit](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/CenterNetResNet34ScalarPalettization2Bit.yaml) | Training Time | 7.71 | 25.66 ± 0.03 | 6.71 |
| [CenterNet (ResNet34 backbone)](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/CenterNetResNet34ScalarPalettization4Bit.mlpackage.zip) | [4 bit](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/palettized/CenterNetResNet34ScalarPalettization4Bit.yaml) | Training Time | 3.94 | 28.14 ± 0.11 | 6.91 |
| [CenterNet (ResNet34 backbone)](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/palettized/CenterNetResNet34ScalarPalettization6Bit.mlpackage.zip) | 6 bit | Post Training | 2.65 | 28.27 | 7.01 |
| [CenterNet (ResNet34 backbone)](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/palettized/CenterNetResNet34ScalarPalettization8Bit.mlpackage.zip) | 8 bit | Post Training | 2.0 | 28.75 | 7.45 |

### Linear 8-Bit Quantization

| Model Name | Config | Optimization Workflow | Compression Ratio | Accuracy | Latency in ms (per batch) |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileNetV2Alpha1.mlpackage.zip) | Float16 | n/a | 1.0 | 71.86 | 0.52 |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/quantized/MobileNetV2Alpha1WeightOnlySymmetricQuantized.mlpackage.zip) | Weight-only | Post Training | 1.92 | 71.78 | 0.47 |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/MobileNetV2Alpha1SymmetricPerChannel.mlpackage.zip) | [Weight & activation](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/MobileNetV2Alpha1SymmetricPerChannel.yaml) | Training Time | 1.92 | 71.66 ± 0.04 | 0.28 |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/ResNet50.mlpackage.zip) | Float16 | n/a | 1.0 | 76.14 | 1.42 |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/quantized/ResNet50WeightOnlySymmetricQuantized.mlpackage.zip) | Weight-only | Post Training | 1.99 | 76.10 | 1.36 |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/ResNet50SymmetricPerChannel.mlpackage.zip) | [Weight & activation](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/ResNet50SymmetricPerChannel.yaml) | Training Time | 1.98 | 76.80 ± 0.05 | 1.47 |
| [MobileViTv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileViTV2Alpha1.mlpackage.zip) | Float16 | n/a | 1.0 | 78.09 | 1.48 |
| [MobileViTv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/quantized/MobileViTV2Alpha1WeightOnlySymmetricQuantized.mlpackage.zip) | Weight-only | Post Training | 1.92 | 77.66 | 1.53 |
| [MobileViTv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/MobileViTV2Alpha1SymmetricPerChannel.mlpackage.zip) | [Weight & activation](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/MobileViTV2Alpha1SymmetricPerChannel.yaml) | Training Time | 1.89 | 76.89 ± 0.07 | 1.35 |


