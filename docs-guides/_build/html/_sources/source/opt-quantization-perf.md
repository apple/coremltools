Performance
============

Since quantization reduces the size of each weight value, the amount of data to be moved is reduced during prediction. 
This can lead to benefits with memory-bottlenecked models.

Quantizing the activations may further ease this memory pressure and may lead to more gains when compared to weight-only 
quantization. However, with activation quantization, you may observe a considerable slowdown in inference for the 
compute units (CPU and sometimes GPU) that employ load-time weight decompression, since activations are not known at 
load time, and they need to be decompressed at runtime, slowing down the inference. Therefore, it is recommended to use
activation quantization only when your model is fully or mostly running on the Neural Engine (NE).

In newer hardware with A17 Pro or M4 chips, such as iPhone 15 Pro, there is increased throughput possible for int8-int8 
compute on Neural Engine, compared to previous versions. This means that activation and weight quantization for networks running on
Neural Engine can give even more latency gains. This can be seen in the [table below](#results) (e.g. 
The ResNet50 model with `W8A8` mode runs considerably faster than its `W16A16` equivalent). 

For the `per-block` weight quantization option added in `iOS18/macOS15`, which is especially useful when employing 
quantization to `4-bits`, one can expect to see great runtime memory gains, as well as latency gains depending on the model,
when it is running on the GPU. On the other hand, if the model is running on the NE, it is recommended to use the 
`per-channel` scales option.

## Performance Benchmarks:

### Methodology:

The latency numbers were captured using the Xcode **Performance** tab, using the `median` statistic. Compute unit 
selection is `all` unless otherwise noted. The latency numbers are sensitive to the device state, and may vary depending
on the device state and build versions. 

- Device: iPhone 14 Pro (A16), unless otherwise mentioned
- iOS build: iOS 17 
- Xcode : Xcode 15

For more details on base models and compression methodology, please refer to docs [here](opt-palettization-perf.md).

### Model Info

| Model Name                    | Task                 | Pre-trained Weights                                                                | Dataset                                                                                                                         | Accuracy Metric    |
|-------------------------------|----------------------|------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------|
| MobileNetv2-1.0               | Image Classification | [Torchvision](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth)       | [ImageNet](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html)                                        | Top-1 Accuracy (%) |
| ResNet50                      | Image Classification | [Torchvision](https://download.pytorch.org/models/resnet50-0676ba61.pth)           | [ImageNet](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html)                                        | Top-1 Accuracy (%) |
| MobileViTv2-1.0               | Image Classification | cvnets                                                                             | [ImageNet](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html)                                        | Top-1 Accuracy (%) |

### Results 

| Model Name                                                                                                                                                                | Config                                                                                                       | Optimization Workflow | Compression Ratio | Accuracy       | Latency in ms (per batch) on iPhone 14 Pro | Latency in ms (per batch) on iPhone 15 Pro |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------------|-------------------|----------------|--------------------------------------------|--------------------------------------------|
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileNetV2Alpha1.mlpackage.zip)                                                       | Float16                                                                                                      | n/a                   | 1.0               | 71.86          | 0.48                                       | 0.49                                       |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/quantized/MobileNetV2Alpha1WeightOnlySymmetricQuantized.mlpackage.zip)     | Weight-only                                                                                                  | Post Training         | 1.92              | 71.78          | 0.45                                       | 0.44                                       |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/MobileNetV2Alpha1SymmetricPerChannel.mlpackage.zip)              | [Weight & activation](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/MobileNetV2Alpha1SymmetricPerChannel.yaml) | Training Time         | 1.92              | 71.66 ± 0.04   | 0.27                                       | 0.20                                       |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/ResNet50.mlpackage.zip)                                                                      | Float16                                                                                                      | n/a                   | 1.0               | 76.14          | 1.52                                       | 1.38                                       |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/quantized/ResNet50WeightOnlySymmetricQuantized.mlpackage.zip)                     | Weight-only                                                                                                  | Post Training         | 1.99              | 76.10          | 1.49                                       | 1.50                                       |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/ResNet50SymmetricPerChannel.mlpackage.zip)                             | [Weight & activation](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/ResNet50SymmetricPerChannel.yaml)       | Training Time         | 1.98              | 76.80 ± 0.05   | 0.94                                       | 0.77                                       |
| [MobileViTv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileViTV2Alpha1.mlpackage.zip)                                                      | Float16                                                                                                      | n/a                   | 1.0               | 78.09          | 1.38                                       | 1.36                                       |
| [MobileViTv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/quantized/MobileViTV2Alpha1WeightOnlySymmetricQuantized.mlpackage.zip)     | Weight-only                                                                                                  | Post Training         | 1.92              | 77.66          | 1.43                                       | 1.37                                       |
| [MobileViTv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/MobileViTV2Alpha1SymmetricPerChannel.mlpackage.zip)              | [Weight & activation](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/MobileViTV2Alpha1SymmetricPerChannel.yaml) | Training Time         | 1.89              | 76.89 ± 0.07   | 1.18                                       | 1.03                                       |

**Note**: The trained and compressed models and the `coremltools.optimize.torch` config files used for compression can be downloaded by clicking the respective links embedded in the model and config names.
