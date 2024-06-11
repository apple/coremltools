Performance
============

- Compared to storing weights in a dense format with float16 precision, sparse representation saves about two bytes of storage for every zero value. Model size goes down linearly with the level of sparsity introduced.

- For a model that is primarily running on the Neural Engine, sparsity typically helps in improving latency. This is made possible by two factors:
  - It reduces the size of weights to be loaded at inference time, which can speed up inference of networks that are weight memory bound. 
  - When a string of consecutive 0s are encountered, the Neural Engine may also be able to skip computations, thereby reducing the amount of computation. This can be achieved by choosing higher levels of unstructured sparsity (e.g. 75% or higher) or block structured sparsity, where zeros occur in blocks of 2 or its multiples. 
  - Note that longer fine-tuning with more data is usually needed to preserve accuracy with larger block sizes and higher levels of sparsity 

- Models with a lot of linear ops can benefit from inference speed-ups on CPU on newer hardware generations, when using ``n:m`` sparsity. Here, out of a block of ``m`` elements, ``n`` are 0s. ``m`` should be a factor of 16 (such as ``3:4``, ``7:8``, ``14:16``, and so on) and ``n/m >= 0.5``.

- Pruning can be applied jointly with quantization and palettization to achieve additional latency and memory savings, over and above those achieved by applying those techniques individually.   


## Performance Benchmarks 

### Methodology

The latency numbers were captured using the Xcode **Performance** tab, using the `median` statistic. Compute unit selection is `all` unless otherwise noted. The latency numbers are sensitive to the device state, and may vary depending on the device state and build versions. 

- Device: iPhone 14 Pro (A16), unless otherwise mentioned
- iOS build: iOS17 
- Xcode : Xcode 15

For more details on base models and compression methodology, please refer to docs [here](opt-palettization-perf.md).

### Results

| Model Name                                                                                                                                                                      | Config                                                                                                                                                      | Optimization Algorithm | Compression Ratio | Latency in ms (per batch) |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|-------------------|---------------------------|
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileNetV2Alpha1.mlpackage.zip)                                                             | Float16                                                                                                                                                     | n/a                    | 1.0               | 0.48                      |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV2Alpha1UnstructuredSparsity50.mlpackage.zip)                    | Unstructured Sparsity 50%                                                                                                                                   | MagnitudePruner        | 1.37              | 0.46                      |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV2Alpha1UnstructuredSparsity75.mlpackage.zip)                    | Unstructured Sparsity 75%                                                                                                                                   | MagnitudePruner        | 1.73              | 0.46                      |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileNetV3Small.mlpackage.zip)                                                            | Float16                                                                                                                                                     | n/a                    | 1.0               | 0.13                      |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV3SmallUnstructuredSparsity50.mlpackage.zip)                   | [Unstructured Sparsity 50%](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV3SmallUnstructuredSparsity50.yaml)| MagnitudePruner        | 1.73              | 0.12                      |
| [MobileNetv3-small](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV3SmallUnstructuredSparsity75.mlpackage.zip)                   | [Unstructured Sparsity 75%](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/MobileNetV3SmallUnstructuredSparsity75.yaml)| MagnitudePruner        | 3.06              | 0.12                      |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/ResNet50.mlpackage.zip)                                                                             | Float16                                                                                                                                                     | n/a                    | 1.0               | 1.52                      |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/ResNet50UnstructuredSparsity50.mlpackage.zip)                                    | [Unstructured Sparsity 50%](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/ResNet50UnstructuredSparsity50.yaml)        | MagnitudePruner        | 1.77              | 1.46                      |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/ResNet50UnstructuredSparsity75.mlpackage.zip)                                    | [Unstructured Sparsity 75%](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/pruned/ResNet50UnstructuredSparsity75.yaml)        | MagnitudePruner        | 3.17              | 1.28                      |


**Note**: The trained and compressed models and the `coremltools.optimize.torch` config files used for compression can be downloaded by clicking the respective links embedded in the model and config names.
