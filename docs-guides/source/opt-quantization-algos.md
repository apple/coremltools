# Quantization Algorithms

Following are the various algorithms available in Core ML Tools to quantize a model:

- Post Training (data free) weight quantization
- Post Training (data calibration) activation quantization
- GPTQ algorithm for weight quantization (post training data calibration)
- Fine tuning based algorithm for quantizing weight and/or activations

## Post Training (data free) weight quantization

This algorithm uses the round-to-nearest (RTN) method to quantize the model weights. This is the fastest approach for quantizing the model weights.

Suggested API(s):
- [coremltools.optimize.torch.quantization.PostTrainingQuantizer](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.PostTrainingQuantizer) (For Torch models)
- [coremltools.optimize.coreml.linear_quantize_weights](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.post_training_quantization.html#coremltools.optimize.coreml.linear_quantize_weights) (For Core ML models)

## Post Training (data calibration) activation quantization
This algorithm quantizes the activations using a calibration dataset. The data is passed through the model and the range of values that the activations take is estimated. This estimate is then used to compute the scale / zero-point using the RTN method for quantizing the activations.
Suggested API(s):
- [coremltools.optimize.torch.quantization.LinearQuantizer](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer) (For Torch models)
- [coremltools.optimize.coreml.experimental.linear_quantize_activations](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.quantization.html#coremltools.optimize.coreml.experimental.linear_quantize_activations)(For Core ML models)

## GPTQ algorithm for weight quantization (post training data calibration)
This algorithm is based on the paper [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323). The layerwise compression paradigm helps to compress a sequential model layer-by-layer by minimizing the quantization error while quantizing the weights. Each layer is compressed by minimizing the [L2 norm](https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm) of the difference between the layer's original outputs and the outputs obtained by using the compressed weights. The outputs are computed using a few samples of training data (around 128 samples are usually sufficient). Once a layer is compressed, the layer's outputs are used as inputs for compressing the next layer.

Suggested API(s):
- [coremltools.optimize.torch.layerwise_compression.LayerwiseCompressor](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.layerwise_compression.LayerwiseCompressor)

## Fine tuning based algorithm for quantizing weight and/or activations

This algorithm is also known as quantization-aware training (QAT) as described in the paper [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/pdf/1712.05877.pdf). QAT allows for quantizing both weights and activations. The model is fine tuned upon simulating quantization on the weights and / or activations to recover the accuracy lost upon quantizing the model. 

Suggested API(s):
- [coremltools.optimize.torch.quantization.LinearQuantizer](https://apple.github.io/coremltools/source/coremltools.optimize.torch.quantization.html#coremltools.optimize.torch.quantization.LinearQuantizer)

```{admonition} PyTorch quantization APIs
You can use PyTorch's quantization APIs directly, and then convert the model to Core ML. However, the converted model performance may not be optimal. The PyTorch API default settings (symmetric asymmetric quantization modes and which ops are quantized) are not optimal for the Core ML stack and Apple hardware. If you use the Core ML Tools `coremltools.optimize.torch` APIs, as described in this section, the correct default settings are applied automatically.
```

### Impact on accuracy with different modes

Weight-only post-training quantization (PTQ) using 8-bit precision with per-channel granularity, typically preserves the accuracy of the model well. If we further compress the model to 4-bit precision, per-block granularity is required to retain the accuracy. 

If the former method (weight-only PTQ 8-bit per-channel) does not work well, then calibration data-based techniques such as GPTQ or fine tuning based methods such as QAT can be explored. 

For activation quantization, the calibration data based approach should work well in most cases. However, if a loss of accuracy is seen, quantizing activation using QAT can be used to recover the lost accuracy.


### Accuracy data

| Model Name                                                                                                                                                            | Config                                                                                                                                                  | Optimization Workflow | Compression Ratio | Accuracy     |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|-------------------|--------------|
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileNetV2Alpha1.mlpackage.zip)                                                   | Float16                                                                                                                                                 | n/a                   | 1.0               | 71.86        |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/quantized/MobileNetV2Alpha1WeightOnlySymmetricQuantized.mlpackage.zip) | Weight-only                                                                                                                                             | PTQ         | 1.92              | 71.78        |
| [MobileNetv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/MobileNetV2Alpha1SymmetricPerChannel.mlpackage.zip)          | [Weight & activation](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/MobileNetV2Alpha1SymmetricPerChannel.yaml) | QAT         | 1.92              | 71.66 ± 0.04 |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/ResNet50.mlpackage.zip)                                                                   | Float16                                                                                                                                                 | n/a                   | 1.0               | 76.14        |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/quantized/ResNet50WeightOnlySymmetricQuantized.mlpackage.zip)                 | Weight-only                                                                                                                                             | PTQ        | 1.99              | 76.10        |
| [ResNet50](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/ResNet50SymmetricPerChannel.mlpackage.zip)                          | [Weight & activation](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/ResNet50SymmetricPerChannel.yaml)          | QAT         | 1.98              | 76.80 ± 0.05 |
| [MobileViTv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/uncompressed/MobileViTV2Alpha1.mlpackage.zip)                                                   | Float16                                                                                                                                                 | n/a                   | 1.0               | 78.09        |
| [MobileViTv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/post_training_compressed/quantized/MobileViTV2Alpha1WeightOnlySymmetricQuantized.mlpackage.zip) | Weight-only                                                                                                                                             | PTQ        | 1.92              | 77.66        |
| [MobileViTv2-1.0](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/MobileViTV2Alpha1SymmetricPerChannel.mlpackage.zip)          | [Weight & activation](https://ml-assets.apple.com/coreml/quantized_models/training_time_compressed/quantized/MobileViTV2Alpha1SymmetricPerChannel.yaml) | QAT         | 1.89              | 76.89 ± 0.07 |



