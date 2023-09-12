```{eval-rst}
.. index:: 
    single: optimization; overview
    single: compression; types of
```

# Overview

Optimization is a rapidly evolving field of machine learning. There are several ways to optimize a deep learning model. For example, you can distill a large model into a smaller model, start from scratch and train a smaller efficient architecture, customize the model architecture for a specific task and hardware, and so on. 

This section focuses specifically on techniques that achieve a smaller model by compressing its weights, and optionally its activations. In particular, this section covers those optimizations that enable you to take a model with `float` precision weights and modify the weights into smaller-sized approximate and lossy representations, allowing you to trade off task accuracy with overall model size and on-device performance.

## Types of Compression

There are lots of different ways to compress weights and activations of a neural network model, and for each kind of compression type, there are different algorithms to achieve it. This is an extremely active area of research and development. Most of these methods result in weights that can be represented in one of the three formats that Core ML supports. For overviews of these formats and examples, see the following:

- Sparse weights: [Pruning](pruning)
- Palettized weights: [Palettization](palettization)  
- 8 bit linear quantization (weights and/or activations): [Linear 8-Bit Quantization](quantization-aware-training)

## When to Compress

You can either directly compress a Core ML model, or compress a model in the source framework during training and then convert. While the former is quicker and can happen without needing data, the latter can preserve accuracy better by fine-tuning with data. To find out more about the two workflows see [Optimization Workflow](optimization-workflow).

## How to Compress

You can compress the model in your source framework and then convert, or use the recommended workflow: `coremltools.optimize.coreml.*` APIs for data free compression or `coremltools.optimize.torch.*` APIs to compress with data and fine tuning.

To learn more on how to use these APIs, see [API Overview](unified-conversion-api).

## Learn More about Accuracy and Performance

The accuracy of the compressed model depends not only on the type of model and the task for which it is trained, but also on the amount of the compression ratio. To learn more about the impact of model compression methods, see [Accuracy and Performance](performance-impact). 

## Software Availability of Optimizations

| OS versions | Optimizations | Core ML Model Type | coremltools API |
| ----------- | ------------- | ------------------ | --------------- |
| iOS15 or lower, macOS12 or lower | palettization, 8 bit quantization | `neuralnetwork` | `ct.models.neural_networks.quantization_utils.* `|
| iOS16, macOS13 | palettization, sparsity, 8 bit quantization | `mlprogram` | `ct.optimize.*` |
| iOS17, macOS14 | iOS16/macOS13 optimizations + 8bit activation quantization, runtime memory & latency improvements | `mlprogram` | `ct.optimize.*` |

```{tip}
You may also find it useful to view the presentation in [WWDC 2023](https://developer.apple.com/videos/play/wwdc2023/10047/), which provides an overview of the optimizations.
```

