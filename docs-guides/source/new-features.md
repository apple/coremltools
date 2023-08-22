# New Features

The following sections describe new features and improvements in the most recent versions of Core ML Tools.

## New in Core ML Tools 7

The [coremltools 7](https://github.com/apple/coremltools) package now includes more APIs for optimizing the models to use less storage space, reduce power consumption, and reduce latency during inference. Key optimization techniques include pruning, quantization, and palettization. 

You can either directly compress a Core ML model, or compress a model in the source framework during training and then convert. While the former is quicker and can happen without needing data, the latter can preserve accuracy better by fine-tuning with data. For details, see [Optimizing Models](optimizing-models).

For a full list of changes, see [Release Notes](#release-notes).

To install Core ML Tools version 7.0b2: 

```shell
pip install coremltools==7.0b2
```

## Previous Versions

The [coremltools 6](https://github.com/apple/coremltools/releases/tag/6.3) package offers the following features to optimize the model conversion process:

- Model compression utilities; see [Compressing Neural Network Weights](quantization-neural-network).
- Float 16 input/output types including image. See [Image Input and Output](image-inputs).

For a full list of changes from `coremltools` 5.2, see [Release Notes](#release-notes).


## Release Notes

Learn about changes to the `coremltools` package from the following release notes:

- [Release Notes (newest)](https://github.com/apple/coremltools/releases/)

For information about previous releases, see the following:

- [Release Notes for coremltools 6.3](https://github.com/apple/coremltools/releases/tag/6.3)
- [Release Notes for coremltools 5.2](https://github.com/apple/coremltools/releases/tag/5.2)
- [Release Notes for coremltools 5.0](https://github.com/apple/coremltools/releases/tag/5.0)
- [Release Notes for coremltools 4.1](https://github.com/apple/coremltools/releases/tag/4.1)
- [Release Notes for coremltools 4.0](https://github.com/apple/coremltools/releases/tag/4.0)
- [Release Notes for coremltools 3.4](https://github.com/apple/coremltools/releases/tag/3.4)
- [All release notes](https://github.com/apple/coremltools/releases)

