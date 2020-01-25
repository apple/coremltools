[![Build Status](https://travis-ci.com/apple/coremltools.svg?branch=master)](#)
[![PyPI Release](https://img.shields.io/pypi/v/coremltools.svg)](#)
[![Python Versions](https://img.shields.io/pypi/pyversions/coremltools.svg)](#)

Core ML Community Tools
=======================

Core ML community tools contains all supporting tools for Core ML model
conversion, editing and validation. This includes deep learning frameworks like
TensorFlow, Keras, Caffe as well as classical machine learning frameworks like
LIBSVB, scikit-learn, and XGBoost.

To get the latest version of coremltools:

```shell
pip install --upgrade coremltools
```

For the latest changes please see the [release notes](https://github.com/apple/coremltools/releases/).

# Table of Contents

* [Neural network conversion](#Neural-network-conversion)
* [Core ML specification](#Core-ML-specification)
* [coremltools user guide and examples](#user-guide-and-examples)
* [Installation from Source](#Installation)


## Neural Network Conversion

[Link](docs/NeuralNetworkGuide.md) to the detailed NN conversion guide.

There are several `converters` available to translate neural networks trained
in various frameworks into the Core ML model format.  Following formats can be
converted to the Core ML `.mlmodel` format through the coremltools python
package (this repo):

- Caffe V1 (`.prototxt`, `.caffemodel` format)
- Keras API (2.2+) (`.h5` format)
- TensorFlow 1 (1.13+) (`.pb` frozen graph def format)
- TensorFlow 2 (`.h5` and `SavedModel` formats)

In addition, there are two more neural network converters build on top of `coremltools`:
- [onnx-coreml](https://github.com/onnx/onnx-coreml): to convert `.onnx` model format. Several frameworks such as PyTorch, MXNet, CaffeV2 etc
provide native export to the ONNX format.
- [tfcoreml](https://github.com/tf-coreml/tf-coreml): to convert TensorFlow models. For producing Core ML models targeting iOS 13 or later,
tfcoreml defers to the TensorFlow converter implemented inside coremltools.
For iOS 12 or earlier, the code path is different and lives entirely in the [tfcoreml](https://github.com/tf-coreml/tf-coreml) package.  

To get an overview on how to use the converters and features such as
post-training quantization using coremltools, please see the [neural network
guide](docs/NeuralNetworkGuide.md).  

## Core ML Specification

- Core ML specification is fully described in a set of protobuf files.
They are all located in the folder `mlmodel/format/`
- For an overview of the Core ML framework API, see [here](https://developer.apple.com/documentation/coreml).
- To find the list of model types supported by Core ML, see [this](https://github.com/apple/coremltools/blob/1fcac9eb087e20bcc91b41bc938112fa91b4e5a8/mlmodel/format/Model.proto#L229)
portion of the `model.proto` file.
- To find the list of neural network layer types supported see [this](https://github.com/apple/coremltools/blob/1fcac9eb087e20bcc91b41bc938112fa91b4e5a8/mlmodel/format/NeuralNetwork.proto#L472)
portion of the `NeuralNetwork.proto` file.
- Auto-generated documentation for all the protobuf files can be found at this [link](https://apple.github.io/coremltools/coremlspecification/)


## User Guide and Examples

- [API documentation](https://apple.github.io/coremltools)
- [Updatable models](examples/updatable_models)
- [Neural network inference examples](examples/neural_network_inference)
- [Neural network guide](docs/NeuralNetworkGuide.md)
- [Miscellaneous How-to code snippets](docs/APIExamples.md)

## Installation

We recommend using virtualenv to use, install, or build coremltools. Be
sure to install virtualenv using your system pip.

```shell
pip install virtualenv
```

The method for installing `coremltools` follows the
[standard python package installation steps](https://packaging.python.org/installing/).
To create a Python virtual environment called `pythonenv` follow these steps:

```shell
# Create a folder for your virtualenv
mkdir mlvirtualenv
cd mlvirtualenv

# Create a Python virtual environment for your Core ML project
virtualenv pythonenv
```

To activate your new virtual environment and install `coremltools` in this
environment, follow these steps:

```
# Active your virtual environment
source pythonenv/bin/activate


# Install coremltools in the new virtual environment, pythonenv
(pythonenv) pip install -U coremltools
```

The package [documentation](https://apple.github.io/coremltools) contains
more details on how to use coremltools.
