[![Build Status](https://img.shields.io/gitlab/pipeline/coremltools1/coremltools/main)](https://gitlab.com/coremltools1/coremltools/-/pipelines?page=1&scope=branches&ref=main)
[![PyPI Release](https://img.shields.io/pypi/v/coremltools.svg)](#)
[![Python Versions](https://img.shields.io/pypi/pyversions/coremltools.svg)](#)

[Core ML Tools](https://apple.github.io/coremltools/docs-guides/source/overview-coremltools.html)
=======================

![Core ML Tools logo](docs/logo.png)

Use [Core ML Tools](https://apple.github.io/coremltools/docs-guides/source/overview-coremltools.html) (*coremltools*) to convert machine learning models from third-party libraries to the Core ML format. This Python package contains the supporting tools for converting models from training libraries such as the following:

* [TensorFlow 1.x](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf)
* [TensorFlow 2.x](https://www.tensorflow.org/api_docs)
* [PyTorch](https://pytorch.org/)
* Non-neural network frameworks:
	* [scikit-learn](https://scikit-learn.org/stable/)
	* [XGBoost](https://xgboost.readthedocs.io/en/latest/)
	* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)

With coremltools, you can:

* Convert trained models to the Core ML format.
* Read, write, and optimize Core ML models.
* Verify conversion/creation (on macOS) by making predictions using Core ML.

After conversion, you can integrate the Core ML models with your app using Xcode.

## Install 8.0 Beta
The [coremltools version 8 beta 2](https://github.com/apple/coremltools/releases/tag/8.0b2) is now out. To install, run the following command in your terminal:
```shell
pip install coremltools==8.0b2
```


## Install Version 7.2
To install the latest non-beta version, run the following command in your terminal:
```shell
pip install -U coremltools
```

## Core ML

[Core ML](https://developer.apple.com/documentation/coreml) is an Apple framework to integrate machine learning models into your app. Core ML provides a unified representation for all models. Your app uses Core ML APIs and user data to make predictions, and to fine-tune models, all on the user’s device. Core ML optimizes on-device performance by leveraging the CPU, GPU, and Neural Engine while minimizing its memory footprint and power consumption. Running a model strictly on the user’s device removes any need for a network connection, which helps keep the user’s data private and your app responsive.

## Resources

To install coremltools, see [Installing Core ML Tools](https://apple.github.io/coremltools/docs-guides/source/installing-coremltools.html). For more information, see the following:

* [Release Notes](https://github.com/apple/coremltools/releases/)
* [Guide and examples](https://apple.github.io/coremltools/docs-guides/index.html)
* [API Reference](https://apple.github.io/coremltools/index.html)
* [Core ML Specification](https://apple.github.io/coremltools/mlmodel/index.html)
* [Building from Source](BUILDING.md)
* [Contribution Guidelines](CONTRIBUTING.md)
