[![Build Status](https://img.shields.io/gitlab/pipeline/zach_nation/coremltools/master)](https://gitlab.com/zach_nation/coremltools/-/pipelines?page=1&scope=branches&ref=master)
[![PyPI Release](https://img.shields.io/pypi/v/coremltools.svg)](#)
[![Python Versions](https://img.shields.io/pypi/pyversions/coremltools.svg)](#)

[Core ML Tools](https://coremltools.readme.io/docs)
=======================

Use *coremltools* to convert machine learning models from third-party libraries to the Core ML format. The Python package contains the supporting tools for converting models from training libraries such as the following:

* [TensorFlow 1.x](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf)
* [TensorFlow 2.x](https://www.tensorflow.org/api_docs)
* [PyTorch](https://pytorch.org/)
* [TensorFlow's Keras APIs](https://keras.io/)
* Non-neural network frameworks:
	* [scikit-learn](https://scikit-learn.org/stable/)
	* [XGBoost](https://xgboost.readthedocs.io/en/latest/)
	* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)

With coremltools, you can do the following:

* Convert trained models to the Core ML format.
* Read, write, and optimize Core ML models.
* Verify conversion/creation (on macOS) by making predictions using Core ML.

After conversion, you can integrate the Core ML models with your app using Xcode.

## Version 5

The coremltools 5 package offers several performance improvements over previous versions, including the following new features: 

* [Core ML model package](https://coremltools.readme.io/docs/new-in-coremltools#save-a-core-ml-model-package): A new model container format that separates the model into components and offers more flexible metadata editing and better source control.
* [ML program](https://coremltools.readme.io/docs/ml-programs): A new model type that represents computation as programmatic instructions, offers more control over the precision of its intermediate tensors and better performance. 

To install coremltools 5, use the following command:

```shell
pip install coremltools==5.0b1
```


## Core ML

[Core ML](https://developer.apple.com/documentation/coreml) is an Apple framework to integrate machine learning models into your app. Core ML provides a unified representation for all models. Your app uses Core ML APIs and user data to make predictions, and to fine-tune models, all on the user’s device. Core ML optimizes on-device performance by leveraging the CPU, GPU, and Neural Engine while minimizing its memory footprint and power consumption. Running a model strictly on the user’s device removes any need for a network connection, which helps keep the user’s data private and your app responsive.

## Resources

To install coremltools, see the [“Installation“ page](https://coremltools.readme.io/docs/installation). For more information, see the following:

* [Release Notes](https://github.com/apple/coremltools/releases/) 
* [Guides and examples](https://coremltools.readme.io/) 
* [API Reference](https://apple.github.io/coremltools/index.html)
* [Core ML Specification](https://mlmodel.readme.io/)
* [Building from Source](BUILDING.md)
* [Contribution Guidelines](CONTRIBUTING.md) 


