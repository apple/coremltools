# What Is Core ML Tools?

The [coremltools](https://github.com/apple/coremltools "apple/coremltools") Python package is the primary way to convert third-party models to Core ML. [Core ML](https://developer.apple.com/documentation/coreml "Core ML Framework") is an Apple framework to integrate machine learning models into your app. 

Use Core ML Tools to convert models from third-party training libraries such as [TensorFlow](https://www.tensorflow.org "TensorFlow") and [PyTorch](https://pytorch.org "PyTorch") to the [Core ML model package format](https://developer.apple.com/documentation/coreml/core_ml_api/updating_a_model_file_to_a_model_package). You can then use Core ML to integrate the models into your app.

```{figure} images/introduction-coremltools.png
:alt: Core ML Tools overview
:align: center
:class: imgnoborder

Convert a third-party model to a Core ML model package file.
```

With Core ML Tools you can:

- Convert trained models from libraries and frameworks such as [TensorFlow](https://www.tensorflow.org "TensorFlow") and [PyTorch](https://pytorch.org "PyTorch") to the Core ML model package format.
- Read, write, and optimize Core ML models to use less storage space, reduce power consumption, and reduce latency during inference.
- Verify creation and conversion by making predictions using Core ML in macOS.

Core ML provides a unified representation for all models. Your app uses Core ML APIs and user data to make predictions, and to fine-tune models, all on the user’s device. Running a model strictly on the user’s device removes any need for a network connection, which helps keep the user’s data private and your app responsive.

Core ML optimizes on-device performance by leveraging the CPU, GPU, and Apple Neural Engine (ANE) while minimizing its memory footprint and power consumption.


## Additional Resources

- The [Machine Learning](https://developer.apple.com/machine-learning/ "Machine Learning") page provides educational material, tutorials, guides, and documentation for Apple developers.
- The [ML & Vision session videos](https://developer.apple.com/videos/frameworks/machine-learning-and-vision "ML & Vision session videos from the World Wide Developer Conference") from the World Wide Developer Conference are a great place to start if you are new to machine learning technology and Core ML.
- The [Core ML documentation](https://developer.apple.com/documentation/coreml "Core ML documentation") walks you through the first steps in developing an app with a machine learning model.
- Try out `coremltools` in your browser with Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ContinuumIO/coreml-demo/HEAD)


## Supported Libraries and Frameworks

You can convert trained models from the following libraries and frameworks to Core ML:

| Model Family      | Supported Packages |
| ----------- | ----------- |
| Neural Networks      | [TensorFlow 1 (1.14.0+)](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf), [TensorFlow 2 (2.1.0+)](https://www.tensorflow.org), [PyTorch (1.13.0+)](https://pytorch.org) |
| Tree Ensembles   | [XGboost (1.1.0)](https://xgboost.readthedocs.io/en/latest/index.html), [scikit-learn (0.18.1)](https://scikit-learn.org/stable/)        |
| Generalized Linear Models   | [scikit-learn (0.18.1)](https://scikit-learn.org/stable/)        |
| Support Vector Machines   | [LIBSVM (3.22)](https://pypi.org/project/libsvm/), [scikit-learn (0.18.1)](https://scikit-learn.org/stable/)        |
| Pipelines (pre- and post-processing)   | [scikit-learn (0.18.1)](https://scikit-learn.org/stable/)        |




