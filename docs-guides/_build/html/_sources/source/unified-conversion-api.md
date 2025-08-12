# Core ML Tools API Overview

Core ML Tools is the [`coremltools`](https://apple.github.io/coremltools/index.html) Python package for macOS (10.13+) and Linux. It includes the [Unified Conversion API](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#module-coremltools.converters._converters_entry) for converting deep learning models and neural networks to [Core ML](https://developer.apple.com/documentation/coreml "Core ML Framework").

For example, you can use the Unified Conversion API to convert TensorFlow and PyTorch source model frameworks to Core ML. For the conversion parameters, see the [`convert()`](https://apple.github.io/coremltools/source/coremltools.converters.convert.html#coremltools.converters._converters_entry.convert) method.

```{note}
This section is about converting neural network models using the Unified Conversion API. For converting other classic models, see [LibSVM](libsvm-conversion), [Scikit-learn](sci-kit-learn-conversion), and [XGBoost](xgboost-conversion) in the "Other Converters" section.
```

For instructions and examples, see the following pages:

- [Converting Deep Learning Models](convert-learning-models)
- [ML Programs](convert-to-ml-program) 
- [Converting from PyTorch](convert-pytorch)
- [Converting from TensorFlow](convert-tensorflow) 
- [Examples](coremltools-examples) 

For common scenarios using conversion options, see the following pages:

- [Model Input and Output Types](model-input-and-output-types) 
- [Image Input and Output](image-inputs) 
- [Classifiers](classifiers)
- [Flexible Input Shapes](flexible-inputs)
- [Composite Operators](composite-operators) 
- [Custom Operators](custom-operators)
- [Graph Passes](graph-passes-intro)

