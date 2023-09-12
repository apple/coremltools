```{eval-rst}
.. index:: 
    single: LibSVM
```

# LibSVM

You can convert a [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) model to the Core ML format using [`libsvm.convert()`](https://apple.github.io/coremltools/source/coremltools.converters.libsvm.html#coremltools.converters.libsvm._libsvm_converter.convert):

```python
# Make a LIBSVM model
import svmutil
problem = svmutil.svm_problem([0,0,1,1], [[0,1], [1,1], [8,9], [7,7]])
libsvm_model = svmutil.svm_train(problem, svmutil.svm_parameter())

# Convert using default input and output names
import coremltools as ct
coreml_model = ct.converters.libsvm.convert(libsvm_model)

# Save the Core ML model to a file.
coreml_model.save('./my_model.mlmodel')

# Convert using user specified input names
coreml_model = ct.converters.libsvm.convert(libsvm_model, input_names=['x', 'y'])
```

For more information, see the [API Reference](https://apple.github.io/coremltools/source/coremltools.converters.libsvm.html#module-coremltools.converters.libsvm._libsvm_converter).
