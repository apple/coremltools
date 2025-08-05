```{eval-rst}
.. index:: 
    single: XGBoost
```

# XGBoost

You can convert a trained [XGBoost](https://en.wikipedia.org/wiki/XGBoost) model to Core ML format using [`xgboost.convert()`](https://apple.github.io/coremltools/source/coremltools.converters.xgboost.html#coremltools.converters.xgboost._tree.convert):

```python
# Convert it with default input and output names
import coremltools as ct
coreml_model = ct.converters.xgboost.convert(model)

# Saving the Core ML model to a file.
coreml_model.save('my_model.mlmodel')
```

For more information, see the [API Reference](https://apple.github.io/coremltools/source/coremltools.converters.xgboost.html#module-coremltools.converters.xgboost._tree).
