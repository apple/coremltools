```{eval-rst}
.. index:: 
    single: Scikit-learn
```

# Scikit-learn

You can convert a [scikit-learn](https://scikit-learn.org/stable/) pipeline, classifier, or regressor to the Core ML format using [`sklearn.convert()`](https://apple.github.io/coremltools/source/coremltools.converters.sklearn.html#coremltools.converters.sklearn._converter.convert):

```python
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load data
data = pd.read_csv('houses.csv')

# Train a model
model = LinearRegression()
model.fit(data[["bedroom", "bath", "size"]], data["price"])

# Convert and save the scikit-learn model
import coremltools as ct
coreml_model = ct.converters.sklearn.convert(
  model, ["bedroom", "bath", "size"], "price")
coreml_model.save('HousePricer.mlmodel')
```

For more information, see the [API Reference](https://apple.github.io/coremltools/source/coremltools.converters.sklearn.html#module-coremltools.converters.sklearn._converter).
