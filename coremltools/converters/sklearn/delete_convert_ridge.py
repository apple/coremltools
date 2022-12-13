import numpy as np
import coremltools as ct
from sklearn.linear_model import Ridge, LinearRegression

n_samples, n_features = 10, 5
rng = np.random.RandomState(0)

y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)

clf = Ridge(alpha=1.0)
clf.fit(X, y)

# print(clf.__dict__)

coreml_model = ct.converters.sklearn.convert(clf)
coreml_model.save('Ridge.mlmodel')

converted_model = ct.models.MLModel('Ridge.mlmodel')

trained_model_predictions = clf.predict(X)