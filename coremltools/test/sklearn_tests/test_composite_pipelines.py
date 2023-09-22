# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import unittest
from distutils.version import StrictVersion

import pandas as pd

from coremltools._deps import _HAS_SKLEARN, _SKLEARN_VERSION
from coremltools.converters.sklearn import convert
from coremltools.models.utils import (_is_macos, _macos_version,
                                      evaluate_regressor, evaluate_transformer)

if _HAS_SKLEARN:
    from sklearn.datasets import load_boston
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler


@unittest.skipIf(not _HAS_SKLEARN, "Missing sklearn. Skipping tests.")
class GradientBoostingRegressorBostonHousingScikitNumericTest(unittest.TestCase):

    @unittest.skipIf(not _HAS_SKLEARN, "Missing sklearn. Skipping tests.")
    @unittest.skipIf(_SKLEARN_VERSION >= StrictVersion("0.22"),
        "categorical_features parameter to OneHotEncoder() deprecated after SciKit Learn 0.22."
    )
    def test_boston_OHE_plus_normalizer(self):
        data = load_boston()

        pl = Pipeline(
            [
                ("OHE", OneHotEncoder(categorical_features=[8], sparse=False)),
                ("Scaler", StandardScaler()),
            ]
        )

        pl.fit(data.data, data.target)

        # Convert the model
        spec = convert(pl, data.feature_names, "out")

        if _is_macos() and _macos_version() >= (10, 13):
            input_data = [dict(zip(data.feature_names, row)) for row in data.data]
            output_data = [{"out": row} for row in pl.transform(data.data)]

            result = evaluate_transformer(spec, input_data, output_data)
            assert result["num_errors"] == 0

    @unittest.skipIf(_SKLEARN_VERSION >= StrictVersion("0.22"),
        "categorical_features parameter to OneHotEncoder() deprecated after SciKit Learn 0.22."
    )
    def _test_boston_OHE_plus_trees(self, loss='ls'):

        data = load_boston()

        pl = Pipeline(
            [
                ("OHE", OneHotEncoder(categorical_features=[8], sparse=False)),
                ("Trees", GradientBoostingRegressor(random_state=1, loss=loss)),
            ]
        )

        pl.fit(data.data, data.target)

        # Convert the model
        spec = convert(pl, data.feature_names, "target")

        if _is_macos() and _macos_version() >= (10, 13):
            # Get predictions
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = pl.predict(data.data)

            # Evaluate it
            result = evaluate_regressor(spec, df, "target", verbose=False)

            assert result["max_error"] < 0.0001

    def test_boston_OHE_plus_trees(self):
        self._test_boston_OHE_plus_trees()

    def test_boston_OHE_plus_trees_with_huber_loss(self):
        self._test_boston_OHE_plus_trees(loss='huber')
