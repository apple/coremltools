# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import unittest

import pandas as pd

from ..utils import load_boston
from coremltools._deps import _HAS_SKLEARN
from coremltools.models.utils import (_is_macos, _macos_version,
                                      evaluate_regressor)

if _HAS_SKLEARN:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder

    from coremltools.converters.sklearn import convert


@unittest.skipIf(not _HAS_SKLEARN, "Missing scikitlearn. Skipping tests.")
class RidgeRegressionScikitTest(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter.
    """

    @classmethod
    def setUpClass(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        scikit_data = load_boston()
        scikit_model = Ridge()
        scikit_model.fit(scikit_data["data"], scikit_data["target"])

        # Save the data and the model
        self.scikit_data = scikit_data
        self.scikit_model = scikit_model

    def test_conversion(self):
        input_names = self.scikit_data["feature_names"]
        spec = convert(self.scikit_model, input_names, "target").get_spec()
        self.assertIsNotNone(spec)

        # Test the model class
        self.assertIsNotNone(spec.description)

        # Test the interface class
        self.assertEqual(spec.description.predictedFeatureName, "target")

        # Test the inputs and outputs
        self.assertEqual(len(spec.description.output), 1)
        self.assertEqual(spec.description.output[0].name, "target")
        self.assertEqual(
            spec.description.output[0].type.WhichOneof("Type"), "doubleType"
        )
        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof("Type"), "doubleType")
        self.assertEqual(
            sorted(input_names), sorted(map(lambda x: x.name, spec.description.input))
        )

        # Test the ridge regression parameters.
        self.assertTrue(
            spec.pipelineRegressor.pipeline.models[-1].HasField("glmRegressor")
        )
        lr = spec.pipelineRegressor.pipeline.models[-1].glmRegressor
        self.assertEqual(lr.offset, self.scikit_model.intercept_)
        self.assertEqual(len(lr.weights), 1)
        self.assertEqual(len(lr.weights[0].value), 13)
        i = 0
        for w in lr.weights[0].value:
            self.assertAlmostEqual(w, self.scikit_model.coef_[i])
            i = i + 1

    def test_conversion_bad_inputs(self):
        # Error on converting an untrained model
        with self.assertRaises(TypeError):
            model = Ridge()
            spec = convert(model, "data", "out")

        # Check the expected class during conversion.
        with self.assertRaises(TypeError):
            model = OneHotEncoder()
            spec = convert(model, "data", "out")

    @unittest.skipUnless(
        _is_macos() and _macos_version() >= (10, 13), "Only supported on macOS 10.13+"
    )
    def test_ridge_regression_evaluation(self):
        """
        Check that the evaluation results are the same in scikit learn and coremltools
        """
        input_names = self.scikit_data["feature_names"]
        df = pd.DataFrame(self.scikit_data["data"], columns=input_names)

        for normalize_value in (True, False):
            cur_model = Ridge()
            cur_model.fit(self.scikit_data["data"], self.scikit_data["target"])
            spec = convert(cur_model, input_names, "target")

            df["target"] = cur_model.predict(self.scikit_data["data"])

            metrics = evaluate_regressor(spec, df)
            self.assertAlmostEqual(metrics["max_error"], 0)
