# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import unittest

import pandas as pd
import pytest

from coremltools._deps import _HAS_SKLEARN, _HAS_XGBOOST
from coremltools.models.utils import (_is_macos, _macos_version,
                                      evaluate_regressor)

if _HAS_XGBOOST:
    import xgboost

    from coremltools.converters import xgboost as xgb_converter

if _HAS_SKLEARN:
    from sklearn.datasets import load_boston
    from sklearn.ensemble import GradientBoostingRegressor

    from coremltools.converters import sklearn as skl_converter


@unittest.skipIf(not _HAS_SKLEARN, "Missing sklearn. Skipping tests.")
class GradientBoostingRegressorBostonHousingScikitNumericTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Load data and train model
        scikit_data = load_boston()
        self.scikit_data = scikit_data
        self.X = scikit_data["data"]
        self.target = scikit_data["target"]
        self.feature_names = scikit_data.feature_names
        self.output_name = "target"

    def _check_metrics(self, metrics, params={}):
        self.assertAlmostEqual(
            metrics["rmse"],
            0,
            delta=1e-5,
            msg="Failed case %s. Results %s" % (params, metrics),
        )
        self.assertAlmostEqual(
            metrics["max_error"],
            0,
            delta=1e-5,
            msg="Failed case %s. Results %s" % (params, metrics),
        )

    def _train_convert_evaluate_assert(self, **scikit_params):
        scikit_model = GradientBoostingRegressor(random_state=1, **scikit_params)
        scikit_model.fit(self.X, self.target)

        # Convert the model
        spec = skl_converter.convert(scikit_model, self.feature_names, self.output_name)

        if _is_macos() and _macos_version() >= (10, 13):
            # Get predictions
            df = pd.DataFrame(self.X, columns=self.feature_names)
            df["target"] = scikit_model.predict(self.X)

            # Evaluate it
            metrics = evaluate_regressor(spec, df, "target", verbose=False)
            self._check_metrics(metrics, scikit_params)

    def test_boston_housing_simple_regression(self):
        self._train_convert_evaluate_assert()

    @pytest.mark.slow
    def test_boston_housing_parameter_stress_test(self):

        options = dict(
            max_depth=[1, 10, None],
            min_samples_split=[2, 0.5],
            min_samples_leaf=[1, 5],
            min_weight_fraction_leaf=[0.0, 0.5],
            max_features=[None, 1],
            max_leaf_nodes=[None, 20],
        )

        # Make a cartesian product of all options
        product = itertools.product(*options.values())
        args = [dict(zip(options.keys(), p)) for p in product]

        print("Testing a total of %s cases. This could take a while" % len(args))
        for it, arg in enumerate(args):
            self._train_convert_evaluate_assert(**arg)


@unittest.skipIf(_macos_version() >= (12, 0), "rdar://problem/84898245")
@unittest.skipIf(not _HAS_XGBOOST, "Missing xgboost. Skipping")
@unittest.skipIf(not _HAS_SKLEARN, "Missing scikit-learn. Skipping tests.")
class XgboostBoosterBostonHousingNumericTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        if not _HAS_XGBOOST:
            return
        if not _HAS_SKLEARN:
            return

        # Load data and train model
        scikit_data = load_boston()
        self.X = scikit_data.data.astype("f").astype("d")
        self.dtrain = xgboost.DMatrix(
            scikit_data.data,
            label=scikit_data.target,
            feature_names=scikit_data.feature_names,
        )
        self.feature_names = scikit_data.feature_names
        self.output_name = "target"

    def _check_metrics(self, metrics, allowed_error={}, params={}):
        """
        Check the metrics
        """
        self.assertAlmostEqual(
            metrics["rmse"],
            allowed_error.get("rmse", 0),
            delta=1e-2,
            msg="Failed case %s. Results %s" % (params, metrics),
        )
        self.assertAlmostEqual(
            metrics["max_error"],
            allowed_error.get("max_error", 0),
            delta=1e-2,
            msg="Failed case %s. Results %s" % (params, metrics),
        )

    def _train_convert_evaluate_assert(self, bt_params={}, allowed_error={}, **params):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        # Train a model
        xgb_model = xgboost.train(bt_params, self.dtrain, **params)

        # Convert the model
        spec = xgb_converter.convert(
            xgb_model, self.feature_names, self.output_name, force_32bit_float=False
        )

        if _is_macos() and _macos_version() >= (10, 13):
            # Get predictions
            df = pd.DataFrame(self.X, columns=self.feature_names)
            df["target"] = xgb_model.predict(self.dtrain)

            # Evaluate it
            metrics = evaluate_regressor(spec, df, target="target", verbose=False)
            self._check_metrics(metrics, allowed_error, bt_params)

    def test_boston_housing_simple_decision_tree_regression(self):
        self._train_convert_evaluate_assert(num_boost_round=1)

    def test_boston_housing_simple_boosted_tree_regression(self):
        self._train_convert_evaluate_assert(num_boost_round=10)

    def test_boston_housing_simple_random_forest_regression(self):
        self._train_convert_evaluate_assert(bt_params={"subsample": 0.5},
                                            allowed_error={"rmse": 0.004, "max_error": 0.09})

    def test_boston_housing_float_double_corner_case(self):
        self._train_convert_evaluate_assert(
            {
                "colsample_bytree": 1,
                "colsample_bylevel": 1,
                "scale_pos_weight": 1,
                "learning_rate": 0.5,
                "max_delta_step": 0,
                "min_child_weight": 1,
                "n_estimators": 1,
                "subsample": 0.5,
                "objective": "reg:linear",
                "max_depth": 5,
            },
            num_boost_round=2,
        )

    @pytest.mark.slow
    def test_boston_housing_parameter_stress_test(self):

        options = dict(
            max_depth=[1, 5],
            learning_rate=[0.1, 0.5],
            n_estimators=[1, 10],
            min_child_weight=[1, 2],
            max_delta_step=[0, 0.1],
            colsample_bytree=[1, 0.5],
            colsample_bylevel=[1, 0.5],
            scale_pos_weight=[1],
            objective=["reg:linear"],
        )

        # Make a cartesian product of all options
        product = itertools.product(*options.values())
        args = [dict(zip(options.keys(), p)) for p in product]

        print("Testing a total of %s cases. This could take a while" % len(args))
        for it, arg in enumerate(args):
            self._train_convert_evaluate_assert(arg)


@unittest.skipIf(_macos_version() >= (12, 0), "rdar://problem/84898245")
@unittest.skipIf(not _HAS_XGBOOST, "Missing xgboost. Skipping")
@unittest.skipIf(not _HAS_SKLEARN, "Missing sklearn. Skipping tests.")
class XGboostRegressorBostonHousingNumericTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """

        # Load data and train model
        scikit_data = load_boston()

        self.X = scikit_data.data
        self.scikit_data = self.X
        self.target = scikit_data.target
        self.feature_names = scikit_data.feature_names
        self.output_name = "target"

    def _check_metrics(self, metrics, params={}, allowed_error={}):
        self.assertAlmostEqual(
            metrics["rmse"],
            allowed_error.get("rmse", 0),
            delta=1e-2,
            msg="Failed case %s. Results %s" % (params, metrics),
        )
        self.assertAlmostEqual(
            metrics["max_error"],
            allowed_error.get("max_error", 0),
            delta=1e-2,
            msg="Failed case %s. Results %s" % (params, metrics),
        )

    def _train_convert_evaluate_assert(self, bt_params={}, allowed_error={}, **params):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        # Train a model
        xgb_model = xgboost.XGBRegressor(**params)
        xgb_model.fit(self.X, self.target)

        # Convert the model (feature_names can't be given because of XGboost)
        spec = xgb_converter.convert(
            xgb_model, self.feature_names, self.output_name, force_32bit_float=False
        )

        if _is_macos() and _macos_version() >= (10, 13):
            # Get predictions
            df = pd.DataFrame(self.X, columns=self.feature_names)
            df["target"] = xgb_model.predict(self.X)

            # Evaluate it
            metrics = evaluate_regressor(spec, df, target="target", verbose=False)
            self._check_metrics(metrics, bt_params, allowed_error)

    def test_boston_housing_simple_boosted_tree_regression(self):
        self._train_convert_evaluate_assert()

    def test_boston_housing_simple_random_forest_regression(self):
        self._train_convert_evaluate_assert(
            allowed_error={"rmse": 0.05, "max_error": 0.81}, subsample=0.5
        )

    def test_boston_housing_simple_decision_tree_regression(self):
        self._train_convert_evaluate_assert(n_estimators=1)

    def test_boston_housing_float_double_corner_case(self):
        self._train_convert_evaluate_assert(
            {
                "colsample_bytree": 1,
                "colsample_bylevel": 1,
                "scale_pos_weight": 1,
                "learning_rate": 0.1,
                "max_delta_step": 0,
                "min_child_weight": 1,
                "n_estimators": 10,
                "subsample": 0.3,
                "objective": "reg:linear",
                "max_depth": 1,
            }
        )

    @pytest.mark.slow
    def test_boston_housing_parameter_stress_test(self):

        options = dict(
            max_depth=[1, 5],
            learning_rate=[0.1, 0.5],
            n_estimators=[1, 10],
            objective=["reg:linear"],
            min_child_weight=[1, 2],
            max_delta_step=[0, 0.1],
            subsample=[1, 0.5, 0.3],
            colsample_bytree=[1, 0.5],
            colsample_bylevel=[1, 0.5],
            scale_pos_weight=[1],
        )

        # Make a cartesian product of all options
        product = itertools.product(*options.values())
        args = [dict(zip(options.keys(), p)) for p in product]

        print("Testing a total of %s cases. This could take a while" % len(args))
        for it, arg in enumerate(args):
            self._train_convert_evaluate_assert(arg)
