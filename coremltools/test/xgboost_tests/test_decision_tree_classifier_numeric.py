# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import unittest
from distutils.version import StrictVersion

import numpy as np
import pandas as pd
import pytest

from coremltools._deps import _HAS_SKLEARN, _SKLEARN_VERSION
from coremltools.models.utils import (_is_macos, _macos_version,
                                      evaluate_classifier)

if _HAS_SKLEARN:
    from sklearn.datasets import load_boston
    from sklearn.tree import DecisionTreeClassifier

    from coremltools.converters import sklearn as skl_converter


@unittest.skipIf(not _HAS_SKLEARN, "Missing sklearn. Skipping tests.")
class DecisionTreeClassificationBostonHousingScikitNumericTest(unittest.TestCase):
    def _check_metrics(self, metrics, params={}):
        self.assertEqual(
            metrics["num_errors"],
            0,
            msg="Failed case %s. Results %s" % (params, metrics),
        )

    def _train_convert_evaluate_assert(self, **scikit_params):
        scikit_model = DecisionTreeClassifier(random_state=1, **scikit_params)
        scikit_model.fit(self.X, self.target)

        # Convert the model
        spec = skl_converter.convert(scikit_model, self.feature_names, self.output_name)

        if _is_macos() and _macos_version() >= (10, 13):
            # Get predictions
            df = pd.DataFrame(self.X, columns=self.feature_names)
            df["target"] = scikit_model.predict(self.X)

            # Evaluate it
            metrics = evaluate_classifier(spec, df)
            self._check_metrics(metrics, scikit_params)


@unittest.skipIf(not _HAS_SKLEARN, "Missing sklearn. Skipping tests.")
class DecisionTreeBinaryClassificationBostonHousingScikitNumericTest(
    DecisionTreeClassificationBostonHousingScikitNumericTest
):
    @classmethod
    def setUpClass(self):
        # Load data and train model
        scikit_data = load_boston()
        self.scikit_data = scikit_data
        self.X = scikit_data.data.astype("f").astype(
            "d"
        )  ## scikit-learn downcasts data
        self.target = 1 * (scikit_data["target"] > scikit_data["target"].mean())
        self.feature_names = scikit_data.feature_names
        self.output_name = "target"

    def test_simple_binary_classifier(self):
        self._train_convert_evaluate_assert()

    @pytest.mark.slow
    def test_binary_classifier_stress_test(self):
        options = dict(
            splitter=["best"],
            max_depth=[1, 10, None],
            min_samples_split=[2, 10, 0.5],
            min_samples_leaf=[1, 5],
            min_weight_fraction_leaf=[0.0, 0.5],
            max_features=[None, 1, 5],
            max_leaf_nodes=[None, 20],
        )
        if _SKLEARN_VERSION < StrictVersion("0.22"): # 'presort' option deprecated >=0.22
            options["presort"] = [False, True]

        # Make a cartesian product of all options
        product = itertools.product(*options.values())
        args = [dict(zip(options.keys(), p)) for p in product]

        print("Testing a total of %s cases. This could take a while" % len(args))
        for it, arg in enumerate(args):
            self._train_convert_evaluate_assert(**arg)


@unittest.skipIf(not _HAS_SKLEARN, "Missing sklearn. Skipping tests.")
class DecisionTreeMultiClassClassificationBostonHousingScikitNumericTest(
    DecisionTreeClassificationBostonHousingScikitNumericTest
):
    @classmethod
    def setUpClass(self):
        # Load data and train model
        scikit_data = load_boston()
        num_classes = 3
        self.X = scikit_data.data.astype("f").astype(
            "d"
        )  ## scikit-learn downcasts data
        t = scikit_data.target
        target = np.digitize(t, np.histogram(t, bins=num_classes - 1)[1]) - 1

        # Save the data and the model
        self.scikit_data = scikit_data
        self.target = target
        self.feature_names = scikit_data.feature_names
        self.output_name = "target"

    def test_simple_multiclass(self):
        self._train_convert_evaluate_assert()

    @pytest.mark.slow
    def test_multiclass_stress_test(self):
        options = dict(
            splitter=["best"],
            max_depth=[1, 10, None],
            min_samples_split=[2, 10, 0.5],
            min_samples_leaf=[1, 5],
            min_weight_fraction_leaf=[0.0, 0.5],
            max_features=[None, 1, 5],
            max_leaf_nodes=[None, 20],
        )
        if _SKLEARN_VERSION < StrictVersion("0.22"): # 'presort' option deprecated >=0.22
            options["presort"] = [False, True]

        # Make a cartesian product of all options
        product = itertools.product(*options.values())
        args = [dict(zip(options.keys(), p)) for p in product]

        print("Testing a total of %s cases. This could take a while" % len(args))
        for it, arg in enumerate(args):
            self._train_convert_evaluate_assert(**arg)
