# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import json
import tempfile
import unittest

import numpy as np

from ..utils import load_boston
from coremltools._deps import _HAS_SKLEARN, _HAS_XGBOOST
from coremltools.converters import sklearn as skl_converter
from coremltools.models.utils import _macos_version

if _HAS_SKLEARN:
    from sklearn.ensemble import GradientBoostingClassifier

if _HAS_XGBOOST:
    import xgboost

    from coremltools.converters import xgboost as xgb_converter


@unittest.skipIf(not _HAS_SKLEARN, "Missing sklearn. Skipping tests.")
class GradientBoostingBinaryClassifierScikitTest(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter.
    """

    @classmethod
    def setUpClass(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        scikit_data = load_boston()
        scikit_model = GradientBoostingClassifier(random_state=1)
        target = scikit_data["target"] > scikit_data["target"].mean()
        scikit_model.fit(scikit_data["data"], target)

        s = 0
        for est in scikit_model.estimators_:
            for e in est:
                s = s + e.tree_.node_count
        self.scikit_model_node_count = s

        # Save the data and the model
        self.scikit_data = scikit_data
        self.scikit_model = scikit_model

    def test_conversion(self):
        input_names = self.scikit_data["feature_names"]
        output_name = "target"
        spec = skl_converter.convert(
            self.scikit_model, input_names, "target"
        ).get_spec()
        self.assertIsNotNone(spec)

        # Test the model class
        self.assertIsNotNone(spec.description)
        self.assertIsNotNone(spec.treeEnsembleClassifier)

        # Test the interface class
        self.assertEqual(spec.description.predictedFeatureName, "target")

        # Test the inputs and outputs
        self.assertEqual(len(spec.description.output), 2)
        self.assertEqual(spec.description.output[0].name, "target")
        self.assertEqual(
            spec.description.output[0].type.WhichOneof("Type"), "int64Type"
        )
        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof("Type"), "doubleType")
        self.assertEqual(
            sorted(input_names), sorted(map(lambda x: x.name, spec.description.input))
        )

        # Test the linear regression parameters.
        tr = spec.pipelineClassifier.pipeline.models[
            1
        ].treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), self.scikit_model_node_count)

    def test_conversion_bad_inputs(self):
        # Error on converting an untrained model
        with self.assertRaises(Exception):
            model = GradientBoostingClassifier()
            spec = skl_converter.convert(model, "data", "out")

        # Check the expected class during conversion.
        from sklearn.preprocessing import OneHotEncoder

        with self.assertRaises(Exception):
            model = OneHotEncoder()
            spec = skl_converter.convert(model, "data", "out")

class GradientBoostingMulticlassClassifierScikitTest(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter.
    """

    @classmethod
    def setUpClass(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        scikit_data = load_boston()
        scikit_model = GradientBoostingClassifier(random_state=1)
        t = scikit_data["target"]
        target = np.digitize(t, np.histogram(t)[1]) - 1
        scikit_model.fit(scikit_data["data"], target)
        self.target = target

        s = 0
        for est in scikit_model.estimators_:
            for e in est:
                s = s + e.tree_.node_count
        self.scikit_model_node_count = s
        
        # Save the data and the model
        self.scikit_data = scikit_data
        self.scikit_model = scikit_model

    def test_conversion(self):
        input_names = self.scikit_data["feature_names"]
        output_name = "target"
        spec = skl_converter.convert(
            self.scikit_model, input_names, "target"
        ).get_spec()
        self.assertIsNotNone(spec)

        # Test the model class
        self.assertIsNotNone(spec.description)
        self.assertEqual(spec.description.predictedFeatureName, "target")

        # Test the inputs and outputs
        self.assertEqual(len(spec.description.output), 2)
        self.assertEqual(spec.description.output[0].name, "target")
        self.assertEqual(
            spec.description.output[0].type.WhichOneof("Type"), "int64Type"
        )

        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof("Type"), "doubleType")
        self.assertEqual(
            sorted(input_names), sorted(map(lambda x: x.name, spec.description.input))
        )

        self.assertEqual(len(spec.pipelineClassifier.pipeline.models), 2)
        tr = spec.pipelineClassifier.pipeline.models[
            -1
        ].treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), self.scikit_model_node_count)

    def test_conversion_bad_inputs(self):
        # Error on converting an untrained model
        with self.assertRaises(Exception):
            model = GradientBoostingClassifier()
            spec = skl_converter.convert(model, "data", "out")

        # Check the expected class during conversion.
        from sklearn.preprocessing import OneHotEncoder

        with self.assertRaises(Exception):
            model = OneHotEncoder()
            spec = skl_converter.convert(model, "data", "out")


@unittest.skipIf(not _HAS_SKLEARN, "Missing sklearn. Skipping tests.")
@unittest.skipIf(not _HAS_XGBOOST, "Skipping, no xgboost")
class GradientBoostingBinaryClassifierXGboostTest(unittest.TestCase):
    """
    Unit test class for testing xgboost converter.
    """

    @classmethod
    def setUpClass(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        scikit_data = load_boston()
        self.xgb_model = xgboost.XGBClassifier()
        target = scikit_data["target"] > scikit_data["target"].mean()
        self.xgb_model.fit(scikit_data["data"], target)

        # Save the data and the model
        self.scikit_data = scikit_data

    def test_conversion(self):
        input_names = self.scikit_data["feature_names"]
        output_name = "target"
        spec = xgb_converter.convert(
            self.xgb_model, input_names, output_name, mode="classifier"
        ).get_spec()
        self.assertIsNotNone(spec)

        # Test the model class
        self.assertIsNotNone(spec.description)
        self.assertIsNotNone(spec.treeEnsembleClassifier)

        # Test the interface class
        self.assertEqual(spec.description.predictedFeatureName, output_name)

        # Test the inputs and outputs
        self.assertEqual(len(spec.description.output), 2)
        self.assertEqual(spec.description.output[0].name, output_name)
        self.assertEqual(
            spec.description.output[0].type.WhichOneof("Type"), "int64Type"
        )
        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof("Type"), "doubleType")
        self.assertEqual(
            sorted(input_names), sorted(map(lambda x: x.name, spec.description.input))
        )

        # Test the linear regression parameters.
        tr = spec.treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)

    def test_conversion_bad_inputs(self):
        # Error on converting an untrained model
        with self.assertRaises(Exception):
            model = xgboost.XGBClassifier()
            spec = xgb_converter.convert(model, "data", "out", mode="classifier")

        # Check the expected class during conversion.
        with self.assertRaises(Exception):
            model = xgboost.XGBRegressor()
            spec = xgb_converter.convert(model, "data", "out", mode="classifier")


@unittest.skipIf(not _HAS_SKLEARN, "Missing sklearn. Skipping tests.")
@unittest.skipIf(not _HAS_XGBOOST, "Skipping, no xgboost")
class GradientBoostingMulticlassClassifierXGboostTest(unittest.TestCase):
    """
    Unit test class for testing xgboost converter.
    """

    @classmethod
    def setUpClass(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        scikit_data = load_boston()
        t = scikit_data["target"]
        target = np.digitize(t, np.histogram(t)[1]) - 1
        dtrain = xgboost.DMatrix(
            scikit_data["data"], label=target, feature_names=scikit_data["feature_names"]
        )
        self.xgb_model = xgboost.train({}, dtrain)
        self.target = target

        # Save the data and the model
        self.scikit_data = scikit_data
        self.n_classes = len(np.unique(self.target))

        # train a booster with special characters in feature names
        x = scikit_data['data']
        # prepare feature names with special chars
        self.feature_names_special_chars = [f'\t"{i}"\n' for i in
                                            range(x.shape[1])]
        # create training dmatrix
        dm = xgboost.DMatrix(x, label=target,
                             feature_names=self.feature_names_special_chars)
        # train booster
        self.xgb_model_special_chars = xgboost.train({}, dm)
        # create XGBClassifier from a copy of trainer booster
        self.xgb_classifier_special_chars = \
            xgboost.XGBClassifier(xgb_model=self.xgb_model_special_chars.copy())
        self.xgb_classifier_special_chars.fit(x, target)

    def test_conversion(self):

        input_names = self.scikit_data["feature_names"]
        output_name = "target"
        spec = xgb_converter.convert(
            self.xgb_model,
            input_names,
            output_name,
            mode="classifier",
            n_classes=self.n_classes,
        ).get_spec()
        self.assertIsNotNone(spec)

        # Test the model class
        self.assertIsNotNone(spec.description)
        self.assertEqual(spec.description.predictedFeatureName, output_name)

        # Test the inputs and outputs
        self.assertEqual(len(spec.description.output), 2)
        self.assertEqual(spec.description.output[0].name, output_name)
        self.assertEqual(
            spec.description.output[0].type.WhichOneof("Type"), "int64Type"
        )

        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof("Type"), "doubleType")
        self.assertEqual(
            sorted(input_names), sorted(map(lambda x: x.name, spec.description.input))
        )

        # Test the linear regression parameters.
        tr = spec.treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)

    def test_conversion_from_file(self):
        import numpy as np

        output_name = "target"
        feature_names = self.scikit_data["feature_names"]

        xgb_model_json = tempfile.mktemp("xgb_tree_model_classifier.json")
        xgb_json_out = self.xgb_model.get_dump(with_stats=True, dump_format="json")
        with open(xgb_model_json, "w") as f:
            json.dump(xgb_json_out, f)
        spec = xgb_converter.convert(
            xgb_model_json,
            feature_names,
            output_name,
            mode="classifier",
            n_classes=self.n_classes,
        ).get_spec()
        self.assertIsNotNone(spec)

        # Test the model class
        self.assertIsNotNone(spec.description)
        self.assertIsNotNone(spec.treeEnsembleRegressor)

        # Test the interface class
        self.assertEqual(spec.description.predictedFeatureName, output_name)

        # Test the inputs and outputs
        self.assertEqual(len(spec.description.output), 2)
        self.assertEqual(spec.description.output[0].name, output_name)
        self.assertEqual(
            spec.description.output[0].type.WhichOneof("Type"), "int64Type"
        )
        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof("Type"), "doubleType")
        self.assertEqual(
            sorted(self.scikit_data["feature_names"]),
            sorted(map(lambda x: x.name, spec.description.input)),
        )

        # Test the linear regression parameters.
        tr = spec.treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)

    def test_conversion_special_characters_in_feature_names(self):
        # this test should fail if conversion function does not implement the
        # special characters in feature names fix

        # test both sklearn wrapper and raw booster
        for model in [self.xgb_model_special_chars, self.xgb_classifier_special_chars]:

            # process as usual
            output_name = "target"
            spec = xgb_converter.convert(
                model,
                self.feature_names_special_chars,
                output_name,
                mode="classifier",
                n_classes=self.n_classes,
            ).get_spec()
            self.assertIsNotNone(spec)

            # Test the model class
            self.assertIsNotNone(spec.description)
            self.assertEqual(spec.description.predictedFeatureName, output_name)

            # Test the inputs and outputs
            self.assertEqual(len(spec.description.output), 2)
            self.assertEqual(spec.description.output[0].name, output_name)
            self.assertEqual(
                spec.description.output[0].type.WhichOneof("Type"), "int64Type"
            )

            for input_type in spec.description.input:
                self.assertEqual(input_type.type.WhichOneof("Type"), "doubleType")
            self.assertEqual(
                sorted(self.feature_names_special_chars), sorted(map(lambda x: x.name, spec.description.input))
            )

            # Test the linear regression parameters.
            tr = spec.treeEnsembleClassifier.treeEnsemble
            self.assertIsNotNone(tr)
