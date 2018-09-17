# Copyright (c) 2018, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from coremltools._deps import HAS_LIGHTGBM, HAS_SKLEARN
from coremltools.models import MLModel

if HAS_LIGHTGBM:
    from coremltools.converters.lightgbm import convert as lightgbm_converter


@unittest.skipIf(not HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
@unittest.skipIf(not HAS_LIGHTGBM, 'Missing LightGBM. Skipping tests.')
class LightGBMTreeClassifierTest(unittest.TestCase):
    """
    Unit test class for testing LightGBM converter.
    """

    def setUp(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        from sklearn.datasets import make_classification
        import lightgbm as lgb

        X, y = make_classification(n_samples=1000, n_features=4,
                                   n_informative=2, n_redundant=0,
                                   random_state=0, shuffle=False)

        lgb_train = lgb.Dataset(X, y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': 0
        }

        # Build native LightGBM model
        self.lightgbm_model = lgb.train(params, lgb_train, num_boost_round=10, valid_sets=lgb_train)

        # Do conversion
        self.spec = lightgbm_converter(self.lightgbm_model, 'data', 'target').get_spec()

        # Load CoreML executable model from the converted CoreML LightGBM spec
        self.model = MLModel(self.spec)

    def test_spec_interface(self):
        self.assertIsNotNone(self.spec)

        # Test the model class
        self.assertIsNotNone(self.spec.description)
        self.assertIsNotNone(self.spec.treeEnsembleClassifier)

        # Test the interface class
        self.assertEqual(self.spec.description.predictedFeatureName, 'classLabel')

        # Test the inputs and outputs
        self.assertEqual(len(self.spec.description.output), 2)
        self.assertEqual(self.spec.description.output[0].name, 'classLabel')  # Should this be 'target'?
        self.assertEqual(self.spec.description.output[1].name, 'classProbability')
        self.assertEqual(self.spec.description.output[0].type.WhichOneof('Type'), 'int64Type')
        self.assertEqual(self.spec.description.output[1].type.WhichOneof('Type'), 'dictionaryType')
        self.assertEqual(len(self.spec.description.input), 4)

        for feature_index in range(4):
            input_type = self.spec.description.input[feature_index]
            self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
            self.assertEqual(input_type.name, 'Column_{}'.format(feature_index))

        # Test actual tree attributes
        tr = self.spec.treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), 508)

    def test_loaded_model(self):
        self.assertIsNotNone(self.model)

    def test_prediction1(self):
        coreml_prediction = self.model.predict({'Column_0': 0.0, 'Column_1': 0.0, 'Column_2': 0.0, 'Column_3': 0.0})
        lightgbm_prediction = self.lightgbm_model.predict([[0, 0, 0, 0]])
        self.assertAlmostEqual(coreml_prediction['classProbability'][1], lightgbm_prediction[0])

    def test_prediction2(self):
        coreml_prediction = self.model.predict({'Column_0': 0.0, 'Column_1': -1.0, 'Column_2': 0.0, 'Column_3': 0.0})
        lightgbm_prediction = self.lightgbm_model.predict([[0, -1, 0, 0]])
        self.assertAlmostEqual(coreml_prediction['classProbability'][1], lightgbm_prediction[0])

    def test_prediction3(self):
        coreml_prediction = self.model.predict({'Column_0': 1.0, 'Column_1': -1.0, 'Column_2': 5.5, 'Column_3': 6.0})
        lightgbm_prediction = self.lightgbm_model.predict([[1, -1, 5.5, 6]])
        self.assertAlmostEqual(coreml_prediction['classProbability'][1], lightgbm_prediction[0])

if __name__ == '__main__':
    unittest.main()
