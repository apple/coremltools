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

        self.lightgbm_model = lgb.train(params, lgb_train, num_boost_round=10, valid_sets=lgb_train)

        # Load

    def test_interface(self):
        spec = lightgbm_converter(self.lightgbm_model, 'data', 'target').get_spec()  # TODO: Should it respect target?
        self.assertIsNotNone(spec)

        # Test the model class
        self.assertIsNotNone(spec.description)
        self.assertIsNotNone(spec.treeEnsembleClassifier)

        # Test the interface class
        self.assertEqual(spec.description.predictedFeatureName, 'classLabel')

        # Test the inputs and outputs
        self.assertEqual(len(spec.description.output), 2)
        self.assertEqual(spec.description.output[0].name, 'classLabel')  # Should this be 'target'?
        self.assertEqual(spec.description.output[1].name, 'classProbability')
        self.assertEqual(spec.description.output[0].type.WhichOneof('Type'), 'int64Type')
        self.assertEqual(spec.description.output[1].type.WhichOneof('Type'), 'dictionaryType')
        self.assertEqual(len(spec.description.input), 4)

        input_type = spec.description.input[0]

        self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
        print('input_type name: {}'.format(input_type.name))
        # self.assertEqual(input_type.name, 'data')  # TODO

        # Test actual tree attributes
        tr = spec.treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), 508)

    def test_prediction(self):
        spec = lightgbm_converter(self.lightgbm_model, 'data', 'target').get_spec()
        model = MLModel(spec)
        self.assertIsNotNone(model)

        coreml_prediction = model.predict({'Column_0': [0.0, 0.0, 0.0, 0.0],
                                           'Column_1': [], 'Column_2': [], 'Column_3': []})  # TODO: Fix this bug

        lightgbm_prediction = self.lightgbm_model.predict([[0, 0, 0, 0]])

        print('CoreML prediction: {}, LightGBM prediction: {}'.format(coreml_prediction, lightgbm_prediction))

        self.assertAlmostEqual(coreml_prediction['classProbability'][1], lightgbm_prediction[0])



if __name__ == '__main__':
    unittest.main()