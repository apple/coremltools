# Created by Caleb Madrigal
# Copyright (c) 2018, FireEye Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import unittest
from coremltools._deps import HAS_LIGHTGBM, HAS_SKLEARN
from coremltools.models import MLModel

if HAS_LIGHTGBM:
    from coremltools.converters.lightgbm import convert as lightgbm_converter


@unittest.skipIf(not HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
@unittest.skipIf(not HAS_LIGHTGBM, 'Missing LightGBM. Skipping tests.')
class LightGBMTreeBinaryClassifierPredictionTest(unittest.TestCase):
    """
    Unit test class for testing LightGBM converter.
    """

    def setUp(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        from sklearn.datasets import make_classification
        import lightgbm as lgb

        self.num_features = 4

        X, y = make_classification(n_samples=1000, n_features=self.num_features,
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
        self.spec = lightgbm_converter(self.lightgbm_model, None, 'target').get_spec()

        # Load CoreML executable model from the converted CoreML LightGBM spec
        self.model = MLModel(self.spec)

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


@unittest.skipIf(not HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
@unittest.skipIf(not HAS_LIGHTGBM, 'Missing LightGBM. Skipping tests.')
class LightGBMTreeBinaryClassifierPredictionArrayInputTest(unittest.TestCase):
    """
    Unit test class for testing LightGBM converter.
    """

    def setUp(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        from sklearn.datasets import make_classification
        import lightgbm as lgb

        self.num_features = 4

        X, y = make_classification(n_samples=1000, n_features=self.num_features,
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

    def test_loaded_model(self):
        self.assertIsNotNone(self.model)

    def test_prediction1(self):
        coreml_prediction = self.model.predict({'data': [0.0, 0.0, 0.0, 0.0]})
        lightgbm_prediction = self.lightgbm_model.predict([[0, 0, 0, 0]])
        self.assertAlmostEqual(coreml_prediction['classProbability'][1], lightgbm_prediction[0])

    def test_prediction2(self):
        coreml_prediction = self.model.predict({'data': [0.0, -1.0, 0.0, 0.0]})
        lightgbm_prediction = self.lightgbm_model.predict([[0, -1, 0, 0]])
        self.assertAlmostEqual(coreml_prediction['classProbability'][1], lightgbm_prediction[0])

    def test_prediction3(self):
        coreml_prediction = self.model.predict({'data': [1.0, -1.0, 5.5, 6.0]})
        lightgbm_prediction = self.lightgbm_model.predict([[1, -1, 5.5, 6]])
        self.assertAlmostEqual(coreml_prediction['classProbability'][1], lightgbm_prediction[0])


@unittest.skipIf(not HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
@unittest.skipIf(not HAS_LIGHTGBM, 'Missing LightGBM. Skipping tests.')
class LightGBMTreeRegressorPredictionTest(unittest.TestCase):
    """
    Unit test class for testing LightGBM converter.
    """

    def setUp(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        from sklearn.datasets import make_regression
        import lightgbm as lgb

        self.num_features = 3

        X_reg, y_reg = make_regression(n_samples=1000, n_features=self.num_features,
                                       n_informative=2, random_state=0, shuffle=False)

        lgb_train_reg = lgb.Dataset(X_reg, y_reg)

        params_reg = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'verbose': 0
        }

        # Build native LightGBM model
        self.lightgbm_model = lgb.train(params_reg,
                                        lgb_train_reg,
                                        num_boost_round=10,
                                        valid_sets=lgb_train_reg)

        # Do conversion
        self.spec = lightgbm_converter(self.lightgbm_model,
                                       ['Column_0', 'Column_1', 'Column_2'],
                                       'target123').get_spec()

        # Load CoreML executable model from the converted CoreML LightGBM spec
        self.model = MLModel(self.spec)

    def test_loaded_model(self):
        self.assertIsNotNone(self.model)

    def test_prediction1(self):
        """regressor([0, 0, 0]) -> -0.21527121893057433"""
        coreml_prediction = self.model.predict({'Column_0': 0.0, 'Column_1': 0.0, 'Column_2': 0.0})
        lightgbm_prediction = self.lightgbm_model.predict([[0, 0, 0]])
        self.assertAlmostEqual(coreml_prediction['target123'], lightgbm_prediction[0])

    def test_prediction2(self):
        """regressor([1, 2, 3]) -> 129.5898800010246"""
        coreml_prediction = self.model.predict({'Column_0': 1.0, 'Column_1': 2.0, 'Column_2': 3.0})
        lightgbm_prediction = self.lightgbm_model.predict([[1, 2, 3]])
        self.assertAlmostEqual(coreml_prediction['target123'], lightgbm_prediction[0])

    def test_prediction3(self):
        """regressor([1, 2, 0]) -> 129.5898800010246"""
        coreml_prediction = self.model.predict({'Column_0': 1.0, 'Column_1': 2.0, 'Column_2': 0.0})
        lightgbm_prediction = self.lightgbm_model.predict([[1, 2, 0]])
        self.assertAlmostEqual(coreml_prediction['target123'], lightgbm_prediction[0])

    def test_prediction4(self):
        """regressor([2, 1, 0]) -> 118.24945326331657"""
        coreml_prediction = self.model.predict({'Column_0': 2.0, 'Column_1': 1.0, 'Column_2': 0.0})
        lightgbm_prediction = self.lightgbm_model.predict([[2, 1, 0]])
        self.assertAlmostEqual(coreml_prediction['target123'], lightgbm_prediction[0])


@unittest.skipIf(not HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
@unittest.skipIf(not HAS_LIGHTGBM, 'Missing LightGBM. Skipping tests.')
class LightGBMTreeMulticlassClassifierPredictionTest(unittest.TestCase):
    """
    Unit test class for testing LightGBM converter.
    """

    def setUp(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        from sklearn.datasets import make_classification
        import lightgbm as lgb

        self.num_features = 3
        self.num_classes = 4

        X_multiclass, y_multiclass = make_classification(n_samples=1000, n_features=self.num_features,
                                                         n_classes=self.num_classes,
                                                         n_informative=self.num_features, n_redundant=0,
                                                         random_state=0, shuffle=False)

        lgb_train_multiclass = lgb.Dataset(X_multiclass, y_multiclass)

        params_multiclass = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': self.num_classes,
            'verbose': 0
        }

        # Build native LightGBM model
        self.lightgbm_model = lgb.train(params_multiclass,
                                        lgb_train_multiclass,
                                        num_boost_round=10,
                                        valid_sets=lgb_train_multiclass)

        # Do conversion
        self.spec = lightgbm_converter(self.lightgbm_model, None, None).get_spec()

        # Load CoreML executable model from the converted CoreML LightGBM spec
        self.model = MLModel(self.spec)

    def test_loaded_model(self):
        self.assertIsNotNone(self.model)

    def test_prediction1(self):
        """multiclass([0, 0, 0]) -> [0.44128488, 0.15541033, 0.24996637, 0.15333842]"""
        coreml_prediction = self.model.predict({'Column_0': 0.0, 'Column_1': 0.0, 'Column_2': 0.0})
        lightgbm_prediction = self.lightgbm_model.predict([[0, 0, 0]])

        # Check probabilities
        for class_id in range(len(lightgbm_prediction)):
            lgbm_pred_for_class = lightgbm_prediction[0][class_id]
            coreml_pred_for_class = coreml_prediction['classProbability'][class_id]
            self.assertAlmostEqual(lgbm_pred_for_class, coreml_pred_for_class)

        # Check label
        lightgbm_predicted_class = max(enumerate(lightgbm_prediction[0]), key=lambda x: x[1])[0]
        self.assertEqual(lightgbm_predicted_class, coreml_prediction['classLabel'])

    def test_prediction2(self):
        """multiclass([-1, 0, 0]) -> [0.18612569, 0.27207182, 0.28398856, 0.25781393]"""
        coreml_prediction = self.model.predict({'Column_0': -1.0, 'Column_1': 0.0, 'Column_2': 0.0})
        lightgbm_prediction = self.lightgbm_model.predict([[-1, 0, 0]])

        # Check probabilities
        for class_id in range(len(lightgbm_prediction)):
            lgbm_pred_for_class = lightgbm_prediction[0][class_id]
            coreml_pred_for_class = coreml_prediction['classProbability'][class_id]
            self.assertAlmostEqual(lgbm_pred_for_class, coreml_pred_for_class)

        # Check label
        lightgbm_predicted_class = max(enumerate(lightgbm_prediction[0]), key=lambda x: x[1])[0]
        self.assertEqual(lightgbm_predicted_class, coreml_prediction['classLabel'])

    def test_prediction3(self):
        """multiclass([-1, 5, 0]) -> [0.11954505, 0.11977181, 0.57066349, 0.19001964]"""
        coreml_prediction = self.model.predict({'Column_0': -1.0, 'Column_1': 5.0, 'Column_2': 0.0})
        lightgbm_prediction = self.lightgbm_model.predict([[-1, 5, 0]])

        # Check probabilities
        for class_id in range(len(lightgbm_prediction)):
            lgbm_pred_for_class = lightgbm_prediction[0][class_id]
            coreml_pred_for_class = coreml_prediction['classProbability'][class_id]
            self.assertAlmostEqual(lgbm_pred_for_class, coreml_pred_for_class)

        # Check label
        lightgbm_predicted_class = max(enumerate(lightgbm_prediction[0]), key=lambda x: x[1])[0]
        self.assertEqual(lightgbm_predicted_class, coreml_prediction['classLabel'])

if __name__ == '__main__':
    unittest.main()
