# Created by Caleb Madrigal
# Copyright (c) 2018, FireEye Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import unittest
from coremltools._deps import HAS_LIGHTGBM, HAS_SKLEARN

if HAS_LIGHTGBM:
    from coremltools.converters.lightgbm import convert as lightgbm_converter


@unittest.skipIf(not HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
@unittest.skipIf(not HAS_LIGHTGBM, 'Missing LightGBM. Skipping tests.')
class LightGBMTreeBinaryClassifierArrayInputTest(unittest.TestCase):
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

    def test_spec_interface(self):
        self.assertIsNotNone(self.spec)

        # Test the model class
        self.assertIsNotNone(self.spec.description)
        self.assertIsNotNone(self.spec.treeEnsembleClassifier)

        # Test the interface class
        self.assertEqual(self.spec.description.predictedFeatureName, 'target')

        # Test the inputs and outputs
        self.assertEqual(len(self.spec.description.output), 2)
        self.assertEqual(self.spec.description.output[0].name, 'target')  # Should this be 'target'?
        self.assertEqual(self.spec.description.output[1].name, 'classProbability')
        self.assertEqual(self.spec.description.output[0].type.WhichOneof('Type'), 'int64Type')
        self.assertEqual(self.spec.description.output[1].type.WhichOneof('Type'), 'dictionaryType')
        self.assertEqual(len(self.spec.description.input), 1)

        self.assertEqual(self.spec.description.input[0].type.WhichOneof('Type'), 'multiArrayType')
        self.assertEqual(self.spec.description.input[0].name, 'data')

        # Test actual tree attributes
        tr = self.spec.treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), 508)


@unittest.skipIf(not HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
@unittest.skipIf(not HAS_LIGHTGBM, 'Missing LightGBM. Skipping tests.')
class LightGBMTreeBinaryClassifierTest(unittest.TestCase):
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

        # Do conversion - None specified as input means feature names will be taken from mode, which
        # for LightGBM will be ['Column_0', 'Column_1', 'Column_2', 'Column_3']
        self.spec = lightgbm_converter(self.lightgbm_model, None, 'target').get_spec()

    def test_spec_interface(self):
        self.assertIsNotNone(self.spec)

        # Test the model class
        self.assertIsNotNone(self.spec.description)
        self.assertIsNotNone(self.spec.treeEnsembleClassifier)

        # Test the interface class
        self.assertEqual(self.spec.description.predictedFeatureName, 'target')

        # Test the inputs and outputs
        self.assertEqual(len(self.spec.description.output), 2)
        self.assertEqual(self.spec.description.output[0].name, 'target')  # Should this be 'target'?
        self.assertEqual(self.spec.description.output[1].name, 'classProbability')
        self.assertEqual(self.spec.description.output[0].type.WhichOneof('Type'), 'int64Type')
        self.assertEqual(self.spec.description.output[1].type.WhichOneof('Type'), 'dictionaryType')
        self.assertEqual(len(self.spec.description.input), self.num_features)

        for feature_index in range(self.num_features):
            input_type = self.spec.description.input[feature_index]
            self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
            self.assertEqual(input_type.name, 'Column_{}'.format(feature_index))

        # Test actual tree attributes
        tr = self.spec.treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), 508)


@unittest.skipIf(not HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
@unittest.skipIf(not HAS_LIGHTGBM, 'Missing LightGBM. Skipping tests.')
class LightGBMTreeRegressorTest(unittest.TestCase):
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
        self.spec = lightgbm_converter(self.lightgbm_model, ['Column_0', 'Column_1', 'Column_2'], 'target').get_spec()

    def test_spec_interface(self):
        self.assertIsNotNone(self.spec)

        # Test the model class
        self.assertIsNotNone(self.spec.description)
        self.assertIsNotNone(self.spec.treeEnsembleClassifier)

        # Test the interface class
        self.assertEqual(self.spec.description.predictedFeatureName, 'target')

        # Test the inputs and outputs
        self.assertEqual(len(self.spec.description.output), 1)
        self.assertEqual(self.spec.description.output[0].name, 'target')
        self.assertEqual(self.spec.description.output[0].type.WhichOneof('Type'), 'doubleType')
        self.assertEqual(len(self.spec.description.input), self.num_features)

        for feature_index in range(self.num_features):
            input_type = self.spec.description.input[feature_index]
            self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
            self.assertEqual(input_type.name, 'Column_{}'.format(feature_index))

        # Test actual tree attributes
        tr = self.spec.treeEnsembleRegressor.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), 610)


@unittest.skipIf(not HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
@unittest.skipIf(not HAS_LIGHTGBM, 'Missing LightGBM. Skipping tests.')
class LightGBMTreeRegressorNoneFeatureInputTest(unittest.TestCase):
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
        self.spec = lightgbm_converter(self.lightgbm_model, None, 'target2').get_spec()

    def test_spec_interface(self):
        self.assertIsNotNone(self.spec)

        # Test the model class
        self.assertIsNotNone(self.spec.description)
        self.assertIsNotNone(self.spec.treeEnsembleClassifier)

        # Test the interface class
        self.assertEqual(self.spec.description.predictedFeatureName, 'target2')

        # Test the inputs and outputs
        self.assertEqual(len(self.spec.description.output), 1)
        self.assertEqual(self.spec.description.output[0].name, 'target2')
        self.assertEqual(self.spec.description.output[0].type.WhichOneof('Type'), 'doubleType')
        self.assertEqual(len(self.spec.description.input), self.num_features)

        for feature_index in range(self.num_features):
            input_type = self.spec.description.input[feature_index]
            self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
            self.assertEqual(input_type.name, 'Column_{}'.format(feature_index))

        # Test actual tree attributes
        tr = self.spec.treeEnsembleRegressor.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), 610)


@unittest.skipIf(not HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
@unittest.skipIf(not HAS_LIGHTGBM, 'Missing LightGBM. Skipping tests.')
class LightGBMTreeMulticlassClassifierTest(unittest.TestCase):
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

        # Do conversion; feature_names = None means extract features from model; target=None defaults to 'classLabel'
        self.spec = lightgbm_converter(self.lightgbm_model, None, None).get_spec()

    def test_spec_interface(self):
        self.assertIsNotNone(self.spec)

        # Test the model class
        self.assertIsNotNone(self.spec.description)
        self.assertIsNotNone(self.spec.treeEnsembleClassifier)

        # Test the interface class
        self.assertEqual(self.spec.description.predictedFeatureName, 'classLabel')

        # Test the inputs and outputs
        self.assertEqual(len(self.spec.description.output), 2)
        self.assertEqual(self.spec.description.output[0].name, 'classLabel')
        self.assertEqual(self.spec.description.output[1].name, 'classProbability')
        self.assertEqual(self.spec.description.output[0].type.WhichOneof('Type'), 'int64Type')
        self.assertEqual(self.spec.description.output[1].type.WhichOneof('Type'), 'dictionaryType')
        self.assertEqual(len(self.spec.description.input), self.num_features)

        for feature_index in range(self.num_features):
            input_type = self.spec.description.input[feature_index]
            self.assertEqual(input_type.type.WhichOneof('Type'), 'doubleType')
            self.assertEqual(input_type.name, 'Column_{}'.format(feature_index))

        # Test actual tree attributes
        tr = self.spec.treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), 2358)


@unittest.skipIf(not HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
@unittest.skipIf(not HAS_LIGHTGBM, 'Missing LightGBM. Skipping tests.')
class LightGBMTreeMulticlassArrayInputClassifierTest(unittest.TestCase):
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

        # Do conversion. None as target defaults target to 'classLabel'
        self.spec = lightgbm_converter(self.lightgbm_model, 'my_data', 'myClassLabel').get_spec()

    def test_spec_interface(self):
        self.assertIsNotNone(self.spec)

        # Test the model class
        self.assertIsNotNone(self.spec.description)
        self.assertIsNotNone(self.spec.treeEnsembleClassifier)

        # Test the interface class
        self.assertEqual(self.spec.description.predictedFeatureName, 'myClassLabel')

        # Test the inputs and outputs
        self.assertEqual(len(self.spec.description.output), 2)
        self.assertEqual(self.spec.description.output[0].name, 'myClassLabel')
        self.assertEqual(self.spec.description.output[1].name, 'classProbability')
        self.assertEqual(self.spec.description.output[0].type.WhichOneof('Type'), 'int64Type')
        self.assertEqual(self.spec.description.output[1].type.WhichOneof('Type'), 'dictionaryType')
        self.assertEqual(len(self.spec.description.input), 1)

        self.assertEqual(self.spec.description.input[0].type.WhichOneof('Type'), 'multiArrayType')
        self.assertEqual(self.spec.description.input[0].name, 'my_data')

        # Test actual tree attributes
        tr = self.spec.treeEnsembleClassifier.treeEnsemble
        self.assertIsNotNone(tr)
        self.assertEqual(len(tr.nodes), 2358)


if __name__ == '__main__':
    unittest.main()
