# Created by Caleb Madrigal
# Copyright (c) 2018, FireEye Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from ...models.tree_ensemble import TreeEnsembleRegressor, TreeEnsembleClassifier
from ..._deps import HAS_LIGHTGBM as _HAS_LIGHTGBM

if _HAS_LIGHTGBM:
    import lightgbm as _lightgbm

LIGHTGBM_DECISION_TYPE_MAP = {
    '<=': 'BranchOnValueLessThanEqual',
    '<': 'BranchOnValueLessThan',
    '>=': 'BranchOnValueGreaterThanEqual',
    '>': 'BranchOnValueGreaterThan',
    '=': 'BranchOnValueEqual',
    '!=': 'BranchOnValueNotEqual'
}


def recurse_tree(coreml_tree, lgbm_tree_dict, tree_id, node_id, current_global_node_id, class_id=None):
    """Traverse through the tree and append to the tree spec."""
    relative_hit_rate = None

    try:
        relative_hit_rate = lgbm_tree_dict['internal_count']
    except KeyError:
        pass

    # Branch node
    if 'leaf_value' not in lgbm_tree_dict:
        decision_type_str = lgbm_tree_dict['decision_type']
        branch_mode = LIGHTGBM_DECISION_TYPE_MAP[decision_type_str]

        feature_index = lgbm_tree_dict['split_feature']
        feature_value = lgbm_tree_dict['threshold']

        if 'left_child' in lgbm_tree_dict:
            left_child = lgbm_tree_dict['left_child']
            current_global_node_id[0] += 1
            left_child_id = current_global_node_id[0]
        else:
            left_child = None
            left_child_id = None
        
        if 'right_child' in lgbm_tree_dict:
            right_child = lgbm_tree_dict['right_child']
            current_global_node_id[0] += 1
            right_child_id = current_global_node_id[0]
        else:
            right_child = None
            right_child_id = None

        if lgbm_tree_dict['default_left']:  # If left is the 'true' branch
            (true_child_id, false_child_id) = (left_child_id, right_child_id)
        else:
            (true_child_id, false_child_id) = (right_child_id, left_child_id)

        missing_value_tracks_true_child = True

        coreml_tree.add_branch_node(tree_id, node_id, feature_index,
                                    feature_value, branch_mode, true_child_id, false_child_id,
                                    relative_hit_rate = relative_hit_rate,
                                    missing_value_tracks_true_child = missing_value_tracks_true_child)

        # Recurse
        if left_child:
            recurse_tree(coreml_tree, lgbm_tree_dict['left_child'], tree_id,
                         left_child_id, current_global_node_id, class_id = class_id)
        if right_child:
            recurse_tree(coreml_tree, lgbm_tree_dict['right_child'], tree_id,
                         right_child_id, current_global_node_id, class_id = class_id)

    # Leaf node
    else:
        value = lgbm_tree_dict['leaf_value']
        if class_id:
            value = {class_id: value}

        coreml_tree.add_leaf_node(tree_id, node_id, value, relative_hit_rate = relative_hit_rate)


def _is_classifier(lightgbm_model):
    """Determines if the lightgbm model is a classifier or regressor.
    This is not pretty, but I didn't see a better way to discriminate between the two."""

    if isinstance(lightgbm_model, _lightgbm.LGBMClassifier):
        return True

    elif isinstance(lightgbm_model, _lightgbm.LGBMRegressor):
        return False

    # If lightgbm.basic.Booster, it's more difficult to differentiate between classifiers and regressors...
    regressor_eval_algorithms = {'l1', 'l2', 'l2_root', 'quantile', 'mape', 'huber', 'fair', 'poisson',
                                 'gamma', 'gamma_deviance', 'tweedie'}
    inner_eval_list = set(lightgbm_model._Booster__name_inner_eval)
    # This is a classifier if any of the regressor algorithms are present in the _Booster__name_inner_eval
    return regressor_eval_algorithms & inner_eval_list == set()


def convert_tree_ensemble(model, feature_names, target):
    """Convert a generic tree model to the protobuf spec.

    This currently supports:
      * Classifier
      * Regressor

    Parameters
    ----------
    model: str | lightgbm.basic.Booster | lightgbm.LGBMClassifier | lightgbm.LGBMRegressor
        Lightgbm model object or path on disk to pickled model object.

    feature_names : list of strings or None
        Names of each of the features. When set to None, the feature names are
        extracted from the model.

    target: str,
        Name of the output column.

    Returns
    -------
    model_spec: An object of type Model_pb.
        Protobuf representation of the model
    """
    if not(_HAS_LIGHTGBM):
        raise RuntimeError('lightgbm not found. lightgbm conversion API is disabled.')

    import pickle

    # If str, assume path to pickled model
    if isinstance(model, str):
        with open(model, 'rb') as f:
            model = pickle.load(f)

    if isinstance(model, (_lightgbm.LGBMClassifier, _lightgbm.LGBMRegressor)):
        lgbm_model_dict = model._Booster.dump_model()  # Produces a python dict representing the model

    elif isinstance(model, _lightgbm.Booster):
        lgbm_model_dict = model.dump_model()  # Produces a python dict representing the model

    else:
        raise ValueError('Model object not recognized; must be one of: lightgbm.Booster, lightgbm.LGBMClassifier, '
                         'lightgbm.LGBMRegressor, or string path to pickled model on disk.')

    trees = lgbm_model_dict['tree_info']
    features = lgbm_model_dict['feature_names']

    # Handle classifier model
    if _is_classifier(model):
        # Determine class labels
        num_classes = lgbm_model_dict['num_class']

        # num_class=1 is a special case indicating binary classification (which really means 2 classes)
        if num_classes == 1:
            num_classes = 2

        class_labels = range(num_classes)

        coreml_tree = TreeEnsembleClassifier(features, class_labels=class_labels, output_features=None)

        # LightGBM uses a 0 default_prediction_value
        if num_classes == 2:
            # Binary classification
            coreml_tree.set_default_prediction_value(0.0)

            # LightGBM appears to always use a Logistic transformer for classifiers
            coreml_tree.set_post_evaluation_transform('Regression_Logistic')
        else:
            # Multiclass classification. This is also how we inform the model of the number of classes.
            coreml_tree.set_default_prediction_value([0.0] * num_classes)

            # LightGBM multiclass uses SoftMax
            coreml_tree.set_post_evaluation_transform('Classification_SoftMax')

        # Actually build the tree
        for lgbm_tree_id, lgbm_tree_dict in enumerate(trees):
            if num_classes == 2:
                class_id = None
            else:
                # If multiclass classification, the value needs to indicate which class is being acted upon,
                # so it must be {class_id: value}. In LightGBM, multiclass classification is done as a series
                # of All-vs-One trees. So, for example, if there are 4 classes and 40 trees, the first 10
                # trees represent a binary classification between "Is this Class 0" or "Any other class".
                #
                # LightGBM simply cycles through the classes for each subsequent tree, so if there are 4 classes,
                # tree 0 will be class 0, tree 1 will be class 1, tree 2 will be class 2, tree 3 will be class 3,
                # tree 4 will be class 0, tree 5 will be class 1, tree 6 will be class 2, tree 7 will be class 3,
                # etc.
                class_id = lgbm_tree_id % num_classes

            recurse_tree(coreml_tree, lgbm_tree_dict['tree_structure'], lgbm_tree_id, node_id=0,
                         current_global_node_id=[0], class_id = class_id)

    # Handle regressor model
    else:
        coreml_tree = TreeEnsembleRegressor(feature_names, target)

        # LightGBM uses a 0 default_prediction_value
        coreml_tree.set_default_prediction_value(0.0)

        # LightGBM appears to always use no transform for regressors
        coreml_tree.set_post_evaluation_transform('NoTransform')

        # Actually build the tree
        for lgbm_tree_id, lgbm_tree_dict in enumerate(trees):
            recurse_tree(coreml_tree, lgbm_tree_dict['tree_structure'], lgbm_tree_id, node_id=0,
                         current_global_node_id=[0])

    return coreml_tree.spec
