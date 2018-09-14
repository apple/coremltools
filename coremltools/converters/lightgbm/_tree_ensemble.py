# Copyright (c) 2018, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from ...models.tree_ensemble import TreeEnsembleRegressor, TreeEnsembleClassifier
from ..._deps import HAS_LIGHTGBM as _HAS_LIGHTGBM

import numpy as _np

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

def recurse_tree(coreml_tree, lgbm_tree_dict, tree_id, node_id, feature_map, current_global_node_id,
                 force_32bit_float):
    """Traverse through the tree and append to the tree spec.
    """
    relative_hit_rate = None

    # print('tree_id = {}, node_id = {}'.format(tree_id, node_id))

    try:
        relative_hit_rate = lgbm_tree_dict['internal_count']
    except KeyError:
        pass

    # Fill node attributes
    if 'leaf_value' not in lgbm_tree_dict:
        decision_type_str = lgbm_tree_dict['decision_type']
        branch_mode = LIGHTGBM_DECISION_TYPE_MAP[decision_type_str]

        split_name = lgbm_tree_dict['split_feature']
        feature_index = split_name if not feature_map else feature_map[split_name]

        # xgboost internally uses float32, but the parsing from json pulls it out
        # as a 64bit double.  To trigger the internal float32 detection in the 
        # tree ensemble compiler, we need to explicitly cast it to a float 32 
        # value, then back to the 64 bit float that protobuf expects.  This is 
        # controlled with the force_32bit_float flag. 
        feature_value = lgbm_tree_dict['threshold']

        if force_32bit_float:
            feature_value = float(_np.float32(feature_value))

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
                         left_child_id, feature_map, current_global_node_id, force_32bit_float)
        if right_child:
            recurse_tree(coreml_tree, lgbm_tree_dict['right_child'], tree_id,
                         right_child_id, feature_map, current_global_node_id, force_32bit_float)

    else:
        value = lgbm_tree_dict['leaf_value']

        if force_32bit_float:
            value = float(_np.float32(value))  

        coreml_tree.add_leaf_node(tree_id, node_id, value, relative_hit_rate = relative_hit_rate)

def convert_tree_ensemble(model, feature_names, target, force_32bit_float):
    """Convert a generic tree model to the protobuf spec.

    This currently supports:
      * Decision tree regression
      * Decision tree classifier

    Parameters
    ----------
    model: str | Booster
        Path on disk where the LightGBM JSON representation of the model is or
        a handle to the LightGBM model.

    feature_names : list of strings or None
        Names of each of the features. When set to None, the feature names are
        extracted from the model.

    target: str,
        Name of the output column.

    force_32bit_float: bool
        If True, then the resulting CoreML model will use 32 bit floats internally.

    Returns
    -------
    model_spec: An object of type Model_pb.
        Protobuf representation of the model
    """
    if not(_HAS_LIGHTGBM):
        raise RuntimeError('xgboost not found. xgboost conversion API is disabled.')
    
    import json
    import os

    if isinstance(model, _lightgbm.Booster):
        lgbm_model_dict = model.dump_model()  # Produces a python dict representing the model

    elif isinstance(model, dict):
        lgbm_model_dict = model

    # Path on the file system where the LightGBM model exists.
    elif isinstance(model, str):
        if not os.path.exists(model):
            raise TypeError("Invalid path %s." % model)
        with open(model) as f:
            lgbm_model_dict = json.load(f)

    feature_names = lgbm_model_dict['feature_names']
    #feature_map = {f:i for i,f in enumerate(feature_names)}

    trees = lgbm_model_dict['tree_info']

    num_classes = lgbm_model_dict['num_class']
    # 1 indicates binary classification (which really means 2 classes)
    if num_classes == 1:
        num_classes = 2

    class_labels = range(num_classes)  # TODO: Find number of output classes

    # coreml_tree = _TreeEnsembleRegressor(feature_names, target)
    #coreml_tree = _TreeEnsembleClassifier(feature_names, target, output_features=('predicted_class', float))
    coreml_tree = TreeEnsembleClassifier(feature_names, class_labels=class_labels, output_features=None)
    coreml_tree.set_default_prediction_value(0.0)  # Correct valid
    for lgbm_tree_id, lgbm_tree_dict in enumerate(trees):
        recurse_tree(coreml_tree, lgbm_tree_dict['tree_structure'], lgbm_tree_id, node_id = 0,
                     feature_map = None, current_global_node_id = [0], force_32bit_float = force_32bit_float)  # TODO: Do we need feature map?

    # LightGBM appears to always use a Logistic transformer for classifiers
    coreml_tree.set_post_evaluation_transform('Regression_Logistic')

    return coreml_tree.spec
