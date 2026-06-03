# Copyright (c) 2026, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as _np

from ..._deps import _HAS_LIGHTGBM
from ...models.tree_ensemble import TreeEnsembleClassifier
from ...models.tree_ensemble import TreeEnsembleRegressor as _TreeEnsembleRegressor

if _HAS_LIGHTGBM:
    import lightgbm as _lightgbm


def _recurse_node(
    mlkit_tree,
    lgb_node,
    tree_id,
    node_id,
    force_32bit_float,
    mode,
    tree_index,
    n_classes,
    id_counter,
    leaf_scale,
):
    """Recursively traverse a LightGBM tree node and append to the CoreML tree spec.

    LightGBM uses a nested dict structure (left_child/right_child) rather than
    XGBoost's flat nodeid references, so we assign node IDs ourselves via id_counter.
    id_counter is a one-element list mutated in place to share state across recursive calls.

    leaf_scale multiplies every leaf value before it is written. It is the binary
    objective's ``sigmoid`` coefficient (1.0 otherwise): because
    ``1 / (1 + exp(-s * sum(leaf))) == 1 / (1 + exp(-sum(s * leaf)))``, pre-scaling
    the leaves lets the fixed Regression_Logistic transform reproduce LightGBM's
    probabilities exactly.
    """
    relative_hit_rate = lgb_node.get("internal_weight") or lgb_node.get("leaf_weight")

    if "split_feature" in lgb_node:
        # Branch node
        decision_type = lgb_node.get("decision_type", "<=")
        if decision_type != "<=":
            # LightGBM uses "==" for categorical splits, where the threshold is a
            # "1||3"-style set of category values rather than a scalar. Core ML's
            # tree ensemble cannot represent set membership in a single branch, so
            # fail loudly instead of crashing on float(threshold) downstream.
            raise NotImplementedError(
                "LightGBM categorical splits (decision_type '%s') are not supported "
                "by the Core ML converter. Train without categorical_feature, or "
                "one-hot encode categorical inputs before training." % decision_type
            )
        feature_index = lgb_node["split_feature"]
        feature_value = lgb_node["threshold"]
        if force_32bit_float:
            feature_value = float(_np.float32(feature_value))

        # LightGBM decision_type is "<=" for numerical splits → BranchOnValueLessThanEqual
        branch_mode = "BranchOnValueLessThanEqual"

        # Assign child IDs now (we claim them from id_counter)
        true_child_id = id_counter[0]
        id_counter[0] += 1
        false_child_id = id_counter[0]
        id_counter[0] += 1

        # default_left=True means missing values go to left child (the true branch)
        missing_value_tracks_true_child = lgb_node.get("default_left", False)

        mlkit_tree.add_branch_node(
            tree_id,
            node_id,
            feature_index,
            feature_value,
            branch_mode,
            true_child_id,
            false_child_id,
            relative_hit_rate=relative_hit_rate,
            missing_value_tracks_true_child=missing_value_tracks_true_child,
        )

        # Recurse into children using the pre-assigned IDs
        _recurse_node(
            mlkit_tree,
            lgb_node["left_child"],
            tree_id,
            true_child_id,
            force_32bit_float,
            mode,
            tree_index,
            n_classes,
            id_counter,
            leaf_scale,
        )
        _recurse_node(
            mlkit_tree,
            lgb_node["right_child"],
            tree_id,
            false_child_id,
            force_32bit_float,
            mode,
            tree_index,
            n_classes,
            id_counter,
            leaf_scale,
        )
    else:
        # Leaf node
        value = lgb_node["leaf_value"]
        if leaf_scale != 1.0:
            value = value * leaf_scale
        if force_32bit_float:
            value = float(_np.float32(value))

        if mode == "classifier" and n_classes > 2:
            value = {tree_index: value}

        mlkit_tree.add_leaf_node(
            tree_id, node_id, value, relative_hit_rate=relative_hit_rate
        )


def convert_tree_ensemble(
    model,
    feature_names,
    target,
    force_32bit_float,
    mode="classifier",
    class_labels=None,
    n_classes=None,
):
    """Convert a LightGBM model to a CoreML protobuf tree ensemble spec.

    Parameters
    ----------
    model : lightgbm.Booster or lightgbm.LGBMClassifier or lightgbm.LGBMRegressor
        Trained LightGBM model.
    feature_names : list of str or None
        Names of input features. When None, taken from the model.
    target : str
        Name of the output feature in the CoreML model.
    force_32bit_float : bool
        Cast threshold/leaf values to float32 to match LightGBM's internal precision.
    mode : str in ['regressor', 'classifier']
        Conversion mode.
    class_labels : list or None
        Class label list for classifiers. Defaults to ``range(n_classes)``.
    n_classes : int or None
        Number of classes. Inferred from model when not provided.

    Returns
    -------
    spec : Model_pb2.Model
        Protobuf spec ready to wrap in ``coremltools.models.MLModel``.
    """
    if not _HAS_LIGHTGBM:
        raise RuntimeError(
            "LightGBM not found. LightGBM conversion API is disabled."
        )
    accepted_modes = ["regressor", "classifier"]
    if mode not in accepted_modes:
        raise ValueError("mode must be one of %s" % accepted_modes)

    # ------------------------------------------------------------------ #
    # Resolve model → Booster and feature_names                           #
    # ------------------------------------------------------------------ #
    if isinstance(model, (_lightgbm.LGBMClassifier, _lightgbm.LGBMRegressor)):
        if isinstance(model, _lightgbm.LGBMClassifier) and mode == "regressor":
            raise ValueError("mode is 'regressor' but a LGBMClassifier was provided.")
        if isinstance(model, _lightgbm.LGBMRegressor) and mode == "classifier":
            raise ValueError("mode is 'classifier' but a LGBMRegressor was provided.")

        booster = model.booster_
        if feature_names is None:
            feature_names = booster.feature_name()

        if isinstance(model, _lightgbm.LGBMClassifier) and n_classes is None:
            n_classes = model.n_classes_

    elif isinstance(model, _lightgbm.Booster):
        booster = model
        if feature_names is None:
            feature_names = booster.feature_name()
    else:
        raise TypeError(
            "Unexpected model type %s. "
            "Expected lightgbm.Booster, LGBMClassifier, or LGBMRegressor." % type(model)
        )

    # ------------------------------------------------------------------ #
    # Dump model to dict                                                   #
    # ------------------------------------------------------------------ #
    dump = booster.dump_model()

    if n_classes is None:
        # num_tree_per_iteration == 1 for binary; > 1 for multiclass
        n_classes = max(dump.get("num_tree_per_iteration", 1), 1)
        if n_classes == 1:
            # binary classification or regression — treat as 2-class for classifier
            if mode == "classifier":
                n_classes = 2

    # ------------------------------------------------------------------ #
    # Build CoreML tree ensemble                                           #
    # ------------------------------------------------------------------ #
    if mode == "classifier":
        if class_labels is None:
            class_labels = list(range(n_classes))
        if len(class_labels) != n_classes:
            raise ValueError(
                "Number of classes in model (%d) does not match "
                "length of class_labels (%d)." % (n_classes, len(class_labels))
            )

        base_prediction = [0.0] if n_classes == 2 else [0.0] * n_classes
        mlkit_tree = TreeEnsembleClassifier(feature_names, class_labels, target)
        mlkit_tree.set_default_prediction_value(base_prediction)
        if n_classes == 2:
            mlkit_tree.set_post_evaluation_transform("Regression_Logistic")
        else:
            mlkit_tree.set_post_evaluation_transform("Classification_SoftMax")
    else:
        mlkit_tree = _TreeEnsembleRegressor(feature_names, target)
        mlkit_tree.set_default_prediction_value(0.0)

    num_tree_per_iter = dump.get("num_tree_per_iteration", 1)

    # LightGBM's binary objective applies a sigmoid coefficient s:
    # proba = 1 / (1 + exp(-s * raw_score)). The dump records it in the objective
    # string, e.g. "binary sigmoid:2". Core ML's Regression_Logistic transform has
    # no such coefficient, so we fold s into the leaf values (see _recurse_node).
    # It only applies to the binary classifier path; multiclass uses softmax.
    leaf_scale = 1.0
    if mode == "classifier" and n_classes == 2:
        objective = dump.get("objective", "") or ""
        for token in objective.split():
            if token.startswith("sigmoid:"):
                try:
                    leaf_scale = float(token.split(":", 1)[1])
                except ValueError:
                    leaf_scale = 1.0

    for tree_id, tree_info in enumerate(dump["tree_info"]):
        tree_index = tree_id % num_tree_per_iter  # class index for multiclass
        root = tree_info["tree_structure"]
        # id_counter[0] is the next free node ID within this tree.
        # Root always gets 0; children are assigned in pre-order.
        id_counter = [1]
        _recurse_node(
            mlkit_tree,
            root,
            tree_id,
            node_id=0,
            force_32bit_float=force_32bit_float,
            mode=mode,
            tree_index=tree_index,
            n_classes=n_classes,
            id_counter=id_counter,
            leaf_scale=leaf_scale,
        )

    return mlkit_tree.spec
