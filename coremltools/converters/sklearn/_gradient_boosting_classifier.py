# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from ._tree_ensemble import convert_tree_ensemble as _convert_tree_ensemble
from ._tree_ensemble import get_input_dimension

from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
import numpy as np

if _HAS_SKLEARN:
    import sklearn.ensemble as _ensemble
    from . import _sklearn_util

    sklearn_class = _ensemble.GradientBoostingClassifier

model_type = "classifier"


def convert(model, feature_names, target):
    """Convert a boosted tree model to protobuf format.

    Parameters
    ----------
    decision_tree : GradientBoostingClassifier
        A trained scikit-learn tree model.

    feature_names: [str]
        Name of the input columns.

    target: str
        Name of the output column.

    Returns
    -------
    model_spec: An object of type Model_pb.
        Protobuf representation of the model
    """
    if not (_HAS_SKLEARN):
        raise RuntimeError(
            "scikit-learn not found. scikit-learn conversion API is disabled."
        )

    _sklearn_util.check_expected_type(model, _ensemble.GradientBoostingClassifier)

    def is_gbr_model(m):
        if len(m.estimators_) == 0:
            return False
        if hasattr(m, "estimators_") and m.estimators_ is not None:
            for t in m.estimators_.flatten():
                if not hasattr(t, "tree_") or t.tree_ is None:
                    return False
            return True
        else:
            return False

    _sklearn_util.check_fitted(model, is_gbr_model)
    post_evaluation_transform = None
    if model.n_classes_ == 2:
        post_evaluation_transform = "Regression_Logistic"
    else:
        post_evaluation_transform = "Classification_SoftMax"
    # Here we enumerate known methods GradientBoostingClassifier use for initializing the raw predictions.
    # Alternatively we can enumerate known estimators/strategies combinations.
    # This covers more combinations with less hacks
    base_prediction = None
    dummy_x = np.zeros((1, model.n_features_))
    for base_init_func in ('_init_decision_function', '_raw_predict_init'):
        if not hasattr(model, base_init_func):
            continue
        raw_predictions = getattr(model, base_init_func)(dummy_x)[0, :]
        if '_init_decision_function' == base_init_func and model.n_classes_ > 2:
            # fix initial default prediction for multiclass classification
            # https://github.com/scikit-learn/scikit-learn/pull/12983
            raw_predictions = np.log(raw_predictions)
        base_prediction = list(raw_predictions)
        break
    if base_prediction is None:
        raise ValueError("We don't support your classifier: cannot initialize base_prediction. "
                         "Please file a bug report.")

    return _MLModel(
        _convert_tree_ensemble(
            model,
            feature_names,
            target,
            mode="classifier",
            base_prediction=base_prediction,
            class_labels=model.classes_,
            post_evaluation_transform=post_evaluation_transform,
        )
    )


def supports_output_scores(model):
    return True


def get_output_classes(model):
    return list(model.classes_)
