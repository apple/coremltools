# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from ..._deps import _HAS_SKLEARN
from ...models import MLModel as _MLModel
from ._tree_ensemble import convert_tree_ensemble as _convert_tree_ensemble, get_input_dimension

if _HAS_SKLEARN:
    import sklearn.ensemble as _ensemble

    from . import _sklearn_util

    sklearn_class = _ensemble.RandomForestRegressor

model_type = "regressor"


def convert(model, feature_names, target):
    """Convert a boosted tree model to protobuf format.

    Parameters
    ----------
    decision_tree : RandomForestRegressor
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

    _sklearn_util.check_expected_type(model, _ensemble.RandomForestRegressor)

    def is_rf_model(m):
        if len(m.estimators_) == 0:
            return False
        if hasattr(m, "estimators_") and m.estimators_ is not None:
            for t in m.estimators_:
                if not hasattr(t, "tree_") or t.tree_ is None:
                    return False
            return True
        else:
            return False

    _sklearn_util.check_fitted(model, is_rf_model)
    return _MLModel(_convert_tree_ensemble(model, feature_names, target))
