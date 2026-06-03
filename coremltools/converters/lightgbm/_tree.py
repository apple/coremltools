# Copyright (c) 2026, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import coremltools as ct
from coremltools import __version__ as ct_version

from ._tree_ensemble import convert_tree_ensemble as _convert_tree_ensemble


def convert(
    model,
    feature_names=None,
    target="target",
    force_32bit_float=True,
    mode="classifier",
    class_labels=None,
    n_classes=None,
):
    """Convert a trained LightGBM model to Core ML format.

    Parameters
    ----------
    model : lightgbm.Booster, lightgbm.LGBMClassifier, or lightgbm.LGBMRegressor
        Trained LightGBM model.

    feature_names : list of str or None
        Names of input features exposed in the Core ML model interface.
        When ``None``, feature names are taken from the model.

    target : str
        Name of the output feature exposed in the Core ML model.

    force_32bit_float : bool
        When ``True``, threshold and leaf values are cast to float32 to match
        LightGBM's internal precision and enable the tree-ensemble compiler's
        float32 optimisation path.

    mode : str in ['regressor', 'classifier']
        Conversion mode. Defaults to ``'classifier'``.

    class_labels : list or None
        Class labels for the classifier output. Defaults to ``range(n_classes)``.

    n_classes : int or None
        Number of output classes. Inferred from the model when not provided.

    Returns
    -------
    model : coremltools.models.MLModel
        Core ML model wrapping a ``TreeEnsembleClassifier`` or
        ``TreeEnsembleRegressor`` spec.

    Notes
    -----
    Models trained with categorical features (LightGBM ``"=="`` splits) are not
    supported and raise ``NotImplementedError``. One-hot encode categorical
    inputs before training to convert such models.

    Examples
    --------
    .. sourcecode:: python

        >>> import coremltools
        >>> coreml_model = coremltools.converters.lightgbm.convert(
        ...     lgbm_model,
        ...     feature_names=feature_names,
        ...     target="walkout_probability",
        ... )
        >>> coreml_model.save("model.mlpackage")
    """
    spec = _convert_tree_ensemble(
        model,
        feature_names,
        target,
        force_32bit_float=force_32bit_float,
        mode=mode,
        class_labels=class_labels,
        n_classes=n_classes,
    )
    coreml_model = ct.models.MLModel(spec)

    try:
        from lightgbm import __version__ as lgbm_version
        source_str = "lightgbm=={0}".format(lgbm_version)
    except Exception:
        source_str = "lightgbm"

    coreml_model.user_defined_metadata[ct.models._METADATA_VERSION] = ct_version
    coreml_model.user_defined_metadata[ct.models._METADATA_SOURCE] = source_str

    return coreml_model
