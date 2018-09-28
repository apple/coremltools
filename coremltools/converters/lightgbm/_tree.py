# Created by Caleb Madrigal
# Copyright (c) 2018, FireEye Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from ._tree_ensemble import convert_tree_ensemble as _convert_tree_ensemble
from ...models import MLModel as _MLModel


def convert(model, feature_names = None, target = 'target'):
    """
    Convert a trained LightGBM model to Core ML format.

    Parameters
    ----------
    model : Booster
        A trained LightGBM tree model.

    feature_names: [str] | str
        Names of input features that will be exposed in the Core ML model
        interface.

        Can be set to one of the following:

        - None for using the feature names from the model.
        - List of names of the input features that should be exposed in the
          interface to the Core ML model. These input features are in the same
          order as the LightGBM model.

    target: str
        Name of the output feature name exposed to the Core ML model.

    Returns
    -------
    model:MLModel
        Returns an MLModel instance representing a Core ML model.

    Examples
    --------
    .. sourcecode:: python

        # Convert it with default input and output names
        >>> import coremltools
        >>> coreml_model = coremltools.converters.lightgbm.convert(model)

        # Saving the Core ML model to a file.
        >>> coreml_model.save('my_model.mlmodel')
    """
    return _MLModel(_convert_tree_ensemble(model, feature_names, target))

