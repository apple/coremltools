# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy as _deepcopy
import platform as _platform
import sys as _sys
import tempfile as _tempfile

from .utils import save_spec as _save_spec
from ..proto import Model_pb2 as _Model_pb2


class _FeatureDescription(object):
    def __init__(self, fd_spec):
        self._fd_spec = fd_spec

    def __repr__(self):
        return "Features(%s)" % ','.join(map(lambda x: x.name, self._fd_spec))

    def __len__(self):
        return len(self._fd_spec)

    def __getitem__(self, key):
        for f in self._fd_spec:
            if key == f.name:
                return f.shortDescription
        raise KeyError("No feature with name %s." % key)

    def __contains__(self, key):
        for f in self._fd_spec:
            if key == f.name:
                return True
        return False

    def __setitem__(self, key, value):
        for f in self._fd_spec:
            if key == f.name:
                f.shortDescription = value
                return
        raise AttributeError("No feature with name %s." % key)

    def __iter__(self):
        for f in self._fd_spec:
            yield f.name

def _get_proxy_from_spec(filename):
    try:
        from ..libcoremlpython import _MLModelProxy
    except:
        _MLModelProxy = None

    if _MLModelProxy:
        return _MLModelProxy(filename)
    else:
        return None

class MLModel(object):
    """
    This class defines the minimal interface to a CoreML object in Python.

    At a high level, the protobuf specification consists of:

    - Model description: Encodes names and type information of the inputs and outputs to the model.
    - Model parameters: The set of parameters required to represent a specific instance of the model.
    - Metadata: Information about the origin, license, and author of the model.

    With this class, you can inspect a CoreML model, modifiy metadata, and make
    predictions for the purposes of testing (on select platforms).

    Examples
    --------
    .. sourcecode:: python

        # Load the model
        >>> model =  MLModel('HousePricer.mlmodel')

        # Set the model metadata
        >>> model.author = 'Author'
        >>> model.license = 'BSD'
        >>> model.short_description = 'Predicts the price of a house in the Seattle area.'

        # Get the interface to the model
        >>> model.input_descriptions
        >>> model.output_description

        # Set feature descriptions manually
        >>> model.input_description['bedroom'] = 'Number of bedrooms'
        >>> model.input_description['bathrooms'] = 'Number of bathrooms'
        >>> model.input_description['size'] = 'Size (in square feet)'

        # Set
        >>> model.output_description['price'] = 'Price of the house'

        # Make predictions
        >>> predictions = model.predict({'bedroom': 1.0, 'bath': 1.0, 'size': 1240})

        # Get the spec of the model
        >>> model.spec

        # Save the model
        >>> model.save('HousePricer.mlmodel')

    See Also
    --------
    predict
    """
    def __init__(self, model):
        """
        Construct an MLModel from a .mlmodel

        Parameters
        ----------
        model: str | Model_pb2
            If a string is given it should be the location of the .mlmodel to load.

        Examples
        --------
        >>> loaded_model = MLModel('my_model_file.mlmodel')
        """
        from .utils import load_spec as _load_spec

        if isinstance(model, str):
            self._spec = _load_spec(model)
            self.__proxy__ = _get_proxy_from_spec(model)
        elif isinstance(model, _Model_pb2.Model):
            self._spec = model
            filename = _tempfile.mktemp(suffix = '.mlmodel')
            _save_spec(model, filename)
            self.__proxy__ = _get_proxy_from_spec(filename)
        else:
            raise TypeError("Expected model to be a .mlmodel file or a Model_pb2 object")

        self._input_description = _FeatureDescription(self._spec.description.input)
        self._output_description = _FeatureDescription(self._spec.description.output)

    @property
    def short_description(self):
        return self._spec.description.metadata.shortDescription

    @short_description.setter
    def short_description(self, short_description):
        self._spec.description.metadata.shortDescription = short_description

    @property
    def input_description(self):
        return self._input_description

    @property
    def output_description(self):
        return self._output_description

    @property
    def user_defined_metadata(self):
        return self._spec.description.metadata.userDefined

    @property
    def author(self):
        return self._spec.description.metadata.author

    @author.setter
    def author(self, author):
        self._spec.description.metadata.author = author

    @property
    def license(self):
        return self._spec.description.metadata.license

    @license.setter
    def license(self, license):
        self._spec.description.metadata.license = license

    def __repr__(self):
        return self._spec.description.__repr__()

    def __str__(self):
        return self.__repr__()

    def save(self, filename):
        """
        Save the model to a .mlmodel format.

        Parameters
        ----------
        location : str
            Target filename for the model.

        See Also
        --------
        coremltools.utils.load_model

        Examples
        --------
        >>> model.save('my_model_file.mlmodel')
        >>> loaded_model = MLModel('my_model_file.mlmodel')
        """
        _save_spec(self._spec, filename)

    def get_spec(self):
        """
        Get a deep copy of the protobuf specification of the model.

        Returns
        -------
        model: Model_pb2
            Protobuf specification of the model.

        Examples
        ----------
        >>> spec = model.get_spec()
        """
        return _deepcopy(self._spec)

    def predict(self, data, useCPUOnly=False, **kwargs):
        """
        Return predictions for the model. The kwargs gets passed into the
        model as a dictionary.

        Parameters
        ----------
        data : dict[str, value]
            Dictionary of data to make predictions from where the keys are
            the names of the input features.

        useCPUOnly : bool
            Set to true to restrict computation to use only the CPU. Defaults to False.

        Returns
        -------
        out : dict[str, value]
            Predictions as a dictionary where each key is the output feature
            name.

        Examples
        --------
        >>> data = {'bedroom': 1.0, 'bath': 1.0, 'size': 1240}
        >>> predictions = model.predict(data)
        """
        if self.__proxy__:
            return self.__proxy__.predict(data,useCPUOnly)
        else:
            if _sys.platform != 'darwin' or float('.'.join(_platform.mac_ver()[0].split('.')[:2])) < 10.13:
                raise Exception('Model prediction is only supported on macOS version 10.13.')
            else:
                raise Exception('Unable to load CoreML.framework. Cannot make predictions.')
