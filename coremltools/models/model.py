# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy as _deepcopy
import os as _os
import shutil as _shutil
import tempfile as _tempfile
import warnings as _warnings


from ..proto import (
    Model_pb2 as _Model_pb2,
    MIL_pb2 as _MIL_pb2
)
from .utils import (
    _create_mlpackage,
    _has_custom_layer,
    _is_macos,
    _macos_version,
    _MLMODEL_EXTENSION,
    _MLPACKAGE_AUTHOR_NAME,
    _MLPACKAGE_EXTENSION,
    _WEIGHTS_DIR_NAME,
    load_spec as _load_spec,
    save_spec as _save_spec
)
from coremltools import ComputeUnit as _ComputeUnit
from coremltools.converters.mil.mil.program import Program as _Program


try:
    from ..libmodelpackage import ModelPackage as _ModelPackage
except:
    _ModelPackage = None


_MLMODEL_FULL_PRECISION = "float32"
_MLMODEL_HALF_PRECISION = "float16"
_MLMODEL_QUANTIZED = "quantized_model"

_VALID_MLMODEL_PRECISION_TYPES = [
    _MLMODEL_FULL_PRECISION,
    _MLMODEL_HALF_PRECISION,
    _MLMODEL_QUANTIZED,
]

# Linear quantization
_QUANTIZATION_MODE_LINEAR_QUANTIZATION = "_linear_quantization"
# Linear quantization represented as a lookup table
_QUANTIZATION_MODE_LOOKUP_TABLE_LINEAR = "_lookup_table_quantization_linear"
# Lookup table quantization generated by K-Means
_QUANTIZATION_MODE_LOOKUP_TABLE_KMEANS = "_lookup_table_quantization_kmeans"
# Custom lookup table quantization
_QUANTIZATION_MODE_CUSTOM_LOOKUP_TABLE = "_lookup_table_quantization_custom"
# Dequantization
_QUANTIZATION_MODE_DEQUANTIZE = "_dequantize_network"  # used for testing
# Symmetric linear quantization
_QUANTIZATION_MODE_LINEAR_SYMMETRIC = "_linear_quantization_symmetric"

_SUPPORTED_QUANTIZATION_MODES = [
    _QUANTIZATION_MODE_LINEAR_QUANTIZATION,
    _QUANTIZATION_MODE_LOOKUP_TABLE_LINEAR,
    _QUANTIZATION_MODE_LOOKUP_TABLE_KMEANS,
    _QUANTIZATION_MODE_CUSTOM_LOOKUP_TABLE,
    _QUANTIZATION_MODE_DEQUANTIZE,
    _QUANTIZATION_MODE_LINEAR_SYMMETRIC,
]

_LUT_BASED_QUANTIZATION = [
    _QUANTIZATION_MODE_LOOKUP_TABLE_LINEAR,
    _QUANTIZATION_MODE_LOOKUP_TABLE_KMEANS,
    _QUANTIZATION_MODE_CUSTOM_LOOKUP_TABLE,
]

_METADATA_VERSION = "com.github.apple.coremltools.version"
_METADATA_SOURCE = "com.github.apple.coremltools.source"



class _FeatureDescription(object):
    def __init__(self, fd_spec):
        self._fd_spec = fd_spec

    def __repr__(self):
        return "Features(%s)" % ",".join(map(lambda x: x.name, self._fd_spec))

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


def _get_proxy_and_spec(filename, compute_units, skip_model_load=False):
    try:
        from ..libcoremlpython import _MLModelProxy
    except Exception:
        _MLModelProxy = None

    filename = _os.path.expanduser(filename)
    specification = _load_spec(filename)

    if _MLModelProxy and not skip_model_load:

        # check if the version is supported
        engine_version = _MLModelProxy.maximum_supported_specification_version()
        if specification.specificationVersion > engine_version:
            # in this case the specification is a newer kind of .mlmodel than this
            # version of the engine can support so we'll not try to have a proxy object
            return (None, specification, None)

        try:
            return (_MLModelProxy(filename, compute_units.name), specification, None)
        except RuntimeError as e:
            _warnings.warn(
                "You will not be able to run predict() on this Core ML model."
                + " Underlying exception message was: "
                + str(e),
                RuntimeWarning,
            )
            return (None, specification, e)

    return (None, specification, None)

def _try_get_weights_dir_path(mlpackage_path):
    """
    Try to find the weights in mlpackage and return the path to the weights directory if found.
    Return None if not found.
    :param mlpackage_path: str, path to the mlpackage directory
    :return: path to the weights directory inside the mlpackage directory
    """
    weights_dir = None
    try:
        if _ModelPackage.isValid(mlpackage_path):
            item_info = _ModelPackage(mlpackage_path).findItemByNameAuthor(_WEIGHTS_DIR_NAME, _MLPACKAGE_AUTHOR_NAME)
            if item_info is not None:
                weights_dir = item_info.path()
    except:
        pass
    return weights_dir


class MLModel(object):
    """
    This class defines the minimal interface to a CoreML object in Python.

    At a high level, the protobuf specification consists of:

    - Model description: Encodes names and type information of the inputs and outputs to the model.
    - Model parameters: The set of parameters required to represent a specific instance of the model.
    - Metadata: Information about the origin, license, and author of the model.

    With this class, you can inspect a CoreML model, modify metadata, and make
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
        >>> model.input_description
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
        >>> spec = model.get_spec()

        # Save the model
        >>> model.save('HousePricer.mlpackage')

        # Load the model from the spec object
        >>> spec = model.get_spec()
        >>> # modify spec (e.g. rename inputs/ouputs etc)
        >>> model = MLModel(spec)
        >>> # if model type is mlprogram, i.e. spec.WhichOneof('Type') == "mlProgram", then:
        >>> model = MLModel(spec, weights_dir=model.weights_dir)

    See Also
    --------
    predict
    """

    def __init__(self, model,
                 useCPUOnly=False,
                 is_temp_package=False,
                 mil_program=None,
                 skip_model_load=False,
                 compute_units=_ComputeUnit.ALL,
                 weights_dir=None):
        """
        Construct an MLModel from an ``.mlmodel``.

        Parameters
        ----------
        model: str or Model_pb2

            For MLProgram, the model can be a path string (``.mlpackage``) or ``Model_pb2``.
            If its a path string, it must point to a directory containing bundle
            artifacts (such as ``weights.bin``).
            If it is of type ``Model_pb2`` (spec), then ``weights_dir`` must also be provided, if the model
            has weights, since to initialize and load the model, both the proto spec and the weights are
            required. Proto spec for an MLProgram, unlike the NeuralNetwork, does not contain the weights,
            they are stored separately. If the model does not have weights, an empty weights_dir can be provided.

            For non mlprogram model types, the model can be a path string (``.mlmodel``) or type ``Model_pb2``,
            i.e. a spec object.

        useCPUOnly: bool
            This parameter is deprecated and will be removed in 6.0. Use the ``compute_units``
            parameter instead.

            The ``compute_units`` parameter overrides any usages of this parameter.

            Set to True to restrict loading of the model to only the CPU. Defaults to False.

        is_temp_package: bool
            Set to true if the input model package dir is temporary and can be
            deleted upon destruction of this class.

        mil_program: coremltools.converters.mil.Program
            Set to the MIL program object, if available.
            It is available whenever an MLModel object is constructed using
            the unified converter API `coremltools.convert() <https://apple.github.io/coremltools/source/coremltools.converters.mil.html#module-coremltools.converters._converters_entry>`_.

        skip_model_load: bool
            Set to True to prevent coremltools from calling into the Core ML framework
            to compile and load the model. In that case, the returned model object cannot
            be used to make a prediction. This flag may be used to load a newer model
            type on an older Mac, to inspect or load/save the spec.
            
            Example: Loading an ML Program model type on a macOS 11, since an ML Program can be
            compiled and loaded only from macOS12+.
            
            Defaults to False.

        compute_units: coremltools.ComputeUnit
            An enum with three possible values:
                - ``coremltools.ComputeUnit.ALL``: Use all compute units available, including the
                  neural engine.
                - ``coremltools.ComputeUnit.CPU_ONLY``: Limit the model to only use the CPU.
                - ``coremltools.ComputeUnit.CPU_AND_GPU``: Use both the CPU and GPU,
                  but not the neural engine.

        weights_dir: str
            Path to the weight directory, required when loading an MLModel of type mlprogram,
            from a spec object, i.e. when the argument ``model`` is of type ``Model_pb2``

        Notes
        -----
        Internally this maintains the following:

        - ``_MLModelProxy``: A pybind wrapper around
          CoreML::Python::Model (see
          `coremltools/coremlpython/CoreMLPython.mm <https://github.com/apple/coremltools/blob/main/coremlpython/CoreMLPython.mm>`_)

        - ``package_path`` (mlprogram only): Directory containing all artifacts (``.mlmodel``,
          weights, and so on).

        - ``weights_dir`` (mlprogram only): Directory containing weights inside the package_path.

        Examples
        --------
        >>> loaded_model = MLModel('my_model.mlmodel')
        >>> loaded_model = MLModel("my_model.mlpackage")
        """
        if useCPUOnly:
            _warnings.warn('The "useCPUOnly" parameter is deprecated and will be removed in 6.0. '
                           'Use the "compute_units" parameter: "compute_units=coremotools.ComputeUnits.CPUOnly".')
            compute_units = _ComputeUnit.CPU_ONLY
        if not isinstance(compute_units, _ComputeUnit):
            raise TypeError('"compute_units" parameter must be of type: coremltools.ComputeUnit')
        self.compute_unit = compute_units

        self.is_package = False
        self.package_path = None
        self._weights_dir = None
        if mil_program is not None:
            if not isinstance(mil_program, _Program):
                raise ValueError("mil_program must be of type 'coremltools.converters.mil.Program'")
        self._mil_program = mil_program

        if isinstance(model, str):
            if _os.path.isdir(model):
                self.is_package = True
                self.package_path = model
                self.is_temp_package = is_temp_package
                self._weights_dir = _try_get_weights_dir_path(model)
            self.__proxy__, self._spec, self._framework_error = _get_proxy_and_spec(
                model, compute_units, skip_model_load=skip_model_load,
            )
        elif isinstance(model, _Model_pb2.Model):
            if model.WhichOneof('Type') == "mlProgram":
                if weights_dir is None:
                    raise Exception('MLModel of type mlProgram cannot be loaded just from the model spec object. '
                                    'It also needs the path to the weights file. Please provide that as well, '
                                    'using the \'weights_dir\' argument.')
                self.is_package = True
                self.is_temp_package = True
                filename = _create_mlpackage(model, weights_dir, copy_weights=True)
                self.package_path = filename
                self._weights_dir = _try_get_weights_dir_path(filename)
            else:
                filename = _tempfile.mktemp(suffix=_MLMODEL_EXTENSION)
                _save_spec(model, filename)

            self.__proxy__, self._spec, self._framework_error = _get_proxy_and_spec(
                filename, compute_units, skip_model_load=skip_model_load,
            )
            try:
                _os.remove(filename)
            except OSError:
                pass
        else:
            raise TypeError(
                "Expected model to be a .mlmodel file, .mlpackage file or a Model_pb2 object"
            )

        self._input_description = _FeatureDescription(self._spec.description.input)
        self._output_description = _FeatureDescription(self._spec.description.output)


    def __del__(self):
        # Cleanup temporary package upon destruction
        if hasattr(self, 'is_package') and self.is_package \
           and hasattr(self, 'is_temp_package') and self.is_temp_package:
            _shutil.rmtree(self.package_path)

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

    @property
    def version(self):
        return self._spec.description.metadata.versionString

    @property
    def weights_dir(self):
        return self._weights_dir

    @version.setter
    def version(self, version_string):
        self._spec.description.metadata.versionString = version_string

    def __repr__(self):
        return self._spec.description.__repr__()

    def __str__(self):
        return self.__repr__()

    def save(self, filename):
        """
        Save the model to a ``.mlmodel`` format. For an MIL program, the filename is
        a package directory containing the ``mlmodel`` and weights.

        Parameters
        ----------
        filename: str
            Target filename / bundle directory for the model. Must have the
            ``.mlmodel`` extension

        Examples
        --------
        >>> model.save('my_model_file.mlmodel')
        >>> loaded_model = MLModel('my_model_file.mlmodel')
        """
        filename = _os.path.expanduser(filename)
        if self.is_package:
            name, ext = _os.path.splitext(filename)
            if not ext:
                filename = "{}{}".format(filename, _MLPACKAGE_EXTENSION)
            elif ext != _MLPACKAGE_EXTENSION:
                raise Exception("For an ML Program, extension must be {} (not {})".format(_MLPACKAGE_EXTENSION, ext))
            if _os.path.exists(filename):
                _shutil.rmtree(filename)
            _shutil.copytree(self.package_path, filename)
        else:
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


    def predict(self, data, useCPUOnly=False):
        """
        Return predictions for the model.

        Parameters
        ----------
        data: dict[str, value]
            Dictionary of data to make predictions from where the keys are
            the names of the input features.

        useCPUOnly: bool
            This parameter is deprecated and will be removed in 6.0. Instead, use the ``compute_units``
            parameter at load time or conversion time (that is, in
            `coremltools.models.MLModel() <https://apple.github.io/coremltools/source/coremltools.models.html#module-coremltools.models.model>`_ or
            `coremltools.convert() <https://apple.github.io/coremltools/source/coremltools.converters.mil.html#module-coremltools.converters._converters_entry>`_).

            Set to True to restrict computation to use only the CPU. Defaults to False.

        Returns
        -------
        out: dict[str, value]
            Predictions as a dictionary where each key is the output feature
            name.

        Examples
        --------
        >>> data = {'bedroom': 1.0, 'bath': 1.0, 'size': 1240}
        >>> predictions = model.predict(data)
        """
        if useCPUOnly:
            _warnings.warn('The "useCPUOnly" parameter is deprecated and will be removed in 6.0. '
                           'Please use the "compute_units" parameter at load time or conversion time, '
                           'i.e. in "coremltools.models.MLModel()" or "coremltools.convert()".')

        if self.is_package and _is_macos() and _macos_version() < (12, 0):
            raise Exception(
                "predict() for .mlpackage is not supported in macOS version older than 12.0."
            )

        if self.__proxy__:
            # Check if the input name given by the user is valid.
            # Although this is checked during prediction inside CoreML Framework,
            # we still check it here to return early and
            # return a more verbose error message
            self._verify_input_name_exists(data)
            return self.__proxy__.predict(data, useCPUOnly)
        else:
            if _macos_version() < (10, 13):
                raise Exception(
                    "Model prediction is only supported on macOS version 10.13 or later."
                )

            try:
                from ..libcoremlpython import _MLModelProxy
            except Exception as e:
                print("Exception loading model proxy: %s\n" % e)
                _MLModelProxy = None
            except:
                print("Exception while loading model proxy.\n")
                _MLModelProxy = None

            if not _MLModelProxy:
                raise Exception("Unable to load CoreML.framework. Cannot make predictions.")
            elif (
                _MLModelProxy.maximum_supported_specification_version()
                < self._spec.specificationVersion
            ):
                engineVersion = _MLModelProxy.maximum_supported_specification_version()
                raise Exception(
                    "The specification has version "
                    + str(self._spec.specificationVersion)
                    + " but the Core ML framework version installed only supports Core ML model specification version "
                    + str(engineVersion)
                    + " or older."
                )
            elif _has_custom_layer(self._spec):
                raise Exception(
                    "This model contains a custom neural network layer, so predict is not supported."
                )
            else:
                if self._framework_error:
                    raise self._framework_error
                else:
                    raise Exception("Unable to load CoreML.framework. Cannot make predictions.")


    def _set_build_info_mil_attributes(self, metadata):
        assert self.is_package
        ml_program_attributes = self._spec.mlProgram.attributes
        build_info_proto_dict = ml_program_attributes["buildInfo"].immediateValue.dictionary

        # Copy the metadata
        for k, v in metadata.items():
            key_pair = _MIL_pb2.DictionaryValue.KeyValuePair()
            key_pair.key.immediateValue.tensor.strings.values.append(k)
            key_pair.value.immediateValue.tensor.strings.values.append(v)
            build_info_proto_dict.values.append(key_pair)


    def _get_mil_internal(self):
        """
        Get a deep copy of the MIL program object, if available.
        It's available whenever an MLModel object is constructed using
        the unified converter API [`coremltools.convert()`](https://apple.github.io/coremltools/source/coremltools.converters.mil.html#coremltools.converters._converters_entry.convert).

        Returns
        -------
        program: coremltools.converters.mil.Program

        Examples
        ----------
        >>> mil_prog = model._get_mil_internal()
        """
        return _deepcopy(self._mil_program)


    def _verify_input_name_exists(self, input_dict):
        model_input_names = [inp.name for inp in self._spec.description.input]
        model_input_names_set = set(model_input_names)
        for given_input in input_dict.keys():
            if given_input not in model_input_names_set:
                err_msg = "Provided key \"{}\", in the input dict, " \
                          "does not match any of the model input name(s), which are: {}"
                raise KeyError(err_msg.format(given_input, ",".join(model_input_names)))
