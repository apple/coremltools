# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Utilities for the entire package.
"""

import copy as _copy
import math as _math
import os as _os
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys
import tempfile as _tempfile
import warnings as _warnings
from collections.abc import Iterable as _Iterable
from functools import lru_cache as _lru_cache
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

import numpy as _np

import coremltools as _ct
from coremltools import _SPECIFICATION_VERSION_IOS_16, _SPECIFICATION_VERSION_IOS_18
from coremltools import ComputeUnit as _ComputeUnit
from coremltools import proto as _proto
from coremltools.converters.mil import mil as _mil
from coremltools.converters.mil.frontend.milproto import load as _milproto_to_pymil
from coremltools.converters.mil.mil import Program as _Program
from coremltools.converters.mil.mil.passes.defs.preprocess import NameSanitizer as _NameSanitizer
from coremltools.converters.mil.mil.passes.defs.randomize import (
    WeightRandomizer as _WeightRandomizer,
)
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass as _AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY as _PASS_REGISTRY

from .._deps import _HAS_SCIPY

_MLMODEL_EXTENSION = ".mlmodel"
_MLPACKAGE_EXTENSION = ".mlpackage"
_MODEL_FILE_NAME = 'model.mlmodel'
_WEIGHTS_FILE_NAME = 'weight.bin'
_WEIGHTS_DIR_NAME = 'weights'
_MLPACKAGE_AUTHOR_NAME = "com.apple.CoreML"

try:
    from ..libmodelpackage import ModelPackage as _ModelPackage
except:
    _ModelPackage = None

if _HAS_SCIPY:
    import scipy.sparse as _sp


def _to_unicode(x):
    if isinstance(x, bytes):
        return x.decode()
    else:
        return x


def _remove_invalid_keys(input_dict, model):
    # make sure that input_dict does not contain an input name, which
    # is not present in the list of model inputs
    input_dict_keys = list(input_dict.keys())
    model_input_names = set([inp.name for inp in model._spec.description.input])
    for k in input_dict_keys:
        if k not in model_input_names:
            del input_dict[k]


def _create_mlpackage(
    proto_spec: _proto.Model_pb2,
    weights_dir: _Optional[str] = None,
    package_path: _Optional[str] = None,
) -> str:
    """

    Parameters
    ----------
    proto_spec
        The proto spec of the model.

    weights_dir
        Copy weights from this path to the ``mlpackage``.

    package_path
        Place the created ``mlpackage`` at this path. Error out if this path is a non-empty directory.

    Returns
    -------
    path to the ``mlpackage``.
    """
    if package_path is None:
        package_path = _tempfile.mkdtemp(suffix=_MLPACKAGE_EXTENSION)
    if _os.path.exists(package_path):
        if _os.listdir(package_path):
            raise FileExistsError(
                f"The package_path is invalid because it's a non-empty directory: {package_path}"
            )
        # If package_path is an empty dir, the ModelPackage load will error out with `manifest.json not found` issue.
        _shutil.rmtree(package_path)

    _, ext = _os.path.splitext(package_path)
    if ext != _MLPACKAGE_EXTENSION:
        raise Exception(
            f"For an ML Package, extension must be {_MLPACKAGE_EXTENSION} (not {ext})"
        )

    package = _ModelPackage(package_path)

    # Save proto to disk as the root model file, and copy into the model package.
    spec_file = _tempfile.NamedTemporaryFile(suffix=_MLMODEL_EXTENSION)
    spec_file.write(proto_spec.SerializeToString())
    spec_file.flush()
    package.setRootModel(spec_file.name, _MODEL_FILE_NAME, _MLPACKAGE_AUTHOR_NAME,
                         "CoreML Model Specification")
    # Spec file is auto cleaned after close, which is fine because it is already added to the model package.
    spec_file.close()

    # Add weights bundle into the model package.
    if weights_dir is not None:
        package.addItem(
            weights_dir,
            _WEIGHTS_DIR_NAME,
            _MLPACKAGE_AUTHOR_NAME,
            "CoreML Model Weights",
        )

    return package_path


def save_spec(spec, filename, auto_set_specification_version=False, weights_dir=None):
    """
    Save a protobuf model specification to file.

    Parameters
    ----------
    spec: Model_pb
        Protobuf representation of the model.

    filename: str
        File path where the spec is saved.

    auto_set_specification_version: bool
        If ``True``, will always try to set specification version automatically.

    weights_dir: str
        Path to the directory containing the weights.bin file. This is required
        when the spec has model type ``mlprogram``. If the ``mlprogram`` does not contain
        any weights, this path can be an empty directory.

    Examples
    --------
    .. sourcecode:: python

        coremltools.utils.save_spec(spec, "HousePricer.mlmodel")
        coremltools.utils.save_spec(spec, "HousePricer.mlpackage")
        coremltools.utils.save_spec(
            spec, "mlprogram_model.mlpackage", weights_dir="/path/to/weights/directory"
        )

    See Also
    --------
    load_spec
    """
    name, ext = _os.path.splitext(filename)

    is_package = False

    if not ext:
        filename = "{}{}".format(filename, _MLMODEL_EXTENSION)
    elif ext == _MLPACKAGE_EXTENSION:
        is_package = True
    elif ext == _MLMODEL_EXTENSION:
        is_package = False
    else:
        raise Exception("Extension must be {} or {} (not {})".format(_MLMODEL_EXTENSION, _MLPACKAGE_EXTENSION, ext))

    if auto_set_specification_version:
        try:
            # always try to downgrade the specification version to the
            # minimal version that supports everything in this mlmodel
            from ..libcoremlpython import _MLModelProxy

            spec = _MLModelProxy.auto_set_specification_version(spec)
        except Exception as e:
            print(e)
            _warnings.warn(
                "Failed to automatic set specification version for this model.",
                RuntimeWarning,
            )

    if is_package:
        if _ModelPackage is None:
            raise Exception(
                "Unable to load libmodelpackage. Cannot save spec"
            )
        if spec.WhichOneof('Type') == "mlProgram" and weights_dir is None:
                raise Exception('spec of type mlProgram cannot be saved without the'
                                ' weights file. Please provide the path to the weights file as well, '
                                'using the \'weights_dir\' argument.')
        _create_mlpackage(spec, weights_dir=weights_dir, package_path=filename)
    else:
        with open(filename, "wb") as f:
            f.write(spec.SerializeToString())


def load_spec(model_path: str) -> _proto.Model_pb2:
    """
    Load a protobuf model specification from file (``mlmodel``) or directory (``mlpackage``).

    Parameters
    ----------
    model_path: Path to the model from which the protobuf spec is loaded.

    Returns
    -------
    model_spec: Model_pb
        Protobuf representation of the model.

    Examples
    --------
    .. sourcecode:: python

        spec = coremltools.utils.load_spec("HousePricer.mlmodel")
        spec = coremltools.utils.load_spec("HousePricer.mlpackage")

    See Also
    --------
    save_spec
    """
    if _os.path.isdir(model_path):
        if _ModelPackage is None:
            raise Exception("Unable to load libmodelpackage. Cannot make save spec.")
        specfile = _ModelPackage(model_path).getRootModel().path()
    else:
        specfile = model_path

    spec = _proto.Model_pb2.Model()
    with open(specfile, "rb") as f:
        spec.ParseFromString(f.read())
    return spec


def _get_nn_layers(spec):
    """
    Returns a list of neural network layers if the model contains any.

    Parameters
    ----------
    spec: Model_pb
        A model protobuf specification.

    Returns
    -------
    [NN layer]
        list of all layers (including layers from elements of a pipeline).

    """

    layers = []
    if spec.WhichOneof("Type") == "pipeline":
        layers = []
        for model_spec in spec.pipeline.models:
            if not layers:
                return _get_nn_layers(model_spec)
            else:
                layers.extend(_get_nn_layers(model_spec))

    elif spec.WhichOneof("Type") in ["pipelineClassifier", "pipelineRegressor"]:
        layers = []
        for model_spec in spec.pipeline.models:
            if not layers:
                return _get_nn_layers(model_spec)
            else:
                layers.extend(_get_nn_layers(model_spec))

    elif spec.neuralNetwork.layers:
        layers = spec.neuralNetwork.layers
    elif spec.neuralNetworkClassifier.layers:
        layers = spec.neuralNetworkClassifier.layers
    elif spec.neuralNetworkRegressor.layers:
        layers = spec.neuralNetworkRegressor.layers

    return layers


def _fp32_to_reversed_fp16_byte_array(fp32_arr):
    raw_fp16 = _np.float16(fp32_arr)
    x = ""
    for fp16 in raw_fp16:
        all_bytes = _np.fromstring(fp16.tobytes(), dtype="int8")
        x += all_bytes[1].tobytes()
        x += all_bytes[0].tobytes()
    return x


def _fp32_to_fp16_byte_array(fp32_arr):
    if _np.amax(fp32_arr) >= 65504 or _np.amin(fp32_arr) <= -65504:
        raise Exception(
            "Model cannot be converted as "
            "it has weights that cannot be represented in "
            "half precision.\n"
        )

    if _sys.byteorder == "little":
        return _np.float16(fp32_arr).tobytes()
    else:
        return _fp32_to_reversed_fp16_byte_array(fp32_arr)


def _wp_to_fp16wp(wp):
    assert wp
    # If the float32 field is empty do nothing.
    if len(wp.floatValue) == 0:
        return
    wp.float16Value = _fp32_to_fp16_byte_array(wp.floatValue)
    del wp.floatValue[:]

def _convert_neural_network_spec_weights_to_fp16(fp_spec):
    from .neural_network.quantization_utils import (
        _QUANTIZATION_MODE_LINEAR_QUANTIZATION,
        _quantize_spec_weights,
    )

    qspec = _quantize_spec_weights(fp_spec, 16, _QUANTIZATION_MODE_LINEAR_QUANTIZATION)
    return qspec


def _convert_neural_network_weights_to_fp16(full_precision_model):
    """
    Utility function to convert a full-precision (float) MLModel to a
    half-precision MLModel (float16).

    Parameters
    ----------
    full_precision_model: MLModel
        Model which will be converted to half precision. Currently conversion
        for only neural network models is supported. If a pipeline model is
        passed in, then all embedded neural network models embedded within
        will be converted.

    Returns
    -------
    model: MLModel
        The converted half precision MLModel.

    """
    spec = full_precision_model.get_spec()
    return _get_model(_convert_neural_network_spec_weights_to_fp16(spec))


def _get_model(spec, compute_units=_ComputeUnit.ALL):
    """
    Utility to get the model and the data.
    """
    from . import MLModel

    if isinstance(spec, MLModel):
        return spec
    else:
        return MLModel(spec, compute_units=compute_units)

def evaluate_regressor(model, data, target="target", verbose=False):
    """
    Evaluate a Core ML regression model and compare against predictions
    from the original framework (for testing correctness of conversion).

    Parameters
    ----------
    model: MLModel or str
        A loaded MLModel or a path to a saved MLModel.

    data: Dataframe
        Test data on which to evaluate the models.

    target: str
       Name of the column in the dataframe to be compared against the prediction.

    verbose: bool
       Set to true for a more verbose output.

    See Also
    --------
    evaluate_classifier

    Examples
    --------
    .. sourcecode:: python

        metrics = coremltools.utils.evaluate_regressor(
            spec, "data_and_predictions.csv", "target"
        )
        print(metrics)
        {"samples": 10, "rmse": 0.0, max_error: 0.0}
    """
    model = _get_model(model)

    if verbose:
        print("")
        print("Other Framework\t\tPredicted\t\tDelta")

    max_error = 0
    error_squared = 0

    for _, row in data.iterrows():
        input_dict = dict(row)
        _remove_invalid_keys(input_dict, model)
        predicted = model.predict(input_dict)[_to_unicode(target)]
        other_framework = row[target]
        delta = predicted - other_framework

        if verbose:
            print("{}\t\t\t\t{}\t\t\t{:0.4f}".format(other_framework, predicted, delta))

        max_error = max(abs(delta), max_error)
        error_squared = error_squared + (delta * delta)

    ret = {
        "samples": len(data),
        "rmse": _math.sqrt(error_squared / len(data)),
        "max_error": max_error,
    }

    if verbose:
        print("results: {}".format(ret))
    return ret


def evaluate_classifier(model, data, target="target", verbose=False):
    """
    Evaluate a Core ML classifier model and compare against predictions
    from the original framework (for testing correctness of conversion).
    Use this evaluation for models that don't deal with probabilities.

    Parameters
    ----------
    filename: list of str or list of MLModel
        File to load the model from, or a loaded
        version of the MLModel.

    data: list of str or list of Dataframe
        Test data on which to evaluate the models (dataframe,
        or path to a CSV file).

    target: str
       Column to interpret as the target column.

    verbose: bool
       Set to true for more verbose output.

    See Also
    --------
    evaluate_regressor, evaluate_classifier_with_probabilities

    Examples
    --------
    .. sourcecode:: python

        metrics = coremltools.utils.evaluate_classifier(
            spec, "data_and_predictions.csv", "target"
        )
        print(metrics)
        {"samples": 10, num_errors: 0}
    """
    model = _get_model(model)
    if verbose:
        print("")
        print("Other Framework\t\tPredicted")

    num_errors = 0

    for _, row in data.iterrows():
        input_dict = dict(row)
        _remove_invalid_keys(input_dict, model)
        predicted = model.predict(input_dict)[_to_unicode(target)]
        other_framework = row[target]
        if predicted != other_framework:
            num_errors += 1

        if verbose:
            print("{}\t\t\t\t{}".format(other_framework, predicted))

    ret = {"num_samples": len(data), "num_errors": num_errors}

    if verbose:
        print("results: {}".format(ret))

    return ret


def evaluate_classifier_with_probabilities(
    model, data, probabilities="probabilities", verbose=False
):
    """
    Evaluate a classifier specification for testing.

    Parameters
    ----------
    filename: [str | Model]
        File to load the model from, or a loaded
        version of the MLModel.

    data: [str | Dataframe]
        Test data on which to evaluate the models (dataframe,
        or path to a CSV file).

    probabilities: str
       Column to interpret as the probabilities column.

    verbose: bool
       Verbosity levels of the predictions.
    """

    model = _get_model(model)
    if verbose:
        print("")
        print("Other Framework\t\tPredicted")

    max_probability_error, num_key_mismatch = 0, 0

    for _, row in data.iterrows():
        input_dict = {k: v for k, v in dict(row).items() if k != probabilities}
        _remove_invalid_keys(input_dict, model)
        predicted_values = model.predict(input_dict)[_to_unicode(probabilities)]
        other_values = row[probabilities]

        if set(predicted_values.keys()) != set(other_values.keys()):
            if verbose:
                print(
                    "Different classes: ",
                    str(predicted_values.keys()),
                    str(other_values.keys()),
                )
            num_key_mismatch += 1
            continue

        for cur_class, cur_predicted_class_values in predicted_values.items():
            delta = cur_predicted_class_values - other_values[cur_class]
            if verbose:
                print(delta, cur_predicted_class_values, other_values[cur_class])

            max_probability_error = max(abs(delta), max_probability_error)

        if verbose:
            print("")

    ret = {
        "num_samples": len(data),
        "max_probability_error": max_probability_error,
        "num_key_mismatch": num_key_mismatch,
    }

    if verbose:
        print("results: {}".format(ret))

    return ret


def rename_feature(
    spec, current_name, new_name, rename_inputs=True, rename_outputs=True
):
    """
    Rename a feature in the specification.

    Parameters
    ----------
    spec: Model_pb
        The specification containing the feature to rename.

    current_name: str
        Current name of the feature. If this feature doesn't exist, the rename
        is a no-op.

    new_name: str
        New name of the feature.

    rename_inputs: bool
        Search for ``current_name`` only in the input features (that is, ignore output
        features).

    rename_outputs: bool
        Search for ``current_name`` only in the output features (that is, ignore input
        features).

    Examples
    --------
    .. sourcecode:: python

        # In-place rename of spec
        model = MLModel("model.mlmodel")
        spec = model.get_spec()
        coremltools.utils.rename_feature(spec, "old_feature", "new_feature_name")
        # re-initialize model
        model = MLModel(spec)
        model.save("model.mlmodel")

        # Rename a spec when the model is an mlprogram, in that case, weights are stored outside of the spec
        model = coremltools.convert(torch_model, convert_to="mlprogram")
        spec = model.get_spec()
        # print info about inputs and outputs
        print(spec.description)
        coremltools.utils.rename_feature(spec, "old_feature", "new_feature_name")
        # re-initialize model
        model = MLModel(spec, weights_dir=model.weights_dir)
        model.save("model.mlpackage")
    """

    if not rename_inputs and not rename_outputs:
        return

    changed_input = False
    changed_output = False

    if rename_inputs:
        for input in spec.description.input:
            if input.name == current_name:
                input.name = new_name
                changed_input = True

    if rename_outputs:
        for output in spec.description.output:
            if output.name == current_name:
                output.name = new_name
                changed_output = True

        if spec.description.predictedFeatureName == current_name:
            spec.description.predictedFeatureName = new_name

        if spec.description.predictedProbabilitiesName == current_name:
            spec.description.predictedProbabilitiesName = new_name

    if not changed_input and not changed_output:
        return

    # Rename internally in NN model
    nn = None
    for nn_type in [
        "neuralNetwork",
        "neuralNetworkClassifier",
        "neuralNetworkRegressor",
    ]:
        if spec.HasField(nn_type):
            nn = getattr(spec, nn_type)

    if nn is not None:
        for layer in nn.layers:
            if rename_inputs:
                for index, name in enumerate(layer.input):
                    if name == current_name:
                        layer.input[index] = new_name
                if rename_outputs:
                    for index, name in enumerate(layer.output):
                        if name == current_name:
                            layer.output[index] = new_name

        if rename_inputs:
            for preprocess_params in nn.preprocessing:
                if preprocess_params.featureName == current_name:
                    preprocess_params.featureName = new_name

        if spec.HasField("neuralNetworkClassifier"):
            if nn.labelProbabilityLayerName == current_name:
                nn.labelProbabilityLayerName = new_name

    # Rename internally for feature vectorizer
    if spec.HasField("featureVectorizer") and rename_inputs:
        for input in spec.featureVectorizer.inputList:
            if input.inputColumn == current_name:
                input.inputColumn = new_name
                changed_input = True

    # Rename for pipeline models
    pipeline = None
    if spec.HasField("pipeline"):
        pipeline = spec.pipeline
    elif spec.HasField("pipelineClassifier"):
        pipeline = spec.pipelineClassifier.pipeline
    elif spec.HasField("pipelineRegressor"):
        pipeline = spec.pipelineRegressor.pipeline

    if pipeline is not None:
        for index, model in enumerate(pipeline.models):
            rename_feature(
                model,
                current_name,
                new_name,
                rename_inputs or (index != 0),
                rename_outputs or (index < len(spec.pipeline.models)),
            )

    # Rename for mlProgram
    if spec.HasField("mlProgram"):
        new_name_sanitized = _NameSanitizer().sanitize_name(new_name)
        if new_name != new_name_sanitized:
            raise ValueError("Input/output names for ML Program must be of the format [a-zA-Z_][a-zA-Z0-9_]*. "
                             "That is, it must start with a letter and only contain numerals, underscore or letters. "
                             "Provided feature name, \"{}\" does not satisfy these requirements.".format(new_name))
        mil = spec.mlProgram
        for function in mil.functions.values():
            for name_value_type in function.inputs:
                if name_value_type.name == current_name:
                    name_value_type.name = new_name
            for block in function.block_specializations.values():
                for i, out_name in enumerate(block.outputs):
                    if out_name == current_name:
                        block.outputs[i] = new_name
                for op in block.operations:
                    for argument in op.inputs.values():
                        for binding in argument.arguments:
                            if binding.HasField("name"):
                                if binding.name == current_name:
                                    binding.name = new_name
                    for name_value_type in op.outputs:
                        if name_value_type.name == current_name:
                            name_value_type.name = new_name


def _sanitize_value(x):
    """
    Performs cleaning steps on the data so various type comparisons can
    be performed correctly.
    """
    if isinstance(x, (str, int, float,)):
        return x
    elif _HAS_SCIPY and _sp.issparse(x):
        return x.todense()
    elif isinstance(x, _np.ndarray):
        return x
    elif isinstance(x, tuple):
        return (_sanitize_value(v) for v in x)
    elif isinstance(x, list):
        return [_sanitize_value(v) for v in x]
    elif isinstance(x, dict):
        return dict((_sanitize_value(k), _sanitize_value(v)) for k, v in x.items())
    else:
        assert False, str(x)


def _element_equal(x, y):
    """
    Performs a robust equality test between elements.
    """
    if isinstance(x, _np.ndarray) or isinstance(y, _np.ndarray):
        try:
            return (abs(_np.asarray(x) - _np.asarray(y)) < 1e-5).all()
        except:
            return False
    elif isinstance(x, dict):
        return (
            isinstance(y, dict)
            and _element_equal(x.keys(), y.keys())
            and all(_element_equal(x[k], y[k]) for k in x.keys())
        )
    elif isinstance(x, float):
        return abs(x - y) < 1e-5 * (abs(x) + abs(y))
    elif isinstance(x, (list, tuple)):
        return x == y
    else:
        return bool(x == y)


def evaluate_transformer(model, input_data, reference_output, verbose=False):
    """
    Evaluate a transformer specification for testing.

    Parameters
    ----------
    spec: list of str or list of MLModel
        File to load the Model from, or a loaded
        version of the MLModel.

    input_data: list of dict
        Test data on which to evaluate the models.

    reference_output: list of dict
        Expected results for the model.

    verbose: bool
        Verbosity levels of the predictions.

    Examples
    --------
    .. sourcecode:: python

        input_data = [{"input_1": 1, "input_2": 2}, {"input_1": 3, "input_2": 3}]
        expected_output = [{"input_1": 2.5, "input_2": 2.0}, {"input_1": 1.3, "input_2": 2.3}]
        metrics = coremltools.utils.evaluate_transformer(
            scaler_spec, input_data, expected_output
        )

    See Also
    --------
    evaluate_regressor, evaluate_classifier
    """
    model = _get_model(model)
    if verbose:
        print(model)
        print("")
        print("Other Framework\t\tPredicted")

    num_errors = 0
    for index, row in enumerate(input_data):
        assert isinstance(row, dict)
        sanitized_row = _sanitize_value(row)
        ref_data = _sanitize_value(reference_output[index])
        if verbose:
            print("Input:\n\t", str(row))
            print("Correct output:\n\t", str(ref_data))

        predicted = _sanitize_value(model.predict(sanitized_row))

        assert isinstance(ref_data, dict)
        assert isinstance(predicted, dict)

        predicted_trimmed = dict((k, predicted[k]) for k in ref_data.keys())

        if verbose:
            print("Predicted:\n\t", str(predicted_trimmed))

        if not _element_equal(predicted_trimmed, ref_data):
            num_errors += 1

    ret = {"num_samples": len(input_data), "num_errors": num_errors}

    if verbose:
        print("results: {}".format(ret))
    return ret


def _has_custom_layer(spec):
    """

    Returns true if the given protobuf specification has a custom layer, and false otherwise.

    Parameters
    ----------
    spec: mlmodel spec

    Returns
    -------

    ``True`` if the protobuf specification contains a neural network with a custom layer, ``False`` otherwise.

    """

    layers = _get_nn_layers(spec)
    for layer in layers:
        if layer.WhichOneof("layer") == "custom":
            return True

    return False


def _get_custom_layer_names(spec):
    """

    Returns a list of ``className`` fields which appear in the given protobuf spec.

    Parameters
    ----------
    spec: mlmodel spec

    Returns
    -------
    set(str)
        A set of unique ``className`` fields of custom layers that appear in the model.

    """
    layers = _get_nn_layers(spec)
    layers_out = set()
    for layer in layers:
        if layer.WhichOneof("layer") == "custom":
            layers_out.add(layer.custom.className)

    return layers_out


def _get_custom_layers(spec):
    """

    Returns a list of all neural network custom layers in the spec.

    Parameters
    ----------
    spec: mlmodel spec

    Returns
    -------
    [NN layer]
        A list of custom layer implementations.
    """
    layers = _get_nn_layers(spec)
    layers_out = []
    for layer in layers:
        if layer.WhichOneof("layer") == "custom":
            layers_out.append(layer)

    return layers_out


def _replace_custom_layer_name(spec, oldname, newname):
    """

    Substitutes ``newname`` for ``oldname`` in the ``className`` field of custom layers. If there are no custom layers, or no
    layers with ``className`` = ``oldname``, then the spec is unchanged.

    Parameters
    ----------
    spec: mlmodel spec

    oldname: str
        The custom layer ``className`` to be replaced.

    newname: str
        The new ``className`` value to replace ``oldname``.

    Returns
    -------
    An mlmodel spec.

    """
    layers = _get_custom_layers(spec)
    for layer in layers:
        if layer.custom.className == oldname:
            layer.custom.className = newname


def _is_macos():
    """Returns True if current platform is MacOS, False otherwise."""
    return _sys.platform == "darwin"


@_lru_cache()
def _macos_version():
    """
    Returns macOS version as a tuple of integers, making it easy to do proper
    version comparisons. On non-Macs, it returns an empty tuple.
    """
    if _is_macos():
        try:
            ver_str = _subprocess.run(["sw_vers", "-productVersion"], stdout=_subprocess.PIPE).stdout.decode('utf-8').strip('\n')
            return tuple([int(v) for v in ver_str.split(".")])
        except:
            raise Exception("Unable to determine the macOS version")
    return ()


def _python_version():
    """
    Return python version as a tuple of integers
    """
    version = _sys.version.split(" ")[0]
    version = list(map(int, list(version.split("."))))
    return tuple(version)


def _get_feature(spec, feature_name):
    for input_feature in spec.description.input:
        if input_feature.name == feature_name:
            return input_feature

    for output_feature in spec.description.output:
        if output_feature.name == feature_name:
            return output_feature

    raise Exception("Feature with name {} does not exist".format(feature_name))


def _get_input_names(spec):
    """
    Returns a list of the names of the inputs to this model.
    :param spec: The model protobuf specification
    :return: list of str A list of input feature names
    """
    retval = [feature.name for feature in spec.description.input]
    return retval


def convert_double_to_float_multiarray_type(spec):
    """
    Convert all double multiarrays feature descriptions (input, output, training input)
    to float multiarrays.

    Parameters
    ----------
    spec: Model_pb
        The specification containing the multiarrays types to convert.

    Examples
    --------
    .. sourcecode:: python

        # In-place convert multiarray type of spec
        spec = mlmodel.get_spec()
        coremltools.utils.convert_double_to_float_multiarray_type(spec)
        model = coremltools.models.MLModel(spec)
    """

    def _convert_to_float(feature):
        if feature.type.HasField("multiArrayType"):
            if feature.type.multiArrayType.dataType == _proto.Model_pb2.ArrayFeatureType.DOUBLE:
                feature.type.multiArrayType.dataType = _proto.Model_pb2.ArrayFeatureType.FLOAT32

    for feature in spec.description.input:
        _convert_to_float(feature)

    for feature in spec.description.output:
        _convert_to_float(feature)

    for feature in spec.description.trainingInput:
        _convert_to_float(feature)

    if spec.WhichOneof("Type") == "pipeline":
        for model_spec in spec.pipeline.models:
            convert_double_to_float_multiarray_type(model_spec)


def compile_model(model: _proto.Model_pb2.Model, destination_path: _Optional[str] = None) -> str:
    """
    Compiles a Core ML model spec.

    Parameters
    ----------
    model: Model_pb2
        Spec/protobuf to compile.

        Note: an ``mlprogam`` which uses a blob file is not supported.

    destination_path: str
        Path where the compiled model will be saved.

    Returns
    -------

    str : Path to compiled model directory
        If the ``destination_path`` is specified, that is the value that will be returned.

    Examples
    --------
    .. sourcecode:: python

        from coremltools.models import CompiledMLModel
        from coremltools.models.utils import compile_model
        from coremltools.proto import Model_pb2

        spec = Model_pb2.Model()
        spec.specificationVersion = 1

        input_ = spec.description.input.add()
        input_.name = "x"
        input_.type.doubleType.MergeFromString(b"")

        output_ = spec.description.output.add()
        output_.name = "y"
        output_.type.doubleType.MergeFromString(b"")
        spec.description.predictedFeatureName = "y"

        lr = spec.glmRegressor
        lr.offset.append(0.1)
        weights = lr.weights.add()
        weights.value.append(2.0)

        compiled_model_path = compile_model(spec)
        model = CompiledMLModel(compiled_model_path)
        y = model.predict({"x": 2})

    See Also
    --------
    coremltools.models.CompiledMLModel
    """
    # Check environment
    if _macos_version() < (10, 13):
        raise Exception("Compiling a Core ML models is only support on macOS 10.13 or higher.")
    try:
        from ..libcoremlpython import _MLModelProxy
    except:
        raise Exception("Unable to compile any Core ML models.")

    # Check model parameter
    if isinstance(model, str):
        raise TypeError("To get a compiled model from a saved MLModel, first load the model, "
                        " then call \"get_compiled_model_path\".")
    if isinstance(model, _ct.models.MLModel):
        raise TypeError("This model has already been compiled. Call \"get_compiled_model_path\""
                        " to get the compiled model.")
    if not isinstance(model, _proto.Model_pb2.Model):
        raise TypeError("Unrecognized input for \"model\" parameter. It should be a spec.")

    # Check file extension of destination_path parameter
    if destination_path is not None and not destination_path.rstrip('/').endswith(".mlmodelc"):
        raise Exception("\"destination_path\" parameter must have \".mlmodelc\" file extension.")

    # Compile model
    with _tempfile.TemporaryDirectory() as save_dir:
        spec_file_path = save_dir + '/spec.mlmodel'
        save_spec(model, spec_file_path)
        original_compiled_model_path =  _MLModelProxy.compileModel(spec_file_path)

    # Move the compiled model if needed
    if destination_path is None:
        return original_compiled_model_path
    _shutil.move(original_compiled_model_path, destination_path)
    return destination_path


def make_pipeline(
        *models: '_ct.models.MLModel',
        compute_units: _Union[None, _ct.ComputeUnit] = None
    ) -> '_ct.models.MLModel':
    """
    Makes a pipeline with the given models.

    Parameters
    ----------
    *models :
        Two or more instances of ``ct.models.MLModel``.

    compute_units :
        The set of processing units that all models in the pipeline can use to make predictions.
        Can be ``None`` or ``coremltools.ComputeUnit``.

        * If ``None``, the ``compute_unit`` will be inferred from the ``compute_unit`` values of the models.
          If all models do not have the same ``compute_unit`` values, this parameter must be specified.

        * ``coremltools.ComputeUnit`` is an enum with four possible values:
            - ``coremltools.ComputeUnit.ALL``: Use all compute units available, including the
              neural engine.
            - ``coremltools.ComputeUnit.CPU_ONLY``: Limit the model to only use the CPU.
            - ``coremltools.ComputeUnit.CPU_AND_GPU``: Use both the CPU and GPU,
              but not the neural engine.
            - ``coremltools.ComputeUnit.CPU_AND_NE``: Use both the CPU and neural engine, but
              not the GPU. Available only for macOS >= 13.0.

    Returns
    -------
    ct.models.MLModel

    Examples
    --------
    .. sourcecode:: python

        my_model1 = ct.models.MLModel("/tmp/m1.mlpackage")
        my_model2 = ct.models.MLModel("/tmp/m2.mlmodel")

        my_pipeline_model = ct.utils.make_pipeline(my_model1, my_model2)

        y = my_pipeline_model.predict({"x": 12})

        my_pipeline_model.save("/tmp/my_pipeline.mlpackage")
        new_my_pipeline = ct.model.MLModel("/tmp/my_pipeline.mlpackage")

    """

    def updateBlobFileName(proto_message, new_path):
        if type(proto_message) == _proto.MIL_pb2.Value:
            # Value protobuf message. This is what might need to be updated.
            if proto_message.WhichOneof('value') == 'blobFileValue':
                assert proto_message.blobFileValue.fileName == "@model_path/weights/weight.bin"
                proto_message.blobFileValue.fileName = new_path
        elif hasattr(proto_message, 'ListFields'):
            # Normal protobuf message
            for f in proto_message.ListFields():
                updateBlobFileName(f[1], new_path)
        elif hasattr(proto_message, 'values'):
            # Protobuf map
            for v in proto_message.values():
                updateBlobFileName(v, new_path)
        elif isinstance(proto_message, _Iterable) and not isinstance(proto_message, str):
            # Repeated protobuf message
            for e in proto_message:
                updateBlobFileName(e, new_path)


    assert len(models) > 1
    if compute_units is not None and not isinstance(compute_units, _ComputeUnit):
        raise TypeError('"compute_units" parameter must be None or of type coremltools.ComputeUnit')

    if compute_units is None:
        all_compute_units_the_same = all(map(
            lambda m: models[0].compute_unit is m.compute_unit,
            models[1:]
        ))
        if not all_compute_units_the_same:
            raise ValueError(
                'Models have different compute_unit values. The "compute_units" parameter must be specified.'
            )
        compute_units = models[0].compute_unit

    if (compute_units == _ComputeUnit.CPU_AND_NE
          and _is_macos()
          and _macos_version() < (13, 0)
          ):
        raise ValueError(
            'coremltools.ComputeUnit.CPU_AND_NE is only available on macOS >= 13.0'
        )

    input_specs = list(map(lambda m: m.get_spec(), models))

    pipeline_spec = _ct.proto.Model_pb2.Model()
    pipeline_spec.specificationVersion = max(
        map(lambda spec: spec.specificationVersion, input_specs)
    )

    # If a later model doesn't get an input from a previous model, it must be
    # an input to the pipeline.
    available_as_input = set()
    for cur_spec in input_specs:
        for cur_input in cur_spec.description.input:
            if cur_input.name not in available_as_input:
                pipeline_spec.description.input.add().MergeFrom(cur_input)
                available_as_input.add(cur_input.name)
        available_as_input.update([i.name for i in cur_spec.description.output])

    # If an output for a model is not used as input for a later model, assume it
    # should be an output to the pipeline.
    used_as_input = set()
    for cur_spec in input_specs[::-1]:     # iterate overs specs in reverse
        for cur_output in cur_spec.description.output:
            if cur_output.name not in used_as_input:
                pipeline_spec.description.output.add().MergeFrom(cur_output)
        used_as_input.update([i.name for i in cur_spec.description.input])

    # Map input shapes to output shapes
    var_name_to_type = {}
    for i in range(len(input_specs) - 1):
        for j in input_specs[i + 1].description.input:
            var_name_to_type[j.name] = j.type

        for j in input_specs[i].description.output:
            # If shape is already present, don't override it
            if j.type.WhichOneof('Type') == 'multiArrayType' and len(j.type.multiArrayType.shape) != 0:
                continue

            if j.name in var_name_to_type:
                j.type.CopyFrom(var_name_to_type[j.name])

    # Update each model's spec to have a unique weight filename
    for i, cur_spec in enumerate(input_specs):
        if cur_spec.WhichOneof("Type") == "mlProgram":
            new_file_path = f"@model_path/weights/{i}-weight.bin"
            updateBlobFileName(cur_spec.mlProgram, new_file_path)
        pipeline_spec.pipeline.models.append(cur_spec)

    mlpackage_path = _create_mlpackage(pipeline_spec)
    dst = mlpackage_path + '/Data/' + _MLPACKAGE_AUTHOR_NAME + '/' + _WEIGHTS_DIR_NAME
    _os.mkdir(dst)

    # Copy and rename each model's weight file
    for i, cur_model in enumerate(models):
        if cur_model.weights_dir is not None:
            weight_file_path = cur_model.weights_dir + "/" + _WEIGHTS_FILE_NAME
            if _os.path.exists(weight_file_path):
                _shutil.copyfile(weight_file_path, dst + f"/{i}-weight.bin")

    return _ct.models.MLModel(pipeline_spec, compute_units=compute_units, weights_dir=dst)


def _convert_model_spec_to_pymil_prog(
    mlmodel: "_ct.models.MLModel",
    specification_version: int,
    pymil_load_func: _Callable,
    skip_model_load: bool = False,
) -> _Program:
    """
    A utility that converts an ``mlprogram`` model into PyMIL program.
    """
    model_spec = mlmodel.get_spec()
    model_type = model_spec.WhichOneof("Type")
    if model_type in (
        "neuralNetwork",
        "neuralNetworkClassifier",
        "neuralNetworkRegressor",
        "pipeline",
        "PipelineClassifier",
        "PipelineRegressor",
    ):
        msg = (
            "coremltools.optimize.coreml are meant to be used only with mlprogram typed coreml models. "
            "This model has type {}. Please use coremltools.models.neural_network.quantization_utils.quantize_weights"
            "instead to compress the weights of the model."
        )
        raise TypeError(msg.format(model_type))
    elif model_type == "mlProgram":
        pass
    else:
        raise TypeError("weight compression not applicable for model type {}".format(model_type))

    prog = pymil_load_func(
        model_spec=model_spec,
        specification_version=specification_version,
        file_weights_dir=mlmodel.weights_dir,
        skip_model_load=skip_model_load,
    )
    return prog


def _apply_graph_pass(
    mlmodel: "_ct.models.MLModel",
    graph_pass: _AbstractGraphPass,
    spec_version: int = _SPECIFICATION_VERSION_IOS_16,
    skip_model_load: bool = False,
    pymil_load_func: _Callable = _milproto_to_pymil.load,
    return_pymil_prog: bool = False,
) -> _Union["_ct.models.MLModel", _Program]:
    # We do the lazy import to prevent circular import
    from coremltools.converters.mil.converter import mil_convert as _mil_convert

    # Utility function which compresses a Core ML model
    # Converts the full precision mlmodel into a pymil program
    model_spec = mlmodel.get_spec()
    specification_version = max(model_spec.specificationVersion, spec_version)
    prog = _convert_model_spec_to_pymil_prog(
        mlmodel, specification_version, pymil_load_func, skip_model_load
    )

    # Apply graph pass.
    print(type(graph_pass))
    assert isinstance(
        graph_pass, _AbstractGraphPass
    ), "graph pass must be an AbstractGraphPass instance"
    graph_pass.apply(prog)

    # An early return can prevent running all other optimization paths triggered by _mil_convert.
    if return_pymil_prog:
        return prog

    # Convert the pymil program back to mlmodel
    compressed_mlmodel = _mil_convert(
        prog,
        convert_to="mlprogram",
        convert_from="milinternal",
        specification_version=specification_version,
        compute_units=mlmodel.compute_unit,
        model_description=model_spec.description,
        skip_model_load=skip_model_load,
    )
    return compressed_mlmodel


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
            item_info = _ModelPackage(mlpackage_path).findItemByNameAuthor(
                _WEIGHTS_DIR_NAME, _MLPACKAGE_AUTHOR_NAME
            )
            if item_info is not None:
                weights_dir = item_info.path()
    except:
        pass
    return weights_dir


class MultiFunctionDescriptor:
    """
    The data class defines how to construct a multifunction model from different model sources.
    The users can use the ``add_function`` method to specify the path to the source ``mlpackage``,
    along with the source and target function names.

    After setting the ``default_function_name`` to the ``MultiFunctionDescriptor`` instance,
    a multifunction model can be exported using the ``save_multifunction`` method.

    Examples
    --------
    .. sourcecode:: python

        from coremltools.utils import MultiFunctionDescriptor, save_multifunction

        # Initialize a MultiFunctionDescriptor instance with functions in an existing mlpackage.
        # desc will constain all functions in "my_model.mlpackage"
        desc = MultiFunctionDescriptor("my_model.mlpackage")

        # Construct a MultiFunctionDescriptor instance from scratch.
        # The below code inserts "main" function from "my_model.mlpackage" as "main_1",
        # and inserts "main" function from "my_model_2.mlpackage" as "main_2".
        desc = MultiFunctionDescriptor()
        desc.add_function(
            model_path="my_model.mlpackage",
            source_function_name="main",
            target_function_name="main_1",
        )
        desc.add_function(
            model_path="my_model_2.mlpackage",
            source_function_name="main",
            target_function_name="main_2",
        )

        # Each MultiFunctionDescriptor must has a default function name before saved
        # as a multifunction mlpackage on disk.
        desc.default_function_name = "main_1"
        save_multifunction(desc, "my_multifunction_model.mlpackage")

    See Also
    --------
    save_multifunction

    """

    def __init__(self, model_path: _Optional[str] = None):
        """
        If ``model_path`` is passed to the constructor, it must be a str pointing to a
        mlpackage on disk. The MultiFunctionDescriptor instance will be initiated
        with functions in ``model_path``.
        """
        self._default_function_name = None
        self._name_to_source_function = {}
        self._modelpath_to_functions = {}
        self._modelpath_to_spec = {}

        if model_path is not None:
            self.add_model(model_path)

    def _functions(self) -> _Dict[str, _Tuple[str, str]]:
        """
        Returns ``self._name_to_source_function``
        """
        return _copy.copy(self._name_to_source_function)

    def _add_modelpath_to_cache(self, model_path: str) -> None:
        """
        Given a mlpackage path ``model_path``, the utils caches related metadata.
        """
        if model_path in self._modelpath_to_functions:
            return

        try:
            spec = load_spec(model_path)
        except Exception as err:
            raise ValueError(f"invalid model_path {model_path} with error {err} while loading.")

        desc = spec.description

        # for the iOS17 and below protobuf, there were no functions field,
        # in which "main" is the only function associated with the model.
        if len(desc.functions) == 0:
            self._modelpath_to_functions[model_path] = ["main"]
        else:
            self._modelpath_to_functions[model_path] = [func.name for func in desc.functions]
        self._modelpath_to_spec[model_path] = spec

    @property
    def default_function_name(self) -> _Union[str, None]:
        return self._default_function_name

    @default_function_name.setter
    def default_function_name(self, val: str) -> None:
        if not isinstance(val, str):
            raise ValueError(f"default_function_name must be type of str. Got {val}.")
        self._default_function_name = val

    def add_function(
        self, model_path: str, src_function_name: str, target_function_name: str
    ) -> None:
        """
        Insert ``src_function_name`` function from ``model_path`` as ``target_function_name``
        function in the multifunction descriptor.
        """
        self._add_modelpath_to_cache(model_path)

        if src_function_name not in self._modelpath_to_functions[model_path]:
            raise ValueError(f"src_function_name {src_function_name} not found in {model_path}.")

        if target_function_name in self._name_to_source_function:
            raise ValueError(f"function {target_function_name} already exist.")

        self._name_to_source_function[target_function_name] = (model_path, src_function_name)

    def add_model(self, model_path: str) -> None:
        """
        Insert all functions in ``model_path`` into the multifunction descriptor.
        Same function names in the original model will be applied.
        """
        self._add_modelpath_to_cache(model_path)

        for func_name in self._modelpath_to_functions[model_path]:
            self.add_function(model_path, func_name, func_name)

    def remove_function(self, function_name: str) -> None:
        """
        Remove function ``function_name`` from the multifunction descriptor.
        """
        if function_name not in self._name_to_source_function:
            raise ValueError(f"function_name {function_name} not found.")
        del self._name_to_source_function[function_name]


def save_multifunction(
    desc: MultiFunctionDescriptor,
    destination_path: str,
):
    """
    Save a MultiFunctionDescriptor instance into a multifunction ``mlpackage``.
    The utility also performs constant deduplication across functions to allow weight sharing.

    Parameters
    ----------
    desc : MultiFunctionDescriptor
        Multifunction descriptor to save on the disk.

    destination_path : str
        The saved ``mlpackage`` model path.

    Examples
    --------
    .. sourcecode:: python

        from coremltools.utils import MultiFunctionDescriptor, save_multifunction

        desc = MultiFunctionDescriptor("my_model_1.mlpackage")
        desc.add_function("my_model_2.mlpackage", "main", "main_2")
        desc.default_function_name = "main_2"

        save_multifunction(desc, "multifunctino_model.mlpackage")

    See Also
    --------
    MultiFunctionDescriptor

    """
    # We do the lazy import to prevent circular import
    from coremltools.converters.mil.converter import mil_convert as _mil_convert

    def get_function_spec(
        spec: _proto.Model_pb2, func_name: str
    ) -> _proto.Model_pb2.FunctionDescription:
        """
        Utils to construct a FunctionDescription from the source spec.
        """
        model_desc = spec.description
        # For single function model, we construct the FunctionDescription ourselves
        if len(model_desc.functions) == 0:
            assert func_name == "main", f"invalid function name {func_name}"
            return _proto.Model_pb2.FunctionDescription(
                input=model_desc.input,
                output=model_desc.output,
                state=model_desc.state,
                predictedFeatureName=model_desc.predictedFeatureName,
                predictedProbabilitiesName=model_desc.predictedProbabilitiesName,
            )
        # For multifunction model, we look for the corresponding FunctionDescription
        for func_desc in model_desc.functions:
            if func_desc.name != func_name:
                continue
            res = _proto.Model_pb2.FunctionDescription()
            res.CopyFrom(func_desc)
            res.name = ""
            return res

    # compile model information: spec / weight_dir
    modelpath_to_spec_and_weightdir = {}
    for k, v in desc._name_to_source_function.items():
        model_path = v[0]
        if model_path in modelpath_to_spec_and_weightdir:
            continue
        spec = desc._modelpath_to_spec[model_path]
        weight_dir = _try_get_weights_dir_path(model_path)
        if weight_dir is None:
            raise ValueError(f"weight_dir for model_path {model_path} not found.")
        modelpath_to_spec_and_weightdir[model_path] = (spec, weight_dir)

    # min spec version to support multi-functions model is iOS18
    # we also make the target spec version the max among the input models
    spec_version = max(
        map(lambda val: val[0].specificationVersion, modelpath_to_spec_and_weightdir.values())
    )
    spec_version = max(spec_version, _SPECIFICATION_VERSION_IOS_18)

    # convert spec into pymil program
    modelpath_to_pymil = {}
    for model_path, (spec, weight_dir) in modelpath_to_spec_and_weightdir.items():
        prog = _milproto_to_pymil.load(
            spec,
            spec_version,
            weight_dir,
        )
        modelpath_to_pymil[model_path] = prog

    # construct a multifunction pymil program
    multifunction_prog = _mil.Program()
    function_to_desc = {}
    for target_func_name, v in desc._name_to_source_function.items():
        model_path = v[0]
        src_func_name = v[1]
        prog = modelpath_to_pymil[model_path]
        multifunction_prog.add_function(target_func_name, prog.functions[src_func_name])

        # get the corresponding function description from the spec
        spec = modelpath_to_spec_and_weightdir[model_path][0]
        function_spec = get_function_spec(spec, src_func_name)
        assert function_spec.name == "", "function_spec should not have name set"
        function_spec.name = target_func_name
        function_to_desc[target_func_name] = function_spec

    # Here we deduplicate the same weights across functions, to allow consts to use
    # the same blob file value when lowered into milproto.
    # By weight sharing, we can make the model size as small as we could.
    graph_pass = _PASS_REGISTRY["common::const_deduplication"]
    graph_pass._deduplicate_const_across_functions(multifunction_prog)

    # set default function name
    default_function_name = desc.default_function_name
    if default_function_name is None:
        raise ValueError(
            "default_function_name must be set for the MultiFunctionDescriptor instance before calling save_multifunction."
        )

    if default_function_name not in multifunction_prog.functions:
        raise ValueError(
            f"default_function_name {default_function_name} not found in the program. Available functions names are {list(multifunction_prog.functions.keys())}"
        )
    multifunction_prog.default_function_name = default_function_name

    # export program into multi-functions CoreML model
    functions = []
    for func in multifunction_prog.functions:
        functions.append(function_to_desc[func])
    model_description = _proto.Model_pb2.ModelDescription(
        functions=functions,
        defaultFunctionName=default_function_name,
    )
    multifunction_prog.skip_all_passes = True
    mlmodel = _mil_convert(
        multifunction_prog,
        convert_to="mlprogram",
        convert_from="milinternal",
        specification_version=spec_version,
        compute_units=_ct.ComputeUnit.CPU_ONLY,
        model_description=model_description,
        export_multi_functions=True,
        skip_model_load=True,
    )
    mlmodel.save(destination_path)


def randomize_weights(mlmodel: "_ct.models.MLModel"):
    """
    Utility function to randomize weights

    Parameters
    ----------
    mlmodel: MLModel
        Model which will be randomized.

    Returns
    -------
    model: MLModel
        The MLModel with randomized weights.

    Examples
    --------
    .. sourcecode:: python

        import coremltools as ct

        model = ct.models.MLModel("my_model.mlpackage")
        randomized_mlmodel = ct.models.utils.randomize_weights(mlmodel)

    """

    randomized_mlmodel = _apply_graph_pass(
        mlmodel, graph_pass=_WeightRandomizer(), skip_model_load=True
    )

    return randomized_mlmodel
