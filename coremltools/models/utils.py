# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Utilities for the entire package.
"""

from collections.abc import Iterable as _Iterable
from functools import lru_cache as _lru_cache
import math as _math
import os as _os
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys
import tempfile as _tempfile
from typing import Optional as _Optional, Union as _Union
import warnings as _warnings

import numpy as _np

import coremltools as _ct
from coremltools import ComputeUnit as _ComputeUnit
from coremltools.converters.mil.mil.passes.defs.preprocess import NameSanitizer as _NameSanitizer
from coremltools.proto import Model_pb2 as _Model_pb2
import coremltools.proto.MIL_pb2 as _mil_proto

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
    proto_spec: _Model_pb2,
    weights_dir: _Optional[str] = None,
    package_path: _Optional[str] = None,
) -> str:
    """
    Args:
        proto_spec: The proto spec of the model.
        weights_dir: Copy weights from this path to the mlpackage.
        package_path: Place the created mlpackage at this path. Error out if this path is a non-empty directory.

    Returns:
        path to the mlpackage
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
        Protobuf representation of the model

    filename: str
        File path where the spec gets saved.

    auto_set_specification_version: bool
        If True, will always try to set specification version automatically.

    weights_dir: str
        Path to the directory containing the weigths.bin file. This is required
        when the spec if of model type mlprogram. If the mlprogram does not contain
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


def load_spec(model_path: str) -> _Model_pb2:
    """
    Load a protobuf model specification from file (mlmodel) or directory (mlpackage).

    Parameters
    ----------
    model_path: Path to the model from which the protobuf spec is loaded.

    Returns
    -------
    model_spec: Model_pb
        Protobuf representation of the model

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

    spec = _Model_pb2.Model()
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
        list of all layers (including layers from elements of a pipeline

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
        _QUANTIZATION_MODE_LINEAR_QUANTIZATION, _quantize_spec_weights)

    qspec = _quantize_spec_weights(fp_spec, 16, _QUANTIZATION_MODE_LINEAR_QUANTIZATION)
    return qspec


def _convert_neural_network_weights_to_fp16(full_precision_model):
    """
    Utility function to convert a full precision (float) MLModel to a
    half precision MLModel (float16).

    Parameters
    ----------
    full_precision_model: MLModel
        Model which will be converted to half precision. Currently conversion
        for only neural network models is supported. If a pipeline model is
        passed in then all embedded neural network models embedded within
        will be converted.

    Returns
    -------
    model: MLModel
        The converted half precision MLModel

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
    Evaluate a CoreML regression model and compare against predictions
    from the original framework (for testing correctness of conversion).

    Parameters
    ----------
    model: MLModel or str
        A loaded MLModel or a path to a saved MLModel

    data: Dataframe
        Test data on which to evaluate the models

    target: str
       Name of the column in the dataframe to be compared against the prediction

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
        File from where to load the model from (OR) a loaded
        version of the MLModel.

    data: list of str or list of Dataframe
        Test data on which to evaluate the models (dataframe,
        or path to a csv file).

    target: str
       Column to interpret as the target column

    verbose: bool
       Set to true for a more verbose output.

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
        File from where to load the model from (OR) a loaded
        version of the MLModel.

    data: [str | Dataframe]
        Test data on which to evaluate the models (dataframe,
        or path to a csv file).

    probabilities: str
       Column to interpret as the probabilities column

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
        Search for `current_name` only in the input features (i.e ignore output
        features)

    rename_outputs: bool
        Search for `current_name` only in the output features (i.e ignore input
        features)

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
        File from where to load the Model from (OR) a loaded
        version of MLModel.

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

    True if the protobuf specification contains a neural network with a custom layer, False otherwise.

    """

    layers = _get_nn_layers(spec)
    for layer in layers:
        if layer.WhichOneof("layer") == "custom":
            return True

    return False


def _get_custom_layer_names(spec):
    """

    Returns a list of className fields which appear in the given protobuf spec

    Parameters
    ----------
    spec: mlmodel spec

    Returns
    -------

    set(str) A set of unique className fields of custom layers that appear in the model.

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

    [NN layer] A list of custom layer implementations
    """
    layers = _get_nn_layers(spec)
    layers_out = []
    for layer in layers:
        if layer.WhichOneof("layer") == "custom":
            layers_out.append(layer)

    return layers_out


def _replace_custom_layer_name(spec, oldname, newname):
    """

    Substitutes newname for oldname in the className field of custom layers. If there are no custom layers, or no
    layers with className=oldname, then the spec is unchanged.

    Parameters
    ----------
    spec: mlmodel spec

    oldname: str The custom layer className to be replaced.

    newname: str The new className value to replace oldname

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
            raise Exception("Unable to detemine the macOS version")
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
    to float multiarrays

    Parameters
    ----------
    spec: Model_pb
        The specification containing the multiarrays types to convert

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
            if (
                feature.type.multiArrayType.dataType
                == _Model_pb2.ArrayFeatureType.DOUBLE
            ):
                feature.type.multiArrayType.dataType = (
                    _Model_pb2.ArrayFeatureType.FLOAT32
                )

    for feature in spec.description.input:
        _convert_to_float(feature)

    for feature in spec.description.output:
        _convert_to_float(feature)

    for feature in spec.description.trainingInput:
        _convert_to_float(feature)

    if spec.WhichOneof("Type") == "pipeline":
        for model_spec in spec.pipeline.models:
            convert_double_to_float_multiarray_type(model_spec)


def compile_model(model: _Union['_ct.models.MLModel', str, _Model_pb2.Model]) -> str:
    """
    Compiles a Core ML model.

    Parameters
    ----------
    model: str, Model_pb2 or MLModel

        str : Path to model to compile

        Model_pb2 : Spec to model to compile

        MLModel : Instantiated Core ML model to compile

    Returns
    -------

    str : Path to compiled model directory

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

    # Check parameter
    if not isinstance(model, (str, _Model_pb2.Model, _ct.models.MLModel)):
        raise Exception("Compiling a Core ML models is only support on macOS 10.13 or higher.")

    # Compile model
    if isinstance(model, (_Model_pb2.Model, _ct.models.MLModel)):
        if isinstance(model, _ct.models.MLModel):
            spec = model.get_spec()
        else:
            spec = model

        with _tempfile.TemporaryDirectory() as save_dir:
            spec_file_path = save_dir + '/spec.mlmodel'
            save_spec(spec, spec_file_path)
            return _MLModelProxy.compileModel(spec_file_path)
    else:
        assert isinstance(model, str)
        model = _os.path.expanduser(model)
        return _MLModelProxy.compileModel(model)


def make_pipeline(*models):
    """
    Makes a pipeline with the given models.

    Parameters
    ----------
    *models
        Two or more instances of ct.models.MLModel.

    Returns
    -------
    ct.models.MLModel

    Examples
    --------
    .. sourcecode:: python

        my_model1 = ct.models.MLModel('/tmp/m1.mlpackage')
        my_model2 = ct.models.MLModel('/tmp/m2.mlmodel')
        
        my_pipeline_model = ct.utils.make_pipeline(my_model1, my_model2)

    """

    def updateBlobFileName(proto_message, new_path):
        if type(proto_message) == _mil_proto.Value:
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
    input_specs = list(map(lambda m: m.get_spec(), models))

    pipeline_spec = _ct.proto.Model_pb2.Model()
    pipeline_spec.specificationVersion = max(
        map(lambda spec: spec.specificationVersion, input_specs)
    )

    # Set pipeline input
    pipeline_spec.description.input.MergeFrom(
        input_specs[0].description.input
    )

    # Set pipeline output
    pipeline_spec.description.output.MergeFrom(
        input_specs[-1].description.output
    )

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

    return _ct.models.MLModel(pipeline_spec, weights_dir=dst)
