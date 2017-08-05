# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Utilities for the entire package.
"""
import json as _json
import math as _math
import numpy as _np
import os as _os
import six as _six

from .._deps import HAS_SKLEARN as _HAS_SKLEARN


if _HAS_SKLEARN:
    import scipy.sparse as _sp


def save_spec(spec, filename):
    """
    Save a protobuf model specification to file.

    Parameters
    ----------
    spec: Model_pb
        Protobuf representation of the model

    filename: str
        File path  where the spec gets saved.

    Examples
    --------
    .. sourcecode:: python

        >>> coremltools.utils.save_spec(spec, 'HousePricer.mlmodel')

    See Also
    --------
    load_spec
    """
    name, ext = _os.path.splitext(filename)
    if not ext:
        filename = "%s.mlmodel" % filename
    else:
        if ext != '.mlmodel':
            raise Exception("Extension must be .mlmodel (not %s)" % ext)

    with open(filename, 'wb') as f:
        s = spec.SerializeToString()
        f.write(s)

def load_spec(filename):
    """
    Load a protobuf model specification from file

    Parameters
    ----------
    filename: str
        Location on disk (a valid filepath) from which the file is loaded
        as a protobuf spec.

    Returns
    -------
    model_spec: Model_pb
        Protobuf representation of the model

    Examples
    --------
    .. sourcecode:: python

        >>> spec = coremltools.utils.load_spec('HousePricer.mlmodel')

    See Also
    --------
    save_spec
    """
    from ..proto import Model_pb2
    spec = Model_pb2.Model()

    with open(filename, 'rb') as f:
        contents = f.read()
        spec.ParseFromString(contents)
        return spec

def _get_model(spec):
    """
    Utility to get the model and the data.
    """
    from . import MLModel
    if isinstance(spec, MLModel):
        return spec
    else:
        return MLModel(spec)


def evaluate_regressor(model, data, target = "target", verbose = False):
    """
    Evaluate a CoreML regression model and compare against predictions
    from the original framework (for testing correctness of conversion)

    Parameters
    ----------
    filename: [str | MLModel]
        File path from which to load the MLModel from (OR) a loaded version of
        MLModel.

    data: [str | Dataframe]
        Test data on which to evaluate the models (dataframe,
        or path to a .csv file).

    target: str
       Name of the column in the dataframe that must be interpreted
       as the target column.

    verbose: bool
       Set to true for a more verbose output.

    See Also
    --------
    evaluate_classifier

    Examples
    --------
    .. sourcecode:: python

        >>> metrics =  coremltools.utils.evaluate_regressor(spec, 'data_and_predictions.csv', 'target')
        >>> print(metrics)
        {"samples": 10, "rmse": 0.0, max_error: 0.0}
    """
    model = _get_model(model)

    if verbose:
        print("")
        print("Other Framework\t\tPredicted\t\tDelta")

    max_error = 0
    error_squared = 0

    for index,row in data.iterrows():
        predicted = model.predict(dict(row))[unicode(target)]
        other_framework = row["prediction"]
        delta = predicted - other_framework

        if verbose:
            print("%s\t\t\t\t%s\t\t\t%0.4f" % (other_framework, predicted, delta))

        max_error = max(abs(delta), max_error)
        error_squared = error_squared + (delta * delta)

    ret = {
        "samples": len(data),
        "rmse": _math.sqrt(error_squared / len(data)),
        "max_error": max_error
    }

    if verbose:
        print("results: %s" % ret)
    return ret

def evaluate_classifier(model, data, target = 'target', verbose = False):
    """
    Evaluate a CoreML classifier model and compare against predictions
    from the original framework (for testing correctness of conversion). Use
    this evaluation for models that don't deal with probabilities.

    Parameters
    ----------
    filename: [str | MLModel]
        File from where to load the model from (OR) a loaded
        version of the MLModel.

    data: [str | Dataframe]
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

        >>> metrics =  coremltools.utils.evaluate_classifier(spec, 'data_and_predictions.csv', 'target')
        >>> print(metrics)
        {"samples": 10, num_errors: 0}
    """
    model = _get_model(model)
    if verbose:
        print("")
        print("Other Framework\t\tPredicted")

    num_errors = 0

    for index,row in data.iterrows():
        predicted = model.predict(dict(row))[unicode(target)]
        other_framework = row["prediction"]
        if predicted != other_framework:
            num_errors += 1

        if verbose:
            print("%s\t\t\t\t%s" % (other_framework, predicted))

    ret = {
        "num_samples": len(data),
        "num_errors": num_errors
    }

    if verbose:
        print("results: %s" % ret)

    return ret


def evaluate_classifier_with_probabilities(model, data, probabilities =
        'probabilities', verbose = False):
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

    for _,row in data.iterrows():
        predicted_values = model.predict(dict(row))[unicode(probabilities)]
        other_values = row[probabilities]

        if set(predicted_values.keys()) != set(other_values.keys()):
            if verbose:
                print("Different classes: ", str(predicted_values.keys()), str(other_values.keys()))
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
        "num_key_mismatch": num_key_mismatch
    }

    if verbose:
        print("results: %s" % ret)

    return ret

def rename_feature(spec, current_name, new_name, rename_inputs=True, rename_outputs=True):
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
        >>> coremltools.utils.rename_feature(spec, 'old_feature', 'new_feature_name')
    """
    from coremltools.models import MLModel

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
    for nn_type in ['neuralNetwork','neuralNetworkClassifier','neuralNetworkRegressor']:
        if spec.HasField(nn_type):
            nn = getattr(spec,nn_type)

    if nn is not None:
        for layer in nn.layers:
            if rename_inputs:
                for index,name in enumerate(layer.input):
                    if name == current_name:
                        layer.input[index] = new_name
                if rename_outputs:
                    for index,name in enumerate(layer.output):
                        if name == current_name:
                            layer.output[index] = new_name

    # Rename internally for feature vectorizer
    if spec.HasField('featureVectorizer') and rename_inputs:
        for input in spec.featureVectorizer.inputList:
            if input.inputColumn == current_name:
                input.inputColumn = new_name
                changed_input = True

    # Rename for pipeline models
    pipeline = None
    if spec.HasField('pipeline'):
        pipeline = spec.pipeline
    elif spec.HasField('pipelineClassifier'):
        pipeline = spec.pipelineClassifier.pipeline
    elif spec.HasField('pipelineRegressor'):
        pipeline = spec.pipelineRegressor.pipeline

    if pipeline is not None:
        for index,model in enumerate(pipeline.models):
            rename_feature(model,
                           current_name,
                           new_name,
                           rename_inputs or (index != 0),
                           rename_outputs or (index < len(spec.pipeline.models)))

def _sanitize_value(x):
    """
    Performs cleaning steps on the data so various type comparisons can
    be performed correctly.
    """
    if isinstance(x, _six.string_types + _six.integer_types + (float,)):
        return x
    elif _HAS_SKLEARN and _sp.issparse(x):
        return x.todense()
    elif isinstance(x, _np.ndarray):
        return x
    elif isinstance(x, tuple):
        return (_sanitize_value(v) for v in x)
    elif isinstance(x, list):
        return [_sanitize_value(v) for v in x]
    elif isinstance(x, dict):
        return dict( (_sanitize_value(k), _sanitize_value(v)) for k, v in x.items())
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
        return (isinstance(y, dict)
                and _element_equal(x.keys(), y.keys())
                and all(_element_equal(x[k], y[k]) for k in x.keys()))
    elif isinstance(x, float):
        return abs(x - y) < 1e-5 * (abs(x) + abs(y))
    elif isinstance(x, (list, tuple)):
        return x == y
    else:
        return bool(x == y)

def evaluate_transformer(model, input_data, reference_output, verbose = False):
    """
    Evaluate a transformer specification for testing.

    Parameters
    ----------
    spec: [str | MLModel]
        File from where to load the Model from (OR) a loaded
        version of MLModel.

    input_data: list[dict]
        Test data on which to evaluate the models.

    reference_output: list[dict]
        Expected results for the model.

    verbose: bool
        Verbosity levels of the predictions.

    Examples
    --------
    .. sourcecode:: python

        >>> input_data = [{'input_1': 1, 'input_2': 2}, {'input_1': 3, 'input_2': 3}]
        >>> expected_output = [{'input_1': 2.5, 'input_2': 2.0}, {'input_1': 1.3, 'input_2': 2.3}]
        >>> metrics = coremltools.utils.evaluate_transformer(scaler_spec, input_data, expected_output)

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

        predicted_trimmed = dict( (k, predicted[k]) for k in ref_data.keys())

        if verbose:
            print("Predicted:\n\t", str(predicted_trimmed))

        if not _element_equal(predicted_trimmed, ref_data):
            num_errors += 1

    ret = {
        "num_samples": len(input_data),
        "num_errors": num_errors
    }

    if verbose:
        print("results: %s" % ret)
    return ret
