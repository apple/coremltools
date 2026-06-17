# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy as _copy

from coremltools.models.utils import _get_model

from .builder import NeuralNetworkBuilder


def make_image_input(
    model,
    input_name,
    is_bgr=False,
    red_bias=0.0,
    blue_bias=0.0,
    green_bias=0.0,
    gray_bias=0.0,
    scale=1.0,
    image_format="NHWC",
):
    """
    Convert input of type multiarray to type image.

    Parameters
    ----------
    model : MLModel
        A Core ML model object. The model must be of type ``neuralNetwork``,
        ``neuralNetworkClassifier``, or ``neuralNetworkRegressor``.

    input_name : str or list of str
        The name of the multiarray input feature (or list of names) to be
        converted to an image.

    is_bgr : bool, optional
        If True, the image is treated as BGR. Otherwise, it is treated as RGB.
        Defaults to False.

    red_bias : float, optional
        The bias value to be subtracted from the red channel. Defaults to 0.0.

    blue_bias : float, optional
        The bias value to be subtracted from the blue channel. Defaults to 0.0.

    green_bias : float, optional
        The bias value to be subtracted from the green channel. Defaults to 0.0.

    gray_bias : float, optional
        The bias value to be subtracted from the grayscale channel (used if
        the image has 1 channel). Defaults to 0.0.

    scale : float, optional
        The scaling factor to multiply the input values by. Defaults to 1.0.

    image_format : str, optional
        The image color/layout format. Can be "NHWC" or "NCHW". Defaults to "NHWC".

    Returns
    -------
    model : MLModel
        The modified MLModel object with the specified input(s) converted to image type.

    Examples
    --------
    .. sourcecode:: python

        import coremltools as ct
        from coremltools.models.neural_network.utils import make_image_input

        # Load a model
        model = ct.models.MLModel("my_model.mlpackage")

        # Convert a multiarray input named "input_data" to an image
        image_model = make_image_input(
            model,
            input_name="input_data",
            red_bias=123.68,
            green_bias=116.78,
            blue_bias=103.94,
            scale=1.0 / 255.0,
            image_format="NCHW",
        )
    """

    spec = model.get_spec()

    if spec.WhichOneof("Type") not in [
        "neuralNetwork",
        "neuralNetworkClassifier",
        "neuralNetworkRegressor",
    ]:
        raise ValueError(
            "Provided model must be of type neuralNetwork, neuralNetworkClassifier or neuralNetworkRegressor"
        )

    if not isinstance(input_name, list):
        input_name = [input_name]

    spec_inputs = [i.name for i in spec.description.input]
    for name in input_name:
        if name not in spec_inputs:
            msg = "Provided input_name: {}, is not an existing input to the model"
            raise ValueError(msg.format(name))

    builder = NeuralNetworkBuilder(spec=spec)
    builder.set_pre_processing_parameters(
        image_input_names=input_name,
        is_bgr=is_bgr,
        red_bias=red_bias,
        green_bias=green_bias,
        blue_bias=blue_bias,
        gray_bias=gray_bias,
        image_scale=scale,
        image_format=image_format,
    )
    return _get_model(spec)


def make_nn_classifier(
    model,
    class_labels,
    predicted_feature_name=None,
    predicted_probabilities_output=None,
):
    """
    Convert a model of type "neuralNetwork" to type "neuralNetworkClassifier".

    Parameters
    ----------
    model : MLModel
        A Core ML model object of type ``neuralNetwork`` to be converted to a classifier.

    class_labels : list of str/int, or str
        List of class labels (strings or integers) or the path to a text file
        containing class labels (one per line).

    predicted_feature_name : str, optional
        The name of the output feature for the predicted class label. If not
        provided, it will use the default name.

    predicted_probabilities_output : str, optional
        The name of the output feature for the class probabilities dictionary.

    Returns
    -------
    model : MLModel
        The converted MLModel object of type ``neuralNetworkClassifier``.

    Examples
    --------
    .. sourcecode:: python

        import coremltools as ct
        from coremltools.models.neural_network.utils import make_nn_classifier

        # Load a neuralNetwork model
        model = ct.models.MLModel("neural_network.mlpackage")

        # Convert to a classifier with a list of labels
        classifier_model = make_nn_classifier(
            model,
            class_labels=["cat", "dog", "bird"],
            predicted_feature_name="classLabel",
            predicted_probabilities_output="classLabel_probs",
        )
    """

    spec = model.get_spec()

    if spec.WhichOneof("Type") != "neuralNetwork":
        raise ValueError('Provided model must be of type "neuralNetwork"')

    # convert type to "neuralNetworkClassifier" and copy messages from "neuralNetwork"
    nn_spec = _copy.deepcopy(spec.neuralNetwork)
    spec.ClearField("neuralNetwork")
    for layer in nn_spec.layers:
        spec.neuralNetworkClassifier.layers.add().CopyFrom(layer)
    for preprocessing in nn_spec.preprocessing:
        spec.neuralNetworkClassifier.preprocessing.add().CopyFrom(preprocessing)
    spec.neuralNetworkClassifier.arrayInputShapeMapping = nn_spec.arrayInputShapeMapping
    spec.neuralNetworkClassifier.imageInputShapeMapping = nn_spec.imageInputShapeMapping
    spec.neuralNetworkClassifier.updateParams.CopyFrom(nn_spec.updateParams)

    # set properties related to classifier
    builder = NeuralNetworkBuilder(spec=spec)
    message = "Class labels must be a list of integers / strings or a file path"
    classes_in = class_labels
    if isinstance(classes_in, str):
        import os

        if not os.path.isfile(classes_in):
            raise ValueError("Path to class labels (%s) does not exist." % classes_in)
        with open(classes_in, "r") as f:
            classes = f.read()
        classes = classes.splitlines()
    elif isinstance(classes_in, list):  # list[int or str]
        classes = classes_in
        assert all(isinstance(x, (int, str)) for x in classes), message
    else:
        raise ValueError(message)

    kwargs = {}
    if predicted_feature_name is not None:
        kwargs["predicted_feature_name"] = predicted_feature_name
    if predicted_probabilities_output is not None:
        kwargs["prediction_blob"] = predicted_probabilities_output
    builder.set_class_labels(classes, **kwargs)

    return _get_model(spec)
