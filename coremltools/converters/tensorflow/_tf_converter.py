# Copyright (c) 2019, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import os.path
from ...models import MLModel
from warnings import warn


class SupportedVersion():
    # Supported iOS Version
    # New OS Version must be added at the end to maintain backward version index
    supported_ios_version = ['11.2', '12', '13']
    IOS_13_VERSION = supported_ios_version.index('13')
    ND_ARRARY_SUPPORT = IOS_13_VERSION

    @staticmethod
    def ios_support_check(target_ios):
        return target_ios in SupportedVersion.supported_ios_version

    @staticmethod
    def is_nd_array_supported(target_ios):
        if not SupportedVersion.ios_support_check(target_ios):
            raise TypeError('{} not supported. Please provide one of target iOS: {}', target_ios, SupportedVersion.supported_ios_version)

        target_ios_index = SupportedVersion.supported_ios_version.index(target_ios)
        return SupportedVersion.ND_ARRARY_SUPPORT <= target_ios_index

    @staticmethod
    def get_supported_ios():
        return SupportedVersion.supported_ios_version


# Checks input and output naming convention
# With `target_ios=13` i.e. new tf-coreml path drops ':' from the input and output
# names
def check_input_output_names(input_name_shape_dict, output_feature_names):
  new_input = []
  old_input = []
  new_output = []
  old_output = []
  for _key in input_name_shape_dict:
    if ':' in _key:
      new_input.append(_key.split(':')[0])
      old_input.append(_key)

  for _output in output_feature_names:
    if ':' in _output:
      new_output.append(_output.split(':')[0])
      old_output.append(_output)

  if len(new_input) > 0 or len(new_output) > 0:
    input_string = ''
    output_string = ''
    if len(new_input) > 0:
      input_string = 'Input: ' + str(new_input) + ' instead of ' + str(old_input) + '\n'
    if len(new_output) > 0:
      output_string = 'Output: ' + str(new_output) + ' instead of ' + str(old_output)

    raise ValueError('with target deployment > \'12\', the converter drops \':\' convention for input and output.'
                     ' Please provide input and output without \':\' e.g. `input` instead of `input:0`\n'
                     'Recommendation: \n {} {}'.format(input_string, output_string))


def _graph_def_from_saved_model_or_keras_model(filename):
    """
    Utility function that returns GraphDef object from the given SavedModel or HDF5 model.
    :param filename: TensorFlow SavedModel directory or Keras HDF5 model (.h5) file.
    :return: TensorFlow GraphDef object.
    """
    try:
        import tensorflow as tf
        from tensorflow.python.keras.saving import saving_utils as _saving_utils
        from tensorflow.python.framework import convert_to_constants as _convert_to_constants
        model = tf.keras.models.load_model(filename)
        tf.keras.backend.set_learning_phase(False)
        func = _saving_utils.trace_model_call(model)
        concrete_func = func.get_concrete_function()
        # concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        frozen_func = _convert_to_constants.convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
    except ImportError as e:
        raise ImportError('Failed to import TensorFlow utilities. {}.'.format(e))
    except Exception as e:
        raise RuntimeError('Failed to load SavedModel or .h5 model. {}.'.format(e))
    return graph_def


def _graph_def_from_concrete_function(concrete_functions):
    """
    Utility function that returns GraphDef object from the given concrete functions.
    :param concrete_functions: list of TensorFlow concrete functions.
    :return: TensorFlow GraphDef object.
    """
    if len(concrete_functions) != 1:
        raise ValueError('This converter can only convert a single ConcreteFunction.')
    try:
        import tensorflow as tf
        from tensorflow.python.framework import convert_to_constants as _convert_to_constants
        from tensorflow.python.eager import function as _function
        frozen_func = _convert_to_constants.convert_variables_to_constants_v2(concrete_functions[0])
        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
    except ImportError as e:
        raise ImportError('Failed to import TensorFlow utilities. {}.'.format(e))
    except Exception as e:
        raise RuntimeError('Failed to load concrete functions(s). {}.'.format(e))
    return graph_def


def convert(filename,
            inputs=None,
            outputs=None,
            image_input_names=None,
            tf_image_format=None,
            is_bgr=False,
            red_bias=0.0,
            green_bias=0.0,
            blue_bias=0.0,
            gray_bias=0.0,
            image_scale=1.0,
            class_labels=None,
            predicted_feature_name=None,
            predicted_probabilities_output='',
            add_custom_layers=False,  # type: bool
            custom_conversion_functions=None,  # type: dict{text, any}
            custom_shape_functions=None,  # type: dict{text, any}
            minimum_ios_deployment_target='13',
            mlmodel_path=None,
            **kwargs):
    """
    Convert TensorFlow model to Core ML format.

    Parameters
    ----------
    filename: str
        Path to the TensorFlow model. Takes in one of the following formats:

        - TensorFlow frozen graph (.pb) model file name
        - TensorFlow tf.keras HDF5 (.h5) model file name
        - TensorFlow SavedModel directory path
        - TensorFlow concrete functions(s)

    inputs: dict(str: list or tuple)
        Model input name and shape pairs.

    outputs: [str]
        Model output names.

    image_input_names: [str] | str
      Input names (a subset of the keys of input_name_shape_dict)
      that can be treated as images by Core ML. All other inputs
      are treated as MultiArrays.
      Input names (a subset of the keys of inputs)
      that can be treated as images by Core ML. All other inputs
      are treated as MultiArrays.
    tf_image_format: str
      Optional and valid if image_input_names is also set. Specify either 'NCHW' or 'NHWC' to set or
      override the image format. If not set, tries to use hints from the graph which may be present in convolution or
      other image-specific layers. Ultimately defaults to NHWC.
    is_bgr: bool | dict():
      Applicable only if image_input_names is specified.
      To specify different values for each image input provide a dictionary with input names as keys and booleans as values.
    red_bias: float | dict()
      Bias value to be added to the red channel of the input image, after applying scale.
      Defaults to 0.0
      Applicable only if image_input_names is specified.
      To specify different values for each image input provide a dictionary with input names as keys.
    blue_bias: float | dict()
      Bias value to be added to the blue channel of the input image, after applying scale.
      Defaults to 0.0
      Applicable only if image_input_names is specified.
      To specify different values for each image input provide a dictionary with input names as keys.
    green_bias: float | dict()
      Bias value to be added to the green channel of the input image, after applying scale.
      Defaults to 0.0
      Applicable only if image_input_names is specified.
      To specify different values for each image input provide a dictionary with input names as keys.
    gray_bias: float | dict()
      Bias value to be added to the input image (in grayscale), after applying scale.
      Defaults to 0.0
      Applicable only if image_input_names is specified.
      To specify different values for each image input provide a dictionary with input names as keys.
    image_scale: float | dict()
      Value by which input images will be scaled before bias is added and
      Core ML model makes a prediction. Defaults to 1.0.
      Applicable only if image_input_names is specified.
      To specify different values for each image input provide a dictionary with input names as keys.
    class_labels: list[int or str] | str
      Class labels (applies to classifiers only) that map the index of the
      output of a neural network to labels in a classifier.
      If the provided class_labels is a string, it is assumed to be a
      file path where classes are parsed as a list of newline separated
      strings.
    predicted_feature_name: str
      Name of the output feature for the class labels exposed in the Core ML
      model (applies to classifiers only). Defaults to 'classLabel'
    predicted_probabilities_output: str
      Name of the neural network output to be interpreted as the predicted
      probabilities of the resulting classes. Typically the output of a
      softmax function.
    add_custom_layers: bool
      Flag to turn on addition of custom CoreML layers for unsupported TF ops or attributes within
      a supported op.
    custom_conversion_functions: dict(): {Text: func(**kwargs)}
      Argument to provide user-defined functions for converting Tensorflow operations (op, for short).
      A dictionary with keys corresponding to the names or types of the TF ops and values as handle to user-defined functions.
      The keys can be either the type of the op or the name of the op. If former, then the function is called whenever the op
      of that type is encountered during conversion. By using op names, specific ops can be targeted which is
      useful for handling unsupported configuration in an op.
      The function receives multiple arguments: TF operation, the CoreML Neural network builder object,
      dictionary containing the op's inputs that are constants and their values (as numpy arrays).
      The function can add custom layers or any other combination of CoreML layers to translate the TF op.
      See "examples/custom_layer_examples.ipynb" jupyter-notebook for examples on using this argument.
    custom_shape_functions: dict(): {Text: func()}
      Argument to provide user-defined functions to compute shape for given op.
      A dictionary with keys corresponding to the type of TF Op and value as handled to user-defined function.
      Function receives `layer specification` and `input shape` as a input.
      output of the function must be output shape for give op. (generally List).
      Custom shape function is required for adding custom layer in Core ML 3.
    minimum_ios_deployment_target: str
      Minimum target deployment iOS version (default: '13'). Supported iOS version options: '11.2', '12', '13'.
      Core ML model produced by the converter will be compatible with the iOS version specified in this
      argument and the versions after it. e.g., if minimum_ios_deployment_target='12', the converter would
      only utilize layers released till iOS 12 (equivalently macOS 10.14, watchOS 5 etc.), and the produced
      model can be deployed to iOS12+ (iOS 12, iOS 13 and later).

      iOS 11.2 (Core ML 0.8): https://github.com/apple/coremltools/releases/tag/v0.8
      iOS 12 (Core ML 2.0): https://github.com/apple/coremltools/releases/tag/v2.0
      iOS 13 (Core ML 3.0): https://github.com/apple/coremltools/releases/tag/3.0
    mlmodel_path: str
      Path to where the generated .mlmodel will be stored
    Returns
    -------
    model: MLModel
        Returns an MLModel instance representing a Core ML model.

    Examples
    --------
    .. code-block:: python

        import coremltools
        from tensorflow.keras.applications import ResNet50

        model = coremltools.converters.tensorflow.convert(
            './model.h5',
             inputs={'input': (1, 224, 224, 3)},
             outputs=['output']
        )

    For more examples, see: https://github.com/apple/coremltools/blob/master/docs/NeuralNetworkGuide.md
    """

    # `tf_model_path` takes in one of the following formats:
    # 1) TensorFlow frozen graph (.pb) model file name
    # 2) TensorFlow tf.keras HDF5 (.h5) model file name
    # 3) TensorFlow SavedModel directory path
    # 4) TensorFlow concrete functions(s)

    invalid_filename_message = ('invalid input tf_model_path: {}!\n'
                                'Supported tf_model_path input format includes:\n'
                                '- Path to TensorFlow frozen graph (.pb) file\n'
                                '- Path to TensorFlow tf.keras model (.h5) file\n'
                                '- Path to TensorFlow SavedModel directory\n'
                                '- List of TensorFlow concrete functions'.format(filename))

    if isinstance(filename, str) and not os.path.exists(filename):
        raise ValueError('invalid input tf_model_path \'{}\' does not exist.'.format(filename))

    if isinstance(filename, str) and os.path.isfile(filename):
        # path to the model file must end with either .pb or .h5 format
        if not (filename.endswith('.pb') or filename.endswith('.h5')):
            raise ValueError(invalid_filename_message)

        if filename.endswith('.h5'):
            filename = _graph_def_from_saved_model_or_keras_model(filename)

    elif isinstance(filename, str) and os.path.isdir(filename):
        filename = _graph_def_from_saved_model_or_keras_model(filename)

    elif isinstance(filename, list):
        filename = _graph_def_from_concrete_function(filename)
    else:
        raise ValueError(invalid_filename_message)

    if inputs is None:
        inputs = {}

    if outputs is None:
        raise ValueError('Output node names must be provided.')

    if tf_image_format is not None:
        warn('tf_image_format not honored when minimum_ios_deployment_target < 13')

    use_cpu_only = kwargs.get('use_cpu_only')
    use_cpu_only = use_cpu_only if use_cpu_only is not None else False

    optional_inputs = kwargs.get('optional_inputs')
    optional_inputs = optional_inputs if optional_inputs is not None else []

    if not SupportedVersion.ios_support_check(minimum_ios_deployment_target):
        raise TypeError('{} not supported. Please provide one of target iOS: {}', minimum_ios_deployment_target,
                        SupportedVersion.get_supported_ios())

    if SupportedVersion.is_nd_array_supported(minimum_ios_deployment_target):
        # Check input and output name for correct convention being used
        check_input_output_names(inputs, outputs)

        # convert from TensorFlow to SSA IR
        from ..nnssa.frontend.tensorflow import load as frontend_load
        ssa = frontend_load(filename, resume_on_errors=False, inputs=inputs, outputs=outputs, **kwargs)

        # convert from SSA IR to Core ML
        from ..nnssa.coreml.ssa_converter import ssa_convert
        model_spec = ssa_convert(ssa,
                                 top_func='main',
                                 inputs=inputs,
                                 outputs=outputs,
                                 image_input_names=image_input_names,
                                 image_format=tf_image_format,
                                 is_bgr=is_bgr,
                                 red_bias=red_bias,
                                 green_bias=green_bias,
                                 blue_bias=blue_bias,
                                 gray_bias=gray_bias,
                                 image_scale=image_scale,
                                 class_labels=class_labels,
                                 predicted_feature_name=predicted_feature_name,
                                 predicted_probabilities_output=predicted_probabilities_output,
                                 add_custom_layers=add_custom_layers,
                                 custom_conversion_functions=custom_conversion_functions,
                                 custom_shape_functions=custom_shape_functions,
                                 optional_inputs=optional_inputs)

        mlmodel = MLModel(model_spec, useCPUOnly=use_cpu_only)

    else:
        import coremltools.converters.tensorflow.legacy.tfcoreml as legacy_converter

        mlmodel = legacy_converter.convert(
            tf_model_path=filename,
            mlmodel_path=mlmodel_path,
            output_feature_names=outputs,
            input_name_shape_dict=inputs,
            image_input_names=image_input_names,
            tf_image_format=tf_image_format,
            is_bgr=is_bgr,
            red_bias=red_bias,
            green_bias=green_bias,
            blue_bias=blue_bias,
            gray_bias=gray_bias,
            image_scale=image_scale,
            class_labels=class_labels,
            predicted_feature_name=predicted_feature_name,
            predicted_probabilities_output=predicted_probabilities_output,
            add_custom_layers=add_custom_layers,
            custom_conversion_functions=custom_conversion_functions,
            custom_shape_functions=custom_shape_functions)


    if mlmodel_path is not None:
        mlmodel.save(mlmodel_path)
    return mlmodel



