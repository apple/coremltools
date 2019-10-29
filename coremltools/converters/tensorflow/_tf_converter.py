# Copyright (c) 2019, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import os.path
from ...models import MLModel


def convert(filename,
            inputs=None,
            outputs=None,
            image_input_names=None,
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
            **kwargs):
    use_cpu_only = kwargs.get('use_cpu_only')
    use_cpu_only = use_cpu_only if use_cpu_only is not None else False

    optional_inputs = kwargs.get('optional_inputs')
    optional_inputs = optional_inputs if optional_inputs is not None else []

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

    # convert from TensorFlow to SSA IR
    try:
        from ..nnssa.frontend.tensorflow import load as frontend_load
        ssa = frontend_load(filename, resume_on_errors=False, inputs=inputs, outputs=outputs, **kwargs)
    except ImportError as e:
        raise ImportError('frontend converter not found. {}'.format(e))
    except Exception as e:
        raise RuntimeError('failed to convert from TensorFlow to IR. {}'.format(e))

    # convert from SSA IR to Core ML
    try:
        from ..nnssa.coreml.ssa_converter import ssa_convert
        model_spec = ssa_convert(ssa,
                                 top_func='main',
                                 inputs=inputs,
                                 outputs=outputs,
                                 image_input_names=image_input_names,
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
    except ImportError as e:
        raise ImportError('backend converter not found. {}'.format(e))
    except Exception as e:
        raise RuntimeError('failed to convert from IR to Core ML. {}'.format(e))

    return MLModel(model_spec, useCPUOnly=use_cpu_only)


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
