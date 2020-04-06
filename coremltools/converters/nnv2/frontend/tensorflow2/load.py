# -*- coding: utf-8 -*-
from __future__ import absolute_import as _
from __future__ import division as _
from __future__ import print_function as _

import logging
import os

import six

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

from coremltools.converters.nnv2.frontend.tensorflow.converter import TFConverter
from coremltools.converters.nnv2.frontend.tensorflow.tf_graph_pass import *  # pylint: disable=unused-wildcard-import,wildcard-import
from coremltools.converters.nnv2.frontend.tensorflow.load import tf_ssa_from_graph


def load(model, debug=False, **kwargs):
    """
    Loads a NetworkEnsemble from a TensorFlow model.

    Parameters
    ----------
    model: Model created with TensorFlow 2.0+
        One of the following model format:
            - TensorFlow tf.Graph object
            - TensorFlow tf.keras HDF5 (.h5) model file name
            - TensorFlow SavedModel directory path
            - TensorFlow list of concrete functions(s)
    debug: bool, optional. Defaults to False.
        This flag should generally be False except for debugging purposes
        for diagnosing conversion errors. Setting this flag to True will
        cause graph pass errors to be ignored, forcefully returning a
        NetworkEnsemble object.
    """
    tf_graph = tf_graph_from_model(model)
    tf_ssa = tf_ssa_from_graph(tf_graph)

    if debug:
        import graphviz
        dot_string = tf_ssa.get_dot_string(
            annotation=True, name_and_op_style=True, highlight_debug_nodes=[])
        graphviz.Source(dot_string).view(filename='/tmp/ssa_before_tf_passes.pdf')

    tf_passes = [
        delete_asserts,
        constant_propagation,
        remove_variable_nodes
    ]

    if debug:

        for p in tf_passes:
            try:
                p(tf_ssa)
            except Exception as e:
                logging.exception('Exception in pass "{}": {}'.format(p, e))
                logging.info("Ignoring exception and continuing to next pass")

        import graphviz
        dot_string = tf_ssa.get_dot_string(
            annotation=True, name_and_op_style=True, highlight_debug_nodes=[])
        graphviz.Source(dot_string).view(filename='/tmp/ssa_after_tf_passes.pdf')
        tf.io.write_graph(tf_graph, '/tmp/', '/tmp/tf_graph.pb', as_text=False)

    else:
        for p in tf_passes:
            p(tf_ssa)

    converter = TFConverter(tf_ssa, **kwargs)
    prog = converter.convert()

    return prog


def tf_graph_from_model(model):
    """
    Extract tf.Graph from model created in TensorFlow 2.x
    """
    msg = 'Expected model format: ' \
          '[SavedModel | [concrete_function] | tf.keras.Model | .h5], got {}'
    if isinstance(model, tf.Graph) and hasattr(model, 'as_graph_def'):
        return model.as_graph_def(add_shapes=True)
    elif isinstance(model, list) or isinstance(model, tf.keras.Model) or isinstance(model, six.string_types):
        cfs = []
        if isinstance(model, list):
            cfs = model
        if isinstance(model, tf.keras.Model):
            input_signature = saving_utils.model_input_signature(
                model, keep_original_batch_size=True)
            func = saving_utils.trace_model_call(model, input_signature)
            cfs = [func.get_concrete_function()]
        elif isinstance(model, six.string_types):
            if not os.path.exists(model):
                raise ValueError('Input model "{}" does not exist'.format(model))
            elif os.path.isfile(model) and model.endswith('.h5'):
                keras_model = tf.keras.models.load_model(model)
                input_signature = saving_utils.model_input_signature(
                    keras_model, keep_original_batch_size=True)
                func = saving_utils.trace_model_call(keras_model, input_signature)
                cfs = [func.get_concrete_function()]
            elif os.path.isdir(model):  # SavedModel directory
                saved_model = tf.saved_model.load(model)
                signatures = saved_model.signatures
                signature_values = signatures.values()
                if not isinstance(signature_values, list):
                    signature_values = list(signature_values)
                cfs = signature_values
            else:
                raise NotImplementedError(msg.format(model))

        if len(cfs) != 1:
            raise NotImplementedError('Only a single concrete function is supported.')

        frozen_func = convert_variables_to_constants_v2(cfs[0], False)
        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)

        # run a Grappler's constant folding pass.
        func_inputs = [t for t in frozen_func.inputs if t.dtype != dtypes.resource]
        graph_def = run_graph_optimizations(
            graph_def,
            func_inputs,
            frozen_func.outputs,
            config=get_grappler_config(['constfold']),
            graph=frozen_func.graph)

        return graph_def
    else:
        raise NotImplementedError(msg.format(model))
