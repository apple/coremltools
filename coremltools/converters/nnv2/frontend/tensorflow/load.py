# -*- coding: utf-8 -*-
from __future__ import absolute_import as _
from __future__ import division as _
from __future__ import print_function as _

import logging
import os

import six
from coremltools.converters.nnv2._deps import HAS_TF1, HAS_TF2

if HAS_TF1:
    import tensorflow as tf

elif HAS_TF2:
    import tensorflow as tf
    from tensorflow.python.framework import dtypes
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    from tensorflow.python.keras.saving import saving_utils
    from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

from .basic_graph_ops import fill_outputs
from .converter import TFConverter
from .parse import graph_def_to_dict
from .tf_graph_pass import *  # pylint: disable=unused-wildcard-import,wildcard-import
from .tfssa import NetworkEnsemble, SSAFunction


def load(model, debug=False, **kwargs):
    """
    Loads a NetworkEnsemble from a TensorFlow model.

    Parameters
    ----------
    model: TensorFlow model
        One of the following model format:
        If TensorFlow 1.x:
            - TensorFlow tf.Graph object
            - TensorFlow frozen graph (.pb) model file name
        If TensorFlow 2.x:
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
    tf_graph = _tf_graph_from_model_v1(model) if HAS_TF1 else _tf_graph_from_model_v2(model)
    tf_ssa = _tf_ssa_from_graph(tf_graph)

    if debug:
        import graphviz
        dot_string = tf_ssa.get_dot_string(
            annotation=True, name_and_op_style=True, highlight_debug_nodes=[])
        graphviz.Source(dot_string).view(filename='/tmp/ssa_before_tf_passes.pdf')

    tf_passes = [
        delete_asserts,
        functionalize_loops,
        constant_propagation,
        cond_to_where,
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


def _tf_graph_from_model_v1(model):
    """
    Extract tf.Graph from model created in TensorFlow 1.x
    """
    msg = 'Expected model format: [tf.Graph | .pb], got {}'
    if isinstance(model, tf.Graph) and hasattr(model, 'as_graph_def'):
        return model.as_graph_def(add_shapes=True)
    elif isinstance(model, six.string_types):
        if not os.path.exists(model):
            raise ValueError('Input model "{}" does not exist'.format(model))
        elif model.endswith('.pb'):
            with tf.io.gfile.GFile(model, 'rb') as f:
                g = tf.GraphDef()
                g.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.graph_util.import_graph_def(g, name='')
            return graph.as_graph_def(add_shapes=True)
        else:
            raise NotImplementedError(msg.format(model))
    else:
        raise NotImplementedError(msg.format(model))


def _tf_graph_from_model_v2(model):
    """
    Extract tf.Graph from model created in TensorFlow 2.x
    """
    msg = 'Expected model format: ' \
          '[tf.Graph | [concrete_function] | .h5 | SavedModel], got {}'
    if isinstance(model, tf.Graph) and hasattr(model, 'as_graph_def'):
        return model.as_graph_def(add_shapes=True)
    elif isinstance(model, list) or isinstance(model, six.string_types):
        cfs = []
        if isinstance(model, list):
            cfs = model
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


def _tf_ssa_from_graph(graph, main_func_name='main'):
    """
    Loads a tf.Graph and transform it into tf_ssa.
    """
    with tf.Graph().as_default():
        tf.graph_util.import_graph_def(graph, name='')

    graph = graph_def_to_dict(graph)
    tensor_array_resource_removal(graph)
    graph = insert_get_tuple(graph)
    graph = fill_outputs(graph)
    delete_disconnected_nodes(graph)

    tf_ssa = NetworkEnsemble()
    tf_ssa.functions[main_func_name] = SSAFunction(graph)
    return tf_ssa
