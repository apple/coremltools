# -*- coding: utf-8 -*-
from __future__ import absolute_import as _
from __future__ import division as _
from __future__ import print_function as _

import logging
import os
import gc

import six

import tensorflow as tf

from .basic_graph_ops import fill_outputs
from .converter import TFConverter
from .tf_graph_pass import *  # pylint: disable=unused-wildcard-import,wildcard-import
from .tfssa import NetworkEnsemble, SSAFunction
from .parse import ParsedTFNode
from coremltools.converters._profile_utils import profile

@profile
def load(model, debug=False, **kwargs):
    """
    Loads a NetworkEnsemble from a TensorFlow model.

    Parameters
    ----------
    model: Model created with TensorFlow 1.x
        One of the following model format:
            - TensorFlow tf.Graph object
            - TensorFlow frozen graph (.pb) model file name
    debug: bool, optional. Defaults to False.
        This flag should generally be False except for debugging purposes
        for diagnosing conversion errors. Setting this flag to True will
        cause graph pass errors to be ignored, forcefully returning a
        NetworkEnsemble object.
    """

    outputs = kwargs.get("outputs", None)
    tf_graph = _tf_graph_from_model(model, outputs)

    if len(tf_graph.node) == 0:
        msg = "tf.Graph should have at least 1 node, Got empty graph."
        raise ValueError(msg)

    tf_ssa = tf_ssa_from_graph(tf_graph)

    del tf_graph
    gc.collect()

    if debug:
        import graphviz
        dot_string = tf_ssa.get_dot_string(
            annotation=True, name_and_op_style=True, highlight_debug_nodes=[])
        graphviz.Source(dot_string).view(filename='/tmp/ssa_before_tf_passes', cleanup=True)

    # Applying frontend passes on tfssa. Note that these are different from
    # passes applied to nnv2 ssa in TF frontend.
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
        graphviz.Source(dot_string).view(filename='/tmp/ssa_after_tf_passes', cleanup=True)
        tf.io.write_graph(tf_graph, '/tmp/', '/tmp/graph_def.pb', as_text=False)

    else:
        for p in tf_passes:
            p(tf_ssa)

    converter = TFConverter(tf_ssa, **kwargs)
    prog = converter.convert()

    return prog


def _tf_graph_from_model(model, outputs=None):
    """
    Extract tf.Graph from model created in TensorFlow 1.x
    """
    msg = 'Expected model format: [tf.Graph | .pb], got {}'
    if isinstance(model, tf.Graph) and hasattr(model, 'as_graph_def'):
        gdef = model.as_graph_def(add_shapes=True)
        if outputs is not None:
            outputs = [i.split(":")[0] for i in outputs]
            gdef = tf.compat.v1.graph_util.extract_sub_graph(
                gdef, outputs
            )
        return gdef
    elif isinstance(model, six.string_types):
        if not os.path.exists(model):
            raise ValueError('Input model "{}" does not exist'.format(model))
        elif model.endswith('.pb'):
            with tf.io.gfile.GFile(model, 'rb') as f:
                g = tf.GraphDef()
                g.ParseFromString(f.read())
                if outputs is not None:
                    outputs = [i.split(":")[0] for i in outputs]
                    g = tf.compat.v1.graph_util.extract_sub_graph(
                        g, outputs
                    )
            with tf.Graph().as_default() as graph:
                tf.graph_util.import_graph_def(g, name='')
            return graph.as_graph_def(add_shapes=True)
        else:
            raise NotImplementedError(msg.format(model))
    else:
        raise NotImplementedError(msg.format(model))


def tf_ssa_from_graph(graph, main_func_name='main'):
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


def graph_def_to_dict(gd):
    ret = {}
    for node in gd.node:
        ret[node.name] = ParsedTFNode(node)
    return ret
