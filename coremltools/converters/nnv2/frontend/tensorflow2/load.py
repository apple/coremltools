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
from tensorflow.python.framework.function_def_to_graph import function_def_to_graph
from tensorflow.python.keras.saving import saving_utils

from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

from .converter import TFConverter
from coremltools.converters.nnv2.frontend.tensorflow.basic_graph_ops import fill_outputs
from coremltools.converters.nnv2.frontend.tensorflow.tf_graph_pass import (
    constant_propagation,
    remove_variable_nodes,
    tensor_array_resource_removal,
    insert_get_tuple,
    delete_disconnected_nodes,
)
from coremltools.converters.nnv2.frontend.tensorflow2.tf_graph_pass import (
    flatten_sub_graph_namespaces, rewrite_control_flow_functions
)
from coremltools.converters.nnv2.frontend.tensorflow.tfssa import NetworkEnsemble, SSAFunction
from coremltools.converters.nnv2.frontend.tensorflow.parse import ParsedTFNode


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
    graph_def = _tf_graph_from_model(model)
    if debug:
        tf.io.write_graph(graph_def, '/tmp/', '/tmp/graph_def.pb', as_text=False)

    if len(graph_def.node) == 0:
        msg = "tf.Graph should have at least 1 node, Got empty graph."
        raise ValueError(msg)

    tf_ssa = _tf_ssa_from_graph_def(graph_def)

    if debug:
        import graphviz
        dot_string = tf_ssa.get_dot_string(
            annotation=True, name_and_op_style=True, highlight_debug_nodes=[])
        graphviz.Source(dot_string).view(filename='/tmp/ssa_before_tf_passes', cleanup=True)

    # Notes:
    # - "flatten_while_loop_namespaces" should be after "constant_propagation"
    #   as it changes node names which constant propagation pass is relying on
    #   to perform session.run(), renamed nodes are not understandable for TF.
    tf_passes = [
        # delete_asserts,  # FIXME: rdar://62472804
        constant_propagation,
        rewrite_control_flow_functions,
        flatten_sub_graph_namespaces,
        remove_variable_nodes
    ]

    if debug:
        for tf_pass in tf_passes:
            try:
                tf_pass(tf_ssa)
            except Exception as e:
                logging.exception('Exception in pass "{}": {}'.format(tf_pass, e))
                logging.info("Ignoring exception and continuing to next pass")
                raise(e)

    else:
        for tf_pass in tf_passes:
            tf_pass(tf_ssa)

    if debug:
        import graphviz
        dot_string = tf_ssa.get_dot_string(
            annotation=True, name_and_op_style=True, highlight_debug_nodes=[])
        graphviz.Source(dot_string).view(filename='/tmp/ssa_after_tf_passes', cleanup=True)

    converter = TFConverter(tf_ssa, **kwargs)
    prog = converter.convert()

    return prog


def _tf_graph_from_model(model):
    """
    Extract tf.Graph from model created in TensorFlow 2.x
    """
    logging.info("Loading model '{}'".format(model))

    msg = 'Expected model format: [SavedModel | [concrete_function] | ' \
          'tf.keras.Model | .h5 | tf.Graph], got {}'
    if isinstance(model, tf.Graph) and hasattr(model, 'as_graph_def'):
        return model.as_graph_def(add_shapes=True)
    elif isinstance(model, list) or \
            isinstance(model, tf.keras.Model) or \
            isinstance(model, six.string_types):
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
            # serialized tf.keras model in HDF5 format on disk
            elif os.path.isfile(model) and model.endswith('.h5'):
                keras_model = tf.keras.models.load_model(model)
                input_signature = saving_utils.model_input_signature(
                    keras_model, keep_original_batch_size=True)
                func = saving_utils.trace_model_call(keras_model, input_signature)
                cfs = [func.get_concrete_function()]
            # SavedModel directory
            elif os.path.isdir(model):
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

        frozen_func = convert_variables_to_constants_v2(cfs[0], lower_control_flow=False)
        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)

        # run a Grappler's constant folding pass.
        func_inputs = [t for t in frozen_func.inputs if t.dtype != dtypes.resource]
        graph_def = run_graph_optimizations(
            graph_def,
            func_inputs,
            frozen_func.outputs,
            config=get_grappler_config(['constfold', 'dependency']),
            graph=frozen_func.graph)

        return graph_def
    else:
        raise NotImplementedError(msg.format(model))


def _tf_ssa_from_graph_def(graph_def, fn_name='main'):
    """
    Loads a GraphDef and transform it into tf_ssa.

    Parameters
    ----------
    graph_def: GraphDef
        TensorFlow GraphDef.
    fn_name: str, optional, defaults to 'main'
        Main function name of the GraphDef.

    Returns
    -------
    tf_ssa: NetworkEnsemble
        NetworkEnsemble object containing one or more SSAFunctions.
    """
    with tf.Graph().as_default() as tf_graph:
        tf.graph_util.import_graph_def(graph_def, name='')

    # sub-graphs' input shapes are required for extracting sub-graphs
    sg_input_shapes = _populate_sub_graph_input_shapes(
        tf_graph, tf_graph._functions)

    # get graph_dict and sub-graphs' inputs / outputs
    graph_dict, inputs, outputs, ret = _dict_from_graph(
        tf_graph, fn_name, sg_input_shapes)

    tf_ssa = NetworkEnsemble()
    for name, graph in graph_dict.items():
        tensor_array_resource_removal(graph)
        graph = insert_get_tuple(graph)
        graph = fill_outputs(graph)
        if name == 'main':  # skip for sub-graphs as input can be also output
            delete_disconnected_nodes(graph)
        tf_ssa.functions[name] = SSAFunction(
            graph, inputs=inputs[name], outputs=outputs[name], ret=ret[name])

    return tf_ssa


def _populate_sub_graph_input_shapes(graph, graph_fns):
    """
    Populate function (sub-graph) input shapes from control flow op's inputs
    Note that the functions (sub-graphs) are not nested but the control flow
    ops are nested. The input shapes are used to extract sub-graphs from the
    parent graph (as the input of function_def_to_graph).

    Parameter
    ---------
    graph: tf.Graph
        TensorFlow graph.
    graph_fns: list of graph functions.
        List of TensorFlow graph functions.

    Returns
    -------
    sg_input_shapes: dict(str: list)
        Dictionary of function (sub-graph) name and input shape pairs.
    """
    sg_input_shapes = {}
    sub_graphs = []
    for op in graph.get_operations():
        if op.type not in {'StatelessIf', 'If', 'StatelessWhile', 'While'}:
            continue

        sg1, sg2 = None, None
        if op.type in {'StatelessIf', 'If'}:
            sg1 = op.get_attr('then_branch').name
            sg2 = op.get_attr('else_branch').name
        if op.type in {'StatelessWhile', 'While'}:
            sg1 = op.get_attr('cond').name
            sg2 = op.get_attr('body').name

        # memorize input shapes for sub-graph conversions
        op_input_shapes = [i.get_shape() for i in op.inputs]
        sg_input_shapes.update({sg1: op_input_shapes, sg2: op_input_shapes})
        sub_graphs += [sg1, sg2]

    for name in sub_graphs:
        sg = graph_fns.get(name)
        fn_def = sg.definition
        op_input_shapes = sg_input_shapes[name]
        op_input_shapes = op_input_shapes[-len(fn_def.signature.input_arg):]
        fn_graph = function_def_to_graph(fn_def, input_shapes=op_input_shapes)
        sg_input_shapes.update(
            _populate_sub_graph_input_shapes(fn_graph, graph_fns))

    return sg_input_shapes


def _dict_from_graph(graph, fn_name='main', sg_input_shapes=None):
    """
    Loads a tf.Graph and transform it into dictionary of ParsedTFNodes.
    Potentially contains multiple functions, in such case, recursively
    resolve functions (sub-graphs).

    Parameters
    ----------
    graph: tf.Graph
        TensorFlow graph.
    fn_name: str, optional, defaults to 'main'
        Function name of the graph.
    sg_input_shapes: dict(str: list)
        Dictionary of name and input shapes for functions / sub-graphs.

    Returns
    -------
    dict(str: dict(str: ParsedTFNode))
        Dictionary of function name and dictionary of node name and
        ParsedTFNode object.
    """
    graph_dict = {fn_name: {}}
    graph_inputs = {fn_name: []}
    graph_outputs = {fn_name: []}
    graph_ret = {fn_name: {}}

    for op in graph.get_operations():
        graph_dict[fn_name].update({op.name: ParsedTFNode(op.node_def)})

    for name, sg in graph._functions.items():
        sg_def = sg.definition
        input_shapes = sg_input_shapes[name]
        input_shapes = input_shapes[-len(sg_def.signature.input_arg):]
        fn_graph = function_def_to_graph(sg_def, input_shapes=input_shapes)

        graph_dict.update(
            _dict_from_graph(fn_graph, name, sg_input_shapes)[0])
        graph_inputs.update(
            {name: [t.name.split(':')[0] for t in fn_graph.inputs]})
        graph_outputs.update(
            {name: [t.name.split(':')[0] for t in fn_graph.outputs]})

        # ret is a mapping from the output arg names from `signature` to the
        # outputs from `node_def` that should be returned by the function.
        sg_def_ret = sg_def.ret
        sg_def_ret['identity_0'] = sg_def_ret.pop('identity')
        graph_ret.update({name: sg_def_ret})

    return graph_dict, graph_inputs, graph_outputs, graph_ret
