# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import logging

from .graphdef_to_ssa import graphdef_to_ssa
from .converter import TFConverter
from .tf_graph_pass import *  # pylint: disable=unused-wildcard-import,wildcard-import

def load(tfgraph, resume_on_errors=False, **kwargs):
    """
    Loads a NetworkEnsemble from a TensorFlow frozen graph.
    tfgraph should either be a TensorFlow Graph object, or a path to a 
    frozen graph.

    Parameters
    ----------
    tfgraph: tf.Graph or str
        Either a path to a frozen graph, or a TensorFlow Graph object
    resume_on_errors : bool, optional. Default False.
        This flag should generally be False except for debugging purposes
        for diagnosing 'unconvertible' graphs. Setting this flag to True
        will cause graph pass errors to be ignored, forcefully returning
        a NetworkEnsemble object.
    inputs: dict or None
        Dictionary containing {name: shape} for each input. When not provided,
        The converter assumes all Placeholder or PlaceholderWithDefault
        as inputs.
    default_shapes: dict or None
        Dictionary containing {name: shape} for each nodes.
    outputs: list of str
        A list of names of output TF nodes.
    """
    if hasattr(tfgraph, 'as_graph_def'):
        gd = tfgraph.as_graph_def(add_shapes=True)
    else:
        gd = tfgraph

    tfssa = graphdef_to_ssa(gd)

    placeholder_shape = kwargs.get("inputs", {})
    if placeholder_shape and isinstance(placeholder_shape,dict) and len(placeholder_shape) > 0:
        graph = tfssa.functions['main'].graph
        required_plhd_nodes = [node for node in graph if
                               graph[node].op == 'Placeholder']
        for name in required_plhd_nodes:
            if name in placeholder_shape:
                graph[name].attr['_output_shapes'] = [placeholder_shape[name]]

    default_shapes = kwargs.get("default_shapes", {})
    if default_shapes and len(default_shapes) > 0:
        graph = tfssa.functions['main'].graph
        for k, v in default_shapes.items():
            graph[k].attr['_output_shapes'] = v

    tf_passes = [
        delete_asserts, functionalize_loops, constant_propagation, cond_to_where,
        remove_variable_nodes
    ]

    if not resume_on_errors:
        for p in tf_passes:
            p(tfssa)
    else:
        for p in tf_passes:
            try:
                p(tfssa)
            except:
                logging.exception("Exception in pass %s", str(p))
                logging.info("Ignoring exception and continuing to next pass")

    converter = TFConverter(tfssa, **kwargs)
    prog = converter.convert()

    return prog
