# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import logging

from .graphdef_to_ssa import graphdef_to_ssa
from .graph_pass import *  # pylint: disable=unused-wildcard-import,wildcard-import
from ..common_pass import common_pass

from coremltools.converters.nnssa.commons.features import Features

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

    ssa = graphdef_to_ssa(gd)

    if not Features.new_ssa() and not Features.nnv2_ssa():
        placeholder_shape = kwargs.get("inputs", {})

        if placeholder_shape and len(placeholder_shape) > 0:
            graph = ssa.functions['main'].graph
            required_plhd_nodes = [node for node in graph if
                graph[node].op == 'Placeholder']
            for name in required_plhd_nodes:
                if name in placeholder_shape:
                    graph[name].attr['_output_shapes'] = [placeholder_shape[name]]

    default_shapes = kwargs.get("default_shapes", {})
    if default_shapes and len(default_shapes) > 0:
        graph = ssa.functions['main'].graph
        for k, v in default_shapes.items():
            graph[k].attr['_output_shapes'] = v

    passes = [
        delete_asserts, functionalize_loops, constant_propagation, cond_to_where,
        remove_variable_nodes, fusedbatchnorm_rewrite, lstmblockcell_rewrite, grublockcell_rewrite
    ]

    if not resume_on_errors:
        for p in passes:
            p(ssa)
    else:
        for p in passes:
            try:
                p(ssa)
            except:
                logging.exception("Exception in pass %s", str(p))
                logging.info("Ignoring exception and continuing to next pass")

    common_pass(ssa, resume_on_errors, **kwargs)

    for f in ssa.functions.values():
        f.find_inputs_and_outputs()
    # check that type inference is complete
    if not resume_on_errors:
        for f in ssa.functions.values():
            for n in f.graph.values():
                assert n.datatype is not None
    return ssa
