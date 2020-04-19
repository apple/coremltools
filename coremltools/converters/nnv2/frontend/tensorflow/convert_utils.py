import logging
from .basic_graph_ops import topsort
from coremltools.converters.nnv2.builtin_types.symbolic import (
        is_symbolic, any_variadic)
from coremltools.converters.nnv2.builtin_types import builtins
from .tf_op_registry import _TF_OPS_REGISTRY
from coremltools.converters.nnv2.nnv2_program.program.var import ListVar

def compatible_shapes(tf_shape, inf_shape):
    def compare_elem(dt, ds):
        if dt is None or dt < 0:
            return True
        elif dt == ds:
            return True
        else:
            return False

    if tf_shape is None or any_variadic(inf_shape):
        return True
    else:
        return all(compare_elem(dt, ds) for dt, ds in zip(tf_shape, inf_shape))

def check_output_shapes(x, node):
    """
    x: list[Var] or tuple[Var]
    node: ParsedTFNode
    """
    if isinstance(x, ListVar):
        # No check on list.
        return
    if not isinstance(x, (list, tuple)):
        x = [x]
    tf_shapes = node.attr.get('_output_shapes', None)
    if tf_shapes is None:
        return
    inf_shapes = []
    for y in x:
        if y is None:
            msg = 'TF convert returns None type in TF node {}'
            raise TypeError(msg.format(node.name))
        if builtins.is_tensor(y.sym_type):
            inf_shapes.append(list(y.shape))
        elif builtins.is_scalar(y.sym_type):
            inf_shapes.append([])
        else:
            msg = 'Output type {} not understood'
            raise ValueError(msg.format(y))

    for t, s in zip(tf_shapes, inf_shapes):
        if not compatible_shapes(t, s):
            msg = "Op {} ({}) type inference ({}) and TF output shape " + \
                    "({}) mismatch"
            raise ValueError(msg.format(node.name, node.op, s, t))


def convert_graph(context, graph, outputs=None):
    """
    Construct Core ML ops corresponding to `graph`.

    Inputs:

    - context (TranscriptContext)

    - graph (dict of str -> ParsedTFNode): op name --> ParsedTFNode

    - outputs (list[str]): List of output names. If outputs is None, the last
      node graph (after topsort) must have op type return.

    Returns:

    list[Var]: the output Vars of the constructed SsaBlock.
    """
    # import here to avoid circular dependencies
    nodes = topsort(graph)

    if outputs is None:
        # infer outputs from return
        last_node = graph[nodes[-1]]
        if last_node.op != 'return':
            msg = 'Expect the last node in graph to be \'return\'; Got {}'
            raise ValueError(msg.format(last_node.op))
        second_last_node = graph[last_node.inputs[0]]
        if second_last_node.op == 'make_tuple':
            outputs = second_last_node.inputs
        else:
            # single output function
            outputs = second_last_node.name

    # Translate the non-placeholder ops.
    num_nodes = len(nodes)
    for i, node_name in enumerate(nodes):
        node = graph[node_name]
        if node.op == 'return':
            continue
        logging.info("[{}/{}] Converting {} op {}".format(
            i + 1, num_nodes, node.op, node.name))

        if node.op == 'NoOp':
            continue
        _add_op = _TF_OPS_REGISTRY.get(node.op, None)
        if _add_op is None:
            msg = "Conversion for TF op '{}' not implemented."
            raise NotImplementedError(msg.format(node.op))
        _add_op(context, node)

        x = context[node.name]
        check_output_shapes(x, node)

    output_is_list = isinstance(outputs, (tuple, list))
    if not output_is_list:
        outputs = [outputs]

    output_vars = []
    for output in outputs:
        x = context[output.split(':')[0]]
        if isinstance(x, (tuple,list)):
            idx = int(output.split(':')[1])
            output_vars.append(x[idx])
        else:
            output_vars.append(x)

    return output_vars if output_is_list else output_vars[0]
