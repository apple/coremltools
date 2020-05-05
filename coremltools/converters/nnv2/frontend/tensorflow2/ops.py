from coremltools.converters.nnv2.frontend.tensorflow.ops import (
    _transpose_NHWC_to_NCHW,
    _transpose_NCHW_to_NHWC,
)

# TF 2.x now imports and registers all TF 1.x op against the new registry
# (separated from TF 1.x registry). Overwrite might needed in case the op
# semantics are different between TF 1.x and TF 2.x.
from coremltools.converters.nnv2.frontend.tensorflow.ops import *
from coremltools.converters.nnv2.frontend.tensorflow.tf_op_registry import register_tf_op


@register_tf_op(override=True)
def FusedBatchNormV3(context, node):
    # Get attributes
    data_format = node.attr.get('data_format', 'NHWC')
    epsilon = node.attr.get('epsilon', None)

    # Get inputs
    x = context[node.inputs[0]]
    scale = context[node.inputs[1]]
    offset = context[node.inputs[2]]
    mean = context[node.inputs[3]]
    variance = context[node.inputs[4]]
    if data_format == 'NHWC':
        # TF's FusedBatchNorm is only for 4D inputs
        x = _transpose_NHWC_to_NCHW(x)
        x = cb.batch_norm(x=x, mean=mean, variance=variance, gamma=scale,
                          beta=offset, epsilon=epsilon)
        x = _transpose_NCHW_to_NHWC(x, node.name)
    else:
        x = cb.batch_norm(x=x, mean=mean, variance=variance, gamma=scale,
                          beta=offset, epsilon=epsilon, name=node.name)
    # Inference only batch norm does not have meaningful outputs for
    # batch_mean, batch_variance etc.
    context.add(node.name, x)


@register_tf_op(tf_alias=['If'], override=True)
def StatelessIf(context, node):
    pred = context[node.inputs[0]][0]
    then_graph = context.get_graph(node.attr.get('then_branch'))
    else_graph = context.get_graph(node.attr.get('else_branch'))

    def then_fn():
        context.stack_func_inputs(context[node.inputs[0]])
        then_output_var = convert_graph(context, then_graph)
        context.unstack_func_inputs()
        return then_output_var

    def else_fn():
        context.stack_func_inputs(context[node.inputs[0]])
        else_output_var = convert_graph(context, else_graph)
        context.unstack_func_inputs()
        return else_output_var

    x = cb.cond(pred=pred, _true_fn=then_fn, _false_fn=else_fn, name=node.name)

    # wraps x as tuple for get_tuple that always follow the cond node.
    x = (x,) if not isinstance(x, (tuple, list)) else x

    context.add(node.name, x)


@register_tf_op(tf_alias=['While'], override=True)
def StatelessWhile(context, node):
    # inputs are loop_counter, max_iterations, [loop_vars]
    loop_vars = context[node.inputs[0]][2:]

    cond_graph = context.get_graph(node.attr.get('cond'))
    body_graph = context.get_graph(node.attr.get('body'))

    def cond(*loop_vars):
        context.stack_func_inputs(loop_vars)
        cond_output_vars = convert_graph(context, cond_graph)
        context.unstack_func_inputs()
        return cond_output_vars

    def body(*loop_vars):
        context.stack_func_inputs(loop_vars)
        body_output_vars = convert_graph(context, body_graph)
        context.unstack_func_inputs()
        return body_output_vars

    x = cb.while_loop(
        _cond=cond, _body=body, loop_vars=loop_vars, name=node.name)

    # wraps x as tuple for get_tuple that always follow the while node.
    x = (x,) if not isinstance(x, (tuple, list)) else x

    context.add(node.name, x)
