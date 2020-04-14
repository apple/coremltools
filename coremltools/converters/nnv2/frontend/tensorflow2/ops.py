from coremltools.converters.nnv2.frontend.tensorflow.ops import (
    _transpose_NHWC_to_NCHW,
    _transpose_NCHW_to_NHWC,
)

# TF 2.x now imports and registers all TF 1.x op against the new registry
# (separated from TF 1.x registry). Overwrite might needed in case the op
# semantics are different between TF 1.x and TF 2.x.
from coremltools.converters.nnv2.frontend.tensorflow.ops import *
from coremltools.converters.nnv2.frontend.tensorflow.tf_op_registry import register_tf_op


@register_tf_op
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
