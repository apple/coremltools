import logging
import six
import numpy as np

from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from .convert_utils import convert_graph
from .tf_op_registry import register_tf_op

def _transpose_NHWC_to_NCHW(x):
    return cb.transpose(x=x, perm=[0, 3, 1, 2])


def _transpose_NCHW_to_NHWC(x, node_name):
    return cb.transpose(x=x, perm=[0, 2, 3, 1], name=node_name)


def _transpose_NDHWC_to_NCDHW(x):
    return cb.transpose(x=x, perm=[0, 4, 1, 2, 3])


def _transpose_NCDHW_to_NDHWC(x, node_name):
    return cb.transpose(x=x, perm=[0, 2, 3, 4, 1], name=node_name)


@register_tf_op(tf_alias=['BiasAdd', 'AddV2'])
def Add(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.add(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Abs(context, node):
    x = context[node.inputs[0]]
    x = cb.abs(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Acos(context, node):
    x = context[node.inputs[0]]
    x = cb.acos(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def All(context, node):
    x = context[node.inputs[0]]
    axes = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_prod(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Any(context, node):
    x = context[node.inputs[0]]
    axes = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_sum(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def ArgMax(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    x = cb.reduce_argmax(x=x, axis=axis, name=node.name)
    context.add(node.name, x)


@register_tf_op
def ArgMin(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    x = cb.reduce_argmin(x=x, axis=axis, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Asin(context, node):
    x = context[node.inputs[0]]
    x = cb.asin(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Atan(context, node):
    x = context[node.inputs[0]]
    x = cb.atan(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Atanh(context, node):
    x = context[node.inputs[0]]
    x = cb.atanh(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def AvgPool(context, node):
    x = context[node.inputs[0]]
    in_shape = x.sym_type.get_shape()
    d_rank = len(in_shape) - 2
    data_format = node.attr.get('data_format', 'NHWC')
    ksize = node.attr.get('ksize', None)
    kernel_sizes = _pool_pads_or_strides(ksize, data_format, d_rank)
    strides = node.attr.get('strides', None)
    if strides is not None:
        strides = _pool_pads_or_strides(strides, data_format, d_rank)
    pad_type = node.attr['padding'].lower()
    if data_format == "NHWC":
        x = _transpose_NHWC_to_NCHW(x)
        x = cb.avg_pool(x=x, kernel_sizes=kernel_sizes, strides=strides,
                        pad_type=pad_type)
        x = _transpose_NCHW_to_NHWC(x, node.name)
    else:
        x = cb.avg_pool(x=x, kernel_sizes=kernel_sizes, strides=strides,
                        pad_type=pad_type, name=node.name)
    context.add(node.name, x)


@register_tf_op
def AvgPool3D(context, node):
    x = context[node.inputs[0]]
    d_rank = x.rank - 2
    data_format = node.attr.get('data_format', 'NDHWC')
    ksize = node.attr.get('ksize', None)
    kernel_sizes = _pool_pads_or_strides(ksize, data_format, d_rank)
    strides = node.attr.get('strides', None)
    if strides is not None:
        strides = _pool_pads_or_strides(strides, data_format, d_rank)
    pad_type = node.attr['padding'].lower()
    if data_format == 'NDHWC':
        x = _transpose_NDHWC_to_NCDHW(x)
        x = cb.avg_pool(x=x, kernel_sizes=kernel_sizes, strides=strides,
                        pad_type=pad_type)
        x = _transpose_NCDHW_to_NDHWC(x, node.name)
    else:
        x = cb.avg_pool(x=x, kernel_sizes=kernel_sizes, strides=strides,
                        pad_type=pad_type, name=node.name)

    context.add(node.name, x)


@register_tf_op
def BatchToSpaceND(context, node):
    x = context[node.inputs[0]]
    block_shape = context[node.inputs[1]].val

    if x.rank != 4:
        raise NotImplementedError('rank of input != 4 is not yet supported')
    if len(block_shape.flatten()) != 2:
        raise NotImplementedError('rank of spatial shape != 2 is not yet supported')
    if block_shape[0] != block_shape[1]:
        raise NotImplementedError('non-equal block shape is not yet supported')
    crops = context[node.inputs[2]].val
    needs_cropping = any(crops.flatten())

    x = cb.transpose(x=x, perm=[3, 0, 1, 2])

    x = cb.depth_to_space(x=x, block_size=block_shape[0])
    if needs_cropping:
        x = cb.crop(
                x=x,
                crop_height=[crops[0][0], crops[0][1]],
                crop_width=[crops[1][0], crops[1][1]],
            )

    x = cb.transpose(x=x, perm=[1, 2, 3, 0], name=node.name)
    context.add(node.name, x)


@register_tf_op
def Ceil(context, node):
    x = context[node.inputs[0]]
    x = cb.ceil(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Const(context, node):
    mode = 'immediate_value'
    # Heuristics to decide between file_value and immediate_value
    if node.value is not None and isinstance(node.value.val, (np.ndarray, np.generic)) and \
            node.value.val.size > 10:
        mode = 'file_value'
    x = cb.const(val=node.value.val, mode=mode, name=node.name)
    context.add(node.name, x)

def _conv2d_strides_or_dilations(name, value, data_format,
        default_value=1):
    if value is None:
        value = default_value
    if not isinstance(value, (int, list)):
        raise ValueError('{} must be an int or list'.format(name))

    if isinstance(value, int):
        return [value] * 2

    if len(value) == 1:
        return value * 2
    if len(value) == 2:
        return value
    if len(value) != 4:
        raise ValueError('{} must have length 1, 2, or 4'.format(name))

    if data_format == "NHWC":
        # Only support stride/dilation along N, C == 1
        if not (value[0] == value[3] == 1):
            raise ValueError('{} along N and C other than 1 not implemented'.format(name))
        return value[1:3]

    # "NCHW"
    if not (value[0] == value[1] == 1):
        raise ValueError('{} along N and C other than 1 not implemented'.format(name))
    return value[2:]

@register_tf_op
def Cos(context, node):
    x = context[node.inputs[0]]
    x = cb.cos(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Cosh(context, node):
    x = context[node.inputs[0]]
    x = cb.cosh(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Equal(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.equal(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Exp(context, node):
    x = context[node.inputs[0]]
    x = cb.exp(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Floor(context, node):
    x = context[node.inputs[0]]
    x = cb.floor(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def FloorDiv(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.floor_div(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Greater(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.greater(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def GreaterEqual(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.greater_equal(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Less(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.less(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def LessEqual(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.less_equal(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Log(context, node):
    x = context[node.inputs[0]]
    x = cb.log(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def LogicalAnd(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.logical_and(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def LogicalNot(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.logical_not(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def LogicalOr(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.logical_or(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def LogicalXor(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.logical_xor(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def LRN(context, node):
    x = context[node.inputs[0]]
    depth_radius = node.attr.get('depth_radius')
    size = (depth_radius * 2) + 1
    alpha = node.attr.get('alpha') * size
    beta = node.attr.get('beta')
    bias = node.attr.get('bias')
    x = _transpose_NHWC_to_NCHW(x)
    x = cb.local_response_norm(x=x, size=size, alpha=alpha, beta=beta, k=bias)
    x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def Maximum(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.maximum(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Minimum(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.minimum(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def FloorMod(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.mod(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Mul(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.mul(x=x, y=y, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Neg(context, node):
    x = context[node.inputs[0]]
    x = cb.mul(x=x, y=-1, name=node.name)
    context.add(node.name, x)


@register_tf_op
def NotEqual(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.not_equal(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Pow(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.pow(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def DepthwiseConv2dNative(context, node):
    # [kH, kW, C_in, multiplier]
    W_hwim = context[node.inputs[1]] # m = multiplier
    # [kH, kW, 1, C_in * multipler]
    shape_hw1o = list(W_hwim.shape[:2]) + [1, W_hwim.shape[2]*W_hwim.shape[3]]
    W_hw1o = cb.reshape(x=W_hwim, shape=shape_hw1o)
    # [C_in * multipler, 1, kH, kW]. Note that C_in * multiplier = C_out in
    # NNv2. C_in / group = 1 in depthwise conv.
    W_o1hw = cb.transpose(x=W_hw1o, perm=[3, 2, 0, 1])
    data_format = node.attr.get('data_format', 'NHWC')
    HW_dilations = _conv2d_strides_or_dilations(
            'dilations', node.attr.get('dilations'), data_format)
    HW_strides = _conv2d_strides_or_dilations(
            'strides', node.attr.get('strides'), data_format)

    pad_type = node.attr.get('padding')
    if pad_type not in ['VALID', 'SAME']:
        raise ValueError("Invalid padding type for tf.nn.depthwise_conv2d")

    pad_type = pad_type.lower()
    x = context[node.inputs[0]]
    C_in = x.shape[-1]
    if data_format == "NHWC":
        x = _transpose_NHWC_to_NCHW(x)
    # Only the last op should have the same name as node.name
    conv_name = node.name + 'x' if data_format == 'NHWC' else node.name
    x = cb.conv(x=x, W=W_o1hw, pad_type=pad_type, strides=HW_strides,
            dilations=HW_dilations, group=C_in, name=conv_name)
    if data_format == "NHWC":
        x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)

@register_tf_op
def Conv2D(context, node):
    W_hwio = context[node.inputs[1]]
    W_oihw = cb.transpose(x=W_hwio, perm=[3, 2, 0, 1])
    data_format = node.attr.get('data_format', 'NHWC')
    HW_dilations = _conv2d_strides_or_dilations(
            'dilations', node.attr.get('dilations'), data_format)
    HW_strides = _conv2d_strides_or_dilations(
            'strides', node.attr.get('strides'), data_format)

    pad_type = node.attr.get('padding')
    if not isinstance(pad_type, six.string_types):
        pad_type = "custom"
        raise NotImplementedError("Custom padding not implemented for TF")
    pad_type = pad_type.lower()
    x = context[node.inputs[0]]
    if data_format == "NHWC":
        x = _transpose_NHWC_to_NCHW(x)
    # Only the last op should have the same name as node.name
    conv_name = node.name + 'x' if data_format == 'NHWC' else node.name
    x = cb.conv(x=x, W=W_oihw, pad_type=pad_type, strides=HW_strides,
            dilations=HW_dilations, name=conv_name)
    if data_format == "NHWC":
        x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def DepthToSpace(context, node):
    x = context[node.inputs[0]]
    block_size = node.attr.get('block_size')
    data_format = node.attr.get('data_format', 'NHWC')
    if data_format == 'NHWC':
        x = _transpose_NHWC_to_NCHW(x)
        x = cb.depth_to_space(x=x, block_size=block_size)
        x = _transpose_NCHW_to_NHWC(x, node.name)
    else:
        x = cb.depth_to_space(x=x, block_size=block_size, name=node.name)
    context.add(node.name, x)


@register_tf_op
def EuclideanNorm(context, node):
    x = context[node.inputs[0]]
    axes = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_l2_norm(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def ExpandDims(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    if axis.op.op_type == "const" and (axis.val is not None
                                       and axis.val.size == 1):
        axis = axis.val[0] if axis.shape == (1,) else axis.val
    else:
        raise ValueError("Expand Dims: Invalid value for parameter axis")
    x = cb.expand_dims(x=x, axis=axis, name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=['FusedBatchNormV2'])
def FusedBatchNorm(context, node):
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
        x = _transpose_NCHW_to_NHWC(x, node.name+':0')
    else:
        x = cb.batch_norm(x=x, mean=mean, variance=variance, gamma=scale,
                          beta=offset, epsilon=epsilon, name=node.name+':0')
    # Inference only batch norm does not have meaningful outputs for
    # batch_mean, batch_variance etc.g
    context.add(node.name, [x, mean, variance])


@register_tf_op
def Fill(context, node):
    shape = context[node.inputs[0]]
    value = context[node.inputs[1]]
    x = cb.fill(shape=shape, value=value, name=node.name)
    context.add(node.name, x)


@register_tf_op
def RealDiv(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.real_div(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Rsqrt(context, node):
    x = context[node.inputs[0]]
    x = cb.rsqrt(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Sub(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.sub(x=x, y=y, name=node.name)
    context.add(node.name, x)

@register_tf_op
def StopGradient(context, node):
    Identity(context, node)

@register_tf_op
def Identity(context, node):
    x = context[node.inputs[0]]
    # TODO rdar://60644469 NNv2 -> NNv1 backend output name uses op output var name, not op.name for all ops
    if len(node.outputs) == 0:
        x = cb.mul(x=x, y=1.0, name=node.name)
        context.add(node.name, x)
    else:
        # Don't change tfssa. Just make downstream ops reference the pre-identity op.
        context.add(node.name, x, is_new_var=False)

@register_tf_op
def Placeholder(context, node):
    # no-op as we add Placeholder separately.
    pass

def _pool_pads_or_strides(tf_spec, data_format, d_rank):
    if tf_spec is None:
        d_spec = [1]*d_rank
    elif not isinstance(tf_spec, list):
        d_spec = [tf_spec]*d_rank
    elif len(tf_spec) == 2:
        d_spec = tf_spec
    elif len(tf_spec) == 4:
        if data_format == 'NHWC':
            d_spec = tf_spec[1:3]
        else:
            d_spec = tf_spec[2:]
    elif len(tf_spec) == 5:
        if data_format == 'NDHWC':
            d_spec = tf_spec[1:4]
        else:
            # NCDHW
            d_spec = tf_spec[2:]
    else:
        raise ValueError('Unsupported tf_spec: %s' % tf_spec)
    return d_spec


@register_tf_op(tf_alias=['BatchMatMul'])
def MatMul(context, node):
    a = context[node.inputs[0]]
    b = context[node.inputs[1]]
    transpose_a = node.attr.get('adj_x', False) or node.attr.get('transpose_a', False)
    transpose_b = node.attr.get('adj_y', False) or node.attr.get('transpose_b', False)
    x = cb.matmul(x=a, y=b, transpose_x=transpose_a,
                  transpose_y=transpose_b, name=node.name)
    context.add(node.name, x)


@register_tf_op
def MaxPool(context, node):
    x = context[node.inputs[0]]
    in_shape = x.sym_type.get_shape()
    d_rank = len(in_shape) - 2
    data_format = node.attr.get('data_format', 'NHWC')
    ksize = node.attr.get('ksize', None)
    kernel_sizes = _pool_pads_or_strides(ksize, data_format, d_rank)
    strides = node.attr.get('strides', None)
    if strides is not None:
        strides = _pool_pads_or_strides(strides, data_format, d_rank)
    pad_type = node.attr['padding'].lower()
    if data_format == "NHWC":
        x = _transpose_NHWC_to_NCHW(x)
        x = cb.max_pool(x=x, kernel_sizes=kernel_sizes, strides=strides,
                        pad_type=pad_type)
        x = _transpose_NCHW_to_NHWC(x, node.name)
    else:
        x = cb.max_pool(x=x, kernel_sizes=kernel_sizes, strides=strides,
                        pad_type=pad_type, name=node.name)
    context.add(node.name, x)

@register_tf_op
def MaxPool3D(context, node):
    x = context[node.inputs[0]]
    d_rank = x.rank - 2
    data_format = node.attr.get('data_format', 'NDHWC')
    ksize = node.attr.get('ksize', None)
    kernel_sizes = _pool_pads_or_strides(ksize, data_format, d_rank)
    strides = node.attr.get('strides', None)
    if strides is not None:
        strides = _pool_pads_or_strides(strides, data_format, d_rank)
    pad_type = node.attr['padding'].lower()
    if data_format == 'NDHWC':
        x = _transpose_NDHWC_to_NCDHW(x)
        x = cb.max_pool(x=x, kernel_sizes=kernel_sizes, strides=strides,
                        pad_type=pad_type)
        x = _transpose_NCDHW_to_NDHWC(x, node.name)
    else:
        x = cb.max_pool(x=x, kernel_sizes=kernel_sizes, strides=strides,
                        pad_type=pad_type, name=node.name)

    context.add(node.name, x)


@register_tf_op
def MatrixBandPart(context, node):
    x = context[node.inputs[0]]
    lower = context[node.inputs[1]]
    upper = context[node.inputs[2]]
    x = cb.band_part(x=x, lower=lower, upper=upper, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Max(context, node):
    x = context[node.inputs[0]]
    axes = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_max(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Min(context, node):
    x = context[node.inputs[0]]
    axes = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_min(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Prod(context, node):
    x = context[node.inputs[0]]
    axes = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_prod(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Cast(context, node):
    Round(context, node)

@register_tf_op
def Round(context, node):
    x = context[node.inputs[0]]
    x = cb.round(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Sign(context, node):
    x = context[node.inputs[0]]
    x = cb.sign(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Sin(context, node):
    x = context[node.inputs[0]]
    x = cb.sin(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Sinh(context, node):
    x = context[node.inputs[0]]
    x = cb.sinh(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Slice(context, node):
    x = context[node.inputs[0]]
    begin = context[node.inputs[1]]
    size = context[node.inputs[2]]
    res = cb.slice_by_size(x=x, begin=begin, size=size, name=node.name)
    context.add(node.name, res)

@register_tf_op
def Sqrt(context, node):
    x = context[node.inputs[0]]
    x = cb.sqrt(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Square(context, node):
    x = context[node.inputs[0]]
    x = cb.square(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def StridedSlice(context, node):
    x = context[node.inputs[0]]
    begin = context[node.inputs[1]]
    end = context[node.inputs[2]]
    stride = context[node.inputs[3]]

    def bitmask_to_array(bit):
        arr = []
        while bit > 0:
            if bit & 1:
                arr.append(True)
            else:
                arr.append(False)
            bit >>= 1
        return arr

    begin_mask = bitmask_to_array(node.attr.get('begin_mask', 0))
    end_mask = bitmask_to_array(node.attr.get('end_mask', 0))
    squeeze_mask = bitmask_to_array(node.attr.get('shrink_axis_mask', 0))
    ellipsis_mask = bitmask_to_array(node.attr.get('ellipsis_mask', 0))

    def _pad_mask(x, begin, end, stride, begin_mask, end_mask, squeeze_mask, ellipsis_mask):
        # This function pad the masks, stride, begin and end to the same rank as the input tensor.
        if begin.rank != 1:
            raise ValueError("begin should be 1-D tensor, got {}-D tensor instead".format(self.begin.rank))
        if end.rank != 1:
            raise ValueError("end should be 1-D tensor, got {}-D tensor instead".format(self.end.rank))

        # check if inputs can be determined
        begin_cache = begin
        end_cache = end
        begin = [] if begin.sym_val is None else begin.sym_val.tolist()
        end = [] if end.sym_val is None else end.sym_val.tolist()
        stride = [] if stride is None else stride.val.tolist()

        # pad masks function
        x_rank = x.rank
        def pad_array(arr, max_rank, idx, default_value):
            '''
            This function pads the arr to x_rank with default_value.
            idx is the index where ellipis_mask = True.
            max_rank is the maximum rank of the masks, stride, begin and end.
            '''
            mask = arr[:]
            mask += [default_value]*(x_rank-len(mask))
            new_mask = []

            for i in range(max_rank):
                num = 1 if i != idx else x_rank-max_rank+1
                new_mask += [mask[i]]*num
            return new_mask

        mask_list = [begin_mask, end_mask, squeeze_mask, ellipsis_mask, stride, begin, end]
        max_rank = max([len(arr) for arr in mask_list])

        # If ellipsis_mask is given, the last element of it would be True
        # Otherwise, we simply pad each mask by appending default value
        if ellipsis_mask != []:
            rank = max_rank
            idx = len(ellipsis_mask)-1
        else:
            rank = x_rank
            idx = -1

        begin_mask = pad_array(begin_mask, rank, idx, False)
        end_mask = pad_array(end_mask, rank, idx, False)
        squeeze_mask = pad_array(squeeze_mask, rank, idx, False)
        ellipsis_mask = pad_array(ellipsis_mask, rank, idx, False)
        stride = pad_array(stride, rank, idx, 1)

        # pad begin and end if they are determined during compile time
        if begin != []:
            begin = pad_array(begin, max_rank, idx, 0)
        if end != []:
            end = pad_array(end, max_rank, idx, 0)

        # make sure begin_mask, end_mask, and stride are consistent with ellipsis mask
        # begin_mask and end_mask should be True, and stride should be 1.
        for i, mask in enumerate(ellipsis_mask):
            if mask:
                begin_mask[i] = True
                end_mask[i] = True
                stride[i] = 1

        # return if begin and end are run-time determined
        if begin == [] and end == []:
            begin = begin_cache
            end   = end_cache

        # check which mask is adding by our default value
        # This happens when the given index is less than the tensor rank,
        # for instance, indexing a 3D tensor A with A[:1, :1] is equivalent to
        # A[:1, :1, :]. In this case we should append True to begin_mask and end_mask

        if ellipsis_mask == [False]*x_rank:
            for i in range(max_rank, x_rank):
                begin_mask[i] = True
                end_mask[i] = True

        return begin, end, stride, begin_mask, end_mask, squeeze_mask

    begin, end, stride, begin_mask, end_mask, squeeze_mask =  \
    _pad_mask(x, begin, end, stride, begin_mask, end_mask, squeeze_mask, ellipsis_mask)

    x = cb.slice_by_index(x=x, name=node.name, begin=begin, end=end, stride=stride,
                          begin_mask=begin_mask, end_mask=end_mask, squeeze_mask=squeeze_mask)

    context.add(node.name, x)

@register_tf_op
def Sum(context, node):
    x = context[node.inputs[0]]
    axes = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_sum(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Tan(context, node):
    x = context[node.inputs[0]]
    x = cb.tan(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def get_tuple(context, node):
    x = context[node.inputs[0]]
    if not isinstance(x, (list, tuple)):
        raise ValueError("Op {} should return multiple output.".format(
            node.inputs[0]))
    idx = node.attr['index']
    context.add(node.name, x[idx], is_new_var=False)

@register_tf_op
def Mean(context, node):
    x = context[node.inputs[0]]
    axes = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_mean(x=x, axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)

@register_tf_op
def MirrorPad(context, node):
    x = context[node.inputs[0]]
    pad = context[node.inputs[1]]
    constant_val = node.attr.get('constant_val', 0.0)

    if pad is None:
        raise ValueError("TF `paddings` in Pad op must be const.")

    mode = node.attr.get('mode', 'reflect').lower()
    in_rank = len(x.sym_type.get_shape())

    if in_rank > 5 or in_rank < 2:
        raise ValueError('Unsupported Pad configuration with input rank {}!'.format(str(in_rank)))

    if pad.val.shape != (in_rank, 2):
        raise ValueError('Padding must have length as input tensor rank.')

    pad = pad.val

    # get axix which is non zero
    non_zero_axis = []
    for i in range(len(pad)):
        if not all(pad[i] == 0):
            non_zero_axis.append(i)

    if len(non_zero_axis) > 2:
        raise ValueError ("Unsupported configuration for Pad layer!")

    # make padding a 2 x 2 tensor if len(non_zero_axis) < 2
    if len(non_zero_axis) == 0:
        non_zero_axis = [0, 1]

    if len(non_zero_axis) == 1:
        if non_zero_axis[0] != len(pad)-1:
            non_zero_axis.append(len(pad)-1)
        else:
            non_zero_axis = [0, non_zero_axis[0]]

    # transpose the input such that the padding dim is the last two
    perm = [i for i in range(in_rank) if i not in non_zero_axis] + non_zero_axis
    x = cb.transpose(x=x, perm=perm, name=node.name + '_transpose_1')
    pad = pad[non_zero_axis,:]
    pad = pad.reshape(-1)
    x = cb.pad(x=x, pad=pad, name=node.name + '_pad', constant_val=constant_val, mode=mode)
    inverse_perm = [-1]*len(perm)
    for i, index in enumerate(perm):
        inverse_perm[index] = i
    x = cb.transpose(x=x, perm=inverse_perm, name=node.name)

    context.add(node.name, x)

@register_tf_op
def Pad(context, node):
    x = context[node.inputs[0]]
    pad = context[node.inputs[1]]
    if pad is None:
        raise ValueError("TF `paddings` in Pad op must be const.")

    mode = node.attr.get('mode', 'constant').lower()
    constant_val = node.attr.get('constant_val', 0.0)
    in_rank = len(x.sym_type.get_shape())

    if in_rank > 5:
        raise ValueError('Unsupported Pad configuration!')

    pad = pad.val.reshape(-1)
    x = cb.pad(x=x, pad=pad, name=node.name, mode=mode, constant_val=constant_val)
    context.add(node.name, x)

@register_tf_op
def Relu(context, node):
    x = context[node.inputs[0]]
    x = cb.relu(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Relu6(context, node):
    x = context[node.inputs[0]]
    x = cb.relu6(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Reshape(context, node):
    x = context[node.inputs[0]]
    new_shape = context[node.inputs[1]]
    x = cb.reshape(x=x, shape=new_shape, name=node.name)
    context.add(node.name, x)

@register_tf_op(tf_alias=['ReverseV2'])
def Reverse(context, node):
    x = context[node.inputs[0]]
    axes = context[node.inputs[1]]
    x = cb.reverse(x=x, axes=axes, name=node.name)
    context.add(node.name, x)

@register_tf_op
def ReverseSequence(context, node):
    x = context[node.inputs[0]]
    lengths = context[node.inputs[1]]
    seq_axis = node.attr.get('seq_dim')
    batch_axis = node.attr.get('batch_dim')
    x = cb.reverse_sequence(x=x, lengths=lengths, seq_axis=seq_axis,
                            batch_axis=batch_axis, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Transpose(context, node):
    x = context[node.inputs[0]]
    perm = context[node.inputs[1]]
    x = cb.transpose(x=x, perm=perm, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Squeeze(context, node):
    x = context[node.inputs[0]]
    axes = node.attr.get('squeeze_dims', [])
    if axes == []:
        axes = None
    x = cb.squeeze(x=x, axes=axes, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Multinomial(context, node):
    x = context[node.inputs[0]]
    size = context[node.inputs[1]]
    x = cb.random_categorical(x=x, size=size, name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=['Elu'])
def ELU(context, node):
    x = context[node.inputs[0]]
    x = cb.elu(x=x, alpha=1.0, name=node.name)
    context.add(node.name, x)

@register_tf_op(tf_alias=['Erf'])
def ERF(context, node):
    x = context[node.inputs[0]]
    x = cb.erf(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op(tf_alias=['LeakyRelu'])
def LeakyReLU(context, node):
    x = context[node.inputs[0]]
    alpha = node.attr["alpha"]
    x = cb.leaky_relu(x=x, alpha=alpha, name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=['SelectV2'])
def Select(context, node):
    cond = context[node.inputs[0]]
    a = context[node.inputs[1]]
    b = context[node.inputs[2]]
    x = cb.select(cond=cond, a=a, b=b, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Sigmoid(context, node):
    x = context[node.inputs[0]]
    x = cb.sigmoid(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Softplus(context, node):
    x = context[node.inputs[0]]
    x = cb.softplus(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Softsign(context, node):
    x = context[node.inputs[0]]
    x = cb.softsign(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Softmax(context, node):
    logit = context[node.inputs[0]]
    axis = node.attr.get('axis')
    x = cb.softmax(logit=logit, axis=axis, name=node.name)
    context.add(node.name, x)

@register_tf_op
def SpaceToBatchND(context, node):
    x = context[node.inputs[0]]
    block_shape = context[node.inputs[1]].val
    if x.rank != 4:
        raise NotImplementedError('rank of input != 4 is not yet supported')
    if len(block_shape.flatten()) != 2:
        raise NotImplementedError('rank of spatial shape != 2 is not yet supported')
    if block_shape[0] != block_shape[1]:
        raise NotImplementedError('non-equal block shape is not yet supported')
    paddings = context[node.inputs[2]].val
    needs_paddings = any(paddings.flatten())

    x = cb.transpose(x=x, perm=[3, 0, 1, 2])

    if needs_paddings:
        x = cb.pad(x=x, pad=paddings.flatten(), mode='constant')

    x = cb.space_to_depth(x=x, block_size=block_shape[0])
    x = cb.transpose(x=x, perm=[1, 2, 3, 0], name=node.name)

    context.add(node.name, x)


@register_tf_op
def SpaceToDepth(context, node):
    x = context[node.inputs[0]]
    block_size = node.attr.get('block_size')
    data_format = node.attr.get('data_format', 'NHWC')
    if data_format == 'NHWC':
        x = _transpose_NHWC_to_NCHW(x)
        x = cb.space_to_depth(x=x, block_size=block_size)
        x = _transpose_NCHW_to_NHWC(x, node.name)
    else:
        x = cb.space_to_depth(x=x, block_size=block_size, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Tanh(context, node):
    x = context[node.inputs[0]]
    x = cb.tanh(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=['TopKV2'])
def TopK(context, node):
    x = context[node.inputs[0]]
    k = context[node.inputs[1]]
    x = cb.topk(x=x, k=k.val, axis=-1, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Cumsum(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    exclusive = node.attr.get('exclusive', False)
    reverse = node.attr.get('reverse', False)
    x = cb.cumsum(x=x, axis=axis, exclusive=exclusive, reverse=reverse, name=node.name)
    context.add(node.name, x)

@register_tf_op
def GatherV2(context, node):
    x = context[node.inputs[0]]
    indices = context[node.inputs[1]]
    axis = context[node.inputs[2]]
    x = cb.gather(x=x, indices=indices, axis=axis, name=node.name)
    context.add(node.name,x)

@register_tf_op
def GatherNd(context, node):
    x = context[node.inputs[0]]
    indices = context[node.inputs[1]]
    x = cb.gather_nd(x=x, indices=indices, name=node.name)
    context.add(node.name,x)

@register_tf_op
def Tile(context, node):
    x = context[node.inputs[0]]
    reps = context[node.inputs[1]]
    x = cb.tile(x=x, reps = reps, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Where(context, node):
    x = context[node.inputs[0]]
    x = cb.non_zero(x=x, name=node.name)
    context.add(node.name, x)

@register_tf_op
def SquaredDifference(context, node):
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    x = cb.sub(x=x, y=y)
    x = cb.square(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Conv2DBackpropInput(context, node):
    # Output shape: [N, H_out, W_out, C_out]
    output_shape = context[node.inputs[0]].val
    # Weight shape: [H, W, C_out, C_in]
    weight = context[node.inputs[1]]
    # Input shape: [N, H_in, W_in, C_in]
    x = context[node.inputs[2]]

    data_format = node.attr.get('data_format', 'NHWC')
    HW_dilations = _conv2d_strides_or_dilations(
                        'dilations', node.attr.get('dilations'), data_format)
    HW_strides = _conv2d_strides_or_dilations(
                        'strides', node.attr.get('strides'), data_format)
    pad_type = node.attr.get('padding')

    if not isinstance(pad_type, six.string_types):
        pad_type = "custom"
        raise NotImplementedError("Custom padding not implemented for TF")

    pad_type = pad_type.lower()
    # CoreML expects input to be in NCHW format
    # Transpose input to NCHW format
    if data_format == "NHWC":
        x = _transpose_NHWC_to_NCHW(x)
        if output_shape is not None:
            output_shape = [output_shape[1], output_shape[2]]
    else:
        if output_shape is not None:
            output_shape = [output_shape[2], output_shape[3]]

    # Only the last op should have the same name as node.name
    conv_name = node.name + 'x' if data_format == 'NHWC' else node.name
    # add Conv Tranpose
    x = cb.conv_transpose(x=x, weight=weight, pad_type=pad_type, output_shape=output_shape, strides=HW_strides,
                          dilations=HW_dilations, name=conv_name)

    # Convert NCHW output back to NHWC format
    if data_format == "NHWC":
        x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def Range(context,node):
    start = context[node.inputs[0]]
    end = context[node.inputs[1]]
    step = context[node.inputs[2]]
    x = cb.range_1d(start=start, end=end, step=step, name=node.name)
    context.add(node.name, x)


@register_tf_op
def OneHot(context,node):
    indices = context[node.inputs[0]]
    depth = context[node.inputs[1]]
    on_value = context[node.inputs[2]]
    off_value = context[node.inputs[3]]
    axis = node.attr.get('axis', -1)
    x = cb.one_hot(indices=indices,one_hot_vector_size=depth, axis = axis,
                  on_value = on_value, off_value = off_value, name=node.name)
    context.add(node.name, x)


@register_tf_op(tf_alias=['NonMaxSuppressionV3'])
def NonMaxSuppression(context, node):
    boxes = context[node.inputs[0]]
    scores = context[node.inputs[1]]
    max_boxes = context[node.inputs[2]]
    iou_threshold = context[node.inputs[3]]
    score_threshold = context[node.inputs[4]]
    if score_threshold.val == float('-inf'):
        # TensorFlow's default value for score_threshold, Core ML does not
        # have float('-inf') support, converted to minimum float32 instead
        score_threshold = -3.4e38
    boxes = cb.expand_dims(x=boxes, axis=0)
    scores = cb.expand_dims(x=scores, axis=0)
    scores = cb.expand_dims(x=scores, axis=-1)
    _, _, x, _ = cb.non_maximum_suppression(
        boxes=boxes,
        scores=scores,
        max_boxes=max_boxes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold)
    x = cb.squeeze(x=x, axes=[0], name=node.name)
    context.add(node.name, x)


@register_tf_op
def Shape(context, node):
    x = context[node.inputs[0]]
    x = cb.shape(x=x, name=node.name)
    context.add(node.name, x)


@register_tf_op
def ResizeNearestNeighbor(context, node):
    # "ResizeNearestNeighbor" op in TF is always in the channel last mode
    # instead of upsample factor, it uses output size, which is the second input
    x = context[node.inputs[0]]

    input_shape = x.shape # (N,Hin,Win,C)
    if len(input_shape) != 4:
        raise ValueError('\"ResizeNearestNeighbor\" op: input rank is not 4')
    Hin, Win = input_shape[1:3]

    if context[node.inputs[1]].val is None:
        raise ValueError(
            '\"ResizeNearestNeighbor\" op: the second input, which is the output size, must be known statically')

    if len(context[node.inputs[1]].val) != 2:
        raise ValueError(
            '\"ResizeNearestNeighbor\" op: the second input, which is the output size, must have 2 elements')

    Hout, Wout = context[node.inputs[1]].val

    if not (isinstance(Hout, np.int32) and isinstance(Wout, np.int32)):
        raise ValueError(
            '\"ResizeNearestNeighbor\" op: the second input, which is the output size, must have elements of type int32')

    if (Hout % Hin > 0 or Wout % Win > 0):
        raise ValueError('\"ResizeNearestNeighbor\" op: fractional upsampling factors not supported')

    scaling_factor_h = int(Hout / Hin)
    scaling_factor_w = int(Wout / Win)

    # first transpose to from channel last to channel first format for coreml
    x = _transpose_NHWC_to_NCHW(x)
    # add the upsample layer
    x = cb.upsample_nearest_neighbor(x=x,
                                     upscale_factor_height=scaling_factor_h,
                                     upscale_factor_width=scaling_factor_w,
                                     name=node.name + '_channel_first_upsample')
    # transpose again
    x = _transpose_NCHW_to_NHWC(x, node.name)

    context.add(node.name, x)


@register_tf_op
def ResizeBilinear(context, node):
    # "ResizeBilinear" op in TF is always in the channel last mode
    # second input is the output size

    x = context[node.inputs[0]]
    input_shape = x.shape  # (N,Hin,Win,C)
    if len(input_shape) != 4:
        raise ValueError('\"ResizeBilinear\" op: input rank is not 4')
    Hin, Win = input_shape[1:3]

    if context[node.inputs[1]].val is None:
        raise ValueError(
            '\"ResizeBilinear\" op: the second input, which is the output size, must be known statically')

    if len(context[node.inputs[1]].val) != 2:
        raise ValueError(
            '\"ResizeBilinear\" op: the second input, which is the output size, must have 2 elements')

    Hout, Wout = context[node.inputs[1]].val

    if not (isinstance(Hout, np.int32) and isinstance(Wout, np.int32)):
        raise ValueError(
            '\"ResizeBilinear\" op: the second input, which is the output size, must have elements of type int32')

    align_corners = node.attr.get('align_corners', False)
    half_pixel_centers = node.attr.get('half_pixel_centers', False)

    # first transpose to from channel last to channel first format for coreml
    x = _transpose_NHWC_to_NCHW(x)

    # add either the resize_bilinear layer or the upsample layer

    # [align_corners = True, half_pixel_centers = False]
    if align_corners and not half_pixel_centers:
        x = cb.resize_bilinear(x=x,
                               target_size_height=Hout,
                               target_size_width=Wout,
                               sampling_mode = "STRICT_ALIGN_CORNERS",
                               name=node.name + '_channel_first_resize_bilinear')

    # [align_corners = False, half_pixel_centers = False]
    elif not align_corners and not half_pixel_centers:
        x = cb.resize_bilinear(x=x,
                               target_size_height=Hout,
                               target_size_width=Wout,
                               sampling_mode = "DEFAULT",
                               name=node.name + '_channel_first_resize_bilinear')

    # [align_corners = False, half_pixel_centers = True]
    elif not align_corners and half_pixel_centers:
        x = cb.upsample_bilinear(x=x,
                                 scale_factor_height= (float(Hout) + 1e-2) / float(Hin),
                                 scale_factor_width= (float(Wout) + 1e-2) / float(Win),
                                 align_corners=False,
                                 name=node.name + '_channel_first_upsample_bilinear')

    else:
        # we should not come here since TF does not support align_corners=True and half_pixel_centers=True
        raise ValueError(
            '\"ResizeBilinear\" op: \"align_corners\" and \"half_pixel_centers\" are both True and this mode is not supported')


    # transpose again
    x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def make_tuple(context, node):
    res = tuple([context[in_name] for in_name in node.inputs])
    context.add(node.name, res)

@register_tf_op
def function_entry(context, node):
    if context.get_func_inputs() is None:
        msg = 'function_entry requires function inputs stored in ' + \
                'context.curr_func_inputs'
        raise ValueError(msg)
    context.add(node.name, context.get_func_inputs())

@register_tf_op(tf_alias=['while'])
def While(context, node):
    # TODO(rdar://60331615): Add/fix TensorFlow 2 control flow
    #
    # TF while will never have break statement, because break can always be
    # transformed into while and condition. Example:
    #
    #   while pred:
    #    a = op1(...)
    #    if a == 0:
    #      break
    #    b = op2(...)
    #
    # is equivalent to
    #
    #   while pred and not break_a:
    #    a = op1(...)
    #    break_a = a == 0
    #    if not break_a:
    #      b = op2(...)

    # node.inputs[0] == 'make_tuple_X' (always a make_tuple)
    loop_vars = context[node.inputs[0]]  # python tuple of Vars
    cond_graph = context.get_graph(node.attr['cond_function'])
    body_graph = context.get_graph(node.attr['body_function'])
    def cond(*loop_vars):
        context.stack_func_inputs(loop_vars)

        # convert_graph uses context to convert cond_graph. During conversion
        # it constructs operations (cb.some_op). Note that cond(*loop_vars) is
        # only evaluated inside while_loop's type_inference(), not here. In
        # other words, we use python's deferred function evaluation to defer
        # the SSA block construction until inside while_loop Operation.
        res = convert_graph(context, cond_graph)
        # Done with translating the function
        context.unstack_func_inputs()
        return res
    def body(*loop_vars):
        context.stack_func_inputs(loop_vars)
        res = convert_graph(context, body_graph)
        # Done with translating the function
        context.unstack_func_inputs()
        return res
    x = cb.while_loop(_cond=cond, _body=body, loop_vars=loop_vars,
            name=node.name)
    # wraps x as tuple for get_tuple that always follow the while node.
    if not isinstance(x, (tuple, list)):
        x = (x,)
    context.add(node.name, x)

@register_tf_op
def iff(context, node):
    pred = context[node.inputs[0]]

    # this is always a tensor, as TF uses one iff op for each returned value.
    #
    # Example TF program:
    #
    #  x = tf.placeholder(tf.float32, shape=(1,))
    #  y = tf.placeholder(tf.float32, shape=(1,))
    #  z = tf.multiply(x, y)
    #  pred = tf.less(tf.math.reduce_mean(x), tf.math.reduce_mean(y))
    #  def true_fn(): return tf.add(x, z), x
    #  def false_fn(): return tf.square(y), z
    #  res = tf.cond(pred, true_fn, false_fn)
    #
    # There will be 2 iffs:
    #
    # iff('cond/pred_id', 'cond/Add', 'cond/Square')
    # iff('cond/pred_id', 'cond/Add/Switch', 'cond/Switch_1')
    #
    # where
    #   'cond/pred_id': pred
    #   'cond/Add': tf.add(x, z)
    #   'cond/Square': tf.square(y)
    #   'cond/Add/Switch': x
    #   'cond/Switch_1': z
    #
    # And both branches are executed, and one of the results will be
    # discarded at iff nodes.
    #
    # Note that the above program would translate to two cond ops, each with
    # two blocks.
    true_output_var = context[node.inputs[1]]
    false_output_var = context[node.inputs[2]]

    def true_fn():
        return cb.identity(x=true_output_var)

    def false_fn():
        return cb.identity(x=false_output_var)

    x = cb.cond(pred=pred, _true_fn=true_fn, _false_fn=false_fn,
                name=node.name)
    context.add(node.name, x)

@register_tf_op
def Concat(context, node):
    values = [context[input] for input in node.inputs[1:]]
    axis = context[node.inputs[0]]
    x = cb.concat(values=values, axis=axis, name=node.name)
    context.add(node.name, x)

@register_tf_op
def ConcatV2(context, node):
    values = [context[input] for input in node.inputs[:-1]]
    axis = context[node.inputs[-1]]
    x = cb.concat(values=values, axis=axis, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Pack(context, node):
    values = [context[name] for name in node.inputs]
    axis = node.attr['axis']
    if axis < 0:
        # TF axis = -1 creates new dim at the end
        axis += values[0].rank + 1
    x = cb.stack(values=values, axis=axis, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Unpack(context, node):
    x = context[node.inputs[0]]
    axis = int(node.attr['axis'])
    num_splits = node.attr.get('num', None)
    if num_splits is None:
        num_splits = x.shape[axis]
    y = cb.split(x=x, num_splits=num_splits, axis=axis, name=node.name + "_unsqueezed")
    output_vars = []
    for i in range(num_splits):
        output_vars.append(cb.squeeze(x=y[i], axes=[axis], name=node.name + ":{}".format(i)))

    context.add(node.name, output_vars)



@register_tf_op
def SplitV(context, node):
    x = context[node.inputs[0]]
    split_sizes = context[node.inputs[1]]
    axis = context[node.inputs[2]]
    if 'num_split' not in node.attr:
        raise ValueError('num_splits not found in TF op {}'.format(node.name))
    num_splits = node.attr['num_split']
    if num_splits == 1:
        Identity(context, node)
    else:
        x = cb.split(x=x, num_splits=num_splits, split_sizes=split_sizes,
                axis=axis, name=node.name)
        context.add(node.name, x)

@register_tf_op
def Split(context, node):
    axis = context[node.inputs[0]]
    x = context[node.inputs[1]]
    if 'num_split' not in node.attr:
        raise ValueError('num_splits not found in TF op {}'.format(node.name))
    num_splits = node.attr['num_split']
    if num_splits == 1:
        if len(node.outputs) == 0:
            x = cb.mul(x=x, y=1.0, name=node.name)
            context.add(node.name, x)
        else:
            # Don't change tfssa. Just make downstream ops reference the pre-identity op.
            context.add(node.name, x, is_new_var=False)
    else:
        x = cb.split(x=x, num_splits=num_splits, axis=axis, name=node.name)
        context.add(node.name, x)
        # TODO (rdar://60358242) If tf.split output is returned, there's no
        # get_tuple nodes. Some graph pass is needed. Example:
        #
        #    x = tf.placeholder(tf.float32, shape=input_shape1)
        #    res = tf.split(x, 3, axis=0)
        #
        # res are ['split:0', 'split:1', 'split']
        #
        # but node.outputs == ['gto_1', 'gto_2', 'gto_3']


@register_tf_op
def CropAndResize(context, node):
    x = context[node.inputs[0]]
    input_shape = x.shape  # (B, h_in, w_in, C)
    if len(input_shape) != 4:
        raise ValueError('\"CropResize\" op: expected input rank 4, got {}'.format(x.rank))
    Hin, Win = input_shape[1:3]

    const_box_info = True
    if context[node.inputs[1]].val is None or context[node.inputs[2]].val is None:
        const_box_info = False
        # TODO: rdar://60540725 ([NNv2] [SSAv2] CropResize layer requires concat on float32 and int32 input)
        raise ValueError('"CropResize" op: expected boxes and box_indices to be known during conversion!')

    crop_size = context[node.inputs[3]].val
    method = 'bilinear' if len(node.inputs) < 5 else context[node.inputs[4]].val
    extrapolation_value = 1.0 if len(node.inputs) < 6 else context[node.inputs[5]].val

    # CoreML index information along with boxes
    if const_box_info:
        boxes = context[node.inputs[1]].val
        box_indices = context[node.inputs[2]].val
        box_indices = np.expand_dims(box_indices, axis=1)
        boxes = np.concatenate([box_indices, boxes], axis=1)
        # CoreML expects boxes/ROI in
        # [N, 1, 5, 1, 1] format
        boxes = boxes.reshape(boxes.shape[0], 1, boxes.shape[1], 1, 1)
    else:
        box_indices = context[node.inputs[2]]
        boxes = context[node.inputs[1]]
        box_indices = cb.expand_dims(x=box_indices, axis=1)
        # rdar://60540725 ([NNv2] [SSAv2] CropResize layer requires concat on float32 and int32 input)
        boxes = cb.concat(values=(box_indices, boxes), axis=1)
        # TODO: Dynamic rank: Use GetShape and select indices dynamically
        boxes = cb.reshape(x=boxes, shape=[boxes.shape[0], 1, boxes.shape[1], 1, 1])

    # Get Height and Width of crop
    h_out, w_out = crop_size[0], crop_size[1]

    # TF `nearest` mode not supported
    method_map = {'bilinear':'ALIGN_CORNERS'}
    if method not in method_map:
        raise ValueError(
            '\"\CropResize\" op: Unsupported method {}. Supports {}'.format(method, method_map.keys()))
    method = method_map[method]

    # TF input format: [B, h_in, w_in, C]
    # CoreML input format: [B, C, h_in, w_in]
    x = _transpose_NHWC_to_NCHW(x)


    # Crop Resize
    x = cb.crop_resize(x=x,
                       roi=boxes,
                       target_height=h_out,
                       target_width=w_out,
                       normalized_coordinates=True,
                       spatial_scale=extrapolation_value,
                       box_coordinate_mode='CORNERS_HEIGHT_FIRST',
                       sampling_mode=method)

    # CoreML output format: [N, 1, C, h_out, w_out]
    # TF output format: [N, h_out, h_out, C]
    x = cb.squeeze(x=x, axes=[1])
    x = _transpose_NCHW_to_NHWC(x, node.name)
    context.add(node.name, x)


@register_tf_op
def NoOp(context, node):
    pass
