import logging
import six
import numpy as np

from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from .convert_utils import convert_graph
from .tf_op_registry import register_tf_op


@register_tf_op(tf_alias=['BiasAdd'])
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
    axis = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_prod(x=x, axes=axis, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Any(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_sum(x=x, axes=axis, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def ArgMax(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    x = cb.reduce_argmax(x=x, axis=axis.val, name=node.name)
    context.add(node.name, x)


@register_tf_op
def ArgMin(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    x = cb.reduce_argmin(x=x, axis=axis.val, name=node.name)
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
        x = cb.transpose(x=x, perm=[0, 3, 1, 2])
        x = cb.avg_pool(x=x, kernel_sizes=kernel_sizes, strides=strides,
                        pad_type=pad_type)
        x = cb.transpose(x=x, perm=[0, 2, 3, 1], name=node.name)
    else:
        x = cb.avg_pool(x=x, kernel_sizes=kernel_sizes, strides=strides,
                        pad_type=pad_type, name=node.name)
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
    x = cb.transpose(x=x, perm=[0, 3, 1, 2])
    x = cb.local_response_norm(x=x, size=size, alpha=alpha, beta=beta, k=bias)
    x = cb.transpose(x=x, perm=[0, 2, 3, 1], name=node.name)
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
        x = cb.transpose(x=x, perm=[0, 3, 1, 2])
    # Only the last op should have the same name as node.name
    conv_name = node.name + 'x' if data_format == 'NHWC' else node.name
    x = cb.conv(x=x, W=W_oihw, pad_type=pad_type, strides=HW_strides,
            dilations=HW_dilations, name=conv_name)
    if data_format == "NHWC":
        x = cb.transpose(x=x, perm=[0, 2, 3, 1], name=node.name)
    context.add(node.name, x)


@register_tf_op
def DepthToSpace(context, node):
    x = context[node.inputs[0]]
    block_size = node.attr.get('block_size')
    data_format = node.attr.get('data_format', 'NHWC')
    if data_format == 'NHWC':
        x = cb.transpose(x=x, perm=[0, 3, 1, 2])
        x = cb.depth_to_space(x=x, block_size=block_size)
        x = cb.transpose(x=x, perm=[0, 2, 3, 1], name=node.name)
    else:
        x = cb.depth_to_space(x=x, block_size=block_size, name=node.name)
    context.add(node.name, x)


@register_tf_op
def EuclideanNorm(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_l2_norm(x=x, axes=axis, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def ExpandDims(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
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
        x = cb.transpose(x=x, perm=[0, 3, 1, 2])
        x = cb.batch_norm(x=x, mean=mean, variance=variance, gamma=scale,
                          beta=offset, epsilon=epsilon)
        x = cb.transpose(x=x, perm=[0, 2, 3, 1], name=node.name)
    else:
        x = cb.batch_norm(x=x, mean=mean, variance=variance, gamma=scale,
                          beta=offset, epsilon=epsilon, name=node.name)
    # Inference only batch norm does not have meaningful outputs for
    # batch_mean, batch_variance etc.
    context.add(node.name, x)


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
def get_tuple(context, node):
    x = context[node.inputs[0]]
    if not isinstance(x, list):
        raise ValueError("Op type {} should return multiple output.".format(
            node.inputs[0].op))
    idx = node.attr['index']
    context.add(node.name, x[idx], is_new_var=False)

@register_tf_op
def Identity(context, node):
    # Don't change tfssa. Just make downstream ops reference the pre-identity op.
    x = context[node.inputs[0]]
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
    return d_spec


@register_tf_op
def MatMul(context, node):
    a = context[node.inputs[0]]
    b = context[node.inputs[1]]
    transpose_a = node.attr.get('transpose_a', False)
    transpose_b = node.attr.get('transpose_b', False)
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
        x = cb.transpose(x=x, perm=[0, 3, 1, 2])
        x = cb.max_pool(x=x, kernel_sizes=kernel_sizes, strides=strides,
                        pad_type=pad_type)
        x = cb.transpose(x=x, perm=[0, 2, 3, 1], name=node.name)
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
    axis = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_min(x=x, axes=axis, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


@register_tf_op
def Prod(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_prod(x=x, axes=axis, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)


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
def Sum(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_sum(x=x, axes=axis, keep_dims=keep_dims, name=node.name)
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
    axis = context[node.inputs[1]]
    keep_dims = node.attr.get('keep_dims', False)
    x = cb.reduce_mean(x=x, axes=axis, keep_dims=keep_dims, name=node.name)
    context.add(node.name, x)

@register_tf_op
def MirrorPad(context, node):
    x = context[node.inputs[0]]
    pad = context[node.inputs[1]]
    if pad is None:
        raise ValueError("TF `paddings` in Pad op must be const.")
    # mode must be 'reflect'. 'symmetry' mode not supported
    mode = node.attr.get('mode', 'reflect').lower()
    constant_val = node.attr.get('constant_val', 0.0)
    in_shape = x.sym_type.get_shape()
    in_rank = len(in_shape)
    # Reflect mode requires padding to be provided for all dimensions
    if in_rank <= 5 and mode == 'reflect':
        pad = pad.val.reshape(-1)
        # Reflect mode is supported only on last two dimension
        # If possible, pass padding of size 4
        if pad.shape[0] > 4:
            if np.all(pad[:-4] == 0):
                pad = pad[-4:]
            else:
                raise ValueError("Padding must be applied for last 2 dimensions in reflect mode! Applied for {}".format(pad.shape[0]))
        x = cb.pad(x=x, pad=pad, name=node.name, mode=mode, constant_val=constant_val)
    else:
        raise ValueError('Unsupported Pad configuration!')
    context.add(node.name, x)

@register_tf_op
def Pad(context, node):
    x = context[node.inputs[0]]
    pad = context[node.inputs[1]]
    if pad is None:
        raise ValueError("TF `paddings` in Pad op must be const.")
    # mode must be one of 'constant', 'reflect' or 'replicate'
    mode = node.attr.get('mode', 'constant').lower()
    constant_val = node.attr.get('constant_val', 0.0)
    in_shape = x.sym_type.get_shape()
    in_rank = len(in_shape)
    if in_rank <= 5 or mode == 'constant':
        pad = pad.val.reshape(-1)
        x = cb.pad(x=x, pad=pad, name=node.name, mode=mode, constant_val=constant_val)
    else:
        raise ValueError('Unsupported Pad configuration!')
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
def Squeeze(context, node):
    x = context[node.inputs[0]]
    axes = node.attr.get('squeeze_dims', [])
    axes = None if axes == [] else axes
    if not axes:
        x = cb.squeeze(x=x, name=node.name)
    else:
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
def SpaceToDepth(context, node):
    x = context[node.inputs[0]]
    block_size = node.attr.get('block_size')
    data_format = node.attr.get('data_format', 'NHWC')
    if data_format == 'NHWC':
        x = cb.transpose(x=x, perm=[0, 3, 1, 2])
        x = cb.space_to_depth(x=x, block_size=block_size)
        x = cb.transpose(x=x, perm=[0, 2, 3, 1], name=node.name)
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
    x, indices = cb.topk(x=x, k=k.val, axis=-1, name=node.name)
    context.add(x.name, x)
    context.add(indices.name, indices)
    context.add(node.name, [x, indices])


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
def Transpose(context, node):
    x = context[node.inputs[0]]
    perm = context[node.inputs[1]]
    x = cb.transpose(x=x, perm=perm, name=node.name)
    context.add(node.name, x)

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
        x = cb.transpose(x=x, perm=[0, 3, 1, 2])
        output_shape = [output_shape[1], output_shape[2]]
    else:
        output_shape = [output_shape[2], output_shape[3]]

    # Only the last op should have the same name as node.name
    conv_name = node.name + 'x' if data_format == 'NHWC' else node.name
    # add Conv Tranpose
    x = cb.conv_transpose(x=x, weight=weight, pad_type=pad_type, output_shape=output_shape, strides=HW_strides,
                          dilations=HW_dilations, name=conv_name)

    # Convert NCHW output back to NHWC format
    if data_format == "NHWC":
        x = cb.transpose(x=x, perm=[0, 2, 3, 1], name=node.name)
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
    x = cb.transpose(x=x, perm=[0, 3, 1, 2])
    # add the upsample layer
    x = cb.upsample_nearest_neighbor(x=x,
                                     upscale_factor_height=scaling_factor_h,
                                     upscale_factor_width=scaling_factor_w,
                                     name=node.name + '_channel_first_upsample')
    # transpose again
    x = cb.transpose(x=x, perm=[0, 2, 3, 1], name=node.name)

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
    x = cb.transpose(x=x, perm=[0, 3, 1, 2])

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
    x = cb.transpose(x=x, perm=[0, 2, 3, 1], name=node.name)
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
def SplitV(context, node):
    x = context[node.inputs[0]]
    split_sizes = context[node.inputs[1]]
    axis = context[node.inputs[2]]
    if 'num_split' not in node.attr:
        raise ValueError('num_splits not found in TF op {}'.format(node.name))
    num_splits = node.attr['num_split']
    x = cb.split(x=x, num_splits=num_splits, split_sizes=split_sizes,
            axis=axis, name=node.name)
    context.add(node.name, x)

@register_tf_op
def Split(context, node):
    axis = context[node.inputs[0]]
    x = context[node.inputs[1]]
    node.attr
    if 'num_split' not in node.attr:
        raise ValueError('num_splits not found in TF op {}'.format(node.name))
    num_splits = node.attr['num_split']
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
