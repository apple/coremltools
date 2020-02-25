import logging
import six
import numpy as np

from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
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


@register_tf_op
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
    if data_format == "NHWC":
        # TF's FusedBatchNorm is only for 4D inputs
        x = cb.transpose(x=x, perm=[0, 3, 1, 2])
    x = cb.batchnorm(x=x, mean=mean, variance=variance, gamma=scale,
            beta=offset, epsilon=epsilon, name=node.name)
    if data_format == "NHWC":
        x = cb.transpose(x=x, perm=[0, 2, 3, 1])
    # Inference only batch norm does not have meaningful outputs for
    # batch_mean, batch_variance etc.
    context.add(node.name, [x, None, None, None, None])


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
    context.add(node.name, x[idx])

@register_tf_op
def Identity(context, node):
    # Don't change tfssa. Just make downstream ops reference the pre-identity op.
    x = context[node.inputs[0]]
    context.add(node.name, x)

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
    if not isinstance(x, list):
        raise ValueError("Op type {} should return multiple output.".format(
            node.inputs[0].op))
    idx = node.attr['index']
    context.add(node.name, x[idx])

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