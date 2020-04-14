import numpy as np
import logging
import six
from coremltools.models import neural_network as neural_network
from coremltools.proto import NeuralNetwork_pb2
from coremltools.converters.nnv2.builtin_types.symbolic import is_variadic
from coremltools.converters.nnv2.nnv2_program.var import Var
from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.nnv2_program.ops.registry import SSAOpRegistry

V2_TO_V1_OP_REGISTRY = {}


def register_v2_op(func):
    f_name = func.__name__
    if f_name in V2_TO_V1_OP_REGISTRY:
        raise ValueError("V2 op {} is already registered.".format(f_name))
    V2_TO_V1_OP_REGISTRY[f_name] = func
    return func

def convert_ops(const_context, builder, ops, outputs):
    """
    const_context: set of str: const name for v1 & v2 (the same)
    builder: neural_network.NeuralNetworkBuilder
    ops: list[Operation], usually from SsaBlock.operations.
    outputs: list[Var]. block outputs
    """

    custom_ops = SSAOpRegistry.custom_ops
    for op in ops:
        if op.op_type in custom_ops:
            mapper = V2_TO_V1_OP_REGISTRY['custom_op']
        elif op.op_type in V2_TO_V1_OP_REGISTRY:
            mapper = V2_TO_V1_OP_REGISTRY[op.op_type]
        else:
            msg = '{} is not implemented for nnv1 backend. block: {}'
            raise ValueError(msg.format(op.op_type, op.enclosing_block))
        # const is globally shared in nnv1.
        mapper(const_context, builder, op)

    for ov in outputs:
        # If block return value is a const, we need to add it.
        if ov.op.op_type == 'const':
            add_const(const_context, builder, ov.name, ov.val)

def make_input(const_context, builder, variables):
    """
    Ensure that variables, if const, are added to builder.

    variables: list[Var] or Var or str. Inputs for an nnv1 layer.

    Returns:
        list[str] or str: variables' names.
    """
    if isinstance(variables, (list, tuple)):
        return [make_input(const_context, builder, v) for v in variables]
    if isinstance(variables, six.string_types):
        return variables

    v = variables # variables is Var
    if v.op is not None and v.op.op_type == 'const' and not v.name in const_context:
        add_const(const_context, builder, v.name, v.val)
    return v.name


def _convert_pool(const_context, builder, op, mode, exclude_padding_from_average=True):
    num_spatial_dimensions = len(op.kernel_sizes.val)
    op_pad = op.pad.val if op.pad is not None else [0] * num_spatial_dimensions * 2
    if num_spatial_dimensions <= 2:
        padding_type = op.pad_type.val.upper()
        # nnv1's add_pool function does not support CUSTOM padding,
        # but VALID padding supports user-defined padding amounts.
        # Therefore we map CUSTOM padding to VALID padding.
        if padding_type == 'CUSTOM':
            padding_type = 'VALID'
        builder.add_pooling(
            name=op.name,
            height=op.kernel_sizes.val[-2],
            width=op.kernel_sizes.val[-1],
            stride_height=op.strides.val[-2],
            stride_width=op.strides.val[-1],
            layer_type=mode.upper(),
            padding_type=padding_type,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.name,
            exclude_pad_area=exclude_padding_from_average,
            padding_top=op_pad[0],
            padding_bottom=op_pad[1],
            padding_left=op_pad[2],
            padding_right=op_pad[3],
            is_global=False,
        )
    elif num_spatial_dimensions == 3:
        builder.add_pooling3d(
            name=op.name,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            pooling_type=mode.upper(),
            kernel_depth=op.kernel_sizes.val[-3],
            kernel_height=op.kernel_sizes.val[-2],
            kernel_width=op.kernel_sizes.val[-1],
            stride_depth=op.strides.val[-3],
            stride_height=op.strides.val[-2],
            stride_width=op.strides.val[-1],
            padding_mode=op.pad_type.val,
            custom_padding_front=op_pad[0],
            custom_padding_back=op_pad[1],
            custom_padding_top=op_pad[2],
            custom_padding_bottom=op_pad[3],
            custom_padding_left=op_pad[4],
            custom_padding_right=op_pad[5],
            average_pooling_count_excludes_padding=exclude_padding_from_average
        )
    else:
        raise ValueError('Unsupported number of spatial dimensions.  Maximum is 3, but got %s'
                         % num_spatial_dimensions)


def _try_convert_global_pool(const_context, builder, op, mode):
    """
    Optional performance optimization pass that tries to lower spatial
    reduce_mean / reduce_max to global_avg_pool / global_max_pool.
    Return True if the lowering happened, otherwise return False to
    continue as normal reduction op.
    """
    rank = op.x.rank
    if is_variadic(rank) or rank not in {4, 5}:
        return False
    keep_dims = op.keep_dims.val
    if op.axes is not None:
        axes = op.axes.val
        axes = sorted([rank + axis if axis < 0 else axis for axis in axes])
        if keep_dims is False:
            return False
        if rank == 4 and tuple(axes) != (2, 3):
            return False
        if rank == 5 and tuple(axes) != (2, 3, 4):
            return False
    builder.add_pooling(
        name=op.name,
        height=0,
        width=0,
        stride_height=0,
        stride_width=0,
        layer_type=mode.upper(),
        padding_type='valid'.upper(),
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        is_global=True,
    )
    return True


def add_const(const_context, builder, name, val):
    """
    const_context (set(str)): const names added to v1 builder. Const names are
    identical between v2 and v1

    name (str): name of const. Should be the same for v1 and v2.
    val: np.ndarray

    No return values as `name` is the name of const in v1.

    Comment: we don't need to add scalar const as they are just fields in
             layer proto message in NNv1.
             If we really need a const scalar, we upcast it to rank-1.

    """
    if name in const_context:
        logging.warning('Const {} was already added.'.format(name))
        return
    if not isinstance(val, (np.ndarray, np.generic)):
        val = np.array([val])
    rank = len(val.shape)
    if rank == 0:
        builder.add_load_constant_nd(
                name=name,
                output_name=name,
                constant_value=val.reshape([1]),
                shape=[1])
    else:
        builder.add_load_constant_nd(
                name=name,
                output_name=name,
                constant_value=val,
                shape=val.shape)
    const_context.add(name)


# Helper routines for recurrent layers
def _expand_dim(builder, node_name, input_name, axes):
    builder.add_expand_dims(
        name=node_name,
        input_name=input_name,
        output_name=node_name,
        axes=axes
    )

def _squeeze(builder, node_name, input_name, axes):
    builder.add_squeeze(
        name=node_name,
        input_name=input_name,
        output_name=node_name,
        axes=axes
    )

def _split(x, sections, axis):
    if x is None:
        return None
    if x.shape[axis] % sections != 0:
        raise ValueError("Cannot split axis {} into {} sections for input of shape {}".format(axis, sections, x.shape))
    return np.split(x, sections, axis=axis)

# Split weights into given number of sections
# This method should be used when weights are combined into
# one matrix for several nodes e.g. Input, forget, cell and output gate
# of LSTM
def _split_weights(w, sections):
    hidden_size = w.shape[-1] // sections
    input_size = w.shape[0] - hidden_size
    w = np.transpose(w, (1, 0))
    w_x = _split(w[:, :input_size], sections=sections, axis=0)
    w_h = _split(w[:, input_size:], sections=sections, axis=0)
    return w_x, w_h

# Split bias into given number of sections
# This method should be used when biases are combined into
# one matrix for several nodes e.g. Input, forget, cell and output gate
# of LSTM
def _split_bias(b, sections):
    if b is None:
        return None
    # Combine input-hidden and hidden-hidden bias
    b = b[0] + b[1]
    b = _split(b, sections=sections, axis=0)
    return b

@register_v2_op
def avg_pool(const_context, builder, op):
    _convert_pool(
        const_context=const_context,
        builder=builder,
        op=op,
        mode='average',
        exclude_padding_from_average=op.exclude_padding_from_average.val)


@register_v2_op
def band_part(const_context, builder, op):
    builder.add_matrix_band_part(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        num_lower=op.lower.val,
        num_upper=op.upper.val,
    )


@register_v2_op
def batch_norm(const_context, builder, op):
    channels = op.x.shape[1]
    gamma = np.array([1.] * channels) if op.gamma is None else op.gamma.val
    beta = np.array([0.] * channels) if op.beta is None else op.beta.val
    builder.add_batchnorm(
        name=op.name,
        channels=channels,
        gamma=gamma,
        beta=beta,
        mean=op.mean.val,
        variance=op.variance.val,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        compute_mean_var=False,
        instance_normalization=False,
        epsilon=op.epsilon.val,
    )


@register_v2_op
def const(const_context, builder, op):
    # const in V2 are added to V1 lazily.
    pass

@register_v2_op
def conv(const_context, builder, op):
    # v2 x: (n, C_in/group, *D_in)
    x_name = make_input(const_context, builder, op.x)
    is_conv1d = op.x.rank == 3
    is_conv2d = op.x.rank == 4
    is_conv3d = op.x.rank == 5
    if not (is_conv1d or is_conv2d or is_conv3d):
        raise ValueError(
            "Input tensor rank '{}' is not one of '{}'.".format(
                op.x.rank,
                (3, 4, 5),
            )
        )
    if is_conv1d:
        x_name = op.name + "expand_dim"
        builder.add_expand_dims(
            name=x_name,
            input_name=op.x.name,
            output_name=x_name,
            axes=3,
        )
    # `x_name` is guaranteed to be (n, C_in/group, H, W) for 1D and 2D convolution

    # W_v1 wil be np.ndarray (if W is const at compile time) or None
    # (if W is not known at compile time).
    weights = None
    input_names = make_input(const_context, builder, [op.x])
    if op.W.val is not None:
        # v2 convolution (conv3d) expects weights to have shape (C_out, C_in/group, spatial_dims)
        # v1 convolution expects (H, W, C_in/group, C_out) or (D, H, W, C_in/group, C_out)
        weights = op.W.val
        if is_conv1d:
            weights = np.expand_dims(op.W.val, 3)
        if is_conv1d or is_conv2d:
            weights = np.transpose(weights, [2, 3, 1, 0])
    else:
        # op.W is not const at compile time.
        # When weight is dynamic, v1 convolution expects weight to be
        # (C_out, C_in/group, H, W)
        # TODO 3D convolution doesn't support dynamic weights:
        if is_conv3d:
            raise ValueError("3D Convolution doesn't support dynamic weights.")
        weights_name = op.W.name
        if is_conv1d:
            weights_name += "_expand_dim"
            builder.add_expand_dims(
                name=weights_name,
                input_name=op.W.name,
                output_name=weights_name,
                axes=3,
            )
        input_names.append(weights_name)

    # padding
    padding_mode = op.pad_type.val
    pad = {}
    if padding_mode == "custom":
        padding_mode = "valid"
        pad["padding_top"] = op.pad.val[0]
        pad["padding_bottom"] = op.pad.val[1]
        if is_conv2d or is_conv3d:
            pad["padding_left"] = op.pad.val[2]
            pad["padding_right"] = op.pad.val[3]
        if is_conv3d:
            pad["padding_front"] = op.pad.val[4]
            pad["padding_back"] = op.pad.val[5]

    # This doesn't work till builder fills in all optional values
    # (rdar://59280101)
    #has_bias = np.sum(np.abs(op.B.val)) > 0
    has_bias = op.B is not None
    groups = op.group.val
    dilations = op.dilations.val.tolist()
    if is_conv1d:
        dilations = dilations + [1]

    if is_conv1d or is_conv2d:
        builder.add_convolution(
            name=op.name,
            kernel_channels=op.W.shape[1],
            output_channels=op.W.shape[0],
            height=op.W.shape[2],
            width=1 if is_conv1d else op.W.shape[3],
            stride_height=op.strides.val[0],
            stride_width=1 if is_conv1d else op.strides.val[1],
            border_mode=padding_mode,
            groups=groups,
            W=weights,
            b=op.B.val if has_bias else None,
            has_bias=has_bias,
            is_deconv=False,
            input_name=input_names,
            output_name=op.outputs[0].name,
            dilation_factors=dilations,
            **pad  # Python 2.7.16 will fail with a syntax error if a comma is included after `**pad`
        )
    if is_conv3d:
        builder.add_convolution3d(
            name=op.name,
            input_channels=op.W.shape[1] * groups,
            output_channels=op.W.shape[0],
            depth=op.W.shape[2],
            height=op.W.shape[3],
            width=op.W.shape[4],
            W=weights,
            b=op.B.val if has_bias else None,
            has_bias=has_bias,
            groups=groups,
            stride_depth=op.strides.val[0],
            stride_height=op.strides.val[1],
            stride_width=op.strides.val[2],
            dilation_depth=dilations[0],
            dilation_height=dilations[1],
            dilation_width=dilations[2],
            padding_mode=padding_mode,
            input_name=input_names,
            output_name=op.name,
            **pad  # Python 2.7.16 will fail with a syntax error if a comma is included after `**pad`
        )


@register_v2_op
def cumsum(const_context, builder, op):
    input_names = make_input(const_context, builder, [op.x])
    builder.add_cumsum(
        name=op.name,
        input_names=input_names,
        output_name=op.name,
        axis=op.axis.val,
        reverse=op.reverse.val,
        exclusive=op.exclusive.val)

def _add_elementwise_unary(const_context, builder, op, mode, output_name=None, **kwargs):
    output_name = output_name if output_name else op.outputs[0].name
    name = output_name if output_name else op.name
    if mode in ["sqrt", "rsqrt", "inverse", "power", "exp", "log", "abs", "threshold"]:
        builder.add_unary(
                name=name,
                input_name=make_input(const_context, builder, op.x),
                output_name=output_name,
                mode=mode,
                **kwargs)
    else:
        add_func = getattr(builder, "add_"+mode, None)
        if add_func is None:
            logging.error('Elementwise unary method {} not found in builder.'.format(mode))
        add_func(name=name,
                 input_name=make_input(const_context, builder, op.x),
                 output_name=output_name,
                 **kwargs)

def _add_elementwise_binary(const_context, builder, op, mode, output_name = None, **kwargs):
    output_name = output_name if output_name else op.outputs[0].name
    name = output_name if output_name else op.name
    if mode in ["add", "multiply"]:
        params = {"name": name, "output_name": output_name, "mode": mode.upper()}
        if op.x.val is not None and op.x.rank == 0:
            params["input_names"] = make_input(const_context, builder, [op.y])
            params["alpha"] = op.x.val
            builder.add_elementwise(**params)
            return
        elif op.y.val is not None and op.y.rank == 0:
            params["input_names"] = make_input(const_context, builder, [op.x])
            params["alpha"] = op.y.val
            builder.add_elementwise(**params)
            return
    elif mode in ["equal", "not_equal"]:
        add_func = getattr(builder, "add_"+mode, None)
        params = {"name": name, "output_name": output_name}
        if op.x.val is not None and op.x.rank == 0:
            params["input_names"] = make_input(const_context, builder, [op.y])
            params["alpha"] = op.x.val
            add_func(**params)
            return
        elif op.y.val is not None and op.y.rank == 0:
            params["input_names"] = make_input(const_context, builder, [op.x])
            params["alpha"] = op.y.val
            add_func(**params)
            return
    elif mode in ["greater_than", "greater_equal", "less_than", "less_equal"]:
        params = {"name": name, "output_name": output_name}
        if op.x.val is not None and op.x.rank == 0:
            params["input_names"] = make_input(const_context, builder, [op.y])
            params["alpha"] = op.x.val
            if "less" in mode:
                params["use_greater_than_equal"] = mode.endswith("_equal")
                builder.add_greater_than(**params)
            elif "greater" in mode:
                params["use_less_than_equal"] = mode.endswith("_equal")
                builder.add_less_than(**params)
            return
        elif op.y.val is not None and op.y.rank == 0:
            params["input_names"] = make_input(const_context, builder, [op.x])
            params["alpha"] = op.y.val
            if "greater" in mode:
                params["use_greater_than_equal"] = mode.endswith("_equal")
                builder.add_greater_than(**params)
            elif "less" in mode:
                params["use_less_than_equal"] = mode.endswith("_equal")
                builder.add_less_than(**params)
            return

    if op.x.val is not None:
        add_const(const_context, builder, op.x.name, op.x.val)
    if op.y.val is not None:
        if mode == "pow":
            _add_elementwise_unary(const_context, builder, op, "power", output_name=output_name,
                    alpha=op.y.val)
            return
        add_const(const_context, builder, op.y.name, op.y.val)

    if mode in ["add", "multiply", "max", "min", "ave"]:
        if op.x.shape == op.y.shape:
            builder.add_elementwise(
                    name=name,
                    input_names=make_input(const_context, builder,
                        [op.x, op.y]),
                    output_name=output_name,
                    mode=mode.upper())
        else:
            add_func = getattr(builder, "add_"+mode+"_broadcastable", None)

            if add_func is None:
                logging.error('Elementwise binary broadcastable method {} not found in builder.'.format(mode))

            add_func(name=name,
                     input_names=make_input(const_context, builder,
                         [op.x, op.y]),
                     output_name=output_name,
                     **kwargs)
    else:
        if mode in ["divide", "floor_div", "mod", "pow", "subtract"]:
            add_func = getattr(builder, "add_"+mode+"_broadcastable", None)
        elif mode == 'less_equal':
            add_func = builder.add_less_than
            kwargs['use_less_than_equal'] = True
        elif mode == 'greater_equal':
            add_func = builder.add_greater_than
            kwargs['use_greater_than_equal'] = True
        else:
            add_func = getattr(builder, "add_"+mode, None)

        if add_func is None:
            msg = 'Elementwise binary method {} not found in builder.'
            raise ValueError(msg.format(mode))

        add_func(name=name,
                 input_names=make_input(const_context, builder, [op.x, op.y]),
                 output_name=output_name,
                 **kwargs)

def _add_logical(const_context, builder, op, mode):
    input_names = []
    input_names.append(make_input(const_context, builder, op.x))
    if mode != "NOT":
        input_names.append(make_input(const_context, builder, op.y))

    builder.add_logical(
            name=op.name,
            input_names=input_names,
            output_name=op.outputs[0].name,
            mode=mode)

@register_v2_op
def abs(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "abs")

@register_v2_op
def acos(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "acos")

@register_v2_op
def add(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "add")

@register_v2_op
def asin(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "asin")

@register_v2_op
def atan(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "atan")

@register_v2_op
def atanh(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "atanh")

@register_v2_op
def cast(const_context, builder, op):
    if op.dtype.val in ["int32", "int64"]:
        _add_elementwise_unary(const_context, builder, op, "floor", output_name=op.name + "_floor")
        _add_elementwise_unary(const_context, builder, op, "ceil", output_name=op.name + "_ceil")

        builder.add_greater_than(
            name = op.name + "_cond",
            input_names = [make_input(const_context, builder, op.x)],
            output_name = op.name + "_cond",
            alpha = 0.0
        )

        builder.add_where_broadcastable(
            name = op.name,
            input_names=[op.name + i for i in ["_cond", "_floor", "_ceil"]],
            output_name=op.outputs[0].name
        )
    elif op.dtype.val in ["fp32", "fp64"]:
        builder.add_activation(
            name=op.name,
            non_linearity='LINEAR',
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            params=[1.0, 0.0]
        )
    else:
        raise NotImplementedError("Parameter dtype of the cast operation can be one of the {}. "
                                  "Provided {}".format(["int32", "int64","fp32", "fp64"], op.dtype.val))

@register_v2_op
def ceil(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "ceil")

@register_v2_op
def clip(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "clip",
            min_value=op.alpha.val, max_value=op.beta.val)

@register_v2_op
def cos(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "cos")

@register_v2_op
def cosh(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "cosh")

@register_v2_op
def equal(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "equal")

@register_v2_op
def exp(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "exp")

@register_v2_op
def exp2(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "exp2")

@register_v2_op
def floor(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "floor")

@register_v2_op
def floor_div(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "floor_div")

@register_v2_op
def greater(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "greater_than")

@register_v2_op
def greater_equal(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "greater_equal")

@register_v2_op
def inverse(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "inverse")

@register_v2_op
def less(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "less_than")

@register_v2_op
def less_equal(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "less_equal")

@register_v2_op
def log(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "log")

@register_v2_op
def logical_and(const_context, builder, op):
    _add_logical(const_context, builder, op, "AND")

@register_v2_op
def logical_not(const_context, builder, op):
    _add_logical(const_context, builder, op, "NOT")

@register_v2_op
def logical_or(const_context, builder, op):
    _add_logical(const_context, builder, op, "OR")

@register_v2_op
def logical_xor(const_context, builder, op):
    _add_logical(const_context, builder, op, "XOR")

@register_v2_op
def maximum(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "max")

@register_v2_op
def minimum(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "min")

@register_v2_op
def mod(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "mod")

@register_v2_op
def mul(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "multiply")

@register_v2_op
def not_equal(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "not_equal")

@register_v2_op
def pow(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "pow")

@register_v2_op
def real_div(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "divide")

@register_v2_op
def round(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "round")

@register_v2_op
def rsqrt(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "rsqrt")

@register_v2_op
def sign(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "sign")

@register_v2_op
def sin(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "sin")

@register_v2_op
def sinh(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "sinh")

@register_v2_op
def slice_by_index(const_context, builder, op):
    rank = op.x.rank
    stride = [1] * rank if op.stride is None else op.stride.val
    begin_mask = [False] * rank if op.begin_mask is None else op.begin_mask.val
    end_mask = [False] * rank if op.end_mask is None else op.end_mask.val
    squeeze_mask = [False] * rank if op.squeeze_mask is None else op.squeeze_mask.val

    builder.add_slice_dynamic(
        name=op.name,
        input_names=make_input(const_context, builder,
            [op.x, op.begin, op.end]),
        output_name=op.outputs[0].name,
        strides=tuple(stride),
        begin_masks=tuple(begin_mask),
        end_masks=tuple(end_mask),
        squeeze_masks=tuple(squeeze_mask)
    )

@register_v2_op
def slice_by_size(const_context, builder, op):
    size = op.size.val
    end_ids = op.name+"_end_ids"
    builder.add_elementwise(
            name=end_ids,
            input_names=make_input(const_context, builder,
                [op.begin, op.size]),
            output_name=end_ids,
            mode="ADD")
    input_names = make_input(const_context, builder,
                             [op.x, op.begin, end_ids])
    builder.add_slice_dynamic(
            name=op.name,
            input_names=input_names,
            output_name=op.outputs[0].name,
            end_masks=[True if s == -1 else False for s in size])

@register_v2_op
def sqrt(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "sqrt")

@register_v2_op
def square(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "power", alpha=2.)

@register_v2_op
def sub(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "subtract")

@register_v2_op
def tan(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "tan")

@register_v2_op
def threshold(const_context, builder, op):
    _add_elementwise_unary(const_context, builder, op, "threshold",
            alpha=op.alpha.val)

@register_v2_op
def depth_to_space(const_context, builder, op):
    builder.add_reorganize_data(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        mode='DEPTH_TO_SPACE',
        block_size=op.block_size.val
    )

@register_v2_op
def expand_dims(const_context, builder, op):
    builder.add_expand_dims(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axes=op.axes.val)


@register_v2_op
def fill(const_context, builder, op):
    if op.shape.val is None:
        builder.add_fill_dynamic(
            name=op.name,
            input_name=make_input(const_context, builder, op.shape),
            output_name=op.outputs[0].name,
            value=op.value.val
        )
    else:
        builder.add_fill_static(
            name=op.name,
            output_name=op.outputs[0].name,
            output_shape=op.shape.val,
            value=op.value.val
        )


@register_v2_op
def random_bernoulli(const_context, builder, op):
    if op.shape.val is None:
        builder.add_random_bernoulli_dynamic(
            name=op.name,
            input_name=make_input(const_context, builder, op.shape),
            output_name=op.outputs[0].name,
            prob=op.prob.val,
            seed=op.seed.val,
        )
    else:
        builder.add_random_bernoulli_static(
            name=op.name,
            output_name=op.outputs[0].name,
            output_shape=op.shape.val,
            prob=op.prob.val,
            seed=op.seed.val,
        )


@register_v2_op
def random_categorical(const_context, builder, op):
    builder.add_categorical_distribution(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        num_samples=op.size.val,
        is_logits=(op.mode.val == 'logits'),
        seed=op.seed.val,
    )


@register_v2_op
def random_normal(const_context, builder, op):
    if op.shape.val is None:
        builder.add_random_normal_dynamic(
            name=op.name,
            input_name=make_input(const_context, builder, op.shape),
            output_name=op.outputs[0].name,
            mean=op.mean.val,
            stddev=op.stddev.val,
            seed=op.seed.val,
        )
    else:
        builder.add_random_normal_static(
            name=op.name,
            output_name=op.outputs[0].name,
            output_shape=op.shape.val,
            mean=op.mean.val,
            stddev=op.stddev.val,
            seed=op.seed.val,
        )


@register_v2_op
def random_uniform(const_context, builder, op):
    if op.shape.val is None:
        builder.add_random_uniform_dynamic(
            name=op.name,
            input_name=make_input(const_context, builder, op.shape),
            output_name=op.outputs[0].name,
            minval=op.low.val,
            maxval=op.high.val,
            seed=op.seed.val,
        )
    else:
        builder.add_random_uniform_static(
            name=op.name,
            output_name=op.outputs[0].name,
            output_shape=op.shape.val,
            minval=op.low.val,
            maxval=op.high.val,
            seed=op.seed.val,
        )


@register_v2_op
def gru(const_context, builder, op):
    make_input(const_context, builder, [op.x, op.initial_h])
    # Input shape: [b, s, I]
    input_name = op.x.name
    # Shape: [b, H]
    initial_h = op.initial_h.name

    w = op.weight.val
    b = op.bias.val if op.bias is not None else None
    direction = op.direction.val
    output_sequence = op.output_sequence.val
    activations = [v.val for v in op.activations]

    # Add expand dims for input, in
    _expand_dim(builder, input_name+'_expanded', input_name, [3, 4])
    input_name += '_expanded'

    if direction not in {"forward", "reverse"}:
        raise ValueError('Unknown direction {} for GRU layer. Supported are forward, reverse'.format(direction))

    # Expand initial_h
    _expand_dim(builder, initial_h+'_expanded', initial_h, [2, 3, 4])
    initial_h += '_expanded'

    # Get weights here
    # weight format: [I+H, 3*H]
    # Split into Input and hidden weights
    # w_x: [I*H, I*H, I*H]
    # w_h: [H*H, H*H, H*H]
    # where, format is [Z, R, O]
    # Z: Update gate, R: Reset gate, O: Output gate
    w_x, w_h = _split_weights(w, sections=3)
    # bias format: [2, 3*H]
    # bias[0]: Input-Hidden bias
    # bias[1]: Hidden-Hidden bias
    # Combine bias into one and split into [Z, R, O] format
    b = _split_bias(b, sections=3)

    input_size = w_x[0].shape[1]
    hidden_size = w_x[0].shape[0]

    # 2 outputs
    # Y  : [s/1, b, h, 1, 1]
    # Y_h: [  1, b, h, 1, 1]
    output_names = [_output.name + '_5d' for _output in op.outputs]
    builder.add_gru(
        name=op.name,
        W_h=w_h,
        W_x=w_x,
        b=b,
        hidden_size=hidden_size,
        input_size=input_size,
        input_names=[input_name, initial_h],
        output_names=output_names,
        inner_activation=activations[0],
        activation=activations[1],
        output_all=output_sequence,
        reverse_input=(direction=="reverse")
    )

    # Squeeze Output
    # to output shape of [Seq Len or 1, Batch Size, Hidden Size]
    _squeeze(builder, op.outputs[0].name, output_names[0], axes=[3, 4])
    # Squeeze Output H and Output C
    # to output shape of [Batch Size, Hidden Size]
    _squeeze(builder, op.outputs[1].name, output_names[1], axes=[0, 3, 4])


@register_v2_op
def squeeze(const_context, builder, op):
    axes = op.axes.val if op.axes is not None else None
    builder.add_squeeze(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axes=axes,
        squeeze_all=axes is None)


@register_v2_op
def topk(const_context, builder, op):
    builder.add_topk(
        name=op.name,
        input_names=make_input(const_context, builder, [op.x]),
        output_names=[op.name + ':0', op.name + ':1'],
        k=op.k.val,
        axis=op.axis.val,
        use_bottom_k=op.ascending.val
    )


@register_v2_op
def l2_pool(const_context, builder, op):
    _convert_pool(
        const_context=const_context,
        builder=builder,
        op=op,
        mode='l2')


@register_v2_op
def linear(const_context, builder, op):
    out_channels, in_channels = op.weight.shape
    has_bias = op.bias.val is not None
    builder.add_inner_product(
            name=op.name,
            W=op.weight.val,
            b=op.bias.val,
            input_channels=in_channels,
            output_channels=out_channels,
            has_bias=has_bias,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name)


@register_v2_op
def matmul(const_context, builder, op):

    input_names = make_input(const_context, builder, [op.x, op.y])
    builder.add_batched_mat_mul(
        name=op.name,
        input_names=input_names,
        output_name=op.outputs[0].name,
        transpose_a=op.transpose_x.val,
        transpose_b=op.transpose_y.val,
    )


@register_v2_op
def max_pool(const_context, builder, op):
    _convert_pool(
        const_context=const_context,
        builder=builder,
        op=op,
        mode='max')


@register_v2_op
def non_zero(const_context, builder, op):
    builder.add_where_nonzero(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name
    )


@register_v2_op
def lstm(const_context, builder, op):
    make_input(const_context, builder, [op.x, op.initial_h, op.initial_c])
    # Input shape [b, s, I]
    input_name = op.x.name
    # Shape: [b, DIRECTION*H]
    initial_h = op.initial_h.name
    initial_c = op.initial_c.name

    w = op.weight.val
    b = op.bias.val if op.bias is not None else None
    direction = op.direction.val
    output_sequence = op.output_sequence.val
    activations = [v.val for v in op.activations]
    peephole = op.peephole.val if op.peephole is not None else None
    clip = op.clip.val

    # Add expand dims for input, in
    _expand_dim(builder, input_name+'_expanded', input_name, [3, 4])
    input_name += '_expanded'

    if direction in {"forward", "reverse"}:
        # Expand initial_h and initial_c
        _expand_dim(builder, initial_h+'_expanded', initial_h, [2, 3, 4])
        initial_h += '_expanded'
        _expand_dim(builder, initial_c+'_expanded', initial_c, [2, 3, 4])
        initial_c += '_expanded'

        # Get weights here
        # weight format: [I+H, 4*H]
        # Split into Input and hidden weights
        # w_x: [I*H, I*H, I*H, I*H]
        # w_h: [H*H, H*H, H*H, H*H]
        # where format is, [input gate, forget gate, cell gate, output gate]
        w_x, w_h = _split_weights(w, sections=4)
        # bias format: [2, 4*H]
        # bias[0]: Input-Hidden bias
        # bias[1]: Hidden-Hidden bias
        b = _split_bias(b, sections=4)
        # peephole format: [3*H]
        # where format is, [input gate, forget gate, output gate]
        peephole = _split(peephole, sections=3, axis=0)

        input_size = w_x[0].shape[1]
        hidden_size = w_x[0].shape[0]

        # 3 outputs
        # Y  : [s/1, b, h, 1, 1]
        # Y_h: [  1, b, h, 1, 1]
        # Y_c: [  1, b, h, 1, 1]
        output_names = [ _output.name + '_5d' for _output in op.outputs]
        builder.add_unilstm(
            name=op.name,
            W_h=w_h,
            W_x=w_x,
            b=b,
            hidden_size=hidden_size,
            input_size=input_size,
            input_names=[input_name, initial_h, initial_c],
            output_names=output_names,
            inner_activation=activations[0],
            cell_state_update_activation=activations[1],
            output_activation=activations[2],
            peep=peephole,
            output_all=output_sequence,
            cell_clip_threshold=clip,
            reverse_input=(direction=="reverse")
        )

        # Squeeze Output
        # to output shape of [Seq Len or 1, Batch Size, Hidden Size]
        _squeeze(builder, op.outputs[0].name, output_names[0], axes=[3, 4])
        # Squeeze Output H and Output C
        # to output shape of [Batch Size, Hidden Size]
        _squeeze(builder, op.outputs[1].name, output_names[1], axes=[0, 3, 4])
        _squeeze(builder, op.outputs[2].name, output_names[2], axes=[0, 3, 4])

    elif direction == "bidirectional":
        # Expand initial_h and initial_c
        _expand_dim(builder, initial_h+'_expanded', initial_h, [2, 3, 4])
        initial_h += '_expanded'
        _expand_dim(builder, initial_c+'_expanded', initial_c, [2, 3, 4])
        initial_c += '_expanded'

        initial_h_f = initial_h + '_forward'
        initial_h_r = initial_h + '_reverse'
        initial_c_f = initial_c + '_forward'
        initial_c_r = initial_c + '_reverse'

        # split input_h and input_c into two parts
        builder.add_split_nd(
            name=op.name+'_split_h',
            input_name=initial_h,
            output_names=[initial_h_f, initial_h_r],
            axis=1
        )
        builder.add_split_nd(
            name=op.name+'_split_c',
            input_name=initial_c,
            output_names=[initial_c_f, initial_c_r],
            axis=1
        )

        # Get weights here
        # weight format: [I+H, 2*4*H] -> [I+H, 4*H (forward):4*H (backward)]
        hidden_size = w.shape[-1] // 8
        input_size  = w.shape[0] - hidden_size
        forward_wts_index = 4*hidden_size
        # f_w_x and r_w_x: [I*H, I*H, I*H, I*H]
        # f_w_h and r_w_h: [H*H, H*H, H*H, H*H]
        # where format is, [input gate, forget gate, cell gate, output gate]
        f_w_x, f_w_h = _split_weights(w[:,:forward_wts_index], sections=4)
        r_w_x, r_w_h = _split_weights(w[:,forward_wts_index:], sections=4)

        # bias format: [2, 2*4*H]
        # bias[0]: Input-Hidden bias
        # bias[1]: Hidden-Hidden bias
        f_b, r_b = None, None
        if b is not None:
            f_b = _split_bias(b[:,:forward_wts_index], sections=4)
            r_b = _split_bias(b[:,forward_wts_index:], sections=4)

        # peephole format: [2*3*H] -> [3*H (forward) : 3*H (backward)]
        if peephole is None:
            f_peephole, r_peephole = None, None
        else:
            f_peephole = _split(peephole[:3*hidden_size], sections=3, axis=0)
            r_peephole = _split(peephole[3*hidden_size:], sections=3, axis=0)

        output_names = [op.outputs[0].name + '_5d',         # Output Y           [s/1, b, 2*h, 1, 1]
                        op.outputs[1].name + '_5d_foward',  # Output Y_h         [  1, b,   h, 1, 1]
                        op.outputs[2].name + '_5d_forward', # Output Y_c         [  1, b,   h, 1, 1]
                        op.outputs[1].name + '_5d_reverse', # Output Y_h_reverse [  1, b,   h, 1, 1]
                        op.outputs[2].name + '_5d_reverse'] # Output Y_c_reverse [  1, b,   h, 1, 1]

        builder.add_bidirlstm(
            name=op.name,
            W_h=f_w_h,
            W_x=f_w_x,
            b=f_b,
            W_h_back=r_w_h,
            W_x_back=r_w_x,
            b_back=r_b,
            hidden_size=hidden_size,
            input_size=input_size,
            input_names=[input_name, initial_h_f, initial_c_f, initial_h_r, initial_c_r],
            output_names=output_names,
            inner_activation=activations[0],
            cell_state_update_activation=activations[1],
            output_activation=activations[2],
            peep=f_peephole,
            peep_back=r_peephole,
            output_all=output_sequence,
            cell_clip_threshold=clip
        )

        # Squeeze Output
        # to output shape of [Seq Len or 1, Batch Size, 2*Hidden Size]
        _squeeze(builder, op.outputs[0].name, output_names[0], axes=[3, 4])

        # Output H is of format
        # 1, Batch_Size, Hidden_Size, 1, 1
        # Concat to make it
        # 1, Batch_Size, 2*Hidden_Size, 1, 1
        builder.add_concat_nd(
            name=op.outputs[1].name + '_5d',
            input_names=[output_names[1], output_names[3]],
            output_name=op.outputs[1].name + '_5d',
            axis=2
        )
        # Output C is of format
        # 1, Batch_Size, Hidden_Size, 1, 1
        builder.add_concat_nd(
            name=op.outputs[2].name + '_5d',
            input_names=[output_names[2], output_names[4]],
            output_name=op.outputs[2].name + '_5d',
            axis=2
        )

        # Squeeze Output H and Output C
        # to output shape of [Batch Size, 2*Hidden Size]
        _squeeze(builder, op.outputs[1].name, op.outputs[1].name + '_5d', axes=[0, 3, 4])
        _squeeze(builder, op.outputs[2].name, op.outputs[2].name + '_5d', axes=[0, 3, 4])
    else:
        raise ValueError('Unknown direction {} for LSTM layer. Supported are forward, reverse or bidirectional'.format(direction))

@register_v2_op
def reshape(const_context, builder, op):
    if op.shape.val is None:
        builder.add_reshape_dynamic(
                name=op.name,
                input_names=make_input(const_context, builder, [op.x,
                    op.shape]),
                output_name=op.outputs[0].name)
    elif -1 in op.shape.val and len(op.shape.val) == op.x.rank:
        # Support 0 in shape.
        builder.add_rank_preserving_reshape(
                name=op.name,
                input_name=make_input(const_context, builder, op.x),
                output_name=op.outputs[0].name,
                output_shape=op.shape.val)
    else:
        if 0 in op.shape.val:
            # Does not support 0 in shape
            msg = 'Use 0 in shape only if len(shape) == x.rank. Report bug.'
            raise ValueError(msg)
        builder.add_reshape_static(
                name=op.name,
                input_name=make_input(const_context, builder, op.x),
                output_name=op.outputs[0].name,
                output_shape=op.shape.val)

@register_v2_op
def reduce_argmax(const_context, builder, op):
    builder.add_argmax(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axis=op.axis.val,
        keepdims=op.keep_dims.val,
    )

@register_v2_op
def reduce_argmin(const_context, builder, op):
    builder.add_argmin(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axis=op.axis.val,
        keepdims=op.keep_dims.val,
    )

def _reduce_axes(const_context, builder, builder_op, op):
    axes = op.axes.val if op.axes is not None else op.axes
    builder_op(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axes=axes,
        keepdims=op.keep_dims.val,
        reduce_all=axes is None
    )

@register_v2_op
def reduce_l1_norm(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_l1, op)

@register_v2_op
def reduce_l2_norm(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_l2, op)

@register_v2_op
def reduce_log_sum(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_logsum, op)

@register_v2_op
def reduce_log_sum_exp(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_logsumexp, op)

@register_v2_op
def reduce_max(const_context, builder, op):
    if not _try_convert_global_pool(const_context, builder, op, mode='max'):
        _reduce_axes(const_context, builder, builder.add_reduce_max, op)


@register_v2_op
def reduce_mean(const_context, builder, op):
    if not _try_convert_global_pool(const_context, builder, op, mode='average'):
        _reduce_axes(const_context, builder, builder.add_reduce_mean, op)

@register_v2_op
def reduce_min(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_min, op)

@register_v2_op
def reduce_prod(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_prod, op)

@register_v2_op
def reduce_sum(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_sum, op)

@register_v2_op
def reduce_sum_square(const_context, builder, op):
    _reduce_axes(const_context, builder, builder.add_reduce_sumsquare, op)


@register_v2_op
def reverse(const_context, builder, op):
    reverse_dim = [False] * op.x.rank
    if op.axes is None:
        reverse_dim = [True] * op.x.rank
    else:
        for axis in op.axes.val:
            reverse_dim[axis] = True
    builder.add_reverse(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        reverse_dim=reverse_dim,
    )


@register_v2_op
def reverse_sequence(const_context, builder, op):
    add_const(const_context, builder, op.lengths.name, op.lengths.val)
    builder.add_reverse_sequence(
        name=op.name,
        input_names=[op.x.name, op.lengths.name],
        output_name=op.outputs[0].name,
        batch_axis=op.batch_axis.val,
        seq_axis=op.seq_axis.val
    )


@register_v2_op
def rnn(const_context, builder, op):
    input_name = make_input(const_context, builder, op.x) # [b, s, I]
    initial_h = make_input(const_context, builder, op.initial_h) # [b, H]

    w = op.weight.val
    b = op.bias.val if op.bias is not None else None
    direction = op.direction.val
    output_sequence = op.output_sequence.val
    activation = op.activation.val

    # Add expand dims for input, in
    _expand_dim(builder, input_name+'_expanded', input_name, [3, 4])
    input_name += '_expanded'

    if direction not in {"forward", "reverse"}:
        raise ValueError('Unknown direction {} for RNN layer. Supported are forward and reverse'.format(direction))

    # Expand initial_h and initial_c
    _expand_dim(builder, initial_h+'_expanded', initial_h, [2, 3, 4])
    initial_h += '_expanded'

    # Get weights here
    # weight format: [I+H, H]
    # Split into Input and hidden weights
    # w_x: (H, I)
    # w_h: (H, H)
    w = w.transpose()
    hidden_size = w.shape[0]
    input_size  = w.shape[-1] - hidden_size
    w_x, w_h = w[:,:input_size], w[:, input_size:]
    # bias format: [2, H]
    # bias[0]: Input-Hidden bias
    # bias[1]: Hidden-Hidden bias
    if b is not None:
        b = b[0] + b[1]

    # 3 outputs
    # Y  : [s/1, b, h, 1, 1]
    # Y_h: [  1, b, h, 1, 1]
    output_names = [ _output.name + '_5d' for _output in op.outputs]
    builder.add_simple_rnn(
        name=op.name,
        W_h=w_h,
        W_x=w_x,
        b=b,
        hidden_size=hidden_size,
        input_size=input_size,
        input_names=[input_name, initial_h],
        output_names=output_names,
        activation=activation,
        output_all=output_sequence,
        reverse_input=(direction=="reverse")
    )

    # Squeeze Output
    # to output shape of [Seq Len or 1, Batch Size, Hidden Size]
    _squeeze(builder, op.outputs[0].name, output_names[0], [3, 4])
    # Squeeze Output H and Output C
    # to output shape of [Batch Size, Hidden Size]
    _squeeze(builder, op.outputs[1].name, output_names[1], [0, 3, 4])

@register_v2_op
def select(const_context, builder, op):
    builder.add_where_broadcastable(
        name=op.name,
        input_names=make_input(const_context, builder, [op.cond, op.a, op.b]),
        output_name=op.outputs[0].name
    )

@register_v2_op
def space_to_depth(const_context, builder, op):
    builder.add_reorganize_data(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        mode='SPACE_TO_DEPTH',
        block_size=op.block_size.val
    )

@register_v2_op
def transpose(const_context, builder, op):
    builder.add_transpose(
            name=op.name,
            axes=op.perm.val,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name)

@register_v2_op
def gather(const_context, builder, op):
    builder.add_gather(
            name=op.name,
            input_names=make_input(const_context, builder, [op.x, op.indices]),
            output_name=op.outputs[0].name,
            axis=op.axis.val)

@register_v2_op
def scatter(const_context, builder, op):
    builder.add_scatter(
            name=op.name,
            input_names=make_input(const_context, builder, [op.data,
                op.indices, op.updates]),
            output_name=op.outputs[0].name,
            axis=op.axis.val,
            mode=op.mode.val.upper())

@register_v2_op
def gather_along_axis(const_context, builder, op):
    builder.add_gather_along_axis(
            name=op.name,
            input_names=make_input(const_context, builder, [op.x, op.indices]),
            output_name=op.outputs[0].name,
            axis=op.axis.val)

@register_v2_op
def scatter_along_axis(const_context, builder, op):
    builder.add_scatter_along_axis(
            name=op.name,
            input_names=make_input(const_context, builder, [op.data,
                op.indices, op.updates]),
            output_name=op.outputs[0].name,
            axis=op.axis.val,
            mode=op.mode.val.upper())

@register_v2_op
def gather_nd(const_context, builder, op):
    builder.add_gather_nd(
            name=op.name,
            input_names=[op.x.name, op.indices.name],
            output_name=op.outputs[0].name)

@register_v2_op
def scatter_nd(const_context, builder, op):
    builder.add_scatter_nd(
            name=op.name,
            input_names=[op.data.name, op.indices.name, op.updates.name],
            output_name=op.outputs[0].name,
            mode=op.mode.val.upper())

@register_v2_op
def tile(const_context, builder, op):
    builder.add_tile(
            name=op.name,
            reps=op.reps.val,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name)

@register_v2_op
def tanh(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='TANH',
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )

@register_v2_op
def scaled_tanh(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='SCALED_TANH',
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        params=[op.alpha.val, op.beta.val]
    )

@register_v2_op
def sigmoid(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='SIGMOID',
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )

@register_v2_op
def sigmoid_hard(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='SIGMOID_HARD',
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        params=[op.alpha.val, op.beta.val]
    )

@register_v2_op
def erf(const_context, builder, op):
    builder.add_erf(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name
    )

@register_v2_op
def thresholded_relu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='THRESHOLDEDRELU',
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        params=op.alpha.val
    )

@register_v2_op
def elu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='ELU',
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        params=op.alpha.val
    )

@register_v2_op
def leaky_relu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='LEAKYRELU',
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        params=[op.alpha.val]
    )

@register_v2_op
def gelu(const_context, builder, op):
    builder.add_gelu(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )

@register_v2_op
def softplus(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='SOFTPLUS',
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
    )

@register_v2_op
def softmax(const_context, builder, op):
    builder.add_softmax_nd(
        name=op.name,
        input_name=op.logit.name,
        output_name=op.outputs[0].name,
        axis=op.axis.val
    )

@register_v2_op
def softplus_parametric(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='PARAMETRICSOFTPLUS',
        input_name=make_input(const_context, builder, op.x),
        input_shape=op.x.shape,
        input_rank=op.x.rank,
        output_name=op.outputs[0].name,
        params=[op.alpha.val, op.beta.val]
    )

@register_v2_op
def softsign(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='SOFTSIGN',
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name
    )

@register_v2_op
def linear_activation(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='LINEAR',
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        params=[op.alpha.val, op.beta.val]
    )

@register_v2_op
def relu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='RELU',
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name
    )

@register_v2_op
def clamped_relu(const_context, builder, op):
    builder.add_clamped_relu(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        alpha=op.alpha.val,
        beta=op.beta.val
    )

@register_v2_op
def relu6(const_context, builder, op):
    builder.add_clip(name=op.name,
                     input_name=make_input(const_context, builder, op.x),
                     output_name=op.outputs[0].name,
                     min_value=0.0, max_value=6.0)

@register_v2_op
def prelu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='PRELU',
        input_name=make_input(const_context, builder, op.x),
        input_shape=op.x.shape,
        input_rank=op.x.rank,
        output_name=op.outputs[0].name,
        params=op.alpha.val
    )

@register_v2_op
def pad(const_context, builder, op):
    pad   = op.pad.val
    mode  = op.mode.val

    if len(pad.shape) != 1:
        raise ValueError('Pad should be a 1D tensor.')
    constant_val = op.constant_val.val

    nnv1_mode_mapping = {"reflect" : "reflection", "replicate" : "replication"}
    mode = nnv1_mode_mapping.get(mode,mode)

    if op.x.rank > 1 and np.all(pad[:-4] == 0):
        # check and map mode
        if mode == "symmetric":
            mode = "reflection"
        pad = pad[-4:]
        left, right = pad[2], pad[3]
        top, bottom = pad[0], pad[1]
        layer = builder.add_padding(
            name=op.name,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            padding_type=mode,
            value=constant_val,
        )
    elif mode == "constant":
        builder.add_constant_pad(
            name=op.name,
            input_names=[op.x.name],
            output_name=op.outputs[0].name,
            value=constant_val,
            pad_to_given_output_size_mode=False,
            pad_amounts=pad
        )
    else:
        raise ValueError("Unsupported mode for Pad layer! {}".format(mode))


@register_v2_op
def instance_norm(const_context, builder, op):
    channels = op.x.shape[1]
    gamma = np.array([1.] * channels) if op.gamma is None else op.gamma.val
    beta = np.array([0.] * channels) if op.beta is None else op.beta.val
    builder.add_batchnorm(
        name=op.name,
        channels=channels,
        gamma=gamma,
        beta=beta,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        compute_mean_var=True,
        instance_normalization=True,
        epsilon=op.epsilon.val,
    )


@register_v2_op
def l2_norm(const_context, builder, op):
    builder.add_l2_normalize(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        epsilon=op.epsilon.val,
    )


@register_v2_op
def layer_norm(const_context, builder, op):
    input_shape = list(op.x.shape)
    axes = None if op.axes is None else op.axes.val
    normalized_shape = input_shape[-len(axes):]
    gamma = np.ones(normalized_shape) if op.gamma is None else op.gamma.val
    beta = np.zeros(normalized_shape) if op.beta is None else op.beta.val
    if len(input_shape) in [2, 3] and len(axes) == 1 and axes[0] == len(input_shape) - 1:
        # Performance enhancement for some models with layer-norm
        builder.add_reshape_static(name=op.name + '_reshape',
                                   input_name=make_input(const_context,
                                       builder, op.x),
                                   output_name=op.x.name + '_reshape',
                                   output_shape=input_shape + [1, 1])

        builder.add_mvn(name=op.x.name + '_mvn',
                        input_name=op.x.name + '_reshape',
                        output_name=op.x.name + '_mvn', across_channels=True,
                        normalize_variance=True, epsilon=op.epsilon.val)

        builder.add_scale(name=op.x.name + '_5d',
                          input_name=op.x.name + '_mvn',
                          output_name=op.x.name + '_5d', W=gamma, b=beta, has_bias=True,
                          shape_scale=[len(gamma)], shape_bias=[len(beta)])

        builder.add_reshape_static(name=op.name,
                                   input_name=op.x.name + '_5d',
                                   output_name=op.outputs[0].name,
                                   output_shape=input_shape)
    else:
        builder.add_layer_normalization(
            name=op.name,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            normalized_shape=normalized_shape,
            gamma=gamma,
            beta=beta,
            eps=op.epsilon.val
        )


@register_v2_op
def local_response_norm(const_context, builder, op):
    builder.add_lrn(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        alpha=op.alpha.val,
        beta=op.beta.val,
        local_size=op.size.val,
        k=op.k.val
    )


@register_v2_op
def conv_transpose(const_context, builder, op):
    x_name = make_input(const_context, builder, op.x)
    out_name = op.outputs[0].name

    # Special handling for 1d conv transpose
    is_conv_transpose_1d = op.x.rank == 3
    if is_conv_transpose_1d:
        x_name = op.name + '_expand_dim'
        out_name = op.name + '_expanded'
        builder.add_expand_dims(
                name=x_name,
                input_name=op.x.name,
                output_name=x_name,
                axes=[3])

    # Input names to be used
    input_names = [x_name]

    # Kernel shape
    # 2D: [H, W, C_out, C_in]
    # 1D: [H, C_out, C_in]
    weight = op.weight.val

    if is_conv_transpose_1d:
        weight = np.expand_dims(op.weight.val, 1)

    # padding
    border_mode = 'valid' if op.pad_type is None else op.pad_type.val
    pad = [0] * 4
    if border_mode == 'custom' or op.pad is not None:
        border_mode = 'valid'
        pad[0] = op.pad.val[0] # Top
        pad[1] = op.pad.val[1] # Bottom
        if not is_conv_transpose_1d:
            pad[2] = op.pad.val[2] # Left
            pad[3] = op.pad.val[3] # Right

    strides = [op.strides.val[0], 1 if is_conv_transpose_1d else op.strides.val[1]]
    group = op.group.val
    has_bias = op.bias is not None
    dilations = [op.dilations.val[0], 1 if is_conv_transpose_1d else op.dilations.val[1]]
    # Get H and W from output shape
    output_shape = None if op.output_shape is None else tuple(op.output_shape.val)
    kernel_channels = weight.shape[3]
    output_channels = weight.shape[2]

    builder.add_convolution(
            name=out_name,
            kernel_channels=kernel_channels,
            output_channels=output_channels,
            height=weight.shape[0],
            width=weight.shape[1],
            stride_height=strides[0],
            stride_width=strides[1],
            border_mode=border_mode,
            groups=group,
            W=np.transpose(weight, (0, 1, 3, 2)),
            b=op.bias.val if has_bias else None,
            has_bias=has_bias,
            is_deconv=True,
            output_shape=output_shape,
            input_name=input_names,
            output_name=out_name,
            dilation_factors=dilations,
            padding_top=pad[0],
            padding_bottom=pad[1],
            padding_left=pad[2],
            padding_right=pad[3],
            )

    # Squeeze added `Width` dimension for 1d case
    if is_conv_transpose_1d:
        x_name = op.name+'expand_dim'
        builder.add_squeeze(
                name=op.name,
                input_name=out_name,
                output_name=op.outputs[0].name,
                axes=[3])


@register_v2_op
def range_1d(const_context, builder, op):
    if op.start.val is not None and op.step.val is not None:
        inputs = [op.end]
    elif op.start.val is None and op.step.val is not None:
        inputs = [op.end, op.start]
    elif op.start.val is not None and op.step.val is None:
        inputs = [op.end, op.start, op.step]
    else:
        inputs = [op.end, op.start, op.step]

    builder.add_range_dynamic(
            name=op.name,
            output_name=op.outputs[0].name,
            input_names=make_input(const_context, builder, inputs),
            start=op.start.val if op.start.val is not None else 0,
            step=op.step.val if op.step.val is not None else 1)

@register_v2_op
def one_hot(const_context, builder, op):
    if op.one_hot_vector_size.val is not None:
        inputs = [op.indices]
    else:
        inputs = [op.indices, op.one_hot_vector_size]

    builder.add_one_hot(
            name = op.name,
            input_names=make_input(const_context, builder, inputs),
            output_name = op.name,
            one_hot_vector_size=op.one_hot_vector_size.val,
            axis=op.axis.val,
            on_value=op.on_value.val,
            off_value=op.off_value.val)

@register_v2_op
def non_maximum_suppression(const_context, builder, op):
    builder.add_nms(
        name=op.name,
        input_names=make_input(const_context, builder, [op.boxes, op.scores]),
        output_names=['{}:{}'.format(op.name, i) for i in range(4)],
        iou_threshold=op.iou_threshold.val,
        score_threshold=op.score_threshold.val,
        max_boxes=op.max_boxes.val,
        per_class_suppression=op.per_class_suppression.val
    )

@register_v2_op
def flatten(const_context, builder, op):
    builder.add_flatten_to_2d(
            name=op.name,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            axis=op.axis.val)


@register_v2_op
def shape(const_context, builder, op):
    builder.add_get_shape(
            name=op.name,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name)


@register_v2_op
def upsample_nearest_neighbor(const_context, builder, op):
    builder.add_upsample(
        name=op.name,
        scaling_factor_h=op.upscale_factor_height.val,
        scaling_factor_w=op.upscale_factor_width.val,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        mode='NN',
    )

@register_v2_op
def upsample_bilinear(const_context, builder, op):
    if op.align_corners.val:
        builder.add_upsample(
            name=op.name,
            scaling_factor_h=op.scale_factor_height.val,
            scaling_factor_w=op.scale_factor_width.val,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            mode='BILINEAR',
            linear_upsample_mode='ALIGN_CORNERS_TRUE'
        )
    else:
        builder.add_upsample(
            name=op.name,
            scaling_factor_h=op.scale_factor_height.val,
            scaling_factor_w=op.scale_factor_width.val,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name,
            mode='BILINEAR',
            linear_upsample_mode='ALIGN_CORNERS_FALSE'
        )

@register_v2_op
def resize_bilinear(const_context, builder, op):
    grid_sampling_mode_map = {}
    grid_sampling_mode_map["STRICT_ALIGN_CORNERS"] = 'STRICT_ALIGN_ENDPOINTS_MODE'
    grid_sampling_mode_map["ALIGN_CORNERS"] = 'ALIGN_ENDPOINTS_MODE'
    grid_sampling_mode_map["DEFAULT"] = 'UPSAMPLE_MODE'
    grid_sampling_mode_map["OFFSET_CORNERS"] = 'ROI_ALIGN_MODE'

    builder.add_resize_bilinear(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        target_height=op.target_size_height.val,
        target_width=op.target_size_width.val,
        mode=grid_sampling_mode_map[op.sampling_mode.val]
    )


@register_v2_op
def cond(const_context, builder, op):
    true_block = op.blocks[0]
    false_block = op.blocks[1]

    branch_layer = builder.add_branch(
            name=op.name,
            input_name=make_input(const_context, builder, op.pred),
            )
    true_builder = neural_network.NeuralNetworkBuilder(
            nn_spec=branch_layer.branch.ifBranch,
            disable_rank5_shape_mapping=True)
    convert_ops(const_context, true_builder, true_block.operations,
            true_block.outputs)

    # Copy block output to cond op output.
    for block_out, op_out in zip(true_block.outputs, op.outputs):
        true_builder.add_copy(
                    name=block_out.name+'_ret_copy',
                    input_name=make_input(const_context, builder, block_out),
                    output_name=op_out.name)

    false_builder = neural_network.NeuralNetworkBuilder(
            nn_spec=branch_layer.branch.elseBranch,
            disable_rank5_shape_mapping=True)
    convert_ops(const_context, false_builder, false_block.operations,
            false_block.outputs)

    for block_out, op_out in zip(false_block.outputs, op.outputs):
        false_builder.add_copy(
                    name=block_out.name+'_ret_copy',
                    input_name=make_input(const_context, builder, block_out),
                    output_name=op_out.name)


@register_v2_op
def while_loop(const_context, builder, op):
    cond_block = op.blocks[0]
    body_block = op.blocks[1]

    # Assume that all loop vars aren't loop invariant (invariant loop vars
    # should've be optimized away in graph passes).
    for v_in, vx_in in zip(op.loop_vars, body_block.inputs):
        assert v_in.name != vx_in.name, 'Loop invariant detected in {}'.format(op)
        builder.add_copy(
                name=vx_in.name+'_input_copy',
                input_name=make_input(const_context, builder, v_in),
                output_name=vx_in.name)

    loop_layer = builder.add_loop(
            name=op.name,
            # max_iterations=0 to use condition network.
            max_iterations=0)

    # Construct while_loop condition
    cond_builder = neural_network.NeuralNetworkBuilder(
            nn_spec=loop_layer.loop.conditionNetwork,
            disable_rank5_shape_mapping=True)
    convert_ops(const_context, cond_builder, cond_block.operations,
            cond_block.outputs)

    loop_layer.loop.conditionVar = cond_block.outputs[0].name

    # while_loop body
    body_builder = neural_network.NeuralNetworkBuilder(
            nn_spec=loop_layer.loop.bodyNetwork,
            disable_rank5_shape_mapping=True)
    convert_ops(const_context, body_builder, body_block.operations,
            body_block.outputs)

    # Also assume all outputs are different from loop inputs (i.e., no loop
    # invariant.)
    for vx_in, vx_out in zip(body_block.inputs, body_block.outputs):
        if vx_in.name == vx_out.name:
            msg = 'Loop invariant var {} detected in block {}'
            logging.warning(msg.format(vx_in.name, body_block))
            continue
        body_builder.add_copy(
                    name=vx_in.name+'_ret_copy',
                    input_name=make_input(const_context, builder, vx_out),
                    output_name=vx_in.name)

@register_v2_op
def identity(const_context, builder, op):
    builder.add_copy(
            name=op.name,
            input_name=make_input(const_context, builder, op.x),
            output_name=op.outputs[0].name)

@register_v2_op
def concat(const_context, builder, op):
    builder.add_concat_nd(
            name=op.name,
            input_names=make_input(const_context, builder, op.values),
            output_name=op.outputs[0].name,
            axis=op.axis.val)

@register_v2_op
def stack(const_context, builder, op):
    builder.add_stack(
            name=op.name,
            input_names=make_input(const_context, builder, op.values),
            output_name=op.outputs[0].name,
            axis=op.axis.val)

@register_v2_op
def split(const_context, builder, op):
    split_sizes = None
    if op.split_sizes is not None:
        if op.split_sizes.val is None:
            raise ValueError('Non-const split_sizes unsupported in NNv1')
        split_sizes = op.split_sizes.val.tolist()
    builder.add_split_nd(
            name=op.name,
            input_name=make_input(const_context, builder, op.x),
            output_names=[v.name for v in op.outputs],
            axis=op.axis.val,
            num_splits=len(op.outputs),
            split_sizes=split_sizes)

@register_v2_op
def argsort(const_context, builder, op):
    axis = op.x.rank + op.axis.val if op.axis.val < 0 else op.axis.val
    builder.add_argsort(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axis=axis,
        descending=(not op.ascending.val)
    )


@register_v2_op
def pixel_shuffle(const_context, builder, op):
    builder.add_reorganize_data(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        mode='PIXEL_SHUFFLE',
        block_size=op.upscale_factor.val
    )


@register_v2_op
def sliding_windows(const_context, builder, op):
    builder.add_sliding_windows(
        name=op.name,
        input_name=make_input(const_context, builder, op.x),
        output_name=op.outputs[0].name,
        axis=op.axis.val,
        window_size=op.size.val,
        step=op.stride.val
    )

@register_v2_op
def crop(const_context, builder, op):
    builder.add_crop(
        name=op.name,
        input_names=[op.x.name],
        output_name=op.name,
        offset=0,
        left=op.crop_width.val[0], right=op.crop_width.val[1],
        top=op.crop_height.val[0], bottom=op.crop_height.val[1],
    )

@register_v2_op
def crop_resize(const_context, builder, op):
    grid_sampling_mode_map = {'STRICT_ALIGN_CORNERS' : 'STRICT_ALIGN_ENDPOINTS_MODE',
                              'ALIGN_CORNERS' : 'ALIGN_ENDPOINTS_MODE',
                              'DEFAULT' : 'UPSAMPLE_MODE',
                              'OFFSET_CORNERS': 'ROI_ALIGN_MODE'
    }

    mode = grid_sampling_mode_map[op.sampling_mode.val]

    input_expanded = op.name + '_x_expand'
    builder.add_expand_dims(
        name=input_expanded,
        input_name=make_input(const_context, builder, op.x),
        output_name=input_expanded,
        axes=[0]
    )
    builder.add_crop_resize(
        name=op.name,
        input_names=make_input(const_context, builder, [input_expanded,
            op.roi]),
        output_name=op.outputs[0].name,
        target_height=op.target_height.val,
        target_width=op.target_width.val,
        mode=mode,
        normalized_roi=op.normalized_coordinates.val,
        box_indices_mode=op.box_coordinate_mode.val,
        spatial_scale=op.spatial_scale.val
    )

@register_v2_op
def custom_op(const_context, builder, op):
    class_name = op.bindings.get('class_name', op.name)
    input_order = op.bindings.get('input_order', [])
    parameters = op.bindings.get('parameters', [])
    weights = op.bindings.get('weights', [])
    description = op.bindings.get('description', "")

    if len(input_order) == 0:
        raise ValueError("Inputs not provided for Custom Layer: {}".format(op.name))

    # Get input names
    input_names = [op.inputs[_name].name for _name in input_order]

    # Get output names
    output_names = [_output.name for _output in op.outputs]

    # Load custom params
    params = NeuralNetwork_pb2.CustomLayerParams()
    params.className = class_name
    params.description = description

    # Load parameters
    for _param in parameters:
        param = op.inputs[_param]
        param_val = param.val
        if builtins.is_bool(param.dtype):
            params.parameters[_param].boolValue = param_val
        elif builtins.is_int(param.dtype):
            params.parameters[_param].intValue = param_val
        elif builtins.is_float(param.dtype):
            params.parameters[_param].doubleValue = param_val
        elif builtins.is_str(param.dtype):
            params.parameters[_param].stringValue = param_val
        else:
            raise ValueError("Unknown parameter type for custom layer- "
                             "Op: {}, Parameter: {}, Type: {}".format(op.name, _param, param.dtype))

    # Load weights
    for _weight in weights:
        wt = params.weights.add()
        wt.floatValue.extend(map(float, _weight))

    # Add a custom layer
    builder.add_custom(
        name=op.name,
        input_names=input_names,
        output_names=output_names,
        custom_proto_spec=params
    )
