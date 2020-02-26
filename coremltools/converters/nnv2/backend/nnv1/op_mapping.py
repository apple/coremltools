import numpy as np
import logging

V2_TO_V1_OP_REGISTRY = {}


def register_v2_op(func):
    V2_TO_V1_OP_REGISTRY[func.__name__] = func
    return func


def _convert_pool(const_context, builder, op, mode):
    builder.add_pooling(
        name=op.name,
        height=op.kernel_sizes.val[-2],
        width=op.kernel_sizes.val[-1],
        stride_height=op.strides.val[-2],
        stride_width=op.strides.val[-1],
        layer_type=mode.upper(),
        padding_type=op.pad_type.val.upper(),
        input_name=op.x.name,
        output_name=op.name,
        exclude_pad_area=True,
        is_global=False,
    )


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
    rank = len(val.shape)
    if rank == 0:
        builder.add_load_constant_nd(
                name=name,
                output_name=name,
                constant_value=val.reshape([1]),
                shape=[1])
    elif rank == 3:
        builder.add_load_constant(
                name=name,
                output_name=name,
                constant_value=val,
                shape=val.shape)
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
def add(const_context, builder, op):
    if op.x.val is not None and op.x.rank > 0:
        add_const(const_context, builder, op.x.name, op.x.val)
    if op.y.val is not None and op.y.rank > 0:
        add_const(const_context, builder, op.y.name, op.y.val)

    if op.x.shape != op.y.shape:
        # Use the braodcast version
        builder.add_add_broadcastable(
                name=op.name,
                input_names=[op.x.name, op.y.name],
                output_name=op.name)
    elif op.x.rank == 0 and op.x.val is not None:
        builder.add_elementwise(
                name=op.name,
                input_names=[op.y.name],
                output_name=op.name,
                alpha=op.x.val,
                mode='ADD')
    elif op.y.rank == 0 and op.y.val is not None:
        builder.add_elementwise(
                name=op.name,
                input_names=[op.x.name],
                output_name=op.name,
                alpha=op.y.val,
                mode='ADD')
    else:
        # x, y are same shape
        builder.add_elementwise(
                name=op.name,
                input_names=[op.x.name, op.y.name],
                output_name=op.name,
                mode='ADD')


@register_v2_op
def avg_pool(const_context, builder, op):
    _convert_pool(
        const_context=const_context,
        builder=builder,
        op=op,
        mode='average')


@register_v2_op
def band_part(const_context, builder, op):
    builder.add_matrix_band_part(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        num_lower=op.lower.val,
        num_upper=op.upper.val,
    )


@register_v2_op
def const(const_context, builder, op):
    # const in V2 are added to V1 lazily.
    pass

@register_v2_op
def conv(const_context, builder, op):
    # v2 x: (n, C_in/group, *D_in)
    x_name = op.x.name
    is_conv1d = op.x.rank == 3
    if is_conv1d:
        x_name = op.name+'expand_dim'
        builder.add_expand_dims(
                name=x_name,
                input_name=op.x.name,
                output_name=x_name,
                axes=3)
    # `x_name` is guaranteed to be (n, C_in/group, H, W)

    # W_v1 wil be np.ndarray (if W is const at compile time) or None
    # (if W is not known at compile time).
    W_v1 = None
    input_names = [op.x.name]
    if op.W.val is not None:
        # v2 conv W: (C_out, C_in/group, spatial_dims)
        # v1 convolution expects (H, W, C_in/group, C_out)
        W_v1 = op.W.val
        if op.W.rank == 3:
            W_v1 = np.expand_dims(op.W.val, 3)
        W_v1 = np.transpose(W_v1, [2, 3, 1, 0])
    else:
        # op.W is not const at compile time.
        W_rank4 = op.W.name+'expand_dim'
        if op.W.rank == 3:
            builder.add_expand_dims(
                    name=W_rank4,
                    input_name=op.W.name,
                    output_name=W_rank4,
                    axes=3)
        W_transposed = op.W.name + 'transposed'
        builder.add_transpose(
                name=W_transposed,
                axes=[2, 3, 1, 0],
                output_name=W_transposed)
        input_names.append(W_transposed)

    # padding
    border_mode = op.pad_type.val
    pad = {}
    if border_mode == 'custom':
        border_mode = 'valid'
        pad['padding_top'] = op.pad.val[0]
        pad['padding_bottom'] = op.pad.val[1]
        if not is_conv1d:
            pad['padding_left'] = op.pad.val[2]
            pad['padding_right'] = op.pad.val[3]

    # This doesn't work till builder fills in all optional values
    # (rdar://59280101)
    #has_bias = np.sum(np.abs(op.B.val)) > 0
    has_bias = op.B is not None
    dilations = op.dilations.val.tolist()
    if is_conv1d:
        dilations = dilations + [1]

    builder.add_convolution(
            name=op.name,
            kernel_channels=op.W.shape[1],
            output_channels=op.W.shape[0],
            height=op.W.shape[2],
            width=1 if is_conv1d else op.W.shape[3],
            stride_height=op.strides.val[0],
            stride_width=1 if is_conv1d else op.strides.val[1],
            border_mode=border_mode,
            groups=op.group.val,
            W=W_v1,
            b=op.B.val if has_bias else None,
            has_bias=has_bias,
            is_deconv=False,
            input_name=input_names,
            output_name=op.name,
            dilation_factors=dilations,
            **pad)

def _add_elementwise_unary(builder, op, mode, **kwargs):
    if mode in ["sqrt", "rsqrt", "inverse", "power", "exp", "log", "abs", "threshold"]:
        builder.add_unary(
                name=op.name,
                input_name=op.x.name,
                output_name=op.name,
                mode=mode,
                **kwargs)
    else:
        add_func = getattr(builder, "add_"+mode, None)
        if add_func is None:
            logging.error('Elementwise unary method {} not found in builder.'.format(mode))
        add_func(name=op.name,
                 input_name=op.x.name,
                 output_name=op.name,
                 **kwargs)

def _add_elementwise_binary(const_context, builder, op, mode, **kwargs):
    if mode in ["add", "multiply"]:
        params = {"name": op.name, "output_name": op.name, "mode": mode.upper()}
        if op.x.val is not None and op.x.rank == 0:
            params["input_names"] = [op.y.name]
            params["alpha"] = op.x.val
            builder.add_elementwise(**params)
            return
        elif op.y.val is not None and op.y.rank == 0:
            params["input_names"] = [op.x.name]
            params["alpha"] = op.y.val
            builder.add_elementwise(**params)
            return
    elif mode in ["equal", "not_equal"]:
        add_func = getattr(builder, "add_"+mode, None)
        params = {"name": op.name, "output_name": op.name}
        if op.x.val is not None and op.x.rank == 0:
            params["input_names"] = [op.y.name]
            params["alpha"] = op.x.val
            add_func(**params)
            return
        elif op.y.val is not None and op.y.rank == 0:
            params["input_names"] = [op.x.name]
            params["alpha"] = op.y.val
            add_func(**params)
            return
    elif mode in ["greater_than", "greater_equal", "less_than", "less_equal"]:
        params = {"name": op.name, "output_name": op.name}
        if op.x.val is not None and op.x.rank == 0:
            params["input_names"] = [op.y.name]
            params["alpha"] = op.x.val
            if "less" in mode:
                params["use_greater_than_equal"] = mode.endswith("_equal")
                builder.add_greater_than(**params)
            elif "greater" in mode:
                params["use_less_than_equal"] = mode.endswith("_equal")
                builder.add_less_than(**params)
            return
        elif op.y.val is not None and op.y.rank == 0:
            params["input_names"] = [op.x.name]
            params["alpha"] = op.y.val
            if "greater" in mode:
                params["use_greater_than_equal"] = mode.endswith("_equal")
                builder.add_greater_than(**params)
            elif "less" in mode:
                params["use_less_than_equal"] = mode.endswith("_equal")
            return

    if op.x.val is not None:
        add_const(const_context, builder, op.x.name, op.x.val)
    if op.y.val is not None:
        if mode == "pow":
            _add_elementwise_unary(builder, "power", alpha=op.y.val)
            return
        add_const(const_context, builder, op.y.name, op.y.val)

    if mode in ["add", "multiply", "max", "min", "ave"]:
        if op.x.shape == op.y.shape:
            builder.add_elementwise(
                    name=op.name,
                    input_names=[op.x.name, op.y.name],
                    output_name=op.name,
                    mode=mode.upper())
        else:
            add_func = getattr(builder, "add_"+mode+"_broadcastable", None)

            if add_func is None:
                logging.error('Elementwise binary broadcastable method {} not found in builder.'.format(mode))

            add_func(name=op.name,
                     input_names=[op.x.name, op.y.name],
                     output_name=op.name,
                     **kwargs)
    else:
        if mode in ["divide", "floor_div", "mod", "pow", "subtract"]:
            add_func = getattr(builder, "add_"+mode+"_broadcastable", None)
        else:
            add_func = getattr(builder, "add_"+mode, None)

        if add_func is None:
            logging.error('Elementwise binary method {} not found in builder.'.format(mode))

        add_func(name=op.name,
                 input_names=[op.x.name, op.y.name],
                 output_name=op.name,
                 **kwargs)

def _add_logical(builder, op, mode):
    input_names = []
    input_names.append(op.x.name)
    if mode != "NOT":
        input_names.append(op.y.name)

    builder.add_logical(
            name=op.name,
            input_names=input_names,
            output_name=op.name,
            mode=mode)

@register_v2_op
def abs(const_context, builder, op):
    _add_elementwise_unary(builder, op, "abs")

@register_v2_op
def acos(const_context, builder, op):
    _add_elementwise_unary(builder, op, "acos")

@register_v2_op
def add(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "add")

@register_v2_op
def asin(const_context, builder, op):
    _add_elementwise_unary(builder, op, "asin")

@register_v2_op
def atan(const_context, builder, op):
    _add_elementwise_unary(builder, op, "atan")

@register_v2_op
def atanh(const_context, builder, op):
    _add_elementwise_unary(builder, op, "atanh")

@register_v2_op
def ceil(const_context, builder, op):
    _add_elementwise_unary(builder, op, "ceil")

@register_v2_op
def clip(const_context, builder, op):
    _add_elementwise_unary(builder, op, "clip", min_value=op.alpha.val, max_value=op.beta.val)

@register_v2_op
def cos(const_context, builder, op):
    _add_elementwise_unary(builder, op, "cos")

@register_v2_op
def cosh(const_context, builder, op):
    _add_elementwise_unary(builder, op, "cosh")

@register_v2_op
def equal(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "equal")

@register_v2_op
def exp(const_context, builder, op):
    _add_elementwise_unary(builder, op, "exp")

@register_v2_op
def exp2(const_context, builder, op):
    _add_elementwise_unary(builder, op, "exp2")

@register_v2_op
def floor(const_context, builder, op):
    _add_elementwise_unary(builder, op, "floor")

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
def less(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "less_than")

@register_v2_op
def less_equal(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "less_equal")

@register_v2_op
def log(const_context, builder, op):
    _add_elementwise_unary(builder, op, "log")

@register_v2_op
def logical_and(const_context, builder, op):
    _add_logical(builder, op, "AND")

@register_v2_op
def logical_not(const_context, builder, op):
    _add_logical(builder, op, "NOT")

@register_v2_op
def logical_or(const_context, builder, op):
    _add_logical(builder, op, "OR")

@register_v2_op
def logical_xor(const_context, builder, op):
    _add_logical(builder, op, "XOR")

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
    _add_elementwise_unary(builder, op, "round")

@register_v2_op
def rsqrt(const_context, builder, op):
    _add_elementwise_unary(builder, op, "rsqrt")

@register_v2_op
def sign(const_context, builder, op):
    _add_elementwise_unary(builder, op, "sign")

@register_v2_op
def sin(const_context, builder, op):
    _add_elementwise_unary(builder, op, "sin")

@register_v2_op
def sinh(const_context, builder, op):
    _add_elementwise_unary(builder, op, "sinh")

@register_v2_op
def sqrt(const_context, builder, op):
    _add_elementwise_unary(builder, op, "sqrt")

@register_v2_op
def sub(const_context, builder, op):
    _add_elementwise_binary(const_context, builder, op, "subtract")

@register_v2_op
def tan(const_context, builder, op):
    _add_elementwise_unary(builder, op, "tan")

@register_v2_op
def threshold(const_context, builder, op):
    _add_elementwise_unary(builder, op, "threshold", alpha=op.alpha.val)

@register_v2_op
def depth_to_space(const_context, builder, op):
    builder.add_reorganize_data(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        mode='DEPTH_TO_SPACE',
        block_size=op.block_size.val
    )

@register_v2_op
def expand_dims(const_context, builder, op):
    builder.add_expand_dims(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axes=[op.axis.val])


@register_v2_op
def fill(const_context, builder, op):
    if op.shape.val is None:
        builder.add_fill_dynamic(
            name=op.name,
            input_name=op.shape.name,
            output_name=op.name,
            value=op.value.val
        )
    else:
        builder.add_fill_static(
            name=op.name,
            output_name=op.name,
            output_shape=op.shape.val,
            value=op.value.val
        )


@register_v2_op
def random_bernoulli(const_context, builder, op):
    if op.shape.val is None:
        builder.add_random_bernoulli_dynamic(
            name=op.name,
            input_name=op.shape.name,
            output_name=op.name,
            prob=op.prob.val,
            seed=op.seed.val,
        )
    else:
        builder.add_random_bernoulli_static(
            name=op.name,
            output_name=op.name,
            output_shape=op.shape.val,
            prob=op.prob.val,
            seed=op.seed.val,
        )


@register_v2_op
def random_categorical(const_context, builder, op):
    builder.add_categorical_distribution(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        num_samples=op.size.val,
        is_logits=(op.mode.val == 'logits'),
        seed=op.seed.val,
    )


@register_v2_op
def random_normal(const_context, builder, op):
    if op.shape.val is None:
        builder.add_random_normal_dynamic(
            name=op.name,
            input_name=op.shape.name,
            output_name=op.name,
            mean=op.mean.val,
            stddev=op.stddev.val,
            seed=op.seed.val,
        )
    else:
        builder.add_random_normal_static(
            name=op.name,
            output_name=op.name,
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
            input_name=op.shape.name,
            output_name=op.name,
            minval=op.low.val,
            maxval=op.high.val,
            seed=op.seed.val,
        )
    else:
        builder.add_random_uniform_static(
            name=op.name,
            output_name=op.name,
            output_shape=op.shape.val,
            minval=op.low.val,
            maxval=op.high.val,
            seed=op.seed.val,
        )


@register_v2_op
def reduce_argmax(const_context, builder, op):
    builder.add_argmax(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axis=op.axis.val,
        keepdims=op.keep_dims.val,
    )


@register_v2_op
def gru(const_context, builder, op):
    # Input shape: [b, s, I]
    input_name = op.x.name
    # Shape: [b, H]
    initial_h = op.initial_h.name

    w = op.weight.val
    b = op.bias.val if op.bias is not None else None
    direction = op.direction.val
    output_sequence = op.output_sequence.val
    activations = op.activations.val

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
    builder.add_squeeze(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axes=op.axes.val,
        squeeze_all=(op.axes.val is None)
    )


@register_v2_op
def transpose(const_context, builder, op):
    builder.add_transpose(
            name=op.name,
            axes=op.perm.val,
            input_name=op.x.name,
            output_name=op.name)


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
            input_name=op.x.name,
            output_name=op.name)


@register_v2_op
def matmul(const_context, builder, op):
    builder.add_batched_mat_mul(
        name=op.name,
        input_names=[op.x.name, op.y.name],
        output_name=op.name,
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
def lstm(const_context, builder, op):
    # Input shape [b, s, I]
    input_name = op.x.name
    # Shape: [b, DIRECTION*H]
    initial_h = op.initial_h.name
    initial_c = op.initial_c.name

    w = op.weight.val
    b = op.bias.val if op.bias is not None else None
    direction = op.direction.val
    output_sequence = op.output_sequence.val
    activations = op.activations.val
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
                input_names=[op.x.name, op.shape.name],
                output_name=op.name)
    elif -1 in op.shape.val and len(op.shape.val) == op.x.rank:
        # Support 0 in shape.
        builder.add_rank_preserving_reshape(
                name=op.name,
                input_name=op.x.name,
                output_name=op.name,
                output_shape=op.shape.val)
    else:
        if 0 in op.shape.val:
            # Does not support 0 in shape
            msg = 'Use 0 in shape only if len(shape) == x.rank. Report bug.'
            raise ValueError(msg)
        builder.add_reshape_static(
                name=op.name,
                input_name=op.x.name,
                output_name=op.name,
                output_shape=op.shape.val)

@register_v2_op
def reduce_argmax(const_context, builder, op):
    builder.add_argmax(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axis=op.axis.val,
        keepdims=op.keep_dims.val,
    )

@register_v2_op
def reduce_argmin(const_context, builder, op):
    builder.add_argmin(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axis=op.axis.val,
        keepdims=op.keep_dims.val,
    )

@register_v2_op
def reduce_l1_norm(const_context, builder, op):
    builder.add_reduce_l1(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axes=op.axes.val,
        keepdims=op.keep_dims.val,
        reduce_all=op.axes.val is None
    )

@register_v2_op
def reduce_l2_norm(const_context, builder, op):
    builder.add_reduce_l2(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axes=op.axes.val,
        keepdims=op.keep_dims.val,
        reduce_all=op.axes.val is None
    )

@register_v2_op
def reduce_log_sum(const_context, builder, op):
    builder.add_reduce_logsum(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axes=op.axes.val,
        keepdims=op.keep_dims.val,
        reduce_all=op.axes.val is None
    )

@register_v2_op
def reduce_log_sum_exp(const_context, builder, op):
    builder.add_reduce_logsumexp(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axes=op.axes.val,
        keepdims=op.keep_dims.val,
        reduce_all=op.axes.val is None
    )

@register_v2_op
def reduce_max(const_context, builder, op):
    # rdar://59609180 (Optimization: mapping reduce to global_pool)
    builder.add_reduce_max(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axes=op.axes.val,
        keepdims=op.keep_dims.val,
        reduce_all=op.axes.val is None
    )

@register_v2_op
def reduce_mean(const_context, builder, op):
    # rdar://59609180 (Optimization: mapping reduce to global_pool)
    builder.add_reduce_mean(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axes=op.axes.val,
        keepdims=op.keep_dims.val,
        reduce_all=op.axes.val is None
    )

@register_v2_op
def reduce_min(const_context, builder, op):
    builder.add_reduce_min(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axes=op.axes.val,
        keepdims=op.keep_dims.val,
        reduce_all=op.axes.val is None
    )

@register_v2_op
def reduce_prod(const_context, builder, op):
    builder.add_reduce_prod(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axes=op.axes.val,
        keepdims=op.keep_dims.val,
        reduce_all=op.axes.val is None
    )

@register_v2_op
def reduce_sum(const_context, builder, op):
    builder.add_reduce_sum(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axes=op.axes.val,
        keepdims=op.keep_dims.val,
        reduce_all=op.axes.val is None
    )

@register_v2_op
def reduce_sum_square(const_context, builder, op):
    builder.add_reduce_sumsquare(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        axes=op.axes.val,
        keepdims=op.keep_dims.val,
        reduce_all=op.axes.val is None
    )

@register_v2_op
def rnn(const_context, builder, op):
    input_name = op.x.name # [b, s, I]
    initial_h = op.initial_h.name # [b, H]

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
def space_to_depth(const_context, builder, op):
    builder.add_reorganize_data(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        mode='SPACE_TO_DEPTH',
        block_size=op.block_size.val
    )

@register_v2_op
def transpose(const_context, builder, op):
    builder.add_transpose(
            name=op.name,
            axes=op.perm.val,
            input_name=op.x.name,
            output_name=op.name)

@register_v2_op
def gather(const_context, builder, op):
    builder.add_gather(
            name=op.name,
            input_names=[op.x.name, op.indices.name],
            output_name=op.name,
            axis=op.axis.val)

@register_v2_op
def scatter(const_context, builder, op):
    builder.add_scatter(
            name=op.name,
            input_names=[op.data.name, op.indices.name, op.updates.name],
            output_name=op.name,
            axis=op.axis.val,
            mode=op.mode.val.upper())

@register_v2_op
def gather_along_axis(const_context, builder, op):
    builder.add_gather_along_axis(
            name=op.name,
            input_names=[op.x.name, op.indices.name],
            output_name=op.name,
            axis=op.axis.val)

@register_v2_op
def scatter_along_axis(const_context, builder, op):
    builder.add_scatter_along_axis(
            name=op.name,
            input_names=[op.data.name, op.indices.name, op.updates.name],
            output_name=op.name,
            axis=op.axis.val,
            mode=op.mode.val.upper())

@register_v2_op
def gather_nd(const_context, builder, op):
    builder.add_gather_nd(
            name=op.name,
            input_names=[op.x.name, op.indices.name],
            output_name=op.name)

@register_v2_op
def scatter_nd(const_context, builder, op):
    builder.add_scatter_nd(
            name=op.name,
            input_names=[op.data.name, op.indices.name, op.updates.name],
            output_name=op.name,
            mode=op.mode.val.upper())

@register_v2_op
def tile(const_context, builder, op):
    builder.add_tile(
            name=op.name,
            reps=op.reps.val,
            input_name=op.x.name,
            output_name=op.name)

@register_v2_op
def tanh(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='TANH',
        input_name=op.x.name,
        output_name=op.name,
    )

@register_v2_op
def scaled_tanh(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='SCALED_TANH',
        input_name=op.x.name,
        output_name=op.name,
        params=[op.alpha.val, op.beta.val]
    )

@register_v2_op
def sigmoid(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='SIGMOID',
        input_name=op.x.name,
        output_name=op.name,
    )

@register_v2_op
def sigmoid_hard(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='SIGMOID_HARD',
        input_name=op.x.name,
        output_name=op.name,
        params=[op.alpha.val, op.beta.val]
    )

@register_v2_op
def erf(const_context, builder, op):
    builder.add_erf(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name
    )

@register_v2_op
def thresholded_relu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='THRESHOLDEDRELU',
        input_name=op.x.name,
        output_name=op.name,
        params=op.alpha.val
    )

@register_v2_op
def elu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='ELU',
        input_name=op.x.name,
        output_name=op.name,
        params=op.alpha.val
    )

@register_v2_op
def leaky_relu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='LEAKYRELU',
        input_name=op.x.name,
        output_name=op.name,
        params=[op.alpha.val]
    )

@register_v2_op
def gelu(const_context, builder, op):
    builder.add_gelu(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
    )

@register_v2_op
def softplus(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='SOFTPLUS',
        input_name=op.x.name,
        output_name=op.name,
    )

@register_v2_op
def softplus_parametric(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='PARAMETRICSOFTPLUS',
        input_name=op.x.name,
        output_name=op.name,
        params=[op.alpha.val, op.beta.val]
    )

@register_v2_op
def softsign(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='SOFTSIGN',
        input_name=op.x.name,
        output_name=op.name
    )

@register_v2_op
def linear_activation(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='LINEAR',
        input_name=op.x.name,
        output_name=op.name,
        params=[op.alpha.val, op.beta.val]
    )

@register_v2_op
def relu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='RELU',
        input_name=op.x.name,
        output_name=op.name
    )

@register_v2_op
def clamped_relu(const_context, builder, op):
    builder.add_clamped_relu(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name,
        alpha=op.alpha.val,
        beta=op.beta.val
    )

@register_v2_op
def relu6(const_context, builder, op):
    builder.add_clamped_relu(
        name=op.name,
        input_name=op.x.name,
        output_name=op.name
    )

@register_v2_op
def prelu(const_context, builder, op):
    builder.add_activation(
        name=op.name,
        non_linearity='PRELU',
        input_name=op.x.name,
        output_name=op.name,
        params=op.alpha.val
    )

@register_v2_op
def squeeze(const_context, builder, op):
    builder.add_squeeze(
            name=op.name,
            axes=op.axes.val if op.axes else None,
            squeeze_all=True if op.axes is None else False,
            input_name=op.x.name,
            output_name=op.name)

@register_v2_op
def pad(const_context, builder, op):
    pad   = op.pad.val
    mode  = op.mode.val
    constant_val = op.constant_val.val

    nnv1_mode_mapping = {"reflect" : "reflection", "replicate" : "replication"}
    mode = nnv1_mode_mapping.get(mode, mode)
    # If all pad values are zeros for for first rank(x) - 2 dimensions
    # Use generic padding instead.
    if op.x.rank > 1 and np.all(pad[:-4] == 0):
        # TODO: <rdar://problem/59771971> [NNv2] Padding layer handling NHWC input
        builder.add_padding(
            name=op.x.name,
            top=pad[-4],
            bottom=pad[-3],
            left=pad[-2],
            right=pad[-1],
            value=constant_val,
            input_name=op.x.name,
            output_name=op.name,
            padding_type=mode
        )
    elif mode == "constant":
        builder.add_constant_pad(
            name=op.x.name,
            input_names=[op.x.name],
            output_name=op.name,
            value=constant_val,
            pad_to_given_output_size_mode=False,
            pad_amounts=pad
        )
    else:
        message = "Expecting either rank 4 with padding last two dimensions (H and W) or general padding with `constant` mode " + \
                   "Provided rank {} input with padding {} and mode {}".format(op.x.rank, pad, mode)
        raise ValueError("Unsupported configuration for Pad layer! {}".format(message))
