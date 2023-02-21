#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import builtins
import numbers

import numpy as _np
import paddle

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.defs._utils import (
    promote_input_dtypes,
    solve_slice_by_index_shape)
from coremltools.converters.mil.mil.types.symbolic import (
    any_symbolic,
    is_symbolic,
)
from coremltools.converters.mil.mil.var import Var

from .._utils import build_einsum_mil
from .paddle_op_registry import _PADDLE_OPS_REGISTRY, register_paddle_op


# This is a magic number in PaddlePaddle. It's used as a default value in many
# functions.
PYPADDLE_MAGIC_DEFAULT = 9223372036854775807
VALUE_CLOSE_TO_INFINITY = 1e+38



def convert_nodes(context, paddle_program):
    """
    Iterate over the nodes of a graph or block and convert to MIL.

    Arguments:
        context: A TranscriptionContext object to pull node inputs and
            assign node outputs.
        graph: An InternalPaddleIRGraph or InternalPaddleIRBlock object.
    """

    for block in paddle_program.blocks:
        for i, op in enumerate(block.ops):
            if op.type in ['feed', 'fetch']:
                continue
            else:
                op_lookup = op.type
                add_op = _PADDLE_OPS_REGISTRY.get(op_lookup, None)

                if add_op is None:
                    raise RuntimeError(
                        "PaddlePaddle convert function for op '{}' not implemented.".format(node.kind)
                    )
                add_op(context, op)


def convert_block(context, block, inputs):
    """Convert a block (sub-graph) to MIL. Conversion happens within a new
        context frame.

        Arguments:
            context: A TranscriptionContext object to pull node inputs and
                assign node outputs.
            block: An InternalPaddleIRBlock object.
            inputs: List of Vars from the outer context that map to the block's
                expected inputs. The number of inputs provided must match the
                number expected by the block.
    """

    assert len(block.inputs) == len(inputs)

    # Start a new context frame.
    context.push((block.inputs, inputs))

    # Add the block ops.
    convert_nodes(context, block)

    # Collect the block outputs.
    outputs = [context[outp] for outp in block.outputs]

    # Return to the previous context frame.
    context.pop()
    return outputs


# Some ops will receive a dtype input as an integer
# which maps to a paddle dtype. The below mapping was found by
# converting test models with different dtypes passed to ones.
NUM_TO_PADDLE_DTYPE = {
    0: paddle.uint8,
    1: paddle.int8,
    2: paddle.int16,
    3: paddle.int32,
    4: paddle.int32,
    5: paddle.float16,
    6: paddle.float32,
    7: paddle.float32,
    11: paddle.bool,
}

NUMPY_DTYPE_TO_PADDLE_NUM = {
    _np.uint8: 0,
    _np.int8: 1,
    _np.int16: 2,
    _np.int32: 3,
    _np.int64: 4,
    _np.float16: 5,
    _np.float32: 6,
    _np.float64: 7,
    _np.bool: 11,
}

NUM_TO_NUMPY_DTYPE = {
    0: _np.uint8,
    1: _np.int8,
    2: _np.int16,
    3: _np.int32,
    4: _np.int32,
    5: _np.float16,
    6: _np.float32,
    7: _np.float32,
    11: _np.bool,
}

NUM_TO_DTYPE_STRING = {
    0: "bool",
    1: "int16",
    2: "int32",
    3: "int32",
    4: "fp32",
    5: "fp32"
}

TYPE_TO_DTYPE_STRING = {
    types.bool: "bool",
    types.fp16: "fp16",
    types.fp32: "fp32",
    types.int32: "int32",
}


def _construct_constant(val, name):
    # Converter cannot handle paddle tensors.
    if isinstance(val, paddle.Tensor):
        val = val.cpu().numpy()

    # MIL casts ints to int32, which can't represent the 64 bit magic number.
    # So we instead represent it with None, and any ops that might get the
    # value will check for None instead.
    if isinstance(val, int) and val == PYPADDLE_MAGIC_DEFAULT:
        val = None

    # PaddlePaddle uses inf
    if val is not None and isinstance(val, numbers.Number) and _np.isinf(val):
        if val < 0:  # neg inf
            # most negative number in fp32
            val = -3.4e+38
        else:  # positive inf
            val = 3.4e+38
    if val is None:
        return None
    else:
        return mb.const(val=val, name=name)


@register_paddle_op()
def silu(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]
    x = mb.silu(x=x, name=output_name)
    context.add(x)


@register_paddle_op
def constant(context, node):
    assert len(node.inputs) == 0
    assert len(node.outputs) == 1

    name = node.name
    val = node.attr["value"]

    const = _construct_constant(val, name)
    context.add(const, paddle_name=name)
    

@register_paddle_op
def shape(context, node):
    x = context[node.input("Input")[0]]
    output_name = node.output("Out")[0]
    shape = mb.shape(x=x, name=output_name)
    context.add(shape)


@register_paddle_op(paddle_alias=['less_than'])
def lt(context, node):
    x = context[node.input("X")[0]]
    y = context[node.input("Y")[0]]
    inputs = [x, y]
    output_name = node.output("Out")[0]
    x, y = promote_input_dtypes(inputs)
    less = mb.less(x=x, y=y, name=output_name)
    context.add(less)


@register_paddle_op(paddle_alias=["transpose2"])
def permute(context, node):
    x = context[node.input("X")[0]]
    axis = node.desc.attr("axis")
    output_name = node.output("Out")[0]
    perm = mb.transpose(x=x, perm=axis, name=output_name)
    context.add(perm)



@register_paddle_op(paddle_alias=["matmul_v2"])
def matmul(context, node):
    x = context[node.input("X")[0]]
    y = context[node.input("Y")[0]]
    output_name = node.output("Out")[0]

    trans_x = node.desc.attr("trans_x")
    trans_y= node.desc.attr("trans_y")

    if trans_x:
        if len(x.shape) == 2:
            x = mb.transpose(x=x, perm=[1,0])
        elif len(x.shape) == 3:
            x = mb.transpose(x=x, perm=[0,2,1])
        elif len(x.shape) == 4:
            x = mb.transpose(x=x, perm=[0,1,3,2])
    if trans_y:
        if len(y.shape) == 2:
            y = mb.transpose(x=y, perm=[1,0])
        elif len(y.shape) == 3:
            y = mb.transpose(x=y, perm=[0,2,1])
        elif len(y.shape) == 4:
            y = mb.transpose(x=y, perm=[0,1,3,2])

    res = mb.matmul(x=x, y=y, name=output_name)
    context.add(res)


@register_paddle_op(paddle_alias=["elementwise_add"])
def add(context, node):

    # TODO (sberardi): 3rd param to aten::add is a scale factor, need to handle that.
    # out=input+alpha x other
    # rdar://60175736
    # if len(add_inputs) > 2 and add_inputs[2].val != 1:
    #     raise ValueError("ADD does not support scale factor param")
    x = context[node.input("X")[0]]
    y = context[node.input("Y")[0]]
    x, y = promote_input_dtypes([x, y])
    output_name = node.output("Out")[0]
    add_node = mb.add(x=x, y=y, name=output_name)
    context.add(add_node)


@register_paddle_op(paddle_alias=["conv2d"])
def _convolution(context, node):
    dilations = node.desc.attr('dilations')
    groups = node.desc.attr('groups')
    strides = node.desc.attr('strides')
    paddings = node.desc.attr('paddings')
    paddings += paddings

    x = context[node.input("Input")[0]]
    weight = context[node.input("Filter")[0]]
    if node.input("Bias"):
        bias = context[node.input("Bias")[0]]
    else:
        bias = None
    
    output_name = node.output('Output')[0]

    kwargs = {
        "x": x,
        "weight": weight,
        "strides": strides,
        "pad_type": "custom",
        "pad": paddings,
        "dilations": dilations,
        "groups": groups,
        "name": output_name,
    }
    # # Bias is optional in PaddlePaddle's convolution.
    if bias is not None:
        kwargs["bias"] = bias

    conv = mb.conv(**kwargs)
    context.add(conv)


@register_paddle_op
def softmax(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]

    axis = node.desc.attr("axis")
    res = mb.softmax(x=x, axis=axis, name=output_name)
    context.add(res)


@register_paddle_op(paddle_alias=["flatten_contiguous_range"])
def flatten(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]

    dims = list(x.shape)

    start_val = node.desc.attr("start_axis")
    end_val = node.desc.attr("stop_axis")

    start = len(dims) + start_val if start_val < 0 else start_val
    end = len(dims) + end_val if end_val < 0 else end_val

    if start > len(dims) or end > len(dims) or start < 0 or end < 0:
        raise ValueError(
            "Invalid start and end. (start, end) == ({}, {})".format(start, end_val)
        )
    if start > end:
        raise ValueError(
            "Start must be before end. (start, end) == ({}, {})".format(start, end_val)
        )
    x_shape = mb.shape(x=x)

    shape1 = mb.slice_by_index(x=x_shape, begin=[0], end=[start])
    shape2 = mb.slice_by_index(x=x_shape, begin=[end + 1], end=[len(dims)])

    flatten_dim = -1
    if not any_symbolic(x.shape):
        flatten_dim = 1
        for dim in dims[start: end + 1]:
            flatten_dim *= dim

    shape = mb.concat(values=(shape1, [flatten_dim], shape2), axis=0)
    shape = mb.cast(x=shape, dtype="int32")
    reshape = mb.reshape(x=x, shape=shape, name=output_name)
    context.add(reshape)


@register_paddle_op
def einsum(context, node):
    operands = [context[k] for k in node.input("Operands")]
    equation = node.desc.attr("equation")
    output_name = node.output("Out")[0]
    x = build_einsum_mil(operands[0], operands[1], equation, output_name)
    context.add(x, output_name)


@register_paddle_op(paddle_alias=["elementwise_floordiv"])
def floor_divide(context, node):
    x = context[node.input("X")[0]]
    y = context[node.input("Y")[0]]
    output_name = node.output("Out")[0]
    res = mb.floor_div(x=x, y=y, name=output_name)
    context.add(res)


@register_paddle_op(paddle_alias=["elementwise_mul"])
def mul(context, node):
    x = context[node.input("X")[0]]
    y = context[node.input("Y")[0]]
    x, y = promote_input_dtypes([x, y])
    output_name = node.output("Out")[0]
    res = mb.mul(x=x, y=y, name=output_name)
    context.add(res)


@register_paddle_op
def scale(context, node):
    x= context[node.input("X")[0]]
    output_name = node.output("Out")[0]

    scale_factor = node.desc.attr("scale")
    bias = node.desc.attr("bias")

    x, scale_factor, bias = promote_input_dtypes([x, scale_factor, bias])

    res = mb.mul(x=x, y=scale_factor)
    res = mb.add(x = res, y = bias, name = output_name)
    context.add(res)


@register_paddle_op(paddle_alias=["elementwise_sub"])
def sub(context, node):
    x = context[node.input("X")[0]]
    y = context[node.input("Y")[0]]
    x, y = promote_input_dtypes([x, y])
    output_name = node.output("Out")[0]
    res = mb.sub(x=x, y=y, name=output_name)
    context.add(res)


@register_paddle_op
def reduce_mean(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]
    axes = node.desc.attr("dim")
    keep_dim = node.desc.attr("keep_dim")
    res = mb.reduce_mean(x=x, axes=axes, keep_dims=keep_dim, name=output_name)
    context.add(res)


@register_paddle_op(paddle_alias=["squeeze2"])
def squeeze(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]
    axes = node.desc.attr("axes")
    if axes:
        res = mb.squeeze(x=x, axes=axes, name=output_name)
    else:
        res = mb.squeeze(x=x, name=output_name)
    context.add(res)

@register_paddle_op
def unsqueeze2(context, node):
    x = context[node.input("X")[0]]

    output_name = node.output("Out")[0]

    axes = node.desc.attr("axes")

    unsqueeze = mb.expand_dims(x=x, axes=axes, name=output_name)
    context.add(unsqueeze)


@register_paddle_op(paddle_alias=["reshape2"])
def view(context, node):
    x = context[node.input("X")[0]]
    shape = [context[i] for i in node.input("ShapeTensor")]
    output_name = node.output("Out")[0]

    if (len(shape)==0):
        shape = node.desc.attr("shape")
        for i in range(len(shape)):
            if shape[i] == 0:
                shape[i] = x.shape[i]

    if (
        isinstance(shape, list)
        and all([isinstance(dim, Var) and len(dim.shape) == 1 for dim in shape])
    ):
        shape = [mb.cast(x=k, dtype="int32") for k in shape]
        shape = mb.concat(values=shape, axis=0)

    shape = mb.cast(x=shape, dtype="int32")
    view = mb.reshape(x=x, shape=shape, name=output_name)
    context.add(view)


@register_paddle_op
def group_norm(context, node):
    x = context[node.input("X")[0]]
    bias = context[node.input("Bias")[0]]
    weight = context[node.input("Scale")[0]]

    output_name = node.output("Y")[0]

    eps = node.desc.attr("epsilon")
    group = node.desc.attr("groups")

    n,c = x.shape[0],x.shape[1] # at minimum (N, C) required
    input_shape = [*x.shape] # n, c, *
    num_groups = builtins.min(group,c)
    new_shape = [n, num_groups, c//num_groups]
    new_shape += [*x.shape[2:]] # adds remaining dims
    num_extra_axes = len(x.shape[2:])
    axes_ = [int(i) for i in range(2, 2 + num_extra_axes + 1)]
    weight_shape, bias_shape = [1,c], [1,c]
    weight_shape += [1 for _ in range(num_extra_axes)]
    bias_shape += [1 for _ in range(num_extra_axes)]
    
    x = mb.reshape(x=x, shape=new_shape)
    mean = mb.reduce_mean(x=x, axes=axes_, keep_dims=True)
    var = _std(x,axes_,True,False,eps)
    x = mb.sub(x=x,y=mean)
    x = mb.real_div(x=x,y=var)
    x = mb.reshape(x=x, shape=input_shape)
    if weight is not None:
        weight = mb.reshape(x=weight, shape=weight_shape)
        x = mb.mul(x=x,y=weight)
    if bias is not None:
        bias = mb.reshape(x=bias, shape=bias_shape)
        x = mb.add(x=x,y=bias)
    context.add(x, output_name)


@register_paddle_op(paddle_alias=["lookup_table_v2"])
def embedding(context, node):
    _input = context[node.input("W")[0]]
    indices = context[node.input("Ids")[0]]
    output_name = node.output("Out")[0]

    indices = mb.cast(x=indices, dtype="int32")

    #  Changing the axis from 0 is not an option in paddle, so we don't expose it
    gather = mb.gather(x=_input, indices=indices, name=output_name)
    context.add(gather)


@register_paddle_op
def concat(context, node):
    x = [context[k] for k in node.input("X")]
    output_name = node.output("Out")[0]
    axis = node.desc.attr("axis")
    res = mb.concat(values=x, axis=axis, name=output_name)
    context.add(res)


@register_paddle_op
def stack(context, node):
    values = [context[k] for k in node.input("X")]
    output_name = node.output("Y")[0]
    axis = node.desc.attr("axis")
    if (len(values)==1):
        res = mb.expand_dims(x=values[0], axes=[axis], name=output_name)
        context.add(res)
        return
    res = mb.stack(values=values, axis=axis, name=output_name)
    context.add(res)


@register_paddle_op
def layer_norm(context, node):
    _input = context[node.input("X")[0]]
    weight = context[node.input("Scale")[0]]
    bias = context[node.input("Bias")[0]]
    output_name = node.output("Y")[0]

    eps = node.desc.attr("epsilon")

    begin_norm_axis = node.desc.attr("begin_norm_axis")

    # cudnn_enable = inputs[5] unused

    layer_norm = mb.layer_norm(
        x=_input,
        axes=list(range(-(len(_input.shape) - begin_norm_axis), 0)),
        gamma=weight,
        beta=bias,
        epsilon=eps,
        name=output_name,
    )
    context.add(layer_norm)


def _get_scales_from_output_size(output_size, input_shape):
    scales = []
    if output_size is not None:
        # output_size will be either
        # (1) A list of Var, and each Var indicates the output size for that dimension
        # (2) A single Var which indicates the whole output size
        # (3) A numpy array

        if isinstance(output_size, list):
            output_size = [x.val[0] for x in output_size]
        if isinstance(output_size, Var):
            output_size = [x for x in output_size.val]
        if isinstance(output_size, _np.ndarray):
            output_size = output_size.tolist()

        # output size is computed using the formula floor (scale * input_size) in Core ML (and PaddlePaddle).
        # Thus, when computing the scales from the output size, we add a small positive constant to the output size
        # to make sure that the floor formula results in the correct output size and not 1 unit smaller.
        # For instance, if output size = 5 and input size = 2, then scale will be 2.5, which can get
        # represented as 2.49999 due to float precision issues, and this might resultin an output size of 4
        # instead of 5, without the epsilon correction.

        if len(output_size) == 1:
            # 1d upsampling
            Hout = output_size[0]
            Hin = input_shape[-1]
            scales_h = Hout / Hin if Hout % Hin == 0 else (Hout + 1e-4) / Hin
            scales = scales_h
        elif len(output_size) == 2:
            # 2d upsampling
            Hout, Wout = output_size[0], output_size[1]
            Hin, Win = input_shape[-2], input_shape[-1]
            scales_h = Hout / Hin if Hout % Hin == 0 else (Hout + 1e-4) / Hin
            scales_w = Wout / Win if Wout % Win == 0 else (Wout + 1e-4) / Win
            scales = [scales_h, scales_w]
        else:
            msg = "Only 1d and 2d unsampling are supported."
            raise NotImplementedError(msg)

    return scales


@register_paddle_op(paddle_alias=["nearest_interp_v2"])
def upsample_nearest2d(context, node):
    x = context[node.input("X")[0]]
    scale_tensor = context[node.input("Scale")[0]] if node.input("Scale") else None
    output_size = [context[k] for k in node.input("SizeTensor")] if node.input("SizeTensor") else None

    output_name = node.output("Out")[0]

    if (
        scale_tensor is not None
        and scale_tensor.val is not None
        # and scale_tensor.rank == 1
        # and scale_tensor.shape[0] == 2
    ):
        # get scale factors from provided inputs
        scale_tensor = scale_tensor.val
        scales_h = scale_tensor[0]
        scales_w = scale_tensor[1]
    elif node.desc.attr("scale"):
        scales_h, scales_w = node.desc.attr("scale")
    elif output_size is not None:
        scales = _get_scales_from_output_size(output_size, x.shape)
        if scales:
            scales_h, scales_w = scales
        else:
            raise ValueError("Failed to infer scale factors from inputs.")
    else:
        raise ValueError("Failed to infer scale factors from inputs.")

    upsample_nearest2d = mb.upsample_nearest_neighbor(
        x=x,
        scale_factor_height=scales_h,
        scale_factor_width=scales_w,
        name=output_name,
    )

    context.add(upsample_nearest2d)
    return


@register_paddle_op(paddle_alias=["cast"])
def type_as(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]

    in_dtype = node.desc.attr("in_dtype")
    out_dtype = node.desc.attr("out_dtype")

    NUM_TO_DTYPE_STRING

    if in_dtype == out_dtype:
        x = mb.identity(x=x, name=output_name)
    else:
        x = mb.cast(x=x, dtype=NUM_TO_DTYPE_STRING[out_dtype], name=output_name)

    context.add(x)


def _get_slice_params(context, data, inputs):
    rank = data.rank
    begin = [0] * rank
    end = [0] * rank
    stride = [1] * rank
    begin_mask = [False] * rank
    end_mask = [False] * rank
    squeeze_mask = [False] * rank

    num_of_slice_set = len(inputs) // 2

    for i in range(num_of_slice_set):
        if inputs[2 * i + 1] is None:
            # This is pure index select
            idx = context[inputs[2 * i]].val
            begin[i] = idx
            squeeze_mask[i] = True
        else:
            # This is a slice
            begin_var = context[inputs[2 * i]]
            end_var = context[inputs[2 * i + 1]]

            if begin_var is None:
                begin_mask[i] = True
            else:
                begin[i] = begin_var.val

            if end_var is None:
                end_mask[i] = True
            else:
                end[i] = end_var.val

    for i in range(num_of_slice_set, rank):
        begin_mask[i] = True
        end_mask[i] = True

    return begin, end, stride, begin_mask, end_mask, squeeze_mask


@register_paddle_op
def _internal_op_tensor_inplace_copy(context, node):
    data = context[node.inputs[0]]
    updates = context[node.inputs[1]]
    begin, end, stride, begin_mask, end_mask, squeeze_mask = _get_slice_params(
        context, data, node.inputs[2:]
    )

    data, updates = promote_input_dtypes([data, updates])
    updated_x = mb.paddle_tensor_assign(
        data=data,
        updates=updates,
        begin=begin,
        end=end,
        stride=stride,
        begin_mask=begin_mask,
        end_mask=end_mask,
        squeeze_mask=squeeze_mask,
        name=node.name,
    )
    context.add(updated_x)

@register_paddle_op
def assign_value(context, node):
    fp32_value = node.desc.attr("fp32_values")
    output_name = node.output("Out")[0]
    output_var = mb.const(val=fp32_value, name=output_name)
    context.add(output_var)


@register_paddle_op
def _internal_op_tensor_inplace_fill(context, node):
    data = context[node.inputs[0]]
    fill_scalar = context[node.inputs[1]]

    begin, end, stride, begin_mask, end_mask, squeeze_mask = _get_slice_params(
        context, data, node.inputs[2:]
    )
    fill_shape = solve_slice_by_index_shape(
        data.shape, begin, end, stride, begin_mask, end_mask, squeeze_mask
    )
    update_values = _np.full(fill_shape, fill_scalar.val)

    data, update_values = promote_input_dtypes([data, update_values])
    updated_x = mb.paddle_tensor_assign(
        data=data,
        updates=update_values,
        begin=begin,
        end=end,
        stride=stride,
        begin_mask=begin_mask,
        end_mask=end_mask,
        squeeze_mask=squeeze_mask,
        name=node.name,
    )
    context.add(updated_x)


def _make_fill_op(size, val, name, dtype):
    assert val is not None
    if isinstance(size, list):
        size = mb.concat(values=size, axis=0)
    fill = mb.fill(shape=size, value=val, name=name)
    return fill


@register_paddle_op
def fill_constant(context, node):
    output_name = node.output("Out")[0]

    if node.input_names and  node.input("ShapeTensorList"):
        shape_tensor_list = [context[k] for k in node.input("ShapeTensorList")]
        shape_tensor_list = [mb.cast(x=shape, dtype="int32") for shape in shape_tensor_list]
        size = mb.concat(values=shape_tensor_list, axis=0)
    elif node.input_names and node.input("ShapeTensor"):
        size = node.input("ShapeTensor")[0]
    else:
        size = node.desc.attr("shape")
    val = node.desc.attr("value")
    dtype = node.desc.attr("dtype")

    if dtype==2 or dtype==3:
        val = _np.array(val, dtype=_np.int32)

    if size:
        result = _make_fill_op(size, val, output_name, dtype)
    else:
        result = mb.const(val=val, name=output_name)
    
    context.add(result)


@register_paddle_op
def sigmoid(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]

    res = mb.sigmoid(x=x, name=output_name)
    context.add(res)


@register_paddle_op
def gelu(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]

    res = mb.gelu(x=x, name=output_name)
    context.add(res)


@register_paddle_op(paddle_alias=["slice"])
def _slice(context, node):
    x = context[node.input("Input")[0]]
    if node.input("StartsTensorList"):
        starts_tensor_list = context[node.input("StartsTensorList")[0]]
    else:
        starts_tensor_list = None
    if node.input("EndsTensorList"):
        ends_tensor_list = context[node.input("EndsTensorList")[0]]
    else:
        ends_tensor_list = None

    start = _np.array(node.desc.attr("starts"), dtype=_np.int32)
    end = _np.array(node.desc.attr("ends"), dtype=_np.int32)
    dim = _np.array(node.desc.attr("axes"), dtype=_np.int32)

    step = 1

    output_name = node.output("Out")[0]
    

    if start.all() == 0 and end is None and step == 1:
        # Handling x[:], just pass through the tensor.
        context.add(x, node.name)
        return

    begin_array = _np.array([0] * len(x.shape))
    if starts_tensor_list is not None:
       begin_array[dim] = starts_tensor_list.sym_val
    else: 
        begin_array[dim] = start
    end_array = _np.array([s if isinstance(s, int) else 0 for s in x.shape])
    end_mask = _np.array([True] * len(x.shape))
    if ends_tensor_list is not None:
        end_array[dim] = ends_tensor_list.sym_val
        end_mask[dim] = False
    elif end is not None:
        end_array[dim] = end
        end_mask[dim] = False


    kwargs = {
        "x": x,
        "begin": begin_array,
        "end": end_array,
        "end_mask": end_mask,
        "name": output_name,
    }

    res = mb.slice_by_index(**kwargs)
    context.add(res)


@register_paddle_op
def split(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")

    
    num = node.desc.attr("num")

    if node.input("AxisTensor"):
        axis_tensor = context[node.input("AxisTensor")[0]]
        axis = axis_tensor.val
        if num > 0:
            total = x.shape[axis]
            size = int(total / num)
            split_sizes = [size] * num
            res = mb.split(x=x, split_sizes=split_sizes, axis=axis)
            for val, name in zip(res, output_name):
                context.add(val, name)
            return
        else:
            raise ValueError("num must be greater than 0")
    else:
        axis = node.desc.attr("axis")
        if num > 0:
            total = x.shape[axis]
            size = int(total / num)
            split_sizes = [size] * num
            res = mb.split(x=x, split_sizes=split_sizes, axis=axis)
            for val, name in zip(res, output_name):
                context.add(val, name)
            return
        else:
            raise ValueError("num must be greater than 0")


def _broadcast(name, tensor, shape):
    if len(shape) > tensor.rank:
        new_dims = len(shape) - tensor.rank
        tensor = mb.expand_dims(x=tensor, axes=list(range(new_dims)))

    reps = []
    for ts, ds in zip(tensor.shape, shape):
        if not is_symbolic(ts) and not is_symbolic(ds) and ds > 0 and ts == 1:
            reps.append(ds)
        else:
            reps.append(1)

    res = mb.tile(x=tensor, reps=reps, name=name)
    return res


@register_paddle_op(paddle_alias=["expand_v2"])
def expand(context, node):
    def _broadcast_dynamic(name, tensor, shape):
        # Add any extra dimensions
        if len(shape) > tensor.rank:
            new_dims = len(shape) - tensor.rank
            tensor = mb.expand_dims(x=tensor, axes=list(range(new_dims)))

        tensor_shape = mb.shape(x=tensor)
        shape = mb.concat(values=shape, axis=0)
        reps = mb.real_div(x=shape, y=tensor_shape)
        reps = mb.cast(x=reps, dtype="int32")
        res = mb.tile(x=tensor, reps=reps, name=name)
        return res
    x = context[node.input("X")[0]]
    if node.input("Shape"):
        shape = context[node.input("Shape")[0]]
    if node.input("expand_shapes_tensor"):
        shape_tensor = context[node.input("expand_shapes_tensor")[0]]
    output_name = node.output("Out")[0]

    shape_attr = node.desc.attr("shape")

    if node.input("Shape"):
        res = _broadcast_dynamic(output_name, x, shape)
    elif node.input("expand_shapes_tensor"):
        res = _broadcast(output_name, x, shape_tensor.val)
    else:
        res = _broadcast_dynamic(output_name, x, shape_attr)
    
    context.add(res)


@register_paddle_op(paddle_alias=["range"])
def arange(context, node):
    start = context[node.input("Start")[0]]
    end = context[node.input("End")[0]]
    step = context[node.input("Step")[0]]

    output_name = node.output("Out")[0]

    # If start, end, and step don't have the same dtype, we cast them to fp32
    int_start = isinstance(start, int) or types.is_int(start.dtype)
    int_end = isinstance(end, int) or types.is_int(end.dtype)
    int_step = isinstance(step, int) or types.is_int(step.dtype)

    if int_start != int_end or int_start != int_step:
        start = mb.cast(x=start, dtype="fp32")
        end = mb.cast(x=end, dtype="fp32")
        step = mb.cast(x=step, dtype="fp32")
    res = mb.range_1d(start=start, end=end, step=step, name=output_name)
    context.add(res)


@register_paddle_op
def arg_max(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]

    axis = node.desc.attr("axis")
    keep_dims = node.desc.attr("keepdims")
    if types.is_int(x.dtype) and x.dtype._width == 64:
        # MIL reduce_argmax doesn't support int64.
        x = mb.cast(x=x, dtype="int32")
    res = mb.reduce_argmax(x=x, axis=axis, keep_dims=keep_dims, name=output_name)
    context.add(res)


@register_paddle_op
def gather_nd(context, node):
    x = context[node.input("X")[0]]
    indices = context[node.input("Index")[0]]
    output_name = node.output("Out")[0]
    res = mb.gather_nd(x=x, indices=indices, name=output_name)
    context.add(res)



@register_paddle_op
def index_select(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    indices = context[node.inputs[2]]
    context.add(mb.gather(x=x, indices=indices, axis=axis, name=node.name))


@register_paddle_op
def repeat(context, node):
    x = context[node.inputs[0]]
    reps = context[node.inputs[1]]
    if isinstance(reps, list):
        reps = mb.concat(values=reps, axis=0)

    if reps.shape[0] > len(x.shape):
        x = mb.expand_dims(x=x, axes=list(range(reps.shape[0] - x.rank)))
    context.add(mb.tile(x=x, reps=reps, name=node.name))


@register_paddle_op
def cos(context, node):
    x = context[node.input("X")[0]]
    output = node.output("Out")[0]
    context.add(mb.cos(x=x, name=output))

    
@register_paddle_op
def exp(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]
    context.add(mb.exp(x=x, name=output_name))


@register_paddle_op
def rsqrt(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]
    context.add(mb.rsqrt(x=x, name=output_name))


@register_paddle_op
def sin(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]
    context.add(mb.sin(x=x, name=output_name))


def _std(x, axes, keep_dim, unbiased, eps):
    need_rescale = False
    if unbiased:
        # If "unbiased" is True,
        # then we need to divide by "N-1" (instead of "N") to compute the mean of (x-E[x])^2
        # for an unbiased estimate of the variance /  standard deviation.
        # In the sequence of MIL ops added below, we first compute the mean using "N", and only if its unbiased
        # we rescale later, the final result.
        # We ignore the "unbiased" flag, if any of the dimensions involved in this operation are dynamic
        # (we could have still handled that case by using "get_shape" etc ops, but we don't do that here,
        # trading performance for numerical accuracy)
        if axes is None:
            if not any_symbolic(x.shape) and _np.prod(x.shape) > 1:
                N = _np.prod(x.shape)
                need_rescale = True
        else:
            dims = []
            # collect dimensions corresponding to "axes"
            for axis in axes:
                dims.append(x.shape[axis])
            if all([not is_symbolic(s) for s in dims]):
                N = _np.prod(dims)
                if N > 1:
                    need_rescale = True
    if need_rescale:
        rescale_factor = _np.sqrt(N / float(N - 1))

    x_mean = mb.reduce_mean(x=x, axes=axes, keep_dims=True)
    x_demeaned = mb.sub(x=x, y=x_mean)
    x_demeaned_square = mb.square(x=x_demeaned)
    x_demeaned_square_mean = mb.reduce_mean(x=x_demeaned_square, axes=axes, keep_dims=keep_dim)
    if eps > 0:
        x_demeaned_square_mean = mb.add(x=x_demeaned_square_mean, y=eps)
    if need_rescale:
        y_before_scale = mb.sqrt(x=x_demeaned_square_mean)
        y = mb.mul(x=y_before_scale, y=rescale_factor)
    else:
        y = mb.sqrt(x=x_demeaned_square_mean)
    return y
