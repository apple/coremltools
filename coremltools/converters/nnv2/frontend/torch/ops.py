import logging

import numpy as np
import torch

from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb

from .internal_graph import *
from .torch_op_registry import register_torch_op

# The pytorch args for many of the below ops were sourced from
# https://github.com/pytorch/pytorch/blob/d971007c291c0ead1003d12cd553d18ddb582207/torch/csrc/jit/mobile/register_mobile_ops.cpp#L216


def decide_immediate_or_file(val):
    if val is not None and isinstance(val, (np.ndarray, np.generic)) and val.size >= 10:
        return "file_value"
    else:
        return "immediate_value"


def _get_inputs(context, node, expected=None):
    """Look up a node's inputs in @context and return them as a list. If
        @expected is not None, also verifies the number of inputs matches the
        value of @expcted.
    """
    inputs = [context[name] for name in node.inputs]
    assert expected is None or len(inputs) == expected
    return inputs


@register_torch_op
def constant(context, node):
    assert len(node.inputs) == 0
    assert len(node.outputs) == 1

    name = node.name
    val = node.val
    # Converter cannot handle torch tensors.
    if isinstance(val, torch.Tensor):
        val = val.numpy()

    mode = decide_immediate_or_file(val)

    if val is None:
        const = None
    else:
        const = cb.const(mode=mode, val=val, name=name)
    context.add(const, torch_name=name)


@register_torch_op
def listconstruct(context, node):
    assert len(node.outputs) == 1

    mode = "immediate_value"
    val = []
    # The ListConstruct op gets its inputs from previously constructed
    # constant ops. Look those up and build the list directly as a
    # constant op. This will leave those constant ops unconnected, and
    # a cleanup pass will remove them.
    for _input in node.inputs:
        x = context[_input]
        val.append(x.val)
    const = cb.const(mode=mode, val=val, name=node.name)
    context.add(const)


@register_torch_op
def t(context, node):
    assert len(node.outputs) == 1

    # PyToch has several tranpose ops that can be emitted. This one is only
    # emitted when .T() is called on a tensor, which means it can only be
    # called on a matrix.
    if len(node.inputs) == 1:
        # Special case where we are transposing a rank 2 tensor (matrix).
        _input = node.inputs[0]
        transpose = cb.transpose(x=context[_input], perm=[1, 0], name=node.name)
        context.add(transpose)
    else:
        raise ValueError("transpose for rank != 2 is unsupported")


@register_torch_op
def matmul(context, node):
    inputs = _get_inputs(context, node, expected=2)
    matmul = cb.matmul(x=inputs[0], y=inputs[1], name=node.name)
    context.add(matmul)


@register_torch_op
def add(context, node):
    add_inputs = _get_inputs(context, node, expected=3)
    assert len(node.outputs) == 1

    # TODO (sberardi): 3rd param to aten::add is a scale factor, need to handle that.
    # out=input+alpha x other
    # rdar://60175736
    if add_inputs[2].val != 1:
        raise ValueError("ADD does not support scale factor param")

    add_node = cb.add(x=add_inputs[0], y=add_inputs[1], name=node.name)
    context.add(add_node)


@register_torch_op
def addmm(context, node):
    # addmm(Tensor input, Tensor mat1, Tensor mat2, Scalar beta=1, Scalar alpha=1)
    # output = beta * input + alpha * mat1 * mat2

    assert len(node.inputs) == 5
    assert len(node.outputs) == 1

    inputs = [context[name] for name in node.inputs]
    bias = inputs[0]
    mat1 = inputs[1]
    mat2 = inputs[2]
    beta = inputs[3]
    alpha = inputs[4]

    if beta.val != 1.0:
        # Apply scaling factor beta to the bias.
        bias = cb.mul(x=beta, y=bias, name=bias.name + "_scaled")
        context.add(bias)

    if alpha.val != 1.0:
        # Apply scaling factor alpha to the input.
        mat1 = cb.mul(x=alpha, y=mat1, name=mat1.name + "_scaled")
        context.add(mat1)

    # NNv2 linear will transpose mat2, but addmm expects that mat1 and mat2
    # can multiply as is. So we add a tranpose.
    mat2 = cb.transpose(x=mat2, perm=[1, 0], name=mat2.name + "_transposed")
    context.add(mat2)

    addmm_node = cb.linear(x=mat1, weight=mat2, bias=bias, name=node.name)
    context.add(addmm_node)


@register_torch_op
def _convolution(context, node):

    inputs = _get_inputs(context, node, expected=12)

    x = inputs[0]
    weight = inputs[1]
    bias = inputs[2]
    strides = inputs[3]

    pad = inputs[4]
    # Need to explicity state L-R, T-B pad
    pad = np.repeat(pad.val, 2) 

    dilations = inputs[5]
    group = inputs[8]

    kwargs = {
        "x": x,
        "W": weight,
        "strides": strides,
        "pad_type": "custom",
        "pad": pad,
        "dilations": dilations,
        "group": group,
        "name": node.name,
    }
    # Bias is optional in PyTorch's convolution.
    if bias:
        kwargs["B"] = bias
    conv = cb.conv(**kwargs)
    context.add(conv)


@register_torch_op
def softmax(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    axis = inputs[1]
    softmax = cb.softmax(x=x, axis=axis, name=node.name)
    context.add(softmax)


@register_torch_op
def flatten(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    dims = list(x.shape)
    start = inputs[1].val
    end_val = inputs[2].val

    total = 1
    if end_val < 0:
        end = len(dims) + end_val
    else:
        end = end_val

    if start > len(dims) or end > len(dims) or start < 0 or end < 0:
        raise ValueError(
            "Invalid start and end. (start, end) == ({}, {})".format(start, end_val)
        )
    if start > end:
        raise ValueError(
            "Start must be before end. (start, end) == ({}, {})".format(start, end_val)
        )
    for dim in dims[start : end + 1]:
        total *= dim
    dims = dims[:start] + [total] + dims[end + 1 :]

    reshape = cb.reshape(x=x, shape=dims, name=node.name)
    context.add(reshape)


@register_torch_op(torch_alias=["relu_"])
def relu(context, node):
    inputs = _get_inputs(context, node, expected=1)

    relu = cb.relu(x=inputs[0], name=node.name)
    context.add(relu)


@register_torch_op
def max_pool2d(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    kernel_sizes = inputs[1]
    strides = inputs[2]
    pad_type = "valid"

    # Need to explicity state L-R, T-B pad
    pad = inputs[3]
    pad = np.repeat(pad.val, 2)
    dilation = inputs[4]
    pool = cb.max_pool(
        x=x,
        kernel_sizes=kernel_sizes,
        strides=strides,
        pad_type=pad_type,
        pad=pad,
        name=node.name,
    )
    context.add(pool)


@register_torch_op
def div(context, node):
    inputs = _get_inputs(context, node, expected=2)

    div = cb.real_div(x=inputs[0], y=inputs[1], name=node.name)
    context.add(div)


@register_torch_op
def mul(context, node):
    inputs = _get_inputs(context, node, expected=2)

    mul = cb.mul(x=inputs[0], y=inputs[1], name=node.name)
    context.add(mul)


@register_torch_op
def pow(context, node):
    inputs = _get_inputs(context, node, expected=2)

    _pow = cb.pow(x=inputs[0], y=inputs[1], name=node.name)
    context.add(_pow)


@register_torch_op
def sub(context, node):
    inputs = _get_inputs(context, node, expected=2)

    sub = cb.sub(x=inputs[0], y=inputs[1], name=node.name)
    context.add(sub)


@register_torch_op
def mean(context, node):
    inputs = _get_inputs(context, node, expected=4)

    # @axes needs to be a list, but if only one axis was specified in the
    # model, it will be constructed as an int. Construct a new constant as a
    # list.
    axes = inputs[1]
    if not isinstance(axes.val, np.ndarray):
        axes = cb.const(val=[axes.val], name=axes.name + "_list")
        context.add(axes)
    keep_dims = inputs[2]
    # Last input to mean is an optional output tensor. We always expect this to
    # be None.
    assert inputs[3] is None
    mean = cb.reduce_mean(x=inputs[0], axes=axes, keep_dims=keep_dims, name=node.name)
    context.add(mean)


@register_torch_op
def size(context, node):
    inputs = _get_inputs(context, node, expected=2)

    # Get the shape of the tensor.
    shape_node = cb.shape(x=inputs[0], name=node.name + "_shape")
    context.add(shape_node)
    # Get the size of the tensor along the input dimension.
    dim = inputs[1]
    size_node = cb.const(val=shape_node.val[dim.val], name=node.name)
    context.add(size_node)


@register_torch_op
def view(context, node):
    inputs = _get_inputs(context, node, expected=2)

    view = cb.reshape(x=inputs[0], shape=inputs[1], name=node.name)
    context.add(view)
    
def adaptive_avg_pool2d(context, node):
    inputs = _get_inputs(context, node, expected=2)

    _input = inputs[0]
    output_size = inputs[1]
    assert isinstance(output_size.val, np.ndarray)
    if tuple(output_size.val) != (1, 1):
        raise ValueError(
            "adaptive_avg_pool2d only supports output size == (1,1). Recived: {}".format(output_size.val)
        )

    # Represent (1,1) output size via @reduce_mean
    # Assume (b,c,h,w) input shape
    axes = cb.const(val=[1, 2], name=node.name + "_axes")
    keep_dims = cb.const(val=True, name=node.name + "_keep_dims")

    reduce_mean = cb.reduce_mean(
        x=_input, axes=axes, keep_dims=keep_dims, name=node.name
    )
    context.add(reduce_mean)


@register_torch_op
def batch_norm(context, node):
    inputs = _get_inputs(context, node, expected=9)
    # inputs skipped:
    #   bool training (5)
    #   float momentum (6)
    #   bool cudnn_enabled (8)
    _input = inputs[0]
    weight = inputs[1]
    bais = inputs[2]
    running_mean = inputs[3]
    running_var = inputs[4]
    eps = inputs[7]
    batch_norm = cb.batch_norm(
        x=_input,
        mean=running_mean,
        variance=running_var,
        gamma=weight,
        beta=bais,
        epsilon=eps,
        name=node.name,
    )
    context.add(batch_norm)


@register_torch_op
def dropout(context, node):
    """Dropout is set as a noOP."""
    logging.warning("Setting dropout to no-op.")
    inputs = _get_inputs(context, node, expected=3)

    _input = inputs[0]
    context.add(_input, torch_name=node.name)


@register_torch_op
def hardtanh_(context, node):
    """Represent hardtanh as a hard sigmoid via the following translation:

        hardtanh(min_val, max_val) = 
            S * hardsigmoid(alpha = 1/S, beta = min_val/S) + min_val
            where S = (max_val - min_val)
    """
    inputs = _get_inputs(context, node, expected=3)
    _input = inputs[0]
    min_val = inputs[1].val
    max_val = inputs[2].val

    beta = cb.const(val=0.0)
    scalar = cb.const(val=max_val - min_val, name=node.name + "_scalar")
    alpha = cb.const(val=1 / (max_val - min_val), name=node.name + "_alpha")
    beta = cb.const(val=-min_val / (max_val - min_val), name=node.name + "_beta")
    offset = cb.const(val=min_val, name=node.name + "_offset")

    sig = cb.sigmoid_hard(
        x=_input, alpha=alpha, beta=beta, name=node.name + "_hard_sigmoid"
    )

    scaled = cb.mul(x=sig, y=scalar, name=node.name + "_scaled")
    offset = cb.add(x=scaled, y=offset, name=node.name)
    context.add(offset)
