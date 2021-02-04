#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging

import math as _math
import numbers
import numpy as _np
from tqdm import tqdm as _tqdm

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.var import Var, ListVar
from coremltools.converters.mil.mil import Placeholder, Symbol
from .internal_graph import *
from .torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.mil.types.symbolic import any_symbolic, is_symbolic
from coremltools.converters.mil.mil.types import is_bool

# The pytorch args for many of the below ops were sourced from
# https://github.com/pytorch/pytorch/blob/d971007c291c0ead1003d12cd553d18ddb582207/torch/csrc/jit/mobile/register_mobile_ops.cpp#L216


# This is a magic number in PyTorch. It's used as a default value in many
# functions.
PYTORCH_MAGIC_DEFAULT = 9223372036854775807


def _all_outputs_present(context, graph):
    """ Returns true if all the symbols in the graph's output list are
        present in context."""
    for outp in graph.outputs:
        try:
            context[outp]
        except ValueError:
            return False
    return True


def convert_nodes(context, graph):
    """Iterate over the nodes of a graph or block and convert to MIL.

        Arguments:
            context: A TranscriptionContext object to pull node inputs and
                assign node outputs.
            graph: An InternalTorchIRGraph or InternalTorchIRBlock object.
    """
    for node in _tqdm(graph.nodes, desc="Converting Frontend ==> MIL Ops", unit=" ops"):
        _add_op = _TORCH_OPS_REGISTRY.get(node.kind, None)
        _logging.info("Converting op {} : {}".format(node.name, node.kind))
        if _add_op is None:
            raise RuntimeError(
                "PyTorch convert function for op '{}' not implemented.".format(node.kind)
            )
        else:
            _add_op(context, node)

        # We've generated all the outputs the graph needs, terminate conversion.
        if _all_outputs_present(context, graph):
            break


def convert_block(context, block, inputs):
    """Convert a block (sub-graph) to MIL. Conversion happens within a new
        context frame.

        Arguments:
            context: A TranscriptionContext object to pull node inputs and
                assign node outputs.
            block: An InternalTorchIRBlock object.
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
# which maps to a torch dtype. The below mapping was found by
# converting test models with different dtypes passed to ones.
NUM_TO_TORCH_DTYPE = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    11: torch.bool,
}

NUMPY_DTYPE_TO_TORCH_NUM = {
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

NUM_TO_DTYPE_STRING = {
    3: "int32",
    4: "int64",
    6: "fp32",
    7: "fp64",
    11: "bool",
}


def decide_immediate_or_file(val):
    if (
        val is not None
        and isinstance(val, (_np.ndarray, _np.generic))
        and val.size >= 10
    ):
        return "file_value"
    return "immediate_value"


def _get_inputs(context, node, expected=None):
    """
    Look up a node's inputs in @context and return them as a list. If
    @expected is not None, also verifies the number of inputs matches the
    value of @expected.
    """
    inputs = [context[name] for name in node.inputs]
    if expected is not None:
        expected = [expected] if not isinstance(expected, (list, tuple)) else expected

        if len(inputs) not in expected:
            raise _logging.warning(
                "node {} ({}) got {} input(s), expected {}".format(
                    node.name, node.kind, len(inputs), expected
                )
            )

    return inputs


def _list_select(shape_var, index):
    """ Sometimes we need to select a specific item from a list. If that item
        is known at compile time, extract it as a const. Otherwise, if it's
        symbolic, use gather.
    """
    # TODO: gather doesn't work when the shape is known size.
    if shape_var.val is not None:
        res = mb.const(val=shape_var.val[index])
    else:
        res = mb.gather(x=shape_var, indices=index)
    return res


def _construct_constant(val, name):
    # Converter cannot handle torch tensors.
    if isinstance(val, torch.Tensor):
        val = val.numpy()

    # MIL casts ints to int32, which can't represent the 64 bit magic number.
    # So we instead represent it with None, and any ops that might get the
    # value will check for None instead.
    if isinstance(val, int) and val == PYTORCH_MAGIC_DEFAULT:
        val = None

    # Pytorch uses inf
    if val is not None and isinstance(val, numbers.Number) \
        and _np.isinf(val):
          if val < 0:  # neg inf
              # most negative number in fp32
              val = -3.4e+38
          else: # positive inf
              val = 3.4e+38
    mode = decide_immediate_or_file(val)
    if val is None:
        return None
    else:
        return mb.const(mode=mode, val=val, name=name)


@register_torch_op
def constant(context, node):
    assert len(node.inputs) == 0
    assert len(node.outputs) == 1

    name = node.name
    val = node.attr["value"]

    const = _construct_constant(val, name)
    context.add(const, torch_name=name)


def _array_construct(context, node, array_type):
    assert len(node.outputs) == 1
    inputs = _get_inputs(context, node)
    scalar_inputs = [
        inp
        for inp in inputs
        if isinstance(inp, Var) and inp.val is not None and len(inp.shape) == 0
    ]

    if len(scalar_inputs) == len(inputs):
        # All the list items are compile-time scalar constants, so let's create
        # a new const that concatenates them.
        mode = "immediate_value"
        val = array_type([inp.val for inp in inputs])
        const = mb.const(mode=mode, val=val, name=node.name)
        context.add(const)
    else:
        # If at least one input to the construct op is non-const, collect
        # the inputs and add them directly to the context. Ops that use this
        # node's output will take the list directly as input.
        context.add(array_type(inputs), node.name)


@register_torch_op
def tupleconstruct(context, node):
    _array_construct(context, node, array_type=tuple)


@register_torch_op
def listconstruct(context, node):
    _array_construct(context, node, array_type=list)


@register_torch_op
def eq(context, node):
    inputs = _get_inputs(context, node, expected=2)
    equal_to = mb.equal(x=inputs[0], y=inputs[1], name=node.name)
    context.add(equal_to)


@register_torch_op
def ne(context, node):
    inputs = _get_inputs(context, node, expected=2)
    equal_to = mb.not_equal(x=inputs[0], y=inputs[1], name=node.name)
    context.add(equal_to)


@register_torch_op
def le(context, node):
    inputs = _get_inputs(context, node, expected=2)
    less_equal = mb.less_equal(x=inputs[0], y=inputs[1], name=node.name)
    context.add(less_equal)


@register_torch_op
def lt(context, node):
    inputs = _get_inputs(context, node, expected=2)
    less = mb.less(x=inputs[0], y=inputs[1], name=node.name)
    context.add(less)


@register_torch_op
def ge(context, node):
    inputs = _get_inputs(context, node, expected=2)
    greater_equal = mb.greater_equal(x=inputs[0], y=inputs[1], name=node.name)
    context.add(greater_equal)


@register_torch_op
def gt(context, node):
    inputs = _get_inputs(context, node, expected=2)
    greater = mb.greater(x=inputs[0], y=inputs[1], name=node.name)
    context.add(greater)


@register_torch_op(torch_alias=["t", "transpose_"])
def transpose(context, node):
    assert len(node.outputs) == 1
    inputs = _get_inputs(context, node)
    x = inputs[0]

    if len(node.inputs) == 1:
        # PyTorch has several transpose ops that can be emitted. This one is only
        # emitted when .t() is called on a tensor, which means it can only be
        # called on a matrix.
        if len(x.shape) > 2:
            raise ValueError("transpose without dims for rank > 2 is unsupported")
        res = mb.transpose(x=x, perm=[1, 0], name=node.name)
    else:
        assert len(inputs) == 3
        ax0 = inputs[1].val
        ax1 = inputs[2].val

        perm = list(range(len(x.shape)))
        perm[ax0] = ax1
        perm[ax1] = ax0

        res = mb.transpose(x=x, perm=perm, name=node.name)
    context.add(res)


@register_torch_op
def permute(context, node):
    inputs = _get_inputs(context, node, expected=2)
    perm = mb.transpose(x=inputs[0], perm=inputs[1], name=node.name)
    context.add(perm)

@register_torch_op
def pixel_shuffle(context, node):
    inputs = _get_inputs(context, node, expected=2)
    perm = mb.pixel_shuffle(x=inputs[0], upscale_factor=inputs[1], name=node.name)
    context.add(perm)


@register_torch_op(torch_alias=["bmm"])
def matmul(context, node):
    inputs = _get_inputs(context, node, expected=2)
    if inputs[1].val is not None and \
            len(inputs[1].shape) == 2 and len(inputs[0].shape) <= 3:
        res = mb.linear(x=inputs[0], weight=_np.transpose(inputs[1].val), name=node.name)
    else:
        res = mb.matmul(x=inputs[0], y=inputs[1], name=node.name)
    context.add(res)

@register_torch_op
def add(context, node):
    add_inputs = _get_inputs(context, node)
    assert len(node.outputs) == 1

    # TODO (sberardi): 3rd param to aten::add is a scale factor, need to handle that.
    # out=input+alpha x other
    # rdar://60175736
    if len(add_inputs) > 2 and add_inputs[2].val != 1:
        raise ValueError("ADD does not support scale factor param")

    add_node = mb.add(x=add_inputs[0], y=add_inputs[1], name=node.name)
    context.add(add_node)

@register_torch_op
def cumsum(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    if is_bool(x.dtype):
        x = mb.cast(x=x, dtype='int32')
    res = mb.cumsum(x=x, axis=inputs[1], name=node.name)
    context.add(res)


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
        bias = mb.mul(x=beta, y=bias, name=bias.name + "_scaled")
        context.add(bias)

    if alpha.val != 1.0:
        # Apply scaling factor alpha to the input.
        mat1 = mb.mul(x=alpha, y=mat1, name=mat1.name + "_scaled")
        context.add(mat1)

    # MIL linear will transpose mat2, but addmm expects that mat1 and mat2
    # can multiply as is. So we add a tranpose.
    mat2 = mb.transpose(x=mat2, perm=[1, 0], name=mat2.name + "_transposed")
    context.add(mat2)

    addmm_node = mb.linear(x=mat1, weight=mat2, bias=bias, name=node.name)
    context.add(addmm_node)


@register_torch_op(torch_alias=["conv2d"])
def _convolution(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    weight = inputs[1]
    bias = inputs[2]
    strides = inputs[3]

    # Expand padding. Torch accepts either an int (for all dimensions) or an n-tuple of ints (one per dimension), but
    # we require a (2 * n)-tuple, where n is the number of spatial dimensions, start and end for each spatial dimension
    pad = inputs[4].val

    if weight.val.ndim in (3, 4):
        # 1D and 2D: Need to explicitly state L-R, T-B pad
        pad = _np.repeat(pad, 2)
    elif weight.val.ndim == 5:
        # 3D: Need to explicitly state F-Bk, L-R, T-B pad
        if type(pad) == int:
            pad = _np.repeat(pad, 6)
        elif len(pad) == 3:
            pad = _np.repeat(pad, 2)
    else:
        raise ValueError(
            "Invalid weight dimension. Must be 3, 4, or 5 for 1D, 2D, or 3D convolution, respectively."
        )

    dilations = inputs[5]
    out_pad = None
    if len(inputs) >= 12:
        transposed = inputs[6].val
        out_pad = inputs[7].val
        group = inputs[8]
    elif len(inputs) == 7:
        transposed = False
        group = inputs[6]
    else:
        raise ValueError(
            "unexpected number of inputs for node {} ({}): {}".format(
                node.name, node.kind, len(inputs)
            )
        )

    kwargs = {
        "x": x,
        "strides": strides,
        "pad_type": "custom",
        "pad": pad,
        "dilations": dilations,
        "groups": group,
        "name": node.name,
    }

    # Bias is optional in PyTorch's convolution.
    if bias is not None:
        kwargs["bias"] = bias

    if transposed is True:
        # Transposed convolution

        # PyTorch weight ordering [Cin, Cout, *D]
        # MIL expects [Cout, Cin, *D]
        perm = _np.arange(len(weight.shape))
        perm[[0, 1]] = perm[[1, 0]]
        weight_transpose = mb.transpose(
            x=weight, perm=perm, name=weight.name + "_transpose"
        )

        # Handle output_padding using pre-pad or post-crop
        pre_pad = [0] * len(pad)
        post_crop = [0] * len(pad)

        if out_pad is not None and any(out_pad):
            output_padding = [0] * len(pad)
            # output padding adds additional padding on one of the side of dimension
            # i.e. bottom from top-bottom,
            #      right  from left-right
            #      back   from front-back
            # CoreML padding structure is similar [top, bottom, left, right]
            # mapping output_padding to simplify further processing!
            #
            # For ConvTranspose2d: [bottom, right] -> [0, b, 0, r]
            output_padding = [0 if i % 2 == 0 else out_pad[i//2] for i in range(len(pad))]
            # TODO: rdar://65588783 ([PyTorch] Define and error out on unsupported configuration for output_padding)
            # error out here with unsupported configuration along with output padding
            if sum(pad) == 0 and any(output_padding):
                raise ValueError("ConvTranspose configuration of padding=0 and output_padding > 0 not supported!")
            post_crop = pad.copy()
            pad *= 0
            for i in range(0, len(pad)):
                if post_crop[i] >= output_padding[i]:
                    post_crop[i] -= output_padding[i]
                else:
                    pre_pad[i] = output_padding[i] - post_crop[i]
            kwargs["pad"] = pre_pad
            if any(pre_pad):
                # Constant pad requires pad to be of length 2*input_rank
                pre_pad = [0] * 2 * (len(x.shape) - 2) + pre_pad
                x = mb.pad(x=x, pad=pre_pad)
                kwargs["x"] = x
            if any(post_crop):
                del kwargs["name"]

        kwargs["weight"] = weight_transpose
        conv = mb.conv_transpose(**kwargs)
        if any(post_crop):
            # TODO: rdar://65575826 (PyTorch converter: output_padding mapping to slice
            # instead of crop layer for 1 and 3D ConvTranspose)
            if len(post_crop) == 2 and conv.rank == 3:
                # Number of elements to crop from right = post_crop[-1].
                # Since slicing supports negative indexing, end_id = -1 * post_crop[-1]
                conv = mb.slice_by_index(x=conv, begin=[0, 0, post_crop[0]], end=[0, 0, -1*post_crop[-1]],
                                         begin_mask=[True, True, False], end_mask=[True, True, False], name=node.name)
            elif len(post_crop) == 4 and conv.rank == 4:
                conv = mb.crop(x=conv, crop_height=post_crop[:2], crop_width=post_crop[2:4], name=node.name)
            else:
                raise ValueError("output_padding is supported only for ConvTranspose1D or ConvTranspose2D!")
    else:
        # Normal convolution
        kwargs["weight"] = weight
        conv = mb.conv(**kwargs)

    context.add(conv)

@register_torch_op
def softmax(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    axis = inputs[1]
    res = mb.softmax(x=x, axis=axis, name=node.name)
    context.add(res)


@register_torch_op
def flatten(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    dims = list(x.shape)
    start_val = inputs[1].val
    end_val = inputs[2].val

    total = 1

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
    for dim in dims[start : end + 1]:
        total *= dim
    dims = dims[:start] + [total] + dims[end + 1 :]

    reshape = mb.reshape(x=x, shape=dims, name=node.name)
    context.add(reshape)


@register_torch_op
def _reshape_from_tensor(context, node):
    inputs = _get_inputs(context, node, expected=2)

    reshape = mb.reshape(x=inputs[0], shape=inputs[1], name=node.name)
    context.add(reshape)

@register_torch_op
def softsign(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.softsign(x=inputs[0], name=node.name)
    context.add(res)

@register_torch_op(torch_alias=["relu_"])
def relu(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.relu(x=inputs[0], name=node.name)
    context.add(res)

@register_torch_op
def prelu(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    alpha = inputs[1].val
    alpha_vec = _np.ones((x.shape[1],))*alpha
    res = mb.prelu(x=x, alpha=alpha_vec, name=node.name)
    context.add(res)

@register_torch_op
def relu6(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.relu6(x=inputs[0], name=node.name)
    context.add(res)

@register_torch_op
def elu(context, node):
    ## Torch port to ATen adds scale and input_scale which is set to 1
    inputs = _get_inputs(context, node, expected=4)

    res = mb.elu(x=inputs[0], alpha = inputs[1], name=node.name)
    context.add(res)

@register_torch_op(torch_alias=["leaky_relu_"])
def leaky_relu(context, node):
    inputs = _get_inputs(context, node, expected=2)

    res = mb.leaky_relu(x=inputs[0], alpha=inputs[1], name=node.name)
    context.add(res)

@register_torch_op
def softplus(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    beta_ = inputs[1].val
    C = x.shape[1]
    alpha_br = _np.repeat(1.0 / beta_, C).astype('float32')
    beta_br = _np.repeat(beta_, C).astype('float32')

    res = mb.softplus_parametric(x=x, alpha = alpha_br, beta = beta_br, name=node.name)
    context.add(res)

def _adjust_pad_for_ceil_mode(input_shape, kernel_sizes, stride_sizes, pad_sizes):
    """ TODO Given an input tensor and pooling parameters, add the extra input
        padding needed to replicate ceil_mode. If no padding is needed, returns
        the original input. Otherwise, returns the Var returned by the new
        padding op.
    """
    new_pad = pad_sizes
    for idx in range(len(input_shape)):
        dim = input_shape[idx]
        kernel = kernel_sizes[idx]
        stride = stride_sizes[idx]
        pad = pad_sizes[idx * 2 : idx * 2 + 2]
        out_numerator = dim + pad[0] + pad[1] - kernel
        remainder = out_numerator % stride
        # Additional padding is added only on one side.
        # https://stackoverflow.com/questions/59906456/in-pytorchs-maxpool2d-is-padding-added-depending-on-ceil-mode
        if remainder > 0:
            # MIL pooling does not support ceil_mode natively, but we can
            # workaround by padding the input appropriately.
            # rdar://60634390
            _logging.warning("pooling padding adjusted to support ceil_mode=True")
            new_pad[2 * idx + 1] += stride - remainder

    return new_pad


def _max_pool(context, node, inputs):
    x = inputs[0]
    kernel_sizes = inputs[1]
    strides = inputs[2]
    if strides.op.op_type == "const"  and (not list(strides.val)):
        strides = mb.const(val=kernel_sizes.val, name=strides.name)

    pad_type = "custom"
    # Need to explicitly state L-R, T-B pad
    pad = inputs[3]
    pad = _np.repeat(pad.val, 2)
    dilation = inputs[4].val
    ceil_mode = inputs[5].val
    if _np.any(dilation > 1):
        # See: rdar://60633736 (Implement dilation for mil op max_pool)
        raise ValueError("@max_pool does not support dilation > 1")
    if ceil_mode is True and list(strides.val) != [1] * len(strides.val):
        # need to adjust padding values if ceil_mode is True
        # ceil_mode only causes any difference though, if the strides are not 1 
        rank = len(pad) // 2
        x_spatial_dimensions = x.shape[-rank:]
        if any_symbolic(x_spatial_dimensions):
            raise ValueError("@max_pool does not support symbolic input spatial shape when ceil_mode is True")
        pad = _adjust_pad_for_ceil_mode(
            x_spatial_dimensions, kernel_sizes.val, strides.val, pad
        )

    pool = mb.max_pool(
        x=x,
        kernel_sizes=kernel_sizes,
        strides=strides,
        pad_type=pad_type,
        pad=pad,
        name=node.name,
    )
    context.add(pool)


@register_torch_op
def max_pool1d(context, node):
    inputs = _get_inputs(context, node, expected=6)
    _max_pool(context, node, inputs)


@register_torch_op
def max_pool2d(context, node):
    inputs = _get_inputs(context, node, expected=6)
    _max_pool(context, node, inputs)


@register_torch_op
def max_pool3d(context, node):
    inputs = _get_inputs(context, node, expected=6)
    _max_pool(context, node, inputs)


@register_torch_op
def div(context, node):
    inputs = _get_inputs(context, node, expected=2)

    res = mb.real_div(x=inputs[0], y=inputs[1], name=node.name)
    context.add(res)


@register_torch_op(torch_alias=["floordiv"])
def floor_divide(context, node):
    inputs = _get_inputs(context, node, expected=2)
    div_res = mb.floor_div(x=inputs[0], y=inputs[1])
    # Pytorch's floor_divide always returns fp32, even if the inputs are int
    res = mb.cast(x=div_res, dtype='fp32', name=node.name)
    context.add(res)

@register_torch_op
def true_divide(context, node):
    inputs = _get_inputs(context, node, expected=2)
    res = mb.real_div(x=inputs[0], y=inputs[1], name=node.name)
    context.add(res)


@register_torch_op
def mul(context, node):
    inputs = _get_inputs(context, node, expected=2)

    for i, input in enumerate(inputs):
        if is_bool(input.dtype):
            inputs[i] = mb.cast(x=inputs[i], dtype="int32")

    res = mb.mul(x=inputs[0], y=inputs[1], name=node.name)
    context.add(res)


@register_torch_op(torch_alias=["pow"])
def pow_(context, node):
    inputs = _get_inputs(context, node, expected=2)

    res = mb.pow(x=inputs[0], y=inputs[1], name=node.name)
    context.add(res)


@register_torch_op(torch_alias=["rsub"])
def sub(context, node):
    inputs = _get_inputs(context, node, expected=[2, 3])
    assert len(node.outputs) == 1

    if node.kind == "rsub":
        # rsub reverses the order of arguments
        y = inputs[0]
        x = inputs[1]
    else:
        x = inputs[0]
        y = inputs[1]

    if len(inputs) > 2:
        alpha = inputs[2].val

        # TODO (sberardi): 3rd param to aten::sub is a scale factor, need to handle that.
        # out=input-alpha x other
        # rdar://60175736
        if alpha != 1:
            raise ValueError("SUB does not support scale factor param")

    res = mb.sub(x=x, y=y, name=node.name)
    context.add(res)


@register_torch_op(torch_alias=["sum"])
def mean(context, node):
    inputs = _get_inputs(context, node)

    kwargs = {"x": inputs[0], "name": node.name}

    # @axes is optional, so omit if None.
    axes = inputs[1]
    if axes is not None:
        # @axes needs to be a list, but if only one axis was specified in the
        # model, it will be constructed as an int. Construct a new constant as a
        # list.
        if not isinstance(axes.val, _np.ndarray):
            axes = mb.const(val=[axes.val], name=axes.name + "_list")
            context.add(axes)
        kwargs["axes"] = axes

    # @keep_dims is optional.
    if len(inputs) >= 3:
        keep_dims = inputs[2]
        kwargs["keep_dims"] = keep_dims

    # Last input to mean is an optional output tensor. We always expect this to
    # be None or absent.
    assert len(inputs) <= 3 or inputs[3] is None
    func = mb.reduce_sum if node.kind == "sum" else mb.reduce_mean
    res = func(**kwargs)
    context.add(res)

@register_torch_op
def squeeze(context, node):
    inputs = _get_inputs(context, node)
    if len(inputs) == 1:
        res = mb.squeeze(x=inputs[0], name=node.name)
    elif len(inputs) == 2:
        squeeze_dim = inputs[1].val
        res = mb.squeeze(x=inputs[0], axes=(squeeze_dim,), name=node.name)
    context.add(res)


@register_torch_op
def unsqueeze(context, node):
    inputs = _get_inputs(context, node, expected=2)
    unsqueeze = mb.expand_dims(x=inputs[0], axes=[inputs[1].val], name=node.name)
    context.add(unsqueeze)


@register_torch_op
def size(context, node):
    inputs = _get_inputs(context, node, expected=[1, 2])

    # Get the shape of the tensor.
    size_node = mb.shape(x=inputs[0], name=node.name + "_shape")
    # Get the size of the tensor along the input dimension.
    if len(node.inputs) == 2:
        dim = inputs[1].val
        size_node = _list_select(size_node, dim)
    context.add(size_node, node.name)


@register_torch_op
def _shape_as_tensor(context, node):
    inputs = _get_inputs(context, node, expected=1)

    # Get the shape of the tensor.
    shape_node = mb.shape(x=inputs[0], name=node.name)
    context.add(shape_node, node.name)



@register_torch_op(torch_alias=["reshape"])
def view(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    shape = inputs[1]

    if isinstance(shape, ListVar):
        length = mb.list_length(ls=shape)
        indices = mb.range_1d(start=0, end=length, step=1)
        shape = mb.list_gather(ls=shape, indices=indices)

    if isinstance(shape, list) and all([isinstance(dim, Var) and len(dim.shape) == 0 for dim in shape]) and any([dim.val is None for dim in shape]):
        shape = mb.concat(values=shape, axis=0)

    view = mb.reshape(x=x, shape=shape, name=node.name)
    context.add(view)


@register_torch_op
def adaptive_avg_pool2d(context, node):
    inputs = _get_inputs(context, node, expected=2)

    _input = inputs[0]
    output_size = inputs[1].val
    assert isinstance(output_size, _np.ndarray)
    output_size = tuple(output_size)

    if output_size == (1, 1):
        # Represent (1,1) output size via @reduce_mean
        # Assume channel first ordering, reduce the last two (HW) dims.
        axes = mb.const(val=[-2, -1], name=node.name + "_axes")
        keep_dims = mb.const(val=True, name=node.name + "_keep_dims")

        avg_pool = mb.reduce_mean(
            x=_input, axes=axes, keep_dims=keep_dims, name=node.name
        )
    elif _input.shape is not None:
        # TODO: The calculations to convert adaptive_pool to standard pool,
        # given a known input size, come from
        # https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work
        # However, as indicated in that SO, this isn't quite how PyTorch
        # computes adaptive pooling, leading to inaccuracies in model outputs.
        # rdar://60900834
        strides = [ind // outd for ind, outd in zip(_input.shape[-2:], output_size)]
        pad_type = "valid"
        # Need to explicity state L-R, T-B pad
        pad = [0, 0, 0, 0]
        dilation = [1, 1]
        kernel_sizes = [
            ind - s * (outd - 1)
            for ind, outd, s in zip(_input.shape[-2:], output_size, strides)
        ]
        avg_pool = mb.avg_pool(
            x=_input,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pad_type=pad_type,
            pad=pad,
            name=node.name,
        )
    else:
        raise ValueError(
            "adaptive_avg_pool2d only supported when input tensor size is known or output size == (1,1). Recived: input size == {}, output size == {}".format(
                _input.shape_str(), output_size,
            )
        )

    context.add(avg_pool)

@register_torch_op
def constant_pad_nd(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    pad = inputs[1]
    if pad.op.op_type == "const":
        pad = pad.val.reshape((-1, 2))[::-1].reshape(-1).tolist()
        missing_dims = x.rank - (len(pad) // 2)
        pad = [0, 0] * missing_dims + pad

    scalar_val = inputs[2] if inputs[2] else 0.0
    if inputs[2] and inputs[2].op.op_type == "const":
        scalar_val = float(scalar_val.val)

    res = mb.pad(x=x, pad=pad, mode="constant", constant_val=scalar_val, name=node.name)
    context.add(res)

@register_torch_op
def adaptive_max_pool2d(context, node):
    inputs = _get_inputs(context, node, expected=2)

    _input = inputs[0]
    output_size = inputs[1].val
    assert isinstance(output_size, _np.ndarray)
    output_size = tuple(output_size)

    if output_size == (1, 1):
        # Represent (1,1) output size via @reduce_max
        # Assume channel first ordering, reduce the last two (HW) dims.
        max_pool = mb.reduce_max(
            x=_input, axes=[-2,-1], keep_dims=True, name=node.name
        )
    elif _input.shape is not None:
        # TODO: The calculations to convert adaptive_pool to standard pool,
        # given a known input size, come from
        # https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work
        # However, as indicated in that SO, this isn't quite how PyTorch
        # computes adaptive pooling, leading to inaccuracies in model outputs.
        # rdar://60900834
        strides = [ind // outd for ind, outd in zip(_input.shape[-2:], output_size)]
        pad_type = "valid"
        # Need to explicity state L-R, T-B pad
        pad = [0, 0, 0, 0]
        dilation = [1, 1]
        kernel_sizes = [
            ind - s * (outd - 1)
            for ind, outd, s in zip(_input.shape[-2:], output_size, strides)
        ]
        max_pool = mb.max_pool(
            x=_input,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pad_type=pad_type,
            pad=pad,
            name=node.name,
        )
    else:
        raise ValueError(
            "adaptive_max_pool2d only supported when input tensor size is known or output size == (1,1). Recived: input size == {}, output size == {}".format(
                _input.shape_str(), output_size,
            )
        )

    context.add(max_pool)



@register_torch_op
def batch_norm(context, node):
    inputs = _get_inputs(context, node, expected=9)
    # inputs skipped:
    #   bool training (5)
    #   float momentum (6)
    #   bool cudnn_enabled (8)
    input_rank = inputs[0].rank
    if input_rank > 4:
        raise NotImplementedError("Translation for BatchNorm3d not supported")

    if input_rank < 2:
        raise ValueError("BatchNorm: Encountered invalid input rank during translation in torch frontend.")

    _input = inputs[0]
    weight = inputs[1]
    bias = inputs[2]
    running_mean = inputs[3]
    running_var = inputs[4]
    eps = inputs[7]
    name = node.name
    if input_rank == 2:
        _input = mb.expand_dims(x=_input, axes=[-1], name=node.name + "_rank2_expansion")
        name = node.name + "_batch_norm_1d"

    batch_norm = mb.batch_norm(
        x=_input,
        mean=running_mean,
        variance=running_var,
        gamma=weight,
        beta=bias,
        epsilon=eps,
        name=name,
    )

    if input_rank == 2:
        batch_norm = mb.squeeze(x=batch_norm, name=node.name, axes=[-1])

    context.add(batch_norm)


@register_torch_op
def instance_norm(context, node):
    inputs = _get_inputs(context, node, expected=9)
    x = inputs[0]
    weight = inputs[1]
    bias = inputs[2]
    eps = inputs[7]
    x = mb.instance_norm(
        x=x,
        gamma=weight,
        beta=bias,
        epsilon=eps,
        name=node.name,
    )
    context.add(x)

@register_torch_op
def group_norm(context, node):
    inputs = _get_inputs(context, node, expected=6)
    x = inputs[0]
    num_groups = inputs[1].val
    weight = inputs[2]
    bias = inputs[3]
    eps = inputs[4]
    n,c,h,w = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
    num_groups = min(num_groups,c)
    x = mb.reshape(x=x, shape=[n,num_groups,c//num_groups,h,w])
    mean = mb.reduce_mean(x=x, axes=[2,3,4], keep_dims=True)
    var = _std(x,[2,3,4],True,False,eps.val)
    x = mb.sub(x=x,y=mean)
    x = mb.real_div(x=x,y=var)
    x = mb.reshape(x=x, shape=[n,c,h,w])
    if weight is not None:
        weight = mb.reshape(x=weight, shape=[1,c,1,1])
        x = mb.mul(x=x,y=weight)
    if bias is not None:
        bias = mb.reshape(x=bias, shape=[1,c,1,1])
        x = mb.add(x=x,y=bias)
    context.add(x,node.name)


@register_torch_op
def embedding(context, node):
    inputs = _get_inputs(context, node)
    _input = inputs[0]
    indices = inputs[1]

    padding_idx = -1
    scale_grad_by_freq = False
    sparse = False
    if len(inputs) >= 3:
        padding_idx = inputs[2].val
    if len(inputs) >= 4:
        scale_grad_by_freq = inputs[3].val
    if len(inputs) >= 5:
        sparse = inputs[4].val

    if padding_idx != -1 or scale_grad_by_freq or sparse:
        _logging.warning(
            "CoreML embedding (gather) layer does not support any "
            "inputs besides the weights and indices. Those given "
            "will be ignored."
        )

    #  Changing the axis from 0 is not an option in torch, so we don't expose it
    gather = mb.gather(x=_input, indices=indices, name=node.name)
    context.add(gather)


@register_torch_op(torch_alias=["hardtanh_"])
def hardtanh(context, node):
    inputs = _get_inputs(context, node, expected=3)
    _input = inputs[0]
    min_val = inputs[1].val
    max_val = inputs[2].val

    res = mb.clip(x=_input, alpha=min_val, beta=max_val, name=node.name)
    context.add(res)


@register_torch_op
def cat(context, node):
    inputs = _get_inputs(context, node)
    axis = 0 if len(inputs) == 1 else inputs[1]
    concat = mb.concat(values=inputs[0], axis=axis, name=node.name)
    context.add(concat)


@register_torch_op
def stack(context, node):
    inputs = _get_inputs(context, node)

    values = inputs[0]

    if len(inputs) < 2:
        axis = 0
    else:
        axis = inputs[1]

    res = mb.stack(values=values, axis=axis, name=node.name)
    context.add(res)


@register_torch_op
def item(context, node):
    inputs = _get_inputs(context, node, expected=1)

    if inputs[0].shape == ():
        # MIL ops that reduce already output a scalar, so no need to do
        # anything.
        res = inputs[0]
    elif _np.all([d == 1 for d in inputs[0].shape]):
        # Item only makes sense when called on a length 1 tensor. We use
        # reduce_max as a workaround for not having a way to extract a scalar
        # from a symbolic tensor.
        res = mb.reduce_max(x=inputs[0], name=node.name)
    else:
        raise ValueError("expected input to be a scalar or a length 1 tensor")
    context.add(res, node.name)


def _cast(context, node, dtype, dtype_name):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    # Input must either be a scalar or a (1 x 1 x ... x 1) tensor
    if not (len(x.shape) == 0 or _np.all([d == 1 for d in x.shape])):
        raise ValueError("input to cast must be either a scalar or a length 1 tensor")

    if x.val is not None:
        # If x is a compile-time constant, directly cast it to @dtype if it's
        # not one already.
        if not isinstance(x.val, dtype):
            res = mb.const(val=dtype(x.val), name=node.name)
        else:
            res = x
    elif x.shape == (1,):
        x = mb.squeeze(x=x, name=node.name + "_item")
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    else:
        if len(x.shape) > 0:
            # TODO: There's no MIL op to extract a value from a symbolic tensor,
            # so as a workaround we use reduce_max to convert it to a scalar.
            x = mb.reduce_max(x=x, name=node.name + "_item")
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    context.add(res, node.name)


@register_torch_op(torch_alias=["bool"])
def _bool(context, node):
    _cast(context, node, bool, "bool")


@register_torch_op(torch_alias=["int"])
def _int(context, node):
    _cast(context, node, int, "int32")


@register_torch_op
def layer_norm(context, node):
    inputs = _get_inputs(context, node, expected=6)
    _input = inputs[0]
    normalized_shape = inputs[1]
    weight = inputs[2]
    bias = inputs[3]
    eps = inputs[4]
    # cudnn_enable = inputs[5] unused

    layer_norm = mb.layer_norm(
        x=_input,
        axes=list(range(-len(normalized_shape.val),0)),
        gamma=weight,
        beta=bias,
        epsilon=eps,
        name=node.name,
    )
    context.add(layer_norm)


@register_torch_op
def numtotensor(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    if x.shape != ():
        raise ValueError(
            "numtotensor expected scalar input, got tensor with shape {}".format(
                x.shape
            )
        )

    if x.val is not None:
        res = mb.const(val=[x.val], name=node.name)
        context.add(res)
    else:
        context.add(x, node.name)

def _ifzo_to_ifoz(weights, name):
    """
        i, f, z, o -> i, f, o, z
        where weights_split[0] == i, etc.
        Used to transform lstm weights from pytorch
        to CoreML format
    """
    split_size = weights.shape[0] // 4
    weights_split = mb.split(x=weights, split_sizes=_np.array([split_size] * 4), axis=0)
    weights_concat = mb.concat(
        values=[weights_split[0], weights_split[1], weights_split[3], weights_split[2]],
        axis=0,
    )
    # make transpose a noOP for 0/1d tensors
    return mb.transpose(
        x=weights_concat, perm=([1, 0] if len(weights.shape) > 1 else [0]), name=name
    )


def _pytorch_hidden_to_coreml_milops(x, name):
    """
        Used to transform lstm state values (hn, cn)
        from pytorch to CoreML format.
    """
    split_size = x.shape[0] // 2
    x_split = mb.split(x=x, split_sizes=_np.array([split_size] * 2), axis=0)
    x_concat = mb.concat(values=[x_split[0], x_split[1]], axis=2,)
    # (4.) See docstring to @lstm
    return mb.squeeze(x=x_concat, axes=_np.array([0]), name=name)


def _add_mil_lstm(input, initial_h, initial_c, weights, has_bias, bidirectional, name):

    '''
    Most of this code is to transform the tensors into
    a shape acceptable by the CoreML implementation of LSTM.

    For weights, biases,  per direction, pytorch uses two tensors:
    (ii, if, ig, io) stacked on top of each other for each layer (tensor 1)
    and (hi, hf, hg, ho) stacked on top of each other for each layer (tensor 2).
    That is,  (W_ii|W_if|W_ig|W_io), of shape (4*hidden_size, input_size) and
    (W_hi|W_hf|W_hg|W_ho), of shape (4*hidden_size, hidden_size).


    The CoreML LSTM op expects two tensors, weight and bias. So
    the tensors for weight and bias are seperated from pytorch's @weights list (1.).
    For each individual weight and bias tensor, the CoreML LSTM op expects the form
    ii, if, io, ig and hi, hf, ho, hg, requiring the ifzo_to_ifoz function (2.).
    Each seperate weight and bias tensor is concatinated to
    form the two weight and bias tensors. (3.)
    In the bidirectional case, the forward and backward weights and biases
    are stacked on top of eachother instead of stored as seperate tensors in
    the @weights list. (4.)

    initial_h and initial_c are list of "num_layers" tensors, each of shape [n_directions, B, H],
    where n_directions = 1 or 2
    whereas the shapes of the initial states to MIL's LSTM, BiLSTM must be [B, H] and [B, 2*H] respectively.
    This means we need to do the following transformations:
    - if its an LSTM (n_directions=1):
            squeeze the first dimension of initial_h/initial_c , before feeding it to MIL's LSTM
    - if its a BiLSTM (n_directions=2):
            - split the input, shape=(2, B, H), to get (1,B,H) and (1,B,H)
            - concatenate to get (1,B,2*H)
            - squeeze to get (B,2*H)
    '''

    if bidirectional:
        if has_bias:
            # (1.)
            biases = weights[2:4] + weights[6:8]
            weights = weights[0:2] + weights[4:6]

            # (2.)
            assert len(biases) == 4
            for index in range(len(biases)):
                biases[index] = _ifzo_to_ifoz(
                    biases[index],
                    name="{}_lstm_bias_reshape_{}".format(name, index),
                )

            # (4.)
            f_stack = mb.stack(values=biases[0:2], axis=0, )
            r_stack = mb.stack(values=biases[2:4], axis=0, )
            # (3.)
            final_biases = mb.concat(
                values=(f_stack, r_stack),
                axis=1,
                name=name + "_lstm_biases_concat",
            )

        # (4.)
        forward_concat = mb.concat(
            values=[weights[0], weights[1]],
            axis=1,
            name=name + "_lstm_weights_forward_concat",
        )
        backward_concat = mb.concat(
            values=[weights[2], weights[3]],
            axis=1,
            name=name + "_lstm_weights_backward_concat",
        )
        # (2.)
        forward_transformed = _ifzo_to_ifoz(
            forward_concat, name=name + "_lstm_forward_weights_ifoz_to_ifzo",
        )
        backward_transformed = _ifzo_to_ifoz(
            backward_concat, name=name + "_lstm_backward_weights_ifoz_to_ifzo"
        )
        # (3.)
        final_weights = mb.concat(
            values=[forward_transformed, backward_transformed],
            axis=1,
            name=name + "_lstm_weights_final_concat",
        )

        # (5.)
        h = _pytorch_hidden_to_coreml_milops(initial_h, name=name + "_lstm_h0_reshaped")
        c = _pytorch_hidden_to_coreml_milops(initial_c, name=name + "_lstm_c0_reshaped")

    else:
        if has_bias:
            # (1.)
            biases = weights[len(weights) // 2:]
            weights = weights[: len(weights) // 2]
            ih_b = biases[0]
            hh_b = biases[1]

            # (2.)
            ih_b_transformed = _ifzo_to_ifoz(
                ih_b, name=name + "_lstm_ih_bias_transformed",
            )
            hh_b_transformed = _ifzo_to_ifoz(
                hh_b, name=name + "_lstm_hh_bias_transformed",
            )

            # (3.)
            final_biases = mb.stack(
                values=(ih_b_transformed, hh_b_transformed),
                axis=0,
                name=name + "_lstm_bias_stacked",
            )

        # (3.)
        weights_concat = mb.concat(
            values=weights, axis=1, name=name + "_lstm_weights_concat"
        )
        # (2.)
        final_weights = _ifzo_to_ifoz(
            weights_concat, name=name + "_lstm_weights_ifoz_to_ifzo",
        )

        # (4.)
        h = mb.squeeze(x=initial_h, axes=_np.array([0]), name=name + "_lstm_h0_squeeze")
        c = mb.squeeze(x=initial_c, axes=_np.array([0]), name=name + "_lstm_c0_squeeze")

    lstm = mb.lstm(
        x=input,
        initial_h=h,
        initial_c=c,
        weight=final_weights,
        bias=(final_biases if has_bias else None),
        direction=("bidirectional" if bidirectional is True else "forward"),
        output_sequence=True,
        name=name,
    )

    return lstm


@register_torch_op
def lstm(context, node):
    inputs = _get_inputs(context, node, expected=9)

    _input = inputs[0]
    h0, c0 = inputs[1]
    weights_list = inputs[2]
    has_bias = inputs[3].val # bool
    num_layers = inputs[4].val
    dropout = inputs[5] # ignored in the translation
    bidirectional = inputs[7].val
    batch_first = inputs[8].val

    '''
    Torch LSTM layer's input shapes: 
    
    (1) first input
        (Seq, B, C) : if batch_first = False
        (B, Seq, C) : if batch_first = True
        
    (2) & (3) initialization states
        (num_layers, B, H) : if bidirectional = False
        (num_layers * 2, B, H) : if bidirectional = True
        
        
    For the MIL LSTM layer, these are the input shapes:
     
    (1) first input: (Seq, B, C)
           this means, if batch_first=True, we need to insert a transpose op first
           
    (2) & (3) initialization states
        MIL's LSTM layer does not natively support the "num_layers" parameters.
        So, when num_layers > 1, we add multiple MIL LSTM ops in a sequence.
        Each of these LSTM ops will take in initialization states in the following shape:
        (B, H) if bidirectional = False
        (B, 2*H) if bidirectional = True
    '''

    if batch_first:
        _input = mb.transpose(x=_input, perm=[1, 0, 2], name=_input.name + "_batch_first_transpose")

    expected_num_weights = 2 * num_layers * (int(bidirectional) + 1) * (int(has_bias) + 1)
    if len(weights_list) != expected_num_weights:
        raise ValueError(
            "Incorrect weights shape for lstm layer: Expected: {}. Recieved {}".format(
                expected_num_weights, len(weights_list)
            )
        )

    # shape of h0 and c0 are (num_layers * n_directions, B, H)
    if num_layers == 1:
        all_initial_h = [h0]   # [(n_directions, B, H)]
        all_initial_c = [c0]   # [(n_directions, B, H)]
    else:
        all_initial_h = mb.split(x=h0, num_splits=num_layers, axis=0) # [(n_directions, B, H)]
        all_initial_c = mb.split(x=c0, num_splits=num_layers, axis=0) # [(n_directions, B, H)]

    n_weights_per_layer = int(len(weights_list) / num_layers)
    x = _input
    h_out_list = []
    c_out_list = []
    for i in range(num_layers):
        if i < num_layers - 1:
            op_name = node.name + "_lstm_layer_{}".format(i)
        else:
            if batch_first:
                op_name = node.name + "_batch_first"
            else:
                op_name = node.name

        lstm_out = _add_mil_lstm(input=x,
                                 initial_h=all_initial_h[i],
                                 initial_c=all_initial_c[i],
                                 weights=weights_list[i * n_weights_per_layer : (i+1) * n_weights_per_layer],
                                 has_bias=has_bias,
                                 bidirectional=bidirectional,
                                 name=op_name,
                                 )
        x = lstm_out[0] # shape of lstm_out[0] == (S,B,H) if bidirectional = True else (S, B, 2*H)
        h_out_list.append(lstm_out[1]) # shape of lstm_out[1] == (B,H) if bidirectional = False else (B, 2*H)
        c_out_list.append(lstm_out[2]) # shape of lstm_out[2] == (B,H) if bidirectional = False else (B, 2*H)


    '''
    For torch, these are the dimensions of the 3 output tensors: 
    (1) output[0] : 
            (Seq, B, H) if batch_first = False, bidirectional = False
            (Seq, B, 2*H) if batch_first = False, bidirectional = True
            (B, Seq, H) if batch_first = True, bidirectional = False
            (B, Seq, 2*H) if batch_first = True, bidirectional = True
            
    (2) & (3) these are the state outputs: 
            (num_layers, B, H) if bidirectional = False
            (num_layers * 2, B, H) if bidirectional = True
            
    MIL lstm layer's output shapes:
    (1) output[0]: 
        (Seq, B, H) if bidirectional = False
        (Seq, B, 2*H) if bidirectional = True
        This means we need a transpose op if batch_first is True
        
    (2) & (3) shapes of the state outputs:
        each MIL LSTM op will produce final state tensors with the following shape: 
        (B, H) if bidirectional = False
        (B, 2*H) if bidirectional = True
        
        stack/expand the final state tensors to match the Torch output
    '''
    for index, (name, output) in enumerate(zip(node.outputs, lstm_out)):
        if index > 0:
            # index > 0 ===> its one of the state outputs (h or c)
            if bidirectional:
                if num_layers == 1:
                    out1, out2 = mb.split(x=output, num_splits=2, axis=1) # each output of shape [B, H] after the split
                    final_out = mb.stack(values=[out1, out2], axis=0, name=name) # [2, B, H]
                    context.add(final_out, name)
                else:
                    out_state_tensors_list = h_out_list if index == 1 else c_out_list # each tensor in the list is of shape (B, 2*H)
                    list_of_tensors_to_stack = []
                    for i in range(num_layers):
                        out1, out2 = mb.split(x=out_state_tensors_list[i], num_splits=2, axis=1) # each output of shape [B, H] after the split
                        out = mb.stack(values=[out1, out2], axis=0)  # [2, B, H]
                        list_of_tensors_to_stack.append(out)
                    final_out = mb.concat(values=list_of_tensors_to_stack, axis=0) # output of shape (num_layers * 2, B, H)
                    context.add(final_out, name)
            else:
                if num_layers == 1:
                    unsqueeze = mb.expand_dims(x=output, axes=[0], name=name)
                    context.add(unsqueeze, name)
                else:
                    out = mb.stack(values=h_out_list if index == 1 else c_out_list, axis=0, name=name)
                    context.add(out, name)
        else:
            if batch_first:
                output = mb.transpose(x=output, perm=[1, 0, 2], name=name)
            context.add(output, name)


def _get_scales_from_output_size(output_size, input_shape):
    scales = []
    if output_size is not None:
        # @output_size will be a list if scales was provided or a
        # single var if output size was provided
        if isinstance(output_size, list):
            output_size = [output_size[0].val, output_size[1].val]
        if isinstance(output_size, Var):
            output_size = [output_size.val[0], output_size.val[1]]

        # output size is computed using the formula
        # floor (scale * input_size) in Core ML (and PyTorch)
        # Thus, when computing the scales from the output size,
        # add a small positive constant to the output size,
        # to make sure that the floor formula results in the correct output
        # size and not 1 unit smaller, due to float precision issues
        # e.g. if output size = 34 and input size = 2, then scale will be
        # 17, which can get represented as 16.9999, resulting in an output size of 33
        # instead of 34, without this correction.
        scales_h = (output_size[0] + 1e-4) / float(input_shape[-2])
        scales_w = (output_size[1] + 1e-4) / float(input_shape[-1])
        scales = [scales_h, scales_w]
    return scales


@register_torch_op
def upsample_bilinear2d(context, node):
    inputs = _get_inputs(context, node)
    _input = inputs[0]
    output_size = inputs[1]
    align_corners = bool(inputs[2].val)

    scales_h, scales_w = None, None
    if output_size is None:
        # get scale factors from provided inputs
        scale_factors = inputs[3].val
        scales_h = scale_factors[0]
        scales_w = scale_factors[1]
    else:
        # infer scale factors from output sizes
        scales = _get_scales_from_output_size(output_size, _input.shape)
        if scales:
            scales_h, scales_w = scales

    if scales_h is None or scales_w is None:
        if len(inputs) == 5:
            # For torch==1.5.0, upsample_bilinear2d has 5 inputs.
            scales_h = inputs[3]
            scales_w = inputs[4]
        else:
            raise ValueError(f"Failed to infer scale factors from inputs.")

    upsample_bilinear = mb.upsample_bilinear(
        x=_input,
        scale_factor_height=scales_h,
        scale_factor_width=scales_w,
        align_corners=align_corners,
        name=node.name,
    )
    context.add(upsample_bilinear)


@register_torch_op
def upsample_nearest2d(context, node):
    inputs = _get_inputs(context, node)
    _input = inputs[0]
    scales_h, scales_w = None, None

    output_size = inputs[1]
    if output_size is None:
        # get scale factors from provided inputs
        scale_factors = inputs[2].val
        scales_h = scale_factors[0]
        scales_w = scale_factors[1]
    else:
        # infer scale factors from output sizes
        scales = _get_scales_from_output_size(output_size, _input.shape)
        if scales:
            scales_h, scales_w = scales

    if scales_h is None or scales_w is None:
        if len(inputs) == 5:
            # For torch==1.5.0, upsample_bilinear2d has 5 inputs.
            scales_h = inputs[3]
            scales_w = inputs[4]
        else:
            raise ValueError(f"Failed to infer scale factors from inputs.")

    upsample_nearest2d = mb.upsample_nearest_neighbor(
        x=_input,
        scale_factor_height=scales_h,
        scale_factor_width=scales_w,
        name=node.name,
    )
    context.add(upsample_nearest2d)


@register_torch_op(torch_alias=["listunpack"])
def tupleunpack(context, node):
    inputs = _get_inputs(context, node, expected=1)
    values = inputs[0]
    # Node input could have been turned into constant array in @tupleconstruct
    if not isinstance(values, tuple) and not isinstance(values, list):
        values = values.val
    if len(values) != len(node.outputs):
        raise ValueError(
            "unpack node expected {} outputs, got {}".format(
                len(node.outputs), len(values)
            )
        )
    assert len(values) == len(node.outputs)
    # @value is either a numpy primitive or a Var object
    for value, output in zip(values, node.outputs):
        if not isinstance(value, Var):
            value = _construct_constant(value, name=output)
        assert isinstance(value, Var)
        context.add(value, output)


@register_torch_op
def loop(context, node):
    """ In TorchIR, a loop looks like:
            %y_1, ..., %y_r = prim::Loop(%max_trip_count, %initial_condition, %x_1, ..., %x_r)
            block0(%i, %a_1, ..., %a_r):
                %b_1, ..., %b_m = some::node(%a_value_from_outer_block, %a_1)
                %iter_condition = some::other_node(%a_2)
                -> (%iter_condition, %b_1, ..., %b_r)

        This translates to pseudo code as:
            y_1, ..., y_r = x_1, ..., x_r
            condition = initial_condition
            i = 0
            while condition and i < max_trip_count:
                a_1, ..., a_r = y_1, ..., y_r

                ############################################################
                # Actual body of the loop
                b_1, ..., b_m = some::node(a_value_from_outside_of_the_loop, a_1)
                iter_condition = some::node(a_2)
                ############################################################

                y_1, ..., y_r = b_1, ..., b_r
                condition = iter_condition
                i += 1

        Which further translates to MIL while_loop as:
            loop_vars = (0, initial_condition, x_1, ..., x_r)
            _cond = {
                return (loop_vars[1] and loop_vars[0] < max_trip_count)
            }
            _body = {
                a_1, ..., a_r = loop_vars[2], ..., loop_vars[-1]
                b_1, ..., b_m = some::node(a_value_from_outside_of_the_loop, a_1)
                iter_condition = some::node(a_2)
                return (loop_vars[0] + 1, iter_condition, b_1, ..., b_r)
            }

        For loops pass True for %initial_condition and %iter_condition
        While loops set %max_trip_count to INT_MAX and %i is unused
    """
    name = node.name
    # inputs[0]: max iter count
    # inputs[1]: initial condition
    # inputs[2]: block input 0
    # ...
    # inputs[N+2]: block input N
    inputs = _get_inputs(context, node)
    max_iter_count = inputs[0]

    # Magic default signals this is a while-only loop, so no iteration count
    # is needed.
    has_iter_count = max_iter_count is not None

    # Create an interation count. This will only be used if this is a for loop.
    iter_count = mb.const(val=0, name=node.name + "_iter")
    # @loop_vars is tuple(iter_count, cond, inputs...)
    loop_vars = tuple([iter_count] + inputs[1:])

    def _loop_cond(*loop_vars):
        cond = loop_vars[1]

        # Check the iteration count if we're keeping track.
        if has_iter_count:
            iter_count = loop_vars[0]
            iter_cond = mb.less(
                x=iter_count, y=max_iter_count, name=node.name + "_cond"
            )
            return mb.logical_and(x=cond, y=iter_cond)
        else:
            return mb.identity(x=cond)

    def _shapes_are_equivalent(shape1, shape2):
        """ Compares two sets of tensor shapes and returns True if they are
            equivalent. That is, they are the same rank, and each dimension
            is the same or symbolic.
        """
        if len(shape1) != len(shape2):
            return False

        # Each dimension must have the same integer length, or else be
        # symbolic.
        all_equivalent = [
            s1 == s2 or (isinstance(s1, Symbol) and isinstance(s2, Symbol))
            for s1, s2 in zip(shape1, shape2)
        ]
        return all_equivalent

    def _loop_body(*loop_vars):
        block = node.blocks[0]
        iter_var = loop_vars[0]
        inputs = (iter_var,) + loop_vars[2:]
        res = convert_block(context, block, inputs)

        for input_var, output_var in zip(loop_vars[2:], res[1:]):
            if not _shapes_are_equivalent(input_var.shape, output_var.shape):
                _logging.warning(
                    "detected change in shape of loop variable. this could lead to incorrect inference results!"
                )
                _logging.warning(
                    "{}:{} -> {}:{}".format(
                        input_var.name,
                        input_var.shape,
                        output_var.name,
                        output_var.shape,
                    )
                )

        # Update the iteration count if we're keeping track.
        if has_iter_count:
            iter_var = mb.add(x=iter_var, y=1, name=iter_var.name + "_inc")
        else:
            iter_var = mb.identity(x=iter_var)

        # Must return tuple with same length and types as @loop_vars.
        return tuple([iter_var,] + res)

    loop = mb.while_loop(
        _cond=_loop_cond, _body=_loop_body, loop_vars=loop_vars, name=name
    )

    # Make sure the loop returned the expected number of outputs. Note that the
    # first two loop outputs are the iteration count and condition.
    assert len(loop) - 2 == len(node.outputs)
    for output_name, output_var in zip(node.outputs, loop[2:]):
        context.add(output_var, torch_name=output_name)


@register_torch_op(torch_alias=["if"])
def _if(context, node):
    """ In TorchIR, a conditional looks like:
            %y_1, ..., %y_r = prim::If(%condition)
            block0():  # TRUE BRANCH, never takes arguments, has to return r outputs
                %t_1, ..., %t_k = some::node(%a_value_from_outer_block)
                -> (%t_1, ..., %t_r)
            block1():  # FALSE BRANCH, never takes arguments, has to return r outputs
                %f_1, ..., %f_m = some::node(%a_value_from_outer_block)
                -> (%f_1, ..., %f_r)

        This translates to pseudo code as:
            if (condition):
                t_1, ..., t_k = some::node(a_value_from_outer_block)
                y_1, ..., y_r = t_1, ..., t_r
            else:
                f_1, ..., f_m = some::node(a_value_from_outer_block)
                y_1, ..., y_r = f_1, ..., f_r

        Which further translates to MIL cond as:
            _true = {
                t_1, ..., t_k = some::node(a_value_from_outer_block)
                return (t_1, ..., t_r)
            }
            _false = {
                f_1, ..., f_m = some::node(a_value_from_outer_block)
                return (f_1, ..., f_m)
            }
    """
    name = node.name
    # inputs[0]: condition
    inputs = _get_inputs(context, node, expected=1)
    condition = inputs[0]

    assert len(node.blocks) == 2
    true_block = node.blocks[0]
    false_block = node.blocks[1]

    def _true_path():
        res = convert_block(context, true_block, [])
        return tuple(res)

    def _false_path():
        res = convert_block(context, false_block, [])
        return tuple(res)

    cond = mb.cond(
        pred=condition, _true_fn=_true_path, _false_fn=_false_path, name=name
    )
    # If the condition only returns one item, wrap it in a tuple.
    if not isinstance(cond, tuple):
        cond = (cond,)

    # Make sure the condition returned the expected number of outputs.
    assert len(cond) == len(node.outputs)
    for output_name, output_var in zip(node.outputs, cond):
        context.add(output_var, torch_name=output_name)


@register_torch_op
def select(context, node):
    inputs = _get_inputs(context, node, expected=3)
    _input = inputs[0]
    dim = inputs[1].val
    index = inputs[2].val

    assert dim.shape == ()
    assert index.shape == ()
    assert _input.val is None

    # NOTE:
    # Each index in @begin_array/@end_array corresponds to a dimension of @_input
    # Each val of those arrays corresponds to the start/end index to slice in that dimension
    begin_array = [0] * len(_input.shape)
    begin_array[dim] = index
    end_array = [s if isinstance(s, int) else 0 for s in _input.shape]
    end_mask = [True] * len(_input.shape)
    if index != -1:
        end_array[dim] = index + 1
        end_mask[dim] = False

    slice_by_index = mb.slice_by_index(
        x=_input,
        begin=begin_array,
        end=end_array,
        end_mask=end_mask,
        name=node.name + "_slice_by_index",
    )
    # Now we squeeze the dimension we have selected from to remove it
    squeeze = mb.squeeze(
        x=slice_by_index, axes=_np.array([dim]), name=node.name + "_squeeze"
    )
    context.add(squeeze, node.name)


@register_torch_op
def ones(context, node):
    inputs = _get_inputs(context, node, expected=[5, 6])
    size = inputs[0]
    # dtype = NUM_TO_TORCH_DTYPE[inputs[1].val] unused
    # layout = inputs[2] unused
    # device = inputs[3] unused
    # requires_grad = inputs[4] unused
    # out = inputs[5] unused
    fill = mb.fill(shape=size, value=1.0, name=node.name)
    context.add(fill)


@register_torch_op
def ones_like(context, node):
    inputs = _get_inputs(context, node, expected=6)
    size = mb.shape(x=inputs[0])
    # dtype = NUM_TO_TORCH_DTYPE[inputs[1].val] unused
    # layout = inputs[2] unused
    # device = inputs[3] unused
    # requires_grad = inputs[4] unused
    # out = inputs[5] unused
    fill = mb.fill(shape=size, value=1.0, name=node.name)
    context.add(fill)


def _avg_pool(context, node, inputs):
    x = inputs[0]
    kernel_sizes = inputs[1]
    strides = inputs[2]
    if strides.op.op_type == "const"  and (not list(strides.val)):
        strides = mb.const(val=kernel_sizes.val, name=strides.name)
    pad_type = "custom"
    # Need to explicitly state L-R, T-B pad
    pad = inputs[3]
    pad = _np.repeat(pad.val, 2)
    ceil_mode = inputs[4]
    if ceil_mode.val is True and list(strides.val) != [1] * len(strides.val):
        # need to adjust padding values if ceil_mode is True
        # ceil_mode only causes any difference though, if the strides are not 1
        rank = len(pad) // 2
        x_spatial_dimensions = x.shape[-rank:]
        if any_symbolic(x_spatial_dimensions):
            raise ValueError("@avg_pool does not support symbolic input spatial shape when ceil_mode is True")
        pad = _adjust_pad_for_ceil_mode(
            x_spatial_dimensions, kernel_sizes.val, strides.val, pad
        )
    include_pad = inputs[5].val

    pool = mb.avg_pool(
        x=x,
        kernel_sizes=kernel_sizes,
        strides=strides,
        pad_type=pad_type,
        pad=pad,
        name=node.name,
        exclude_padding_from_average=not include_pad,
    )
    context.add(pool)


@register_torch_op
def avg_pool1d(context, node):
    inputs = _get_inputs(context, node, expected=6)
    _avg_pool(context, node, inputs)


@register_torch_op
def avg_pool2d(context, node):
    inputs = _get_inputs(context, node, expected=7)
    divisor_override = inputs[6]
    if divisor_override is not None:
        raise ValueError("divisor_override is not supported for avg_pool2d")
    _avg_pool(context, node, inputs)


@register_torch_op
def avg_pool3d(context, node):
    inputs = _get_inputs(context, node, expected=7)
    divisor_override = inputs[6]
    if divisor_override is not None:
        raise ValueError("divisor_override is not supported for avg_pool3d")
    _avg_pool(context, node, inputs)


@register_torch_op
def log_softmax(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    axis = inputs[1]
    out = inputs[2]  # Ignored.
    assert out is None
    res = mb.softmax(x=x, axis=axis, name=node.name + "_softmax")
    res = mb.log(x=res, name=node.name)
    context.add(res)


@register_torch_op
def sigmoid(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.sigmoid(x=inputs[0], name=node.name)
    context.add(res)

@register_torch_op
def hardsigmoid(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.sigmoid_hard(x=inputs[0], alpha=1.0/6, beta=0.5, name=node.name)
    context.add(res)

@register_torch_op
def gelu(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.gelu(x=inputs[0], name=node.name)
    context.add(res)


@register_torch_op(torch_alias=["slice"])
def _slice(context, node):
    inputs = _get_inputs(context, node, expected=5)
    x = inputs[0]
    dim = inputs[1].val

    if inputs[2] and inputs[2].val is not None:
        start = inputs[2].val
    elif isinstance(inputs[2], Var):
        start = inputs[2]
    else:
        start = 0

    if inputs[3] and inputs[3].val is not None:
        end = inputs[3].val
    elif isinstance(inputs[3], Var):
        end = inputs[3]
    else:
        end = None

    step = inputs[4].val

    if start == 0 and end is None and step == 1:
        # Handling x[:], just pass through the tensor.
        context.add(x, node.name)
        return

    begin_array = [0] * len(x.shape)
    begin_array[dim] = start
    end_array = [s if isinstance(s, int) else 0 for s in x.shape]
    end_mask = [True] * len(x.shape)
    if end is not None:
        end_array[dim] = end
        end_mask[dim] = False

    if isinstance(start, Var):
        begin_array = mb.concat(values=begin_array, axis=0)

    if isinstance(end, Var):
        end_array = mb.concat(values=end_array, axis=0)

    kwargs = {
        "x": x,
        "begin": begin_array,
        "end": end_array,
        "end_mask": end_mask,
        "name": node.name,
    }

    if step != 1:
        stride_array = _np.array([1] * len(x.shape))
        stride_array[dim] = step
        kwargs["stride"] = stride_array

    res = mb.slice_by_index(**kwargs)
    context.add(res)


@register_torch_op(torch_alias=["split_with_sizes"])
def split(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    split_sizes = inputs[1]
    dim = inputs[2].val

    if not isinstance(split_sizes.val, _np.ndarray):
        shape = mb.shape(x=x)
        dim_size = _list_select(shape, dim)
        # MIL split op needs the size of each split to be given explicitly.
        num_whole_splits = mb.floor_div(x=dim_size, y=split_sizes)
        remainder = mb.mod(x=dim_size, y=split_sizes)

        # MIL doesn't have a way of turning a scalar into a tensor (list write
        # only supports tensors). As a workaround, we create a constant [1]
        # tensor and multiply it by the scalar value, thus creating a tensor
        # with the scalar value in it.
        tmp = mb.const(val=[1])
        whole_sizes = mb.mul(x=tmp, y=split_sizes)
        reps = mb.mul(x=tmp, y=num_whole_splits)
        whole_sizes = mb.tile(x=whole_sizes, reps=reps)
        if remainder.val == 0:
            split_sizes = whole_sizes
        else:
            partial_size = mb.mul(x=tmp, y=remainder)
            split_sizes = mb.concat(values=[whole_sizes, partial_size], axis=0)
    res = mb.split(x=x, split_sizes=split_sizes, axis=dim, name=node.name)
    context.add(res, torch_name=node.name)


@register_torch_op
def unbind(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    dim = inputs[1].val
    split_sizes = [1]*x.shape[dim]
    res = mb.split(x=x, split_sizes=split_sizes, axis=dim, name=node.name)
    res = [mb.squeeze(x=x, axes=[dim]) for x in res]
    context.add(res, torch_name=node.name)

@register_torch_op
def to(context, node):
    # @non_blocking and @copy are unused
    inputs = _get_inputs(context, node)

    if len(inputs) == 8:
        _input = inputs[0]
        dtype = inputs[1].val
    elif len(inputs) == 7:
        _input = inputs[0]
        dtype = inputs[1].val
    elif len(inputs) == 6:
        _input = inputs[0]
        device = inputs[1]
        dtype = inputs[2].val
        # non_blocking = inputs[3]
        # copy = inputs[4]
        # memory_format = inputs[5] # usually None
    elif len(inputs) == 5:
        _input = inputs[0]
        dtype = NUMPY_DTYPE_TO_TORCH_NUM[inputs[1].val.dtype.type] if isinstance(inputs[1].val, _np.ndarray) else inputs[1].val
        # non_blocking = inputs[2]
        # copy = inputs[3]
        # memory_format = inputs[4]
    elif len(inputs) == 4:
        _input = inputs[0]
        dtype = inputs[1].val
        # non_blocking = inputs[2]
        # copy = inputs[3]
    elif len(inputs) == 3:
        # Since @non_blocking and @copy are unused, add back to context
        _input = inputs[0]
        # non_blocking = inputs[1]
        # copy = inputs[2]
        context.add(_input, torch_name=node.name)
        return
    else:
        raise ValueError(
            "Received invalid arguments for PyTorch conversion of op {}".format(node)
        )

    torch_dtype = NUM_TO_TORCH_DTYPE[dtype]
    if isinstance(_input, Var) and _input.val is not None:
        _input = _input.val
        # numpy -> torch -> torch cast -> numpy
        # This path is needed to use the mapping of passed in dtypes to torch dtypes.
        casted_input = torch.tensor(_input).type(torch_dtype).cpu().numpy()
        res = mb.const(mode="immediate_value", val=casted_input, name=node.name)
    else:
        res = mb.cast(x=_input, dtype=NUM_TO_DTYPE_STRING[dtype], name=node.name)
    context.add(res)


@register_torch_op
def erf(context, node):
    inputs = _get_inputs(context, node, expected=1)
    _input = inputs[0]
    erf = mb.erf(x=_input, name=node.name)
    context.add(erf)


@register_torch_op(torch_alias=["scalarimplicit"])
def implicittensortonum(context, node):
    inputs = _get_inputs(context, node, expected=1)
    _input = inputs[0]

    if _input.shape == (): #already a scalar
        context.add(_input, node.name)
    else:
        assert _input.shape == (1,)
        # shape: (1,) -> ()
        squeeze = mb.squeeze(x=_input, name=node.name)
        context.add(squeeze)


@register_torch_op
def constantchunk(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    # ConstantChunk gets its parameters as attributes of the node.
    chunks = node.attr["chunks"]
    dim = node.attr["dim"]

    total = x.shape[dim]
    size = int(_math.ceil(float(total) / float(chunks)))
    split_sizes = [size] * int(_math.floor(total / size))
    remainder = total - sum(split_sizes)
    if remainder > 0:
        split_sizes.append(remainder)

    res = mb.split(x=x, split_sizes=split_sizes, axis=dim, name=node.name)
    for val, name in zip(res, node.outputs):
        context.add(val, name)


def _expand(context, name, tensor, shape):
    reps = [ds if ds > 0 and ts == 1 else 1 for ts, ds in zip(tensor.shape, shape)]
    res = mb.tile(x=tensor, reps=reps, name=name)
    context.add(res)


@register_torch_op
def expand(context, node):
    # PyTorch 1.6+ has 3 inputs while older version has 2
    inputs = _get_inputs(context, node, expected=[2, 3])

    x = inputs[0]
    shape = inputs[1].val

    _expand(context, node.name, x, shape)


@register_torch_op
def expand_as(context, node):
    # PyTorch 1.6+ has 3 inputs while older version has 2
    inputs = _get_inputs(context, node, expected=[2, 3])
    x = inputs[0]
    other = inputs[1]

    _expand(context, node.name, x, other.shape)


@register_torch_op
def arange(context, node):
    inputs = _get_inputs(context, node)
    # dtype = inputs[-4]
    # layout = inputs[-3]
    # device = inputs[-2]
    # pin_memory = inputs[-1]
    if len(inputs) == 5:
        # inputs are [end, dtype, layout, device, pin_memory]
        start = 0
        end = inputs[0]
        step = 1
    elif len(inputs) == 6:
        # inputs are [start, end, dtype, layout, device, pin_memory]
        start = inputs[0]
        end = inputs[1]
        step = 1
    elif len(inputs) == 7:
        # inputs are [start, end, step, dtype, layout, device, pin_memory]
        start = inputs[0]
        end = inputs[1]
        step = inputs[2]
    else:
        raise ValueError(
            "arange must have exactly 5, 6, or 7 inputs, got {}".format(len(inputs))
        )

    res = mb.range_1d(start=start, end=end, step=step, name=node.name)
    context.add(res)


@register_torch_op(torch_alias=["masked_fill_"])
def masked_fill(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    mask = inputs[1]
    value = inputs[2]
    # @mb.select does not properly broadcast scalar input, so as a workaround
    # we create a full sized tensor.
    # rdar://61463562

    if types.is_int(value.dtype):
        # @mb.fill cannot handle value with dtype integer
        # so we cast the value.
        value = mb.cast(x=value, dtype="fp32")
    shape = mb.shape(x=x, name=node.name + "_shape")
    value = mb.fill(shape=shape, value=value, name=node.name + "_value")
    res = mb.select(cond=mask, a=value, b=x, name=node.name)
    context.add(res)


@register_torch_op
def meshgrid(context, node):
    """
    For N input tensors, a meshgrid is constructed by viewing each tensor as an N-dimension tensor
    with values in the dimension corresponding it its order in the args. (a.)
    Then, it is expanded along dimensions corresponding to the dimensions of each
    1d tensor in the order that they were passed in. (b.)

    Each output tensor is put into a tuple that is returned. These tuples form
    N, N-dimenional grids, where the ith grid is defined as expanding the ith input over
    dimensions defined by the other inputs.
    """
    inputs = _get_inputs(context, node)
    if len(inputs) < 2:
        raise ValueError("Requires > 2 tensor inputs.")

    # scalar inputs will be considered 1d tensors
    tensor_inputs = []
    for tensor_var in inputs:
        if not isinstance(tensor_var.val, _np.ndarray):
            tensor_inputs.append(_np.array(tensor_var.val))
        else:
            tensor_inputs.append(_np.array(tensor_var))

    if any([len(tensor_var.shape) > 1 for tensor_var in inputs]):
        raise ValueError("meshgrid recieved non-1d tensor.")

    dim_tuple = tuple(tensor_var.shape[0] for tensor_var in inputs)

    grids = []
    size = len(inputs)
    for i in range(size):
        view_shape = [1] * size
        view_shape[i] = -1
        view_shape = tuple(view_shape)
        tensor = torch.tensor(inputs[i].val).cpu()
        # (a.) in docstring
        view = mb.reshape(
            x=inputs[i], shape=view_shape, name=node.name + "_view_" + str(i)
        )

        # (b.) in docstring
        reps = [
            ds if ds > 0 and ts == 1 else 1 for ts, ds in zip(view.shape, dim_tuple)
        ]
        expand = mb.tile(x=view, reps=reps, name=node.name + "_expand_" + str(i))
        grids.append(expand)

    context.add(tuple(grids), node.name)


# Defines all the nodes that are noOps
@register_torch_op(
    torch_alias=[
        "dropout",
        "dropout_",
        "feature_dropout",
        "contiguous",
        "device",
        "detach",
        "clone",
    ]
)
def noop(context, node):
    _logging.info("Setting pytorch op: {} to no-op.".format(node))
    inputs = _get_inputs(context, node)
    _input = inputs[0]
    context.add(_input, torch_name=node.name)


@register_torch_op
def argmax(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    axis = inputs[1]
    keep_dims = inputs[2]
    res = mb.reduce_argmax(x=x, axis=axis, keep_dims=keep_dims, name=node.name)
    context.add(res)


@register_torch_op
def zeros(context, node):
    inputs = _get_inputs(context, node, expected=5)
    size = inputs[0].val
    dtype = inputs[1].val
    # layout = inputs[2] unused
    # device = inputs[3] unused
    # pin_memory = inputs[4] unused

    torch_dtype = NUM_TO_TORCH_DTYPE[dtype]
    zeros_array = torch.zeros(tuple(size)).type(torch_dtype).numpy()
    const = mb.const(mode="immediate_value", val=zeros_array, name=node.name)
    context.add(const)


@register_torch_op
def max(context, node):
    inputs = _get_inputs(context, node, expected=3)
    _input = inputs[0]
    dim = inputs[1].val
    keepdim = inputs[2].val

    values = mb.reduce_max(x=_input, axes=[dim], keep_dims=keepdim)
    indices = mb.reduce_argmax(x=_input, axis=dim, keep_dims=keepdim)
    assert len(node.outputs) == 2
    values_name = node.outputs[0]
    indices_name = node.outputs[1]
    context.add(values, torch_name=values_name)
    context.add(indices, torch_name=indices_name)


@register_torch_op
def argsort(context, node):
    inputs = _get_inputs(context, node, expected=3)
    ascending = mb.logical_not(x=inputs[2])
    argsort = mb.argsort(x=inputs[0], axis=inputs[1], ascending=ascending, name=node.name)
    context.add(argsort)


@register_torch_op
def sort(context, node):
    inputs = _get_inputs(context, node)
    _input = inputs[0]
    axis = inputs[1].val
    descending = inputs[2].val
    # NOTE: This is actually descending
    # rdar://62901267 (argsort ascending is actually descending)
    indices = mb.argsort(x=_input, axis=axis, ascending=descending)
    values = mb.gather_along_axis(x=_input, indices=indices, axis=axis)

    values_name = node.outputs[0]
    indices_name = node.outputs[1]
    context.add(values, torch_name=values_name)
    context.add(indices, torch_name=indices_name)


@register_torch_op
def append(context, node):
    # Note: by applying torchir_passes.transform_inplace_ops the meaning of
    # this op is changed from the original TorchIR. This op expects a python
    # list or MIL List as its first input. If an MIL List, the second input
    # must be a tensor of whatever shape the List expects. If not an MIL List,
    # the second input can by anything. The result will be the second input
    # joined to the first input, either by list_write if an MIL list, or
    # append if a python list.
    inputs = _get_inputs(context, node, expected=2)
    ls = inputs[0]
    value = inputs[1]

    if isinstance(ls, list):
        context.add(ls + [value], node.name)
    elif isinstance(ls, ListVar):
        index = mb.list_length(ls=ls, name=node.name + "_index")
        res = mb.list_write(ls=ls, index=index, value=value, name=node.name)
        context.add(res)
    else:
        raise ValueError("can only append to Python list or MIL ListVar, got {}.".format(type(inputs[0])))


@register_torch_op
def gather(context, node):
    inputs = _get_inputs(context, node)
    res = mb.gather_along_axis(x=inputs[0], indices=inputs[2], axis=inputs[1], name=node.name)
    context.add(res)

@register_torch_op
def index_select(context, node):
    x = context[node.inputs[0]]
    axis = context[node.inputs[1]]
    indices = context[node.inputs[2]]
    context.add(mb.gather(x=x, indices=indices, axis=axis, name=node.name))

@register_torch_op(torch_alias=["abs"])
def _abs(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.abs(x=inputs[0], name=node.name))

@register_torch_op
def repeat(context, node):
    x = context[node.inputs[0]]
    reps = context[node.inputs[1]]
    context.add(mb.tile(x=x, reps=reps, name=node.name))

@register_torch_op
def acos(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.acos(x=inputs[0], name=node.name))

@register_torch_op
def acosh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.acosh(x=inputs[0], name=node.name))

@register_torch_op
def asin(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.asin(x=inputs[0], name=node.name))

@register_torch_op
def atan(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.atan(x=inputs[0], name=node.name))

@register_torch_op
def atanh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.atanh(x=inputs[0], name=node.name))

@register_torch_op
def ceil(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.ceil(x=inputs[0], name=node.name))

@register_torch_op
def clamp(context, node):
    inputs = _get_inputs(context, node, expected=3)
    min_val = inputs[1] if inputs[1] else _np.finfo(_np.float32).min
    max_val = inputs[2] if inputs[2] else _np.finfo(_np.float32).max
    context.add(mb.clip(x=inputs[0], alpha=min_val, beta=max_val, name=node.name))


@register_torch_op
def cos(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.cos(x=inputs[0], name=node.name))

@register_torch_op
def cosh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.cosh(x=inputs[0], name=node.name))

@register_torch_op
def exp(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.exp(x=inputs[0], name=node.name))

@register_torch_op
def exp2(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.exp2(x=inputs[0], name=node.name))

@register_torch_op
def floor(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.floor(x=inputs[0], name=node.name))

@register_torch_op
def reciprocal(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.inverse(x=inputs[0], name=node.name))

@register_torch_op
def log(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.log(x=inputs[0], name=node.name))

@register_torch_op(torch_alias=["round"])
def _round(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.round(x=inputs[0], name=node.name))

@register_torch_op
def rsqrt(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.rsqrt(x=inputs[0], name=node.name))

@register_torch_op
def sin(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.sin(x=inputs[0], name=node.name))

@register_torch_op
def sinh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.sinh(x=inputs[0], name=node.name))

@register_torch_op
def asinh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.asinh(x=inputs[0], name=node.name))

@register_torch_op
def sqrt(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.sqrt(x=inputs[0], name=node.name))

@register_torch_op
def square(context, node):
    inputs = _get_inputs(context, node, expected=1)
    # mb.square is not supported in some backend
    context.add(mb.mul(x=inputs[0], y=inputs[0], name=node.name))

@register_torch_op
def tan(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.tan(x=inputs[0], name=node.name))

@register_torch_op
def tanh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.tanh(x=inputs[0], name=node.name))

@register_torch_op
def threshold(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    alpha = inputs[1]
    threshold_val = inputs[2]

    # Simple case (threshold_val == alpha)
    if alpha.val == threshold_val.val:
        threshold_node = mb.threshold(x=x, alpha=alpha, name=node.name)
        context.add(threshold_node)
        return

    # Complex case (threshold_val != threshold)
    threshold_node = mb.threshold(x=x, alpha=alpha, name=node.name + '_threshold')
    context.add(threshold_node)

    gt_node = mb.greater_equal(x=alpha, y=x, name=node.name + '_ge')
    context.add(gt_node)
    gt_node_32 = mb.cast(x=gt_node, dtype="fp32", name=node.name + '_ge32')

    mul_node = mb.linear_activation(x=gt_node_32, alpha=float(threshold_val.val - alpha.val),
                                    name=node.name + '_mul')
    context.add(mul_node)

    final_node = mb.add(x=mul_node, y=threshold_node, name=node.name)
    context.add(final_node)

@register_torch_op
def sign(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.sign(x=inputs[0], name=node.name))

@register_torch_op
def is_floating_point(context, node):
    inputs = _get_inputs(context, node, expected=1)
    is_float = types.is_float(inputs[0].dtype)
    context.add(mb.const(val=is_float, name=node.name))

@register_torch_op
def where(context, node):
    inputs = _get_inputs(context, node, expected=3)
    context.add(mb.select(cond=inputs[0], a=inputs[1], b=inputs[2], name=node.name))

@register_torch_op
def neg(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.mul(x=inputs[0], y=-1, name=node.name))


@register_torch_op
def topk(context, node):
    inputs = _get_inputs(context, node)
    kwargs = {"name": node.name, "x": inputs[0], "k": inputs[1]}

    if len(inputs) > 6:
        raise Exception("Number of inputs to topk exceeds 6")
    # optional: @axis
    if len(inputs) > 2:
        if inputs[2] is not None:
            kwargs["axis"] = inputs[2].val

    # optional: @ascending
    if len(inputs) > 3:
        largest = inputs[3].val
        kwargs["ascending"] = not largest

    # last inputs to topk are optional - sorted and out.
    if len(inputs) > 4:
        if inputs[4].val is False:
            raise Exception("Unsupported value for argument 'sorted' in topk. Supported values: True, but input "
                            "is {}".format(inputs[4].val))
    if len(inputs) > 5:
        if inputs[5] is not None:
            raise Exception("Unsupported value for argument 'out' in topk. Supported values: None, but input "
                            "is {}".format(inputs[5].val))

    res = mb.topk(**kwargs)
    values_name = node.outputs[0]
    indices_name = node.outputs[1]
    context.add(res[0], torch_name=values_name)
    context.add(res[1], torch_name=indices_name)



def _std(x,axes,keep_dim,unbiased,eps):
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
        rescale_factor = _np.sqrt(N/float(N-1))

    x_mean = mb.reduce_mean(x=x, axes=axes, keep_dims=True)
    x_demeaned = mb.sub(x=x, y=x_mean)
    x_demeaned_square = mb.square(x=x_demeaned)
    x_demeaned_square_mean = mb.reduce_mean(x=x_demeaned_square, axes=axes, keep_dims=keep_dim)
    if eps>0:
        x_demeaned_square_mean = mb.add(x=x_demeaned_square_mean,y=eps)
    if need_rescale:
        y_before_scale = mb.sqrt(x=x_demeaned_square_mean)
        y = mb.mul(x=y_before_scale, y=rescale_factor)
    else:
        y = mb.sqrt(x=x_demeaned_square_mean)
    return y

@register_torch_op
def std(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    if not (len(inputs) == 2 or len(inputs) == 4):
        raise ValueError("Number of inputs to the 'std' op must be"
                         "2 or 4")

    keep_dim = False
    axes = None
    if len(inputs) == 2:
        unbiased = inputs[1].val
    if len(inputs) == 4:
        axes = inputs[1].val
        if isinstance(axes, int):
            axes = [axes]
        unbiased = inputs[2].val
        keep_dim = inputs[3].val

    y = _std(x,axes,keep_dim,unbiased,0)
    context.add(y,node.name)

@register_torch_op
def copy_(context, node):
    inputs = _get_inputs(context, node, expected=3)
    context.add(mb.identity(x=inputs[0], name=node.name))

@register_torch_op
def dtype(context, node):
    inputs = _get_inputs(context, node, expected=1)
    dtype_str = inputs[0].dtype.__name__
    context.add(mb.const(val=dtype_str, name=node.name))

@register_torch_op
def tensor(context, node):
    inputs = _get_inputs(context, node, expected=4)
    val = inputs[0].val # element val to fill
    msg_prefix = 'torch::tensor {} '.format(node.name)
    if val is None:
        raise ValueError(msg_prefix + 'val is None')
    dtype_str = inputs[1].val
    if dtype_str != 'fp32':
        raise NotImplementedError(msg_prefix + \
            'Unsupported dtype: {}'.format(dtype_str))
    # inputs[3] is a bool (not sure what it is)
    shape = mb.shape(x=inputs[2], name=node.name+"_shape")
    context.add(mb.fill(shape=shape, value=val, name=node.name))

@register_torch_op
def log10(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    log_x = mb.log(x=x)
    context.add(mb.mul(x=log_x, y=1/_np.log(10.0)), node.name)

@register_torch_op
def reflection_pad2d(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    torch_pad = inputs[1].val
    pad_flipped = torch_pad.reshape((-1, 2))[::-1].ravel()
    pad = _np.pad(pad_flipped, (len(x.shape) * 2 - len(pad_flipped), 0))
    context.add(mb.pad(x=x, pad=pad, mode='reflect'), node.name)


@register_torch_op
def reflection_pad1d(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    torch_pad = inputs[1].val
    pad_flipped = torch_pad.reshape((-1, 2))[::-1].ravel()
    pad = _np.pad(pad_flipped, (len(x.shape) * 2 - len(pad_flipped), 0))
    context.add(mb.pad(x=x, pad=pad, mode='reflect'), node.name)


@register_torch_op
def replication_pad2d(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    torch_pad = inputs[1].val
    pad_flipped = torch_pad.reshape((-1, 2))[::-1].ravel()
    pad = _np.pad(pad_flipped, (len(x.shape) * 2 - len(pad_flipped), 0))
    context.add(mb.pad(x=x, pad=pad, mode='replicate'), node.name)


@register_torch_op
def replication_pad1d(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    torch_pad = inputs[1].val
    pad_flipped = torch_pad.reshape((-1, 2))[::-1].ravel()
    pad = _np.pad(pad_flipped, (len(x.shape) * 2 - len(pad_flipped), 0))
    context.add(mb.pad(x=x, pad=pad, mode='replicate'), node.name)

