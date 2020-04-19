import logging

import math
import numpy as np
import torch

from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.program.var import Var
from coremltools.converters.nnv2.nnv2_program.program.program import Placeholder
from .internal_graph import *
from .torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op

# The pytorch args for many of the below ops were sourced from
# https://github.com/pytorch/pytorch/blob/d971007c291c0ead1003d12cd553d18ddb582207/torch/csrc/jit/mobile/register_mobile_ops.cpp#L216


# This is a magic number in PyTorch. It's used as a default value in many
# functions.
PYTORCH_MAGIC_DEFAULT = 9223372036854775807


def convert_nodes(context, graph):
    """Iterate over the nodes of a graph or block and convert to NNv2.

        Arguments:
            context: A TranscriptionContext object to pull node inputs and
                assign node outputs.
            graph: An InternalTorchIRGraph or InternalTorchIRBlock object.
    """
    for node in graph.nodes:
        _add_op = _TORCH_OPS_REGISTRY.get(node.kind, None)
        logging.debug("Converting op {}".format(node.kind))
        if _add_op is None:
            raise RuntimeError(
                "Pytorch convert function for op {} not implemented".format(node.kind)
            )
        else:
            _add_op(context, node)


def convert_block(context, block, inputs):
    """Convert a block (sub-graph) to NNv2. Conversion happens within a new
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


# Some ops will recieve a dtype input as an integer
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


def decide_immediate_or_file(val):
    if val is not None and isinstance(val, (np.ndarray, np.generic)) and val.size >= 10:
        return "file_value"
    return "immediate_value"


def _get_inputs(context, node, expected=None):
    """Look up a node's inputs in @context and return them as a list. If
        @expected is not None, also verifies the number of inputs matches the
        value of @expcted.
    """
    inputs = [context[name] for name in node.inputs]
    if expected is not None and len(inputs) != expected:
        raise ValueError(
            "node {} ({}) got {} input(s), expected {}".format(
                node.name, node.kind, len(inputs), expected
            )
        )
    return inputs


def _construct_constant(val, name):
    # Converter cannot handle torch tensors.
    if isinstance(val, torch.Tensor):
        val = val.numpy()

    # NNv2 casts ints to int32, which can't represent the 64 bit magic number.
    # So we instead represent it with None, and any ops that might get the
    # value will check for None instead.
    if isinstance(val, int) and val == PYTORCH_MAGIC_DEFAULT:
        val = None

    mode = decide_immediate_or_file(val)
    if val is None:
        return None
    else:
        return cb.const(mode=mode, val=val, name=name)


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
    constant_inputs = [inp for inp in inputs if inp.val is not None]

    if len(constant_inputs) == len(inputs):
        # All the list items are compile-time constants, so let's create a new
        # const that concatenates them.
        mode = "immediate_value"
        val = array_type([inp.val for inp in inputs])
        const = cb.const(mode=mode, val=val, name=node.name)
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
    equal_to = cb.equal(x=inputs[0], y=inputs[1], name=node.name)
    context.add(equal_to)


@register_torch_op
def le(context, node):
    inputs = _get_inputs(context, node, expected=2)
    less_equal = cb.less_equal(x=inputs[0], y=inputs[1], name=node.name)
    context.add(less_equal)


@register_torch_op
def lt(context, node):
    inputs = _get_inputs(context, node, expected=2)
    less = cb.less(x=inputs[0], y=inputs[1], name=node.name)
    context.add(less)


@register_torch_op
def ge(context, node):
    inputs = _get_inputs(context, node, expected=2)
    greater_equal = cb.greater_equal(x=inputs[0], y=inputs[1], name=node.name)
    context.add(greater_equal)


@register_torch_op
def gt(context, node):
    inputs = _get_inputs(context, node, expected=2)
    greater = cb.greater(x=inputs[0], y=inputs[1], name=node.name)
    context.add(greater)


@register_torch_op(torch_alias=["t", "transpose_"])
def transpose(context, node):
    assert len(node.outputs) == 1
    inputs = _get_inputs(context, node)
    x = inputs[0]

    if len(node.inputs) == 1:
        # PyTorch has several tranpose ops that can be emitted. This one is only
        # emitted when .t() is called on a tensor, which means it can only be
        # called on a matrix.
        if len(x.shape) > 2:
            raise ValueError("transpose without dims for rank > 2 is unsupported")
        res = cb.transpose(x=x, perm=[1, 0], name=node.name)
    else:
        assert len(inputs) == 3
        ax0 = inputs[1].val
        ax1 = inputs[2].val

        perm = list(range(len(x.shape)))
        perm[ax0] = ax1
        perm[ax1] = ax0

        res = cb.transpose(x=x, perm=perm, name=node.name)
    context.add(res)


@register_torch_op
def permute(context, node):
    inputs = _get_inputs(context, node, expected=2)
    perm = cb.transpose(x=inputs[0], perm=inputs[1], name=node.name)
    context.add(perm)


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

    # Expand padding. Torch accepts either an int (for all dimensions) or an n-tuple of ints (one per dimension), but
    # we require a (2 * n)-tuple, where n is the number of spatial dimensions, start and end for each spatial dimension
    pad = inputs[4]
    if weight.val.ndim in (3, 4):
        # 1D and 2D: Need to explicitly state L-R, T-B pad
        pad = np.repeat(pad.val, 2)
    elif weight.val.ndim == 5:
        # 3D: Need to explicitly state F-Bk, L-R, T-B pad
        if type(pad.val) == int:
            pad = np.repeat(pad.val, 6)
        elif len(pad.val) == 3:
            pad = np.repeat(pad.val, 2)
    else:
        raise ValueError(
            "Invalid weight dimension. Must be 3, 4, or 5 for 1D, 2D, or 3D convolution, respectively."
        )

    dilations = inputs[5]
    transposed = inputs[6]
    out_pad = inputs[7]  # unused
    group = inputs[8]

    if any([v != 0 for v in out_pad.val]):
        raise ValueError(
            "convolution does not support output_padding (given {})".format(out_pad)
        )

    kwargs = {
        "x": x,
        "strides": strides,
        "pad_type": "custom",
        "pad": pad,
        "dilations": dilations,
        "group": group,
        "name": node.name,
    }

    if transposed.val is True:
        # Transposed convolution

        # PyTorch weight ordering [Cin, Cout, H, W]
        # NNv2 expects [H, W, Cout, Cin]
        weight_transpose = cb.transpose(
            x=weight, perm=[2, 3, 1, 0], name=weight.name + "_transpose"
        )
        kwargs["weight"] = weight_transpose
        if bias is not None:
            kwargs["bias"] = bias
        conv = cb.conv_transpose(**kwargs)
    else:
        # Normal convolution

        kwargs["W"] = weight
        # Bias is optional in PyTorch's convolution.
        if bias is not None:
            kwargs["B"] = bias
        conv = cb.conv(**kwargs)

    context.add(conv)


@register_torch_op
def softmax(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    axis = inputs[1]
    softmax = cb.softmax(logit=x, axis=axis, name=node.name)
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
            # NNv2 pooling does not support ceil_mode natively, but we can
            # workaround by padding the input appropriately.
            # rdar://60634390
            logging.warn("padding adjusted to support ceil_mode=True")
            new_pad[2 * idx + 1] += stride - remainder

    return new_pad


@register_torch_op
def max_pool2d(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    kernel_sizes = inputs[1]
    strides = inputs[2]
    pad_type = "custom"

    # Need to explicity state L-R, T-B pad
    pad = inputs[3]
    pad = np.repeat(pad.val, 2)
    dilation = inputs[4].val
    ceil_mode = inputs[5].val
    if np.any(dilation > 1):
        # See: rdar://60633736 (Implement dilation for nnv2 op max_pool)
        raise ValueError("@max_pool2d does not support dilation > 1")
    if ceil_mode is True:
        pad = _adjust_pad_for_ceil_mode(
            x.shape[-2:], kernel_sizes.val, strides.val, pad
        )

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


@register_torch_op(torch_alias="rsub")
def sub(context, node):
    inputs = _get_inputs(context, node, expected=3)
    assert len(node.outputs) == 1

    # TODO (sberardi): 3rd param to aten::sub is a scale factor, need to handle that.
    # out=input-alpha x other
    # rdar://60175736
    if inputs[2].val != 1:
        raise ValueError("SUB does not support scale factor param")

    sub = cb.sub(x=inputs[0], y=inputs[1], name=node.name)
    context.add(sub)


@register_torch_op
def mean(context, node):
    inputs = _get_inputs(context, node)

    kwargs = {"x": inputs[0], "name": node.name}

    # @axes is optional, so omit if None.
    axes = inputs[1]
    if axes is not None:
        # @axes needs to be a list, but if only one axis was specified in the
        # model, it will be constructed as an int. Construct a new constant as a
        # list.
        if not isinstance(axes.val, np.ndarray):
            axes = cb.const(val=[axes.val], name=axes.name + "_list")
            context.add(axes)
        kwargs["axes"] = axes

    # @keep_dims is optional.
    if len(inputs) >= 3:
        keep_dims = inputs[2]
        kwargs["keep_dims"] = keep_dims

    # Last input to mean is an optional output tensor. We always expect this to
    # be None or absent.
    assert len(inputs) <= 3 or inputs[3] is None
    mean = cb.reduce_mean(**kwargs)
    context.add(mean)


@register_torch_op
def squeeze(context, node):
    inputs = _get_inputs(context, node)
    if len(inputs) == 1:
        squeeze = cb.squeeze(x=inputs[0], name=node.name)
    elif len(inputs) == 2:
        squeeze_dim = inputs[1].val
        squeeze = cb.squeeze(x=inputs[0], axes=(squeeze_dim,), name=node.name)
    context.add(squeeze)


@register_torch_op
def unsqueeze(context, node):
    inputs = _get_inputs(context, node, expected=2)
    unsqueeze = cb.expand_dims(x=inputs[0], axes=[inputs[1].val], name=node.name)
    context.add(unsqueeze)


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


@register_torch_op
def adaptive_avg_pool2d(context, node):
    inputs = _get_inputs(context, node, expected=2)

    _input = inputs[0]
    output_size = inputs[1].val
    assert isinstance(output_size, np.ndarray)
    output_size = tuple(output_size)

    if output_size == (1, 1):
        # Represent (1,1) output size via @reduce_mean
        # Assume channel first ordering, reduce the last two (HW) dims.
        axes = cb.const(val=[-2, -1], name=node.name + "_axes")
        keep_dims = cb.const(val=True, name=node.name + "_keep_dims")

        avg_pool = cb.reduce_mean(
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
        avg_pool = cb.avg_pool(
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
def embedding(context, node):
    inputs = _get_inputs(context, node)
    _input = inputs[0]
    indices = inputs[1]
    if len(inputs) > 2:
        logging.warning(
            "CoreML embedding (gather) layer does not support any "
            "inputs besides the weights and indices. Those given "
            "will be ignored."
        )
    # inputs skipped:
    #  padding_idx (2)
    #  scale_grad_by_freq (3)
    #  sparse (4)
    #  Changing the axis from 0 is not an option in torch, so we don't expose it
    gather = cb.gather(x=_input, indices=indices, name=node.name)
    context.add(gather)


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


@register_torch_op
def cat(context, node):
    inputs = _get_inputs(context, node)

    values = inputs[0]
    if len(values) == 1:
        # Only one item to "concatenate", so treat it as a no-OP. Otherwise,
        # NNv1 concatND layer will complain it only has one input.
        context.add(values[0], node.name)
        return

    if len(inputs) < 2:
        axis = 0
    else:
        axis = inputs[1]

    concat = cb.concat(values=values, axis=axis, name=node.name)
    context.add(concat)


@register_torch_op
def item(context, node):
    inputs = _get_inputs(context, node, expected=1)

    # Item is used to convert a rank 1 tensor to a scalar. NNv2 ops that
    # reduce already output a scalar, so this is a NOP. But to make sure,
    # check that it actually is a scalar.
    if inputs[0].shape != ():
        raise ValueError("expected input to be a scalar")
    context.add(inputs[0], torch_name=node.name)


@register_torch_op(torch_alias=["bool"])
def _bool(context, node):
    inputs = _get_inputs(context, node, expected=1)

    x = inputs[0]
    # TODO: this is a hack, we'll be able to use the cast op once it is
    # complete (rdar://problem/61168016)
    if x.val is not None and not isinstance(x.val, bool):
        x = cb.const(val=bool(x.val), name=node.name)
    context.add(x, node.name)


@register_torch_op(torch_alias=["int"])
def _int(context, node):
    inputs = _get_inputs(context, node, expected=1)

    x = inputs[0]
    # TODO: this is a hack, we'll be able to use the cast op once it is
    # complete (rdar://problem/61168016)
    if x.val is not None and not isinstance(x.val, int):
        x = cb.const(val=int(x.val), name=node.name)
    context.add(x, node.name)


@register_torch_op
def layer_norm(context, node):
    inputs = _get_inputs(context, node, expected=6)
    _input = inputs[0]
    normalized_shape = inputs[1]
    weight = inputs[2]
    bias = inputs[3]
    eps = inputs[4]
    # cudnn_enable = inputs[5] unused
    layer_norm = cb.layer_norm(
        x=_input,
        axes=normalized_shape,
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
    assert x.shape == ()
    res = cb.const(val=[x.val], name=node.name)
    context.add(res)


@register_torch_op
def upsample_bilinear2d(context, node):
    inputs = _get_inputs(context, node)
    _input = inputs[0]
    output_size = inputs[1]
    align_corners = bool(inputs[2].val)
    if len(inputs) == 5:
        scales_h = inputs[3]
        scales_w = inputs[4]
    elif len(inputs) == 3:
        scales_h = None
        scales_w = None
    else:
        raise ValueError(
            "Invalid number of args in Pytorch conversion op for node: {}".format(node)
        )

    assert output_size is None or scales_h is None

    if output_size:
        assert len(output_size.val) == 2
        # output size is computed using the formula
        # floor (scale * input_size) in Core ML (and Pytorch)
        # Thus, when computing the scales from the output size,
        # add a small positive constant to the output size,
        # to make sure that the floor formula results in the correct output
        # size and not 1 unit smaller, due to float precision issues
        # e.g. if output size = 34 and input size = 2, then scale will be
        # 17, which can get represented as 16.9999, resulting in an output size of 33
        # instead of 34, without this correction.
        scales_h = (output_size.val[0] + 1e-4) / float(_input.shape[-2])
        scales_w = (output_size.val[1] + 1e-4) / float(_input.shape[-1])

    upsample_bilinear = cb.upsample_bilinear(
        x=_input,
        scale_factor_height=scales_h,
        scale_factor_width=scales_w,
        align_corners=align_corners,
        name=node.name,
    )
    context.add(upsample_bilinear)


@register_torch_op(torch_alias=["listunpack"])
def tupleunpack(context, node):
    inputs = _get_inputs(context, node, expected=1)
    values = inputs[0]
    # Node input could have been turned into constant array in @tupleconstruct
    if not isinstance(values, tuple) and not isinstance(values, list):
        values = values.val
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

        Which further translates to NNv2 while_loop as:
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
    has_iter_count = max_iter_count is not None and max_iter_count.val >= 0

    # Create an interation count. This will only be used if this is a for loop.
    iter_count = cb.const(val=0, name=node.name + "_iter")
    # @loop_vars is tuple(iter_count, cond, inputs...)
    loop_vars = tuple([iter_count] + inputs[1:])

    def _loop_cond(*loop_vars):
        cond = loop_vars[1]

        # Check the iteration count if we're keeping track.
        if has_iter_count:
            iter_count = loop_vars[0]
            iter_cond = cb.less(
                x=iter_count, y=max_iter_count, name=node.name + "_cond"
            )
            return cb.logical_and(x=cond, y=iter_cond)
        else:
            return cb.identity(x=cond)

    def _loop_body(*loop_vars):
        block = node.blocks[0]
        iter_var = loop_vars[0]
        inputs = (iter_var,) + loop_vars[2:]
        res = convert_block(context, block, inputs)

        # Update the iteration count if we're keeping track.
        if has_iter_count:
            iter_var = cb.add(x=iter_var, y=1, name=iter_var.name + "_inc")
        else:
            iter_var = cb.identity(x=iter_var)

        # Must return tuple with same length and types as @loop_vars.
        return tuple([iter_var,] + res)

    loop = cb.while_loop(
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

        Which further translates to NNv2 cond as:
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

    cond = cb.cond(
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
    begin_array = np.array([0] * len(_input.shape))
    begin_array[dim] = index
    end_array = np.array(_input.shape)
    end_array[dim] = index + 1

    slice_by_index = cb.slice_by_index(
        x=_input, begin=begin_array, end=end_array, name=node.name + "_slice_by_index"
    )
    # Now we squeeze the dimension we have selected from to remove it
    squeeze = cb.squeeze(
        x=slice_by_index, axes=np.array([dim]), name=node.name + "_squeeze"
    )
    context.add(squeeze, node.name)


@register_torch_op
def ones(context, node):
    inputs = _get_inputs(context, node, expected=6)
    size = inputs[0]
    # dtype = NUM_TO_TORCH_DTYPE[inputs[1].val] unused
    # layout = inputs[2] unused
    # device = inputs[3] unused
    # requires_grad = inputs[4] unused
    # out = inputs[5] unused
    fill = cb.fill(shape=size, value=1.0, name=node.name)
    context.add(fill)


@register_torch_op
def ones_like(context, node):
    inputs = _get_inputs(context, node, expected=6)
    size = cb.shape(x=inputs[0])
    # dtype = NUM_TO_TORCH_DTYPE[inputs[1].val] unused
    # layout = inputs[2] unused
    # device = inputs[3] unused
    # requires_grad = inputs[4] unused
    # out = inputs[5] unused
    fill = cb.fill(shape=size, value=1.0, name=node.name)
    context.add(fill)


def _avg_pool(context, node, inputs):
    x = inputs[0]
    kernel_sizes = inputs[1]
    strides = inputs[2]
    pad_type = "custom"
    # Need to explicity state L-R, T-B pad
    pad = inputs[3]
    pad = np.repeat(pad.val, 2)
    ceil_mode = inputs[4]
    if ceil_mode.val is True:
        rank = len(pad) // 2
        pad = _adjust_pad_for_ceil_mode(
            x.shape[-rank:], kernel_sizes.val, strides.val, pad
        )
    include_pad = inputs[5].val

    pool = cb.avg_pool(
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
def log_softmax(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    axis = inputs[1]
    out = inputs[2]  # Ignored.
    assert out is None
    res = cb.softmax(logit=x, axis=axis, name=node.name + "_softmax")
    res = cb.log(x=res, name=node.name)
    context.add(res)


@register_torch_op
def sigmoid(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = cb.sigmoid(x=inputs[0], name=node.name)
    context.add(res)


@register_torch_op
def gelu(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = cb.gelu(x=inputs[0], name=node.name)
    context.add(res)


@register_torch_op(torch_alias=["slice"])
def _slice(context, node):
    inputs = _get_inputs(context, node, expected=5)
    x = inputs[0]
    dim = inputs[1].val
    start = inputs[2].val
    end = inputs[3].val if inputs[3] is not None else None
    step = inputs[4].val

    begin_array = np.array([0] * len(x.shape))
    begin_array[dim] = start
    end_array = np.array(x.shape)
    if end != None:
        end_array[dim] = end

    kwargs = {"x": x, "begin": begin_array, "end": end_array, "name": node.name}

    if step != 1:
        stride_array = np.array([1] * len(x.shape))
        stride_array[dim] = step
        kwargs["stride"] = stride_array

    res = cb.slice_by_index(**kwargs)
    context.add(res)


@register_torch_op(torch_alias=["split_with_sizes"])
def split(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    split_sizes = inputs[1]
    dim = inputs[2]

    if not isinstance(split_sizes.val, np.ndarray):
        # NNv2 needs the size of each split to be given explicitly.
        num_whole_splits = x.shape[dim.val] // split_sizes.val
        remainder = x.shape[dim.val] % split_sizes.val
        split_sizes = [split_sizes.val] * num_whole_splits
        if remainder > 0:
            split_sizes += [remainder]
    res = cb.split(x=x, split_sizes=split_sizes, axis=dim, name=node.name)
    context.add(res, torch_name=node.name)


@register_torch_op
def to(context, node):
    # @non_blocking and @copy are unused
    inputs = _get_inputs(context, node)
    if len(inputs) == 5:
        _input = inputs[0]
        device = inputs[1]
        dtype = inputs[2].val
        # non_blocking = inputs[3]
        # copy = inputs[4]
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
            "Received invalid arguments for Pytorch conversion of op {}".format(node)
        )

    torch_dtype = NUM_TO_TORCH_DTYPE[dtype]
    if isinstance(_input, Var):
        _input = _input.val

    # numpy -> torch -> torch cast -> numpy
    # This path is needed to use the mapping of passed in dtypes to torch dtypes.
    casted_input = torch.tensor(_input).type(torch_dtype).numpy()
    const = cb.const(mode="immediate_value", val=casted_input, name=node.name)
    context.add(const)


@register_torch_op
def floor(context, node):
    inputs = _get_inputs(context, node, expected=1)
    _input = inputs[0]
    floor = cb.floor(x=_input, name=node.name)
    context.add(floor)


@register_torch_op
def erf(context, node):
    inputs = _get_inputs(context, node, expected=1)
    _input = inputs[0]
    erf = cb.erf(x=_input, name=node.name)
    context.add(erf)


@register_torch_op
def implicittensortonum(context, node):
    inputs = _get_inputs(context, node, expected=1)
    _input = inputs[0]
    assert _input.shape == (1,)
    # shape: (1,) -> ()
    squeeze = cb.squeeze(x=_input, name=node.name)
    context.add(squeeze)


@register_torch_op
def constantchunk(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    # ConstantChunk gets its parameters as attributes of the node.
    chunks = node.attr["chunks"]
    dim = node.attr["dim"]

    total = x.shape[dim]
    size = int(math.ceil(float(total) / float(chunks)))
    split_sizes = [size] * int(math.floor(total / size))
    remainder = total - sum(split_sizes)
    if remainder > 0:
        split_sizes.append(remainder)

    res = cb.split(x=x, split_sizes=split_sizes, axis=dim, name=node.name)
    for val, name in zip(res, node.outputs):
        context.add(val, name)


def _expand(context, name, tensor, shape):
    reps = [ds if ds > 0 and ts == 1 else 1 for ts, ds in zip(tensor.shape, shape)]
    res = cb.tile(x=tensor, reps=reps, name=name)
    context.add(res)


@register_torch_op
def expand(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    shape = inputs[1].val

    _expand(context, node.name, x, shape)


@register_torch_op
def expand_as(context, node):
    inputs = _get_inputs(context, node, expected=2)
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
        start = inputs[0].val
        end = inputs[1].val
        step = 1
    elif len(inputs) == 7:
        # inputs are [start, end, step, dtype, layout, device, pin_memory]
        start = inputs[0].val
        end = inputs[1].val
        step = inputs[2].val
    else:
        raise ValueError(
            "arange must have exactly 5, 6, or 7 inputs, got {}".format(len(inputs))
        )

    res = cb.range_1d(start=start, end=end, step=step, name=node.name)
    context.add(res)


@register_torch_op(torch_alias=["masked_fill_"])
def masked_fill(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    mask = inputs[1]
    value = inputs[2]
    # cb.select does not properly broadcast scalar input, so as a workaround
    # we create a full sized tensor.
    # rdar://61463562
    value = cb.fill(shape=x.shape, value=value, name=node.name + "_value")
    res = cb.select(cond=mask, a=value, b=x, name=node.name)
    context.add(res)


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
        if not isinstance(tensor_var.val, np.ndarray):
            tensor_inputs.append(np.array(tensor_var.val))
        else:
            tensor_inputs.append(np.array(tensor_var))

    if any([len(tensor_var.shape) > 1 for tensor_var in inputs]):
        raise ValueError("meshgrid recieved non-1d tensor.")

    dim_tuple = tuple(tensor_var.shape[0] for tensor_var in inputs)

    grids = []
    size = len(inputs)
    for i in range(size):
        view_shape = [1] * size
        view_shape[i] = -1
        view_shape = tuple(view_shape)
        tensor = torch.tensor(inputs[i].val)
        # (a.) in docstring
        view = cb.reshape(
            x=inputs[i], shape=view_shape, name=node.name + "_view_" + str(i)
        )

        # (b.) in docstring
        reps = [
            ds if ds > 0 and ts == 1 else 1 for ts, ds in zip(view.shape, dim_tuple)
        ]
        expand = cb.tile(x=view, reps=reps, name=node.name + "_expand_" + str(i))
        grids.append(expand)

    context.add(tuple(grids), node.name)


@register_torch_op
def tanh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    _input = inputs[0]
    tanh = cb.tanh(x=_input, name=node.name)
    context.add(tanh)


# Defines all the nodes that are noOps
@register_torch_op(
    torch_alias=[
        "dropout",
        "dropout_",
        "feature_dropout",
        "contiguous",
        "device",
        "detach",
    ]
)
def noop(context, node):
    logging.warning("Setting pytorch op: {} to no-op.".format(node))
    inputs = _get_inputs(context, node)
    _input = inputs[0]
    context.add(_input, torch_name=node.name)
