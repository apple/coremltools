#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import builtins
import math as _math
import numbers
from collections.abc import Iterable

import numpy as _np
import paddle
from tqdm import tqdm as _tqdm

from coremltools import _logger as logger
from coremltools.converters.mil._deployment_compatibility import (
    AvailableTarget as target,
)
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Symbol, types
from coremltools.converters.mil.mil.block import (
    curr_opset_version,
    is_current_opset_version_compatible_with,
)
from coremltools.converters.mil.mil.ops.defs._utils import (
    MAX_SIZE_CONSTANT_FOLDING, promote_input_dtypes,
    solve_slice_by_index_shape)
from coremltools.converters.mil.mil.types import is_bool
from coremltools.converters.mil.mil.types.symbolic import (
    any_symbolic,
    is_compatible_symbolic_vector,
    is_symbolic,
)
from coremltools.converters.mil.mil.var import ListVar, Var

from .._utils import value_at, build_einsum_mil
from .paddle_op_registry import _PADDLE_OPS_REGISTRY, register_paddle_op


# The pypaddle args for many of the below ops were sourced from
# https://github.com/pypaddle/pypaddle/blob/d971007c291c0ead1003d12cd553d18ddb582207/paddle/csrc/jit/mobile/register_mobile_ops.cpp#L216


# This is a magic number in PyPaddle. It's used as a default value in many
# functions.
PYPADDLE_MAGIC_DEFAULT = 9223372036854775807
VALUE_CLOSE_TO_INFINITY = 1e+38


def _all_outputs_present(context, graph):
    """
    Returns true if all the symbols in the graph's output list are
    present in context.
    """
    for outp in graph.outputs:
        try:
            context[outp]
        except ValueError:
            return False
    return True


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
                # inputs = {}
                # outputs = {}
                # for ipt in op.input_names:
                #     inputs[ipt] = op.input(ipt)
                # for opt in op.output_names:
                #     outputs[opt] = op.output(opt)
                # node = self.make_node(op, inputs, outputs,
                op_lookup = op.type
                # if op_lookup.endswith("_"):
                #     # This is an "in place" op.
                #     # Look up the standard op instead by removing underscore.
                #     op_lookup = op_lookup[:-1]
                add_op = _PADDLE_OPS_REGISTRY.get(op_lookup, None)

                # logger.info("Converting op {} : {}".format(node.name, node.kind))
                if add_op is None:
                    raise RuntimeError(
                        "PyPaddle convert function for op '{}' not implemented.".format(node.kind)
                    )
                add_op(context, op)

                # We've generated all the outputs the graph needs, terminate conversion.
                # if _all_outputs_present(context, graph):
                #     break


    # for node in _tqdm(graph.nodes, desc="Converting PyPaddle Frontend ==> MIL Ops", unit=" ops"):
    #     op_lookup = node.kind
    #     if op_lookup.endswith("_"):
    #         # This is an "in place" op.
    #         # Look up the standard op instead by removing underscore.
    #         op_lookup = op_lookup[:-1]
    #     add_op = _PADDLE_OPS_REGISTRY.get(op_lookup, None)

    #     logger.info("Converting op {} : {}".format(node.name, node.kind))
    #     if add_op is None:
    #         raise RuntimeError(
    #             "PyPaddle convert function for op '{}' not implemented.".format(node.kind)
    #         )
    #     add_op(context, node)

    #     # We've generated all the outputs the graph needs, terminate conversion.
    #     if _all_outputs_present(context, graph):
    #         break


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


def _get_inputs(context, node, expected=None, min_expected=None):
    """
    Look up a node's inputs in @context and return them as a list. If
    @expected is not None, also verifies the number of inputs matches the
    value of @expected.
    """
    inputs = [context[name] for name in node.inputs]
    if expected is not None:
        expected = [expected] if not isinstance(expected, (list, tuple)) else expected

        if len(inputs) not in expected:
            raise ValueError(
                "node {} ({}) got {} input(s), expected {}".format(
                    node.name, node.kind, len(inputs), expected
                )
            )
    if min_expected is not None:
        if len(inputs) < min_expected:
            raise ValueError(
                "node {} ({}) got {} input(s), expected minimum {} inputs".format(
                    node.name, node.kind, len(inputs), min_expected
                )
            )

    return inputs


def _list_select(shape_var, index):
    """
    Sometimes we need to select a specific item from a list. If that item
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
    # Converter cannot handle paddle tensors.
    if isinstance(val, paddle.Tensor):
        val = val.cpu().numpy()

    # MIL casts ints to int32, which can't represent the 64 bit magic number.
    # So we instead represent it with None, and any ops that might get the
    # value will check for None instead.
    if isinstance(val, int) and val == PYPADDLE_MAGIC_DEFAULT:
        val = None

    # Pypaddle uses inf
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


@register_paddle_op
def affine_grid_generator(context, node):
    # rdar://73165386 (Improve error handling of coremltools "affine" op PyPaddle conversion.)

    affine_op_name = node.name
    theta, size, align_corners = _get_inputs(context, node, expected=3)

    # note: only add consts here as PyPaddle uses affine_grid + grid_sampler together
    is_theta_const = theta.val is not None
    if is_theta_const:
        context.add(mb.const(val=theta.val, name="{}_theta".format(affine_op_name)))
    else:  # theta is dynamic input, keep track of it's name
        context.add(mb.const(val=theta.name, name="{}_theta".format(affine_op_name)))

    context.add(mb.const(val=size.val, name="{}_size".format(affine_op_name)))
    context.add(mb.const(val=align_corners.val, name="{}_align_corners".format(affine_op_name)))


@register_paddle_op
def grid_sampler(context, node):
    affine_op_name = node.inputs[1]
    # https://github.com/pypaddle/pypaddle/blob/00d432a1ed179eff52a9d86a0630f623bf20a37a/aten/src/ATen/native/GridSampler.h#L10-L11
    m_mode = {0: "bilinear", 1: "nearest"}
    m_padding_mode = {0: "constant", 1: "border", 2: "reflection"}

    # add `resample` if grid/coordinates is in input, otherwise,
    # add `affine` to generate grid from `affine_grid_generator`.
    if affine_op_name in context:  # add `resample` op
        inputs = _get_inputs(context, node, expected=5)
        sampling_mode = m_mode[inputs[2].val]
        padding_mode = m_padding_mode[inputs[3].val]
        align_corners = inputs[4].val

        # When align_corners=False, padding_mode is corresponding to Core ML's symmetric
        if padding_mode == "reflection" and align_corners is False:
            padding_mode = "symmetric"

        x = mb.resample(
            x=inputs[0],
            coordinates=inputs[1],
            sampling_mode=sampling_mode,
            padding_mode=padding_mode,
            padding_value=0.0,
            coordinates_mode="normalized_minus_one_to_one",
            align_corners=align_corners,
            name=node.name,
        )
        context.add(x)
    else:  # add `affine` op instead
        x = context[node.inputs[0]]
        # inputs from `affine_grid_generator`
        affine_theta = context["{}_theta".format(affine_op_name)]
        affine_size = context["{}_size".format(affine_op_name)]
        affine_align_corners = context["{}_align_corners".format(affine_op_name)]

        # affine_theta.val is either name string (dynamic input) or np.ndarray (static values)
        # see `affine_grid_generator` for details.
        is_theta_const = not isinstance(affine_theta.val, str)
        if is_theta_const:
            transform_matrix = _np.reshape(affine_theta.val, (affine_theta.shape[0], 6))
        else:  # theta is dynamic input, add `reshape` op to PyMIL
            transform_matrix = mb.reshape(
                x=context[affine_theta.val],
                shape=(-1, 6),
                name=node.name + "_theta_reshape",
            )

        # inputs from `grid_sampler`
        sampling_mode = m_mode[context[node.inputs[2]].val]
        padding_mode = m_padding_mode[context[node.inputs[3]].val]
        align_corners = context[node.inputs[4]].val

        if sampling_mode != "bilinear":
            raise NotImplementedError("'sampling_mode' not supported.")

        if padding_mode != "constant":
            raise NotImplementedError("'padding_mode' not supported.")

        if affine_align_corners.val != align_corners:
            raise ValueError(
                "Op 'affine_grid_generator' and 'grid_sampler' must agree on 'align_corners'."
            )

        x = mb.affine(
            x=x,
            transform_matrix=transform_matrix,
            output_height=affine_size.val[2],
            output_width=affine_size.val[3],
            sampling_mode=sampling_mode,
            padding_mode=padding_mode,
            padding_value=0.0,
            coordinates_mode="normalized_minus_one_to_one",
            align_corners=align_corners,
            name=node.name,
        )
        context.add(x)


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
def cosine_similarity(context, node):
    inputs = _get_inputs(context, node, expected=4)
    dim = inputs[-2].val
    eps = inputs[-1].val
    xy = mb.mul(x=inputs[0], y=inputs[1])
    sum_xy = mb.reduce_sum(x=xy, axes=[dim])

    xx = mb.mul(x=inputs[0], y=inputs[0])
    sum_xx = mb.reduce_sum(x=xx, axes=[dim])
    yy = mb.mul(x=inputs[1], y=inputs[1])
    sum_yy = mb.reduce_sum(x=yy, axes=[dim])

    mul_sum_xy = mb.mul(x=sum_xx, y=sum_yy)
    div_12 = mb.maximum(x=mul_sum_xy, y=eps * eps)
    div_sqrt = mb.sqrt(x=div_12)

    cs = mb.real_div(x=sum_xy, y=div_sqrt, name=node.name)
    context.add(cs)


@register_paddle_op
def selu(context, node):
    ALPHA = 1.6732632423543772
    SCALE = 1.0507009873554805

    x = _get_inputs(context, node, expected=1)[0]
    x = mb.elu(x=x, alpha=ALPHA)
    x = mb.mul(x=x, y=SCALE, name=node.name)
    context.add(x)


@register_paddle_op
def dot(context, node):
    inputs = _get_inputs(context, node, expected=2)
    xy = mb.mul(x=inputs[0], y=inputs[1])
    sum_xy = mb.reduce_sum(x=xy, axes=[0])
    context.add(sum_xy, node.name)


@register_paddle_op
def mv(context, node):
    inputs = _get_inputs(context, node, expected=2)
    expand = mb.expand_dims(x=inputs[1], axes=[-1], name=node.name + "_expanded")
    mv = mb.matmul(x=inputs[0], y=expand, name=node.name + "_mv")
    res = mb.squeeze(x=mv, axes=[-1], name=node.name)
    context.add(res)


@register_paddle_op
def outer(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = mb.reshape(x=inputs[0], shape=[-1, 1])
    y = mb.reshape(x=inputs[1], shape=[1, -1])
    res = mb.matmul(x=x, y=y, name=node.name)
    context.add(res)


@register_paddle_op
def cross(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    y = inputs[1]
    dim = inputs[2]

    x1 = mb.gather(x=x, indices=[1, 2, 0], axis=dim, name="x1")
    x2 = mb.gather(x=x, indices=[2, 0, 1], axis=dim, name="x2")
    y1 = mb.gather(x=y, indices=[1, 2, 0], axis=dim, name="y1")
    y2 = mb.gather(x=y, indices=[2, 0, 1], axis=dim, name="y2")
    m1 = mb.mul(x=x1, y=y2)
    m2 = mb.mul(x=x2, y=y1)
    z = mb.sub(x=m1, y=m2, name=node.name)
    context.add(z)


@register_paddle_op
def frobenius_norm(context, node):
    x, dim, keep_dims = _get_inputs(context, node, expected=3)
    result = mb.reduce_l2_norm(x=x, axes=dim, keep_dims=keep_dims, name=node.name)
    context.add(result)


@register_paddle_op
def norm(context, node):
    x, num, dim, keep_dims = _get_inputs(context, node, expected=4)
    assert x is not None and keep_dims is not None and num is not None and dim is not None
    temp = _vector_norm(x=x, order=num, dim=dim, keep_dims=keep_dims, name=node.name)
    context.add(temp)


def _vector_norm(x, order, dim, keep_dims, name):
    if order.val == 0:
        # sum(x!=0)
        x = mb.cast(x=x, dtype="fp32")
        temp = mb.not_equal(x=x, y=0.)
        temp = mb.cast(x=temp, dtype='int32')
        temp = mb.reduce_sum(x=temp, axes=dim, keep_dims=keep_dims, name=name)
    elif order.val > VALUE_CLOSE_TO_INFINITY:
        # max(abs(x))
        temp = mb.abs(x=x)
        temp = mb.reduce_max(x=temp, axes=dim, keep_dims=keep_dims, name=name)
    elif order.val < -VALUE_CLOSE_TO_INFINITY:
        # min(abs(x))
        temp = mb.abs(x=x)
        temp = mb.reduce_min(x=temp, axes=dim, keep_dims=keep_dims, name=name)
    else:
        # sum(abs(x)^{order})^{(1 / order)}
        temp = mb.abs(x=x)
        x, y = promote_input_dtypes([temp, order.val])
        temp = mb.pow(x=x, y=y)
        temp = mb.reduce_sum(x=temp, axes=dim, keep_dims=keep_dims)
        temp = mb.pow(x=temp, y=1.0 / order.val, name=name)
    return temp
    
@register_paddle_op
def _weight_norm(context, node):
    v, g, dim = _get_inputs(context, node, expected=3)

    # Determine axes for L2 norm
    if dim.val == -1:
        axes = None
    else:
        axes = list(range(v.rank))
        dim = dim.val
        if dim >= 0:
            axes.remove(dim)
        else:
            axes.remove(v.rank + dim)

    # Calculate L2 norm of v
    temp = mb.pow(x=v, y=2.)
    temp = mb.reduce_sum(x=temp, axes=axes, keep_dims=True)
    norm = mb.pow(x=temp, y=1./2)

    inverse_norm = mb.inverse(x=norm)
    direction = mb.mul(x=v, y=inverse_norm)
    result = mb.mul(x=g, y=direction, name=node.name)
    context.add(result)



def _matrix_norm(x, order, dim, keep_dims, name):
    if order.val == 1:
        # min(sum(abs(x), dim=0))
        temp = mb.abs(x=x)
        temp = mb.reduce_sum(x=temp, axes=[dim[0]], keep_dims=True)
        temp = mb.reduce_max(x=temp, axes=dim, keep_dims=keep_dims, name=name)
    elif order.val == -1:
        # min(sum(abs(x), dim=0))
        temp = mb.abs(x=x)
        temp = mb.reduce_sum(x=temp, axes=[dim[0]], keep_dims=True)
        temp = mb.reduce_min(x=temp, axes=dim, keep_dims=keep_dims, name=name)
    elif order.val == "fro":
        # sum(x**2)**1/2
        temp = mb.reduce_l2_norm(x=x, axes=dim, keep_dims=keep_dims, name=name)
    elif order.val > VALUE_CLOSE_TO_INFINITY:
        # max(sum(abs(x), dim=1))
        temp = mb.abs(x=x)
        temp = mb.reduce_sum(x=temp, axes=[dim[1]], keep_dims=True)
        temp = mb.reduce_max(x=temp, axes=dim, keep_dims=keep_dims, name=name)
    elif order.val < -VALUE_CLOSE_TO_INFINITY:
        # min(sum(abs(x), dim=1))
        temp = mb.abs(x=x)
        temp = mb.reduce_sum(x=temp, axes=[dim[1]], keep_dims=True)
        temp = mb.reduce_min(x=temp, axes=dim, keep_dims=keep_dims, name=name)
    else:
        raise RuntimeError("Matrix norm is not defined for the current inputs")
    return temp


@register_paddle_op
def linalg_vector_norm(context, node):
    x, order, dim, keep_dims, _ = _get_inputs(context, node, expected=5)
    assert x is not None and keep_dims is not None and order is not None
    temp = _vector_norm(x=x, order=order, dim=dim, keep_dims=keep_dims, name=node.name)
    context.add(temp)


@register_paddle_op
def linalg_matrix_norm(context, node):
    x, order, dim, keep_dims, _ = _get_inputs(context, node, expected=5)
    assert x is not None and keep_dims is not None and order is not None and dim is not None
    assert len(dim.val) == 2
    temp = _matrix_norm(x=x, order=order, dim=dim.val, keep_dims=keep_dims, name=node.name)
    context.add(temp)


@register_paddle_op
def linalg_norm(context, node):
    x, order, dim, keep_dims, _ = _get_inputs(context, node, expected=5)
    assert x is not None and keep_dims is not None
    if dim is None:
        dim = _np.arange(x.rank)
    else:
        dim = dim.val
    if order is None:
        temp = mb.reduce_l2_norm(x=x, axes=dim, keep_dims=keep_dims, name=node.name)
    elif len(dim) == 2:
        temp = _matrix_norm(
            x=x, order=order, dim=dim, keep_dims=keep_dims, name=node.name
        )
    else:
        temp = _vector_norm(x=x, order=order, dim=dim, keep_dims=keep_dims, name=node.name)
    context.add(temp)


@register_paddle_op
def hardswish(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]

    w = mb.thresholded_relu(x=x, alpha=-3.0)
    y = mb.sigmoid_hard(
        x=w, alpha=1.0 / 6, beta=0.5
    )  # ``y = min(max(alpha * x + beta, -1), 1)
    result = mb.mul(x=w, y=y, name=node.name)

    context.add(result)
    

@register_paddle_op
def reshape_as(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    ref = inputs[1]
    shape = mb.shape(x=ref)
    result = mb.reshape(x=x, shape=shape, name=node.name)
    context.add(result)

@register_paddle_op
def shape(context, node):
    x = context[node.input("Input")[0]]
    output_name = node.output("Out")[0]
    shape = mb.shape(x=x, name=output_name)
    context.add(shape)



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
        val = array_type([inp.val for inp in inputs])
        const = mb.const(val=val, name=node.name)
        context.add(const)
    else:
        # If at least one input to the construct op is non-const, collect
        # the inputs and add them directly to the context. Ops that use this
        # node's output will take the list directly as input.
        context.add(array_type(inputs), node.name)


@register_paddle_op
def tupleconstruct(context, node):
    _array_construct(context, node, array_type=tuple)


@register_paddle_op
def listconstruct(context, node):
    _array_construct(context, node, array_type=list)


@register_paddle_op
def eq(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    y = inputs[1]
    if is_bool(x.dtype):
        x = mb.cast(x=x, dtype='int32')
    if is_bool(y.dtype):
        y = mb.cast(x=y, dtype='int32')
    x, y = promote_input_dtypes([x, y])
    equal_to = mb.equal(x=x, y=y, name=node.name)
    context.add(equal_to)


@register_paddle_op
def ne(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    y = inputs[1]
    if is_bool(x.dtype):
        x = mb.cast(x=x, dtype='int32')
    if is_bool(y.dtype):
        y = mb.cast(x=y, dtype='int32')
    equal_to = mb.not_equal(x=x, y=y, name=node.name)
    context.add(equal_to)


@register_paddle_op
def le(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = promote_input_dtypes(inputs)
    less_equal = mb.less_equal(x=x, y=y, name=node.name)
    context.add(less_equal)


@register_paddle_op
def lt(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = promote_input_dtypes(inputs)
    less = mb.less(x=x, y=y, name=node.name)
    context.add(less)


@register_paddle_op
def ge(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = promote_input_dtypes(inputs)
    greater_equal = mb.greater_equal(x=x, y=y, name=node.name)
    context.add(greater_equal)


@register_paddle_op
def gt(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = promote_input_dtypes(inputs[:2])
    greater = mb.greater(x=x, y=y, name=node.name)
    context.add(greater)


@register_paddle_op(paddle_alias=["t"])
def transpose(context, node):
    assert len(node.outputs) == 1
    inputs = _get_inputs(context, node)
    x = inputs[0]

    if len(node.inputs) == 1:
        # PyPaddle has several transpose ops that can be emitted. This one is only
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


@register_paddle_op(paddle_alias=["transpose2"])
def permute(context, node):
    x = context[node.input("X")[0]]
    axis = node.desc.attr("axis")
    output_name = node.output("Out")[0]
    perm = mb.transpose(x=x, perm=axis, name=output_name)
    context.add(perm)


@register_paddle_op
def frac(context, node):
    # Frac(x) = x - floor(abs(x)) * sign(x)

    x = _get_inputs(context, node, expected=1)[0]
    floor_abs = mb.floor(x=mb.abs(x=x))
    sign_abs_floor = mb.mul(x=floor_abs, y=mb.sign(x=x))
    res = mb.sub(x=x, y=sign_abs_floor)
    context.add(res, paddle_name=node.name)


@register_paddle_op
def pixel_shuffle(context, node):
    inputs = _get_inputs(context, node, expected=2)
    perm = mb.pixel_shuffle(x=inputs[0], upscale_factor=inputs[1], name=node.name)
    context.add(perm)


@register_paddle_op
def pixel_unshuffle(context, node):
    inputs = _get_inputs(context, node, expected=2)
    downscale_factor = _np.uint32(inputs[1].val)
    perm = mb.pixel_unshuffle(x=inputs[0], downscale_factor=downscale_factor, name=node.name)
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
    # add_inputs = _get_inputs(context, node)
    # assert len(node.outputs) == 1

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


@register_paddle_op
def cumsum(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    if is_bool(x.dtype):
        x = mb.cast(x=x, dtype='int32')
    res = mb.cumsum(x=x, axis=inputs[1], name=node.name)
    context.add(res)


@register_paddle_op
def addmm(context, node):
    # addmm(Tensor input, Tensor mat1, Tensor mat2, Scalar beta=1, Scalar alpha=1)
    # output = beta * input + alpha * mat1 * mat2

    assert len(node.outputs) == 1
    inputs = _get_inputs(context, node, expected=5)
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


@register_paddle_op
def linear(context, node):
    inputs = _get_inputs(context, node, expected=[2, 3])
    x = inputs[0]
    W = inputs[1]
    bias = inputs[2] if len(node.inputs) == 3 else None
    res = mb.linear(x=x, weight=W, bias=bias, name=node.name)
    context.add(res)


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

    # inputs = _get_inputs(context, node)

    # x = inputs[0]
    # PyPaddle and MIL has same weight layout
    # Conv: [Cout, Cin, *D]
    # ConvTranspose: [Cin, Cout, *D]
    # weight = inputs[1]
    # bias = inputs[2]
    # strides = inputs[3]

    # # Expand padding. Paddle accepts either an int (for all dimensions) or an n-tuple of ints (one per dimension), but
    # # we require a (2 * n)-tuple, where n is the number of spatial dimensions, start and end for each spatial dimension
    # pad = inputs[4].val

    # if len(weight.shape) in (3, 4):
    #     # 1D and 2D: Need to explicitly state L-R, T-B pad
    #     pad = _np.repeat(pad, 2)
    # elif len(weight.shape) == 5:
    #     # 3D: Need to explicitly state F-Bk, L-R, T-B pad
    #     if type(pad) == int:
    #         pad = _np.repeat(pad, 6)
    #     elif len(pad) == 3:
    #         pad = _np.repeat(pad, 2)
    # else:
    #     raise ValueError(
    #         "Invalid weight dimension. Must be 3, 4, or 5 for 1D, 2D, or 3D convolution, respectively."
    #     )

    # dilations = inputs[5]
    # out_pad = None
    # if len(inputs) >= 12:
    #     transposed = inputs[6].val
    #     out_pad = inputs[7].val
    #     group = inputs[8]
    # elif len(inputs) == 7:
    #     transposed = False
    #     group = inputs[6]
    # else:
    #     raise ValueError(
    #         "unexpected number of inputs for node {} ({}): {}".format(
    #             node.name, node.kind, len(inputs)
    #         )
    #     )

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
    # # Bias is optional in PyPaddle's convolution.
    if bias is not None:
        kwargs["bias"] = bias

    # if transposed is True:
    #     # Transposed convolution
    #     # Handle output_padding using pre-pad or post-crop
    #     pre_pad = [0] * len(pad)
    #     post_crop = [0] * len(pad)

    #     if out_pad is not None and any(out_pad):
    #         output_padding = [0] * len(pad)
    #         # output padding adds additional padding on one of the side of dimension
    #         # i.e. bottom from top-bottom,
    #         #      right  from left-right
    #         #      back   from front-back
    #         # CoreML padding structure is similar [top, bottom, left, right]
    #         # mapping output_padding to simplify further processing!
    #         #
    #         # For ConvTranspose2d: [bottom, right] -> [0, b, 0, r]
    #         output_padding = [
    #             0 if i % 2 == 0 else out_pad[i // 2] for i in range(len(pad))
    #         ]
    #         if sum(pad) == 0 and any(output_padding):
    #             raise ValueError(
    #                 "ConvTranspose configuration of padding=0 and output_padding > 0 not supported!"
    #             )
    #         post_crop = pad.copy()
    #         pad *= 0
    #         for i in range(0, len(pad)):
    #             if post_crop[i] >= output_padding[i]:
    #                 post_crop[i] -= output_padding[i]
    #             else:
    #                 pre_pad[i] = output_padding[i] - post_crop[i]
    #         kwargs["pad"] = pre_pad
    #         if any(pre_pad):
    #             # Constant pad requires pad to be of length 2*input_rank
    #             pre_pad = [0] * 2 * (len(x.shape) - 2) + pre_pad
    #             x = mb.pad(x=x, pad=pre_pad)
    #             kwargs["x"] = x
    #         if any(post_crop):
    #             del kwargs["name"]

    #     conv = mb.conv_transpose(**kwargs)
    #     if any(post_crop):
    #         # TODO: rdar://65575826 (PyPaddle converter: output_padding mapping to slice
    #         # instead of crop layer for 1 and 3D ConvTranspose)
    #         if len(post_crop) == 2 and conv.rank == 3:
    #             # Number of elements to crop from right = post_crop[-1].
    #             # Since slicing supports negative indexing, end_id = -1 * post_crop[-1]
    #             conv = mb.slice_by_index(
    #                 x=conv,
    #                 begin=[0, 0, post_crop[0]],
    #                 end=[0, 0, -1 * post_crop[-1]],
    #                 begin_mask=[True, True, False],
    #                 end_mask=[True, True, False],
    #                 name=node.name,
    #             )
    #         elif len(post_crop) == 4 and conv.rank == 4:
    #             conv = mb.crop(
    #                 x=conv,
    #                 crop_height=post_crop[:2],
    #                 crop_width=post_crop[2:4],
    #                 name=node.name,
    #             )
    #         else:
    #             raise ValueError(
    #                 "output_padding is supported only for ConvTranspose1D or ConvTranspose2D!"
    #             )
    # else:
    #     # Normal convolution
    conv = mb.conv(**kwargs)
    context.add(conv)


# Convolution with "same" padding
@register_paddle_op
def _convolution_mode(context, node):
    inputs = _get_inputs(context, node, expected=7)
    assert inputs[4].val == "same"

    context.add(
        mb.conv(
            x=inputs[0],
            weight=inputs[1],
            bias=inputs[2],
            strides=inputs[3],
            pad_type="same",
            dilations=inputs[5],
            groups=inputs[6],
            name=node.name,
        )
    )


@register_paddle_op
def softmax(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]

    axis = node.desc.attr("axis")
    res = mb.softmax(x=x, axis=axis, name=output_name)
    context.add(res)


@register_paddle_op
def flatten(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    dims = list(x.shape)
    start_val = inputs[1].val
    end_val = inputs[2].val

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
    reshape = mb.reshape(x=x, shape=shape, name=node.name)
    context.add(reshape)


@register_paddle_op
def _reshape_from_tensor(context, node):
    inputs = _get_inputs(context, node, expected=2)

    reshape = mb.reshape(x=inputs[0], shape=inputs[1], name=node.name)
    context.add(reshape)


@register_paddle_op
def softsign(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.softsign(x=inputs[0], name=node.name)
    context.add(res)


@register_paddle_op()
def relu(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.relu(x=inputs[0], name=node.name)
    context.add(res)


@register_paddle_op
def prelu(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    alpha = inputs[1]
    # In the MIL backend, it assumes that the inputs of prelu should have
    # at least rank 3, i.e. [batch, channel, spatial_dims*].
    if x.rank >= 2:
        alpha = alpha.val
        alpha = _np.ones((x.shape[1],)) * alpha

    if x.rank <= 2:
        axes = [1, 2] if x.rank == 1 else [2]
        x = mb.expand_dims(x=x, axes=axes)
        x = mb.prelu(x=x, alpha=alpha)
        res = mb.squeeze(x=x, axes=axes, name=node.name)
    else:
        res = mb.prelu(x=x, alpha=alpha, name=node.name)

    context.add(res)


@register_paddle_op
def linspace(context, node):
    inputs = _get_inputs(context, node, min_expected=3)

    start = inputs[0]
    end = inputs[1]
    nums = inputs[2]
    start = mb.cast(x=start, dtype="fp32")
    end = mb.cast(x=end, dtype="fp32")

    if start.val is not None and end.val is not None and nums.val is not None:
        start_val = start.val
        end_val = end.val
        nums_val = nums.val
        if nums_val < MAX_SIZE_CONSTANT_FOLDING:
            res = mb.const(val=_np.linspace(start_val, end_val, nums_val), name=node.name)
            context.add(res)
            return

    if nums.val is None:
        msg = "Dynamic steps input for paddle.linspace is not supported. Please use paddle.arange instead"
        raise NotImplementedError(msg)
    else:
        if nums.val == 1:
            res = mb.expand_dims(x=start, axes=[0], name=node.name)
        else:
            # step = (end - start) / (nums - 1)
            x = mb.sub(x=end, y=start)
            y = mb.sub(x=nums, y=1)
            x = mb.cast(x=x, dtype="fp32")
            y = mb.cast(x=y, dtype="fp32")
            step = mb.real_div(x=x, y=y)

            # Note that the range_1d op excluded the end point,
            # so we have to add the end back to the resulting array.
            arange = mb.range_1d(end=end, start=start, step=step)
            new_end = mb.expand_dims(x=end, axes=[0])
            res = mb.concat(values=[arange, new_end], axis=0, name=node.name)
    context.add(res)


@register_paddle_op
def relu6(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.relu6(x=inputs[0], name=node.name)
    context.add(res)


@register_paddle_op
def einsum(context, node):
    a = context[node.inputs[1]][0]
    b = context[node.inputs[1]][1]
    equation = context[node.inputs[0]].val
    x = build_einsum_mil(a, b, equation, node.name)
    context.add(x)


@register_paddle_op
def eye(context, node):
    inputs = _get_inputs(context, node, expected=[5, 6])
    if len(inputs) == 5:
        eye = _np.eye(inputs[0].val)
    if len(inputs) == 6:
        eye = _np.eye(inputs[0].val, inputs[1].val)
    eye = mb.const(val=eye, name=node.name)
    context.add(eye)


@register_paddle_op
def elu(context, node):
    ## Paddle port to ATen adds scale and input_scale which is set to 1
    inputs = _get_inputs(context, node, expected=4)

    res = mb.elu(x=inputs[0], alpha=inputs[1], name=node.name)
    context.add(res)


@register_paddle_op()
def leaky_relu(context, node):
    inputs = _get_inputs(context, node, expected=2)

    res = mb.leaky_relu(x=inputs[0], alpha=inputs[1], name=node.name)
    context.add(res)


@register_paddle_op()
def rrelu(context, node):
    inputs = _get_inputs(context, node, expected=5)

    # Alpha in evaluation mode is just the average between upper and lower.
    lower_alpha = inputs[1]
    upper_alpha = inputs[2]
    alpha = (lower_alpha.val + upper_alpha.val) / 2

    res = mb.leaky_relu(x=inputs[0], alpha=alpha, name=node.name)
    context.add(res)


@register_paddle_op
def softplus(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    beta_ = inputs[1].val
    C = x.shape[1]
    alpha_br = _np.repeat(1.0 / beta_, C).astype('float32')
    beta_br = _np.repeat(beta_, C).astype('float32')

    res = mb.softplus_parametric(x=x, alpha=alpha_br, beta=beta_br, name=node.name)
    context.add(res)


@register_paddle_op
def mish(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]

    softplus = mb.softplus(x=x)
    tanh = mb.tanh(x=softplus)
    res = mb.mul(x=x, y=tanh, name=node.name)
    context.add(res)


def _adjust_pad_for_ceil_mode(input_shape, kernel_size, stride_sizes, pad_sizes):
    """ Given an input tensor and pooling parameters, add the extra input
        padding needed to replicate ceil_mode.
        MIL 3D pooling does not support ceil_mode natively, but we can
        workaround by padding the input appropriately.

        PyPaddle output size formula for pooling:
        (reference: https://github.com/pypaddle/pypaddle/blob/375c30a7177442fb9d6de7516a9ae4031ae324c4/aten/src/ATen/native/Pool.h#L28)

        When ceil mode is True:
            out_dim = floor((in_dim + pad_l + pad_r - kernel_size + (stride-1)) / stride) + 1
            if (out_dim-1) * stride >= in_dim + pad_l and (pad_l > 0 or pad_r > 0):
                out_dim = out_dim - 1
        When ceil mode is False:
            out_dim = floor((in_dim + pad_l + pad_r - kernel_size) / stride) + 1


        # follow the approach here to calculate padding:
        # https://github.com/pypaddle/pypaddle/blob/edf751ca2fededecdd9366874c761431c0f61f01/aten/src/ATen/native/mkldnn/Pooling.cpp#L121
        # which keeps increasing the pad_r value until the output size without the ceil mode matches that of the ceil mode
    """

    def _calculate_pool_output_size(in_dim, kernel, stride, pad_l, pad_r, ceil_mode):
        if ceil_mode:
            out_dim = _math.floor((in_dim + pad_r + pad_l - kernel + stride - 1) / stride) + 1
            if (out_dim - 1) * stride >= in_dim + pad_l and (pad_l > 0 or pad_r > 0):
                out_dim = out_dim - 1
        else:
            out_dim = _math.floor((in_dim + pad_r + pad_l - kernel) / stride) + 1
        return out_dim

    new_pad = pad_sizes.copy()
    for idx in range(len(input_shape)):
        if is_symbolic(input_shape[idx]):
            logger.warning(
                "pooling padding adjusted to support ceil_mode=True, for symbolic dimension."
                "Output shape of the pool op maybe be wrong for certain input shapes."
            )
            new_pad[2 * idx + 1] += stride_sizes[idx] - 1
        else:
            out_dim_with_ceil_mode = _calculate_pool_output_size(
                input_shape[idx],
                kernel_size[idx],
                stride_sizes[idx],
                pad_sizes[2 * idx],
                pad_sizes[2 * idx + 1],
                True,
            )
            is_equal = False
            while not is_equal:
                out_dim_without_ceil_mode = _calculate_pool_output_size(
                    input_shape[idx],
                    kernel_size[idx],
                    stride_sizes[idx],
                    new_pad[2 * idx],
                    new_pad[2 * idx + 1],
                    False,
                )
                is_equal = True
                if out_dim_without_ceil_mode < out_dim_with_ceil_mode:
                    new_pad[2 * idx + 1] += 1
                    is_equal = False

    return new_pad


def _max_pool(context, node, inputs):
    x = inputs[0]
    kernel_sizes = inputs[1]
    strides = inputs[2]
    if strides.op.op_type == "const" and (not list(strides.val)):
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
    spatial_rank = len(pad) // 2
    if spatial_rank > 2 and ceil_mode is True and list(strides.val) != [1] * len(strides.val):
        # since MIL does not support ceil_mode for 3D pool,
        # need to adjust padding values if ceil_mode is True
        # ceil_mode only causes any difference though, if the strides are not 1
        x_spatial_dimensions = x.shape[-spatial_rank:]
        pad = _adjust_pad_for_ceil_mode(x_spatial_dimensions, kernel_sizes.val, strides.val, pad)

    pool = mb.max_pool(
        x=x,
        kernel_sizes=kernel_sizes,
        strides=strides,
        pad_type=pad_type,
        pad=pad,
        name=node.name,
        ceil_mode=ceil_mode if spatial_rank <= 2 else False,
    )
    context.add(pool)


@register_paddle_op
def max_pool1d(context, node):
    inputs = _get_inputs(context, node, expected=6)
    _max_pool(context, node, inputs)


@register_paddle_op
def max_pool2d(context, node):
    inputs = _get_inputs(context, node, expected=6)
    _max_pool(context, node, inputs)


@register_paddle_op
def max_pool3d(context, node):
    inputs = _get_inputs(context, node, expected=6)
    _max_pool(context, node, inputs)


@register_paddle_op
def minimum(context, node):
    inputs = _get_inputs(context, node, expected=2)
    assert len(node.outputs) == 1
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    out = mb.minimum(x=x, y=y, name=node.name)
    context.add(out)


@register_paddle_op
def clamp_min(context, node):
    x = _get_inputs(context, node, expected=2)
    x = mb.clip(x=x[0], alpha=x[1], beta=_np.inf, name=node.name)
    context.add(x)


@register_paddle_op
def maximum(context, node):
    inputs = _get_inputs(context, node, expected=2)
    assert len(node.outputs) == 1
    x = context[node.inputs[0]]
    y = context[node.inputs[1]]
    out = mb.maximum(x=x, y=y, name=node.name)
    context.add(out)


@register_paddle_op
def div(context, node):
    inputs = _get_inputs(context, node, expected=[2, 3])

    if len(inputs) > 2 and inputs[2] is not None:
        rounding_mode = inputs[2].val
        if rounding_mode == "floor":
            # round towards negative infinity
            # e.g.:
            # values before floor: [2.6, -3.4, -3.6]
            # values after floor: [2, -4, -4]
            res = mb.floor_div(x=inputs[0], y=inputs[1], name=node.name)
        elif rounding_mode == "trunc":
            # round towards 0
            # e.g.:
            # values before trunc: [2.6, -3.4, -3.6]
            # values after trunc: [2, -3, -3]
            x = mb.cast(x=inputs[0], dtype="fp32")
            y = mb.cast(x=inputs[1], dtype="fp32")
            z = mb.real_div(x=x, y=y)
            s = mb.sign(x=z)
            all_positive = mb.mul(x=z, y=s)
            all_positive_floor = mb.floor(x=all_positive)
            res = mb.mul(x=all_positive_floor, y=s, name=node.name)
        else:
            raise NotImplementedError(
                'rounding mode "{}" not supported in the "div" op'.format(rounding_mode)
            )
    else:
        x = mb.cast(x=inputs[0], dtype="fp32")
        y = mb.cast(x=inputs[1], dtype="fp32")
        res = mb.real_div(x=x, y=y, name=node.name)
    context.add(res)


@register_paddle_op(paddle_alias=["elementwise_floordiv"])
def floor_divide(context, node):
    x = context[node.input("X")[0]]
    y = context[node.input("Y")[0]]
    output_name = node.output("Out")[0]
    res = mb.floor_div(x=x, y=y, name=output_name)
    context.add(res)

@register_paddle_op
def true_divide(context, node):
    inputs = _get_inputs(context, node, expected=2)
    res = mb.real_div(x=inputs[0], y=inputs[1], name=node.name)
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

@register_paddle_op()
def pow(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = promote_input_dtypes(inputs)
    res = mb.pow(x=x, y=y, name=node.name)
    context.add(res)


@register_paddle_op(paddle_alias=["rsub"])
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

    x, y = promote_input_dtypes([x, y])
    res = mb.sub(x=x, y=y, name=node.name)
    context.add(res)


@register_paddle_op(
    paddle_alias=[
        "sum",
        "logsumexp",
    ])
def mean(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    if types.is_bool(x.dtype):
        # TODO: In the future when MIL op supports bool, we need to use curr_opset_version to decide
        # if we want to cast or not.
        x = mb.cast(x=x, dtype="fp32")
    kwargs = {"x": x, "name": node.name}

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
    if node.kind == "sum":
        res = mb.reduce_sum(**kwargs)
    elif node.kind == "logsumexp":
        res = mb.reduce_log_sum_exp(**kwargs)
    else:
        res = mb.reduce_mean(**kwargs)
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
    # inputs = _get_inputs(context, node, expected=2)
    x = context[node.input("X")[0]]

    output_name = node.output("Out")[0]

    axes = node.desc.attr("axes")

    unsqueeze = mb.expand_dims(x=x, axes=axes, name=output_name)
    context.add(unsqueeze)


@register_paddle_op
def size(context, node):
    inputs = _get_inputs(context, node, expected=[1, 2])

    # Get the shape of the tensor.
    size_node = mb.shape(x=inputs[0], name=node.name + "_shape")
    # Get the size of the tensor along the input dimension.
    if len(node.inputs) == 2:
        dim = inputs[1].val
        size_node = _list_select(size_node, dim)
    context.add(size_node, node.name)


@register_paddle_op
def _shape_as_tensor(context, node):
    inputs = _get_inputs(context, node, expected=1)

    # Get the shape of the tensor.
    shape_node = mb.shape(x=inputs[0], name=node.name)
    context.add(shape_node, node.name)


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
    # if isinstance(shape, ListVar):
    #     length = mb.list_length(ls=shape)
    #     indices = mb.range_1d(start=0, end=length, step=1)
    #     shape = mb.list_gather(ls=shape, indices=indices)

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
        # https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pypaddle-work
        # However, as indicated in that SO, this isn't quite how PyPaddle
        # computes adaptive pooling, leading to inaccuracies in model outputs.
        # rdar://60900834
        strides = [ind // outd for ind, outd in zip(_input.shape[-2:], output_size)]
        pad_type = "valid"
        # Need to explicity state L-R, T-B pad
        pad = [0, 0, 0, 0]
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


@register_paddle_op(paddle_alias=['constant_pad_nd'])
def pad(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]

    pad = inputs[1]
    if pad.val is not None:
        pad = pad.val.reshape((-1, 2))[::-1].reshape(-1).tolist()
        missing_dims = x.rank - (len(pad) // 2)
        pad = [0, 0] * missing_dims + pad

    if len(inputs) == 4:
        mode = inputs[2].val
        assert mode in ('constant', 'reflect', 'replicate')
        val_index = 3
    else:
        mode = 'constant'
        val_index = 2

    scalar_val = inputs[val_index] if inputs[val_index] else 0.0
    if inputs[val_index] and inputs[val_index].op.op_type == "const":
        scalar_val = float(scalar_val.val)

    res = mb.pad(x=x, pad=pad, mode=mode, constant_val=scalar_val, name=node.name)
    context.add(res)


@register_paddle_op
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
            x=_input, axes=[-2, -1], keep_dims=True, name=node.name
        )
    elif _input.shape is not None:
        # TODO: The calculations to convert adaptive_pool to standard pool,
        # given a known input size, come from
        # https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pypaddle-work
        # However, as indicated in that SO, this isn't quite how PyPaddle
        # computes adaptive pooling, leading to inaccuracies in model outputs.
        # rdar://60900834
        strides = [ind // outd for ind, outd in zip(_input.shape[-2:], output_size)]
        pad_type = "valid"
        # Need to explicity state L-R, T-B pad
        pad = [0, 0, 0, 0]
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


@register_paddle_op
def batch_norm(context, node):
    inputs = _get_inputs(context, node, expected=9)
    # inputs skipped:
    #   float momentum (6)
    #   bool cudnn_enabled (8)
    input_rank = inputs[0].rank
    if input_rank < 2 or input_rank > 5:
        raise ValueError(
            "BatchNorm: Encountered invalid input rank during translation in paddle frontend."
        )

    _input = inputs[0]
    weight = inputs[1]
    bias = inputs[2]
    running_mean = inputs[3]
    running_var = inputs[4]
    training = inputs[5].val
    eps = inputs[7]

    # If training = True, the mean and variance of the current batch of data are used to normalize the input data.
    # If training = False, data statistics running_mean and running_var are used instead.
    # Note that, even in the evaluation mode (after calling model.eval()), the training parameter can still be true
    # and it just refers to a different computation as mentioned above.

    # helper functions for different type of batch norm
    def _add_batch_norm_dynamic():
        x = _input
        
        if training or (running_mean is None) or (running_var is None):
            axes = [axis for axis in range(x.rank) if axis != 1]
            mean = mb.reduce_mean(x=x, axes=axes, keep_dims=True)
            num = mb.sub(x=x, y=mean)
            square = mb.mul(x=num, y=num)
            variance = mb.reduce_mean(x=square, axes=axes, keep_dims=True)
            shape = mb.shape(x=variance)
        else:
            shape = [1] * x.rank
            shape[1] = -1 if any_symbolic(running_mean.shape) else running_mean.shape[0]
            mean = mb.reshape(x=running_mean, shape=shape)
            num = mb.sub(x=x, y=mean)
            variance = mb.reshape(x=running_var, shape=shape)

        variance_add_epsilon = mb.add(x=variance, y=eps)
        sqrt = mb.sqrt(x=variance_add_epsilon)
        
        name = node.name if weight is None and bias is None else node.name + "_div"
        x = mb.real_div(x=num, y=sqrt, name=name)
        
        if weight is not None:
            weight_reshape = mb.reshape(x=weight, shape=shape)
            name = node.name if bias is None else node.name + "_mul"
            x = mb.mul(x=x, y=weight_reshape, name=name)
            
        if bias is not None:
            bias_reshape = mb.reshape(x=bias, shape=shape)
            x = mb.add(x=x, y=bias_reshape, name=node.name)

        context.add(x)

    def _add_batch_norm_1d():
        # first expand the 3d tensor to 4d, and call the standard mb.batch_norm
        x = mb.expand_dims(x=_input, axes=[-1], name=node.name + "_rank2_expansion")
        bn = mb.batch_norm(
            x=x,
            mean=running_mean,
            variance=running_var,
            gamma=weight,
            beta=bias,
            epsilon=eps,
            name=node.name + "_batch_norm_1d",
        )
        bn = mb.squeeze(x=bn, name=node.name, axes=[-1])
        context.add(bn)

    def _add_batch_norm():
        bn = mb.batch_norm(
            x=_input,
            mean=running_mean,
            variance=running_var,
            gamma=weight,
            beta=bias,
            epsilon=eps,
            name=node.name,
        )
        context.add(bn)

    is_batch_norm_1d_rank_2 = input_rank == 2

    if training or running_mean.val is None or running_var.val is None or weight is None or bias is None:
        _add_batch_norm_dynamic()
    elif is_batch_norm_1d_rank_2:
        _add_batch_norm_1d()
    else:
        _add_batch_norm()


@register_paddle_op
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

    # padding_idx = -1
    # scale_grad_by_freq = False
    # sparse = False
    # if len(inputs) >= 3:
    #     padding_idx = inputs[2].val
    # if len(inputs) >= 4:
    #     scale_grad_by_freq = inputs[3].val
    # if len(inputs) >= 5:
    #     sparse = inputs[4].val

    # if padding_idx != -1 or scale_grad_by_freq or sparse:
    #     logger.warning(
    #         "CoreML embedding (gather) layer does not support any "
    #         "inputs besides the weights and indices. Those given "
    #         "will be ignored."
    #     )

    indices = mb.cast(x=indices, dtype="int32")

    #  Changing the axis from 0 is not an option in paddle, so we don't expose it
    gather = mb.gather(x=_input, indices=indices, name=output_name)
    context.add(gather)


@register_paddle_op()
def hardtanh(context, node):
    inputs = _get_inputs(context, node, expected=3)
    _input = inputs[0]
    min_val = inputs[1].val
    max_val = inputs[2].val

    res = mb.clip(x=_input, alpha=min_val, beta=max_val, name=node.name)
    context.add(res)


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

    # if (len(values)==1):
    #     res = mb.expand_dims(x=values[0], axes=[axis], name=output_name)
    #     context.add(res)
    #     return

    
    res = mb.stack(values=values, axis=axis, name=output_name)
    context.add(res)


@register_paddle_op
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


@register_paddle_op(paddle_alias=["bool"])
def _bool(context, node):
    _cast(context, node, bool, "bool")


@register_paddle_op(paddle_alias=["int"])
def _int(context, node):
    _cast(context, node, int, "int32")


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


@register_paddle_op
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
        Used to transform lstm weights from pypaddle
        to CoreML format
    """
    split_size = weights.shape[0] // 4
    weights_split = mb.split(x=weights, split_sizes=_np.array([split_size] * 4), axis=0)
    return mb.concat(
        values=[weights_split[0], weights_split[1], weights_split[3], weights_split[2]],
        axis=0,
    )


def _pypaddle_hidden_to_coreml_milops(x, name):
    """
        Used to transform lstm state values (hn, cn)
        from pypaddle to CoreML format.
    """
    split_size = x.shape[0] // 2
    x_split = mb.split(x=x, split_sizes=_np.array([split_size] * 2), axis=0)
    x_concat = mb.concat(
        values=[x_split[0], x_split[1]],
        axis=2,
    )
    # (4.) See docstring to @lstm
    return mb.squeeze(x=x_concat, axes=_np.array([0]), name=name)


def _add_gru_layer(_input, h0, wi, wh, bi, bh, h_list_name, h_name):
    """
        Add a single GRU layer.
        Please note that the CoreML GRU has different definition from Paddle,
        so we cannot use mb.gru, and need to implement it with while loop.
        To be more specific, in CoreML:

        o_t = activation(W_{io} x_t + r_t * W_{ho} h_(t−1) + b_{o})

        while paddle has
        o_t = activation(W_{io} x_t + b_{io} + r_t * (W_{ho} h_(t−1) + b_{ho}))

        Inputs:
            _input : (seq_len, batch_size, input_dim)
            h0 : (1, batch_size, hidden_dim)
            wi : (3*hidden_dim, input_dim) for the first layer, else (3*hidden_dim, hidden_dim)
            wh : (3*hidden_dim, hidden_dim)
            bi : (3*hidden_dim)
            bh : (3*hidden_dim)

        Return:
            h_list : the list contains all hidden states for each time step
                     with shape (seq_len, batch_size, hidden_dim)
            h : the last hidden state, with shape (1, batch_size, hidden_dim
    """

    # split the weights and bias
    w_ir, w_iz, w_in = _np.split(wi, 3)
    w_hr, w_hz, w_hn = _np.split(wh, 3)
    b_ir, b_iz, b_in = _np.split(bi, 3)
    b_hr, b_hz, b_hn = _np.split(bh, 3)

    # allocate hlist
    # hlist : (seq_len, batch_size, hidden_dim)
    x_shape = mb.shape(x=_input)
    seq_len = mb.slice_by_index(x=x_shape, begin=[0], end=[1])
    h_shape = mb.shape(x=h0)
    h_shape = mb.slice_by_index(x=h_shape, begin=[1], end=[3])
    h_list_shape = mb.concat(values=[seq_len, h_shape], axis=0)
    h_list = mb.fill(shape=h_list_shape)

    # concate h0 to h_list
    # h_list: (seq_len + 1, batch_size, hidden_dim)
    h_list = mb.concat(values=[h0, h_list], axis=0)

    def cond(i, h_list):
        return mb.less(x=i, y=seq_len)

    def body(i, h_list):
        # slice for the x and state for time step i
        # the resulting shape:
        # xt : (batch_size, input_dim)
        # h_prev : (batch_size, hidden_dim)

        xt = mb.gather(x=_input, indices=i, axis=0)
        h_prev = mb.gather(x=h_list, indices=i, axis=0)

        xt = mb.squeeze(x=xt, axes=[0])
        h_prev = mb.squeeze(x=h_prev, axes=[0])

        # rt = sigmoid(wir * xt + whr * h_prev + bir + bhr)
        # rt : (batch_size, hidden_dim)
        rt_1 = mb.linear(x=xt, weight=w_ir, bias=b_ir)
        rt_2 = mb.linear(x=h_prev, weight=w_hr, bias=b_hr)
        rt = mb.add(x=rt_1, y=rt_2)
        rt = mb.sigmoid(x=rt)

        # zt = sigmoid(wiz * xt + whz * h_prev + biz + bhz)
        # zt : (batch_size, hidden_dim)
        zt_1 = mb.linear(x=xt, weight=w_iz, bias=b_iz)
        zt_2 = mb.linear(x=h_prev, weight=w_hz, bias=b_hz)
        zt = mb.add(x=zt_1, y=zt_2)
        zt = mb.sigmoid(x=zt)

        # nt = tanh(win * xt + bin + rt(whn * h_prev + bhn))
        # nt : (batch_size, hidden_dim)
        nt_1 = mb.linear(x=xt, weight=w_in, bias=b_in)
        nt_2 = mb.linear(x=h_prev, weight=w_hn, bias=b_hn)
        nt_2 = mb.mul(x=rt, y=nt_2)
        nt = mb.add(x=nt_1, y=nt_2)
        nt = mb.tanh(x=nt)

        # h = (1-zt) * nt + zt* h_prev
        # h : (batch_size, hidden_dim)
        h_1 = mb.sub(x=1., y=zt)
        h_1 = mb.mul(x=h_1, y=nt)
        h_2 = mb.mul(x=zt, y=h_prev)
        h = mb.add(x=h_1, y=h_2)

        # update counter
        counter = mb.add(x=i, y=1)

        # update h and h_list
        h = mb.expand_dims(x=h, axes=[0])
        h_list = mb.scatter(data=h_list, indices=counter, updates=h)

        return (
            counter,
            h_list,
        )

    _, h_list = mb.while_loop(
        _cond=cond, _body=body, loop_vars=([0], h_list),
    )

    # slice h0 out of h_list
    h_list = mb.slice_by_index(
        x=h_list,
        begin=[1, 0, 0],
        end=[0, 0, 0],
        begin_mask=[False, True, True],
        end_mask=[True, True, True],
        name=h_list_name,
    )

    # get the last state of h_list
    if seq_len.val is None or seq_len.val > 1:
        h = mb.slice_by_index(
            x=h_list,
            begin=[-1, 0, 0],
            end=[-2, 0, 0],
            begin_mask=[False, True, True],
            end_mask=[False, True, True],
            stride=[-1, 1, 1],
            name=h_name,
        )
    else:
        h = h_list

    return h_list, h


@register_paddle_op
def gru(context, node):
    inputs = _get_inputs(context, node, expected=9)

    _input = inputs[0]
    h0 = inputs[1]
    weights_list = inputs[2]
    has_bias = inputs[3].val
    num_layers = inputs[4].val
    dropout = inputs[5]
    bidirectional = inputs[7].val
    batch_first = inputs[8].val

    # For each layer of GRU, the layout of the weights list is [Wi, Wh, bi, bh] with has_bias == True,
    # and is [Wi, Wh] with bias == False.
    # If bidirectional == True, the list is double up, corresponding to forward and backward direction.
    expected_num_weights = 2 * num_layers * (int(has_bias) + 1) * (int(bidirectional) + 1)
    if len(weights_list) != expected_num_weights:
        raise ValueError(
            "Incorrect weights shape for gru layer: Expected: {}. Recieved {}".format(
                expected_num_weights, len(weights_list)
            )
        )

    # Transpose the input data to (seq_len, batch_size, input_dim) if batch_first == True
    if batch_first:
        _input = mb.transpose(x=_input, perm=[1, 0, 2])

    # iterate through all the layers
    x = _input
    state_out_list = []

    def _get_weights_and_bias(weights_list, index, num_layers, has_bias, bidirectional, mode):
        num_weights_per_layer = len(weights_list) // num_layers
        weights = weights_list[
            num_weights_per_layer * index : num_weights_per_layer * (index + 1)
        ]

        if bidirectional:
            weights_f, weights_r = (
                weights[: num_weights_per_layer // 2],
                weights[num_weights_per_layer // 2 :],
            )
            assert len(weights_f) == len(weights_r)
        else:
            weights_f, weights_r = weights, []

        if mode == "forward":
            weights = weights_f
        elif mode == "reverse":
            weights = weights_r

        wi, wh = weights[0].val, weights[1].val

        if has_bias:
            bi, bh = weights[2].val, weights[3].val
        else:
            hidden_dim = wh.shape[1]
            bi, bh = _np.zeros(3 * hidden_dim), _np.zeros(3 * hidden_dim)

        return wi, wh, bi, bh

    def _get_initial_state(h0, i, bidirectional, mode):

        if mode == "forward":
            return mb.slice_by_index(
                x=h0,
                begin=[(1 + int(bidirectional)) * i, 0, 0],
                end=[(1 + int(bidirectional)) * i + 1, 0, 0],
                begin_mask=[False, True, True],
                end_mask=[False, True, True],
            )
        if mode == "reverse":
            assert bidirectional
            return mb.slice_by_index(
                x=h0,
                begin=[2 * i + 1, 0, 0],
                end=[2 * (i + 1), 0, 0],
                begin_mask=[False, True, True],
                end_mask=[False, True, True],
            )

    seq_output_name = node.outputs[0]  # output sequence name
    state_output_name = node.outputs[1]  # output state name

    for i in range(num_layers):
        # get layer names
        x_name = seq_output_name + "_layer_" + str(i) if i < num_layers - 1 else seq_output_name
        h_name = state_output_name + '_layer_' + str(i) if num_layers > 0 else state_output_name

        if batch_first:
            x_name += "_tmp"

        if bidirectional:
            x_f_name = x_name + '_forward'
            h_f_name = h_name + '_forward'
            x_r_name = x_name + '_backward'
            h_r_name = h_name + '_backward'
        else:
            x_f_name = x_name
            h_f_name = h_name

        # forward direction
        x_f = x
        wi_f, wh_f, bi_f, bh_f = _get_weights_and_bias(
            weights_list, i, num_layers, has_bias, bidirectional, "forward"
        )
        initial_h_f = _get_initial_state(h0, i, bidirectional, "forward")
        x_f, h_f = _add_gru_layer(x_f, initial_h_f, wi_f, wh_f, bi_f, bh_f, x_f_name, h_f_name)

        # reverse direction
        if bidirectional:
            x_r = mb.reverse(x=x, axes=[0])
            wi_r, wh_r, bi_r, bh_r = _get_weights_and_bias(
                weights_list, i, num_layers, has_bias, bidirectional, "reverse"
            )
            initial_h_r = _get_initial_state(h0, i, bidirectional, "reverse")
            x_r, h_r = _add_gru_layer(
                x_r,
                initial_h_r,
                wi_r,
                wh_r,
                bi_r,
                bh_r,
                x_r_name + "_reverse",
                h_r_name,
            )
            x_r = mb.reverse(x=x_r, axes=[0], name=x_r_name)

            # concate output from forward and reverse direction
            x = mb.concat(values=[x_f, x_r], axis=2, name=x_name)
            h = mb.concat(values=[h_f, h_r], axis=0, name=h_name)
        else:
            x = x_f
            h = h_f

        state_out_list.append(h)

    # rnn output
    if batch_first:
        x = mb.transpose(x=x, perm=[1, 0, 2], name=seq_output_name)
    context.add(x, seq_output_name)

    # state output
    if len(state_out_list) > 1:
        h = mb.concat(values=state_out_list, axis=0, name=state_output_name)
    context.add(h, state_output_name)


def _add_simple_rnn(context, node, activation):
    inputs = _get_inputs(context, node, expected=9)

    '''
    Batch size: B
    Sequence length: S
    Input dimension: C
    Hidden dimension: H

    (1) _input : (B, S, C) if batch_first == True, else (S, B, C)
    (2) h0: (num_layers, B, H)
    '''
    _input = inputs[0]
    h0 = inputs[1]
    weights_list = inputs[2]
    has_bias = inputs[3].val
    num_layers = inputs[4].val
    dropout = inputs[5]
    bidirectional = inputs[7].val
    batch_first = inputs[8].val

    # We only support uni-directional simple RNN now
    if bidirectional:
        raise NotImplementedError("Bidirectional simple RNN not supported.")

    expected_num_weights = 2 * num_layers * (int(has_bias) + 1)
    if len(weights_list) != expected_num_weights:
        raise ValueError(
            "Incorrect weights shape for lstm layer: Expected: {}. Recieved {}".format(
                expected_num_weights, len(weights_list)
            )
        )

    # Transpose the input data to (S, B, C) if batch_first == True
    if batch_first:
        _input = mb.transpose(x=_input, perm=[1, 0, 2])

    state_out_list = []
    out = _input

    for i in range(num_layers):
        if has_bias:
            weight_ih = weights_list[4 * i]
            weight_hh = weights_list[4 * i + 1]
            bias = mb.add(x=weights_list[4 * i + 2], y=weights_list[4 * i + 3])
        else:
            weight_ih = weights_list[2 * i]
            weight_hh = weights_list[2 * i + 1]
            bias = None

        # get the initial state
        initial_h = mb.slice_by_index(
            x=h0,
            begin=[i, 0, 0],
            end=[0, 0, 0],
            stride=[1, 1, 1],
            begin_mask=[False, True, True],
            end_mask=[False, True, True],
            squeeze_mask=[True, False, False],
        )

        # get the RNN output for each unit
        out, state = mb.rnn(
            x=out,
            initial_h=initial_h,
            weight_ih=weight_ih,
            weight_hh=weight_hh,
            bias=bias,
            output_sequence=True,
            activation=activation,
        )

        # append state to lists which will stack later
        state_out_list.append(state)

    # rnn output
    output_name = node.outputs[0]
    if batch_first:
        out = mb.transpose(x=out, perm=[1, 0, 2], name=output_name)
    else:
        out = mb.identity(x=out, name=output_name)
    context.add(out, output_name)

    # stack the states into a single tensor
    state_output_name = node.outputs[1]
    if num_layers == 1:
        state = mb.expand_dims(x=state_out_list[0], axes=[0], name=state_output_name)
    else:
        state = mb.stack(values=state_out_list, axis=0, name=state_output_name)
    context.add(state, state_output_name)


@register_paddle_op
def rnn_tanh(context, node):
    _add_simple_rnn(context, node, "tanh")


@register_paddle_op
def rnn_relu(context, node):
    _add_simple_rnn(context, node, "relu")


def _add_mil_lstm(input, initial_h, initial_c, weights, has_bias, bidirectional, name):
    '''
    Most of this code is to transform the tensors into
    a shape acceptable by the CoreML implementation of LSTM.

    For weights, biases,  per direction, pypaddle uses two tensors:
    (ii, if, ig, io) stacked on top of each other for each layer (tensor 1)
    and (hi, hf, hg, ho) stacked on top of each other for each layer (tensor 2).
    That is,  (W_ii|W_if|W_ig|W_io), of shape (4*hidden_size, input_size) and
    (W_hi|W_hf|W_hg|W_ho), of shape (4*hidden_size, hidden_size).


    The CoreML LSTM op expects two tensors, weight and bias. So
    the tensors for weight and bias are seperated from pypaddle's @weights list (1.).
    For bias tensor, the CoreML LSTM op expects the form ii, if, io, ig and hi, hf, ho, hg,
    requiring the ifzo_to_ifoz function. Further adding input and hidden bias into one (2.).
    Similar to bias, input and hidden weight requires different layout. (3.)

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
            f_b = mb.add(x=biases[0], y=biases[1], )
            r_b = mb.add(x=biases[2], y=biases[3], )

        # (3.)
        f_ih_w = _ifzo_to_ifoz(
            weights[0], name=name + "_lstm_forward_ih_weights_ifoz_to_ifzo",
        )
        f_hh_w = _ifzo_to_ifoz(
            weights[1], name=name + "_lstm_forward_hh_weights_ifoz_to_ifzo",
        )
        r_ih_w = _ifzo_to_ifoz(
            weights[2], name=name + "_lstm_reverse_ih_weights_ifoz_to_ifzo",
        )
        r_hh_w = _ifzo_to_ifoz(
            weights[3], name=name + "_lstm_reverse_hh_weights_ifoz_to_ifzo",
        )

        h = _pypaddle_hidden_to_coreml_milops(initial_h, name=name + "_lstm_h0_reshaped")
        c = _pypaddle_hidden_to_coreml_milops(initial_c, name=name + "_lstm_c0_reshaped")
        return mb.lstm(x=input,
                       initial_h=h,
                       initial_c=c,
                       weight_ih=f_ih_w,
                       weight_hh=f_hh_w,
                       weight_ih_back=r_ih_w,
                       weight_hh_back=r_hh_w,
                       bias=(f_b if has_bias else None),
                       bias_back=(r_b if has_bias else None),
                       direction="bidirectional",
                       output_sequence=True,
                       name=name)
    else:
        if has_bias:
            # (1.)
            biases = weights[len(weights) // 2:]
            weights = weights[: len(weights) // 2]
            # (2.)
            b = mb.add(x=biases[0], y=biases[1], )
            b = _ifzo_to_ifoz(
                b, name=name + "_lstm_bias_transformed",
            )
        # (3.)
        f_ih_w = _ifzo_to_ifoz(
            weights[0], name=name + "_lstm_ih_weights_ifoz_to_ifzo",
        )
        f_hh_w = _ifzo_to_ifoz(
            weights[1], name=name + "_lstm_hh_weights_ifoz_to_ifzo",
        )

        h = mb.squeeze(x=initial_h, axes=_np.array([0]), name=name + "_lstm_h0_squeeze")
        c = mb.squeeze(x=initial_c, axes=_np.array([0]), name=name + "_lstm_c0_squeeze")

        return mb.lstm(x=input,
                       initial_h=h,
                       initial_c=c,
                       weight_ih=f_ih_w,
                       weight_hh=f_hh_w,
                       bias=(b if has_bias else None),
                       direction="forward",
                       output_sequence=True,
                       name=name)


@register_paddle_op
def lstm(context, node):
    inputs = _get_inputs(context, node, expected=9)

    _input = inputs[0]

    # there are two cases here,
    # (1) the input tensor is a PackedSequence object,
    # in this case, the second input of the lstm layer is the batch_size (MIL Var).
    # (2) the input tensor is a normal tensor,
    # in this case, the second input is an array.
    # As the result, we can use the second input to identify which category the graph is.

    has_batch_sizes = not isinstance(inputs[1], Iterable)
    if has_batch_sizes:
        batch_sizes = inputs[1]
        h0, c0 = inputs[2]
        weights_list = inputs[3]
        has_bias = inputs[4].val
        num_layers = inputs[5].val
        dropout = inputs[6]
        bidirectional = inputs[8].val
        # the output of the _pack_padded_sequence is always in the layout of batch first
        batch_first = True
    else:
        h0, c0 = inputs[1]
        weights_list = inputs[2]
        has_bias = inputs[3].val
        num_layers = inputs[4].val
        dropout = inputs[5]
        bidirectional = inputs[7].val
        batch_first = inputs[8].val

    '''
    Paddle LSTM layer's input shapes:

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
        all_initial_h = [h0]  # [(n_directions, B, H)]
        all_initial_c = [c0]  # [(n_directions, B, H)]
    else:
        all_initial_h = mb.split(
            x=h0, num_splits=num_layers, axis=0
        )  # [(n_directions, B, H)]
        all_initial_c = mb.split(
            x=c0, num_splits=num_layers, axis=0
        )  # [(n_directions, B, H)]

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

        lstm_out = _add_mil_lstm(
            input=x,
            initial_h=all_initial_h[i],
            initial_c=all_initial_c[i],
            weights=weights_list[
                i * n_weights_per_layer : (i + 1) * n_weights_per_layer
            ],
            has_bias=has_bias,
            bidirectional=bidirectional,
            name=op_name,
        )
        # shape of lstm_out[0] == (S,B,H) if bidirectional = True else (S, B, 2*H)
        x = lstm_out[0]
        # shape of lstm_out[1] == (B,H) if bidirectional = False else (B, 2*H)
        h_out_list.append(lstm_out[1])
        # shape of lstm_out[2] == (B,H) if bidirectional = False else (B, 2*H)
        c_out_list.append(lstm_out[2])

    '''
    For paddle, these are the dimensions of the 3 output tensors:
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

        stack/expand the final state tensors to match the Paddle output
    '''
    for index, (name, output) in enumerate(zip(node.outputs, lstm_out)):
        if index > 0:
            # index > 0 ===> its one of the state outputs (h or c)
            if bidirectional:
                if num_layers == 1:
                    out1, out2 = mb.split(
                        x=output, num_splits=2, axis=1
                    )  # each output of shape [B, H] after the split
                    final_out = mb.stack(
                        values=[out1, out2], axis=0, name=name
                    )  # [2, B, H]
                    context.add(final_out, name)
                else:
                    out_state_tensors_list = (
                        h_out_list if index == 1 else c_out_list
                    )  # each tensor in the list is of shape (B, 2*H)
                    list_of_tensors_to_stack = []
                    for i in range(num_layers):
                        out1, out2 = mb.split(
                            x=out_state_tensors_list[i], num_splits=2, axis=1
                        )  # each output of shape [B, H] after the split
                        out = mb.stack(values=[out1, out2], axis=0)  # [2, B, H]
                        list_of_tensors_to_stack.append(out)
                    final_out = mb.concat(
                        values=list_of_tensors_to_stack, axis=0, name=name
                    )  # output of shape (num_layers * 2, B, H)
                    context.add(final_out, name)
            else:
                if num_layers == 1:
                    unsqueeze = mb.expand_dims(x=output, axes=[0], name=name)
                    context.add(unsqueeze, name)
                else:
                    out = mb.stack(
                        values=h_out_list if index == 1 else c_out_list,
                        axis=0,
                        name=name,
                    )
                    context.add(out, name)
        else:
            if batch_first:
                output = mb.transpose(x=output, perm=[1, 0, 2], name=name)
            context.add(output, name)


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

        # output size is computed using the formula floor (scale * input_size) in Core ML (and PyPaddle).
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


def _is_float_value(x, threshold=0.001):
    return x - _math.floor(x) > threshold


@register_paddle_op
def upsample_linear1d(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    output_size = inputs[1]
    align_corners = bool(inputs[2].val)
    scale = inputs[3]

    scale_factor = None

    if scale is not None and scale.val is not None and scale.shape == (1,):
        # Get the scale factor from provided inputs
        # This happens when recompute_scale_factor = False
        scale_factor = scale.val[0]

        # Currently, we are not supporting recompute_scale_factor = False, align_corners = False with float output size
        _, _, h = x.shape
        if not is_symbolic(h):
            # For the static input shape, we can compute the output size beforehand, and check if it is a float value
            output_size = h * scale_factor
            is_float = _is_float_value(output_size)
        else:
            # For the dynamic input shape, we check if the scale factor itself is float
            is_float = _is_float_value(scale_factor)

        if is_float and not align_corners:
            msg = (
                "recompute_scale_factor = False, align_corners = False with float output size is "
                + "not supported for the upsample op {}".format(node.name)
            )
            raise NotImplementedError(msg)

    elif isinstance(output_size, list):
        # When the input shape is dynamic and recompute_scale_factor = True,
        # we need to trace the graph to find the scale factor.
        x = mb.expand_dims(x=x, axes=[3])
        x = mb.paddle_upsample_bilinear(
            x=x,
            output_height=output_size[0],
            output_width=1,
            align_corners=align_corners,
        )
        x = mb.squeeze(x=x, axes=[3], name=node.name)
        context.add(x)
        return

    elif output_size.val is not None:
        # Infer the scale factor from the provided output size
        scale_factor = _get_scales_from_output_size(output_size, x.shape)

    # Expand the input to a 4d tensor, and use MIL's upsample_bilinear op
    x = mb.expand_dims(x=x, axes=[3])
    x = mb.upsample_bilinear(
        x=x,
        scale_factor_height=scale_factor,
        scale_factor_width=1.,
        align_corners=align_corners,
    )
    x = mb.squeeze(x=x, axes=[3], name=node.name)
    context.add(x)


@register_paddle_op
def upsample_bilinear2d(context, node):
    inputs = _get_inputs(context, node)
    _input = inputs[0]
    output_size = inputs[1]
    align_corners = bool(inputs[2].val)
    scale_factors = inputs[3]

    scales_h, scales_w = None, None

    if (
        scale_factors is not None
        and scale_factors.val is not None
        and scale_factors.rank == 1
        and scale_factors.shape[0] == 2
    ):
        # get scale factors from provided inputs
        # this happens when recompute_scale_factor = False
        scale_factors = scale_factors.val
        scales_h = scale_factors[0]
        scales_w = scale_factors[1]

        # currently, we are not supporting recompute_scale_factor = False, align_corners = False with float output size
        _, _, h, w = _input.shape
        if not is_symbolic(h) and not is_symbolic(w):
            # For the static input shape, we can compute the output size beforehand
            output_h = h * scales_h
            output_w = w * scales_w
            is_h_float = _is_float_value(output_h)
            is_w_float = _is_float_value(output_w)

        else:
            # For the dynamic input shape, we check if the scale factor itself is float
            is_h_float = _is_float_value(scales_h)
            is_w_float = _is_float_value(scales_w)

        if (is_h_float or is_w_float) and not align_corners:
            msg = (
                "recompute_scale_factor = False, align_corners = False with float output size is "
                + "not supported for the upsample op {}".format(node.name)
            )
            raise NotImplementedError(msg)

    elif (
        isinstance(output_size, list)
        and output_size[0].val is None
        and output_size[1].val is None
    ):
        # the input shape is dynamic and recompute_scale_factor = True
        # need to trace the graph to find the scale factor
        # we define a paddle front end op mb.paddle_upsample_bilinear to resolve the const scaling factor
        paddle_upsample_bilinear = mb.paddle_upsample_bilinear(
            x=_input,
            output_height=output_size[0],
            output_width=output_size[1],
            align_corners=align_corners,
            name=node.name,
        )
        context.add(paddle_upsample_bilinear)
        return
    else:
        # infer scale factors from output sizes
        # This happens when recompute_scale_factor = True or the output_size is specified
        scales = _get_scales_from_output_size(output_size, _input.shape)
        if scales:
            scales_h, scales_w = scales

    if scales_h is None or scales_w is None:
        if len(inputs) == 5:
            # For paddle==1.5.0, upsample_bilinear2d has 5 inputs.
            scales_h = inputs[3]
            scales_w = inputs[4]
        else:
            raise ValueError("Failed to infer scale factors from inputs.")

    upsample_bilinear = mb.upsample_bilinear(
        x=_input,
        scale_factor_height=scales_h,
        scale_factor_width=scales_w,
        align_corners=align_corners,
        name=node.name,
    )
    context.add(upsample_bilinear)


@register_paddle_op
def upsample_nearest1d(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    output_size = inputs[1]
    scale = inputs[2]

    scale_factor = None

    if scale is not None and scale.val is not None and scale.shape == (1,):
        # Get the scale factor from provided inputs
        # This happens when recompute_scale_factor = False
        scale_factor = scale.val[0]

    elif isinstance(output_size, list):
        # When the input shape is dynamic and recompute_scale_factor = True,
        # we need to trace the graph to find the scale factor.
        x = mb.expand_dims(x=x, axes=[3])
        x = mb.paddle_upsample_nearest_neighbor(
            x=x,
            output_height=output_size[0],
            output_width=1,
        )
        x = mb.squeeze(x=x, axes=[3], name=node.name)
        context.add(x)
        return
    else:
        # Infer scale factors from output sizes
        scale_factor = _get_scales_from_output_size(output_size, x.shape)

    x = mb.expand_dims(x=x, axes=[3])
    x = mb.upsample_nearest_neighbor(
        x=x,
        scale_factor_height=scale_factor,
        scale_factor_width=1.,
    )
    x = mb.squeeze(x=x, axes=[3], name=node.name)
    context.add(x)


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

    # if (
    #     scale_factors is not None
    #     and scale_factors.val is not None
    #     and scale_factors.rank == 1
    #     and scale_factors.shape[0] == 2
    # ):
    #     # get scale factors from provided inputs
    #     scale_factors = scale_factors.val
    #     scales_h = scale_factors[0]
    #     scales_w = scale_factors[1]
    # elif (
    #     isinstance(output_size, list)
    #     and output_size[0].val is None
    #     and output_size[1].val is None
    # ):
    #     # the input shape is dynamic and recompute_scale_factor = True
    #     # need to trace the graph to find the scale factor
    #     # we define a paddle front end op mb.paddle_upsample_nearest_neighbor to resolve the const scaling factor
    #     paddle_upsample_nearest2d = mb.paddle_upsample_nearest_neighbor(
    #         x=_input,
    #         output_height=output_size[0],
    #         output_width=output_size[1],
    #         name=node.name,
    #     )
    #     context.add(paddle_upsample_nearest2d)
    #     return
    # else:
    #     # infer scale factors from output sizes
    #     scales = _get_scales_from_output_size(output_size, _input.shape)
    #     if scales:
    #         scales_h, scales_w = scales

    # if scales_h is None or scales_w is None:
    #     if len(inputs) == 5:
    #         # For paddle==1.5.0, upsample_bilinear2d has 5 inputs.
    #         scales_h = inputs[3]
    #         scales_w = inputs[4]
    #     else:
    #         raise ValueError("Failed to infer scale factors from inputs.")

    # upsample_nearest2d = mb.upsample_nearest_neighbor(
    #     x=_input,
    #     scale_factor_height=scales_h,
    #     scale_factor_width=scales_w,
    #     name=node.name,
    # )
    # context.add(upsample_nearest2d)


@register_paddle_op(paddle_alias=["listunpack"])
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


@register_paddle_op
def loop(context, node):
    """ In PaddleIR, a loop looks like:
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
                logger.warning(
                    "detected change in shape of loop variable. this could lead to incorrect inference results!"
                )
                logger.warning(
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
        return tuple(
            [
                iter_var,
            ]
            + res
        )

    loop = mb.while_loop(
        _cond=_loop_cond, _body=_loop_body, loop_vars=loop_vars, name=name
    )

    # Make sure the loop returned the expected number of outputs. Note that the
    # first two loop outputs are the iteration count and condition.
    assert len(loop) - 2 == len(node.outputs)
    for output_name, output_var in zip(node.outputs, loop[2:]):
        context.add(output_var, paddle_name=output_name)


@register_paddle_op(paddle_alias=["if"])
def _if(context, node):
    """ In PaddleIR, a conditional looks like:
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
    if not isinstance(cond, (tuple, list)):
        cond = (cond,)

    # Make sure the condition returned the expected number of outputs.
    assert len(cond) == len(node.outputs)
    for output_name, output_var in zip(node.outputs, cond):
        context.add(output_var, paddle_name=output_name)


@register_paddle_op
def select(context, node):
    inputs = _get_inputs(context, node, expected=3)
    _input = inputs[0]
    dim = inputs[1].val
    index = inputs[2].val

    assert dim.shape == ()
    assert index.shape == ()

    # NOTE:
    # Each index in @begin_array/@end_array corresponds to a dimension of @_input
    # Each val of those arrays corresponds to the start/end index to slice in that dimension
    rank = _input.rank
    begin_array = [0] * rank
    begin_array[dim] = index
    end_array = [s if isinstance(s, int) else 0 for s in _input.shape]
    end_mask = [True] * rank
    squeeze_mask = [False] * rank
    squeeze_mask[dim] = True

    if index != -1:
        end_array[dim] = index + 1
        end_mask[dim] = False

    slice_by_index = mb.slice_by_index(
        x=_input,
        begin=begin_array,
        end=end_array,
        end_mask=end_mask,
        squeeze_mask=squeeze_mask,
        name=node.name,
    )
    context.add(slice_by_index)


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
        # if inputs[1].dtype not in TYPE_TO_DTYPE_STRING:
        #     raise NotImplementedError(
        #         "Tensor type {} cast is not supported.".format(inputs[1].dtype)
        #     )
        x = mb.cast(x=x, dtype=NUM_TO_DTYPE_STRING[out_dtype], name=output_name)

    context.add(x)


@register_paddle_op
def nonzero(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    nonzero = mb.non_zero(x=x, name=node.name)
    context.add(nonzero)


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


@register_paddle_op
def index_put(context, node):
    inputs = _get_inputs(context, node, expected=4)
    x = inputs[0]
    indices = inputs[1]
    values = inputs[2]
    accumulate = inputs[3].val
    rank = x.rank
    mode = "add" if accumulate else "update"

    indices_type = indices[0].sym_type.get_primitive()

    if types.is_bool(indices_type):
        assert len(indices) == 1, "Unsupported index_put_ usage."
        indices = indices[0]
        assert indices.shape == x.shape, "indices shape must equal to input shape for index put operation."
        indices = mb.cast(x=indices, dtype="int32")
        indices = mb.non_zero(x=indices)

    if types.is_int(indices_type):
        if len(indices) > 1:
            indices = mb.stack(values=indices, axis=rank - 1)
        else:
            indices = mb.expand_dims(x=indices[0], axes=[-1])

    if len(values.shape) == 0:
        values = mb.expand_dims(x=values, axes=[0])

    if values.rank == 1 and values.shape[0] == 1:
        reps = value_at(mb.shape(x=indices), 0)
        reps = mb.expand_dims(x=reps, axes=[0])
        values = mb.tile(x=values, reps=reps)

    result = mb.scatter_nd(data=x, indices=indices, updates=values, mode=mode, name=node.name)
    context.add(result)


@register_paddle_op
def index(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    indices = inputs[1]
    rank = x.rank

    """
    we support two kinds of paddle tensor indexing now
    """
    """
    Case 1: indices are bool tensor indicating index selection
    Ex:
        a = paddle.randn(1,2,3,4)
        b = a[a > 0.1]

    For this case, indices is a list with length 1, containing a single bool tensor with the same shape
    as the input tensor.

    The true value indicates whether the element should be selected.
    The output b is a 1-D vector with shape (N), where N is the number of elements satisfying condition > 0.1
    """
    if (
        len(indices) == 1
        and indices[0] is not None
        and indices[0].sym_type.get_primitive() == types.bool
        and indices[0].shape == x.shape
    ):
        indices = indices[0]
        x_reshape = mb.reshape(x=x, shape=[-1])
        indices = mb.cast(x=indices, dtype="int32")
        indices_reshape = mb.reshape(x=indices, shape=[-1])

        # the resulting non_zeros_indices has shape [N, 1],
        # where N is the number of non-zero element
        non_zeros_indices = mb.non_zero(x=indices_reshape)
        non_zeros_indices = mb.squeeze(x=non_zeros_indices, axes=[1])  # [N]

        # gather the element from the flatten vector
        select_x = mb.gather(x=x_reshape, indices=non_zeros_indices, axis=0, name=node.name)
        context.add(select_x)
        return

    """
    Case 2: Pure index selection
    Ex # 1:
        a = paddle.rand(1,2,3,4)
        index = paddle.tensor([0, 1])
        b = a[:,:,:,index]

        In this case, indices is a list [None, None, None, [0, 1]]]. The None element means the corresponding
        dimension is masked.

        b has shape (1,2,3,2).

    Ex # 2:
        a = paddle.rand(1,2,3,4)
        index = paddle.tensor([0, 1])
        b = a[:,index,:,index]

        In this case, indices is a list [None, [0,1], None, [0,1]]

        b has shape (2,1,3)

    Ex # 3:
        a = paddle.rand(1,2,3,4)
        index_1 = paddle.tensor([0, 1])
        index_2 = paddle.tensor([0, 1])
        b = a[:,index_1,index_2,:]

        indices is a list [None, [0, 1], [0, 1], None]

        b has shape (1,2,4)

    EX # 4:
        a = paddle.rand(4,5)
        index_1 = [True, True, False, False]
        index_2 = [False, True, True, False, False]
        b = a[index_1, index_2]

        indices is a list [[True, True, False, False], [False, True, True, False, False]]

        b has shape (2, )

    Note that, in pypaddle, the indices can be broadcasable. And it is NOT supported right now.
    """

    # get the index axes
    indices = indices + [None] * (x.rank - len(indices))
    indices_axes = []
    valid_indices = []
    for i, index in enumerate(indices):
        if index is not None:
            indices_axes.append(i)
            valid_indices.append(index)

    # If all elements in indices is None, simpily return the original tensor.
    if len(indices_axes) == 0:
        x = mb.identity(x=x, name=node.name)
        context.add(x)
        return

    # convert all indices to int type
    for i, indice in enumerate(valid_indices):
        if indice is not None and types.is_bool(indice.dtype):
            indice = mb.non_zero(x=indice)
            indice = mb.squeeze(x=indice, axes=[1])
        valid_indices[i] = indice

    # For the single index axis case, we can use mb.gather directly
    if len(indices_axes) == 1:
        axis = indices_axes[0]
        x = mb.gather(x=x, indices=valid_indices[0], axis=axis, name=node.name)
        context.add(x)
        return

    # For multiple index axes case, we now assume that all the index have equal shape
    for index in valid_indices:
        if not is_compatible_symbolic_vector(index.shape, valid_indices[0].shape):
            raise NotImplementedError("Broadcasable tensor index not supported.")

    # First stack the index together
    indices_rank = valid_indices[0].rank
    indices = mb.stack(values=valid_indices, axis=indices_rank)

    # transpose the input tensor to gather the slicing index in front
    is_connected = True
    for i in range(1, len(indices_axes)):
        if indices_axes[i] != indices_axes[i - 1] + 1:
            is_connected = False
            break

    name = node.name + "_transpose" if is_connected else node.name
    perm = indices_axes + [axis for axis in range(x.rank) if axis not in indices_axes]
    x = mb.transpose(x=x, perm=perm)
    x = mb.gather_nd(x=x, indices=indices, name=name)

    # if the index axes are connect, we need to transpose it back
    if is_connected:
        new_dimensions = list(range(indices_axes[0], indices_axes[0] + indices_rank))
        new_perm = new_dimensions + [
            axis
            for axis in range(rank + indices_rank - len(indices_axes))
            if axis not in new_dimensions
        ]
        perm_back = [new_perm.index(axis) for axis in range(len(new_perm))]
        x = mb.transpose(x=x, perm=perm_back, name=node.name)
    context.add(x)


@register_paddle_op
def ones(context, node):
    inputs = _get_inputs(context, node, expected=[5, 6])
    size = inputs[0]
    # dtype = NUM_TO_PADDLE_DTYPE[inputs[1].val] unused
    # layout = inputs[2] unused
    # device = inputs[3] unused
    # requires_grad = inputs[4] unused
    # out = inputs[5] unused
    if isinstance(size, list):
        size = mb.concat(values=size, axis=0)
    fill = mb.fill(shape=size, value=1.0, name=node.name)
    context.add(fill)


@register_paddle_op
def ones_like(context, node):
    inputs = _get_inputs(context, node, expected=6)
    x = inputs[0]
    if is_current_opset_version_compatible_with(target.iOS16):
        fill = mb.fill_like(ref_tensor=x, value=1.0, name=node.name)
    else:
        size = mb.shape(x=x)
        # dtype = NUM_TO_PADDLE_DTYPE[inputs[1].val] unused
        # layout = inputs[2] unused
        # device = inputs[3] unused
        # requires_grad = inputs[4] unused
        # out = inputs[5] unused
        fill = mb.fill(shape=size, value=1.0, name=node.name)
    context.add(fill)


def _make_fill_op(size, val, name, dtype):
    assert val is not None
    if isinstance(size, list):
        size = mb.concat(values=size, axis=0)
    fill = mb.fill(shape=size, value=val, name=name)
    return fill


@register_paddle_op
def fill_constant(context, node):
    output_name = node.output("Out")[0]

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
def full_like(context, node):
    inputs = _get_inputs(context, node, expected=7)
    x = inputs[0]
    val = inputs[1].val
    if is_current_opset_version_compatible_with(target.iOS16):
        result = mb.fill_like(ref_tensor=x, value=val, name=node.name)
    else:
        size = mb.shape(x=inputs[0])
        result = _make_fill_op(size, val, node.name)
    context.add(result)


@register_paddle_op
def new_full(context, node):
    # The difference between "new_full" and "full" is that the "new_full" is called from
    # an existing tensor: tensor.new_full(size, fill_value), while the "full" is called
    # from the paddle API: paddle.full(size, fill_value).
    # But they are basically doing the same thing.
    inputs = _get_inputs(context, node)
    size = inputs[1]
    val = inputs[2].val
    result = _make_fill_op(size, val, node.name)
    context.add(result)
    
@register_paddle_op
def randint(context, node):
    inputs = _get_inputs(context, node, expected=8)
    low = mb.cast(x=inputs[0], dtype="fp32")
    high = mb.cast(x=inputs[1], dtype="fp32")
    shape = inputs[2]
    rand_uniform = mb.random_uniform(shape=shape, low=low, high=high)
    rand_int = mb.cast(x=rand_uniform, dtype="int32", name=node.name)
    context.add(rand_int)


@register_paddle_op
def bitwise_not(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    dtype = x.dtype
    if types.is_int(dtype):
        x = mb.add(x=x, y=1)
        x = mb.mul(x=x, y=-1, name=node.name)
    elif types.is_bool(dtype):
        x = mb.logical_not(x=x, name=node.name)
    else:
        raise ValueError("Not supported type {} found for 'bitwise_not' op".format(dtype))
    context.add(x)


def _avg_pool(context, node, inputs):
    x = inputs[0]
    kernel_sizes = inputs[1]
    strides = inputs[2]
    if strides.op.op_type == "const" and (not list(strides.val)):
        strides = mb.const(val=kernel_sizes.val, name=strides.name)
    pad_type = "custom"
    # Need to explicitly state L-R, T-B pad
    pad = inputs[3]
    pad = _np.repeat(pad.val, 2)
    ceil_mode = inputs[4].val
    include_pad = inputs[5].val

    spatial_rank = len(pad) // 2
    if spatial_rank > 2 and ceil_mode is True and list(strides.val) != [1] * len(strides.val):
        # since MIL does not support ceil_mode for 3D pool,
        # need to adjust padding values if ceil_mode is True
        # ceil_mode only causes any difference though, if the strides are not 1
        x_spatial_dimensions = x.shape[-spatial_rank:]
        new_pad = _adjust_pad_for_ceil_mode(
            x_spatial_dimensions, kernel_sizes.val, strides.val, pad
        )
        if _np.sum(_np.abs(new_pad - pad)) > 1e-3:
            if include_pad:
                raise ValueError('pool3D with ceil mode=True and include_pad=True not supported')
        pad = new_pad

    pool = mb.avg_pool(
        x=x,
        kernel_sizes=kernel_sizes,
        strides=strides,
        pad_type=pad_type,
        pad=pad,
        name=node.name,
        exclude_padding_from_average=not include_pad,
        ceil_mode=ceil_mode if spatial_rank <= 2 else False,
    )
    context.add(pool)


@register_paddle_op
def avg_pool1d(context, node):
    inputs = _get_inputs(context, node, expected=6)
    _avg_pool(context, node, inputs)


@register_paddle_op
def avg_pool2d(context, node):
    inputs = _get_inputs(context, node, expected=7)
    divisor_override = inputs[6]
    if divisor_override is not None:
        raise ValueError("divisor_override is not supported for avg_pool2d")
    _avg_pool(context, node, inputs)


@register_paddle_op
def avg_pool3d(context, node):
    inputs = _get_inputs(context, node, expected=7)
    divisor_override = inputs[6]
    if divisor_override is not None:
        raise ValueError("divisor_override is not supported for avg_pool3d")
    _avg_pool(context, node, inputs)


@register_paddle_op
def log_softmax(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    axis = inputs[1]
    out = inputs[2]  # Ignored.
    assert out is None
    res = mb.softmax(x=x, axis=axis, name=node.name + "_softmax")
    res = mb.log(x=res, name=node.name)
    context.add(res)


@register_paddle_op(paddle_alias=["nll_loss_nd"])
def nll_loss(context, node):
    inputs = _get_inputs(context, node, expected=5)

    x = inputs[0]
    target = inputs[1]
    weight = inputs[2]
    reduction = inputs[3]
    ignore_index = inputs[4]

    # mapping for reduction
    reduction_mapping = {0: "none", 1: "mean", 2: "sum"}
    reduction = reduction_mapping[reduction.val]

    # compute the weights loss
    batch_size = x.shape[0]

    # only support weight and ignore_index both None
    if weight is not None:
        raise NotImplementedError("Only unity weight is supported for NLLLoss.")
    if ignore_index.val != -100:
        raise NotImplementedError("ignore index not supported for NLLLoss.")

    x = mb.cast(x=x, dtype="fp32")
    x = mb.mul(x=x, y=-1.)
    range_indices = mb.range_1d(end=batch_size, start=0, step=1)
    total_indices = mb.stack(values=[range_indices, target], axis=1)
    loss = mb.gather_nd(x=x, indices=total_indices)

    # reduction type
    if reduction == "none":
        out = mb.identity(x=loss, name=node.name)
    elif reduction == "sum":
        out = mb.reduce_sum(x=loss, axes=[0], keep_dims=False, name=node.name)
    elif reduction == "mean":
        out = mb.real_div(x=loss, y=_np.float(batch_size))
        out = mb.reduce_sum(x=out, axes=[0], keep_dims=False, name=node.name)
    else:
        raise NotImplementedError("Unsupported reduction type for NLLLoss.")

    context.add(out)


@register_paddle_op
def sigmoid(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]

    res = mb.sigmoid(x=x, name=output_name)
    context.add(res)


@register_paddle_op()
def hardsigmoid(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.sigmoid_hard(x=inputs[0], alpha=1.0 / 6, beta=0.5, name=node.name)
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

    # inputs = _get_inputs(context, node, expected=5)
    # x = inputs[0]
    # dim = inputs[1].val

    # if inputs[2] and inputs[2].val is not None:
    #     start = inputs[2].val
    # elif isinstance(inputs[2], Var):
    #     start = inputs[2]
    # else:
    #     start = 0

    # if inputs[3] and inputs[3].val is not None:
    #     end = inputs[3].val
    # elif isinstance(inputs[3], Var):
    #     end = inputs[3]
    # else:
    #     end = None

    # step = inputs[4].val

    

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

    # if isinstance(start, Var):
    #     begin_array = mb.concat(values=begin_array, axis=0)

    # if isinstance(end, Var):
    #     end_array = mb.concat(values=end_array, axis=0)

    kwargs = {
        "x": x,
        "begin": begin_array,
        "end": end_array,
        "end_mask": end_mask,
        "name": output_name,
    }

    # if step != 1:
    #     stride_array = _np.array([1] * len(x.shape))
    #     stride_array[dim] = step
    #     kwargs["stride"] = stride_array

    res = mb.slice_by_index(**kwargs)
    context.add(res)


@register_paddle_op
def split(context, node):
    x = context[node.input("X")[0]]
    # sections_tensor_list = [context[k] for k in node.input("SectionsTensorList")]
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

    #     elif sections_tensor_list:
    #         section = [s.val for s in sections_tensor_list]
    #     elif node.desc.attr("sections"):
    #         section = node.desc.attr("sections")
    #     else:
    #         section = node.desc.attr("sections")
    # else:


    # inputs = _get_inputs(context, node, expected=3)
    # x = inputs[0]
    # split_sizes = inputs[1]
    # dim = inputs[2].val

    # if not isinstance(split_sizes.val, _np.ndarray):
    #     shape = mb.shape(x=x)
    #     dim_size = _list_select(shape, dim)
    #     # MIL split op needs the size of each split to be given explicitly.
    #     num_whole_splits = mb.floor_div(x=dim_size, y=split_sizes)
    #     remainder = mb.mod(x=dim_size, y=split_sizes)

    #     # MIL doesn't have a way of turning a scalar into a tensor (list write
    #     # only supports tensors). As a workaround, we create a constant [1]
    #     # tensor and multiply it by the scalar value, thus creating a tensor
    #     # with the scalar value in it.
    #     tmp = mb.const(val=[1])
    #     whole_sizes = mb.mul(x=tmp, y=split_sizes)
    #     reps = mb.mul(x=tmp, y=num_whole_splits)
    #     whole_sizes = mb.tile(x=whole_sizes, reps=reps)
    #     if remainder.val == 0:
    #         split_sizes = whole_sizes
    #     else:
    #         partial_size = mb.mul(x=tmp, y=remainder)
    #         split_sizes = mb.concat(values=[whole_sizes, partial_size], axis=0)
    # res = mb.split(x=x, split_sizes=split_sizes, axis=dim, name=node.name)
    # context.add(res, paddle_name=node.name)


@register_paddle_op
def unbind(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    dim = inputs[1].val
    split_sizes = [1] * x.shape[dim]
    if len(split_sizes) == 1:
        res = [mb.squeeze(x=x, axes=[dim])]
    else:
        res = mb.split(x=x, split_sizes=split_sizes, axis=dim, name=node.name)
        res = [mb.squeeze(x=x, axes=[dim]) for x in res]
    context.add(res, paddle_name=node.name)


@register_paddle_op
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
        dtype = inputs[2].val
        # non_blocking = inputs[3]
        # copy = inputs[4]
        # memory_format = inputs[5] # usually None
    elif len(inputs) == 5:
        _input = inputs[0]
        dtype = (
            NUMPY_DTYPE_TO_PADDLE_NUM[inputs[1].val.dtype.type]
            if isinstance(inputs[1].val, _np.ndarray)
            else inputs[1].val
        )
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
        context.add(_input, paddle_name=node.name)
        return
    else:
        raise ValueError(
            "Received invalid arguments for PyPaddle conversion of op {}".format(node)
        )

    paddle_dtype = NUM_TO_PADDLE_DTYPE[dtype]
    if isinstance(_input, Var) and _input.val is not None:
        _input = _input.val
        # numpy -> paddle -> paddle cast -> numpy
        # This path is needed to use the mapping of passed in dtypes to paddle dtypes.
        casted_input = paddle.tensor(_input).type(paddle_dtype).cpu().numpy()
        res = mb.const(val=casted_input, name=node.name)
    else:
        res = mb.cast(x=_input, dtype=NUM_TO_DTYPE_STRING[dtype], name=node.name)
    context.add(res)


@register_paddle_op
def erf(context, node):
    inputs = _get_inputs(context, node, expected=1)
    _input = inputs[0]
    erf = mb.erf(x=_input, name=node.name)
    context.add(erf)


@register_paddle_op(paddle_alias=["scalarimplicit"])
def implicittensortonum(context, node):
    inputs = _get_inputs(context, node, expected=1)
    _input = inputs[0]

    if _input.shape == ():  # already a scalar
        context.add(_input, node.name)
    else:
        assert _input.shape == (1,)
        # shape: (1,) -> ()
        squeeze = mb.squeeze(x=_input, name=node.name)
        context.add(squeeze)


@register_paddle_op
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


# @register_paddle_op
# def expand(context, node):
#     def _broadcast_dynamic(name, tensor, shape):
#         # Add any extra dimensions
#         if len(shape) > tensor.rank:
#             new_dims = len(shape) - tensor.rank
#             tensor = mb.expand_dims(x=tensor, axes=list(range(new_dims)))

#         tensor_shape = mb.shape(x=tensor)
#         shape = mb.concat(values=shape, axis=0)
#         reps = mb.real_div(x=shape, y=tensor_shape)
#         reps = mb.cast(x=reps, dtype="int32")
#         res = mb.tile(x=tensor, reps=reps, name=name)
#         return res


#     # PyPaddle 1.6+ has 3 inputs while older version has 2
#     inputs = _get_inputs(context, node, expected=[2, 3])

#     x = inputs[0]
#     shape = inputs[1]

#     if isinstance(shape, list):
#         res = _broadcast_dynamic(node.name, x, shape)
#     else:
#         res = _broadcast(node.name, x, shape.val)
#     context.add(res)

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


@register_paddle_op
def expand_as(context, node):
    # PyPaddle 1.6+ has 3 inputs while older version has 2
    inputs = _get_inputs(context, node, expected=[2, 3])
    x = inputs[0]
    other = inputs[1]

    res = _broadcast(node.name, x, other.shape)
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


@register_paddle_op()
def masked_fill(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    mask = inputs[1]
    value = inputs[2]
    # @mb.select does not properly broadcast scalar input, so as a workaround
    # we create a full sized tensor.

    if types.is_int(value.dtype):
        # @mb.fill cannot handle value with dtype integer
        # so we cast the value.
        value = mb.cast(x=value, dtype="fp32")

    if not types.is_bool(mask.dtype):
        # cond must be bool type
        mask = mb.cast(x=mask, dtype="bool")

    shape = mb.shape(x=x, name=node.name + "_shape")
    value = mb.fill(shape=shape, value=value, name=node.name + "_value")
    res = mb.select(cond=mask, a=value, b=x, name=node.name)
    context.add(res)


@register_paddle_op
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
    if len(inputs) == 1:
        inputs = inputs[0]
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
@register_paddle_op(
    paddle_alias=[
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
    logger.info("Setting pypaddle op: {} to no-op.".format(node))
    inputs = _get_inputs(context, node)
    _input = inputs[0]
    context.add(_input, paddle_name=node.name)


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


@register_paddle_op(paddle_alias=["empty_like"])
def zeros_like(context, node):
    inputs = _get_inputs(context, node, expected=6)
    x = inputs[0]
    dtype = inputs[1].val
    shape = mb.shape(x=x)
    np_type = NUM_TO_NUMPY_DTYPE[dtype]

    if shape.val is not None:
        shape = shape.val
        zeros = _np.zeros(shape).astype(np_type)
        zeros_like = mb.const(val=zeros, name=node.name)
    else:
        value = np_type(0)
        if is_current_opset_version_compatible_with(target.iOS16):
            zeros_like = mb.fill_like(ref_tensor=x, value=value, name=node.name)
        else:
            zeros_like = mb.fill(shape=shape, value=value, name=node.name)

    context.add(zeros_like)


@register_paddle_op(paddle_alias=["empty"])
def zeros(context, node):
    inputs = _get_inputs(context, node)
    size = inputs[0]
    dtype = inputs[1].val
    if isinstance(size, list):
        # the size is dynamic
        size = mb.concat(values=size, axis=0)
        dtype = inputs[1].val
        np_type = NUM_TO_NUMPY_DTYPE[dtype]
        value = np_type(0)
        zeros = mb.fill(shape=size, value=value, name=node.name)
    else:
        # the size is static
        size = size.val
        dtype = inputs[1].val
        # layout = inputs[2] unused
        # device = inputs[3] unused
        # pin_memory = inputs[4] unused

        paddle_dtype = NUM_TO_PADDLE_DTYPE[dtype]
        zeros_array = paddle.zeros(tuple(size)).type(paddle_dtype).numpy()
        zeros = mb.const(val=zeros_array, name=node.name)

    context.add(zeros)


@register_paddle_op(paddle_alias=["new_empty"])
def new_zeros(context, node):
    inputs = _get_inputs(context, node)
    shape = inputs[1]
    if isinstance(shape, list):
        # when the size is dynamic, it is a list of pymil scalar,
        # we need to concat them first to get a shape.
        shape = mb.concat(values=shape, axis=0)
    context.add(mb.fill(shape=shape, value=0., name=node.name))


@register_paddle_op
def dim(context, node):
    inputs = _get_inputs(context, node)
    shape = mb.shape(x=inputs[0])
    rank = mb.shape(x=shape)
    context.add(value_at(rank, 0, node.name))


@register_paddle_op
def min(context, node):
    inputs = _get_inputs(context, node)

    # mimic functionality from https://pypaddle.org/docs/stable/generated/paddle.min.html
    if len(inputs) == 1:
        value = mb.reduce_min(x=inputs[0], axes=None, name=node.name)
        context.add(value)
    elif len(inputs) == 2 and inputs[0].shape == inputs[1].shape:
        value = mb.minimum(x=inputs[0], y=inputs[1], name=node.name)
        context.add(value)
    else:
        assert(len(inputs) <= 3)
        _input = inputs[0]
        dim = inputs[1].val
        keepdim = inputs[2].val if len(inputs) == 3 else False

        values = mb.reduce_min(x=_input, axes=[dim], keep_dims=keepdim)
        indices = mb.reduce_argmin(x=_input, axis=dim, keep_dims=keepdim)
        assert len(node.outputs) == 2
        values_name = node.outputs[0]
        indices_name = node.outputs[1]
        context.add(values, paddle_name=values_name)
        context.add(indices, paddle_name=indices_name)


@register_paddle_op
def max(context, node):
    inputs = _get_inputs(context, node)

    # mimic functionality from https://pypaddle.org/docs/stable/generated/paddle.max.html
    if len(inputs) == 1:
        value = mb.reduce_max(x=inputs[0], axes=None, name=node.name)
        context.add(value)
    elif len(inputs) == 2 and inputs[0].shape == inputs[1].shape:
        value = mb.maximum(x=inputs[0], y=inputs[1], name=node.name)
        context.add(value)
    else:
        assert(len(inputs) <= 3)
        _input = inputs[0]
        dim = inputs[1].val
        keepdim = inputs[2].val if len(inputs) == 3 else False

        values = mb.reduce_max(x=_input, axes=[dim], keep_dims=keepdim)
        indices = mb.reduce_argmax(x=_input, axis=dim, keep_dims=keepdim)
        assert len(node.outputs) == 2
        values_name = node.outputs[0]
        indices_name = node.outputs[1]
        context.add(values, paddle_name=values_name)
        context.add(indices, paddle_name=indices_name)


@register_paddle_op
def argsort(context, node):
    inputs = _get_inputs(context, node, expected=3)
    ascending = mb.logical_not(x=inputs[2])
    argsort = mb.argsort(x=inputs[0], axis=inputs[1], ascending=ascending, name=node.name)
    context.add(argsort)


@register_paddle_op
def sort(context, node):
    inputs = _get_inputs(context, node)
    _input = inputs[0]
    axis = inputs[1].val
    ascending = not inputs[2].val
    indices_name = node.outputs[1]
    values_name = node.outputs[0]
    indices = mb.argsort(x=_input, axis=axis, ascending=ascending, name=indices_name)
    values = mb.gather_along_axis(x=_input, indices=indices, axis=axis, name=values_name)
    context.add(values, paddle_name=values_name)
    context.add(indices, paddle_name=indices_name)


@register_paddle_op
def append(context, node):
    # Note: by applying paddleir_passes.transform_inplace_ops the meaning of
    # this op is changed from the original PaddleIR. This op expects a python
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
        raise ValueError(
            "can only append to Python list or MIL ListVar, got {}.".format(
                type(inputs[0])
            )
        )


@register_paddle_op
def gather(context, node):
    inputs = _get_inputs(context, node)
    res = mb.gather_along_axis(x=inputs[0], indices=inputs[2], axis=inputs[1], name=node.name)
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


@register_paddle_op(paddle_alias=["abs"])
def _abs(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.abs(x=inputs[0], name=node.name))


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
def acos(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.acos(x=inputs[0], name=node.name))


@register_paddle_op
def acosh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.acosh(x=inputs[0], name=node.name))


@register_paddle_op
def asin(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.asin(x=inputs[0], name=node.name))


@register_paddle_op
def atan(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.atan(x=inputs[0], name=node.name))


@register_paddle_op
def atanh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.atanh(x=inputs[0], name=node.name))


@register_paddle_op
def ceil(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.ceil(x=inputs[0], name=node.name))


@register_paddle_op
def clamp(context, node):
    inputs = _get_inputs(context, node, expected=3)
    min_val = inputs[1] if inputs[1] else _np.finfo(_np.float32).min
    max_val = inputs[2] if inputs[2] else _np.finfo(_np.float32).max
    context.add(mb.clip(x=inputs[0], alpha=min_val, beta=max_val, name=node.name))


@register_paddle_op
def triu(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    diagonal = inputs[1]
    diagonal = 0 if diagonal is None else diagonal.val
    if diagonal <= 0:
        res = mb.band_part(x=x, lower=-diagonal, upper=-1, name=node.name)
    else:
        y = mb.band_part(x=x, lower=-1, upper=diagonal - 1)
        res = mb.sub(x=x, y=y, name=node.name)
    context.add(res)


@register_paddle_op
def tril(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    diagonal = inputs[1]
    diagonal = 0 if diagonal is None else diagonal.val
    if diagonal >= 0:
        res = mb.band_part(x=x, lower=-1, upper=diagonal, name=node.name)
    else:
        y = mb.band_part(x=x, lower=-diagonal - 1, upper=-1)
        res = mb.sub(x=x, y=y, name=node.name)
    context.add(res)


@register_paddle_op
def cos(context, node):
    x = context[node.input("X")[0]]
    output = node.output("Out")[0]
    context.add(mb.cos(x=x, name=output))


@register_paddle_op
def cosh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.cosh(x=inputs[0], name=node.name))

    
@register_paddle_op
def exp(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]
    context.add(mb.exp(x=x, name=output_name))


@register_paddle_op
def exp2(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.exp2(x=inputs[0], name=node.name))


@register_paddle_op
def floor(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.floor(x=inputs[0], name=node.name))


@register_paddle_op
def reciprocal(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.inverse(x=inputs[0], name=node.name))


@register_paddle_op
def log(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.log(x=inputs[0], name=node.name))


@register_paddle_op(paddle_alias=["round"])
def _round(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.round(x=inputs[0], name=node.name))


@register_paddle_op
def rsqrt(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.rsqrt(x=inputs[0], name=node.name))


@register_paddle_op
def sin(context, node):
    x = context[node.input("X")[0]]
    output_name = node.output("Out")[0]
    context.add(mb.sin(x=x, name=output_name))


@register_paddle_op
def sinh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.sinh(x=inputs[0], name=node.name))


@register_paddle_op
def asinh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.asinh(x=inputs[0], name=node.name))


@register_paddle_op
def sqrt(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.sqrt(x=inputs[0], name=node.name))


@register_paddle_op
def square(context, node):
    inputs = _get_inputs(context, node, expected=1)
    # mb.square is not supported in some backend
    context.add(mb.mul(x=inputs[0], y=inputs[0], name=node.name))


@register_paddle_op
def tan(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.tan(x=inputs[0], name=node.name))


@register_paddle_op
def tanh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.tanh(x=inputs[0], name=node.name))


@register_paddle_op
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

    mul_node = mb.linear_activation(
        x=gt_node_32,
        alpha=float(threshold_val.val - alpha.val),
        beta=0.,
        name=node.name + '_mul'
    )
    context.add(mul_node)

    final_node = mb.add(x=mul_node, y=threshold_node, name=node.name)
    context.add(final_node)


@register_paddle_op
def sign(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.sign(x=inputs[0], name=node.name))


@register_paddle_op
def is_floating_point(context, node):
    inputs = _get_inputs(context, node, expected=1)
    is_float = types.is_float(inputs[0].dtype)
    context.add(mb.const(val=is_float, name=node.name))


@register_paddle_op()
def logical_and(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = inputs
    x = mb.cast(x=x, dtype="bool")
    y = mb.cast(x=y, dtype="bool")
    context.add(mb.logical_and(x=x, y=y, name=node.name))


@register_paddle_op()
def logical_or(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = inputs
    x = mb.cast(x=x, dtype="bool")
    y = mb.cast(x=y, dtype="bool")
    context.add(mb.logical_or(x=x, y=y, name=node.name))


@register_paddle_op()
def logical_xor(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = inputs
    x = mb.cast(x=x, dtype="bool")
    y = mb.cast(x=y, dtype="bool")
    context.add(mb.logical_xor(x=x, y=y, name=node.name))


def _nonzero_as_tuple(context, node, x):
    '''
    Calculates the non-zero elements of x then slices results by each inner index.
    '''
    non_zero = mb.non_zero(x=x)

    result = []
    for i in range(x.rank):
        result.append(
            mb.slice_by_index(
                x=non_zero,
                begin=[0, i],
                end=[-1, -1], # Ignored, but required
                end_mask=[True, False],
                squeeze_mask=[False, True]
            )
        )

    context.add(result, node.name)


@register_paddle_op
def where(context, node):
    inputs = _get_inputs(context, node)
    
    if len(inputs) == 1:
        _nonzero_as_tuple(context, node, inputs[0])
        return
        
    assert len(inputs) == 3
    cond = inputs[0]
    if not types.is_bool(cond.dtype):
        # cond must be bool type
        cond = mb.cast(x=cond, dtype="bool")
    if not any([any_symbolic(x.shape) for x in inputs[:3]]):
        # broadcast all tensors to the same shape
        broadcast_inputs = _broadcast_tensors([cond, inputs[1], inputs[2]])
        result = mb.select(
            cond=broadcast_inputs[0],
            a=broadcast_inputs[1],
            b=broadcast_inputs[2],
            name=node.name,
        )
    else:
        result = mb.select(cond=cond, a=inputs[1], b=inputs[2], name=node.name)
    context.add(result)
    
    
@register_paddle_op
def nonzero_numpy(context, node):
    inputs = _get_inputs(context, node, expected=1)
    _nonzero_as_tuple(context, node, inputs[0])


@register_paddle_op
def neg(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x, y = promote_input_dtypes([inputs[0], -1])
    context.add(mb.mul(x=x, y=y, name=node.name))


@register_paddle_op
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
    sort = True
    if len(inputs) > 4:
        if inputs[4].val is False and curr_opset_version() < target.iOS16:
            raise Exception("For opset <= iOS16, only sorted=True supported for the topk")
        sort = inputs[4].val

    if len(inputs) > 5:
        if inputs[5] is not None:
            raise Exception(
                "Unsupported value for argument 'out' in topk. Supported values: None, but input "
                "is {}".format(inputs[5].val)
            )

    if is_current_opset_version_compatible_with(target.iOS16):
        kwargs["sort"] = sort

    res = mb.topk(**kwargs)
    values_name = node.outputs[0]
    indices_name = node.outputs[1]
    context.add(res[0], paddle_name=values_name)
    context.add(res[1], paddle_name=indices_name)


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


@register_paddle_op
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

    y = _std(x, axes, keep_dim, unbiased, 0)
    context.add(y, node.name)


@register_paddle_op
def copy(context, node):
    inputs = _get_inputs(context, node, expected=[2, 3])
    context.add(mb.identity(x=inputs[0], name=node.name))


@register_paddle_op
def dtype(context, node):
    inputs = _get_inputs(context, node, expected=1)
    dtype_str = inputs[0].dtype.__name__
    context.add(mb.const(val=dtype_str, name=node.name))


@register_paddle_op
def tensor(context, node):
    def _make_tensor(list_of_tensor, name, rank):
        if rank == 6:
            raise NotImplementedError("CoreML only supports tensor rank <= 5.")
        if not isinstance(list_of_tensor, list):
            return list_of_tensor
        values = [
            _make_tensor(x, name + "_r_" + str(i), rank + 1)
            for i, x in enumerate(list_of_tensor)
        ]
        if len(values) == 1:
            return mb.expand_dims(x=values[0], axes=[0], name=name)
        return mb.stack(values=values, axis=0, name=name)

    inputs = _get_inputs(context, node, expected=4)

    # Case 1: Using paddle.tensor to create a const tensor
    # For example:
    # paddle.tensor([[[0, 0], [0, 10], [5, 10], [5, 0]]], dtype=paddle.float32)
    val = inputs[0]
    if isinstance(val, list):
        context.add(_make_tensor(val, node.name, 1))
        return

    if val.shape != ():
        context.add(mb.identity(x=val, name=node.name))
        return

    # Case 2: Create a tensor filled with a single value
    val = val.val  # element val to fill
    msg_prefix = 'paddle::tensor {} '.format(node.name)
    if val is None:
        raise ValueError(msg_prefix + 'val is None')
    dtype_str = inputs[1].val
    if dtype_str != "fp32":
        raise NotImplementedError(
            msg_prefix + "Unsupported dtype: {}".format(dtype_str)
        )
    # inputs[3] is a bool (not sure what it is)
    shape = mb.shape(x=inputs[2], name=node.name + "_shape")
    context.add(mb.fill(shape=shape, value=val, name=node.name))


"""
Pack and unpack op in pypaddle.
The typical pattern is as following

>>> seq = paddle.tensor([[1,2,0], [3,0,0], [4,5,6]])
>>> lens = [2, 1, 3]
>>> packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
>>> packed
PackedSequence(data=tensor([4, 1, 3, 5, 2, 6]), batch_sizes=tensor([3, 2, 1]),
               sorted_indices=tensor([2, 0, 1]), unsorted_indices=tensor([1, 2, 0]))
>>> seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
>>> seq_unpacked
tensor([[1, 2, 0],
        [3, 0, 0],
        [4, 5, 6]])
>>> lens_unpacked
tensor([2, 1, 3])

source from https://pypaddle.org/docs/stable/generated/paddle.nn.utils.rnn.pad_packed_sequence.html
"""


@register_paddle_op
def _pack_padded_sequence(context, node):
    # The implementation of this op is not efficient. Raise a warning.
    logger.warning("Encountered a _pack_padded_sequence layer. The implementation of translating pack/unpack op\
        in pypaddle is not efficient due to the current limitation of CoreML. Removing the pack-unpack logic \
        and use a fixed batch size model is recommended.")

    inputs = _get_inputs(context, node, expected=3)
    tensor_name, batch_sizes_name = node.outputs
    tensor_input = inputs[0]
    batch_sizes = inputs[1]
    batch_first = inputs[2].val

    # by assuming that the output of this op is always feed in lstm layer,
    # we enforce the layout to be Batch * seq_length * Feature.
    if not batch_first:
        tensor_input = mb.transpose(x=tensor_input, perm=[1, 0, 2])
    context.add(mb.identity(x=tensor_input, name=tensor_name))

    # add the batch_sizes in the context, so that _pad_packed_sequence can
    # find it later.
    context.add(mb.identity(x=batch_sizes, name=batch_sizes_name))


@register_paddle_op
def _pad_packed_sequence(context, node):
    # The implementation of this op is not efficient. Raise a warning.
    logger.warning("Encountered a _pad_packed_sequence layer. The implementation of translating pack/unpack op\
        in pypaddle is not efficient due to the current limitation of CoreML. Removing the pack-unpack logic \
        and use a fixed batch size model is recommended.")
    inputs = _get_inputs(context, node)

    # seq_lengths denotes the actual sequence length for each batch.
    # pad denotes the padding value for those data which has shorter length.
    input_tensor = inputs[0]
    seq_lengths = inputs[1]
    batch_first = inputs[2].val
    pad = inputs[3].val

    # we only support pack and unpack translation for static tensor shape,
    # i.e., the three dimensions are all known during compile time.
    if any([is_symbolic(x) for x in input_tensor.shape]):
        raise NotImplementedError("Only static shape of PackedSequence object is supported.")

    # the input always has batch first layout.
    # padded_seq_len denotes the maximum sequence length across batches.
    batch, padded_seq_len, input_dim = input_tensor.shape
    assert seq_lengths.rank == 1
    assert batch == seq_lengths.shape[0]

    # we iterate through the batch, pad each data, and concate them into a single tensor in the end,
    # which is the total_tensor here.
    # Say the input_tensor has shape [batch , padded_seq_len, input_dim],
    # and the seq_lengths = [len_1, len_2, len_3].
    # Note that in pypaddle, the seq_lengths must be decreasing in order, len_1 >= len_2 >= len_3.
    total_tensor = []

    for i in range(batch):
        # slice for each data
        # x has shape [padded_seq_len, input_dim]
        x = mb.slice_by_index(
            x=input_tensor,
            begin=[i, 0, 0],
            end=[0, 0, 0],
            stride=[1, 1, 1],
            begin_mask=[False, True, True],
            end_mask=[False, True, True],
            squeeze_mask=[True, False, False],
        )

        # get the unpadded sequence,
        # if the unpadded sequence has length seq_length,
        # x would have shape [seq_length, input_dim].
        # For example, the first data would result in a [len_1, input_dim] tensor.
        seq_length = mb.cast(x=value_at(seq_lengths, i), dtype="int32")
        concate_values = [seq_length, input_dim]
        end_index = mb.concat(values=concate_values, axis=0)
        x = mb.slice_by_index(
            x=x,
            begin=[0, 0],
            end=end_index,
            stride=[1, 1],
            begin_mask=[True, True],
            end_mask=[False, True],
        )

        # get the padding part of the data
        # Note that we always add one dummy padding in the end with shape [padded_seq_len - seq_length + 1, input_dim].
        # The reason is that for the case when seq_length = padded_seq_len,
        # coreml cannot handle the empty tensor.
        pad_length = mb.sub(x=padded_seq_len + 1, y=seq_length)
        concate_values = [pad_length, input_dim]
        shape = mb.concat(values=concate_values, axis=0)
        pad_values = mb.fill(shape=shape, value=pad)

        # concate the unpadded sequence and the padding data
        # the resulting tensor would have shape [padded_seq_len + 1, input_dim]
        x, pad_values = promote_input_dtypes([x, pad_values])
        concate_values = [x, pad_values]
        add_values = mb.concat(values=concate_values, axis=0)

        # trim the dummy padding tensor
        # the output would have shpae [padded_seq_len, input_dim]
        x = mb.slice_by_index(
            x=add_values,
            begin=[0, 0],
            end=[padded_seq_len, 0],
            stride=[1, 1],
            begin_mask=[True, True],
            end_mask=[False, True],
        )

        # add it to total tensor
        total_tensor.append(x)

    # transpose the tensor if batch_first = False
    if not batch_first:
        x = mb.stack(values=total_tensor, axis=0)
        x = mb.transpose(x=x, perm=[1, 0, 2], name=node.name)
    else:
        x = mb.stack(values=total_tensor, axis=0, name=node.name)

    context.add(x)


@register_paddle_op
def log10(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    log_x = mb.log(x=x)
    context.add(mb.mul(x=log_x, y=1 / _np.log(10.0)), node.name)


@register_paddle_op
def log2(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    log_x = mb.log(x=x)
    context.add(mb.mul(x=log_x, y=1 / _np.log(2.0)), node.name)


@register_paddle_op
def flip(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = mb.reverse(x=inputs[0], axes=inputs[1], name=node.name)
    context.add(x, node.name)


@register_paddle_op(paddle_alias=["reflection_pad1d"])
def reflection_pad2d(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    paddle_pad = inputs[1].val
    pad_flipped = paddle_pad.reshape((-1, 2))[::-1].ravel()
    pad = _np.pad(pad_flipped, (len(x.shape) * 2 - len(pad_flipped), 0))
    context.add(mb.pad(x=x, pad=pad, mode='reflect'), node.name)


@register_paddle_op(paddle_alias=["replication_pad1d"])
def replication_pad2d(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    paddle_pad = inputs[1].val
    pad_flipped = paddle_pad.reshape((-1, 2))[::-1].ravel()
    pad = _np.pad(pad_flipped, (len(x.shape) * 2 - len(pad_flipped), 0))
    context.add(mb.pad(x=x, pad=pad, mode='replicate'), node.name)


def _broadcast_tensors(tensors):
    def _solve_broadcast_shape(shapes):
        rank = _np.max([len(shape) for shape in shapes])
        shapes = [[1] * (rank - len(shape)) + shape for shape in shapes]
        result_shape = []
        for i in range(rank):
            dims = [shapes[j][i] for j in range(len(tensors))]
            if any_symbolic(dims):
                # rdar://85559497 (Handle dynamic shapes inputs broadcast for pypaddle)
                raise NotImplementedError(
                    "Only static shaped inputs are supported for paddle.broadcast_tensors conversion."
                )
            result_shape.append(_np.max(dims))
        return result_shape

    if len(tensors) == 1:
        return tensors

    # solve the broadcast shape
    input_shapes = [list(x.shape) for x in tensors]
    broadcast_shape = _solve_broadcast_shape(input_shapes)

    # do the broadcasting
    results = []
    for tensor in tensors:
        name = tensor.name + "_after_broadcast"
        results.append(_broadcast(name, tensor, broadcast_shape))
    return results


@register_paddle_op
def broadcast_tensors(context, node):
    inputs = _get_inputs(context, node)
    context.add(_broadcast_tensors(inputs[0]), node.name)


def _scatter(context, inputs, mode, name):
    data = inputs[0]
    axis = inputs[1].val
    indices = inputs[2]
    updates = inputs[3]
    if types.is_scalar(updates.sym_type):
        updates = mb.fill(shape=indices.shape, value=updates.val, name=name)
    result = mb.scatter_along_axis(data=data, indices=indices, updates=updates,
                                   axis=axis, mode=mode, name=name)
    context.add(result)


@register_paddle_op
def scatter(context, node):
    inputs = _get_inputs(context, node)
    assert len(inputs) in (4, 5)

    # Determine reduce/mode parameter
    if len(inputs) == 5:
        mode = inputs[4].val
        if mode == 'multiply':
            mode = 'mul'
        else:
            assert mode == 'add'
    else:
        mode = 'update'

    _scatter(context, inputs, mode, node.name)


@register_paddle_op
def scatter_add(context, node):
    inputs = _get_inputs(context, node)
    _scatter(context, inputs, 'add', node.name)


@register_paddle_op
def baddbmm(context, node):
    """
    baddbmm(Tensor input, Tensor batch1, Tensor batch2, Scalar beta=1, Scalar alpha=1)
    output = beta * input + alpha * batch1 * batch2

    Notice that batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
    If batch1 is a (b×n×m) tensor, batch2 is a (b×m×p) tensor, then input must be broadcastable with a (b×n×p) tensor
    and out will be a (b×n×p) tensor.
    """
    assert len(node.outputs) == 1
    inputs = _get_inputs(context, node, expected=5)
    bias, batch1, batch2, beta, alpha = inputs

    if beta.val != 1.0:
        # Apply scaling factor beta to the bias.
        bias = mb.mul(x=beta, y=bias, name=bias.name + "_scaled")
        context.add(bias)

    if alpha.val != 1.0:
        # Apply scaling factor alpha to the input.
        batch1 = mb.mul(x=alpha, y=batch1, name=batch1.name + "_scaled")
        context.add(batch1)

    bmm_node = mb.matmul(x=batch1, y=batch2, name=node.name + "_bmm")
    context.add(bmm_node)

    baddbmm_node = mb.add(x=bias, y=bmm_node, name=node.name)
    context.add(baddbmm_node)


@register_paddle_op
def glu(context, node):
    """
    glu(Tensor input, Scalar dim=-1)
    Applies the gated linear unit function GLU(a,b)=a⊗σ(b) where a is the first half of the input matrices and b is the
    second half.
    """
    assert len(node.outputs) == 1
    inputs = _get_inputs(context, node, expected=2)
    input, axis = inputs

    first_half, second_half = mb.split(x=input, num_splits=2, axis=axis.val, name=node.name + "_split")
    context.add(first_half)
    context.add(second_half)

    sigmoid_second_half = mb.sigmoid(x=second_half, name=second_half.name + "_sigmoid")
    context.add(sigmoid_second_half)

    glu_node = mb.mul(x=first_half, y=sigmoid_second_half, name=node.name)
    context.add(glu_node)


@register_paddle_op
def hstack(context, node):
    """
    hstack(List[Tensor] tensors, Optional[Tensor] out)
    Stack tensors in sequence horizontally (column wise). This is equivalent to concatenation along the first axis for
    1-D tensors, and along the second axis for all other tensors.
    """
    inputs = _get_inputs(context, node)
    tensors = inputs[0]
    input_shapes = [list(x.shape) for x in tensors]
    # Concatenates along the first axis for 1-D tensors, and along the second axis for all other tensors.
    axis = 0 if len(input_shapes[0]) == 1 else 1
    hstack_node = mb.concat(values=tensors, axis=axis, name=node.name)
    context.add(hstack_node)


@register_paddle_op
def remainder(context, node):
    """
    remainder(Tensor dividend, Tensor divisor, Optional[Tensor] out)
    Computes Python’s modulus operation entrywise. The result has the same sign as the divisor and its absolute value
    is less than that of divisor. It may also be defined in terms of paddle.div() as:
    remainder(a, b) == a - a.div(b, rounding_mode="floor") * b
    """
    # Don't specify `expected` because the parameter `out` is optional.
    inputs = _get_inputs(context, node)
    dividend, divisor = inputs[0], inputs[1]
    div_node = mb.floor_div(x=dividend, y=divisor, name=node.name + "_div")
    context.add(div_node)
    scaled_div = mb.mul(x=div_node, y=divisor, name=div_node.name + "_scaled")
    context.add(scaled_div)
    remainder_node = mb.sub(x=dividend, y=scaled_div, name=node.name)
    context.add(remainder_node)


@register_paddle_op
def hann_window(context, node):
    inputs = _get_inputs(context, node, expected=[5, 6])
    if inputs[0].val is None:
        raise NotImplementedError("variable 'window_length' not supported.")

    periodic = True
    if len(inputs) == 6:
        if inputs[1].val is None:
            raise NotImplementedError("variable 'periodic' not supported.")
        if not inputs[1].val:
            periodic = False

    size = (inputs[0].val,)
    if inputs[0].val <= 1:
        one = mb.fill(shape=size, value=1.0, name=node.name)
        context.add(one)
        return

    ones = mb.fill(shape=size, value=1.0)
    cum = mb.cumsum(x=ones, axis=0)
    seq = mb.sub(x=cum, y=ones)
    pi = mb.fill(shape=size, value=_math.pi)
    window_length_float = mb.cast(x=inputs[0], dtype="fp32")
    if not periodic:
        window_length_float = mb.sub(x=window_length_float, y=ones)
    denominator = mb.fill(shape=size, value=window_length_float)
    numerator = mb.mul(x=seq, y=pi)
    frac = mb.real_div(x=numerator, y=denominator)
    sin = mb.sin(x=frac)
    sin_sq = mb.mul(x=sin, y=sin, name=node.name)
    context.add(sin_sq)

@register_paddle_op
def trace(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    dims = mb.shape(x=x)
    dim0 = value_at(dims, 0)
    dim1 = value_at(dims, 1)
    min_dim = mb.minimum(x=dim0, y=dim1)
    indices = mb.range_1d(end=min_dim, start=0, step=1)
    indices = mb.stack(values=[indices, indices], axis=1)
    diagonal = mb.gather_nd(x=x, indices=indices)
    trace = mb.reduce_sum(x=diagonal, name=node.name)
    context.add(trace)

@register_paddle_op
def roll(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    shift = inputs[1]
    if inputs[2].val:
        raise NotImplementedError('"dims" parameter of "roll" op is not supported.')
    shape = mb.shape(x=x)
    flatten = mb.reshape(x=x, shape=[-1])

    dim_prod = mb.reduce_prod(x=shape)
    start_idx = mb.sub(x=dim_prod, y=shift)
    indices0 = mb.range_1d(end=dim_prod, start=start_idx, step=1)
    indices1 = mb.range_1d(end=start_idx, start=0, step=1)
    indices = mb.concat(values=[indices0, indices1], axis=0)
    x = mb.gather(x=flatten, indices=indices, name=node.name)
    x = mb.reshape(x=x, shape=shape, name=node.name)
    context.add(x)


@register_paddle_op
def im2col(context, node):
    """
    This op is used by paddle.nn.Unfold, which extracts sliding local blocks from a batched input
    tensor.

    It only supports rank-4 tensor (consistent with PyPaddle) with dilation and stride both set to 1.
    More flexbible dilation and stride support will be added in the future.
    """
    inputs = _get_inputs(context, node, expected=5)
    input_data = inputs[0]
    kernel_size = inputs[1].val
    dilation = inputs[2].val
    padding = inputs[3].val
    stride = inputs[4].val

    if input_data.rank != 4:
        raise ValueError("Only supports rank=4 input data for im2col (unfold).")
    if not (dilation[0] == 1 and dilation[1] == 1):
        raise ValueError("Only supports dilation=1 for im2col (unfold).")
    if not (stride[0] == 1 and stride[1] == 1):
        raise ValueError("Only supports stride=1 for im2col (unfold).")

    if padding[0] != 0 or padding[1] != 0:
        # The order for padding is reversed for unfold compared to what is used for padding, i.e.,
        # pad expects the layout (padding_left, padding_right, padding_top, padding_bottom). But in
        # the functions from `unfold` to `im2col`, it uses the first index to be the vertical
        # padding and second for horizontal: (padding[1], padding[1], padding[0], padding[0]).
        pad_dim = _np.flip(padding).repeat(2)
        pad_dim = pad_dim.reshape((-1, 2))[::-1].reshape(-1).tolist()
        missing_dims = input_data.rank - (len(pad_dim) // 2)
        pad_dim = [0, 0] * missing_dims + pad_dim
        input_data = mb.pad(x=input_data, pad=pad_dim, mode="constant")

    N, C, H, W = input_data.shape

    # Get total number of blocks. It follows the formula at paddle.nn.Unfold documentation.
    sptial_size = (H, W)
    block_count = 1
    for i in range(2):
        block_count *= (
            _np.floor(
                (sptial_size[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]
            ).astype(_np.int32)
            + 1
        )

    # Get batch block indices.
    batch_idx = _np.arange(N)[:, None, None] * C * H * W

    # Get starting block indices.
    start_idx = _np.arange(kernel_size[0])[None, :, None] * W + _np.arange(
        kernel_size[1]
    )

    # Generate depth indices.
    channel_index = H * W * _np.arange(C)
    start_idx = (channel_index[None, :, None] + _np.ravel(start_idx)).reshape(
        (-1, kernel_size[0], kernel_size[1])
    )

    # Get offsetted indices across the height and width of input array.
    row_extent = H - kernel_size[0] + 1
    col_extent = W - kernel_size[1] + 1
    offset_idx = _np.arange(row_extent)[None, :, None] * W + _np.arange(col_extent)
    indices = _np.ravel(start_idx)[:, None] + _np.ravel(offset_idx)

    # Gather batches together.
    indices = batch_idx + indices
    input_data = mb.reshape(x=input_data, shape=[-1])
    gathered_data = mb.gather_along_axis(
        x=input_data, indices=indices.reshape(-1), axis=0
    )
    block_size = C * kernel_size[0] * kernel_size[1]
    output = mb.reshape(
        x=gathered_data, shape=(N, block_size, block_count), name=node.name
    )

    context.add(output)
