#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import builtins
import math as _math
import numbers
import re
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple, Union

import numpy as _np
import numpy as np
import torch
from tqdm import tqdm as _tqdm

from coremltools import _logger as logger
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target
from coremltools.converters.mil.frontend import _utils
from coremltools.converters.mil.frontend.milproto.load import TranscriptionContext
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Symbol, types
from coremltools.converters.mil.mil.block import is_current_opset_version_compatible_with
from coremltools.converters.mil.mil.ops.defs._utils import (
    MAX_SIZE_CONSTANT_FOLDING,
    promote_input_dtypes,
    solve_slice_by_index_shape,
)
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry
from coremltools.converters.mil.mil.scope import ScopeInfo, ScopeSource
from coremltools.converters.mil.mil.types import is_bool, nptype_from_builtin
from coremltools.converters.mil.mil.types.symbolic import any_symbolic, is_symbolic
from coremltools.converters.mil.mil.types.type_mapping import builtin_to_string
from coremltools.converters.mil.mil.var import ListVar, Var

from .._utils import build_einsum_mil, value_at
from .internal_graph import InternalTorchIRGraph, InternalTorchIRNode
from .torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from .utils import (
    NUM_TO_DTYPE_STRING,
    NUM_TO_NUMPY_DTYPE,
    NUM_TO_TORCH_DTYPE,
    NUMPY_DTYPE_TO_TORCH_NUM,
    TORCH_DTYPE_TO_NUM,
    TORCH_EXPORT_BASED_FRONTENDS,
    TYPE_TO_DTYPE_STRING,
    TorchFrontend,
    dtype_to_32bit,
)

# The pytorch args for many of the below ops were sourced from
# https://github.com/pytorch/pytorch/blob/d971007c291c0ead1003d12cd553d18ddb582207/torch/csrc/jit/mobile/register_mobile_ops.cpp#L216


# Max int64 value. Used as a default value in many PyTorch functions.
PYTORCH_DEFAULT_VALUE = 2**63 - 1

VALUE_CLOSE_TO_INFINITY = 1e+38

TORCH_STRING_ARGS = {
    # div rounding mode
    "floor",
    "trunc",
    # conv padding
    "same",
    "valid",

    # meshgrid indexing
    "ij",
    "xy",

    # pad mode
    "circular",
    "constant",
    "reflect",
    "replicate",

    # norm
    "fro",

    # searchsorted side
    "left",
    "right",
}


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


def convert_nodes(
    context: TranscriptionContext,
    graph: InternalTorchIRGraph,
    early_exit: Optional[bool] = True,
) -> None:
    """
    Iterate over the nodes of a graph or block and convert to MIL.

    Arguments:
        context: A TranscriptionContext object to pull node inputs and assign node outputs.
        graph: An InternalTorchIRGraph or InternalTorchIRBlock object.
    """
    for node in _tqdm(graph.nodes, desc="Converting PyTorch Frontend ==> MIL Ops", unit=" ops"):
        try:
            convert_single_node(context, node)
        except Exception as e:
            scope_names = node.get_scope_info()[0]
            op_location = '/'.join(scope_names)
            logger.error(f"\n\nERROR - converting '{node.kind}' op (located at: '{op_location}'):\n")
            raise e     # re-raise exception

        if early_exit and _all_outputs_present(context, graph):
            # We've generated all the outputs the graph needs, terminate conversion.
            break


def convert_single_node(context: TranscriptionContext, node: InternalTorchIRNode) -> None:
    """
    Converts a single lowered PyTorch op to MIL.

    Arguments:
        context: A TranscriptionContext object to pull node inputs and assign node outputs.
        node: lowered PyTorch op to convert.
    """
    op_lookup = node.kind
    add_op = _TORCH_OPS_REGISTRY.get_func(op_lookup)
    if add_op is None:
        if re.match(r".*_dynamic", op_lookup):
            raise RuntimeError(
                f"PyTorch convert function for op '{op_lookup}' not implemented.\n"
                "Dynamic quantized models are not supported by Core ML.\n"
                "Please use static quantization or the APIs in coremltools.optimize to quantize/compress models."
            )
        else:
            raise RuntimeError(
                f"PyTorch convert function for op '{op_lookup}' not implemented."
            )

    logger.info("Converting op {} : {}".format(node.name, op_lookup))

    scopes = []
    if context.frontend == TorchFrontend.TORCHSCRIPT:
        scope_name, scope_type = node.get_scope_info()
        scopes = [
            ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=scope_type),
            ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_NAME, data=scope_name),
        ]
    elif context.frontend in TORCH_EXPORT_BASED_FRONTENDS:
        scopes = [
            ScopeInfo(source=ScopeSource.EXIR_STACK_TRACE, data=[node.meta.get("stack_trace")])
        ]
        if context.frontend == TorchFrontend.EXECUTORCH:
            scopes.append(
                ScopeInfo(
                    source=ScopeSource.EXIR_DEBUG_HANDLE, data=[node.meta.get("debug_handle")]
                )
            )
    else:
        raise ValueError(f"Invalid PyTorch frontend {context.frontend}")

    with mb.scope(*scopes):
        if context.frontend == TorchFrontend.TORCHSCRIPT:
            context.quant_context.maybe_handle_quantized_inputs(node)

        # Only torch script needs to prepare
        if context.frontend == TorchFrontend.TORCHSCRIPT:
            context.convert_input_to_tensor_type(node)

        add_op(context, node)
        if _TORCH_OPS_REGISTRY.is_inplace_op(op_lookup):
            context.process_inplace_op(node)


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


def _assert_torch_dtype_num_is_not_complex_number(num):
    # 9 is torch.complex64 or torch.cfloat
    # 10 is torch.complex128 or torch.cdouble
    assert num is None or num.val is None or num.val not in (9, 10), \
        "This op does not support complex number dtype."


def _get_bindings(context, alist) -> List[Var]:
    """
    This utility is needed in order to handle following cases:
        With EXIR,
        - Some of the inputs can be literals (like axis, perms) and thus can be of types: list, int etc.
        - An Input Parameter of an op could be a list/tuple similar to our concat layer
    """
    results = []

    for i in alist:
        if isinstance(i, str):
            """
            torch.export args can be type of string:
            1. torch script strictly has name: variable mapping
            2. certain torch.export args can be strings,
               e.g. pad type. Since those args were equivalently enums, we introduced TORCH_STRING_ARGS
            3. some torch.export args can be arbitrary string, hence we just raise an warning.

            rdar://138953006 ([Torch.Export] Enforce `Name: Variable` Mapping in Context)
            The above radar is tracking if all string can fit in the context.
            """
            if i in context:
                results.append(context[i])
            elif i in TORCH_STRING_ARGS:
                results.append(i)
            else:
                results.append(i)
                logger.warning(
                    f"Binding {i} is neither a name of exisitng var in context, "
                    "nor a torch string argument."
                )
        elif isinstance(i, (list, tuple)) and all(isinstance(j, int) for j in i):
            results.append(mb.const(val=i))
        elif isinstance(i, (list, tuple)):
            results.append(_get_bindings(context, i))
        elif isinstance(i, (int, float)):
            results.append(mb.const(val=i))
        elif i is None:
            results.append(None)
        else:
            raise NotImplementedError(f"Binding of inputs of type {type(i)} not handled yet")

    return results


def _get_inputs(
    context,
    node,
    expected: Union[int, List, Tuple, Dict[TorchFrontend, int]] = None,
    min_expected: Union[int, Dict[TorchFrontend, int]] = None,
) -> List[Var]:
    """
    Look up a node's inputs in @context and return them as a list. If
    @expected is not None, also verifies the number of inputs matches the
    value of @expected.
    """

    def check_if_number_of_inputs_expected(num_inputs: int, expected: Union[int, List, Tuple]) -> None:
        expected = [expected] if isinstance(expected, int) else expected
        if num_inputs not in expected:
            raise ValueError(
                f"node {node.name} ({node.kind}) got {num_inputs} input(s), expected {expected}"
            )

    def check_if_number_of_inputs_more_than_min_expected(num_inputs: int, min_expected: int) -> None:
        if num_inputs < min_expected:
            raise ValueError(
                f"node {node.name} ({node.kind}) got {num_inputs} input(s), "
                f"expected minimum {min_expected} inputs"
            )

    inputs = _get_bindings(context, node.inputs)

    if expected is not None:
        if isinstance(expected, dict):
            if context.frontend in expected:
                check_if_number_of_inputs_expected(len(inputs), expected[context.frontend])
        else:
            check_if_number_of_inputs_expected(len(inputs), expected)

    if min_expected is not None:
        if isinstance(min_expected, dict):
            if context.frontend in min_expected:
                check_if_number_of_inputs_more_than_min_expected(len(inputs), min_expected[context.frontend])
        else:
            check_if_number_of_inputs_more_than_min_expected(len(inputs), min_expected)

    return inputs


def _get_kwinputs(context, node, keyword: str, default: Optional[List[Var]] = None) -> List[Var]:
    if node.kwinputs is None:
        return default
    else:
        bindings = node.kwinputs.get(keyword)
        if bindings is None:
            return default
        else:
            return _get_bindings(context, bindings)


def _list_select(shape_var, index):
    """
    Sometimes we need to select a specific item from a list. If that item
    is known at compile time, extract it as a const. Otherwise, if it's
    symbolic, use gather.
    """
    if shape_var.can_be_folded_to_const():
        res = mb.const(val=shape_var.val[index])
    else:
        if is_current_opset_version_compatible_with(target.iOS17):
            # IOS17 `gather` requires non-negative indices.
            index = mb.select(
                cond=mb.greater_equal(x=index, y=0),
                a=index,
                b=mb.add(x=index, y=value_at(mb.shape(x=shape_var), 0)),
            )
        res = mb.gather(x=shape_var, indices=index)
    return res

def _is_const(var, optional=False):
    """
    Check if a var is a const.
    It could be `const` or `constexpr_` ops.
    """
    if optional and var is None:
        return True
    if isinstance(var, np.ndarray):
        return True
    return (
        var is not None
        and isinstance(var, Var)
        and var.op is not None
        and (
            var.op.op_type.startswith("constexpr_")
            or (var.op.op_type == "dequantize" and var.op.can_materialize_val())
            or var.val is not None
        )
    )

def _create_linear_layer(x, w, bias):
    """
    Utility to translate linear layer.
    Since the linear layer can only take `const` or `constexpr_` weight as input,
    for other cases, we implement the linear layer through matmul.

    For instance, given a torch model with an int8 weight:

    int8_weight -> transpose -> reshape -> linear

    If we directly use `mb.linear`, it is going to produce compilation error at the runtime.
    """
    if _is_const(w) and _is_const(bias, optional=True):
        return mb.linear(x=x, weight=w, bias=bias)
    res = mb.matmul(x=x, y=w, transpose_y=True)
    if bias is not None:
        res = mb.add(x=res, y=bias)
    return res

def _construct_constant(val, name):
    # Converter cannot handle torch tensors.
    if isinstance(val, torch.Tensor):
        val = val.cpu().numpy()

    # MIL casts ints to int32, which can't represent PyTorch's default value.
    # So we instead represent it with None, and any ops that might get the
    # value will check for None instead.
    if isinstance(val, int) and val == PYTORCH_DEFAULT_VALUE:
        val = None

    # Pytorch uses inf
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


def _cast_to(_input: Union[Var, np.ndarray], dtype_str: str, node_name: str) -> Var:
    """Create a Var to cast from input to target dtype."""
    valid_dtypes = SSAOpRegistry._get_core_op_cls("cast").supported_dtypes()
    if dtype_str in valid_dtypes:
        res = mb.cast(x=_input, dtype=dtype_str)
    else:
        np_dtype = types.nptype_from_builtin(types.string_to_builtin(dtype_str))
        if _np.issubdtype(np_dtype, _np.integer):
            target_dtype = "int32"
        elif _np.issubdtype(np_dtype, _np.floating):
            target_dtype = "fp32"
        else:
            raise ValueError(f"Unsupported `to` op ({node_name}) with target dtype {np_dtype}")
        logger.warning(
            f"The {dtype_str} is not supported by cast op. Will do best-effort cast to {target_dtype}"
        )
        res = mb.cast(x=_input, dtype=target_dtype)
    return res


@register_torch_op
def native_dropout(context, node):
    if context.frontend in TORCH_EXPORT_BASED_FRONTENDS:
        inputs = _get_inputs(context, node, min_expected=2)
        context.add((inputs[0],), node.name)
    else:
        raise ValueError(f"native_dropout should only appear in EXIR, but got {context.frontend}")


@register_torch_op
def affine_grid_generator(context, node):
    # rdar://73165386 (Improve error handling of coremltools "affine" op PyTorch conversion.)

    affine_op_name = node.name
    theta, size, align_corners = _get_inputs(context, node, expected=3)

    # note: only add consts here as PyTorch uses affine_grid + grid_sampler together
    is_theta_const = theta.val is not None
    if is_theta_const:
        context.add(mb.const(val=theta.val, name="{}_theta".format(affine_op_name)))
    else:  # theta is dynamic input, keep track of it's name
        context.add(mb.const(val=theta.name, name="{}_theta".format(affine_op_name)))

    context.add(mb.const(val=size.val, name="{}_size".format(affine_op_name)))
    context.add(mb.const(val=align_corners.val, name="{}_align_corners".format(affine_op_name)))


@register_torch_op
def grid_sampler(context, node):
    affine_op_name = node.inputs[1]
    # https://github.com/pytorch/pytorch/blob/00d432a1ed179eff52a9d86a0630f623bf20a37a/aten/src/ATen/native/GridSampler.h#L10-L11
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


@register_torch_op
def silu(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = mb.silu(x=inputs[0], name=node.name)
    context.add(x)


@register_torch_op
def constant(context, node):
    assert len(node.inputs) == 0
    assert len(node.outputs) == 1

    name = node.name
    val = node.attr["value"]

    const = _construct_constant(val, name)
    context.add(const, torch_name=name)


@register_torch_op
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


@register_torch_op
def selu(context, node):
    ALPHA = 1.6732632423543772
    SCALE = 1.0507009873554805

    x = _get_inputs(context, node, expected=1)[0]
    x = mb.elu(x=x, alpha=ALPHA)
    x = mb.mul(x=x, y=SCALE, name=node.name)
    context.add(x)


@register_torch_op
def dot(context, node):
    inputs = _get_inputs(context, node, expected=2)
    xy = mb.mul(x=inputs[0], y=inputs[1])
    sum_xy = mb.reduce_sum(x=xy, axes=[0])
    context.add(sum_xy, node.name)


@register_torch_op
def mv(context, node):
    inputs = _get_inputs(context, node, expected=2)
    expand = mb.expand_dims(x=inputs[1], axes=[-1], name=node.name + "_expanded")
    mv = mb.matmul(x=inputs[0], y=expand, name=node.name + "_mv")
    res = mb.squeeze(x=mv, axes=[-1], name=node.name)
    context.add(res)


@register_torch_op
def outer(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = mb.reshape(x=inputs[0], shape=[-1, 1])
    y = mb.reshape(x=inputs[1], shape=[1, -1])
    res = mb.matmul(x=x, y=y, name=node.name)
    context.add(res)


@register_torch_op
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


@register_torch_op
def frobenius_norm(context, node):
    x, dim, keep_dims = _get_inputs(context, node, expected=3)
    result = mb.reduce_l2_norm(x=x, axes=dim, keep_dims=keep_dims, name=node.name)
    context.add(result)


@register_torch_op
def norm(context, node):
    x, num, dim, keep_dims = _get_inputs(context, node, expected=4)
    assert x is not None and keep_dims is not None and num is not None and dim is not None
    temp = _vector_norm(x=x, order=num, dim=dim, keep_dims=keep_dims, name=node.name)
    context.add(temp)


def _vector_norm(x, order, dim, keep_dims, name):
    if isinstance(order, Var):
        order = order.val
    # 0 norm is special
    if order == 0:
        # sum(x!=0)
        x = mb.cast(x=x, dtype="fp32")
        temp = mb.not_equal(x=x, y=0.)
        temp = mb.cast(x=temp, dtype='int32')
        temp = mb.reduce_sum(x=temp, axes=dim, keep_dims=keep_dims, name=name)
    # infinity norm is special
    elif order > VALUE_CLOSE_TO_INFINITY:
        # max(abs(x))
        temp = mb.abs(x=x)
        temp = mb.reduce_max(x=temp, axes=dim, keep_dims=keep_dims, name=name)
    # -infinity norm is special
    elif order < -VALUE_CLOSE_TO_INFINITY:
        # min(abs(x))
        temp = mb.abs(x=x)
        temp = mb.reduce_min(x=temp, axes=dim, keep_dims=keep_dims, name=name)
    # Although 2 norm can fit in the general formula,
    # since it is very common, we have tailored kernel for it
    elif order == 2:
        temp = mb.reduce_l2_norm(x=x, axes=dim, keep_dims=keep_dims, name=name)
    # use general formula to compute all other norms
    else:
        # sum(abs(x)^{order})^{(1 / order)}
        temp = mb.abs(x=x)
        x, y = promote_input_dtypes([temp, order])
        temp = mb.pow(x=x, y=y)
        temp = mb.reduce_sum(x=temp, axes=dim, keep_dims=keep_dims)
        temp = mb.pow(x=temp, y=1.0 / order, name=name)
    return temp


@register_torch_op
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


@register_torch_op
def narrow(context, node):
    x, dim, start, length = _get_inputs(context, node, expected=4)

    begin = [0] * len(x.shape)
    begin[dim.val] = start.val

    end = list(x.shape)
    end[dim.val] = start.val + length.val

    context.add(
        mb.slice_by_index(x=x, begin=begin, end=end, name=node.name)
    )


@register_torch_op
def linalg_vector_norm(context, node):
    inputs = _get_inputs(
        context,
        node,
        expected={TorchFrontend.TORCHSCRIPT: 5},
        min_expected={TorchFrontend.TORCHEXPORT: 1, TorchFrontend.EXECUTORCH: 1},
    )
    nargs = len(inputs)

    x = inputs[0]
    order = 2 if nargs < 2 else inputs[1]
    dim = None if nargs < 3 else inputs[2]
    keepdim = False if nargs < 4 else inputs[3]
    assert x is not None and keepdim is not None and order is not None

    temp = _vector_norm(x=x, order=order, dim=dim, keep_dims=keepdim, name=node.name)
    context.add(temp)


@register_torch_op
def linalg_matrix_norm(context, node):
    x, order, dim, keep_dims, _ = _get_inputs(context, node, expected=5)
    assert x is not None and keep_dims is not None and order is not None and dim is not None
    assert len(dim.val) == 2
    temp = _matrix_norm(x=x, order=order, dim=dim.val, keep_dims=keep_dims, name=node.name)
    context.add(temp)


@register_torch_op(torch_alias=["linalg_norm.ord_str"])
def linalg_norm(context, node):
    inputs = _get_inputs(
        context,
        node,
        expected={TorchFrontend.TORCHSCRIPT: 5},
        min_expected={TorchFrontend.TORCHEXPORT: 1, TorchFrontend.EXECUTORCH: 1},
    )
    nargs = len(inputs)

    x = inputs[0]
    order = None if nargs < 2 else inputs[1]
    dim = None if nargs < 3 else inputs[2]
    keepdim = False if nargs < 4 else inputs[3]

    assert x is not None and keepdim is not None
    if dim is None:
        dim = _np.arange(x.rank)
    else:
        dim = dim.val

    if order is None or order == "fro":
        temp = mb.reduce_l2_norm(x=x, axes=dim, keep_dims=keepdim, name=node.name)
    elif len(dim) == 2:
        temp = _matrix_norm(x=x, order=order, dim=dim, keep_dims=keepdim, name=node.name)
    else:
        temp = _vector_norm(x=x, order=order, dim=dim, keep_dims=keepdim, name=node.name)
    context.add(temp)


@register_torch_op
def hardswish(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]

    w = mb.thresholded_relu(x=x, alpha=-3.0)
    y = mb.sigmoid_hard(
        x=w, alpha=1.0 / 6, beta=0.5
    )  # ``y = min(max(alpha * x + beta, -1), 1)
    result = mb.mul(x=w, y=y, name=node.name)

    context.add(result)


@register_torch_op
def reshape_as(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    ref = inputs[1]
    shape = mb.shape(x=ref)
    result = mb.reshape(x=x, shape=shape, name=node.name)
    context.add(result)


@register_torch_op
def unflatten(context, node):
    x, dim_var, unflattened_size_var = _get_inputs(context, node, expected=3)

    dim = dim_var.val
    if dim is None:
        raise ValueError("In 'unflatten' op, the 'dim' must be provided.")
    if dim < 0:
        dim += x.rank

    x_shape = mb.shape(x=x)
    pre_shape = mb.slice_by_index(x=x_shape, begin=[0], end=[dim])
    post_shape = mb.slice_by_index(x=x_shape, begin=[dim + 1], end=[len(x.shape)])
    target_shape = mb.concat(values=(pre_shape, unflattened_size_var, post_shape), axis=0)
    target_shape = mb.cast(x=target_shape, dtype="int32")
    y = mb.reshape(x=x, shape=target_shape, name=node.name)
    context.add(y)


def _array_construct(context, node, array_type):
    assert len(node.outputs) == 1
    inputs = _get_inputs(context, node)
    scalar_inputs = [
        inp
        for inp in inputs
        if isinstance(inp, Var) and inp.can_be_folded_to_const() and len(inp.shape) == 0
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


@register_torch_op
def tupleconstruct(context, node):
    _array_construct(context, node, array_type=tuple)


@register_torch_op
def listconstruct(context, node):
    _array_construct(context, node, array_type=list)


@register_torch_op
def eq(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    y = inputs[1]
    if is_bool(x.dtype):
        x = mb.cast(x=x, dtype="int32")
    if is_bool(y.dtype):
        y = mb.cast(x=y, dtype="int32")
    x, y = promote_input_dtypes([x, y])
    equal_to = mb.equal(x=x, y=y, name=node.name)
    context.add(equal_to)


@register_torch_op
def ne(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    y = inputs[1]
    if is_bool(x.dtype):
        x = mb.cast(x=x, dtype="int32")
    if is_bool(y.dtype):
        y = mb.cast(x=y, dtype="int32")
    x, y = promote_input_dtypes([x, y])
    equal_to = mb.not_equal(x=x, y=y, name=node.name)
    context.add(equal_to)


@register_torch_op
def le(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = promote_input_dtypes(inputs)
    less_equal = mb.less_equal(x=x, y=y, name=node.name)
    context.add(less_equal)


@register_torch_op
def lt(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = promote_input_dtypes(inputs)
    less = mb.less(x=x, y=y, name=node.name)
    context.add(less)


@register_torch_op
def ge(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = promote_input_dtypes(inputs)
    greater_equal = mb.greater_equal(x=x, y=y, name=node.name)
    context.add(greater_equal)


@register_torch_op
def gt(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = promote_input_dtypes(inputs[:2])
    greater = mb.greater(x=x, y=y, name=node.name)
    context.add(greater)


@register_torch_op(torch_alias=["t", "numpy_t"])
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


@register_torch_op(torch_alias=["permute"])
def permute_copy(context, node):
    inputs = _get_inputs(context, node, expected=2)
    perm = mb.transpose(x=inputs[0], perm=inputs[1], name=node.name)
    context.add(perm)


@register_torch_op
def frac(context, node):
    # Frac(x) = x - floor(abs(x)) * sign(x)

    x = _get_inputs(context, node, expected=1)[0]
    floor_abs = mb.floor(x=mb.abs(x=x))
    sign_abs_floor = mb.mul(x=floor_abs, y=mb.sign(x=x))
    res = mb.sub(x=x, y=sign_abs_floor)
    context.add(res, torch_name=node.name)


@register_torch_op
def pixel_shuffle(context, node):
    inputs = _get_inputs(context, node, expected=2)
    perm = mb.pixel_shuffle(x=inputs[0], upscale_factor=inputs[1], name=node.name)
    context.add(perm)


@register_torch_op
def pixel_unshuffle(context, node):
    inputs = _get_inputs(context, node, expected=2)
    downscale_factor = _np.uint32(inputs[1].val)
    perm = mb.pixel_unshuffle(x=inputs[0], downscale_factor=downscale_factor, name=node.name)
    context.add(perm)


def _construct_matmul(x: Var, y: Var, name: Optional[str] = None) -> Var:
    if (len(y.shape) == 2 and len(x.shape) <= 3) and (_is_const(y) or y.is_descendant_of_const):
        linear_x, weight = x, y
        transposed_weight = mb.transpose(x=weight, perm=(1, 0))
        res = mb.linear(x=linear_x, weight=transposed_weight, name=name)
    else:
        x, y = promote_input_dtypes([x, y])
        res = mb.matmul(x=x, y=y, name=name)
    return res


@register_torch_op(torch_alias=["bmm", "mm"])
def matmul(context, node):
    x, y = _get_inputs(context, node, expected=2)
    res = _construct_matmul(x, y, node.name)
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
    x, y = add_inputs[:2]
    if types.is_bool(x.dtype) and types.is_bool(y.dtype):
        add_node = mb.logical_or(x=x, y=y, name=node.name)
    elif types.is_complex(x.dtype) or types.is_complex(y.dtype):
        x_real = mb.complex_real(data=x) if types.is_complex(x.dtype) else x
        x_imag = mb.complex_imag(data=x) if types.is_complex(x.dtype) else 0.0
        y_real = mb.complex_real(data=y) if types.is_complex(y.dtype) else y
        y_imag = mb.complex_imag(data=y) if types.is_complex(y.dtype) else 0.0
        add_node = mb.complex(real_data=mb.add(x=x_real, y=y_real), imag_data=mb.add(x=x_imag, y=y_imag), name=node.name)
    else:
        x, y = promote_input_dtypes([x, y])
        add_node = mb.add(x=x, y=y, name=node.name)
    context.add(add_node)


@register_torch_op
def addmm(context, node):
    # addmm(Tensor x, Tensor mat1, Tensor mat2, Scalar beta=1, Scalar alpha=1)
    # output = beta * x + alpha * (mat1 @ mat2)

    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=(3, 4, 5))
        nargs = len(inputs)

        x = inputs[0]
        mat1 = inputs[1]
        mat2 = inputs[2]

        beta = inputs[3] if nargs > 3 else 1.0
        alpha = inputs[4] if nargs > 4 else 1.0

        return x, mat1, mat2, beta, alpha

    def _parse_keyword_args(context, node, beta, alpha) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return beta, alpha

        beta = _get_kwinputs(context, node, "beta", default=[beta])[0]
        alpha = _get_kwinputs(context, node, "alpha", default=[alpha])[0]

        return beta, alpha

    x, mat1, mat2, beta, alpha = _parse_positional_args(context, node)
    beta, alpha = _parse_keyword_args(context, node, beta, alpha)
    if isinstance(beta, Var):
        beta = beta.val
    if isinstance(alpha, Var):
        alpha = alpha.val

    if beta != 1.0:
        # Apply beta scaling factor to the input.
        x = mb.mul(x=x, y=beta)

    matmul = mb.matmul(x=mat1, y=mat2)

    if alpha != 1.0:
        # Apply alpha scaling factor to the matrix multiplicaiton
        matmul = mb.mul(x=alpha, y=matmul)

    result = mb.add(x=x, y=matmul, name=node.name)
    context.add(result)


@register_torch_op
def baddbmm(context, node):
    """
    baddbmm(Tensor input, Tensor batch1, Tensor batch2, Scalar beta=1, Scalar alpha=1)
    output = beta * input + alpha * batch1 * batch2

    Notice that batch1 and batch2 must be 3-D tensors each containing the same number of matrices.
    If batch1 is a (b×n×m) tensor, batch2 is a (b×m×p) tensor, then input must be broadcastable with a (b×n×p) tensor
    and out will be a (b×n×p) tensor.
    """

    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=(3, 4, 5))
        nargs = len(inputs)

        bias = inputs[0]
        batch1 = inputs[1]
        batch2 = inputs[2]

        beta = inputs[3] if nargs > 3 else 1.0
        alpha = inputs[4] if nargs > 4 else 1.0

        return bias, batch1, batch2, beta, alpha

    def _parse_keyword_args(context, node, beta, alpha) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return beta, alpha

        beta = _get_kwinputs(context, node, "beta", default=[beta])[0]
        alpha = _get_kwinputs(context, node, "alpha", default=[alpha])[0]

        return beta, alpha

    bias, batch1, batch2, beta, alpha = _parse_positional_args(context, node)
    beta, alpha = _parse_keyword_args(context, node, beta, alpha)
    if isinstance(beta, Var):
        beta = beta.val
    if isinstance(alpha, Var):
        alpha = alpha.val

    if alpha != 1.0:
        # Apply scaling factor alpha to the input.
        batch1 = mb.mul(x=alpha, y=batch1, name=batch1.name + "_scaled")
        context.add(batch1)

    bmm_node = mb.matmul(x=batch1, y=batch2, name=node.name + "_bmm")

    if beta != 0.0 or bias.shape != bmm_node.shape:
        context.add(bmm_node)
        if beta != 1.0:
            # Torch supports integers, so convert to float before
            if beta.dtype != bias.dtype:
                beta = mb.cast(x=beta, dtype=types.builtin_to_string(bias.dtype))
            # Apply scaling factor beta to the bias.
            bias = mb.mul(x=beta, y=bias, name=bias.name + "_scaled")
            context.add(bias)

        baddbmm_node = mb.add(x=bias, y=bmm_node, name=node.name)
        context.add(baddbmm_node)
    else:
        bmm_node.name = node.name
        context.add(bmm_node)


@register_torch_op
def linear(context, node):
    inputs = _get_inputs(context, node, expected=[2, 3])
    x = inputs[0]
    W = inputs[1]
    bias = inputs[2] if len(node.inputs) == 3 else None
    if bias is not None:
        x, W, bias = promote_input_dtypes([x, W, bias])
    else:
        x, W = promote_input_dtypes([x, W])

    res = _create_linear_layer(x, W, bias)
    context.add(res, torch_name=node.name)


@register_torch_op(
    torch_alias=[
        "convolution",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv1d.padding",
        "conv2d.padding",
        "conv3d.padding",
        "conv_transpose1d",
        "conv_transpose2d.input",
        "conv_transpose3d.input",
    ]
)
def _convolution(context, node):
    default_torch_padding = "valid" if node.kind.endswith(".padding") else 0

    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            min_expected={
                TorchFrontend.TORCHSCRIPT: 7,
                TorchFrontend.TORCHEXPORT: 2,
                TorchFrontend.EXECUTORCH: 2,
            },
        )
        nargs = len(inputs)

        x = inputs[0]
        # PyTorch and MIL has same weight layout
        # Conv: [Cout, Cin, *D]
        # ConvTranspose: [Cin, Cout, *D]
        weight = inputs[1]
        x, weight = promote_input_dtypes([x, weight])

        bias = inputs[2] if nargs > 2 else None
        stride = inputs[3] if nargs > 3 else 1
        padding = inputs[4] if nargs > 4 else default_torch_padding

        if node.kind in ("_convolution", "convolution"):
            dilation = inputs[5] if nargs > 5 else 1
            transposed = inputs[6].val if nargs > 6 else False
            out_padding = inputs[7] if nargs > 7 else 0
            groups = inputs[8] if nargs > 8 else 1
        elif re.match(r"conv_transpose[123]d.*", node.kind):
            out_padding = inputs[5] if nargs > 5 else 0
            groups = inputs[6] if nargs > 6 else 1
            dilation = inputs[7] if nargs > 7 else 1
            transposed = True
        else:
            dilation = inputs[5] if nargs > 5 else 1
            groups = inputs[6] if nargs > 6 else 1
            transposed = False
            out_padding = 0

        return x, weight, bias, stride, padding, dilation, groups, transposed, out_padding

    def _parse_keyword_args(
        context, node, bias, stride, padding, dilation, groups, out_padding
    ) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return bias, stride, padding, dilation, groups, out_padding

        bias = _get_kwinputs(context, node, "bias", default=[bias])[0]
        stride = _get_kwinputs(context, node, "stride", default=[stride])[0]
        padding = _get_kwinputs(context, node, "padding", default=[padding])[0]
        dilation = _get_kwinputs(context, node, "dilation", default=[dilation])[0]
        groups = _get_kwinputs(context, node, "groups", default=[groups])[0]
        out_padding = _get_kwinputs(context, node, "out_padding", default=[out_padding])[0]

        return bias, stride, padding, dilation, groups, out_padding

    def _translate_torch_args(node, weight, stride, padding, dilation, groups, out_padding):
        spatial_rank = weight.rank - 2

        # Core ML strides comes from torch stride
        if isinstance(stride, Var):
            stride = stride.val
            assert stride is not None, "torch conv stride must be constant"
        # Torch stride is an int (for all spatial dims) or an n-tuple of ints (one per spatial dim)
        # Core ML requires an n-tuple
        if isinstance(stride, int) or len(stride) == 1:
            strides = _np.array([np.squeeze(stride)] * spatial_rank)
        else:
            strides = stride
        # 1 is Core ML default value, so using None is preferred
        if _np.all(strides == 1):
            strides = None

        # Core ML pad_type and pad come from torch padding
        # For torch conv op .padding variants, torch padding is a string,
        # with possible values ("valid", "same")
        if node.kind.endswith(".padding"):
            pad_type = padding
            if isinstance(pad_type, Var):
                assert pad_type.val is not None
                pad_type = pad_type.val
            assert pad_type in ("valid", "same")
            # Core ML pad is None for pad_type "valid" / "same"
            pad = None
        # For other torch conv op variants, torch padding is
        # an int (for all spatial dims) or an n-tuple of ints (one per spatial dim)
        else:
            if isinstance(padding, Var):
                padding = padding.val
                assert padding is not None, "torch conv padding must be constant"
            # Core ML requires a (2 * n)-tuple, start and end for each spatial dim
            if isinstance(padding, int) or len(padding) == 1:
                pad = _np.array([np.squeeze(padding)] * (2 * spatial_rank))
            else:
                assert len(padding) == spatial_rank
                pad = _np.repeat(padding, 2)
            # Create Core ML pad_type according to Core ML pad
            if _np.all(pad == 0):
                pad_type = "valid"
                # 0 is Core ML default value, so using None is preferred
                pad = None
            else:
                pad_type = "custom"

        # Core ML dilations comes from torch dilation
        if isinstance(dilation, Var):
            dilation = dilation.val
            assert dilation is not None, "torch conv dilation must be constant"
        # Torch dilation is an int (for all spatial dims) or an n-tuple of ints (one per spatial dim)
        # Core ML requires an n-tuple
        if isinstance(dilation, int) or len(dilation) == 1:
            dilations = _np.array([np.squeeze(dilation)] * spatial_rank)
        else:
            dilations = dilation
        # 1 is Core ML default value, so using None is preferred
        if _np.all(dilations == 1):
            dilations = None

        # Core ML groups is torch groups
        if isinstance(groups, Var):
            groups = groups.val
            assert groups is not None, "torch conv groups must be constant"
        # 1 is Core ML default value, so using None is preferred
        if groups == 1:
            groups = None

        if isinstance(out_padding, Var):
            out_padding = out_padding.val
            assert out_padding is not None, "torch out_padding must be constant"
        # 0 is Core ML default value, so using None is preferred
        if _np.all(out_padding == 0):
            out_padding = None

        return strides, pad_type, pad, dilations, groups, out_padding

    (
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        transposed,
        out_padding,
    ) = _parse_positional_args(context, node)
    bias, stride, padding, dilation, groups, out_padding = _parse_keyword_args(
        context, node, bias, stride, padding, dilation, groups, out_padding
    )
    strides, pad_type, pad, dilations, groups, out_padding = _translate_torch_args(
        node, weight, stride, padding, dilation, groups, out_padding
    )

    kwargs = {
        "x": x,
        "weight": weight,
        "pad_type": pad_type,
        "name": node.name,
    }
    if bias is not None:
        kwargs["bias"] = bias
    if pad_type == "custom":
        kwargs["pad"] = pad
    if strides is not None:
        kwargs["strides"] = strides
    if dilations is not None:
        kwargs["dilations"] = dilations
    if groups is not None:
        kwargs["groups"] = groups

    if transposed is True:
        pad_len = 2 * (weight.rank - 2)
        # Transposed convolution
        # Handle output_padding using pre-pad or post-crop
        pre_pad = [0] * pad_len
        post_crop = [0] * pad_len

        if out_padding is not None and any(out_padding):
            output_padding = [0] * pad_len
            # output padding adds additional padding on one of the side of dimension
            # i.e. bottom from top-bottom,
            #      right  from left-right
            #      back   from front-back
            # Core ML padding structure is similar [top, bottom, left, right]
            # mapping output_padding to simplify further processing!
            #
            # For ConvTranspose2d: [bottom, right] -> [0, b, 0, r]
            output_padding = [0 if i % 2 == 0 else out_padding[i // 2] for i in range(pad_len)]
            if sum(pad) == 0 and any(output_padding):
                raise ValueError(
                    "ConvTranspose configuration of padding=0 and output_padding > 0 not supported!"
                )
            post_crop = pad.copy()
            pad *= 0
            for i in range(0, pad_len):
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

        conv = mb.conv_transpose(**kwargs)
        if any(post_crop):
            # TODO: rdar://65575826 (PyTorch converter: output_padding mapping to slice
            # instead of crop layer for 1 and 3D ConvTranspose)
            if len(post_crop) == 2 and conv.rank == 3:
                # Number of elements to crop from right = post_crop[-1].
                # Since slicing supports negative indexing, end_id = -1 * post_crop[-1]
                conv = mb.slice_by_index(
                    x=conv,
                    begin=[0, 0, post_crop[0]],
                    end=[0, 0, -1 * post_crop[-1]],
                    begin_mask=[True, True, False],
                    end_mask=[True, True, False],
                    name=node.name,
                )
            elif len(post_crop) == 4 and conv.rank == 4:
                conv = mb.crop(
                    x=conv,
                    crop_height=post_crop[:2],
                    crop_width=post_crop[2:4],
                    name=node.name,
                )
            else:
                raise ValueError(
                    "output_padding is supported only for ConvTranspose1D or ConvTranspose2D!"
                )
    else:
        # Normal convolution
        conv = mb.conv(**kwargs)
    context.add(conv)


# Convolution with "same, valid" padding
@register_torch_op
def _convolution_mode(context, node):
    inputs = _get_inputs(context, node, expected=7)
    mode = inputs[4].val

    context.add(
        mb.conv(
            x=inputs[0],
            weight=inputs[1],
            bias=inputs[2],
            strides=inputs[3],
            pad_type=mode,
            dilations=inputs[5],
            groups=inputs[6],
            name=node.name,
        )
    )


@register_torch_op(torch_alias=["_softmax"])
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


@register_torch_op
def relu(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.relu(x=inputs[0], name=node.name)
    context.add(res)


@register_torch_op
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


@register_torch_op
def linspace(context, node):
    inputs = _get_inputs(context, node, min_expected=3)

    start = inputs[0]
    end = inputs[1]
    nums = inputs[2]
    start = mb.cast(x=start, dtype="fp32")
    end = mb.cast(x=end, dtype="fp32")

    if start.can_be_folded_to_const() and end.can_be_folded_to_const() and nums.can_be_folded_to_const():
        start_val = start.val
        end_val = end.val
        nums_val = nums.val
        if nums_val < MAX_SIZE_CONSTANT_FOLDING:
            res = mb.const(val=_np.linspace(start_val, end_val, nums_val), name=node.name)
            context.add(res)
            return

    if nums.val is None:
        msg = "Dynamic steps input for torch.linspace is not supported. Please use torch.arange instead"
        raise NotImplementedError(msg)
    else:
        if nums.val == 1:
            res = mb.expand_dims(x=start, axes=[0], name=node.name)
        else:
            # step = (end - start) / (nums - 1)
            x = mb.sub(x=end, y=start)
            y = mb.sub(x=nums, y=1)
            y = mb.cast(x=y, dtype="fp32")
            step = mb.real_div(x=x, y=y)

            arange = mb.range_1d(start=0.0, end=mb.cast(x=nums, dtype="fp32"), step=1.0)
            scaled_arange = mb.mul(x=arange, y=step)
            res = mb.add(x=scaled_arange, y=start, name=node.name)

    context.add(res)


@register_torch_op
def relu6(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.relu6(x=inputs[0], name=node.name)
    context.add(res)


@register_torch_op
def einsum(context, node):
    if context.frontend == TorchFrontend.TORCHSCRIPT:
        vars = context[node.inputs[1]]
        vars = promote_input_dtypes(vars)
        equation = context[node.inputs[0]].val
    else:
        equation = node.inputs[0]
        if isinstance(equation, str) and equation in context:
            equation = context[equation].val
        tensor_names = node.inputs[1]
        if isinstance(tensor_names, str) and tensor_names in context:
            vars = context[tensor_names]
        else:
            assert isinstance(tensor_names, tuple)
            vars = [context[tensor_name] for tensor_name in tensor_names]
    x = build_einsum_mil(vars, equation, node.name)
    context.add(x)


@register_torch_op(torch_alias=["eye.m"])
def eye(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: [5, 6]},
            min_expected={TorchFrontend.TORCHEXPORT: 1, TorchFrontend.EXECUTORCH: 1},
        )
        nargs = len(inputs)

        n = inputs[0].val
        if context.frontend == TorchFrontend.TORCHSCRIPT:
            if nargs == 6:
                m = inputs[1].val
            else:
                m = None
        else:
            if node.kind == "eye.m":
                m = inputs[1].val
            else:
                m = None

        return n, m

    n, m = _parse_positional_args(context, node)

    # TODO: rdar://104400568 ([PyTorch] Use MIL ops to construct the eye matrix in order to avoid directly folding the input into a const)
    if m is None:
        eye = _np.eye(n)
    else:
        eye = _np.eye(n, m)
    eye = mb.const(val=eye, name=node.name)
    context.add(eye)


@register_torch_op
def elu(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: 4},
            min_expected={TorchFrontend.TORCHEXPORT: 1, TorchFrontend.EXECUTORCH: 1},
        )
        nargs = len(inputs)

        x = inputs[0]
        alpha = 1 if nargs < 2 else inputs[1]

        return x, alpha

    x, alpha = _parse_positional_args(context, node)

    res = mb.elu(x=x, alpha=alpha, name=node.name)
    context.add(res)


@register_torch_op
def leaky_relu(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=1)
        nargs = len(inputs)

        x = inputs[0]
        negative_slope = inputs[1] if nargs > 1 else 0.01

        return x, negative_slope

    def _parse_keyword_args(context, node, negative_slope) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return negative_slope

        negative_slope = _get_kwinputs(context, node, "negative_slope", default=[negative_slope])[0]
        return negative_slope

    x, negative_slope = _parse_positional_args(context, node)
    negative_slope = _parse_keyword_args(context, node, negative_slope)

    res = mb.leaky_relu(x=x, alpha=negative_slope, name=node.name)
    context.add(res)


@register_torch_op
def rrelu(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: 5},
            min_expected={TorchFrontend.TORCHEXPORT: 1, TorchFrontend.EXECUTORCH: 1},
        )
        nargs = len(inputs)

        x = inputs[0]
        lower = 0.125 if nargs < 2 else inputs[1].val
        upper = 1.0 / 3.0 if nargs < 3 else inputs[2].val

        return x, lower, upper

    x, lower, upper = _parse_positional_args(context, node)

    # Alpha in evaluation mode is just the average between upper and lower.
    alpha = (lower + upper) / 2

    res = mb.leaky_relu(x=x, alpha=alpha, name=node.name)
    context.add(res)


@register_torch_op
def softplus(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: 3},
            min_expected={TorchFrontend.TORCHEXPORT: 1, TorchFrontend.EXECUTORCH: 1},
        )
        nargs = len(inputs)

        x = inputs[0]
        beta = float(inputs[1].val) if nargs > 1 else 1.0
        threshold = float(inputs[2].val) if nargs > 2 else 20.0

        return x, beta, threshold

    x, beta, threshold = _parse_positional_args(context, node)

    if beta == 1:
        # this is the special case that Core ML softplus handles
        res = mb.softplus(x=x, name=node.name)
    else:
        if x.rank == 4:
            # can use Core ML softplus_parametric
            C = x.shape[1]
            alpha_br = _np.repeat(1.0 / beta, C).astype("float32")
            beta_br = _np.repeat(beta, C).astype("float32")
            res = mb.softplus_parametric(x=x, alpha=alpha_br, beta=beta_br, name=node.name)
        else:
            # have to generally decompose
            beta_mul_x = mb.mul(x=beta, y=x)
            softplus = mb.softplus(x=beta_mul_x)
            res = mb.real_div(x=softplus, y=beta, name=node.name)
    context.add(res)


@register_torch_op
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

        PyTorch output size formula for pooling:
        (reference: https://github.com/pytorch/pytorch/blob/375c30a7177442fb9d6de7516a9ae4031ae324c4/aten/src/ATen/native/Pool.h#L28)

        When ceil mode is True:
            out_dim = floor((in_dim + pad_l + pad_r - kernel_size + (stride-1)) / stride) + 1
            if (out_dim-1) * stride >= in_dim + pad_l and (pad_l > 0 or pad_r > 0):
                out_dim = out_dim - 1
        When ceil mode is False:
            out_dim = floor((in_dim + pad_l + pad_r - kernel_size) / stride) + 1


        # follow the approach here to calculate padding:
        # https://github.com/pytorch/pytorch/blob/edf751ca2fededecdd9366874c761431c0f61f01/aten/src/ATen/native/mkldnn/Pooling.cpp#L121
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


@register_torch_op(
    torch_alias=[
        "max_pool2d",
        "max_pool3d",
        "max_pool1d_with_indices",
        "max_pool2d_with_indices",
        "max_pool3d_with_indices",
    ]
)
def max_pool1d(context, node):
    inputs = _get_inputs(context, node, min_expected=3)

    x = inputs[0]
    kernel_sizes = inputs[1]
    strides = inputs[2]
    if strides.op.op_type == "const" and (not list(strides.val)):
        strides = mb.const(val=kernel_sizes.val, name=strides.name)

    pad_type = "custom"

    pad = np.array([0] * (kernel_sizes.shape[0] * 2)) if len(inputs) < 4 else _np.repeat(inputs[3].val, 2)
    dilation = np.array([1] * kernel_sizes.shape[0]) if len(inputs) < 5 else inputs[4].val
    ceil_mode = False if len(inputs) < 6 else inputs[5].val

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

    if re.match(r"max_pool[123]d_with_indices", node.kind):
        # TODO(rdar://117038432) ([Executorch] Handle/Bind other outputs of `max_pool2d_with_indices` op during lowering)
        context.add((pool, None), torch_name=node.name)
    else:
        context.add(pool)


@register_torch_op(torch_alias=["min.other"])
def minimum(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = promote_input_dtypes(inputs)
    assert len(node.outputs) == 1

    out = mb.minimum(x=x, y=y, name=node.name)
    context.add(out)


@register_torch_op
def clamp_max(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = inputs[0], inputs[1]
    assert x.dtype == y.dtype
    out = mb.minimum(x=x, y=y, name=node.name)
    context.add(out)

@register_torch_op
def clamp_min(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = inputs[0], inputs[1]
    assert x.dtype == y.dtype
    out = mb.maximum(x=x, y=y, name=node.name)
    context.add(out)


@register_torch_op(torch_alias=["max.other"])
def maximum(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = promote_input_dtypes(inputs)
    assert len(node.outputs) == 1

    out = mb.maximum(x=x, y=y, name=node.name)
    context.add(out)


@register_torch_op(torch_alias=["truediv"])
def div(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=[2, 3])
        nargs = len(inputs)

        x = mb.cast(x=inputs[0], dtype="fp32")
        y = mb.cast(x=inputs[1], dtype="fp32")
        if nargs < 3:
            rounding_mode = None
        else:
            rounding_mode_var = inputs[2]
            if rounding_mode_var is None:
                rounding_mode = None
            else:
                rounding_mode = rounding_mode_var.val

        return x, y, rounding_mode

    def _parse_keyword_args(context, node, rounding_mode) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return rounding_mode

        rounding_mode = _get_kwinputs(context, node, "rounding_mode", default=[rounding_mode])[0]
        return rounding_mode

    x, y, rounding_mode = _parse_positional_args(context, node)
    rounding_mode = _parse_keyword_args(context, node, rounding_mode)

    if rounding_mode is not None:
        if rounding_mode == "floor":
            # round towards negative infinity
            # e.g.:
            # values before floor: [2.6, -3.4, -3.6]
            # values after floor: [2, -4, -4]
            res = mb.floor_div(x=x, y=y, name=node.name)
        elif rounding_mode == "trunc":
            # round towards 0
            # e.g.:
            # values before trunc: [2.6, -3.4, -3.6]
            # values after trunc: [2, -3, -3]
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
        res = mb.real_div(x=x, y=y, name=node.name)

    context.add(res)


@register_torch_op(torch_alias=["floordiv"])
def floor_divide(context, node):
    inputs = _get_inputs(context, node, expected=2)
    inputs = promote_input_dtypes(inputs)
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
    x, y = promote_input_dtypes(inputs)
    if types.is_bool(x.dtype) and types.is_bool(y.dtype):
        res = mb.logical_and(x=x, y=y, name=node.name)
    else:
        res = mb.mul(x=x, y=y, name=node.name)
    context.add(res)


@register_torch_op
def pow(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = promote_input_dtypes(inputs)
    res = mb.pow(x=x, y=y, name=node.name)
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

    x, y = promote_input_dtypes([x, y])
    res = mb.sub(x=x, y=y, name=node.name)
    context.add(res)


# Various torch reduction ops are unified in this same translation logic
@register_torch_op(
    torch_alias=[
        "mean.dim",
        "sum",
        "sum.dim_intlist",
        "logsumexp",
        "all",
        "all.dim",
        "all.dims",
        "any",
        "any.dim",
        "any.dims",
    ]
)
def mean(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=1)
        nargs = len(inputs)

        if context.frontend == TorchFrontend.TORCHSCRIPT:
            x = inputs[0]
            dim = inputs[1] if nargs > 1 else None
            keepdim = inputs[2] if nargs > 2 else False
        else:
            if node.kind in ("mean", "sum", "all", "any"):
                x = inputs[0]
                dim = None
                keepdim = None
                # although we would not use dtype, still parse it
                dtype = inputs[1] if nargs > 1 else None
            else:
                x = inputs[0]
                dim = inputs[1] if nargs > 1 else None
                keepdim = inputs[2] if nargs > 2 else False
        return x, dim, keepdim

    def _parse_keyword_args(context, node, keepdim) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return keepdim

        keepdim = _get_kwinputs(context, node, "keepdim", default=[keepdim])[0]
        return keepdim

    def _translate_torch_args(dim, keepdim) -> Tuple[Var]:
        # torch dim means Core ML axes
        axes = None
        if dim is not None:
            # Core ML axes needs to be a list, but if only one dim was specified in torch,
            # it will be constructed as an int, so we construct a new constant as a list
            if not isinstance(dim.val, _np.ndarray):
                axes = mb.const(val=[dim.val], name=dim.name + "_list")
            elif dim.val.shape == (0,):
                axes = None
            else:
                axes = dim.val

        # torch keepdim means Core ML keep_dims, and Core ML defaults to False
        keep_dims = None
        if keepdim is not None:
            if isinstance(keepdim, Var):
                keepdim = keepdim.val
            if keepdim:
                keep_dims = True

        return axes, keep_dims

    x, dim, keepdim = _parse_positional_args(context, node)
    keepdim = _parse_keyword_args(context, node, keepdim)
    axes, keep_dims = _translate_torch_args(dim, keepdim)

    kwargs = {}
    if axes is not None:
        kwargs["axes"] = axes
    if keep_dims is not None:
        kwargs["keep_dims"] = keep_dims

    node_kind = node.kind.split(".")[0]
    if node_kind in ("all", "any"):
        x = mb.cast(x=x, dtype="int32")
        kwargs["x"] = x
        if node_kind == "all":
            res = mb.reduce_min(**kwargs)
        else:
            res = mb.reduce_max(**kwargs)
        res = mb.cast(x=res, dtype="bool", name=node.name)
    else:
        if node_kind == "mean":
            reduction_op_type = "reduce_mean"
        elif node_kind == "sum":
            reduction_op_type = "reduce_sum"
        else:
            assert node_kind == "logsumexp"
            reduction_op_type = "reduce_log_sum_exp"
        if (
            builtin_to_string(x.dtype)
            not in SSAOpRegistry._get_core_op_cls(reduction_op_type).supported_dtypes()
        ):
            x = mb.cast(x=x, dtype="fp32")
        kwargs["x"] = x
        kwargs["name"] = node.name
        res = getattr(mb, reduction_op_type)(**kwargs)

    context.add(res)


@register_torch_op(torch_alias=["logcumsumexp", "_logcumsumexp"])
def cumsum(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=2)
        nargs = len(inputs)

        x = inputs[0]
        dim = inputs[1]
        dtype = inputs[2] if nargs > 2 else None

        return x, dim, dtype

    def _parse_keyword_args(context, node, dtype) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dtype

        dtype = _get_kwinputs(context, node, "dtype", default=[dtype])[0]
        return dtype

    x, dim, dtype = _parse_positional_args(context, node)
    dtype = _parse_keyword_args(context, node, dtype)
    if dtype is not None:
        dtype = NUM_TO_DTYPE_STRING[dtype.val]

    if is_bool(x.dtype):
        x = mb.cast(x=x, dtype="int32")
    if node.kind == "cumsum":
        if dtype is not None and builtin_to_string(x.dtype) != dtype:
            x = mb.cast(x=x, dtype=dtype)
        res = mb.cumsum(x=x, axis=dim, name=node.name)
    else:
        assert node.kind in ("logcumsumexp", "_logcumsumexp")
        exp = mb.exp(x=x)
        cumsumexp = mb.cumsum(x=exp, axis=dim)
        res = mb.log(x=cumsumexp, name=node.name)

    context.add(res)


@register_torch_op(
    torch_alias=[
        "squeeze.dim",
        "squeeze.dims",
        "squeeze_copy",
        "squeeze_copy.dim",
        "squeeze_copy.dims",
    ]
)
def squeeze(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=(1, 2))
        nargs = len(inputs)
        x = inputs[0]
        dims = inputs[1] if nargs > 1 else None
        return x, dims

    def _translate_torch_args(x: Var, dims: Var) -> Tuple[int]:
        if isinstance(dims, Var):
            dims = dims.val
        if dims is None:
            return None

        # wrap `dims` into a tuple
        try:
            dims = (int(dims),)
        except:
            pass
        # squeeze all dims is the default behaviour
        if list(dims) == [*range(x.rank)]:
            return None
        # CPU fails non-single-dim squeeze (rdar://124555262)
        # so let us filter out the non-single dims
        filtered_dims = []
        for dim in dims:
            if x.shape[dim] == 1:
                filtered_dims.append(int(dim))
        return tuple(filtered_dims)

    x, dims = _parse_positional_args(context, node)
    axes = _translate_torch_args(x, dims)

    if axes is None:
        res = mb.squeeze(x=x, name=node.name)
        context.add(res)
    elif axes == ():
        # no axis to be squeezed, noop
        context.add(x, torch_name=node.name)
    else:
        res = mb.squeeze(x=x, axes=axes, name=node.name)
        context.add(res)


@register_torch_op(torch_alias=["unsqueeze_copy"])
def unsqueeze(context, node):
    inputs = _get_inputs(context, node, expected=2)
    unsqueeze = mb.expand_dims(x=inputs[0], axes=[inputs[1].val], name=node.name)
    context.add(unsqueeze)


@register_torch_op(torch_alias=["sym_size"])
def size(context, node):
    inputs = _get_inputs(context, node, expected=[1, 2])
    x = inputs[0]

    # Get the shape of the tensor.
    if types.is_complex(x.dtype):
        size_node = mb.complex_shape(x=inputs[0], name=node.name + "_shape")
    else:
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


@register_torch_op(torch_alias=["view_copy", "_unsafe_view", "reshape"])
def view(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    shape = inputs[1]

    if isinstance(shape, Var) and np.prod(shape.shape) == 0:
        # Reshape to empty shape (works only for scalar) is a no op
        assert (
            np.prod(x.shape) <= 1
        ), "Reshape to empty shape works only for scalar and single-element tensor"
        context.add(mb.identity(x=x, name=node.name))
        return

    if isinstance(shape, ListVar):
        length = mb.list_length(ls=shape)
        indices = mb.range_1d(start=0, end=length, step=1)
        shape = mb.list_gather(ls=shape, indices=indices)

    if isinstance(shape, list) and all(
        [isinstance(dim, Var) and len(dim.shape) == 0 for dim in shape]
    ):
        int_shape = []
        for size in shape:
            if size.dtype == types.int32:
                int_size = size
            else:
                int_size = mb.cast(x=size, dtype="int32")
            int_shape.append(int_size)
        shape = mb.concat(values=int_shape, axis=0)

    shape = mb.cast(x=shape, dtype="int32")

    if types.is_complex(x.dtype):
        real, imag = (mb.reshape(x=x, shape=shape, name=node.name) for x in (mb.complex_real(data=x), mb.complex_imag(data=x)))
        view = mb.complex(real_data=real, imag_data=imag, name=node.name)
    else:
        view = mb.reshape(x=x, shape=shape, name=node.name)

    context.add(view)


@register_torch_op(torch_alias=["constant_pad_nd"])
def pad(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: [3, 4]},
            min_expected={TorchFrontend.TORCHEXPORT: 2, TorchFrontend.EXECUTORCH: 2},
        )
        nargs = len(inputs)
        if context.frontend == TorchFrontend.TORCHSCRIPT:
            assert (node.kind == "pad") == (nargs == 4)
            assert (node.kind == "constant_pad_nd") == (nargs == 3)

        x = inputs[0]
        pad = inputs[1]
        if pad.val is not None:
            pad = pad.val.reshape((-1, 2))[::-1].reshape(-1).tolist()
            missing_dims = x.rank - (len(pad) // 2)
            pad = [0, 0] * missing_dims + pad

        if node.kind == "pad":
            mode = "constant"
            if nargs > 2:
                if isinstance(inputs[2], str):
                    mode = inputs[2]
                else:
                    if isinstance(inputs[2], Var) and inputs[2].val is not None:
                        mode = inputs[2].val
                    else:
                        raise ValueError(
                            "if pad mode is specified, then it must either be a string, "
                            "or a constant pymil variable"
                        )
            assert mode in ("circular", "constant", "reflect", "replicate")
            scalar_val = inputs[3] if nargs > 3 else 0.0
        else:
            mode = "constant"
            scalar_val = inputs[2] if nargs > 2 else 0.0
        if scalar_val is None:
            scalar_val = 0.0
        elif isinstance(scalar_val, Var):
            assert scalar_val.val is not None
            scalar_val = float(scalar_val.val)

        return x, pad, mode, scalar_val

    x, pad, mode, scalar_val = _parse_positional_args(context, node)

    if types.is_complex(x.dtype):
        real, imag = (mb.pad(x=x, pad=pad, mode=mode, constant_val=scalar_val, name=node.name) for x in (mb.complex_real(data=x), mb.complex_imag(data=x)))
        res = mb.complex(real_data=real, imag_data=imag, name=node.name)
    else:
        x, scalar_val = promote_input_dtypes([x, scalar_val])
        res = mb.pad(x=x, pad=pad, mode=mode, constant_val=scalar_val, name=node.name)
    context.add(res)


@register_torch_op
def adaptive_avg_pool1d(context, node):
    _adaptive_pool1d(context, node, mb.reduce_mean)


@register_torch_op(torch_alias=["_adaptive_avg_pool2d"])
def adaptive_avg_pool2d(context, node):
    _adaptive_pool2d(context, node, mb.avg_pool, mb.reduce_mean)


def _adaptive_pool1d(context, node, reduce_op):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    assert len(inputs[1].val) == 1
    out_length = inputs[1].val[0]

    if len(x.shape) == 3:
        # 3D input
        begin_prefix = [0, 0]
        end_prefix = [x.shape[0], x.shape[1]]
        out_shape = [x.shape[0], x.shape[1], out_length]
    else:
        # 2D input
        assert len(x.shape) == 2
        begin_prefix = [0]
        end_prefix = [x.shape[0]]
        out_shape = [x.shape[0], out_length]

    pool_results = []
    for start, end in _get_kernel_indexes_1d_for_adaptive_pooling(x.shape[-1], out_length):
        cur_kernel = mb.slice_by_index(
            x=x,
            begin=begin_prefix + [start],
            end=end_prefix+[end],
        )
        cur_result = reduce_op(
            x=cur_kernel,
            axes=[-1],
            keep_dims=True
        )
        pool_results.append(cur_result)

    context.add(
        mb.reshape(
            x=mb.concat(values=pool_results, axis=-1),
            shape=out_shape,
            name=node.name,
        )
    )


@register_torch_op
def adaptive_max_pool1d(context, node):
    _adaptive_pool1d(context, node, mb.reduce_max)


@register_torch_op
def adaptive_max_pool2d(context, node):
    _adaptive_pool2d(context, node, mb.max_pool, mb.reduce_max)


def _get_kernel_indexes_1d_for_adaptive_pooling(
        in_dimension: int,
        out_dimension: int) -> List[Tuple[int, int]]:
    results = []
    for i in range(out_dimension):
        start = _math.floor(i * in_dimension / out_dimension)
        end = _math.ceil((i + 1) * in_dimension / out_dimension)
        results.append((start, end))
    return results


def _adaptive_pool2d_non_fixed_kernel_size_and_stride(x, output_shape, name, reduce_op):
    '''
    If the input dimension is not evenly divisible by the output dimension, then the
    stride and kernel size used by PyTorch is not fixed. This is true for both the
    height and width dimension.
    '''

    pool_results = []
    for s2, e2 in _get_kernel_indexes_1d_for_adaptive_pooling(x.shape[-2], output_shape[0]):
        for s3, e3 in _get_kernel_indexes_1d_for_adaptive_pooling(x.shape[-1], output_shape[1]):
            begin = tuple([0] * (x.rank - 2) + [s2, s3])
            end = x.shape[: x.rank - 2] + (e2, e3)
            cur_kernel = mb.slice_by_index(x=x, begin=begin, end=end)
            cur_result = reduce_op(
                x=cur_kernel,
                axes=[-2, -1],
                keep_dims=True
            )
            pool_results.append(cur_result)
    pool_result = mb.concat(values=pool_results, axis=-1)

    shape = x.shape[: x.rank - 2] + output_shape
    return mb.reshape(x=pool_result, shape=shape, name=name)


def _adaptive_pool2d(context, node, pool_op, reduce_op):
    # Get input tensor and output shape
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    output_shape = inputs[1].val
    assert isinstance(output_shape, _np.ndarray) and len(output_shape) == 2
    output_shape = tuple(output_shape)

    if output_shape == (1, 1):
        # Represent (1,1) output size with global reduce op
        result = reduce_op(x=x, axes=[-2, -1], keep_dims=True, name=node.name)
    elif x.shape is None or any_symbolic(x.shape):
        raise ValueError(
            "Adaptive pooling is only supported when input tensor size is known or output size == (1,1). "
            "Received: input size == {}, output size == {}".format(
                x.shape_str(), output_shape,
            )
        )
    elif x.shape[-2] % output_shape[-2] == 0 and x.shape[-1] % output_shape[-1] == 0:
        # Stride and and kernel size is fixed
        strides = [ind // outd for ind, outd in zip(x.shape[-2:], output_shape)]
        kernel_sizes = [
            ind - s * (outd - 1)
            for ind, outd, s in zip(x.shape[-2:], output_shape, strides)
        ]
        if x.rank < 4:
            axes = [*range(4 - x.rank)]
            x_expanded = mb.expand_dims(x=x, axes=axes)
            result_expanded = pool_op(
                x=x_expanded,
                kernel_sizes=kernel_sizes,
                strides=strides,
                pad_type="valid",
            )
            result = mb.squeeze(x=result_expanded, axes=axes, name=node.name)
        else:
            result = pool_op(
                x=x,
                kernel_sizes=kernel_sizes,
                strides=strides,
                pad_type="valid",
                name=node.name,
            )
    else:
        result = _adaptive_pool2d_non_fixed_kernel_size_and_stride(
            x, output_shape, node.name, reduce_op
        )

    context.add(result)


@register_torch_op(
    torch_alias=[
        "_native_batch_norm_legit_no_training",
        "_native_batch_norm_legit.no_stats",
    ]
)
def batch_norm(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected=(6, 7, 9),
        )
        nargs = len(inputs)

        if node.kind == "_native_batch_norm_legit.no_stats":
            # inputs skipped:
            #     float momentum (4)
            _input = inputs[0]
            weight = inputs[1]
            bias = inputs[2]
            training = inputs[3].val
            eps = inputs[5]
            running_mean = None
            running_var = None
        elif node.kind == "_native_batch_norm_legit_no_training":
            assert nargs == 7, "torch _native_batch_norm_legit_no_training has 7 args"
            # inputs skipped:
            #     float momentum (5)
            _input = inputs[0]
            weight = inputs[1]
            bias = inputs[2]
            running_mean = inputs[3]
            running_var = inputs[4]
            eps = inputs[6]
            training = False
        else:
            assert node.kind == "batch_norm"
            assert nargs == 9, "torch batch_norm has 9 args"
            # inputs skipped:
            #     float momentum (6)
            #     bool cudnn_enabled (8)
            _input = inputs[0]
            weight = inputs[1]
            bias = inputs[2]
            running_mean = inputs[3]
            running_var = inputs[4]
            training = inputs[5].val
            eps = inputs[7]

        return _input, weight, bias, eps, running_mean, running_var, training

    _input, weight, bias, eps, running_mean, running_var, training = _parse_positional_args(
        context, node
    )

    input_rank = _input.rank
    if input_rank < 2 or input_rank > 5:
        raise ValueError(
            "BatchNorm: Encountered invalid input rank during translation in torch frontend."
        )

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

        return x

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
        return bn

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
        return bn

    is_batch_norm_1d_rank_2 = input_rank == 2

    if training or running_mean.val is None or running_var.val is None or weight is None or bias is None:
        bn = _add_batch_norm_dynamic()
    elif is_batch_norm_1d_rank_2:
        bn = _add_batch_norm_1d()
    else:
        bn = _add_batch_norm()

    if node.kind == "_native_batch_norm_legit_no_training":
        context.add((bn, None, None), torch_name=node.name)
    else:
        context.add(bn, torch_name=node.name)


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


def _group_norm_impl(x: Var, num_groups: int, weight: Var, bias: Var, eps: float) -> Var:
    n, c = x.shape[0], x.shape[1]  # at minimum (N, C) required
    num_groups = builtins.min(num_groups, c)
    new_shape = [n, num_groups, c // num_groups]
    # optimization for non symbolic shapes. This get rids of 3 mil ops that required on dynamic shapes
    if not any_symbolic(x.shape[2:]):
        new_shape += [*x.shape[2:]]  # adds remaining dims
        input_shape = [*x.shape]  # n, c, *
    else:
        input_shape = mb.shape(x=x)
        input_shape_sliced = mb.slice_by_size(x=input_shape, begin=[2], size=[-1])  # x_shape[2:]
        new_shape = mb.concat(values=[new_shape, input_shape_sliced], axis=0)

    num_extra_axes = len(x.shape[2:])
    axes_ = [int(i) for i in range(2, 2 + num_extra_axes + 1)]
    weight_shape, bias_shape = [1, c], [1, c]
    weight_shape += [1 for _ in range(num_extra_axes)]
    bias_shape += [1 for _ in range(num_extra_axes)]

    x = mb.reshape(x=x, shape=new_shape)
    mean = mb.reduce_mean(x=x, axes=axes_, keep_dims=True)
    var = _var(x, axes=axes_, keep_dims=True, unbiased=False)
    var_plus_eps = mb.add(x=var, y=eps)
    std = mb.sqrt(x=var_plus_eps)
    x = mb.sub(x=x, y=mean)
    x = mb.real_div(x=x, y=std)
    x = mb.reshape(x=x, shape=input_shape)
    if weight is not None:
        weight = mb.reshape(x=weight, shape=weight_shape)
        x = mb.mul(x=x, y=weight)
    if bias is not None:
        bias = mb.reshape(x=bias, shape=bias_shape)
        x = mb.add(x=x, y=bias)

    return x


@register_torch_op
def group_norm(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: 6},
            min_expected={TorchFrontend.TORCHEXPORT: 2, TorchFrontend.EXECUTORCH: 2},
        )
        nargs = len(inputs)

        x = inputs[0]
        num_groups = inputs[1].val

        weight = inputs[2] if nargs > 2 else None
        bias = inputs[3] if nargs > 3 else None
        eps = inputs[4].val if nargs > 4 else 1e-5

        return x, num_groups, weight, bias, eps

    def _parse_keyword_args(context, node, weight, bias) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return weight, bias

        weight = _get_kwinputs(context, node, "weight", default=[weight])[0]
        bias = _get_kwinputs(context, node, "bias", default=[bias])[0]
        return weight, bias

    x, num_groups, weight, bias, eps = _parse_positional_args(context, node)
    weight, bias = _parse_keyword_args(context, node, weight, bias)

    result = _group_norm_impl(x, num_groups, weight, bias, eps)
    context.add(result, node.name)


@register_torch_op
def native_group_norm(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=8)

        x = inputs[0]
        weight = inputs[1]
        bias = inputs[2]
        N = inputs[3].val
        C = inputs[4].val
        HxW = inputs[5].val
        group = inputs[6].val
        eps = inputs[7].val

        return x, weight, bias, N, C, HxW, group, eps

    x, weight, bias, N, C, HxW, group, eps = _parse_positional_args(context, node)
    assert x.shape[0] == N
    assert x.shape[1] == C

    result = _group_norm_impl(x, group, weight, bias, eps)
    context.add((result, None, None), node.name)


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
        logger.warning(
            "Core ML embedding (gather) layer does not support any "
            "inputs besides the weights and indices. Those given "
            "will be ignored."
        )

    indices = mb.cast(x=indices, dtype="int32")

    #  Changing the axis from 0 is not an option in torch, so we don't expose it
    gather = mb.gather(x=_input, indices=indices, name=node.name)
    context.add(gather)


@register_torch_op
def hardtanh(context, node):
    inputs = _get_inputs(context, node, expected=3)
    _input = inputs[0]
    min_val = inputs[1].val
    max_val = inputs[2].val

    res = mb.clip(x=_input, alpha=min_val, beta=max_val, name=node.name)
    context.add(res)


@register_torch_op(torch_alias=["concat"])
def cat(context, node):
    def is_tensor_empty(var: Var) -> bool:
        return np.any([size == 0 for size in var.shape])

    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=1)
        nargs = len(inputs)

        xs = inputs[0]
        # PyTorch can have empty tensor, which is then ignored
        # However, CoreML does not allow such empty tensor, so remove them now
        if np.any([is_tensor_empty(x) for x in xs]):
            filtered_xs = [x for x in xs if not is_tensor_empty(x)]
            xs = filtered_xs if len(filtered_xs) > 0 else [xs[0]]

        dim = inputs[1] if nargs > 1 else 0

        return xs, dim

    def _parse_keyword_args(context, node, dim) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dim

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        return dim

    xs, dim = _parse_positional_args(context, node)
    dim = _parse_keyword_args(context, node, dim)

    concat = mb.concat(values=promote_input_dtypes(xs), axis=dim, name=node.name)
    context.add(concat)


@register_torch_op
def stack(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=1)
        nargs = len(inputs)

        tensors = inputs[0]
        dim = inputs[1] if nargs > 1 else 0

        return tensors, dim

    def _parse_keyword_args(context, node, dim) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dim

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        return dim

    tensors, dim = _parse_positional_args(context, node)
    dim = _parse_keyword_args(context, node, dim)
    if isinstance(dim, Var):
        dim = dim.val

    if len(tensors) == 1:
        res = mb.expand_dims(x=tensors[0], axes=[dim], name=node.name)
    else:
        res = mb.stack(values=tensors, axis=dim, name=node.name)
    context.add(res)


@register_torch_op
def tile(context, node):
    x, dims = _get_inputs(context, node, expected=2)

    # The torch.tile only supports tuple of ints for "dims", not Tensor. So it will not be dynamic.
    if dims is None or dims.val is None:
        raise ValueError("The `dims` input for torch.tile must be static (tuple of ints).")

    dims_num = dims.shape[0]
    if dims_num < x.rank:
        # When the number of elements in dims is smaller than rank of x, ones are prepended.
        prepend_ones = np.array([1] * (x.rank - dims_num))
        dims = mb.concat(values=(prepend_ones, dims), axis=0)

    res = mb.tile(x=x, reps=dims, name=node.name)
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

    if x.can_be_folded_to_const():
        # If x is a compile-time constant, directly cast it to @dtype if it's
        # not one already.
        if not isinstance(x.val, dtype):
            res = mb.const(val=dtype(x.val), name=node.name)
        else:
            res = x
    elif len(x.shape) > 0:
        x = mb.squeeze(x=x, name=node.name + "_item")
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    else:
        res = mb.cast(x=x, dtype=dtype_name, name=node.name)
    context.add(res, node.name)


@register_torch_op(torch_alias=["bool"])
def _bool(context, node):
    _cast(context, node, bool, "bool")


@register_torch_op(torch_alias=["int"])
def _int(context, node):
    _cast(context, node, int, "int32")


@register_torch_op(torch_alias=["native_layer_norm"])
def layer_norm(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: [5, 6]},
            min_expected={TorchFrontend.TORCHEXPORT: 2, TorchFrontend.EXECUTORCH: 2},
        )
        nargs = len(inputs)

        x, normalized_shape = inputs[:2]

        weight = inputs[2] if nargs > 2 else None
        bias = inputs[3] if nargs > 3 else None
        eps = inputs[4] if nargs > 4 else None
        return x, normalized_shape, weight, bias, eps

    x, normalized_shape, weight, bias, eps = _parse_positional_args(context, node)

    layer_norm = mb.layer_norm(
        x=x,
        axes=list(range(-len(normalized_shape.val), 0)),
        gamma=weight,
        beta=bias,
        epsilon=eps,
        name=node.name,
    )

    if node.kind == "native_layer_norm":
        # TODO(rdar://117038370) ([Executorch] Handle/Bind other outputs of `native_layer_norm` op during lowering)
        context.add((layer_norm, None, None), torch_name=node.name)
    else:
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

    if x.can_be_folded_to_const():
        res = mb.const(val=[x.val], name=node.name)
        context.add(res)
    else:
        context.add(x, node.name)


def _ifzo_to_ifoz(weights, name):
    """
    i, f, z, o -> i, f, o, z
    where weights_split[0] == i, etc.
    Used to transform lstm weights from pytorch
    to Core ML format
    """
    split_size = weights.shape[0] // 4
    weights_split = mb.split(x=weights, split_sizes=_np.array([split_size] * 4), axis=0)
    return mb.concat(
        values=[weights_split[0], weights_split[1], weights_split[3], weights_split[2]],
        axis=0,
    )


def _pytorch_hidden_to_coreml_milops(x, name):
    """
    Used to transform lstm state values (hn, cn)
    from pytorch to Core ML format.
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
    Please note that the Core ML GRU has different definition from Torch,
    so we cannot use mb.gru, and need to implement it with while loop.
    To be more specific, in Core ML:

    o_t = activation(W_{io} x_t + r_t * W_{ho} h_(t−1) + b_{o})

    while torch has
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


@register_torch_op
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
            "Incorrect weights shape for gru layer: Expected: {}. Received {}".format(
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
            "Incorrect weights shape for lstm layer: Expected: {}. Received {}".format(
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


@register_torch_op
def rnn_tanh(context, node):
    _add_simple_rnn(context, node, "tanh")


@register_torch_op
def rnn_relu(context, node):
    _add_simple_rnn(context, node, "relu")


def _add_mil_lstm(input, initial_h, initial_c, weights, has_bias, bidirectional, name):
    """
    Most of this code is to transform the tensors into
    a shape acceptable by the Core ML implementation of LSTM.

    For weights, biases,  per direction, pytorch uses two tensors:
    (ii, if, ig, io) stacked on top of each other for each layer (tensor 1)
    and (hi, hf, hg, ho) stacked on top of each other for each layer (tensor 2).
    That is,  (W_ii|W_if|W_ig|W_io), of shape (4*hidden_size, input_size) and
    (W_hi|W_hf|W_hg|W_ho), of shape (4*hidden_size, hidden_size).


    The Core ML LSTM op expects two tensors, weight and bias. So
    the tensors for weight and bias are separated from pytorch's @weights list (1.).
    For bias tensor, the Core ML LSTM op expects the form ii, if, io, ig and hi, hf, ho, hg,
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
    """

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

        h = _pytorch_hidden_to_coreml_milops(initial_h, name=name + "_lstm_h0_reshaped")
        c = _pytorch_hidden_to_coreml_milops(initial_c, name=name + "_lstm_c0_reshaped")
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


@register_torch_op
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
            "Incorrect weights shape for lstm layer: Expected: {}. Received {}".format(
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
            output_size = [x.val for x in output_size]
        if isinstance(output_size, Var):
            output_size = [x for x in output_size.val]
        if isinstance(output_size, _np.ndarray):
            output_size = output_size.tolist()

        # output size is computed using the formula floor (scale * input_size) in Core ML (and PyTorch).
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


@register_torch_op(torch_alias=["upsample_linear1d.vec"])
def upsample_linear1d(context, node):
    """
    MIL only has upsample_bilinear (i.e. upsample 2d),
    so we will use the data dim as height and expand a dummy width dim
    to call MIL upsample_bilinear then squeeze the dummy width dim
    """

    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: 4},
            min_expected={TorchFrontend.TORCHEXPORT: 3, TorchFrontend.EXECUTORCH: 3},
        )
        nargs = len(inputs)

        x = inputs[0]
        output_size = inputs[1]
        align_corners = bool(inputs[2].val)

        if context.frontend == TorchFrontend.TORCHSCRIPT:
            scales = None if nargs < 4 else inputs[3]
        else:
            suffix = node.kind.split(".")[-1]
            if suffix == "vec":
                scale_factors = None if nargs < 4 else inputs[3]
                if scale_factors is not None:
                    scales = scale_factors[0]
                else:
                    scales = None
            else:
                scales = None if nargs < 4 else inputs[3]

        return x, output_size, align_corners, scales

    def _parse_keyword_args(context, node, scales) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return scales

        scales = _get_kwinputs(context, node, "scales", default=[scales])[0]

        return scales

    def _translate_torch_args(x, output_size, align_corners, scales) -> Var:
        scales_h = None
        scales_w = None

        if scales is not None:
            # Get the scale factor from provided inputs
            # This happens when recompute_scale_factor = False
            scales_h = scales.val
            if isinstance(scales_h, np.ndarray):
                scales_h = scales_h.reshape(-1)[0]

            h = x.shape[-1]
            # Currently, we are not supporting recompute_scale_factor = False, align_corners = False with float output size
            if not is_symbolic(h):
                # For the static input shape, we can compute the output size beforehand, and check if it is a float value
                output_size = h * scales_h
                is_float = _is_float_value(output_size)
            else:
                # For the dynamic input shape, we check if the scale factor itself is float
                is_float = _is_float_value(scales_h)

            if is_float and not align_corners:
                raise NotImplementedError(
                    "recompute_scale_factor = False, align_corners = False with float output size "
                    f"is not supported for the upsample op {node.name}"
                )

        elif isinstance(output_size, Var) and output_size.val is not None:
            # Infer the scale factor from the provided output size
            scales_h = _get_scales_from_output_size(output_size, x.shape)

        if scales_h is not None:
            if _is_float_value(scales_h):
                scales_w = 1.0
            else:
                scales_h = int(scales_h)

        return scales_h, scales_w

    x, output_size, align_corners, scales = _parse_positional_args(context, node)
    scales = _parse_keyword_args(context, node, scales)
    scales_h, scales_w = _translate_torch_args(x, output_size, align_corners, scales)

    x = mb.expand_dims(x=x, axes=[3])
    if scales_h is not None:
        x = mb.upsample_bilinear(
            x=x,
            scale_factor_height=scales_h,
            scale_factor_width=scales_w,
            align_corners=align_corners,
        )
    else:
        # When the input shape is dynamic and recompute_scale_factor = True,
        # we need to trace the graph to find the scale factor.
        assert (
            isinstance(output_size, list) and len(output_size) == 1
        ), "for dynamic shape torch should give [output_size]"
        x = mb.torch_upsample_bilinear(
            x=x,
            output_height=output_size[0],
            output_width=1,
            align_corners=align_corners,
        )
    x = mb.squeeze(x=x, axes=[3], name=node.name)
    context.add(x)


@register_torch_op(
    torch_alias=[
        "upsample_bilinear2d.vec",
        "_upsample_bilinear2d_aa",
        "_upsample_bilinear2d_aa.vec",
    ],
)
def upsample_bilinear2d(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: (4, 5)},
            min_expected={TorchFrontend.TORCHEXPORT: 3, TorchFrontend.EXECUTORCH: 3},
        )
        nargs = len(inputs)

        x = inputs[0]
        output_size = inputs[1]
        align_corners = bool(inputs[2].val)

        if context.frontend == TorchFrontend.TORCHSCRIPT:
            if nargs == 4:
                scale_factors = inputs[3]
                if scale_factors is None:
                    scales_h = None
                    scales_w = None
                else:
                    scales_h = scale_factors.val[0]
                    scales_w = scale_factors.val[1]
            else:
                assert nargs == 5, "Starting from torch 1.5.0, upsample_bilinear2d has 5 inputs"
                scales_h = inputs[3]
                scales_w = inputs[4]
        else:
            suffix = node.kind.split(".")[-1]
            if suffix == "vec":
                scale_factors = None if nargs < 4 else inputs[3]
                if scale_factors is None:
                    scales_h = None
                    scales_w = None
                else:
                    scales_h = scale_factors[0]
                    scales_w = scale_factors[1]
            else:
                scales_h = None if nargs < 4 else inputs[3]
                scales_w = None if nargs < 5 else inputs[4]

        return x, output_size, align_corners, scales_h, scales_w

    def _parse_keyword_args(context, node, scales_h, scales_w) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return scales_h, scales_w

        scales_h = _get_kwinputs(context, node, "scales_h", default=[scales_h])[0]
        scales_w = _get_kwinputs(context, node, "scales_w", default=[scales_w])[0]

        return scales_h, scales_w

    def _translate_torch_args(x, output_size, align_corners, scales_h, scales_w) -> Var:
        if scales_h is not None and scales_w is not None:
            # get scale factors from provided inputs
            # this happens when recompute_scale_factor = False
            if isinstance(scales_h, Var):
                scales_h = scales_h.val
            if isinstance(scales_w, Var):
                scales_w = scales_w.val
        elif isinstance(output_size, Var) and output_size.val is not None:
            # infer scale factors from output sizes
            # This happens when recompute_scale_factor = True or the output_size is specified
            scales = _get_scales_from_output_size(output_size, x.shape)
            if scales:
                scales_h, scales_w = scales
        return scales_h, scales_w

    x, output_size, align_corners, scales_h, scales_w = _parse_positional_args(context, node)
    scales_h, scales_w = _parse_keyword_args(context, node, scales_h, scales_w)
    scales_h, scales_w = _translate_torch_args(x, output_size, align_corners, scales_h, scales_w)

    if scales_h is not None and scales_w is not None:
        # currently, we are not supporting recompute_scale_factor = False, align_corners = False with float output size
        _, _, h, w = x.shape
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
            if x.val is not None and (output_size is None or output_size.val is not None):
                if output_size is not None:
                    output_size = torch.tensor(output_size.val)
                upsample_res = (
                    torch.nn.functional.upsample_bilinear(
                        torch.tensor(x.val), output_size, [scales_h, scales_w]
                    )
                    .detach()
                    .numpy()
                )
                context.add(mb.const(val=upsample_res, name=node.name))
                return
            raise NotImplementedError(
                "recompute_scale_factor = False, align_corners = False with float output size "
                f"is not supported for the upsample op {node.name}"
            )

        upsample_bilinear = mb.upsample_bilinear(
            x=x,
            scale_factor_height=scales_h,
            scale_factor_width=scales_w,
            align_corners=align_corners,
            name=node.name,
        )
    else:
        # the input shape is dynamic and recompute_scale_factor = True
        # need to trace the graph to find the scale factor
        # we define a torch front end op mb.torch_upsample_bilinear to resolve the const scaling factor
        assert (
            isinstance(output_size, list) and len(output_size) == 2
        ), "for dynamic shape torch should give [output_size_h, output_size_w]"
        upsample_bilinear = mb.torch_upsample_bilinear(
            x=x,
            output_height=output_size[0],
            output_width=output_size[1],
            align_corners=align_corners,
            name=node.name,
        )
    context.add(upsample_bilinear)


@register_torch_op(torch_alias=["upsample_nearest1d.vec"])
def upsample_nearest1d(context, node):
    """
    MIL only has upsample_nearest_neighbor 2d,
    so we will use the data dim as height and expand a dummy width dim
    to call MIL upsample_nearest_neighbor then squeeze the dummy width dim
    """

    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: 3},
            min_expected={TorchFrontend.TORCHEXPORT: 2, TorchFrontend.EXECUTORCH: 2},
        )
        nargs = len(inputs)

        x = inputs[0]
        output_size = inputs[1]

        if context.frontend == TorchFrontend.TORCHSCRIPT:
            scales = None if nargs < 3 else inputs[2]
        else:
            suffix = node.kind.split(".")[-1]
            if suffix == "vec":
                scale_factors = None if nargs < 3 else inputs[2]
                if scale_factors is not None:
                    scales = scale_factors[0]
                else:
                    scales = None
            else:
                scales = None if nargs < 3 else inputs[2]

        return x, output_size, scales

    def _parse_keyword_args(context, node, scales) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return scales

        scales = _get_kwinputs(context, node, "scales", default=[scales])[0]

        return scales

    def _translate_torch_args(x, output_size, scales) -> Var:
        scales_h = None
        scales_w = None

        if scales is not None:
            # Get the scale factor from provided inputs
            # This happens when recompute_scale_factor = False
            scales_h = scales.val
            if isinstance(scales_h, np.ndarray):
                scales_h = scales_h.reshape(-1)[0]
        elif isinstance(output_size, Var) and output_size.val is not None:
            # Infer the scale factor from the provided output size
            scales_h = _get_scales_from_output_size(output_size, x.shape)

        if scales_h is not None:
            if _is_float_value(scales_h):
                scales_w = 1.0
            else:
                scales_h = int(scales_h)

        return scales_h, scales_w

    x, output_size, scales = _parse_positional_args(context, node)
    scales = _parse_keyword_args(context, node, scales)
    scales_h, scales_w = _translate_torch_args(x, output_size, scales)

    x = mb.expand_dims(x=x, axes=[3])
    if scales_h is not None:
        x = mb.upsample_nearest_neighbor(
            x=x,
            scale_factor_height=scales_h,
            scale_factor_width=scales_w,
        )
    else:
        # When the input shape is dynamic and recompute_scale_factor = True,
        # we need to trace the graph to find the scale factor.
        assert (
            isinstance(output_size, list) and len(output_size) == 1
        ), "for dynamic shape torch should give [output_size]"
        x = mb.torch_upsample_nearest_neighbor(
            x=x,
            output_height=output_size[0],
            output_width=1,
        )
    x = mb.squeeze(x=x, axes=[3], name=node.name)
    context.add(x)


@register_torch_op(torch_alias=["upsample_nearest2d.vec"])
def upsample_nearest2d(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: (3, 4)},
            min_expected={TorchFrontend.TORCHEXPORT: 2, TorchFrontend.EXECUTORCH: 2},
        )
        nargs = len(inputs)

        x = inputs[0]
        output_size = inputs[1]

        if context.frontend == TorchFrontend.TORCHSCRIPT:
            if nargs == 3:
                scale_factors = inputs[2]
                if scale_factors is None:
                    scales_h = None
                    scales_w = None
                else:
                    scales_h = scale_factors.val[0]
                    scales_w = scale_factors.val[1]
            else:
                assert nargs == 4, "Starting from torch 1.5.0, upsample_nearest2d has 4 inputs"
                scales_h = inputs[2]
                scales_w = inputs[3]
        else:
            suffix = node.kind.split(".")[-1]
            if suffix == "vec":
                scale_factors = None if nargs < 3 else inputs[2]
                if scale_factors is None:
                    scales_h = None
                    scales_w = None
                else:
                    scales_h = scale_factors[0]
                    scales_w = scale_factors[1]
            else:
                scales_h = None if nargs < 3 else inputs[2]
                scales_w = None if nargs < 4 else inputs[3]

        return x, output_size, scales_h, scales_w

    def _parse_keyword_args(context, node, scales_h, scales_w) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return scales_h, scales_w

        scales_h = _get_kwinputs(context, node, "scales_h", default=[scales_h])[0]
        scales_w = _get_kwinputs(context, node, "scales_w", default=[scales_w])[0]

        return scales_h, scales_w

    def _translate_torch_args(x, output_size, scales_h, scales_w) -> Var:
        if scales_h is not None and scales_w is not None:
            # get scale factors from provided inputs
            # this happens when recompute_scale_factor = False
            if isinstance(scales_h, Var):
                scales_h = scales_h.val
            if isinstance(scales_w, Var):
                scales_w = scales_w.val

        elif isinstance(output_size, Var) and output_size.val is not None:
            # infer scale factors from output sizes
            # This happens when recompute_scale_factor = True or the output_size is specified
            scales = _get_scales_from_output_size(output_size, x.shape)
            if scales:
                scales_h, scales_w = scales

        return scales_h, scales_w

    x, output_size, scales_h, scales_w = _parse_positional_args(context, node)
    scales_h, scales_w = _parse_keyword_args(context, node, scales_h, scales_w)
    scales_h, scales_w = _translate_torch_args(x, output_size, scales_h, scales_w)

    if scales_h is not None and scales_w is not None:
        upsample_nearest2d = mb.upsample_nearest_neighbor(
            x=x,
            scale_factor_height=scales_h,
            scale_factor_width=scales_w,
            name=node.name,
        )
    else:
        # the input shape is dynamic and recompute_scale_factor = True
        # need to trace the graph to find the scale factor
        # we define a torch front end op mb.torch_upsample_nearest_neighbor to resolve the const scaling factor
        upsample_nearest2d = mb.torch_upsample_nearest_neighbor(
            x=x,
            output_height=output_size[0],
            output_width=output_size[1],
            name=node.name,
        )
    context.add(upsample_nearest2d)


@register_torch_op(torch_alias=["listunpack"])
def tupleunpack(context, node):
    inputs = _get_inputs(context, node, expected=1)
    values = inputs[0]

    # Node input could have been turned into constant array in @tupleconstruct
    if not isinstance(values, (tuple, list)):
        if values.val is not None:
            values = values.val
        else:
            # The `values` could be a single Var with symbolic val.
            values = [values]

    if len(values) != len(node.outputs):
        raise ValueError(f"unpack node expected {len(node.outputs)} outputs, got {len(values)}")

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
        context.add(output_var, torch_name=output_name)


@register_torch_op
def _unique2(context, node):
    (x, sorted, return_inverse, return_counts)  = _get_inputs(context, node, expected=4)

    # Unsupported case
    if sorted.val is not True:
        raise NotImplementedError("sorted=False not supported for unique op")

    x_flatten = mb.reshape(x=x, shape=[-1])

    # Sort flattened input
    indices = mb.argsort(x=x_flatten, ascending=True)
    x_sorted = mb.gather_along_axis(x=x_flatten, indices=indices)

    # Subtract n_th+1 element from n_th element
    neg_inf = np.float32(-np.inf)
    x_sorted = mb.cast(x=x_sorted, dtype="fp32")
    x_sorted_shifted  = mb.pad(x=x_sorted, pad=[1, 0], constant_val=neg_inf)
    x_sorted_padded = mb.pad(x=x_sorted, pad=[0, 1], mode="replicate")
    diff = mb.sub(x=x_sorted_padded, y=x_sorted_shifted)

    # Get non-zero element after subtraction to determine unique values
    non_zero_indices = mb.non_zero(x=diff)
    unique_values_unsqueeze = mb.gather(x=x_sorted, indices=non_zero_indices)
    unique_values = mb.squeeze(x = unique_values_unsqueeze)

    # Add unique values to output and see if we're done.
    context.add(unique_values, torch_name=node.outputs[0])
    if return_counts.val is False and return_inverse.val is False:
        # only the unique values are needed
        return

    # Calculate a UxN boolean tensor, where:
    #     U - number of unique values
    #     N - number of input elements
    num_unique_values = mb.shape(x=unique_values)
    x_tile = mb.tile(x=x_flatten, reps=num_unique_values)
    tile_shape = mb.concat(values=(num_unique_values, mb.shape(x=x_flatten)), axis=0)
    x_tile = mb.reshape(x=x_tile, shape=tile_shape)
    unique_values_unsqueeze = mb.cast(x=unique_values_unsqueeze, dtype="int32")
    x_tile, unique_values_unsqueeze = promote_input_dtypes([x_tile, unique_values_unsqueeze])
    diff = mb.sub(x=x_tile, y=unique_values_unsqueeze)
    bool_tensor = mb.logical_not(x=mb.cast(x=diff, dtype="bool"))

    if return_inverse.val is True:
        # Get indices
        range = mb.range_1d(start=0, end=mb.squeeze(x=num_unique_values), step=1)
        indices = mb.matmul(x=range, y=mb.cast(x=bool_tensor, dtype="int32"))
        indices = mb.reshape(x=indices, shape=mb.shape(x=x))
        context.add(indices, torch_name=node.outputs[1])

    if return_counts.val is True:
        # Get counts
        counts = mb.reduce_sum(x=mb.cast(x=bool_tensor, dtype='int32'), axes=(-1,))
        context.add(counts, torch_name=node.outputs[2])


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
    if not isinstance(cond, (tuple, list)):
        cond = (cond,)

    # Make sure the condition returned the expected number of outputs.
    assert len(cond) == len(node.outputs)
    for output_name, output_var in zip(node.outputs, cond):
        context.add(output_var, torch_name=output_name)


@register_torch_op(torch_alias=["select_copy"])
def select(context, node):
    inputs = _get_inputs(context, node, expected=3)
    _input = inputs[0]
    dim = inputs[1].val
    index = inputs[2]

    assert dim.shape == ()

    # NOTE:
    # Each index in @begin_array/@end_array corresponds to a dimension of @_input
    # Each val of those arrays corresponds to the start/end index to slice in that dimension
    rank = _input.rank

    begin_array = [0] * rank
    if index.val is None:
        # index value not known till runtime
        begin_array[dim] = index
        begin_array = mb.concat(values=begin_array, axis=0)
    else:
        # index value known now
        assert index.val.shape == ()
        begin_array[dim] = index.val

    end_array = [s if isinstance(s, int) else 0 for s in _input.shape]
    end_mask = [True] * rank
    squeeze_mask = [False] * rank
    squeeze_mask[dim] = True

    if index.val != -1:
        if index.val is None:
            # index value not know till runtime
            temp = mb.add(x=index, y=1)
            end_array[dim] = temp
            end_array = mb.concat(values=end_array, axis=0)
        else:
            end_array[dim] = index.val + 1
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


@register_torch_op
def getitem(context, node):
    inputs = _get_inputs(context, node, expected=2)

    assert inputs[1].val is not None, "Only static item selection supported"

    try:
        index = int(inputs[1].val)
    except:
        raise AssertionError(
            f"Index into python list/tuple needs to be integer. Provided value: {inputs[1].val}"
        )

    if not isinstance(inputs[0], (list, tuple)):
        # For single object with index 0, return this object
        if isinstance(inputs[0], Var) and index == 0:
            context.add(inputs[0], torch_name=node.name)
            return
        # Otherwise undefined
        else:
            raise AssertionError("Item selection is supported only on python list/tuple objects")

    out = inputs[0][index]

    assert out is not None, (
        f"coremltools lowering does not handle/bind value at index {index}. "
        "Please inspect the lowering of parent op for its return value"
    )

    context.add(out, torch_name=node.name)


@register_torch_op
def type_as(context, node):
    inputs = _get_inputs(context, node, expected=2)

    if inputs[0].dtype == inputs[1].dtype:
        x = mb.identity(x=inputs[0], name=node.name)
    else:
        x = inputs[0]
        if inputs[1].dtype not in TYPE_TO_DTYPE_STRING:
            raise NotImplementedError(
                "Tensor type {} cast is not supported.".format(inputs[1].dtype)
            )
        x = mb.cast(x=x, dtype=TYPE_TO_DTYPE_STRING[inputs[1].dtype], name=node.name)

    context.add(x)


@register_torch_op
def nonzero(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    nonzero = mb.non_zero(x=x, name=node.name)
    context.add(nonzero)


def _get_slice_params(context, data, inputs):
    def _expand_list_to_rank_1(arr):
        """
        We make the elements in begin and end rank 1,
        so the pattern of ``squeeze -> expand_dims`` can be removed
        by the ``fuse_squeeze_expand_dims`` graph pass.
        """
        for i, val in enumerate(arr):
            if isinstance(val, Var):
                if val.rank == 0:
                    arr[i] = mb.expand_dims(x=val, axes=[0])
            else:
                arr[i] = np.array([val])
        return arr

    rank = data.rank
    begin = [0] * rank
    end = [0] * rank
    stride = [1] * rank
    begin_mask = [False] * rank
    end_mask = [False] * rank
    squeeze_mask = [False] * rank

    num_of_slice_set = len(inputs) // 3

    for i in range(num_of_slice_set):
        if inputs[3 * i + 1] is None:
            # This is pure index select
            idx = context[inputs[3 * i]]
            if idx.val is not None:
                idx = idx.val
            begin[i] = idx
            squeeze_mask[i] = True
        else:
            # This is a slice
            begin_var = context[inputs[3 * i]]
            end_var = context[inputs[3 * i + 1]]
            stride_var = context[inputs[3 * i + 2]]

            if begin_var is None:
                begin_mask[i] = True
            else:
                begin[i] = begin_var

            if end_var is None:
                end_mask[i] = True
            else:
                end[i] = end_var

            if stride_var is None:
                stride[i] = 1
            else:
                stride[i] = stride_var.val

    for i in range(num_of_slice_set, rank):
        begin_mask[i] = True
        end_mask[i] = True

    begin = _expand_list_to_rank_1(begin)
    eng = _expand_list_to_rank_1(end)
    begin = mb.concat(values=begin, axis=0)
    end = mb.concat(values=end, axis=0)

    return begin, end, stride, begin_mask, end_mask, squeeze_mask


def _translate_torch_tensor_assign(
    x: Var,
    updates: Var,
    begin: Var,
    end: Var,
    stride=None,
    begin_mask=None,
    end_mask=None,
    squeeze_mask=None,
    name=None,
):
    translation_kwargs = {}
    if stride is not None:
        translation_kwargs["stride"] = stride
    if begin_mask is not None:
        translation_kwargs["begin_mask"] = begin_mask
    if end_mask is not None:
        translation_kwargs["end_mask"] = end_mask
    if squeeze_mask is not None:
        translation_kwargs["squeeze_mask"] = squeeze_mask
    if name is not None:
        translation_kwargs["name"] = name

    if is_current_opset_version_compatible_with(target.iOS18):
        # slice_update is not supporting scalar update at runtime.
        # Until this radar is fixed: rdar://128221986 ([Feature][Slice_update] The backend is not supporting scalar update for the slice_update op),
        # we have a workaround to expand scalar update to a 1-D tensor.
        if updates.rank == 0:
            # Since the workaround uses the compile-time value of begin and end,
            # so we do the validation first.
            is_begin_or_end_dynamic = False
            for var in [begin, end]:
                if isinstance(var, Var) and var.val is None:
                    is_begin_or_end_dynamic = True
            if is_begin_or_end_dynamic or any_symbolic(x.shape):
                return mb.torch_tensor_assign(
                    x=x,
                    updates=updates,
                    begin=begin,
                    end=end,
                    **translation_kwargs,
                )

            # First pick up the ``dim`` in which ``squeeze_mask[dim] = True``,
            # and do the following transformation:
            # 1. set ``squeeze_mask[dim] = False``
            # 2. set both ``begin_mask`` and ``end_mask`` to ``False``
            # 3. make ``end = begin + 1``
            dim = None
            for i, val in enumerate(squeeze_mask):
                if val is True:
                    dim = i
                    break
            squeeze_mask[dim] = False
            begin_mask = [False] * x.rank
            end_mask = [False] * x.rank

            if isinstance(begin, Var):
                begin = begin.val
            if isinstance(end, Var):
                end = end.val

            # convert negative indexes to positive indexes
            begin = [val if val >= 0 else val + x.shape[i] for i, val in enumerate(begin)]
            end = mb.add(x=begin, y=1)

            # expand updates to 1D tensor
            updates = mb.expand_dims(x=updates, axes=[0])

        return mb.slice_update(
            x=x,
            update=updates,
            begin=begin,
            end=end,
            **translation_kwargs,
        )

    return mb.torch_tensor_assign(
        x=x,
        updates=updates,
        begin=begin,
        end=end,
        **translation_kwargs,
    )


@register_torch_op
def _internal_op_tensor_inplace_copy(context, node):
    data = context[node.inputs[0]]
    updates = context[node.inputs[1]]
    begin, end, stride, begin_mask, end_mask, squeeze_mask = _get_slice_params(
        context, data, node.inputs[2:]
    )

    data, updates = promote_input_dtypes([data, updates])
    updated_x = _translate_torch_tensor_assign(
        x=data,
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


@register_torch_op
def _internal_op_tensor_inplace_fill(context, node):
    data = context[node.inputs[0]]
    fill_scalar = context[node.inputs[1]]

    if len(node.inputs) == 2 and fill_scalar.val is not None:
        shape = mb.shape(x=data)
        if isinstance(fill_scalar.val, _np.ndarray):
            fill = mb.fill(shape=shape, value=fill_scalar.val.item())
        else:
            fill = mb.fill(shape=shape, value=fill_scalar)
        casted = mb.cast(x=fill, dtype=TYPE_TO_DTYPE_STRING[data.dtype], name=node.name)
        context.add(casted)
        return

    begin, end, stride, begin_mask, end_mask, squeeze_mask = _get_slice_params(
        context, data, node.inputs[2:]
    )
    if begin.val is None or end.val is None or any_symbolic(data.shape):
        raise ValueError("_internal_op_tensor_inplace_fill does not support dynamic index")

    fill_shape = solve_slice_by_index_shape(
        data.shape, begin.val, end.val, stride, begin_mask, end_mask, squeeze_mask
    )
    update_values = _np.full(fill_shape, fill_scalar.val)

    data, update_values = promote_input_dtypes([data, update_values])

    updated_x = _translate_torch_tensor_assign(
        x=data,
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


@register_torch_op
def select_scatter(context, node):
    inputs = _get_inputs(context, node, expected=4)
    x = inputs[0]
    updates = inputs[1]
    dim = inputs[2].val
    if dim is None:
        raise ValueError("Only compile time known dim supported yet")
    index = inputs[3]

    # mb.torch_tensor_assign handles multi-dim slicing
    # so we need to create slice specifications for all other dimensions
    begin = [0] * x.rank
    begin[dim] = index
    begin = mb.concat(values=begin, axis=0)
    end = mb.shape(x=x)

    # and squeeze dim to do pure indexing on it
    squeeze_mask = [False] * x.rank
    squeeze_mask[dim] = True

    updated_x = _translate_torch_tensor_assign(
        x=x,
        updates=updates,
        begin=begin,
        end=end,
        squeeze_mask=squeeze_mask,
        name=node.name,
    )
    context.add(updated_x)


@register_torch_op
def slice_scatter(context, node):
    inputs = _get_inputs(context, node, min_expected=2)
    x, updates = promote_input_dtypes(inputs[0:2])
    x_shape = mb.shape(x=x)
    rank = x.rank

    # sanitize and validate dim
    dim = 0 if len(inputs) <= 2 else inputs[2].val
    if dim is None:
        raise ValueError("Only compile time known dim supported yet")
    if dim < 0:
        dim = dim + x.rank
    assert 0 <= dim and dim < x.rank, f"invalid dim: {dim}"

    # sanitize start
    start = 0 if len(inputs) <= 3 else inputs[3]
    if start is None:
        start = 0

    # sanitize end
    shape_at_dim = value_at(x_shape, dim)
    if len(inputs) <= 4:
        end = shape_at_dim
    else:
        end = inputs[4]
        if end is not None:
            end = mb.minimum(x=inputs[4], y=shape_at_dim)
        else:
            end = shape_at_dim

    # get step given different number of inputs
    step = 1 if len(inputs) <= 5 else inputs[5]

    # mb.torch_tensor_assign handles multi-dim slicing
    # so we need to pad start, end, step from scalar to x.rank
    starts = [0] * x.rank
    starts[dim] = start
    starts = mb.concat(values=starts, axis=0)

    ends = [value_at(x_shape, i) if i != dim else end for i in range(rank)]
    ends = mb.concat(values=ends, axis=0)

    steps = [1] * x.rank
    steps[dim] = step
    steps = mb.concat(values=steps, axis=0)

    updated_x = _translate_torch_tensor_assign(
        x=x,
        updates=updates,
        begin=starts,
        end=ends,
        stride=steps,
        begin_mask=None,
        end_mask=None,
        squeeze_mask=None,
        name=node.name,
    )
    context.add(updated_x)


@register_torch_op
def index_put(context, node):
    inputs = _get_inputs(context, node, min_expected=3)
    x = inputs[0]
    indices = inputs[1]
    values = inputs[2]
    accumulate = False if len(inputs) < 4 else inputs[3].val
    mode = "add" if accumulate else "update"

    assert isinstance(indices, list), "indices must be a list of tensors"
    # Usually indices is a list of non-None tensors, so we stack them and feed to mb.scatter_nd
    # However, when there exists a whole slice (i.e. :), that index is represented as None
    if any(map(lambda index: index is None, indices)):
        # We have 2 ways to translate such torch.index_put, both have pros and cons
        # 1. mb.scatter_nd
        #    * pro: can handle accumulate or update
        #    * con: can only have whole slice at last dimensions
        # 2. mb.torch_tensor_assign
        #    * pro: can have whole slice at arbitrary dimension
        #    * con: can only handle update
        # Here we use mb.torch_tensor_assign
        # TODO: explore how can we cover as many torch.index_put cases as possible
        if accumulate:
            raise NotImplementedError(
                "If there existed any whole slice (e.g. : in x[:, 0]), "
                "only torch.index_put(..., accumulate=False) handled yet"
            )

        begin = [0] * x.rank
        end = list(x.shape)
        stride = [1] * x.rank
        begin_mask = [True] * x.rank
        end_mask = [True] * x.rank
        # note: in torch slice, an indexed dim becomes size 1, rather than squeezed, e.g.
        #     x = torch.zeros((2, 3))
        #     y = x[:, 1]
        # we will get y.shape as (2, 1)
        is_dim_unity = [False] * x.rank
        for dim, index in enumerate(indices):
            if index is not None:
                if len(index.shape) > 0:
                    index = mb.squeeze(x=index)
                begin[dim] = index
                end[dim] = mb.add(x=index, y=1)
                begin_mask[dim] = False
                end_mask[dim] = False
                is_dim_unity[dim] = True
        begin = mb.concat(values=begin, axis=0)
        end = mb.concat(values=end, axis=0)

        expected_values_shape = []
        for dim in range(x.rank):
            expected_values_shape.append(1 if is_dim_unity[dim] else x.shape[dim])
        expected_values_shape = tuple(expected_values_shape)

        if values.shape != expected_values_shape:
            values = _broadcast(values.name + "_broadcasted", values, expected_values_shape)

        updated_x = _translate_torch_tensor_assign(
            x=x,
            updates=values,
            begin=begin,
            end=end,
            stride=stride,
            begin_mask=begin_mask,
            end_mask=end_mask,
            squeeze_mask=[False] * x.rank,
            name=node.name,
        )
        context.add(updated_x)
        return

    indices_type = indices[0].sym_type.get_primitive()
    if types.is_bool(indices_type):
        # indices
        if len(indices) != 1:
            raise AssertionError("Unsupported index_put_ usage.")
        indices = indices[0]
        if indices.shape != x.shape[: len(indices.shape)]:
            raise AssertionError(
                f"indices shape {indices.shape} must match input shape {x.shape} "
                "for index put operation."
            )
        indices = mb.cast(x=indices, dtype="int32")
        indices = mb.non_zero(x=indices)

        # if the indices is all False,
        # we translate the op into identity
        if 0 in indices.shape:
            result = mb.identity(x=x, name=node.name)
            context.add(result)
            return

        # values
        if values.shape == ():
            values = mb.expand_dims(x=values, axes=[0])
        if values.rank == 1 and values.shape[0] == 1:
            reps = value_at(mb.shape(x=indices), 0)
            reps = mb.expand_dims(x=reps, axes=[0])
            values = mb.tile(x=values, reps=reps)
    elif types.is_int(indices_type):
        # indices
        if len(indices) > 1:
            indices = mb.stack(values=indices, axis=indices[0].rank)
        else:
            indices = mb.expand_dims(x=indices[0], axes=[-1])
        # values
        expected_values_shape = indices.shape[:-1] + x.shape[indices.shape[-1] :]
        if values.shape != expected_values_shape:
            values = _broadcast(values.name + "_broadcasted", values, expected_values_shape)
    else:
        raise ValueError(f"Only bool and int index handled yet, but got {indices_type}")

    if is_current_opset_version_compatible_with(target.iOS17):
        # IOS17 `scatter_nd` behaviour is undefined for negative indices.
        cond = mb.greater_equal(x=indices, y=0)
        x_shape = mb.shape(x=x)
        indices_shape = mb.shape(x=indices)
        indices_last_dim = value_at(indices_shape, indices.rank - 1)
        indices_last_dim_expand = mb.expand_dims(x=indices_last_dim, axes=[0])
        slice_shape = mb.slice_by_size(x=x_shape, begin=[0], size=indices_last_dim_expand)
        indices = mb.select(
            cond=cond,
            a=indices,
            b=mb.add(x=indices, y=slice_shape),
        )

    # The `scatter_nd` op doesn't support bool for data/updates, so we need to convert them to int32.
    if types.is_bool(x.dtype):
        x = mb.cast(x=x, dtype="int32")
    if types.is_bool(values.dtype):
        values = mb.cast(x=values, dtype="int32")

    result = mb.scatter_nd(data=x, indices=indices, updates=values, mode=mode, name=node.name)
    context.add(result)


@register_torch_op(torch_alias=["_unsafe_index"])
def index(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = inputs[0]
    indices = inputs[1]
    rank = x.rank

    """
    Case 1: A single boolean index selection
    Ex:
        a = torch.rand(2, 3, 4)
        b = torch.rand(3, 4)
        index = b > 0.1
        c = a[:, b]

    For this case, the only non-None tensor is with dtype bool
    The true value indicates whether the element should be selected among the masked axes
    The output c is a tensor with shape (2, N), where N is the number of elements of b satisfying condition > 0.1
    """
    boolean_indices_axis = []
    for i, index in enumerate(indices):
        if index is not None and types.is_bool(index.dtype):
            boolean_indices_axis.append(i)
    if len(boolean_indices_axis) == 1:
        # get the True element indices
        axis = boolean_indices_axis[0]
        axes = list(range(axis, axis + index.rank))
        index = indices[axis]
        index = mb.non_zero(x=index)

        # transpose the masked axes to the beginning
        perm = axes + [i for i in range(rank) if i not in axes]
        x = mb.transpose(x=x, perm=perm)
        x = _utils._construct_gather_op("gather_nd", x, index)

        # transpose the tensor back
        perm_back = list(range(1, x.rank))
        perm_back.insert(axis, 0)
        res = mb.transpose(x=x, perm=perm_back, name=node.name)
        context.add(res)
        return

    """
    Case 2: Pure index selection
    Ex # 1 [Single dimension selection]:
        a = torch.rand(1,2,3,4)
        index = torch.tensor([0, 1])
        b = a[:,:,:,index]

        In this case, indices is a list [None, None, None, [0, 1]]]. The None element means the corresponding
        dimension is masked.

        b has shape (1,2,3,2).

    Ex # 2 [Multiple disconnected dimensions selection]:
        a = torch.rand(1,2,3,4)
        index = torch.tensor([0, 1])
        b = a[:,index,:,index]

        In this case, indices is a list [None, [0,1], None, [0,1]]

        b has shape (2,1,3),
        where b[0,:,:] = a[:,0,:,0] and b[1,:,:] = a[:,1,:,1]

    Ex # 3 [Multiple connected dimensions selection]:
        a = torch.rand(1,2,3,4)
        index_1 = torch.tensor([0, 1])
        index_2 = torch.tensor([0, 1])
        b = a[:,index_1,index_2,:]

        indices is a list [None, [0, 1], [0, 1], None]

        b has shape (1,2,4),
        where b[:,0,:] = a[:,0,0,:] and b[:,1,:] = a[:,1,1,:]

    Ex # 4 [Selection with boolean masks]:
        a = torch.rand(4,5)
        index_1 = [True, True, False, False]
        index_2 = [False, True, True, False, False]
        b = a[index_1, index_2]

        indices is a list [[True, True, False, False], [False, True, True, False, False]]

        In this case, index_1 and index_2 are interpreted as mask by indices of True,
        index_1 -> [0, 1]
        index_2 -> [1, 2]

        b has shape (2,),
        where b[0] = a[0, 1] and b[1] = a[1, 2]

    Ex # 5 [Broadcast selection]:
        a = torch.rand(1,2,3,4)
        index_1 = torch.tensor([0, 1])
        index_2 = torch.tensor([0])
        b = a[:,index_1,index_2,:]

        indices is a list [None, [0, 1], [0], None]

        In this case, index_2 is going to be broadcasted to [0, 0]

        b has shape (1,2,4),
        where b[:,0,:] = a[:,0,0,:] and b[:,1,:] = a[:,1,0,:]

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
        indices = valid_indices[0]
        if is_current_opset_version_compatible_with(target.iOS17):
            # IOS17 `gather` behaviour is undefined for negative indices.
            indices = mb.select(
                cond=mb.greater_equal(x=indices, y=0),
                a=indices,
                b=mb.add(x=indices, y=value_at(mb.shape(x=x), axis)),
            )
        x = _utils._construct_gather_op("gather", x, indices, axis, name=node.name)
        context.add(x)
        return

    # For multiple index axes case, we delegate broadcast to np if there is no dynamic shape.
    if all(not any_symbolic(idx.shape) for idx in valid_indices):
        broadcasted_shape = _np.broadcast_shapes(*[idx.shape for idx in valid_indices])
        for i, index in enumerate(valid_indices):
            if (index.shape != broadcasted_shape) and index.val is not None:
                new_val = _np.broadcast_to(index.val, broadcasted_shape)
                valid_indices[i] = mb.const(
                    val=new_val, name=index.name + "_broadcasted"
                )
    valid_indices = [mb.cast(x=index, dtype="int32") for index in valid_indices]

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

    if is_current_opset_version_compatible_with(target.iOS17):
        # IOS17 `gather_nd` behaviour is undefined for negative indices.
        cond = mb.greater_equal(x=indices, y=0)
        x_shape = mb.shape(x=x)
        indices_shape = mb.shape(x=indices)
        indices_last_dim = value_at(indices_shape, indices.rank - 1)
        indices_last_dim_expand = mb.expand_dims(x=indices_last_dim, axes=[0])
        slice_shape = mb.slice_by_size(x=x_shape, begin=[0], size=indices_last_dim_expand)
        indices = mb.select(
            cond=cond,
            a=indices,
            b=mb.add(x=indices, y=slice_shape),
        )
    x = _utils._construct_gather_op("gather_nd", x, indices, name=name)

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


@register_torch_op
def ones(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var, Optional[Var]]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: [5, 6]},
            min_expected={TorchFrontend.TORCHEXPORT: 1, TorchFrontend.EXECUTORCH: 1},
        )
        size = inputs[0]
        dtype = inputs[1] if (len(inputs) > 1 and inputs[1] is not None) else None
        return size, dtype

    def _parse_keyword_args(context, node, dtype) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dtype
        dtype = _get_kwinputs(context, node, "dtype", default=[dtype])[0]
        return dtype

    size, dtype = _parse_positional_args(context, node)
    dtype = _parse_keyword_args(context, node, dtype)

    if isinstance(size, list):
        size = mb.concat(values=size, axis=0)
    res = mb.fill(shape=size, value=1.0)
    if dtype is not None:
        res = _cast_to(res, NUM_TO_DTYPE_STRING[dtype.val], node.name)
    context.add(res, node.name)


@register_torch_op
def ones_like(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var, Optional[Var]]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: 6},
            min_expected={TorchFrontend.TORCHEXPORT: 1, TorchFrontend.EXECUTORCH: 1},
        )
        x = inputs[0]
        dtype = None
        if len(inputs) > 1 and inputs[1] is not None:
            dtype = inputs[1]
        return x, dtype

    def _parse_keyword_args(context, node, dtype) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dtype
        dtype = _get_kwinputs(context, node, "dtype", default=[dtype])[0]
        return dtype

    x, dtype = _parse_positional_args(context, node)
    dtype = _parse_keyword_args(context, node, dtype)

    if is_current_opset_version_compatible_with(target.iOS16):
        res = mb.fill_like(ref_tensor=x, value=1.0)
    else:
        res = mb.fill(shape=mb.shape(x=x), value=1.0)
        # By default use input x's dtype.
        dtype_str = NUM_TO_DTYPE_STRING[dtype.val] if dtype is not None else types.builtin_to_string(x.dtype)
        res = _cast_to(res, dtype_str, node.name)
    context.add(res, node.name)


@register_torch_op
def fill(context, node):
    inputs = _get_inputs(context, node, expected=2)
    shape = mb.shape(x=inputs[0])
    value = inputs[1].val
    result = mb.fill(shape=shape, value=value, name=node.name)
    context.add(result)


def _make_fill_op(size, val, name):
    assert val is not None
    if isinstance(size, list):
        size = mb.concat(values=size, axis=0)
    if types.is_float(size.dtype):
        size = mb.cast(x=size, dtype="int32")
    fill = mb.fill(shape=size, value=val, name=name)
    return fill


@register_torch_op
def full(context, node):
    inputs = _get_inputs(context, node, min_expected=2)

    size = inputs[0]

    # dtype could be torch.dtype or an integer that maps to a numpy.dtype
    dtype = None
    if len(inputs) < 3 or inputs[2] is None:
        dtype = np.float32
    elif isinstance(inputs[2].val, torch.dtype):
        dtype = NUM_TO_NUMPY_DTYPE[TORCH_DTYPE_TO_NUM[inputs[2].val]]
    elif isinstance(inputs[2].val, (int, np.generic)):
        dtype = NUM_TO_NUMPY_DTYPE[inputs[2].val]
    else:
        raise ValueError(f"unsupported type {type(inputs[2].val)}.")

    val = dtype(inputs[1].val)

    result = _make_fill_op(size, val, node.name)
    context.add(result)


@register_torch_op
def full_like(context, node):
    inputs = _get_inputs(context, node, min_expected=2)
    x = inputs[0]
    val = inputs[1].val

    if is_current_opset_version_compatible_with(target.iOS16):
        result = mb.fill_like(ref_tensor=x, value=val, name=node.name)
    else:
        size = mb.shape(x=inputs[0])
        result = _make_fill_op(size, val, node.name)
    context.add(result)


@register_torch_op
def new_full(context, node):
    # The difference between "new_full" and "full" is that the "new_full" is called from
    # an existing tensor: tensor.new_full(size, fill_value), while the "full" is called
    # from the torch API: torch.full(size, fill_value).
    # But they are basically doing the same thing.
    inputs = _get_inputs(context, node)
    size = inputs[1]
    val = inputs[2].val
    result = _make_fill_op(size, val, node.name)
    context.add(result)


@register_torch_op(torch_alias=["randint.low"])
def randint(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=2)
        if context.frontend == TorchFrontend.TORCHSCRIPT or node.kind == "randint.low":
            low = mb.cast(x=inputs[0], dtype="fp32")
            high = mb.cast(x=inputs[1], dtype="fp32")
            shape = inputs[2].val
        else:
            assert node.kind == "randint"
            low = 0.0
            high = mb.cast(x=inputs[0], dtype="fp32")
            shape = inputs[1].val
        return low, high, shape

    low, high, shape = _parse_positional_args(context, node)
    rand_uniform = mb.random_uniform(shape=shape, low=low, high=high)
    rand_int = mb.cast(x=rand_uniform, dtype="int32", name=node.name)
    context.add(rand_int)

@register_torch_op
def rand(context, node):
    shape, _, dtype, _, _ = _get_inputs(context, node)
    dtype = NUM_TO_DTYPE_STRING[TORCH_DTYPE_TO_NUM[dtype.val]] if dtype else "fp32"
    low, high = mb.cast(x=0.0, dtype=dtype), mb.cast(x=1.0, dtype=dtype)
    rand_uniform = mb.random_uniform(shape=shape, low=low, high=high)
    context.add(rand_uniform, node.name)


@register_torch_op
def rand_like(context, node):
    inputs = _get_inputs(
        context,
        node,
        expected={TorchFrontend.TORCHSCRIPT: 6},
        min_expected={TorchFrontend.TORCHEXPORT: 1, TorchFrontend.EXECUTORCH: 1},
    )
    x = inputs[0]
    dtype = inputs[1] if len(inputs) >= 2 else None
    dtype = NUM_TO_DTYPE_STRING[dtype.val] if dtype else types.builtin_to_string(x.dtype)
    low, high = mb.cast(x=0.0, dtype=dtype), mb.cast(x=1.0, dtype=dtype)
    shape = mb.shape(x=x)
    rand_uniform = mb.random_uniform(shape=shape, low=low, high=high)
    context.add(rand_uniform, node.name)

@register_torch_op
def randn(context, node):
    inputs = _get_inputs(context, node, expected=[5, 6])

    shape = inputs[0]
    dtype = inputs[1]
    _assert_torch_dtype_num_is_not_complex_number(dtype)
    rand_normal = mb.random_normal(shape=shape)
    rand_fp32 = mb.cast(x=rand_normal, dtype="fp32", name=node.name)
    context.add(rand_fp32)

@register_torch_op
def randn_like(context, node):
    inputs = _get_inputs(context, node, expected=6)
    x = inputs[0]
    dtype = inputs[1]
    _assert_torch_dtype_num_is_not_complex_number(dtype)
    shape = mb.shape(x=x)
    rand_normal = mb.random_normal(shape=shape)
    rand_fp32 = mb.cast(x=rand_normal, dtype="fp32", name=node.name)
    context.add(rand_fp32)

@register_torch_op
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


@register_torch_op(torch_alias=["and"])
def bitwise_and(context, node):
    inputs = _get_inputs(context, node)

    input_dtypes = [i.dtype for i in inputs]
    if all(types.is_bool(input_dtype) for input_dtype in input_dtypes):
        logical_and(context, node)
    else:
        raise NotImplementedError(
            f"The `bitwise_and` op only supports boolean input, but get {input_dtypes}."
        )


@register_torch_op
def logical_not(context, node):
    # There is an optional `out` parameter in torch.logical_not.
    inputs = _get_inputs(context, node, expected=[1, 2])
    x = inputs[0]
    if not types.is_bool(x.dtype):
        x = mb.cast(x=x, dtype="bool")
    res = mb.logical_not(x=x, name=node.name)
    context.add(res)


def _avg_pool(context, node, inputs):
    x = inputs[0]

    kernel_sizes = inputs[1]

    strides = kernel_sizes  # default strides = kernel sizes
    if len(inputs) > 2:
        strides = inputs[2]
        # TorchScript may give us empty stride, in such case
        # we still default strides to kernel sizes, but name conform to TorchScript
        if strides.op.op_type == "const" and (not list(strides.val)):
            strides = mb.const(val=kernel_sizes.val, name=strides.name)

    pad_type = "custom"
    # Need to explicitly state L-R, T-B pad
    pad = None if len(inputs) < 4 else _np.repeat(inputs[3].val, 2)

    ceil_mode = False if len(inputs) < 5 else inputs[4].val

    include_pad = True if len(inputs) < 6 else inputs[5].val

    spatial_rank = 0 if pad is None else len(pad) // 2
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


@register_torch_op
def avg_pool1d(context, node):
    inputs = _get_inputs(
        context,
        node,
        expected={TorchFrontend.TORCHSCRIPT : 6},
        min_expected={TorchFrontend.TORCHEXPORT: 2, TorchFrontend.EXECUTORCH: 2},
    )
    _avg_pool(context, node, inputs)


@register_torch_op
def avg_pool2d(context, node):
    inputs = _get_inputs(
        context,
        node,
        min_expected={
            TorchFrontend.TORCHSCRIPT: 6,
            TorchFrontend.TORCHEXPORT: 2,
            TorchFrontend.EXECUTORCH: 2,
        },
    )
    divisor_override = None if len(inputs) < 7 else inputs[6]
    if divisor_override is not None:
        raise ValueError("divisor_override is not supported for avg_pool2d")
    _avg_pool(context, node, inputs)


@register_torch_op
def avg_pool3d(context, node):
    inputs = _get_inputs(
        context,
        node,
        expected={TorchFrontend.TORCHSCRIPT : 7},
        min_expected={TorchFrontend.TORCHEXPORT: 2, TorchFrontend.EXECUTORCH: 2},
    )
    divisor_override = None if len(inputs) < 7 else inputs[6]
    if divisor_override is not None:
        raise ValueError("divisor_override is not supported for avg_pool3d")
    _avg_pool(context, node, inputs)


@register_torch_op(torch_alias=["_log_softmax"])
def log_softmax(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=2)
        nargs = len(inputs)

        x = inputs[0]
        axis = inputs[1]
        # input 2 is dtype, so we ignore

        return x, axis

    x, axis = _parse_positional_args(context, node)

    res = mb.softmax(x=x, axis=axis, name=node.name + "_softmax")
    res = mb.log(x=res, name=node.name)
    context.add(res)


@register_torch_op(torch_alias=["nll_loss_nd"])
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
    class_num = x.shape[1]

    # only support weight and ignore_index both None
    if weight is not None:
        raise NotImplementedError("Only unity weight is supported for NLLLoss.")
    if ignore_index.val != -100:
        raise NotImplementedError("ignore index not supported for NLLLoss.")

    x = mb.cast(x=x, dtype="fp32")
    x = mb.mul(x=x, y=-1.)

    target = mb.cast(x=target, dtype="int32")
    labels = mb.one_hot(indices=target, one_hot_vector_size=class_num)
    labels = mb.cast(x=labels, dtype="fp32")
    loss = mb.mul(x=x, y=labels)
    loss = mb.reduce_sum(x=loss, axes=[1])

    # reduction type
    if reduction == "none":
        out = mb.identity(x=loss, name=node.name)
    elif reduction == "sum":
        out = mb.reduce_sum(x=loss, axes=[0], keep_dims=False, name=node.name)
    elif reduction == "mean":
        out = mb.real_div(x=loss, y=_np.float32(batch_size))
        out = mb.reduce_sum(x=out, axes=[0], keep_dims=False, name=node.name)
    else:
        raise NotImplementedError("Unsupported reduction type for NLLLoss.")

    context.add(out)


@register_torch_op
def sigmoid(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.sigmoid(x=inputs[0], name=node.name)
    context.add(res)


@register_torch_op
def hardsigmoid(context, node):
    inputs = _get_inputs(context, node, expected=1)

    res = mb.sigmoid_hard(x=inputs[0], alpha=1.0 / 6, beta=0.5, name=node.name)
    context.add(res)


@register_torch_op
def gelu(context, node):
    inputs = _get_inputs(context, node)
    assert len(inputs) in (1, 2)
    mode = None
    if len(inputs) == 2:
        approximate = inputs[1].val
        if approximate == "tanh":
            mode = "TANH_APPROXIMATION"
        else:
            assert approximate == "none"
    res = mb.gelu(x=inputs[0], mode=mode, name=node.name)
    context.add(res)


@register_torch_op(torch_alias=["_slice", "slice_copy"])
def slice(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected=(1, 2, 3, 4, 5),
        )
        nargs = len(inputs)

        x = inputs[0]
        dim = inputs[1].val if nargs > 1 else 0
        start = None
        if nargs > 2:
            start = inputs[2]
            if isinstance(start, Var) and start.val is not None:
                start = start.val
        end = None
        if nargs > 3:
            end = inputs[3]
            if isinstance(end, Var) and end.val is not None:
                end = end.val
        step = inputs[4].val if nargs > 4 else 1
        return x, dim, start, end, step

    x, dim, start, end, step = _parse_positional_args(context, node)

    # torch.export may have kwargs
    if context.frontend == TorchFrontend.TORCHEXPORT:
        if dim == 0:
            dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        if start is None:
            start = _get_kwinputs(context, node, "start", default=[start])[0]
        if end is None:
            end = _get_kwinputs(context, node, "end", default=[end])[0]
        if step == 1:
            step = _get_kwinputs(context, node, "step", default=[step])[0]

    # torch start = None means Core ML start = 0
    if start is None:
        start = 0

    # dim must be constant
    if isinstance(dim, Var):
        dim = dim.val
        assert dim is not None

    if start == 0 and end is None and step == 1:
        # Handling x[:], just pass through the tensor.
        x_identity = mb.identity(x=x, name=node.name)
        context.add(x_identity)
        return

    begin_array = [0] * len(x.shape)
    begin_array[dim] = start
    end_array = [s if isinstance(s, int) else 0 for s in x.shape]
    end_mask = [True] * len(x.shape)

    if end is not None:
        end_array[dim] = end
        # if end >= x.shape[dim], then end can be ignored, i.e. end_mask[dim] = True
        end_mask[dim] = True if isinstance(end, int) and end >= x.shape[dim] else False

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


@register_torch_op(torch_alias=["split_with_sizes", "split_with_sizes_copy"])
def split(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=2)
        nargs = len(inputs)

        x = inputs[0]
        split_sizes = inputs[1]
        dim = inputs[2] if nargs > 2 else 0
        return x, split_sizes, dim

    def _parse_keyword_args(context, node, dim) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dim

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        return dim

    def _translate_torch_args(dim) -> Var:
        if isinstance(dim, Var):
            dim = dim.val
        return dim

    x, split_sizes, dim = _parse_positional_args(context, node)
    dim = _parse_keyword_args(context, node, dim)
    dim = _translate_torch_args(dim)

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
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=(1, 2))
        nargs = len(inputs)

        x = inputs[0]
        dim = inputs[1] if nargs > 1 else 0

        return x, dim

    x, dim = _parse_positional_args(context, node)
    # torch.export may have kwargs
    if context.frontend == TorchFrontend.TORCHEXPORT:
        if dim == 0:
            dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
    if isinstance(dim, Var):
        dim = dim.val

    split_sizes = [1] * x.shape[dim]
    if len(split_sizes) == 1:
        res = [mb.squeeze(x=x, axes=[dim])]
    else:
        res = mb.split(x=x, split_sizes=split_sizes, axis=dim, name=node.name)
        res = [mb.squeeze(x=x, axes=[dim]) for x in res]
    context.add(res, torch_name=node.name)


@register_torch_op(torch_alias = ["_to_copy"])
def to(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context, node, expected={TorchFrontend.TORCHSCRIPT: (1, 2, 3, 4, 5, 6, 7, 8)}
        )
        nargs = len(inputs)

        _input = inputs[0]

        # There are a lot of variants of `to` op.
        # - When len(inputs) is 7 or 8, we only care about the first two params (input and dtype).
        # - When len(inputs) == 6, the parameter is (input, _, dtype, non_blocking, copy, memory_format)
        # - When len(inputs) == 5, the parameter is (input, dtype, non_blocking, copy, memory_format)
        # - When len(inputs) == 4, the parameter is (input, dtype, non_blocking, copy)
        # - When len(inputs) == 3, the parameter is (input, non_blocking, copy)
        # We only use `input` and `dtype`, and `non_blocking` and `copy` are unused.
        if context.frontend == TorchFrontend.TORCHSCRIPT:
            dtype: Optional[Var]
            if nargs in (4, 5, 7, 8):
                dtype = inputs[1]
            elif nargs == 6:
                dtype = inputs[2]
            else:
                dtype = None

            if dtype is None:
                return _input, dtype
            elif types.is_scalar(dtype.sym_type) and dtype.val is not None:
                dtype = dtype.val
            else:
                # When the val of dtype is not available, bridge from the np dtype.
                np_type = nptype_from_builtin(dtype.dtype)
                dtype = NUMPY_DTYPE_TO_TORCH_NUM[np_type]

        # clearly distinguish each variant of max
        else:
            if node.kind in ("to.dtype", "_to_copy") and nargs > 1:
                dtype = inputs[1]
            else:
                dtype = None

        return _input, dtype

    def _parse_keyword_args(context, node, dtype) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dtype

        dtype = _get_kwinputs(context, node, "dtype", default=[dtype])[0]
        return dtype

    _input, dtype = _parse_positional_args(context, node)
    dtype = _parse_keyword_args(context, node, dtype)

    if dtype is None:
        # When dtype is None, it means the input's dtype is already the target dtype.
        context.add(_input, torch_name=node.name)
        return
    else:
        if isinstance(dtype, Var):
            dtype = dtype.val

    if isinstance(_input, Var) and _input.can_be_folded_to_const():
        # numpy -> torch -> torch cast -> numpy
        # This path is needed to use the mapping of passed in dtypes to torch dtypes.
        torch_dtype = dtype_to_32bit(NUM_TO_TORCH_DTYPE[dtype])
        res = mb.const(val=torch.tensor(_input.val).type(torch_dtype).cpu().numpy())
    else:
        dtype_str = NUM_TO_DTYPE_STRING[dtype]
        res = _cast_to(_input, dtype_str, node.name)
    context.add(res, node.name)


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

    if _input.shape == ():  # already a scalar
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


@register_torch_op(torch_alias=["expand_copy"])
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


    # PyTorch 1.6+ has 3 inputs while older version has 2
    inputs = _get_inputs(context, node, expected=[2, 3])

    x = inputs[0]
    shape = inputs[1]

    if isinstance(shape, list):
        res = _broadcast_dynamic(node.name, x, shape)
    else:
        res = _broadcast(node.name, x, shape.val)
    context.add(res)


@register_torch_op
def expand_as(context, node):
    # PyTorch 1.6+ has 3 inputs while older version has 2
    inputs = _get_inputs(context, node, expected=[2, 3])
    x = inputs[0]
    other = inputs[1]

    res = _broadcast(node.name, x, other.shape)
    context.add(res)


@register_torch_op(
    torch_alias=[
        "atleast_2d",
        "atleast_3d",
        "atleast_1d.sequence",
        "atleast_2d.sequence",
        "atleast_3d.sequence",
    ]
)
def atleast_1d(context, node):
    def _maybe_expand_dims(x: Var, rank: int, name: Optional[str] = None) -> Var:
        if x.rank < rank:
            if rank == 3:
                if x.rank == 2:
                    axes = [2]
                elif x.rank == 1:
                    axes = [0, 2]
                else:
                    axes = [0, 1, 2]
            else:
                axes = [*range(rank - x.rank)]
            kwargs = {"x": x, "axes": axes}
            if name is not None:
                kwargs["name"] = name
            x = mb.expand_dims(**kwargs)
        return x

    inputs = _get_inputs(context, node, expected=1)[0]
    rank = int(node.kind[8])
    assert rank in (1, 2, 3)

    if isinstance(inputs, (tuple, list)):
        results = []
        for x in inputs:
            results.append(_maybe_expand_dims(x, rank))
    else:
        assert isinstance(inputs, Var)
        x = inputs
        results = _maybe_expand_dims(x, rank, node.name)

    context.add(results, torch_name=node.name)


def _arange(
    context,
    node_name: str,
    start: Var,
    end: Var,
    step: Var,
):
    # torch may have mixed precision, including mixing float and int,
    # but Core ML needs these inputs to have uniform dtype
    start, end, step = promote_input_dtypes([start, end, step])

    res = mb.range_1d(start=start, end=end, step=step, name=node_name)
    context.add(res)


@register_torch_op(torch_alias=["arange.start"])
def arange(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=1)
        nargs = len(inputs)

        if context.frontend == TorchFrontend.TORCHSCRIPT:
            # dtype = inputs[-4]
            # layout = inputs[-3]
            # device = inputs[-2]
            # pin_memory = inputs[-1]
            if nargs == 1 or nargs == 5:
                # inputs are [end] or [end, dtype, layout, device, pin_memory]
                start = 0
                end = inputs[0]
                step = 1
            elif nargs == 6:
                # inputs are [start, end, dtype, layout, device, pin_memory]
                start = inputs[0]
                end = inputs[1]
                step = 1
            elif nargs == 7:
                # inputs are [start, end, step, dtype, layout, device, pin_memory]
                start = inputs[0]
                end = inputs[1]
                step = inputs[2]
            else:
                raise ValueError(f"arange must have exactly 5, 6, or 7 inputs, got {nargs}")
        else:
            if re.match(r"arange\.start.*", node.kind):
                start = inputs[0]
                assert nargs > 1, "arange.start has at least 2 positional args: start, end"
                end = inputs[1]
                if node.kind == "arange.start_step":
                    step = inputs[2] if nargs > 2 else 1
                else:
                    step = 1
            else:
                start = 0
                end = inputs[0]
                step = 1

        return start, end, step

    def _parse_keyword_args(context, node, step) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return step

        step = _get_kwinputs(context, node, "step", default=[step])[0]
        return step

    start, end, step = _parse_positional_args(context, node)
    step = _parse_keyword_args(context, node, step)

    _arange(context, node.name, start, end, step)


@register_torch_op(torch_alias=["arange.start_step"])
def arange_start_step(context, node):
    inputs = _get_inputs(context, node)
    start = inputs[0]
    end = inputs[1]
    step = 1 if len(inputs) < 3 else inputs[2]

    _arange(context, node.name, start, end, step)


@register_torch_op
def masked_fill(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    mask = inputs[1]
    value = inputs[2]

    if not types.is_bool(mask.dtype):
        # cond must be bool type
        mask = mb.cast(x=mask, dtype="bool")

    if value.dtype != x.dtype:
        value = mb.cast(x=value, dtype=builtin_to_string(x.dtype))

    value, x = promote_input_dtypes([value, x])
    res = mb.select(cond=mask, a=value, b=x, name=node.name)
    context.add(res)


@register_torch_op(torch_alias=["meshgrid.indexing"])
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

    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=[1, 2])
        nargs = len(inputs)

        tensor_inputs = inputs[0]
        indexing = inputs[1].val if nargs > 1 else "ij"
        return tensor_inputs, indexing

    def _check_args(tensor_inputs, indexing) -> None:
        assert isinstance(tensor_inputs, (list, tuple))
        if len(tensor_inputs) < 2:
            raise ValueError("Requires >= 2 tensor inputs.")
        if any([len(tensor_var.shape) > 1 for tensor_var in tensor_inputs]):
            raise ValueError("meshgrid received non-1d tensor.")

        if indexing not in ("ij", "xy"):
            raise ValueError(f"indexing mode {indexing} not supported")

    tensor_inputs, indexing = _parse_positional_args(context, node)
    # torch.export may have kwargs
    if context.frontend == TorchFrontend.TORCHEXPORT:
        if indexing == "ij":
            indexing = _get_kwinputs(context, node, "indexing", default=[indexing])[0]
    _check_args(tensor_inputs, indexing)

    dim_tuple = tuple(tensor_var.shape[0] for tensor_var in tensor_inputs)

    grids = []
    size = len(tensor_inputs)
    for i in range(size):
        view_shape = [1] * size
        view_shape[i] = -1
        view_shape = tuple(view_shape)
        # (a.) in docstring
        view = mb.reshape(
            x=tensor_inputs[i], shape=view_shape, name=node.name + "_view_" + str(i)
        )

        # (b.) in docstring
        reps = [
            ds if ds > 0 and ts == 1 else 1 for ts, ds in zip(view.shape, dim_tuple)
        ]
        res = mb.tile(x=view, reps=reps, name=node.name + "_expand_" + str(i))

        # transpose the first two dimensions for "xy" indexing
        if indexing == "xy":
            perm = [1, 0] + list(range(2, size))
            res = mb.transpose(x=res, perm=perm, name=node.name + "_transpose_" + str(i))

        grids.append(res)

    context.add(tuple(grids), node.name)


# Defines all the nodes that are noOps
@register_torch_op(
    torch_alias=[
        "_assert_async.msg",
        "_assert_scalar",
        "_local_scalar_dense",
        "alias_copy",
        "clone",
        "contiguous",
        "detach",
        "device",
        "dropout",
        "feature_dropout",
        "lift_fresh",
        "lift_fresh_copy",
        "sym_constrain_range",
        "sym_constrain_range_for_size",
    ]
)
def noop(context, node):
    logger.info(f"Setting pytorch op: {node.kind} to no-op.")
    # These noops do not produce output
    if node.kind in ("_assert_scalar", "sym_constrain_range", "sym_constrain_range_for_size"):
        return
    # Other noops return input as output
    else:
        inputs = _get_inputs(context, node)
        _input = inputs[0]
        context.add(_input, torch_name=node.name)


@register_torch_op
def argmax(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=(1, 2, 3, 4))
        nargs = len(inputs)

        x = inputs[0]

        dim = inputs[1] if nargs > 1 else None
        keepdim = inputs[2] if nargs > 2 else False

        # When node.kind == argmax.out, there can be 1 more arg `Tensor(a!) out`,
        # which is for in-place mutation, so we ignore it since Core ML is functional
        return x, dim, keepdim

    def _parse_keyword_args(context, node, dim, keepdim) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dim, keepdim

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        keepdim = _get_kwinputs(context, node, "keepdim", default=[keepdim])[0]

        return dim, keepdim

    x, dim, keepdim = _parse_positional_args(context, node)
    dim, keepdim = _parse_keyword_args(context, node, dim, keepdim)
    if isinstance(dim, Var):
        dim = dim.val
    if isinstance(keepdim, Var):
        keepdim = keepdim.val

    if types.is_int(x.dtype) and x.dtype._width == 64:
        # MIL reduce_argmax doesn't support int64.
        x = mb.cast(x=x, dtype="int32")
    res = mb.reduce_argmax(x=x, axis=dim, keep_dims=keepdim, name=node.name)
    context.add(res)


@register_torch_op(torch_alias=["empty_like"])
def zeros_like(context, node):
    inputs = _get_inputs(
        context,
        node,
        expected={TorchFrontend.TORCHSCRIPT: 6},
        min_expected={TorchFrontend.TORCHEXPORT: 1, TorchFrontend.EXECUTORCH: 1},
    )
    x = inputs[0]
    src_np_type = nptype_from_builtin(x.dtype)
    if len(inputs) > 1 and inputs[1] and inputs[1].val:
        dtype = inputs[1].val
        dst_np_type = NUM_TO_NUMPY_DTYPE[dtype]
    else:
        dst_np_type = src_np_type

    shape = mb.shape(x=x)
    if shape.can_be_folded_to_const():
        shape = shape.val
        zeros = _np.zeros(shape).astype(dst_np_type)
        zeros_like = mb.const(val=zeros, name=node.name)
    else:
        if src_np_type == np.bool_:
            zeros = mb.logical_xor(x=x, y=x)
        else:
            zeros = mb.sub(x=x, y=x)
        if src_np_type != dst_np_type:
            num = NUMPY_DTYPE_TO_TORCH_NUM[dst_np_type]
            zeros_like = mb.cast(x=zeros, dtype=NUM_TO_DTYPE_STRING[num], name=node.name)
        else:
            zeros.name = node.name
            zeros_like = zeros

    context.add(zeros_like)


@register_torch_op(torch_alias=["empty", "empty.memory_format"])
def zeros(context, node):
    inputs = _get_inputs(context, node, min_expected=1)
    size = inputs[0]

    if len(inputs) > 1 and inputs[1] is not None:
        dtype = inputs[1].val
    else:
        dtype = torch.get_default_dtype()
        assert dtype in (torch.float32, torch.float64)
        dtype = 6

    if isinstance(size, list) or not size.can_be_folded_to_const():
        # the size is dynamic or this zeros op cannot be folded into const.
        size = mb.concat(values=size, axis=0) if isinstance(size, list) else size
        np_type = NUM_TO_NUMPY_DTYPE[dtype]
        zeros = mb.fill(shape=size, value=np_type(0), name=node.name)
    else:
        # the size is static and this zeros op can be folded into const.
        size = size.val
        # layout = inputs[2] unused
        # device = inputs[3] unused
        # pin_memory = inputs[4] unused
        torch_dtype = dtype_to_32bit(NUM_TO_TORCH_DTYPE[dtype])
        zeros_array = torch.zeros(tuple(size)).type(torch_dtype).numpy()
        zeros = mb.const(val=zeros_array, name=node.name)

    context.add(zeros)


@register_torch_op(torch_alias=["new_empty"])
def new_zeros(context, node):
    inputs = _get_inputs(context, node)
    shape = inputs[1]
    if isinstance(shape, list):
        # when the size is dynamic, it is a list of pymil scalar,
        # we need to concat them first to get a shape.
        shape = mb.concat(values=shape, axis=0)
    context.add(mb.fill(shape=shape, value=0., name=node.name))


@register_torch_op
def scalar_tensor(context, node):
    x = _get_inputs(context, node, expected=[1, 5])[0]
    res = mb.identity(x=x, name=node.name)
    context.add(res)


@register_torch_op
def dim(context, node):
    inputs = _get_inputs(context, node)
    shape = mb.shape(x=inputs[0])
    rank = mb.shape(x=shape)
    context.add(value_at(rank, 0, node.name))


def _add_max_min(context, node, reduce_op, reduce_arg_op, alias_op):
    if context.frontend == TorchFrontend.TORCHSCRIPT:
        # mimic functionality from https://pytorch.org/docs/stable/generated/torch.min.html
        # mimic functionality from https://pytorch.org/docs/stable/generated/torch.max.html

        inputs = _get_inputs(context, node, expected=[1, 2, 3])
        if len(inputs) == 1:
            value = reduce_op(x=inputs[0], axes=None, name=node.name)
            context.add(value)
        elif len(inputs) == 2:
            value = alias_op(x=inputs[0], y=inputs[1], name=node.name)
            context.add(value)
        elif len(inputs) == 3:
            _input = inputs[0]
            dim = inputs[1].val
            keepdim = inputs[2].val

            values = reduce_op(x=_input, axes=[dim], keep_dims=keepdim)
            indices = reduce_arg_op(x=_input, axis=dim, keep_dims=keepdim)
            assert len(node.outputs) == 2
            values_name = node.outputs[0]
            indices_name = node.outputs[1]
            context.add(values, torch_name=values_name)
            context.add(indices, torch_name=indices_name)

    else:
        # clearly distinguish each variant of max

        def _parse_positional_args(context, node) -> Tuple[Var]:
            inputs = _get_inputs(context, node, min_expected=1)
            nargs = len(inputs)

            x = inputs[0]
            dim = None if nargs < 2 else inputs[1].val
            keepdim = False if nargs < 3 else inputs[2].val

            return x, dim, keepdim

        x, dim, keepdim = _parse_positional_args(context, node)

        func_suffix = node.kind.split(".")
        if len(func_suffix) == 1:
            value = reduce_op(x=x, axes=None, name=node.name)
            context.add(value)
        elif func_suffix[-1] == "dim":
            values = reduce_op(x=x, axes=[dim], keep_dims=keepdim)
            indices = reduce_arg_op(x=x, axis=dim, keep_dims=keepdim)
            context.add((values, indices), torch_name=node.name)


@register_torch_op(torch_alias=["max.dim"])
def max(context, node):
    _add_max_min(context, node, mb.reduce_max, mb.reduce_argmax, mb.maximum)


@register_torch_op(torch_alias=["min.dim"])
def min(context, node):
    _add_max_min(context, node, mb.reduce_min, mb.reduce_argmin, mb.minimum)


def _add_amax_amin(context, node, reduce_op):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=(1, 2, 3))
        nargs = len(inputs)

        x = inputs[0]
        dim = inputs[1] if nargs > 1 else []
        keepdim = inputs[2] if nargs > 2 else False

        return x, dim, keepdim

    def _parse_keyword_args(context, node, dim, keepdim) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dim, keepdim

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        keepdim = _get_kwinputs(context, node, "keepdim", default=[keepdim])[0]

        return dim, keepdim

    def _translate_torch_args(dim) -> Var:
        if isinstance(dim, Var):
            dim = dim.val
        if dim is None or len(dim) == 0:
            axes = None
        else:
            if isinstance(dim, int):
                axes = [dim]
            else:
                axes = [axis for axis in dim]
        return axes

    x, dim, keepdim = _parse_positional_args(context, node)
    dim, keepdim = _parse_keyword_args(context, node, dim, keepdim)
    axes = _translate_torch_args(dim)

    result = reduce_op(x=x, axes=axes, keep_dims=keepdim, name=node.name)
    context.add(result)

@register_torch_op
def amax(context, node):
    _add_amax_amin(context, node, mb.reduce_max)

@register_torch_op
def amin(context, node):
    _add_amax_amin(context, node, mb.reduce_min)


@register_torch_op
def argsort(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: 3},
            min_expected={TorchFrontend.TORCHEXPORT: 1, TorchFrontend.EXECUTORCH: 1},
        )
        nargs = len(inputs)

        x = inputs[0]
        dim = inputs[1] if nargs > 1 else -1
        descending = inputs[2] if nargs > 2 else False

        return x, dim, descending

    def _parse_keyword_args(context, node, dim, descending) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dim, descending

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        descending = _get_kwinputs(context, node, "descending", default=[descending])[0]

        return dim, descending

    x, dim, descending = _parse_positional_args(context, node)
    dim, descending = _parse_keyword_args(context, node, dim, descending)

    ascending = mb.logical_not(x=descending)
    argsort = mb.argsort(x=x, axis=dim, ascending=ascending, name=node.name)
    context.add(argsort)


@register_torch_op
def sort(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: 3},
            min_expected={TorchFrontend.TORCHEXPORT: 1, TorchFrontend.EXECUTORCH: 1},
        )
        nargs = len(inputs)

        x = inputs[0]
        dim = inputs[1] if nargs > 1 else -1
        descending = inputs[2] if nargs > 2 else False

        return x, dim, descending

    def _parse_keyword_args(context, node, dim, descending) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dim, descending

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        descending = _get_kwinputs(context, node, "descending", default=[descending])[0]

        return dim, descending

    x, dim, descending = _parse_positional_args(context, node)
    dim, descending = _parse_keyword_args(context, node, dim, descending)

    ascending = mb.logical_not(x=descending)
    if context.frontend == TorchFrontend.TORCHSCRIPT:
        indices_name = node.outputs[1]
        values_name = node.outputs[0]
        indices = mb.argsort(x=x, axis=dim, ascending=ascending, name=indices_name)
        values = mb.gather_along_axis(x=x, indices=indices, axis=dim, name=values_name)
        context.add(values, torch_name=values_name)
        context.add(indices, torch_name=indices_name)
    else:
        indices = mb.argsort(x=x, axis=dim, ascending=ascending)
        values = mb.gather_along_axis(x=x, indices=indices, axis=dim)
        context.add((values, indices), torch_name=node.name)


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
        raise ValueError(
            "can only append to Python list or MIL ListVar, got {}.".format(
                type(inputs[0])
            )
        )


@register_torch_op
def gather(context, node):
    inputs = _get_inputs(context, node)
    res = mb.gather_along_axis(x=inputs[0], indices=inputs[2], axis=inputs[1], name=node.name)
    context.add(res)


@register_torch_op
def index_select(context, node):
    x, axis, indices = _get_inputs(context, node, expected=3)
    context.add(mb.gather(x=x, indices=indices, axis=axis, name=node.name))


@register_torch_op(torch_alias=["abs"])
def _abs(context, node):
    x = _get_inputs(context, node, expected=1)[0]
    if types.is_complex(x.dtype):
        context.add(mb.complex_abs(x=x, name=node.name))
    else:
        context.add(mb.abs(x=x, name=node.name))


@register_torch_op
def repeat(context, node):
    x, reps = _get_inputs(context, node, expected=2)
    if isinstance(reps, list):
        reps = mb.concat(values=reps, axis=0)

    if reps.shape[0] > len(x.shape):
        x = mb.expand_dims(x=x, axes=list(range(reps.shape[0] - x.rank)))
    context.add(mb.tile(x=x, reps=reps, name=node.name))


@register_torch_op(torch_alias=["repeat_interleave.self_tensor", "repeat_interleave.self_int"])
def repeat_interleave(context, node):
    """
    For now, we only support scalar repeats + None or 0 dim
    """

    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: 4},
            min_expected={TorchFrontend.TORCHEXPORT: 2, TorchFrontend.EXECUTORCH: 2},
        )
        nargs = len(inputs)

        x = inputs[0]
        repeats = inputs[1]
        dim = inputs[2] if nargs > 2 else None
        return x, repeats, dim

    def repeat_interleave_dim0(x: Var, repeats_val: int, name: str = None) -> Var:
        """
        on a high level:
             x
             | tile in dim 0
             v
            [x, x, ...]
             | reshape to split the repeats
             v
            [[x],
             [x],
             ...]
             | transpose(1, 0)
             V
            [x^T, x^T, ...]
             | flatten
             V
            result
        """

        translation_kwargs = {}
        if name is not None:
            translation_kwargs["name"] = name

        x_shape = mb.shape(x=x)

        reps = [1] * x.rank
        reps[0] = repeats_val
        x_tiled = mb.tile(x=x, reps=reps)

        split_reps_shape = mb.concat(values=([repeats_val], x_shape), axis=0)
        x_reshaped = mb.reshape(x=x_tiled, shape=split_reps_shape)

        perm = [*range(x.rank + 1)]
        perm[0] = 1
        perm[1] = 0
        x_transposed = mb.transpose(x=x_reshaped, perm=perm)

        x_unaffected_sizes = mb.slice_by_index(x=x_shape, begin=[1], end=[x.rank])
        result_shape = mb.concat(values=([-1], x_unaffected_sizes), axis=0)
        result = mb.reshape(x=x_transposed, shape=result_shape, **translation_kwargs)

        return result

    x, repeats, dim = _parse_positional_args(context, node)
    # torch.export may have kwargs
    if context.frontend == TorchFrontend.TORCHEXPORT:
        if dim is None:
            dim = _get_kwinputs(context, node, "dim", default=[dim])[0]

    repeats_val = repeats.val
    if isinstance(repeats_val, np.ndarray):
        repeats_val0 = np.expand_dims(repeats_val, 0).reshape(-1)[0]
        if np.any(repeats_val != repeats_val0):
            raise NotImplementedError(
                "Conversion for torch.repeat_interleave with Tensor repeats has not been implemented"
            )
        repeats_val = repeats_val0

    is_dim_0 = True
    # This would operate on the flattened input tensor
    if dim is None:
        x = mb.reshape(x=x, shape=(-1,))
    else:
        dim_val = dim.val
        assert dim_val is not None, "torch.repeat_interleave uses static dim"
        if dim_val < 0:
            dim_val += x.rank
        # non-0 dim requires additional pre and post treatment
        if dim_val != 0:
            is_dim_0 = False

    # quick return: repeat 1 is noop
    if repeats_val == 1:
        context.add(x, torch_name=node.name)
        return

    if is_dim_0:
        result = repeat_interleave_dim0(x, repeats_val, node.name)
    else:
        # pre treatment: permute to have dim 0
        perm2dim0 = [dim_val]
        for i in range(x.rank):
            if i != dim_val:
                perm2dim0.append(i)
        x = mb.transpose(x=x, perm=perm2dim0)

        result_of_dim0 = repeat_interleave_dim0(x, repeats_val)

        # post treatment: permute back to original dim
        perm_back = [0] * x.rank
        for i in range(x.rank):
            perm_back[perm2dim0[i]] = i
        result = mb.transpose(x=result_of_dim0, perm=perm_back, name=node.name)

    context.add(result)


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
def atan2(context, node):
    """
    atan2(Tensor y, Tensor x)
    Element-wise arctangent of y / x with consideration of the quadrant
    Returns a new tensor with the signed angles in radians between vector (x, y) and vector (1, 0)

    On a high level:
    1. atan(y / x) to get the angle in [-pi / 2, pi / 2]
    2. analyze quadrant to determine the angle in [-pi, pi]

    Reference PyTorch code https://gist.github.com/nikola-j/b5bb6b141b8d9920318677e1bba70466
    def my_atan2(y, x):
        pi = torch.from_numpy(np.array([np.pi])).to(y.device, y.dtype)
        ans = torch.atan(y / x)
        ans += ((y > 0) & (x < 0)) * pi
        ans -= ((y < 0) & (x < 0)) * pi
        ans *= (1 - ((y > 0) & (x == 0)) * 1.0)
        ans += ((y > 0) & (x == 0)) * (pi / 2)
        ans *= (1 - ((y < 0) & (x == 0)) * 1.0)
        ans += ((y < 0) & (x == 0)) * (-pi / 2)
        return ans
    """
    inputs = _get_inputs(context, node, expected=2)
    y = inputs[0]
    x = inputs[1]
    if not types.is_float(y.dtype):
        y = mb.cast(x=y, dtype="fp32")
    if not types.is_float(x.dtype):
        x = mb.cast(x=x, dtype="fp32")

    # basic logical expressions
    y_less_0 = mb.less(x=y, y=0.0)
    y_greater_0 = mb.greater(x=y, y=0.0)
    x_less_0 = mb.less(x=x, y=0.0)
    x_equal_0 = mb.equal(x=x, y=0.0)

    # combined logical expressions
    ygreater0_and_xless0 = mb.logical_and(x=y_greater_0, y=x_less_0)
    yless0_and_xless0 = mb.logical_and(x=y_less_0, y=x_less_0)
    ygreater0_and_xequal0 = mb.logical_and(x=y_greater_0, y=x_equal_0)
    yless0_and_xequal0 = mb.logical_and(x=y_less_0, y=x_equal_0)

    # bool -> fp32 for numeric operation
    ygreater0_and_xless0_numeric = mb.cast(x=ygreater0_and_xless0, dtype="fp32")
    yless0_and_xless0_numeric = mb.cast(x=yless0_and_xless0, dtype="fp32")
    ygreater0_and_xequal0_numeric = mb.cast(x=ygreater0_and_xequal0, dtype="fp32")
    yless0_and_xequal0_numeric = mb.cast(x=yless0_and_xequal0, dtype="fp32")

    # quadrant modification coefficients
    coeff1 = mb.mul(x=ygreater0_and_xless0_numeric, y=_np.pi)
    coeff2 = mb.mul(x=yless0_and_xless0_numeric, y=_np.pi)
    coeff3 = mb.sub(x=1.0, y=ygreater0_and_xequal0_numeric)
    coeff4 = mb.mul(x=ygreater0_and_xequal0_numeric, y=_np.pi / 2.0)
    coeff5 = mb.sub(x=1.0, y=yless0_and_xequal0_numeric)
    coeff6 = mb.mul(x=yless0_and_xequal0_numeric, y=-_np.pi / 2.0)

    # if -1e-8 < x < 1e-8, x += 2e-8 to avoid y / 0
    # this shift makes atan2(0, 0) = 0, which is consistent with PyTorch torch.atan2
    x0left = mb.greater(x=x, y=-1e-8)
    x0right = mb.less(x=x, y=1e-8)
    x0 = mb.logical_and(x=x0left, y=x0right)
    x0numeric = mb.cast(x=x0, dtype="fp32")
    safe_shift = mb.mul(x=x0numeric, y=2e-8)
    x_safe = mb.add(x=x, y=safe_shift)

    # compute atan(y / x)
    ydx = mb.real_div(x=y, y=x_safe)
    atan2_1 = mb.atan(x=ydx)

    # analyze quadrant
    atan2_2 = mb.add(x=atan2_1, y=coeff1)
    atan2_3 = mb.sub(x=atan2_2, y=coeff2)
    atan2_4 = mb.mul(x=atan2_3, y=coeff3)
    atan2_5 = mb.add(x=atan2_4, y=coeff4)
    atan2_6 = mb.mul(x=atan2_5, y=coeff5)
    context.add(mb.add(x=atan2_6, y=coeff6, name=node.name))


@register_torch_op
def atanh(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.atanh(x=inputs[0], name=node.name))


@register_torch_op
def ceil(context, node):
    inputs = _get_inputs(context, node, expected=1)
    context.add(mb.ceil(x=inputs[0], name=node.name))


@register_torch_op(torch_alias=["clip"])
def clamp(context, node):
    inputs = _get_inputs(context, node, expected=[1,2,3])
    x = inputs[0]
    min_val = inputs[1] if (len(inputs) > 1 and inputs[1]) else mb.const(val=_np.finfo(_np.float32).min)
    max_val = inputs[2] if (len(inputs) > 2 and inputs[2]) else mb.const(val=_np.finfo(_np.float32).max)
    x, min_val, max_val = promote_input_dtypes([x, min_val, max_val])

    if min_val.val is not None and max_val.val is not None and min_val.val >= max_val.val:
        # When min >= max, PyTorch sets all values to max.
        context.add(mb.fill(shape=mb.shape(x=x), value=max_val.val, name=node.name))
        return

    if min_val.val is None or max_val.val is None or not types.is_float(x.dtype):
        # The `mb.clip` only support const alpha and beta, and requires the input to be float tensor.
        # For other cases, need to use `mb.minimum` and `mb.maximum`.
        res = mb.minimum(x=x, y=max_val)
        res = mb.maximum(x=res, y=min_val)
    else:
        res = mb.clip(x=x, alpha=min_val, beta=max_val)

    context.add(res, node.name)


@register_torch_op
def triu(context, node):
    assert context.frontend != TorchFrontend.EXECUTORCH, "triu is not a core aten op"
    inputs = _get_inputs(
        context,
        node,
        expected={
            TorchFrontend.TORCHSCRIPT: 2,
            TorchFrontend.TORCHEXPORT: [1, 2],
        },
    )
    x = inputs[0]
    if len(inputs) > 1 and inputs[1] is not None and inputs[1].val is not None:
        diagonal = inputs[1].val
    else:
        diagonal = 0
    if diagonal <= 0:
        res = mb.band_part(x=x, lower=-diagonal, upper=-1)
    else:
        y = mb.band_part(x=x, lower=-1, upper=diagonal - 1)
        use_bool = False
        if types.is_bool(x.dtype):
            # The `mb.sub` op doesn't support bool.
            use_bool = True
            x = mb.cast(x=x, dtype="int32")
            y = mb.cast(x=y, dtype="int32")
        res = mb.sub(x=x, y=y)
        if use_bool:
            res = mb.cast(x=res, dtype="bool")
    context.add(res, node.name)


@register_torch_op
def tril(context, node):
    assert context.frontend != TorchFrontend.EXECUTORCH, "tril is not a core aten op"
    inputs = _get_inputs(
        context,
        node,
        expected={
            TorchFrontend.TORCHSCRIPT: 2,
            TorchFrontend.TORCHEXPORT: [1, 2],
        },
    )
    x = inputs[0]
    if len(inputs) > 1 and inputs[1] is not None and inputs[1].val is not None:
        diagonal = inputs[1].val
    else:
        diagonal = 0
    if diagonal >= 0:
        res = mb.band_part(x=x, lower=-1, upper=diagonal)
    else:
        y = mb.band_part(x=x, lower=-diagonal - 1, upper=-1)
        use_bool = False
        if types.is_bool(x.dtype):
            # The `mb.sub` op doesn't support bool.
            use_bool = True
            x = mb.cast(x=x, dtype="int32")
            y = mb.cast(x=y, dtype="int32")
        res = mb.sub(x=x, y=y)
        if use_bool:
            res = mb.cast(x=res, dtype="bool")
    context.add(res, node.name)


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
def expm1(context, node):
    inputs = _get_inputs(context, node, expected=1)
    exp = mb.exp(x=inputs[0])
    expm1 = mb.sub(x=exp, y=1.0, name=node.name)
    context.add(expm1)


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
    x = inputs[0]
    if types.is_int(x.dtype):
        x = mb.cast(x=x, dtype="fp32")
    context.add(mb.log(x=x, name=node.name))


@register_torch_op
def log1p(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    if types.is_int(x.dtype):
        x = mb.cast(x=x, dtype="fp32")
    context.add(mb.log(x=x, epsilon=1.0, name=node.name))


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

    mul_node = mb.linear_activation(
        x=gt_node_32,
        alpha=float(threshold_val.val - alpha.val),
        beta=0.,
        name=node.name + '_mul'
    )
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
def logical_and(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = inputs
    x = mb.cast(x=x, dtype="bool")
    y = mb.cast(x=y, dtype="bool")
    context.add(mb.logical_and(x=x, y=y, name=node.name))

@register_torch_op
def logical_or(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x, y = inputs
    x = mb.cast(x=x, dtype="bool")
    y = mb.cast(x=y, dtype="bool")
    context.add(mb.logical_or(x=x, y=y, name=node.name))


@register_torch_op
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


@register_torch_op(torch_alias=["where.self"])
def where(context, node):
    inputs = _get_inputs(context, node)

    if len(inputs) == 1:
        _nonzero_as_tuple(context, node, inputs[0])
        return

    assert len(inputs) == 3
    cond, a, b = inputs
    a, b = promote_input_dtypes([a, b])
    if not types.is_bool(cond.dtype):
        # cond must be bool type
        cond = mb.cast(x=cond, dtype="bool")
    if not any([any_symbolic(x.shape) for x in (cond, a, b)]):
        # broadcast all tensors to the same shape
        cond, a, b = _broadcast_tensors([cond, a, b])
    result = mb.select(cond=cond, a=a, b=b, name=node.name)
    context.add(result)


@register_torch_op
def nonzero_numpy(context, node):
    inputs = _get_inputs(context, node, expected=1)
    _nonzero_as_tuple(context, node, inputs[0])


@register_torch_op
def neg(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x, y = promote_input_dtypes([inputs[0], -1])
    context.add(mb.mul(x=x, y=y, name=node.name))


@register_torch_op
def topk(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=(2, 3, 4, 5, 6, 7))
        nargs = len(inputs)

        x = inputs[0]
        k = inputs[1]

        dim = inputs[2] if nargs > 2 else -1
        largest = inputs[3] if nargs > 3 else True
        sorted = inputs[4] if nargs > 4 else True

        # When node.kind == topk.values, there can be 2 more args
        # `Tensor(a!) values` and `Tensor(b!) indices`, which are for in-place mutation,
        # so we ignore them since Core ML is functional
        return x, k, dim, largest, sorted

    def _parse_keyword_args(context, node, dim, largest, sorted) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dim, largest, sorted

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        largest = _get_kwinputs(context, node, "largest", default=[largest])[0]
        sorted = _get_kwinputs(context, node, "sorted", default=[sorted])[0]

        return dim, largest, sorted

    def _translate_torch_args(dim, largest, sorted) -> Tuple[Var]:
        if isinstance(dim, Var):
            dim = dim.val

        if isinstance(largest, Var):
            largest = largest.val

        if isinstance(sorted, Var):
            sorted = sorted.val
        if not sorted and not is_current_opset_version_compatible_with(target.iOS16):
            raise Exception("For opset <= iOS16, only sorted=True supported for the topk")

        return dim, not largest, sorted

    x, k, dim, largest, sorted = _parse_positional_args(context, node)
    dim, largest, sorted = _parse_keyword_args(context, node, dim, largest, sorted)
    axis, ascending, sort = _translate_torch_args(dim, largest, sorted)

    kwargs = {"name": node.name, "x": x, "k": k, "axis": axis, "ascending": ascending}
    if is_current_opset_version_compatible_with(target.iOS16):
        kwargs["sort"] = sort
    # if axis is not None:
    #     kwargs["axis"] = axis
    # if ascending is not None and ascending:
    #     kwargs["ascending"] = ascending
    # if sort is not None and not sort:
    #     kwargs["sort"] = sort

    if kwargs["k"].val is None:
        res = _utils.dynamic_topk(
            x=kwargs["x"], k=kwargs["k"], axis=kwargs["axis"], ascending=kwargs["ascending"]
        )
    else:
        res = mb.topk(**kwargs)
    if context.frontend == TorchFrontend.TORCHSCRIPT:
        values_name = node.outputs[0]
        indices_name = node.outputs[1]
        context.add(res[0], torch_name=values_name)
        context.add(res[1], torch_name=indices_name)
    else:
        context.add(res, torch_name=node.name)


def _var(
    x: Var,
    axes: Tuple[int] = None,
    keep_dims: bool = False,
    unbiased: bool = None,
    correction: float = None,
):
    if unbiased is not None and correction is not None:
        raise ValueError("Cannot specify both unbiased and correction in var")

    # compute biased variance
    x_mean = mb.reduce_mean(x=x, axes=axes, keep_dims=True)
    x_demeaned = mb.sub(x=x, y=x_mean)
    x_demeaned_square = mb.square(x=x_demeaned)
    variance = mb.reduce_mean(x=x_demeaned_square, axes=axes, keep_dims=keep_dims)

    # debias / correct if requested
    if unbiased or correction is not None:
        shape = mb.shape(x=x)
        if axes is None:
            numel = mb.reduce_prod(x=shape)
        else:
            sizes = mb.concat(values=[value_at(shape, axis) for axis in axes], axis=0)
            numel = mb.reduce_prod(x=sizes)
        numel_fp = mb.cast(x=numel, dtype="fp32")
        if unbiased:
            numel_minus_1_fp = mb.sub(x=numel_fp, y=1.0)
            scale = mb.real_div(x=numel_fp, y=numel_minus_1_fp)
        else:
            correction_fp = mb.cast(x=correction, dtype="fp32")
            numel_corrected_fp = mb.sub(x=numel_fp, y=correction_fp)
            scale = mb.real_div(x=numel_fp, y=numel_corrected_fp)
        variance = mb.mul(x=variance, y=scale)

    return variance


@register_torch_op
def numel(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    x = mb.shape(x=x)
    x = mb.reduce_prod(x=x, axes=[0], name=node.name)
    context.add(x)


@register_torch_op(torch_alias=["var.dim"])
def var(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=(1, 2, 3, 4))
        nargs = len(inputs)

        x = inputs[0]

        if context.frontend == TorchFrontend.TORCHSCRIPT:
            # torch.jit.trace does not distinguish by name `std` and `std.dim`,
            # instead by nargs 2 or 4
            keepdim = False
            dim = None
            if len(inputs) == 2:
                unbiased = inputs[1]
            if len(inputs) == 4:
                dim = inputs[1]
                unbiased = inputs[2]
                keepdim = inputs[3]
        else:
            if node.kind == "var":
                unbiased = inputs[1] if nargs > 1 else True
                dim = None
                keepdim = False
            else:
                assert node.kind == "var.dim"
                assert nargs > 1
                dim = inputs[1]
                unbiased = inputs[2] if nargs > 2 else True
                keepdim = inputs[3] if nargs > 3 else False

        return x, dim, unbiased, keepdim

    def _parse_keyword_args(context, node, dim, unbiased, keepdim) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dim, unbiased, keepdim

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        unbiased = _get_kwinputs(context, node, "unbiased", default=[unbiased])[0]
        keepdim = _get_kwinputs(context, node, "keepdim", default=[keepdim])[0]

        return dim, unbiased, keepdim

    def _translate_torch_args(dim, unbiased, keepdim) -> Tuple[Var]:
        if isinstance(dim, Var):
            dim = dim.val
        try:
            dim = (int(dim),)
        except:
            pass

        if isinstance(unbiased, Var):
            unbiased = unbiased.val

        if isinstance(keepdim, Var):
            keepdim = keepdim.val

        return dim, unbiased, keepdim

    x, dim, unbiased, keepdim = _parse_positional_args(context, node)
    dim, unbiased, keepdim = _parse_keyword_args(context, node, dim, unbiased, keepdim)
    axes, unbiased, keep_dims = _translate_torch_args(dim, unbiased, keepdim)

    y = _var(x, axes=axes, keep_dims=keep_dims, unbiased=unbiased)
    context.add(y, node.name)


@register_torch_op(torch_alias=["var.correction"])
def var_correction(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=(1, 2, 3, 4))
        nargs = len(inputs)

        x = inputs[0]
        dim = inputs[1] if nargs > 1 else None
        correction = inputs[2] if nargs > 2 else None
        keepdim = inputs[3] if nargs > 3 else False

        return x, dim, correction, keepdim

    def _parse_keyword_args(context, node, dim, correction, keepdim) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dim, correction, keepdim

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        correction = _get_kwinputs(context, node, "correction", default=[correction])[0]
        keepdim = _get_kwinputs(context, node, "keepdim", default=[keepdim])[0]

        return dim, correction, keepdim

    def _translate_torch_args(dim, correction, keepdim) -> Tuple[Var]:
        if isinstance(dim, Var):
            dim = dim.val
        try:
            dim = (int(dim),)
        except:
            pass

        if isinstance(correction, Var):
            correction = correction.val

        if isinstance(keepdim, Var):
            keepdim = keepdim.val

        return dim, correction, keepdim

    x, dim, correction, keepdim = _parse_positional_args(context, node)
    dim, correction, keepdim = _parse_keyword_args(context, node, dim, correction, keepdim)
    axes, correction, keep_dims = _translate_torch_args(dim, correction, keepdim)

    y = _var(x, axes=axes, keep_dims=keep_dims, correction=correction)
    context.add(y, node.name)


@register_torch_op(torch_alias=["std.dim"])
def std(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={
                TorchFrontend.TORCHSCRIPT: (2, 4),
                TorchFrontend.TORCHEXPORT: (1, 2, 3, 4),
                TorchFrontend.EXECUTORCH: (1, 2, 3, 4),
            },
        )
        nargs = len(inputs)

        x = inputs[0]

        if context.frontend == TorchFrontend.TORCHSCRIPT:
            # torch.jit.trace does not distinguish by name `std` and `std.dim`,
            # instead by nargs 2 or 4
            keepdim = False
            dim = None
            if len(inputs) == 2:
                unbiased = inputs[1]
            if len(inputs) == 4:
                dim = inputs[1]
                unbiased = inputs[2]
                keepdim = inputs[3]
        else:
            if node.kind == "std":
                unbiased = inputs[1] if nargs > 1 else True
                dim = None
                keepdim = False
            else:
                assert node.kind == "std.dim"
                assert nargs > 1
                dim = inputs[1]
                unbiased = inputs[2] if nargs > 2 else True
                keepdim = inputs[3] if nargs > 3 else False

        return x, dim, unbiased, keepdim

    def _parse_keyword_args(context, node, dim, unbiased, keepdim) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dim, unbiased, keepdim

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        unbiased = _get_kwinputs(context, node, "unbiased", default=[unbiased])[0]
        keepdim = _get_kwinputs(context, node, "keepdim", default=[keepdim])[0]

        return dim, unbiased, keepdim

    def _translate_torch_args(dim, unbiased, keepdim) -> Tuple[Var]:
        if isinstance(dim, Var):
            dim = dim.val
        try:
            dim = (int(dim),)
        except:
            pass

        if isinstance(unbiased, Var):
            unbiased = unbiased.val

        if isinstance(keepdim, Var):
            keepdim = keepdim.val

        return dim, unbiased, keepdim

    x, dim, unbiased, keepdim = _parse_positional_args(context, node)
    dim, unbiased, keepdim = _parse_keyword_args(context, node, dim, unbiased, keepdim)
    axes, unbiased, keep_dims = _translate_torch_args(dim, unbiased, keepdim)

    variance = _var(x, axes=axes, keep_dims=keep_dims, unbiased=unbiased)
    standard_deviation = mb.sqrt(x=variance)
    context.add(standard_deviation, node.name)


@register_torch_op(torch_alias=["std.correction"])
def std_correction(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=(1, 2, 3, 4))
        nargs = len(inputs)

        x = inputs[0]
        dim = inputs[1] if nargs > 1 else None
        correction = inputs[2] if nargs > 2 else None
        keepdim = inputs[3] if nargs > 3 else False

        return x, dim, correction, keepdim

    def _parse_keyword_args(context, node, dim, correction, keepdim) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dim, correction, keepdim

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        correction = _get_kwinputs(context, node, "correction", default=[correction])[0]
        keepdim = _get_kwinputs(context, node, "keepdim", default=[keepdim])[0]

        return dim, correction, keepdim

    def _translate_torch_args(dim, correction, keepdim) -> Tuple[Var]:
        if isinstance(dim, Var):
            dim = dim.val
        try:
            dim = (int(dim),)
        except:
            pass

        if isinstance(correction, Var):
            correction = correction.val

        if isinstance(keepdim, Var):
            keepdim = keepdim.val

        return dim, correction, keepdim

    x, dim, correction, keepdim = _parse_positional_args(context, node)
    dim, correction, keepdim = _parse_keyword_args(context, node, dim, correction, keepdim)
    axes, correction, keep_dims = _translate_torch_args(dim, correction, keepdim)

    variance = _var(x, axes=axes, keep_dims=keep_dims, correction=correction)
    standard_deviation = mb.sqrt(x=variance)
    context.add(standard_deviation, node.name)


@register_torch_op
def copy(context, node):
    inputs = _get_inputs(context, node, expected=[2, 3])
    assert (
        context.frontend != TorchFrontend.TORCHSCRIPT
    ), (
        "In torch script frontend, by graph pass `generate_tensor_assignment_ops`, "
        "`torch.copy_` should have been replaced with `_internal_op_tensor_inplace_copy`"
    )
    if context.frontend in TORCH_EXPORT_BASED_FRONTENDS:
        src = inputs[1]
        if inputs[0].shape != src.shape:
            _, src = _broadcast_tensors(inputs[: 2])
        result = mb.identity(x=src, name=node.name)
    else:
        raise ValueError(f"Invalid PyTorch frontend {context.frontend}")
    context.add(result)


@register_torch_op
def dtype(context, node):
    inputs = _get_inputs(context, node, expected=1)
    dtype_str = inputs[0].dtype.__name__
    context.add(mb.const(val=dtype_str, name=node.name))


@register_torch_op
def tensor(context, node):
    def _make_tensor(list_of_tensor, name, rank):
        if rank == 6:
            raise NotImplementedError("Core ML only supports tensor rank <= 5.")
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

    # Case 1: Using torch.tensor to create a const tensor
    # For example:
    # torch.tensor([[[0, 0], [0, 10], [5, 10], [5, 0]]], dtype=torch.float32)
    val = inputs[0]
    if isinstance(val, list):
        context.add(_make_tensor(val, node.name, 1))
        return

    if inputs[2] is None:
        context.add(mb.identity(x=val, name=node.name))
        return

    # Case 2: Create a tensor filled with a single value
    val = val.val  # element val to fill
    msg_prefix = 'torch::tensor {} '.format(node.name)
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
Pack and unpack op in pytorch.
The typical pattern is as following

>>> seq = torch.tensor([[1,2,0], [3,0,0], [4,5,6]])
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

source from https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html
"""


@register_torch_op
def _pack_padded_sequence(context, node):
    # The implementation of this op is not efficient. Raise a warning.
    logger.warning(
        "Encountered a _pack_padded_sequence layer. The implementation of translating pack/unpack op\
        in pytorch is not efficient due to the current limitation of Core ML. Removing the pack-unpack logic \
        and use a fixed batch size model is recommended."
    )

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


@register_torch_op
def _pad_packed_sequence(context, node):
    # The implementation of this op is not efficient. Raise a warning.
    logger.warning(
        "Encountered a _pad_packed_sequence layer. The implementation of translating pack/unpack op\
        in pytorch is not efficient due to the current limitation of Core ML. Removing the pack-unpack logic \
        and use a fixed batch size model is recommended."
    )
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
    # Note that in pytorch, the seq_lengths must be decreasing in order, len_1 >= len_2 >= len_3.
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
        # the output would have shape [padded_seq_len, input_dim]
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


@register_torch_op
def log10(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    log_x = mb.log(x=x)
    context.add(mb.mul(x=log_x, y=1 / _np.log(10.0)), node.name)


@register_torch_op
def log2(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    log_x = mb.log(x=x)
    context.add(mb.mul(x=log_x, y=1 / _np.log(2.0)), node.name)


@register_torch_op
def flip(context, node):
    inputs = _get_inputs(context, node, expected=2)
    x = mb.reverse(x=inputs[0], axes=inputs[1], name=node.name)
    context.add(x, node.name)


@register_torch_op(torch_alias=["reflection_pad1d"])
def reflection_pad2d(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    torch_pad = inputs[1].val
    pad_flipped = torch_pad.reshape((-1, 2))[::-1].ravel()
    pad = _np.pad(pad_flipped, (len(x.shape) * 2 - len(pad_flipped), 0))
    context.add(mb.pad(x=x, pad=pad, mode='reflect'), node.name)


@register_torch_op(torch_alias=["replication_pad1d"])
def replication_pad2d(context, node):
    inputs = _get_inputs(context, node)
    x = inputs[0]
    torch_pad = inputs[1].val
    pad_flipped = torch_pad.reshape((-1, 2))[::-1].ravel()
    pad = _np.pad(pad_flipped, (len(x.shape) * 2 - len(pad_flipped), 0))
    context.add(mb.pad(x=x, pad=pad, mode='replicate'), node.name)

def _solve_broadcast_shape(shapes: List[List[int]]) -> List[np.ndarray]:
    rank = _np.max([len(shape) for shape in shapes])
    shapes = [[1] * (rank - len(shape)) + shape for shape in shapes]
    result_shape = []
    for i in range(rank):
        dims = [shapes[j][i] for j in range(len(shapes))]
        if any_symbolic(dims):
            # rdar://85559497 (Handle dynamic shapes inputs broadcast for pytorch)
            symbols = set()
            integers = set()
            for dim in dims:
                if is_symbolic(dim):
                    symbols.add(dim)
                else:
                    integers.add(dim)
            # Integers can be safely ignored
            if integers == {1} or integers == set():
                result_dim = list(symbols)[0]
                result_shape.append(result_dim)
                # In principle, there must be only 1 symbol
                # In practise, since our symbol propagation is imperfect,
                # we may see multiple symbols, even if they must equal to each other / 1
                if len(symbols) != 1:
                    logger.warning(f"Recklessly broadcast {symbols} to {result_dim}")
            # In principle, in such case the symbols must be 1 or equal to the integer
            # In practise, since our symbol propagation is imperfect,
            # we may still see symbols, even if they must equal to max integer / 1
            else:
                result_dim = _np.max(list(integers))
                result_shape.append(result_dim)
                logger.warning(f"Recklessly broadcast {symbols} and {integers} to {result_dim}")
        else:
            result_shape.append(_np.max(dims))
    return result_shape

def _broadcast_tensors(tensors):
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


@register_torch_op
def broadcast_tensors(context, node):
    inputs = _get_inputs(context, node)
    context.add(_broadcast_tensors(inputs[0]), node.name)


def _scatter(context, inputs, mode, name):
    data = inputs[0]
    axis = inputs[1].val
    indices = inputs[2]
    updates = inputs[3]
    if types.is_scalar(updates.sym_type):
        updates = mb.fill(shape=indices.shape, value=updates.val)
    result = mb.scatter_along_axis(
        data=data, indices=indices, updates=updates, axis=axis, mode=mode, name=name
    )
    context.add(result)


@register_torch_op(
    torch_alias=["scatter.src", "scatter.value", "scatter.reduce", "scatter.value_reduce"]
)
def scatter(context, node):
    inputs = _get_inputs(context, node, expected=(4, 5))

    reduce = inputs[4].val if len(inputs) > 4 else "update"
    if context.frontend in TORCH_EXPORT_BASED_FRONTENDS:
        # torch.export may have `mode` in kwarg `reduce`
        reduce = _get_kwinputs(context, node, "reduce", default=[reduce])[0]
    if isinstance(reduce, Var):
        reduce = reduce.val

    mode = reduce
    if mode == "multiply":
        mode = "mul"
    assert mode in ("update", "add", "mul")

    _scatter(context, inputs, mode, node.name)


@register_torch_op
def scatter_add(context, node):
    inputs = _get_inputs(context, node)
    _scatter(context, inputs, 'add', node.name)


@register_torch_op
def glu(context, node):
    """
    glu(Tensor input, Scalar dim=-1)
    Applies the gated linear unit function GLU(a,b)=a⊗σ(b) where a is the first half of the input matrices and b is the
    second half.
    """

    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=(1, 2))
        nargs = len(inputs)

        x = inputs[0]
        dim = inputs[1] if nargs > 1 else -1
        return x, dim

    def _parse_keyword_args(context, node, dim) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return dim

        dim = _get_kwinputs(context, node, "dim", default=[dim])[0]
        return dim

    x, dim = _parse_positional_args(context, node)
    dim = _parse_keyword_args(context, node, dim)
    if isinstance(dim, Var):
        dim = dim.val

    first_half, second_half = mb.split(x=x, num_splits=2, axis=dim, name=node.name + "_split")
    sigmoid_second_half = mb.sigmoid(x=second_half, name=second_half.name + "_sigmoid")
    glu_node = mb.mul(x=first_half, y=sigmoid_second_half, name=node.name)
    context.add(glu_node)


@register_torch_op
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


@register_torch_op
def remainder(context, node):
    """
    remainder(Tensor dividend, Tensor divisor, Optional[Tensor] out)
    Computes Python’s modulus operation entrywise. The result has the same sign as the divisor and its absolute value
    is less than that of divisor. It may also be defined in terms of torch.div() as:
    remainder(a, b) == a - a.div(b, rounding_mode="floor") * b
    """
    # Don't specify `expected` because the parameter `out` is optional.
    inputs = _get_inputs(context, node)
    dividend, divisor = promote_input_dtypes([inputs[0], inputs[1]])
    div_node = mb.floor_div(x=dividend, y=divisor, name=node.name + "_div")
    context.add(div_node)
    scaled_div = mb.mul(x=div_node, y=divisor, name=div_node.name + "_scaled")
    context.add(scaled_div)
    remainder_node = mb.sub(x=dividend, y=scaled_div, name=node.name)
    context.add(remainder_node)


@register_torch_op
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


@register_torch_op
def mse_loss(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    y = inputs[1]
    reduction = inputs[2].val

    diff = mb.sub(x=x, y=y)

    if reduction == 0:
        # reduction is "none"
        res = mb.mul(x=diff, y=diff, name=node.name)
        context.add(res)
        return

    square = mb.mul(x=diff, y=diff)
    if reduction == 1:
        # reduction is "mean"
        res = mb.reduce_mean(x=square, axes=None, name=node.name)

    elif reduction == 2:
        # reduction is "sum"
        res = mb.reduce_sum(x=square, axes=None, name=node.name)
    else:
        raise ValueError("Reduction is not supported")

    context.add(res)


@register_torch_op(torch_alias=["diagonal_copy"])
def diagonal(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=1)
        nargs = len(inputs)

        x = inputs[0]
        offset = inputs[1] if nargs > 1 else 0
        dim1 = inputs[2] if nargs > 2 else 0
        dim2 = inputs[3] if nargs > 3 else 1

        return x, offset, dim1, dim2

    def _parse_keyword_args(context, node, offset, dim1, dim2) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return offset, dim1, dim2

        offset = _get_kwinputs(context, node, "offset", default=[offset])[0]
        dim1 = _get_kwinputs(context, node, "dim1", default=[dim1])[0]
        dim2 = _get_kwinputs(context, node, "dim2", default=[dim2])[0]

        return offset, dim1, dim2

    x, offset, dim1, dim2 = _parse_positional_args(context, node)
    offset, dim1, dim2 = _parse_keyword_args(context, node, offset, dim1, dim2)

    if offset == 0 and dim1 == 0 and dim2 == 1:
        diagonal = mb.band_part(x=x, lower=0, upper=0, name=node.name)
    else:
        raise NotImplementedError("Only offset == 0 and dim1 == 0 and dim2 == 1 handled")

    context.add(diagonal)


@register_torch_op
def trace(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    diagonal = mb.band_part(x=x, lower=0, upper=0)
    trace = mb.reduce_sum(x=diagonal, name=node.name)
    context.add(trace)


@register_torch_op
def roll(context, node):
    inputs = _get_inputs(context, node, expected=3)
    x = inputs[0]
    shift = inputs[1].val
    dims = inputs[2].val
    origin_shape = mb.shape(x=x)

    need_flatten = len(dims) == 0

    if need_flatten:
        # The tensor is flattened before rolling
        x = mb.reshape(x=x, shape=[-1])
        dims = [0]

    shape = mb.shape(x=x)

    for s, i in zip(shift, dims):
        dim = value_at(shape, i)
        s = mb.mod(x=s, y=dim)
        start_idx = mb.sub(x=dim, y=s)
        indices0 = mb.range_1d(end=dim, start=start_idx, step=1)
        indices1 = mb.range_1d(end=start_idx, start=0, step=1)
        indices = mb.concat(values=[indices0, indices1], axis=0)
        x = mb.gather(x=x, indices=indices, axis=i)

    if need_flatten:
        x = mb.reshape(x=x, shape=origin_shape)

    context.add(x, node.name)


def _construct_unfold_indices(N, C, H, W, kernel_size, stride):
    """
    A utility function to construct indices for torch.unfold (im2col),
    assuming the torch.unfold input `x` to be contiguous
    """

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
    offset_idx = _np.arange(0, row_extent, stride[0])[None, :, None] * W + _np.arange(0, col_extent, stride[1])
    indices = _np.ravel(start_idx)[:, None] + _np.ravel(offset_idx)

    # Get batch block indices.
    batch_idx = _np.arange(N)[:, None, None] * C * H * W
    indices = batch_idx + indices

    return indices.reshape(-1)


@register_torch_op
def im2col(context, node):
    """
    Extract sliding local blocks from a batched input tensor (rank=4).

    torch.nn.functional.unfold aims to be the general version: im2col is the rank=4 case of unfold.
    PyTorch currently only supports rank=4 input: torch.nn.functional.unfold redispatches to at::im2col,
    which is why coremltools needs im2col to convert torch.nn.functional.unfold.

    We currently only support rank=4 input (consistent with PyTorch) and dilation set to 1.
    More flexbible dilation support will be added in the future.

    Reference https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    """
    inputs = _get_inputs(context, node, expected=5)
    x = inputs[0]
    kernel_size = inputs[1].val
    dilation = inputs[2].val
    padding = inputs[3].val
    stride = inputs[4].val

    if x.rank != 4:
        raise ValueError("Only supports rank=4 input data for im2col (unfold).")
    if not (dilation[0] == 1 and dilation[1] == 1):
        raise ValueError("Only supports dilation=1 for im2col (unfold).")

    # for simplicity, we explicitly pad; TODO: implicit padding would be more efficient
    # torch.unfold padding has different semantics
    # * for torch.unfold
    #   x.shape[i + x.rank - padding.rank] = padding[i] + x.shape[i + x.rank - padding.rank] + padding[i]
    #   taking x.rank = 4 and padding.rank = 2 as an example:
    #       x.shape[0 + 4 - 2] = padding[0] + x.shape[0 + 4 - 2] + padding[0]
    #       x.shape[1 + 4 - 2] = padding[1] + x.shape[1 + 4 - 2] + padding[1]
    # * for mb.pad(x=x, pad=pad, mode="constant")
    #   x.shape[i] = pad[2 * i] + x.shape[i] + pad[2 * i + 1]
    # * for torch.nn.functional.pad
    #   x.shape[-1] = padding[0] +x.shape[-1] + padding[1]
    #   x.shape[-2] = padding[2] +x.shape[-1] + padding[3]
    #   ...
    #   x.shape[-i] = padding[2 * i - 2] + x.shape[-i] + padding[2 * i - 1]
    # so we need to convert torch.unfold padding to mb.pad(mode="constant") pad
    missing_dims = x.rank - len(padding)
    pad = [0, 0] * missing_dims + _np.array(padding).repeat(2).tolist()
    x = mb.pad(x=x, pad=pad, mode="constant")

    N, C, H, W = x.shape

    # Get total number of blocks. It follows the formula at torch.nn.Unfold documentation.
    sptial_size = (H, W)
    block_count = 1
    for i in range(2):
        block_count *= _np.floor(
            # the original formula is
            #     (sptial_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]
            # since we have explicitly padded, we no longer add 2 * padding[i] to sptial_size[i]
            (sptial_size[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]
            + 1
        ).astype(_np.int32)

    """
    The implementation below assumes x to be contiguous
    """

    indices = _construct_unfold_indices(N, C, H, W, kernel_size, stride)

    x = mb.reshape(x=x, shape=[-1])
    gathered_data = mb.gather_along_axis(x=x, indices=indices, axis=0)
    block_size = C * kernel_size[0] * kernel_size[1]
    output = mb.reshape(
        x=gathered_data, shape=(N, block_size, block_count), name=node.name
    )

    context.add(output)


@register_torch_op
def col2im(context, node):
    """
    Combines an array of sliding local blocks into a large containing tensor.

    torch.nn.functional.fold aims to be the general version:
    col2im is the "2 output spatial dimensions" case of fold.

    PyTorch currently only supports col2im: torch.nn.functional.fold redispatches to at::col2im,
    which is why coremltools needs col2im to convert torch.nn.functional.fold.

    We currently only support col2im (consistent with PyTorch) and:
    * dilation set to 1
    * padding set to 0
    * stride set to kernel_size
    * output_size is divisible by kernel_size

    More flexbible support will be added in the future.

    Reference https://pytorch.org/docs/stable/generated/torch.nn.Fold.html
    """

    inputs = _get_inputs(context, node, expected=6)
    x = inputs[0]
    output_size = inputs[1].val
    kernel_size = inputs[2].val
    dilation = inputs[3].val
    padding = inputs[4].val
    stride = inputs[5].val

    if len(output_size) != 2:
        raise ValueError("Only supports 2 output spatial dimensions for col2im (fold).")
    if not (dilation[0] == 1 and dilation[1] == 1):
        raise ValueError("Only supports dilation=1 for col2im (fold).")
    if not (padding[0] == 0 and padding[1] == 0):
        raise ValueError("Only supports padding=0 for col2im (fold).")
    # In Pytorch, if multiple entries unfold to same location, then in folding they are accumulated
    # In Core ML, however, there is no such op to perform this accumulation,
    # so we cowardly refuse to convert if accumulation happens
    # TODO: we may be able to support accumulation if x has certain symmetry (e.g. output by im2col)
    #       by multiplying the repeat times of each entry
    if any(stride != kernel_size):
        raise ValueError("Only supports stride = kernel_size for col2im (fold).")
    # We implement fold as an inverse to unfold
    # i.e. a gather with indices that are inverse to unfold gather indices
    # This works only if there is no edge leftover
    if any(output_size % kernel_size != 0):
        raise ValueError("Only supports output_size % kernel_size = 0 for col2im (fold).")

    N, block_size, block_count = x.shape
    C = int(block_size / _np.prod(kernel_size))
    H, W = output_size

    """
    The implementation below assumes x to be contiguous
    """

    # inverse unfold indices
    indices_unfold = _construct_unfold_indices(N, C, H, W, kernel_size, stride)
    indices = _np.empty(indices_unfold.shape, dtype=np.int32)
    for i in range(indices.shape[0]):
        indices[indices_unfold[i]] = i

    # perform gather with fold indices
    x_flatten = mb.reshape(x=x, shape=(-1,))
    y_flatten_with_extra = mb.gather_along_axis(x=x_flatten, indices=indices)
    y_flatten = mb.slice_by_index(x=y_flatten_with_extra, begin=(0,), end=(N * C * H * W,))
    y = mb.reshape(x=y_flatten, shape=(N, C, H, W), name=node.name)

    context.add(y)


@register_torch_op
def complex(context, node):
    real_part, imag_part = _get_inputs(context, node, expected=2)
    result = mb.complex(real_data=real_part, imag_data=imag_part)
    context.add(result, node.name)


@register_torch_op
def real(context, node):
    input_data = _get_inputs(context, node, expected=1)[0]
    if types.is_complex(input_data.dtype):
        real_part = mb.complex_real(data=input_data)
        context.add(real_part, node.name)
    else:
        context.add(input_data, node.name)


@register_torch_op
def imag(context, node):
    input_data = _get_inputs(context, node, expected=1)[0]
    if not types.is_complex(input_data.dtype):
        # Keep consistent with PyTorch.
        raise ValueError("The `imag` op only supports complex input.")
    real_part = mb.complex_imag(data=input_data)
    context.add(real_part, node.name)


@register_torch_op
def view_as_real(context, node):
    input_data = _get_inputs(context, node, expected=1)[0]
    if not types.is_complex(input_data.dtype):
        raise ValueError(f"view_as_real only supports complex input, but got {types.builtin_to_string(input_data.dtype)}")

    real_part = mb.complex_real(data=input_data)
    imag_part = mb.complex_imag(data=input_data)
    result = mb.stack(values=[real_part, imag_part], axis=-1)
    context.add(result, node.name)


@register_torch_op
def fft_fft(context, node):
    """Lowers torch.fft.fft by the dialect op `complex_fft` from complex_dialect_ops.py."""
    input_data, n, dim, norm = _get_inputs(context, node, expected=[4])
    fft_res = mb.complex_fft(data=input_data, n=n, dim=dim, norm=norm)
    context.add(fft_res, node.name)


@register_torch_op
def fft_fftn(context, node):
    """Lowers torch.fft.fftn by the dialect op `complex_fftn` from complex_dialect_ops.py."""
    input_data, shapes, dims, norm = _get_inputs(context, node, expected=[4])
    fft_res = mb.complex_fftn(data=input_data, shapes=shapes, dims=dims, norm=norm)
    context.add(fft_res, node.name)


@register_torch_op
def fft_rfft(context, node):
    """Lowers torch.fft.rfft by the dialect op `complex_rfft` from complex_dialect_ops.py."""
    input_data, n, dim, norm = _get_inputs(context, node, expected=[4])
    rfft_res = mb.complex_rfft(data=input_data, n=n, dim=dim, norm=norm)
    context.add(rfft_res, node.name)


@register_torch_op
def fft_rfftn(context, node):
    """Lowers torch.fft.rfftn by the dialect op `complex_rfftn` from complex_dialect_ops.py."""
    input_data, shapes, dims, norm = _get_inputs(context, node, expected=[4])
    rfft_res = mb.complex_rfftn(data=input_data, shapes=shapes, dims=dims, norm=norm)
    context.add(rfft_res, node.name)


@register_torch_op
def fft_ifft(context, node):
    """Lowers torch.fft.ifft by the dialect op `complex_ifft` from complex_dialect_ops.py."""
    input_data, n, dim, norm = _get_inputs(context, node, expected=[4])
    ifft_res = mb.complex_ifft(data=input_data, n=n, dim=dim, norm=norm)
    context.add(ifft_res, node.name)


@register_torch_op
def fft_ifftn(context, node):
    """Lowers torch.fft.ifftn by the dialect op `complex_ifftn` from complex_dialect_ops.py."""
    input_data, shapes, dims, norm = _get_inputs(context, node, expected=[4])
    ifftn_res = mb.complex_ifftn(data=input_data, shapes=shapes, dims=dims, norm=norm)
    context.add(ifftn_res, node.name)


@register_torch_op
def fft_irfft(context, node):
    """Lowers torch.fft.irfft by the dialect op `complex_irfft` from complex_dialect_ops.py."""
    input_data, n, dim, norm = _get_inputs(context, node, expected=[4])
    irfft_res = mb.complex_irfft(data=input_data, n=n, dim=dim, norm=norm)
    context.add(irfft_res, node.name)


@register_torch_op
def fft_irfftn(context, node):
    """Lowers torch.fft.irfftn by the dialect op `complex_irfftn` from complex_dialect_ops.py."""
    input_data, shapes, dims, norm = _get_inputs(context, node, expected=[4])
    irfftn_res = mb.complex_irfftn(data=input_data, shapes=shapes, dims=dims, norm=norm)
    context.add(irfftn_res, node.name)

@register_torch_op
def stft(context, node):
    """
    Lowers torch.stft with the dialect op `complex_stft` from complex_dialect_ops.py
    """
    input_data, n_fft, hop_length, win_length, window, normalized, onesided, _ = _get_inputs(context, node, min_expected=2)
    if types.is_complex(input_data.dtype):
        onesided = False # pytorch defaults onesided to False for complex inputs
    stft_res = mb.complex_stft(
        input=input_data,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=normalized,
        onesided=onesided)
    context.add(stft_res, node.name)

@register_torch_op(torch_alias=["torchvision::nms"])
def torchvision_nms(context, node):
    inputs = _get_inputs(context, node, expected=3)
    boxes, scores = promote_input_dtypes([inputs[0], inputs[1]])
    iou_threshold = inputs[2].val
    # Use float min to avoid boxes being pruned by scores in MIL NMS op.
    score_threshold = (
        _np.finfo(_np.float16).min if boxes.dtype._width == 16 else _np.finfo(_np.float32).min
    )

    box_num = boxes.shape[0]
    if is_symbolic(box_num):
        # When the number of boxes is unknown at compile time, use a large number to avoid valid
        # boxes got pruned. We don't use _np.iinfo(_np.int32).max here because it triggers the MIL
        # NMS op segment fault.
        box_num = 10000

    # The boxes' coordinates from PyTorch input is (x1, y1, x2, y2) format with 0 <= x1 < x2 and
    # 0 <= y1 < y2. However, the MIL NMS op expects CENTER_SIZE_WIDTH_FIRST format, which is
    # (x, y, width, height) where (x, y) is the center coordinate.
    x1, y1, x2, y2 = mb.split(x=boxes, num_splits=4, axis=-1)
    # For numerical stability, use x1+(x2-x1)/2 instead of (x1+x2)/2 to calculate center coordinate.
    width = mb.sub(x=x2, y=x1)
    height = mb.sub(x=y2, y=y1)
    center_x = mb.add(x=x1, y=mb.real_div(x=width, y=2.0))
    center_y = mb.add(x=y1, y=mb.real_div(x=height, y=2.0))
    boxes = mb.concat(values=[center_x, center_y, width, height], axis=-1)

    # Expand dims to construct the batch dim and score class dim expected by MIL NMS op.
    boxes = mb.expand_dims(x=boxes, axes=[0])
    scores = mb.expand_dims(x=scores, axes=[0, -1])

    if not is_current_opset_version_compatible_with(target.iOS17):
        _, _, indices, valid_outputs = mb.non_maximum_suppression(
            boxes=boxes,
            scores=scores,
            max_boxes=box_num,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )

        indices = mb.squeeze(x=indices, axes=[0])
        valid_outputs = mb.squeeze(x=valid_outputs, axes=[0])
        range = mb.range_1d(end=valid_outputs, start=0, step=1)
        indices = mb.cast(x=indices, dtype="fp32")
        valid_indices = mb.gather(x=indices, indices=range, axis=0)
        valid_indices = mb.cast(x=valid_indices, dtype="int32", name=node.name)
        context.add(valid_indices)
    else:
        # In IOS17, the MIL NMS op's inputs are ordered with number of boxes in the last dimension.
        boxes = mb.transpose(x=boxes, perm=[0, 2, 1])
        scores = mb.transpose(x=scores, perm=[0, 2, 1])

        # In IOS17, the MIL NMS op's last output (number of valid boxes in each batch) gets removed.
        _, _, indices = mb.non_maximum_suppression(
            boxes=boxes,
            scores=scores,
            max_boxes=box_num,
            iou_threshold=iou_threshold,
        )

        # Remove invalid indices (the padded -1 indices).
        valid_outputs = mb.reduce_sum(
            x=mb.cast(x=mb.greater(x=indices, y=-1), dtype="int32"), axes=[-1]
        )
        valid_indices = mb.slice_by_size(
            x=mb.squeeze(x=indices, axes=[0]),
            begin=mb.fill_like(ref_tensor=valid_outputs, value=0),
            size=valid_outputs,
            name=node.name,
        )
        context.add(valid_indices)


@register_torch_op
def tupleindex(context, node):
    tuple_input, index_input = _get_inputs(context, node, expected=2)
    context.add(tuple_input[index_input.val], node.name)


@register_torch_op(
    torch_alias=[
        "_scaled_dot_product_flash_attention_for_cpu",
        "coreml.sdpa",
        "coreml::sdpa",
    ]
)
def scaled_dot_product_attention(context, node):
    """
    Input shapes/types:
    - query : (target_seq, d) or (B, target_seq, d) or (B, h, target_seq, d) or (B,.., target_seq, d)
    - key : (source_seq, d) or (B, source_seq, d) or (B, h, source_seq, d) or (B,.., source_seq, d)
    - value: (source_seq, d_v) or (B, source_seq, d_v) or (B, h, source_seq, d_v) or (B,.., source_seq, d_v)
    - attn_mask : (target_seq, source_seq) or (B, target_seq, source_seq) or (B, h, target_seq, source_seq) or
                  (B, ..., target_seq, source_seq)
    - is_causal : bool
    - scale : optional float

    Output shape: (target_seq, d_v) or (B,...,target_seq, d_v)

    output = softmax(scale*Q*K^transpose + mask) * V

    Currently, Core ML does not support dropout, so it has to be either None or 0

    See details at:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """

    def _get_batch_dims(x: Var) -> List[int]:
        return list(x.shape)[:-2]

    def _broadcast_tensor_to_same_batch_dims(x: Var, batch_dims: List[int]) -> Var:
        broadcast_shape = batch_dims + list(x.shape[-2:])
        return _broadcast(x.name + "_broadcast_same_batch_dims", x, broadcast_shape)

    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, min_expected=3)
        nargs = len(inputs)

        q, k, v = inputs[:3]

        if node.kind == "scaled_dot_product_attention":
            attn_mask = inputs[3] if nargs > 3 else None
            dropout = inputs[4] if nargs > 4 else 0.0
            is_causal = inputs[5].val if nargs > 5 else False
            scale = inputs[6] if nargs > 6 else None
        elif node.kind == "_scaled_dot_product_flash_attention_for_cpu":
            dropout = inputs[3] if nargs > 3 else 0.0
            is_causal = inputs[4].val if nargs > 4 else False
            attn_mask = inputs[5] if nargs > 5 else None
            scale = inputs[6] if nargs > 6 else None
        else:
            assert node.kind in ("coreml.sdpa", "coreml::sdpa")
            attn_mask = inputs[3] if nargs > 3 else None
            dropout = 0.0
            is_causal = False
            scale = None

        return q, k, v, attn_mask, dropout, is_causal, scale

    def _parse_keyword_args(context, node, attn_mask, scale) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return attn_mask, scale

        attn_mask = _get_kwinputs(context, node, "attn_mask", default=[attn_mask])[0]
        scale = _get_kwinputs(context, node, "scale", default=[scale])[0]

        return attn_mask, scale

    def _check_args(q, k, v, attn_mask, dropout, is_causal) -> None:
        if attn_mask is not None and is_causal:
            raise ValueError(
                "scaled_dot_product_attention op: "
                "attn_mask cannot be provided when is_causal is set to True."
            )

        if dropout is not None:
            if isinstance(dropout, Var):
                if dropout.val is None:
                    raise NotImplementedError(
                        "A variable dropout probability is specified. Since Core ML "
                        "does not support dropout yet, we cowardly refuse to convert it"
                    )
                else:
                    dropout = dropout.val
            if dropout != 0.0:
                raise ValueError(
                    "A non-zero dropout probability is specified. Since Core ML "
                    "does not support dropout yet, we cannot convert it"
                )

        # check that ranks of q, k, v and attn_mask match
        if k.rank != q.rank:
            raise ValueError(
                "Rank of query and key do not match in scaled_dot_product_attention torch op"
            )
        if v.rank != q.rank:
            raise ValueError(
                "Rank of query and value do not match in scaled_dot_product_attention torch op"
            )

    def _construct_causal_attn_mask(query_var: Var, key_var: Var) -> Var:
        # create mask of shape (target_seq, source_seq)
        # s.t the lower triangular of the matrix are all Trues
        # and the strict upper triangular are all Falses
        target_seq = query_var.shape[-2]
        source_seq = key_var.shape[-2]
        if is_symbolic(target_seq) or is_symbolic(source_seq):
            target_seq = value_at(mb.shape(x=query_var), -2)
            source_seq = value_at(mb.shape(x=key_var), -2)
        mask_shape = mb.concat(values=(target_seq, source_seq), axis=0)
        all_trues = mb.fill(shape=mask_shape, value=True)
        mask = mb.band_part(x=all_trues, lower=-1, upper=0)
        return mask

    def _cast_bool_attn_mask(attn_mask: Var, query_var: Var) -> Var:
        """
        compute float mask as (1 - cast(bool_mask)) * -30k
        """
        assert is_bool(attn_mask.dtype)

        mask = mb.cast(x=attn_mask, dtype=types.builtin_to_string(query_var.dtype))
        compliment_of_mask = mb.sub(x=1.0, y=mask)
        # Use a big enough but not easily fp16-overflow number (e.g. -3e4) as -inf
        float_mask = mb.mul(x=-3e4, y=compliment_of_mask)
        return float_mask

    def _translate_torch_args(q, k, attn_mask, is_causal, can_use_fused_sdpa) -> Var:
        mask = attn_mask
        if is_causal:
            mask = _construct_causal_attn_mask(q, k)
        if mask is not None and is_bool(mask.dtype) and not can_use_fused_sdpa:
            # In Core ML, only fused sdpa can use bool attention mask
            # i.e. the decomposition has to use floating point mask
            mask = _cast_bool_attn_mask(mask, q)
        return mask

    q, k, v, attn_mask, dropout, is_causal, scale = _parse_positional_args(context, node)
    attn_mask, scale = _parse_keyword_args(context, node, attn_mask, scale)
    _check_args(q, k, v, attn_mask, dropout, is_causal)

    # Since ios18, Core ML supports scaled_dot_product_attention op
    # It does not have scale, though
    can_use_fused_sdpa = is_current_opset_version_compatible_with(target.iOS18) and scale is None
    mask = _translate_torch_args(q, k, attn_mask, is_causal, can_use_fused_sdpa)

    if can_use_fused_sdpa:
        # ios18 scaled_dot_product_attention only supports rank >= 3
        is_rank_2 = q.rank == 2

        if is_rank_2:
            q = mb.expand_dims(x=q, axes=[0])
            k = mb.expand_dims(x=k, axes=[0])
            v = mb.expand_dims(x=v, axes=[0])

        # broadcast the batch_dims to the same shape
        # note that, we only support the broadcast if the batch_dim is static
        q_batch = _get_batch_dims(q)
        k_batch = _get_batch_dims(k)
        v_batch = _get_batch_dims(v)

        if not any_symbolic(q_batch + k_batch + v_batch):
            b_dims = _solve_broadcast_shape([q_batch, k_batch, v_batch])
            q = _broadcast_tensor_to_same_batch_dims(q, b_dims)
            k = _broadcast_tensor_to_same_batch_dims(k, b_dims)
            v = _broadcast_tensor_to_same_batch_dims(v, b_dims)

        # directly translated into iOS18 sdpa op
        res = mb.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=mask, name=node.name
        )

        if is_rank_2:
            res = mb.squeeze(x=res, axes=[0], name=node.name)

    else:
        res = _utils._decompose_scaled_dot_product_attention(q, k, v, mask, node.name, scale=scale)

    context.add(res)


@register_torch_op
def fliplr(context, node):
    """
    Flip tensor in the left/right direction.

    Flip the entries in each row in the left/right direction. Columns are preserved, but appear in a
    different order than before.
    It's equivalent to TF's reverse op but with axes always be [1].
    """
    x = _get_inputs(context, node, expected=1)[0]
    res = mb.reverse(x=x, axes=[1], name=node.name)
    context.add(res)


@register_torch_op
def multinomial(context, node):
    x = context[node.inputs[0]]
    num_samples = context[node.inputs[1]].val
    replacement = context[node.inputs[2]].val
    if num_samples is None:
        raise ValueError("In torch.multinomial op, num_samples must be const")
    if num_samples > 1 and not replacement:
        raise ValueError("When num_samples is larger than 1, only replacement=True is supported.")
    # Based on PyTorch documentations, the input to `torch.multinomial` is probability, not logit.
    x = mb.random_categorical(x=x, size=num_samples, mode="probs", name=node.name)
    context.add(x)


@register_torch_op
def linalg_inv(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    if x.val is None:
        raise NotImplementedError(
            "For non-const input, the lowering for Torch `linalg_inv` op is not supported"
        )
    context.add(mb.const(val=np.linalg.inv(x.val), name=node.name))


@register_torch_op
def isnan(context, node):
    x = _get_inputs(context, node, expected=1)[0]
    # Find indices of NaN based on "NaN is never equal to itself".
    nan_indices = mb.not_equal(x=x, y=x, name=node.name)
    context.add(nan_indices)


@register_torch_op
def nan_to_num(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: 4},
            min_expected={TorchFrontend.TORCHEXPORT: 1, TorchFrontend.EXECUTORCH: 1},
        )
        nargs = len(inputs)

        x = inputs[0]
        nan = inputs[1] if nargs > 1 else None
        posinf = inputs[2] if nargs > 2 else None
        neginf = inputs[3] if nargs > 3 else None

        return x, nan, posinf, neginf

    def _parse_keyword_args(context, node, nan, posinf, neginf) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return nan, posinf, neginf

        nan = _get_kwinputs(context, node, "nan", default=[nan])[0]
        posinf = _get_kwinputs(context, node, "posinf", default=[posinf])[0]
        neginf = _get_kwinputs(context, node, "neginf", default=[neginf])[0]

        return nan, posinf, neginf

    def _translate_torch_args(x, nan, posinf, neginf) -> Tuple[Var]:
        if nan is None:
            nan = 0.0
        else:
            if isinstance(nan, Var):
                nan = nan.val

        if posinf is None:
            posinf = types.type_mapping.builtin_to_range(x.dtype).high
        else:
            if isinstance(posinf, Var):
                posinf = posinf.val

        if neginf is None:
            neginf = types.type_mapping.builtin_to_range(x.dtype).low
        else:
            if isinstance(neginf, Var):
                neginf = neginf.val

        return nan, posinf, neginf

    def _replace_values_by_bool_mask(data: Var, mask: Var, new_value: Union[int, float]):
        """Replace the position in data where mask has True element to new_value."""
        indices = mb.non_zero(x=mb.cast(x=mask, dtype="int32"))

        # Expand the replacement value to the compatible shape for scatter_nd.
        replacement_values = mb.expand_dims(x=new_value, axes=[0])
        reps = mb.expand_dims(x=value_at(mb.shape(x=indices), 0), axes=[0])
        replacement_values = mb.tile(x=replacement_values, reps=reps)

        # Replace all nan to the corresponding values.
        return mb.scatter_nd(data=data, indices=indices, updates=replacement_values, mode="update")

    x, nan, posinf, neginf = _parse_positional_args(context, node)
    nan, posinf, neginf = _parse_keyword_args(context, node, nan, posinf, neginf)
    nan, posinf, neginf = _translate_torch_args(x, nan, posinf, neginf)

    if x.val is not None:
        res = mb.const(val=np.nan_to_num(x.val, nan=nan, posinf=posinf, neginf=neginf))
    else:
        # Find indices of NaN based on "NaN is never equal to itself".
        nan_indices_mask = mb.not_equal(x=x, y=x)
        res = _replace_values_by_bool_mask(x, nan_indices_mask, nan)

        # Find indices of Inf based on "Inf times zero becomes NaN".
        x_times_zero = mb.mul(x=x, y=0.0)
        inf_indices_mask = mb.not_equal(x=x_times_zero, y=x_times_zero)
        posinf_indices_mask = mb.logical_and(x=inf_indices_mask, y=mb.greater(x=x, y=0.0))
        neginf_indices_mask = mb.logical_and(x=inf_indices_mask, y=mb.less(x=x, y=0.0))
        res = _replace_values_by_bool_mask(res, posinf_indices_mask, posinf)
        res = _replace_values_by_bool_mask(res, neginf_indices_mask, neginf)

    context.add(res, node.name)


@register_torch_op
def cumprod(context, node):
    inputs = _get_inputs(context, node, min_expected=2)
    x = inputs[0]
    dim = inputs[1].val
    # dtype may be the 3rd input, but we will not use it

    size = x.shape[dim]
    if is_symbolic(size):
        raise NotImplementedError(
            "For symbolic shape input, the lowering for Torch `cumprod` op is not supported"
        )
    for idx in range(1, size):
        # For each element, multiply it with the previous element.
        prev_element = mb.gather(x=x, indices=[idx - 1], axis=dim)
        x = mb.scatter(data=x, indices=[idx], updates=prev_element, axis=dim, mode="mul")

    context.add(x, node.name)


@register_torch_op
def searchsorted(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(
            context,
            node,
            expected={TorchFrontend.TORCHSCRIPT: 6},
            min_expected={TorchFrontend.TORCHEXPORT: 2, TorchFrontend.EXECUTORCH: 2},
        )
        nargs = len(inputs)

        sorted_sequence = inputs[0]
        values = inputs[1]
        # we will not use `out_int32`
        right = inputs[3] if nargs > 3 else False
        side = inputs[4] if nargs > 4 else None
        # we will not use `sorter`

        return sorted_sequence, values, right, side

    def _parse_keyword_args(context, node, right, side) -> Tuple[Var]:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return right, side

        right = _get_kwinputs(context, node, "right", default=[right])[0]
        side = _get_kwinputs(context, node, "side", default=[side])[0]

        return right, side

    def _translate_torch_args(right, side) -> Tuple[Var]:
        if side is not None:
            if isinstance(side, Var):
                side = side.val
            # The `side` parameter is preferred than `right` in torch.
            right = side == "right"
        else:
            # If side is not specified, use the `right` parameter to determine.
            if right is None:
                right = False
            else:
                if isinstance(right, Var):
                    right = right.val
        return right

    sorted_sequence, values, right, side = _parse_positional_args(context, node)
    right, side = _parse_keyword_args(context, node, right, side)
    right = _translate_torch_args(right, side)

    if sorted_sequence.rank != values.rank:
        raise NotImplementedError(
            "Not support `searchsorted` op with different ranks of "
            f"`sorted_sequence` ({sorted_sequence.rank}) and `values` ({values.rank})."
        )
    if is_symbolic(values.shape[-1]):
        raise NotImplementedError(
            "Not support `searchsorted` op for `values` with symbolic last dim."
        )

    # Assume `sorted_sequence` has shape [..., M] and `values` has shape [..., N]
    # First tile the sorted_sequence to [..., N, M], and make values to [..., N, 1].
    # Then count the number of smaller elements in tiles `sorted_sequence` for each element in `values`.
    # It will get the result index with shape [..., N].
    tile_reps = [1] * (sorted_sequence.rank + 1)
    tile_reps[-2] = values.shape[-1]
    tiled_sorted_sequence = mb.tile(x=mb.expand_dims(x=sorted_sequence, axes=[-2]), reps=tile_reps)
    values = mb.expand_dims(x=values, axes=[-1])
    if right:
        count_smaller_num = mb.greater_equal(x=values, y=tiled_sorted_sequence)
    else:
        count_smaller_num = mb.greater(x=values, y=tiled_sorted_sequence)
    res = mb.reduce_sum(x=mb.cast(x=count_smaller_num, dtype="int32"), axes=[-1], keep_dims=False)
    context.add(res, node.name)


@register_torch_op
def one_hot(context, node):
    def _parse_positional_args(context, node) -> Tuple[Var]:
        inputs = _get_inputs(context, node, expected=(1, 2))
        nargs = len(inputs)

        labels = inputs[0]
        num_classes = inputs[1] if nargs > 1 else -1

        return labels, num_classes

    def _parse_keyword_args(context, node, num_classes) -> Var:
        # Only torch.export may have kwargs
        if context.frontend not in TORCH_EXPORT_BASED_FRONTENDS:
            return num_classes

        num_classes = _get_kwinputs(context, node, "num_classes", default=[num_classes])[0]

        return num_classes

    labels, num_classes = _parse_positional_args(context, node)
    num_classes = _parse_keyword_args(context, node, num_classes)
    if isinstance(num_classes, Var):
        num_classes = num_classes.val

    res = mb.one_hot(indices=labels, one_hot_vector_size=num_classes, name=node.name)
    context.add(res)
