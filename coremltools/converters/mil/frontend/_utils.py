#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import math as math
from typing import List, Optional, Union

import numpy as np

from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target
from coremltools.converters.mil.input_types import InputType
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Var, types
from coremltools.converters.mil.mil.block import is_current_opset_version_compatible_with
from coremltools.converters.mil.mil.ops.defs._utils import (
    parse_einsum_equation,
    promote_input_dtypes,
)
from coremltools.converters.mil.mil.types.symbolic import any_symbolic, is_symbolic


def value_at(x: Var, idx: int, name=None, before_op=None):
    """
    input x: 1D tensor (vector).
    return value at index idx: x[idx].
    Could specify the name of the returned MIL scalar tensor as well.
    """
    assert x.rank == 1
    args = {
        "x": x,
        "begin": [idx],
        "end": [0],
        "squeeze_mask": [True],
    }
    if name is not None:
        args["name"] = name
    if before_op is not None:
        args["before_op"] = before_op
    return mb.slice_by_index(**args)


def _construct_gather_op(
    op_type: str, x: Var, indices: Var, axis: Var = None, name: str = None
) -> Var:
    """
    This utility is a more general gather in the sense that:
    1. Both mb.gather and mb.gather_nd are handled
    2. x is allowed to be bool, while mb.gather and mb.gather_nd only allow float or int
    """
    assert (
        op_type in {"gather", "gather_nd"}
    ), f"This utility only handles gather or gather_nd, but got {op_type}"
    if op_type == "gather_nd":
        assert axis is None, "mb.gather_nd should not have input axis"

    # if is gathering bool:
    #     cast bool input to a smallest supported dtype to gather, then cast back gather result
    #     the back cast carries the specified name
    # else:
    #     usual gather, and gather carries the specified name
    is_gathering_bool = x.dtype == types.bool
    if is_gathering_bool:
        gather_name_kwarg = {}
        cast_name_kwarg = {} if name is None else {"name": name}
    else:
        gather_name_kwarg = {} if name is None else {"name": name}

    if is_gathering_bool:
        work_dtype = "int8" if is_current_opset_version_compatible_with(target.iOS17) else "fp16"
        x = mb.cast(x=x, dtype=work_dtype)

    if op_type == "gather":
        if types.is_float(indices.dtype):
            indices = mb.cast(x=indices, dtype="int32")
        result = mb.gather(x=x, indices=indices, axis=axis, **gather_name_kwarg)
    else:
        result = mb.gather_nd(x=x, indices=indices, **gather_name_kwarg)

    if is_gathering_bool:
        result = mb.cast(x=result, dtype="bool", **cast_name_kwarg)

    return result


def _reverse_input_einsum_eq(equation: str) -> str:
    """
    Reverse the input order of the einsum equation
    e.g.:
    input : "nchw,nwhu->nchu"
    returns : "nwhu,nchw->nchu"
    """
    input_output_strings = equation.split('->')
    assert len(input_output_strings) == 2, "invalid equation"
    input_strings = input_output_strings[0].split(',')
    assert len(input_strings) == 2, "invalid equation"
    equation = input_strings[1] + ',' + input_strings[0] + '->' + input_output_strings[1]
    return equation


def build_einsum_mil(vars: List[Var], equation: str, name: str) -> Var:
    """
    Get MIL variables as input and build a variable using MIL builder, that
    contains the output of the einsum equation

    :param vars:
        - List[var]
        - list of input variables
    :param equation:
        - str
        - the einsum equation
    :param name:
        - str
        - name tp be assigned to the output var

    :return:
        - var
        - output var that contains the einsum result
    """

    ## TODO: rdar://73851694 (Update einsum op translation to support generic cases)
    equation = equation.replace(" ", "")
    parsed_vectors = parse_einsum_equation(equation)

    if len(vars) != 2:
        return solve_generic_einsum(list(parsed_vectors), vars, name)

    equation_rev = _reverse_input_einsum_eq(equation)
    parsed_vectors_rev = parse_einsum_equation(equation_rev)

    def _swap(a, b):
        return b, a

    a_var, b_var = vars
    is_dynamic = any([any_symbolic(var.shape) for var in vars])
    # list of equations supported for explicit mil translations
    vec_bnqd_bnkd_bnqk = (
        [0, 1, 2, 3],
        [0, 1, 4, 3],
        [0, 1, 2, 4],
    )  # equation == "bnqd,bnkd->bnqk"
    vec_bhcq_bhck_bhqk = (
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
    )  # equation == "bhcq,bhck->bhqk"
    vec_abc_cd_abd = ([0, 1, 2], [2, 3], [0, 1, 3])  # equation == "abc,cd->abd"
    vec_abc_cde_abde = (
        [0, 1, 2],
        [2, 3, 4],
        [0, 1, 3, 4],
    )  # equation == "abc,cde->abde"
    vec_btnh_bfnh_bnft = (
        [0, 1, 2, 3],
        [0, 4, 2, 3],
        [0, 2, 4, 1],
    )  # equation == "btnh,bfnh->bnft"
    vec_bnft_btnh_bfnh = (
        [0, 1, 2, 3],
        [0, 3, 1, 4],
        [0, 2, 1, 4],
    )  # equation == "bnft,btnh->bfnh"
    vec_abcd_cde_abe = (
        [0, 1, 2, 3],
        [2, 3, 4],
        [0, 1, 4],
    )  # equation == "abcd,cde->abe"
    vec_nchw_nwhu_nchu = (
        [0, 1, 2, 3],
        [0, 3, 2, 4],
        [0, 1, 2, 4],
    )  # equation == "nchw,nwhu->nchu"
    vec_chw_whu_chu = ([0, 1, 2], [2, 1, 3], [0, 1, 3])  # equation == "chw,whu->chu"

    # add the op(s) corresponding to the equation
    if vec_bnqd_bnkd_bnqk in [parsed_vectors, parsed_vectors_rev]:
        if parsed_vectors_rev == vec_bnqd_bnkd_bnqk:
            a_var, b_var = _swap(a_var, b_var)
        x = mb.matmul(x=a_var, y=b_var, transpose_x=False, transpose_y=True, name=name)
    elif vec_bhcq_bhck_bhqk in [parsed_vectors, parsed_vectors_rev]:
        if parsed_vectors_rev == vec_bhcq_bhck_bhqk:
            a_var, b_var = _swap(a_var, b_var)
        x = mb.matmul(x=a_var, y=b_var, transpose_x=True, transpose_y=False, name=name)
    elif vec_abc_cd_abd in [parsed_vectors, parsed_vectors_rev]:
        if parsed_vectors_rev == vec_abc_cd_abd:
            a_var, b_var = _swap(a_var, b_var)
        x = mb.matmul(x=a_var, y=b_var, transpose_x=False, transpose_y=False, name=name)
    elif vec_abc_cde_abde in [parsed_vectors, parsed_vectors_rev] and not is_dynamic:
        if parsed_vectors_rev == vec_abc_cde_abde:
            a_var, b_var = _swap(a_var, b_var)
        x_1 = mb.reshape(x=a_var, shape=[a_var.shape[0] * a_var.shape[1], a_var.shape[2]])
        x_2 = mb.reshape(x=b_var, shape=[b_var.shape[0], b_var.shape[1] * b_var.shape[2]])
        x = mb.matmul(x=x_1, y=x_2, transpose_x=False, transpose_y=False)
        x = mb.reshape(
            x=x, shape=[a_var.shape[0], a_var.shape[1], b_var.shape[1], b_var.shape[2]], name=name
        )
    elif vec_btnh_bfnh_bnft in [parsed_vectors, parsed_vectors_rev]:
        if parsed_vectors_rev == vec_btnh_bfnh_bnft:
            a_var, b_var = _swap(a_var, b_var)
        x_1 = mb.transpose(x=a_var, perm=[0, 2, 1, 3])
        x_2 = mb.transpose(x=b_var, perm=[0, 2, 1, 3])
        x = mb.matmul(x=x_2, y=x_1, transpose_x=False, transpose_y=True, name=name)
    elif vec_bnft_btnh_bfnh in [parsed_vectors, parsed_vectors_rev]:
        if parsed_vectors_rev == vec_bnft_btnh_bfnh:
            a_var, b_var = _swap(a_var, b_var)
        b_var = mb.transpose(x=b_var, perm=[0, 2, 1, 3])
        x = mb.matmul(x=a_var, y=b_var, transpose_x=False, transpose_y=False)
        x = mb.transpose(x=x, perm=[0, 2, 1, 3], name=name)
    elif vec_abcd_cde_abe in [parsed_vectors, parsed_vectors_rev] and not is_dynamic:
        if parsed_vectors_rev == vec_abcd_cde_abe:
            a_var, b_var = _swap(a_var, b_var)
        x_1 = mb.reshape(x=a_var, shape=[a_var.shape[0], a_var.shape[1], a_var.shape[2] * a_var.shape[3]])
        x_2 = mb.reshape(x=b_var, shape=[b_var.shape[0] * b_var.shape[1], b_var.shape[2]])
        x = mb.matmul(x=x_1, y=x_2, transpose_x=False, transpose_y=False, name=name)
    elif vec_nchw_nwhu_nchu in [parsed_vectors, parsed_vectors_rev]:
        if parsed_vectors == vec_nchw_nwhu_nchu:
            x = mb.einsum(values=(a_var, b_var), equation=equation, name=name)
        else:
            x = mb.einsum(values=(b_var, a_var), equation=equation_rev, name=name)
    elif vec_chw_whu_chu in [parsed_vectors, parsed_vectors_rev]:
        if parsed_vectors == vec_chw_whu_chu:
            x = mb.einsum(values=(a_var, b_var), equation=equation, name=name)
        else:
            x = mb.einsum(values=(b_var, a_var), equation=equation_rev, name=name)
    else:
        x = solve_generic_einsum(list(parsed_vectors), [a_var, b_var], name)

    return x


def is_symbolic_dim_in_prog(prog):
    '''
    Takes in a MIL program object, checks if any of the tensors in it contain a symbolic dimension.
    Returns true if it does.

    :param prog: coremltools.converters.mil.Program
    :return: bool
    '''
    def _does_block_contain_symbolic_shape(block):
        for op in block.operations:
            for b in op.blocks:
                if _does_block_contain_symbolic_shape(b):
                    return True
            for out in op.outputs:
                if types.is_tensor(out.sym_type):
                    shape = out.sym_type.get_shape()
                    if any_symbolic(shape):
                        return True
                elif types.is_scalar(out.sym_type) or types.is_str(out.sym_type):
                    if is_symbolic(out.val):
                        return True
                elif types.is_list(out.sym_type):
                    if types.is_tensor(out.elem_type):
                        if any_symbolic(out.elem_type.get_shape()):
                            return True
                    else:
                        raise NotImplementedError("\'{}\' type in a list not handled".format(out.elem_type))
                else:
                    raise NotImplementedError("\'{}\' type is not handled".format(out.sym_type))
        return False

    for f in prog.functions.values():
        if _does_block_contain_symbolic_shape(f):
            return True
    return False


def get_output_names(outputs) -> Optional[List[str]]:
    """
    :param: list[ct.TensorType/ct.ImageType]
    :return: list[str] or None
    """
    output_names = None
    if outputs is not None:
        assert all([isinstance(t, InputType) for t in outputs]), \
            "outputs must be a list of ct.ImageType or ct.TensorType"
        output_names = [t.name for t in outputs]
        if all([name is None for name in output_names]):
            output_names = None
    return output_names


# This is a workaround in Core ML for topk with dynamic `k`:
#     * Core ML topk supports only constant `k`
#     * Luckily, Core ML gather supports dynamic `end`, so we workaround by argsort then gather
# This leads to a slightly different behaviour, though: top-k elements are always sorted
def dynamic_topk(
    x: Var, k: Var, axis: int, ascending: Optional[bool] = False, name: Optional[str] = None
):
    assert k.val is None, "Please use mb.topk directly if k is compile time known"

    indices = mb.argsort(x=x, axis=axis, ascending=ascending)
    if name is None:
        values = mb.gather_along_axis(x=x, indices=indices, axis=axis)
    else:
        values = mb.gather_along_axis(x=x, indices=indices, axis=axis, name=name)

    k_indices = mb.range_1d(end=k, start=0, step=1)
    values = mb.gather(x=values, indices=k_indices, axis=axis)
    if name is None:
        indices = mb.gather(x=indices, indices=k_indices, axis=axis)
    else:
        indices = mb.gather(x=indices, indices=k_indices, axis=axis, name=name)

    return values, indices


def solve_diagonal_einsum(parsed_vectors, vars):
    def solve_diagonal_einsum_one_step(parsed_vector, x):
        for i in range(len(parsed_vector)):
            for j in range(i + 1, len(parsed_vector)):
                if parsed_vector[i] != parsed_vector[j]:
                    continue

                perm = list(range(len(parsed_vector)))
                duplicated_indices = [j for j in range(len(parsed_vector)) if parsed_vector[j] == parsed_vector[i]]
                for i, j in enumerate(duplicated_indices):
                    perm[i], perm[j] = perm[j], perm[i]
                    parsed_vector[i], parsed_vector[j] = parsed_vector[j], parsed_vector[i]

                dims = mb.shape(x=x)
                dim_length = value_at(dims, duplicated_indices[0])

                indices = mb.range_1d(end=dim_length, start=0, step=1)
                indices = mb.stack(values=[indices] * len(duplicated_indices), axis=1)
                x = mb.transpose(x=x, perm=perm)
                x = mb.gather_nd(x=x, indices=indices)
                ret_parsed_vector = [parsed_vector[0]] + parsed_vector[len(duplicated_indices):]
                return ret_parsed_vector, x

    for i in range(len(vars)):
        while len(parsed_vectors[i]) != len(set(parsed_vectors[i])):
            parsed_vector, var = solve_diagonal_einsum_one_step(parsed_vectors[i], vars[i])
            parsed_vectors[i] = parsed_vector
            vars[i] = var
    return tuple(parsed_vectors), vars


def solve_sum_einsum(parsed_vectors, vars):
    """
    Apply reduce_sum for axes before binary einsum calculation if enable.

    e.g.:
    input : "abce,acd->ae"
    returns : "ace,ac->ae"

    In this example, since each of those axes is only used by one var and does not appear in the output,
    axes `b` and `d` can be reduced before binary einsum.
    """

    def solve_sum_einsum_one_step(src_axes, used_by_other_axes, x):
        dst_axes = []
        for axis in src_axes:
            if axis not in used_by_other_axes:
                continue
            dst_axes.append(axis)
        summed_axis_indices = [i for i in range(len(src_axes)) if src_axes[i] not in dst_axes]
        if summed_axis_indices:
            x = mb.reduce_sum(x=x, axes=summed_axis_indices)
        return dst_axes, x

    ret_parsed_vectors = []
    parsed_vectors = list(parsed_vectors)
    for i, var in enumerate(vars):
        used_by_other_axes = []
        for j, parsed_vector in enumerate(parsed_vectors):
            if i != j:
                used_by_other_axes += parsed_vector
        dst_axes, var = solve_sum_einsum_one_step(parsed_vectors[i], used_by_other_axes, vars[i])
        ret_parsed_vectors.append(dst_axes)
        vars[i] = var
    ret_parsed_vectors.append(parsed_vectors[-1])
    return ret_parsed_vectors, vars


def get_perm_transpose_einsum(src_axes: List[int], dst_axes: List[int]) -> List[int]:
    """
    :param src_axes: list[int]
    :param dst_axes: list[int]
    :return: list[int]
    """
    return [src_axes.index(s) for s in dst_axes]


def solve_transpose_einsum(src_parsed_vector: List[int], dst_parsed_vector: List[int], var: Var, name: str) -> Var:
    return mb.transpose(x=var, perm=get_perm_transpose_einsum(src_parsed_vector, dst_parsed_vector), name=name)


def solve_generic_einsum(parsed_vectors, vars, name) -> Var:
    """
    :param parsed_vectors: list[list[int]]
    :param vars:
        - list[var]
        - input variables
    :param name:
        - str
        - name to be assigned to the output var

    :return:
        - var
        - output var that contains the einsum result
    """

    parsed_vectors, vars = solve_diagonal_einsum(parsed_vectors, vars)
    parsed_vectors, vars = solve_sum_einsum(parsed_vectors, vars)
    if len(vars) == 1:
        return solve_transpose_einsum(parsed_vectors[0], parsed_vectors[1], vars[0], name)
    while len(vars) >= 2:
        out_vector = []
        input_symbols = list(itertools.chain.from_iterable(parsed_vectors[:2]))
        for symbol in itertools.chain.from_iterable(parsed_vectors[2:]):
            if symbol in input_symbols and symbol not in out_vector:
                out_vector.append(symbol)
        temp_parsed_vectors = [parsed_vectors[0], parsed_vectors[1], out_vector]
        parsed_vectors[0] = out_vector
        parsed_vectors.pop(1)
        vars[0] = solve_binary_generic_einsum(temp_parsed_vectors, vars[0], vars[1], name if len(vars) == 2 else None)
        vars.pop(1)
    return vars[0]


def solve_binary_generic_einsum(parsed_vectors, a_var, b_var, name) -> Var:
    def _concat_dims(dims, none_if_empty=False):
        if len(dims) == 0:
            if none_if_empty:
                return None
            else:
                return 1
        return mb.concat(values=dims, axis=0)

    a_axes, b_axes, out_axes = parsed_vectors

    a_dims = mb.shape(x=a_var)
    b_dims = mb.shape(x=b_var)

    batched_axes = []
    reduced_axes = []
    a_unique_axes = []
    b_unique_axes = []

    batch_dims = []
    reduce_dims = []
    a_unique_dims = []
    b_unique_dims = []

    for i, a_axis in enumerate(a_axes):
        a_dim = value_at(a_dims, i)
        if a_axis in b_axes:
            if a_axis in out_axes:
                batched_axes.append(a_axis)
                batch_dims.append(a_dim)
            else:
                reduced_axes.append(a_axis)
                reduce_dims.append(a_dim)
        else:
            a_unique_axes.append(a_axis)
            a_unique_dims.append(a_dim)
    concat_batch_dims = _concat_dims(batch_dims, True)
    # if there is no dim to reduce, then add a dummy dim,
    # so mb.matmul will reduce the dummy dim to achieve outer product
    concat_reduce_dims = _concat_dims(reduce_dims)
    # if there is no dim of `a` remains, then add a dummy dim for `a` as a matrix dim,
    # otherwise mb.matmul may mistake the batch dim of `a` as the matrix dim
    concat_a_unique_dims = _concat_dims(a_unique_dims)

    for i, b_axis in enumerate(b_axes):
        b_dim = value_at(b_dims, i)
        if b_axis not in a_axes:
            b_unique_axes.append(b_axis)
            b_unique_dims.append(b_dim)
    # if there is no dim of `b` remains, then add a dummy dim for `b`,
    # otherwise mb.matmul may mistake the batch dim of `b` as a matrix dim
    concat_b_unique_dims = _concat_dims(b_unique_dims)

    a_transpose_axes = batched_axes + a_unique_axes + reduced_axes
    a = mb.transpose(x=a_var, perm=get_perm_transpose_einsum(a_axes, a_transpose_axes))
    a_reshape_dims = _concat_dims(
        [mb.reduce_prod(x=x) for x in [concat_batch_dims, concat_a_unique_dims, concat_reduce_dims] if x is not None])
    a = mb.reshape(x=a, shape=a_reshape_dims)

    b_transpose_axes = batched_axes + reduced_axes + b_unique_axes
    b = mb.transpose(x=b_var, perm=get_perm_transpose_einsum(b_axes, b_transpose_axes))
    b_reshape_dims = _concat_dims(
        [mb.reduce_prod(x=x) for x in [concat_batch_dims, concat_reduce_dims, concat_b_unique_dims] if x is not None])
    b = mb.reshape(x=b, shape=b_reshape_dims)

    ab = mb.matmul(x=a, y=b)
    concat_batch_dims = _concat_dims(batch_dims, True)
    concat_a_unique_dims = _concat_dims(a_unique_dims, True)
    concat_b_unique_dims = _concat_dims(b_unique_dims, True)
    ab_reshaped_dims = _concat_dims(
        [
            x
            for x in [concat_batch_dims, concat_a_unique_dims, concat_b_unique_dims]
            if x is not None
        ],
        True,
    )
    # Removes excessive dimensions for scalar output
    if ab_reshaped_dims is None:
        if name is None:
            return mb.squeeze(x=ab)
        else:
            return mb.squeeze(x=ab, name=name)
    # Reshape tensor output to specified output shape
    else:
        ab = mb.reshape(x=ab, shape=ab_reshaped_dims)
        ab_reshaped_axes = batched_axes + a_unique_axes + b_unique_axes
        if name is None:
            ab = mb.transpose(x=ab, perm=get_perm_transpose_einsum(ab_reshaped_axes, out_axes))
        else:
            ab = mb.transpose(x=ab, perm=get_perm_transpose_einsum(ab_reshaped_axes, out_axes), name=name)
        return ab


def _decompose_scaled_dot_product_attention(
    q: Var,
    k: Var,
    v: Var,
    mask: Var,
    name: str,
    scale: Optional[Var] = None,
    before_op: Optional[Operation] = None,
) -> Var:
    # scale the query input
    embed_size = q.shape[-1]
    if is_symbolic(embed_size):
        raise ValueError(
            "The embedding size, i.e. last dimension of the shape of query tensor"
            " cannot be symbolic, in scaled_dot_product_attention op"
        )

    q, k, v = promote_input_dtypes([q, k, v])
    if scale is None:
        multiplicative_scale_factor = 1 / math.sqrt(embed_size)
        if types.builtin_to_string(q.dtype) == "fp16":
            multiplicative_scale_factor = np.float16(multiplicative_scale_factor)
    else:
        multiplicative_scale_factor = scale
    q = mb.mul(x=q, y=multiplicative_scale_factor, before_op=before_op)

    # multiply query and key input tensors
    # shape of output: (target_seq, source_seq) or (B,...,target_seq, source_seq)
    attn_weights = mb.matmul(x=q, y=k, transpose_y=True, before_op=before_op)

    # add mask if applicable
    if mask is not None:
        attn_weights = mb.add(x=attn_weights, y=mask, before_op=before_op)

    # do softmax
    attn_weights_normalized = mb.softmax(x=attn_weights, axis=-1, before_op=before_op)

    # multiply attn_weights and value tensor
    res = mb.matmul(x=attn_weights_normalized, y=v, name=name, before_op=before_op)
    return res


def _construct_constexpr_dequant_op(
    quantized_weights: np.ndarray,
    zero_point: Optional[Union[Var, np.ndarray, np.generic]],
    scale: Union[Var, np.ndarray, np.generic],
    axis: Optional[Union[Var, int]] = None,
    name: Optional[str] = None,
    before_op: Optional[Operation] = None,
) -> Var:
    """
    Constructs the constexpr op to represent the quantized weight.

    Use constexpr_affine_dequantize for pre-iOS18 and constexpr_blockwise_shift_scale for others.
    """
    if not is_current_opset_version_compatible_with(target.iOS18):
        # The constexpr_affine_dequantize op requires axis.
        if axis is None:
            # Infer the axis based on scale's shape.
            non_single_dim = [dim for dim, dim_size in enumerate(scale.shape) if dim_size > 1]
            if len(non_single_dim) > 2:
                raise ValueError(
                    "The constexpr_affine_dequantize op doesn't support scale which "
                    "have more than one non-single dimensions. Got scale with shape "
                    f"{scale.shape}"
                )
            # Empty non_single_dim means per-tensor quantization, just use a dummy axis.
            axis = 0 if len(non_single_dim) == 0 else non_single_dim[0]
        if isinstance(axis, int):
            axis = np.int32(axis)

        # The constexpr_affine_dequantize op requires zero_point.
        if zero_point is None:
            zero_point = np.zeros_like(scale).astype(quantized_weights.dtype)

        # The constexpr_affine_dequantize op requires scale and zero_point to have rank 0 or 1.
        if isinstance(scale, (np.ndarray, np.generic)):
            scale = np.squeeze(scale)
        if isinstance(zero_point, (np.ndarray, np.generic)):
            zero_point = np.squeeze(zero_point)
        if len(scale.shape) > 1 or len(zero_point.shape) > 1:
            raise ValueError(
                "The more fine-grained quantization (such as blockwise) is only supported since iOS18."
                "Please set minimum_deployment_target to iOS18 for using it."
            )

        kwargs = {
            "quantized_data": quantized_weights,
            "zero_point": zero_point,
            "scale": scale,
            "axis": axis,
        }
        if name is not None:
            kwargs["name"] = name
        if before_op is not None:
            kwargs["before_op"] = before_op
        return mb.constexpr_affine_dequantize(**kwargs)

    # For iOS18 constexpr_blockwise_shift_scale op, the data/scale/offset need to have same rank.
    if len(quantized_weights.shape) != len(scale.shape):
        if axis is not None:
            target_shape = [1] * len(quantized_weights.shape)
            target_shape[axis] = quantized_weights.shape[axis]
        else:
            target_shape = list(scale.shape) + [1] * (
                len(quantized_weights.shape) - len(scale.shape)
            )
        if np.prod(scale.shape) != np.prod(target_shape):
            raise ValueError(
                "Unable to infer scale's shape. Please provide a scale that has the "
                "same rank as the weight."
            )
        scale = scale.reshape(target_shape)

    # Check the value range to determine the true data type (such as int4/uint4).
    sub_byte_type = (
        types.uint4
        if types.numpy_type_to_builtin_type(quantized_weights.dtype).is_unsigned()
        else types.int4
    )
    sub_byte_range = types.type_mapping._TYPES_TO_RANGE[sub_byte_type]
    if (
        np.max(quantized_weights) <= sub_byte_range.high
        and np.min(quantized_weights) >= sub_byte_range.low
    ):
        quantized_weights = quantized_weights.astype(types.nptype_from_builtin(sub_byte_type))

    kwargs = {
        "data": quantized_weights,
        "scale": scale,
    }
    if zero_point is not None and np.any(zero_point):
        # Only pass the offset parameter when not all elements in `zero_point` are zeroes.
        zero_point = zero_point.reshape(scale.shape)
        # When zero_point is integer, it's required to have the same dtype as the quantized weight.
        if np.issubdtype(zero_point.dtype, np.integer):
            zero_point = zero_point.astype(quantized_weights.dtype)
        kwargs["offset"] = zero_point
    if name is not None:
        kwargs["name"] = name
    if before_op is not None:
        kwargs["before_op"] = before_op
    return mb.constexpr_blockwise_shift_scale(**kwargs)


def _construct_constexpr_lut_op(
    indices: Union[Var, np.ndarray],
    lut: Union[Var, np.ndarray],
    vector_axis: Optional[Union[Var, int]] = None,
    name: Optional[str] = None,
    before_op: Optional[Operation] = None,
) -> Var:
    """
    Constructs the constexpr op to represent the palettized weight, using different versions of `constexpr_lut_to_dense`
    op based on the opset version.

    The input `indices`, `lut` and `vector_axis` (if provided) should follow iOS18 `constexpr_lut_to_dense` op's def.
    """
    # Avoid circular import
    from coremltools.optimize.coreml import _utils as optimize_utils

    kwargs = {"indices": indices, "lut": lut}
    if name is not None:
        kwargs["name"] = name
    if before_op is not None:
        kwargs["before_op"] = before_op

    if is_current_opset_version_compatible_with(target.iOS18):
        if vector_axis is not None:
            kwargs["vector_axis"] = vector_axis
        if not isinstance(lut, Var):
            # Adjust dtype if necessary.
            num_palettes = lut.shape[-2]
            nbits = int(math.log2(num_palettes))
            target_np_dtype = types.nptype_from_builtin(types.string_to_builtin(f"uint{nbits}"))
            kwargs["indices"] = indices.astype(target_np_dtype)
    else:
        if isinstance(lut, Var):
            lut: np.ndarray = lut.val
        lut_params = optimize_utils.LutParams(indices=indices, lut=lut, vector_axis=vector_axis)
        lut_params: optimize_utils.LutParamsIos16 = optimize_utils.ios18_lut_params_to_ios16(
            lut_params
        )
        kwargs.update(lut_params._asdict())

    return mb.constexpr_lut_to_dense(**kwargs)
