#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import itertools

from typing import List, Optional

from coremltools.converters.mil.input_types import InputType
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Var, types
from coremltools.converters.mil.mil.ops.defs._utils import parse_einsum_equation
from coremltools.converters.mil.mil.types.symbolic import any_symbolic, is_symbolic


def value_at(x: Var, idx: int, name=None, before_op=None):
    """
    input x: 1D tensor (vector).
    return value at index idx. x[idx].
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


def _reverse_input_einsum_eq(equation: str) -> str:
    """
    Reverse the input order of the einsum eqaution
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
