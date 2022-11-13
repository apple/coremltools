#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.input_types import InputType
from coremltools.converters.mil.mil import Builder as mb, types
from coremltools.converters.mil.mil.ops.defs._utils import parse_einsum_equation
from coremltools.converters.mil.mil.types.symbolic import any_symbolic, is_symbolic


def value_at(x, idx, name=None):
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
    return mb.slice_by_index(**args)


def _reverse_input_einsum_eq(equation):
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


def build_einsum_mil(a_var, b_var, equation, name):
    """
    Get MIL variables as input and build a variable using MIL builder, that
    contains the output of the einsum equation

    :param a_var:
        - var
        - first input variable
    :param b_var:
        - var
        - second input variable
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
    equation_rev = _reverse_input_einsum_eq(equation)
    parsed_vectors_rev = parse_einsum_equation(equation_rev)

    def _swap(a, b):
        return b, a

    is_dynamic = any_symbolic(a_var.shape) or any_symbolic(b_var.shape)
    if parsed_vectors == ([0,1,2,3],[0,1,4,3],[0,1,2,4]) or parsed_vectors_rev == ([0,1,2,3],[0,1,4,3],[0,1,2,4]): # equation == "bnqd,bnkd->bnqk"
        if parsed_vectors_rev == ([0,1,2,3],[0,1,4,3],[0,1,2,4]):
            a_var, b_var = _swap(a_var, b_var)
        x = mb.matmul(x=a_var, y=b_var, transpose_x=False, transpose_y=True, name=name)
    elif parsed_vectors == ([0,1,2],[2,3],[0,1,3]) or parsed_vectors_rev == ([0,1,2],[2,3],[0,1,3]): # equation == "abc,cd->abd"
        if parsed_vectors_rev == ([0,1,2],[2,3],[0,1,3]):
            a_var, b_var = _swap(a_var, b_var)
        x = mb.matmul(x=a_var, y=b_var, transpose_x=False, transpose_y=False, name=name)
    elif (parsed_vectors == ([0,1,2],[2,3,4],[0,1,3,4]) or parsed_vectors_rev == ([0,1,2],[2,3,4],[0,1,3,4])) and not is_dynamic: # equation == "abc,cde->abde"
        if parsed_vectors_rev == ([0,1,2],[2,3,4],[0,1,3,4]):
            a_var, b_var = _swap(a_var, b_var)
        x_1 = mb.reshape(x=a_var, shape=[a_var.shape[0] * a_var.shape[1], a_var.shape[2]])
        x_2 = mb.reshape(x=b_var, shape=[b_var.shape[0], b_var.shape[1] * b_var.shape[2]])
        x = mb.matmul(x=x_1, y=x_2, transpose_x=False, transpose_y=False)
        x = mb.reshape(
            x=x, shape=[a_var.shape[0], a_var.shape[1], b_var.shape[1], b_var.shape[2]], name=name
        )
    elif parsed_vectors == ([0,1,2,3],[0,4,2,3],[0,2,4,1]) or parsed_vectors_rev == ([0,1,2,3],[0,4,2,3],[0,2,4,1]): # equation == "BTNH,BFNH->BNFT"
        if parsed_vectors_rev == ([0,1,2,3],[0,4,2,3],[0,2,4,1]):
            a_var, b_var = _swap(a_var, b_var)
        x_1 = mb.transpose(x=a_var, perm=[0, 2, 1, 3])
        x_2 = mb.transpose(x=b_var, perm=[0, 2, 1, 3])
        x = mb.matmul(x=x_2, y=x_1, transpose_x=False, transpose_y=True, name=name)
    elif parsed_vectors == ([0,1,2,3],[0,3,1,4],[0,2,1,4]) or parsed_vectors_rev == ([0,1,2,3],[0,3,1,4],[0,2,1,4]): # equation == "BNFT,BTNH->BFNH"
        if parsed_vectors_rev == ([0,1,2,3],[0,3,1,4],[0,2,1,4]):
            a_var, b_var = _swap(a_var, b_var)
        b_var = mb.transpose(x=b_var, perm=[0, 2, 1, 3])
        x = mb.matmul(x=a_var, y=b_var, transpose_x=False, transpose_y=False)
        x = mb.transpose(x=x, perm=[0, 2, 1, 3], name=name)
    elif (parsed_vectors == ([0,1,2,3],[2,3,4],[0,1,4]) or parsed_vectors_rev == ([0,1,2,3],[2,3,4],[0,1,4])) and not is_dynamic: # equation == "abcd,cde->abe"
        if parsed_vectors_rev == ([0,1,2,3],[2,3,4],[0,1,4]):
            a_var, b_var = _swap(a_var, b_var)
        x_1 = mb.reshape(x=a_var, shape=[a_var.shape[0], a_var.shape[1], a_var.shape[2] * a_var.shape[3]])
        x_2 = mb.reshape(x=b_var, shape=[b_var.shape[0] * b_var.shape[1], b_var.shape[2]])
        x = mb.matmul(x=x_1, y=x_2, transpose_x=False, transpose_y=False, name=name)
    elif parsed_vectors == ([0,1,2,3],[0,3,2,4],[0,1,2,4]) or parsed_vectors_rev == ([0,1,2,3],[0,3,2,4],[0,1,2,4]): # equation == "nchw,nwhu->nchu"
        if parsed_vectors == ([0,1,2,3],[0,3,2,4],[0,1,2,4]):
            x = mb.einsum(values=(a_var, b_var), equation=equation, name=name)
        else:
            x = mb.einsum(values=(b_var, a_var), equation=equation_rev, name=name)
    elif parsed_vectors == ([0,1,2],[2,1,3],[0,1,3]) or parsed_vectors_rev == ([0,1,2],[2,1,3],[0,1,3]): # equation == "chw,whu->chu"
        if parsed_vectors == ([0,1,2],[2,1,3],[0,1,3]):
            x = mb.einsum(values=(a_var, b_var), equation=equation, name=name)
        else:
            x = mb.einsum(values=(b_var, a_var), equation=equation_rev, name=name)
    else:
        x = solve_generic_einsum(parsed_vectors, a_var, b_var, name)

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


def get_output_names(outputs):
    """
    :param: list[ct.TensorType/ct.ImageType]
    :return: list[str]
    """
    output_names = None
    if outputs is not None:
        assert all([isinstance(t, InputType) for t in outputs]), \
            "outputs must be a list of ct.ImageType or ct.TensorType"
        output_names = [t.name for t in outputs]
        if all([name is None for name in output_names]):
            output_names = None
    return output_names


def solve_generic_einsum(parsed_vectors, a_var, b_var, name):
    """
    :param parsed_vectors: list[list[int]]
    :param a_var:
        - var
        - first input variable
    :param b_var:
        - var
        - second input variable
    :param name:
        - str
        - name to be assigned to the output var

    :return:
        - var
        - output var that contains the einsum result
    """

    def _get_perm(src_axes, dst_axes):
        """
        :param src_axes: list[int]
        :param dst_axes: list[int]
        :return: list[int]
        """
        return [src_axes.index(s) for s in dst_axes]

    def _concat_dims(dims, none_if_empty=False):
        if len(dims) == 0:
            if none_if_empty:
                return None
            else:
                return 1
        return mb.concat(values=dims, axis=0)

    a_axes, b_axes, out_axes = parsed_vectors

    if len(a_axes) > len(set(a_axes)) or len(b_axes) > len(set(b_axes)):
        raise ValueError(
            "Generic einsum does not support trace operation."
        )

    if not out_axes:
        raise ValueError(
            "Generic einsum does not support scalar output."
        )

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
    concat_batch_dims = _concat_dims(batch_dims)
    concat_reduce_dims = _concat_dims(reduce_dims)
    concat_a_unique_dims = _concat_dims(a_unique_dims)

    for i, b_axis in enumerate(b_axes):
        b_dim = value_at(b_dims, i)
        if b_axis not in a_axes:
            b_unique_axes.append(b_axis)
            b_unique_dims.append(b_dim)
    concat_b_unique_dims = _concat_dims(b_unique_dims)

    a_transpose_axes = batched_axes + a_unique_axes + reduced_axes
    a = mb.transpose(x=a_var, perm=_get_perm(a_axes, a_transpose_axes))
    a_reshape_dims = _concat_dims(
        [mb.reduce_prod(x=x) for x in [concat_batch_dims, concat_a_unique_dims, concat_reduce_dims] if x is not None])
    a = mb.reshape(x=a, shape=a_reshape_dims)

    b_transpose_axes = batched_axes + reduced_axes + b_unique_axes
    b = mb.transpose(x=b_var, perm=_get_perm(b_axes, b_transpose_axes))
    b_reshape_dims = _concat_dims(
        [mb.reduce_prod(x=x) for x in [concat_batch_dims, concat_reduce_dims, concat_b_unique_dims] if x is not None])
    b = mb.reshape(x=b, shape=b_reshape_dims)

    ab = mb.matmul(x=a, y=b)
    concat_batch_dims = _concat_dims(batch_dims, True)
    concat_a_unique_dims = _concat_dims(a_unique_dims, True)
    concat_b_unique_dims = _concat_dims(b_unique_dims, True)
    ab_reshaped_dims = _concat_dims(
        [x for x in [concat_batch_dims, concat_a_unique_dims, concat_b_unique_dims] if x is not None])
    ab = mb.reshape(x=ab, shape=ab_reshaped_dims)
    ab_reshaped_axes = batched_axes + a_unique_axes + b_unique_axes
    ab = mb.transpose(x=ab, perm=_get_perm(ab_reshaped_axes, out_axes), name=name)
    return ab
