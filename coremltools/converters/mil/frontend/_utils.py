#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb, types
from coremltools.converters.mil.mil.ops.defs._utils import parse_einsum_equation
from coremltools.converters.mil.mil.types.symbolic import any_symbolic, is_symbolic


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

    parsed_vectors = parse_einsum_equation(equation)
    equation_rev = _reverse_input_einsum_eq(equation)
    parsed_vectors_rev = parse_einsum_equation(equation_rev)

    def _swap(a, b):
        return b, a

    if parsed_vectors == ([0,1,2,3],[0,1,4,3],[0,1,2,4]) or parsed_vectors_rev == ([0,1,2,3],[0,1,4,3],[0,1,2,4]): # equation == "bnqd,bnkd->bnqk"
        if parsed_vectors_rev == ([0,1,2,3],[0,1,4,3],[0,1,2,4]):
            a_var, b_var = _swap(a_var, b_var)
        x = mb.matmul(x=a_var, y=b_var, transpose_x=False, transpose_y=True, name=name)
    elif parsed_vectors == ([0,1,2],[2,3],[0,1,3]) or parsed_vectors_rev == ([0,1,2],[2,3],[0,1,3]): # equation == "abc,cd->abd"
        if parsed_vectors_rev == ([0,1,2],[2,3],[0,1,3]):
            a_var, b_var = _swap(a_var, b_var)
        x = mb.matmul(x=a_var, y=b_var, transpose_x=False, transpose_y=False, name=name)
    elif parsed_vectors == ([0,1,2],[2,3,4],[0,1,3,4]) or parsed_vectors_rev == ([0,1,2],[2,3,4],[0,1,3,4]): # equation == "abc,cde->abde"
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
    elif parsed_vectors == ([0,1,2,3],[2,3,4],[0,1,4]) or parsed_vectors_rev == ([0,1,2,3],[2,3,4],[0,1,4]): # equation == "abcd,cde->abe"
        if parsed_vectors_rev == ([0,1,2,3],[2,3,4],[0,1,4]):
            a_var, b_var = _swap(a_var, b_var)
        x_1 = mb.reshape(x=a_var, shape=[a_var.shape[0], a_var.shape[1], a_var.shape[2] * a_var.shape[3]])
        x_2 = mb.reshape(x=b_var, shape=[b_var.shape[0] * b_var.shape[1], b_var.shape[2]])
        x = mb.matmul(
                x=x_1, y=x_2, transpose_x=False, transpose_y=False, name=name
        )
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
        raise NotImplementedError(
            "Einsum unsupported equation format: ", equation
        )

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
