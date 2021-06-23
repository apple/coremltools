#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math

import coremltools.converters

from coremltools.converters.mil.mil import get_new_symbol
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from ._op_reqs import *

def broadcast_shapes(shape_x, shape_y):
    """
    Check and broadcast given input shapes.
    :param shape_x: tuple of int or symbols
        Shape of the first tensor (possibly symbolic).
    :param shape_y: tuple of int or symbols
        Shape of the second tensor (possibly symbolic).
    :return: tuple of int or symbols
        Result from broadcast.
    """
    shape_x = tuple(shape_x)
    shape_y = tuple(shape_y)
    if len(shape_x) < len(shape_y):
        shape_x = tuple([1] * (len(shape_y) - len(shape_x))) + shape_x
    if len(shape_y) < len(shape_x):
        shape_y = tuple([1] * (len(shape_x) - len(shape_y))) + shape_y

    ret_shapes = list()
    for i in range(len(shape_x)):
        x_unknown = is_symbolic(shape_x[i])
        y_unknown = is_symbolic(shape_y[i])
        if shape_x[i] == 1:
            ret_shapes.append(shape_y[i])
        elif shape_y[i] == 1:
            ret_shapes.append(shape_x[i])
        elif not y_unknown and shape_y[i] > 1:
            if not x_unknown and shape_x[i] != shape_y[i]:
                raise ValueError(
                    "Incompatible dim {} in shapes {} vs. {}".format(
                        i, shape_x, shape_y
                    )
                )
            ret_shapes.append(shape_y[i])
        elif not x_unknown and shape_x[i] > 1:
            if not y_unknown and shape_x[i] != shape_y[i]:
                raise ValueError(
                    "Incompatible dim {} in shapes {} vs. {}".format(
                        i, shape_x, shape_y
                    )
                )
            ret_shapes.append(shape_x[i])
        elif x_unknown or y_unknown:
            ret_shapes.append(get_new_symbol())
        else:
            assert shape_x[i] == shape_y[i]
            ret_shapes.append(shape_x[i])

    return tuple(ret_shapes)


def promoted_primitive_type(type1, type2):
    """
    Given a pair of tensor or primitive types, find the smallest type that can store an instance
    of their primitive type.
    """
    ptype1 = type1.get_primitive() if types.is_tensor(type1) else type1
    ptype2 = type2.get_primitive() if types.is_tensor(type2) else type2
    return types.promote_types(ptype1, ptype2)


def effective_kernel(kernel_shape, dilations):
    """

    Args:
        kernel_shape: tuple[int] representing the kernel shape in each
            given dimension.
        dilations: tuple[int] representing the dilation of the kernel
            in each given dimension.  Must be the same length as
            kernel_shape, and is assumed to give the dimensions in
            the same order as kernel_shape

    Returns: tuple[int] representing the effective shape of the kernel
        in each given dimension, with each dimension in the order given,
        taking into account dilation.
        See http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions
        Note that a dilation of 1 is equivalent to having no dilation.

    """
    if len(kernel_shape) != len(dilations):
        raise ValueError(
            "kernel_shape ({}) and dilations ({}) must be the same length".format(
                len(kernel_shape), len(dilations)
            )
        )
    return [(k - 1) * d + 1 for k, d in zip(kernel_shape, dilations)]


def aggregated_pad(
    pad_type,
    kernel_shape,
    input_shape=None,
    strides=None,
    dilations=None,
    custom_pad=None,
):
    """
    Args
        pad_type: string. Must be one of ('same', 'valid', 'custom')

        kernel_shape: [kH, kW, ...]: spatial kernel dims (excluding channels)

        input_shape: [iH, iW, ...]: spatial input dims (excluding channels)
            Required iff pad_type == 'same'

        strides: [sH, sW, ...]: spatial strides (excluding channels)
            Required iff pad_type == 'same'

        dilations: [dH, dW, ...]: dilations (excluding channels)
            If not provided, defaults to [1, 1, ...], effectively no dilation.

        custom_pad: Required iff pad_type == 'custom'.
            custom_pad[2*i], custom_pad[2*i+1] are before/after custom padding
            for spatial dim i.


    Returns:
        A list of total (before + after) padding for each spatial dimension in kernel_shape.
    """
    num_spatial_dims = len(kernel_shape)
    if dilations is None:
        dilations = [1] * num_spatial_dims
    elif len(dilations) != num_spatial_dims:
        raise ValueError(
            "dilations must have same length as kernel_shape ({}, but got {})".format(
                num_spatial_dims, len(dilations)
            )
        )
    if pad_type == "same":
        if input_shape is None or len(input_shape) != num_spatial_dims:
            raise ValueError(
                "For SAME padding input_shape must not be None and must have "
                "same length as kernel_shape ({}, but got {})".format(
                    num_spatial_dims,
                    len(input_shape) if input_shape is not None else "None",
                )
            )
        if strides is None or len(strides) != num_spatial_dims:
            raise ValueError(
                "For SAME padding strides must not be None and must have "
                "same length as kernel_shape ({}, but got {})".format(
                    num_spatial_dims, len(strides) if strides is not None else "None"
                )
            )
        effective_ks = effective_kernel(kernel_shape, dilations)
        return [
            int(max(0, s * math.ceil(float(i) / float(s)) - i + k - s))
            if not is_symbolic(i) else get_new_symbol()
            for i, k, s in zip(input_shape, effective_ks, strides)
        ]
    if pad_type == "valid":
        return [0] * num_spatial_dims
    if pad_type == "custom":
        if custom_pad is None or len(custom_pad) != 2 * num_spatial_dims:
            raise ValueError("Invalid custom_pad.")
        return [
            custom_pad[2 * d] + custom_pad[2 * d + 1] for d in range(num_spatial_dims)
        ]
    raise ValueError('Invalid padding pad_type "{}"'.format(pad_type))


def spatial_dimensions_out_shape(
    pad_type, input_shape, kernel_shape, strides, dilations=None, custom_pad=None, ceil_mode=False,
):
    """
    Args
        pad_type: string. Must be one of ('same', 'valid', 'custom')

        input_shape: [iH, iW, ...]: spatial input dims (excluding channels)
            Required iff pad_type == 'same'

        kernel_shape: [kH, kW, ...]: spatial kernel dims (excluding channels)

        strides: [sH, sW, ...]: spatial strides (excluding channels)
            Required iff pad_type == 'same'

        dilations: [dH, dW, ...]: dilations (excluding channels)
            If not provided, defaults to [1, 1, ...], effectively no dilation.

        custom_pad: Required iff pad_type == 'custom'.
            custom_pad[2*i], custom_pad[2*i+1] are before/after custom padding
            for spatial dim i.

        ceil_mode: determines the padding and output shape.
             When ceil mode is True:
                out_dim = floor((in_dim + pad_l + pad_r - kernel_size + (stride-1)) / stride) + 1
                if (out_dim-1) * stride >= in_dim + pad_l and (pad_l > 0 or pad_r > 0):
                    out_dim = out_dim - 1
            When ceil mode is False:
                out_dim = floor((in_dim + pad_l + pad_r - kernel_size) / stride) + 1

    Returns:
        A list of spatial output sizes for each spatial dimension of kernel_shape.

    """
    num_spatial_dims = len(kernel_shape)
    if dilations is None:
        dilations = [1] * num_spatial_dims
    if custom_pad is None:
        custom_pad = [0] * num_spatial_dims * 2
    if not (
        len(input_shape)
        == len(kernel_shape)
        == len(strides)
        == len(dilations)
        == len(custom_pad) / 2
    ):
        raise ValueError(
            "input_shape (length {}), kernel_shape (length {}), "
            "strides (length {}), dilations (length {}), and "
            "custom_pad (length {}) divided by two must all be "
            "the same length".format(
                len(input_shape),
                len(kernel_shape),
                len(strides),
                len(dilations),
                len(custom_pad),
            )
        )

    pad = aggregated_pad(
        pad_type=pad_type,
        kernel_shape=kernel_shape,
        input_shape=input_shape,
        strides=strides,
        dilations=dilations,
        custom_pad=custom_pad,
    )
    effective_ks = effective_kernel(kernel_shape, dilations)
    out_shape = []
    for r in range(num_spatial_dims):
        if is_symbolic(input_shape[r]):
            out_shape.append(get_new_symbol())
        else:
            if not ceil_mode:
                out_shape.append(math.floor((input_shape[r] + pad[r] - effective_ks[r]) / strides[r] + 1))
            else:
                out_dim = math.floor((input_shape[r] + pad[r] - effective_ks[r] + strides[r] - 1) / strides[r] + 1)
                if (out_dim - 1) * strides[r] >= input_shape[r] + pad[r]/2 and pad[r] > 0:
                    out_dim = out_dim - 1
                out_shape.append(out_dim)
    return out_shape


def parse_einsum_equation(equation):
    """
    Args
        equation : str

     parse the equation in the following manner:
     (running example: "nchw,nwhr->nchr")

    step 1: split the equation with delimiter "->"
        e.g.: this will give "nchw,nwhr" and "nchr"

    step 2: split the LHS equation string with delimiter ","
        e.g.: this will give input1 : "nchw", input2: "nwhr"

    step 3: map each character to a unique integer, which is incremented.
            Iterate over input1, input2 and output, in that order.
            e.g.: input 1, i.e., "nchw" will give vector {0,1,2,3}
                  input 2, i.e, "nwhr" will produce {0,3,2,4}
                  output , i.e. "nchr" will produce {0,1,2,4}

    return vectors corresponding to the 2 inputs and the output
    """
    input_output_str = equation.split('->')
    assert len(input_output_str) == 2, "unsupported einsum equation {}".format(equation)
    input_str = input_output_str[0]
    output_str = input_output_str[1]

    inputs = input_str.split(',')
    assert len(inputs) == 2, "unsupported einsum equation {}".format(equation)
    input1_str = inputs[0]
    input2_str = inputs[1]

    input1_vec = [-1 for i in range(len(input1_str))]
    input2_vec = [-1 for i in range(len(input2_str))]
    output_vec = [-1 for i in range(len(output_str))]
    map_char_to_int = {}

    def _update_vec(str, vec, map_char_to_int, index):
        for i, s in enumerate(str):
            if s not in map_char_to_int:
                map_char_to_int[s] = index
                index += 1
            vec[i] = map_char_to_int[s]
        return index

    index = _update_vec(input1_str, input1_vec, map_char_to_int, 0)
    index = _update_vec(input2_str, input2_vec, map_char_to_int, index)
    index = _update_vec(output_str, output_vec, map_char_to_int, index)

    return input1_vec, input2_vec, output_vec
