#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math
import numbers
from typing import List, Tuple


import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Var, get_new_symbol, types
from coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_unary import (
    cast as cast_op_class,
)
from coremltools.converters.mil.mil.types import builtin_to_string, promote_dtypes
from coremltools.converters.mil.mil.types.symbolic import is_symbolic

MAX_SIZE_CONSTANT_FOLDING = 1024 * 1024 / 4 # When a fp32 const takes over 1MB, we won't create a const op for that

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

    def raise_incompatible_dim_exception():
        raise ValueError(
            "Incompatible dim {} in shapes {} vs. {}".format(
                i, shape_x, shape_y
            )
        )

    shape_x = tuple(shape_x)
    shape_y = tuple(shape_y)
    if len(shape_x) < len(shape_y):
        shape_x = tuple([1] * (len(shape_y) - len(shape_x))) + shape_x
    if len(shape_y) < len(shape_x):
        shape_y = tuple([1] * (len(shape_x) - len(shape_y))) + shape_y

    ret_shapes = list()
    for i in range(len(shape_x)):
        if shape_x[i] == shape_y[i]:
            ret_shapes.append(shape_x[i])
        else:
            is_x_unknown = is_symbolic(shape_x[i])
            is_y_unknown = is_symbolic(shape_y[i])
            if shape_x[i] == 1:
                ret_shapes.append(shape_y[i])
            elif shape_y[i] == 1:
                ret_shapes.append(shape_x[i])
            elif not is_y_unknown and shape_y[i] > 1:
                if not is_x_unknown and shape_x[i] != shape_y[i]:
                    raise_incompatible_dim_exception()
                ret_shapes.append(shape_y[i])
            elif not is_x_unknown and shape_x[i] > 1:
                if not is_y_unknown and shape_x[i] != shape_y[i]:
                    raise_incompatible_dim_exception()
                ret_shapes.append(shape_x[i])
            elif is_x_unknown or is_y_unknown:
                ret_shapes.append(get_new_symbol())
            else:
                raise_incompatible_dim_exception()

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
        pad_type: string. Must be one of ('same', 'same_lower', 'valid', 'custom')

        kernel_shape: [kH, kW, ...]: spatial kernel dims (excluding channels)

        input_shape: [iH, iW, ...]: spatial input dims (excluding channels)
            Required iff pad_type in ['same', 'same_lower']

        strides: [sH, sW, ...]: spatial strides (excluding channels)
            Required iff pad_type in ['same', 'same_lower']

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
    if pad_type in ["same", "same_lower"]:
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
        pad_type: string. Must be one of ('same', 'same_lower', 'valid', 'custom')

        input_shape: [iH, iW, ...]: spatial input dims (excluding channels)
            Required iff pad_type in ['same', 'same_lower']

        kernel_shape: [kH, kW, ...]: spatial kernel dims (excluding channels)

        strides: [sH, sW, ...]: spatial strides (excluding channels)
            Required iff pad_type in ['same', 'same_lower']

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


def parse_einsum_equation(equation: str) -> Tuple[List]:
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
    _update_vec(output_str, output_vec, map_char_to_int, index)

    return input1_vec, input2_vec, output_vec

def compute_gather(params, indices, axis, batch_dims):
    """
    This utility function computes the gather operation with batch_dims supported.
    """
    def compute_gather_helper(params, indices, axis):
        scalar_indices = isinstance(indices, numbers.Integral)
        if scalar_indices:
            res = np.take(params, [indices], axis)
            res2 = np.squeeze(res, axis=axis)
            if isinstance(res2, np.ndarray) and len(res2.shape) == 0:
                # res2 is a scalar, but represented as np.array(symbol,
                # dtype=np.object) which np.squeeze can't remove.
                return res2.item()
            return res2
        return np.take(params, indices, axis)

    if batch_dims == 0:
        return compute_gather_helper(params, indices, axis)

    params_shape = params.shape
    indices_shape = indices.shape
    batch_shape = params_shape[:batch_dims]

    params_new_shape = [np.prod(batch_shape)] + list(params_shape[batch_dims:])
    indices_new_shape = [np.prod(batch_shape)] + list(indices_shape[batch_dims:])
    params_reshape = np.reshape(params, params_new_shape)
    indices_reshape = np.reshape(indices, indices_new_shape)

    res = []
    for p, i in zip(params_reshape, indices_reshape):
        res.append(compute_gather_helper(p, i, axis - batch_dims))
    res = np.stack(res)
    res_new_shape = tuple(batch_shape) + tuple(res.shape[1:])
    return np.reshape(res, res_new_shape)

def promote_input_dtypes(input_vars):
    """
    This utility function promotes all input variables to the same data type.
    It is used to homogenize inputs to an op such as matmul / elementwise_binary,
    and not the inputs to a function itself.
    """
    def _is_same_dtype(dtype1, dtype2):
        return builtin_to_string(dtype1) == builtin_to_string(dtype2)

    def _promoted_var(var, promoted_dtype):
        if var.val is None:
            x = mb.cast(
                x=var, dtype=builtin_to_string(promoted_dtype), name=var.name + "_promoted")
        else:
            const_value_after_cast = cast_op_class.get_cast_value(var, builtin_to_string(promoted_dtype))
            x = mb.const(val=const_value_after_cast, name=var.name + "_promoted")
        return x

    for i, var in enumerate(input_vars):
        if not isinstance(var, Var):
            input_vars[i] = mb.const(val=var)

    promoted_dtype = promote_dtypes([var.dtype for var in input_vars])

    for i, var in enumerate(input_vars):
        if not _is_same_dtype(var.dtype, promoted_dtype):
            input_vars[i] = _promoted_var(var, promoted_dtype)

    return input_vars


def solve_slice_by_index_shape(x_shape, begin, end, stride, begin_mask, end_mask, squeeze_mask):
    """
    Helper function to solve the shape of tensor slicing.
    """
    ret_shape = []

    if begin is None or len(begin) == 0:
        begin = [None] * len(x_shape)
    if end is None or len(end) == 0:
        end = [None] * len(x_shape)

    # solve for shape inference
    for idx in range(len(x_shape)):
        # skip if we want to squeeze the dimension
        if squeeze_mask[idx]:
            continue

        # for those a[:] cases
        if begin_mask[idx] and end_mask[idx]:
            if is_symbolic(x_shape[idx]):
                if stride[idx] == -1 or stride[idx] == 1:
                    ret_shape.append(x_shape[idx])
                else:
                    ret_shape.append(get_new_symbol())
            else:
                num = np.ceil(float(x_shape[idx]) / abs(stride[idx])).astype(
                    np.int32
                )
                ret_shape.append(num)
            continue

        # for symbolic case
        if is_symbolic(x_shape[idx]):
            ret_shape.append(get_new_symbol())
            continue

        # for single-element extraction case
        if x_shape[idx] == 1:
            ret_shape.append(1)
            continue

        # when begin and end are not determined
        if begin[idx] is None and not begin_mask[idx]:
            ret_shape.append(get_new_symbol())
            continue
        if end[idx] is None and not end_mask[idx]:
            ret_shape.append(get_new_symbol())
            continue

        # parse negative dimention
        if begin[idx] is not None and begin[idx] < 0:
            begin[idx] = max(0, begin[idx] + x_shape[idx])
        if end[idx] is not None and end[idx] < 0:
            end[idx] = max(0, end[idx] + x_shape[idx])

        # compute shape
        low, high = [0, x_shape[idx]] if stride[idx] > 0 else [-1, x_shape[idx] - 1]
        begin_idx, end_idx = (
            [begin[idx], end[idx]] if stride[idx] > 0 else [end[idx], begin[idx]]
        )
        is_begin_mask, is_end_mask = (
            [begin_mask[idx], end_mask[idx]]
            if stride[idx] > 0
            else [end_mask[idx], begin_mask[idx]]
        )
        if is_begin_mask:
            begin_idx = low
        end_idx = high if is_end_mask else min(end_idx, high)
        num = np.ceil(float(end_idx - begin_idx) / abs(stride[idx])).astype(
            np.int32
        )
        ret_shape.append(max(0, num))

    return ret_shape
