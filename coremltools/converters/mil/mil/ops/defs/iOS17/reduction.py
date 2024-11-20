#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import DefaultInputs, InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.reduction import reduce_arg as _reduce_arg_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


class reduce_arg(_reduce_arg_iOS15):
    _VALID_OUTPUT_DTYPES = ("int32", "uint16")

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        keep_dims=TensorInputType(const=True, optional=True, type_domain=types.bool),
        output_dtype=TensorInputType(const=True, optional=True, type_domain=types.str),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def default_inputs(self):
        return DefaultInputs(
            axis=-1,
            keep_dims=False,
            output_dtype="int32",
        )

    def type_inference(self):
        reduced_shape = self._find_reduced_shape()
        output_dtype = self.output_dtype.val.lower()
        if output_dtype not in self._VALID_OUTPUT_DTYPES:
            raise ValueError(
                f'Invalid "output_dtype" {output_dtype}. Only support {self._VALID_OUTPUT_DTYPES}'
            )
        return types.tensor(types.string_to_builtin(output_dtype), tuple(reduced_shape))


@register_op(opset_version=_IOS17_TARGET)
class reduce_argmax(reduce_arg):
    """
    Computes the indices of the maximum value across dimensions of a tensor.
    In case of ties, the identity of the return value is not guaranteed.

    The differences between this version and the iOS 15 :py:class:`~.iOS15.reduction.reduce_argmax` are:
        * The output supports uint16 dtype.
        * New optional input ``output_dtype``.

    Parameters
    ----------
    x: <\\*, T> (Required)
        * Must be 1-dimensional or higher.

    axis: const<i32> (Optional)
        * The dimension to reduce. Default is ``-1``.

    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` by removing the dimension
          specified in ``axis``.
        * If ``True``, retain reduced axis with length ``1``.

    output_dtype: const<str> (Optional)
        * Possible values: ``uint16``, ``int32``.
        * If set, then value type inference will output using that dtype.
        * Default is ``int32``.

    Returns
    -------
    <\\*, U>

    Attributes
    ----------
    T: fp16, fp32, i32
    U: int32, uint16
    """

    def get_operator(self):
        return np.argmax


@register_op(opset_version=_IOS17_TARGET)
class reduce_argmin(reduce_arg):
    """
    Computes the indices of the minimum value across dimensions of a tensor.
    In case of ties, the identity of the return value is not guaranteed.

    The differences between this version and the iOS 15 :py:class:`~.iOS15.reduction.reduce_argmin` are:
        * The output supports uint16 dtype.
        * New optional input ``output_dtype``.

    Parameters
    ----------
    x: <\\*, T> (Required)
        * Must be 1-dimensional or higher.

    axis: const<i32> (Optional)
        * The dimension to reduce. Default is ``-1``.

    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` by removing the dimension specified
          in ``axis``, otherwise retain reduced axis with length ``1``.

    output_dtype: const<str> (Optional)
        * Possible values: ``uint16``, ``int32``.
        * If set, then value type inference will output using that dtype.
        * Default is ``int32``.

    Returns
    -------
    <\\*, U>

    Attributes
    ----------
    T: fp16, fp32, i32
    U: int32, uint16
    """

    def get_operator(self):
        return np.argmin
