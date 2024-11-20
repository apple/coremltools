#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math

import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import DefaultInputs, InputSpec, TensorInputType
from coremltools.converters.mil.mil.operation import SYMBOL, VALUE, Operation, precondition
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.types import nptype_from_builtin
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.mil.types.type_mapping import (
    string_to_builtin,
    string_to_nptype,
)


def _maintain_shape(x, y):
    # numpy converts rank 0 tensors to scalars
    if x.ndim == 0:
        # convert back to rank 0 tensor
        return np.array(y)
    return y


class elementwise_unary(Operation):
    """
    Elementwise Unary Op Superclass
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        return self.x.sym_type

class elementwise_unary_with_int(Operation):
    """
    Elementwise Unary Op Superclass
    """
    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def type_inference(self):
        return self.x.sym_type

"""
Elementwise unary op implementation(s)
"""

@register_op
class abs(elementwise_unary_with_int):
    """
    Return the absolute values of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.abs(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class acos(elementwise_unary):
    """
    Return the inverse cosine values of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.arccos(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class asin(elementwise_unary):
    """
    Return the inverse sine of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.arcsin(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class atan(elementwise_unary):
    """
    Return the inverse tangent of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.arctan(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class atanh(elementwise_unary):
    """
    Return the inverse hyperbolic tangent values of the input
    ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.arctanh(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class ceil(elementwise_unary):
    """
    Return the ceil values of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.ceil(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class clip(Operation):
    """
    Clip the values in the input ``x`` to ``[alpha, beta]``, element-wise.
    Any values less than ``alpha`` are set to ``alpha``, and any values greater
    than ``beta`` are set to ``beta``.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)
    alpha: const T (Required)
    beta: const T (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        alpha=TensorInputType(const=True, type_domain="T"),
        beta=TensorInputType(const=True, type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.minimum(np.maximum(self.x.val, self.alpha.val), self.beta.val)


@register_op
class cos(elementwise_unary):
    """
    Return cosine of ``x`` element-wise. Input domain is ``(-inf, inf)`` and
    output range is ``[-1,1]``.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.cos(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class cosh(elementwise_unary):
    """
    Return hyperbolic cosine of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.cosh(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class erf(elementwise_unary):
    """
    Return the gauss error function of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        erf_vector_function = np.vectorize(math.erf)
        return erf_vector_function(self.x.val)


@register_op
class exp(elementwise_unary):
    """
    Return e^x, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.exp(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class exp2(elementwise_unary_with_int):
    """
    Return 2^x, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.exp2(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class floor(elementwise_unary):
    """
    Return the floor of the input ``x``, element-wise, the same as rounding
    towards negative infinity.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.floor(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class inverse(Operation):
    """
    Return the reciprocal value of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)
    epsilon: const T (Optional, default=1e-4)
        * This is a small constant that is added to the input, before taking its
          inverse, for stability.
        * ``y = 1 / (x + epsilon)``.

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        epsilon=TensorInputType(const=True, optional=True, type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def default_inputs(self):
        return DefaultInputs(
            epsilon=nptype_from_builtin(self.x.dtype)(1e-4),
        )

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.array(np.reciprocal(self.x.val + self.epsilon.val), copy=False)


@register_op
class log(Operation):
    """
    Return the natural logarithm value of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)
    epsilon: const T (Optional, default=1e-45)
        * This is a small constant that is added to the input, before taking log.
        * ``y = log(x + epsilon)``.

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        epsilon=TensorInputType(const=True, optional=True, type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def default_inputs(self):
        return DefaultInputs(
            epsilon=nptype_from_builtin(self.x.dtype)(1e-45)
        )

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.log(self.x.val + self.epsilon.val)
        return _maintain_shape(self.x.val, result)


@register_op
class logical_not(Operation):
    """
    Return the value of NOT the input ``x``, element-wise. (``1`` for true, ``0``
    for false in numeric domain.) A numeric value ``t`` is evaluated to true
    ``iff t != 0``.

    Parameters
    ----------
    x: tensor<[\\*d], bool> (Required)

    Returns
    -------
    tensor<[\\*d], bool>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain=types.bool),
    )

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.logical_not(self.x.val)

    def type_inference(self):
        return self.x.sym_type


@register_op
class round(elementwise_unary):
    """
    Return the round value of the input ``x`` to nearest integer, element-wise.
    ``0.5`` is rounded to ``0``.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.round(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class rsqrt(Operation):
    """
    Return the reciprocal value of the square root of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)
    epsilon: const T (Optional, default=1e-12)
        * This is a small constant that is added to the input, before applying the
          ``rsqrt`` function, for stability.
        * ``y = 1 / sqrt(x + epsilon)``.

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        epsilon=TensorInputType(const=True, optional=True, type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def default_inputs(self):
        return DefaultInputs(
            epsilon=nptype_from_builtin(self.x.dtype)(1e-12),
        )

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        result = 1.0 / np.sqrt(self.x.val + self.epsilon.val)
        return _maintain_shape(self.x.val, result)


@register_op
class sign(elementwise_unary_with_int):
    """
    Return the sign value of the input ``x``, element-wise.

    All elements in the output will be either ``-1`` or ``1``, or zero if the input ``x`` is zero.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.sign(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class sin(elementwise_unary):
    """
    Return the sine value of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.sin(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class sinh(elementwise_unary):
    """
    Return the hyperbolic sine value of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.sinh(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class sqrt(elementwise_unary):
    """
    Returns the square root value of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.sqrt(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class square(elementwise_unary_with_int):
    """
    Return ``x^2``, element-wise.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.square(self.x.val)


@register_op
class tan(elementwise_unary):
    """
    Return the tangent value of the input ``x``, element-wise. Both input and output
    ranges are ``(-inf, inf)``.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.tan(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class tanh(elementwise_unary):
    """
    Return the hyperbolic tangent value of the input ``x``, element-wise. Both input
    and output ranges are ``(-inf, inf)`` while output range is ``[-1,  1]``.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.tanh(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op
class threshold(Operation):
    """
    Set a lower bound ``alpha`` to the values in the input ``x``, element-wise.
    Any values less than ``alpha`` are set to ``alpha``.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)
    alpha: const T (Required)

    Returns
    -------
    tensor<[\\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        alpha=TensorInputType(const=True, type_domain="T"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.maximum(self.x.val, self.alpha.val)


@register_op
class cast(Operation):
    """
    Cast the input ``x`` to the new type ``dtype``.

    Parameters
    ----------
    x: tensor<[\\*d], T> (Required)
    dtype: const str (Required)
        * Can be one of the following types: ``int32``, ``fp16``, ``fp32``, ``bool``.

    Returns
    -------
    tensor<[\\*d], dtype>
        * A tensor of the same shape as ``x``, with type ``dtype``.

    Attributes
    ----------
    T: i32, fp16, fp32, bool.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        dtype=TensorInputType(const=True, type_domain=types.str)
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def type_inference(self):
        if self.dtype.val not in self.supported_dtypes():
            raise NotImplementedError(
                "Parameter dtype of the cast operation can be one of the {}. "
                "Provided {}".format(self.supported_dtypes(), self.dtype.val)
            )

        if not types.is_tensor(self.x.sym_type):
            return string_to_builtin(self.dtype.val)

        ret_shape = self.x.shape
        return types.tensor(string_to_builtin(self.dtype.val), ret_shape)

    @precondition(allow=VALUE | SYMBOL)
    def value_inference(self):
        return self.get_cast_value(self.x, self.dtype.val)

    @classmethod
    def get_cast_value(cls, input_var, dtype_val):
        if dtype_val not in cls.supported_dtypes():
            raise NotImplementedError(
                "Parameter dtype of the cast operation can be one of the {}. "
                "Provided {}".format(cls.supported_dtypes(), dtype_val)
            )

        if input_var.val is None:
            if (
                input_var.sym_val is not None
                and not is_symbolic(input_var.sym_val)
                and len(input_var.sym_val.shape) == 1
            ):
                result = [
                    np.array(val).astype(dtype=string_to_nptype(dtype_val)).item()
                    if not is_symbolic(val)
                    else val
                    for val in input_var.sym_val
                ]
                return np.array(result)
            return None

        if hasattr(input_var.val, "astype"):
            return input_var.val.astype(dtype=string_to_nptype(dtype_val))
        else:
            return string_to_nptype(dtype_val)(input_var.val)
