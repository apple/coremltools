#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math
import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.operation import Operation, precondition, SYMBOL, VALUE
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.input_type import (
    DefaultInputs,
    FloatInputType,
    InputSpec,
    ScalarOrTensorInputType,
    StringInputType
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
    input_spec = InputSpec(x=ScalarOrTensorInputType(),)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

"""
Elementwise unary op implementation(s)
"""

@register_op(doc_str="")
class abs(elementwise_unary):
    """
    Return the absolute values of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.abs(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class acos(elementwise_unary):
    """
    Return the inverse cosine values of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.arccos(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class asin(elementwise_unary):
    """
    Return the inverse sine of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.arcsin(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class atan(elementwise_unary):
    """
    Return the inverse tangent of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.arctan(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class atanh(elementwise_unary):
    """
    Return the inverse hyperbolic tangent values of the input
    ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.arctanh(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class ceil(elementwise_unary):
    """
    Return the ceil values of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.ceil(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class clip(Operation):
    """
    Clip the values in the input ``x`` to ``[alpha, beta]``, element-wise.
    Any values less than ``alpha`` are set to ``alpha``, and any values greater
    than ``beta`` are set to ``beta``.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)
    alpha: const f32 (Required)
    beta: const f32 (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(),
        alpha=FloatInputType(const=True),
        beta=FloatInputType(const=True),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.minimum(np.maximum(self.x.val, self.alpha.val), self.beta.val)


@register_op(doc_str="")
class cos(elementwise_unary):
    """
    Return cosine of ``x`` element-wise. Input domain is ``(-inf, inf)`` and
    output range is ``[-1,1]``.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], T>

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.cos(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class cosh(elementwise_unary):
    """
    Return hyperbolic cosine of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.cosh(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class erf(elementwise_unary):
    """
    Return the gauss error function of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        erf_vector_function = np.vectorize(math.erf)
        return erf_vector_function(self.x.val)


@register_op(doc_str="")
class exp(elementwise_unary):
    """
    Return e^x, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.exp(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class exp2(elementwise_unary):
    """
    Return 2^x, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.exp2(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class floor(elementwise_unary):
    """
    Return the floor of the input ``x``, element-wise, the same as rounding
    towards negative infinity.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.floor(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class inverse(Operation):
    """
    Return the reciprocal value of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)
    epsilon: const fp32 (Optional, default=1e-4)
        * This is a small constant that is added to the input, before taking its
          inverse, for stability.
        * ``y = 1 / (x + epsilon)``.

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(),
        epsilon=FloatInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            epsilon=1e-4,
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.reciprocal(self.x.val + self.epsilon.val)


@register_op(doc_str="")
class log(Operation):
    """
    Return the natural logarithm value of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)
    epsilon: const fp32 (Optional, default=1e-45)
        * This is a small constant that is added to the input, before taking log.
        * ``y = log(x + epsilon)``.

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(),
        epsilon=FloatInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            epsilon=1e-45)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.log(self.x.val + self.epsilon.val)


@register_op(doc_str="")
class logical_not(elementwise_unary):
    """
    Return the value of NOT the input ``x``, element-wise. (``1`` for true, ``0``
    for false in numeric domain.) A numeric value ``t`` is evaluated to true
    ``iff t != 0``.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: bool
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.logical_not(self.x.val)


@register_op(doc_str="")
class round(elementwise_unary):
    """
    Return the round value of the input ``x`` to nearest integer, element-wise.
    ``0.5`` is rounded to ``0``.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.round(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class rsqrt(Operation):
    """
    Return the reciprocal value of the square root of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)
    epsilon: const fp32 (Optional, default=1e-12)
        * This is a small constant that is added to the input, before applying the
          ``rsqrt`` function, for stability.
        * ``y = 1 / sqrt(x + epsilon)``.

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(),
        epsilon=FloatInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            epsilon=1e-12,
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        result = 1.0 / np.sqrt(self.x.val + self.epsilon.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class sign(elementwise_unary):
    """
    Return the sign value of the input ``x``, element-wise.

    All elements in the output will be either ``-1``. or ``1``.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.sign(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class sin(elementwise_unary):
    """
    Return the sine value of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.sin(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class sinh(elementwise_unary):
    """
    Return the hyperbolic sine value of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.sinh(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class sqrt(elementwise_unary):
    """
    Returns the square root value of the input ``x``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.sqrt(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class square(elementwise_unary):
    """
    Return ``x^2``, element-wise.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.square(self.x.val)


@register_op(doc_str="")
class tan(elementwise_unary):
    """
    Return the tangent value of the input ``x``, element-wise. Both input and output
    ranges are ``(-inf, inf)``.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.tan(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class tanh(elementwise_unary):
    """
    Return the hyperbolic tangent value of the input ``x``, element-wise. Both input
    and output ranges are ``(-inf, inf)`` while output range is ``[-1,  1]``.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        result = np.tanh(self.x.val)
        return _maintain_shape(self.x.val, result)


@register_op(doc_str="")
class threshold(Operation):
    """
    Set a lower bound ``alpha`` to the values in the input ``x``, element-wise.
    Any values less than ``alpha`` are set to ``alpha``.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)
    alpha: const fp32 (Required)

    Returns
    -------
    tensor<[\*d], f32>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(), alpha=FloatInputType(const=True),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.maximum(self.x.val, self.alpha.val)


@register_op(doc_str="")
class cast(Operation):
    """
    Cast the input ``x`` to the new type ``dtype``.

    Parameters
    ----------
    x: tensor<[\*d], T> (Required)
    dtype: const str (Required)
        * Can be one of the following types: ``int32``, ``int64``, ``fp32``, ``fp64``.

    Returns
    -------
    tensor<[\*d], dtype>
        * A tensor of the same shape as ``x``, with type ``dtype``.

    Attributes
    ----------
    T: i32, i64, fp16, fp32, fp64, bool.
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(), dtype=StringInputType(const=True)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def type_inference(self):
        type_map = {
            "int32": types.int32,
            "int64": types.int64,
            "fp16": types.fp16,
            "fp32": types.fp32,
            "fp64": types.fp64,
            "bool": types.bool,
        }

        if self.dtype.val not in type_map.keys():
            raise NotImplementedError(
                "Parameter dtype of the cast operation can be one of the {}. "
                "Provided {}".format(type_map.keys(), self.dtype.val)
            )

        if not types.is_tensor(self.x.sym_type):
            return type_map[self.dtype.val]

        ret_shape = self.x.shape
        return types.tensor(type_map[self.dtype.val], ret_shape)

    @precondition(allow=VALUE | SYMBOL)
    def value_inference(self):
        return self.get_cast_value(self.x, self.dtype.val)

    @staticmethod
    def get_cast_value(input_var, dtype_val):
        type_map = {
            "int32": np.int32,
            "int64": np.int64,
            "fp16": np.float16,
            "fp32": np.float32,
            "fp64": np.float64,
            "bool": np.bool,
        }

        if dtype_val not in type_map.keys():
            raise NotImplementedError(
                "Parameter dtype of the cast operation can be one of the {}. "
                "Provided {}".format(type_map.keys(), dtype_val)
            )

        if input_var.val is None:
            if input_var.sym_val is not None and not is_symbolic(input_var.sym_val) and len(input_var.sym_val.shape) == 1:
                result = [np.array(val).astype(dtype=type_map[dtype_val]).item() if not is_symbolic(val) else val for val in input_var.sym_val]
                return np.array(result)
            return None

        if not types.is_tensor(input_var.sym_type):
            return input_var.val.astype(dtype=type_map[dtype_val])
        else:
            return np.array(input_var.val).astype(dtype=type_map[dtype_val])
