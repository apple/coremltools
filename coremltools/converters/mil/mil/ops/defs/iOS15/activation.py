#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math
import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import (
    DefaultInputs,
    FloatInputType,
    InputSpec,
    IntInputType,
    ScalarOrTensorInputType,
    StringInputType,
    TensorInputType,
)
from coremltools.converters.mil.mil.operation import Operation, precondition, VALUE
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op

from .elementwise_unary import elementwise_unary


@register_op
class clamped_relu(Operation):
    """
    If ``x >= 0`` return elementwise ``min(beta, x)``, otherwise return
    ``min(beta, alpha * x)``.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
    alpha: const fp32 (Required)
    beta: const fp32 (Required)

    Returns
    -------
    tensor<\*?, T>
        * A tensor of the same type and shape as ``x``.

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

    @precondition(allow=VALUE)
    def value_inference(self):
        x = np.minimum(np.maximum(self.x.val, 0), self.beta.val)
        y = np.minimum(np.minimum(self.x.val, 0) * self.alpha.val, self.beta.val)
        return x + y

    def type_inference(self):
        return self.x.sym_type


@register_op
class elu(Operation):
    """
    If ``x > 0`` return elementwise ``x``, otherwise return ``alpha * (e^x - 1)``.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
    alpha: const fp32 (Optional)
        * Default is ``1``.

    Returns
    -------
    tensor<\*?, T>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(),
        alpha=FloatInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            alpha=1.,
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        b = np.copy(self.x.val)
        b[b < 0] = self.alpha.val * (np.exp(b[b < 0]) - 1)
        return b

    def type_inference(self):
        return self.x.sym_type


@register_op
class gelu(Operation):
    """
    Return the elementwise Gaussian error linear unit activation function for ``x``.
    
    You can use ``EXACT``, ``TANH_APPROXIMATION``, or ``SIGMOID_APPROXIMATION`` values
    based on the following formulas:
    
    * ``EXACT``:
    
    .. math::
       f(x) = 0.5x\\left ( 1+\\rm{erf}\\left ( \\frac{x}{\\sqrt{2}} \\right ) \\right )
    
    * ``TANH_APPROXIMATION``:
    
    .. math::
       f(x) = 0.5x\\left ( 1+\\rm{tanh}\\left ( \\sqrt{2/\\pi}\\left ( x + 0.044715x^3 \\right ) \\right ) \\right )
    
    * ``SIGMOID_APPROXIMATION``:
    
    .. math::
       f(x) = x*\\rm{sigmoid}(1.702x)

    
    Parameters
    ----------
    x: tensor<\*?, T> (Required)
    
    mode: const str (Optional)
        * Use ``'EXACT'``, ``'TANH_APPROXIMATION'``, or ``'SIGMOID_APPROXIMATION'`` for ``str``.
        * Default is ``'EXACT'``.

    Returns
    -------
    tensor<\*?, T>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(),
        mode=StringInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            mode="EXACT",
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        if self.mode.val == "TANH_APPROXIMATION":
            a = np.sqrt(2 / np.pi) * (self.x.val + 0.044715 * np.power(self.x.val, 3))
            return 0.5 * self.x.val * (1 + np.tanh(a))
        elif self.mode.val == "SIGMOID_APPROXIMATION":
            return self.x.val * (1 / (1 + np.exp(-(1.702 * self.x.val))))
        else:
            sqaure_root_of_2 = np.sqrt(2)
            vfunc = np.vectorize(lambda x: 0.5 * x * (1 + math.erf(x / sqaure_root_of_2)))
            return vfunc(self.x.val)

    def type_inference(self):
        allowed_values = {"EXACT", "TANH_APPROXIMATION", "SIGMOID_APPROXIMATION"}
        if self.mode.val not in allowed_values:
            msg = '"gelu" op: unrecognized value of mode: "{}". Allowed values are {}'
            raise ValueError(msg.format(self.mode.val, allowed_values))
        return self.x.sym_type


@register_op
class leaky_relu(Operation):
    """
    If ``x >= 0`` apply ``x`` elementwise, otherwise apply ``alpha * x`` elementwise.

    Parameters
    ----------
    x: <*?, T> (Required)
    alpha: const fp32 (Optional)
        * Default is ``0.01``.

    Returns
    -------
    tensor<\*?, fp32>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(),
        alpha=FloatInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            alpha=0.01,
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        b = np.copy(self.x.val)
        b[b < 0] *= self.alpha.val
        return b

    def type_inference(self):
        return self.x.sym_type


@register_op
class linear_activation(Operation):
    """
    Apply elementwise ``x * alpha + beta``.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
    alpha: const fp32 (Required)
    beta: const fp32 (Optional)
        * Default is ``0``.

    Returns
    -------
    tensor<\*?, T>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(),
        alpha=FloatInputType(const=True),
        beta=FloatInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            beta=0.,
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return self.alpha.val * self.x.val + self.beta.val

    def type_inference(self):
        return self.x.sym_type


@register_op
class prelu(Operation):
    """
    Where ``i = 1 ... C``, if ``x_i > 0``, return ``x_i`` , otherwise return ``alpha_i * x_i``.

    Parameters
    ----------
    x: tensor<[B, C, 1..3], T> (Required)
        * x must have rank 4 or rank 3 or rank 5, i.e. a shape of (B,C,H) or (B,C,H,W) or (B,C,D,H,W)
    alpha: const tensor<[C], T>, (Required)
        * The length of alpha must match the second dimension of x (channel dimension)

    Returns
    -------
    tensor<[B, C, 1..3], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp32, fp16
    """

    input_spec = InputSpec(
        x=TensorInputType(), 
        alpha=TensorInputType(const=True),)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        alpha_br = self.alpha.val
        for i in range(1, len(self.x.shape)):
            alpha_br = np.expand_dims(alpha_br, i)
        x_pos = np.maximum(self.x.val, 0)
        b = np.minimum(self.x.val, 0)
        return x_pos + b * alpha_br

    def type_inference(self):
        if self.x.rank not in (3, 4, 5):
            raise ValueError("prelu op: x must be rank 3 or 4 or 5, instead it is of rank {}".format(len(self.x.shape)))
        if len(self.alpha.val.shape) != 1:
            raise ValueError("alpha should be rank 1")
        if self.x.shape[1] != self.alpha.val.shape[0]:
            raise ValueError(
                "Size of dimension 1 of alpha should be the same as "
                + "the size of dimension 1 of x."
            )
        if self.x.rank in (3, 5):
            # check whether all alpha values are the same or not
            are_values_same = np.where(np.abs(self.alpha.val - self.alpha.val[0]) > 1e-5)[0].size == 0
            if not are_values_same:
                raise ValueError("prelu op: rank 3 or rank 5 input is only supported when all the values of alpha are same,"
                                 "which is not the case here")
        return self.x.sym_type


@register_op
class relu(elementwise_unary):
    """
    Return elementwise-applied rectified linear activation: ``min(x, 0)``.

    Parameters
    ----------
    x: tensor<\*?, fp32> (Required)

    Returns
    -------
    tensor<\*?, fp32>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.maximum(self.x.val, 0)


@register_op
class relu6(elementwise_unary):
    """
    Return elementwise-applied rectified linear activation: ``min(max(x, 0), 6)``.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)

    Returns
    -------
    tensor<\*?, T>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.minimum(np.maximum(self.x.val, 0), 6)


@register_op
class scaled_tanh(Operation):
    """
    Return ``alpha * tanh(beta * x)`` elementwise.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
        * Input range is ``(-inf, inf)``.
    alpha: const fp32 (Optional)
        * Default is ``1``.
    beta: const fp32 (Optional)
        * Default is ``1``.

    Returns
    -------
    tensor<\*?, fp32>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(),
        alpha=FloatInputType(const=True, optional=True),
        beta=FloatInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            alpha=1,
            beta=1,
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return self.alpha.val * np.tanh(self.x.val * self.beta.val)

    def type_inference(self):
        return self.x.sym_type


@register_op
class sigmoid(elementwise_unary):
    """
    Return ``sigmoid(x)`` elementwise.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)

    Returns
    -------
    tensor<\*?, T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return 1 / (1 + np.exp(-self.x.val))


@register_op
class sigmoid_hard(Operation):
    """
    Return ``min( max( alpha * x + beta, 0 ), 1 )`` elementwise.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
    alpha: const fp32 (Optional)
        * Default is ``0.2``.
    beta: const fp32 (Optional)
        * Default is ``0.5``.

    Returns
    -------
    tensor<\*?, fp32>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(),
        alpha=FloatInputType(const=True, optional=True),
        beta=FloatInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            alpha=0.2,
            beta=0.5,
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.minimum(
            np.maximum((self.alpha.val * self.x.val) + self.beta.val, 0), 1
        )

    def type_inference(self):
        return self.x.sym_type
        
        
@register_op()
class silu(Operation):
    """
    Sigmoid Linear Unit, elementwise apply the SiLU or Swish operation ``x * sigmoid(x)``.

    Parameters
    ----------
    x: tensor<\*, T>

    Returns
    -------
    tensor<\*, T>

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(x=TensorInputType(),)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def type_inference(self):
        return types.tensor(self.x.dtype, tuple(self.x.shape))


@register_op
class softplus(elementwise_unary):
    """
    Return ``log( 1 + e^x )`` elementwise.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)

    Returns
    -------
    tensor<\*?, T>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.log(1 + np.exp(-np.abs(self.x.val))) + np.maximum(self.x.val, 0)


@register_op
class softplus_parametric(Operation):
    """
    Return ``alpha_i * log( 1 + e^( beta_i * x_i ) )``, where ``i = 1 ... C``.

    Parameters
    ----------
    x: tensor<[b, C, n, m], T> (Required)
    alpha: const tensor<[C], fp32> (Required)
    beta: const tensor<[C], fp32> (Required)

    Returns
    -------
    tensor<[b, C, n, m], T>
        * A tensor of the same shape as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        alpha=TensorInputType(const=True),
        beta=TensorInputType(const=True),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        alpha_br = np.copy(self.alpha.val)
        beta_br = np.copy(self.beta.val)
        for i in range(1, len(self.x.val.shape)):
            alpha_br = np.expand_dims(alpha_br, i)
            beta_br = np.expand_dims(beta_br, i)
        return alpha_br * np.log(1 + np.exp(self.x.val * beta_br))

    def type_inference(self):
        if len(self.x.shape) < 3:
            raise ValueError("x should be at least rank 3")
        if len(self.alpha.val.shape) != 1:
            raise ValueError("alpha should be rank 1")
        if self.x.shape[1] != self.alpha.val.shape[0]:
            raise ValueError(
                "Size of dimension 0 of alpha should be the same as "
                + "the size of dimension 1 of x."
            )
        if len(self.beta.val.shape) != 1:
            raise ValueError("beta should be rank 1")
        if self.x.shape[1] != self.beta.val.shape[0]:
            raise ValueError(
                "Size of dimension 0 of beta should be the same as "
                + "the size of dimension 1 of x."
            )
        return self.x.sym_type


@register_op
class softmax(Operation):
    """
    Return ``exp(x) / tf.reduce_sum(tf.exp(x), axis)``.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
    axis: const i32 (Optional)
        * Default is ``-1``.

    Returns
    -------
    tensor<\*?, fp32>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        axis=IntInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            axis=-1,
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        x = self.x.val
        axis = self.axis.val
        max_vals = np.max(x, axis=axis, keepdims=True)
        temp = np.exp(x - max_vals)
        return temp / np.sum(temp, axis=axis, keepdims=True)

@register_op
class softsign(elementwise_unary):
    """
    Return ``x / ( 1 + |x| )`` applied elementwise.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)

    Returns
    -------
    tensor<\*?, T>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @precondition(allow=VALUE)
    def value_inference(self):
        return self.x.val / (1 + np.abs(self.x.val))


@register_op
class thresholded_relu(Operation):
    """
    Return ``x`` if ``x >= alpha``, otherwise return ``0``.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
    alpha: const fp32 (Optional)
        * Default is ``1``.

    Returns
    -------
    tensor<\*, T>
        * A tensor of the same shape and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(),
        alpha=FloatInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            alpha=1.,
            )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        y = self.x.val
        y[y < self.alpha.val] = 0
        return y
