#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from ._op_reqs import *
from coremltools.converters.mil.mil.types.symbolic import (
    any_symbolic,
)

@register_op(doc_str="")
class batch_norm(Operation):
    """
    Normalize input tensor ``x`` by ``mean`` and ``variance``, and optionally apply a
    scale ``gamma`` and an offset ``beta``:
    
    .. math::
       y_i = \\gamma_i \\dfrac{ (x_i - mean_i)}{\\sqrt{variance_i + epsilon}} + beta_i \\;,\\;i=1,....,C
    
    The ``mean``, ``variance``, ``gamma``, and ``beta``
    must be 1-D tensors whose lengths are equal to the second axis (the "depth"
    or "channel" dimension) of ``x``.
    
    Parameters
    ----------
    x: tensor<[n,C,*D], T> (Required)
        * ``3 <= rank <= 4``.
        * ``*D`` refers to the spatial dimensions, ``1 <= rank(*D) <= 2``.
        * ``n`` is the batch dimension.
    mean: const tensor<[C], T> (Required)
    variance: const tensor<[C], T> (Required)
    gamma: const tensor<[C], T> (Optional)
        * Optional scale applied to normalized tensor.
        * Default is all ones.
    beta: const tensor<[C], T> (Optional)
        * Optional offset applied to normalized tensor.
        * Default is all zeros.
    epsilon: const fp32 (Optional)
        * Default is ``1e-5``.
    
    Returns
    -------
    tensor<[n,C,*D], T>
        * Output tensor has the same shape and type as the input ``x``.

    Attributes
    ----------
    T: fp32
    """
    
    input_spec = InputSpec(
        x=TensorInputType(),
        mean=TensorInputType(const=True),
        variance=TensorInputType(const=True),
        gamma=TensorInputType(const=True, optional=True),
        beta=TensorInputType(const=True, optional=True),
        epsilon=FloatInputType(const=True, default=1e-5),
    )

    def __init__(self, **kwargs):
        super(batch_norm, self).__init__(**kwargs)

    def type_inference(self):
        x_shape = self.x.shape
        return types.tensor(types.fp32, tuple(x_shape))


@register_op(doc_str="")
class instance_norm(Operation):
    """
    Apply instance normalization to the n-dimensional input tensor.
    
    Parameters
    ----------
    x: tensor<[n,C,*D], T>  (Required)
        * ``3 <= rank(x) <= 4``.
        * ``*D`` refers to the spatial dimensions, ``1 <= rank(*D) <= 2``.
        * ``n`` is the batch dimension.
    gamma: const tensor<[C], T> (Optional)
        * Optional scale applied to normalized tensor.
        * Default to all ones.
    beta: const tensor<[C], T> (Optional)
        * Optional offset applied to normalized tensor.
        * Default to all zeros.
    epsilon: const f32 (Optional)
        * Default to ``1e-5``.
    
    Returns
    -------
    tensor<[n,C,*D], T>
        * Output tensor has the same shape and type as the input ``x``.
    """
    
    input_spec = InputSpec(
        x=TensorInputType(),
        gamma=TensorInputType(const=True, optional=True),
        beta=TensorInputType(const=True, optional=True),
        epsilon=FloatInputType(const=True, default=1e-5),
    )

    def __init__(self, **kwargs):
        super(instance_norm, self).__init__(**kwargs)

    def type_inference(self):
        x_shape = self.x.shape
        return types.tensor(types.fp32, tuple(x_shape))


@register_op(doc_str="")
class l2_norm(Operation):
    """
    Apply L2 normalization to the n-dimensional input tensor. That is, divide the input
    tensor by the square root of the sum of squares of all elements of the input.
    
    .. math::
       x_i \\leftarrow \\dfrac{x_i}{\\sqrt{\\sum{x_i^2} + \\epsilon}}
    
    
    Parameters
    ----------
    x: tensor<[*D,C,H,W], T> (Required)
        * Input tensor, ``rank(x) >= 3``.
        * ``*D`` refers to the spatial dimensions, ``rank(*D) >= 0``.
        * ``n`` is the batch dimension.
        * For ranks greater than 3, the leading dimensions, starting from ``0`` to ``-4`` (inclusive),
          are all treated as batch.
    epsilon: const fp32 (Optional)
        * Small constant to avoid division by ``0``.
        * Optional, defaults to ``1e-6``.

    Returns
    -------
    tensor<[*D,C,H,W], T>
        * Same type and shape as the input tensor ``x``.
    
    Attributes
    ----------
    T: fp32
    """
    
    input_spec = InputSpec(
        x=TensorInputType(),
        epsilon=FloatInputType(const=True, default=1e-6),
    )

    def __init__(self, **kwargs):
        super(l2_norm, self).__init__(**kwargs)

    def type_inference(self):
        x_shape = self.x.shape
        return types.tensor(types.fp32, tuple(x_shape))


@register_op(doc_str="")
class layer_norm(Operation):
    """
    Apply layer normalization to the n-dimensional input tensor:
    
    .. math::
       out = gamma * (input - E[x]) / sqrt(Var[x] + epsilon) + beta
    
    
    Parameters
    ----------
    x: tensor<*?, T> (Required)
        * Input tensor.
    axes: const<[K], i32> (Optional)
        * Dimensions to perform layer normalization.
        * Default is ``None`` (all dimensions).
    gamma: const tensor<[K], T> (Optional)
        * if provided, the shape must be be ``x.shape[axes]``,
        *  for instance, if with input ``x`` with shape ``(3,4,5,6)`` and ``axes = [2,3]``,
          gamma must have shape ``(5,6)``.
        * Default is all ones.
    beta: const tensor<[K], T> (Optional)
        * Same shape as gamma.
        * Default is all zeros.
    epsilon: const fp32 (Optional)
        * Small constant to avoid division by ``0``.
        * Default is ``1e-5``.
    
    Returns
    -------
    tensor<*?, T>:
     * Tensor with same shape and type as the input tensor ``x``.

    Attributes
    ----------
    T: fp32
    """
    
    input_spec = InputSpec(
        x=TensorInputType(),
        axes=IntTensorInputType(const=True, optional=True),
        gamma=TensorInputType(const=True, optional=True),
        beta=TensorInputType(const=True, optional=True),
        epsilon=FloatInputType(const=True, default=1e-5),
    )

    def __init__(self, **kwargs):
        super(layer_norm, self).__init__(**kwargs)

    @staticmethod
    def _is_compatible_shape(shapea, shapeb):
        if not len(shapea) == len(shapeb):
            return False
        for a,b in zip(shapea, shapeb):
            if any_symbolic([a,b]):
                continue
            if a != b:
                return False
        return True

    def type_inference(self):
        rank = self.x.rank

        # check valid axes
        positive_axes = [axis + rank if axis < 0 else axis for axis in self.axes.val]
        if not all([axis >= 0 and axis < rank for axis in positive_axes]):
            raise ValueError("axes must in the range of [-x.rank, x.rank-1].")

        # check shape of gamma and beta
        normalized_shape = [self.x.shape[i] for i in range(rank) if i in positive_axes]
        if self.gamma is not None and not layer_norm._is_compatible_shape(list(self.gamma.shape), normalized_shape):
            raise ValueError("Expect shape {} for gamma, but get shape {} instead".format(normalized_shape, self.gamma.shape))

        if self.beta is not None and not layer_norm._is_compatible_shape(list(self.gamma.shape), normalized_shape):
            raise ValueError("Expect shape {} for beta, but get shape {} instead".format(normalized_shape, self.beta.shape))

        x_shape = self.x.shape
        return types.tensor(types.fp32, tuple(x_shape))


    @precondition(allow=VALUE)
    def value_inference(self):
        def np_layer_norm(x, axes, gamma, beta, epsilon=1e-5):
            rank = len(x.shape)
            axes = [axis + rank if axis < 0 else axis for axis in axes]
            normalized_shape = [x.shape[i] if i in axes else 1 for i in range(rank)]
            gamma = np.ones(shape=normalized_shape) if gamma is None else np.reshape(gamma, normalized_shape)
            beta = np.zeros(shape=normalized_shape) if beta is None else np.reshape(beta, normalized_shape)
            num = x - np.mean(x, axis=tuple(axes), keepdims=True)
            dem = np.sqrt(
                np.sum(np.square(num), axis=tuple(axes), keepdims=True)
                / np.prod(normalized_shape)
                + epsilon
            )
            return num / dem * gamma + beta

        _axes = self.x.shape if self.axes is None else self.axes.val
        _gamma = None if self.gamma is None else self.gamma.val
        _beta = None if self.beta is None else self.beta.val
        return np_layer_norm(self.x.val, _axes, _gamma, _beta, self.epsilon.val)


@register_op(doc_str="")
class local_response_norm(Operation):
    """
    Apply local response normalization to the n-dimensional input tensor:
    
    .. math::
       x_i \\leftarrow \\dfrac{x_i}{\\left ( k + \\dfrac{\\alpha}{C} \\sum_j x_j^2 \\right )^\\beta}
    
    
    Parameters
    ----------
    x: tensor<[n,C,*D], T> (Required)
        * Input tensor, ``3 <= rank(x) <= 4``.
        * ``*D`` refers to the spatial dimensions, ``1 <= rank(*D) <= 2``.
        * ``n`` is the batch dimension.
    size: const i32 (Required)
        * Amount of neighboring channels to normalize.
    alpha: const fp32 (Optional)
        * Scale factor.
        * Default is ``1e-4``.
    beta: const fp32 (Optional)
        * An exponent.
        * Default is ``0.75``.
    k: const fp32 (Optional)
        * Additive factor.
        * Default is ``1.0``.
    
    Returns
    -------
    tensor<[n,C,*D], T>
        * Same type and shape as the input tensor ``x``.
    
    Attributes
    ----------
    T: fp32
    """
    
    input_spec = InputSpec(
        x=TensorInputType(),
        size=IntInputType(const=True),
        alpha=FloatInputType(const=True, default=1e-4),
        beta=FloatInputType(const=True, default=0.75),
        k=FloatInputType(const=True, default=1.0),
    )

    def __init__(self, **kwargs):
        super(local_response_norm, self).__init__(**kwargs)

    def type_inference(self):
        x_shape = self.x.shape
        return types.tensor(types.fp32, tuple(x_shape))
