#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import numpy as np

from ._op_reqs import register_op
from coremltools.converters.mil.mil import (
    Operation,
    precondition,
    types,
    VALUE
)
from coremltools.converters.mil.mil.input_type import (
    BoolInputType,
    DefaultInputs,
    InputSpec,
    IntInputType,
    IntTensorInputType,
    TensorInputType
)


class ReductionAxes(Operation):
    """
    Reduction Op Superclasses
    """
    input_spec = InputSpec(
        x=TensorInputType(),
        axes=IntTensorInputType(const=True, optional=True),
        keep_dims=BoolInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            axes=None,
            keep_dims=False,
            )

    def __init__(self, **kwargs):
        super(ReductionAxes, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = self.x.shape
        axes = self.axes.val if self.axes is not None else None
        if axes is None:
            axes = range(self.x.rank)
        keep_dims = self.keep_dims.val

        reduced_shape = list(x_shape)
        if keep_dims:
            for i in axes:
                reduced_shape[i] = 1
        else:
            # sort reverse so we can delete shape elements back to front
            axes = [axis if axis >= 0 else axis + len(reduced_shape) for axis in axes]
            for i in sorted(axes)[::-1]:
                reduced_shape.pop(i)
        if len(reduced_shape) == 0:
            return x_type  # scalar

        return types.tensor(x_type, tuple(reduced_shape))

    @precondition(allow=VALUE)
    def value_inference(self):
        axes = tuple(self.axes.val) if self.axes is not None else None
        return self.get_operator()(self.x.val, axis=axes, keepdims=self.keep_dims.val)

    def get_operator(self):
        raise NotImplementedError()


class ReductionAxis(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        axis=IntInputType(const=True, optional=True),
        keep_dims=BoolInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            axis=-1,
            keep_dims=False,
            )

    def __init__(self, **kwargs):
        super(ReductionAxis, self).__init__(**kwargs)

    def _find_reduced_shape(self):
        x_shape = self.x.shape
        axis = self.axis.val

        reduced_shape = list(x_shape)
        axis = axis if axis >= 0 else axis + len(reduced_shape)
        if self.keep_dims.val:
            reduced_shape[axis] = 1
        else:
            reduced_shape.pop(axis)
        return reduced_shape

    def type_inference(self):
        x_type = self.x.dtype
        reduced_shape = self._find_reduced_shape_and_axis()
        return types.tensor(x_type, tuple(reduced_shape))

    @precondition(allow=VALUE)
    def value_inference(self):
        tmp = self.get_operator()(self.x.val, axis=self.axis.val)
        reduced_shape = self._find_reduced_shape()
        if self.keep_dims.val:
            tmp = np.reshape(tmp, reduced_shape)
        return tmp

    def get_operator(self):
        raise NotImplementedError()


"""
Reduction op implementations
"""


@register_op(doc_str="TODO")
class reduce_arg(ReductionAxis):
    def __init__(self, **kwargs):
        super(reduce_arg, self).__init__(**kwargs)

    def type_inference(self):
        x_shape = self.x.shape
        axis = self.axis.val

        reduced_shape = list(x_shape)
        axis = axis if axis >= 0 else axis + len(reduced_shape)
        if self.keep_dims.val:
            reduced_shape[axis] = 1
        else:
            reduced_shape.pop(axis)

        return types.tensor(types.int32, tuple(reduced_shape))


@register_op(doc_str="")
class reduce_argmax(reduce_arg):
    """
    Computes the indices of the maximum value across dimensions of a tensor.
    In case of ties, the identity of the return value is not guaranteed.
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Must be 1-dimensional or higher.
    
    axis: const<i32> (Optional)
        * The dimension to reduce. Default is ``-1``.
    
    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` by removing the dimension
          specified in ``axis``. If ``True``, retain reduced axis with length ``1``.
    
    Returns
    -------
    <\*, int32>
    
    Attributes
    ----------
    T: f32, int32
    
    References
    ----------
    See `tf.math.argmax <https://www.tensorflow.org/api_docs/python/tf/math/argmax>`_.
    
    """
    
    def __init__(self, **kwargs):
        super(reduce_argmax, self).__init__(**kwargs)

    def get_operator(self):
        return np.argmax


@register_op(doc_str="")
class reduce_argmin(reduce_arg):
    """
    Computes the indices of the minimum value across dimensions of a tensor.
    In case of ties, the identity of the return value is not guaranteed.
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Must be 1-dimensional or higher.
    
    axis: const<i32> (Optional)
        * The dimension to reduce. Default is ``-1``.
    
    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` by removing the dimension specified
          in ``axis``, otherwise retain reduced axis with length ``1``.
    
    Returns
    -------
    <\*, int32>
    
    Attributes
    ----------
    T: f32, int32
    
    References
    ----------
    See `tf.math.argmin <https://www.tensorflow.org/api_docs/python/tf/math/argmin>`_.
    
    """
    
    def __init__(self, **kwargs):
        super(reduce_argmin, self).__init__(**kwargs)

    def get_operator(self):
        return np.argmin


@register_op(doc_str="")
class reduce_l1_norm(ReductionAxes):
    """
    Computes the L1 normalization of elements across given dimensions of the input tensor.
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Must be 1-dimensional or higher.
    
    axes: const<K,i32> (Optional, default="None", reduce on all axes.)
        * The dimensions to reduce.
    
    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` for each entry in ``axes``,
          otherwise retain reduced axes with length ``1``.
    
    Returns
    -------
    <\*,T>
        * Scalar or tensor: The reduced tensor.
    
    Attributes
    ----------
    T: f32, int32
    
    References
    ----------
    See `reduce_mean <https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean?version=stable>`_.
    
    """
    
    def __init__(self, **kwargs):
        super(reduce_l1_norm, self).__init__(**kwargs)

    def get_operator(self):
        def l1_norm(x, axis=None, keepdims=False):
            return np.sum(np.abs(x), axis=axis, keepdims=keepdims)

        return l1_norm


@register_op(doc_str="")
class reduce_l2_norm(ReductionAxes):
    """
    Computes the L2 normalization of elements across given dimensions of the input tensor.
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Must be 1-dimensional or higher.
    
    axes: const<K,i32> (Optional, default="None", reduce on all axes.)
        * The dimensions to reduce.
    
    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` for each entry in ``axes``,
          otherwise retain reduced axes with length ``1``.
    
    Returns
    -------
    <\*,T>
        * Scalar or tensor: The reduced tensor.
    
    Attributes
    ----------
    T: f32, int32
    
    """
    
    def __init__(self, **kwargs):
        super(reduce_l2_norm, self).__init__(**kwargs)

    def get_operator(self):
        def l2_norm(x, axis=None, keepdims=False):
            return np.sqrt(np.sum(np.square(x), axis=axis, keepdims=keepdims))

        return l2_norm


@register_op(doc_str="")
class reduce_log_sum(ReductionAxes):
    """
    Computes the natural logarithm of the sum of all the elements across given dimensions
    of the input tensor.
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Must be 1-dimensional or higher.
    
    axes: const<K,i32> (Optional, default="None", reduce on all axes.)
        * The dimensions to reduce.
    
    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` for each entry in ``axes``,
          otherwise retain reduced axes with length ``1``.
    
    Returns
    -------
    <\*,T>
        * Scalar or tensor: The reduced tensor.
    
    Attributes
    ----------
    T: f32, int32
    
    """
    
    def __init__(self, **kwargs):
        super(reduce_log_sum, self).__init__(**kwargs)

    def get_operator(self):
        def log_sum(x, axis=None, keepdims=False):
            return np.log(np.sum(x, axis=axis, keepdims=keepdims))

        return log_sum


@register_op(doc_str="")
class reduce_log_sum_exp(ReductionAxes):
    """
    Computes the natural logarithm of the sum of the exponentials of the elements across
    given dimensions of the input tensor. It is a smooth approximation of the maximum
    function, more numerically stable than ``log(sum(exp(input)))``. It avoids
    overflows caused by taking the ``exp`` of large inputs and underflows caused by
    taking the ``log`` of small inputs.
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Must be 1-dimensional or higher.
    
    axes: const<K,i32> (Optional, default="None", reduce on all axes.)
        * The dimensions to reduce.
    
    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` for each entry in ``axes``,
          otherwise retain reduced axes with length ``1``.
    
    Returns
    -------
    <\*,T>
        * Scalar or tensor: The reduced tensor.
    
    Attributes
    ----------
    T: f32, int32
    
    References
    ----------
    See `tf.math.reduce_logsumexp <https://www.tensorflow.org/api_docs/python/tf/math/reduce_logsumexp>`_.
    
    """
    
    def __init__(self, **kwargs):
        super(reduce_log_sum_exp, self).__init__(**kwargs)

    def get_operator(self):
        def operator(a, axis=None, keepdims=False):
            max_values = np.amax(a, axis=axis, keepdims=True)
            temp = np.exp(a - max_values)

            if not keepdims:
                max_values = np.squeeze(max_values, axis=axis)

            sum = np.sum(temp, axis=axis, keepdims=keepdims)
            result = np.log(sum)
            return result + max_values

        return operator


@register_op(doc_str="")
class reduce_max(ReductionAxes):
    """
    Computes the maximum of elements across given dimensions of the input tensor.
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Must be 1-dimensional or higher.
    
    axes: const<K,i32> (Optional, default="None", reduce on all axes.)
        * The dimensions to reduce.
    
    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` for each entry in ``axes``,
          otherwise retain reduced axes with length ``1``.
    
    Returns
    -------
    <\*,T>
        * Scalar or tensor: The reduced tensor.
    
    Attributes
    ----------
    T: f32, int32
    
    """
    
    def __init__(self, **kwargs):
        super(reduce_max, self).__init__(**kwargs)

    def get_operator(self):
        return np.max


@register_op(doc_str="")
class reduce_mean(ReductionAxes):
    """
    Computes the mean of elements across given dimensions of the input tensor.
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Must be 1-dimensional or higher.
    
    axes: const<K,i32> (Optional, default="None", reduce on all axes.)
        * The dimensions to reduce.
    
    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` for each entry in ``axes``,
          otherwise retain reduced axes with length ``1``.
    
    Returns
    -------
    <\*,T>
        * Scalar or tensor: The reduced tensor.
    
    Attributes
    ----------
    T: f32, int32
    
    References
    ----------
    For an example, see `tf.math.reduce_mean <https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean?version=stable>`_.
    """
    
    def __init__(self, **kwargs):
        super(reduce_mean, self).__init__(**kwargs)

    def get_operator(self):
        return np.mean


@register_op(doc_str="")
class reduce_min(ReductionAxes):
    """
    Computes the minimum of elements across given dimensions of the input tensor.
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Must be 1-dimensional or higher.
    
    axes: const<K,i32> (Optional, default="None", reduce on all axes.)
        * The dimensions to reduce.
    
    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` for each entry in ``axes``,
          otherwise retain reduced axes with length ``1``.
    
    Returns
    -------
    <\*,T>
        * Scalar or tensor: The reduced tensor.
    
    Attributes
    ----------
    T: f32, int32
    
    """
    
    def __init__(self, **kwargs):
        super(reduce_min, self).__init__(**kwargs)

    def get_operator(self):
        return np.min


@register_op(doc_str="")
class reduce_prod(ReductionAxes):
    """
    Computes the product of elements across given dimensions of the input tensor.
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Must be 1-dimensional or higher.
    
    axes: const<K,i32> (Optional, default="None", reduce on all axes.)
        * The dimensions to reduce.
    
    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` for each entry in ``axes``,
          otherwise retain reduced axes with length ``1``.
    
    Returns
    -------
    <\*,T>
        * Scalar or tensor: The reduced tensor.
    
    Attributes
    ----------
    T: f32, int32
    
    """
    
    def __init__(self, **kwargs):
        super(reduce_prod, self).__init__(**kwargs)

    def get_operator(self):
        return np.prod


@register_op(doc_str="")
class reduce_sum(ReductionAxes):
    """
    Computes the sum of elements across given dimensions of the input tensor.
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Must be 1-dimensional or higher.
    
    axes: const<K,i32> (Optional, default="None", reduce on all axes.)
        * The dimensions to reduce.
    
    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` for each entry in ``axes``,
          otherwise retain reduced axes with length ``1``.
    
    Returns
    -------
    <\*,T>
        * Scalar or tensor: The reduced tensor.
    
    Attributes
    ----------
    T: f32, int32
    
    """
    
    def __init__(self, **kwargs):
        super(reduce_sum, self).__init__(**kwargs)

    def get_operator(self):
        return np.sum


@register_op(doc_str="")
class reduce_sum_square(ReductionAxes):
    """
    Computes the sum of squares of elements across given dimensions of the input tensor.
    
    Parameters
    ----------
    x: <\*,T> (Required)
        * Must be 1-dimensional or higher.
    
    axes: const<K,i32> (Optional, default="None", reduce on all axes.)
        * The dimensions to reduce.
    
    keep_dims: const<bool> (Optional, default=False)
        * If ``False``, the rank is reduced by ``1`` for each entry in ``axes``,
          otherwise retain reduced axes with length ``1``.
    
    Returns
    -------
    <\*,T>
        * Scalar or tensor: The reduced tensor.
    
    Attributes
    ----------
    T: f32, int32
    
    """
    
    def __init__(self, **kwargs):
        super(reduce_sum_square, self).__init__(**kwargs)

    def get_operator(self):
        def sum_squre(x, axis=None, keepdims=False):
            return np.sum(np.square(x), axis=axis, keepdims=keepdims)

        return sum_squre
