#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import DefaultInputs, InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.scatter_gather import scatter as _scatter_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.scatter_gather import (
    scatter_along_axis as _scatter_along_axis_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.scatter_gather import (
    scatter_nd as _scatter_nd_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS16.scatter_gather import gather as _gather_iOS16
from coremltools.converters.mil.mil.ops.defs.iOS16.scatter_gather import (
    gather_along_axis as _gather_along_axis_iOS16,
)
from coremltools.converters.mil.mil.ops.defs.iOS16.scatter_gather import (
    gather_nd as _gather_nd_iOS16,
)
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


@register_op(opset_version=_IOS17_TARGET)
class scatter(_scatter_iOS15):
    """
    Scatter ``updates`` to ``data`` at locations ``indices`` at dimension ``axis``
    by the operation ``mode``.

    This section documents only the differences between this version and the
    iOS 15 :py:class:`~.iOS15.scatter_gather.scatter`. The major differences are as follows:

    - Input parameter ``indices`` now supports only positive values -- negative values
      are considered out-of-bound. If support for negative indices is required, they must be
      explicitly converted to positive values using the following::

        index = iOS17.select(index >= 0, index, index + max_index)

    - New input parameter called ``validate_indices`` has been added to all scatter ops. Its behavior is as follows:
       - If ``True``, it raises a runtime (possibly also a compile-time) exception
         for out-of-bound values of the ``indices`` parameter.
       - If ``False``, absolutely no checking is performed for out-of-bound values
         of ``indices`` either at compile or runtime. Behavior for out-of-bound indices
         is undefined but memory safe.


    Parameters
    ----------
    data: tensor<\\*D, T> (Required)
    indices: tensor<[C], i32> (Required)
        * 1-D tensor.
    updates: tensor<\\*K, T> (Required)
        * ``K = data.shape[:axis] + [len(indices)] + data.shape[axis+1:]``.
    axis: const i32 (Optional)
        * Default to ``0``.
    mode: const string (Optional)
        * Can be the following modes: ``add``, ``div``, ``max``, ``min``, ``mul``, ``sub``, ``update``.
        * Default value is ``update``.
    validate_indices: const bool (Optional)
        * If ``True``, it raises a runtime (possibly also a compile-time) exception
          for out-of-bound values of the ``indices`` parameter.
        * If ``False``, absolutely no checking is performed for out-of-bound values
          of ``indices`` either at compile or runtime. Behavior for out-of-bound indices
          is undefined but memory safe.
        * Default value is ``False``.

    Returns
    -------
    tensor<\\*D, T>
        * With the same type and shape as input ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
        indices=TensorInputType(type_domain=types.int32),
        updates=TensorInputType(type_domain="T"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        mode=TensorInputType(const=True, optional=True, type_domain=types.str),
        validate_indices=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    def default_inputs(self):
        return DefaultInputs(
            axis=0,
            mode="add",
            validate_indices=False,
        )

    def type_inference(self):
        result = super().type_inference()
        if self.validate_indices.val:
            indices = self.indices.val
            if indices is not None:
                if np.count_nonzero(
                    np.logical_or(indices < 0, indices >= self.data.shape[self.axis.val])
                ):
                    raise IndexError(
                        f"Indices is out of bounds for `{self.op_type}` node {self.name}. "
                        f"Expected indices between [0, {self.data.shape[self.axis.val]}), but got {indices}."
                    )
        return result


@register_op(opset_version=_IOS17_TARGET)
class scatter_along_axis(_scatter_along_axis_iOS15):
    """
    Scatter ``updates`` to ``data`` at locations ``indices`` along ``axis`` dimension
    using the ``mode`` operation.

    The major differences from the previous version are illustrated in :py:class:`scatter`.
    For more information, see the iOS 15 :py:class:`~.iOS15.scatter_gather.scatter_along_axis`.

    Parameters
    ----------
    data: tensor<\\*D, T> (Required)
    indices: tensor<\\*K, i32> (Required)
        * ``rank(indices) == rank(data)``.
    updates: tensor<\\*K, T> (Required)
        * Must be the same shape as ``indices``.
    axis: const i32 (Optional)
        * Default to ``0``.
    mode: const string (Optional)
        * Default to ``add``.
        * Can be the following modes: ``add``, ``div``, ``max``, ``min``, ``mul``, ``sub``, ``update``.
    validate_indices: const bool (Optional)
        * If ``True``, it raises a runtime (possibly also a compile-time) exception
          for out-of-bound values of the ``indices`` parameter.
        * If ``False``, absolutely no checking is performed for out-of-bound values
          of ``indices`` either at compile or runtime. Behavior for out-of-bound indices
          is undefined but memory safe.
        * Default value is ``False``.

    Returns
    -------
    tensor<\\*D, T>
        * With the same type and shape as input ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
        indices=TensorInputType(type_domain=types.int32),
        updates=TensorInputType(type_domain="T"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        mode=TensorInputType(const=True, optional=True, type_domain=types.str),
        validate_indices=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    def default_inputs(self):
        return DefaultInputs(
            axis=0,
            mode="add",
            validate_indices=False,
        )

    def type_inference(self):
        result = super().type_inference()
        if self.validate_indices.val:
            indices = self.indices.val
            if indices is not None:
                if np.count_nonzero(
                    np.logical_or(indices < 0, indices >= self.data.shape[self.axis.val])
                ):
                    raise IndexError(
                        f"Indices is out of bounds for `{self.op_type}` node {self.name}. "
                        f"Expected indices between [0, {self.data.shape[self.axis.val]}), but got {indices}."
                    )
        return result


@register_op(opset_version=_IOS17_TARGET)
class scatter_nd(_scatter_nd_iOS15):
    """
    Scatter ``updates`` to ``data`` at locations ``indices``.

    The major differences from the previous version are illustrated in :py:class:`scatter`.
    For more information, see the iOS 15 :py:class:`~.iOS15.scatter_gather.scatter_nd`.

    Parameters
    ----------
    data: tensor<\\*D, T> (Required)
    indices: tensor<\\*K, i32> (Required)
    updates: tensor<\\*K, T> (Required)
        * Must be the shape as ``K[:-1]+data.shape[K[-1]:]``.
    mode: const string (Optional)
        * Default to ``add``.
        * Can be the following modes: ``add``, ``div``, ``max``, ``min``, ``mul``, ``sub``, ``update``.
    validate_indices: const bool (Optional)
        * If ``True``, it raises a runtime (possibly also a compile-time) exception for out-of-bound values of
          the ``indices`` parameter.
        * If ``False``, absolutely no checking is performed for out-of-bound values of ``indices``
          either at compile or runtime. Behavior for out-of-bound indices is undefined but memory safe.
        * Default value is ``False``.

    Returns
    -------
    tensor<\\*D, T>
        * A tensor with the same shape and type as ``data``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
        indices=TensorInputType(type_domain=types.int32),
        updates=TensorInputType(type_domain="T"),
        mode=TensorInputType(const=True, optional=True, type_domain=types.str),
        validate_indices=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    def default_inputs(self):
        return DefaultInputs(
            mode="add",
            validate_indices=False,
        )

    def type_inference(self):
        result = super().type_inference()
        if self.validate_indices.val:
            indices = self.indices.val
            upper_bound = self.data.shape
            if indices is not None:
                if np.count_nonzero(np.logical_or(indices < 0, indices >= upper_bound)):
                    raise IndexError(
                        f"Indices is out of bounds for `{self.op_type}` node {self.name}. "
                        f"Expected indices between [0, {upper_bound}), but got {indices}."
                    )
        return result


@register_op(opset_version=_IOS17_TARGET)
class gather(_gather_iOS16):
    """
    Gather slices from input ``x`` along dimension ``axis`` according to ``indices``,
    similar to `tf.gather_nd <https://www.tensorflow.org/api_docs/python/tf/gather_nd>`_.

    This section documents only the differences between this version and the
    iOS 16 :py:class:`~.iOS16.scatter_gather.gather`. The major differences are as follows:

    - Input parameter ``x`` adds support for ``int16``, ``uint16``, ``int8``, and ``uint8``.
    - Input parameter ``indices`` adds support for ``int8`` and ``uint8``.
    - Input parameter ``indices`` now supports only positive values -- negative values
      are considered out-of-bound. If support for negative indices is required, they must be
      explicitly converted to positive values, using the following::

         index = iOS17.select(index >= 0, index, index + max_index)

    - New input parameter called ``validate_indices`` has been added to all gather ops. Its behavior is as follows:
       - If ``True``, it raises a runtime (possibly also a compile-time) exception for
         out-of-bound values of the ``indices`` parameter.
       - If ``False``, absolutely no checking is performed for out-of-bound values of ``indices``
         either at compile or runtime. Behavior for out-of-bound indices is undefined but memory safe.

    Parameters
    ----------
    x: tensor<\\*D, T> (Required)
    indices: tensor<\\*N, I> (Required)
        * Indices values may be negative. More precisely, ``-D[axis]<= v < D[axis]`` for ``v`` in ``indices``.
    axis: const i32 (Optional. Default=``0``)
        * Negative axis is supported.
    batch_dims: const i32 (Optional. Default=``0``)
        * The number of batch dimensions.
    validate_indices: const bool (Optional)
        * If ``True``, it raises a runtime (possibly also a compile-time) exception
          for out-of-bound values of the ``indices`` parameter.
        * If ``False``, absolutely no checking is performed for out-of-bound values
          of ``indices`` either at compile or runtime. Behavior for out-of-bound indices
          is undefined but memory safe.
        * Default value is ``False``.

    Returns
    -------
    tensor<\\*K, T>
        * Where ``K = D[:axis] + N[batch_dims:] + D[axis+1:]``.

    Attributes
    ----------
    T: fp16, fp32, int32, int16, uint16, int8, uint8
    I: int32, int16, uint16, int8, uint8
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        indices=TensorInputType(type_domain="I"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        batch_dims=TensorInputType(const=True, optional=True, type_domain=types.int32),
        validate_indices=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int32,
            types.int16,
            types.uint16,
            types.int8,
            types.uint8,
        ),
        "I": (types.int32, types.int16, types.uint16, types.int8, types.uint8),
    }

    def default_inputs(self):
        return DefaultInputs(axis=0, batch_dims=0, validate_indices=False)

    def type_inference(self):
        result = super().type_inference()
        if self.validate_indices.val:
            indices = self.indices.val
            if indices is not None:
                if np.count_nonzero(
                    np.logical_or(indices < 0, indices >= self.x.shape[self.axis.val])
                ):
                    raise IndexError(
                        f"Indices is out of bounds for `{self.op_type}` node {self.name}. "
                        f"Expected indices between [0, {self.x.shape[self.axis.val]}), but got {indices}."
                    )
        return result


@register_op(opset_version=_IOS17_TARGET)
class gather_along_axis(_gather_along_axis_iOS16):
    """
    Take the values along ``axis`` at locations ``indices``.

    The major differences from the previous version are illustrated in :py:class:`gather`.
    For more information, see the iOS 16 :py:class:`~.iOS16.scatter_gather.gather_along_axis`.

    Parameters
    ----------
    x: tensor<\\*D, T> (Required)
    indices: tensor<\\*K, I> (Required)
        * ``rank(indices) == rank(x)``.
    axis: const i32 (Optional):
        * Default to ``0``.
    validate_indices: const bool (Optional)
        * If ``True``, it raises a runtime (possibly also a compile-time) exception for out-of-bound values of
          the ``indices`` parameter.
        * If ``False``, absolutely no checking is performed for out-of-bound values of ``indices``
          either at compile or runtime. Behavior for out-of-bound indices is undefined but memory safe.
        * Default value is ``False``.

    Returns
    -------
    tensor<\\*D, T>:
        * Output tensor has the same shape as ``indices``.

    Attributes
    ----------
    T: fp16, fp32, int32, int16, uint16, int8, uint8
    I: int32, int16, uint16, int8, uint8
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        indices=TensorInputType(type_domain="I"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        validate_indices=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int32,
            types.int16,
            types.uint16,
            types.int8,
            types.uint8,
        ),
        "I": (types.int32, types.int16, types.uint16, types.int8, types.uint8),
    }

    def default_inputs(self):
        return DefaultInputs(
            axis=0,
            validate_indices=False,
        )

    def type_inference(self):
        result = super().type_inference()
        if self.validate_indices.val:
            indices = self.indices.val
            if indices is not None:
                upper_bound = self.x.shape[self.axis.val]
                if np.count_nonzero(np.logical_or(indices < 0, indices >= upper_bound)):
                    raise IndexError(
                        f"Indices is out of bounds for `{self.op_type}` node {self.name}. "
                        f"Expected indices between [0, {upper_bound}), but got {indices}."
                    )
        return result


@register_op(opset_version=_IOS17_TARGET)
class gather_nd(_gather_nd_iOS16):
    """
    Gather slices from ``x`` according to ``indices``, similar to `tf.gather_nd`.

    The major differences from the previous version are illustrated in :py:class:`gather`.
    For more information, see the iOS 16 :py:class:`~.iOS16.scatter_gather.gather_nd`.

    Parameters
    ----------
    x: tensor<\\*D, T> (Required)
    indices: tensor<\\*K, I> (Required)
    batch_dims: const i32 (Optional. Default=``0``)
        * The number of batch dimensions.
    validate_indices: const bool (Optional)
        * If ``True``, it raises a runtime (possibly also a compile-time) exception for out-of-bound values of
          the ``indices`` parameter.
        * If ``False``, absolutely no checking is performed for out-of-bound values of ``indices``
          either at compile or runtime. Behavior for out-of-bound indices is undefined but memory safe.
        * Default value is ``False``.

    Returns
    -------
    tensor<\\*V, T>
        * ``V = K[:-1] + D[batch_dims + K[-1]:]``, where ``D = x.shape`` and ``K = indices.shape``.

    Attributes
    ----------
    T: fp16, fp32, int32, int16, uint16, int8, uint8
    I: int32, int16, uint16, int8, uint8
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        indices=TensorInputType(type_domain="I"),
        batch_dims=TensorInputType(const=True, optional=True, type_domain=types.int32),
        validate_indices=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int32,
            types.int16,
            types.uint16,
            types.int8,
            types.uint8,
        ),
        "I": (types.int32, types.int16, types.uint16, types.int8, types.uint8),
    }

    def default_inputs(self):
        return DefaultInputs(
            batch_dims=0,
            validate_indices=False,
        )

    def type_inference(self):
        result = super().type_inference()
        if self.validate_indices.val:
            indices = self.indices.val
            upper_bound = self.x.shape
            if indices is not None:
                if np.count_nonzero(np.logical_or(indices < 0, indices >= upper_bound)):
                    raise IndexError(
                        f"Indices is out of bounds for `{self.op_type}` node {self.name}. "
                        f"Expected indices between [0, {upper_bound}), but got {indices}."
                    )
        return result
