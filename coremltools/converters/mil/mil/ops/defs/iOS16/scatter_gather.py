#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Operation, types
from coremltools.converters.mil.mil.input_type import DefaultInputs, InputSpec, TensorInputType
from coremltools.converters.mil.mil.operation import SYMBOL, VALUE, precondition
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs._utils import compute_gather
from coremltools.converters.mil.mil.ops.defs.iOS15.scatter_gather import (
    gather_along_axis as _gather_along_axis_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS16 import _IOS16_TARGET


@register_op(opset_version=_IOS16_TARGET)
class gather(Operation):
    """
    The iOS16 version.
    This section documents only the differences between this version and the
    iOS 15 :py:class:`~.iOS15.scatter_gather.gather`.

    This version supports ``batch_dims``, similar to `tf.gather <https://www.tensorflow.org/api_docs/python/tf/gather>`_.
    Input parameter ``indices`` now supports ``int16`` and ``uint16``.

    Parameters
    ----------
    x: tensor<\\*D, T> (Required)
    indices: tensor<\\*N, I> (Required)
        * Indices values may be negative. More precisely, ``-D[axis]<= v < D[axis]`` for ``v`` in ``indices``.
    axis: const i32 (Optional. Default=``0``)
        * Negative axis is supported.
    batch_dims: const i32 (Optional. Default=``0``)
        * The number of batch dimensions.

    Returns
    -------
    tensor<\\*K, T>
        * Where ``K = D[:axis] + N[batch_dims:] + D[axis+1:]``.

    Attributes
    ----------
    T: fp16, fp32, i32
    I: uint16, int16, int32

    References
    ----------
    See `tf.gather <https://www.tensorflow.org/api_docs/python/tf/gather>`_.

    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        indices=TensorInputType(type_domain="I"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        batch_dims=TensorInputType(const=True, optional=True, type_domain=types.int32)
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
        "I": (types.int32, types.uint16, types.int16),
    }

    def default_inputs(self):
        return DefaultInputs(
            axis=0,
            batch_dims=0,
        )

    @precondition(allow=VALUE | SYMBOL)
    def value_inference(self):
        x = self.x.sym_val
        indices = self.indices.val
        if indices is None:
            # only allow x to be symbolic. indices cannot.
            return None
        return compute_gather(
            params=self.x.sym_val,
            indices=self.indices.val,
            axis=self.axis.val,
            batch_dims=self.batch_dims.val,
        )

    def type_inference(self):
        # validate parameters
        if self.axis.val < -self.x.rank or self.axis.val >= self.x.rank:
            raise IndexError(
                "Axis value {} is out of bounds for {} node {}".format(
                    self.axis.val, self.op_type, self.name
                )
            )
        if self.batch_dims.val >= self.x.rank:
            raise ValueError(
                "batch_dims {} must be less than x.rank {} for node {}".format(
                    self.batch_dims.val, self.x.rank, self.name
                )
            )
        if self.batch_dims.val > self.indices.rank:
            raise ValueError(
                "batch_dims {} must be less or equal to than indices.rank {} for node {}".format(
                    self.batch_dims.val, self.indices.rank, self.name
                )
            )

        output_rank = self.x.rank - 1 + self.indices.rank - self.batch_dims.val
        if output_rank == 0:
            # output scalar
            return self.x.dtype

        # compute output shape
        axis = self.axis.val
        axis = axis if axis >= 0 else axis + self.x.rank
        batch_dims = self.batch_dims.val
        out_shape = self.x.shape[:axis] + self.indices.shape[batch_dims:] + self.x.shape[axis + 1 :]

        return types.tensor(self.x.dtype, out_shape)


@register_op(opset_version=_IOS16_TARGET)
class gather_along_axis(_gather_along_axis_iOS15):
    """
    The iOS16 version.
    The only difference between this version and the iOS 15 :py:class:`~.iOS15.scatter_gather.gather_along_axis`.
    is that input parameter ``indices`` now supports ``int16`` and ``uint16``.

    Parameters
    ----------
    x: tensor<\\*D, T> (Required)
    indices: tensor<\\*K, I> (Required)
    axis: const i32 (Optional):
        * Default to ``0``.

    Returns
    -------
    tensor<\\*D, T>:
        * Output tensor has the same shape as ``indices``.

    Attributes
    ----------
    T: fp16, fp32, i32
    I: uint16, int16, int32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        indices=TensorInputType(type_domain="I"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
        "I": (types.int32, types.uint16, types.int16),
    }


@register_op(opset_version=_IOS16_TARGET)
class gather_nd(Operation):
    """
    The iOS16 version.
    This section documents only the differences between this version and the
    iOS 15 :py:class:`~.iOS15.scatter_gather.gather_nd`.

    This version supports ``batch_dims``.
    Input parameter ``indices`` now supports ``int16`` and ``uint16``.

    Parameters
    ----------
    x: tensor<\\*D, T> (Required)
    indices: tensor<\\*K, I> (Required)
    batch_dims: const i32 (Optional. Default=``0``)
        * The number of batch dimensions.

    Returns
    -------
    tensor<\\*V, T>
        * ``V = K[:-1] + D[batch_dims + K[-1]:]``, where ``D = x.shape`` and ``K = indices.shape``.

    Attributes
    ----------
    T: fp16, fp32, i32
    I: uint16, int16, int32

    References
    ----------
    See `tf.gather_nd <https://www.tensorflow.org/api_docs/python/tf/gather_nd>`_.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        indices=TensorInputType(type_domain="I"),
        batch_dims=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
        "I": (types.int32, types.uint16, types.int16),
    }

    def default_inputs(self):
        return DefaultInputs(
            batch_dims=0,
        )

    def type_inference(self):
        batch_dims = self.batch_dims.val
        indices_depth = self.indices.shape[-1]
        if indices_depth > self.x.rank - batch_dims:
            msg = "For node {}, indices.shape[-1] ({}) + batch_dims ({}) must be smaller or equal to the input rank {}".format(
                    self.name, indices_depth, batch_dims, self.x.rank
            )
            raise ValueError(msg)
        out_type = self.x.dtype
        out_shape = self.indices.shape[:-1] + self.x.shape[batch_dims+indices_depth:]
        return types.tensor(out_type, out_shape)
