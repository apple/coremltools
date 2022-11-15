#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Operation, types
from coremltools.converters.mil.mil.input_type import (DefaultInputs,
                                                       InputSpec,
                                                       TensorInputType)
from coremltools.converters.mil.mil.operation import (SYMBOL, VALUE,
                                                      precondition)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs._utils import compute_gather
from coremltools.converters.mil.mil.types.symbolic import (
    is_compatible_symbolic_vector)


@register_op
class gather(Operation):
    """
    Gather slices from input ``x`` along dimension ``axis`` according to ``indices``,
    similar to `tf.gather <https://www.tensorflow.org/api_docs/python/tf/gather>`_.

    * If ``indices`` is scalar (0-D):

    .. math::
       output[p_0, ..., p_{axis-1}, ~~~~~~~~~~~~~~~~~~~~~~~~ p_{axis+1}, ..., p_{rank(x)-1}] =
    .. math::
       x[p_0, ..., p_{axis-1}, ~~~~~~~~~ indices, ~~~~~~~~ p_{axis+1}, ..., p_{rank(x)-1}]

    Where ``rank(x)`` is the rank of ``x``. The ``output`` has rank ``rank(x) - 1``.

    * If ``indices`` is 1-D tensor:

    .. math::
       output[p_0, ..., p_{axis-1}, ~~~~~~~~~~~~~ i, ~~~~~~~~~~~~~ p_{axis+1}, ..., p_{rank(*D)-1}] =
    .. math::
       x[p_0, ..., p_{axis-1}, ~~~~~~~~ indices[i], ~~~~~~~~ p_{axis+1}, ..., p_{rank(*D)-1}]

    The output has rank ``rank(x)``.

    * In general:

    .. math::
       output[p_0, ..., p_{axis-1}, ~~~~~~~~ i_0, ..., i_{M-1}, ~~~~~~~~ p_{axis+1}, ..., p_{rank(x)-1}] =
    .. math::
       x[p_0, ..., p_{axis-1}, ~~~~~~~ indices[i_0, ..., i_{M-1}], ~~~~~~~ p_{axis+1}, ..., p_{rank(x)-1}]

    Where ``M = rank(indices)``.

    Parameters
    ----------
    x: tensor<\*D, T> (Required)
    indices: tensor<\*N, i32> (Required)
        * Indices values may be negative. More precisely, ``-D[axis]<= v < D[axis]`` for ``v`` in ``indices``.
    axis: const i32 (Optional. Default=``0``)
        * Negative axis is supported.

    Returns
    -------
    tensor<\*K, T>
        * Where ``K = D[:axis] + N + D[axis+1:]``.

    Attributes
    ----------
    T: fp16, fp32, i32

    References
    ----------
    See `tf.gather <https://www.tensorflow.org/api_docs/python/tf/gather>`_.

    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        indices=TensorInputType(type_domain=types.int32),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )
    
    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def default_inputs(self):
        return DefaultInputs(
            axis=0,
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
                batch_dims=0
            )

    def type_inference(self):
        out_type = self.x.dtype

        if self.axis.val < -self.x.rank or self.axis.val >= self.x.rank:
            raise IndexError(
                "Axis value {} is out of bounds for {} node {}".format(
                    self.axis.val, self.op_type, self.name
                )
            )

        output_rank = self.x.rank - 1 + self.indices.rank
        if output_rank == 0:
            # output scalar
            return out_type

        axis = self.axis.val
        axis = axis if axis >= 0 else axis + self.x.rank
        out_shape = self.x.shape[:axis] + self.indices.shape + self.x.shape[axis + 1 :]
        return types.tensor(out_type, out_shape)


@register_op
class scatter(Operation):
    """
    Scatter ``updates`` to ``data`` at locations ``indices`` at dimension ``axis``
    by operation ``mode``.

    Example: ``mode == update``.

    * For ``i`` in ``[0, len(indices)]``:

    .. math::
       output[p_0, ..., p_{axis-1}, indice[i], p_{axis+1}, ..., p_D] =
    .. math::
       updates[p_0, ..., p_{axis-1}, i, p_{axis+1}, ..., p_D]

    * For ``j != i``:

    .. math::
       output[p_0, ..., p_{axis-1}, j, p_{axis+1}, ..., p_D] =
    .. math::
       data[p_0, ..., p_{axis-1}, j, p_{axis+1}, ..., p_D]

    Example: ``mode == add``.

    * For ``i`` in ``[0, len(indices)]``:

    .. math::
       output[p_0, ..., p_{axis-1}, indice[i], p_{axis+1}, ..., p_D] =
    .. math::
       updates[p_0, ..., p_{axis-1}, i, p_{axis+1}, ..., p_D] +
    .. math::
       x[p_0, ..., p_{axis-1}, indice[i], p_{axis+1}, ..., p_D]

    * For ``j != i``:

    .. math::
       output[p_0, ..., p_{axis-1}, j, p_{axis+1}, ..., p_D] =
    .. math::
       data[p_0, ..., p_{axis-1}, j, p_{axis+1}, ..., p_D]

    Parameters
    ----------
    data: tensor<\*D, T> (Required)
    indices: tensor<[C], i32> (Required)
        * 1-D tensor.
    updates: tensor<\*K, T> (Required)
        * ``K = data.shape[:axis] + [len(indices)] + data.shape[axis+1:]``.
    axis: const i32 (Optional)
        * Default to ``0``.
    mode: const string (Optional)
        * Can be the following modes: ``update``, ``add``, ``sub``, ``mul``,
          ``div``, ``max``, ``min``.
        * Default value is ``update``.

    Returns
    -------
    tensor<\*D, T>
        * With the same type and shape as input ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32

    For example:
        data = [[1, 2, 3], [4, 5, 6]]
        indices = [1, 0]
        updates = [[5, 6, 7], [8, 9, 10]]
        axis = 0
        mode = "update"

    produces:
       [[9, 11, 13], [9, 11, 13]]
    """

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
        indices=TensorInputType(type_domain=types.int32),
        updates=TensorInputType(type_domain="T"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        mode=TensorInputType(const=True, optional=True, type_domain=types.str),
    )
    
    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def default_inputs(self):
        return DefaultInputs(
            axis=0,
            mode="add",
            )

    def type_inference(self):
        if self.axis.val < -self.data.rank or self.axis.val >= self.data.rank:
            raise IndexError(
                "Axis value {} is out of bounds for {} node {}".format(
                    self.axis.val, self.op_type, self.name
                )
            )

        axis = self.axis.val
        axis = axis if axis >= 0 else axis + self.data.rank
        expected_updates_shape = (
            self.data.shape[:axis] + self.indices.shape + self.data.shape[axis + 1 :]
        )

        err = "Updates shape {} is incorrect. It should be {}.".format(self.updates.shape, expected_updates_shape)
        assert is_compatible_symbolic_vector(
            self.updates.shape, tuple(expected_updates_shape)
        ), err

        return self.data.sym_type


@register_op
class gather_along_axis(Operation):
    """
    Take the values along ``axis`` at locations ``indices``.

    .. math::
       idx = indices[p_0, ..., p_{axis-1}, i, p_{axis+1}, ..., p_D]
    .. math::
       output[p_0, ..., p_{axis-1}, i, p_{axis+1}, ..., p_D] = = x[p_0, ..., p_{axis-1}, idx, p_{axis+1}, ..., p_D]

    Parameters
    ----------
    x: tensor<\*D, T> (Required)
    indices: tensor<\*K, i32> (Required)
        * ``rank(indices) == rank(x)``.
    axis: const i32 (Optional):
        * Default to ``0``.

    Returns
    -------
    tensor<\*D, T>:
        * Output tensor has the same shape as ``indices``.

    Attributes
    ----------
    T: fp16, fp32, i32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        indices=TensorInputType(type_domain=types.int32),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )
    
    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def default_inputs(self):
        return DefaultInputs(
            axis=0,
            )

    @precondition(allow=VALUE)
    def value_inference(self):
        x = self.x.val
        indices = self.indices.val
        axis = self.axis.val
        return np.take_along_axis(x, indices, axis)

    def type_inference(self):

        if self.x.rank != self.indices.rank:
            raise ValueError(
                "Rank mismatch between input and indices. \
                              Input rank: {}, indices rank: {}".format(
                    self.x.rank, self.indices.rank
                )
            )

        if self.axis.val < -self.x.rank or self.axis.val >= self.x.rank:
            raise IndexError(
                "Axis value {} is out of bounds for {} node {}".format(
                    self.axis.val, self.op_type, self.name
                )
            )

        axis = self.axis.val
        axis = axis if axis >= 0 else axis + self.x.rank

        for i in range(self.x.rank):
            if i != axis:
                assert self.x.shape[i] == self.indices.shape[i]

        return types.tensor(self.x.dtype, self.indices.shape)


@register_op
class scatter_along_axis(Operation):
    """
    Scatter ``updates`` to ``data`` at locations ``indices`` along ``axis`` dimension
    using ``mode`` operation.

    Example: ``mode == update``.

    * For ``i`` in ``[0, len(indices)]``:

    .. math::
       idx = indices[p_0, ..., p_{axis-1}, i, p_{axis+1}, ..., p_D]
    .. math::
       output[p_0, ..., p_{axis-1}, idx, p_{axis+1}, ..., p_D] =
    .. math::
       updates[p_0, ..., p_{axis-1}, i, p_{axis+1}, ..., p_D]

    * For ``j! = i``:

    .. math::
       output[p_0, ..., p_{axis-1}, j, p_{axis+1}, ..., p_D] =
    .. math::
       data[p_0, ..., p_{axis-1}, j, p_{axis+1}, ..., p_D]

    Example: ``mode == add``.

    * For ``i`` in ``[0, len(indices)]``:

    .. math::
       idx = indices[p_0, ..., p_{axis-1}, i, p_{axis+1}, ..., p_D]
    .. math::
       output[p_0, ..., p_{axis-1}, idx, p_{axis+1}, ..., p_D] =
    .. math::
       updates[p_0, ..., p_{axis-1}, i, p_{axis+1}, ..., p_D] +
    .. math::
       x[p_0, ..., p_{axis-1}, indice[i], p_{axis+1}, ..., p_D]

    * For ``j! = i``:

    .. math::
       output[p_0, ..., p_{axis-1}, j, p_{axis+1}, ..., p_D] =
    .. math::
       data[p_0, ..., p_{axis-1}, j, p_{axis+1}, ..., p_D]

    Parameters
    ----------
    data: tensor<\*D, T> (Required)
    indices: tensor<\*K, i32> (Required)
        * ``rank(indices) == rank(data)``.
    updates: tensor<\*K, T> (Required)
        * Must be the same shape as ``indices``.
    axis: const i32 (Optional)
        * Default to ``0``.
    mode: const string (Optional)
        * Default to ``add``.
        * Can be the following modes: ``update``, ``add``, ``sub``, ``mul``,
          ``div``, ``max``, ``min``.

    Returns
    -------
    tensor<\*D, T>
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
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def default_inputs(self):
        return DefaultInputs(
            axis=0,
            mode="add",
            )

    @precondition(allow=VALUE)
    def value_inference(self):
        data = np.copy(self.data.val)
        indices = self.indices.val
        updates = self.updates.val
        axis = self.axis.val
        np_output = data
        np.put_along_axis(np_output, indices, updates, axis=axis)
        return np_output

    def type_inference(self):
        if self.axis.val < -self.data.rank or self.axis.val >= self.data.rank:
            raise IndexError(
                "Axis value {} is out of bounds for {} node {}".format(
                    self.axis.val, self.op_type, self.name
                )
            )

        axis = self.axis.val
        axis = axis if axis >= 0 else axis + self.data.rank

        assert is_compatible_symbolic_vector(
            self.indices.shape, self.updates.shape
        )
        assert self.data.rank == self.indices.rank
        for i in range(self.data.rank):
            if i != axis:
                assert self.data.shape[i] == self.indices.shape[i]

        return self.data.sym_type


@register_op
class gather_nd(Operation):
    """
    Gather slices from ``x`` according to ``indices``, similar to `tf.gather_nd <https://www.tensorflow.org/api_docs/python/tf/gather_nd>`_.

    The ``indices`` is a K-dim tensor, where ``indices[i_0,...,i_{K-2}]`` defines a slice
    of ``x``:

    .. math::
       output[i_0, ..., i_{K-2}]= x[indices[i_0, ..., i_{K-2}]]

    Where ``K = rank(indices)`` and ``x[indices[i_0, ..., i_{K-2}]]`` has rank
    ``rank(x) - indices.shape[-1]``.

    Parameters
    ----------
    x: tensor<\*D, T> (Required)
    indices: tensor<\*K, i32> (Required)

    Returns
    -------
    tensor<\*V, T>
        * ``V = K[:-1] + D[K[-1]:]``, where ``D = x.shape`` and ``K = indices.shape``.

    Attributes
    ----------
    T: fp16, fp32, i32

    References
    ----------
    See `tf.gather_nd <https://www.tensorflow.org/api_docs/python/tf/gather_nd>`_.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        indices=TensorInputType(type_domain=types.int32),
        )
        
    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def type_inference(self):
        assert self.indices.shape[-1] <= self.x.rank
        out_type = self.x.dtype
        out_shape = self.indices.shape[:-1] + self.x.shape[self.indices.shape[-1] :]
        return types.tensor(out_type, out_shape)


@register_op
class scatter_nd(Operation):
    """
    Scatter ``updates`` to ``data`` at locations ``indices``.

    The ``indices`` is a K-dim tensor, where ``indices[i_0,...,i_{K-2}]`` defines a
    slice of ``data``, ``K = rank(indices)``, and ``data[indices[i_0, ..., i_{K-2}]]``
    has rank ``rank(data) - indices.shape[-1]``.

    * Example: ``mode == update``: The ``output`` is set to ``data`` initially, and
      the op updates ``output`` as follows:

    .. math::
       output[indices[i_0, ..., i_{K-2}]]= updates[indices[i_0, ..., i_{K-2}]]

    * Example: ``mode == add``. The update rule is:

    .. math::
       output[indices[i_0, ..., i_{K-2}]] += updates[indices[i_0, ..., i_{K-2}]]

    Parameters
    ----------
    data: tensor<\*D, T> (Required)
    indices: tensor<\*K, i32> (Required)
    updates: tensor<\*K, T> (Required)
        * Must be the shape as ``K[:-1]+data.shape[K[-1]:]``.
    mode: const string (Optional)
        * Default to ``add``.
        * Can be the following modes: ``update``, ``add``, ``sub``, ``mul``,
          ``div``, ``max``, ``min``.

    Returns
    -------
    tensor<\*D, T>
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
    )
    
    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def default_inputs(self):
        return DefaultInputs(
            mode="add",
            )

    def type_inference(self):
        assert self.indices.shape[-1] <= self.data.rank
        expected_updates_shape = (
            self.indices.shape[:-1] + self.data.shape[self.indices.shape[-1] :]
        )
        assert is_compatible_symbolic_vector(
            self.updates.shape, tuple(expected_updates_shape)
        )
        return self.data.sym_type
