#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging
import numpy as np
import sympy as sm

from coremltools.converters.mil.mil.types.symbolic import (
    is_symbolic,
    isscalar,
    any_symbolic,
    any_variadic,
)
from coremltools.converters.mil.mil import (
    get_new_symbol,
    get_new_variadic_symbol,
    Operation,
    precondition,
    SYMBOL,
    types,
    VALUE,
)
from coremltools.converters.mil.mil.input_type import (
    BoolTensorInputType,
    DefaultInputs,
    InputSpec,
    IntInputType,
    IntTensorInputType,
    ScalarOrTensorInputType,
    TensorInputType
)
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op

def _solve_slice_by_index_shape(x_shape, begin, end, stride, begin_mask, end_mask, squeeze_mask):
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
                continue
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


@register_op(doc_str="")
class depth_to_space(Operation):
    """
    Rearrange elements in a tensor from depth (channel) into spatial dimensions.

    Parameters
    ----------
    x: tensor<[n, C, H, W], T> (Required)
        * Input tensor of rank ``4``.
    block_size: const i32 (Required)
        * The size of the spatial block. Must be greater than ``1`` and divisible by
          channel dimension ``C``.

    Returns
    -------
    tensor<[n, C / block_size^2, H x block_size, W x block_size], T>
        * Where ``b`` is the block size.

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        block_size=IntInputType(const=True),
        )

    def __init__(self, **kwargs):
        super(depth_to_space, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        n, c, h, w = self.x.shape
        bs = self.block_size.val
        ret_shape = (n, c // (bs * bs), h * bs, w * bs)
        return types.tensor(x_type, ret_shape)


@register_op(doc_str="")
class expand_dims(Operation):
    """
    Insert a single-dimension in a 1-D or higher tensor at each axis in axes.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
        * Scalar or tensor.
    axes: const tensor<[K], i32> Required
        * ``K`` is the number of dimensions expanded.
        * Insert single dimension at dimension index at each axes.
        * Negative value to index from the end. ``-d-1 <= axis <= d``
          where ``d`` is the rank of ``x``.

    Returns
    -------
    tensor<\*(rank(x)+K), T>
        * Same type as the input ``x`` with rank ``rank(x)+K``.

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(),
        axes=IntTensorInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(expand_dims, self).__init__(**kwargs)

    def type_inference(self):
        x_rank = self.x.rank
        x_type = self.x.dtype
        x_shape = list(self.x.shape)
        axes = self.axes.val
        out_rank = x_rank + len(axes)

        for axis in axes:
            if axis <= -out_rank - 1 or axis >= out_rank:
                msg = 'Axis value {} is out of bounds for {} node "{}" of shape {}'
                raise IndexError(
                    msg.format(axis, self.op_type, self.name, self.x.shape)
                )

        ret_shape = x_shape
        axes = sorted([out_rank + axis if axis < 0 else axis for axis in axes])
        for axis in axes:
            ret_shape.insert(axis, 1)

        return types.tensor(x_type, tuple(ret_shape))

    @precondition(allow=VALUE)
    def value_inference(self):
        axes = self.axes.val
        out_rank = self.x.rank + len(axes)

        for axis in axes:
            if axis <= -out_rank - 1 or axis >= out_rank:
                msg = 'Axis value {} is out of bounds for {} node "{}" of shape {}'
                raise IndexError(
                    msg.format(axis, self.op_type, self.name, self.x.shape)
                )

        axes = sorted([out_rank + axis if axis < 0 else axis for axis in axes])
        ret_shape = list(self.x.shape)
        for axis in axes:
            ret_shape.insert(axis, 1)
        return np.reshape(self.x.val, ret_shape)


def reshape_with_symbol(v, shape):
    """
    Perform basic reshape if v is symbolic (not array of symbols).
    """
    if is_symbolic(v):
        return np.array(v).reshape(shape)
    shape = [int(s) for s in shape]
    return v.reshape(shape)


@register_op(doc_str="")
class reshape(Operation):
    """
    Return a tensor that has the same values as ``x`` with shape ``shape``.
    ``shape`` must have the same volume (number of elements) as ``x``.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)

        * A n-D tensor or a scalar.
        * If ``x`` is fixed rank (and possibly contains symbolic dimension),
          shape may contain elements that are not positive integers (see below).
        * If ``x`` is variadic rank, shape can only contain positive integers.

    shape: tensor<[K], i32> (Required)

        A 1-D tensor, with elements from the following:

            * Positive integers.
            * Symbols: All but one symbol in shape must be present in ``x.shape``.
              The new symbol that is not present in ``x.shape`` represent a dimension
              such that the total size remains constant. Symbol is illegal
              if ``x`` is variadic rank.
            * ``-1``: ``-1`` introduces a new symbol (see Symbols). Therefore, ``-1`` is
              allowed if all symbols in the shape appear in ``x.shape``. ``-1`` is illegal
              if ``x`` is variadic rank.
            * ``0``: If ``K == rank(x)`` then ``0`` means inheriting from the corresponding
              dimension in ``x.shape``. ``0`` is illegal if ``x`` is variadic rank.

    Returns
    -------
    tensor<\*?, T>
        * Tensor with shape determined by the input shape.

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(
        x=ScalarOrTensorInputType(),
        shape=IntTensorInputType(),
        )

    def __init__(self, **kwargs):
        super(reshape, self).__init__(**kwargs)

    def type_inference(self):
        if any_symbolic(self.shape.shape):
            # We can't infer any shape if shape has variable length.
            return types.tensor(self.x.dtype, (get_new_variadic_symbol(),))

        # shape has fixed length here.
        if self.shape.sym_val is None:
            shape = tuple([get_new_symbol() for _ in range(self.shape.shape[0])])
            return types.tensor(self.x.dtype, shape)
        t, _ = self._get_type_val()
        return t

    @precondition(allow=VALUE | SYMBOL)
    def value_inference(self):
        _, val = self._get_type_val()
        return val

    def _get_type_val(self):
        x_type = self.x.dtype
        x_shape = self.x.shape
        x_vol = np.prod(x_shape)
        # shape is const, and thus sym_val is not None
        sym_shape = self.shape.sym_val
        sym_shape = [get_new_symbol() if d == -1 else d for d in sym_shape]
        try:
            ret_shape = reshape.enforce_volumetric_constraint(x_vol, sym_shape)
        except:
            ret_shape = sym_shape
        ret_val = None
        if self.x.val is not None and all(isscalar(a) and not is_symbolic(a) for a in ret_shape):
            ret_val = reshape_with_symbol(self.x.val, ret_shape)
        return types.tensor(x_type, tuple(ret_shape)), ret_val

    @staticmethod
    def enforce_volumetric_constraint(left_volume, inshape):
        left_symbols = set()
        if is_symbolic(left_volume):
            left_symbols = left_volume.free_symbols
        # Generally, we want to solve for right in terms of left. But this
        # is kinda annoying actually.
        shape = list(inshape)

        # Handling when reshape is given 0 instead of actual input
        # input tensor shape: [4, 3, 2], reshape:[0, -1], output tensor shape: [4, 6]
        if shape.count(-1) > 1:
            raise ValueError(
                "Reshape op supports only one dimension to be -1. Given {}".format(
                    shape.count(-1)
                )
            )

        infer_dim_index = shape.index(-1) if -1 in shape else None
        right_volume = 1
        for i in shape:
            if i != -1:
                right_volume = right_volume * i

        if infer_dim_index:
            shape[infer_dim_index] = left_volume // right_volume

        if not is_symbolic(right_volume):
            return shape

        constraints = [left_volume - right_volume]
        solve_for = [s for s in shape if is_symbolic(s)]

        for rightsym in solve_for:
            sol = sm.solve(constraints, [rightsym], dict=True)
            if not isinstance(sol, list):
                sol = [sol]
            # look for an acceptable solution
            for s in sol:
                if 0 in s.values():
                    continue
                for i in range(len(shape)):
                    if shape[i] in s:
                        v = s[shape[i]]
                        if len(v.free_symbols - left_symbols) > 0:
                            continue
                        try:
                            shape[i] = int(v)
                        except:
                            shape[i] = v
        return shape


@register_op(doc_str="")
class reverse(Operation):
    """
    Reverse the order of the input tensor ``x`` along specified ``axes``(dimensions).

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
        * Input tensor.

    axes: const<D, i32> (Optional)
        * Dimension(s) to reverse. Each axis must be in the range ``[-rank(x), rank(x))``.
        * Defaults to None (reverse on all dimensions).

    Returns
    -------
    tensor<\*?, T>
        * Same type and shape as the input tensor.

    Attributes
    ----------
    T: fp32

    References
    ----------
    See `tf.reverse <https://www.tensorflow.org/api_docs/python/tf/reverse>`_
    and `TORCH <https://pytorch.org/docs/stable/torch.html#torch.flip>`_.
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        axes=IntTensorInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            axes=None,
            )

    def __init__(self, **kwargs):
        super(reverse, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        res = self.x.val
        axes = self.axes.val if self.axes is not None else range(self.x.rank)
        for axis in axes:
            res = np.flip(res, axis=axis)
        return res


@register_op(doc_str="")
class reverse_sequence(Operation):
    """
    Reverse variable length slices for specified axes / dimensions of the input
    tensor. This op first slices input tensor along the ``batch_axis`` dimension, then
    partially reverses the elements along the ``seq_axis`` for the first ``lengths[i]``
    elements.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
        * Input tensor.
    lengths: tensor<L, i32> (Required)
        * 1-dimensional tensor of length ``x.shape[batch_axis]`` specifying the length
          of the sequence to reverse.
        * Values must be in range ``[0, x.shape[seq_axis]]``.
    seq_axis: const<i32> (Optional)
        * The dimension to reverse.
        * Defaults to ``0``.
    batch_axis: const<i32> (Optional)
        * Dimension for slicing.
        * Defaults to ``0``.

    Returns
    -------
    tensor<\*?, T>
        * Same type and shape as the input tensor.

    Attributes
    ----------
    T: fp32

    References
    ----------
    `tf.reverse_sequence <https://www.tensorflow.org/api_docs/python/tf/reverse_sequence>`_

    """

    input_spec = InputSpec(
        x=TensorInputType(),
        lengths=IntTensorInputType(),
        seq_axis=IntInputType(const=True, optional=True),
        batch_axis=IntInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            seq_axis=0,
            batch_axis=0)

    def __init__(self, **kwargs):
        super(reverse_sequence, self).__init__(**kwargs)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        raise NotImplementedError("TODO")


@register_op(doc_str="")
class slice_by_index(Operation):
    """
    Method for numpy style indexing and slicing.
    Suppose we have a tensor ``x``, this method achieves:
    ``result = x[begin[0]: end[0]: stride[0], begin[1]: end[1]: stride[1], ...]``
    Note this method does not support pure indexing. You would need to do squeeze if indexing is intended.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
        * Input tensor
    begin: tensor<[rank<x>], i32> (Required)
        * Starting index for the dimension of slicing.
    end: tensor<[rank(x)], i32> (Required)
        * Ending index for the dimension of slicing.
    stride: tensor<[rank(x)], i32> (Optional)
        * Default as all ``1``s.
        * Stride for the dimension of slicing.
    begin_mask: tensor<[rank(x)], bool> (Optional)
        * Default to all ``False``.
        * If ``begin_mask[i]==True``, neglect ``begin[i]``, and set ``begin[i]`` to ``0``.
    end_mask: tensor<[rank(x)], bool> (Optional)
        * Default to all ``False``.
        * If ``end_mask[i]==True``, neglect ``end[i]``, and set ``end[i]`` to ``x.shape[i]``.
    squeeze_mask: tensor<[rank(x)], bool> (Optional)
        * Default to all ``False``.
        * If ``squeeze_mask[i]==true``, neglect ``end[i]``, and do the pure index at ``begin[i]``.

    Returns
    -------
    tensor<*?, T>
        - Scalar or tensor.

    Attributes
    ----------
    T: fp32

    """

    input_spec = InputSpec(
        x=TensorInputType(),
        begin=IntTensorInputType(),
        end=IntTensorInputType(),
        stride=IntTensorInputType(const=True, optional=True),
        begin_mask=BoolTensorInputType(const=True, optional=True),
        end_mask=BoolTensorInputType(const=True, optional=True),
        squeeze_mask=BoolTensorInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            stride=None,
            begin_mask=None,
            end_mask=None,
            squeeze_mask=None,
            )

    def __init__(self, **kwargs):
        super(slice_by_index, self).__init__(**kwargs)

    def type_inference(self):

        # get tensor and set default value
        begin = self.begin.val
        end = self.end.val
        x_rank = self.x.rank
        stride = self.stride.val if self.stride is not None else [1] * x_rank
        begin_mask = (
            self.begin_mask.val if self.begin_mask is not None else [False] * x_rank
        )
        end_mask = self.end_mask.val if self.end_mask is not None else [False] * x_rank
        squeeze_mask = (
            self.squeeze_mask.val if self.squeeze_mask is not None else [False] * x_rank
        )

        # solve shape
        x_shape = self.x.shape
        ret_shape = _solve_slice_by_index_shape(x_shape, begin, end, stride, begin_mask, end_mask, squeeze_mask)

        if len(ret_shape) == 0:
            # Scalar case.
            return self.x.dtype
        else:
            return types.tensor(self.x.dtype, tuple(ret_shape))

    def value_inference(self):
        if self.x.sym_val is None or self.begin.val is None or self.end.val is None:
            return None
        x_shape = self.x.shape
        begin = [int(i) for i in list(self.begin.val[:])]
        end = [int(i) for i in list(self.end.val[:])]
        stride = [1] * self.x.rank if self.stride is None else self.stride.val
        begin_mask = (
            [False] * self.x.rank if self.begin_mask is None else self.begin_mask.val
        )
        end_mask = [False] * self.x.rank if self.end_mask is None else self.end_mask.val
        squeeze_mask = (
            [False] * self.x.rank
            if self.squeeze_mask is None
            else self.squeeze_mask.val
        )

        slices = []
        for idx, mask in enumerate(begin_mask):
            if mask:
                begin[idx] = None
        for idx, mask in enumerate(end_mask):
            if mask:
                end[idx] = None
        squeeze_axes = []
        for idx, mask in enumerate(squeeze_mask):
            if mask:
                end[idx] = None
                stride[
                    idx
                ] = 2147483647  # We slice out only 1 element by setting stride to INF
                squeeze_axes.append(idx)
        for idx in range(self.x.rank):
            slices.append(slice(begin[idx], end[idx], stride[idx]))

        slices = tuple(slices)
        res = self.x.sym_val[slices]

        # remove squeezed axes
        if len(squeeze_axes) > 0:
            if len(squeeze_axes) == len(res.shape):
                if len(res) == 0:
                    logging.warning("%s seems to be a 0 sized tensor", self.name)
                    return np.array([])
                res = res.tolist()[0]
                if is_symbolic(res):
                    return res
                elif self.x.dtype == types.int32 or self.x.dtype == types.int64:
                    res = np.int32(res)
                elif self.x.dtype == types.float or self.x.dtype == types.double:
                    res = np.float32(res)
                else:
                    raise ValueError(
                        "Unable to convert type {}".format(self.x.sym_val.dtype)
                    )
            else:
                res = np.squeeze(res, axis=tuple(squeeze_axes))
        return res


@register_op(doc_str="")
class slice_by_size(Operation):
    """
    Slice input tensor starting from the given ``begin`` index and by
    the amount specified by the ``size`` input, for each dimension.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
        * Input tensor.
    begin: tensor<[rank(x)], i32> Required
        * The begin index for slice.
    size: tensor<[rank(x)], i32> Required
        * The size that is to be sliced. If ``size`` is ``-1``,
          all the remaining elements starting with "begin" are sliced.

    Returns
    -------
    tensor<\*?, T>
        * Scalar or tensor.

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        begin=IntTensorInputType(),
        size=IntTensorInputType(),
    )

    def __init__(self, **kwargs):
        super(slice_by_size, self).__init__(**kwargs)

    def type_inference(self):
        if self.begin.rank != 1:
            raise ValueError(
                "begin should be 1-D tensor, got {}-D tensor instead".format(
                    self.begin.rank
                )
            )
        if self.size.rank != 1:
            raise ValueError(
                "size should be 1-D tensor, got {}-D tensor instead".format(
                    self.size.rank
                )
            )
        if self.x.rank != self.begin.shape[0]:
            raise ValueError(
                "Length of begin {} doesn't equal to input rank {}.".format(
                    len(self.begin.shape[0]), len(self.x.rank)
                )
            )
        if self.x.rank != self.size.shape[0]:
            raise ValueError(
                "Length of size {} doesn't equal to input rank {}.".format(
                    len(self.size.shape[0]), len(self.x.rank)
                )
            )

        x_shape = self.x.shape
        ret_shape = []
        if self.size.sym_val is None:
            ret_shape = [get_new_symbol() for _ in range(self.x.rank)]
            return types.tensor(self.x.dtype, tuple(ret_shape))

        for idx, s in enumerate(self.size.sym_val):
            if is_symbolic(s):
                ret_shape.append(s)
            elif s != -1:
                ret_shape.append(s)
            elif self.begin.sym_val is not None:
                ret_shape.append(x_shape[idx] - self.begin.sym_val[idx])
            else:
                ret_shape.append(get_new_symbol())

        return types.tensor(self.x.dtype, tuple(ret_shape))

    @precondition(allow=VALUE | SYMBOL)
    def value_inference(self):
        if any_symbolic(self.begin.sym_val):
            return None
        if any_symbolic(self.size.sym_val):
            return None
        if self.x.val is None:
            return None
        slices = []
        for i in range(self.x.rank):
            begin_val = self.begin.val[i]
            if begin_val < 0:
                if is_symbolic(self.x.shape[i]):
                    return None
                begin_val += self.x.shape[i]
            if self.size.val[i] > 0:
                slices.append(slice(begin_val, begin_val + self.size.val[i]))
            else:
                slices.append(slice(begin_val, None, None))
        return self.x.val[tuple(slices)]


@register_op(doc_str="")
class space_to_depth(Operation):
    """
    Rearrange elements in a tensor from spatial into depth (channel) dimension.

    Parameters
    ----------
    x: tensor<[n, C, H, W], T> (Required)
        * Input tensor of rank ``4``.
    block_size: const<i32> (Required)
        * The size of the spatial block. Must be greater than ``1`` and divisible
          by spatial dimensions ``H, W``.

    Returns
    -------
    tensor<[n, C x block_size^2, H / block_size, W / block_size], T>
        * Where ``b`` is the block size.

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        block_size=IntInputType(const=True),)

    def __init__(self, **kwargs):
        super(space_to_depth, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        n, c, h, w = self.x.shape
        bs = self.block_size.val
        ret_shape = (n, c * (bs * bs), h // bs, w // bs)
        return types.tensor(x_type, ret_shape)


@register_op(doc_str="")
class squeeze(Operation):
    """
    Remove single-dimension dimensions in a 1-D or higher tensor.

    Parameters
    ----------
    x: tensor<\*?,T> (Required)
        * Must be at least 1-D.
    axes: const<K,i32> (Optional)
        * Axes to squeeze out.
        * Default to remove all single-dimensions.

    Returns
    -------
    tensor<\*(rank(x)-K),T>
        * Tensor with same type as input ``x`` and rank ``rank(x)-K``.

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        axes=IntTensorInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(
            axes=None,
            )

    def __init__(self, **kwargs):
        super(squeeze, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        x_shape = self.x.shape
        squeezed_shape = list(x_shape)
        if self.axes is None:
            # Squeeze all single-dim, assuming symbolic dims != 1
            squeezed_shape = [s for s in squeezed_shape if s != 1]
        else:
            axes = self.axes.val
            axes = [axis if axis >= 0 else axis + self.x.rank for axis in axes]
            for i in sorted(axes)[::-1]:  # descending order
                if len(squeezed_shape) <= i:
                    raise ValueError(
                        "Cannot squeeze dim {} for shape"
                        + " {}".format(i, squeezed_shape)
                    )
                squeezed_shape.pop(i)

        return types.tensor(x_type, tuple(squeezed_shape)) if len(squeezed_shape) != 0 else x_type

    @precondition(allow=VALUE)
    def value_inference(self):
        if self.x.val is None:
            return None
        if self.axes is None:
            val =  np.squeeze(self.x.val)
        else:
            val = np.squeeze(self.x.val, axis=tuple(self.axes.val))
        return val if val.shape != () else self.x.val[0]

@register_op(doc_str="")
class transpose(Operation):
    """
    Permute tensor ``x`` dimensions according to ``perm``.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)
        * Must be at least 1-D. ``x`` may have a symbolic shape.
    perm: const<[rank(x)], i32> (Required)
        * Permutation order. -rank(x) <= perm[I] < rank(x) for all perm entries.

    Returns
    -------
    tensor<\*?,T>
        * Tensor with same rank and type as ``x``.

    Attributes
    ----------
    T: fp32

    References
    ----------
    `torch.Tensor.permute <https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute>`_
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        perm=IntTensorInputType(const=True),)

    def __init__(self, **kwargs):
        super(transpose, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        perm = self.perm.val
        x_shape = np.array(self.x.shape)
        if len(perm) != self.x.rank:
            msg = "perm should have the same length as rank(x): {} != {}"
            raise ValueError(msg.format(len(perm), self.x.rank))
        if self.x.rank == 0:
            return self.x.sym_type  # scalar cannot be transposed
        if any_variadic(self.x.shape):
            ret_shape = get_new_variadic_symbol()
        else:
            ret_shape = x_shape[perm]
        return types.tensor(x_type, tuple(ret_shape))

    @precondition(allow=VALUE | SYMBOL)
    def value_inference(self):
        return np.transpose(self.x.val, axes=self.perm.val)


@register_op(doc_str="")
class pixel_shuffle(Operation):
    """
    Rearrange elements in a tensor from depth (channel) into spatial dimensions.
    Equivalent to PyTorch's ``PixelShuffle``.

    Parameters
    ----------
    x: tensor<[n, C x f^2, H, W], T> (Required)
        * Input tensor of rank ``4``.
    upscale_factor: const<i32>
        * Factor to increase spatial resolution by.

    Returns
    -------
    tensor<[n, C, H x f, W x f], T>
        * Where ``f`` is the upscale factor.

    Attributes
    ----------
    T: fp32

    References
    ----------
    `torch.nn.PixelShuffle <https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html?highlight=pixel%20shuffle#torch.nn.PixelShuffle>`_
    """

    input_spec = InputSpec(
        x=TensorInputType(), upscale_factor=IntInputType(const=True),
    )

    def __init__(self, **kwargs):
        super(pixel_shuffle, self).__init__(**kwargs)

    def type_inference(self):
        x_type = self.x.dtype
        n, c, h, w = self.x.shape
        f = self.upscale_factor.val
        ret_shape = (n, c // (f * f), h * f, w * f)
        return types.tensor(x_type, ret_shape)


@register_op(doc_str="")
class sliding_windows(Operation):
    """
    Return a tensor containing all windows of ``size``, separated by stride along the
    given ``axis``.

    Parameters
    ----------
    x: tensor<[\*d0, d_axis, *dn], T>
        * Input tensor.

    axis: const<i32>
        * Axis to perform the operation.

    size: const<i32>
        * Number of elements in the sliding window.

    stride: const<i32> Optional
        * Default to ``1``.
        * The stride of the input elements in the sliding window.

    Returns
    -------
    tensor<[\*d0, d_axis - size // stride + 1, size, \*dn], T>
        * The output will be a tensor of rank ``N+1`` where ``N`` is the input tensor
          rank.

    Attributes
    ----------
    T: fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(),
        axis=IntInputType(const=True),
        size=IntInputType(const=True),
        stride=IntInputType(const=True, optional=True),
    )

    def default_inputs(self):
        return DefaultInputs(stride=1)

    def __init__(self, **kwargs):
        super(sliding_windows, self).__init__(**kwargs)

    def type_inference(self):
        x_shape = self.x.shape
        axis = self.axis.val
        size = self.size.val
        stride = self.stride.val
        ret_shape = list(x_shape)
        ret_shape[axis] = (x_shape[axis] - size) // stride + 1
        pos_axis = axis if axis >= 0 else axis + self.x.rank
        ret_shape.insert(pos_axis + 1, size)
        return types.tensor(self.x.dtype, tuple(ret_shape))
