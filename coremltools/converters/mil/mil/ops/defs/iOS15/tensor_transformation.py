#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import List

import numpy as np
import sympy as sm

from coremltools import _logger as logger
from coremltools.converters.mil.mil import (
    Operation,
    get_new_symbol,
    get_new_variadic_symbol,
    precondition,
    types,
)
from coremltools.converters.mil.mil.input_type import DefaultInputs, InputSpec, TensorInputType
from coremltools.converters.mil.mil.operation import SYMBOL, VALUE
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs._utils import (
    get_param_val,
    get_squeeze_axes,
    solve_slice_by_index_shape,
    solve_slice_by_index_slice,
)
from coremltools.converters.mil.mil.types.symbolic import (
    any_symbolic,
    any_variadic,
    is_symbolic,
    isscalar,
)


@register_op
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
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        block_size=TensorInputType(const=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        x_type = self.x.dtype
        n, c, h, w = self.x.shape
        bs = self.block_size.val
        ret_shape = (n, c // (bs * bs), h * bs, w * bs)
        return types.tensor(x_type, ret_shape)


@register_op
class expand_dims(Operation):
    """
    Insert a single-dimension in a 1-D or higher tensor at each axis in axes.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
        * Scalar or tensor.
    axes: const tensor<[K], i32> Required
        * ``K`` is the number of dimensions expanded.
        * Insert single dimension at dimension index at each axes.
        * Negative value to index from the end. ``-d-1 <= axis <= d``
          where ``d`` is the rank of ``x``.

    Returns
    -------
    tensor<\\*(rank(x)+K), T>
        * Same type as the input ``x`` with rank ``rank(x)+K``.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axes=TensorInputType(const=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

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


@register_op
class reshape(Operation):
    """
    Return a tensor that has the same values as ``x`` with shape ``shape``.
    ``shape`` must have the same volume (number of elements) as ``x``.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)

        * An n-D tensor or a scalar.
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
            * ``-1``: ``-1`` introduces a new symbol (see Symbols above). Therefore, ``-1`` is
              allowed if all symbols in the shape appear in ``x.shape``. ``-1`` is illegal
              if ``x`` is variadic rank.
            * ``0``: If ``K == rank(x)`` then ``0`` means inheriting from the corresponding
              dimension in ``x.shape``. ``0`` is illegal if ``x`` is variadic rank.

    Returns
    -------
    tensor<\\*?, T>
        * Tensor with shape determined by the input shape.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        shape=TensorInputType(type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

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
        count_neg_one = np.count_nonzero(self.shape.sym_val == -1)
        if count_neg_one > 1:
            raise ValueError(
                f"Reshape op supports only one dimension to be -1, "
                f"but got {count_neg_one} dimensions be -1."
            )

        if not any_symbolic(self.x.shape) and self.shape.val is not None:
            ret_shape = self._infer_shape_static()
        else:
            ret_shape = self._infer_shape_dynamic()

        ret_val = None
        if self.x.sym_val is not None and all(
            isscalar(a) and not is_symbolic(a) for a in ret_shape
        ):
            ret_val = reshape_with_symbol(self.x.sym_val, ret_shape)
        return types.tensor(self.x.dtype, tuple(ret_shape)), ret_val

    @staticmethod
    def replace_zeros_in_shape(from_shape: List[int], to_shape: List[int]) -> List[int]:
        """Replaces 0s in `to_shape` by the corresponding dims in `from_shape`."""
        if to_shape.count(0):
            if len(from_shape) != len(to_shape):
                raise ValueError(
                    f"When there is 0 in shape, the rank of x ({len(from_shape)}) "
                    f"must equal to the target shape len ({len(to_shape)})."
                )
            to_shape = [s if s != 0 else from_shape[dim] for dim, s in enumerate(to_shape)]
        return to_shape

    @staticmethod
    def replace_neg_one_in_shape(from_shape: List[int], to_shape: List[int]) -> List[int]:
        """Replaces -1 in `to_shape` by the corresponding dims in `from_shape`."""
        if to_shape.count(-1):
            neg_one_idx = to_shape.index(-1)
            total_element_num = np.prod(from_shape)
            remain_element_num = np.prod(
                [dim for idx, dim in enumerate(to_shape) if idx != neg_one_idx]
            )
            infer_dim = total_element_num // remain_element_num
            to_shape[neg_one_idx] = infer_dim
        return to_shape

    def _infer_shape_static(self):
        from_shape = list(self.x.shape)
        to_shape = list(self.shape.val)
        to_shape = self.replace_zeros_in_shape(from_shape, to_shape)
        to_shape = self.replace_neg_one_in_shape(from_shape, to_shape)
        if np.prod(from_shape) != np.prod(to_shape):
            raise ValueError(
                f"Invalid target shape in `reshape` op ({from_shape} to {list(self.shape.val)})."
            )
        return to_shape

    def _infer_shape_dynamic(self):
        x_vol = np.prod(self.x.shape)
        # shape is const, and thus sym_val is not None
        sym_shape = self.shape.sym_val
        sym_shape = [get_new_symbol() if d == -1 else d for d in sym_shape]
        try:
            ret_shape = reshape.enforce_volumetric_constraint(x_vol, sym_shape)
        except:
            ret_shape = sym_shape
        return ret_shape

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


@register_op
class reverse(Operation):
    """
    Reverse the order of the input tensor ``x`` along specified ``axes`` (dimensions).

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
        * Input tensor.

    axes: const<D, i32> (Optional)
        * Dimension(s) to reverse. Each axis must be in the range ``[-rank(x), rank(x))``.
        * Defaults to None (reverse on all dimensions).

    Returns
    -------
    tensor<\\*?, T>
        * Same type and shape as the input tensor.

    Attributes
    ----------
    T: fp16, fp32, i32, bool

    References
    ----------
    See `tf.reverse <https://www.tensorflow.org/api_docs/python/tf/reverse>`_
    and `TORCH <https://pytorch.org/docs/stable/torch.html#torch.flip>`_.
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axes=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def default_inputs(self):
        return DefaultInputs(
            axes=None,
            )

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        res = self.x.val
        axes = self.axes.val if self.axes is not None else range(self.x.rank)
        for axis in axes:
            res = np.flip(res, axis=axis)
        return res


@register_op
class reverse_sequence(Operation):
    """
    Reverse variable length slices for specified axes / dimensions of the input
    tensor. This op first slices input tensor along the ``batch_axis`` dimension, then
    partially reverses the elements along the ``seq_axis`` for the first ``lengths[i]``
    elements.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
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
    tensor<\\*?, T>
        * Same type and shape as the input tensor.

    Attributes
    ----------
    T: fp16, fp32, i32, bool

    References
    ----------
    `tf.reverse_sequence <https://www.tensorflow.org/api_docs/python/tf/reverse_sequence>`_

    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        lengths=TensorInputType(type_domain=types.int32),
        seq_axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        batch_axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def default_inputs(self):
        return DefaultInputs(
            seq_axis=0,
            batch_axis=0)

    def type_inference(self):
        return self.x.sym_type

    @precondition(allow=VALUE)
    def value_inference(self):
        raise NotImplementedError("TODO")


@register_op
class slice_by_index(Operation):
    """
    Method for numpy style indexing and slicing.
    With a tensor ``x``, this method achieves the following:

    ``result = x[begin[0]: end[0]: stride[0], begin[1]: end[1]: stride[1], ...]``

    Note: This method does not support pure indexing. You would need to do a
    squeeze if indexing is intended.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
        * Input tensor
    begin: tensor<[rank(x)], i32> (Required)
        * Starting index for the dimension of slicing.
    end: tensor<[rank(x)], i32> (Required)
        * Ending index for the dimension of slicing.
    stride: tensor<[rank(x)], i32> (Optional)
        * Default is all ``1``.
        * Stride for the dimension of slicing.
    begin_mask: tensor<[rank(x)], bool> (Optional)
        * Default to all ``False``.
        * If ``begin_mask[i]==True``, ignores ``begin[i]``, and set ``begin[i]`` to ``0``.
    end_mask: tensor<[rank(x)], bool> (Optional)
        * Default to all ``False``.
        * If ``end_mask[i]==True``, ignores ``end[i]``, and set ``end[i]`` to ``x.shape[i]``.
    squeeze_mask: tensor<[rank(x)], bool> (Optional)
        * Default to all ``False``.
        * If ``squeeze_mask[i]==true``, ignores ``end[i]``, and do the pure index at ``begin[i]``.

    Returns
    -------
    tensor<\\*?, T>
        - Scalar or tensor.

    Attributes
    ----------
    T: fp16, fp32, i32, bool

    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        begin=TensorInputType(type_domain=types.int32),
        end=TensorInputType(type_domain=types.int32),
        stride=TensorInputType(const=True, optional=True, type_domain=types.int32),
        begin_mask=TensorInputType(const=True, optional=True, type_domain=types.bool),
        end_mask=TensorInputType(const=True, optional=True, type_domain=types.bool),
        squeeze_mask=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def default_inputs(self):
        return DefaultInputs(
            stride=None,
            begin_mask=None,
            end_mask=None,
            squeeze_mask=None,
            )

    def type_inference(self):
        # solve shape
        ret_shape = solve_slice_by_index_shape(
            self.x.shape,
            self.begin.val,
            self.end.val,
            get_param_val(self.stride),
            get_param_val(self.begin_mask),
            get_param_val(self.end_mask),
            get_param_val(self.squeeze_mask),
        )

        if len(ret_shape) == 0:
            # Scalar case.
            return self.x.dtype
        else:
            return types.tensor(self.x.dtype, tuple(ret_shape))

    def value_inference(self):
        if self.x.sym_val is None or self.begin.val is None or self.end.val is None:
            return None

        # solve the data slices and slice tensor
        slices = solve_slice_by_index_slice(
            self.x.shape,
            self.begin.val,
            self.end.val,
            get_param_val(self.stride),
            get_param_val(self.begin_mask),
            get_param_val(self.end_mask),
            get_param_val(self.squeeze_mask),
        )
        res = self.x.sym_val[slices]

        # remove squeeze_axes
        squeeze_axes = get_squeeze_axes(get_param_val(self.squeeze_mask), self.x.rank)
        if len(squeeze_axes) > 0:
            if len(squeeze_axes) == len(res.shape):
                if len(res) == 0:
                    logger.warning("%s seems to be a 0 sized tensor", self.name)
                    return np.array([])
                res = np.squeeze(res).tolist()
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


@register_op
class slice_by_size(Operation):
    """
    Slice input tensor starting from the given ``begin`` index and by
    the amount specified by the ``size`` input, for each dimension.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
        * Input tensor.
    begin: tensor<[rank(x)], i32> (Required)
        * The begin index for slice.
    size: tensor<[rank(x)], i32> (Required)
        * The size that is to be sliced. If ``size`` is ``-1``,
          all the remaining elements starting with "begin" are sliced.

    Returns
    -------
    tensor<\\*?, T>
        * Scalar or tensor.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        begin=TensorInputType(type_domain=types.int32),
        size=TensorInputType(type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

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


@register_op
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
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        block_size=TensorInputType(const=True, type_domain=types.int32)
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        x_type = self.x.dtype
        n, c, h, w = self.x.shape
        bs = self.block_size.val
        ret_shape = (n, c * (bs * bs), h // bs, w // bs)
        return types.tensor(x_type, ret_shape)

@register_op
class space_to_batch(Operation):
    """
    Rearrange elements in a tensor from spatial into batch dimensions.

    Parameters
    ----------
    x: tensor<[n, C, H, W], T> (Required)
        * Input tensor must have rank ``4``.
        * The first and the second dimension are batch, channel; respectively.
        * The remaining dimensions ``(H, W)`` are treated as "spatial dimensions".
    block_shape: const tensor<[2], i32> (Required)
        * The length of the ``block_shape`` must be ``2``.
        * It defines the shapes of the block in which the spatial dimensions are divided.
    paddings: const tensor<[2, 2], i32> (Required)
        * It must have shape ``(2, 2)``.
        * It defines the padding for each spatial dimension.

    Returns
    -------
    tensor<[new_n, C, new_H, new_W], T>
        * ``new_n = n * block_shape[0] * block_shape[1]``
        * ``new_H = (H + paddings[0][0] + padding[0][1])/block_shape[0]``
        * ``new_W = (W + paddings[1][0] + padding[1][1])/block_shape[1]``
        * The output has the same rank as the input.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        block_shape=TensorInputType(const=True, type_domain=types.int32),
        paddings=TensorInputType(const=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        x_shape = self.x.shape
        block_shape = self.block_shape.val
        paddings = self.paddings.val

        if self.x.rank != 4:
            msg = "Input to space_to_batch op must be rank 4. Instead got an input with rank {}".format(self.x.rank)
            raise ValueError(msg)

        if paddings.shape != (block_shape.shape[0], 2):
            msg = "block_shape and paddings must have shape [2], [2, 2] accordingly in the space_to_batch op. "\
            "Got {}, {}.".format(block_shape.shape, paddings.shape)
            raise ValueError(msg)

        m = block_shape.shape[0]
        if m != 2:
            msg = "space_to_batch op only supports spatial dimensions = 2. Got {}".format(m)
            raise ValueError(msg)

        b = x_shape[0]
        c = x_shape[1]
        spatial_shape = x_shape[2:2+m]

        if self.x.rank != m + 2:
            raise ValueError("The input rank of space_to_batch op must exactly be " \
                             "len(block_shape){} + 2! Got {}".format(self.block_shape.val, self.x.rank))

        padded_spatial_shape = [x + paddings[i][0] + paddings[i][1] for i, x in enumerate(spatial_shape)]
        new_b = b * np.prod(block_shape)
        new_spatial_shape = [padded_spatial_shape[i]/block_shape[i] for i in range(m)]
        ret_shape = [new_b, c] + new_spatial_shape
        x_type = self.x.dtype

        return types.tensor(x_type, ret_shape)

@register_op
class batch_to_space(Operation):
    """
    Rearrange elements in a tensor from batch into spatial dimensions.

    Parameters
    ----------
    x: tensor<[n, C, H, W], T> (Required)
        * Input tensor must have rank ``4``.
        * The first and the second dimension are batch, channel; respectively.
        * The remaining dimensions ``(H, W)`` are treated as "spatial dimensions".
    block_shape: const tensor<[2], i32> (Required)
        * The length of the ``block_shape`` must be ``2``.
        * It defines the shapes of the block in which the spatial dimensions are multiplied.
    crops: const tensor<[2, 2], i32> (Required)
        * It must have shape ``(2, 2)``.
        * It defines the amount to crop from each spatial dimension.

    Returns
    -------
    tensor<[new_n, C, new_H, new_W], T>
        * ``new_n = n / (block_shape[0] * block_shape[1])``
        * ``new_H = (H * block_shape[0]) - paddings[0][0] - padding[0][1]``
        * ``new_W = (W * block_shape[1]) - paddings[1][0] - padding[1][1]``
        * The output has the same rank as the input.

    Attributes
    ----------
    T: fp16, fp32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        block_shape=TensorInputType(const=True, type_domain=types.int32),
        crops=TensorInputType(const=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        x_shape = self.x.shape
        block_shape = self.block_shape.val
        crops = self.crops.val

        if self.x.rank != 4:
            msg = "Input to batch_to_space op must be rank 4. Instead got an input with rank {}".format(self.x.rank)
            raise ValueError(msg)

        if crops.shape != (block_shape.shape[0], 2):
            msg = "block_shape and crops must have shape [2], [2, 2] accordingly in the batch_to_space op. "\
            "Got {}, {}.".format(block_shape.shape, crops.shape)
            raise ValueError(msg)

        m = block_shape.shape[0]
        if m != 2:
            msg = "batch_to_space op only supports spatial dimensions = 2. Got {}".format(m)
            raise ValueError(msg)

        b = x_shape[0]
        c = x_shape[1]
        spatial_shape = x_shape[2:2+m]

        if self.x.rank != m + 2:
            raise ValueError("The input rank of batch_to_space op must exactly be " \
                             "len(block_shape){} + 2! Got {}".format(self.block_shape.val, self.x.rank))

        if not is_symbolic(b) and  b % np.prod(block_shape) != 0:
            msg = ("Batch size must be perfectly divided by the product of block_shape. Got batch size {}, and block_shape {}."
            ).format(b, block_shape)
            raise ValueError(msg)

        new_b = b / np.prod(block_shape)
        new_spatial_shape = [spatial_shape[i] * block_shape[i] for i in range(m)]
        cropped_spatial_shape = [x - crops[i][0] - crops[i][1] for i, x in enumerate(new_spatial_shape)]
        ret_shape = [new_b, c] + cropped_spatial_shape
        x_type = self.x.dtype

        return types.tensor(x_type, ret_shape)

@register_op
class squeeze(Operation):
    """
    Remove single-dimension dimensions in a 1-D or higher tensor.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
        * Must be at least 1-D.
    axes: const<K,i32> (Optional)
        * Axes to squeeze out.
        * The behaviour of squeezing non-single dimensions follow PyTorch instead of NumPy, where
          it ignores non-single dimensions instead of erroring out. More specifically, if x has
          shape (2, 3, 4) and axes is [0, 1], the output will be a tensor with shape (2, 3, 4).
        * Default to remove all single-dimensions.

    Returns
    -------
    tensor<\\*(rank(x)-K), T>
        * Tensor with same type as input ``x`` and rank ``rank(x)-K``.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axes=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

    def default_inputs(self):
        return DefaultInputs(
            axes=None,
            )

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
                        f"Invalid axis {i} in squeeze. The axis should be smaller than {len(squeezed_shape)}"
                    )
                if squeezed_shape[i] == 1:
                    # Only remove the dim_size=1 dimension.
                    squeezed_shape.pop(i)

        return types.tensor(x_type, tuple(squeezed_shape)) if len(squeezed_shape) != 0 else x_type

    @precondition(allow=VALUE)
    def value_inference(self):
        if self.x.val is None:
            return None
        if self.axes is None:
            val = np.squeeze(self.x.val)
        else:
            val = np.squeeze(self.x.val, axis=tuple(self.axes.val))
        return val if val.shape != () else self.x.val[0]

@register_op
class transpose(Operation):
    """
    Permute tensor ``x`` dimensions according to ``perm``.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
        * Must be at least 1-D. ``x`` may have a symbolic shape.
    perm: const<[rank(x)], i32> (Required)
        * Permutation order. -rank(x) <= perm[I] < rank(x) for all perm entries.

    Returns
    -------
    tensor<\\*?, T>
        * Tensor with same rank and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32, bool

    References
    ----------
    `torch.Tensor.permute <https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute>`_
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        perm=TensorInputType(const=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32, types.bool),
    }

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

    @precondition(allow=VALUE)
    def value_inference(self):
        return np.transpose(self.x.val, axes=self.perm.val)


@register_op
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
    T: fp16, fp32

    References
    ----------
    `torch.nn.PixelShuffle <https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html?highlight=pixel%20shuffle#torch.nn.PixelShuffle>`_
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        upscale_factor=TensorInputType(const=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32),
    }

    def type_inference(self):
        x_type = self.x.dtype
        n, c, h, w = self.x.shape
        f = self.upscale_factor.val
        ret_shape = (n, c // (f * f), h * f, w * f)
        return types.tensor(x_type, ret_shape)


@register_op
class sliding_windows(Operation):
    """
    Return a tensor containing all windows of ``size``, separated by stride along the
    given ``axis``.

    Parameters
    ----------
    x: tensor<[\\*d0, d_axis, *dn], T>
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
    tensor<[\\*d0, d_axis - size // stride + 1, size, \\*dn], T>
        * The output will be a tensor of rank ``N+1`` where ``N`` is the input tensor
          rank.

    Attributes
    ----------
    T: fp16, fp32, int32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axis=TensorInputType(const=True, type_domain=types.int32),
        size=TensorInputType(const=True, type_domain=types.int32),
        stride=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
    }

    def default_inputs(self):
        return DefaultInputs(stride=1)

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
