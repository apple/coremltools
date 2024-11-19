#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import List

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType, TupleInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation import (
    expand_dims as _expand_dims_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation import (
    reshape as _reshape_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation import (
    reverse as _reverse_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation import (
    reverse_sequence as _reverse_sequence_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation import (
    slice_by_index as _slice_by_index_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation import (
    slice_by_size as _slice_by_size_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation import (
    sliding_windows as _sliding_windows_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation import (
    squeeze as _squeeze_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation import (
    transpose as _transpose_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS16.tensor_transformation import (
    reshape_like as _reshape_like_iOS16,
)
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


@register_op(opset_version=_IOS17_TARGET)
class reshape(_reshape_iOS15):
    """
    Return a tensor that has the same values as ``x`` with shape ``shape``.
    ``shape`` must have the same volume (number of elements) as ``x``.

    The major differences between this version and the iOS 15 :py:class:`~.iOS15.tensor_transformation.reshape` are as follows:

    - When the ``shape`` contains ``0``,
      the restriction about ``K == rank(x)`` is no longer enforced. Each ``0`` in ``shape`` will match the
      corresponding dimension in ``x.shape``, counting from the rightmost element. So ``shape[i]``
      matches ``input[j]`` if ``length(shape)-i == rank(input)-j``. If a ``0`` is out of range, assign ``1``
      (equivalent to ``expand_dims`` for ``x.shape``).

      More specifically, when ``x.shape`` is ``[2, 50]`` and ``shape`` is ``[1, 0, -1, 0]``, it will error out
      in iOS 15 or iOS 16 because ``x`` has rank ``2`` while the ``len`` of ``shape`` is ``4``. In iOS 17, the result will
      have ``shape`` ``[1, 1, 2, 50]``, because the rightmost ``0`` will be changed to the rightmost dim of
      ``x.shape``, which is ``50``. There is no other ``0`` that has a corresponding dim in ``x.shape``, so it is set
      as ``1``. Finally, the ``-1`` is calculated based on knowing dimensions that produce ``2``.

    - Support more data types, including int8, uint8, int16, uint16 for ``x`` and int8, int16 for
      ``shape``.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)

        * An ``n-D`` tensor or a scalar.
        * If ``x`` has a fixed rank (and possibly contains symbolic dimension),
          ``shape`` may contain elements that are not positive integers (see below).
        * If ``x`` has a variadic rank, ``shape`` can only contain positive integers.

    shape: tensor<[K], U> (Required)

        A 1-D tensor, with elements from the following:

            * Positive integers.
            * Symbols: All but one symbol in ``shape`` must be present in ``x.shape``.
              The new symbol that is not present in ``x.shape`` represents a dimension
              such that the total size remains constant. Symbol is illegal
              if ``x`` has a variadic rank.
            * ``-1``: ``-1`` introduces a new symbol (see Symbols). Therefore, ``-1`` is
              allowed if all symbols in the ``shape`` appear in ``x.shape``. ``-1`` is illegal
              if ``x`` has a variadic rank.
            * ``0``: It will match the corresponding dimension in ``x.shape``. See the previous
              description of different behaviors with iOS 17.

    Returns
    -------
    tensor<\\*?, T>
        * Tensor with shape determined by the input shape.

    Attributes
    ----------
    T: fp16, fp32, int8, uint8, int16, uint16, int32, bool
    U: int8, int16, int32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        shape=TensorInputType(type_domain="U"),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.uint8,
            types.int16,
            types.uint16,
            types.int32,
            types.bool,
        ),
        "U": (types.int8, types.int16, types.int32),
    }

    @staticmethod
    def replace_zeros_in_shape(from_shape: List[int], to_shape: List[int]) -> List[int]:
        """
        Replaces 0s in `to_shape` by the corresponding dims in `from_shape`.

        Overrides IOS15's method to demonstrate IOS17's different behaviours.
        """
        if to_shape.count(0):
            # To do right alignment, we reverse the input and do left alignment instead.
            from_shape_reversed = from_shape[::-1]
            to_shape_reversed = to_shape[::-1]
            for idx, to_element in enumerate(to_shape_reversed):
                if to_element == 0:
                    to_shape_reversed[idx] = (
                        from_shape_reversed[idx] if idx < len(from_shape_reversed) else 1
                    )
            # Reverse the result back to make the right alignment.
            to_shape = to_shape_reversed[::-1]
        return to_shape


@register_op(opset_version=_IOS17_TARGET)
class reshape_like(_reshape_like_iOS16):
    """
    Reshape a tensor to an output shape specified by some or all dimensions of a tuple of reference tensors ``ref_tensors``.

    The major difference between this version and the iOS 15 :py:class:`~.iOS16.tensor_transformation.reshape_like`
    is that input ``x`` and ``ref_tensors`` supports more data types: int8, uint8, int16, uint16.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
        * The input tensor to be reshaped.

    ref_tensors: Tuple[tensor<\\*?, R>] (Required)
        * A tuple of tensors that define the output shape.

    begins: Tuple[const<int32>] (Required)
        * A tuple of integers specifying the begin index into the shape vector of the corresponding ``ref_tensor``.

    ends: Tuple[const<int32>] (Required)
        * A tuple of integers specifying the end index into the shape vector of the corresponding ``ref_tensor``.

    end_masks: Tuple[const<bool>] (Required)
        * If ``True``, select all axes from the begin index until the end of the corresponding ``ref_tensor``, as in
          ``ref_tensors[i].shape[begins[i]:]``.

    Returns
    -------
    tensor<\\*?, T>
        * Same type as input tensor ``x``.
        * Output shape is computed by ``ref_tensors``, ``begins``, ``ends``, and ``end_masks``.

    Attributes
    ----------
    T: fp16, fp32, int8, int16, int32, uint8, uint16, bool
    R: fp16, fp32, int8, int16, int32, uint8, uint16, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        ref_tensors=TupleInputType(),
        begins=TupleInputType(),
        ends=TupleInputType(),
        end_masks=TupleInputType(),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.int16,
            types.int32,
            types.uint8,
            types.uint16,
            types.bool,
        ),
    }


@register_op(opset_version=_IOS17_TARGET)
class expand_dims(_expand_dims_iOS15):
    """
    Insert a single-dimension in a 1-D or higher tensor at each axis in axes.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.tensor_transformation.expand_dims`
    is that input ``x`` supports more data types: int8, uint8, int16, uint16.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
        * Scalar or tensor.
    axes: const tensor<[K], int32> Required
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
    T: fp16, fp32, int8, int16, int32, uint8, uint16, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axes=TensorInputType(const=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.int16,
            types.int32,
            types.uint8,
            types.uint16,
            types.bool,
        ),
    }


@register_op(opset_version=_IOS17_TARGET)
class squeeze(_squeeze_iOS15):
    """
    Remove single-dimension dimensions in a 1-D or higher tensor.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.tensor_transformation.squeeze`
    is that input ``x`` supports more data types: int8, uint8, int16, uint16.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
        * Must be at least 1-D.
    axes: const<K,int32> (Optional)
        * Axes to squeeze out.
        * Default to remove all single-dimensions.

    Returns
    -------
    tensor<\\*(rank(x)-K), T>
        * Tensor with same type as input ``x`` and rank ``rank(x)-K``.

    Attributes
    ----------
    T: fp16, fp32, int8, int16, int32, uint8, uint16, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axes=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.int16,
            types.int32,
            types.uint8,
            types.uint16,
            types.bool,
        ),
    }


@register_op(opset_version=_IOS17_TARGET)
class reverse(_reverse_iOS15):
    """
    Reverse the order of the input tensor ``x`` along specified ``axes`` (dimensions).

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.tensor_transformation.reverse`
    is that input ``x`` supports more data types: int8, uint8, int16, uint16.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
        * Input tensor.

    axes: const<D, int32> (Optional)
        * Dimension(s) to reverse. Each axis must be in the range ``[-rank(x), rank(x))``.
        * Defaults to None (reverse on all dimensions).

    Returns
    -------
    tensor<\\*?, T>
        * Same type and shape as the input tensor.

    Attributes
    ----------
    T: fp16, fp32, int8, int16, int32, uint8, uint16, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axes=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.int16,
            types.int32,
            types.uint8,
            types.uint16,
            types.bool,
        ),
    }


@register_op(opset_version=_IOS17_TARGET)
class reverse_sequence(_reverse_sequence_iOS15):
    """
    Reverse variable length slices for specified axes / dimensions of the input
    tensor. This op first slices input tensor along the ``batch_axis`` dimension, then
    partially reverses the elements along the ``seq_axis`` for the first ``lengths[i]``
    elements.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.tensor_transformation.reverse_sequence`
    is that input supports more data types:
    - ``x`` additionally supports int8, uint8, int16, uint16
    - ``lengths`` additionally supports int8, int16

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
        * Input tensor.
    lengths: tensor<L, U> (Required)
        * 1-dimensional tensor of length ``x.shape[batch_axis]`` specifying the length
          of the sequence to reverse.
        * Values must be in range ``[0, x.shape[seq_axis]]``.
    seq_axis: const<int32> (Optional)
        * The dimension to reverse.
        * Defaults to ``0``.
    batch_axis: const<int32> (Optional)
        * Dimension for slicing.
        * Defaults to ``0``.

    Returns
    -------
    tensor<\\*?, T>
        * Same type and shape as the input tensor.

    Attributes
    ----------
    T: fp16, fp32, int8, int16, int32, uint8, uint16, bool
    U: int8, int16, int32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        lengths=TensorInputType(type_domain="U"),
        seq_axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        batch_axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.int16,
            types.int32,
            types.uint8,
            types.uint16,
            types.bool,
        ),
        "U": (types.int8, types.int16, types.int32),
    }


@register_op(opset_version=_IOS17_TARGET)
class sliding_windows(_sliding_windows_iOS15):
    """
    Return a tensor containing all windows of ``size``, separated by stride along the
    given ``axis``.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.tensor_transformation.sliding_windows`
    is that input ``x`` supports more data types: int8, uint8, int16, uint16.

    Parameters
    ----------
    x: tensor<[\\*d0, d_axis, *dn], T>
        * Input tensor.

    axis: const<int32>
        * Axis to perform the operation.

    size: const<int32>
        * Number of elements in the sliding window.

    stride: const<int32> Optional
        * Default to ``1``.
        * The stride of the input elements in the sliding window.

    Returns
    -------
    tensor<[\\*d0, d_axis - size // stride + 1, size, \\*dn], T>
        * The output will be a tensor of rank ``N+1`` where ``N`` is the input tensor
          rank.

    Attributes
    ----------
    T: fp16, fp32, int8, int16, int32, uint8, uint16, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        axis=TensorInputType(const=True, type_domain=types.int32),
        size=TensorInputType(const=True, type_domain=types.int32),
        stride=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.int16,
            types.int32,
            types.uint8,
            types.uint16,
            types.bool,
        ),
    }


@register_op(opset_version=_IOS17_TARGET)
class transpose(_transpose_iOS15):
    """
    Permute tensor ``x`` dimensions according to ``perm``.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.tensor_transformation.transpose`
    is that input ``x`` supports more data types: int8, uint8, int16, uint16.

    Parameters
    ----------
    x: tensor<\\*?, T> (Required)
        * Must be at least 1-D. ``x`` may have a symbolic shape.
    perm: const<[rank(x)], i32> (Required)
        * Permutation order. -rank(x) <= perm[I] < rank(x) for all perm entries.

    Returns
    -------
    tensor<\\*?,T>
        * Tensor with same rank and type as ``x``.

    Attributes
    ----------
    T: fp16, fp32, int8, int16, int32, uint8, uint16, bool
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        perm=TensorInputType(const=True, type_domain=types.int32),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.int16,
            types.int32,
            types.uint8,
            types.uint16,
            types.bool,
        ),
    }


@register_op(opset_version=_IOS17_TARGET)
class slice_by_index(_slice_by_index_iOS15):
    """
    Method for numpy style indexing and slicing.
    With a tensor ``x``, this method achieves the following:

    ``result = x[begin[0]: end[0]: stride[0], begin[1]: end[1]: stride[1], ...]``

    The differences between this version and the iOS 15 :py:class:`~.iOS15.tensor_transformation.slice_by_index`
    is that additional data types are supported for ``x``, ``begin``, ``end``, and ``stride``.
    See Parameters and Attributes sections for details.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
        * Input tensor
    begin: tensor<[rank(x)], U> (Required)
        * Starting index for the dimension of slicing.
    end: tensor<[rank(x)], U> (Required)
        * Ending index for the dimension of slicing.
    stride: tensor<[rank(x)], U> (Optional)
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
        * If ``squeeze_mask[i]==True``, ignores ``end[i]``, and do the pure index at ``begin[i]``.

    Returns
    -------
    tensor<\\*?, T>
        - Scalar or tensor.

    Attributes
    ----------
    T: bool, fp16, fp32, int8, int16, int32, uint8, uint16
    U: int8, int16, int32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        begin=TensorInputType(type_domain="U"),
        end=TensorInputType(type_domain="U"),
        stride=TensorInputType(const=True, optional=True, type_domain="U"),
        begin_mask=TensorInputType(const=True, optional=True, type_domain=types.bool),
        end_mask=TensorInputType(const=True, optional=True, type_domain=types.bool),
        squeeze_mask=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.int16,
            types.int32,
            types.uint8,
            types.uint16,
            types.bool,
        ),
        "U": (types.int8, types.int16, types.int32),
    }


@register_op(opset_version=_IOS17_TARGET)
class slice_by_size(_slice_by_size_iOS15):
    """
    Slice input tensor starting from the given ``begin`` index and by
    the amount specified by the ``size`` input, for each dimension.

    The differences between this version and the iOS 15 :py:class:`~.iOS15.tensor_transformation.slice_by_size`
    is that additional data types are supported for ``x``, ``begin``, and ``size``.
    See Parameters and Attributes sections for details.

    Parameters
    ----------
    x: tensor<*?, T> (Required)
        * Input tensor.
    begin: tensor<[rank(x)], U> Required
        * The begin index for slice.
    size: tensor<[rank(x)], U> Required
        * The size that is to be sliced. If ``size`` is ``-1``,
          all the remaining elements starting with "begin" are sliced.

    Returns
    -------
    tensor<\\*?, T>
        * Scalar or tensor.

    Attributes
    ----------
    T: bool, fp16, fp32, int8, int16, int32, uint8, uint16
    U: int8, int16, int32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        begin=TensorInputType(type_domain="U"),
        size=TensorInputType(type_domain="U"),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.int16,
            types.int32,
            types.uint8,
            types.uint16,
            types.bool,
        ),
        "U": (types.int8, types.int16, types.int32),
    }
