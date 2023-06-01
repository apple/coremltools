#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import List

from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation import (
    reshape as _reshape_iOS15,
)
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


@register_op(opset_version=_IOS17_TARGET)
class reshape(_reshape_iOS15):
    """
    Return a tensor that has the same values as ``x`` with shape ``shape``.
    ``shape`` must have the same volume (number of elements) as ``x``.

    The major difference between this version and the iOS 15 :py:class:`~.iOS15.tensor_transformation.reshape` is as follows:

    When the ``shape`` contains ``0``,
    the restriction about ``K == rank(x)`` is no longer enforced. Each ``0`` in ``shape`` will match the
    corresponding dimension in ``x.shape``, counting from the rightmost element. So ``shape[i]``
    matches ``input[j]`` if ``length(shape)-i == rank(input)-j``. If a ``0`` is out of range, assign ``1``
    (equivalent to ``expand_dims`` for ``x.shape``).

    More specifically, when ``x.shape`` is ``[2, 50]`` and ``shape`` is ``[1, 0, -1, 0]``, it will error out
    in iOS 15 or iOS 16 because ``x`` has rank ``2`` while the ``len`` of ``shape`` is ``4``. In iOS 17, the result will
    have ``shape`` ``[1, 1, 2, 50]``, because the rightmost ``0`` will be changed to the rightmost dim of
    ``x.shape``, which is ``50``. There is no other ``0`` that has a corresponding dim in ``x.shape``, so it is set
    as ``1``. Finally, the ``-1`` is calculated based on knowing dimensions that produce ``2``.

    Parameters
    ----------
    x: tensor<\*?, T> (Required)

        * An ``n-D`` tensor or a scalar.
        * If ``x`` has a fixed rank (and possibly contains symbolic dimension),
          ``shape`` may contain elements that are not positive integers (see below).
        * If ``x`` has a variadic rank, ``shape`` can only contain positive integers.

    shape: tensor<[K], i32> (Required)

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
    tensor<\*?, T>
        * Tensor with shape determined by the input shape.

    Attributes
    ----------
    T: fp16, fp32, i32, bool
    """

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
