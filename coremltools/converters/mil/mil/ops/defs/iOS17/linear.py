#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS15.linear import linear as _linear_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS15.linear import matmul as _matmul_iOS15
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


@register_op(opset_version=_IOS17_TARGET)
class linear(_linear_iOS15):
    """
    A version of ``linear`` for iOS 17+. The only difference between this version and the
    iOS 15 :py:class:`~.iOS15.linear.linear` is that the ``weight`` and ``bias`` may have a
    different dtype than the input/output.

    Parameters
    ----------
    x: tensor<[\*D, D_in], T> (Required)
        * ``1 <= rank <= 3``.
        * ``0 <= rank(*D) <= 2``.
    weight: const tensor<[D_out, D_in], U> (Required)
    bias: const tensor<[D_out], U> (Optional)
        * Default to ``0``.

    Returns
    -------
    tensor<[\*D, D_out], T>
        * Same rank as the input ``x``.

    Attributes
    ----------
    T: fp16, fp32, i32
    U: fp16, fp32, i32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        weight=TensorInputType(const=True, type_domain="U"),
        bias=TensorInputType(const=True, optional=True, type_domain="U"),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
        "U": (types.fp16, types.fp32, types.int32),
    }


@register_op(opset_version=_IOS17_TARGET)
class matmul(_matmul_iOS15):
    """
    A version of ``matmul`` for iOS 17+. The only difference between this version and the
    iOS 15 :py:class:`~.iOS15.linear.matmul` is that the ``x`` and ``y`` can have a different
    dtypes when one of them is const.

    Parameters
    ----------
    x: tensor<[\*, K1], T> (Required)
        * ``x`` must be 1-D or higher.
    y: tensor<[\*, K2], U> (Required)
        * ``y`` must be 1-D or higher.
    transpose_x: const bool (Optional)
        * Default to ``False``.
        * Use ``True`` to transpose the last two dimensions of ``x`` before multiplication.
          It has no effect when ``x`` is 1-D.
    transpose_y: const bool (Optional)
        * Default to ``False``.
        * Use ``True`` to transpose the last two dimensions of ``y`` before multiplication.
          It has no effect when ``y`` is 1-D.

    Returns
    -------
    tensor<\*, V>
        * Scalar or tensor output.
        * When ``x`` and ``y`` are both const or both non-const, it should follow ios15 behavior
          that ``x``, ``y``, and ``output`` all have the same dtype.
          When one of x and y is const, the output dtype should be the same as the non-const one.

    Attributes
    ----------
    T: fp16, fp32, i32
    U: fp16, fp32, i32
    """

    input_spec = InputSpec(
        x=TensorInputType(type_domain="T"),
        y=TensorInputType(type_domain="U"),
        transpose_x=TensorInputType(const=True, optional=True, type_domain=types.bool),
        transpose_y=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (types.fp16, types.fp32, types.int32),
        "U": (types.fp16, types.fp32, types.int32),
    }

    def type_inference(self):
        x_is_const = self.x.op is not None and self.x.op.op_type == "const"
        y_is_const = self.y.op is not None and self.y.op.op_type == "const"

        if x_is_const == y_is_const and self.x.dtype != self.y.dtype:
            is_const_str = "const" if x_is_const else "non-const"
            raise ValueError(
                f'In op "matmul", when x and y are both {is_const_str}, their dtype '
                f"need to match, but got x as {types.builtin_to_string(self.x.dtype)} "
                f"and y as {types.builtin_to_string(self.y.dtype)}"
            )

        inferred_type = super().type_inference()
        if x_is_const != y_is_const:
            # The output dtype should be the same as the non-const one.
            output_dtype = self.x.dtype if y_is_const else self.y.dtype
            inferred_type = types.tensor(output_dtype, inferred_type.get_shape())

        return inferred_type
