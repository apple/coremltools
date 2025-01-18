# Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Optional, Tuple

import numpy as np

from coremltools.converters.mil.mil import Var, types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.operation import VALUE, Operation, precondition
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET
from coremltools.optimize import _utils as optimize_utils


def _check_scale_zp_shapes(input_data, scale, zero_point, axis):
    def assert_vector_size_same_as_axial_dimension(param, axis_dim_size, name):
        if param.rank == 1 and param.shape[0] != axis_dim_size:
            raise ValueError(
                "Parameter {}, if vector, needs to have same size as the dimension size along the parameter input".format(
                    name
                )
            )

    if scale.rank == 0:
        # ios17.dequantize doesn't want axis defined for scalar quant params.
        if axis is not None:
            raise ValueError("axis should not be provided to quantize if scale/zp are scalars")
        if zero_point is not None and zero_point.rank != 0:
            raise ValueError("zero_point should be a scalar if scale is a scalar")
    elif scale.rank == 1:
        if axis is None or axis.val is None:
            raise ValueError("axis should be provided to quantize if scale/zp are not scalars")
        if axis.val < -input_data.rank or axis.val >= input_data.rank:
            raise ValueError(
                "Parameter axis needs to be in the range -input.rank <= axis < input.rank"
            )

        input_axis_dim_size = input_data.shape[axis.val]
        assert_vector_size_same_as_axial_dimension(scale, input_axis_dim_size, "scale")
        if zero_point is not None:
            if zero_point.rank != 1:
                raise ValueError("zero_point should be a vector if scale is a vector")
            assert_vector_size_same_as_axial_dimension(
                zero_point, input_axis_dim_size, "zero_point"
            )
    else:
        raise ValueError("Params scale & zero_point should both be scalars or vectors")


def _prepare_scale_and_zero_point(
    input_data: Var, scale: Var, zero_point: Optional[Var], axis: Optional[Var]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    input_data = input_data.val
    scale = scale.val
    if zero_point is not None:
        zero_point = zero_point.val
    if axis is not None:
        axis = axis.val

    scale = optimize_utils.promote_rank_to_same_as_data(scale, input_data, axis)
    if zero_point is not None:
        zero_point = optimize_utils.promote_rank_to_same_as_data(zero_point, input_data, axis)
    return scale, zero_point


@register_op(opset_version=_IOS17_TARGET)
class quantize(Operation):
    """
    Performs affine/linear quantization on an input tensor.

    The original data comes from the first "input".
    The other parameters -- ``scale``, ``zero_point``, and ``axis`` -- describe how
    quantization should occur::

        quantized_data = clip(round(input / scale) + zero_point)

    Parameters
    ----------
    input: tensor<SrcT, [1..]> (Required)

    zero_point: const tensor<DstT, [0..1]> (Optional)
        * The ``zero_point`` can be either a scalar or a vector. If not provided, it is
          assumed to be ``0``.
        * The ``zero_point`` follows similar broadcasting rules and size constraints as ``scale``.

    scale: const tensor<SrcT, [0..1]> (Required)
        * The ``scale`` can be either a scalar or a vector.
        * If ``scale`` is a vector, for implementation, it is broadcasted to the following shape:
            - The rank of ``scale`` becomes the same as the rank of the input.
            - Constraint: ``size(scale-vector) == input.shape[axis]``.
            - For ``i == axis``, ``scale.shape[i] == input.shape[i]``.
            - For ``i != axis``, ``scale.shape == 1``.
            - For example:
                - Assume ``input.shape = (2, 3, 4, 5)`` and ``axis = 1``.
                - If ``scale`` is a vector, then ``scale.size`` needs to be equal to
                  ``input.shape[axis]``; that is, equal to ``3``.
                - This is broadcasted to ``(1, 3, 1, 1)``.

    output_dtype: const tensor<string, []> (Required)
        * This parameter can take ``"uint8"``, ``"int8"`` as values.
        * The ``output_dtype`` value must match the ``zero_point`` dtype.

    axis: const tensor<int32, []> (Optional)

    Returns
    -------
    tensor<DstT, [1..]>

    Attributes
    ----------
    SrcT: fp16, fp32
    DstT: uint8, int8
    """

    input_spec = InputSpec(
        input=TensorInputType(type_domain="SrcT"),
        zero_point=TensorInputType(const=True, optional=True, type_domain="DstT"),
        scale=TensorInputType(const=True, type_domain="SrcT"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
        output_dtype=TensorInputType(const=True, type_domain=types.str),
    )

    type_domains = {
        "SrcT": (types.fp16, types.fp32),
        "DstT": (types.uint8, types.int8),
    }

    def type_inference(self):
        out_dtype = types.string_to_builtin(self.output_dtype.val)
        if out_dtype not in self.type_domains["DstT"]:
            raise ValueError(
                '"quantize" op: unrecognized output dtype "{}"'.format(self.output_dtype.val)
            )

        if self.zero_point is not None:
            if out_dtype != self.zero_point.dtype:
                raise ValueError(
                    "output_dtype & zero_point dtype mismatch: {}, {}".format(
                        self.output_dtype.val, types.builtin_to_string(self.zero_point.dtype)
                    )
                )

        if np.all(self.scale.val == 0):
            raise ValueError("quantize op: scale cannot be 0")

        _check_scale_zp_shapes(self.input, self.scale, self.zero_point, self.axis)

        return types.tensor(out_dtype, self.input.shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        scale, zero_point = _prepare_scale_and_zero_point(
            self.input, self.scale, self.zero_point, self.axis
        )
        return optimize_utils.quantize_by_scale_and_zp(
            self.input.val, scale, zero_point, types.string_to_builtin(self.output_dtype.val)
        )


@register_op(opset_version=_IOS17_TARGET)
class dequantize(Operation):
    """
    Performs dequantization on an input tensor with affine/linear quantization.

    The quantized data comes from the first "input".
    The other parameters -- ``scale``, ``zero_point``, and ``axis`` -- describe how
    unquantized values can be extracted from it,
    using the following equation for affine/linear quantization::

        unquantized_data = scale * (input - zero_point)

    Parameters
    ----------
    input: tensor<SrcT, [1..]> (Required)

    zero_point: const tensor<SrcT, [0..1]> (Optional)
        * The ``zero_point`` can be either a scalar or a vector. If not provided,
          it is assumed to be ``0``.
        * The ``zero_point`` follows similar broadcasting rules and size constraints as ``scale``.

    scale: const tensor<DstT, [0..1]> (Required)
        * The ``scale`` can be either a scalar or a vector.
        * If ``scale`` is a vector, for implementation, it is broadcasted to the following shape:
            - The rank of ``scale`` becomes the same as the rank of the input.
            - Constraint: ``size(scale-vector) == input.shape[axis]``.
            - For ``i == axis``, ``scale.shape[i] == input.shape[i]``.
            - For ``i != axis``, ``scale.shape == 1``.
            - For example:
                - Assume ``input.shape = (2, 3, 4, 5)`` and ``axis = 1``.
                - If ``scale`` is a vector, then ``scale.size`` needs to be equal to
                  ``input.shape[axis]``; that is, equal to ``3``.
                - This is broadcasted to ``(1, 3, 1, 1)``.

    axis: const tensor<int32, []> (Optional)

    Returns
    -------
    tensor<DstT, [1..]>

    Attributes
    ----------
    SrcT: uint8, int8
    DstT: fp16, fp32
    """

    input_spec = InputSpec(
        input=TensorInputType(type_domain="SrcT"),
        zero_point=TensorInputType(const=True, optional=True, type_domain="SrcT"),
        scale=TensorInputType(const=True, type_domain="DstT"),
        axis=TensorInputType(const=True, optional=True, type_domain=types.int32),
    )

    type_domains = {
        "DstT": (types.fp16, types.fp32),
        "SrcT": (types.uint8, types.int8),
    }

    def type_inference(self):
        _check_scale_zp_shapes(self.input, self.scale, self.zero_point, self.axis)
        return types.tensor(self.scale.dtype, self.input.shape)

    def can_materialize_val(self) -> bool:
        if self.input.val is None:
            return False
        if self.scale.val is None:
            return False
        if self.zero_point is not None and self.zero_point.val is None:
            return False
        if self.axis is not None and self.axis.val is None:
            return False
        return True

    def materialized_val_inference(self) -> np.ndarray:
        if not self.can_materialize_val():
            return None

        scale, zero_point = _prepare_scale_and_zero_point(
            self.input, self.scale, self.zero_point, self.axis
        )
        return optimize_utils.dequantize_by_scale_and_zp(self.input.val, scale, zero_point)
