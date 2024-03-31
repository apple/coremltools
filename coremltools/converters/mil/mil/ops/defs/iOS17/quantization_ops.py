# Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, TensorInputType
from coremltools.converters.mil.mil.operation import VALUE, Operation, precondition
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS17 import _IOS17_TARGET


def _rank_promoted_to_same_as_data(data, axis, param):
    """
    Reshapes `param` to be the same shape as `data`.
    """
    if axis is not None:
        axis = axis if axis >= 0 else axis + len(data.shape)
    if len(param.shape) == 0:
        return np.reshape(param, np.ones(len(data.shape), np.int32))
    else:
        axes = [i for i in range(len(data.shape)) if i != axis]
        return np.expand_dims(param, axis=tuple(axes))


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
        if out_dtype not in {types.int8, types.uint8}:
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

        _check_scale_zp_shapes(self.input, self.scale, self.zero_point, self.axis)

        return types.tensor(out_dtype, self.input.shape)

    @precondition(allow=VALUE)
    def value_inference(self):
        original_data = self.input.val
        if self.zero_point is not None:
            zero_point = self.zero_point.val
        else:
            zero_point = np.int8(0) if self.output_dtype.val == "int8" else np.uint8(0)
        scale = self.scale.val
        axis = None
        if self.axis is not None:
            axis = self.axis.val
        dtype_info = np.iinfo(zero_point.dtype)

        sc = _rank_promoted_to_same_as_data(original_data, axis, scale)
        zp = _rank_promoted_to_same_as_data(original_data, axis, zero_point)
        val = np.clip(
            np.around(original_data / sc) + zp.astype(np.float32), dtype_info.min, dtype_info.max
        )
        return val.astype(zero_point.dtype)


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

        quantized_data = self.input.val
        if self.zero_point is not None:
            zero_point = self.zero_point.val
        else:
            zero_point = np.int8(0) if self.input.dtype == types.int8 else np.uint8(0)
        scale = self.scale.val
        axis = None
        if self.axis is not None:
            axis = self.axis.val

        sc = _rank_promoted_to_same_as_data(quantized_data, axis, scale)
        zp = _rank_promoted_to_same_as_data(quantized_data, axis, zero_point)
        val = sc * (quantized_data.astype(np.float32) - zp.astype(np.float32))
        return val.astype(scale.dtype)
