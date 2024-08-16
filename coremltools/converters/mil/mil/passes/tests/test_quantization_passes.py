#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
from typing import Tuple

import numpy as np
import parameterized
import pytest
from mock import patch

import coremltools as ct
import coremltools.converters.mil.mil.types as types
from coremltools._deps import _HAS_TORCH, _IS_MACOS, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.defs import quantization
from coremltools.converters.mil.mil.passes.defs.quantization import add_fp16_cast
from coremltools.converters.mil.mil.types import numpy_type_to_builtin_type
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    get_op_types_in_program,
)

if _HAS_TORCH:
    import torch
    import torch.nn as nn

np.random.seed(1818)


class TestTensorwiseAffineDequantizeConstElimination:
    @pytest.mark.parametrize("axis", (None, 0, 1, -1))
    def test_eliminate_transpose(self, axis):
        """
        Input graph:
            data -> constexpr_affine_dequantize -> transpose

        Output graph:
            new_data -> constexpr_affine_dequantize

        where new_data is the value after applying transpose to data
        """
        SHAPE = (1, 2, 3, 4)
        quantized_data = np.random.randint(0, 256, SHAPE).astype(np.int8)
        if axis is None:
            axis = 0  # although tensor-wise, constexpr_affine_dequantize requires a (dummy) axis
            scale = np.random.rand()
            zero_point = np.random.randint(-127, 128, dtype=np.int8)
        else:
            size = SHAPE[axis]
            scale = np.random.rand(size)
            zero_point = np.random.randint(-127, 128, size, dtype=np.int8)

        @mb.program(input_specs=[], opset_version=ct.target.iOS16)
        def prog():
            res = mb.constexpr_affine_dequantize(
                quantized_data=quantized_data,
                axis=axis,
                scale=scale,
                zero_point=zero_point,
            )
            return mb.transpose(x=res, perm=(2, 0, 1, 3))

        apply_pass_and_basic_check(prog, "common::merge_affine_dequantize_with_consecutive_ops")
        assert get_op_types_in_program(prog) == ["constexpr_affine_dequantize"]

        new_op = prog.find_ops(op_type="constexpr_affine_dequantize", exactly_one=True)[0]
        expected_quantized_data = np.transpose(quantized_data, (2, 0, 1, 3))
        np.testing.assert_array_equal(new_op.quantized_data.val, expected_quantized_data)

    def test_eliminate_reshape(self):
        """
        Input graph:
            data -> constexpr_affine_dequantize -> reshape

        Output graph:
            new_data -> constexpr_affine_dequantize

        where new_data is the value after applying reshape to data
        """
        quantized_data = np.random.randint(0, 256, (1, 2, 3, 4)).astype(np.int8)

        @mb.program(input_specs=[], opset_version=ct.target.iOS16)
        def prog():
            res = mb.constexpr_affine_dequantize(
                quantized_data=quantized_data,
                axis=0,
                scale=8.9,
                zero_point=np.int8(34),
            )
            return mb.reshape(x=res, shape=(3, -1))

        apply_pass_and_basic_check(prog, "common::merge_affine_dequantize_with_consecutive_ops")
        assert get_op_types_in_program(prog) == ["constexpr_affine_dequantize"]

        new_op = prog.find_ops(op_type="constexpr_affine_dequantize", exactly_one=True)[0]
        expected_quantized_data = np.reshape(quantized_data, (3, 8))
        np.testing.assert_array_equal(new_op.quantized_data.val, expected_quantized_data)

    def test_eliminate_expand_dims(self):
        """
        Input graph:
            data -> constexpr_affine_dequantize -> expand_dims

        Output graph:
            new_data -> constexpr_affine_dequantize

        where new_data is the value after applying expand_dims to data
        """
        quantized_data = np.random.randint(0, 256, (2, 3, 4)).astype(np.int8)

        @mb.program(input_specs=[], opset_version=ct.target.iOS16)
        def prog():
            res = mb.constexpr_affine_dequantize(
                quantized_data=quantized_data,
                axis=0,
                scale=8.9,
                zero_point=np.int8(34),
            )
            return mb.expand_dims(x=res, axes=(0, 2, 4))

        apply_pass_and_basic_check(prog, "common::merge_affine_dequantize_with_consecutive_ops")
        assert get_op_types_in_program(prog) == ["constexpr_affine_dequantize"]

        new_op = prog.find_ops(op_type="constexpr_affine_dequantize", exactly_one=True)[0]
        expected_quantized_data = np.expand_dims(quantized_data, axis=(0, 2, 4))
        np.testing.assert_array_equal(new_op.quantized_data.val, expected_quantized_data)

    @pytest.mark.parametrize("axis", [(0, 3), None])
    def test_eliminate_squeeze(self, axis):
        """
        Input graph:
            data -> constexpr_affine_dequantize -> squeeze

        Output graph:
            new_data -> constexpr_affine_dequantize

        where new_data is the value after applying squeeze to data
        """
        quantized_data = np.random.randint(0, 256, (1, 2, 3, 1, 4)).astype(np.int8)

        @mb.program(input_specs=[], opset_version=ct.target.iOS16)
        def prog():
            res = mb.constexpr_affine_dequantize(
                quantized_data=quantized_data,
                axis=0,
                scale=8.9,
                zero_point=np.int8(34),
            )
            return mb.squeeze(x=res, axes=axis)

        apply_pass_and_basic_check(prog, "common::merge_affine_dequantize_with_consecutive_ops")
        assert get_op_types_in_program(prog) == ["constexpr_affine_dequantize"]

        new_op = prog.find_ops(op_type="constexpr_affine_dequantize", exactly_one=True)[0]
        expected_quantized_data = np.squeeze(quantized_data, axis=axis)
        np.testing.assert_array_equal(new_op.quantized_data.val, expected_quantized_data)

    def test_eliminate_multiple_ops(self):
        """
        Input graph:
            data -> constexpr_affine_dequantize -> transpose ->
            reshape -> expand_dims -> squeeze

        Output graph:
            new_data -> constexpr_affine_dequantize

        where new_data is the value after applying the same chain of transformations to data
        """
        quantized_data = np.random.randint(0, 256, (1, 2, 3, 4)).astype(np.int8)

        @mb.program(input_specs=[], opset_version=ct.target.iOS16)
        def prog():
            res = mb.constexpr_affine_dequantize(
                quantized_data=quantized_data,
                axis=0,
                scale=8.9,
                zero_point=np.int8(34),
            )
            res = mb.transpose(x=res, perm=(1, 0, 3, 2))
            res = mb.reshape(x=res, shape=(8, 3))
            res = mb.expand_dims(x=res, axes=(0, 2, 4))
            return mb.squeeze(x=res, axes=(2,))

        apply_pass_and_basic_check(prog, "common::merge_affine_dequantize_with_consecutive_ops")
        assert get_op_types_in_program(prog) == ["constexpr_affine_dequantize"]

        new_op = prog.find_ops(op_type="constexpr_affine_dequantize", exactly_one=True)[0]

        expected_quantized_data = np.transpose(quantized_data, (1, 0, 3, 2))
        expected_quantized_data = np.reshape(expected_quantized_data, (8, 3))
        expected_quantized_data = np.expand_dims(expected_quantized_data, (0, 2, 4))
        expected_quantized_data = np.squeeze(expected_quantized_data, (2,))

        np.testing.assert_array_equal(new_op.quantized_data.val, expected_quantized_data)

    def test_negative_non_linked_list_pattern(self):
        """
        If ``quantized_data`` feeds into multiple ``constexpr_affine_dequantize`` ops,
        the graph will not be changed.
        """
        quantized_data = np.random.randint(0, 256, (2, 3, 4)).astype(np.int8)

        @mb.program(input_specs=[], opset_version=ct.target.iOS16)
        def prog():
            data = mb.const(val=quantized_data)
            x = mb.constexpr_affine_dequantize(
                quantized_data=data,
                axis=0,
                scale=8.9,
                zero_point=np.int8(34),
            )
            y = mb.constexpr_affine_dequantize(
                quantized_data=data,
                axis=0,
                scale=8.1,
                zero_point=np.int8(56),
            )
            return mb.transpose(x=x, perm=(1, 0, 2)), mb.reshape(x=y, shape=(24,))

        apply_pass_and_basic_check(prog, "common::merge_affine_dequantize_with_consecutive_ops")
        assert get_op_types_in_program(prog) == [
            "constexpr_affine_dequantize",
            "constexpr_affine_dequantize",
            "transpose",
            "reshape",
        ]

    def test_eliminate_connected_outputs(self):
        """
        The optimization stops when the node is a block output
        """
        quantized_data = np.random.randint(0, 256, (2, 3, 4)).astype(np.int8)

        @mb.program(input_specs=[], opset_version=ct.target.iOS16)
        def prog():
            x = mb.constexpr_affine_dequantize(
                quantized_data=quantized_data,
                axis=0,
                scale=8.9,
                zero_point=np.int8(34),
            )
            x = mb.transpose(x=x, perm=(1, 0, 2))
            x = mb.reshape(x=x, shape=(2, 2, 3, 2))
            y = mb.transpose(x=x, perm=(0, 3, 2, 1))
            return x, y

        apply_pass_and_basic_check(prog, "common::merge_affine_dequantize_with_consecutive_ops")
        assert get_op_types_in_program(prog) == [
            "constexpr_affine_dequantize",
            "transpose",
        ]

        new_op = prog.find_ops(op_type="constexpr_affine_dequantize", exactly_one=True)[0]
        expected_quantized_data = np.transpose(quantized_data, (1, 0, 2))
        expected_quantized_data = np.reshape(expected_quantized_data, (2, 2, 3, 2))
        np.testing.assert_array_equal(new_op.quantized_data.val, expected_quantized_data)

        transpose_op = prog.find_ops(op_type="transpose", exactly_one=True)[0]
        assert transpose_op.perm.val.tolist() == [0, 3, 2, 1]


class QuantizationBaseTest:
    @staticmethod
    def generate_random_quantization_params(
        float_dtype: np.dtype,
        quant_dtype: np.dtype,
        input_shape: Tuple[int],
        is_zp_present: bool = True,
        is_axis_present: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        return scale, zero point, axis
        """

        input_rank = len(input_shape)
        low, high = (-128, 128) if quant_dtype == np.int8 else (0, 256)

        scale = None
        zero_point = None
        axis = (
            np.random.randint(-input_rank, input_rank, dtype=np.int32) if is_axis_present else None
        )
        if is_axis_present:
            scale = np.random.rand(input_shape[axis]).astype(float_dtype)
            if is_zp_present:
                zero_point = np.random.randint(
                    low=low, high=high, size=input_shape[axis], dtype=quant_dtype
                )
        else:
            scale = np.array(np.random.rand()).astype(float_dtype)
            if is_zp_present:
                zero_point = np.array(np.random.randint(low=low, high=high, dtype=quant_dtype))

        return scale, zero_point, axis

    @staticmethod
    def generate_random_quantize_input(
        float_dtype: np.dtype,
        quant_dtype: np.dtype,
        scale: np.ndarray,
        zero_point: np.ndarray,
        axis: int,
        shape: Tuple[int],
    ) -> np.ndarray:
        assert float_dtype == scale.dtype
        if zero_point is not None:
            assert quant_dtype == zero_point.dtype
        if axis is not None:
            assert shape[axis] == scale.shape[0]
        if zero_point is not None and axis is not None:
            assert shape[axis] == zero_point.shape[0]

        low, high = (-128, 128) if quant_dtype == np.int8 else (0, 256)
        x_q = np.random.randint(low=low, high=high, size=shape, dtype=np.int32)
        if axis is None:
            if zero_point is None:
                x_fp = x_q * scale
            else:
                x_fp = (x_q - zero_point) * scale
        else:
            # prepare broadcast shape for latter dequantize
            broadcastable_shape = np.ones(len(shape), dtype=np.int32)
            broadcastable_shape[axis] = shape[axis]

            broadcasted_scale = np.reshape(scale, broadcastable_shape)

            if zero_point is None:
                x_fp = x_q * broadcasted_scale
            else:
                broadcasted_zero_point = np.reshape(zero_point, broadcastable_shape)
                x_fp = (x_q - broadcasted_zero_point) * broadcasted_scale

        return float_dtype(x_fp)


class TestIntOpCanonicalization:
    @pytest.mark.parametrize("op_type", ["reshape"])
    def test_canonicalize_int_op(self, op_type):
        """
        Input graph:

            input -> quantize -> dequantize -> int op -> quantize -> dequantize -> output

        Output graph:

            input -> quantize -> int op -> dequantize -> output
        """
        input_shape = (5, 6)
        output_shape = (5, 2, 3)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)], opset_version=ct.target.iOS17)
        def prog(x):
            quantize_0 = mb.quantize(input=x, scale=0.1, output_dtype="int8")
            dequantize_1 = mb.dequantize(input=quantize_0, scale=0.1)
            if op_type == "reshape":
                reshape = mb.reshape(x=dequantize_1, shape=output_shape)
            quantize_1 = mb.quantize(input=reshape, scale=0.1, output_dtype="int8")
            dequantize_2 = mb.dequantize(input=quantize_1, scale=0.1)
            return dequantize_2

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::int_op_canonicalization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prev_prog) == [
            "quantize",
            "dequantize", "reshape", "quantize",
            "dequantize",
        ]
        assert get_op_types_in_program(prog) == ["quantize", "reshape", "dequantize"]

        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={block.outputs[0].name: output_shape},
            minimum_deployment_target=ct.target.iOS17,
            backend=("mlprogram", "fp32"),
        )

    @pytest.mark.parametrize("all_are_int", (True, False))
    def test_canonicalize_versatile_inputs(self, all_are_int):
        """
        Input graph:

                                             |-> int op 0 if all_are_int else add -> quantize -> dequantize -> output_0
            input -> quantize -> dequantize -|
                                             |-> int op 1 -> quantize -> dequantize -> output_1

        Output graph:

            if all_are_int:

                                   |-> int op 0 -> dequantize -> output_0
                input -> quantize -|
                                   |-> int op 1 -> dequantize -> output_1

            else:

                                   |-> dequantize -> add -> quantize -> dequantize -> output_0
                input -> quantize -|
                                   |-> int op 1 -> dequantize -> output_1
        """
        input_shape = (5, 6)
        output_shape = (5, 2, 3)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)], opset_version=ct.target.iOS17)
        def prog(x):
            quantize_0 = mb.quantize(input=x, scale=0.1, output_dtype="int8")
            dequantize_1 = mb.dequantize(input=quantize_0, scale=0.1)

            # int op 0 (here reshape) path
            if all_are_int:
                reshape = mb.reshape(x=dequantize_1, shape=output_shape)
                quantize_1_0 = mb.quantize(input=reshape, scale=0.1, output_dtype="int8")
                dequantize_2_0 = mb.dequantize(input=quantize_1_0, scale=0.1)
            # float op (here add) path
            else:
                add = mb.add(x=dequantize_1, y=1.0)
                quantize_1_0 = mb.quantize(input=add, scale=0.1, output_dtype="int8")
                dequantize_2_0 = mb.dequantize(input=quantize_1_0, scale=0.1)

            # int op 1 (here reshape) path
            reshape = mb.reshape(x=dequantize_1, shape=output_shape)
            quantize_1_1 = mb.quantize(input=reshape, scale=0.1, output_dtype="int8")
            dequantize_2_1 = mb.dequantize(input=quantize_1_1, scale=0.1)

            return (
                dequantize_2_0,
                dequantize_2_1,
            )

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::int_op_canonicalization")
        if all_are_int:
            _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
            assert get_op_types_in_program(prev_prog) == [
                "quantize", "dequantize",
                "reshape", "quantize", "dequantize",
                "reshape", "quantize", "dequantize",
            ]
            assert get_op_types_in_program(prog) == [
                "quantize",
                "reshape", "dequantize",
                "reshape", "dequantize",
            ]
        else:
            assert get_op_types_in_program(prev_prog) == [
                "quantize", "dequantize",
                "add", "quantize", "dequantize",
                "reshape", "quantize", "dequantize",
            ]
            assert get_op_types_in_program(prog) == [
                "quantize",
                "dequantize", "add", "quantize", "dequantize",
                "reshape", "dequantize",
            ]

        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={
                block.outputs[0].name: output_shape if all_are_int else input_shape,
                block.outputs[1].name: output_shape,
            },
            minimum_deployment_target=ct.target.iOS17,
            backend=("mlprogram", "fp32"),
        )

    def test_canonicalize_consecutive_int_ops(self):
        """
        Input graph:

            input -> quantize -> dequantize -> int op 0 -> quantize -> dequantize -> int op 1 -> quantize -> dequantize -> output

        Output graph:

            input -> quantize -> int op 0 -> int op 1 -> dequantize -> output
        """
        input_shape = (5, 6)
        activation_shape = (10, 3)
        output_shape = (5, 2, 3)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)], opset_version=ct.target.iOS17)
        def prog(x):
            quantize_0 = mb.quantize(input=x, scale=0.1, output_dtype="int8")

            dequantize_1 = mb.dequantize(input=quantize_0, scale=0.1)
            reshape0 = mb.reshape(x=dequantize_1, shape=activation_shape)
            quantize_1 = mb.quantize(input=reshape0, scale=0.1, output_dtype="int8")

            dequantize_2 = mb.dequantize(input=quantize_1, scale=0.1)
            reshape1 = mb.reshape(x=dequantize_2, shape=output_shape)
            quantize_2 = mb.quantize(input=reshape1, scale=0.1, output_dtype="int8")

            dequantize_3 = mb.dequantize(input=quantize_2, scale=0.1)
            return dequantize_3

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::int_op_canonicalization")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prev_prog) == [
            "quantize",
            "dequantize", "reshape", "quantize",
            "dequantize", "reshape", "quantize",
            "dequantize",
        ]
        assert get_op_types_in_program(prog) == ["quantize", "reshape", "reshape", "dequantize"]

        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={block.outputs[0].name: output_shape},
            minimum_deployment_target=ct.target.iOS17,
            backend=("mlprogram", "fp32"),
        )

    def test_canonicalize_block_output_input(self):
        """
        Input graph:

                                             |-> output_0
            input -> quantize -> dequantize -|
                                             |-> int op -> quantize -> dequantize -> output_1

        Output graph:

                               |-> dequantize -> output_0
            input -> quantize -|
                               |-> int op -> dequantize -> output_1
        """
        input_shape = (5, 6)
        output_shape = (5, 2, 3)

        @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)], opset_version=ct.target.iOS17)
        def prog(x):
            quantize_0 = mb.quantize(input=x, scale=0.1, output_dtype="int8")
            dequantize_1 = mb.dequantize(input=quantize_0, scale=0.1)

            reshape = mb.reshape(x=dequantize_1, shape=output_shape)
            quantize_1 = mb.quantize(input=reshape, scale=0.1, output_dtype="int8")
            dequantize_2 = mb.dequantize(input=quantize_1, scale=0.1)

            return dequantize_1, dequantize_2

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::int_op_canonicalization")
        assert get_op_types_in_program(prev_prog) == [
            "quantize", "dequantize",
            "reshape", "quantize", "dequantize",
        ]
        assert get_op_types_in_program(prog) == [
            "quantize",
            "dequantize",
            "reshape", "dequantize",
        ]

        assert_model_is_valid(
            prog,
            {"x": input_shape},
            expected_output_shapes={
                block.outputs[0].name: input_shape,
                block.outputs[1].name: output_shape,
            },
            minimum_deployment_target=ct.target.iOS17,
            backend=("mlprogram", "fp32"),
        )

    # TODO (rdar://112297858): test the case where `int_op_canonicalization`
    # refuses to transform because the "int op" is from an older iOS version
    # that does not support int8 and uint8


class TestNullifyRedundantQuantizationZeroPoint:
    @staticmethod
    def np_dtype_to_str(np_dtype: np.dtype) -> str:
        NP_DTYPE_TO_STR = {np.int8: "int8", np.uint8: "uint8"}
        return NP_DTYPE_TO_STR.get(np_dtype)

    @staticmethod
    def shift_128(input: np.ndarray, quant_dtype: np.dtype) -> np.ndarray:
        """
        shift const input according to zero point shift and dtype change:
            int8: -128 -> 0, int8 -> uint8
            uint8: 128 -> 0, uint8 -> int8
        """
        shifted_input: np.ndarray
        if quant_dtype == np.int8:
            shifted_input = np.uint8(np.int16(input) + 128)
        else:
            shifted_input = np.int8(np.int16(input) - 128)
        return shifted_input

    @pytest.mark.parametrize(
        "quant_dtype, is_axis_present",
        itertools.product(
            (np.int8, np.uint8),
            (True, False),
        ),
    )
    def test_optimize_zp0_quantize(self, quant_dtype, is_axis_present):
        """
        initial graph:
            input -> quantize(zero_point=0) -> dequantize(scale=1) -> output

        final graph:
            input -> quantize() -> dequantize(scale=1) -> output
        """

        SHAPE = (1, 3)

        rank = len(SHAPE)
        axis = np.random.randint(-rank, rank) if is_axis_present else None

        scale = np.random.rand(SHAPE[axis]) if is_axis_present else np.random.rand()

        zero_point = np.zeros(SHAPE[axis], dtype=quant_dtype) if is_axis_present else quant_dtype(0)

        @mb.program(input_specs=[mb.TensorSpec(shape=SHAPE)])
        def prog(x):
            quantized = mb.quantize(
                input=x,
                scale=scale,
                zero_point=zero_point,
                axis=axis,
                output_dtype=self.np_dtype_to_str(quant_dtype),
            )
            dequantized = mb.dequantize(
                input=quantized,
                scale=1.0,
            )
            return dequantized

        assert get_op_types_in_program(prog) == ["quantize", "dequantize"]
        quantize_op = prog.find_ops(op_type="quantize")[0]
        assert np.all(quantize_op.zero_point.val == 0)

        _, _, block = apply_pass_and_basic_check(prog, "common::nullify_redundant_quantization_zero_point")
        assert get_op_types_in_program(prog) == ["quantize", "dequantize"]
        quantize_op = prog.find_ops(op_type="quantize")[0]
        assert quantize_op.zero_point is None

        assert_model_is_valid(
            prog,
            {"x": SHAPE},
            minimum_deployment_target=ct.target.iOS17,
            backend=("mlprogram", "fp32"),
            expected_output_shapes={block.outputs[0].name: SHAPE},
        )

    @pytest.mark.parametrize(
        "quant_dtype, is_axis_present",
        itertools.product(
            (np.int8, np.uint8),
            (True, False),
        ),
    )
    def test_optimize_zp0_dequantize(self, quant_dtype, is_axis_present):
        """
        initial graph:
            input -> quantize(scale=1) -> dequantize(zero_point=0) -> output

        final graph:
            input -> quantize(scale=1) -> dequantize() -> output
        """

        SHAPE = (6, 5, 4, 3, 2)

        rank = len(SHAPE)
        axis = np.random.randint(-rank, rank) if is_axis_present else None

        scale = np.random.rand(SHAPE[axis]) if is_axis_present else np.random.rand()

        zero_point = np.zeros(SHAPE[axis], dtype=quant_dtype) if is_axis_present else quant_dtype(0)

        @mb.program(input_specs=[mb.TensorSpec(shape=SHAPE)])
        def prog(x):
            quantized = mb.quantize(
                input=x,
                scale=1.0,
                output_dtype=self.np_dtype_to_str(quant_dtype),
            )
            dequantized = mb.dequantize(
                input=quantized,
                scale=scale,
                zero_point=zero_point,
                axis=axis,
            )
            return dequantized

        assert get_op_types_in_program(prog) == ["quantize", "dequantize"]
        dequantize_op = prog.find_ops(op_type="dequantize")[0]
        assert np.all(dequantize_op.zero_point.val == 0)

        _, _, block = apply_pass_and_basic_check(
            prog, "common::nullify_redundant_quantization_zero_point"
        )
        assert get_op_types_in_program(prog) == ["quantize", "dequantize"]
        dequantize_op = prog.find_ops(op_type="dequantize")[0]
        assert dequantize_op.zero_point is None

        assert_model_is_valid(
            prog,
            {"x": SHAPE},
            minimum_deployment_target=ct.target.iOS17,
            backend=("mlprogram", "fp32"),
            expected_output_shapes={block.outputs[0].name: SHAPE},
        )

    @pytest.mark.parametrize(
        "quant_dtype, is_axis_present",
        itertools.product(
            (np.int8, np.uint8),
            (True, False),
        ),
    )
    def test_optimize_zp128_quantize_dequantize(self, quant_dtype, is_axis_present):
        """
        initial graph:
            input -> quantize(zero_point=±128) -> dequantize(zero_point=±128) -> output

        final graph:
            input -> quantize() -> dequantize() -> output
        """

        SHAPE = (2, 3)

        rank = len(SHAPE)
        axis = np.random.randint(-rank, rank) if is_axis_present else None

        scale_quantize = np.random.rand(SHAPE[axis]) if is_axis_present else np.random.rand()
        scale_dequantize = np.random.rand(SHAPE[axis]) if is_axis_present else np.random.rand()

        zero_point_value = -128 if quant_dtype == np.int8 else 128
        zero_point = (
            np.full(SHAPE[axis], zero_point_value, dtype=quant_dtype)
            if is_axis_present
            else quant_dtype(zero_point_value)
        )

        @mb.program(input_specs=[mb.TensorSpec(shape=SHAPE)])
        def prog(x):
            quantized = mb.quantize(
                input=x,
                scale=scale_quantize,
                zero_point=zero_point,
                axis=axis,
                output_dtype=self.np_dtype_to_str(quant_dtype),
            )
            dequantized = mb.dequantize(
                input=quantized,
                scale=scale_dequantize,
                zero_point=zero_point,
                axis=axis,
            )
            return dequantized

        assert get_op_types_in_program(prog) == ["quantize", "dequantize"]
        quantize_op = prog.find_ops(op_type="quantize")[0]
        dequantize_op = prog.find_ops(op_type="dequantize")[0]
        assert np.all(quantize_op.zero_point.val == zero_point_value)
        assert np.all(dequantize_op.zero_point.val == zero_point_value)

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::nullify_redundant_quantization_zero_point"
        )
        assert get_op_types_in_program(prog) == ["quantize", "dequantize"]
        quantize_op = prog.find_ops(op_type="quantize")[0]
        dequantize_op = prog.find_ops(op_type="dequantize")[0]
        assert quantize_op.zero_point is None
        assert dequantize_op.zero_point is None

        assert_model_is_valid(
            prog,
            {"x": SHAPE},
            minimum_deployment_target=ct.target.iOS17,
            backend=("mlprogram", "fp32"),
            expected_output_shapes={block.outputs[0].name: SHAPE},
        )

        prev_model = ct.convert(prev_prog, minimum_deployment_target=ct.target.iOS17)
        model = ct.convert(prog, minimum_deployment_target=ct.target.iOS17)

        x = np.random.rand(*SHAPE)
        prev_output = list(prev_model.predict({"x": x}).values())[0]
        output = list(model.predict({"x": x}).values())[0]
        assert np.all(prev_output == output)

    @pytest.mark.parametrize(
        "quant_dtype, is_axis_present",
        itertools.product(
            (np.int8, np.uint8),
            (True, False),
        ),
    )
    def test_optimize_zp128_const_dequantize(self, quant_dtype, is_axis_present):
        """
        initial graph:
            input -----------------------|
                                         |-> add -> output
            dequantize(zero_point=±128) -|

        apply nullify_redundant_quantization_zero_point:
            input --------|
                          |-> add -> output
            dequantize() -|

        final graph:
            input -----------------------|
                                         |-> add -> output
            constexpr_affine_dequantize -|
        """

        SHAPE = (2, 5, 3)

        quantized = (
            np.random.randint(low=-128, high=128, size=SHAPE, dtype=quant_dtype)
            if quant_dtype == np.int8
            else np.random.randint(low=0, high=256, size=SHAPE, dtype=quant_dtype)
        )

        rank = len(SHAPE)
        axis = np.random.randint(-rank, rank) if is_axis_present else None

        scale = np.random.rand(SHAPE[axis]) if is_axis_present else np.random.rand()

        zero_point_value = -128 if quant_dtype == np.int8 else 128
        zero_point = (
            np.full(SHAPE[axis], zero_point_value, dtype=quant_dtype)
            if is_axis_present
            else quant_dtype(zero_point_value)
        )

        @mb.program(input_specs=[mb.TensorSpec(shape=SHAPE)])
        def prog(x):
            dequantized = mb.dequantize(
                input=quantized,
                scale=scale,
                zero_point=zero_point,
                axis=axis,
            )
            # Core ML cannot have a model with idle input and constant outputs
            # so append an `add` op to make the model valid
            result = mb.add(x=x, y=dequantized)
            return result

        assert get_op_types_in_program(prog) == ["dequantize", "add"]
        dequantize_op = prog.find_ops(op_type="dequantize")[0]
        assert np.all(dequantize_op.input.val == quantized)
        assert np.all(dequantize_op.zero_point.val == zero_point_value)

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::nullify_redundant_quantization_zero_point"
        )
        assert get_op_types_in_program(prog) == ["dequantize", "add"]
        dequantize_op = prog.find_ops(op_type="dequantize")[0]
        assert np.all(dequantize_op.input.val == self.shift_128(quantized, quant_dtype))
        assert dequantize_op.zero_point is None

        _, _, block = apply_pass_and_basic_check(prog, "common::dequantize_to_constexpr")
        assert get_op_types_in_program(prog) == ["constexpr_affine_dequantize", "add"]

        assert_model_is_valid(
            prog,
            {"x": SHAPE},
            minimum_deployment_target=ct.target.iOS17,
            backend=("mlprogram", "fp32"),
            expected_output_shapes={block.outputs[0].name: SHAPE},
        )

        prev_model = ct.convert(prev_prog, minimum_deployment_target=ct.target.iOS17)
        model = ct.convert(prog, minimum_deployment_target=ct.target.iOS17)

        x = np.random.rand(*SHAPE)
        prev_output = list(prev_model.predict({"x": x}).values())[0]
        output = list(model.predict({"x": x}).values())[0]
        assert np.all(prev_output == output)

    @pytest.mark.parametrize(
        "quant_dtype, is_axis_present",
        itertools.product(
            (np.int8, np.uint8),
            (True, False),
        ),
    )
    def test_keep_mismatching_quantize_dequantize(self, quant_dtype, is_axis_present):
        """
        initial graph:
            input -> quantize(zero_point=±128 + perturbation) -> dequantize(zero_point=±128) -> output

        final graph:
            input -> quantize(zero_point=±128 + perturbation) -> dequantize(zero_point=±128) -> output

        perturbation may also be applied to dequantize
        """

        SHAPE = (2, 3)

        rank = len(SHAPE)
        axis = np.random.randint(-rank, rank) if is_axis_present else None

        scale_quantize = np.random.rand(SHAPE[axis]) if is_axis_present else np.random.rand()
        scale_dequantize = np.random.rand(SHAPE[axis]) if is_axis_present else np.random.rand()

        zero_point_value = -128 if quant_dtype == np.int8 else 128
        perturbation = np.random.randint(1, 10, dtype=quant_dtype)
        zero_point = (
            np.full(SHAPE[axis], zero_point_value, dtype=quant_dtype)
            if is_axis_present
            else quant_dtype(zero_point_value)
        )
        zero_point_perturbed = quant_dtype(zero_point + perturbation)

        perturb_quantize = np.random.rand() < 0.5
        if perturb_quantize:
            zero_point_quantize = zero_point_perturbed
            zero_point_dequantize = zero_point
        else:
            zero_point_quantize = zero_point
            zero_point_dequantize = zero_point_perturbed

        @mb.program(input_specs=[mb.TensorSpec(shape=SHAPE)])
        def prog(x):
            quantized = mb.quantize(
                input=x,
                scale=scale_quantize,
                zero_point=zero_point_quantize,
                axis=axis,
                output_dtype=self.np_dtype_to_str(quant_dtype),
            )
            dequantized = mb.dequantize(
                input=quantized,
                scale=scale_dequantize,
                zero_point=zero_point_dequantize,
                axis=axis,
            )
            return dequantized

        assert get_op_types_in_program(prog) == ["quantize", "dequantize"]
        quantize_op = prog.find_ops(op_type="quantize")[0]
        dequantize_op = prog.find_ops(op_type="dequantize")[0]
        if perturb_quantize:
            assert np.all(quantize_op.zero_point.val == zero_point_perturbed)
            assert np.all(dequantize_op.zero_point.val == zero_point)
        else:
            assert np.all(quantize_op.zero_point.val == zero_point)
            assert np.all(dequantize_op.zero_point.val == zero_point_perturbed)

        _, _, block = apply_pass_and_basic_check(
            prog, "common::nullify_redundant_quantization_zero_point"
        )
        assert get_op_types_in_program(prog) == ["quantize", "dequantize"]
        quantize_op = prog.find_ops(op_type="quantize")[0]
        dequantize_op = prog.find_ops(op_type="dequantize")[0]
        if perturb_quantize:
            assert np.all(quantize_op.zero_point.val == zero_point_perturbed)
            assert np.all(dequantize_op.zero_point.val == zero_point)
        else:
            assert np.all(quantize_op.zero_point.val == zero_point)
            assert np.all(dequantize_op.zero_point.val == zero_point_perturbed)

        assert_model_is_valid(
            prog,
            {"x": SHAPE},
            minimum_deployment_target=ct.target.iOS17,
            backend=("mlprogram", "fp32"),
            expected_output_shapes={block.outputs[0].name: SHAPE},
        )


class TestDequantizeQuantizePairElimination:
    @staticmethod
    def generate_scale_zp_axis(shape, is_zp_present, is_axis_present):
        rank = len(shape)

        axis = None
        if is_axis_present:
            axis = np.random.randint(-rank, rank, dtype=np.int32)

        scale = np.random.rand(shape[axis]) if is_axis_present else np.random.rand()

        zero_point = None
        if is_zp_present:
            zero_point = (
                np.random.randint(-128, 128, shape[axis], dtype=np.int8)
                if is_axis_present
                else np.random.randint(-128, 128, dtype=np.int8)
            )

        return scale, zero_point, axis

    @pytest.mark.parametrize(
        "is_zp_present, is_axis_present",
        itertools.product(
            (True, False),
            (True, False),
        ),
    )
    def test_eliminate_identical_dequantize_quantize(self, is_zp_present, is_axis_present):
        """
        Input graph:
            input -> quantize0 -> dequantize1 -> quantize1 -> dequantize2 -> add -> quantize2 -> dequantize3 -> output

        Output graph:
            input -> quantize0 -> dequantize2 -> add -> quantize2 -> dequantize3 -> output
        """

        SHAPE = (2, 3)
        scale, zero_point, axis = self.generate_scale_zp_axis(SHAPE, is_zp_present, is_axis_present)

        @mb.program(input_specs=[mb.TensorSpec(shape=SHAPE, dtype=types.fp32)])
        def prog(x):
            # quantize input
            quantized_0 = mb.quantize(
                input=x, scale=scale, zero_point=zero_point, axis=axis, output_dtype="int8"
            )
            # redundant dequantize-quantize pair
            dequantized_1 = mb.dequantize(
                input=quantized_0, scale=scale, zero_point=zero_point, axis=axis
            )
            quantized_1 = mb.quantize(
                input=dequantized_1,
                scale=scale,
                zero_point=zero_point,
                axis=axis,
                output_dtype="int8",
            )
            # dequantize-op-quantize sandwich
            dequantized_2 = mb.dequantize(
                input=quantized_1, scale=scale, zero_point=zero_point, axis=axis
            )
            y = mb.add(x=dequantized_2, y=dequantized_2)
            quantized_2 = mb.quantize(input=y, scale=0.1, output_dtype="int8")
            # dequantize output
            dequantized_3 = mb.dequantize(input=quantized_2, scale=0.1)
            return dequantized_3

        prev_prog, _, _ = apply_pass_and_basic_check(
            prog, "common::dequantize_quantize_pair_elimination"
        )
        assert get_op_types_in_program(prev_prog) == [
            "quantize",
            "dequantize",
            "quantize",
            "dequantize",
            "add",
            "quantize",
            "dequantize",
        ]
        # As expected, dequantize_1 -> quantize_1 gets eliminated.
        # On the other hand, even with same scales and zero points and axes,
        # quantize_0 -> dequantize_2 and quantize_2 -> dequantize_3 are kept.
        assert get_op_types_in_program(prog) == [
            "quantize",
            "dequantize",
            "add",
            "quantize",
            "dequantize",
        ]

    @pytest.mark.parametrize(
        "is_zp_present, is_axis_present, is_shifted_zp_present",
        itertools.product(
            (True, False),
            (True, False),
            (True, False),
        ),
    )
    def test_keep_unidentical_dequantize_quantize(
        self, is_zp_present, is_axis_present, is_shifted_zp_present
    ):
        """
        Input graph:
            input -> quantize0 -> dequantize1(scale1, zp1) -> quantize1(scale2, zp2) -> dequantize2 -> add -> quantize2 -> dequantize3 -> output

        Nothing changes when dequantize1 and quantize1 have different parameters
        """

        SHAPE = (2, 3, 5)
        scale, zero_point, axis = self.generate_scale_zp_axis(SHAPE, is_zp_present, is_axis_present)

        @mb.program(input_specs=[mb.TensorSpec(shape=SHAPE, dtype=types.fp32)])
        def prog(x):
            # quantize input
            quantized_0 = mb.quantize(
                input=x, scale=scale, zero_point=zero_point, axis=axis, output_dtype="int8"
            )
            # non-redundant dequantize-quantize pair
            # this pattern can emerge from a (future) graph pass
            dequantized_1 = mb.dequantize(
                input=quantized_0, scale=scale, zero_point=zero_point, axis=axis
            )
            if is_zp_present:
                # input graph:
                #     dequantize -> add(y=const) -> quantize
                # output graph:
                #     dequantize -> quantize(zero_point += const / scale)
                if is_shifted_zp_present:
                    shifted_zero_point = (
                        (zero_point + 1.0).astype(np.int8)
                        if is_axis_present
                        else np.int8(zero_point + 1.0)
                    )
                    quantized_1 = mb.quantize(
                        input=dequantized_1,
                        scale=scale,
                        zero_point=shifted_zero_point,
                        axis=axis,
                        output_dtype="int8",
                    )
                else:
                    quantized_1 = mb.quantize(
                        input=dequantized_1, scale=scale, axis=axis, output_dtype="int8"
                    )
            else:
                # input graph:
                #     dequantize(zero_point=0) -> mul(y=const) -> quantize(zero_point=0)
                # output graph:
                #     dequantize(zero_point=0) -> quantize(scale /= const, zero_point=0)
                quantized_1 = mb.quantize(
                    input=dequantized_1, scale=scale / 2.0, axis=axis, output_dtype="int8"
                )
            # dequantize-op-quantize sandwich
            dequantized_2 = mb.dequantize(
                input=quantized_1, scale=scale, zero_point=zero_point, axis=axis
            )
            y = mb.add(x=dequantized_2, y=dequantized_2)
            quantized_2 = mb.quantize(input=y, scale=0.1, output_dtype="int8")
            # dequantize output
            dequantized_3 = mb.dequantize(input=quantized_2, scale=0.1)
            return dequantized_3

        prev_prog, _, _ = apply_pass_and_basic_check(
            prog, "common::dequantize_quantize_pair_elimination"
        )
        assert get_op_types_in_program(prev_prog) == [
            "quantize",
            "dequantize",
            "quantize",
            "dequantize",
            "add",
            "quantize",
            "dequantize",
        ]
        # nothing gets eliminated
        assert get_op_types_in_program(prog) == [
            "quantize",
            "dequantize",
            "quantize",
            "dequantize",
            "add",
            "quantize",
            "dequantize",
        ]

    @pytest.mark.parametrize(
        "is_zp_present, is_axis_present",
        itertools.product(
            (True, False),
            (True, False),
        ),
    )
    def test_keep_block_output_dequantize(self, is_zp_present, is_axis_present):
        """
        Input graph:
            input -> quantize0 -> dequantize1 -> quantize1 -> dequantize2 -> add -> quantize2 -> dequantize3 -> output

        Nothing changes when dequantize1 is a block output
        """

        SHAPE = (2, 3)
        scale, zero_point, axis = self.generate_scale_zp_axis(SHAPE, is_zp_present, is_axis_present)

        @mb.program(input_specs=[mb.TensorSpec(shape=SHAPE, dtype=types.fp32)])
        def prog(x):
            # quantize input
            quantized_0 = mb.quantize(
                input=x, scale=scale, zero_point=zero_point, axis=axis, output_dtype="int8"
            )
            # redundant dequantize-quantize pair
            dequantized_1 = mb.dequantize(
                input=quantized_0, scale=scale, zero_point=zero_point, axis=axis
            )
            quantized_1 = mb.quantize(
                input=dequantized_1,
                scale=scale,
                zero_point=zero_point,
                axis=axis,
                output_dtype="int8",
            )
            # dequantize-op-quantize sandwich
            dequantized_2 = mb.dequantize(
                input=quantized_1, scale=scale, zero_point=zero_point, axis=axis
            )
            y = mb.add(x=dequantized_2, y=dequantized_2)
            quantized_2 = mb.quantize(input=y, scale=0.1, output_dtype="int8")
            # dequantize output
            dequantized_3 = mb.dequantize(input=quantized_2, scale=0.1)
            return dequantized_1, dequantized_3

        prev_prog, _, _ = apply_pass_and_basic_check(
            prog, "common::dequantize_quantize_pair_elimination"
        )
        assert get_op_types_in_program(prev_prog) == [
            "quantize",
            "dequantize",
            "quantize",
            "dequantize",
            "add",
            "quantize",
            "dequantize",
        ]
        # nothing gets eliminated
        assert get_op_types_in_program(prog) == [
            "quantize",
            "dequantize",
            "quantize",
            "dequantize",
            "add",
            "quantize",
            "dequantize",
        ]

    @pytest.mark.parametrize(
        "is_zp_present, is_axis_present",
        itertools.product(
            (True, False),
            (True, False),
        ),
    )
    def test_keep_multichildren_dequantize(self, is_zp_present, is_axis_present):
        """
        Input graph:
                                               |-> quantize1 -> dequantize2 -> add -> quantize2 -> dequantize3 -> output1
            input -> quantize0 -> dequantize1 -|
                                               |-> mul -> quantize -> dequantize -> output2

        Output graph:
                        |-> dequantize2 -> add -> quantize2 -> dequantize3 -> output1
            input -> quantize0 -> dequantize1 -|
                                               |-> mul -> quantize -> dequantize -> output2

        As `dequantize1` has multiple children, we don't eliminate it, but can remove the child `quantize1`.
        """

        SHAPE = (2, 3)
        scale, zero_point, axis = self.generate_scale_zp_axis(SHAPE, is_zp_present, is_axis_present)

        @mb.program(input_specs=[mb.TensorSpec(shape=SHAPE, dtype=types.fp32)])
        def prog(x):
            # quantize input
            quantized_0 = mb.quantize(
                input=x, scale=scale, zero_point=zero_point, axis=axis, output_dtype="int8"
            )
            # redundant dequantize-quantize pair
            dequantized_1 = mb.dequantize(
                input=quantized_0, scale=scale, zero_point=zero_point, axis=axis
            )
            quantized_1 = mb.quantize(
                input=dequantized_1,
                scale=scale,
                zero_point=zero_point,
                axis=axis,
                output_dtype="int8",
            )
            # dequantize-op-quantize sandwich
            dequantized_2 = mb.dequantize(
                input=quantized_1, scale=scale, zero_point=zero_point, axis=axis
            )
            y = mb.add(x=dequantized_2, y=dequantized_2)
            quantized_2 = mb.quantize(input=y, scale=0.1, output_dtype="int8")
            # dequantize output
            dequantized_3 = mb.dequantize(input=quantized_2, scale=0.1)

            # now add another usage of dequantized_1
            z = mb.mul(x=dequantized_1, y=dequantized_1)
            quantized_z = mb.quantize(input=z, scale=0.2, output_dtype="int8")
            dequantized_z = mb.dequantize(input=quantized_z, scale=0.2)

            return dequantized_3, dequantized_z

        prev_prog, _, _ = apply_pass_and_basic_check(
            prog, "common::dequantize_quantize_pair_elimination"
        )
        assert get_op_types_in_program(prev_prog) == [
            "quantize",
            "dequantize",
            "quantize",
            "dequantize",
            "add",
            "quantize",
            "dequantize",
            "mul",
            "quantize",
            "dequantize",
        ]
        # The `quantize` before `add` got eliminated.
        assert get_op_types_in_program(prog) == [
            "quantize",
            "dequantize",
            "dequantize",
            "add",
            "quantize",
            "dequantize",
            "mul",
            "quantize",
            "dequantize",
        ]


@pytest.mark.skipif(ct.utils._macos_version() < (14, 0), reason="Requires Core ML 7")
class TestDistributiveQuantizedBinaryOpScaleNormalization(QuantizationBaseTest):
    @pytest.mark.parametrize(
        "op_type, has_relu_fusion, input_rank, is_axis_x_present",
        itertools.product(
            ("add", "sub"),
            (True, False),
            (1, 3, 5),
            (True, False),
        ),
    )
    def test_normalize(self, op_type, has_relu_fusion, input_rank, is_axis_x_present):
        """
        Input graph:
            x -> quantize(scale_x) -> dequantize(scale_x) -|
                                                           |-> add/sub (-> relu) -> dequantize(scale_z) -> output
            y -> quantize(scale_y) -> dequantize(scale_y) -|

        Output graph:
            x -> quantize(scale_x) -> dequantize(scale_x/scale_y) -|
                                                                   |-> add/sub (-> relu) -> dequantize(scale_z/scale_y) -> output
            y -> quantize(scale_y) -> dequantize(1.0)             -|

        x and y may get swapped to have the one with scalar scale being new "y"
        """

        # if axis_x is present, then axis_y is not, vice versa,
        # so that one of scale_x or scale_y is scalar
        SHAPE = np.random.randint(1, 5, size=input_rank, dtype=np.int32)
        scale_x, zero_point_x, axis_x = self.generate_random_quantization_params(
            np.float32, np.int8, SHAPE, True, is_axis_x_present
        )
        scale_y, zero_point_y, axis_y = self.generate_random_quantization_params(
            np.float32, np.int8, SHAPE, True, not is_axis_x_present
        )
        scale_z, zero_point_z, axis_z = self.generate_random_quantization_params(
            np.float32, np.int8, SHAPE, True, True
        )

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=SHAPE, dtype=types.fp32),
                mb.TensorSpec(shape=SHAPE, dtype=types.fp32),
            ]
        )
        def prog(x, y):
            # quantize input
            quantize_x = mb.quantize(
                input=x, scale=scale_x, zero_point=zero_point_x, axis=axis_x, output_dtype="int8"
            )
            quantize_y = mb.quantize(
                input=y, scale=scale_y, zero_point=zero_point_y, axis=axis_y, output_dtype="int8"
            )
            # quantized binary op
            dequantize_x = mb.dequantize(
                input=quantize_x, scale=scale_x, zero_point=zero_point_x, axis=axis_x
            )
            dequantize_y = mb.dequantize(
                input=quantize_y, scale=scale_y, zero_point=zero_point_y, axis=axis_y
            )
            z = None
            if op_type == "add":
                z = mb.add(x=dequantize_x, y=dequantize_y)
            elif op_type == "sub":
                z = mb.sub(x=dequantize_x, y=dequantize_y)
            else:
                raise ValueError("unsupported op type")
            if has_relu_fusion:
                z = mb.relu(x=z)
            quantize_z = mb.quantize(
                input=z, scale=scale_z, zero_point=zero_point_z, axis=axis_z, output_dtype="int8"
            )
            # dequantize output
            dequantize_z = mb.dequantize(
                input=quantize_z, scale=scale_z, zero_point=zero_point_z, axis=axis_z
            )
            return dequantize_z

        # dequantize_x, dequantize_y, z
        prev_prog, _, _ = apply_pass_and_basic_check(
            prog, "common::distributive_quantized_binary_op_scale_normalization"
        )
        # dequantize_x, dequantize_y, dequantize_x_normalized, dequantize_y_normalized, z
        _, _, _ = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        # dequantize_x_normalized, dequantize_y_normalized, z

        scale_prev_dequantize_x = prev_prog.find_ops(op_type="dequantize")[0].scale.val
        scale_prev_dequantize_y = prev_prog.find_ops(op_type="dequantize")[1].scale.val
        scale_prev_quantize_z = prev_prog.find_ops(op_type="quantize")[-1].scale.val
        assert np.all(scale_prev_dequantize_x == scale_x)
        assert np.all(scale_prev_dequantize_y == scale_y)
        assert np.all(scale_prev_quantize_z == scale_z)

        scale_dequantize_x = prog.find_ops(op_type="dequantize")[0].scale.val
        scale_dequantize_y = prog.find_ops(op_type="dequantize")[1].scale.val
        scale_quantize_z = prog.find_ops(op_type="quantize")[-1].scale.val
        # if axis_x is present, then scale_y gets normalized
        # else, scale_x gets normalized, and x and y will get swapped
        assert np.all(
            scale_dequantize_x == scale_x / scale_y if is_axis_x_present else scale_y / scale_x
        )
        assert np.all(scale_dequantize_y == 1.0)
        assert np.all(
            scale_quantize_z == scale_z / scale_y if is_axis_x_present else scale_z / scale_x
        )

        prev_model = ct.convert(
            prev_prog,
            source="milinternal",
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS17,
        )
        model = ct.convert(
            prog,
            source="milinternal",
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS17,
        )

        x = self.generate_random_quantize_input(
            np.float32, np.int8, scale_x, zero_point_x, axis_x, SHAPE
        )
        y = self.generate_random_quantize_input(
            np.float32, np.int8, scale_y, zero_point_y, axis_y, SHAPE
        )
        prev_output = list(prev_model.predict({"x": x, "y": y}).values())[0]
        output = list(model.predict({"x": x, "y": y}).values())[0]
        assert np.all(prev_output == output)

    def test_normalize_versatile_inputs(self):
        """
        Input graph:
                                               |-> exp -> dequantize(scale_z)
                                               |
            x -> quantize(scale_x) -> dequantize(scale_x) -|
                                                           |-> add -> dequantize(scale_z) -> output
            y -> quantize(scale_y) -> dequantize(scale_y) -|

        Output graph:
                         |-> dequantize(scale_x) -> exp -> dequantize(scale_z)
                         |
            x -> quantize(scale_x) -> dequantize(scale_x/scale_y) -|
                                                                   |-> add -> dequantize(scale_z/scale_y) -> output
            y -> quantize(scale_y) -> dequantize(1.0)             -|
        """

        SHAPE = (2, 1)
        scale_x, zero_point_x, axis_x = np.float32(0.2), None, None
        scale_y, zero_point_y, axis_y = np.float32(0.3), None, None
        scale_z, zero_point_z, axis_z = np.float32(0.5), None, None

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=SHAPE, dtype=types.fp32),
                mb.TensorSpec(shape=SHAPE, dtype=types.fp32),
            ]
        )
        def prog(x, y):
            # quantize input
            quantize_x = mb.quantize(
                input=x, scale=scale_x, zero_point=zero_point_x, axis=axis_x, output_dtype="uint8"
            )
            quantize_y = mb.quantize(
                input=y, scale=scale_y, zero_point=zero_point_y, axis=axis_y, output_dtype="uint8"
            )
            # quantized binary op
            dequantize_x = mb.dequantize(
                input=quantize_x, scale=scale_x, zero_point=zero_point_x, axis=axis_x
            )
            dequantize_y = mb.dequantize(
                input=quantize_y, scale=scale_y, zero_point=zero_point_y, axis=axis_y
            )
            z = mb.add(x=dequantize_x, y=dequantize_y)
            quantize_z = mb.quantize(
                input=z, scale=scale_z, zero_point=zero_point_z, axis=axis_z, output_dtype="uint8"
            )
            # another quantized op
            z1 = mb.exp(x=dequantize_x)
            quantize_z1 = mb.quantize(
                input=z1, scale=scale_z, zero_point=zero_point_z, axis=axis_z, output_dtype="uint8"
            )
            # dequantize output
            dequantize_z = mb.dequantize(
                input=quantize_z, scale=scale_z, zero_point=zero_point_z, axis=axis_z
            )
            dequantize_z1 = mb.dequantize(
                input=quantize_z1, scale=scale_z, zero_point=zero_point_z, axis=axis_z
            )
            return dequantize_z, dequantize_z1

        prev_prog, _, _ = apply_pass_and_basic_check(
            prog, "common::distributive_quantized_binary_op_scale_normalization"
        )
        _, _, _ = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        scale_prev_dequantize_x = prev_prog.find_ops(op_type="dequantize")[0].scale.val
        scale_prev_dequantize_y = prev_prog.find_ops(op_type="dequantize")[1].scale.val
        scale_prev_quantize_z = prev_prog.find_ops(op_type="quantize")[-2].scale.val
        assert np.all(scale_prev_dequantize_x == scale_x)
        assert np.all(scale_prev_dequantize_y == scale_y)
        assert np.all(scale_prev_quantize_z == scale_z)

        scale_dequantize_x_to_z1 = prog.find_ops(op_type="dequantize")[0].scale.val
        scale_dequantize_x_to_z = prog.find_ops(op_type="dequantize")[1].scale.val
        scale_dequantize_y = prog.find_ops(op_type="dequantize")[2].scale.val
        scale_quantize_z = prog.find_ops(op_type="quantize")[-2].scale.val
        assert np.all(scale_dequantize_x_to_z1 == scale_x)
        assert np.all(scale_dequantize_x_to_z == scale_x / scale_y)
        assert np.all(scale_dequantize_y == 1.0)
        assert np.all(scale_quantize_z == scale_z / scale_y)

    def test_skip_0_scale(self):
        """
        Input graph:
            x -> quantize(eps) -> dequantize(eps) -|
                                                   |-> add -> dequantize -> output
            y -> quantize(eps) -> dequantize(eps) -|

        Nothing changes due to underflow scale
        """

        # consider anything underflows fp16 to be 0
        SHAPE = (1, 2)
        scale_x, zero_point_x, axis_x = np.float32(5e-8), None, None
        scale_y, zero_point_y, axis_y = np.float32(-5e-8), None, None
        scale_z, zero_point_z, axis_z = np.float32(0.8), None, None

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=SHAPE, dtype=types.fp32),
                mb.TensorSpec(shape=SHAPE, dtype=types.fp32),
            ]
        )
        def prog(x, y):
            # quantize input
            quantize_x = mb.quantize(
                input=x, scale=scale_x, zero_point=zero_point_x, axis=axis_x, output_dtype="uint8"
            )
            quantize_y = mb.quantize(
                input=y, scale=scale_y, zero_point=zero_point_y, axis=axis_y, output_dtype="uint8"
            )
            # quantized binary op
            dequantize_x = mb.dequantize(
                input=quantize_x, scale=scale_x, zero_point=zero_point_x, axis=axis_x
            )
            dequantize_y = mb.dequantize(
                input=quantize_y, scale=scale_y, zero_point=zero_point_y, axis=axis_y
            )
            z = mb.add(x=dequantize_x, y=dequantize_y)
            quantize_z = mb.quantize(
                input=z, scale=scale_z, zero_point=zero_point_z, axis=axis_z, output_dtype="uint8"
            )
            # dequantize output
            dequantize_z = mb.dequantize(
                input=quantize_z, scale=scale_z, zero_point=zero_point_z, axis=axis_z
            )
            return dequantize_z

        prev_prog, _, _ = apply_pass_and_basic_check(
            prog, "common::distributive_quantized_binary_op_scale_normalization"
        )

        scale_prev_dequantize_x = prev_prog.find_ops(op_type="dequantize")[0].scale.val
        scale_prev_dequantize_y = prev_prog.find_ops(op_type="dequantize")[1].scale.val
        scale_prev_quantize_z = prev_prog.find_ops(op_type="quantize")[-1].scale.val
        assert np.all(scale_prev_dequantize_x == scale_x)
        assert np.all(scale_prev_dequantize_y == scale_y)
        assert np.all(scale_prev_quantize_z == scale_z)

        scale_dequantize_x = prog.find_ops(op_type="dequantize")[0].scale.val
        scale_dequantize_y = prog.find_ops(op_type="dequantize")[1].scale.val
        scale_quantize_z = prog.find_ops(op_type="quantize")[-1].scale.val
        assert np.all(scale_dequantize_x == scale_x)
        assert np.all(scale_dequantize_y == scale_y)
        assert np.all(scale_quantize_z == scale_z)

    @pytest.mark.parametrize("input_rank", (1, 2, 5))
    def test_skip_2_vector_scales(self, input_rank):
        """
        Input graph:
            x -> quantize(scale_x) -> dequantize(scale_x) -|
                                                           |-> add -> dequantize(scale_z) -> output
            y -> quantize(scale_y) -> dequantize(scale_y) -|

        Nothing changes when both scale_x and scale_y are vectors
        """

        # axis_x and axis_y are both present
        SHAPE = np.random.randint(1, 5, size=input_rank, dtype=np.int32)
        scale_x, zero_point_x, axis_x = self.generate_random_quantization_params(
            np.float16, np.uint8, SHAPE, False, True
        )
        scale_y, zero_point_y, axis_y = self.generate_random_quantization_params(
            np.float16, np.uint8, SHAPE, False, True
        )
        scale_z, zero_point_z, axis_z = self.generate_random_quantization_params(
            np.float16, np.uint8, SHAPE, False, False
        )

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=SHAPE, dtype=types.fp16),
                mb.TensorSpec(shape=SHAPE, dtype=types.fp16),
            ]
        )
        def prog(x, y):
            # quantize input
            quantize_x = mb.quantize(
                input=x, scale=scale_x, zero_point=zero_point_x, axis=axis_x, output_dtype="uint8"
            )
            quantize_y = mb.quantize(
                input=y, scale=scale_y, zero_point=zero_point_y, axis=axis_y, output_dtype="uint8"
            )
            # quantized binary op
            dequantize_x = mb.dequantize(
                input=quantize_x, scale=scale_x, zero_point=zero_point_x, axis=axis_x
            )
            dequantize_y = mb.dequantize(
                input=quantize_y, scale=scale_y, zero_point=zero_point_y, axis=axis_y
            )
            z = mb.add(x=dequantize_x, y=dequantize_y)
            quantize_z = mb.quantize(
                input=z, scale=scale_z, zero_point=zero_point_z, axis=axis_z, output_dtype="uint8"
            )
            # dequantize output
            dequantize_z = mb.dequantize(
                input=quantize_z, scale=scale_z, zero_point=zero_point_z, axis=axis_z
            )
            return dequantize_z

        prev_prog, _, _ = apply_pass_and_basic_check(
            prog, "common::distributive_quantized_binary_op_scale_normalization"
        )

        scale_prev_dequantize_x = prev_prog.find_ops(op_type="dequantize")[0].scale.val
        scale_prev_dequantize_y = prev_prog.find_ops(op_type="dequantize")[1].scale.val
        scale_prev_quantize_z = prev_prog.find_ops(op_type="quantize")[-1].scale.val
        assert np.all(scale_prev_dequantize_x == scale_x)
        assert np.all(scale_prev_dequantize_y == scale_y)
        assert np.all(scale_prev_quantize_z == scale_z)

        scale_dequantize_x = prog.find_ops(op_type="dequantize")[0].scale.val
        scale_dequantize_y = prog.find_ops(op_type="dequantize")[1].scale.val
        scale_quantize_z = prog.find_ops(op_type="quantize")[-1].scale.val
        assert np.all(scale_dequantize_x == scale_x)
        assert np.all(scale_dequantize_y == scale_y)
        assert np.all(scale_quantize_z == scale_z)


class TestDequantizeToConstexpr:
    @pytest.mark.parametrize(
        "float_dtype, quant_dtype, is_scalar, is_zp_present",
        itertools.product(
            (np.float32, np.float16),
            (np.int8, np.uint8),
            (True, False),
            (True, False),
        ),
    )
    def test_dequantize_const_to_constexpr(
        self, float_dtype, quant_dtype, is_scalar, is_zp_present
    ):
        """
        Input graph:
            input -> dequantize -> output

        Output graph:
            input -> constexpr_affine_dequantize -> output
        """

        @mb.program(input_specs=[])
        def prog():
            y = None
            if is_scalar:
                if is_zp_present:
                    y = mb.dequantize(
                        input=np.array([10, 11], dtype=quant_dtype),
                        scale=float_dtype(0.1),
                        zero_point=quant_dtype(2),
                    )
                else:
                    y = mb.dequantize(
                        input=np.array([13, 14, 15], dtype=quant_dtype), scale=float_dtype(0.2)
                    )
            else:
                if is_zp_present:
                    y = mb.dequantize(
                        input=np.array([[10, 11], [12, 13], [14, 15]], dtype=quant_dtype),
                        scale=np.array([0.1, 0.2, 0.3], dtype=float_dtype),
                        zero_point=np.array([6, 7, 8], dtype=quant_dtype),
                        axis=0,
                    )
                else:
                    y = mb.dequantize(
                        input=np.array([[19, 20, 21], [22, 23, 24]], dtype=quant_dtype),
                        scale=np.array([0.4, 0.5, 0.6], dtype=float_dtype),
                        axis=1,
                    )
            return y

        assert get_op_types_in_program(prog) == ["dequantize"]
        dequantize_op = prog.find_ops(op_type="dequantize")[0]
        assert dequantize_op.outputs[0].val is None
        assert dequantize_op.can_materialize_val()

        apply_pass_and_basic_check(prog, "common::dequantize_to_constexpr")
        assert get_op_types_in_program(prog) == ["constexpr_affine_dequantize"]

    @pytest.mark.parametrize(
        "float_dtype, quant_dtype, is_scalar, is_zp_present",
        itertools.product(
            (np.float32, np.float16),
            (np.int8, np.uint8),
            (True, False),
            (True, False),
        ),
    )
    def test_dequantize_variable_unchanged(
        self, float_dtype, quant_dtype, is_scalar, is_zp_present
    ):
        """
        Input graph:
            input -> dequantize -> output

        Output graph:
            input -> dequantize -> output
        """

        if is_scalar:
            if is_zp_present:

                @mb.program(
                    input_specs=[
                        mb.TensorSpec(
                            shape=(1, 2, 3, 4, 5), dtype=numpy_type_to_builtin_type(quant_dtype)
                        )
                    ]
                )
                def prog(x):
                    y = mb.dequantize(input=x, scale=float_dtype(0.1), zero_point=quant_dtype(1))
                    return y

            else:

                @mb.program(
                    input_specs=[
                        mb.TensorSpec(
                            shape=(4, 3, 2, 1), dtype=numpy_type_to_builtin_type(quant_dtype)
                        )
                    ]
                )
                def prog(x):
                    y = mb.dequantize(input=x, scale=float_dtype(0.2))
                    return y

        else:
            if is_zp_present:

                @mb.program(
                    input_specs=[
                        mb.TensorSpec(shape=(3, 2), dtype=numpy_type_to_builtin_type(quant_dtype))
                    ]
                )
                def prog(x):
                    y = mb.dequantize(
                        input=x,
                        scale=np.array([0.1, 0.2, 0.3], dtype=float_dtype),
                        zero_point=np.array([1, 2, 3], dtype=quant_dtype),
                        axis=0,
                    )
                    return y

            else:

                @mb.program(
                    input_specs=[
                        mb.TensorSpec(shape=(2, 3), dtype=numpy_type_to_builtin_type(quant_dtype))
                    ]
                )
                def prog(x):
                    y = mb.dequantize(
                        input=x,
                        scale=np.array([0.4, 0.5, 0.6], dtype=float_dtype),
                        axis=1,
                    )
                    return y

        assert get_op_types_in_program(prog) == ["dequantize"]

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::dequantize_to_constexpr"
        )
        assert get_op_types_in_program(prog) == ["dequantize"]


@pytest.mark.skipif(ct.utils._macos_version() < (15, 0), reason="Only supported on macOS 15+")
class TestReorderLutPerChannelScale:
    @staticmethod
    def _verify_numerical(prev_prog, prog, block, input_shape, rtol=1e-7, atol=0.0):
        # Verify the numerical output matches before and after the reordering.
        prev_model = ct.convert(
            prev_prog,
            pass_pipeline=ct.PassPipeline.EMPTY,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
        )
        model = ct.convert(
            prog,
            pass_pipeline=ct.PassPipeline.EMPTY,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
        )
        output_name = block.outputs[0].name
        x_val = np.random.rand(*input_shape).astype(np.float16)
        input_dict = {"x": x_val}
        prev_output = prev_model.predict(input_dict)[output_name]
        output = model.predict(input_dict)[output_name]
        np.testing.assert_allclose(prev_output, output, rtol=rtol, atol=atol)

    @staticmethod
    def _get_lut_pcs_weight(shape: Tuple[int, ...], nbits=4, scale_axis: int = 0):
        """Get a specific shape of weight produced by lut with per-channel-scale (pcs)."""
        num_palette = 2**nbits
        np_dtype = types.nptype_from_builtin(types.string_to_builtin(f"uint{nbits}"))
        indices = np.arange(np.prod(shape)).reshape(shape).astype(np_dtype)
        lut_shape = shape + (num_palette, 1)
        lut = np.arange(np.prod(lut_shape)).reshape(lut_shape).astype(np.float16)

        lut_op = mb.constexpr_lut_to_dense(indices=indices, lut=lut)
        scale_shape = [1] * len(shape)
        scale_shape[scale_axis] = shape[scale_axis]
        scale_shape = tuple(scale_shape)
        scale_val = np.arange(1, np.prod(scale_shape) + 1).reshape(scale_shape).astype(np.float16)
        return mb.constexpr_blockwise_shift_scale(
            data=lut_op,
            scale=scale_val,
        )

    @pytest.mark.parametrize(
        "input_shape, has_bias", itertools.product([(4, 3), (2, 3, 2), (1, 2, 3, 4)], [True, False])
    )
    def test_reorder_scale_linear(self, input_shape: Tuple[int, ...], has_bias: bool):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=input_shape, dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            scaled_weight = self._get_lut_pcs_weight((2, input_shape[-1]))
            bias = np.array([20, 50], dtype=np.float16) if has_bias else None
            output = mb.linear(x=x, weight=scaled_weight, bias=bias)
            return mb.add(x=output, y=np.float16(1.0))

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::reorder_lut_per_channel_scale", skip_essential_scope_check=True
        )
        assert get_op_types_in_program(prev_prog) == [
            "constexpr_lut_to_dense",
            "constexpr_blockwise_shift_scale",
            "linear",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["constexpr_lut_to_dense", "linear", "mul", "add"]
        self._verify_numerical(prev_prog, prog, block, input_shape)

    @pytest.mark.parametrize(
        "use_y_as_weight, transpose_x, transpose_y",
        itertools.product([True, False], [True, False], [True, False]),
    )
    def test_reorder_scale_matmul(self, use_y_as_weight, transpose_x, transpose_y):
        input_shape = (3, 4)

        @mb.program(
            input_specs=[mb.TensorSpec(shape=input_shape, dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            if use_y_as_weight:
                if transpose_x:  # x shape is (4, 3)
                    weight_shape = (2, 3) if transpose_y else (3, 2)
                else:  # x shape is (3, 4)
                    weight_shape = (2, 4) if transpose_y else (4, 2)
                scaled_weight = self._get_lut_pcs_weight(
                    weight_shape, scale_axis=0 if transpose_y else 1
                )
                output = mb.matmul(
                    x=x, y=scaled_weight, transpose_x=transpose_x, transpose_y=transpose_y
                )
            else:
                if transpose_y:  # y shape is (4, 3)
                    weight_shape = (4, 2) if transpose_x else (2, 4)
                else:  # y shape is (3, 4)
                    weight_shape = (3, 2) if transpose_x else (2, 3)
                scaled_weight = self._get_lut_pcs_weight(
                    weight_shape, scale_axis=1 if transpose_x else 0
                )
                output = mb.matmul(
                    x=scaled_weight, y=x, transpose_x=transpose_x, transpose_y=transpose_y
                )
            return mb.add(x=output, y=np.float16(1.0))

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::reorder_lut_per_channel_scale", skip_essential_scope_check=True
        )
        assert get_op_types_in_program(prev_prog) == [
            "constexpr_lut_to_dense",
            "constexpr_blockwise_shift_scale",
            "matmul",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["constexpr_lut_to_dense", "matmul", "mul", "add"]
        self._verify_numerical(prev_prog, prog, block, input_shape)

    @pytest.mark.parametrize(
        "pad_type, has_bias, has_strides_dilations",
        itertools.product(["valid", "same", "same_lower", "custom"], [True, False], [True, False]),
    )
    def test_reorder_scale_conv(self, pad_type, has_bias, has_strides_dilations):
        input_shape = (4, 3, 4, 3)

        @mb.program(
            input_specs=[mb.TensorSpec(shape=input_shape, dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            scaled_weight = self._get_lut_pcs_weight((2, 3, 2, 2), nbits=6)
            bias = np.array([20, 50], dtype=np.float16) if has_bias else None
            pad = [1, 1, 1, 1] if pad_type == "custom" else None
            strides = [1, 2] if has_strides_dilations else None
            dilations = [1, 2] if has_strides_dilations else None
            output = mb.conv(
                x=x,
                weight=scaled_weight,
                strides=strides,
                pad_type=pad_type,
                pad=pad,
                dilations=dilations,
                bias=bias,
            )
            return mb.add(x=output, y=np.float16(1.0))

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::reorder_lut_per_channel_scale", skip_essential_scope_check=True
        )
        assert get_op_types_in_program(prev_prog) == [
            "constexpr_lut_to_dense",
            "constexpr_blockwise_shift_scale",
            "conv",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["constexpr_lut_to_dense", "conv", "mul", "add"]
        self._verify_numerical(prev_prog, prog, block, input_shape)

    @pytest.mark.parametrize(
        "input_shape, has_bias", itertools.product([(4, 3), (2, 3, 2), (1, 2, 3, 4)], [True, False])
    )
    def test_reorder_multiple_usages(self, input_shape: Tuple[int, ...], has_bias: bool):
        """The scaled weight is used by multiple ops."""

        @mb.program(
            input_specs=[mb.TensorSpec(shape=input_shape, dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            scaled_weight = self._get_lut_pcs_weight((2, input_shape[-1]))
            bias = np.array([20, 50], dtype=np.float16) if has_bias else None
            linear_output = mb.linear(x=x, weight=scaled_weight, bias=bias)
            matmul_output = mb.matmul(x=x, y=scaled_weight, transpose_x=False, transpose_y=True)
            return mb.add(x=linear_output, y=matmul_output)

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::reorder_lut_per_channel_scale", skip_essential_scope_check=True
        )
        assert get_op_types_in_program(prev_prog) == [
            "constexpr_lut_to_dense",
            "constexpr_blockwise_shift_scale",
            "linear",
            "matmul",
            "add",
        ]
        assert get_op_types_in_program(prog) == [
            "constexpr_lut_to_dense",
            "linear",
            "mul",
            "matmul",
            "mul",
            "add",
        ]
        self._verify_numerical(prev_prog, prog, block, input_shape)

    def test_reorder_not_happen(self):
        """The scale won't be moved when the scaled weight is used in unsupported ops."""

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(4, 16), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            scaled_weight = self._get_lut_pcs_weight((2, 16))
            linear_output1 = mb.linear(x=x, weight=scaled_weight)
            add_out = mb.add(x=scaled_weight, y=np.float16(1.0))
            linear_output2 = mb.linear(x=x, weight=add_out)
            return mb.add(x=linear_output1, y=linear_output2)

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::reorder_lut_per_channel_scale", skip_essential_scope_check=True
        )
        assert get_op_types_in_program(prog) == get_op_types_in_program(prev_prog)


class TestFP16CastTransform:
    def assertEqual(self, first, second):
        """A convenience method to migrate from unittest (self.assertEqual) to pytest."""
        assert first == second

    def test_single_input_to_single_operation(self):
        """
        Input graph:
            input -> square -> output

        Output graph:
            input -> cast(fp32->fp16) -> square -> cast(fp16->fp32) -> output
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.square(x=x)
            return x

        self.assertEqual(get_op_types_in_program(prog), ["square"])

        apply_pass_and_basic_check(
            prog, quantization.FP16ComputePrecision(op_selector=lambda op: True)
        )
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["cast", "square", "cast"])

        # Asserting first cast configuration
        cast_1 = block.find_ops(op_type="cast")[0]
        self.assertEqual(cast_1.dtype.val, "fp16")
        self.assertEqual(len(cast_1.outputs), 1)
        self.assertEqual(len(cast_1.outputs[0].child_ops), 1)
        self.assertEqual(cast_1.outputs[0].child_ops[0].op_type, "square")

        # Asserting second cast configuration
        cast_2 = block.find_ops(op_type="cast")[1]
        self.assertEqual(cast_2.dtype.val, "fp32")
        self.assertEqual(len(cast_2.outputs), 1)
        self.assertEqual(len(cast_2.outputs[0].child_ops), 0)

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20)},
        )

    @parameterized.parameterized.expand([[1.0], [-1.0]])
    def test_inf(self, sign):
        """
        Input graph:
            input -> add(±2e38) -> tanh -> output

        Output graph:
            input -> cast(fp32->fp16) -> add(±inf) -> tanh -> cast(fp16->fp32) -> output
        """

        SHAPE = (2, 3)

        @mb.program(input_specs=[mb.TensorSpec(shape=SHAPE)])
        def prog(x):
            y = mb.add(x=x, y=np.float32(sign * 2e38))
            z = mb.tanh(x=y)
            return z

        assert get_op_types_in_program(prog) == ["add", "tanh"]

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::add_fp16_cast")
        apply_pass_and_basic_check(prog, "common::cast_optimization")
        apply_pass_and_basic_check(prog, "common::const_elimination")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["cast", "add", "tanh", "cast"]
        cast_to_fp16, cast_to_fp32 = prog.find_ops(op_type="cast")
        assert cast_to_fp16.dtype.val == "fp16"
        assert cast_to_fp32.dtype.val == "fp32"

        output_name = block.outputs[0].name
        assert_model_is_valid(prog, {"x": SHAPE}, expected_output_shapes={output_name: SHAPE})

        prev_model = ct.convert(prev_prog)
        model = ct.convert(prog)

        x = 65500.0 * np.random.rand(*SHAPE)
        prev_output = prev_model.predict({"x": x})[output_name]
        output = model.predict({"x": x})[output_name]
        np.allclose(prev_output, output)

    def test_fp16_overflow(self):
        """
        Input graph:
            input -> clip(-77777, 88888) -> output

        Nothing gets changed due to fp16 overflow
        """

        SHAPE = (2, 1, 3, 7, 5)

        @mb.program(input_specs=[mb.TensorSpec(shape=SHAPE)])
        def prog(x):
            y = mb.clip(x=x, alpha=np.float32(-77777), beta=np.float32(88888))
            return y

        assert get_op_types_in_program(prog) == ["clip"]

        apply_pass_and_basic_check(prog, "common::add_fp16_cast")

        assert get_op_types_in_program(prog) == ["clip"]

    def test_divide_by_zero_operation(self):
        """
        Input graph:
            input ------|
                        |-> div -> output
            const(eps) -|

        Output graph:
            input ------> cast(fp32->fp16) -|
                                            |-> div -> cast(fp16->fp32) -> output
            const(eps) -> cast(fp32->fp16) -|
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            eps = mb.const(val=1e-10)
            x = mb.real_div(x=x, y=eps)
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, quantization.FP16ComputePrecision(op_selector=lambda op: True)
        )

        mlmodel = ct.convert(prog, compute_units=ct.ComputeUnit.CPU_ONLY)
        input_dict = {"x": np.random.rand(10, 20) * 1e-3}

        if _IS_MACOS:
            prediction = mlmodel.predict(input_dict)
            assert not np.isnan(prediction["real_div_0"]).any()
            assert np.isfinite(prediction["real_div_0"]).all()

    def test_multiple_inputs_to_single_operation(self):
        """
        Input graph:
            input1 -|
                    |-> concat -> output
            input2 -|

        Output graph:
            input1 -> cast(fp32->fp16) -|
                                        |-> concat -> cast(fp16->fp32) -> output
            input2 -> cast(fp32->fp16) -|
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20)), mb.TensorSpec(shape=(10, 20))])
        def prog(x, y):
            x = mb.concat(values=(x, y), axis=0)
            return x

        self.assertEqual(get_op_types_in_program(prog), ["concat"])

        apply_pass_and_basic_check(
            prog, quantization.FP16ComputePrecision(op_selector=lambda op: True)
        )
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["cast", "cast", "concat", "cast"])

        # Asserting first cast configuration
        cast_1 = block.find_ops(op_type="cast")[0]
        self.assertEqual(cast_1.dtype.val, "fp16")
        self.assertEqual(len(cast_1.outputs), 1)
        self.assertEqual(len(cast_1.outputs[0].child_ops), 1)
        self.assertEqual(cast_1.outputs[0].child_ops[0].op_type, "concat")

        # Asserting second cast configuration
        cast_2 = block.find_ops(op_type="cast")[1]
        self.assertEqual(cast_2.dtype.val, "fp16")
        self.assertEqual(len(cast_2.outputs), 1)
        self.assertEqual(len(cast_2.outputs[0].child_ops), 1)
        self.assertEqual(cast_2.outputs[0].child_ops[0].op_type, "concat")

        # Asserting third cast configuration
        cast_3 = block.find_ops(op_type="cast")[2]
        self.assertEqual(cast_3.dtype.val, "fp32")
        self.assertEqual(len(cast_3.outputs), 1)
        self.assertEqual(len(cast_3.outputs[0].child_ops), 0)

        assert_model_is_valid(
            prog,
            {"x": (10, 20), "y": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (20, 20)},
        )

    def test_multiple_outputs_from_single_operation(self):
        """
        Input graph:
                            |-> output_1
            input -> split -|
                            |-> output_2

        Output graph:
                                                |-> cast(fp16->fp32) -> output_1
            input -> cast(fp32->fp16) -> split -|
                                                |-> cast(fp16->fp32) -> output_2
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.split(x=x, axis=0, num_splits=2)
            return x

        self.assertEqual(get_op_types_in_program(prog), ["split"])

        apply_pass_and_basic_check(
            prog, quantization.FP16ComputePrecision(op_selector=lambda op: True)
        )
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["cast", "split", "cast", "cast"])

        # Asserting first cast configuration
        cast_1 = block.find_ops(op_type="cast")[0]
        self.assertEqual(cast_1.dtype.val, "fp16")
        self.assertEqual(len(cast_1.outputs), 1)
        self.assertEqual(len(cast_1.outputs[0].child_ops), 1)
        self.assertEqual(cast_1.outputs[0].child_ops[0].op_type, "split")

        # Asserting second cast configuration
        cast_2 = block.find_ops(op_type="cast")[1]
        self.assertEqual(cast_2.dtype.val, "fp32")
        self.assertEqual(len(cast_2.outputs), 1)
        self.assertEqual(len(cast_2.outputs[0].child_ops), 0)

        # Asserting third cast configuration
        cast_3 = block.find_ops(op_type="cast")[2]
        self.assertEqual(cast_3.dtype.val, "fp32")
        self.assertEqual(len(cast_3.outputs), 1)
        self.assertEqual(len(cast_3.outputs[0].child_ops), 0)

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (5, 20), block.outputs[1].name: (5, 20)},
        )

    def test_single_input_to_multiple_operations(self):
        """
        Input graph:
                   |-> square -> output_1
            input -|
                   |->  relu  -> output_2

        Output graph:
                                       |-> square -> cast(fp16->fp32) -> output_1
            input -> cast(fp32->fp16) -|
                                       |->  relu  -> cast(fp16->fp32) -> output_2
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            y = mb.square(x=x)
            z = mb.relu(x=x)
            return y, z

        self.assertEqual(get_op_types_in_program(prog), ["square", "relu"])

        apply_pass_and_basic_check(
            prog, quantization.FP16ComputePrecision(op_selector=lambda op: True)
        )
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        self.assertEqual(get_op_types_in_program(prog), ["cast", "square", "cast", "relu", "cast"])

        # Asserting first cast configuration
        cast_1 = block.find_ops(op_type="cast")[0]
        self.assertEqual(cast_1.dtype.val, "fp16")
        self.assertEqual(len(cast_1.outputs), 1)
        self.assertEqual(len(cast_1.outputs[0].child_ops), 2)
        self.assertEqual(cast_1.outputs[0].child_ops[0].op_type, "square")
        self.assertEqual(cast_1.outputs[0].child_ops[1].op_type, "relu")

        # Asserting second cast configuration
        cast_2 = block.find_ops(op_type="cast")[1]
        self.assertEqual(cast_2.dtype.val, "fp32")
        self.assertEqual(len(cast_2.outputs), 1)
        self.assertEqual(len(cast_2.outputs[0].child_ops), 0)

        # Asserting third cast configuration
        cast_3 = block.find_ops(op_type="cast")[2]
        self.assertEqual(cast_3.dtype.val, "fp32")
        self.assertEqual(len(cast_3.outputs), 1)
        self.assertEqual(len(cast_3.outputs[0].child_ops), 0)

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20),
            },
        )

    def test_duplicate_output_vars(self):
        """
        Input graph:
                           |-> output_1
            input -> relu -|
                           |-> output_2

        Output graph:
                                                                   |-> output_1
            input -> cast(fp32->fp16) -> relu -> cast(fp16->fp32) -|
                                                                   |-> output_2
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2))])
        def prog(x):
            relu1 = mb.relu(x=x)
            return relu1, relu1

        _, _, block = apply_pass_and_basic_check(
            prog, quantization.FP16ComputePrecision(op_selector=lambda op: True)
        )
        self.assertEqual(get_op_types_in_program(prog), ["cast", "relu", "cast"])

        assert_model_is_valid(
            prog,
            {"x": (1, 2)},
            expected_output_shapes={block.outputs[0].name: (1, 2), block.outputs[1].name: (1, 2)},
            backend=("mlprogram", "fp16"),
        )

    @pytest.mark.parametrize(
        "opset_version, op_name",
        itertools.product(
            [None, ct.target.iOS17],
            ["inverse", "log", "rsqrt"],
        ),
    )
    def test_epsilon_mixed_precision(self, opset_version, op_name):
        """The IOS17+ elementwise unary ops with epsilon support mixed precision."""

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))], opset_version=opset_version)
        def prog(x):
            return getattr(mb, op_name)(x=x, epsilon=0.1)

        _, _, block = apply_pass_and_basic_check(prog, "common::add_fp16_cast")

        expected_ops = ["cast", "cast", op_name, "cast"]
        if opset_version is not None and opset_version >= ct.target.iOS17:
            # Allow mixed precision, so the epsilon is not cast to fp16.
            expected_ops = ["cast", op_name, "cast"]
        assert get_op_types_in_program(prog) == expected_ops

        assert_model_is_valid(
            prog,
            {"x": (2, 3)},
            expected_output_shapes={block.outputs[0].name: (2, 3)},
            backend=("mlprogram", "fp16"),
            minimum_deployment_target=opset_version,
        )


class TestTransformFunctionSignatures:
    @staticmethod
    def test_empty():
        """
        Case where the input var is also a block output.
        """
        # case 1
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            return x

        graph_pass = add_fp16_cast()
        block = prog.functions["main"]
        graph_pass.transform_function_signatures(block)
        apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == []
        assert block.inputs["x"].dtype == types.fp16
        assert len(block.outputs) == 1
        assert block.outputs[0].dtype == types.fp16
        assert block.outputs[0] is block.inputs["x"]

        # case 2
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            return x, mb.relu(x=x), x, x

        graph_pass = add_fp16_cast()
        block = prog.functions["main"]
        graph_pass.transform_function_signatures(block)

        assert block.inputs["x"].dtype == types.fp16
        assert len(block.outputs) == 4

        assert block.outputs[0].dtype == types.fp16
        assert block.outputs[2].dtype == types.fp16
        assert block.outputs[3].dtype == types.fp16

        assert block.outputs[1].dtype == types.fp32

        assert block.outputs[0] is block.inputs["x"]
        assert block.outputs[2] is block.inputs["x"]
        assert block.outputs[3] is block.inputs["x"]

        assert all([x.dtype == types.fp16 for x in block.output_types])

        assert get_op_types_in_program(prog) == ["cast", "relu"]
        cast_op = block.find_ops(op_type="cast")[0]
        assert cast_op.dtype.val == "fp32"

    @staticmethod
    def test_simple():
        """
        Input graph:

            input(fp32) -> relu -> output

        Output graph:

            input(fp16) -> cast(dtype="fp32") -> relu -> output,

            with function.output_types = [ct.TesorType(dtype=types.fp16)]

        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            return mb.relu(x=x)

        graph_pass = add_fp16_cast()
        block = prog.functions["main"]
        graph_pass.transform_function_signatures(block)

        assert block.inputs["x"].dtype == types.fp16

        assert get_op_types_in_program(prog) == ["cast", "relu"]
        cast_op = block.find_ops(op_type="cast")[0]
        assert cast_op.dtype.val == "fp32"

        assert len(block.outputs) == 1
        assert block.outputs[0].dtype == types.fp32

        assert len(block.output_types) == 1
        assert block.output_types[0].dtype == types.fp16

    @staticmethod
    def test_simple_2():
        """
        Input graph:

            input(fp32) -> identity -> cast(dtype="int32") -> output_1
                               |
                               .-> output_2

        Output graph:

            input(fp16) -> cast(dtype="fp32") -> identity -> cast(dtype="int32")  -> output_1
                                                      |
                                                      .-> output_2,

            with function.output_types = [ct.TesorType(dtype=types.int32), ct.TesorType(dtype=types.fp16)]

        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            x = mb.identity(x=x)
            return mb.cast(x=x, dtype="int32"), x

        graph_pass = add_fp16_cast()
        block = prog.functions["main"]
        graph_pass.transform_function_signatures(block)

        assert block.inputs["x"].dtype == types.fp16

        assert get_op_types_in_program(prog) == ["cast", "identity", "cast"]
        cast_ops = block.find_ops(op_type="cast")
        assert cast_ops[0].dtype.val == "fp32"
        assert cast_ops[1].dtype.val == "int32"

        assert len(block.outputs) == 2
        assert block.outputs[0].dtype == types.int32
        assert block.outputs[1].dtype == types.fp32

        assert len(block.output_types) == 2
        assert block.output_types[0].dtype == types.int32
        assert block.output_types[1].dtype == types.fp16


class TestInt32CastToInt16:
    @pytest.mark.parametrize(
        "x_dtype, dynamic, has_neg, opset_version",
        itertools.product(
            [np.int32, np.float32],
            [True, False],
            [True, False],
            [ct.target.iOS15, ct.target.iOS16, ct.target.iOS17],
        ),
    )
    def test_gather_int16_indices(self, x_dtype, dynamic, has_neg, opset_version):
        @mb.program(opset_version=opset_version)
        def prog_static():
            params = np.array([[1, 2, 3], [4, 5, 6]], dtype=x_dtype)
            indices = np.array([-2, 0] if has_neg else [1, 0], dtype=np.int32)
            return mb.gather(x=params, indices=indices, axis=-1)

        @mb.program(
            [
                mb.TensorSpec(shape=(2, 3), dtype=types.numpy_type_to_builtin_type(x_dtype)),
                mb.TensorSpec(shape=(2,), dtype=types.int32),
            ],
            opset_version=opset_version,
        )
        def prog_dynamic(x, indices):
            return mb.gather(x=x, indices=indices, axis=0)

        prog = prog_dynamic if dynamic else prog_static
        assert get_op_types_in_program(prog) == ["gather"]

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::add_int16_cast")

        if opset_version <= ct.target.iOS16:
            # iOS15 gather op's ``indices`` doesn't support int16, so this pass doesn't have effect.
            # iOS16 cast op doesn't support int16, so this pass doesn't have effect.
            assert get_op_types_in_program(prog) == get_op_types_in_program(prev_prog)
        else:
            # When input ``x`` is float32, the output is also float32, so no cast for output.
            # When input ``x`` is int32 and cast to int16, the output will also be int16, so there
            # is another cast op to cast it back to int32.
            expected_ops = ["cast", "gather"]
            if x_dtype == np.int32:
                expected_ops = ["cast", "cast", "gather", "cast"]
            assert get_op_types_in_program(prog) == expected_ops
            indices_cast_op_idx = 1 if x_dtype == np.int32 else 0
            cast_op = block.find_ops(op_type="cast")[indices_cast_op_idx]
            assert cast_op.dtype.val == "int16" if has_neg else "uint16"
            assert len(cast_op.outputs) == 1
            assert len(cast_op.outputs[0].child_ops) == 1
            assert cast_op.outputs[0].child_ops[0].op_type == "gather"
            assert cast_op.outputs[0] == block.find_ops(op_type="gather")[0].indices

        if not dynamic:
            np.testing.assert_allclose(
                np.array([[2, 1], [5, 4]], dtype=np.float32),
                prog.functions["main"].find_ops(op_type="gather")[0].outputs[0].val,
                atol=1e-04,
                rtol=1e-05,
            )

    def test_gather_int16_scalar_indices(self):
        @mb.program(input_specs=[], opset_version=ct.target.iOS17)
        def prog_static():
            params = np.array([1, 2, 3, 4], dtype=np.int32)
            res = mb.gather(x=params, indices=0, axis=0, batch_dims=0, validate_indices=False)
            return res

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(4,), dtype=types.int32)],
            opset_version=ct.target.iOS17,
        )
        def prog_dynamic(x):
            return mb.gather(x=x, indices=0, axis=0)

        for prog in (prog_static, prog_dynamic):
            assert get_op_types_in_program(prog) == ["gather"]
            prev_prog, _, block = apply_pass_and_basic_check(prog, "common::add_int16_cast")
            expected_ops = ["cast", "cast", "gather", "cast"]
            assert get_op_types_in_program(prog) == expected_ops

    @pytest.mark.parametrize(
        "x_dtype, dynamic, has_neg, opset_version",
        itertools.product(
            [np.int32, np.float32],
            [True, False],
            [True, False],
            [ct.target.iOS15, ct.target.iOS16, ct.target.iOS17],
        ),
    )
    def test_gather_along_axis_int16_indices(self, x_dtype, dynamic, has_neg, opset_version):
        @mb.program(opset_version=opset_version)
        def prog_static():
            params = np.array([[1, 2, 3], [4, 5, 6]], dtype=x_dtype)
            indices = np.array(
                [[-2, 0, -2], [-2, -2, 0]] if has_neg else [[1, 0, 1], [1, 1, 0]], dtype=np.int32
            )
            return mb.gather_along_axis(x=params, indices=indices, axis=-1)

        @mb.program(
            [
                mb.TensorSpec(shape=(2, 3), dtype=types.numpy_type_to_builtin_type(x_dtype)),
                mb.TensorSpec(shape=(2, 3), dtype=types.int32),
            ],
            opset_version=opset_version,
        )
        def prog_dynamic(x, indices):
            return mb.gather_along_axis(x=x, indices=indices, axis=0)

        prog = prog_dynamic if dynamic else prog_static
        assert get_op_types_in_program(prog) == ["gather_along_axis"]

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::add_int16_cast")

        if opset_version <= ct.target.iOS16:
            # iOS15 gather op's ``indices`` doesn't support int16, so this pass doesn't have effect.
            # iOS16 cast op doesn't support int16, so this pass doesn't have effect.
            assert get_op_types_in_program(prog) == get_op_types_in_program(prev_prog)
        else:
            # When input ``x`` is float32, the output is also float32, so no cast for output.
            # When input ``x`` is int32 and cast to int16, the output will also be int16, so there
            # is another cast op to cast it back to int32.
            expected_ops = ["cast", "gather_along_axis"]
            if x_dtype == np.int32:
                expected_ops = ["cast", "cast", "gather_along_axis", "cast"]
            assert get_op_types_in_program(prog) == expected_ops
            indices_cast_op_idx = 1 if x_dtype == np.int32 else 0
            cast_op = block.find_ops(op_type="cast")[indices_cast_op_idx]
            assert cast_op.dtype.val == "int16" if has_neg else "uint16"
            assert len(cast_op.outputs) == 1
            assert len(cast_op.outputs[0].child_ops) == 1
            assert cast_op.outputs[0].child_ops[0].op_type == "gather_along_axis"
            assert cast_op.outputs[0] == block.find_ops(op_type="gather_along_axis")[0].indices

        if not dynamic:
            np.testing.assert_allclose(
                np.array([[2, 1, 2], [5, 5, 4]], dtype=np.float32),
                prog.functions["main"].find_ops(op_type="gather_along_axis")[0].outputs[0].val,
                atol=1e-04,
                rtol=1e-05,
            )

    @pytest.mark.parametrize("overflow", [True, False])
    def test_gather_dynamic_overflow_int16(self, overflow):
        """Dynamic input indices should also be cast if x dim size doesn't overflow int16 range."""

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(32769 if overflow else 2, 3)),
                mb.TensorSpec(shape=(2,), dtype=types.int32),
            ],
            opset_version=ct.target.iOS17,
        )
        def prog(x, indices):
            return mb.gather(x=x, indices=indices, axis=0)

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::add_int16_cast")
        if overflow:
            assert get_op_types_in_program(prog) == get_op_types_in_program(prev_prog)
        else:
            assert get_op_types_in_program(prog) == ["cast", "gather"]
            cast_op = block.find_ops(op_type="cast")[0]
            assert cast_op.dtype.val == "int16"
            assert cast_op.outputs[0] == block.find_ops(op_type="gather")[0].indices

    @pytest.mark.parametrize("overflow_uint16", [True, False])
    def test_gather_static_overflow_int16(self, overflow_uint16):
        """Indices cannot be represented by int16 range, but might be represented by uint16."""
        max_index = 65536 if overflow_uint16 else 32768

        @mb.program(opset_version=ct.target.iOS17)
        def prog():
            params = np.array([[1, 2]] * (max_index + 1), dtype=np.float32)
            indices = np.array([max_index, 0], dtype=np.int32)
            return mb.gather(x=params, indices=indices, axis=0)

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::add_int16_cast")
        if overflow_uint16:
            assert get_op_types_in_program(prog) == get_op_types_in_program(prev_prog)
        else:
            assert get_op_types_in_program(prog) == ["cast", "gather"]
            cast_op = block.find_ops(op_type="cast")[0]
            assert cast_op.dtype.val == "uint16"
            assert cast_op.outputs[0] == block.find_ops(op_type="gather")[0].indices

    @pytest.mark.parametrize(
        "dtype, opset_version",
        itertools.product(
            [types.int32, types.fp32],
            [ct.target.iOS15, ct.target.iOS16, ct.target.iOS17, ct.target.iOS18],
        ),
    )
    def test_squeeze(self, dtype, opset_version):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 1), dtype=dtype)],
            opset_version=opset_version,
        )
        def prog(x):
            return mb.squeeze(x=x)

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::add_int16_cast")

        if opset_version < ct.target.iOS17:
            # Prior to iOS 17, `squeeze` does not support int16, so this pass has no effect
            assert get_op_types_in_program(prog) == get_op_types_in_program(prev_prog)
        else:
            if dtype == types.int32:
                # When `x` is int32, it will be cast to int16 then feed into `squeeze`,
                # then `squeeze(x)` will be cast back to int32 for output
                assert get_op_types_in_program(prog) == ["cast", "squeeze", "cast"]
                cast_int16, cast_int32 = block.find_ops(op_type="cast")
                assert cast_int16.dtype.val == "int16"
                assert cast_int32.dtype.val == "int32"
                assert cast_int16.outputs[0].child_ops[0].op_type == "squeeze"
            else:
                # When `x` is float, this int pass has no effect
                assert get_op_types_in_program(prog) == ["squeeze"]

    @patch(
        "coremltools.converters.mil.mil.passes.defs.quantization.add_int16_cast._PREFER_INT16_OPS",
        set(),
    )
    def test_int16_no_effect(self):
        """After patching the pass, no op should be cast to int16"""

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(2, 3)), mb.TensorSpec(shape=(2,), dtype=types.int32)],
            opset_version=ct.target.iOS17,
        )
        def prog(x, indices):
            return mb.gather(x=x, indices=indices, axis=0)

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::add_int16_cast")
        assert get_op_types_in_program(prog) == get_op_types_in_program(prev_prog)

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        "compute_precision, num_embeddings, minimum_deployment_target, symbolic",
        itertools.product(
            [ct.precision.FLOAT16, ct.precision.FLOAT32],
            [10, 32769],
            [ct.target.iOS15, ct.target.iOS16, ct.target.iOS17],
            [True, False],
        ),
    )
    def test_int16_embedding_e2e(
        self, compute_precision, num_embeddings, minimum_deployment_target, symbolic
    ):
        """End-to-end conversion from a torch embedding model."""

        class EmbeddingModel(nn.Module):
            def __init__(self):
                super(EmbeddingModel, self).__init__()
                self.embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=2)

            def forward(self, x):
                return self.embedding(x)

        input_data = np.random.randint(low=0, high=num_embeddings, size=(3, 5))
        input_data = torch.from_numpy(input_data)
        model = EmbeddingModel()
        model.eval()
        traced_model = torch.jit.trace(model, input_data)
        input_shape = (ct.RangeDim(1, 32), ct.RangeDim(1, 32)) if symbolic else input_data.shape
        converted_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=input_shape, name="input", dtype=np.int32)],
            convert_to="mlprogram",
            compute_precision=compute_precision,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=minimum_deployment_target,
        )
        prog = converted_model._mil_program

        # The embedding layer is lowered to `gather` op.
        expected_ops = ["gather"]
        if (
            compute_precision == ct.precision.FLOAT16
            and minimum_deployment_target < ct.target.iOS16
        ):
            # Cast from fp16 to fp32 because fp16 is not supported in I/O before iOS16.
            expected_ops.append("cast")
        if (
            minimum_deployment_target >= ct.target.iOS17
            and compute_precision == ct.precision.FLOAT16
            and num_embeddings <= np.iinfo(np.int16).max
        ):
            # The int16 cast only happens for iOS17+ with fp16 precision and there is no overflow.
            expected_ops.insert(0, "cast")
            cast_op = prog["main"].find_ops(op_type="cast")[0]
            assert cast_op.dtype.val == "int16"
            assert cast_op.outputs[0] == prog["main"].find_ops(op_type="gather")[0].indices
        assert get_op_types_in_program(prog) == expected_ops
