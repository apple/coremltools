#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
from typing import Tuple

import numpy as np
import parameterized
import pytest

import coremltools as ct
import coremltools.converters.mil.mil.types as types
from coremltools._deps import _IS_MACOS
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.defs import quantization
from coremltools.converters.mil.mil.types import numpy_type_to_builtin_type
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    get_op_types_in_program,
)

np.random.seed(1818)


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

            return dequantize_2_0, dequantize_2_1, 

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

        Nothing changes when dequantize1 has multiple children
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
        # nothing gets eliminated
        assert get_op_types_in_program(prog) == [
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

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::dequantize_to_constexpr"
        )
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
        input_dict = {"x": np.random.rand(10, 20)}

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
