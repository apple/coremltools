#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import itertools
import unittest

import numpy as np
import pytest
from mock import patch

import coremltools as ct
from coremltools.converters.mil import mil
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Symbol, get_new_symbol, types
from coremltools.converters.mil.mil.passes.defs.cleanup import topological_reorder
from coremltools.converters.mil.mil.passes.defs.cleanup.remove_redundant_ops import (
    remove_redundant_ops,
)
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    assert_op_count_match,
    assert_same_output_names,
    get_op_names_in_program,
    get_op_types_in_program,
)

from .test_passes import _VALIDATE_MODEL, CONSTEXPR_FUNCS, CONSTEXPR_OPS


class TestConstDeduplication:
    def test_const_deduplication(self):
        BATCH_DIM = 5
        SEQUENCE_LENGTH = 4
        ENCODING_DIM = 256
        EMBEDDING_DIM = 128
        weight = np.random.rand(EMBEDDING_DIM, ENCODING_DIM)
        bias = np.random.rand(EMBEDDING_DIM)

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(BATCH_DIM, SEQUENCE_LENGTH, ENCODING_DIM)),
                mb.TensorSpec(shape=(BATCH_DIM, SEQUENCE_LENGTH, ENCODING_DIM)),
            ]
        )
        def prog(q, k):
            q_e = mb.linear(x=q, weight=weight, bias=bias)
            k_e = mb.linear(x=k, weight=weight, bias=bias)
            attention = mb.matmul(x=q_e, y=k_e, transpose_y=True)
            return attention

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::const_deduplication")
        assert_op_count_match(prev_prog, expect=6, op="const")
        assert_op_count_match(prog, expect=4, op="const")

    @pytest.mark.parametrize(
        "constexpr_op",
        CONSTEXPR_OPS,
    )
    def test_constexpr_deduplication(self, constexpr_op):
        BATCH_DIM = 5
        SEQUENCE_LENGTH = 4
        ENCODING_DIM = 256
        EMBEDDING_DIM = 128

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(BATCH_DIM, SEQUENCE_LENGTH, ENCODING_DIM)),
                mb.TensorSpec(shape=(BATCH_DIM, SEQUENCE_LENGTH, ENCODING_DIM)),
            ]
        )
        def prog(q, k):
            weight_q = CONSTEXPR_FUNCS[constexpr_op]((EMBEDDING_DIM, ENCODING_DIM), seed=19)
            weight_k = CONSTEXPR_FUNCS[constexpr_op]((EMBEDDING_DIM, ENCODING_DIM), seed=19)
            bias_q = CONSTEXPR_FUNCS[constexpr_op]((EMBEDDING_DIM,), seed=29)
            bias_k = CONSTEXPR_FUNCS[constexpr_op]((EMBEDDING_DIM,), seed=29)
            q_e = mb.linear(x=q, weight=weight_q, bias=bias_q)
            k_e = mb.linear(x=k, weight=weight_k, bias=bias_k)
            attention = mb.matmul(x=q_e, y=k_e, transpose_y=True)
            return attention

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::const_deduplication")
        assert_op_count_match(prev_prog, expect=4, op=constexpr_op)
        assert_op_count_match(prog, expect=2, op=constexpr_op)

    def test_const_deduplication_as_outputs(self):
        """
        If the duplicated constants are block outputs, we should not remove them.
        """
        # case 1:
        # const_2 can be eliminated since it is not block output
        const = np.random.rand(40, 20, 30)

        @mb.program(
            input_specs=[
                mb.TensorSpec(
                    shape=(
                        40,
                        20,
                        30,
                    )
                )
            ]
        )
        def prog(x):
            const_1 = mb.const(val=const, name="const_1")
            const_2 = mb.const(val=const, name="const_2")
            x = mb.relu(x=x)
            x = mb.add(x=x, y=const_2)
            return x, const_1

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::const_deduplication")
        assert_op_count_match(prev_prog, expect=2, op="const")
        assert_op_count_match(prog, expect=1, op="const")
        assert prog.functions["main"].outputs[1].name == "const_1"

        # case 2:
        # const_2 can not be eliminated since it is a block output
        const = np.random.rand(40, 20, 30)

        @mb.program(
            input_specs=[
                mb.TensorSpec(
                    shape=(
                        40,
                        20,
                        30,
                    )
                )
            ]
        )
        def prog(x):
            const_1 = mb.const(val=const, name="const_1")
            const_2 = mb.const(val=const, name="const_2")
            x = mb.relu(x=x)
            x = mb.add(x=x, y=const_2)
            return x, const_1, const_2

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::const_deduplication")
        assert_op_count_match(prev_prog, expect=2, op="const")
        assert_op_count_match(prog, expect=2, op="const")
        assert prog.functions["main"].outputs[1].name == "const_1"
        assert prog.functions["main"].outputs[2].name == "const_2"

    @pytest.mark.skip("rdar://109374995 consts are not shared across blocks")
    def test_const_deduplication_multiple_blocks(self):
        weight = np.random.rand(5, 3, 2, 2)

        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 3, 8, 8))])
        def prog(x):
            def _true_fn():
                return mb.conv(x=x, weight=weight, pad_type="valid")

            def _false_fn():
                y = mb.mul(x=x, y=2.0)
                return mb.conv(x=y, weight=weight, pad_type="valid")

            x_gt_0_tensor = mb.greater(x=x, y=0.0)
            x_gt_0 = mb.slice_by_index(x=x_gt_0_tensor, begin=(0, 0, 0, 0), end=(1, 1, 1, 1))
            return mb.cond(pred=x_gt_0, _true_fn=_true_fn, _false_fn=_false_fn)

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::const_deduplication")
        assert_op_count_match(prev_prog, expect=8, op="const")
        assert_op_count_match(prog, expect=6, op="const")


class TestConstElimination:
    def test_const_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            a = np.random.rand(2, 4).astype(np.float32)
            double_a = mb.add(x=a, y=a)
            return mb.add(x=x, y=double_a)

        assert_op_count_match(prog, expect=2, op="const")
        prev_prog = copy.deepcopy(prog)
        PASS_REGISTRY["common::const_elimination"](prog)
        assert_same_output_names(prev_prog, prog)
        assert_op_count_match(prog, expect=3, op="const")

        if _VALIDATE_MODEL:
            assert_model_is_valid(prog, {"x": (2, 4)})

    def test_const_elimination_nonreplaceable(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            a = np.random.rand(2, 4).astype(np.float16)
            constexpr_a = mb.constexpr_cast(source_val=a, output_dtype="fp32")
            double_a = mb.add(x=constexpr_a, y=a.astype(np.float32))
            return mb.add(x=x, y=double_a)

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::const_elimination")
        assert get_op_types_in_program(prev_prog) == ["constexpr_cast", "add", "add"]
        # Not fold into const because the upstream constexpr_cast op is non-replaceable.
        assert get_op_types_in_program(prog) == ["constexpr_cast", "add", "add"]

    def test_force_const_eliminate_nonreplaceable_ops(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3,), dtype=types.int32)])
        def prog(x):
            a = np.random.rand(2, 3, 5).astype(np.float16)
            constexpr_a = mb.constexpr_cast(source_val=a, output_dtype="fp32")
            double_a = mb.add(x=constexpr_a, y=a.astype(np.float32))
            a_shape = mb.shape(x=double_a)
            return mb.add(x=x, y=a_shape)

        assert get_op_types_in_program(prog) == ["constexpr_cast", "add", "shape", "add"]

        apply_pass_and_basic_check(prog, "common::const_elimination")
        # still fold shape into const regardless the non-replaceable upstream
        # constexpr_cast op, since it only provides a shape
        assert get_op_types_in_program(prog) == ["constexpr_cast", "add", "add"]

        apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        # constexpr_cast(a) and add(a, a) no longer contributes to output,
        # so they should get dead code eliminated
        assert get_op_types_in_program(prog) == ["add"]

    def test_force_const_eliminate_nonreplaceable_ops_case_2(self):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1,), dtype=types.int32),
                mb.TensorSpec(shape=(2,), dtype=types.int32),
            ],
            opset_version=ct.target.iOS17,
        )
        def prog(x, y):
            a = np.random.rand(2, 3, 5).astype(np.float16)
            constexpr_a = mb.constexpr_cast(source_val=a, output_dtype="fp32")

            reshape_shape = mb.concat(values=[y, [5]], axis=0)
            reshape = mb.reshape(x=constexpr_a, shape=reshape_shape)
            a_shape = mb.shape(x=reshape)
            a_shape_int16 = mb.cast(x=a_shape, dtype="int16")

            # Even though the gather ops has constexpr_cast op as upstream,
            # it can still be removed by const elimination.
            gather = mb.gather(
                x=a_shape,
                indices=[
                    2,
                ],
                axis=0,
            )
            gather_int32 = mb.cast(x=gather, dtype="int32")
            return mb.add(x=x, y=gather)

        assert get_op_types_in_program(prog) == [
            "constexpr_cast",
            "concat",
            "reshape",
            "shape",
            "cast",
            "gather",
            "cast",
            "add",
        ]

        apply_pass_and_basic_check(prog, "common::const_elimination")
        # still const-folding gather into const regardless the non-replaceable upstream
        # constexpr_cast op, since it only provides the meta data (shape)
        assert get_op_types_in_program(prog) == [
            "constexpr_cast",
            "concat",
            "reshape",
            "shape",
            "cast",
            "add",
        ]

        apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(prog) == ["add"]

    @patch(
        "coremltools.converters.mil.mil.passes.defs.cleanup.const_elimination._skip_const_by_size",
        1000,
    )
    def test_const_elimination_larger_than_threshold(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            # Construct a 10 x 10 matrix (100 elements) which is smaller than the threshold (1000).
            tmp = mb.range_1d(start=0, end=10, step=1)
            tmp_x = mb.reshape(x=tmp, shape=[-1, 1])
            tmp_y = mb.reshape(x=tmp, shape=[1, -1])
            return mb.matmul(x=tmp_x, y=tmp_y)

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog_large_const_size(x):
            # Construct a 100 x 100 matrix (10000 elements) which is larger than the threshold (1000).
            tmp = mb.range_1d(start=0, end=100, step=1)
            tmp_x = mb.reshape(x=tmp, shape=[-1, 1])
            tmp_y = mb.reshape(x=tmp, shape=[1, -1])
            return mb.matmul(x=tmp_x, y=tmp_y)

        prev_prog, _, _ = apply_pass_and_basic_check(prog, "common::const_elimination")
        assert get_op_types_in_program(prev_prog) == [
            "range_1d",
            "reshape",
            "reshape",
            "matmul",
        ]
        # All ops (range_1d, reshape, matmul) constructing that 10x10 matrix is folded into a const.
        assert get_op_types_in_program(prog) == []

        prev_prog_large_const_size, _, _ = apply_pass_and_basic_check(
            prog_large_const_size, "common::const_elimination"
        )
        assert get_op_types_in_program(prev_prog_large_const_size) == [
            "range_1d",
            "reshape",
            "reshape",
            "matmul",
        ]
        # The matmul op constructing the large matrix is kept due to size larger than threshold.
        assert get_op_types_in_program(prog_large_const_size) == ["matmul"]


class TestDeadCodeElimination:
    def test_dead_code_elimination(self):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(2, 4)),
                mb.TensorSpec(shape=(2, 4)),
            ]
        )
        def program0(x, y):
            # following three unused op should be eliminated
            a = mb.const(val=np.zeros(shape=(1,)))
            b = mb.const(val=np.zeros(shape=(1,)))
            _ = mb.add(x=a, y=b)
            return mb.add(x=x, y=y)

        assert_op_count_match(program0, expect=4)
        prev_prog = copy.deepcopy(program0)
        PASS_REGISTRY["common::dead_code_elimination"](program0)
        assert_same_output_names(prev_prog, program0)
        assert_op_count_match(program0, expect=1)

        if _VALIDATE_MODEL:
            assert_model_is_valid(program0, {"x": (2, 4), "y": (2, 4)})

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def program1(x):
            weights_val = np.random.rand(4, 2).T.astype(np.float32)
            weights = mb.const(val=weights_val)
            bias_val = np.random.rand(2).astype(np.float32)
            bias = mb.const(val=bias_val)

            # unused op and its inputs should be eliminated
            weights_for_matmul = mb.transpose(x=weights, perm=[1, 0])
            mb.matmul(x=x, y=weights_for_matmul)

            return mb.linear(x=x, weight=weights, bias=bias)

        assert_op_count_match(program1, expect=8)
        prev_prog = copy.deepcopy(program1)
        PASS_REGISTRY["common::dead_code_elimination"](program1)
        assert_same_output_names(prev_prog, program1)
        assert_op_count_match(program1, expect=3)

        if _VALIDATE_MODEL:
            assert_model_is_valid(program1, {"x": (2, 4)})


class TestDedupOpAndVarNames(unittest.TestCase):
    def test_unchanged(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            x = mb.reshape(x=x, shape=(1, 8), name="reshape")
            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::dedup_op_and_var_names")

        self.assertEqual(get_op_types_in_program(prev_prog), ["reshape"])
        self.assertEqual(get_op_names_in_program(prev_prog), ["reshape"])

        self.assertEqual(get_op_types_in_program(prog), ["reshape"])
        self.assertEqual(get_op_names_in_program(prog), ["reshape"])

        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (1, 8)},
        )

    def test_op_name_duplicated_once(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16", name="castop")
            x = mb.cast(x=x, dtype="fp32", name="castop")
            x = mb.square(x=x, name="square_last")
            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::dedup_op_and_var_names")

        self.assertEqual(get_op_types_in_program(prev_prog), ["cast", "cast", "square"])
        self.assertEqual(get_op_names_in_program(prev_prog), ["castop", "castop", "square_last"])

        self.assertEqual(get_op_types_in_program(prog), ["cast", "cast", "square"])
        self.assertEqual(get_op_names_in_program(prog), ["castop", "castop_1", "square_last"])

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20)},
        )

    def test_op_name_duplicated_many(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16", name="castop")
            x = mb.cast(x=x, dtype="fp16", name="castop")
            x = mb.cast(x=x, dtype="int32", name="castop_2")
            x = mb.cast(x=x, dtype="fp16", name="castop")
            x = mb.cast(x=x, dtype="fp32", name="castop_2")
            x = mb.square(x=x, name="square")
            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::dedup_op_and_var_names")

        self.assertEqual(
            get_op_types_in_program(prev_prog), ["cast", "cast", "cast", "cast", "cast", "square"]
        )
        self.assertEqual(
            get_op_names_in_program(prev_prog),
            ["castop", "castop", "castop_2", "castop", "castop_2", "square"],
        )

        self.assertEqual(
            get_op_types_in_program(prog), ["cast", "cast", "cast", "cast", "cast", "square"]
        )
        self.assertEqual(
            get_op_names_in_program(prog),
            ["castop", "castop_1", "castop_2", "castop_3", "castop_2_1", "square"],
        )

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (10, 20)},
        )

    def test_input_name_shadow(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            # op name "x" results in output var name "x", which shadows prog
            # input var name "x"
            x = mb.transpose(x=x, perm=[1, 0], name="x")
            x = mb.relu(x=x, name="relu")
            return x

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::dedup_op_and_var_names")
        self.assertEqual(get_op_types_in_program(prev_prog), ["transpose", "relu"])
        self.assertEqual(get_op_names_in_program(prev_prog), ["x", "relu"])

        self.assertEqual(get_op_types_in_program(prog), ["transpose", "relu"])
        self.assertEqual(get_op_names_in_program(prog), ["x", "relu"])

        op = prog["main"].find_ops(op_type="transpose")[0]
        self.assertEqual("x_1", op.outputs[0].name)

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (20, 10)},
        )

    def test_nested_block(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1,))])
        def prog(x):
            def true_fn():
                # returns var with name x shadows input 'x'
                return mb.add(x=x, y=1.0, name="x")

            def false_fn():
                # two ops with name "x"
                return mb.add(x=x, y=-1.0, name="x")

            pred = mb.equal(x=mb.squeeze(x=x), y=1.0)
            return mb.cond(pred=pred, _true_fn=true_fn, _false_fn=false_fn)

        cond_op = prog.functions["main"].operations[-1]
        assert cond_op.blocks[0].outputs[0].name == "x"
        assert cond_op.blocks[1].outputs[0].name == "x"
        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::dedup_op_and_var_names")
        cond_op = prog.functions["main"].operations[-1]
        assert cond_op.blocks[0].outputs[0].name == "x_1"
        assert cond_op.blocks[1].outputs[0].name == "x_2"

        assert_model_is_valid(
            prog,
            {"x": (1,)},
            expected_output_shapes={block.outputs[0].name: (1,)},
        )


class TestExpandDynamicLinear:
    def test_keep_static_weight_static_bias(self):
        X_SHAPE = (2, 5)
        WEIGHT_SHAPE = (3, X_SHAPE[-1])

        bias_shape = (WEIGHT_SHAPE[0],)
        output_shape = (X_SHAPE[0], WEIGHT_SHAPE[0])

        quantized_weight = np.random.randint(-127, 128, WEIGHT_SHAPE, np.int8)
        quantized_bias = np.random.randint(-127, 128, bias_shape, np.int8)

        @mb.program(
            input_specs=[mb.TensorSpec(shape=X_SHAPE)],
            opset_version=ct.target.iOS16,
        )
        def prog(x):
            weight = mb.constexpr_affine_dequantize(
                quantized_data=quantized_weight,
                scale=1.2,
                zero_point=np.int8(3),
                axis=0,
            )
            bias = mb.constexpr_affine_dequantize(
                quantized_data=quantized_bias,
                scale=4.5,
                zero_point=np.int8(6),
                axis=0,
            )
            return mb.linear(x=x, weight=weight, bias=bias)

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::expand_dynamic_linear")
        assert get_op_types_in_program(prev_prog) == [
            "constexpr_affine_dequantize",
            "constexpr_affine_dequantize",
            "linear",
        ]
        assert get_op_types_in_program(prog) == get_op_types_in_program(prev_prog)
        assert_model_is_valid(
            prog,
            {"x": X_SHAPE},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=("mlprogram", "fp16"),
            minimum_deployment_target=ct.target.iOS16,
        )

    def test_expand_static_weight_dynamic_bias(self):
        X_SHAPE = (2, 5)
        WEIGHT_SHAPE = (3, X_SHAPE[-1])

        bias_shape = (WEIGHT_SHAPE[0],)
        output_shape = (X_SHAPE[0], WEIGHT_SHAPE[0])

        weight = np.random.rand(*WEIGHT_SHAPE)
        quantized_bias = np.random.randint(-127, 128, bias_shape, np.int8)

        @mb.program(
            input_specs=[mb.TensorSpec(shape=X_SHAPE)],
            opset_version=ct.target.iOS16,
        )
        def prog(x):
            bias = mb.constexpr_affine_dequantize(
                quantized_data=quantized_bias,
                scale=1.2,
                zero_point=np.int8(3),
                axis=0,
            )
            screwed_bias = mb.exp(x=bias)
            return mb.linear(x=x, weight=weight, bias=screwed_bias)

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::expand_dynamic_linear")
        assert get_op_types_in_program(prev_prog) == [
            "constexpr_affine_dequantize",
            "exp",
            "linear",
        ]
        assert get_op_types_in_program(prog) == [
            "constexpr_affine_dequantize",
            "exp",
            "linear",
            "add",
        ]
        assert_model_is_valid(
            prog,
            {"x": X_SHAPE},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=("mlprogram", "fp16"),
            minimum_deployment_target=ct.target.iOS16,
        )

    def test_expand_dynamic_weight_static_zero_bias(self):
        X_SHAPE = (2, 5)
        WEIGHT_SHAPE = (3, X_SHAPE[-1])

        output_shape = (X_SHAPE[0], WEIGHT_SHAPE[0])

        quantized_weight = np.random.randint(-127, 128, WEIGHT_SHAPE, np.int8)

        @mb.program(
            input_specs=[mb.TensorSpec(shape=X_SHAPE)],
            opset_version=ct.target.iOS16,
        )
        def prog(x):
            weight = mb.constexpr_affine_dequantize(
                quantized_data=quantized_weight,
                scale=1.2,
                zero_point=np.int8(3),
                axis=0,
            )
            screwed_weight = mb.exp(x=weight)
            return mb.linear(x=x, weight=screwed_weight)

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::expand_dynamic_linear")
        assert get_op_types_in_program(prev_prog) == [
            "constexpr_affine_dequantize",
            "exp",
            "linear",
        ]
        assert get_op_types_in_program(prog) == [
            "constexpr_affine_dequantize",
            "exp",
            "matmul",
        ]
        assert_model_is_valid(
            prog,
            {"x": X_SHAPE},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=("mlprogram", "fp16"),
            minimum_deployment_target=ct.target.iOS16,
        )

    def test_expand_dynamic_weight_static_compressed_zero_bias(self):
        X_SHAPE = (2, 5)
        WEIGHT_SHAPE = (3, X_SHAPE[-1])

        bias_shape = (WEIGHT_SHAPE[0],)
        output_shape = (X_SHAPE[0], WEIGHT_SHAPE[0])

        quantized_weight = np.random.randint(-127, 128, WEIGHT_SHAPE, np.int8)
        quantized_bias = np.random.randint(-127, 128, bias_shape, np.int8)

        @mb.program(
            input_specs=[mb.TensorSpec(shape=X_SHAPE)],
            opset_version=ct.target.iOS16,
        )
        def prog(x):
            weight = mb.constexpr_affine_dequantize(
                quantized_data=quantized_weight,
                scale=1.2,
                zero_point=np.int8(3),
                axis=0,
            )
            bias = mb.constexpr_affine_dequantize(
                quantized_data=quantized_bias,
                scale=np.random.rand(*bias_shape),
                zero_point=quantized_bias,
                axis=0,
            )
            screwed_weight = mb.exp(x=weight)
            return mb.linear(x=x, weight=screwed_weight, bias=bias)

        original_prog, _, _ = apply_pass_and_basic_check(prog, "common::expand_dynamic_linear")
        expanded_prog, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        assert get_op_types_in_program(original_prog) == [
            "constexpr_affine_dequantize",
            "constexpr_affine_dequantize",
            "exp",
            "linear",
        ]
        assert get_op_types_in_program(expanded_prog) == [
            "constexpr_affine_dequantize",
            "constexpr_affine_dequantize",
            "exp",
            "matmul",
        ]
        assert get_op_types_in_program(prog) == [
            "constexpr_affine_dequantize",
            "exp",
            "matmul",
        ]

        assert_model_is_valid(
            prog,
            {"x": X_SHAPE},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=("mlprogram", "fp16"),
            minimum_deployment_target=ct.target.iOS16,
        )

    def test_expand_dynamic_weight_static_nonzero_bias(self):
        X_SHAPE = (2, 5)
        WEIGHT_SHAPE = (3, X_SHAPE[-1])

        bias_shape = (WEIGHT_SHAPE[0],)
        output_shape = (X_SHAPE[0], WEIGHT_SHAPE[0])

        quantized_weight = np.random.randint(-127, 128, WEIGHT_SHAPE, np.int8)
        bias = np.random.rand(*bias_shape)

        @mb.program(
            input_specs=[mb.TensorSpec(shape=X_SHAPE)],
            opset_version=ct.target.iOS16,
        )
        def prog(x):
            weight = mb.constexpr_affine_dequantize(
                quantized_data=quantized_weight,
                scale=1.2,
                zero_point=np.int8(3),
                axis=0,
            )
            screwed_weight = mb.exp(x=weight)
            return mb.linear(x=x, weight=screwed_weight, bias=bias)

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::expand_dynamic_linear")
        assert get_op_types_in_program(prev_prog) == [
            "constexpr_affine_dequantize",
            "exp",
            "linear",
        ]
        assert get_op_types_in_program(prog) == [
            "constexpr_affine_dequantize",
            "exp",
            "matmul",
            "add",
        ]
        assert_model_is_valid(
            prog,
            {"x": X_SHAPE},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=("mlprogram", "fp16"),
            minimum_deployment_target=ct.target.iOS16,
        )

    def test_expand_dynamic_weight_dynamic_bias(self):
        X_SHAPE = (2, 5)
        WEIGHT_SHAPE = (3, X_SHAPE[-1])

        bias_shape = (WEIGHT_SHAPE[0],)
        output_shape = (X_SHAPE[0], WEIGHT_SHAPE[0])

        quantized_weight = np.random.randint(-127, 128, WEIGHT_SHAPE, np.int8)
        quantized_bias = np.random.randint(-127, 128, bias_shape, np.int8)

        @mb.program(
            input_specs=[mb.TensorSpec(shape=X_SHAPE)],
            opset_version=ct.target.iOS16,
        )
        def prog(x):
            weight = mb.constexpr_affine_dequantize(
                quantized_data=quantized_weight,
                scale=1.2,
                zero_point=np.int8(3),
                axis=0,
            )
            bias = mb.constexpr_affine_dequantize(
                quantized_data=quantized_bias,
                scale=1.2,
                zero_point=np.int8(3),
                axis=0,
            )
            screwed_weight = mb.exp(x=weight)
            screwed_bias = mb.exp(x=bias)
            return mb.linear(x=x, weight=screwed_weight, bias=screwed_bias)

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::expand_dynamic_linear")
        assert get_op_types_in_program(prev_prog) == [
            "constexpr_affine_dequantize",
            "constexpr_affine_dequantize",
            "exp",
            "exp",
            "linear",
        ]
        assert get_op_types_in_program(prog) == [
            "constexpr_affine_dequantize",
            "constexpr_affine_dequantize",
            "exp",
            "exp",
            "matmul",
            "add",
        ]
        assert_model_is_valid(
            prog,
            {"x": X_SHAPE},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=("mlprogram", "fp16"),
            minimum_deployment_target=ct.target.iOS16,
        )


class TestReduceMeanFusion:
    def test_valid_pattern1(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[-1, -2], keep_dims=True)
            x1 = mb.mul(x=1.0 / 30, y=x1)
            return x1

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_reduce_mean")
        assert get_op_types_in_program(prev_prog) == ["reduce_sum", "mul"]
        assert get_op_types_in_program(prog) == ["reduce_mean"]
        assert_model_is_valid(
            prog,
            {"x": (3, 5, 6)},
            expected_output_shapes={block.outputs[0].name: (3, 1, 1)},
        )

    def test_valid_pattern2(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 5))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[0], keep_dims=False)
            x1 = mb.real_div(x=x1, y=4.0)
            return x1

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_reduce_mean")
        assert get_op_types_in_program(prev_prog) == ["reduce_sum", "real_div"]
        assert get_op_types_in_program(prog) == ["reduce_mean"]
        assert_model_is_valid(
            prog,
            {"x": (4, 5)},
            expected_output_shapes={block.outputs[0].name: (5,)},
        )

    def test_invalid_pattern1(self):
        """
        The mul does not correspond to "1/count"
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[-1, -2], keep_dims=True)
            x1 = mb.mul(x=5.0, y=x1)
            return x1

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_reduce_mean")
        assert get_op_types_in_program(prog) == ["reduce_sum", "mul"]

    def test_invalid_pattern2(self):
        """
        The div does not correspond to "count"
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[-1, -2], keep_dims=True)
            x1 = mb.real_div(x=x1, y=31.0)
            return x1

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::fuse_reduce_mean")
        assert get_op_types_in_program(prog) == ["reduce_sum", "real_div"]

    def test_invalid_pattern3(self):
        """
        One of the reduction dim is symbolic
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, get_new_symbol(), 6))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[-1, -2], keep_dims=True)
            x1 = mb.real_div(x=x1, y=30.0)
            return x1

        pass_name = "common::fuse_reduce_mean"
        PASS_REGISTRY[pass_name](prog)
        assert get_op_types_in_program(prog) == ["reduce_sum", "real_div"]

    def test_invalid_pattern4(self):
        """
        output of reduce_sum is model output
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[-1, -2], keep_dims=True)
            y1 = mb.real_div(x=x1, y=30.0)
            return y1, x1

        pass_name = "common::fuse_reduce_mean"
        PASS_REGISTRY[pass_name](prog)
        assert get_op_types_in_program(prog) == ["reduce_sum", "real_div"]

    def test_invalid_pattern5(self):
        """
        output of reduce_sum is feeding into another op
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5, 6))])
        def prog(x):
            x1 = mb.reduce_sum(x=x, axes=[-1, -2], keep_dims=True)
            y1 = mb.real_div(x=x1, y=30.0)
            y2 = mb.mul(x=x1, y=10.0)
            y3 = mb.add(x=y1, y=y2)
            return y3

        pass_name = "common::fuse_reduce_mean"
        PASS_REGISTRY[pass_name](prog)
        assert get_op_types_in_program(prog) == ["reduce_sum", "real_div", "mul", "add"]


class TestLoopInvariantElimination:
    def test_loop_invariant_elimination1(self):
        """
        Invariant pattern: Block input vars are returned as block output vars.
        """

        def body(a, b):
            return mb.add(x=a, y=b), b

        def cond(a, b):
            a_mean = mb.reduce_mean(x=a, axes=[0, 1])
            b_mean = mb.reduce_mean(x=b, axes=[0, 1])
            return mb.less(x=a_mean, y=b_mean)

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2)),
                mb.TensorSpec(shape=(1, 2)),
            ]
        )
        def prog(a, b):
            # b is loop invariant
            return mb.while_loop(_cond=cond, _body=body, loop_vars=(a, b))

        while_op = prog.find_ops(op_type="while_loop", exactly_one=True)[0]
        assert len(while_op.blocks[0].inputs) == 2
        assert len(while_op.outputs) == 2
        assert len(while_op.loop_vars) == 2
        assert while_op.blocks[0].inputs[0].name == "a_x0"
        assert while_op.blocks[0].inputs[1].name == "b_x0"

        prev_prog = copy.deepcopy(prog)
        PASS_REGISTRY["common::loop_invariant_elimination"](prog)
        assert_same_output_names(prev_prog, prog)

        while_op = prog.find_ops(op_type="while_loop", exactly_one=True)[0]
        assert len(while_op.blocks[0].inputs) == 1
        assert len(while_op.outputs) == 1
        assert len(while_op.loop_vars) == 1
        assert while_op.blocks[0].inputs[0].name == "a_x0"

        if _VALIDATE_MODEL:
            assert_model_is_valid(prog, {"a": (1, 2), "b": (1, 2)})

    def test_loop_invariant_elimination2(self):
        """
        Invariant pattern: Block outputs var from outside of the block
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2)),
                mb.TensorSpec(shape=(1, 2)),
            ]
        )
        def prog(a, b):
            def body(a, bx):
                return mb.add(x=a, y=b), b

            def cond(a, bx):
                a_mean = mb.reduce_mean(x=a, axes=[0, 1])
                b_mean = mb.reduce_mean(x=bx, axes=[0, 1])
                return mb.less(x=a_mean, y=b_mean)

            # b is loop invariant
            return mb.while_loop(_cond=cond, _body=body, loop_vars=(a, b))

        while_op = prog.find_ops(op_type="while_loop", exactly_one=True)[0]
        assert len(while_op.blocks[0].inputs) == 2
        assert len(while_op.outputs) == 2
        assert len(while_op.loop_vars) == 2
        assert while_op.blocks[0].inputs[0].name == "a_x0"
        assert while_op.blocks[0].inputs[1].name == "b_x0"

        prev_prog = copy.deepcopy(prog)
        PASS_REGISTRY["common::loop_invariant_elimination"](prog)
        assert_same_output_names(prev_prog, prog)

        while_op = prog.find_ops(op_type="while_loop", exactly_one=True)[0]
        assert len(while_op.blocks[0].inputs) == 1
        assert len(while_op.outputs) == 1
        assert len(while_op.loop_vars) == 1
        assert while_op.blocks[0].inputs[0].name == "a_x0"

        if _VALIDATE_MODEL:
            assert_model_is_valid(prog, {"a": (1, 2), "b": (1, 2)})


class TestNoopElimination:
    @pytest.mark.parametrize("is_block_output", ((True, False)))
    def test_identity(self, is_block_output):
        """
        Input graph:

            input -> identity -> (add 1.0 if not is_block_output) -> output

        Output graph:

            if is_block_output:
                input -> identity -> output
            else:
                input -> add 1.0 -> output
        """
        SHAPE = (2, 3)

        @mb.program(input_specs=[mb.TensorSpec(shape=SHAPE)])
        def prog(x):
            y = mb.identity(x=x)
            if not is_block_output:
                y = mb.add(x=y, y=1.0)
            return y

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        if is_block_output:
            assert get_op_types_in_program(prev_prog) == ["identity"]
            assert get_op_types_in_program(prog) == ["identity"]
        else:
            assert get_op_types_in_program(prev_prog) == ["identity", "add"]
            assert get_op_types_in_program(prog) == ["add"]

        output_name = block.outputs[0].name
        assert_model_is_valid(
            prog,
            {"x": SHAPE},
            expected_output_shapes={output_name: SHAPE},
        )

    @pytest.mark.parametrize(
        "op_type, pos, val",
        itertools.product(
            ["add", "mul", "floor_div", "pow", "real_div", "sub"],
            ["x", "y"],
            [0.0, 1.0, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        ),
    )
    def test_elementwise_elimination(self, op_type, pos, val):
        if "div" in op_type and np.prod(val) == 0:
            return
        if "pow" in op_type and (val != 0 or val != 1):
            return

        test_op = getattr(mb, op_type)

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            if pos == "x":
                r1 = test_op(x=val, y=x)
            else:
                r1 = test_op(x=x, y=val)
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        original_program = [op_type, "relu"]
        new_program = original_program
        if op_type in {"add"}:
            if val == 0.0 or val == [0.0, 0.0, 0.0, 0.0]:
                new_program = ["relu"]
        elif op_type in {"mul"}:
            if val == 1.0 or val == [1.0, 1.0, 1.0, 1.0]:
                new_program = ["relu"]
        elif op_type in {"real_div"}:
            if pos == "y" and (val == 1.0 or val == [1.0, 1.0, 1.0, 1.0]):
                new_program = ["relu"]
        elif op_type in {"pow", "floor_div"}:
            if pos == "y" and (val == 1.0 or val == [1.0, 1.0, 1.0, 1.0]):
                new_program = ["relu"]
        elif op_type in {"sub"}:
            if pos == "y" and (val == 0.0 or val == [0.0, 0.0, 0.0, 0.0]):
                new_program = ["relu"]

        assert get_op_types_in_program(prev_prog) == original_program
        assert get_op_types_in_program(prog) == new_program
        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (2, 4)},
        )

    def test_elementwise_broadcast(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=[4])])
        def prog(x):
            r1 = mb.add(x=x, y=[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        original_program = ["add", "relu"]

        assert get_op_types_in_program(prev_prog) == original_program
        assert get_op_types_in_program(prog) == original_program
        assert_model_is_valid(
            prog,
            {"x": [4]},
            expected_output_shapes={block.outputs[0].name: (2, 4)},
        )

    def test_elementwise_elimination_fill(self):
        """
        When fill layer with dynamic shape is fed to elementwise-binary operation,
        even though the tensor can't be materialized at conversion time but no-op
        elimination can still be performed based on fill-value
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, get_new_symbol()))])
        def prog(x):
            shape = mb.shape(x=x)
            y = mb.fill(value=0.0, shape=shape)
            x = mb.add(x=x, y=y)
            return mb.relu(x=x)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["shape", "fill", "add", "relu"]
        assert get_op_types_in_program(prog) == ["shape", "fill", "relu"]

        apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["relu"]

        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (2, 4)},
        )

    def test_reshape_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            r1 = mb.reshape(x=x, shape=[1, 8])
            mb.reshape(x=r1, shape=[1, 8])
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["reshape", "reshape", "relu"]
        assert get_op_types_in_program(prog) == ["reshape", "relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (1, 8)},
        )

    def test_oneway_split_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            r1 = mb.split(x=x, num_splits=1, axis=-1)
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["split", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (2, 4)},
        )

    def test_full_split_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            r1 = mb.split(x=x, split_sizes=[4], axis=-1)
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["split", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (2, 4)},
        )

    def test_slicebysize_full_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            r1 = mb.slice_by_size(x=x, begin=[0, 0], size=[2, 4])
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["slice_by_size", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (2, 4)},
        )

    def test_slicebysize_to_end_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            r1 = mb.slice_by_size(x=x, begin=[0, 0], size=[-1, -1])
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["slice_by_size", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (2, 4)},
        )

    def test_slicebyindex_full_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            r1 = mb.slice_by_index(x=x, begin=[0, 0], end=[2, 4])
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (2, 4)},
        )

    def test_slicebyindex_negative_stride(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            r1 = mb.slice_by_index(
                x=x,
                begin=[0, 0],
                end=[0, 0],
                stride=[1, -1],
                begin_mask=[True, True],
                end_mask=[True, True],
            )
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", "relu"]
        assert get_op_types_in_program(prog) == ["slice_by_index", "relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (2, 4)},
        )

    @pytest.mark.parametrize(
        "begin_mask, end_mask",
        itertools.product(
            itertools.product([True, False], [True, False]),
            itertools.product([True, False], [True, False]),
        ),
    )
    def test_slicebyindex_mask_elimination(self, begin_mask, end_mask):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 4))])
        def prog(x):
            begin = [1, 1]
            end = [1, 1]
            for i in range(2):
                if not begin_mask[i]:
                    begin[i] = 0
                if not end_mask[i]:
                    end[i] = 4
            r1 = mb.slice_by_index(
                x=x, begin=begin, end=end, begin_mask=begin_mask, end_mask=end_mask
            )
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["slice_by_index", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (4, 4)},
            expected_output_shapes={block.outputs[0].name: (4, 4)},
        )

    def test_pad_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            r1 = mb.pad(x=x, pad=[0, 0, 0, 0])
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["pad", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (2, 4)},
        )

    def test_keep_pad(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            r1 = mb.pad(x=x, pad=[4, 4, 2, 2])
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["pad", "relu"]
        assert get_op_types_in_program(prog) == ["pad", "relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (10, 8)},
        )

    def test_tile_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            r1 = mb.tile(x=x, reps=[1, 1])
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["tile", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (2, 4)},
        )

    def test_keep_tile(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            r1 = mb.tile(x=x, reps=[2, 2])
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["tile", "relu"]
        assert get_op_types_in_program(prog) == ["tile", "relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (4, 8)},
        )

    def test_upsample_nearest_neighbor_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 2, 4))])
        def prog(x):
            r1 = mb.upsample_nearest_neighbor(x=x)
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["upsample_nearest_neighbor", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (3, 2, 4)},
            expected_output_shapes={block.outputs[0].name: (3, 2, 4)},
        )

    def test_upsample_bilinear_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 2, 4))])
        def prog(x):
            r1 = mb.upsample_bilinear(x=x)
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["upsample_bilinear", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (3, 2, 4)},
            expected_output_shapes={block.outputs[0].name: (3, 2, 4)},
        )

    def test_resize_bilinear_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 2, 4))])
        def prog(x):
            r1 = mb.resize_bilinear(x=x, target_size_height=2, target_size_width=4)
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["resize_bilinear", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (3, 2, 4)},
            expected_output_shapes={block.outputs[0].name: (3, 2, 4)},
        )

    def test_crop_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 2, 4))])
        def prog(x):
            r1 = mb.crop(x=x, crop_height=[0, 0], crop_width=[0, 0])
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["crop", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (3, 2, 4)},
            expected_output_shapes={block.outputs[0].name: (3, 2, 4)},
        )

    def test_linear_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 4))])
        def prog(x):
            r1 = mb.linear_activation(x=x, alpha=1.0, beta=0.0)
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["linear_activation", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 4)},
            expected_output_shapes={block.outputs[0].name: (2, 4)},
        )

    def test_transpose_elimination(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 4))])
        def prog(x):
            r1 = mb.transpose(x=x, perm=[0, 1, 2])
            return mb.relu(x=r1)

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::noop_elimination")
        assert get_op_types_in_program(prev_prog) == ["transpose", "relu"]
        assert get_op_types_in_program(prog) == ["relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 3, 4)},
            expected_output_shapes={block.outputs[0].name: (2, 3, 4)},
        )


class TestRemoveRedundantOps:
    def test_redundant_ops_just_after_input_valid_pattern_1(self):
        """
        Input graph:
        input----->transpose(perm=[0, 2, 1])--->add---> add ---> out
               |                                 ^       ^
               |                                 |       |
               |---->transpose(perm=[0, 2, 1])----       |
               |                                         |
               |                                         |
               |---->transpose(perm=[0, 2, 1])------------

        Output graph:
        input----->transpose(perm=[0, 2, 1])--->add---> add ----> out
                                    |            ^       ^
                                    |            |       |
                                    |-------------       |
                                    |                    |
                                    |--------------------
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x1 = mb.transpose(x=x, perm=[0, 2, 1])
            x2 = mb.transpose(x=x, perm=[0, 2, 1])
            x3 = mb.transpose(x=x, perm=[0, 2, 1])
            z = mb.add(x=x1, y=x2)
            z = mb.add(x=z, y=x3)
            return z

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::remove_redundant_ops")
        assert get_op_types_in_program(prev_prog) == [
            "transpose",
            "transpose",
            "transpose",
            "add",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["transpose", "add", "add"]
        assert_model_is_valid(
            prog,
            {"x": (2, 3, 5)},
            expected_output_shapes={block.outputs[0].name: (2, 5, 3)},
        )

    def test_redundant_ops_just_after_input_valid_pattern_2(self):
        """
        Input graph:
        input----->leaky_relu(alpha=0.3)--->add---> add ---> out
               |                             ^       ^
               |                             |       |
               |----->leaky_relu(alpha=0.3)---       |
               |                                     |
               |                                     |
               |---->leaky_relu(alpha=0.3)------------

        Output graph:
        input--------->leaky_relu(alpha=0.3)--->add---> add ----> out
                                    |            ^       ^
                                    |            |       |
                                    |-------------       |
                                    |                    |
                                    |---------------------
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x1 = mb.leaky_relu(x=x, alpha=0.3)
            x2 = mb.leaky_relu(x=x, alpha=0.3)
            x3 = mb.leaky_relu(x=x, alpha=0.3)
            z = mb.add(x=x1, y=x2)
            z = mb.add(x=z, y=x3)
            return z

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::remove_redundant_ops")
        assert get_op_types_in_program(prev_prog) == [
            "leaky_relu",
            "leaky_relu",
            "leaky_relu",
            "add",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["leaky_relu", "add", "add"]
        assert_model_is_valid(
            prog,
            {"x": (2, 3, 5)},
            expected_output_shapes={block.outputs[0].name: (2, 3, 5)},
        )

    def test_redundant_ops_just_after_input_valid_pattern_3(self):
        """
        Input graph:
        input----->leaky_relu(alpha=0.4)--->add---> add ---> out
               |                             ^       ^
               |                             |       |
               |----->leaky_relu(alpha=0.3)---       |
               |                                     |
               |                                     |
               |---->leaky_relu(alpha=0.3)------------

        Output graph:
        input----->leaky_relu(alpha=0.4)--->add---> add ---> out
               |                             ^       ^
               |                             |       |
               |----->leaky_relu(alpha=0.3)----------
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x1 = mb.leaky_relu(x=x, alpha=0.4)
            x2 = mb.leaky_relu(x=x, alpha=0.3)
            x3 = mb.leaky_relu(x=x, alpha=0.3)
            z = mb.add(x=x1, y=x2)
            z = mb.add(x=z, y=x3)
            return z

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::remove_redundant_ops")
        assert get_op_types_in_program(prev_prog) == [
            "leaky_relu",
            "leaky_relu",
            "leaky_relu",
            "add",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["leaky_relu", "leaky_relu", "add", "add"]

        leaky_relu_ops = block.find_ops(op_type="leaky_relu")
        assert leaky_relu_ops[0].alpha.val == np.float32(0.4)
        assert leaky_relu_ops[1].alpha.val == np.float32(0.3)

    def test_redundant_ops_just_after_input_invalid_pattern_1(self):
        """
        input----->transpose(perm=[0, 2, 1])---> reshape(shape=[-1]) -----> add ---> out
               |                                                             ^
               |                                                             |
               |---->transpose(perm=[1, 0, 2])----> reshape(shape=[-1])------
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x1 = mb.transpose(x=x, perm=[0, 2, 1])
            x2 = mb.transpose(x=x, perm=[1, 0, 2])
            x1 = mb.reshape(x=x1, shape=[-1])
            x2 = mb.reshape(x=x2, shape=[-1])
            z = mb.add(x=x1, y=x2)
            return z

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::remove_redundant_ops")
        assert get_op_types_in_program(prev_prog) == [
            "transpose",
            "transpose",
            "reshape",
            "reshape",
            "add",
        ]
        assert get_op_types_in_program(prog) == [
            "transpose",
            "transpose",
            "reshape",
            "reshape",
            "add",
        ]
        assert_model_is_valid(
            prog,
            {"x": (2, 3, 5)},
            expected_output_shapes={block.outputs[0].name: (30,)},
        )

    def test_redundant_ops_just_after_input_invalid_pattern_2(self):
        """
        input----->leaky_relu(alpha=0.3) -----> add ---> out
               |                                 ^
               |                                 |
               |---->leaky_relu(alpha=0.4)-------

        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x1 = mb.leaky_relu(x=x, alpha=0.3)
            x2 = mb.leaky_relu(x=x, alpha=0.4)
            z = mb.add(x=x1, y=x2)
            return z

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::remove_redundant_ops")
        assert get_op_types_in_program(prev_prog) == ["leaky_relu", "leaky_relu", "add"]
        assert get_op_types_in_program(prog) == ["leaky_relu", "leaky_relu", "add"]
        assert_model_is_valid(
            prog,
            {"x": (2, 3, 5)},
            expected_output_shapes={block.outputs[0].name: (2, 3, 5)},
        )

    def test_redundant_ops_just_after_input_invalid_pattern_3(self):
        """
        test case, when inputs of 1 op is a subset of the inputs of the other op

        input----->layer_norm1 -----> add ---> out
               |                       ^
               |                       |
               |---->layer_norm2-------

        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 2))])
        def prog(x):
            x1 = mb.layer_norm(x=x, axes=[2], epsilon=1e-4)
            gamma_val = np.array([1.0, 1.0], dtype=np.float32)
            beta_val = np.array([1.0, 0.0], dtype=np.float32)
            x2 = mb.layer_norm(x=x, axes=[2], epsilon=1e-4, gamma=gamma_val, beta=beta_val)
            z = mb.add(x=x1, y=x2)
            return z

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::remove_redundant_ops")
        assert get_op_types_in_program(prev_prog) == ["layer_norm", "layer_norm", "add"]
        assert get_op_types_in_program(prog) == ["layer_norm", "layer_norm", "add"]
        assert_model_is_valid(
            prog,
            {"x": (1, 3, 2)},
            expected_output_shapes={block.outputs[0].name: (1, 3, 2)},
        )

    @staticmethod
    def _make_repeated_conv_prog(redundant_conv=True, out_channel=2):
        prog = mil.Program()
        func_inputs = {"x": mb.placeholder(shape=[1, 4, 5, 5])}
        with Function(func_inputs) as ssa_fun:
            x = ssa_fun.inputs["x"]
            x = mb.relu(x=x)
            W = np.random.rand(out_channel, 4, 3, 3)
            if redundant_conv:
                bias = np.random.rand(out_channel)
                x1 = mb.conv(x=x, weight=W, bias=bias, pad_type="same", strides=[1, 1])
                x2 = mb.conv(x=x, weight=W, bias=bias, pad_type="same", strides=[1, 1])
            else:
                x1 = mb.conv(
                    x=x, weight=W, bias=np.random.rand(out_channel), pad_type="same", strides=[1, 1]
                )
                x2 = mb.conv(
                    x=x, weight=W, bias=np.random.rand(out_channel), pad_type="same", strides=[1, 1]
                )
            x1 = mb.relu(x=x1)
            x2 = mb.relu(x=x2)
            x1 = mb.avg_pool(x=x1, kernel_sizes=[2, 2], strides=[1, 1], pad_type="same")
            z = mb.concat(values=(x1, x2), axis=-3)
            ssa_fun.set_outputs([z])
        prog.add_function("main", ssa_fun)
        return prog

    def test_redundant_ops_inside_graph_valid_pattern(self):
        """
        Input graph:
        input--> relu--------->conv------>relu----> pool ---> concat ---> out
                 |                                              ^
                 |                                              |
                 |---->conv---->relu----------------------------

        Output graph:
        input-> relu--->conv------>relu----> pool ---> concat ---> out
                                    |                   ^
                                    |                   |
                                    |-------------------
        """
        prog = self._make_repeated_conv_prog(redundant_conv=True)

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::remove_redundant_ops")
        assert get_op_types_in_program(prev_prog) == [
            "relu",
            "conv",
            "conv",
            "relu",
            "relu",
            "avg_pool",
            "concat",
        ]
        assert get_op_types_in_program(prog) == ["relu", "conv", "relu", "avg_pool", "concat"]
        assert_model_is_valid(
            prog,
            {"x": (1, 4, 5, 5)},
            expected_output_shapes={block.outputs[0].name: (1, 4, 5, 5)},
        )

    def test_redundant_ops_inside_graph_with_large_const(self):
        """
        For the large constants, they need to be deduplicated by the const_deduplication first.
        This test is making sure the converter is not doing any "brutal force" comparison.

        Input graph:
        input--> relu--------->conv------>relu----> pool ---> concat ---> out
                 |                                              ^
                 |                                              |
                 |---->conv---->relu----------------------------

        Output graph:
        input-> relu--->conv------>relu----> pool ---> concat ---> out
                                    |                   ^
                                    |                   |
                                    |-------------------
        """
        # The remove_redundant_ops is not doing brutal force array comparison
        prog = self._make_repeated_conv_prog(redundant_conv=True, out_channel=10)
        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::remove_redundant_ops")
        ops_in_prev_prog = [
            "relu",
            "conv",
            "conv",
            "relu",
            "relu",
            "avg_pool",
            "concat",
        ]
        assert get_op_types_in_program(prev_prog) == ops_in_prev_prog
        assert get_op_types_in_program(prog) == ops_in_prev_prog

        # We need to first run the const_deduplication pass.
        prog = self._make_repeated_conv_prog(redundant_conv=True, out_channel=10)
        _, _, block = apply_pass_and_basic_check(prog, "common::const_deduplication")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")
        _, _, block = apply_pass_and_basic_check(prog, "common::remove_redundant_ops")

        assert get_op_types_in_program(prog) == ["relu", "conv", "relu", "avg_pool", "concat"]
        assert_model_is_valid(
            prog,
            {"x": (1, 4, 5, 5)},
            expected_output_shapes={block.outputs[0].name: (1, 20, 5, 5)},
        )

    def test_redundant_ops_inside_graph_invalid_pattern(self):
        """
        input--->relu--------->conv1------>relu----> pool ---> concat ---> out
                  |                                              ^
                  |                                              |
                  |---->conv2---->relu---------------------------
        """
        prog = self._make_repeated_conv_prog(redundant_conv=False)

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::remove_redundant_ops")
        assert get_op_types_in_program(prev_prog) == [
            "relu",
            "conv",
            "conv",
            "relu",
            "relu",
            "avg_pool",
            "concat",
        ]
        assert get_op_types_in_program(prog) == [
            "relu",
            "conv",
            "conv",
            "relu",
            "relu",
            "avg_pool",
            "concat",
        ]
        assert_model_is_valid(
            prog,
            {"x": (1, 4, 5, 5)},
            expected_output_shapes={block.outputs[0].name: (1, 4, 5, 5)},
        )

    def test_redundant_op_as_output_valid_pattern_1(self):
        """
        Input graph:
        input--------->relu------> out1
               |
               |
               |---->relu---->tanh---> out2

        Output graph:
        input--------->relu------> out1
                             |
                             |
                             |---->tanh---> out2
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x1 = mb.relu(x=x)
            x2 = mb.relu(x=x)
            return x1, mb.tanh(x=x2)

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::remove_redundant_ops")
        assert get_op_types_in_program(prev_prog) == ["relu", "relu", "tanh"]
        assert get_op_types_in_program(prog) == ["relu", "tanh"]
        assert_model_is_valid(
            prog,
            {"x": (2, 3, 5)},
            expected_output_shapes={
                block.outputs[0].name: (2, 3, 5),
                block.outputs[1].name: (2, 3, 5),
            },
        )

    def test_redundant_op_as_output_invalid_pattern_1(self):
        """
        Input graph:
        input--------->relu------> out1
               |
               |
               |---->relu---> out2

        "common::remove_redundant_ops" pass does not remove ops if their outputs
        are block outputs.
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x1 = mb.relu(x=x)
            x2 = mb.relu(x=x)
            return x1, x2

        prev_prog, _, block = apply_pass_and_basic_check(
            prog,
            "common::remove_redundant_ops",
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "relu"]
        assert get_op_types_in_program(prog) == ["relu", "relu"]
        assert_model_is_valid(
            prog,
            {"x": (2, 3, 5)},
            expected_output_shapes={
                block.outputs[0].name: (2, 3, 5),
                block.outputs[1].name: (2, 3, 5),
            },
        )

    def test_cond_block_program(self):
        """
        - Test identical ops within different blocks are not removed. The "relu" op inside true and
        false blocks are not removed since they are in different blocks.
        - Test ops that have blocks inside them are not removed. There are two cond ops here,
        with identical inputs but they are not removed, since they are ops that have nested block
        inside them.
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,))])
        def prog(x):
            x1 = mb.cast(x=x, dtype="bool")

            def true_fn():
                x = mb.shape(x=x1)
                x = mb.cast(x=x, dtype="fp32")
                return mb.add(x=x, y=1.0)

            def false_fn():
                x = mb.shape(x=x1)
                x = mb.cast(x=x, dtype="fp32")
                return mb.add(x=x, y=-1.0)

            z1 = mb.cond(pred=x1, _true_fn=true_fn, _false_fn=false_fn)
            z2 = mb.cond(pred=x1, _true_fn=true_fn, _false_fn=false_fn)
            z = mb.add(x=z1, y=z2)
            return z

        prev_prog, _, block = apply_pass_and_basic_check(
            prog,
            "common::remove_redundant_ops",
        )
        assert get_op_types_in_program(prev_prog) == ["cast", "cond", "cond", "add"]
        assert get_op_types_in_program(prog) == ["cast", "cond", "cond", "add"]
        cond_op = prog.find_ops(op_type="cond")[0]
        assert cond_op.blocks[0].operations[0].op_type == "shape"
        assert cond_op.blocks[1].operations[0].op_type == "shape"
        assert_model_is_valid(
            prog,
            {"x": (1,)},
            expected_output_shapes={block.outputs[0].name: (1,)},
        )

    def test_concat_op_pattern(self):
        """
        Input graph:
                          ---------------> concat ------> log ------> out1
                         |                   ^
                         |                   |
        input--------->relu------> concat ------> relu----> out2
                 |                  ^        |
                 |                  |        |
                 |---->tanh--------------------

        Output graph:
                                     |------>log ------> out1
                                     |
                                     |
        input--------->relu------> concat ------> relu----> out2
                 |                  ^
                 |                  |
                 |---->tanh---------
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 5))])
        def prog(x):
            x1 = mb.relu(x=x)
            x2 = mb.tanh(x=x)
            c1 = mb.concat(values=(x1, x2), axis=0)
            c2 = mb.concat(values=(x1, x2), axis=0)
            z1 = mb.log(x=c1)
            z2 = mb.relu(x=c2)
            return z1, z2

        prev_prog, _, block = apply_pass_and_basic_check(
            prog,
            "common::remove_redundant_ops",
        )
        assert get_op_types_in_program(prev_prog) == [
            "relu",
            "tanh",
            "concat",
            "concat",
            "log",
            "relu",
        ]
        assert get_op_types_in_program(prog) == ["relu", "tanh", "concat", "log", "relu"]
        assert_model_is_valid(
            prog,
            {"x": (10, 5)},
            expected_output_shapes={block.outputs[0].name: (20, 5), block.outputs[1].name: (20, 5)},
        )

    def test_multiple_redundant_child_ops_pattern(self):
        """
        Input graph

        input -------------> reshape ----------> add ---------> out1
                  |                               ^
                  |                               |
                  |-------> reshape ---------------
                  |
                  |------> slice_by_size-----> add ----------> out2
                  |                             ^
                  |                             |
                  |------> slice_by_size -------

        Output graph

        input -------------> reshape ----------> add ------------> out1
          |                              |        ^
          |                              |        |
          |                              |---------
          |
          |------> slice_by_size----------> add -----------------> out2
                        |                    ^
                        |                    |
                        |---------------------

        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 5, 4))])
        def prog(x):
            x1 = mb.reshape(x=x, shape=[5, 2, -1])
            x2 = mb.reshape(x=x, shape=[5, 2, -1])
            x3 = mb.slice_by_size(x=x, begin=[0, 0, 1], size=[2, 4, 3])
            x4 = mb.slice_by_size(x=x, begin=[0, 0, 1], size=[2, 4, 3])
            z1 = mb.add(x=x1, y=x2)
            z2 = mb.add(x=x3, y=x4)
            return z1, z2

        prev_prog, _, block = apply_pass_and_basic_check(
            prog,
            "common::remove_redundant_ops",
        )
        assert get_op_types_in_program(prev_prog) == [
            "reshape",
            "reshape",
            "slice_by_size",
            "slice_by_size",
            "add",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["reshape", "slice_by_size", "add", "add"]
        assert_model_is_valid(
            prog,
            {"x": (10, 5, 4)},
            expected_output_shapes={
                block.outputs[0].name: (5, 2, 20),
                block.outputs[1].name: (2, 4, 3),
            },
        )

    def test_random_distribution_op_invalid_pattern(self):
        """
        Identical random ops are not removed

        input----->cast---->random_uniform------> add ---> out
                    |                              ^
                    |                              |
                    |---->random_uniform------------
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(3,))])
        def prog(shape):
            shape = mb.cast(x=shape, dtype="int32")
            x1 = mb.random_uniform(shape=shape, low=0.0, high=1.0, seed=11)
            x2 = mb.random_uniform(shape=shape, low=0.0, high=1.0, seed=11)
            return mb.add(x=x1, y=x2)

        prev_prog, _, block = apply_pass_and_basic_check(
            prog,
            "common::remove_redundant_ops",
        )
        assert get_op_types_in_program(prev_prog) == [
            "cast",
            "random_uniform",
            "random_uniform",
            "add",
        ]
        assert get_op_types_in_program(prog) == ["cast", "random_uniform", "random_uniform", "add"]

    def test_nonreplaceable_vars(self):
        """
        Nonreplaceable vars shouldn't be removed, e.g. palettized weights

        const_1----->add---->add_1------|
                      |                 |
                    input              add---->output
                      |                 |
        const_2----->add---->add_2------|
        """

        def _constexpr_lut_to_dense():
            lut_data = np.array(
                [-19.0, 4.0, 0.0, -1.0, 1.0, 3.0, 5.0, -8.0, 19, 13, 42, 4.5, 5.4, 2.0, -6, -7]
            ).astype(np.float32)
            indices = np.array([212, 21]).astype(np.uint8)
            shape = np.array([4, 1]).astype(np.uint32)
            return mb.constexpr_lut_to_dense(lut=lut_data, indices=indices, shape=shape)

        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 1))])
        def prog(x):
            constexpr_1 = _constexpr_lut_to_dense()
            constexpr_2 = _constexpr_lut_to_dense()
            c = mb.add(x=constexpr_1, y=x)
            d = mb.add(x=constexpr_2, y=x)
            return mb.add(x=c, y=d)

        prev_prog, _, _ = apply_pass_and_basic_check(
            prog,
            "common::remove_redundant_ops",
        )
        assert get_op_types_in_program(prev_prog) == get_op_types_in_program(prog)

    def test_redundant_ops_time_complexity(self):
        """
        Test the graph pass doesn't re-run right away after detecting a redundant pattern,
        in order to keep time complexity low.

        In this example, a program with 26 ops is first traversed, and 5 relu ops are removed.
        At the time of second traversal, there are only 21 remaining ops.
        As the result, the total ops of visited is 26 + 21 = 47.
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x = mb.cos(x=x)
            for i in range(5):
                x1 = mb.relu(x=x)
                x2 = mb.relu(x=x)
                z = mb.add(x=x1, y=x2)
                z = mb.add(x=z, y=x2)
                x = mb.sin(x=x)
            return x

        graph_pass = remove_redundant_ops()
        graph_pass.apply(prog)

        assert get_op_types_in_program(prog) == ["cos"] + ["relu", "add", "add", "sin"] * 5
        assert graph_pass._num_of_visited_ops == 47

    def test_redundant_ops_time_complexity_pattern_2(self):
        """
        Test the graph pass doesn't re-run right away after detecting a redundant pattern,
        in order to keep time complexity low.

        In this example, there are three groups of identical leaky_relu ops can be removed,
        and the algorithm should be run in the fashion that only goes through the
        program twice. As the result, the total ops visited is:

        8 + (8 - 3) = 13
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3, 5))])
        def prog(x):
            x = mb.cos(x=x)
            x1 = mb.leaky_relu(x=x, alpha=0.2)
            x2 = mb.leaky_relu(x=x, alpha=0.2)
            x3 = mb.leaky_relu(x=x, alpha=0.3)
            x4 = mb.leaky_relu(x=x, alpha=0.3)
            x5 = mb.leaky_relu(x=x, alpha=0.4)
            x6 = mb.leaky_relu(x=x, alpha=0.4)
            return mb.sin(x=x6)

        graph_pass = remove_redundant_ops()
        graph_pass.apply(prog)

        assert get_op_types_in_program(prog) == ["cos"] + ["leaky_relu"] * 3 + ["sin"]
        assert graph_pass._num_of_visited_ops == 13


class TestRemoveSymbolicReshape:
    def test_remove_symbolic_reshape(self):
        sym_b = Symbol("s0")
        original_shape = (sym_b, Symbol("s1"), 2)
        reshape_name = "reshape"

        @mb.program(input_specs=[mb.TensorSpec(shape=(sym_b, 4))])
        def prog(x):
            # const cannot represent symbolic values. Use _const_symbolic
            shape = mb._const_symbolic(val=original_shape)
            return mb.reshape(x=x, shape=shape, name=reshape_name)

        reshape_op = prog.find_ops(prefix=reshape_name, op_type="reshape", exactly_one=True)[0]
        shape_var = reshape_op.shape
        reshaped_var = reshape_op.outputs[0]
        assert np.all(shape_var.sym_val == original_shape)
        assert np.all(reshaped_var.shape == (sym_b, 2, 2))

        # Note: we cannot deepcopy prog with symbol.
        prev_outputs = [o.name for o in prog["main"].outputs]
        PASS_REGISTRY["common::remove_symbolic_reshape"](prog)
        curr_outputs = [o.name for o in prog["main"].outputs]
        assert curr_outputs == prev_outputs

        reshape_op = prog.find_ops(prefix=reshape_name, op_type="reshape", exactly_one=True)[0]
        shape_var = reshape_op.shape
        reshaped_var = reshape_op.outputs[0]
        # shape param cannot be symbolic after the pass
        assert np.all(shape_var.sym_val == (-1, 2, 2))
        # output shape is still symbolic
        assert np.all(reshaped_var.shape == (sym_b, 2, 2))

        if _VALIDATE_MODEL:
            assert_model_is_valid(prog, {"x": (3, 4)})


class TestTopologicalReorder:
    def test_move_sink_casts_to_the_end(self):
        """
        Input graph:
            x (input) ---> square ---> cast (output)
            |
            | -----------> log ------> cast (output)
            |
            | -----------> relu -----> cast ----> relu (output)
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x1 = mb.square(x=x)
            x2 = mb.cast(x=x1, dtype="fp32")
            x3 = mb.log(x=x)
            x4 = mb.cast(x=x3, dtype="fp32")
            x5 = mb.relu(x=x)
            x6 = mb.cast(x=x5, dtype="fp32")
            x7 = mb.relu(x=x6)
            return x2, x4, x7

        assert get_op_types_in_program(prog) == [
            "cast",
            "square",
            "cast",
            "log",
            "cast",
            "relu",
            "cast",
            "relu",
        ]

        apply_pass_and_basic_check(prog, "common::topological_reorder")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == [
            "cast",
            "square",
            "log",
            "relu",
            "cast",
            "relu",
            "cast",
            "cast",
        ]

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20),
                block.outputs[2].name: (10, 20),
            },
        )

    def test_move_sink_cast_transpose_to_the_end(self):
        """
        Input graph:
            x (input) ---> square ---> transpose ---> cast (output)
            |
            | -----------> log ------> transpose ---> cast (output)
            |
            | -----------> relu -----> cast ----> relu (output)
            |
            | -----------> relu (output)
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x1 = mb.square(x=x)
            x1_t = mb.transpose(x=x1, perm=[1, 0])
            x2 = mb.cast(x=x1_t, dtype="fp32")
            x3 = mb.log(x=x)
            x3_t = mb.transpose(x=x3, perm=[1, 0])
            x4 = mb.cast(x=x3_t, dtype="fp32")
            x5 = mb.relu(x=x)
            x6 = mb.cast(x=x5, dtype="fp32")
            x7 = mb.relu(x=x6)
            x8 = mb.relu(x=x)
            return x2, x4, x7, x8

        assert get_op_types_in_program(prog) == [
            "cast",
            "square",
            "transpose",
            "cast",
            "log",
            "transpose",
            "cast",
            "relu",
            "cast",
            "relu",
            "relu",
        ]

        apply_pass_and_basic_check(prog, "common::topological_reorder")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == [
            "cast",
            "square",
            "log",
            "relu",
            "cast",
            "relu",
            "relu",
            "transpose",
            "cast",
            "transpose",
            "cast",
        ]

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (20, 10),
                block.outputs[1].name: (20, 10),
                block.outputs[2].name: (10, 20),
                block.outputs[3].name: (10, 20),
            },
        )

    def test_move_multiple_uses_overlapping(self):
        """
        Input graph:
            x (input) ---> cast ---> cast (output)
                           |
                           |-------> transpose ---> transpose (output)
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x1 = mb.cast(x=x, dtype="fp16")
            x2 = mb.cast(x=x1, dtype="fp32")
            x3 = mb.transpose(x=x1, perm=[1, 0])
            x4 = mb.transpose(x=x3, perm=[1, 0])
            return x2, x4

        assert get_op_types_in_program(prog) == ["cast", "cast", "transpose", "transpose"]

        apply_pass_and_basic_check(prog, "common::topological_reorder")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == ["cast", "transpose", "transpose", "cast"]

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (10, 20),
            },
        )

    def test_move_split_to_first_use(self):
        """
        Input graph:
            x (input) ---> split ---> square ---> add (output)
            |                |                     |
            |                | --------------------|
            |
            | -----------> square --------------> relu (output)
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            s1, s2 = mb.split(x=x, num_splits=2, axis=0)
            x2 = mb.square(x=x)
            x3 = mb.relu(x=x2)
            s1_1 = mb.square(x=s1)
            s3 = mb.add(x=s1_1, y=s2)
            return x3, s3

        assert get_op_types_in_program(prog) == ["split", "square", "relu", "square", "add"]

        block = prog.functions["main"]
        # Reorder `split` op to test op with multiple output case
        topological_reorder._move_operations_to_the_end_block(block, ["split"])
        assert get_op_types_in_program(prog) == ["square", "relu", "split", "square", "add"]

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (10, 20),
                block.outputs[1].name: (5, 20),
            },
        )

    def test_move_transpose_before_subblock(self):
        """
        Input graph:
            x (input) ---> cast ---> transpose ---> cast (output)
            |
            | -----------> square ------> transpose (x1_t) ---> cast (output)
            |
            | -----------> squeeze ----> equal ----> squeeze
                                                        |
                                          (true) <--- /  \ ---> (false)
                                            |                        |
                                            |      /<-(x1_t)->\      |
                                           add  <-/            \--> add
                                            |---------> | <---------|
                                                        |
                                                       add ---> cast (output)
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x1 = mb.square(x=x)
            x1_t = mb.transpose(x=x1, perm=[1, 0])

            def true_fn():
                return mb.add(x=x1_t, y=np.float16(1), name="x2")

            def false_fn():
                return mb.add(x=x1_t, y=np.float16(2), name="x2")

            is_one = mb.equal(x=mb.squeeze(x=x), y=np.float16(1.0))
            pred = mb.squeeze(x=is_one)
            x3 = mb.cond(pred=pred, _true_fn=true_fn, _false_fn=false_fn)
            x4 = mb.add(x=x1_t, y=x3)
            x5 = mb.cast(x=x4, dtype="fp32")
            return x5

        apply_pass_and_basic_check(prog, "common::topological_reorder")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == [
            "cast",
            "square",
            "squeeze",
            "equal",
            "squeeze",
            "transpose",
            "cond",
            "add",
            "cast",
        ]

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={block.outputs[0].name: (20, 10)},
        )

    def test_cast_transpose_already_at_the_end(self):
        """
        Input graph:
            x (input) ---> square ---> transpose ---> cast (output)
            |
            | -----------> log ------> transpose ---> cast (output)
            |
            | -----------> relu -----> cast ----> relu (output)
            |
            | -----------> relu (output)
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.cast(x=x, dtype="fp16")
            x1 = mb.square(x=x)
            x3 = mb.log(x=x)
            x5 = mb.relu(x=x)
            x6 = mb.cast(x=x5, dtype="fp32")
            x7 = mb.relu(x=x6)
            x8 = mb.relu(x=x)
            x1_t = mb.transpose(x=x1, perm=[1, 0])
            x2 = mb.cast(x=x1_t, dtype="fp32")
            x3_t = mb.transpose(x=x3, perm=[1, 0])
            x4 = mb.cast(x=x3_t, dtype="fp32")
            return x2, x4, x7, x8

        assert get_op_types_in_program(prog) == [
            "cast",
            "square",
            "log",
            "relu",
            "cast",
            "relu",
            "relu",
            "transpose",
            "cast",
            "transpose",
            "cast",
        ]

        apply_pass_and_basic_check(prog, "common::topological_reorder")
        _, _, block = apply_pass_and_basic_check(prog, "common::dead_code_elimination")

        assert get_op_types_in_program(prog) == [
            "cast",
            "square",
            "log",
            "relu",
            "cast",
            "relu",
            "relu",
            "transpose",
            "cast",
            "transpose",
            "cast",
        ]

        assert_model_is_valid(
            prog,
            {"x": (10, 20)},
            expected_output_shapes={
                block.outputs[0].name: (20, 10),
                block.outputs[1].name: (20, 10),
                block.outputs[2].name: (10, 20),
                block.outputs[3].name: (10, 20),
            },
        )
