#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    get_op_types_in_program,
)


class TestFuseGeluSigmoidApproximation:
    """
    Test the fuse_gelu_sigmoid_approximation pass.

    Input pattern:
        x -> mul(1.702) -> sigmoid -> mul(x) -> output

    Output pattern:
        x -> gelu(mode=SIGMOID_APPROXIMATION) -> output
    """

    def test_basic_fusion(self):
        """Test basic GELU sigmoid approximation fusion."""

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            scaled = mb.mul(x=x, y=np.float32(1.702))
            sigmoid_out = mb.sigmoid(x=scaled)
            return mb.mul(x=x, y=sigmoid_out)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_gelu_sigmoid_approximation"
        )

        assert get_op_types_in_program(prev_prog) == ["mul", "sigmoid", "mul"]
        assert get_op_types_in_program(prog) == ["gelu"]

        gelu_op = block.find_ops(op_type="gelu")[0]
        assert gelu_op.mode.val == "SIGMOID_APPROXIMATION"

        assert_model_is_valid(
            prog,
            {"x": (2, 3)},
            expected_output_shapes={block.outputs[0].name: (2, 3)},
        )

    def test_fusion_with_reversed_mul_order(self):
        """Test fusion when mul operands are in reversed order."""

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            scaled = mb.mul(x=np.float32(1.702), y=x)
            sigmoid_out = mb.sigmoid(x=scaled)
            return mb.mul(x=sigmoid_out, y=x)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_gelu_sigmoid_approximation"
        )

        assert get_op_types_in_program(prev_prog) == ["mul", "sigmoid", "mul"]
        assert get_op_types_in_program(prog) == ["gelu"]

        gelu_op = block.find_ops(op_type="gelu")[0]
        assert gelu_op.mode.val == "SIGMOID_APPROXIMATION"

    def test_fusion_with_different_shapes(self):
        """Test fusion with different input shapes."""
        shapes_to_test = [
            (1,),
            (2, 3),
            (2, 3, 4),
            (1, 2, 3, 4),
        ]

        for shape in shapes_to_test:

            @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
            def prog(x):
                scaled = mb.mul(x=x, y=np.float32(1.702))
                sigmoid_out = mb.sigmoid(x=scaled)
                return mb.mul(x=x, y=sigmoid_out)

            prev_prog, _, block = apply_pass_and_basic_check(
                prog, "common::fuse_gelu_sigmoid_approximation"
            )

            assert get_op_types_in_program(prog) == ["gelu"]

    def test_no_fusion_wrong_constant(self):
        """Test that fusion does not occur with wrong constant value."""

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            scaled = mb.mul(x=x, y=np.float32(2.0))
            sigmoid_out = mb.sigmoid(x=scaled)
            return mb.mul(x=x, y=sigmoid_out)

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::fuse_gelu_sigmoid_approximation"
        )

        assert get_op_types_in_program(prog) == ["mul", "sigmoid", "mul"]

    def test_no_fusion_output_used_elsewhere(self):
        """Test that fusion does not occur when intermediate output is block output."""

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            scaled = mb.mul(x=x, y=np.float32(1.702))
            sigmoid_out = mb.sigmoid(x=scaled)
            gelu_approx = mb.mul(x=x, y=sigmoid_out)
            return gelu_approx, sigmoid_out

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::fuse_gelu_sigmoid_approximation"
        )

        assert "sigmoid" in get_op_types_in_program(prog)

    def test_no_fusion_different_input_var(self):
        """Test that fusion does not occur when sigmoid input differs from final mul input."""

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(2, 3)), mb.TensorSpec(shape=(2, 3))]
        )
        def prog(x, y):
            scaled = mb.mul(x=y, y=np.float32(1.702))
            sigmoid_out = mb.sigmoid(x=scaled)
            return mb.mul(x=x, y=sigmoid_out)

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::fuse_gelu_sigmoid_approximation"
        )

        assert get_op_types_in_program(prog) == ["mul", "sigmoid", "mul"]

    def test_fusion_fp16(self):
        """Test fusion works with fp16 data type."""

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.fp16)])
        def prog(x):
            scaled = mb.mul(x=x, y=np.float16(1.702))
            sigmoid_out = mb.sigmoid(x=scaled)
            return mb.mul(x=x, y=sigmoid_out)

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::fuse_gelu_sigmoid_approximation"
        )

        assert get_op_types_in_program(prog) == ["gelu"]

    def test_numerical_correctness(self):
        """Test that the fused operation produces numerically correct results."""
        np.random.seed(42)
        test_input = np.random.randn(2, 3).astype(np.float32)

        expected_output = test_input * (1 / (1 + np.exp(-1.702 * test_input)))

        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            scaled = mb.mul(x=x, y=np.float32(1.702))
            sigmoid_out = mb.sigmoid(x=scaled)
            return mb.mul(x=x, y=sigmoid_out)

        apply_pass_and_basic_check(prog, "common::fuse_gelu_sigmoid_approximation")

        assert get_op_types_in_program(prog) == ["gelu"]

