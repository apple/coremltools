#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import numpy as np
import pytest
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.defs.iOS15 import elementwise_unary


# Mock class to simulate the input_var behavior
class MockInputVar:
    def __init__(self, val, sym_type):
        self.val = val
        self.sym_type = sym_type


class TestCast:
    NUMPY_DTYPE_TO_STRING = {
        np.int32: "int32",
        np.float16: "fp16",
        np.float32: "fp32",
        np.bool_: "bool",
    }

    @pytest.mark.parametrize(
        "value, dtype",
        itertools.product(
            [2.0, (0.0, 1.0)],
            [np.int32, np.float16, np.float32, np.bool_],
        ),
    )
    def test_cast(self, value, dtype):
        input_var = MockInputVar(val=value, sym_type=None)
        output = elementwise_unary.cast.get_cast_value(
            input_var, self.NUMPY_DTYPE_TO_STRING[dtype]
        )
        expected_output = dtype(value)
        np.testing.assert_array_equal(output, expected_output)


class TestEpsilonForDtype:
    """`log` and `rsqrt` default to epsilons that underflow fp16.

    1e-45 and 1e-12 are representable in fp32 but cast to exactly 0 in fp16, so
    the stabilizer disappeared for the dtype most Core ML models actually use
    and `log(0)` / `rsqrt(0)` folded to -inf / inf.
    """

    @pytest.mark.parametrize(
        "builtin_dtype, np_dtype, value",
        [
            (types.fp32, np.float32, 1e-12),
            (types.fp32, np.float32, 1e-45),
            (types.fp16, np.float16, 1e-12),
            (types.fp16, np.float16, 1e-45),
        ],
    )
    def test_epsilon_is_never_zero(self, builtin_dtype, np_dtype, value):
        epsilon = elementwise_unary._epsilon_for_dtype(builtin_dtype, value)

        assert epsilon.dtype == np_dtype
        assert epsilon > 0, f"{value} underflowed to zero for {np_dtype.__name__}"

    @pytest.mark.parametrize("value", [1e-12, 1e-45])
    def test_fp32_is_unchanged(self, value):
        """Both defaults are representable in fp32, so nothing should move."""
        assert elementwise_unary._epsilon_for_dtype(types.fp32, value) == np.float32(value)

    @pytest.mark.parametrize("value", [1e-12, 1e-45])
    def test_fp16_falls_back_to_the_smallest_subnormal(self, value):
        expected = np.nextafter(np.float16(0), np.float16(1))
        assert elementwise_unary._epsilon_for_dtype(types.fp16, value) == expected


class TestFp16EpsilonKeepsFoldedValuesFinite:
    """The behavioural consequence, independent of how the epsilon is produced.

    Folding `rsqrt(0)` or `log(0)` must not yield inf / -inf just because the
    tensor is fp16. fp32 already behaved; fp16 did not.
    """

    @pytest.mark.parametrize(
        "op_name, np_dtype",
        itertools.product(["rsqrt", "log"], [np.float16, np.float32]),
    )
    def test_folded_value_is_finite_at_zero(self, op_name, np_dtype):
        from coremltools.converters.mil.mil import Builder as mb

        x = np.array([0.0, 1e-8], dtype=np_dtype)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,))])
        def prog(unused):
            return getattr(mb, op_name)(x=x)

        func = list(prog.functions.values())[0]
        op = [o for o in func.operations if o.op_type == op_name][0]
        folded = np.array(op.outputs[0].val)

        assert np.isfinite(folded).all(), (
            f"{op_name} folded to {folded} for {np_dtype.__name__}; "
            f"epsilon was {np.array(op.epsilon.val)}"
        )
