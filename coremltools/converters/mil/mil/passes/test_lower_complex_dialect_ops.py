#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    get_op_types_in_program,
)

np.random.seed(9)


class TestLowerComplexDialectOps:
    def test_lower_complex_real(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3))])
        def prog(x):
            complex_data = mb.complex(real_data=x, imag_data=x)
            real_data = mb.complex_real(data=complex_data)
            return real_data

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::lower_complex_dialect_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["complex", "complex_real"]
        assert get_op_types_in_program(prog) == ["identity"]

        inputs = {"x": (1, 2, 3)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 3)},
        )

    def test_lower_fft(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3))])
        def prog(x):
            fft_res = mb.complex_fft(data=x)
            real_data = mb.complex_real(data=fft_res)
            return real_data

        prev_prog, _, block = apply_pass_and_basic_check(
            prog, "common::lower_complex_dialect_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["complex_fft", "complex_real"]
        after_pass_op_types_set = set(get_op_types_in_program(prog))
        # Verifies that the complex dialect ops got lowered to core ops.
        assert "complex_fft" not in after_pass_op_types_set
        assert "complex_real" not in after_pass_op_types_set

        inputs = {"x": (1, 2, 3)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 3)},
        )
