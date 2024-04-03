#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy

import numpy as np
import pytest

from coremltools import ComputeUnit
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.defs.lower_complex_dialect_ops import (
    _calculate_dft_matrix,
)
from coremltools.converters.mil.mil.scope import ScopeInfo, ScopeSource
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    ct_convert,
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

        prev_prog, _, block = apply_pass_and_basic_check(prog, "common::lower_complex_dialect_ops")
        assert get_op_types_in_program(prev_prog) == ["complex", "complex_real"]
        assert get_op_types_in_program(prog) == ["identity"]

        inputs = {"x": (1, 2, 3)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 3)},
        )

    def test_lower_fft_with_scope(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3))])
        def prog(x):
            with mb.scope(ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["m1"])):
                fft_res = mb.complex_fft(data=x)
            with mb.scope(ScopeInfo(source=ScopeSource.TORCHSCRIPT_MODULE_TYPE, data=["m2"])):
                return mb.complex_real(data=fft_res)
        prog._add_essential_scope_source(ScopeSource.TORCHSCRIPT_MODULE_TYPE)

        apply_pass_and_basic_check(
            prog,
            "common::lower_complex_dialect_ops",
            skip_essential_scope_check=True,  # this graph pass introduces two subgraphs, while only one of them is used.
        )
        apply_pass_and_basic_check(
            prog,
            "common::dead_code_elimination",
        )

        # since the _replace_var is operated on the output of complex_real, so the scope info should be "m2"
        block = prog.functions["main"]
        for op in block.operations:
            assert op.scopes == {
                ScopeSource.TORCHSCRIPT_MODULE_TYPE: ["m2"],
                ScopeSource.COREMLTOOLS_GRAPH_PASS: ["lower_complex_dialect_ops"],
            }

    def test_lower_fft(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3))])
        def prog(x):
            fft_res = mb.complex_fft(data=x)
            real_data = mb.complex_real(data=fft_res)
            return real_data

        # Test the apply_pass_and_basic_check utils has the ability to catch errors regarding incomplete scope information
        with pytest.raises(
            ValueError, match="is missing essential scopes ScopeSource.TORCHSCRIPT_MODULE_TYPE"
        ):
            prev_prog, _, block = apply_pass_and_basic_check(
                copy.deepcopy(prog),
                "common::lower_complex_dialect_ops",
            )

        prev_prog, _, block = apply_pass_and_basic_check(
            prog,
            "common::lower_complex_dialect_ops",
            skip_essential_scope_check=True,  # this graph pass introduces two subgraphs, while only one of them is used.
        )
        assert get_op_types_in_program(prev_prog) == ["complex_fft", "complex_real"]
        after_pass_op_types_set = set(get_op_types_in_program(prog))
        # Verifies that the complex dialect ops got lowered to core ops.
        assert "complex_fft" not in after_pass_op_types_set
        assert "complex_real" not in after_pass_op_types_set

        apply_pass_and_basic_check(
            prog,
            "common::dead_code_elimination",
        )
        # Verifies that the complex dialect ops got lowered to core ops.
        assert "complex_fft" not in after_pass_op_types_set
        assert "complex_real" not in after_pass_op_types_set

        inputs = {"x": (1, 2, 3)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 3)},
        )

    @pytest.mark.parametrize("onesided", [True, False])
    def test_calculate_dft_matrix(self, onesided):
        expected_C = np.zeros((16, 16))
        expected_S = np.zeros((16, 16))

        _range = np.arange(16)
        for k in range(16):
            expected_C[k, :] = np.cos(2 * np.pi * k * _range / 16)
            expected_S[k, :] = np.sin(2 * np.pi * k * _range / 16)

        if onesided:
            expected_C = expected_C[:9]
            expected_S = expected_S[:9]

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,))])
        def prog(x):
            return _calculate_dft_matrix(x, onesided=onesided)

        model = ct_convert(
            program=prog, convert_to=("neuralnetwork", "fp32"), compute_units=ComputeUnit.CPU_ONLY
        )
        p = model.predict({"x": np.array([16.0])})
        cos_matrix, sin_matrix = p["cos_0"], p["sin_0"]

        np.testing.assert_allclose(expected_C, cos_matrix, atol=1e-04, rtol=1e-05)
        np.testing.assert_allclose(expected_S, sin_matrix, atol=1e-04, rtol=1e-05)
