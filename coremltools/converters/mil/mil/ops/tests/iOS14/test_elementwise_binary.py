#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import itertools

import numpy as np
import pytest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.ops.tests.iOS14 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import ssa_fn


class TestElementwiseBinary:
    # All in this test share the same backends
    @pytest.mark.parametrize(
        "compute_unit, backend, mode",
        itertools.product(
            compute_units,
            backends,
            [
                "add",
                "floor_div",
                "maximum",
                "minimum",
                "mod",
                "mul",
                "pow",
                "real_div",
                "sub",
            ],
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, mode):
        if mode == "add":
            x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[0, 4, 0], [8, 0, 12]], dtype=np.float32)

            build = lambda x, y: mb.add(x=x, y=y)
        elif mode == "floor_div":
            x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
            y = np.array([[11, 12, 13], [14, 15, 16]], dtype=np.float32)
            expected_outputs = np.array([[0, 1, 2], [2, 3, 3]], dtype=np.float32)

            build = lambda x, y: mb.floor_div(x=x, y=y)
        elif mode == "maximum":
            x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

            build = lambda x, y: mb.maximum(x=x, y=y)
        elif mode == "minimum":
            x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)

            build = lambda x, y: mb.minimum(x=x, y=y)
        elif mode == "mod":
            x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
            y = np.array([[11, 12, 13], [14, 15, 16]], dtype=np.float32)
            expected_outputs = np.array([[10, 8, 4], [12, 5, 12]], dtype=np.float32)

            build = lambda x, y: mb.mod(x=x, y=y)
        elif mode == "mul":
            x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[-1, 4, -9], [16, -25, 36]], dtype=np.float32)

            build = lambda x, y: mb.mul(x=x, y=y)
        elif mode == "pow":
            x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[1, 4, 0.037], [256, 0.00032, 46656]], dtype=np.float32)

            build = lambda x, y: mb.pow(x=x, y=y)
        elif mode == "real_div":
            x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
            y = np.array([[11, 12, 13], [14, 15, 16]], dtype=np.float32)
            expected_outputs = np.array(
                [[0.90909091, 1.66666667, 2.30769231], [2.85714286, 3.33333333, 3.75]],
                dtype=np.float32,
            )

            build = lambda x, y: mb.real_div(x=x, y=y)
        elif mode == "sub":
            x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[2, 0, 6], [0, 10, 0]], dtype=np.float32)

            build = lambda x, y: mb.sub(x=x, y=y)

        expected_output_types = (2, 3, types.fp32)
        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "y": mb.placeholder(shape=y.shape),
        }
        input_values = {"x": x, "y": y}
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    def test_output_dim_for_same_symbolic_dim_inputs(self):
        symbolic_input_shape = (get_new_symbol(), 4, 5)

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=symbolic_input_shape),
                mb.TensorSpec(shape=symbolic_input_shape),
            ]
        )
        def prog(x, y):
            return mb.add(x=x, y=y)

        add_op = prog.find_ops(op_type="add")[0]
        output_shape = add_op.outputs[0].shape
        if output_shape != symbolic_input_shape:
            raise AssertionError(
                "Invalid Output shape {}. Should instead be {}".format(
                    output_shape, symbolic_input_shape
                )
            )

    @ssa_fn
    def test_builder_add(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[0, 4, 0], [8, 0, 12]], dtype=np.float32)
        v = mb.add(x=x, y=y)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)

    @ssa_fn
    def test_builder_floor_div(self):
        x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
        y = np.array([[11, 12, 13], [14, 15, 16]], dtype=np.float32)
        expected_outputs = np.array([[0, 1, 2], [2, 3, 3]], dtype=np.float32)
        v = mb.floor_div(x=x, y=y)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)

    @ssa_fn
    def test_builder_maximum(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        v = mb.maximum(x=x, y=y)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)

    @ssa_fn
    def test_builder_minimum(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = mb.minimum(x=x, y=y)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)

    @ssa_fn
    def test_builder_mod(self):
        x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
        y = np.array([[11, 12, 13], [14, 15, 16]], dtype=np.float32)
        expected_outputs = np.array([[10, 8, 4], [12, 5, 12]], dtype=np.float32)
        v = mb.mod(x=x, y=y)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)

    @ssa_fn
    def test_builder_mul(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[-1, 4, -9], [16, -25, 36]], dtype=np.float32)
        v = mb.mul(x=x, y=y)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)

    @ssa_fn
    def test_builder_pow(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 4, 0.037], [256, 0.00032, 46656]], dtype=np.float32)
        v = mb.pow(x=x, y=y)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)

    @ssa_fn
    def test_builder_real_div(self):
        x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
        y = np.array([[11, 12, 13], [14, 15, 16]], dtype=np.float32)
        expected_outputs = np.array(
            [[0.90909091, 1.66666667, 2.30769231], [2.85714286, 3.33333333, 3.75]],
            dtype=np.float32,
        )
        v = mb.real_div(x=x, y=y)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)

    @ssa_fn
    def test_builder_real_div_both_ints(self):
        x = np.array([5], dtype=np.int32)
        y = np.array([2], dtype=np.int32)
        expected_outputs = np.array([2], dtype=np.int32)
        v = mb.real_div(x=x, y=y)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)
        assert isinstance(v.val[0], (float, np.int32))
        # make sure the dtype is float
        assert types.is_int(v.dtype)
        # make sure the symbolic type matches the value type
        assert v._sym_type.get_primitive() == v._sym_val.get_primitive()

    @ssa_fn
    def test_builder_sub(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[2, 0, 6], [0, 10, 0]], dtype=np.float32)
        v = mb.sub(x=x, y=y)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_real_div_int_builder_to_backend(self, compute_unit, backend):
        """
        For the neuralnetwork backend, the real_div is producing float output even for int inputs,
        while the mlprogram backend produces int type output.
        """
        x = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
        y = np.array([[11, 12, 13], [14, 15, 16]], dtype=np.float32)

        if backend.backend == "neuralnetwork":
            dtype = np.float32
        else:
            dtype = np.int32
        expected_outputs = np.array(x / y, dtype=dtype)

        build = lambda x, y: mb.real_div(x=x, y=y)

        expected_output_types = (2, 3, types.int32)
        input_placeholders = {
            "x": mb.placeholder(shape=x.shape, dtype=types.int32),
            "y": mb.placeholder(shape=y.shape, dtype=types.int32),
        }
        input_values = {"x": x, "y": y}
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestEqual:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "y": mb.placeholder(shape=y.shape),
        }
        input_values = {"x": x, "y": y}

        def build(x, y):
            return mb.equal(x=x, y=y), mb.equal(x=-3.0, y=y)

        expected_output_types = [
            (2, 3, types.bool),
            (2, 3, types.bool),
        ]
        expected_outputs = [
            np.array([[0, 1, 0], [1, 0, 1]], dtype=bool),
            np.array([[0, 0, 1], [0, 0, 0]], dtype=bool),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[0, 1, 0], [1, 0, 1]], dtype=bool)
        v = mb.equal(x=x_val, y=y_val)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)


class TestGreater:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "y": mb.placeholder(shape=y.shape),
        }
        input_values = {"x": x, "y": y}

        def build(x, y):
            return mb.greater(x=x, y=y), mb.greater(x=x, y=3.5)

        expected_output_types = [
            (2, 3, types.bool),
            (2, 3, types.bool),
        ]
        expected_outputs = [
            np.array([[1, 0, 1], [0, 1, 0]], dtype=bool),
            np.array([[0, 0, 0], [1, 1, 1]], dtype=bool),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
        v = mb.greater(x=x_val, y=y_val)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)


class TestGreaterEqual:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "y": mb.placeholder(shape=y.shape),
        }
        input_values = {"x": x, "y": y}

        def build(x, y):
            return mb.greater_equal(x=x, y=y), mb.greater_equal(x=x, y=3.5)

        expected_output_types = [
            (2, 3, types.bool),
            (2, 3, types.bool),
        ]
        expected_outputs = [
            np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
            np.array([[0, 0, 0], [1, 1, 1]], dtype=bool),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 1, 1], [1, 1, 1]], dtype=bool)
        v = mb.greater_equal(x=x_val, y=y_val)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)


class TestLess:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "y": mb.placeholder(shape=y.shape),
        }
        input_values = {"x": x, "y": y}

        def build(x, y):
            return mb.less(x=x, y=y)

        expected_output_types = (2, 3, types.bool)
        expected_outputs = np.array([[0, 0, 0], [0, 0, 0]], dtype=bool)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke2(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x.shape)}
        input_values = {"x": x}

        def build(x):
            # y is const
            y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
            return mb.less(x=x, y=y)

        expected_output_types = (2, 3, types.bool)
        expected_outputs = np.array([[0, 0, 0], [0, 0, 0]], dtype=bool)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_broadcast(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x.shape)}
        input_values = {"x": x}

        def build(x):
            # y is const
            return mb.less(x=x, y=3.5)

        expected_output_types = (2, 3, types.bool)
        expected_outputs = np.array([[1, 1, 1], [0, 0, 0]], dtype=bool)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[0, 0, 0], [0, 0, 0]], dtype=bool)
        v = mb.less(x=x_val, y=y_val)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)


class TestLessEqual:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "y": mb.placeholder(shape=y.shape),
        }
        input_values = {"x": x, "y": y}

        def build(x, y):
            return mb.less_equal(x=x, y=y)

        expected_output_types = (2, 3, types.bool)
        expected_outputs = np.array([[0, 1, 0], [1, 0, 1]], dtype=bool)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[0, 1, 0], [1, 0, 1]], dtype=bool)
        v = mb.less_equal(x=x_val, y=y_val)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)


class TestNotEqual:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "y": mb.placeholder(shape=y.shape),
        }
        input_values = {"x": x, "y": y}

        def build(x, y):
            return mb.not_equal(x=x, y=y)

        expected_output_types = (2, 3, types.bool)
        expected_outputs = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
        v = mb.not_equal(x=x_val, y=y_val)
        np.testing.assert_allclose(expected_outputs, v.val, atol=1e-04, rtol=1e-05)
