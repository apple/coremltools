#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools._deps import _HAS_TORCH, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.ops.tests.iOS14 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import (
    UNK_SYM,
    UNK_VARIADIC,
    construct_inputs_from_placeholders,
    run_compare_builder,
)
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import ssa_fn

if _HAS_TORCH:
    import torch


class TestDepthToSpace:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        # original input type is (1, 4, 1, 1, fp32)
        val = np.array([[[[9.0]], [[5.0]], [[1.0]], [[3.0]]]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return [mb.depth_to_space(x=x, block_size=2)]

        expected_output_types = (1, 1, 2, 2, types.fp32)
        expected_outputs = np.array([[[[9.0, 5.0], [1.0, 3.0]]]], dtype=np.float32)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestSpaceToBatch:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        # original input type is (2, 1, 2, 4, fp32)
        val = np.array(
            [[[[1, 2, 3, 4], [5, 6, 7, 8]]], [[[9, 10, 11, 12], [13, 14, 15, 16]]]],
            dtype=np.float32,
        )
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return [mb.space_to_batch(x=x, block_shape=[2, 2], paddings=[[0, 0], [2, 0]])]

        expected_output_types = (8, 1, 1, 3, types.fp32)
        expected_outputs = np.array(
            [
                [[[0, 1, 3]]],
                [[[0, 9, 11]]],
                [[[0, 2, 4]]],
                [[[0, 10, 12]]],
                [[[0, 5, 7]]],
                [[[0, 13, 15]]],
                [[[0, 6, 8]]],
                [[[0, 14, 16]]],
            ],
            dtype=np.float32,
        )

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestBatchToSpace:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        # original input type is (8, 1, 1, 3, fp32)
        val = np.array(
            [
                [[[0, 1, 3]]],
                [[[0, 9, 11]]],
                [[[0, 2, 4]]],
                [[[0, 10, 12]]],
                [[[0, 5, 7]]],
                [[[0, 13, 15]]],
                [[[0, 6, 8]]],
                [[[0, 14, 16]]],
            ],
            dtype=np.float32,
        )
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return [mb.batch_to_space(x=x, block_shape=[2, 2], crops=[[0, 0], [2, 0]])]

        expected_output_types = (2, 1, 2, 4, types.fp32)
        expected_outputs = np.array(
            [[[[1, 2, 3, 4], [5, 6, 7, 8]]], [[[9, 10, 11, 12], [13, 14, 15, 16]]]],
            dtype=np.float32,
        )

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestExpandDims:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return [
                mb.expand_dims(x=x, axes=[0]),
                mb.expand_dims(x=x, axes=[1]),
                mb.expand_dims(x=x, axes=[2]),
                mb.expand_dims(x=x, axes=[-1]),
                mb.expand_dims(x=x, axes=[0, 1]),
                mb.expand_dims(x=x, axes=[-2, -1]),
            ]

        expected_output_types = [
            (1, 2, 3, types.fp32),
            (2, 1, 3, types.fp32),
            (2, 3, 1, types.fp32),
            (2, 3, 1, types.fp32),
            (1, 1, 2, 3, types.fp32),
            (2, 3, 1, 1, types.fp32),
        ]
        expected_outputs = [
            np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32),
            np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.float32),
            np.array([[[1], [2], [3]], [[4], [5], [6]]], dtype=np.float32),
            np.array([[[1], [2], [3]], [[4], [5], [6]]], dtype=np.float32),
            np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.float32),
            np.array([[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]]], dtype=np.float32),
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

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_symbolic(self, compute_unit, backend):
        s0 = get_new_symbol()

        input_placeholders = {
            "x": mb.placeholder(shape=(2, s0)),
        }

        def build(x):
            return [
                mb.expand_dims(x=x, axes=[-1]),
                mb.expand_dims(x=x, axes=[1]),
            ]

        expected_output_types = [
            (2, s0, 1, types.fp32),
            (2, 1, s0, types.fp32),
        ]
        expected_outputs = [
            np.array([[[1], [2], [3]], [[4], [5], [6]]], dtype=np.float32),
            np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.float32),
        ]

        input_values = {
            "x": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        }
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, 10)
            if backend.backend == "mlprogram"
            else None,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.random.rand(1, 6)
        v1 = mb.expand_dims(x=x_val, axes=[2])
        np.testing.assert_allclose(np.expand_dims(x_val, 2), v1.val, atol=1e-04, rtol=1e-05)

        v2 = mb.expand_dims(x=x_val, axes=[-1])
        np.testing.assert_allclose(np.expand_dims(x_val, -1), v2.val, atol=1e-04, rtol=1e-05)

        v3 = mb.expand_dims(x=x_val, axes=[-1, -2])
        ref = np.expand_dims(np.expand_dims(x_val, -1), -1)
        np.testing.assert_allclose(ref, v3.val, atol=1e-04, rtol=1e-05)

        v4 = mb.expand_dims(x=x_val, axes=[0, -1, -2])
        np.testing.assert_allclose(
            np.reshape(x_val, (1, 1, 6, 1, 1)), v4.val, atol=1e-04, rtol=1e-05
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank_and_axis",
        itertools.product(
            compute_units,
            backends,
            [(rank, axis) for rank in range(1, 5) for axis in range(-rank - 1, rank + 1)],
        ),
    )
    def test_builder_to_backend_programmatic_one_axis(self, compute_unit, backend, rank_and_axis):
        rank, axis = rank_and_axis
        x_shape = np.random.randint(low=2, high=6, size=rank)
        input_placeholders = {"x": mb.placeholder(shape=x_shape)}
        input_values = {"x": np.random.sample(x_shape).astype(np.float32)}

        def build(x):
            return mb.expand_dims(x=x, axes=[axis])

        adjusted_axis = axis if axis >= 0 else rank + axis + 1
        x_shape = list(x_shape)
        out_shape = x_shape[:adjusted_axis] + [1] + x_shape[adjusted_axis:]
        expected_output_types = tuple(out_shape[:]) + (types.fp32,)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            np.expand_dims(input_values["x"], axis),
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank_and_axes",
        itertools.product(
            compute_units,
            backends,
            [
                (3, [0, 1]),
                (3, [1, 0]),
                (3, [-2, -1]),
                (3, [-1, -2]),
                (2, [-3, -1]),
                (2, [-3, 1, -1]),
                (2, [-2, 0]),
                (1, [-1, -2, -3, -4]),
                (1, [0, -1]),
                (1, [0, 1, -2, -1]),
            ],
        ),
    )
    def test_builder_to_backend_programmatic_multiple_axes(
        self, compute_unit, backend, rank_and_axes
    ):
        rank, axes = rank_and_axes
        x_shape = np.random.randint(low=1, high=6, size=rank)
        input_placeholders = {"x": mb.placeholder(shape=x_shape)}
        input_values = {"x": np.random.sample(x_shape).astype(np.float32)}

        def build(x):
            return mb.expand_dims(x=x, axes=axes)

        out_shape = list(x_shape)
        out_rank = rank + len(axes)
        pos_axes = sorted([out_rank + axis if axis < 0 else axis for axis in axes])
        for axis in pos_axes:
            out_shape.insert(axis, 1)

        expected_outputs = np.reshape(input_values["x"], out_shape)
        expected_output_types = tuple(out_shape) + (types.fp32,)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestReshape:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return [
                mb.reshape(x=x, shape=[3, 2]),
                mb.reshape(x=x, shape=[2, -1]),
                mb.reshape(x=x, shape=[2, 1, 1, 3]),
            ]

        expected_output_types = [
            (3, 2, types.fp32),
            (2, 3, types.fp32),
            (2, 1, 1, 3, types.fp32),
        ]
        expected_outputs = [
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.array([[[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]]], dtype=np.float32),
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
        t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        r = mb.reshape(x=t, shape=[3, 2])
        expected_r = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        np.testing.assert_allclose(expected_r, r.val, atol=1e-04, rtol=1e-05)
        r2 = mb.reshape(x=t, shape=[2, -1])
        expected_r2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        np.testing.assert_allclose(expected_r2, r2.val, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_symbolic(self, compute_unit, backend):
        s0 = get_new_symbol()
        s_len = get_new_symbol()

        input_placeholders = {
            "x": mb.placeholder(shape=(2, s0)),
            "shape": mb.placeholder(shape=(3,), dtype=types.int32),
            "shape2": mb.placeholder(shape=(s_len,), dtype=types.int32),
        }

        def build(x, shape, shape2):
            return [
                mb.reshape(x=x, shape=[2, -1]),
                mb.reshape(x=x, shape=[1, -1]),
                mb.reshape(x=x, shape=[2, 1, 1, -1]),
                mb.reshape(x=x, shape=shape),
                mb.reshape(x=x, shape=shape2),
            ]

        expected_output_types = [
            (2, s0, types.fp32),
            (1, 2 * s0, types.fp32),
            (2, 1, 1, s0, types.fp32),
            (UNK_SYM, UNK_SYM, UNK_SYM, types.fp32),
            (UNK_VARIADIC, types.fp32),
        ]
        expected_outputs = [
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32),
            np.array([[[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]]], dtype=np.float32),
            np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.float32),
            np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.float32),
        ]

        input_values = {
            "x": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            "shape": np.array([2, 1, 3], dtype=np.float32),
            "shape2": np.array([2, 1, 3], dtype=np.float32),
        }

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, 10)
            if backend.backend == "mlprogram"
            else None,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_too_many_neg_ones(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        with pytest.raises(ValueError, match="Reshape op supports only one dimension to be -1"):
            mb.reshape(x=x, shape=[-1, -1])

    @ssa_fn
    def test_invalid_target_shape(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        with pytest.raises(ValueError, match="Invalid target shape in `reshape` op"):
            mb.reshape(x=x, shape=[4, -1])

    @ssa_fn
    def test_invalid_target_shape_with_zero(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        with pytest.raises(ValueError, match="Invalid target shape in `reshape` op"):
            mb.reshape(x=x, shape=[0, 7])

    @staticmethod
    def test_value_inference_with_symbolic_values():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(get_new_symbol(), get_new_symbol()), dtype=types.fp32)
            ]
        )
        def prog(x):
            shape = mb.shape(x=x)
            res = mb.reshape(x=shape, shape=(1, 2))
            res_sym_val = res.sym_val
            assert res_sym_val is not None
            assert res_sym_val.shape == (1, 2)
            assert res_sym_val[0][0] == shape.sym_val[0]
            assert res_sym_val[0][1] == shape.sym_val[1]
            return res

class TestReverse:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        val = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return [mb.reverse(x=x), mb.reverse(x=x, axes=[0])]

        expected_output_types = [(2, 3, types.fp32), (2, 3, types.fp32)]
        expected_outputs = [
            np.array([[6.0, -5.0, 4.0], [-3.0, 2.0, -1.0]], dtype=np.float32),
            np.array([[4.0, -5.0, 6.0], [-1.0, 2.0, -3.0]], dtype=np.float32),
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
        val = np.array([[-1.0, 7.0, -3.0], [4.0, -5.0, 8.0]], dtype=np.float32)
        res = mb.reverse(x=val, axes=[0])
        np.testing.assert_allclose(np.flip(val, axis=0), res.val, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_symbolic(self, compute_unit, backend):
        s0 = get_new_symbol()

        val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=(s0, 3))}
        input_values = {"x": val}

        def build(x):
            return [
                mb.reverse(x=x, axes=[1]),
                mb.reverse(x=x, axes=[0]),
            ]

        expected_output_types = [
            (s0, 3, types.fp32),
            (s0, 3, types.fp32),
        ]
        expected_outputs = [
            np.array([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]], dtype=np.float32),
            np.array([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], dtype=np.float32),
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


class TestReverseSequence:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x_val = np.array(
            [
                [1, 2, 3, 4, 5, 0, 0, 0],
                [1, 2, 0, 0, 0, 0, 0, 0],
                [1, 2, 3, 4, 0, 0, 0, 0],
                [1, 2, 3, 4, 5, 6, 7, 8],
            ],
            dtype=np.float32,
        )
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        input_values = {"x": x_val}

        def build(x):
            return [
                mb.reverse_sequence(x=x, lengths=[7, 2, 3, 5], seq_axis=1, batch_axis=0),
            ]

        expected_output_types = [
            (4, 8, types.fp32),
        ]
        expected_outputs = [
            np.array(
                [
                    [0, 0, 5, 4, 3, 2, 1, 0],
                    [2, 1, 0, 0, 0, 0, 0, 0],
                    [3, 2, 1, 4, 0, 0, 0, 0],
                    [5, 4, 3, 2, 1, 6, 7, 8],
                ],
                dtype=np.float32,
            )
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

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_symbolic(self, compute_unit, backend):
        s0 = get_new_symbol()

        x_val = np.array(
            [
                [1, 2, 3, 4, 5, 0, 0, 0],
                [1, 2, 0, 0, 0, 0, 0, 0],
                [1, 2, 3, 4, 0, 0, 0, 0],
                [1, 2, 3, 4, 5, 6, 7, 8],
            ],
            dtype=np.float32,
        )
        input_placeholders = {"x": mb.placeholder(shape=(4, s0))}
        input_values = {"x": x_val}

        def build(x):
            return [
                mb.reverse_sequence(x=x, lengths=[7, 2, 3, 5], seq_axis=1, batch_axis=0),
            ]

        expected_output_types = [
            (4, s0, types.fp32),
        ]
        expected_outputs = [
            np.array(
                [
                    [0, 0, 5, 4, 3, 2, 1, 0],
                    [2, 1, 0, 0, 0, 0, 0, 0],
                    [3, 2, 1, 4, 0, 0, 0, 0],
                    [5, 4, 3, 2, 1, 6, 7, 8],
                ],
                dtype=np.float32,
            )
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, 10)
            if backend.backend == "mlprogram"
            else None,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestSliceByIndex:
    @pytest.mark.parametrize(
        "compute_unit, backend, x_dtype, idx_dtype",
        itertools.product(
            compute_units,
            backends,
            (np.float16, np.float32, np.int32),
            (np.int32,),
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, x_dtype, idx_dtype):
        x_builtin_dtype = types.numpy_type_to_builtin_type(x_dtype)
        idx_builtin_dtype = types.numpy_type_to_builtin_type(idx_dtype)

        x_val = np.array(list(range(24))).reshape((2, 3, 4)).astype(x_dtype)
        begin_val = np.array([1, 1, 1], dtype=idx_dtype)
        end_val = np.array([2, 3, 3], dtype=idx_dtype)
        input_placeholders = {
            "x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype),
            "begin": mb.placeholder(shape=begin_val.shape, dtype=idx_builtin_dtype),
            "end": mb.placeholder(shape=end_val.shape, dtype=idx_builtin_dtype),
        }
        input_values = {"x": x_val, "begin": begin_val, "end": end_val}

        def build(x, begin, end):
            begin_c = mb.const(val=begin_val)
            end_c = mb.const(val=end_val)
            return [
                mb.slice_by_index(x=x, begin=begin, end=end),
                mb.slice_by_index(x=x, begin=begin_c, end=end_c),
            ]

        expected_output_types = [(UNK_SYM, UNK_SYM, UNK_SYM, x_builtin_dtype)] * 2
        expected_outputs = [np.array([[[17, 18], [21, 22]]], dtype=x_dtype)] * 2
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    def test_type_inference(self):
        s0 = get_new_symbol()
        s1 = get_new_symbol()
        s2 = get_new_symbol()

        input_placeholders = {
            "x": mb.placeholder(shape=(10, s0, s1, s2)),
        }

        def build(x):
            return [
                mb.slice_by_index(
                    x=x, begin=[2, 5, 6, 12], end=[6, 9, 20, -9], stride=[2, 1, 2, 1]
                ),
                mb.slice_by_index(
                    x=x,
                    begin=[-2, -5, -3, 9],
                    end=[-6, -9, -6, -7],
                    stride=[-2, -1, -2, 1],
                ),
                mb.slice_by_index(
                    x=x,
                    begin=[0, 0, 0, 0],
                    end=[-6, -9, 3, -2],
                    stride=[-2, -3, 1, 2],
                    begin_mask=[True, True, True, True],
                    end_mask=[False, False, False, False],
                ),
                mb.slice_by_index(
                    x=x,
                    begin=[-2, 5, -1, -7],
                    end=[0, 0, 0, 0],
                    stride=[-2, -3, 1, -2],
                    begin_mask=[False, False, False, False],
                    end_mask=[True, True, True, True],
                ),
                mb.slice_by_index(
                    x=x, begin=[4, -1, 0, -5], end=[4, -1, 0, -5], stride=[1, -1, 2, -2]
                ),
                mb.slice_by_index(
                    x=x,
                    begin=[0, -1, 0, 2],
                    end=[2, 0, 0, 2],
                    begin_mask=[False, False, False, False],
                    end_mask=[False, True, True, False],
                    stride=[1, 2, -2, 1],
                ),
                mb.slice_by_index(
                    x=x,
                    begin=[0, 2, -3, 0],
                    end=[1, 3, -4, 4],
                    begin_mask=[False, False, False, False],
                    end_mask=[False, False, False, False],
                    stride=[1, 1, -1, 1],
                ),
            ]

        expected_output_types = [
            (2, UNK_SYM, UNK_SYM, UNK_SYM, types.fp32),
            (2, UNK_SYM, UNK_SYM, UNK_SYM, types.fp32),
            (3, UNK_SYM, UNK_SYM, UNK_SYM, types.fp32),
            (5, UNK_SYM, 1, UNK_SYM, types.fp32),
            (0, 0, 0, 0, types.fp32),
            (2, 1, 1, 0, types.fp32),
            (1, 1, 1, UNK_SYM, types.fp32),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            expected_output_types=expected_output_types,
            frontend_only=True,
        )

    @pytest.mark.xfail(reason="rdar://99664032")
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_single_element_edge_case(self, compute_unit, backend):
        x_val = np.array(list(range(6))).reshape((1, 3, 2)).astype(np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=x_val.shape),
        }
        input_values = {"x": x_val}

        def build(x):
            return mb.slice_by_index(
                x=x,
                begin=[-1, 0, 0],
                end=[-2, 0, 0],
                stride=[-1, 1, 1],
                begin_mask=[False, True, True],
                end_mask=[False, True, True],
            )

        expected_output_types = [(1, 3, 2, types.fp32)]
        expected_outputs = [np.array([[[0, 1], [2, 3], [4, 5]]], dtype=np.float32)]
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
    def test_builder_eval_scalar_output_corner_cases(self):
        x1 = np.array([2.0])
        x2 = np.array([[[[1.0], [3.0]]]])
        v = [
            mb.slice_by_index(
                x=x1,
                begin=[
                    0,
                ],
                end=[0],
                squeeze_mask=[True],
            ),
            mb.slice_by_index(
                x=x2,
                begin=[0, 0, 0, 0],
                end=[0, 0, 0, 0],
                squeeze_mask=[True, True, True, True],
            ),
        ]
        assert v[0].val.shape == ()
        assert v[0].val == 2
        assert v[1].val.shape == ()
        assert v[1].val == 1

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array(list(range(24))).reshape((2, 3, 4))
        v = [
            mb.slice_by_index(x=x_val, begin=[1, 1, 1], end=[2, 2, 2]),  # x_val[1:2, 1:2, 1:2]
            mb.slice_by_index(
                x=x_val, begin=[1, 1, 1], end=[2, 3, 4], stride=[1, 1, 2]
            ),  #  x_val[1:2, 1:3, 1:4:2]
            mb.slice_by_index(
                x=x_val, begin=[-3, -3, -3], end=[-1, -1, -1]
            ),  # x_val[-3:-1, -3:-1, -3:-1]
            mb.slice_by_index(
                x=x_val, begin=[0, 0, -3], end=[-1, -2, -2]
            ),  # x_val[0:-1, 0:-2, -3:-2]
            mb.slice_by_index(
                x=x_val, begin=[-1, -1, -1], end=[0, 1, -3], stride=[-2, -1, -3]
            ),  # x_val[-1:0:-2, -1:1:-1, -1:-3:-3]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 1, 1],
                end=[2, 3, 4],
                stride=[1, 1, 2],
                begin_mask=[True, False, True],
            ),  # x_val[:2, 1:3, :4:2]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 1, 1],
                end=[2, 3, 4],
                stride=[1, 1, 2],
                begin_mask=[True, False, True],
                end_mask=[True, True, False],
            ),  # x_val[:, 1:, :4:2]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 1, 1],
                end=[2, 3, 4],
                stride=[1, 1, 2],
                begin_mask=[False, False, True],
                end_mask=[True, False, False],
                squeeze_mask=[False, True, False],
            ),  # x_val[1::1, 1, :3:2]
            mb.slice_by_index(
                x=x_val,
                begin=[0, 0, 0],
                end=[0, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[True, True, True],
                end_mask=[True, True, True],
            ),  # x_val[:, :, :]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 1, 1],
                end=[2, 2, 0],
                stride=[1, 1, 1],
                squeeze_mask=[False, False, True],
            ),  # x_val[1:2, 1:2, 1]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 0, 0],
                end=[2, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[False, True, True],
                end_mask=[False, True, True],
            ),  # x_val[1:2, ...]
            mb.slice_by_index(
                x=x_val,
                begin=[0, 0, 0],
                end=[0, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[True, True, True],
                end_mask=[True, True, True],
            ),  # x_val[...]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 0, 1],
                end=[2, 0, 2],
                stride=[1, 1, 1],
                begin_mask=[False, True, False],
                end_mask=[False, True, False],
            ),  # x_val[1:2, ..., 1:2]
            mb.slice_by_index(
                x=x_val,
                begin=[0, 0, 1],
                end=[0, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[True, True, False],
                end_mask=[True, True, False],
                squeeze_mask=[False, False, True],
            ),  # x_val[..., 1]
            mb.slice_by_index(
                x=x_val,
                begin=[0, 0, 0],
                end=[0, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[False, False, True],
                end_mask=[False, False, True],
                squeeze_mask=[True, True, False],
            ),  # x_val[0, 0, :]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 0, 0],
                end=[2, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[False, True, True],
                end_mask=[False, True, True],
            ),  # x_val[1:2]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 1, 0],
                end=[2, 2, 0],
                stride=[1, 1, 1],
                begin_mask=[False, False, True],
                end_mask=[False, False, True],
            ),  # x_val[1:2, 1:2]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 0, 0],
                end=[0, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[False, True, True],
                end_mask=[False, True, True],
                squeeze_mask=[True, False, False],
            ),  # x_val[1]
            mb.slice_by_index(
                x=x_val,
                begin=[0, 0, 0],
                end=[0, 0, 0],
                begin_mask=[True, True, True],
                end_mask=[True, True, True],
            ),  # x_val[:]
            mb.slice_by_index(
                x=x_val,
                begin=[0, 0, 0],
                end=[0, 0, 0],
                stride=[1, 1, -1],
                begin_mask=[True, True, True],
                end_mask=[True, True, True],
            ),  # x_val[..., ::-1]
        ]
        ans = [
            x_val[1:2, 1:2, 1:2],
            x_val[1:2, 1:3, 1:4:2],
            x_val[-3:-1, -3:-1, -3:-1],
            x_val[0:-1, 0:-2, -3:-2],
            x_val[-1:0:-2, -1:1:-1, -1:-3:-3],
            x_val[:2, 1:3, :4:2],
            x_val[:, 1:, :4:2],
            x_val[1::1, 1, :3:2],
            x_val[:, :, :],
            x_val[1:2, 1:2, 1],
            x_val[1:2, ...],
            x_val[...],
            x_val[1:2, ..., 1:2],
            x_val[..., 1],
            x_val[0, 0, :],
            x_val[1:2],
            x_val[1:2, 1:2],
            x_val[1],
            x_val[:],
            x_val[..., ::-1],
        ]
        for idx in range(len(v)):
            assert ans[idx].shape == v[idx].shape
            np.testing.assert_allclose(ans[idx], v[idx].val, atol=1e-04, rtol=1e-05)

    @staticmethod
    @pytest.mark.skipif(ct.utils._macos_version() < (14, 0), reason="Bug fixed in macOS 14")
    def test_slice_by_index():
        INPUT_SHAPE = (1, 2, 8, 16)

        @mb.program(input_specs=[mb.TensorSpec(shape=INPUT_SHAPE)])
        def prog(x):
            x = mb.slice_by_index(
                x=x,
                begin=[0, 0, 0, 0],
                end=[1, 2, 8, 12],
                stride=[1, 1, 2, 2],
                begin_mask=None,
                end_mask=None,
                squeeze_mask=None,
            )
            return x

        x = np.random.rand(*INPUT_SHAPE)

        # slice by index is x[begin[0]: end[0]: stride[0], begin[1]: end[1]: stride[1], ...]
        y_numpy = x[0:1:1, 0:2:1, 0:8:2, 0:12:2]

        model = ct.convert(prog, source="milinternal", convert_to="neuralnetwork")
        y_neuralnetwork = list(model.predict({"x": x}).values())[0]
        np.testing.assert_allclose(y_numpy, y_neuralnetwork)

        model = ct.convert(
            prog,
            source="milinternal",
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )

        y_mlprogram = list(model.predict({"x": x}).values())[0]
        assert y_numpy.shape == y_mlprogram.shape
        np.testing.assert_allclose(y_numpy, y_mlprogram)

    @staticmethod
    @pytest.mark.skipif(ct.utils._macos_version() < (14, 0), reason="Bug fixed in macOS 14")
    def test_slice_by_index_slice_squeeze_separate():
        INPUT_SHAPE = (1, 2, 8, 16)

        @mb.program(input_specs=[mb.TensorSpec(shape=INPUT_SHAPE)])
        def prog(x):
            x = mb.slice_by_index(
                x=x,
                begin=[0, 0, 0, 0],
                end=[1, 2, 8, 12],
                stride=[1, 1, 1, 2],
                begin_mask=None,
                end_mask=None,
                squeeze_mask=[True, False, False, False],
            )
            return x

        x = np.random.rand(*INPUT_SHAPE)

        # slice by index is x[begin[0]: end[0]: stride[0], begin[1]: end[1]: stride[1], ...]
        # and squeeze dim 0
        y_numpy = x[0:1:1, 0:2:1, 0:8:1, 0:12:2]
        y_numpy = np.squeeze(y_numpy, axis=0)

        model = ct.convert(prog, source="milinternal", convert_to="neuralnetwork")
        y_neuralnetwork = list(model.predict({"x": x}).values())[0]

        assert y_numpy.shape == y_neuralnetwork.shape
        np.testing.assert_allclose(y_numpy, y_neuralnetwork)

        model = ct.convert(prog, source="milinternal", convert_to="mlprogram")
        y_mlprogram = list(model.predict({"x": x}).values())[0]
        # TODO: rdar://103365766 MLProgram does not apply squeeze_mask.
        # np.testing.assert_allclose(y_numpy, y_mlprogram)


class TestSliceBySize:
    @pytest.mark.parametrize(
        "compute_unit, backend, size_val, x_dtype, idx_dtype",
        itertools.product(
            compute_units,
            backends,
            ([1, 2, 3], [-1, 2, -1]),
            (np.float16, np.float32, np.int32),
            (np.int32,),
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, size_val, x_dtype, idx_dtype):
        def build(x, begin):
            return mb.slice_by_size(x=x, begin=begin, size=np.array(size_val, dtype=idx_dtype))

        x_builtin_dtype = types.numpy_type_to_builtin_type(x_dtype)
        idx_builtin_dtype = types.numpy_type_to_builtin_type(idx_dtype)

        x_val = np.array(list(range(24))).reshape((2, 3, 4)).astype(x_dtype)
        begin_val = np.array([1, 1, 1], dtype=idx_dtype)
        input_placeholders = {
            "x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype),
            "begin": mb.placeholder(shape=begin_val.shape, dtype=idx_builtin_dtype),
        }
        input_values = {"x": x_val, "begin": begin_val}

        expected_outputs = np.array([[[17, 18, 19], [21, 22, 23]]], dtype=x_dtype)
        expected_output_types = tuple([dim if dim != -1 else UNK_SYM for dim in size_val]) + (
            x_builtin_dtype,
        )

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
        x = np.array(list(range(24))).reshape(2, 3, 4)
        v_1 = mb.slice_by_size(x=x, begin=(0, 1, 0), size=(-1, -1, -1))
        v_2 = mb.slice_by_size(x=x, begin=(0, 1, 0), size=(-1, -1, 3))
        v_3 = mb.slice_by_size(x=x, begin=(0, -2, 0), size=(-1, -1, 3))
        np.testing.assert_allclose(x[:, 1:, :], v_1.val, atol=1e-04, rtol=1e-05)
        np.testing.assert_allclose(x[:, 1:, :3], v_2.val, atol=1e-04, rtol=1e-05)
        np.testing.assert_allclose(x[:, -2:, :3], v_3.val, atol=1e-04, rtol=1e-05)


class TestSpaceToDepth:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        # original input type is (1, 1, 2, 2, fp32)
        val = np.array([[[[7.0, 9.0], [4.0, 6.0]]]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return [mb.space_to_depth(x=x, block_size=2)]

        expected_output_types = (1, 4, 1, 1, types.fp32)
        expected_outputs = np.array([[[[7.0]], [[9.0]], [[4.0]], [[6.0]]]], dtype=np.float32)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestSqueeze:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([[[[1], [2], [3]]]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x.shape)}

        input_values = {"x": x}

        def build(x):
            return [
                mb.squeeze(x=x, axes=(-1,)),
                mb.squeeze(x=x, axes=(-3, 0)),
                mb.squeeze(x=x, axes=(0, 1, 3)),
                mb.squeeze(x=x),
            ]

        expected_output_types = [
            (1, 1, 3, types.fp32),
            (3, 1, types.fp32),
            (3, types.fp32),
            (3, types.fp32),
        ]

        expected_outputs = [
            np.array([[[1, 2, 3]]], dtype=np.float32),
            np.array([[1], [2], [3]], dtype=np.float32),
            np.array([1, 2, 3], dtype=np.float32),
            np.array([1, 2, 3], dtype=np.float32),
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
        x = np.array([[[[1], [2], [3]], [[4], [5], [6]]]], dtype=np.float32)
        v = mb.squeeze(x=x, axes=(-4, 3))
        np.testing.assert_allclose(np.squeeze(x, axis=(-4, 3)), v.val, atol=1e-04, rtol=1e-05)

    @ssa_fn
    def test_builder_eval_rank_0(self):
        x = np.array([1], dtype=np.float32)
        v = mb.squeeze(x=x)
        assert v.shape == ()
        assert type(v.val) == np.float32
        assert np.isclose(np.squeeze(x), v.val)


class TestTranspose:
    @pytest.mark.parametrize(
        "compute_unit, backend, is_symbolic",
        itertools.product(
            compute_units,
            backends,
            [True, False],
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, is_symbolic):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        input_shape = x.shape
        if is_symbolic:
            input_shape = [get_new_symbol(), get_new_symbol()]

        input_placeholders = {"x": mb.placeholder(shape=input_shape)}

        input_values = {"x": x}

        def build(x):
            return [
                mb.transpose(x=x, perm=(0, 1)),
                mb.transpose(x=x, perm=(1, 0)),
                mb.transpose(x=x, perm=(-1, 0)),
                mb.transpose(x=x, perm=(-2, -1)),
            ]

        d0 = input_shape[0]
        d1 = input_shape[1]
        expected_output_types = [
            (d0, d1, types.fp32),
            (d1, d0, types.fp32),
            (d1, d0, types.fp32),
            (d0, d1, types.fp32),
        ]

        expected_outputs = [x, x.T, x.T, x]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, 10)
            if backend.backend == "mlprogram"
            else None,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        v = mb.transpose(x=x, perm=(1, 0))
        np.testing.assert_allclose(x.T, v.val, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_symbolic(self, compute_unit, backend):
        s0 = get_new_symbol()

        input_placeholders = {
            "x": mb.placeholder(shape=(2, s0)),
        }

        def build(x):
            return [
                mb.transpose(x=x, perm=[1, 0]),
            ]

        expected_output_types = [
            (s0, 2, types.fp32),
        ]
        expected_outputs = [
            np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32),
        ]

        input_values = {
            "x": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        }

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, 10)
            if backend.backend == "mlprogram"
            else None,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestPixelShuffle:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        # original input type is (1, 4, 1, 1, fp32)
        val = np.array([[[[9.0]], [[5.0]], [[1.0]], [[3.0]]]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return [mb.pixel_shuffle(x=x, upscale_factor=2)]

        expected_output_types = (1, 1, 2, 2, types.fp32)
        expected_outputs = np.array([[[[9.0, 5.0], [1.0, 3.0]]]], dtype=np.float32)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.skipif(not testing_reqs._HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        "compute_unit, backend, shape, upscale_factor",
        itertools.product(
            compute_units,
            backends,
            [(1, 16, 1, 1), (2, 16, 3, 3), (1, 32, 1, 1)],
            [2, 4],
        ),
    )
    def test_builder_to_backend_stress(self, compute_unit, backend, shape, upscale_factor):
        val = np.random.rand(*shape)
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return [mb.pixel_shuffle(x=x, upscale_factor=upscale_factor)]

        torch_pixel_shuffle = torch.nn.PixelShuffle(upscale_factor)
        expected_outputs = [torch_pixel_shuffle(torch.Tensor(val)).numpy()]
        expected_output_types = [o.shape[:] + (types.fp32,) for o in expected_outputs]
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestSlidingWindows:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        # original input type is (1, 4, 1, 1, fp32)
        val = np.array([[[[9.0]], [[5.0]], [[1.0]], [[3.0]]]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return [mb.sliding_windows(x=x, axis=1, size=2)]

        expected_output_types = (1, 3, 2, 1, 1, types.fp32)
        expected_outputs = np.array(
            [[[[[9.0]], [[5.0]]], [[[5.0]], [[1.0]]], [[[1.0]], [[3.0]]]]],
            dtype=np.float32,
        )

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
        "compute_unit, backend, rank_and_axis, size, stride",
        itertools.product(
            compute_units,
            backends,
            [(rank, axis) for rank in range(1, 5) for axis in range(-rank, rank)],
            [1, 2],
            [1, 2],
        ),
    )
    def test_builder_to_backend_stress(self, compute_unit, backend, rank_and_axis, size, stride):
        def np_sliding_windows(a, np_axis, np_size, np_stride):
            n = (a.shape[np_axis] - np_size) // np_stride + 1
            x_shape = list(a.shape)
            x_shape[np_axis] = n
            if np_axis < 0:
                np_axis += len(x_shape)
            x_shape.insert(np_axis + 1, np_size)
            strides = list(a.strides)
            eff_stride = strides[np_axis] * np_stride
            strides.insert(np_axis, eff_stride)
            return np.lib.stride_tricks.as_strided(a, x_shape, strides)

        rank, axis = rank_and_axis
        shape = np.random.randint(low=2, high=5, size=rank)
        val = np.random.rand(*shape)
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return [mb.sliding_windows(x=x, axis=axis, size=size, stride=stride)]

        expected_outputs = [np_sliding_windows(val, np_axis=axis, np_size=size, np_stride=stride)]
        expected_output_types = [o.shape[:] + (types.fp32,) for o in expected_outputs]
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestConcat:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        t1 = np.array([[1, 2], [4, 5]], dtype=np.float32)
        t2 = np.array([[7, 8]], dtype=np.float32)

        input_placeholders = {
            "x": mb.placeholder(shape=t1.shape),
            "y": mb.placeholder(shape=t2.shape),
        }
        input_values = {"x": t1, "y": t2}

        def build(x, y):
            return (mb.concat(values=(x, y), axis=0),)

        expected_output_types = [
            (3, 2, types.fp32),
        ]
        expected_outputs = [
            np.array([[1, 2], [4, 5], [7, 8]], dtype=np.float32),
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

    @pytest.mark.parametrize(
        "compute_unit, backend, rank, n_inputs, negative_index",
        itertools.product(
            compute_units,
            backends,
            [1, 2, 3, 4, 5],
            [2, 3],
            [False, True],
        ),
    )
    def test_builder_to_backend_stress_interleave(
        self, compute_unit, backend, rank, n_inputs, negative_index
    ):
        def np_concat_interleave(arrays, axis):
            step = len(arrays)
            in_shape = arrays[0].shape
            out_shape = list(in_shape)
            if axis < 0:
                axis += len(in_shape)
            out_shape[axis] = step * in_shape[axis]
            concat_tensor = np.empty(tuple(out_shape), dtype=np.float32)
            for i in range(step):
                if rank == 5:
                    if axis == 4:
                        concat_tensor[:, :, :, :, i::step] = arrays[i]
                    if axis == 3:
                        concat_tensor[:, :, :, i::step, :] = arrays[i]
                    if axis == 2:
                        concat_tensor[:, :, i::step, :, :] = arrays[i]
                    if axis == 1:
                        concat_tensor[:, i::step, :, :, :] = arrays[i]
                    if axis == 0:
                        concat_tensor[i::step, :, :, :, :] = arrays[i]
                if rank == 4:
                    if axis == 3:
                        concat_tensor[:, :, :, i::step] = arrays[i]
                    if axis == 2:
                        concat_tensor[:, :, i::step, :] = arrays[i]
                    if axis == 1:
                        concat_tensor[:, i::step, :, :] = arrays[i]
                    if axis == 0:
                        concat_tensor[i::step, :, :, :] = arrays[i]
                if rank == 3:
                    if axis == 2:
                        concat_tensor[:, :, i::step] = arrays[i]
                    if axis == 1:
                        concat_tensor[:, i::step, :] = arrays[i]
                    if axis == 0:
                        concat_tensor[i::step, :, :] = arrays[i]
                if rank == 2:
                    if axis == 1:
                        concat_tensor[:, i::step] = arrays[i]
                    if axis == 0:
                        concat_tensor[i::step, :] = arrays[i]
                if rank == 1:
                    concat_tensor[i::step] = arrays[i]
            return concat_tensor

        input_shape = [4, 2, 3, 6, 5]
        for axis in range(rank):
            if negative_index:
                axis = axis - rank
            shape = tuple(input_shape[:rank])
            t1 = np.random.normal(size=shape).astype(np.float32)
            t2 = np.random.normal(size=shape).astype(np.float32)
            all_input_arrs = [t1, t2]
            input_placeholders = {
                "x": mb.placeholder(shape=t1.shape),
                "y": mb.placeholder(shape=t2.shape),
            }
            input_values = {"x": t1, "y": t2}
            if n_inputs == 3:
                t3 = np.random.normal(size=shape).astype(np.float32)
                input_placeholders["z"] = mb.placeholder(shape=t3.shape)
                input_values["z"] = t3
                all_input_arrs.append(t3)

            def build_2_inputs(x, y):
                return (mb.concat(values=(x, y), axis=axis, interleave=True),)

            def build_3_inputs(x, y, z):
                return (mb.concat(values=(x, y, z), axis=axis, interleave=True),)

            np_out = np_concat_interleave(all_input_arrs, axis)
            expected_output_types = [np_out.shape + (types.fp32,)]
            expected_outputs = [np_out]

            run_compare_builder(
                build_3_inputs if n_inputs == 3 else build_2_inputs,
                input_placeholders,
                input_values,
                expected_output_types,
                expected_outputs,
                compute_unit=compute_unit,
                backend=backend,
            )

    @ssa_fn
    def test_builder_eval(self):
        values = [
            np.random.rand(1, 1, 6, 2),
            np.random.rand(1, 1, 3, 2),
        ]
        v = mb.concat(values=values, axis=2)
        np.testing.assert_allclose(np.concatenate(values, 2), v.val, atol=1e-04, rtol=1e-05)

    @ssa_fn
    def test_builder_eval_failure(self):
        values = [
            np.random.rand(1, 1, 6, 2),
            np.random.rand(1, 1, 3, 1),
        ]
        with pytest.raises(ValueError):
            mb.concat(values=values, axis=2)


class TestSplit:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        t = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

        input_placeholders = {
            "x": mb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            return mb.split(x=x, num_splits=2, axis=1) + mb.split(x=x, split_sizes=[1, 2], axis=0)

        expected_output_types = [
            (3, 1, types.fp32),
            (3, 1, types.fp32),
            (1, 2, types.fp32),
            (2, 2, types.fp32),
        ]
        expected_outputs = [
            np.array([[1], [3], [5]], dtype=np.float32),
            np.array([[2], [4], [6]], dtype=np.float32),
            np.array([[1, 2]], dtype=np.float32),
            np.array([[3, 4], [5, 6]], dtype=np.float32),
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
        t = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        vs = mb.split(x=t, num_splits=3, axis=0)
        es = np.split(t, [1, 2, 3], axis=0)
        for v, e in zip(vs, es):
            np.testing.assert_allclose(e, v.val, atol=1e-04, rtol=1e-05)


class TestStack:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        t1 = np.array([1, 2, 3], dtype=np.float32)
        t2 = np.array([7, 8, 9], dtype=np.float32)

        input_placeholders = {
            "x": mb.placeholder(shape=t1.shape),
            "y": mb.placeholder(shape=t2.shape),
        }
        input_values = {"x": t1, "y": t2}

        def build(x, y):
            return [
                mb.stack(values=(x, y), axis=0),
                mb.stack(values=(x, y), axis=1),
                mb.stack(values=(x, y), axis=-1),
            ]

        expected_output_types = [
            (2, 3, types.fp32),
            (3, 2, types.fp32),
            (3, 2, types.fp32),
        ]
        expected_outputs = [
            np.array([[1, 2, 3], [7, 8, 9]], dtype=np.float32),
            np.array([[1, 7], [2, 8], [3, 9]], dtype=np.float32),
            np.array([[1, 7], [2, 8], [3, 9]], dtype=np.float32),
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
        values = [
            np.random.rand(1, 1, 3, 2).astype(np.float32),
            np.random.rand(1, 1, 3, 2).astype(np.float32),
        ]
        v = mb.stack(values=values, axis=2)
        np.testing.assert_allclose(np.stack(values, 2), v.val, atol=1e-04, rtol=1e-05)
