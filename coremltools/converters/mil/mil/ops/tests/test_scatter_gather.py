#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools._deps import _HAS_TF_2, MSG_TF2_NOT_FOUND
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.testing_reqs import backends, compute_units
from coremltools.converters.mil.testing_utils import ssa_fn

from .testing_utils import run_compare_builder

if _HAS_TF_2:
    import tensorflow as tf


class TestScatter:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        indices = np.array([1, 0], dtype=np.int32)
        updates = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.float32)
        input_placeholders = {
            "data": mb.placeholder(shape=data.shape),
            "indices": mb.placeholder(shape=indices.shape, dtype=types.int32),
            "updates": mb.placeholder(shape=updates.shape),
        }

        input_values = {"data": data, "indices": indices, "updates": updates}

        def build(data, indices, updates):
            return (mb.scatter(data=data, indices=indices, updates=updates),)

        expected_output_types = (2, 3, types.fp32)

        expected_outputs = np.array([[9, 11, 13], [9, 11, 13]], dtype=np.float32)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.skipif(not _HAS_TF_2, reason=MSG_TF2_NOT_FOUND)
    @pytest.mark.parametrize(
        "compute_unit, backend, rankData_rankIndices, accumulate_mode",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 2),
                (2, 1),
                (3, 2),
                (2, 3),
                (2, 2),
                (1, 1),
                (3, 3),
                (3, 3),
                (3, 3),
                (1, 3),
                (3, 1),
                (3, 1),
            ],
            ["update", "add", "sub", "mul", "div", "max", "min"],
        ),
    )
    def test_builder_to_backend_programmatic(
        self, compute_unit, backend, rankData_rankIndices, accumulate_mode
    ):
        data_rank, indices_rank = rankData_rankIndices
        data_shape = np.random.randint(low=2, high=5, size=data_rank)
        indices_shape = np.random.randint(low=2, high=5, size=indices_rank)
        updates_shape = list(indices_shape) + list(data_shape[1:])

        data = np.random.rand(*data_shape).astype(np.float32)
        updates = np.random.rand(*updates_shape).astype(np.float32)
        indices = np.random.randint(0, data_shape[0], size=indices_shape).astype(
            np.int32
        )

        def build(data, indices, updates):
            return mb.scatter(
                data=data, indices=indices, updates=updates, mode=accumulate_mode
            )

        tf_output = tf.Variable(data)
        if accumulate_mode == "update":
            tf.compat.v1.scatter_update(tf_output, indices, updates)
        if accumulate_mode == "add":
            tf.compat.v1.scatter_add(tf_output, indices, updates)
        if accumulate_mode == "sub":
            tf.compat.v1.scatter_sub(tf_output, indices, updates)
        if accumulate_mode == "mul":
            tf.compat.v1.scatter_mul(tf_output, indices, updates)
        if accumulate_mode == "div":
            tf.compat.v1.scatter_div(tf_output, indices, updates)
        if accumulate_mode == "max":
            tf.compat.v1.scatter_max(tf_output, indices, updates)
        if accumulate_mode == "min":
            tf.compat.v1.scatter_min(tf_output, indices, updates)
        expected_output = tf_output.numpy()

        input_placeholders = {
            "data": mb.placeholder(shape=data.shape),
            "indices": mb.placeholder(shape=indices.shape, dtype=types.int32),
            "updates": mb.placeholder(shape=updates.shape),
        }

        input_values = {"data": data, "indices": indices, "updates": updates}

        expected_output_types = tuple(data_shape[:]) + (types.fp32,)
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestScatterAlongAxis:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        indices = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.int32)
        updates = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.float32)
        input_placeholders = {
            "data": mb.placeholder(shape=data.shape),
            "indices": mb.placeholder(shape=indices.shape, dtype=types.int32),
            "updates": mb.placeholder(shape=updates.shape),
        }

        input_values = {"data": data, "indices": indices, "updates": updates}

        def build(data, indices, updates):
            return mb.scatter_along_axis(
                data=data, indices=indices, updates=updates, axis=0, mode="update"
            )

        expected_output_types = (2, 3, types.fp32)

        expected_outputs = np.array([[1, 6, 10], [8, 9, 7]], dtype=np.float32)

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
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        indices = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.int32)
        updates = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.float32)
        v = mb.scatter_along_axis(
            data=x, indices=indices, updates=updates, axis=0, mode="update"
        )
        np.testing.assert_allclose(np.array([[1, 6, 10], [8, 9, 7]], dtype=np.float32), v.val, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "compute_unit, backend, rank_axis",
        itertools.product(
            compute_units,
            backends,
            [(rank, axis) for rank in range(1, 5) for axis in range(-rank, rank)],
        ),
    )
    def test_builder_to_backend_programmatic(self, compute_unit, backend, rank_axis):
        rank, axis = rank_axis
        data_shape = np.random.randint(low=2, high=8, size=rank)
        indices_shape = np.copy(data_shape)
        indices_shape[axis] = np.random.randint(low=1, high=8)
        updates_shape = indices_shape

        data = np.random.rand(*data_shape).astype(np.float32)
        updates = np.random.rand(*updates_shape).astype(np.float32)
        indices = np.random.randint(
            -data_shape[axis], data_shape[axis], size=indices_shape
        ).astype(np.int32)

        def build(data, indices, updates):
            return mb.scatter_along_axis(
                data=data, indices=indices, updates=updates, axis=axis, mode="update"
            )

        input_placeholders = {
            "data": mb.placeholder(shape=data.shape),
            "indices": mb.placeholder(shape=indices.shape, dtype=types.int32),
            "updates": mb.placeholder(shape=updates.shape),
        }

        input_values = {"data": data, "indices": indices, "updates": updates}

        expected_output_types = tuple(data_shape[:]) + (types.fp32,)

        np_output = np.copy(data)
        np.put_along_axis(np_output, indices, updates, axis=axis)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            np_output,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestScatterNd:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        indices = np.array([[1, 0], [0, 2]], dtype=np.int32)
        updates = np.array([5, 10], dtype=np.float32)
        input_placeholders = {
            "data": mb.placeholder(shape=data.shape),
            "indices": mb.placeholder(shape=indices.shape, dtype=types.int32),
            "updates": mb.placeholder(shape=updates.shape),
        }

        input_values = {"data": data, "indices": indices, "updates": updates}

        def build(data, indices, updates):
            return (mb.scatter_nd(data=data, indices=indices, updates=updates),)

        expected_output_types = (2, 3, types.fp32)

        expected_outputs = np.array([[1, 2, 13], [9, 5, 6]], dtype=np.float32)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.skipif(not _HAS_TF_2, reason=MSG_TF2_NOT_FOUND)
    @pytest.mark.parametrize(
        "compute_unit, backend, rankData_rankIndices, accumulate_mode",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 2),
                (2, 2),
                (3, 2),
                (2, 3),
                (1, 4),
                (5, 2),
                (2, 5),
                (4, 3),
                (3, 4),
                (2, 4),
                (4, 2),
                (1, 5),
            ],
            ["update", "add", "sub"],
        ),
    )
    def test_builder_to_backend_programmatic(
        self, compute_unit, backend, rankData_rankIndices, accumulate_mode
    ):
        data_rank, indices_rank = rankData_rankIndices
        data_shape = np.random.randint(low=2, high=5, size=data_rank)
        indices_shape = np.random.randint(low=2, high=5, size=indices_rank)
        indices_shape[-1] = np.random.randint(low=1, high=data_rank + 1)
        updates_shape = list(indices_shape[:-1]) + list(data_shape[indices_shape[-1] :])

        data = np.random.rand(*data_shape).astype(np.float32)
        updates = np.random.rand(*updates_shape).astype(np.float32)
        indices_list = []
        for i in range(indices_shape[-1]):
            indices_list.append(
                np.random.randint(0, data_shape[i], size=indices_shape[:-1])
            )

        indices = np.stack(indices_list, axis=-1).astype(np.int32)

        def build(data, indices, updates):
            return mb.scatter_nd(
                data=data, indices=indices, updates=updates, mode=accumulate_mode
            )

        tf_output = tf.Variable(data)
        if accumulate_mode == "update":
            tf.compat.v1.scatter_nd_update(tf_output, indices, updates)
        if accumulate_mode == "add":
            tf.compat.v1.scatter_nd_add(tf_output, indices, updates)
        if accumulate_mode == "sub":
            tf.compat.v1.scatter_nd_sub(tf_output, indices, updates)
        expected_output = tf_output.numpy()

        input_placeholders = {
            "data": mb.placeholder(shape=data.shape),
            "indices": mb.placeholder(shape=indices.shape, dtype=types.int32),
            "updates": mb.placeholder(shape=updates.shape),
        }

        input_values = {"data": data, "indices": indices, "updates": updates}

        expected_output_types = tuple(data_shape[:]) + (types.fp32,)
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestGather:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        indices = np.array([1, 0], dtype=np.int32)
        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "indices": mb.placeholder(shape=indices.shape, dtype=types.int32),
        }

        input_values = {"x": x, "indices": indices}

        def build(x, indices):
            return [
                mb.gather(x=x, indices=indices, axis=0),
                mb.gather(x=x, indices=indices, axis=1),
                mb.gather(x=x, indices=indices, axis=-2),
                mb.gather(x=x, indices=indices, axis=-1),
                mb.gather(x=x, indices=indices),
                # mb.gather(x=x, indices=1), #shape of scalar indices is incorrect.
                # mb.gather(x=x, indices=1, axis=1), #Scalar index passes on axis=0 but fails on axis=1,
                # Need to handle rank 0 correctly, rdar://73160449
            ]

        expected_output_types = [
            (2, 3, types.fp32),
            (2, 2, types.fp32),
            (2, 3, types.fp32),
            (2, 2, types.fp32),
            (2, 3, types.fp32),
            # (3, types.fp32),
        ]

        expected_outputs = [
            np.array([[4, 5, 6], [1, 2, 3]], dtype=np.float32),
            np.array([[2, 1], [5, 4]], dtype=np.float32),
            np.array([[4, 5, 6], [1, 2, 3]], dtype=np.float32),
            np.array([[2, 1], [5, 4]], dtype=np.float32),
            np.array([[4, 5, 6], [1, 2, 3]], dtype=np.float32),
            # np.array([4, 5, 6], dtype=np.float32),
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
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_builder_to_backend_smoke_iOS16(self, compute_unit, backend):
        if backend[0] == "neuralnetwork":
            pytest.skip("nn backend not supported")
        if ct.utils._macos_version() < (13, 0):
            pytest.skip("batch_dims not supported in macOS12 or older.")

        x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
        indices = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 0]]], dtype=np.int32)

        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "indices": mb.placeholder(shape=indices.shape, dtype=types.int32),
        }

        input_values = {"x": x, "indices": indices}

        def build(x, indices):
            return [
                mb.gather(x=x, indices=indices, axis=1, batch_dims=0),
                mb.gather(x=x, indices=indices, axis=1, batch_dims=1),
                mb.gather(x=x, indices=indices, axis=2, batch_dims=0),
                mb.gather(x=x, indices=indices, axis=2, batch_dims=1),
                mb.gather(x=x, indices=indices, axis=2, batch_dims=2),
            ]

        expected_output_types = [
            (2, 2, 2, 2, 3, types.fp32),
            (2, 2, 2, 3, types.fp32),
            (2, 2, 2, 2, 2, types.fp32),
            (2, 2, 2, 2, types.fp32),
            (2, 2, 2, types.fp32),
        ]

        expected_outputs = [
            np.array([[[[[ 4,  5,  6],
                         [ 1,  2,  3]],
                        [[ 1,  2,  3],
                         [ 4,  5,  6]]],
                       [[[ 4,  5,  6],
                         [ 1,  2,  3]],
                        [[ 1,  2,  3],
                         [ 1,  2,  3]]]],
                      [[[[10, 11, 12],
                         [ 7,  8,  9]],
                        [[ 7,  8,  9],
                         [10, 11, 12]]],
                       [[[10, 11, 12],
                         [ 7,  8,  9]],
                        [[ 7,  8,  9],
                         [ 7,  8,  9]]]]], dtype=np.float32
            ), 
            np.array([[[[ 4,  5,  6],
                        [ 1,  2,  3]],
                       [[ 1,  2,  3],
                        [ 4,  5,  6]]],
                      [[[10, 11, 12],
                        [ 7,  8,  9]],
                       [[ 7,  8,  9],
                        [ 7,  8,  9]]]], dtype=np.float32
            ),
            np.array([[[[[ 2,  1],
                         [ 1,  2]],
                        [[ 2,  1],
                         [ 1,  1]]],
                       [[[ 5,  4],
                         [ 4,  5]],
                        [[ 5,  4],
                         [ 4,  4]]]],
                      [[[[ 8,  7],
                         [ 7,  8]],
                        [[ 8,  7],
                         [ 7,  7]]],
                       [[[11, 10],
                         [10, 11]],
                        [[11, 10],
                         [10, 10]]]]], dtype=np.float32
            ),
            np.array([[[[ 2,  1],
                        [ 1,  2]],
                       [[ 5,  4],
                        [ 4,  5]]],
                      [[[ 8,  7],
                        [ 7,  7]],
                       [[11, 10],
                        [10, 10]]]], dtype=np.float32
            ),
            np.array([[[ 2,  1],
                       [ 4,  5]],
                      [[ 8,  7],
                       [10, 10]]], dtype=np.float32
            ),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            minimum_deployment_target=ct.target.iOS16,
        )


    def test_builder_eval_iOS16(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, ), dtype=types.fp32)], opset_version=ct.target.iOS16)
        def prog(x):
            params = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
            indices = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 0]]], dtype=np.int32)
            res = mb.gather(x=params, indices=indices, axis=2, batch_dims=2)
            return res

        main_func = prog.functions["main"]
        gather_ops = main_func.find_ops(op_type="gather")[0]

        np.testing.assert_allclose(
            np.array([[[ 2,  1], [ 4,  5]], [[ 8,  7], [10, 10]]], dtype=np.float32), 
            gather_ops.outputs[0].val, 
            atol=1e-04, 
            rtol=1e-05
        )

    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_embedding_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        indices = np.array([1, 0], dtype=np.int32)
        input_placeholders = {
            "indices": mb.placeholder(shape=indices.shape, dtype=types.int32),
        }

        input_values = {"indices": indices}

        def build(indices):
            return [
                mb.gather(x=x, indices=indices, axis=0),
                mb.gather(x=x, indices=indices, axis=-2),
            ]

        expected_output_types = [
            (2, 3, types.fp32),
            (2, 3, types.fp32),
        ]

        expected_outputs = [
            np.array([[4, 5, 6], [1, 2, 3]], dtype=np.float32),
            np.array([[4, 5, 6], [1, 2, 3]], dtype=np.float32),
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
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        indices = np.array([1, 0], dtype=np.int32)
        v = mb.gather(x=x, indices=indices, axis=-1)
        np.testing.assert_allclose(np.array([[2, 1], [5, 4]], dtype=np.float32), v.val, atol=1e-04, rtol=1e-05)


class TestGatherAlongAxis:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        indices = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.int32)
        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "indices": mb.placeholder(shape=indices.shape, dtype=types.int32),
        }

        input_values = {"x": x, "indices": indices}

        def build(x, indices):
            return [
                mb.gather_along_axis(x=x, indices=indices, axis=0),
                mb.gather_along_axis(x=x, indices=indices, axis=1),
                mb.gather_along_axis(x=x, indices=indices, axis=-2),
                mb.gather_along_axis(x=x, indices=indices, axis=-1),
                mb.gather_along_axis(x=x, indices=indices),
            ]

        expected_output_types = [
            (2, 3, types.fp32),
            (2, 3, types.fp32),
            (2, 3, types.fp32),
            (2, 3, types.fp32),
            (2, 3, types.fp32),
        ]

        expected_outputs = [
            np.array([[4, 2, 6], [4, 5, 3]], dtype=np.float32),
            np.array([[2, 1, 2], [5, 5, 4]], dtype=np.float32),
            np.array([[4, 2, 6], [4, 5, 3]], dtype=np.float32),
            np.array([[2, 1, 2], [5, 5, 4]], dtype=np.float32),
            np.array([[4, 2, 6], [4, 5, 3]], dtype=np.float32),
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
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        indices = np.array([[1, 0, 1], [0, 0, 1]], dtype=np.int32)
        v = mb.gather_along_axis(x=x, indices=indices, axis=0)
        np.testing.assert_allclose(np.array([[4, 2, 6], [1, 2, 6]], dtype=np.float32), v.val, atol=1e-04, rtol=1e-05)

    @pytest.mark.parametrize(
        "compute_unit, backend, rank_axis",
        itertools.product(
            compute_units,
            backends,
            [(rank, axis) for rank in range(1, 5) for axis in range(-rank, rank)],
        ),
    )
    def test_builder_to_backend_programmatic(self, compute_unit, backend, rank_axis):
        if backend[0] == "mlprogram" and compute_unit != ct.ComputeUnit.CPU_ONLY:
            pytest.xfail("rdar://97398875 (TestGatherAlongAxis failing on mlprgram + GPU)")
        rank, axis = rank_axis
        x_shape = np.random.randint(low=2, high=8, size=rank)
        indices_shape = np.copy(x_shape)
        indices_shape[axis] = np.random.randint(low=1, high=8)

        x = np.random.rand(*x_shape).astype(np.float32)
        indices = np.random.randint(
            -x_shape[axis], x_shape[axis], size=indices_shape
        ).astype(np.int32)

        def build(x, indices):
            return mb.gather_along_axis(x=x, indices=indices, axis=axis)

        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "indices": mb.placeholder(shape=indices.shape, dtype=types.int32),
        }

        input_values = {"x": x, "indices": indices}

        expected_output_types = tuple(indices_shape[:]) + (types.fp32,)
        expected_output = np.take_along_axis(x, indices, axis=axis)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestGatherNd:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        indices = np.array([[1, 0], [0, 2]], dtype=np.int32)
        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "indices": mb.placeholder(shape=indices.shape, dtype=types.int32),
        }

        input_values = {"x": x, "indices": indices}

        def build(x, indices):
            return (mb.gather_nd(x=x, indices=indices),)

        expected_output_types = (2, types.fp32)
        expected_outputs = np.array([4, 3], dtype=np.float32)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            frontend_only=False,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_builder_to_backend_smoke_iOS16(self, compute_unit, backend):
        if backend[0] == "neuralnetwork":
            pytest.skip("nn backend not supported")

        if ct.utils._macos_version() < (13, 0):
            pytest.skip("batch_dims not supported in macOS12 or older.")

        x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
        indices = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 0]]], dtype=np.int32)

        input_placeholders = {
            "x": mb.placeholder(shape=x.shape),
            "indices": mb.placeholder(shape=indices.shape, dtype=types.int32),
        }

        input_values = {"x": x, "indices": indices}

        def build(x, indices):
            return [
                mb.gather_nd(x=x, indices=indices, batch_dims=0),
                mb.gather_nd(x=x, indices=indices, batch_dims=1),
            ]

        expected_output_types = [
            (2, 2, 3, types.fp32),
            (2, 2, types.fp32)
        ]

        expected_outputs = [
            np.array([[[7, 8, 9],
                       [4, 5, 6]],
                      [[7, 8, 9],
                       [1, 2, 3]]], dtype=np.float32
            ),
            np.array([[ 4,  2],
                      [10,  7]], dtype=np.float32
            ),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            minimum_deployment_target=ct.target.iOS16,
        )
