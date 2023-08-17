#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.tests.iOS16 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import (
    mark_api_breaking,
    run_compare_builder,
)
from coremltools.converters.mil.testing_reqs import compute_units


class TestGather:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_builder_to_backend_smoke_batch_dims(self, compute_unit, backend):
        # TODO MAKE SURE RUN ON IOS17
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
            np.array(
                [
                    [
                        [[[4, 5, 6], [1, 2, 3]], [[1, 2, 3], [4, 5, 6]]],
                        [[[4, 5, 6], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]],
                    ],
                    [
                        [[[10, 11, 12], [7, 8, 9]], [[7, 8, 9], [10, 11, 12]]],
                        [[[10, 11, 12], [7, 8, 9]], [[7, 8, 9], [7, 8, 9]]],
                    ],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [[[4, 5, 6], [1, 2, 3]], [[1, 2, 3], [4, 5, 6]]],
                    [[[10, 11, 12], [7, 8, 9]], [[7, 8, 9], [7, 8, 9]]],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [[[[2, 1], [1, 2]], [[2, 1], [1, 1]]], [[[5, 4], [4, 5]], [[5, 4], [4, 4]]]],
                    [
                        [[[8, 7], [7, 8]], [[8, 7], [7, 7]]],
                        [[[11, 10], [10, 11]], [[11, 10], [10, 10]]],
                    ],
                ],
                dtype=np.float32,
            ),
            np.array(
                [[[[2, 1], [1, 2]], [[5, 4], [4, 5]]], [[[8, 7], [7, 7]], [[11, 10], [10, 10]]]],
                dtype=np.float32,
            ),
            np.array([[[2, 1], [4, 5]], [[8, 7], [10, 10]]], dtype=np.float32),
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
        "backend",
        backends,
    )
    def test_builder_eval_batch_dims(self, backend):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)],
            opset_version=backend.opset_version,
        )
        def prog(x):
            params = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
            indices = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 0]]], dtype=np.int32)
            res = mb.gather(x=params, indices=indices, axis=2, batch_dims=2)
            return res

        main_func = prog.functions["main"]
        gather_ops = main_func.find_ops(op_type="gather")[0]

        np.testing.assert_allclose(
            np.array([[[2, 1], [4, 5]], [[8, 7], [10, 10]]], dtype=np.float32),
            gather_ops.outputs[0].val,
            atol=1e-04,
            rtol=1e-05,
        )


class TestGatherNd:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_builder_to_backend_smoke_batch_dims(self, compute_unit, backend):
        # TODO MAKE SURE RUN ON IOS17
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

        expected_output_types = [(2, 2, 3, types.fp32), (2, 2, types.fp32)]

        expected_outputs = [
            np.array([[[7, 8, 9], [4, 5, 6]], [[7, 8, 9], [1, 2, 3]]], dtype=np.float32),
            np.array([[4, 2], [10, 7]], dtype=np.float32),
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

    @mark_api_breaking(breaking_opset_version=ct.target.iOS17)
    @pytest.mark.parametrize(
        "backend, indices_val",
        itertools.product(backends, [[[-1], [2]], [[1], [3]]]),
    )
    def test_builder_invalid_indices(self, backend, indices_val):
        def prog(x):
            params = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            indices = np.array(indices_val, dtype=np.int32)
            res = mb.gather_nd(x=params, indices=indices, batch_dims=1)
            return res

        mb.program(
            input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)],
            opset_version=backend.opset_version,
        )(prog)
