#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.tests.iOS14.test_scatter_gather import (
    TestGatherAlongAxis as _TestGatherAlongAxis_iOS14,
)
from coremltools.converters.mil.mil.ops.tests.iOS14.test_scatter_gather import (
    TestScatterAlongAxis as _TestScatterAlongAxis_iOS14,
)
from coremltools.converters.mil.mil.ops.tests.iOS17 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.testing_reqs import compute_units


class TestScatter:
    @pytest.mark.parametrize(
        "compute_unit, backend, indices_val, validate_indices, dynamic",
        itertools.product(
            compute_units,
            backends,
            [[-1, 0], [10, 0]],  # One negative indices, another out-of-range indices.
            [True, False],
            [True, False],
        ),
    )
    def test_ios17_invalid_indices(
        self, compute_unit, backend, indices_val, validate_indices, dynamic
    ):
        def build_static(data, updates):
            return (
                mb.scatter(
                    data=data,
                    indices=np.array(indices_val, dtype=np.int32),
                    updates=updates,
                    validate_indices=validate_indices,
                ),
            )

        def build_dynamic(data, indices, updates):
            return (
                mb.scatter(
                    data=data, indices=indices, updates=updates, validate_indices=validate_indices
                ),
            )

        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        updates = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.float32)
        input_placeholders = {
            "data": mb.placeholder(shape=data.shape),
            "updates": mb.placeholder(shape=updates.shape),
        }
        input_values = {"data": data, "updates": updates}
        if dynamic:
            indices = np.array(indices_val, dtype=np.int32)
            input_placeholders["indices"] = mb.placeholder(shape=indices.shape, dtype=types.int32)
            input_values["indices"] = indices

        if not validate_indices:
            # When not validate indices, negative or out-of-bound indices behavior is undefined.
            expected_error = AssertionError
            expected_error_msg = "Not equal"
        elif dynamic:
            # In PyMIL's validation, the `validate_indices` will only validate indices whose values are
            # known during op insertion, so it will not error out at PyMIL layer, but instead, rely on
            # the backend to do the validation after compilation.
            expected_error = RuntimeError
            expected_error_msg = (
                "Error computing NN outputs",
                "Unable to compute the prediction using a neural network model",
            )
        else:
            # The negative or out-of-bound indices will error out when validate_indices is set.
            expected_error = IndexError
            expected_error_msg = "Indices is out of bounds"

        with pytest.raises(expected_error) as excinfo:
            run_compare_builder(
                build_dynamic if dynamic else build_static,
                input_placeholders,
                input_values,
                expected_output_types=(2, 3, types.fp32),
                expected_outputs=np.array([[9, 11, 13], [9, 11, 13]], dtype=np.float32),
                compute_unit=compute_unit,
                backend=backend,
            )
            if not isinstance(expected_error_msg, tuple):
                expected_error_msg = expected_error_msg
            assert any([err in str(excinfo.value) for err in expected_error_msg])


class TestScatterAlongAxis:
    @pytest.mark.parametrize(
        "compute_unit, backend, rank_axis",
        itertools.product(
            compute_units,
            backends,
            [(rank, axis) for rank in range(1, 5) for axis in range(-rank, rank)],
        ),
    )
    def test_builder_to_backend_programmatic(self, compute_unit, backend, rank_axis):
        _TestScatterAlongAxis_iOS14._test_builder_to_backend_programmatic(
            compute_unit, backend, rank_axis, force_non_negative_indices=True
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, indices_val, dynamic",
        itertools.product(
            compute_units,
            backends,
            [[[-1, 0, 1], [1, 1, 0]], [[1, 10, 1], [1, 1, 0]]],
            [True, False],
        ),
    )
    def test_ios17_invalid_indices(self, compute_unit, backend, indices_val, dynamic):
        def build_static(data, updates):
            return (
                mb.scatter_along_axis(
                    data=data,
                    indices=np.array(indices_val, dtype=np.int32),
                    updates=updates,
                    validate_indices=True,
                ),
            )

        def build_dynamic(data, indices, updates):
            return mb.scatter_along_axis(
                data=data,
                indices=indices,
                updates=updates,
                axis=0,
                mode="update",
                validate_indices=True,
            )

        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        updates = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.float32)
        input_placeholders = {
            "data": mb.placeholder(shape=data.shape),
            "updates": mb.placeholder(shape=updates.shape),
        }
        input_values = {"data": data, "updates": updates}
        if dynamic:
            indices = np.array(indices_val, dtype=np.int32)
            input_placeholders["indices"] = mb.placeholder(shape=indices.shape, dtype=types.int32)
            input_values["indices"] = indices

        if dynamic:
            expected_error = RuntimeError
            expected_error_msg = (
                "Error computing NN outputs",
                "Unable to compute the prediction using a neural network model",
            )
        else:
            # The negative or out-of-bound indices will error out when validate_indices is set.
            expected_error = IndexError
            expected_error_msg = "Indices is out of bounds"

        # The negative or out-of-bound indices will error out when validate_indices is set.
        with pytest.raises(expected_error) as excinfo:
            run_compare_builder(
                build_dynamic if dynamic else build_static,
                input_placeholders,
                input_values,
                expected_output_types=(2, 3, types.fp32),
                expected_outputs=np.array([[1, 6, 10], [8, 9, 7]], dtype=np.float32),
                compute_unit=compute_unit,
                backend=backend,
            )
            if not isinstance(expected_error_msg, tuple):
                expected_error_msg = expected_error_msg
            assert any([err in str(excinfo.value) for err in expected_error_msg])


class TestScatterNd:
    @pytest.mark.parametrize(
        "compute_unit, backend, indices_val, dynamic",
        itertools.product(
            compute_units, backends, [[[1, 0], [0, -1]], [[1, 0], [0, 3]]], [True, False]
        ),
    )
    def test_ios17_invalid_indices(self, compute_unit, backend, indices_val, dynamic):
        def build_static(data, updates):
            return (
                mb.scatter_nd(
                    data=data,
                    indices=np.array(indices_val, dtype=np.int32),
                    updates=updates,
                    validate_indices=True,
                ),
            )

        def build_dynamic(data, indices, updates):
            return (
                mb.scatter_nd(data=data, indices=indices, updates=updates, validate_indices=True),
            )

        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        updates = np.array([5, 10], dtype=np.float32)
        input_placeholders = {
            "data": mb.placeholder(shape=data.shape),
            "updates": mb.placeholder(shape=updates.shape),
        }
        input_values = {"data": data, "updates": updates}
        if dynamic:
            indices = np.array(indices_val, dtype=np.int32)
            input_placeholders["indices"] = mb.placeholder(shape=indices.shape, dtype=types.int32)
            input_values["indices"] = indices

        if dynamic:
            expected_error = RuntimeError
            expected_error_msg = (
                "Error computing NN outputs",
                "Unable to compute the prediction using a neural network model",
            )
        else:
            # The negative or out-of-bound indices will error out when validate_indices is set.
            expected_error = IndexError
            expected_error_msg = "Indices is out of bounds"

        with pytest.raises(expected_error) as excinfo:
            run_compare_builder(
                build_dynamic if dynamic else build_static,
                input_placeholders,
                input_values,
                expected_output_types=(2, 3, types.fp32),
                expected_outputs=np.array([[1, 2, 13], [9, 5, 6]], dtype=np.float32),
                compute_unit=compute_unit,
                backend=backend,
            )
            if not isinstance(expected_error_msg, tuple):
                expected_error_msg = expected_error_msg
            assert any([err in str(excinfo.value) for err in expected_error_msg])


class TestGather:
    @pytest.mark.parametrize(
        "backend, indices_val, validate_indices",
        itertools.product(backends, [[-1, 0], [0, 3]], [True, False]),
    )
    def test_builder_invalid_indices_iOS17(self, backend, indices_val, validate_indices):
        def prog(x):
            params = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            indices = np.array(indices_val, dtype=np.int32)
            res = mb.gather(x=params, indices=indices, axis=-1, validate_indices=validate_indices)
            return res

        if validate_indices:
            with pytest.raises(IndexError, match="Indices is out of bounds for `gather` node"):
                mb.program(
                    input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)],
                    opset_version=backend.opset_version,
                )(prog)
        elif any([idx > 2 for idx in indices_val]):
            # If the indices are not validated during type inference for IOS17, the `gather` op's
            # value inference will raise error for out-of-bound index.
            with pytest.raises(IndexError, match="index 3 is out of bounds for axis 1 with size 3"):
                mb.program(
                    input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)],
                    opset_version=backend.opset_version,
                )(prog)
        else:
            mb.program(
                input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)],
                opset_version=backend.opset_version,
            )(prog)


class TestGatherAlongAxis:
    @pytest.mark.parametrize(
        "compute_unit, backend, rank_axis",
        itertools.product(
            compute_units,
            backends,
            [(rank, axis) for rank in range(1, 5) for axis in range(-rank, rank)],
        ),
    )
    def test_builder_to_backend_programmatic(self, compute_unit, backend, rank_axis):
        _TestGatherAlongAxis_iOS14._test_builder_to_backend_programmatic(
            compute_unit, backend, rank_axis, True
        )

    @pytest.mark.parametrize(
        "backend, indices_val, validate_indices",
        itertools.product(
            backends,
            [[[1, 0, -1], [0, 0, 1]], [[1, 0, 1], [0, 0, 2]]],
            [True, False],
        ),
    )
    def test_builder_invalid_indices(self, backend, indices_val, validate_indices):
        def prog(x):
            params = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            indices = np.array(indices_val, dtype=np.int32)
            res = mb.gather_along_axis(
                x=params, indices=indices, axis=0, validate_indices=validate_indices
            )
            return res

        if validate_indices:
            with pytest.raises(
                IndexError, match="Indices is out of bounds for `gather_along_axis` node"
            ):
                mb.program(
                    input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)],
                    opset_version=backend.opset_version,
                )(prog)
        elif any([idx > 1 for sub_indices in indices_val for idx in sub_indices]):
            # If the indices are not validated during type inference for IOS17, the `gather` op's
            # value inference will raise error for out-of-bound index.
            with pytest.raises(IndexError, match="index 2 is out of bounds for axis 0 with size 2"):
                mb.program(
                    input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)],
                    opset_version=backend.opset_version,
                )(prog)
        else:
            mb.program(
                input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)],
                opset_version=backend.opset_version,
            )(prog)

class TestGatherNd:
    @pytest.mark.parametrize(
        "backend, indices_val, validate_indices",
        itertools.product(
            backends,
            [[[-1], [2]], [[1], [3]]],
            [True, False],
        ),
    )
    def test_builder_invalid_indices(self, backend, indices_val, validate_indices):
        def prog(x):
            params = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            indices = np.array(indices_val, dtype=np.int32)
            res = mb.gather_nd(
                x=params, indices=indices, batch_dims=1, validate_indices=validate_indices
            )
            return res

        if validate_indices:
            with pytest.raises(IndexError, match="Indices is out of bounds for `gather_nd` node"):
                mb.program(
                    input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)],
                    opset_version=backend.opset_version,
                )(prog)
        else:
            mb.program(
                input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)],
                opset_version=backend.opset_version,
            )(prog)
