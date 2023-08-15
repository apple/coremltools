#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.ops.tests.iOS15 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import (
    mark_api_breaking,
    run_compare_builder,
)
from coremltools.converters.mil.testing_reqs import compute_units


class TestAffine:
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x_val = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32).reshape([1, 1, 2, 2])
        transform_matrix_val = np.array(
            [-1.0, -2.0, -3.7, -1.0, 3.5, 1.2], dtype=np.float32
        ).reshape([1, 6])

        input_placeholder_dict = {
            "x": mb.placeholder(shape=x_val.shape),
            "transform_matrix": mb.placeholder(shape=transform_matrix_val.shape),
        }
        input_value_dict = {"x": x_val, "transform_matrix": transform_matrix_val}

        def build(x, transform_matrix):
            return [
                mb.affine(
                    x=x,
                    transform_matrix=transform_matrix,
                    output_height=3,
                    output_width=3,
                    sampling_mode="bilinear",
                    padding_mode="constant",
                    padding_value=0.0,
                    coordinates_mode="normalized_minus_one_to_one",
                    align_corners=True,
                ),
                mb.affine(
                    x=x,
                    transform_matrix=transform_matrix,
                    output_height=2,
                    output_width=5,
                    sampling_mode="bilinear",
                    padding_mode="constant",
                    padding_value=0.0,
                    coordinates_mode="normalized_minus_one_to_one",
                    align_corners=True,
                ),
            ]

        expected_output_types = [
            (1, 1, 3, 3, types.fp32),
            (1, 1, 2, 5, types.fp32),
        ]
        expected_outputs = [
            np.array(
                [10.752501, 2.5025, 0.0, 1.9799997, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ).reshape([1, 1, 3, 3]),
            np.array(
                [10.752501, 5.94, 2.5025, 0.44000006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ).reshape([1, 1, 2, 5]),
        ]

        run_compare_builder(
            build,
            input_placeholder_dict,
            input_value_dict,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestUpsampleNearestNeighborFractionalScales:
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        if compute_unit != ct.ComputeUnit.CPU_ONLY:
            pytest.xfail(
                "rdar://97398448 (TestUpsampleNearestNeighborFractionalScales failing on GPU)"
            )

        x_val = np.array([1.5, -2.5, 3.5], dtype=np.float32).reshape([1, 1, 1, 3])
        input_placeholder_dict = {"x": mb.placeholder(shape=x_val.shape)}
        input_value_dict = {"x": x_val}

        def build(x):
            return [
                mb.upsample_nearest_neighbor(
                    x=x,
                    scale_factor_height=1.0,
                    scale_factor_width=1.0,
                ),
                mb.upsample_nearest_neighbor(
                    x=x, scale_factor_height=3.17, scale_factor_width=0.67
                ),
                mb.upsample_nearest_neighbor(
                    x=x,
                    scale_factor_height=2.0,
                    scale_factor_width=1.12,
                ),
            ]

        expected_output_types = [
            (1, 1, 1, 3, types.fp32),
            (1, 1, 3, 2, types.fp32),
            (1, 1, 2, 3, types.fp32),
        ]
        expected_outputs = [
            x_val,
            np.array([1.5, -2.5, 1.5, -2.5, 1.5, -2.5], dtype=np.float32).reshape([1, 1, 3, 2]),
            np.array([1.5, -2.5, 3.5, 1.5, -2.5, 3.5], dtype=np.float32).reshape([1, 1, 2, 3]),
        ]

        run_compare_builder(
            build,
            input_placeholder_dict,
            input_value_dict,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestResample:
    @staticmethod
    def _test_builder_to_backend_smoke(compute_unit, backend, coordinates_dtype, expected_cast_ops):
        x_ = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32).reshape([1, 1, 2, 2])
        coordinates_ = (
            np.array([-1.0, -2.0, -3.7, -1.0, 0.0, 0.0, 3.5, 1.2], dtype=np.float32)
            .reshape([1, 2, 2, 2])
            .astype(coordinates_dtype)
        )
        if np.issubdtype(coordinates_dtype, np.integer):
            coordinates_ = (
                np.array([0, 0, 1, 1, 0, 0, 1, 1]).reshape([1, 2, 2, 2]).astype(coordinates_dtype)
            )
        expected_output_type = (1, 1, 2, 2, types.fp32)

        def build_0(x, coordinates):
            return mb.resample(
                x=x,
                coordinates=coordinates,
                sampling_mode="bilinear",
                padding_mode="constant",
                padding_value=6.17,
                coordinates_mode="normalized_minus_one_to_one",
                align_corners=True,
            )

        expected_output_0 = np.array([8.585, 6.17, 27.5, 6.17], dtype=np.float32)
        if np.issubdtype(coordinates_dtype, np.integer):
            expected_output_0 = np.array([27.5, 44.0, 27.5, 44.0], dtype=np.float32)
        expected_output_0 = expected_output_0.reshape(expected_output_type[:-1])

        def build_1(x, coordinates):
            return mb.resample(
                x=x,
                coordinates=coordinates,
                sampling_mode="nearest",
                padding_mode="border",
                padding_value=-1.0,
                coordinates_mode="unnormalized",
                align_corners=False,
            )

        expected_output_1 = np.array([11.0, 11.0, 11.0, 44.0], dtype=np.float32)
        if np.issubdtype(coordinates_dtype, np.integer):
            expected_output_1 = np.array([11.0, 44.0, 11.0, 44.0], dtype=np.float32)
        expected_output_1 = expected_output_1.reshape(expected_output_type[:-1])

        def build_2(x, coordinates):
            return mb.resample(
                x=x,
                coordinates=coordinates,
                sampling_mode="bilinear",
                padding_mode="reflection",
                padding_value=-1.0,
                coordinates_mode="normalized_zero_to_one",
                align_corners=True,
            )

        expected_output_2 = np.array([22.0, 36.3, 11.0, 34.1], dtype=np.float32)
        if np.issubdtype(coordinates_dtype, np.integer):
            expected_output_2 = np.array([11.0, 44.0, 11.0, 44.0], dtype=np.float32)
        expected_output_2 = expected_output_2.reshape(expected_output_type[:-1])

        def build_3(x, coordinates):
            return mb.resample(
                x=x,
                coordinates=coordinates,
                sampling_mode="nearest",
                padding_mode="symmetric",
                padding_value=-1.0,
                coordinates_mode="normalized_zero_to_one",
                align_corners=False,
            )

        expected_output_3 = np.array([22.0, 33.0, 11.0, 33.0], dtype=np.float32)
        if np.issubdtype(coordinates_dtype, np.integer):
            expected_output_3 = np.array([11.0, 44.0, 11.0, 44.0], dtype=np.float32)
        expected_output_3 = expected_output_3.reshape(expected_output_type[:-1])

        for build, expected_output in zip(
            [build_0, build_1, build_2, build_3],
            [
                expected_output_0,
                expected_output_1,
                expected_output_2,
                expected_output_3,
            ],
        ):
            # Need to create placeholders inside for loop to avoid interfere with each other.
            input_placeholder_dict = {
                "x": mb.placeholder(shape=x_.shape),
                "coordinates": mb.placeholder(
                    shape=coordinates_.shape,
                    dtype=types.numpy_type_to_builtin_type(coordinates_dtype),
                ),
            }
            input_value_dict = {"x": x_, "coordinates": coordinates_}

            mlmodel = run_compare_builder(
                build,
                input_placeholder_dict,
                input_value_dict,
                expected_output_type,
                expected_output,
                compute_unit=compute_unit,
                backend=backend,
            )
            prog = mlmodel._mil_program
            number_of_cast = len(prog["main"].find_ops(op_type="cast"))
            assert number_of_cast == expected_cast_ops

    @mark_api_breaking(breaking_opset_version=ct.target.iOS16)
    @pytest.mark.parametrize(
        "compute_unit, backend, coordinates_dtype",
        itertools.product(
            compute_units,
            backends,
            (np.int32, np.float32),
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, coordinates_dtype):
        self._test_builder_to_backend_smoke(compute_unit, backend, coordinates_dtype, 2)


class TestResizeBilinear:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([0, 1], dtype=np.float32).reshape(1, 1, 2)
        input_placeholder_dict = {"x": mb.placeholder(shape=x.shape)}
        input_value_dict = {"x": x}

        def build_mode_4(x):
            return mb.resize_bilinear(
                x=x,
                target_size_height=1,
                target_size_width=5,
                sampling_mode="UNALIGN_CORNERS",
            )

        expected_output_type = expected_output_type = (1, 1, 5, types.fp32)
        expected_output = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float32).reshape(1, 1, 5)

        run_compare_builder(
            build_mode_4,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestCropResize:
    @mark_api_breaking(breaking_opset_version=ct.target.iOS17)
    @pytest.mark.parametrize(
        "compute_unit, backend, is_symbolic",
        itertools.product(compute_units, backends, [True, False]),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, is_symbolic):
        if compute_unit != ct.ComputeUnit.CPU_ONLY:
            pytest.xfail("rdar://97398582 (TestCropResize failing on mlprogram + GPU)")
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)

        input_shape = list(x.shape)
        placeholder_input_shape = input_shape
        if is_symbolic:
            # set batch and channel dimension symbolic
            placeholder_input_shape[0] = get_new_symbol()
            placeholder_input_shape[1] = get_new_symbol()

        input_placeholder_dict = {"x": mb.placeholder(shape=placeholder_input_shape)}
        input_value_dict = {"x": x}
        N = 1
        roi = np.array([[1, 1, 2, 2]], dtype=np.float32).reshape(1, 1, 4, 1, 1)
        roi_normalized = np.array([[0, 0.0, 0.0, 1.0 / 3, 1.0 / 3]], dtype=np.float32).reshape(
            1, 1, 5, 1, 1
        )
        roi_invert = np.array([[2, 2, 1, 1]], dtype=np.float32).reshape(1, 1, 4, 1, 1)

        def build(x):
            return mb.crop_resize(
                x=x,
                roi=roi_invert,
                target_width=2,
                target_height=2,
                normalized_coordinates=True,
                box_coordinate_mode="CORNERS_HEIGHT_FIRST",
                sampling_mode="UNALIGN_CORNERS",
            )

        expected_output_type = (
            N,
            placeholder_input_shape[0],
            placeholder_input_shape[1],
            2,
            2,
            types.fp32,
        )

        expected_output = np.array([3.5, 5.5, 11.5, 13.5], dtype=np.float32).reshape(1, 1, 1, 2, 2)

        run_compare_builder(
            build,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )
