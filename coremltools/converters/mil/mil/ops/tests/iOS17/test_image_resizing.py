#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.tests.iOS15.test_image_resizing import (
    TestResample as _TestResampleIos15,
)
from coremltools.converters.mil.mil.ops.tests.iOS17 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import UNK_SYM, run_compare_builder
from coremltools.converters.mil.testing_reqs import compute_units


class TestCropResize:
    @pytest.mark.parametrize(
        "compute_unit, backend, N",
        itertools.product(compute_units, backends, [1, 3]),
    )
    def test_builder_to_backend_ios17(self, compute_unit, backend, N):
        """For iOS17+ the `roi` input is replaced by `boxes` and `box_indices`."""
        x = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        boxes = np.array([1, 1, 2, 2], dtype=np.float32).reshape(1, 4)
        box_indices = None
        normalized_coordinates = False
        if N == 3:
            boxes = np.array(
                [
                    [0.1, 0.3, 1.3, 1.0],
                    [0.5, 1.8, 1.0, 0.3],
                    [0.0, 0.4, 0.6, 0.7],
                ],
                dtype=np.float32,
            )
            box_indices = np.array([0] * 3, dtype=np.int32)
            normalized_coordinates = True

        def build(x):
            return mb.crop_resize(
                x=x,
                boxes=boxes,
                box_indices=box_indices,
                target_width=2,
                target_height=2,
                normalized_coordinates=normalized_coordinates,
                box_coordinate_mode="CORNERS_HEIGHT_FIRST",
                sampling_mode="ALIGN_CORNERS",
                pad_value=10.0,
            )

        expected_outputs = [np.array([6, 7, 10, 11], dtype=np.float32).reshape(1, 1, 2, 2)]
        if N == 3:
            expected_outputs = [
                np.array(
                    [3.1, 5.2, 10.0, 10.0, 10.0, 7.899, 10.0, 13.9, 2.2, 3.1, 9.4, 10.3],
                    dtype=np.float32,
                ).reshape(3, 1, 2, 2)
            ]

        run_compare_builder(
            build,
            input_placeholders={"x": mb.placeholder(shape=(1, 1, 4, 4))},
            input_values={"x": x},
            expected_output_types=[(N, 1, 2, 2, types.fp32)],
            expected_outputs=expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_builder_eval_ios17_invalid(self, backend):
        x = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        three_boxes = np.array(
            [
                [0.1, 0.3, 1.3, 1.0],
                [0.5, 1.8, 1.0, 0.3],
                [0.0, 0.4, 0.6, 0.7],
            ],
            dtype=np.float32,
        )
        with pytest.raises(
            ValueError,
            match='N dimension of "boxes" \(3\) should not be greater '
            'than the B dimension of "x" \(1\)',
        ):

            @mb.program(input_specs=[], opset_version=backend.opset_version)
            def prog():
                return mb.crop_resize(x=x, boxes=three_boxes)

        one_box = np.array([1, 1, 2, 2], dtype=np.float32).reshape(1, 4)
        indices_out_of_bound = np.array([10], dtype=np.int32)
        with pytest.raises(
            ValueError,
            match='input "box_indices" should not have values >= B '
            "dimension of x \(1\), but got \[10\]",
        ):

            @mb.program(input_specs=[], opset_version=backend.opset_version)
            def prog():
                return mb.crop_resize(x=x, boxes=one_box, box_indices=indices_out_of_bound)

        indices_two_dim = np.array([[0]], dtype=np.int32)
        with pytest.raises(
            ValueError, match='input "box_indices" must has shape \[1\], but got \(1, 1\)'
        ):

            @mb.program(input_specs=[], opset_version=backend.opset_version)
            def prog():
                return mb.crop_resize(x=x, boxes=one_box, box_indices=indices_two_dim)

        x_rank5 = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4, 1)
        with pytest.raises(
            ValueError, match='input to the "crop_resize" op must be of rank 4, but got 5'
        ):

            @mb.program(input_specs=[], opset_version=backend.opset_version)
            def prog():
                return mb.crop_resize(x=x_rank5, boxes=one_box)


class TestResample(_TestResampleIos15):
    @pytest.mark.parametrize(
        "compute_unit, backend, coordinates_dtype",
        itertools.product(
            compute_units,
            backends,
            (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.float16, np.float32),
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, coordinates_dtype):
        # The fp16 precision will have two casts inserted for input/output
        expected_cast_ops = 2 if backend.precision == "fp16" else 0
        if backend.precision == "fp16" and coordinates_dtype == np.float32:
            # The coordinates also cast to fp16.
            expected_cast_ops += 1
        if coordinates_dtype not in (np.int32, np.float16, np.float32):
            # For dtype not supported in CoreML I/O, a cast will be inserted.
            expected_cast_ops += 1
        self._test_builder_to_backend_smoke(
            compute_unit, backend, coordinates_dtype, expected_cast_ops
        )


class TestResize:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_resize_nearest_neighbor(self, compute_unit, backend):
        def build_model(x):
            return mb.resize(
                x=x,
                shape=[1, 1, 3, 2],
                resized_dims=np.uint32(2),
                interpolation_mode="NEAREST_NEIGHBOR",
                sampling_mode="DEFAULT",
            )

        x_val = np.array([-6.174, 9.371], dtype=np.float32).reshape([1, 1, 1, 2, 1])
        input_placeholder_dict = {"x": mb.placeholder(shape=x_val.shape)}
        input_value_dict = {"x": x_val}
        expected_output_types = [(1, 1, 1, 3, 2, types.fp32)]
        expected_outputs = [
            np.array([[-6.174, -6.174, 9.371, 9.371, 9.371, 9.371]], dtype=np.float32).reshape(
                [1, 1, 1, 3, 2]
            )
        ]

        run_compare_builder(
            build_model,
            input_placeholder_dict,
            input_value_dict,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_resize_nearest_neighbor_dynamic_shape(self, compute_unit, backend):
        def build_model(x, shape):
            return mb.resize(
                x=x,
                shape=shape,
                resized_dims=np.uint32(2),
                interpolation_mode="NEAREST_NEIGHBOR",
                sampling_mode="DEFAULT",
            )

        x_val = np.array([-6.174, 9.371], dtype=np.float32).reshape([1, 1, 2, 1])
        shape_val = np.array([1, 1, 3, 2], dtype=np.int32)
        input_placeholder_dict = {
            "x": mb.placeholder(shape=x_val.shape, dtype=types.fp32),
            "shape": mb.placeholder(shape=shape_val.shape, dtype=types.int32),
        }
        input_value_dict = {"x": x_val, "shape": shape_val}
        expected_output_types = [(1, 1, UNK_SYM, UNK_SYM, types.fp32)]
        expected_outputs = [
            np.array([[-6.174, -6.174, 9.371, 9.371, 9.371, 9.371]], dtype=np.float32).reshape(
                [1, 1, 3, 2]
            )
        ]

        run_compare_builder(
            build_model,
            input_placeholder_dict,
            input_value_dict,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_resize_linear(self, compute_unit, backend):
        def build_model(x):
            return mb.resize(
                x=x,
                shape=[1, 1, 5],
                resized_dims=np.uint32(2),
                interpolation_mode="LINEAR",
                sampling_mode="DEFAULT",
            )

        x_val = np.array([0, 1], dtype=np.float32).reshape([1, 1, 2])
        input_placeholder_dict = {"x": mb.placeholder(shape=x_val.shape)}
        input_value_dict = {"x": x_val}
        expected_output_types = [(1, 1, 5, types.fp32)]
        expected_outputs = [np.array([[0, 0.4, 0.8, 1, 1]], dtype=np.float32).reshape([1, 1, 5])]

        run_compare_builder(
            build_model,
            input_placeholder_dict,
            input_value_dict,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_resize_linear_dynamic_shape(self, compute_unit, backend):
        def build_model(x, shape):
            return mb.resize(
                x=x,
                shape=shape,
                resized_dims=np.uint32(2),
                interpolation_mode="LINEAR",
                sampling_mode="DEFAULT",
            )

        x_val = np.array([0, 1], dtype=np.float32).reshape([1, 1, 1, 2])
        shape_val = np.array([3, 1, 5], dtype=np.int32)
        input_placeholder_dict = {
            "x": mb.placeholder(shape=x_val.shape, dtype=types.fp32),
            "shape": mb.placeholder(shape=shape_val.shape, dtype=types.int32),
        }
        input_value_dict = {"x": x_val, "shape": shape_val}
        expected_output_types = [(1, 1, UNK_SYM, UNK_SYM, types.fp32)]
        expected_outputs = [np.array([[0, 0.4, 0.8, 1, 1]], dtype=np.float32).reshape([1, 1, 1, 5])]

        run_compare_builder(
            build_model,
            input_placeholder_dict,
            input_value_dict,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_resize_invalid_parameter(self, compute_unit, backend):
        def build_invalid_interpolation_mode(x):
            return mb.resize(
                x=x,
                shape=[1, 1, 5],
                resized_dims=np.uint32(2),
                interpolation_mode="DUMMY",
                sampling_mode="DEFAULT",
            )

        def build_invalid_sampling_mode(x):
            return mb.resize(
                x=x,
                shape=[1, 1, 5],
                resized_dims=np.uint32(2),
                interpolation_mode="LINEAR",
                sampling_mode="DUMMY",
            )

        def build_invalid_target_shape(x):
            return mb.resize(
                x=x,
                shape=[1, 1, 1, 5],
                resized_dims=np.uint32(2),
                interpolation_mode="LINEAR",
                sampling_mode="DEFAULT",
            )

        x_val = np.array([0, 1], dtype=np.float32).reshape([1, 1, 2])
        input_placeholder_dict = {"x": mb.placeholder(shape=x_val.shape)}
        input_value_dict = {"x": x_val}

        with pytest.raises(ValueError, match="Invalid interpolation_mode"):
            run_compare_builder(
                build_invalid_interpolation_mode,
                input_placeholder_dict,
                input_value_dict,
                compute_unit=compute_unit,
                backend=backend,
            )

        with pytest.raises(ValueError, match="Invalid sampling_mode"):
            run_compare_builder(
                build_invalid_sampling_mode,
                input_placeholder_dict,
                input_value_dict,
                compute_unit=compute_unit,
                backend=backend,
            )

        with pytest.raises(ValueError, match="The shape's size \(4\) must <= x's rank \(3\)"):
            run_compare_builder(
                build_invalid_target_shape,
                input_placeholder_dict,
                input_value_dict,
                compute_unit=compute_unit,
                backend=backend,
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, interpolation_mode",
        itertools.product(compute_units, backends, ("LINEAR",)),
    )
    def test_resize_inherit_shape(self, compute_unit, backend, interpolation_mode):
        def build_model(x):
            return mb.resize(
                x=x,
                shape=[1, 0, 0, 0],
                resized_dims=np.uint32(3),
                interpolation_mode=interpolation_mode,
                sampling_mode="DEFAULT",
            )

        pytest.xfail("rdar://112418424 Backend failed when input shape has 0.")

        x_val = np.array([-6.174, 9.371], dtype=np.float32).reshape([1, 1, 1, 2, 1])
        input_placeholder_dict = {"x": mb.placeholder(shape=x_val.shape)}
        input_value_dict = {"x": x_val}
        expected_output_types = [(1, 1, 1, 2, 1, types.fp32)]
        expected_outputs = [np.array([-6.174, 9.371], dtype=np.float32).reshape([1, 1, 1, 2, 1])]

        run_compare_builder(
            build_model,
            input_placeholder_dict,
            input_value_dict,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, interpolation_mode",
        itertools.product(compute_units, backends, ("LINEAR", "NEAREST_NEIGHBOR")),
    )
    def test_resize_inherit_shape_dynamic(self, compute_unit, backend, interpolation_mode):
        def build_model(x, shape):
            return mb.resize(
                x=x,
                shape=shape,
                resized_dims=np.uint32(2),
                interpolation_mode=interpolation_mode,
                sampling_mode="DEFAULT",
            )

        pytest.xfail("rdar://112418424 Backend failed when input shape has 0.")

        x_val = np.array([0, 1], dtype=np.float32).reshape([1, 1, 1, 2])
        shape_val = np.array([1, 0, 0], dtype=np.int32)
        input_placeholder_dict = {
            "x": mb.placeholder(shape=x_val.shape, dtype=types.fp32),
            "shape": mb.placeholder(shape=shape_val.shape, dtype=types.int32),
        }
        input_value_dict = {"x": x_val, "shape": shape_val}
        expected_output_types = [(1, 1, UNK_SYM, UNK_SYM, types.fp32)]
        expected_outputs = [np.array([[0, 1]], dtype=np.float32).reshape([1, 1, 1, 2])]

        run_compare_builder(
            build_model,
            input_placeholder_dict,
            input_value_dict,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )
