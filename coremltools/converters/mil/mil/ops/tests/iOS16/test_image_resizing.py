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
from coremltools.converters.mil.mil.ops.tests.iOS15.test_image_resizing import (
    TestResample as _TestResample_iOS15,
)
from coremltools.converters.mil.mil.ops.tests.iOS16 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import (
    mark_api_breaking,
    run_compare_builder,
)
from coremltools.converters.mil.testing_reqs import compute_units


class TestUpsampleBilinear:
    @pytest.mark.parametrize(
        "compute_unit, backend, align_corners, half_pixel_centers",
        itertools.product(
            compute_units,
            backends,
            [True, False],
            [True, False, None],
        ),
    )
    def test_builder_to_backend_smoke_iOS16(
        self, compute_unit, backend, align_corners, half_pixel_centers
    ):
        if align_corners and half_pixel_centers:
            pytest.skip("Invalid configuration of align_corners and half_pixel_centers")

        x = np.array([1, 2], dtype=np.float32).reshape(1, 1, 1, 2)
        input_placeholder_dict = {"x": mb.placeholder(shape=x.shape)}
        input_value_dict = {"x": x}

        def build_upsample_bilinear(x):
            return mb.upsample_bilinear(
                x=x,
                scale_factor_height=2,
                scale_factor_width=3,
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
            )

        expected_output_type = (1, 1, 2, 6, types.fp32)

        if half_pixel_centers is None:
            half_pixel_centers = not align_corners

        if align_corners and not half_pixel_centers:
            expected_output = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        elif not align_corners and half_pixel_centers:
            expected_output = [
                1.0,
                1.0,
                1.33334,
                1.66667,
                2.0,
                2.0,
                1.0,
                1.0,
                1.33334,
                1.66667,
                2.0,
                2.0,
            ]
        elif not align_corners and not half_pixel_centers:
            expected_output = [
                1.0,
                1.33334,
                1.66667,
                2.0,
                2.0,
                2.0,
                1.0,
                1.33334,
                1.66667,
                2.0,
                2.0,
                2.0,
            ]
        else:
            raise ValueError("align_corners and half_pixel_centers cannot be both True")

        expected_output = [np.array(expected_output, dtype=np.float32).reshape(1, 1, 2, 6)]

        run_compare_builder(
            build_upsample_bilinear,
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
        "compute_unit, backend, pad_value",
        itertools.product(compute_units, backends, [0.0, 1.0, 10.0]),
    )
    def test_builder_to_backend_ios16(self, compute_unit, backend, pad_value):
        """For iOS16+ the crop_resize op supports pad_value."""
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)

        roi = np.array(
            [
                [0, 0.1, 0.3, 1.3, 1],
                [0, 0.5, 1.8, 1.0, 0.3],
                [0, 0.0, 0.4, 0.6, 0.7],
            ],
            dtype=np.float32,
        ).reshape(3, 1, 5, 1, 1)

        def build(x):
            return mb.crop_resize(
                x=x,
                roi=roi,
                target_width=2,
                target_height=2,
                normalized_coordinates=True,
                box_coordinate_mode="CORNERS_HEIGHT_FIRST",
                sampling_mode="ALIGN_CORNERS",
                pad_value=pad_value,
            )

        expected_output_type = [
            (3, 1, 1, 2, 2, types.fp32),
        ]
        expected_output = [
            np.array(
                [
                    3.1,
                    5.2,
                    pad_value,
                    pad_value,
                    pad_value,
                    7.899,
                    pad_value,
                    13.9,
                    2.2,
                    3.1,
                    9.4,
                    10.3,
                ],
                dtype=np.float32,
            ).reshape(3, 1, 1, 2, 2),
        ]

        input_placeholder_dict = {"x": mb.placeholder(shape=(1, 1, 4, 4))}
        input_value_dict = {"x": x}

        run_compare_builder(
            build,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestResample:
    @pytest.mark.parametrize(
        "compute_unit, backend, coordinates_dtype",
        itertools.product(
            compute_units,
            backends,
            (np.int32, np.float16, np.float32),
        ),
    )
    def test_builder_to_backend_smoke_iOS16(self, compute_unit, backend, coordinates_dtype):
        # The fp16 precision will have two casts inserted for input/output
        expected_cast_ops = 2 if backend.precision == "fp16" else 0
        if backend.precision == "fp16" and coordinates_dtype == np.float32:
            # The coordinates also cast to fp16.
            expected_cast_ops += 1
        _TestResample_iOS15._test_builder_to_backend_smoke(
            compute_unit, backend, coordinates_dtype, expected_cast_ops
        )
