#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import functools
import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools._deps import _HAS_TORCH, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.ops.tests.iOS14 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import (
    mark_api_breaking,
    run_compare_builder,
)
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import random_gen

if _HAS_TORCH:
    import torch

class TestResizeNearestNeighbor:
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x_val = np.array([0.37, 6.17], dtype=np.float32).reshape([1, 1, 2, 1])
        input_placeholder_dict = {"x": mb.placeholder(shape=x_val.shape)}
        input_value_dict = {"x": x_val}

        def build_model(x):
            return [
                mb.resize_nearest_neighbor(
                    x=x,
                    target_size_height=2,
                    target_size_width=1,
                ),
                mb.resize_nearest_neighbor(
                    x=x,
                    target_size_height=2,
                    target_size_width=3,
                ),
            ]

        expected_output_types = [
            (1, 1, 2, 1, types.fp32),
            (1, 1, 2, 3, types.fp32),
        ]
        expected_outputs = [
            x_val,
            np.array([0.37, 0.37, 0.37, 6.17, 6.17, 6.17], dtype=np.float32).reshape([1, 1, 2, 3]),
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

class TestResizeBilinear:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        if backend.backend == "mlprogram":
            pytest.xfail(
                "Seg fault: rdar://78343191 ((MIL GPU) Core ML Tools Unit Test failures [failure to load or Seg fault])"
            )

        if backend.backend == "neuralnetwork" and compute_unit == ct.ComputeUnit.CPU_ONLY:
            pytest.xfail(
                "rdar://85318710 (Coremltools Smoke test on ResizeBilinear failing on NNv1 backend.)"
            )

        x = np.array([0, 1], dtype=np.float32).reshape(1, 1, 2)
        input_placeholder_dict = {"x": mb.placeholder(shape=x.shape)}
        input_value_dict = {"x": x}

        def build_mode_0(x):
            return mb.resize_bilinear(
                x=x,
                target_size_height=1,
                target_size_width=5,
                sampling_mode="STRICT_ALIGN_CORNERS",
            )

        expected_output_type = (1, 1, 5, types.fp32)
        expected_output = np.array([0, 0.25, 0.5, 0.75, 1], dtype=np.float32).reshape(1, 1, 5)

        run_compare_builder(
            build_mode_0,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

        def build_mode_2(x):
            return mb.resize_bilinear(
                x=x, target_size_height=1, target_size_width=5, sampling_mode="DEFAULT"
            )

        expected_output = np.array([0, 0.4, 0.8, 1, 1], dtype=np.float32).reshape(1, 1, 5)

        run_compare_builder(
            build_mode_2,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

        def build_mode_3(x):
            return mb.resize_bilinear(
                x=x,
                target_size_height=1,
                target_size_width=5,
                sampling_mode="OFFSET_CORNERS",
            )

        expected_output = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32).reshape(1, 1, 5)

        run_compare_builder(
            build_mode_3,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

class TestUpsampleBilinear:
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

        def build_upsample_integer(x):
            return mb.upsample_bilinear(
                x=x, scale_factor_height=1, scale_factor_width=3, align_corners=True
            )

        expected_output_type = (1, 1, 6, types.fp32)
        expected_output = np.array([0, 0.2, 0.4, 0.6, 0.8, 1], dtype=np.float32).reshape(1, 1, 6)

        run_compare_builder(
            build_upsample_integer,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

        def build_upsample_fractional(x):
            return mb.upsample_bilinear(
                x=x, scale_factor_height=1.0, scale_factor_width=2.6, align_corners=False
            )

        expected_output_type = (1, 1, 5, types.fp32)
        expected_output = np.array([0, 0.1, 0.5, 0.9, 1], dtype=np.float32).reshape(1, 1, 5)

        run_compare_builder(
            build_upsample_fractional,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        "compute_unit, backend, input_shape, scale_factor, align_corners, recompute_scale_factor",
        itertools.product(
            compute_units,
            backends,
            [(2, 5, 10, 22)],
            [(3, 4), (2.5, 2.0), (0.5, 0.75)],
            [True, False],
            [True, False],
        ),
    )
    def test_builder_to_backend_stress(
        self,
        compute_unit,
        backend,
        input_shape,
        scale_factor,
        align_corners,
        recompute_scale_factor,
    ):
        scale_factor_height, scale_factor_width = scale_factor
        _, _, height, width = input_shape
        height = height * scale_factor_height
        width = width * scale_factor_width
        is_h_float = height - np.floor(height) > 0.001
        is_w_float = width - np.floor(width) > 0.001

        # Currently, MIL is not suporting recompute_scale_factor=False + align_corners=False
        # with fractional output size
        if not recompute_scale_factor and not align_corners and (is_h_float or is_w_float):
            pytest.xfail("rdar://81124053 (Support recompute_scale_factor)")

        def _get_torch_upsample_prediction(
            x, scale_factor=(2, 2), align_corners=False, recompute_scale_factor=True
        ):
            x = torch.from_numpy(x)
            out = torch.nn.functional.interpolate(
                x,
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=align_corners,
                recompute_scale_factor=recompute_scale_factor,
            )
            return out.numpy()

        x = random_gen(input_shape, rand_min=-100, rand_max=100)
        torch_pred = _get_torch_upsample_prediction(
            x,
            scale_factor=scale_factor,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
        )

        input_placeholder_dict = {"x": mb.placeholder(shape=x.shape)}
        input_value_dict = {"x": x}

        def build_upsample(x):
            return mb.upsample_bilinear(
                x=x,
                scale_factor_height=scale_factor[0],
                scale_factor_width=scale_factor[1],
                align_corners=align_corners,
            )

        expected_output_type = torch_pred.shape + (types.fp32,)
        run_compare_builder(
            build_upsample,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            torch_pred,
            compute_unit=compute_unit,
            backend=backend,
            rtol=0.5,
        )


class TestUpsampleNearestNeighbor:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([1.5, 2.5, 3.5], dtype=np.float32).reshape([1, 1, 1, 3])
        input_placeholder_dict = {"x": mb.placeholder(shape=x.shape)}
        input_value_dict = {"x": x}

        def build(x):
            return mb.upsample_nearest_neighbor(x=x, scale_factor_height=1, scale_factor_width=2)

        expected_output_type = (1, 1, 1, 6, types.fp32)
        expected_output = np.array([1.5, 1.5, 2.5, 2.5, 3.5, 3.5], dtype=np.float32).reshape(
            [1, 1, 1, 6]
        )

        run_compare_builder(
            build,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestCrop:
    @pytest.mark.parametrize(
        "compute_unit, backend, is_symbolic",
        itertools.product(compute_units, backends, compute_units),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, is_symbolic):
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

        def build(x):
            return mb.crop(x=x, crop_height=[0, 1], crop_width=[1, 1])

        expected_output_type = (
            placeholder_input_shape[0],
            placeholder_input_shape[1],
            3,
            2,
            types.fp32,
        )
        expected_output = np.array([2, 3, 6, 7, 10, 11], dtype=np.float32).reshape(1, 1, 3, 2)

        run_compare_builder(
            build,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, C, H, W",
        itertools.product(
            compute_units,
            backends,
            [x for x in range(2, 4)],
            [x for x in range(5, 8)],
            [x for x in range(8, 10)],
        ),
    )
    def test_builder_to_backend_stress(self, compute_unit, backend, C, H, W):
        input_shape = (1, C, H, W)
        x = np.random.random(input_shape)

        crop_h = [np.random.randint(H)]
        crop_h.append(np.random.randint(H - crop_h[0]))
        crop_w = [np.random.randint(W)]
        crop_w.append(np.random.randint(W - crop_w[0]))

        input_placeholder_dict = {"x": mb.placeholder(shape=input_shape)}
        input_value_dict = {"x": x}

        def build(x):
            return mb.crop(x=x, crop_height=crop_h, crop_width=crop_w)

        expected_output_type = (
            1,
            C,
            H - crop_h[0] - crop_h[1],
            W - crop_w[0] - crop_w[1],
            types.fp32,
        )
        expected_output = x[:, :, crop_h[0] : H - crop_h[1], crop_w[0] : W - crop_w[1]]

        run_compare_builder(
            build,
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
        if backend.backend == "mlprogram" and compute_unit != ct.ComputeUnit.CPU_ONLY:
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

        def build(x, mode=0):
            if mode == 0:
                return mb.crop_resize(
                    x=x,
                    roi=roi,
                    target_width=2,
                    target_height=2,
                    normalized_coordinates=False,
                    box_coordinate_mode="CORNERS_HEIGHT_FIRST",
                    sampling_mode="ALIGN_CORNERS",
                )

            elif mode == 1:
                return mb.crop_resize(
                    x=x,
                    roi=roi,
                    target_width=4,
                    target_height=4,
                    normalized_coordinates=False,
                    box_coordinate_mode="CORNERS_HEIGHT_FIRST",
                    sampling_mode="ALIGN_CORNERS",
                )

            elif mode == 2:
                return mb.crop_resize(
                    x=x,
                    roi=roi,
                    target_width=1,
                    target_height=1,
                    normalized_coordinates=False,
                    box_coordinate_mode="CORNERS_HEIGHT_FIRST",
                    sampling_mode="ALIGN_CORNERS",
                )

            elif mode == 3:
                return mb.crop_resize(
                    x=x,
                    roi=roi_normalized,
                    target_width=2,
                    target_height=2,
                    normalized_coordinates=True,
                    box_coordinate_mode="CORNERS_HEIGHT_FIRST",
                    sampling_mode="ALIGN_CORNERS",
                )

            elif mode == 4:
                return mb.crop_resize(
                    x=x,
                    roi=roi_invert,
                    target_width=2,
                    target_height=2,
                    normalized_coordinates=False,
                    box_coordinate_mode="CORNERS_HEIGHT_FIRST",
                    sampling_mode="ALIGN_CORNERS",
                )

        expected_output_type = [
            (
                N,
                placeholder_input_shape[0],
                placeholder_input_shape[1],
                2,
                2,
                types.fp32,
            ),
            (
                N,
                placeholder_input_shape[0],
                placeholder_input_shape[1],
                4,
                4,
                types.fp32,
            ),
            (
                N,
                placeholder_input_shape[0],
                placeholder_input_shape[1],
                1,
                1,
                types.fp32,
            ),
            (
                N,
                placeholder_input_shape[0],
                placeholder_input_shape[1],
                2,
                2,
                types.fp32,
            ),
            (
                N,
                placeholder_input_shape[0],
                placeholder_input_shape[1],
                2,
                2,
                types.fp32,
            ),
        ]
        expected_output = [
            np.array([6, 7, 10, 11], dtype=np.float32).reshape(1, 1, 1, 2, 2),
            np.array(
                [
                    [6, 6.333333, 6.66666, 7],
                    [7.333333, 7.666666, 8, 8.333333],
                    [8.666666, 9, 9.3333333, 9.666666],
                    [10, 10.333333, 10.666666, 11],
                ],
                dtype=np.float32,
            ).reshape(1, 1, 1, 4, 4),
            np.array([8.5], dtype=np.float32).reshape(1, 1, 1, 1, 1),
            np.array([1, 2, 5, 6], dtype=np.float32).reshape(1, 1, 1, 2, 2),
            np.array([11, 10, 7, 6], dtype=np.float32).reshape(1, 1, 1, 2, 2),
        ]

        for mode in range(5):
            run_compare_builder(
                functools.partial(build, mode=mode),
                input_placeholder_dict,
                input_value_dict,
                expected_output_type[mode],
                expected_output[mode],
                compute_unit=compute_unit,
                backend=backend,
            )
