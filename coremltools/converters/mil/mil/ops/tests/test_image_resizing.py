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
from coremltools.converters.mil.testing_reqs import backends, compute_units
from coremltools.converters.mil.testing_utils import random_gen
from coremltools.models.utils import _macos_version

from .testing_utils import run_compare_builder

if _HAS_TORCH:
    import torch


class TestAffine:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        if backend[0] == "neuralnetwork":
            pytest.skip("nn backend not supported")

        x_val = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32).reshape(
            [1, 1, 2, 2]
        )
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


class TestResample:
    @pytest.mark.parametrize(
        "compute_unit, backend, minimum_deployment_target",
        itertools.product(
            compute_units,
            backends,
            [ct.target.iOS15, ct.target.iOS16],
        )
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, minimum_deployment_target):
        if backend[0] == "neuralnetwork":
            pytest.skip("nn backend not supported")
        if minimum_deployment_target == ct.target.iOS16 and _macos_version() < (13, 0):
            pytest.skip("New functionality in macOS13/iOS16")

        x_ = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32).reshape([1, 1, 2, 2])
        coordinates_ = np.array(
            [-1.0, -2.0, -3.7, -1.0, 0.0, 0.0, 3.5, 1.2], dtype=np.float32
        ).reshape([1, 2, 2, 2])

        input_placeholder_dict = {
            "x": mb.placeholder(shape=x_.shape),
            "coordinates": mb.placeholder(shape=coordinates_.shape),
        }
        input_value_dict = {"x": x_, "coordinates": coordinates_}
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

        expected_output_0 = np.array(
            [8.585, 6.17, 27.5, 6.17], dtype=np.float32
        ).reshape(expected_output_type[:-1])

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

        expected_output_1 = np.array(
            [11.0, 11.0, 11.0, 44.0], dtype=np.float32
        ).reshape(expected_output_type[:-1])

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

        expected_output_2 = np.array(
            [22.0, 36.3, 11.0, 34.1], dtype=np.float32
        ).reshape(expected_output_type[:-1])

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

        expected_output_3 = np.array(
            [22.0, 33.0, 11.0, 33.0], dtype=np.float32
        ).reshape(expected_output_type[:-1])

        for build, expected_output in zip(
            [build_0, build_1, build_2, build_3],
            [
                expected_output_0,
                expected_output_1,
                expected_output_2,
                expected_output_3,
            ],
        ):
            mlmodel = run_compare_builder(
                build,
                input_placeholder_dict,
                input_value_dict,
                expected_output_type,
                expected_output,
                compute_unit=compute_unit,
                backend=backend,
                minimum_deployment_target=minimum_deployment_target,
            )
            prog = mlmodel._mil_program
            number_of_cast = len(prog["main"].find_ops(op_type="cast"))
            # for the new iOS16 resample op, the coordinates is cast to fp16
            if minimum_deployment_target == ct.target.iOS15:
                assert number_of_cast == 2
            elif minimum_deployment_target == ct.target.iOS16:
                assert number_of_cast == 3
            else:
                raise ValueError("Unrecognized target {}".format(minimum_deployment_target))


class TestResizeNearestNeighbor:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x_val = np.array([0.37, 6.17], dtype=np.float32).reshape([1, 1, 2, 1])
        input_placeholder_dict = {"x": mb.placeholder(shape=x_val.shape)}
        input_value_dict = {"x": x_val}

        def build_model(x):
            return [
                mb.resize_nearest_neighbor(
                    x=x, target_size_height=2, target_size_width=1,
                ),
                mb.resize_nearest_neighbor(
                    x=x, target_size_height=2, target_size_width=3,
                ),
            ]

        expected_output_types = [
            (1, 1, 2, 1, types.fp32),
            (1, 1, 2, 3, types.fp32),
        ]
        expected_outputs = [
            x_val,
            np.array([0.37, 0.37, 0.37, 6.17, 6.17, 6.17], dtype=np.float32).reshape(
                [1, 1, 2, 3]
            ),
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


class TestUpsampleNearestNeighborFractionalScales:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        if backend[0] == "neuralnetwork":
            pytest.skip("nn backend not supported")
        
        if backend[0] == "mlprogram" and compute_unit != ct.ComputeUnit.CPU_ONLY:
            pytest.xfail("rdar://97398448 (TestUpsampleNearestNeighborFractionalScales failing on GPU)")

        x_val = np.array([1.5, -2.5, 3.5], dtype=np.float32).reshape([1, 1, 1, 3])
        input_placeholder_dict = {"x": mb.placeholder(shape=x_val.shape)}
        input_value_dict = {"x": x_val}

        def build(x):
            return [
                mb.upsample_nearest_neighbor(
                    x=x, scale_factor_height=1.0, scale_factor_width=1.0,
                ),
                mb.upsample_nearest_neighbor(
                    x=x, scale_factor_height=3.17, scale_factor_width=0.67
                ),
                mb.upsample_nearest_neighbor(
                    x=x, scale_factor_height=2.0, scale_factor_width=1.12,
                ),
            ]

        expected_output_types = [
            (1, 1, 1, 3, types.fp32),
            (1, 1, 3, 2, types.fp32),
            (1, 1, 2, 3, types.fp32),
        ]
        expected_outputs = [
            x_val,
            np.array([1.5, -2.5, 1.5, -2.5, 1.5, -2.5], dtype=np.float32).reshape(
                [1, 1, 3, 2]
            ),
            np.array([1.5, -2.5, 3.5, 1.5, -2.5, 3.5], dtype=np.float32).reshape(
                [1, 1, 2, 3]
            ),
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


class TestResizeBilinear:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        if backend[0] == "mlprogram":
            pytest.xfail("Seg fault: rdar://78343191 ((MIL GPU) Core ML Tools Unit Test failures [failure to load or Seg fault])")

        if backend[0] == "neuralnetwork" and compute_unit == ct.ComputeUnit.CPU_ONLY:
            pytest.xfail("rdar://85318710 (Coremltools Smoke test on ResizeBilinear failing on NNv1 backend.)")

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
        expected_output = np.array([0, 0.25, 0.5, 0.75, 1], dtype=np.float32).reshape(
            1, 1, 5
        )

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

        expected_output = np.array([0, 0.4, 0.8, 1, 1], dtype=np.float32).reshape(
            1, 1, 5
        )

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

        expected_output = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32).reshape(
            1, 1, 5
        )

        run_compare_builder(
            build_mode_3,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

        if backend[0] != "neuralnetwork":
            def build_mode_4(x):
                return mb.resize_bilinear(
                    x=x,
                    target_size_height=1,
                    target_size_width=5,
                    sampling_mode="UNALIGN_CORNERS",
                )

            expected_output = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float32).reshape(
                1, 1, 5
            )

            run_compare_builder(
                build_mode_4,
                input_placeholder_dict,
                input_value_dict,
                expected_output_type,
                expected_output,
                compute_unit=compute_unit,
                backend=backend,
            )


class TestUpsampleBilinear:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([0, 1], dtype=np.float32).reshape(1, 1, 2)
        input_placeholder_dict = {"x": mb.placeholder(shape=x.shape)}
        input_value_dict = {"x": x}

        def build_upsample_integer(x):
            return mb.upsample_bilinear(
                x=x, scale_factor_height=1, scale_factor_width=3
            )

        expected_output_type = (1, 1, 6, types.fp32)
        expected_output = np.array(
            [0, 0.2, 0.4, 0.6, 0.8, 1], dtype=np.float32
        ).reshape(1, 1, 6)

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
        expected_output = np.array([0, 0.1, 0.5, 0.9, 1], dtype=np.float32).reshape(
            1, 1, 5
        )

        run_compare_builder(
            build_upsample_fractional,
            input_placeholder_dict,
            input_value_dict,
            expected_output_type,
            expected_output,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, align_corners, half_pixel_centers",
        itertools.product(
            compute_units,
            backends,
            [True, False],
            [True, False],
        )
    )
    def test_builder_to_backend_smoke_iOS16(self, compute_unit, backend, align_corners, half_pixel_centers):
        if backend[0] == "neuralnetwork" or ct.utils._macos_version() < (13, 0):
            pytest.skip("The new half_pixel_centers argument only available in iOS16")

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
        
        if align_corners and not half_pixel_centers:
            expected_output = [1., 1.2, 1.4, 1.6, 1.8, 2., 1., 1.2, 1.4, 1.6, 1.8, 2.]
        elif not align_corners and half_pixel_centers:
            expected_output = [1., 1., 1.33334, 1.66667, 2., 2., 1., 1., 1.33334, 1.66667, 2., 2.]
        elif not align_corners and not half_pixel_centers:
            expected_output = [1., 1.33334, 1.66667, 2., 2., 2., 1., 1.33334, 1.66667, 2., 2., 2.]
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
            minimum_deployment_target=ct.target.iOS16,
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
        self, compute_unit, backend, input_shape, scale_factor, align_corners, recompute_scale_factor
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

        def _get_torch_upsample_prediction(x, scale_factor=(2, 2), align_corners=False, recompute_scale_factor=True):
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
        )


class TestUpsampleNearestNeighbor:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x = np.array([1.5, 2.5, 3.5], dtype=np.float32).reshape([1, 1, 1, 3])
        input_placeholder_dict = {"x": mb.placeholder(shape=x.shape)}
        input_value_dict = {"x": x}

        def build(x):
            return mb.upsample_nearest_neighbor(
                x=x, scale_factor_height=1, scale_factor_width=2
            )

        expected_output_type = (1, 1, 1, 6, types.fp32)
        expected_output = np.array(
            [1.5, 1.5, 2.5, 2.5, 3.5, 3.5], dtype=np.float32
        ).reshape([1, 1, 1, 6])

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
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_builder_to_backend_smoke_pad_value(self, compute_unit, backend):
        if backend[0] == "neuralnetwork":
            pytest.skip("pad_mode only supported on iOS16 or above")
            
        if ct.utils._macos_version() < (13, 0):
            pytest.skip("pad_value not supported in macOS12 or older.")

        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32,
        ).reshape(1, 1, 4, 4)

        roi = np.array([
            [0, 0.1, 0.3, 1.3, 1],
            [0, 0.5, 1.8, 1., 0.3],
            [0, 0.0, 0.4, 0.6, 0.7],
        ], dtype=np.float32).reshape(3, 1, 5, 1, 1)
        
        def build(x):
            return mb.crop_resize(
                    x=x,
                    roi=roi,
                    target_width=2,
                    target_height=2,
                    normalized_coordinates=True,
                    box_coordinate_mode="CORNERS_HEIGHT_FIRST",
                    sampling_mode="ALIGN_CORNERS",
                    pad_value=10.0,
            )
        
        expected_output_type = [
            (3, 1, 1, 2, 2, types.fp32),
        ]
        expected_output = [
            np.array([ 3.1, 5.2, 10, 10, 10, 7.899, 10, 13.9, 2.2, 3.1, 9.4, 10.3], dtype=np.float32).reshape(3, 1, 1, 2, 2),
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
            minimum_deployment_target=ct.target.iOS16,
        )
        
        
    @pytest.mark.parametrize(
        "compute_unit, backend, is_symbolic",
        itertools.product(compute_units, backends, compute_units),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, is_symbolic):
        if backend[0] == "mlprogram" and compute_unit != ct.ComputeUnit.CPU_ONLY:
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
        roi_normalized = np.array(
            [[0, 0.0, 0.0, 1.0 / 3, 1.0 / 3]], dtype=np.float32
        ).reshape(1, 1, 5, 1, 1)
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

            elif mode == 5:
                return mb.crop_resize(
                        x=x,
                        roi=roi_invert,
                        target_width=2,
                        target_height=2,
                        normalized_coordinates=True,
                        box_coordinate_mode="CORNERS_HEIGHT_FIRST",
                        sampling_mode="UNALIGN_CORNERS",
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
            np.array([3.5, 5.5, 11.5, 13.5], dtype=np.float32).reshape(1, 1, 1, 2, 2),
        ]

        for mode in range(6):
            # nn-proto does not support UNALIGN_CORNERS
            if not (backend[0] == 'neuralnetwork' and mode == 5):
                run_compare_builder(
                    functools.partial(build, mode=mode),
                    input_placeholder_dict,
                    input_value_dict,
                    expected_output_type[mode],
                    expected_output[mode],
                    compute_unit=compute_unit,
                    backend=backend,
                )
