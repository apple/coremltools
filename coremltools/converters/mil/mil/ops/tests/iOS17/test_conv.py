#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools._deps import _HAS_TORCH, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil.mil.ops.tests.iOS14.test_conv import TestConv as _TestConvIos14
from coremltools.converters.mil.mil.ops.tests.iOS14.test_conv import (
    TestConvTranspose as _TestTestConvTransposeIos14,
)
from coremltools.converters.mil.mil.ops.tests.iOS17 import backends
from coremltools.converters.mil.testing_reqs import compute_units


class TestConv(_TestConvIos14):
    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "conv_dim",
                "config",
                "x_weight_dtype",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            ["conv1d", "conv2d", "conv3d"],
            [
                {
                    "padding": (1, 1, 1),
                    "DHWKdKhKw": (10, 12, 14, 3, 2, 4),
                    "stride": (2, 1, 1),
                    "dilation": (1, 1, 1),
                    "has_bias": False,
                    "groups": 1,
                    "symbolic": False,
                },
                {
                    "padding": (2, 2, 2),
                    "DHWKdKhKw": (10, 12, 14, 3, 2, 4),
                    "stride": (2, 2, 2),
                    "dilation": (2, 1, 1),
                    "has_bias": False,
                    "groups": 2,
                    "symbolic": True,
                },
                {
                    "padding": (1, 1, 1),
                    "DHWKdKhKw": (5, 5, 5, 2, 2, 2),
                    "stride": (2, 2, 2),
                    "dilation": (2, 1, 1),
                    "has_bias": True,
                    "groups": 1,
                    "symbolic": True,
                },
                {
                    "padding": (2, 2, 2),
                    "DHWKdKhKw": (5, 5, 5, 2, 2, 2),
                    "stride": (2, 1, 1),
                    "dilation": (1, 1, 1),
                    "has_bias": True,
                    "groups": 2,
                    "symbolic": False,
                },
            ],
            [
                (np.float32, np.float32),
                (np.float16, np.float16),
                (np.float16, np.float32),
                (np.float32, np.float16),
            ],
        ),
    )
    def test_builder_to_backend_stress(
        self,
        compute_unit,
        backend,
        conv_dim,
        config,
        x_weight_dtype,
    ):
        if (
            backend.backend == "mlprogram"
            and backend.precision == "fp16"
            and backend.opset_version == ct.target.iOS17
            and conv_dim == "conv2d"
            and config
            == {
                "padding": (1, 1, 1),
                "DHWKdKhKw": (5, 5, 5, 2, 2, 2),
                "stride": (2, 2, 2),
                "dilation": (2, 1, 1),
                "has_bias": True,
                "groups": 1,
                "symbolic": True,
            }
            and x_weight_dtype == (np.float32, np.float16)
        ):
            pytest.xfail("rdar://124260627 ([CI] Two tests are random failing on CI)")

        super().test_builder_to_backend_stress(
            compute_unit, backend, conv_dim, config, x_weight_dtype
        )


class TestConvTranspose(_TestTestConvTransposeIos14):
    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "conv_dim",
                "config",
                "x_weight_dtype",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            ["conv1d", "conv2d", "conv3d"],
            [
                {
                    "padding": (1, 2, 3),
                    "DHWKdKhKw": (10, 12, 14, 3, 2, 4),
                    "stride": (2, 1, 1),
                    "dilation": (1, 1, 1),
                    "has_bias": False,
                    "groups": 1,
                    "test_symbolic": False,
                    "test_output_shape": True,
                },
                {
                    "padding": (2, 2, 2),
                    "DHWKdKhKw": (10, 12, 14, 3, 2, 4),
                    "stride": (2, 2, 2),
                    "dilation": (2, 1, 1),
                    "has_bias": False,
                    "groups": 2,
                    "test_symbolic": True,
                    "test_output_shape": False,
                },
                {
                    "padding": (1, 2, 3),
                    "DHWKdKhKw": (7, 7, 7, 2, 2, 2),
                    "stride": (2, 2, 2),
                    "dilation": (2, 1, 1),
                    "has_bias": True,
                    "groups": 1,
                    "test_symbolic": True,
                    "test_output_shape": False,
                },
                {
                    "padding": (2, 2, 2),
                    "DHWKdKhKw": (7, 7, 7, 2, 2, 2),
                    "stride": (2, 1, 1),
                    "dilation": (1, 1, 1),
                    "has_bias": True,
                    "groups": 2,
                    "test_symbolic": False,
                    "test_output_shape": False,
                },
            ],
            [
                (np.float32, np.float32),
                (np.float16, np.float16),
                (np.float16, np.float32),
                (np.float32, np.float16),
            ],
        ),
    )
    def test_builder_to_backend_stress(
        self,
        compute_unit,
        backend,
        conv_dim,
        config,
        x_weight_dtype,
    ):
        super().test_builder_to_backend_stress(
            compute_unit, backend, conv_dim, config, x_weight_dtype
        )
