#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools._deps import _HAS_TF_2, _HAS_TORCH, MSG_TF2_NOT_FOUND, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil.mil.ops.tests.iOS14.test_normalization import (
    TestNormalizationBatchNorm as _TestNormalizationBatchNormIos14,
)
from coremltools.converters.mil.mil.ops.tests.iOS14.test_normalization import (
    TestNormalizationInstanceNorm as _TestNormalizationInstanceNormIos14,
)
from coremltools.converters.mil.mil.ops.tests.iOS14.test_normalization import (
    TestNormalizationL2Norm as _TestNormalizationL2NormIos14,
)
from coremltools.converters.mil.mil.ops.tests.iOS14.test_normalization import (
    TestNormalizationLayerNorm as _TestNormalizationLayerNormIos14,
)
from coremltools.converters.mil.mil.ops.tests.iOS14.test_normalization import (
    TestNormalizationLocalResponseNorm as _TestNormalizationLocalResponseNormIos14,
)
from coremltools.converters.mil.mil.ops.tests.iOS17 import backends
from coremltools.converters.mil.testing_reqs import compute_units


class TestNormalizationBatchNorm(_TestNormalizationBatchNormIos14):
    @pytest.mark.parametrize(
        "compute_unit, backend, x_param_dtype",
        itertools.product(
            compute_units,
            backends,
            [
                (np.float16, np.float16),
                (np.float32, np.float32),
                (np.float16, np.float32),
                (np.float32, np.float16),
            ],
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, x_param_dtype):
        super().test_builder_to_backend_smoke(compute_unit, backend, x_param_dtype)


class TestNormalizationInstanceNorm(_TestNormalizationInstanceNormIos14):
    @pytest.mark.parametrize(
        "compute_unit, backend, x_param_dtype",
        itertools.product(
            compute_units,
            backends,
            [
                (np.float16, np.float16),
                (np.float32, np.float32),
                (np.float16, np.float32),
                (np.float32, np.float16),
            ],
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, x_param_dtype):
        super().test_builder_to_backend_smoke(compute_unit, backend, x_param_dtype)

    @pytest.mark.parametrize(
        "compute_unit, backend, x_param_dtype",
        itertools.product(
            compute_units,
            backends,
            [
                (np.float16, np.float16),
                (np.float32, np.float32),
                (np.float16, np.float32),
                (np.float32, np.float16),
            ],
        ),
    )
    def test_builder_to_backend_smoke_with_gamma_and_beta(
        self, compute_unit, backend, x_param_dtype
    ):
        super().test_builder_to_backend_smoke_with_gamma_and_beta(
            compute_unit, backend, x_param_dtype
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        "rank, compute_unit, backend, epsilon, x_param_dtype",
        itertools.product(
            [3, 4],
            compute_units,
            backends,
            [1e-3, 1e-5, 1e-10],
            [
                (np.float16, np.float16),
                (np.float32, np.float32),
                (np.float16, np.float32),
                (np.float32, np.float16),
            ],
        ),
    )
    def test_builder_to_backend_stress(self, rank, compute_unit, backend, epsilon, x_param_dtype):
        super().test_builder_to_backend_stress(rank, compute_unit, backend, epsilon, x_param_dtype)


class TestNormalizationL2Norm(_TestNormalizationL2NormIos14):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, epsilon, x_param_dtype",
        itertools.product(
            compute_units,
            backends,
            [3, 4, 5],
            [1e-4, 5.7],
            [
                (np.float16, np.float16),
                (np.float32, np.float32),
                (np.float16, np.float32),
                (np.float32, np.float16),
            ],
        ),
    )
    def test_builder_to_backend_stress(self, compute_unit, backend, rank, epsilon, x_param_dtype):
        super().test_builder_to_backend_stress(compute_unit, backend, rank, epsilon, x_param_dtype)


class TestNormalizationLayerNorm(_TestNormalizationLayerNormIos14):
    @pytest.mark.skipif(not _HAS_TF_2, reason=MSG_TF2_NOT_FOUND)
    @pytest.mark.parametrize(
        "compute_unit, backend, rank_and_axes, epsilon, x_param_dtype",
        itertools.product(
            compute_units,
            backends,
            [[3, [0, 2]], [3, [-2]], [4, [0, 1, 3]], [5, [0, 4]], [5, [-5, -4, -3, -2, -1]]],
            [0.0001, 0.01],
            [
                (np.float16, np.float16),
                (np.float32, np.float32),
                (np.float16, np.float32),
                (np.float32, np.float16),
            ],
        ),
    )
    def test_builder_to_backend_stress_keras(
        self, compute_unit, backend, rank_and_axes, epsilon, x_param_dtype
    ):
        super().test_builder_to_backend_stress_keras(
            compute_unit, backend, rank_and_axes, epsilon, x_param_dtype
        )


class TestNormalizationLocalResponseNorm(_TestNormalizationLocalResponseNormIos14):
    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, size, alpha, beta, k, x_param_dtype",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(3, 6)],
            [2, 3, 5],
            [0.0001, 0.01],
            [0.75, 1.0],
            [1.0, 2.0],
            [
                (np.float16, np.float16),
                (np.float32, np.float32),
                (np.float16, np.float32),
                (np.float32, np.float16),
            ],
        ),
    )
    def test_builder_to_backend_stress(
        self, compute_unit, backend, rank, size, alpha, beta, k, x_param_dtype
    ):
        super().test_builder_to_backend_stress(
            compute_unit, backend, rank, size, alpha, beta, k, x_param_dtype
        )
