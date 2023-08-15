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
from coremltools.converters.mil.mil.ops.tests.iOS15 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import (
    mark_api_breaking,
    run_compare_builder,
)
from coremltools.converters.mil.testing_reqs import compute_units


class TestReshape:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_reshape_with_zero(self, compute_unit, backend):
        t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return [
                mb.reshape(x=x, shape=[0, -1]),
                mb.reshape(x=x, shape=[0, 3]),
                mb.reshape(x=x, shape=[-1, 0]),
            ]

        expected_output_types = [
            (2, 3, types.fp32),
            (2, 3, types.fp32),
            (2, 3, types.fp32),
        ]
        expected_outputs = [
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
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
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_reshape_with_zero_different_len(self, compute_unit, backend):
        t = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return [
                mb.reshape(x=x, shape=[1, 0, -1, 0]),
            ]

        expected_output_types = [
            (1, 1, 2, 3, types.fp32),
        ]
        expected_outputs = [
            np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.float32),
        ]

        with pytest.raises(
            ValueError,
            match="When there is 0 in shape, the rank of x .* must "
            "equal to the target shape len",
        ):
            run_compare_builder(
                build,
                input_placeholders,
                input_values,
                expected_output_types,
                expected_outputs,
                compute_unit=compute_unit,
                backend=backend,
            )
