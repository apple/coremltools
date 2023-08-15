#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.tests.iOS16 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder

compute_units = testing_reqs.compute_units


class TestFillLike:
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        shape = (2, 1, 3)
        x_val = np.zeros(shape=shape, dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape, dtype=types.int32)}

        input_values = {"x": x_val}

        def build(x):
            return mb.fill_like(ref_tensor=x, value=1.0)

        expected_output_types = [(2, 1, 3, types.fp32)]
        expected_outputs = [np.full(shape=shape, fill_value=1.0)]

        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestTopK:
    @pytest.mark.parametrize(
        "compute_unit, backend, return_indices, sort",
        itertools.product(
            compute_units,
            backends,
            [True, False],
            [True, False],
        ),
    )
    def test_builder_to_backend_smoke_iOS16(self, compute_unit, backend, return_indices, sort):
        val = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return mb.topk(x=x, k=2, axis=1, return_indices=return_indices, sort=sort)

        expected_output_types = [
            (2, 2, types.fp32),
            (2, 2, types.int32),
        ]
        expected_outputs = [
            np.array([[2.0, -1.0], [6.0, 4.0]], dtype=np.float32),
            np.array([[1, 0], [2, 0]], dtype=np.float32),
        ]

        if not return_indices:
            expected_output_types = expected_output_types[:1]
            expected_outputs = expected_outputs[:1]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )
