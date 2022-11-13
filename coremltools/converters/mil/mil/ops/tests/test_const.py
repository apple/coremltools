#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types

from .testing_utils import run_compare_builder

backends = testing_reqs.backends
compute_units = testing_reqs.compute_units


class TestConst:
    @pytest.mark.parametrize(
        "compute_unit, backend, dtype", itertools.product(
            compute_units,
            backends,
            [
                np.int32,
                np.int64,
                np.float16,
                np.float32,
                np.float64,
            ]
        )
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, dtype):
        if backend[0] == "mlprogram" and dtype in [np.uint8, np.int8, np.uint32]:
            pytest.skip("Data type not supported")

        t = np.random.randint(0, 5, (4, 2)).astype(np.float32)
        constant = np.random.randint(0, 5, (4, 2)).astype(dtype)
        input_placeholders = {
            "x": mb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            y = mb.const(val=constant)
            y = mb.cast(x=y, dtype='fp32')
            return mb.add(x=x, y=y)

        expected_output_types = (4, 2, types.fp32)
        expected_outputs = t + constant.astype(np.float32)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )
