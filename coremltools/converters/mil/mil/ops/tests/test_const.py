#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import itertools
import numpy as np
import pytest

from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil import Builder as mb, types

from .testing_utils import run_compare_builder

backends = testing_reqs.backends


class TestConst:
    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend, dtype", itertools.product(
            [True, False],
            backends,
            [
                np.uint8,
                np.int8,
                np.uint16,
                np.int16,
                np.uint32,
                np.int32,
                np.uint64,
                np.int64,
                np.float32,
                np.float64,
            ]
        )
    )
    def test_builder_to_backend_smoke(self, use_cpu_for_conversion, backend, dtype):
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("rdar://78343191 ((MIL GPU) Core ML Tools Unit Test failures [failure to load or Seg fault])")

        t = np.random.randint(0, 5, (4, 2)).astype(np.float32)
        constant = np.random.randint(0, 5, (4, 2)).astype(dtype)
        input_placeholders = {
            "x": mb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            y = mb.const(val=constant)
            x = mb.cast(x=x, dtype='int32')
            z = mb.add(x=x, y=y)
            return mb.cast(x=z, dtype='fp32')

        expected_output_types = (4, 2, types.fp32)
        expected_outputs = t + constant.astype(np.float32)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            use_cpu_only=use_cpu_for_conversion,
            frontend_only=False,
            backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion,
        )
