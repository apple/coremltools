#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from scipy import special
import scipy
from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.testing_reqs import *

from .testing_utils import run_compare_builder

backends = testing_reqs.backends


class TestConst:
    @pytest.mark.parametrize(
        "use_cpu_only, backend, dtype", itertools.product(
            [True],
            backends,
            [np.float32, np.int32]
        )
    )
    def test_builder_to_backend_smoke(self, use_cpu_only, backend, dtype):
        t = np.random.randint(0, 100, (100, 2)).astype(np.float32)
        constant = np.random.randint(0, 100, (100, 2)).astype(dtype)
        input_placeholders = {
            "x": mb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            y = mb.const(val=constant, mode="file_value")
            x = mb.cast(x=x, dtype='int32')
            z = mb.add(x=x, y=y)
            return mb.cast(x=z, dtype='fp32')

        expected_output_types = (100, 2, types.fp32)
        expected_outputs = t + constant.astype(np.float32)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )