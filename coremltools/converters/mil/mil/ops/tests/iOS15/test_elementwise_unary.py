#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import numpy as np
import pytest
from coremltools.converters.mil.mil.ops.defs.iOS15 import elementwise_unary


# Mock class to simulate the input_var behavior
class MockInputVar:
    def __init__(self, val, sym_type):
        self.val = val
        self.sym_type = sym_type


class TestCast:
    NUMPY_DTYPE_TO_STRING = {
        np.int32: "int32",
        np.float16: "fp16",
        np.float32: "fp32",
        np.bool_: "bool",
    }

    @pytest.mark.parametrize(
        "value, dtype",
        itertools.product(
            [2.0, (0.0, 1.0)],
            [np.int32, np.float16, np.float32, np.bool_],
        ),
    )
    def test_cast(self, value, dtype):
        input_var = MockInputVar(val=value, sym_type=None)
        output = elementwise_unary.cast.get_cast_value(
            input_var, self.NUMPY_DTYPE_TO_STRING[dtype]
        )
        expected_output = dtype(value)
        np.testing.assert_array_equal(output, expected_output)
