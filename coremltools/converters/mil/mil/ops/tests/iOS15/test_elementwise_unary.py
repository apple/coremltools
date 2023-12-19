#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from coremltools.converters.mil.mil.ops.defs.iOS15 import elementwise_unary


# Mock class to simulate the input_var behavior
class MockInputVar:
    def __init__(self, val, sym_type):
        self.val = val
        self.sym_type = sym_type


class TestCast:
    def test_cast_float_without_tensor(self):
        input_var = MockInputVar(val=1.0, sym_type=None)
        output = elementwise_unary.cast.get_cast_value(input_var, "fp32")
        expected_output = np.array(1.0, dtype=np.float32)
        np.testing.assert_array_equal(output, expected_output)
