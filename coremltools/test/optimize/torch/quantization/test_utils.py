#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch

from coremltools.optimize.torch.quantization._utils import get_quant_range


@pytest.mark.parametrize("n_bits", list(range(2, 8)))
@pytest.mark.parametrize("dtype", [torch.quint8, torch.uint8, torch.qint8, torch.int8])
def test_quant_range(dtype, n_bits):
    quant_min, quant_max = get_quant_range(n_bits, dtype)
    signed_expected_values = {
        2: [-2, 1],
        3: [-4, 3],
        4: [-8, 7],
        5: [-16, 15],
        6: [-32, 31],
        7: [-64, 63],
        8: [-128, 127],
    }
    unsigned_expected_values = {
        2: [0, 3],
        3: [0, 7],
        4: [0, 15],
        5: [0, 31],
        6: [0, 63],
        7: [0, 127],
        8: [0, 256],
    }
    if dtype in [torch.quint8, torch.uint8]:
        assert quant_min == unsigned_expected_values[n_bits][0]
        assert quant_max == unsigned_expected_values[n_bits][1]
    else:
        assert quant_min == signed_expected_values[n_bits][0]
        assert quant_max == signed_expected_values[n_bits][1]
