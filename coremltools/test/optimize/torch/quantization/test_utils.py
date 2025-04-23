#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch
import torch.ao.quantization as _aoquant

from coremltools.optimize.torch.quantization._utils import (
    get_n_bits_from_range,
    get_quant_range,
    is_per_channel_quant,
    is_pytorch_defined_observer,
    is_symmetric_quant,
)
from coremltools.optimize.torch.quantization.modules.observers import (
    EMAMinMaxObserver,
    NoopObserver,
)


@pytest.mark.parametrize(
    "qscheme, expected_result",
    [
        (torch.per_tensor_affine, False),
        (torch.per_tensor_symmetric, False),
        (torch.per_channel_affine, True),
        (torch.per_channel_symmetric, True),
    ],
)
def test_quantization_utils_is_per_channel_quant(qscheme, expected_result):
    result = is_per_channel_quant(qscheme)
    assert result == expected_result


@pytest.mark.parametrize(
    "qscheme, expected_result",
    [
        (torch.per_tensor_affine, False),
        (torch.per_tensor_symmetric, True),
        (torch.per_channel_affine, False),
        (torch.per_channel_symmetric, True),
    ],
)
def test_quantization_utils_is_symmetric_quant(qscheme, expected_result):
    result = is_symmetric_quant(qscheme)
    assert result == expected_result


@pytest.mark.parametrize(
    "observer_cls, args, expected_result",
    [
        (_aoquant.MinMaxObserver, None, True),
        (_aoquant.PerChannelMinMaxObserver, None, True),
        (_aoquant.MovingAverageMinMaxObserver, None, True),
        (_aoquant.MovingAveragePerChannelMinMaxObserver, None, True),
        (_aoquant.HistogramObserver, None, True),
        (_aoquant.PlaceholderObserver, None, True),
        (_aoquant.NoopObserver, None, True),
        (
            _aoquant.FixedQParamsObserver,
            (torch.ones((1,), dtype=torch.float), torch.zeros((1,), dtype=torch.int)),
            True,
        ),
        (NoopObserver, None, True),  # Subclass of PyTorch defined observer class
        (EMAMinMaxObserver, None, False),
    ],
)
def test_quantization_utils_is_pytorch_defined_observer(observer_cls, args, expected_result):
    observer = None
    if args is not None:
        observer = observer_cls(*args)
    else:
        observer = observer_cls()

    result = is_pytorch_defined_observer(observer)
    assert result == expected_result


@pytest.mark.parametrize("n_bits", list(range(2, 8)))
@pytest.mark.parametrize("dtype", [torch.quint8, torch.uint8, torch.qint8, torch.int8])
def test_quantization_utils_get_quant_range(dtype, n_bits):
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


@pytest.mark.parametrize("n_bits", list(range(2, 8)))
@pytest.mark.parametrize("dtype", [torch.quint8, torch.uint8, torch.qint8, torch.int8])
def test_quantization_utils_get_n_bits_from_range(dtype, n_bits):
    quant_min, quant_max = get_quant_range(n_bits, dtype)
    output_n_bits = get_n_bits_from_range(quant_min, quant_max)
    assert output_n_bits == n_bits
