#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch

from coremltools.optimize.torch.layerwise_compression._quant import Quantizer


@pytest.mark.parametrize(
    "quantizer, expected_scale, expected_zp",
    [
        (
            Quantizer(n_bits=4, per_channel=True, symmetric=True),
            torch.tensor([[0.48], [1.4]]),
            torch.tensor([[8.0], [8.0]]),
        ),
        (
            Quantizer(n_bits=4, per_channel=False, symmetric=True),
            torch.tensor([[1.4], [1.4]]),
            torch.tensor([[8.0], [8.0]]),
        ),
        (
            Quantizer(n_bits=4, per_channel=True, symmetric=False),
            torch.tensor([[0.32], [0.76]]),
            torch.tensor([[11.0], [1.0]]),
        ),
        (
            Quantizer(n_bits=4, per_channel=False, symmetric=False),
            torch.tensor([[0.94], [0.94]]),
            torch.tensor([[4.0], [4.0]]),
        ),
        (
            Quantizer(n_bits=8, per_channel=True, symmetric=True),
            torch.tensor([[0.028], [0.0824]]),
            torch.tensor([[128.0], [128.0]]),
        ),
        (
            Quantizer(n_bits=8, per_channel=False, symmetric=True),
            torch.tensor([[0.0824], [0.0824]]),
            torch.tensor([[128.0], [128.0]]),
        ),
        (
            Quantizer(n_bits=8, per_channel=True, symmetric=False),
            torch.tensor([[0.0188], [0.0447]]),
            torch.tensor([[191.0], [20.0]]),
        ),
        (
            Quantizer(n_bits=8, per_channel=False, symmetric=False),
            torch.tensor([[0.0553], [0.0553]]),
            torch.tensor([[65.0], [65.0]]),
        ),
    ],
)
def test_find_params(quantizer, expected_scale, expected_zp):
    # input
    x = torch.tensor([[1.2, -3.6, 0.4], [-0.9, 1.5, 10.5]])
    # fine quantization params
    quantizer.find_params(x, weight=True)
    # compare
    assert torch.all(
        torch.isclose(quantizer.scale, expected_scale, rtol=0.001, atol=0.001)
    )
    assert torch.all(torch.isclose(quantizer.zero_point, expected_zp))


@pytest.mark.parametrize(
    "input, weight, expected_shape",
    [
        (torch.rand(2, 3), True, (2, 1)),
        (torch.rand(2, 3, 4), True, (2, 1, 1)),
        (torch.rand(2, 3, 4, 5), True, (2, 1, 1, 1)),
        (torch.rand(2, 3), False, (1, 3)),
        (torch.rand(2, 3, 4), False, (1, 1, 4)),
        (torch.rand(2, 3, 4, 5), False, (1, 3, 1, 1)),
    ],
)
def test_find_params_reshape(input, weight, expected_shape):
    quantizer = Quantizer(n_bits=4, per_channel=True, symmetric=True)
    quantizer.find_params(input, weight=weight)
    assert quantizer.scale.shape == expected_shape
