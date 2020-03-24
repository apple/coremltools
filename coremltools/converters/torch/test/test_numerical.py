import itertools
import numpy as np
import pytest
import torch
import torch.nn as nn

import coremltools

from test_utils import *


class TestTorchNumerical:
    """Class containing numerical correctness tests for TorchIR -> CoreML op
        conversion.
    """

    @pytest.fixture
    def set_random_seeds(self):
        torch.manual_seed(1)
        np.random.seed(1)

    @staticmethod
    def run_numerical_test(input_size, model, places=5):
        input_data = torch.rand(input_size)
        model.eval()
        torch_model = torch.jit.trace(model, input_data)
        convert_and_compare(torch_model, input_data, places=places)

    @pytest.mark.parametrize(
        "in_features, out_features", itertools.product([10, 25, 100], [3, 6]),
    )
    def test_addmm(self, in_features, out_features):
        model = nn.Linear(in_features, out_features)
        self.run_numerical_test((1, in_features), model)

    @pytest.mark.parametrize(
        "num_features, eps", itertools.product([5, 2, 1], [0.1, 1e-05]),
    )
    def test_batchnorm(self, num_features, eps):
        model = nn.BatchNorm2d(num_features, eps)
        self.run_numerical_test((1, num_features, 5, 5), model)

    @pytest.mark.parametrize(
        "height, width, in_channels, out_channels, kernel_size, stride, padding, dilation",
        itertools.product(
            [5, 6], [5, 7], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3]
        ),
    )
    def test_convolution2d(
        self,
        height,
        width,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups=1,
    ):
        model = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.run_numerical_test((1, in_channels, height, width), model)

