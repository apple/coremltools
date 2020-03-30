import itertools

import numpy as np
import pytest
import torch
import torch.nn as nn

import coremltools
from .testing_utils import *


class ModuleWrapper(nn.Module):
    def __init__(self, function, kwargs):
        super().__init__()
        self.function = function
        self.kwargs = kwargs

    def forward(self, x):
        return self.function(x, **self.kwargs)


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
        convert_and_compare(torch_model, input_data, atol=10.0 ** -places)

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

    def test_for_loop(self):
        class TestLayer(nn.Module):
            def __init__(self):
                super(TestLayer, self).__init__()

            def forward(self, x):
                x = 2.0 * x
                return x

        class TestNet(nn.Module):
            input_size = (64,)

            def __init__(self):
                super(TestNet, self).__init__()
                layer = TestLayer()
                self.layer = torch.jit.trace(layer, torch.rand(self.input_size))

            def forward(self, x):
                for _ in range(7):
                    x = self.layer(x)
                return x

        model = TestNet().eval()
        torch_model = torch.jit.script(model)

        self.run_numerical_test(model.input_size, torch_model)

    def test_while_loop(self):
        class TestLayer(nn.Module):
            def __init__(self):
                super(TestLayer, self).__init__()

            def forward(self, x):
                x = 0.5 * x
                return x

        class TestNet(nn.Module):
            input_size = (1,)

            def __init__(self):
                super(TestNet, self).__init__()
                layer = TestLayer()
                self.layer = torch.jit.trace(layer, torch.rand(self.input_size))

            def forward(self, x):
                while x > 0.01:
                    x = self.layer(x)
                return x

        model = TestNet().eval()
        torch_model = torch.jit.script(model)

        self.run_numerical_test(model.input_size, torch_model)

    def test_if(self):
        class TestLayer(nn.Module):
            def __init__(self):
                super(TestLayer, self).__init__()

            def forward(self, x):
                x = torch.mean(x)
                return x

        class TestNet(nn.Module):
            input_size = (64,)

            def __init__(self):
                super(TestNet, self).__init__()
                layer = TestLayer()
                self.layer = torch.jit.trace(layer, torch.rand(self.input_size))

            def forward(self, x):
                m = self.layer(x)
                if m < 0:
                    scale = -2.0
                else:
                    scale = 2.0
                x = scale * x
                return x

        model = TestNet().eval()
        torch_model = torch.jit.script(model)

        self.run_numerical_test(model.input_size, torch_model)

    # TODO: Get this test passing via:
    # rdar://60942015 (Implement New Ops (See Description) and add numerical upsample test)
    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "scales_h, scales_w, output_size, align_corners",
        [
            x
            for x in itertools.product(
                [None], [None], [(10, 10), (1, 1), (20, 20)], [1, 0]
            )
        ]
        + [x for x in itertools.product([2, 3], [4, 5], [None], [1, 0])],
    )
    def test_upsample_bilinear2d(self, scales_h, scales_w, output_size, align_corners):
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate,
            {
                "output_size": output_size,
                "scale_factor": (scales_h, scales_w),
                "mode": "bilinear",
                "align_corners": align_corners,
            },
        )
        self.run_numerical_test(input_shape, model)

    @pytest.mark.parametrize(
        "input_shape, eps",
        itertools.product([(1, 3, 15, 15), (1, 1, 1, 1)], [1e-5, 1e-9]),
    )
    def test_layer_norm(self, input_shape, eps):
        model = nn.LayerNorm(input_shape, eps=eps)
        self.run_numerical_test(input_shape, model)

    @pytest.mark.parametrize(
        "input_shape, eps",
        itertools.product([(1, 3, 15, 15), (1, 1, 1, 1)], [1e-5, 1e-9]),
    )
    def test_batch_norm(self, input_shape, eps):
        model = nn.BatchNorm2d(input_shape[-3], eps=eps)
        self.run_numerical_test(input_shape, model)
