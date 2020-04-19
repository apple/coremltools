import itertools

import numpy as np
import pytest
import torch
import torch.nn as nn

import coremltools
from .testing_utils import *


class ModuleWrapper(nn.Module):
    def __init__(self, function, kwargs):
        super(ModuleWrapper, self).__init__()
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

    @pytest.mark.parametrize(
        "in_features, out_features", itertools.product([10, 25, 100], [3, 6]),
    )
    def test_addmm(self, in_features, out_features):
        model = nn.Linear(in_features, out_features)
        run_numerical_test((1, in_features), model)

    @pytest.mark.parametrize(
        "num_features, eps", itertools.product([5, 2, 1], [0.1, 1e-05]),
    )
    def test_batchnorm(self, num_features, eps):
        model = nn.BatchNorm2d(num_features, eps)
        run_numerical_test((1, num_features, 5, 5), model)

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
        run_numerical_test((1, in_channels, height, width), model)

    @pytest.mark.parametrize(
        "height, width, in_channels, out_channels, kernel_size, stride, padding, dilation",
        itertools.product(
            [5, 6], [5, 7], [1, 3], [1, 3], [1, 3], [2, 3], [0, 1], [1, 3]
        ),
    )
    def test_convolution_transpose2d(
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
        model = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        run_numerical_test((1, in_channels, height, width), model)

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

        run_numerical_test(model.input_size, torch_model)

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

        run_numerical_test(model.input_size, torch_model)

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

        run_numerical_test(model.input_size, torch_model)

    @pytest.mark.parametrize(
        "output_size, align_corners",
        [
            x
            for x in itertools.product(
                [(10, 10), (1, 1), (20, 20), (2, 3), (190, 170)], [True, False]
            )
        ]
    )
    def test_bilinear2d_interpolate_with_output_size(self, output_size, align_corners):
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate,
            {
                "size": output_size,
                "scale_factor": None,
                "mode": "bilinear",
                "align_corners": align_corners,
            },
        )
        run_numerical_test(input_shape, model)

    @pytest.mark.parametrize(
        "scales_h, scales_w, align_corners",
        [
            x
            for x in itertools.product(
                [2, 3, 4.5], [4, 5, 5.5], [True, False]
            )
        ]
    )
    def test_bilinear2d_interpolate_with_scales(self, scales_h, scales_w, align_corners):
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate,
            {
                "size": None,
                "scale_factor": (scales_h, scales_w),
                "mode": "bilinear",
                "align_corners": align_corners,
            },
        )
        run_numerical_test(input_shape, model)

    @pytest.mark.parametrize(
        "input_shape, eps",
        itertools.product([(1, 3, 15, 15), (1, 1, 1, 1)], [1e-5, 1e-9]),
    )
    def test_layer_norm(self, input_shape, eps):
        model = nn.LayerNorm(input_shape, eps=eps)
        run_numerical_test(input_shape, model)

    @pytest.mark.parametrize(
        "input_shape, eps",
        itertools.product([(1, 3, 15, 15), (1, 1, 1, 1)], [1e-5, 1e-9]),
    )
    def test_batch_norm(self, input_shape, eps):
        model = nn.BatchNorm2d(input_shape[-3], eps=eps)
        run_numerical_test(input_shape, model)

    @pytest.mark.xfail(reason="rdar://problem/61064173")
    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, pad, include_pad",
        itertools.product(
            [(1, 3, 15), (1, 1, 7), (1, 3, 10)],
            [1, 2, 3],
            [1, 2],
            [0, 1],
            [True, False],
        ),
    )
    def test_avg_pool1d(self, input_shape, kernel_size, stride, pad, include_pad):
        if pad > kernel_size / 2:
            # Because this test is xfail, we have to fail rather than
            # just return here, otherwise these test cases unexpectedly pass.
            # This can be changed to `return` once the above radar
            # is fixed and the test is no longer xfail.
            raise ValueError("pad must be less than half the kernel size")
        model = nn.AvgPool1d(kernel_size, stride, pad, False, include_pad)
        run_numerical_test(input_shape, model)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, pad, include_pad",
        itertools.product(
            [(1, 3, 15, 15), (1, 1, 7, 7), (1, 3, 10, 10)],
            [1, 2, 3],
            [1, 2],
            [0, 1],
            [True, False],
        ),
    )
    def test_avg_pool2d(self, input_shape, kernel_size, stride, pad, include_pad):
        if pad > kernel_size / 2:
            return
        model = nn.AvgPool2d(kernel_size, stride, pad, False, include_pad)
        run_numerical_test(input_shape, model)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, pad, include_pad",
        itertools.product(
            [(1, 3, 15, 15), (1, 1, 7, 7), (1, 3, 10, 10)],
            [3],
            [1, 2],
            [0, 1],
            [True, False],
        ),
    )
    def test_avg_pool2d_ceil_mode(
        self, input_shape, kernel_size, stride, pad, include_pad
    ):
        if pad > kernel_size / 2:
            return
        model = nn.AvgPool2d(kernel_size, stride, pad, True, include_pad)
        run_numerical_test(input_shape, model)

    @pytest.mark.xfail(
        reason="Pytorch convert function for op max_pool1d not implemented, "
        "we will also likely run into rdar://problem/61064173"
    )
    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, pad",
        itertools.product(
            [(1, 3, 15), (1, 1, 7), (1, 3, 10)], [1, 2, 3], [1, 2], [0, 1],
        ),
    )
    def test_max_pool1d(self, input_shape, kernel_size, stride, pad):
        if pad > kernel_size / 2:
            # Because this test is xfail, we have to fail rather than
            # just return here, otherwise these test cases unexpectedly pass.
            # This can be changed to `return` once the above radar
            # is fixed and the test is no longer xfail.
            raise ValueError("pad must be less than half the kernel size")
        model = nn.MaxPool1d(kernel_size, stride, pad, ceil_mode=False)
        run_numerical_test(input_shape, model)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, pad",
        itertools.product(
            [(1, 3, 15, 15), (1, 1, 7, 7), (1, 3, 10, 10)], [1, 2, 3], [1, 2], [0, 1],
        ),
    )
    def test_max_pool2d(self, input_shape, kernel_size, stride, pad):
        if pad > kernel_size / 2:
            return
        model = nn.MaxPool2d(kernel_size, stride, pad, ceil_mode=False)
        run_numerical_test(input_shape, model)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, pad",
        itertools.product(
            [(1, 3, 15, 15), (1, 1, 7, 7), (1, 3, 10, 10)], [3], [1, 2], [0, 1],
        ),
    )
    def test_max_pool2d_ceil_mode(self, input_shape, kernel_size, stride, pad):
        if pad > kernel_size / 2:
            return
        model = nn.MaxPool2d(kernel_size, stride, pad, ceil_mode=True)
        run_numerical_test(input_shape, model)

    # This tests an edge case where the list of tensors to concatenate only
    # has one item. NNv1 throws an error for this case, hence why we have to
    # run through the full conversion process to test it.
    def test_cat(self):
        class TestNet(nn.Module):
            def __init__(self):
                super(TestNet, self).__init__()

            def forward(self, x):
                x = torch.cat((x,), axis=1)
                return x

        model = TestNet()
        run_numerical_test((1, 3, 16, 16), model)
