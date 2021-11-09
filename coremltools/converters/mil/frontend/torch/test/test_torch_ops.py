#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import sys
import itertools
import numpy as np
import pytest
import torch.nn as nn

from .testing_utils import contains_op, ModuleWrapper, TorchBaseTest
from coremltools import RangeDim
from coremltools.models.utils import _python_version
from coremltools.models.utils import _macos_version
from coremltools.converters.mil import testing_reqs

from coremltools import TensorType
from coremltools._deps import version_lt

pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 8), reason="Segfault with Python 3.8+"
)  # rdar://problem/65730375



backends = testing_reqs.backends
torch = pytest.importorskip("torch")
torch.manual_seed(30)
np.random.seed(30)

# Set of common shapes for testing. Not all layers support 1D, so these two
# set of shapes are kept separate
COMMON_SHAPES = [(1, 10), (1, 5, 6), (1, 3, 5, 6), (1, 3, 4, 5, 6)]
COMMON_SHAPES_ALL = [(1, )] + COMMON_SHAPES


class TestAffineGrid(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, x_shape_and_target_size, "
        "sampling_mode, padding_mode, align_corners",
        itertools.product(
            backends,
            [
                # shape format: (Batch, Channel, Height, Width)
                [(1, 1, 3, 3), (1, 1, 3, 3)],  # no size change
                [(2, 3, 5, 5), (2, 3, 3, 2)],  # down-sampling
                [(3, 1, 6, 6), (3, 1, 8, 8)],  # up-sampling
            ],
            ["bilinear"],
            ["zeros"],
            [True],
        ),
    )
    def test(
        self,
        backend,
        x_shape_and_target_size,
        sampling_mode,
        padding_mode,
        align_corners,
    ):
        if backend[0] == "neuralnetwork":
            pytest.xfail("nn backend not supported")

        x_shape, target_size = x_shape_and_target_size
        theta = torch.rand((x_shape[0], 2, 3))

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.affine_grid = torch.nn.functional.affine_grid
                self.grid_sample = torch.nn.functional.grid_sample

            def forward(self, x):
                grid = self.affine_grid(
                    theta=theta, size=target_size, align_corners=align_corners,
                )
                x = self.grid_sample(
                    x,
                    grid=grid,
                    mode=sampling_mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )
                return x

        model = TestModule()
        self.run_compare_torch(x_shape, model, backend=backend)


class TestGridSample(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, data_grid_shapes, mode, padding_mode, align_corners",
        itertools.product(
            backends,
            [
                # Input shape format: (Batch, C, Hin, Win)
                # Grid shape format: (Batch, Hout, Wout, 2)
                [(1, 1, 3, 3), (1, 3, 3, 2)],  # no size change
                [(2, 3, 5, 5), (2, 3, 3, 2)],  # down-sampling
                [(3, 1, 6, 6), (3, 8, 8, 2)],  # up-sampling
            ],
            ["bilinear", "nearest"],
            ["zeros", "border", "reflection"],
            [True, False],
        ),
    )
    def test(
        self,
        backend,
        data_grid_shapes,
        mode,
        padding_mode,
        align_corners,
    ):
        if backend[0] == "neuralnetwork":
            pytest.xfail("nn backend not supported")

        params = {
            "mode": mode,
            "padding_mode": padding_mode,
            "align_corners": align_corners,
        }
        model = ModuleWrapper(
            function=torch.nn.functional.grid_sample, kwargs=params
        )
        self.run_compare_torch(data_grid_shapes, model, backend=backend)


class TestNLLLoss(TorchBaseTest):
    @pytest.mark.parametrize(
        "reduction, backend",
        itertools.product(
            ["none", "sum", "mean"],
            backends,
        ),
    )
    def test_nllloss(
        self,
        reduction,
        backend,
    ):
        class NLLLossModel(nn.Module):
            def __init__(self):
                super(NLLLossModel, self).__init__()
                self.loss = nn.NLLLoss(reduction=reduction)

            def forward(self, x, target):
                loss = self.loss(x, target)
                return loss

        x = torch.randn(3, 5)
        target = torch.tensor([1, 0, 4])
        inputs = (x, target)

        model = NLLLossModel()
        expected_results = model(*inputs)

        self.run_compare_torch(
            inputs, model, expected_results, input_as_shape=False, backend=backend,
        )


class TestArgSort(TorchBaseTest):
    @pytest.mark.parametrize(
        "shape, axis, descending, backend",
        itertools.product(
            COMMON_SHAPES,
            [-1, 0],
            [True, False],
            backends
        )
    )
    def test_argsort(self, shape, axis, descending, backend):
        model = ModuleWrapper(
            function=torch.argsort, kwargs={"dim": axis, "descending": descending}
        )
        TorchBaseTest.run_compare_torch(shape, model, backend=backend)

class TestSort(TorchBaseTest):
    @pytest.mark.parametrize(
        "shape, axis, descending, backend",
        itertools.product(
            COMMON_SHAPES,
            [-1, 0],
            [True, False],
            backends
        )
    )
    def test_sort(self, shape, axis, descending, backend):
        model = ModuleWrapper(
            function=torch.sort, kwargs={"dim": axis, "descending": descending}
        )
        TorchBaseTest.run_compare_torch(shape, model, backend=backend)


class TestNorms(TorchBaseTest):
    @pytest.mark.parametrize(
        "shape, backend, keepdim",
        itertools.product(
            COMMON_SHAPES,
            backends,
            [True, False]
        )
    )
    def test_frobenius_norm(self, shape, backend, keepdim):
        num_dims = len(shape)
        for dim in range(-num_dims, num_dims):
            model = ModuleWrapper(
                function=torch.norm, kwargs={'keepdim': keepdim, 'dim': dim}
            )
            TorchBaseTest.run_compare_torch(shape, model, backend=backend)

    @pytest.mark.parametrize(
        "shape, backend, p, keepdim",
        itertools.product(
            COMMON_SHAPES,
            backends,
            [1, 2, -1, 3, np.inf, -np.inf],
            [True, False]
        )
    )
    def test_number_norm(self, shape, backend, p, keepdim):
        for dim in (-1, 0, 1):
            model = ModuleWrapper(
                function=torch.norm, kwargs={'p': p, 'keepdim': keepdim, 'dim': dim}
            )
            TorchBaseTest.run_compare_torch(shape, model, backend=backend, places=2)


class TestBatchNorm(TorchBaseTest):
    @pytest.mark.parametrize(
        "num_features, eps, affine, backend",
        itertools.product([5, 3, 1], [0.1, 1e-05], [True, False], backends),
    )
    def test_batchnorm(self, num_features, eps, affine, backend):
        model = nn.BatchNorm2d(num_features, eps, affine=affine)
        self.run_compare_torch((6, num_features, 5, 5), model, backend=backend)

    @pytest.mark.parametrize(
        "affine, backend",
        itertools.product([True, False], backends),
    )
    def test_batchnorm_2d_with_conv(self, affine, backend):
        class CRNNBase(nn.Module):
            def __init__(self, ch_in, ch_out, kernel_size=3):
                super(CRNNBase, self).__init__()
                self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size)
                self.norm = nn.BatchNorm2d(ch_out, affine=affine)

            def forward(self, x):
                x = self.conv(x)
                x = self.norm(x)
                return x
        model = CRNNBase(ch_in=6, ch_out=16)
        self.run_compare_torch((1, 6, 15, 30), model, backend=backend)

    @pytest.mark.parametrize(
        "num_features, eps, affine, dynamic_input, backend",
        itertools.product([5, 1], [0.1, 1e-05], [True, False], ["None", "Batch", "Height", "Width", "Depth", "All"], backends),
    )
    def test_batchnorm_3d(self, num_features, eps, affine, dynamic_input, backend):
        model = nn.BatchNorm3d(num_features, eps, affine=affine)
        input_shape = (6, num_features, 2, 3, 4)
        if dynamic_input == "None":
            self.run_compare_torch(
                input_shape,
                model,
                backend=backend
            )
        else:
            if dynamic_input == "Batch":
                converter_input_type = [TensorType(shape=(6, num_features, 2, 3, 4), dtype=np.float32)]
                converter_input_type = [TensorType(shape=(RangeDim(1, 10), num_features, 2, 3, 4), dtype=np.float32)]
            elif dynamic_input == "Height":
                converter_input_type = [TensorType(shape=(6, num_features, RangeDim(1, 10), 3, 4), dtype=np.float32)]
            elif dynamic_input == "Width":
                converter_input_type = [TensorType(shape=(6, num_features, 2, RangeDim(1, 10), 4), dtype=np.float32)]
            elif dynamic_input == "Depth":
                converter_input_type = [TensorType(shape=(6, num_features, 2, 3, RangeDim(1, 10)), dtype=np.float32)]
            elif dynamic_input == "All":
                converter_input_type = [TensorType(shape=(RangeDim(1, 10),
                                                          num_features,
                                                          RangeDim(1, 10),
                                                          RangeDim(1, 10),
                                                          RangeDim(1, 10)),
                                                   dtype=np.float32)]
            self.run_compare_torch(
                input_shape,
                model,
                backend=backend,
                converter_input_type=converter_input_type
            )

    @pytest.mark.parametrize(
        "rank, num_features, eps, training, backend",
        itertools.product([3, 4, 5], [5, 1], [0.1, 1e-05], [True, False], backends),
    )
    def test_batchnorm_dynamic(self, rank, num_features, eps, training, backend):
        model = ModuleWrapper(
            nn.functional.batch_norm,
            {"training": training, "eps": eps,},
        )
        input_shape = [6, num_features, 3, 4, 5]
        input_shape = input_shape[:rank]
        _input = torch.randn(*input_shape)
        _mean = torch.randn(num_features)
        _var = torch.randn(num_features)

        inputs = (_input, _mean, _var)
        expected_results = model(*inputs)

        self.run_compare_torch(
            inputs, model, expected_results, input_as_shape=False, backend=backend,
        )

    @pytest.mark.parametrize(
        "affine, backend",
        itertools.product([True, False], backends),
    )
    def test_batchnorm_1d_with_conv(self, affine, backend):
        class CRNNBase(nn.Module):
            def __init__(self, ch_in, ch_out, kernel_size=3):
                super(CRNNBase, self).__init__()
                self.conv = nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size)
                self.norm = nn.BatchNorm1d(ch_out, affine=affine)

            def forward(self, x):
                x = self.conv(x)
                x = self.norm(x)
                return x
        model = CRNNBase(ch_in=6, ch_out=16)
        self.run_compare_torch((1, 6, 15), model, backend=backend)

    @pytest.mark.parametrize(
        "shape, eps, affine, backend",
        itertools.product([(1, 10), (4, 6), (10, 1)], [0.1, 1e-05], [True, False], backends),
    )
    def test_batchnorm1d_rank2(self, shape, eps, affine, backend):
        N,C = shape
        batchnorm = nn.BatchNorm1d(C, eps=eps, affine=affine).eval()
        self.run_compare_torch(
            (N, C), batchnorm, backend=backend,
        )

    @pytest.mark.parametrize(
        "shape, eps, affine, backend",
        itertools.product([(4, 8, 2), (1, 5, 3), (5, 10, 1), (6, 1, 4)], [0.1, 1e-05], [True, False], backends),
    )
    def test_batchnorm1d_rank3(self, shape, eps, affine, backend):
        N,C,L = shape
        batchnorm = nn.BatchNorm1d(C, eps=eps, affine=affine).eval()
        self.run_compare_torch(
            (N, C, L), batchnorm, backend=backend,
        )

class TestInstanceNorm(TorchBaseTest):
    @pytest.mark.parametrize(
        "num_features, eps, backend",
        itertools.product([5, 2, 1], [0.1, 1e-05], backends),
    )
    def test_instancenorm(self, num_features, eps, backend):
        model = nn.InstanceNorm2d(num_features, eps)
        self.run_compare_torch((6, num_features, 5, 5), model, backend=backend)

    @pytest.mark.parametrize("num_features, backend",
                             itertools.product([5, 2, 1], backends),
    )
    def test_instancenorm_1d(self, num_features, backend):
        model = nn.InstanceNorm1d(num_features)
        self.run_compare_torch((6, num_features, 10), model, backend=backend)

class TestGroupNorm(TorchBaseTest):
    @pytest.mark.parametrize(
        "group_features, eps,affine, backend",
        itertools.product([(16, 32), (1, 1)], [0.1, 1e-05],[True, False], backends),
    )
    def test_groupnorm(self, group_features, eps, affine, backend):
        model = nn.GroupNorm(group_features[0],group_features[1], eps=eps, affine=affine)
        self.run_compare_torch((6, group_features[1], 5, 5), model, backend=backend)

class TestLinear(TorchBaseTest):
    @pytest.mark.parametrize(
        "in_features, out_features, bias, backend",
        itertools.product([5], [10], [True, False], backends),
    )
    def test_linear_rank1_input(self, in_features, out_features, bias, backend):
        model = nn.Linear(in_features, out_features, bias=bias)
        self.run_compare_torch((in_features,), model, backend=backend)

    @pytest.mark.parametrize(
        "in_features, out_features, bias, backend",
        itertools.product([10, 25], [3, 6], [True, False], backends),
    )
    def test_linear_rank2_input(self, in_features, out_features, bias, backend):
        model = nn.Linear(in_features, out_features, bias=bias)
        self.run_compare_torch((1, in_features), model, backend=backend)

    @pytest.mark.parametrize(
        "in_features, out_features, bias, backend",
        itertools.product([10], [6], [True, False], backends),
    )
    def test_linear_rank3_input(self, in_features, out_features, bias, backend):
        model = nn.Linear(in_features, out_features, bias=bias)
        self.run_compare_torch((1, 3, in_features), model, backend=backend)

    @pytest.mark.parametrize(
        "in_features, out_features, bias, backend",
        itertools.product([10], [6], [True, False], backends),
    )
    def test_linear_rank4_input(self, in_features, out_features, bias, backend):
        model = nn.Linear(in_features, out_features, bias=bias)
        self.run_compare_torch((1, 5, 3, in_features), model, backend=backend)


class TestConv(TorchBaseTest):
    @pytest.mark.parametrize(
        "height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, backend",
        [ (*param, bend) for param, bend in itertools.product([
             (5, 3, 1, 1, 1, 2, 0, 1),
             (3, 3, 1, 1, 1, 2, 1, 3),
             (4, 3, 3, 3, 2, 2, 0, 1),
             (7, 3, 3, 3, 1, 3, 0, 1),
             (5, 5, 3, 3, 1, 3, 0, 1),
             (3, 5, 3, 3, 1, 3, 0, 1),
             (3, 5, 3, 3, 1, 3, 1, 3),
             (7, 5, 3, 3, 2, 3, 1, 3),
           ], backends)
        ],
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
        backend,
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
        self.run_compare_torch((1, in_channels, height, width), model,
                           backend=backend)

class TestDynamicConv(TorchBaseTest):
    @pytest.mark.parametrize(
        "width, in_channels, out_channels, kernel_size, stride, padding, backend",
        [ (*param, bend) for param, bend in itertools.product([
             (5, 1, 1, 1, 2, 1),
             (3, 1, 1, 1, 2, 3),
             (4, 3, 3, 1, 2, 1),
             (7, 3, 3, 1, 3, 1),
             (5, 3, 3, 2, 2, 1),
             (3, 3, 3, 1, 3, 1),
             (3, 3, 3, 1, 3, 3),
             (7, 3, 3, 3, 1, 3),
           ], backends)
        ],
    )
    def test_convolution1d(
        self,
        width,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        backend,
        groups=1,
    ):
        if backend[0] == 'mlprogram':
            pytest.xfail("Not supported on ML Program backend")

        class DynamicConv(nn.Module):
            def __init__(self):
                super(DynamicConv, self).__init__()

            def forward(self, input_data, weights):
                return nn.functional.conv1d(
                    input_data,
                    weights,
                    stride=stride,
                    padding=padding
                )

        model = DynamicConv()
        self.run_compare_torch([(1, in_channels, width), (out_channels, int(in_channels/groups), kernel_size)],
            model, backend=backend)

    @pytest.mark.parametrize(
        "height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, backend",
        [ (*param, bend) for param, bend in itertools.product([
             (5, 3, 1, 1, 1, 2, 0, 1),
             (3, 3, 1, 1, 1, 2, 1, 3),
             (4, 3, 3, 3, 1, 2, 0, 1),
             (7, 3, 3, 3, 1, 3, 0, 1),
             (5, 5, 3, 3, 2, 1, 0, 1),
             (3, 5, 3, 3, 1, 3, 0, 1),
             (3, 5, 3, 3, 1, 3, 1, 3),
             (7, 5, 3, 3, 2, 3, 1, 3),
           ], backends)
        ],
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
        backend,
        groups=1,
    ):

        class DynamicConv(nn.Module):
            def __init__(self):
                super(DynamicConv, self).__init__()

            def forward(self, input_data, weights):
                return nn.functional.conv2d(
                    input_data,
                    weights,
                    stride=stride,
                    padding=padding
                )

        model = DynamicConv()
        self.run_compare_torch([(1, in_channels, height, width), (out_channels, int(in_channels/groups), kernel_size, kernel_size)],
            model, backend=backend)

class TestConvTranspose(TorchBaseTest):
    @pytest.mark.parametrize(
        "width, in_channels, out_channels, kernel_size, stride, padding, dilation, backend",
        [ (*param, bend) for param, bend in itertools.product([
             (3, 1, 1, 1, 2, 0, 1),
             (3, 1, 1, 1, 2, 1, 3),
             (3, 3, 3, 1, 2, 0, 1),
             (3, 3, 3, 1, 3, 0, 1),
             (5, 3, 3, 1, 3, 0, 1),
             (5, 3, 3, 1, 3, 0, 1),
             (5, 3, 3, 1, 3, 1, 3),
             (5, 3, 3, 1, 3, 1, 3),
           ], backends)
        ],
    )
    def test_convolution_transpose1d(
            self,
            width,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            backend,
            groups=1,
    ):
        model = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.run_compare_torch((1, in_channels, width), model, backend=backend)


    @pytest.mark.parametrize(
        "height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, backend",
        [ (*param, bend) for param, bend in itertools.product([
             (5, 5, 1, 1, 1, 2, 0, 1),
             (5, 5, 1, 1, 1, 2, 1, 3),
             (5, 5, 3, 3, 1, 2, 0, 1),
             (5, 5, 3, 3, 1, 3, 0, 1),
             (6, 5, 3, 3, 1, 3, 0, 1),
             (6, 5, 3, 3, 1, 3, 0, 1),
             (6, 5, 3, 3, 1, 3, 1, 3),
             (6, 5, 3, 3, 1, 3, 1, 3),
           ], backends)
        ],
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
        backend,
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
        self.run_compare_torch((1, in_channels, height, width), model,
                           backend=backend)

    @pytest.mark.parametrize(
        "dynamic_input, backend",
        itertools.product(
            [True, False], backends
        ),
    )
    def test_convolution_transpose2d_dynamic_input(
        self,
        dynamic_input,
        backend,
    ):
        in_channels = 5
        model = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=10,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=3,
        )
        in_height = 256
        in_width = 512
        input_shape = (1, in_channels, in_height, in_width)

        if dynamic_input:
            converter_input_type = [TensorType(shape=(1, in_channels, RangeDim(256, -1), RangeDim(256, -1)), dtype=np.float32)]
            self.run_compare_torch(
                input_shape,
                model,
                backend=backend,
                converter_input_type=converter_input_type
            )
        else:
            self.run_compare_torch(
                input_shape,
                model,
                backend=backend
            )

    @pytest.mark.parametrize(
        "height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, output_padding, backend",
        [ (*param, bend) for param, bend in itertools.product([
             (5, 5, 1, 1, 1, 2, 1, 1, 1),
             (5, 5, 1, 1, 1, 2, 2, 3, 2),
             (5, 5, 3, 3, 1, 2, 0, 1, 0),
             (5, 5, 3, 3, 1, 3, 1, 1, 1),
             (6, 5, 3, 3, 1, 3, 2, 1, 2),
             (6, 5, 3, 3, 1, 3, 1, 1, 1),
             (6, 5, 3, 3, 1, 3, 2, 3, 2),
             (6, 5, 3, 3, 1, 3, 3, 3, 3),
           ], backends)
        ]+ [
            pytest.param(
                5, 5, 1, 1, 3, 4, 1, 1, 2, "neuralnetwork", marks=pytest.mark.xfail
            ),
            pytest.param(
                5, 5, 1, 1, 3, 2, 1, 3, 2, "neuralnetwork", marks=pytest.mark.xfail
            ),
        ],
    )
    def test_convolution_transpose2d_output_padding(
        self,
        height,
        width,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        output_padding,
        backend,
        groups=1,
    ):

        # Output padding must be less than either stride or dilation
        # Skip testing invalid combinations
        if isinstance(output_padding, int):
            if output_padding >= stride and output_padding >= dilation:
                return
        elif isinstance(output_padding, tuple):
            for _output_padding in output_padding:
                if _output_padding >= stride and _output_padding >= dilation:
                    return

        model = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
        )
        self.run_compare_torch((1, in_channels, height, width), model,
                           backend=backend)


    @pytest.mark.parametrize(
        "depth, height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, backend",
        [ (*param, bend) for param, bend in itertools.product([
             (3, 5, 5, 1, 1, 1, 2, 0, 1),
             (3, 5, 5, 1, 1, 1, 2, 1, 3),
             (3, 5, 5, 3, 3, 1, 2, 0, 1),
             (3, 5, 5, 3, 3, 1, 1, 0, 2),
             (4, 6, 5, 3, 3, 1, 3, 0, 1),
             (4, 6, 5, 3, 3, 1, 3, 1, 2),
             (4, 6, 5, 3, 3, 1, 3, 1, 3),
           ], backends)
        ],
    )
    def test_convolution_transpose3d(
             self,
             depth,
             height,
             width,
             in_channels,
             out_channels,
             kernel_size,
             stride,
             padding,
             dilation,
             backend,
    ):
        model = nn.ConvTranspose3d(
             in_channels=in_channels,
             out_channels=out_channels,
             kernel_size=kernel_size,
             stride=stride,
             padding=padding,
             dilation=dilation,
         )
        self.run_compare_torch((1, in_channels, depth, height, width), model,
                           backend=backend)


class TestCond(TorchBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends)
    )
    def test_cond(self, use_cpu_for_conversion, backend):
        if backend[0] == "mlprogram":
            pytest.skip("rdar://81169758 (Cond tests hang on mlprogram backend)")
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("rdar://78343191 ((MIL GPU) Core ML Tools Unit Test failures [failure to load or Seg fault])")

        class TestNet(nn.Module):
            def forward(self, x):
                if torch.squeeze(x) < 10.:
                    return x*10.
                else:
                    return x*2.

        model = TestNet().eval()
        torch_model = torch.jit.script(model)

        self.run_compare_torch(torch.tensor([1.]), torch_model,
            input_as_shape=False, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion)
        self.run_compare_torch(torch.tensor([11.]), torch_model,
            input_as_shape=False, backend=backend,
            use_cpu_for_conversion=use_cpu_for_conversion)

class TestLoop(TorchBaseTest):
    @pytest.mark.parametrize("backend", backends)
    def test_for_loop(self, backend):
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

        self.run_compare_torch(model.input_size, torch_model, backend=backend)

    @pytest.mark.parametrize("backend", backends)
    def test_while_loop(self, backend):
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

        self.run_compare_torch(model.input_size, torch_model, backend=backend)


class TestUpsample(TorchBaseTest):
    @pytest.mark.parametrize(
        "output_size, align_corners, backend",
         itertools.product(
            [(10, 10), (1, 1), (2, 3), (190, 170)],
            [True, False],
            backends,
         )
    )
    def test_upsample_bilinear2d_with_output_size(
        self, output_size, align_corners, backend
    ):
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate,
            {"size": output_size, "mode": "bilinear", "align_corners": align_corners,},
        )
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "scales_h, scales_w, align_corners, recompute_scale_factor, backend",
         itertools.product(
            [2, 0.5, 4.1], [3, 0.5, 5.3], [True, False], [True, False], backends
         )
    )
    def test_upsample_bilinear2d_with_scales(
        self, scales_h, scales_w, align_corners, recompute_scale_factor, backend
    ):
        def _is_float_value(x, threshold=0.001):
            return x - np.floor(x) > threshold

        Height = 8
        Width = 22
        input_shape = (1, 3, Height, Width)
        output_h = Height * scales_h
        output_w = Width * scales_w
        is_h_float = _is_float_value(output_h)
        is_w_float = _is_float_value(output_w)

        if (is_h_float or is_w_float) and not align_corners and not recompute_scale_factor:
            pytest.xfail("rdar://81124053 (Support recompute_scale_factor)")

        model = ModuleWrapper(
            nn.functional.interpolate,
            {
                "scale_factor": (scales_h, scales_w),
                "mode": "bilinear",
                "align_corners": align_corners,
                "recompute_scale_factor": recompute_scale_factor,
            },
        )
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "output_size, backend",
         itertools.product(
           [(10, 10), (190, 170)], backends
         )
    )
    def test_upsample_nearest2d_with_output_size(self, output_size, backend):
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate, {"size": output_size, "mode": "nearest"},
        )
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "scales_h, scales_w, backend",
        itertools.product([2, 3, 4.5], [4, 5, 5.5], backends),
    )
    def test_upsample_nearest2d_with_scales(self, scales_h, scales_w, backend):
        if backend[0] == "neuralnetwork":
            if isinstance(scales_h, float) or isinstance(scales_w, float):
                return  # Skip fractional scale factors tests for neuralnetwork

        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate,
            {"scale_factor": (scales_h, scales_w), "mode": "nearest"},
        )
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "scales_h, scales_w, backend",
        itertools.product([2, 3], [4, 5], backends),
    )
    def test_upsample_nearest2d_with_scales_dynamic(self, scales_h, scales_w, backend):
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate,
            {"scale_factor": (scales_h, scales_w), "mode": "nearest", "recompute_scale_factor": True,},
        )
        converter_input_type = [TensorType(shape=(1, 3, RangeDim(), RangeDim()), dtype=np.float32)]
        mlmodel = self.run_compare_torch(input_shape, model,
                               backend=backend,
                               converter_input_type=converter_input_type)[1]

        # also check if the scale factor are integers
        if backend[0] == 'neuralnetwork':
            for layer in mlmodel._spec.neuralNetwork.layers:
                if layer.WhichOneof('layer') == "upsample":
                    assert len(layer.upsample.fractionalScalingFactor) == 0


    @pytest.mark.parametrize(
        "scales_h, scales_w, align_corners, recompute_scale_factor, backend",
        itertools.product(
           [2, 3.6], [4, 0.7], [True, False], [True, False], backends
        )
    )
    def test_upsample_bilinear2d_with_scales_dynamic(
        self, scales_h, scales_w, align_corners, recompute_scale_factor, backend
    ):
        def _is_float_value(x, threshold=0.001):
            return x - np.floor(x) > threshold

        is_h_float = _is_float_value(scales_h)
        is_w_float = _is_float_value(scales_w)
        input_shape = (1, 3, 9, 22)

        if (is_h_float or is_w_float) and not align_corners and not recompute_scale_factor:
            pytest.xfail("rdar://81124053 (Support recompute_scale_factor)")

        model = ModuleWrapper(
            nn.functional.interpolate,
            {
                "scale_factor": (scales_h, scales_w),
                "mode": "bilinear",
                "align_corners": align_corners,
                "recompute_scale_factor": recompute_scale_factor,
            },
        )
        converter_input_type = [TensorType(shape=(1, 3, RangeDim(default=9), RangeDim(default=22)), dtype=np.float32)]
        mlmodel = self.run_compare_torch(input_shape, model,
                               backend=backend,
                               converter_input_type=converter_input_type)[1]

        # also check if the scale factor are integers
        if backend[0] == 'neuralnetwork' and not is_h_float and not is_w_float:
            for layer in mlmodel._spec.neuralNetwork.layers:
                if layer.WhichOneof('layer') == "upsample":
                    assert len(layer.upsample.fractionalScalingFactor) == 0

class TestBranch(TorchBaseTest):
    @pytest.mark.parametrize("backend", backends)
    def test_if(self, backend):
        if backend[0] == 'mlprogram':
            pytest.xfail("Not supported on ML Program backend")

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

        self.run_compare_torch(model.input_size, torch_model, backend=backend)


class TestAvgPool(TorchBaseTest):

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, padding, ceil_mode, include_pad, backend",
        [ (*param, bend) for param, bend in itertools.product([
             ((1, 3, 5), 1, 1, 0, True, True),
             ((1, 3, 5), 3, 1, 0, False, True),
             ((1, 3, 5), 1, 2, 1, False, False),
             ((1, 3, 5), 3, 2, 1, False, True),
             ((1, 3, 5), 1, 2, 0, False, True),
             ((1, 3, 10), 1, 1, 1, False, False),
             ((1, 3, 10), 3, 1, 0, False, False),
             ((1, 3, 10), 1, 2, 1, True, True),
             ((1, 3, 10), 3, 2, 0, True, False),
             ((1, 3, 10), 1, 1, 1, True, True),
           ], backends)
        ],
    )
    def test_avg_pool1d(
        self, input_shape, kernel_size, stride, padding, ceil_mode, include_pad, backend
    ):
        if padding > kernel_size / 2:
            return
        if kernel_size == 1 and stride == 2 and padding == 0 and ceil_mode and input_shape[-1] % 2 == 0:
            pytest.xfail(reason="rdar://73894185 (CoreML sometimes returns 'nan's "
                                "for avg_pool when ceil_mode is True and kernel=1,stride=2,pad=0)")

        model = nn.AvgPool1d(
            kernel_size,
            stride,
            padding,
            ceil_mode=ceil_mode,
            count_include_pad=include_pad,
        )
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, padding, ceil_mode, include_pad, backend",
        [ (*param, bend) for param, bend in itertools.product([
             ((1, 3, 5, 5), 1, 1, 0, True, True),
             ((1, 3, 5, 5), 3, 1, 0, False, True),
             ((1, 3, 5, 5), 1, 2, 1, False, False),
             ((1, 3, 5, 5), 3, 2, 1, False, True),
             ((1, 3, 5, 5), 1, 2, 0, False, True),
             ((1, 3, 10, 10), 1, 1, 1, False, False),
             ((1, 3, 10, 10), 3, 1, 0, False, False),
             ((1, 3, 10, 10), 1, 2, 1, True, True),
             ((1, 3, 10, 10), 3, 2, 0, True, False),
             ((1, 3, 10, 10), 1, 1, 1, True, True),
           ], backends)
        ],
    )
    def test_avg_pool2d(
        self, input_shape, kernel_size, stride, padding, ceil_mode, include_pad, backend
    ):
        if padding > kernel_size / 2:
            return
        if kernel_size == 1 and stride == 2 and padding == 0 and ceil_mode and \
                (input_shape[-2] % 2 == 0 or input_shape[-1] % 2 == 0):
            pytest.xfail(reason="rdar://73894185 (CoreML sometimes returns 'nan's "
                                "for avg_pool when ceil_mode is True and kernel=1,stride=2,pad=0)")
        model = nn.AvgPool2d(
            kernel_size,
            stride,
            padding,
            ceil_mode=ceil_mode,
            count_include_pad=include_pad,
        )
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, padding, ceil_mode, include_pad, backend",
        [ (*param, bend) for param, bend in itertools.product([
             ((1, 3, 11, 5, 5), 1, 1, 0, True, True),
             ((1, 3, 11, 5, 5), 3, 1, 0, False, True),
             ((1, 3, 11, 5, 5), 1, 2, 1, False, False),
             ((1, 3, 11, 5, 5), 3, 2, 1, False, True),
             ((1, 3, 11, 5, 5), 1, 2, 0, False, True),
             ((1, 3, 6, 10, 10), 1, 1, 1, False, False),
             ((1, 3, 6, 10, 10), 3, 1, 0, False, False),
             ((1, 3, 6, 10, 10), 1, 2, 1, True, True),
             ((1, 3, 6, 10, 10), 3, 2, 0, True, False),
             ((1, 3, 6, 10, 10), 1, 1, 1, True, True),
           ], backends)
        ]
    )
    def test_avg_pool3d(
        self, input_shape, kernel_size, stride, padding, ceil_mode, include_pad, backend
    ):
        if padding > kernel_size / 2:
            return
        if kernel_size == 1 and stride == 2 and padding == 0 and ceil_mode and \
                (input_shape[-3] % 2 == 0 or input_shape[-2] % 2 == 0 or input_shape[-1] % 2 == 0):
            pytest.xfail(reason="rdar://73894185 (CoreML sometimes returns 'nan's "
                                "for avg_pool when ceil_mode is True and kernel=1,stride=2,pad=0)")
        if include_pad and ceil_mode and stride > 1:
            # skip: MIL/CoreML does not support this configuration
            # rdar://73723194
            return
        model = nn.AvgPool3d(
            kernel_size,
            stride,
            padding,
            ceil_mode=ceil_mode,
            count_include_pad=include_pad,
        )
        self.run_compare_torch(input_shape, model, backend=backend)

class TestAdaptiveMaxPool(TorchBaseTest):
    @pytest.mark.parametrize(
        "output_size, magnification, delta, depth, backend",
        itertools.product(
            [(1,1), (3,2)],
            [1, 2, 7],
            [0, 11],
            [1, 2, 3],
            backends,
        ),
    )
    def test_adaptive_max_pool2d(
            self, output_size, magnification, delta, depth, backend
    ):
        # input_size = output_size * magnification + delta
        input_size = (delta + magnification * output_size[0], delta + magnification * output_size[1])
        # since coremltools reproduces PyTorch's kernel sizes and
        # offsets for adaptive pooling layers only when input_size is
        # a multiple of output_size, we expect failures otherwise
        if not (input_size[0] % output_size[0]  == 0 and input_size[1] % output_size[1] == 0):
            pytest.xfail("Test should fail because input_size is not a multiple of output_size")
        n = 1
        in_shape = (n,depth) + input_size
        model = nn.AdaptiveMaxPool2d(
            output_size
        )
        self.run_compare_torch(in_shape, model, backend=backend)

class TestMaxPool(TorchBaseTest):

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, padding, ceil_mode, backend",
        itertools.product(
            [(1, 3, 15), (1, 1, 7)],
            [1, 3],
            [1, 2],
            [0, 1],
            [True, False],
            backends,
        ),
    )
    def test_max_pool1d(
        self, input_shape, kernel_size, stride, padding, ceil_mode, backend
    ):
        if padding > kernel_size / 2:
            return
        if ceil_mode > 0 and padding == 0 and kernel_size == 1 and stride == 2:
            if input_shape[-1] % 2 == 0:
                # TODO: is this a valid case?
                # in this case, torch adds "-inf" values at the border, post max pool operation
                return

        model = nn.MaxPool1d(
            kernel_size,
            stride,
            padding,
            dilation=1,
            return_indices=False,
            ceil_mode=ceil_mode,
        )
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, padding, ceil_mode, backend",
        itertools.product(
            [(1, 3, 15, 15), (1, 1, 7, 7)],
            [1, 3],
            [1, 2],
            [0, 1],
            [True, False],
            backends,
        ),
    )
    def test_max_pool2d(
        self, input_shape, kernel_size, stride, padding, ceil_mode, backend
    ):
        if padding > kernel_size / 2:
            return
        if ceil_mode > 0 and padding == 0 and kernel_size == 1 and stride == 2:
            for r in range(2,4):
                if input_shape[r] % 2 == 0:
                    # TODO: is this a valid case?
                    # in this case, torch adds "-inf" values at the border, post max pool operation
                    return

        model = nn.MaxPool2d(
            kernel_size,
            stride,
            padding,
            dilation=1,
            return_indices=False,
            ceil_mode=ceil_mode,
        )
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, padding, ceil_mode, backend",
        itertools.product(
            [(1, 3, 11, 3, 11), (1, 1, 7, 4, 7)],
            [1, 3],
            [1, 2],
            [0, 1],
            [True, False],
            backends,
        ),
    )
    def test_max_pool3d(
        self, input_shape, kernel_size, stride, padding, ceil_mode, backend
    ):
        if padding > kernel_size / 2:
            return
        if ceil_mode > 0 and padding == 0 and kernel_size == 1 and stride == 2:
            for r in range(2,5):
                if input_shape[r] % 2 == 0:
                    # TODO: is this a valid case?
                    # in this case, torch adds "-inf" values at the border, post max pool operation
                    return

        model = nn.MaxPool3d(
            kernel_size,
            stride,
            padding,
            dilation=1,
            return_indices=False,
            ceil_mode=ceil_mode,
        )
        self.run_compare_torch(input_shape, model, backend=backend)

class TestMaximumMinimum(TorchBaseTest):

    @pytest.mark.parametrize(
        "input_shape, mode, backend",
        itertools.product(
            [(2, 5, 7, 3), (3, 2, 9)],
            ["minimum", "maximum"],
            backends,
        ),
    )
    def test_minimum_maximum(self, input_shape, mode, backend):
        class TestModel(torch.nn.Module):
            def forward(self, x, y):
                if mode == "minimum":
                    return torch.minimum(x, y)
                elif mode == "maximum":
                    return torch.maximum(x, y)
                else:
                    raise ValueError("Unsupported mode: {mode}".format(mode=mode))

        model = TestModel()
        self.run_compare_torch([input_shape] * 2, model, backend=backend)

class TestPoolSymbolicInput(TorchBaseTest):
    def test_max_pool(self):
        model = nn.MaxPool2d(
            kernel_size=1,
            stride=2,
            padding=0,
            dilation=1,
            ceil_mode=True,
        )
        input_shape = (1, 1, 11, 11)
        converter_input_type = [TensorType(shape=(1, 1, RangeDim(), RangeDim()), dtype=np.float32)]
        self.run_compare_torch(input_shape, model,
                               backend=backends[0],
                               converter_input_type=converter_input_type)

    def test_avg_pool(self):
        model = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
            padding=1,
            count_include_pad=True,
            ceil_mode=True,
        )
        input_shape = (1, 2, 15, 15)
        converter_input_type = [TensorType(shape=(1, 2, RangeDim(), RangeDim()), dtype=np.float32)]
        self.run_compare_torch(input_shape, model,
                               backend=backends[0],
                               converter_input_type=converter_input_type)


class TestLSTM(TorchBaseTest):
    @pytest.mark.parametrize(
        "input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, backend",
        [ (*param, bend) for param, bend in itertools.product([
             (1, 1, 1, True, True, 0.3, True),
             (1, 1, 1, False, True, 0.3, False),
             (1, 1, 1, False, True, 0.3, True),
             (3, 1, 5, True, False, 0.3, False),
             (3, 1, 5, True, True, 0.3, True),
             (3, 7, 5, True, False, 0.3, False),
             (3, 7, 5, False, True, 0.3, True),
             (3, 7, 5, False, True, 0.3, False),
           ], backends)
        ],
    )
    def test_lstm(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
        backend,
    ):
        model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        SEQUENCE_LENGTH = 3
        BATCH_SIZE = 2
        model.eval()

        num_directions = int(bidirectional) + 1

        if batch_first:
            _input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, input_size)
        else:
            _input = torch.randn(SEQUENCE_LENGTH, BATCH_SIZE, input_size)

        h0 = torch.randn(num_layers * num_directions, BATCH_SIZE, hidden_size)
        c0 = torch.randn(num_layers * num_directions, BATCH_SIZE, hidden_size)

        inputs = (_input, (h0, c0))
        expected_results = model(*inputs)
        self.run_compare_torch(
            inputs, model, expected_results, input_as_shape=False, backend=backend,
        )

class TestRNN(TorchBaseTest):
    @pytest.mark.parametrize(
        "input_size, hidden_size, num_layers, bias, batch_first, dropout, activation, backend",
        [ (*param, bend) for param, bend in itertools.product([
             (1, 1, 1, True, True, 0.3, "tanh"),
             (1, 1, 1, False, True, 0.3, "relu"),
             (1, 1, 1, False, True, 0.3, "tanh"),
             (3, 1, 5, True, False, 0.3, "relu"),
             (3, 1, 5, True, True, 0.3, "tanh"),
             (3, 7, 5, True, False, 0.3, "relu"),
             (3, 7, 5, False, True, 0.3, "relu"),
             (3, 7, 5, False, True, 0.3, "tanh"),
           ], backends)
        ],
    )
    def test_rnn(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        activation,
        backend,
    ):
        SEQUENCE_LENGTH = 10
        BATCH_SIZE = 3
        model = nn.RNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bias=bias,
                    batch_first=batch_first,
                    dropout=dropout,
                    nonlinearity=activation,
                    bidirectional=False,     # bi-directional simple RNN not supported
                )
        model.eval()
        num_directions = 1

        if batch_first:
            _input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, input_size)
        else:
            _input = torch.randn(SEQUENCE_LENGTH, BATCH_SIZE, input_size)

        h0 = torch.randn(num_layers * num_directions, BATCH_SIZE, hidden_size)
        inputs = (_input, h0)
        expected_results = model(*inputs)

        self.run_compare_torch(
            inputs, model, expected_results, input_as_shape=False, backend=backend,
        )

class TestGRU(TorchBaseTest):
    @pytest.mark.parametrize(
        "input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, backend",
        [ (*param, bend) for param, bend in itertools.product([
             (1, 1, 1, True, True, 0.3, True),
             (1, 1, 1, False, True, 0.3, True),
             (1, 1, 1, False, True, 0.3, False),
             (3, 1, 5, True, False, 0.3, False),
             (3, 1, 5, True, True, 0.3, True),
             (3, 7, 5, True, False, 0.3, False),
             (3, 7, 5, False, True, 0.3, False),
             (3, 7, 5, False, True, 0.3, True),
           ], backends)
        ],
    )
    def test_gru(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
        backend,
    ):
        SEQUENCE_LENGTH = 10
        BATCH_SIZE = 3
        model = nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bias=bias,
                    batch_first=batch_first,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
        model.eval()
        num_directions = int(bidirectional) + 1

        if batch_first:
            _input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, input_size)
        else:
            _input = torch.randn(SEQUENCE_LENGTH, BATCH_SIZE, input_size)

        h0 = torch.randn(num_layers * num_directions, BATCH_SIZE, hidden_size)

        inputs = (_input, h0)
        expected_results = model(*inputs)

        self.run_compare_torch(
            inputs, model, expected_results, input_as_shape=False, backend=backend,
        )

class TestLSTMWithPackedSequence(TorchBaseTest):
    @pytest.mark.parametrize(
        "pack_batch_first, pad_batch_first, LSTM_batch_first, pad_value, backend",
        itertools.product(
            [True, False], [True, False], [True, False], [-1,0], backends
        ),
    )
    def test_lstm(
        self,
        pack_batch_first,
        pad_batch_first,
        LSTM_batch_first,
        pad_value,
        backend,
    ):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        input_size = 4
        hidden_size = 6
        num_layers = 1

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=LSTM_batch_first,
                    bidirectional=False,
                    dropout=0.0,
                )

            def forward(self, batch_in, seq_lengths):
                packed_input = pack_padded_sequence(batch_in, seq_lengths, batch_first=pack_batch_first)
                output_packed, (hidden, _) = self.lstm(packed_input)
                output, _ = pad_packed_sequence(output_packed, padding_value=pad_value, batch_first=pad_batch_first)
                return output

        SEQUENCE_LENGTH = 10
        BATCH_SIZE = 3
        model = Encoder()
        model.eval()

        if pack_batch_first:
            _input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, input_size)
        else:
            _input = torch.randn(SEQUENCE_LENGTH, BATCH_SIZE, input_size)

        seq_lengths = torch.tensor([10, 5, 1], dtype=int)

        inputs = (_input, seq_lengths)
        expected_results = model(*inputs)
        self.run_compare_torch(
            inputs, model, expected_results, input_as_shape=False, backend=backend,
        )

# Workaround for GitHub Issue #824
# i.e. the return h_n/c_n for a converted BLSTM are mangled.
# Therefore, just look at output 'y' (for now) which is correct.
class StripCellAndHidden(nn.Module):
    def __init__(self,flagReturnTuple_):
        super(StripCellAndHidden, self).__init__()
        self.flagReturnTuple = flagReturnTuple_

    def forward(self,x):
        # Pass tuple, not tensor, to avoid issue in coremltools/converters/mil/frontend/torch/test/testing_utils.py on "if not expected_results:"
        # Pass tensor when we need input for LSTM #2 as part of nn.Sequential()
        return tuple(x[0]) if self.flagReturnTuple else x[0]

# Check GitHub Issue #810, assume num_layers == 2 and bidirectional == True
class TestStackedBLSTM(TorchBaseTest):
    @pytest.mark.parametrize(
        "input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, backend",
        itertools.product([7], [5], [2], [True, False], [True, False], [0.3], [True], backends),
    )
    def test_lstm(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
        backend,
    ):
        model = nn.Sequential(
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=True),
            StripCellAndHidden(False),
            nn.LSTM(
                input_size=2*hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=True),
            StripCellAndHidden(True)
        )

        SEQUENCE_LENGTH = 3
        BATCH_SIZE = 2

        # (seq_len, batch, input_size)
        if batch_first:
            _input = torch.rand(BATCH_SIZE, SEQUENCE_LENGTH, input_size)
        else:
            _input = torch.randn(SEQUENCE_LENGTH, BATCH_SIZE, input_size)

        # Do not use h_0/c_0 input and do not check h_n/c_n output, GitHub Issue #824
        expected_results = model(_input)

        self.run_compare_torch(_input, model, expected_results,
                           input_as_shape=False, backend=backend)


class TestConcat(TorchBaseTest):
    # This tests an edge case where the list of tensors to concatenate only
    # has one item. NN throws an error for this case, hence why we have to
    # run through the full conversion process to test it.
    @pytest.mark.parametrize("backend", backends)
    def test_cat(self, backend):
        class TestNet(nn.Module):
            def __init__(self):
                super(TestNet, self).__init__()

            def forward(self, x):
                x = torch.cat((x,), axis=1)
                return x

        model = TestNet()
        self.run_compare_torch((1, 3, 16, 16), model, backend=backend)

class TestFull(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [1, 3],
        ),
    )
    def test_full_dynamic(self, backend, rank):
        class FullDynamicModel(nn.Module):
            def __init__(self):
                super(FullDynamicModel, self).__init__()

            def forward(self, x):
                if rank == 1:
                    h = x[0]
                    x = torch.zeros(h)
                elif rank == 3:
                    h, w, d = x[0], x[1], x[2]
                    x = torch.zeros(h, w, d)
                return torch.full(x.shape, fill_value=3.14)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        torch_in = torch.tensor(input_shape)
        model = FullDynamicModel().eval()
        torch_out = model(torch_in)
        self.run_compare_torch(torch_in, model, expected_results=torch_out,
                           input_as_shape=False, backend=backend)

    @pytest.mark.parametrize("shape_val, backend",
        itertools.product(
            [
                [(1,), 0.],
                [(2, 3), 3.1415],
                [(1, 1, 2, 5, 1), -2.],
            ],
            backends,
            )
        )
    def test_full_static(self, shape_val, backend):
        shape, val = shape_val
        class FullStaticModel(nn.Module):
            def __init__(self):
                super(FullStaticModel, self).__init__()

            def forward(self, x):
                return torch.full(x.shape, fill_value=val)

        self.run_compare_torch(shape, FullStaticModel().eval(), backend=backend)

class TestOnes(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [1, 3],
        ),
    )
    def test_ones_dynamic(self, backend, rank):
        class OnesDynamicModel(nn.Module):
            def __init__(self):
                super(OnesDynamicModel, self).__init__()

            def forward(self, x):
                if rank == 1:
                    h = x[0]
                    x = torch.zeros(h)
                elif rank == 3:
                    h, w, d = x[0], x[1], x[2]
                    x = torch.zeros(h, w, d)
                return torch.ones(x.shape)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        torch_in = torch.tensor(input_shape)
        model = OnesDynamicModel().eval()
        torch_out = model(torch_in)
        self.run_compare_torch(torch_in, model, expected_results=torch_out,
                           input_as_shape=False, backend=backend)

    @pytest.mark.parametrize("shape, backend",
        itertools.product(
            [(1,), (2, 3), (1, 1, 2, 5, 1)],
            backends,
            )
        )
    def test_ones_static(self, shape, backend):
        class OnesStaticModel(nn.Module):
            def __init__(self):
                super(OnesStaticModel, self).__init__()

            def forward(self, x):
                return torch.ones(x.shape)

        self.run_compare_torch(shape, OnesStaticModel().eval(), backend=backend)

class TestTypeAs(TorchBaseTest):
    @pytest.mark.parametrize("backend, type",
        itertools.product(
            backends,
            ["int32", "float16", "float32", "bool"]
            )
        )
    def test_type_as(self, backend, type):
        class TestNet(nn.Module):
            def __init__(self):
                super(TestNet, self).__init__()

            def forward(self, x, y):
                return x.type_as(y)

        model = TestNet()
        type_map = {
            "int32": torch.int32,
            "float16": torch.float16,
            "float32": torch.float32,
            "bool": torch.bool,
        }
        input = [
            torch.Tensor([0,1,2,3]).to(torch.float32),
            torch.Tensor([2,3]).to(type_map[type]),
        ]
        self.run_compare_torch(input, model, backend=backend, input_as_shape=False)


class TestReduction(TorchBaseTest):
    @pytest.mark.parametrize(
        "input_shape, dim, keepdim, mode, backend",
        itertools.product([(2, 2), (1, 1)], [0, 1], [True, False], ["min", "max"], backends)
    )
    def test_min_max(self, input_shape, dim, keepdim, mode, backend):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()

            def forward(self, x):
                if mode == "min":
                    return torch.min(x, dim=dim, keepdim=keepdim)
                elif mode == "max":
                    return torch.max(x, dim=dim, keepdim=keepdim)
                else:
                    raise ValueError("Unsupported mode: {mode}".format(mode=mode))

        input_data = torch.rand(input_shape)
        model = TestModel()
        # rdar://62681982 (Determine the output names of MLModels)
        expected_results = model(input_data)[::-1]
        self.run_compare_torch(
            input_data,
            model,
            expected_results=expected_results,
            input_as_shape=False,
            backend=backend,
        )


class TestLayerNorm(TorchBaseTest):
    @pytest.mark.parametrize(
        "input_shape, eps, backend",
        itertools.product([(1, 3, 15, 15), (1, 1, 1, 1)], [1e-5, 1e-7], backends),
    )
    def test_layer_norm(self, input_shape, eps, backend):
        model = nn.LayerNorm(input_shape, eps=eps)
        self.run_compare_torch(input_shape, model, backend=backend)


class TestPixelShuffle(TorchBaseTest):
    @pytest.mark.parametrize(
        "batch_size, CHW, r, backend",
        itertools.product([1, 3], [(1, 4, 4), (3, 2, 3)], [2, 4], backends),
    )
    def test_pixel_shuffle(self, batch_size, CHW, r, backend):
        C, H, W = CHW
        input_shape = (batch_size, C * r * r, H, W)
        model = nn.PixelShuffle(upscale_factor=r)
        self.run_compare_torch(input_shape, model, backend=backend)


class TestExpand(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, shapes",
        itertools.product(
            backends,
            [[(2, 1), (2, 2)], [(3, 1), (-1, 4)], [(1, 3, 4, 4), (3, 3, 4, 4)], [(4,), (3, 4)], [(3, 2), (1, 2, -1, 2)]]
        ),
    )
    def test_expand(self, backend, shapes):
        input_shape, output_shape = shapes

        class TestModel(torch.nn.Module):
            def forward(self, x):
                return x.expand(*output_shape)

        model = TestModel()

        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "backend, input_shapes",
        itertools.product(
            backends,
            [[(2, 1), (2, 2)], [(3, 1), (3, 4)], [(1, 3, 4, 4), (3, 3, 4, 4)], [(4,), (1, 3, 4)]]
        ),
    )
    def test_expand_as(self, backend, input_shapes):
        class TestModel(torch.nn.Module):
            def forward(self, x, y):
                return x.expand_as(y)

        model = TestModel()

        self.run_compare_torch(input_shapes, model, backend=backend)


class TestExpandDims(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank_and_axis",
        itertools.product(
            backends,
            [
                (rank, axis)
                for rank in range(1, 5)
                for axis in range(-rank - 1, rank + 1)
            ],
        ),
    )
    def test_unsqueeze(self, backend, rank_and_axis):
        rank, axis = rank_and_axis
        input_shape = tuple(np.random.randint(low=2, high=10, size=rank))
        model = ModuleWrapper(function=torch.unsqueeze, kwargs={"dim": axis})
        self.run_compare_torch(input_shape, model, backend=backend)

class TestEinsum(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, equation, reverse_input_order",
        itertools.product(
            backends,
            ["abcd,adce->abce",
             "abc,cbd->abd",
             "bnqd,bnkd->bnqk",
             "abc,cd->abd",
             "abc,cde->abde",
             "btnh,bfnh->bnft",
             "bnft,btnh->bfnh",
             "abcd,cde->abe"],
            [False, True]
        ),
    )
    def test_einsum(self, backend, equation, reverse_input_order):
        class TestEinsum(nn.Module):
            def __init__(self):
                super(TestEinsum, self).__init__()

            def forward(self, x, y):
                return torch.einsum(equation, x, y)

        if equation == "abcd,adce->abce":
            input_shapes = [[3, 4, 2, 6], [3, 6, 2, 2]]
        elif equation == "abc,cbd->abd":
            input_shapes = [[4, 2, 6], [6, 2, 2]]
        elif equation == "bnqd,bnkd->bnqk":
            input_shapes = [[1,2,3,4], [1,2,4,4]]
        elif equation == "abc,cd->abd":
            input_shapes = [[2,3,4], [4,5]]
        elif equation ==  "abc,cde->abde":
            input_shapes = [[2,3,4], [4,5,6]]
        elif equation == "btnh,bfnh->bnft":
            input_shapes = [[1,2,3,4], [1,5,3,4]]
        elif equation == "bnft,btnh->bfnh":
            input_shapes = [[1,2,3,4], [1,4,2,6]]
        elif equation == "abcd,cde->abe":
            input_shapes = [[1,2,3,4], [3,4,6]]
        else:
            raise ValueError("unrecognized equation")

        if reverse_input_order:
            input_output_strings = equation.split('->')
            input_strings = input_output_strings[0].split(',')
            equation = input_strings[1] + ',' + input_strings[0] + '->' + input_output_strings[1]
            input_shapes = [input_shapes[1], input_shapes[0]]

        model = TestEinsum()
        self.run_compare_torch(input_shapes, model, backend=backend, input_as_shape=True)

class TestSqueeze(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank_and_axis",
        itertools.product(
            backends,
            [(2, 1), (2, 0), (3, 1), (3, None), (4, None), (4, 2), (5, None), (5, -1),],
        ),
    )
    def test_squeeze(self, backend, rank_and_axis):
        rank, axis = rank_and_axis
        input_shape = list(np.random.randint(low=2, high=10, size=rank))
        if axis is not None:
            input_shape[axis] = 1
        else:
            input_shape[0] = 1
        input_shape = tuple(input_shape)
        model = ModuleWrapper(
            function=torch.squeeze, kwargs={"dim": axis} if axis else {}
        )
        self.run_compare_torch(input_shape, model, backend=backend)

class TestCumSum(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, axis",
        itertools.product(
            backends,
            [-1, 0, 1, 2, 3],
        ),
    )
    def test_cumsum(self, backend, axis):
        input_shape = list(np.random.randint(low=2, high=10, size=4))
        input_shape = tuple(input_shape)
        model = ModuleWrapper(
            function=torch.cumsum, kwargs={"dim": axis}
        )
        self.run_compare_torch(input_shape, model, backend=backend)


class TestReshape(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, output_shape",
        itertools.product(backends, [(3, 2), (2, -1), (2, 1, 1, 3),],),
    )
    def test_reshape(self, backend, output_shape):
        input_shape = (2, 3)
        model = ModuleWrapper(function=torch.reshape, kwargs={"shape": output_shape})
        self.run_compare_torch(input_shape, model, backend=backend)


class TestFlatten(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, start_dim",
        itertools.product(backends, [2,-2],),
    )
    def test_reshape(self, backend, start_dim):
        input_shape = (2, 3, 4, 5)
        model = ModuleWrapper(function=torch.flatten, kwargs={"start_dim": start_dim})
        self.run_compare_torch(input_shape, model, backend=backend)


class TestGather(TorchBaseTest):
    @pytest.mark.parametrize(
        "rank_and_axis, backend",
        itertools.product([(i, j) for i in range(1, 6) for j in range(0, i)], backends),
    )
    def test_gather_along_axis(self, rank_and_axis, backend):
        rank, axis = rank_and_axis
        params_shape = np.random.randint(low=2, high=5, size=rank)
        indices_shape = np.copy(params_shape)
        indices_shape[axis] = np.random.randint(low=1, high=8)
        indices = np.random.randint(0, params_shape[axis], size=indices_shape)
        params_shape, indices_shape = tuple(params_shape), tuple(indices_shape)
        model = ModuleWrapper(
            function=torch.gather,
            kwargs={"dim": axis, "index": torch.from_numpy(indices)},
        )
        self.run_compare_torch([params_shape], model, backend=backend)


class TestActivation(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(backends, COMMON_SHAPES_ALL),
    )
    def test_relu(self, backend, shape):
        model = nn.ReLU().eval()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

        model = ModuleWrapper(nn.functional.relu_)
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(backends, COMMON_SHAPES_ALL),
    )
    def test_relu6(self, backend, shape):
        model = nn.ReLU6().eval()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, alpha", itertools.product(backends, [0.1, 0.25, 2.0]),
    )
    def test_prelu(self, backend, alpha):
        input_shape = (1, 5, 6, 7)
        C = input_shape[1]
        model = nn.PReLU(C, alpha).eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape, alpha",
        itertools.product(backends,
            COMMON_SHAPES_ALL,
            [0.1, 2.0, 1.4]
        )
    )
    def test_leaky_relu(self, backend, shape, alpha):
        model = nn.LeakyReLU(negative_slope=alpha).eval()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

        model = ModuleWrapper(nn.functional.leaky_relu_, {'negative_slope': alpha})
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(backends, COMMON_SHAPES_ALL),
    )
    def test_softmax(self, backend, shape):
        model = nn.Softmax().eval()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, range_val",
        itertools.product(
            backends, [(-1.0, 1.0), (0.0, 0.1), (1.0, 3.0), (-1.0, 6.0)]
        ),
    )
    def test_hardtanh(self, backend, range_val):
        input_shape = (1, 10, 4, 5)
        model = nn.Hardtanh(range_val[0], range_val[1]).eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

        model = ModuleWrapper(nn.functional.hardtanh_,
                     {'min_val': range_val[0], 'max_val': range_val[1]})
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape, alpha",
        itertools.product(backends,
            COMMON_SHAPES_ALL,
            [0.1, 2.0, 1.4]
        )
    )
    def test_elu(self, backend, shape, alpha):
        model = nn.ELU(alpha).eval()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(backends, COMMON_SHAPES_ALL)
    )
    def test_gelu(self, backend, shape):
        model = nn.GELU().eval()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.skipif(_python_version() < (3, 6), reason="requires python 3.6")
    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(backends, COMMON_SHAPES_ALL),
    )
    def test_erf(self, backend, shape):

        class ERFActivation(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.erf(x)

        model = ERFActivation().eval()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(backends,
            [(1, 10), (1, 3, 5), (1, 5, 6, 7), (1, 3, 4, 5, 6)]
        ),
    )
    def test_sigmoid(self, backend, shape):
        model = nn.Sigmoid().eval()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.skipif(_python_version() < (3, 6), reason="requires python 3.6")
    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(backends, COMMON_SHAPES_ALL)
    )
    def test_sigmoid_hard(self, backend, shape):
        model = nn.Hardsigmoid().eval()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, beta, threshold", itertools.product(backends, [1, 2, 5], [5, 10, 20]),
    )
    @pytest.mark.skipif(
        _macos_version() <= (10, 15),
        reason="Parametric SoftPlus segfaults on macOS 10.15 and below.",
    )
    def test_softplus(self, backend, beta, threshold):
        input_shape = (1, 10, 5, 15)
        model = nn.Softplus(beta, threshold).eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(backends, COMMON_SHAPES_ALL)
    )
    def test_softsign(self, backend, shape):
        model = nn.Softsign().eval()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.skipif(
        condition=version_lt(torch, "1.7.0"),
        reason="torch.nn.SiLU available only in PyTorch 1.7.0+",
    )
    @pytest.mark.parametrize(
        "shape, backend",
        itertools.product([(1, 10), (1, 3, 4), (1, 4, 5, 6)], backends),
    )
    def test_silu(self, shape, backend):
        model = ModuleWrapper(function=torch.nn.functional.silu)
        self.run_compare_torch([shape], model, backend=backend)

    @pytest.mark.parametrize(
        "rounding_mode, backend",
        itertools.product([None, "floor", "trunc"], backends),
    )
    def test_div(self, rounding_mode, backend):
        model = ModuleWrapper(function=torch.div,
                              kwargs={"rounding_mode": rounding_mode})
        x1 = torch.from_numpy(np.array([2.3, 2.6, -3.6, -3.2], dtype=np.float32))
        x2 = torch.from_numpy(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        out = torch.div(x1, x2, rounding_mode=rounding_mode)
        self.run_compare_torch(
            [x1, x2],
            model,
            backend=backend,
            input_as_shape=False,
            expected_results=out,
        )


class TestElementWiseUnary(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, shape, op_string",
        itertools.product(
            backends,
            [(1, 3, 5, 8)],
            [
                "abs",
                "acos",
                "asin",
                "atan",
                "ceil",
                "cos",
                "cosh",
                "exp",
                "floor",
                "round",
                "sin",
                "sinh",
                "sqrt",
                "square",
                "tan",
                "tanh",
                "sign",
            ],
        ),
    )
    def test_elementwise_no_params(self, backend, shape, op_string):
        if not contains_op(torch, op_string):
            return
        op_func = getattr(torch, op_string)
        model = ModuleWrapper(function=op_func)
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape, clamp_range",
        itertools.product(
            backends,
            [(1, 3, 5, 8)],
            [(0.0, 1.0), (-1.0, 0.5), (0.2, 0.7), (None, 4.0), (-3.0, None)],
        ),
    )
    def test_clamp(self, backend, shape, clamp_range):
        params_dict = {}
        if clamp_range[0] is not None:
            params_dict["min"] = clamp_range[0]
        if clamp_range[1] is not None:
            params_dict["max"] = clamp_range[1]

        model = ModuleWrapper(torch.clamp, params_dict)
        self.run_compare_torch(
            shape, model, backend=backend, rand_range=(-5, 5)
        )

    @pytest.mark.parametrize(
        "backend, shape, threshold",
        itertools.product(
            backends,
            [(1, 3, 5, 8)],
            [(0.0, 0.0), (0.5, 0.5), (0.5, 10), (0.9, 0.0)]
        ),
    )
    def test_threshold(self, backend, shape, threshold):
        model = torch.nn.Threshold(threshold[0], threshold[1]).eval()
        self.run_compare_torch(
            shape, model, backend=backend,
            use_cpu_for_conversion=True, # TODO: change this to False (rdar://78343191)
        )

    @pytest.mark.parametrize(
        "backend, shape, op_string",
        itertools.product(
            backends,
            [(1, 3, 5, 8)],
            [
                "log",
                "rsqrt",
                "reciprocal",
            ],
        ),
    )
    def test_elementwise_numerically_stable(self, backend, shape, op_string):
        op_func = getattr(torch, op_string)
        model = ModuleWrapper(function=op_func)
        self.run_compare_torch(
            shape, model, backend=backend, rand_range=(20, 100)
        )

class TestMatMul(TorchBaseTest):

    @pytest.mark.parametrize("backend", backends)
    def test_bmm(self, backend):
        shape_x, shape_y = (3,4,5), (3,5,6)
        model = ModuleWrapper(function=torch.bmm)
        self.run_compare_torch(
            [shape_x, shape_y], model, backend=backend,
        )


class TestSplit(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, split_size_or_sections, dim",
        itertools.product(backends, [1, 2, [1, 4]], [0, -2]),
    )
    def test_split(self, backend, split_size_or_sections, dim):
        input_shape = (5, 2)
        model = ModuleWrapper(function=torch.split,
                              kwargs={"split_size_or_sections": split_size_or_sections, "dim": dim})
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "backend, split_sizes, dim",
        itertools.product(backends, [[1, 4], [3, 2]], [-1, -2]),
    )
    def test_split_with_sizes(self, backend, split_sizes, dim):
        input_shape = (5, 5)
        model = ModuleWrapper(function=torch.split_with_sizes,
                              kwargs={"split_sizes": split_sizes, "dim": dim})
        self.run_compare_torch(input_shape, model, backend=backend)


class TestUnbind(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, dim",
        itertools.product(backends,[0,1,2]),
    )
    def test_unbind(self, backend, dim):
        input_shape = (3, 3, 4)
        model = ModuleWrapper(function=torch.unbind,
                              kwargs={"dim": dim})
        self.run_compare_torch(input_shape, model, backend=backend)


class TestTranspose(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, shape, dims",
        itertools.product(backends, COMMON_SHAPES, [(0, 1), (-2, -1), (1, 0), (-1, -2)]),
    )
    def test(self, backend, shape, dims):
        model = ModuleWrapper(function=torch.transpose,
                              kwargs={"dim0": dims[0], "dim1": dims[1]})
        self.run_compare_torch(shape, model, backend=backend)

class TestTo(TorchBaseTest):
    @pytest.mark.parametrize(
        "use_cpu_for_conversion, backend", itertools.product([True, False], backends,)
    )
    def test_cast_bug(self, use_cpu_for_conversion, backend):
        if backend[0] == "mlprogram" and not use_cpu_for_conversion:
            pytest.xfail("rdar://78343191 ((MIL GPU) Core ML Tools Unit Test failures [failure to load or Seg fault])")

        if backend[0] == "mlprogram" and use_cpu_for_conversion:
            pytest.xfail("numerical mismatch : rdar://78952850")

        class TestModel(torch.nn.Module):
            def forward(self, spans, embedding):
                spans = spans.float().relu().int()

                max1, _ = torch.max(spans, dim=1, keepdim=False)
                max1, _ = torch.max(max1, dim=1, keepdim=False)
                max2, _ = torch.max(embedding, dim=1, keepdim=False)
                max2, _ = torch.max(max2, dim=1, keepdim=False)
                sigmoided_scores = max1 + max2
                return sigmoided_scores

        model = TestModel()
        self.run_compare_torch([(1, 21, 2), (1, 6, 384)], model, backend=backend,
                               use_cpu_for_conversion=use_cpu_for_conversion)# [spans.shape, embedding.shape]

class TestSlice(TorchBaseTest):
    @pytest.mark.skipif(_python_version() < (3, 6), reason="requires python 3.6")
    @pytest.mark.parametrize(
        "backend", backends,
    )
    def test_dynamic_slice(self, backend):
        class DynamicSlicer(torch.nn.Module):
            def __init__(self):
                super(DynamicSlicer, self).__init__()

            def forward(self, x, context_length):
                return x[context_length:, :, :]

        class Model(torch.nn.Module):

            def __init__(self):
                super(Model, self).__init__()
                self.tokens_embedding = torch.nn.Embedding(10, 10, 0)
                self.context_embedding = torch.nn.Embedding(10, 10, 0)
                self.dynamic_slicer = DynamicSlicer()

            def forward(self, tokens, context, context_length):
                # CoreML requires rank1~5 input, so we use rank 1 for
                # context-length
                tokens_embeddings = self.tokens_embedding(tokens)
                context_embeddings = self.context_embedding(context)
                embeddings = torch.cat((context_embeddings, tokens_embeddings), dim=0)
                embeddings = self.dynamic_slicer(embeddings,
                        torch.squeeze(context_length))

                return embeddings

        model = Model()
        batch_size = 5
        inputs = [ TensorType(name="tokens", shape=(10, batch_size), dtype=np.int64),
                   TensorType(name="context", shape=(3, batch_size), dtype=np.int64),
                   TensorType(name="context_length", shape=(1,), dtype=np.int32),
                   ]
        self.run_compare_torch(inputs, model, rand_range=(0, 8),
                               backend=backend, use_scripting=False)


class TestRepeat(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(backends, list(range(1, 6))),
    )
    def test_repeat(self, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        repeats = np.random.randint(low=2, high=4, size=rank)
        input_shape = tuple(input_shape)

        model = ModuleWrapper(function=lambda x: x.repeat(*repeats))
        self.run_compare_torch(input_shape, model, backend=backend)

class TestStd(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, unbiased",
        itertools.product(backends, [True, False]),
    )
    def test_std_2_inputs(self, backend, unbiased):
        model = ModuleWrapper(function=torch.std,
                              kwargs={"unbiased": unbiased})
        x = torch.randn(1, 5, 10) * 3
        out = torch.std(x, unbiased=unbiased).unsqueeze(0)
        self.run_compare_torch(x, model, expected_results=out,
                           input_as_shape=False, backend=backend)


    @pytest.mark.parametrize(
        "backend, unbiased, dim, keepdim",
        itertools.product(backends, [True, False], [[0,2], [1], [2]], [True, False]),
    )
    def test_std_4_inputs(self, backend, unbiased, dim, keepdim):
        model = ModuleWrapper(function=torch.std,
                              kwargs={"unbiased": unbiased, "dim" : dim, "keepdim": keepdim})
        input_shape = (2, 5, 10)
        self.run_compare_torch(input_shape, model, backend=backend)

class TestZeros(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [1, 3],
        ),
    )
    def test_zeros_like_static(self, backend, rank):
        if backend[0] == 'mlprogram':
            pytest.xfail("Not supported with ML Program backend")

        class ZerosLikeStaticModel(nn.Module):
            def __init__(self):
                super(ZerosLikeStaticModel, self).__init__()

            def forward(self, x):
                return torch.zeros_like(x)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        input_shape = tuple(input_shape)
        model = ZerosLikeStaticModel()
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [1, 3],
        ),
    )
    def test_zeros_like_dynamic(self, backend, rank):
        if backend[0] == 'mlprogram':
            pytest.xfail("Not supported with ML Program backend")

        class ZerosLikeDynamicModel(nn.Module):
            def __init__(self):
                super(ZerosLikeDynamicModel, self).__init__()

            def forward(self, x):
                if rank == 1:
                    h = x[0]
                    x = torch.zeros(h)
                elif rank == 3:
                    h, w, d = x[0], x[1], x[2]
                    x = torch.zeros(h, w, d)
                return torch.zeros_like(x)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        torch_in = torch.tensor(input_shape)
        model = ZerosLikeDynamicModel()
        torch_out = model(torch_in)
        self.run_compare_torch(torch_in, model, expected_results=torch_out,
                           input_as_shape=False, backend=backend)

    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [1, 3],
        ),
    )
    def test_zeros_static(self, backend, rank):
        if backend[0] == 'mlprogram':
            pytest.xfail("Not supported with ML Program backend")

        class ZerosStaticModel(nn.Module):
            def __init__(self):
                super(ZerosStaticModel, self).__init__()

            def forward(self, x):
                if rank == 1:
                    return torch.zeros(1)
                elif rank == 3:
                    return torch.zeros(2, 3, 5)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        input_shape = tuple(input_shape)
        model = ZerosStaticModel()
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [1, 3],
        ),
    )
    def test_zeros_dynamic(self, backend, rank):
        if backend[0] == 'mlprogram':
            pytest.xfail("Not supported with ML Program backend")

        class ZerosDynamicModel(nn.Module):
            def __init__(self):
                super(ZerosDynamicModel, self).__init__()

            def forward(self, x):
                if rank == 1:
                    h = x[0]
                    x = torch.zeros(h)
                elif rank == 3:
                    h, w, d = x[0], x[1], x[2]
                    x = torch.zeros(h, w, d)
                return x

        input_shape = np.random.randint(low=2, high=6, size=rank)
        torch_in = torch.tensor(input_shape)
        model = ZerosDynamicModel()
        torch_out = model(torch_in)
        self.run_compare_torch(torch_in, model, expected_results=torch_out,
                           input_as_shape=False, backend=backend)

class TestTopk(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, largest, shape_dim_k",
        itertools.product(
            backends,
            [True, False],
            [
             ((4, 6, 7, 3), -1, 2),
             ((10, 3, 4), 2, 2),
             ((5,), 0, 2)
             ],
        ),
    )
    def test_topk(self, backend, largest, shape_dim_k):
        input_shape = shape_dim_k[0]
        dim = shape_dim_k[1]
        k = shape_dim_k[2]

        class TopkModel(nn.Module):
            def __init__(self):
                super(TopkModel, self).__init__()

            def forward(self, x):
                return torch.topk(x, k, dim=dim, largest=largest)

        input_data = torch.rand(input_shape)
        model = TopkModel()
        expected_results = model(input_data)
        expected_results = [expected_results.values, expected_results.indices]
        self.run_compare_torch(
            input_data,
            model,
            expected_results=expected_results,
            input_as_shape=False,
            backend=backend,
        )

class TestLog10(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank", itertools.product(backends, range(1, 6)),
    )
    def test_log10(self, backend, rank):

        class Log10Model(nn.Module):
            def __init__(self):
                super(Log10Model, self).__init__()

            def forward(self, x):
                return torch.log10(x)

        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = Log10Model()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

class TestFlip(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank_dim",
        itertools.product(
            backends,
            [
                (1, [0]),
                (2, [0, 1]),
                (3, [1]),
                (4, [0, 1, 2, 3])
            ]
        ),
    )
    def test_flip(self, backend, rank_dim):

        rank, dim = rank_dim
        class FlipModel(nn.Module):
            def __init__(self):
                super(FlipModel, self).__init__()

            def forward(self, x):
                return torch.flip(x, dim)

        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = FlipModel()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

class TestWhere(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [(2, 6), (3, 4, 5)]
        ),
    )
    def test_where_test1(self, backend, shape):

        class WhereModel(nn.Module):
            def __init__(self):
                super(WhereModel, self).__init__()

            def forward(self, x, y):
                return torch.where(x > 0.5, x, y)

        input_shape = [shape, shape]
        model = WhereModel()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [(2, 6), (3, 4, 5)]
        ),
    )
    def test_where_test2(self, backend, shape):

        class WhereModel(nn.Module):
            def __init__(self):
                super(WhereModel, self).__init__()

            def forward(self, cond, x, y):
                return torch.where(cond, x, y)

        cond = torch.rand(*shape) > 0.5
        inputs = [cond, torch.rand(*shape), torch.rand(*shape)]
        model = WhereModel()
        expected_results = model(*inputs)
        self.run_compare_torch(
            inputs,
            model,
            backend=backend,
            expected_results=expected_results,
            input_as_shape=False,
        )

class TestSelect(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, dim_index",
        itertools.product(
            backends,
            [
                [0, 0],
                [1, 1],
                [-1, -1],
            ]
        ),
    )
    def test_select(self, backend, dim_index):
        dim, index = dim_index

        class SelectModel(nn.Module):
            def __init__(self):
                super(SelectModel, self).__init__()

            def forward(self, x):
                return x.select(dim, index)

        input_shape = (1,2,3)
        model = SelectModel()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

class TestNonZero(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [1, 3],
        ),
    )
    def test_non_zero(self, backend, rank):

        if rank == 1:
            input_shape = (10)
            zeros_indices = np.array([1, 4, 7, 9])
        elif rank == 3:
            input_shape = (2, 7, 3)
            zeros_indices = np.array([1, 12, 33, 40])

        input = np.arange(np.prod(input_shape)).astype(np.float32)
        input[zeros_indices] = 0
        input = np.reshape(input, input_shape)
        input = torch.tensor(input)

        model = ModuleWrapper(
            torch.nonzero,
        )

        self.run_compare_torch(input, model,
            input_as_shape=False, backend=backend)

class TestTensorAssign(TorchBaseTest):

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_tensor_assign_case_1(self, backend):
        # single dimension assignment for a 1D tensor
        class TensorAssignModel(torch.nn.Module):
            def __init__(self):
                super(TensorAssignModel, self).__init__()

            def forward(self, x):
                x[0] = 0
                x[1] = 1
                y = x + 1
                x[1] = 2 * y[1]
                return x, y
        shape = (5,)
        model = TensorAssignModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_tensor_assign_case_2(self, backend):
        # single dimension assignment for two 1D tensors
        class TensorAssignModel(torch.nn.Module):
            def __init__(self):
                super(TensorAssignModel, self).__init__()

            def forward(self, x, y):
                x[0] = 0
                y[1] = 2
                y = x + y
                x = 2 * y
                y[3] = x[1] + 5
                y[0] = x[0] * 10
                z = x + y
                return z, x, y
        shape = (5,)
        model = TensorAssignModel()
        self.run_compare_torch(
            [shape, shape], model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (5,4),
                (5,4,3),
            ]
        ),
    )
    def test_tensor_assign_case_3(self, backend, shape):
        # broadcast assignment for two n-D tensors
        class TensorAssignModel(torch.nn.Module):
            def __init__(self):
                super(TensorAssignModel, self).__init__()

            def forward(self, x, y):
                x[0] = 0
                x[3] = 1
                y[2] = 2
                return x

        model = TensorAssignModel()
        self.run_compare_torch(
            [shape, shape], model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_itensor_assign_case_4(self, backend):
        # single dimension assignment for two n-D tensors
        class TensorAssignModel(torch.nn.Module):
            def __init__(self):
                super(TensorAssignModel, self).__init__()

            def forward(self, x, y):
                x[0] = torch.tensor([1.,2.,3.,4.])
                x[3] = 1
                y[0] = x[0]
                return x, y
        shape = (5,4)
        model = TensorAssignModel()
        self.run_compare_torch(
            [shape, shape], model, backend=backend,
        )


    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_tensor_assign_case_5(self, backend):
        # slice dimension assigment
        class TensorAssignModel(torch.nn.Module):
            def __init__(self):
                super(TensorAssignModel, self).__init__()

            def forward(self, x):
                x[:,1] = torch.tensor([1., 2.])
                return x
        shape = (2,10)
        model = TensorAssignModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_tensor_assign_case_6(self, backend):
        # a more complicated slice dimension assigment
        class TensorAssignModel(torch.nn.Module):
            def __init__(self):
                super(TensorAssignModel, self).__init__()

            def forward(self, x):
                x[:,1,:] = torch.tensor([1., 2., 3., 4., 5., 6.]).view(2,3)
                return x
        shape = (2,10,3)
        model = TensorAssignModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

class TestIndexPut(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_index_put_case_1(self, backend):
        class IndexPutModel(torch.nn.Module):
            def __init__(self):
                super(IndexPutModel, self).__init__()

            def forward(self, x, y):
                y = x + 1
                mask = torch.tensor([True, False, False, False, True, True]).view(3,2)
                x[mask] = y[mask]
                return x
        shape = (3,2)
        model = IndexPutModel()
        self.run_compare_torch(
            [shape, shape], model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            backends,
            [0, 1],
        ),
    )
    def test_index_put_case_2(self, backend, rank):
        class IndexPutModel(torch.nn.Module):
            def __init__(self):
                super(IndexPutModel, self).__init__()

            def forward(self, x):
                mask = torch.tensor([True, False, False, False, True, True]).view(3,2)
                if rank == 0:
                    x[mask] = 0.
                if rank == 1:
                    x[mask] = torch.tensor([1.])
                return x
        shape = (3,2)
        model = IndexPutModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend",
        backends,
    )
    def test_index_put_case_3(self, backend):
        pytest.xfail("rdar://84892125 (Empty tensors handling for non_zero, tile and scatter_nd)")
        class IndexPutModel(torch.nn.Module):
            def __init__(self):
                super(IndexPutModel, self).__init__()

            def forward(self, x, y):
                mask = y > 1
                x[y > 1] = 0.
                return x

        inputs = [
            torch.Tensor([1., 2., 3., 4., 5., 6]),
            torch.Tensor([0., 0., 0., 0., 0., 0.]),
        ]
        model = IndexPutModel()
        self.run_compare_torch(
            inputs, model, backend=backend, input_as_shape=False,
        )

class TestIndex(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (10,),
                (3, 4, 5, 6),
            ]
        ),
    )
    def test_index_bool_index(self, backend, shape):
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                return x[x > 0.5]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2),
                (3, 4, 5, 6),
            ]
        ),
    )
    def test_index_int_index_case_1(self, backend, shape):
        # all elements are selected
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 2:
                    return x[:, :]
                elif len(shape) == 4:
                    return x[:]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2),
                (3, 4, 5, 6),
            ]
        ),
    )
    def test_index_int_index_case_2(self, backend, shape):
        # only one axis is sliced
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 2:
                    index = torch.tensor([0])
                    return x[index, :]
                elif len(shape) == 4:
                    index = torch.tensor([1, 2])
                    return x[:, :, index]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_3(self, backend, shape):
        # only two axes are sliced, and connected
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    index_1 = torch.tensor([0])
                    index_2 = torch.tensor([1])
                    return x[index_1, index_2, :]

                elif len(shape) == 4:
                    index_1 = torch.tensor([0, 1, 1])
                    index_2 = torch.tensor([2, 1, 0])
                    return x[:, index_1, index_2, :]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_4(self, backend, shape):
        # only two axes are sliced, and not connected
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    index_1 = torch.tensor([0])
                    index_2 = torch.tensor([1])
                    return x[index_1, :,index_2]

                elif len(shape) == 4:
                    index_1 = torch.tensor([0, 1, 1])
                    index_2 = torch.tensor([3, 3, 4])
                    return x[index_1, :, :, index_2]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_5(self, backend, shape):
        # all axes are sliced
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    index_1 = torch.tensor([0])
                    index_2 = torch.tensor([1])
                    index_3 = torch.tensor([2])
                    return x[index_1, index_2, index_3]

                elif len(shape) == 4:
                    index_1 = torch.tensor([0, 1, 1, 0, 0])
                    index_2 = torch.tensor([1, 2, 0, 0, 0])
                    index_3 = torch.tensor([0, 1, 2, 3, 3])
                    index_4 = torch.tensor([2, 1, 0, 4, 4])
                    return x[index_1, index_2, index_3, index_4]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2),
                (3, 4, 5, 6),
            ]
        ),
    )
    def test_index_int_index_case_6(self, backend, shape):
        # only one axis is sliced + nd mode
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 2:
                    index = torch.tensor([0,0,0,0,0,0])
                    index = index.view(2, 3)
                    return x[index, :]
                elif len(shape) == 4:
                    index = torch.tensor([0,1,2,3,0,1])
                    index = index.view(3, 2)
                    return x[:, index]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_7(self, backend, shape):
        # two axes are sliced, and connected + nd mode
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    index_1 = torch.tensor([0,0,0,0,0,0,0,0]).view(4,2)
                    index_2 = torch.tensor([1,0,0,0,1,1,1,1]).view(4,2)
                    return x[index_1, index_2, :]

                elif len(shape) == 4:
                    index_1 = torch.tensor([0,0,2,2,1,1,2,0]).view(2,4)
                    index_2 = torch.tensor([0,1,2,3,0,1,2,3]).view(2,4)
                    return x[:, index_1, index_2, :]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_8(self, backend, shape):
        # two axes are sliced, and not connected + nd mode
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    index_1 = torch.tensor([0,0,0,0,0,0,0,0]).view(2,4)
                    index_2 = torch.tensor([1,0,0,2,2,1,1,1]).view(2,4)
                    return x[index_1, :,index_2]

                elif len(shape) == 4:
                    index_1 = torch.tensor([0,1,1,1,1,1,0,0]).view(4,2)
                    index_2 = torch.tensor([0,1,2,3,4,0,1,2]).view(4,2)
                    return x[index_1, :, :, index_2]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_9(self, backend, shape):
        # one axis is sliced through bool mask
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    return x[:, [True, False], :]

                elif len(shape) == 4:
                    return x[[True, False], :, :, :]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, shape",
        itertools.product(
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_10(self, backend, shape):
        # multiple axes are sliced through bool masks
        class IndexModel(torch.nn.Module):
            def __init__(self):
                super(IndexModel, self).__init__()

            def forward(self, x):
                if len(shape) == 3:
                    return x[[True], [True, False], [False, True, False]]

                elif len(shape) == 4:
                    return x[[True, True], :, [True, True, False, False], [True, False, False, True, False]]

        model = IndexModel()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

class TestPad(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank, mode",
        itertools.product(backends, range(3, 5), ['reflect', 'replicate'])
    )
    def test_pad_reflect_replicate(self, backend, rank: int, mode: str):
        if rank == 3:
            pad_len = 2
            input_shape = (5, 10, 10)
        elif rank == 4:
            pad_len = 4
            input_shape = (10, 5, 5, 10)
        else:
            raise NotImplementedError("Only 3D, 4D padding with non-constant padding are supported for now")
        max_pad = min(input_shape[-1], input_shape[-2])
        pad = list(np.random.randint(low=0, high=max_pad,
                                     size=pad_len))
        model = ModuleWrapper(function=torch.nn.functional.pad,
                              kwargs={"pad": pad, "mode": mode})
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(backends, range(1, 6))
    )
    def test_pad_constant(self, backend, rank: int):
        if rank > 5:
            raise NotImplementedError("Only supports < 6D constant padding")
        val = float(np.random.random(1))
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        pad_dims = np.random.randint(low=1, high=rank+1)
        pad = list(np.random.randint(low=0, high=10,
                                     size=pad_dims*2))
        model = ModuleWrapper(function=torch.nn.functional.pad,
                              kwargs={"pad": pad, "mode": "constant", "value": val})
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize("backend", backends)
    def test_constant_pad_1d(self, backend):
        input_shape = (3, 4, 5)
        model = torch.nn.ConstantPad1d((5, 6), 3.5).eval()
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize("backend", backends)
    def test_constant_pad_2d(self, backend):
        input_shape = (3, 4, 5, 6)
        model = torch.nn.ConstantPad2d((5, 6, 3, 8), 3.5).eval()
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize("backend", backends)
    def test_constant_pad_3d(self, backend):
        input_shape = (3, 4, 5, 6, 2)
        model = torch.nn.ConstantPad3d((5, 6, 3, 8, 2, 4), 3.5).eval()
        self.run_compare_torch(input_shape, model, backend=backend)

class TestMeshgrid(TorchBaseTest):
    @pytest.mark.parametrize(
        "rows, cols, dtype, inp_mode, backend",
        itertools.product(
            [1, 2, 3], [1, 2, 3], [torch.int, torch.float], ["norm", "list"], backends
        ),
    )
    def test_meshgrid(
        self,
        rows,
        cols,
        dtype,
        inp_mode,
        backend,
    ):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()

            def forward(self, rows, cols):
                if inp_mode == "norm":
                    return torch.meshgrid(rows, cols)
                elif inp_mode == "list":
                    return torch.meshgrid([rows, cols])
                else:
                    raise ValueError("Unsupported mode: {mode}".format(mode=inp_mode))

        inputs = (
            torch.arange(start=0, end=rows, step=1, dtype=dtype),
            torch.arange(start=0, end=cols, step=1, dtype=dtype)
        )
        model = TestModel().eval()
        expected_results = model(*inputs)
        self.run_compare_torch(
            inputs, model, expected_results, input_as_shape=False, backend=backend,
        )

class TestSacatterAdd(TorchBaseTest):
    @pytest.mark.parametrize(
        "shapes_dims, backend",
        itertools.product(
            [
                [(10,), (0, -1)],
                [(2, 3), (1, -1)],
                [(2, 3, 4, 5), (0, -2)],
            ],
            backends
        ),
    )
    def test_scatter_add(self, shapes_dims, backend):
        shapes, dims = shapes_dims
        for dim in dims:

            class TestModel(nn.Module):
                def __init__(self):
                    super(TestModel, self).__init__()
                    self.source = torch.rand(*(shapes))
                    self.index = torch.randint(0, shapes[dim], size=shapes)

                def forward(self, x):
                    index = torch.tensor(self.index)
                    return x.scatter_add_(dim, self.index, self.source)

            self.run_compare_torch(shapes, TestModel().eval(), backend=backend)

class TestBroadcastTensors(TorchBaseTest):
    @pytest.mark.parametrize(
        "shapes, backend",
        itertools.product(
            [(1,), (1, 2)],
            backends
        ),
    )
    def test_one_tensor(self, shapes, backend):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()

            def forward(self, a):
                return torch.broadcast_tensors(a)
        self.run_compare_torch(shapes, TestModel().eval(), backend=backend)

    @pytest.mark.parametrize(
        "shapes, backend",
        itertools.product(
            [
                [(2,1), (1,3)],
                [(5,1,4,1), (3,1,1)],
                [(1,), (3,1,7)],
                [(2,1), (4,3,2,1,)]
            ],
            backends
        ),
    )
    def test_two_tensors(self, shapes, backend):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()

            def forward(self, a, b):
                return torch.broadcast_tensors(a, b)
        self.run_compare_torch(shapes, TestModel().eval(), backend=backend)

    @pytest.mark.parametrize(
        "shapes, backend",
        itertools.product(
            [
                [(2,1), (1,3), (1,), (1,1)],
                [(5,1,4,1), (3,1,1), (1,), (4,8)],
                [(1,), (2,1), (3,2,1), (5,4,3,2,1)],
            ],
            backends
        ),
    )
    def test_four_tensors(self, shapes, backend):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()

            def forward(self, a, b, c, d):
                return torch.broadcast_tensors(a, b, c, d)
        self.run_compare_torch(shapes, TestModel().eval(), backend=backend)
