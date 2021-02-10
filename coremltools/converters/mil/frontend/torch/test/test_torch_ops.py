#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import sys
import itertools
import numpy as np
from coremltools.models.utils import _python_version
from coremltools.models.utils import _macos_version
from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.testing_reqs import *
from .testing_utils import *
from coremltools import TensorType, ImageType, RangeDim

backends = testing_reqs.backends

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 8), reason="Segfault with Python 3.8+"
)  # rdar://problem/65730375


torch.manual_seed(30)
np.random.seed(30)


class TestArgSort(TorchBaseTest):
    @pytest.mark.parametrize(
        "rank, axis, descending, backend",
        itertools.product(
            [rank for rank in range(1, 6)],
            [-1, 0],
            [True, False],
            backends
        )
    )
    def test_argsort(self, rank, axis, descending, backend):
        shape = tuple(np.random.randint(low=1, high=4, size=rank))
        model = ModuleWrapper(
            function=torch.argsort, kwargs={"dim": axis, "descending": descending}
        )
        TorchBaseTest.run_compare_torch(shape, model, backend=backend)


class TestBatchNorm(TorchBaseTest):
    @pytest.mark.parametrize(
        "num_features, eps, backend",
        itertools.product([5, 3, 2, 1], [0.1, 1e-05, 1e-9], backends),
    )
    def test_batchnorm(self, num_features, eps, backend):
        model = nn.BatchNorm2d(num_features, eps)
        self.run_compare_torch((6, num_features, 5, 5), model, backend=backend)

    @pytest.mark.parametrize("backend", backends)
    def test_batchnorm_1d(self, backend):
        class CRNNBase(nn.Module):
            def __init__(self, ch_in, ch_out, kernel_size=3, use_bn=True):
                super(CRNNBase, self).__init__()
                self.conv = nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size)
                self.norm = nn.BatchNorm1d(ch_out)

            def forward(self, x):
                x = self.conv(x)
                x = self.norm(x)
                return x
        model = CRNNBase(ch_in=6, ch_out=16)
        self.run_compare_torch((1, 6, 15), model, backend=backend)

    @pytest.mark.parametrize(
        "shape, eps, backend",
        itertools.product([(4, 8), (1, 5), (5, 10), (6, 1)], [0.1, 1e-05, 1e-9], backends),
    )
    def test_batchnorm1d_rank2(self, shape, eps, backend):
        N,C = shape
        batchnorm = nn.BatchNorm1d(C, eps=eps).eval()
        self.run_compare_torch(
            (N, C), batchnorm, backend=backend,
        )

    @pytest.mark.parametrize(
        "shape, eps, backend",
        itertools.product([(4, 8, 2), (1, 5, 3), (5, 10, 1), (6, 1, 4)], [0.1, 1e-05, 1e-9], backends),
    )
    def test_batchnorm1d_rank3(self, shape, eps, backend):
        N,C,L = shape
        batchnorm = nn.BatchNorm1d(C, eps=eps).eval()
        self.run_compare_torch(
            (N, C, L), batchnorm, backend=backend,
        )

class TestInstanceNorm(TorchBaseTest):
    @pytest.mark.parametrize(
        "num_features, eps, backend",
        itertools.product([5, 3, 2, 1], [0.1, 1e-05, 1e-09], backends),
    )
    def test_instancenorm(self, num_features, eps, backend):
        if backend == "nn_proto" and eps == 1e-09:
            return
        model = nn.InstanceNorm2d(num_features, eps)
        self.run_compare_torch((6, num_features, 5, 5), model, backend=backend)

    @pytest.mark.parametrize("num_features, backend",
                             itertools.product([5, 3, 2, 1], backends),
    )
    def test_instancenorm_1d(self, num_features, backend):
        model = nn.InstanceNorm1d(num_features)
        self.run_compare_torch((6, num_features, 10), model, backend=backend)

class TestGroupNorm(TorchBaseTest):
    @pytest.mark.parametrize(
        "group_features, eps,affine, backend",
        itertools.product([(16,32), (32,64), (1,1)], [0.1, 1e-05, 1e-09],[True, False], backends),
    )
    def test_groupnorm(self, group_features, eps, affine, backend):
        if backend == "nn_proto" and eps == 1e-09:
            return
        model = nn.GroupNorm(group_features[0],group_features[1], eps=eps, affine=affine)
        self.run_compare_torch((6, group_features[1], 5, 5), model, backend=backend)

class TestLinear(TorchBaseTest):
    @pytest.mark.parametrize(
        "in_features, out_features, backend",
        itertools.product([10, 25, 100], [3, 6], backends),
    )
    def test_addmm(self, in_features, out_features, backend):
        model = nn.Linear(in_features, out_features)
        self.run_compare_torch((1, in_features), model, backend=backend)

    @pytest.mark.parametrize(
        "in_features, out_features, backend",
        itertools.product([5], [10], backends),
    )
    def test_linear_rank1_input(self, in_features, out_features, backend):
        model = nn.Linear(in_features, out_features)
        self.run_compare_torch((in_features,), model, backend=backend)


class TestConv(TorchBaseTest):
    @pytest.mark.parametrize(
        "height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, backend",
        itertools.product(
            [5, 6], [5, 7], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], backends
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


class TestConvTranspose(TorchBaseTest):
    @pytest.mark.parametrize(
        "width, in_channels, out_channels, kernel_size, stride, padding, dilation, backend",
        itertools.product(
            [5, 7], [1, 3], [1, 3], [1, 3], [2, 3], [0, 1], [1, 3], backends
        ),
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
        itertools.product(
            [5, 6], [5, 7], [1, 3], [1, 3], [1, 3], [2, 3], [0, 1], [1, 3], backends
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

    # TODO: rdar://65588783 ([PyTorch] Define and error out on unsupported configuration for output_padding)
    # TODO: rdar://65550420 (Add Image Resizing (crop, upsample, resize_bilinear) layers to the MIL backend)
    @pytest.mark.parametrize(
        "height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, output_padding, backend",
        list(
            itertools.product(
                [10],
                [10],
                [1, 3],
                [1, 3],
                [1, 3],
                [1, 2, 3],
                [1, 3],
                [1, 2],
                [1, 2, (1, 2)],
                ["nn_proto"],
            )
        )
        + [
            pytest.param(
                5, 5, 1, 1, 3, 4, 1, 1, 2, "nn_proto", marks=pytest.mark.xfail
            ),
            pytest.param(
                5, 5, 1, 1, 3, 2, 1, 3, 2, "nn_proto", marks=pytest.mark.xfail
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
        itertools.product(
            [3, 4], [5, 6], [5, 7], [1, 3], [1, 3], [1, 3], [2, 3], [0, 1], [1, 3], backends
        ),
    )
    @pytest.mark.skip(reason="rdar://65198011 (Re-enable Conv3dTranspose and DynamicTile unit tests)")
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
    @pytest.mark.parametrize("backend", backends)
    def test_cond(self, backend):
        in_features = 1
        out_features = 2
        class TestNet(nn.Module):
            def forward(self, x):
                if torch.squeeze(x) < 10.:
                    return x*10.
                else:
                    return x*2.

        model = TestNet().eval()
        torch_model = torch.jit.script(model)

        self.run_compare_torch(torch.tensor([1.]), torch_model,
            input_as_shape=False, backend=backend)
        self.run_compare_torch(torch.tensor([11.]), torch_model,
            input_as_shape=False, backend=backend)

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
        [
            x
            for x in itertools.product(
                [(10, 10), (1, 1), (20, 20), (2, 3), (190, 170)],
                [True, False],
                backends,
            )
        ],
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
        "scales_h, scales_w, align_corners, backend",
        [
            x
            for x in itertools.product(
                [2, 3, 4.5], [4, 5, 5.5], [True, False], backends
            )
        ],
    )
    def test_upsample_bilinear2d_with_scales(
        self, scales_h, scales_w, align_corners, backend
    ):
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate,
            {
                "scale_factor": (scales_h, scales_w),
                "mode": "bilinear",
                "align_corners": align_corners,
            },
        )
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "output_size, backend",
        [
            x
            for x in itertools.product(
                [(10, 10), (30, 20), (20, 20), (20, 30), (190, 170)], backends
            )
        ],
    )
    def test_upsample_nearest2d_with_output_size(self, output_size, backend):
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate, {"size": output_size, "mode": "nearest"},
        )
        self.run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize(
        "scales_h, scales_w, backend",
        [x for x in itertools.product([2, 3, 5, 4.5], [4, 5, 2, 5.5], backends)],
    )
    def test_upsample_nearest2d_with_scales(self, scales_h, scales_w, backend):
        if backend == "nn_proto":
            if isinstance(scales_h, float) or isinstance(scales_w, float):
                return  # Skip fractional scale factors tests for nn_proto

        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate,
            {"scale_factor": (scales_h, scales_w), "mode": "nearest"},
        )
        self.run_compare_torch(input_shape, model, backend=backend)


class TestBranch(TorchBaseTest):
    @pytest.mark.parametrize("backend", backends)
    def test_if(self, backend):
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
    # rdar://66066001 (PyTorch converter: enable ceil_mode=True tests for pooling ops)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, padding, ceil_mode, include_pad, backend",
        itertools.product(
            [(1, 3, 15), (1, 1, 7), (1, 3, 10)],
            [1, 2, 3],
            [1, 2],
            [0, 1],
            [False],
            [True, False],
            backends,
        ),
    )
    def test_avg_pool1d(
        self, input_shape, kernel_size, stride, padding, ceil_mode, include_pad, backend
    ):
        if padding > kernel_size / 2:
            return
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
        itertools.product(
            [(1, 3, 15, 15), (1, 1, 7, 7), (1, 3, 10, 10)],
            [1, 2, 3],
            [1, 2],
            [0, 1],
            [False],
            [True, False],
            backends,
        ),
    )
    def test_avg_pool2d(
        self, input_shape, kernel_size, stride, padding, ceil_mode, include_pad, backend
    ):
        if padding > kernel_size / 2:
            return
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
        itertools.product(
            [(1, 3, 11, 3, 11), (1, 1, 7, 4, 7), (1, 3, 6, 6, 3)],
            [1, 2, 3],
            [1, 2],
            [0, 1],
            [False],
            [True, False],
            backends,
        ),
    )
    def test_avg_pool3d(
        self, input_shape, kernel_size, stride, padding, ceil_mode, include_pad, backend
    ):
        if padding > kernel_size / 2:
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
            [(1,1), (3,2),(3,6),(32,32)],
            [1,2,4,5,6,7],
            [0,11],
            [1,2,3],
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
    # rdar://66066001 (PyTorch converter: enable ceil_mode=True tests for pooling ops)

    @pytest.mark.parametrize(
        "input_shape, kernel_size, stride, padding, ceil_mode, backend",
        itertools.product(
            [(1, 3, 15), (1, 1, 7), (1, 3, 10)],
            [1, 2, 3],
            [1, 2],
            [0, 1],
            [False],
            backends,
        ),
    )
    def test_max_pool1d(
        self, input_shape, kernel_size, stride, padding, ceil_mode, backend
    ):
        if padding > kernel_size / 2:
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
            [(1, 3, 15, 15), (1, 1, 7, 7), (1, 3, 10, 10)],
            [1, 2, 3],
            [1, 2],
            [0, 1],
            [False],
            backends,
        ),
    )
    def test_max_pool2d(
        self, input_shape, kernel_size, stride, padding, ceil_mode, backend
    ):
        if padding > kernel_size / 2:
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
            [(1, 3, 11, 3, 11), (1, 1, 7, 4, 7), (1, 3, 6, 6, 3)],
            [1, 2, 3],
            [1, 2],
            [0, 1],
            [False],
            backends,
        ),
    )
    def test_max_pool3d(
        self, input_shape, kernel_size, stride, padding, ceil_mode, backend
    ):
        if padding > kernel_size / 2:
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


class TestLSTM(TorchBaseTest):
    @pytest.mark.parametrize(
        "input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, backend",
        itertools.product(
            [7], [5], [1, 2, 3], [True, False], [False, True], [0.3], [False, True], backends
        ),
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

        num_directions = int(bidirectional) + 1

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


class TestReduction(TorchBaseTest):
    @pytest.mark.parametrize(
        "input_shape, dim, keepdim, backend",
        itertools.product([(2, 2), (1, 1)], [0, 1], [True, False], backends),
    )
    def test_max(self, input_shape, dim, keepdim, backend):
        class TestMax(nn.Module):
            def __init__(self):
                super(TestMax, self).__init__()

            def forward(self, x):
                return torch.max(x, dim=dim, keepdim=keepdim)

        input_data = torch.rand(input_shape)
        model = TestMax()
        # TODO: Expected results are flipped due to naming issue:
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
        itertools.product([(1, 3, 15, 15), (1, 1, 1, 1)], [1e-5, 1e-9], backends),
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
            [[(2, 1), (2, 2)], [(3, 1), (-1, 4)], [(1, 3, 4, 4), (3, 3, 4, 4)]]
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
            [[(2, 1), (2, 2)], [(3, 1), (3, 4)], [(1, 3, 4, 4), (3, 3, 4, 4)]]
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
    # TODO: <rdar://66239973> Add dynamic & rank preserving reshape tests for pytorch
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
    @pytest.mark.xfail(
        reason="Load constant not copied properly for integer valued constants. Enable after eng/PR-65551506 is merged",
        run=False,
    )
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
        "backend, rank", itertools.product(backends, range(1, 6)),
    )
    def test_relu(self, backend, rank):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = nn.ReLU().eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

        model = ModuleWrapper(nn.functional.relu_)
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, rank", itertools.product(backends, range(1, 6)),
    )
    def test_relu6(self, backend, rank):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = nn.ReLU6().eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, alpha", itertools.product(backends, [0.1, 0.25, 2.0]),
    )
    def test_prelu(self, backend, alpha):
        input_shape = tuple(np.random.randint(low=5, high=10, size=4))
        C = input_shape[1]
        model = nn.PReLU(C, alpha).eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, rank, alpha",
        itertools.product(backends, range(1, 6), [0.1, 2.0, 1.5]),
    )
    def test_leaky_relu(self, backend, rank, alpha):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = nn.LeakyReLU(negative_slope=alpha).eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

        model = ModuleWrapper(nn.functional.leaky_relu_, {'negative_slope': alpha})
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, rank", itertools.product(backends, range(1, 6)),
    )
    def test_softmax(self, backend, rank):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = nn.Softmax().eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, rank, range_val",
        itertools.product(
            backends, range(1, 6), [(-1.0, 1.0), (0.0, 0.1), (1.0, 3.0), (-1.0, 6.0)]
        ),
    )
    def test_hardtanh(self, backend, rank, range_val):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
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
        "backend, rank, alpha",
        itertools.product(backends, range(1, 6), [0.1, 2.0, 1.5]),
    )
    def test_elu(self, backend, rank, alpha):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = nn.ELU(alpha).eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    # rdar://problem/66557565
    @pytest.mark.parametrize(
        "backend, rank", itertools.product(['nn_proto'], range(1, 6)),
    )
    def test_gelu(self, backend, rank):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = nn.GELU().eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.skipif(_python_version() < (3, 6), reason="requires python 3.6")
    @pytest.mark.parametrize(
        "backend, rank", itertools.product(backends, range(1, 6)),
    )
    def test_erf(self, backend, rank):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))

        class ERFActivation(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.erf(x)

        model = ERFActivation().eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, rank", itertools.product(backends, range(1, 6)),
    )
    def test_sigmoid(self, backend, rank):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = nn.Sigmoid().eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.skipif(_python_version() < (3, 6), reason="requires python 3.6")
    @pytest.mark.parametrize(
        "backend, rank", itertools.product(backends, range(1, 6)),
    )
    def test_sigmoid_hard(self, backend, rank):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = nn.Hardsigmoid().eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, beta, threshold", itertools.product(backends, [1, 2, 5], [5, 10, 20]),
    )
    @pytest.mark.skipif(
        _macos_version() <= (11,),
        reason="Parametric SoftPlus segfaults on macOS 10.15 and below. (rdar://problem/66555235)",
    )
    def test_softplus(self, backend, beta, threshold):
        input_shape = (1, 10, 5, 15)
        model = nn.Softplus(beta, threshold).eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    # rdar://problem/66557565
    @pytest.mark.parametrize(
        "backend, rank", itertools.product(['nn_proto'], range(1, 6)),
    )
    def test_softsign(self, backend, rank):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = nn.Softsign().eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )


class TestElementWiseUnary(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank, op_string",
        itertools.product(
            backends,
            [4],
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
    def test_elementwise_no_params(self, backend, rank, op_string):
        if not contains_op(torch, op_string):
            return
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        op_func = getattr(torch, op_string)
        model = ModuleWrapper(function=op_func)
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    ## TODO (rdar://66577921): Needs to move to test_elementwise_no_params after backend is added
    @pytest.mark.parametrize(
        "backend, rank",
        itertools.product(
            ['nn_proto'],
            [4],
        ),
    )
    @pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher")
    def test_square(self, backend, rank):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = ModuleWrapper(function=torch.square)
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, rank, clamp_range",
        itertools.product(
            backends,
            [4],
            [(0.0, 1.0), (-1.0, 0.5), (0.2, 0.7), (None, 4.0), (-3.0, None)],
        ),
    )
    def test_clamp(self, backend, rank, clamp_range):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        params_dict = {}
        if clamp_range[0] is not None:
            params_dict["min"] = clamp_range[0]
        if clamp_range[1] is not None:
            params_dict["max"] = clamp_range[1]

        model = ModuleWrapper(torch.clamp, params_dict)
        self.run_compare_torch(
            input_shape, model, backend=backend, rand_range=(-5, 5)
        )

    @pytest.mark.parametrize(
        "backend, rank, threshold",
        itertools.product(
            ['nn_proto'], # rdar://66597974 Renable for all backends due to missing cast
            [4],
            [(0.0, 0.0), (0.5, 0.5), (0.5, 10), (0.9, 0.0)]
        ),
    )
    def test_threshold(self, backend, rank, threshold):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = torch.nn.Threshold(threshold[0], threshold[1]).eval()
        self.run_compare_torch(
            input_shape, model, backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, rank, op_string",
        itertools.product(
            backends,
            [4],
            [
                "log",
                "rsqrt",
                "reciprocal",
            ],
        ),
    )
    def test_elementwise_numerically_stable(self, backend, rank, op_string):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        op_func = getattr(torch, op_string)
        model = ModuleWrapper(function=op_func)
        self.run_compare_torch(
            input_shape, model, backend=backend, rand_range=(20, 100)
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
        "backend, rank, dims",
        itertools.product(backends, list(range(2, 6)),
                          [(0, 1), (-2, -1), (1, 0), (-1, -2)]),
    )
    def test(self, backend, rank, dims):
        input_shape = tuple(np.random.randint(low=1, high=4, size=rank))
        model = ModuleWrapper(function=torch.transpose,
                              kwargs={"dim0": dims[0], "dim1": dims[1]})
        self.run_compare_torch(input_shape, model, backend=backend)

class TestConstantPad(TorchBaseTest):
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

class TestTo(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend", backends,
    )
    def test_cast_bug(self, backend):
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
        self.run_compare_torch([(1, 21, 2), (1, 6, 384)], model, backend=backend)# [spans.shape, embedding.shape]

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
                tokens_embeddings = self.tokens_embedding(tokens)
                context_embeddings = self.context_embedding(context)
                embeddings = torch.cat((context_embeddings, tokens_embeddings), dim=0)
                embeddings = self.dynamic_slicer(embeddings, context_length)

                return embeddings

        model = Model()
        batch_size = 5
        inputs = [ TensorType(name="tokens", shape=(10, batch_size), dtype=np.int64),
                   TensorType(name="context", shape=(3, batch_size), dtype=np.int64),
                   TensorType(name="context_length", shape=(), dtype=np.int32),
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

class TestTopk(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, largest, shape_dim_k",
        itertools.product(
            backends,
            [True, False],
            [
             ((4, 6, 7, 3), -1, 2),
             ((10, 3, 4), 2, 2),
             ((10, 5), -2, 3),
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

class TestPad(TorchBaseTest):
    @pytest.mark.parametrize(
        "backend, rank, mode",
        itertools.product(backends, range(3, 5), ['reflect', 'replicate'])
    )
    def test_pad_reflect_replicate(self, backend, rank: int, mode: str):
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        if rank == 3:
            pad_len = 2
        elif rank == 4:
            pad_len = 4
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
