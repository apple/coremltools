#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import platform
from typing import Tuple

import numpy as np
import pytest
import torch.nn as nn

import coremltools as ct
from coremltools import RangeDim, Shape, TensorType
from coremltools._deps import version_lt
from coremltools.converters.mil import testing_reqs
from coremltools.models.utils import _macos_version, _python_version

from .testing_utils import (
    ModuleWrapper,
    TorchBaseTest,
    contains_op,
    generate_input_data,
)

backends = testing_reqs.backends
compute_units = testing_reqs.compute_units

torch = pytest.importorskip("torch")
torch.manual_seed(30)
np.random.seed(30)

# Set of common shapes for testing. Not all layers support 1D, so these two
# set of shapes are kept separate
COMMON_SHAPES = [(1, 10), (1, 5, 6), (1, 3, 5, 6), (1, 3, 4, 5, 6)]
COMMON_SHAPES_ALL = [(1,)] + COMMON_SHAPES


class TestScriptedModels(TorchBaseTest):

    @staticmethod
    def get_while_loop_model():
        class TestLayer(nn.Module):
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

        return TestNet().eval()

    @staticmethod
    def get_cond_model():
        class TestNet(nn.Module):
            def forward(self, x):
                if torch.squeeze(x) < 10.0:
                    return x * 10.0
                else:
                    return x * 2.0

        return TestNet().eval()

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_while_loop(self, compute_unit, backend):
        model = TestScriptedModels.get_while_loop_model()
        self.run_compare_torch(
            model.input_size,
            model,
            backend=backend,
            compute_unit=compute_unit,
            use_scripting=True
        )

    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_cond(self, compute_unit, backend):
        torch_model = TestScriptedModels.get_cond_model()

        self.run_compare_torch(
            torch.tensor([1.]),
            torch_model,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit,
            use_scripting=True
        )

        self.run_compare_torch(
            torch.tensor([11.]),
            torch_model,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit,
            use_scripting=True
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_for_loop(self, compute_unit, backend):
        class TestLayer(nn.Module):
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

        self.run_compare_torch(
            model.input_size,
            model,
            backend=backend,
            compute_unit=compute_unit,
            use_scripting=True
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_if(self, compute_unit, backend):
        class TestLayer(nn.Module):
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

        self.run_compare_torch(
            model.input_size,
            model,
            backend=backend,
            compute_unit=compute_unit,
            use_scripting=True
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_linear(self, compute_unit, backend):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        model = Model().eval()

        self.run_compare_torch(
            torch.tensor([[1.0, 2.0]]),
            model,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit,
            use_scripting=True,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_conv(self, compute_unit, backend):
        pytest.xfail(
            "rdar://88194776 ([Converter] coremltools is not working with scripted torch convolution model)"
        )
        model = torch.nn.Conv2d(
            in_channels=2,
            out_channels=3,
            kernel_size=1,
            padding="same",
            stride=1,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.run_compare_torch(
            (1, 2, 4, 5),
            model,
            backend=backend,
            compute_unit=compute_unit,
            use_scripting=True,
        )


class TestMean(TorchBaseTest):
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_with_flexible_shape(self, compute_unit, backend):
        if backend[0] == 'mlprogram' and _macos_version() < (13, 0):
            pytest.xfail(
                "Issue fixed in iOS16/macOS13: https://github.com/apple/coremltools/issues/1420"
            )

        class Model(nn.Module):
            def forward(self, x):
                return torch.mean(x, dim=(2, 3), keepdim=True)

        model = Model()
        shape = (1, 3, 256, 256)
        converter_input_type = [
            TensorType(shape=Shape(shape=[1, 3, RangeDim(), RangeDim()], default=shape))
        ]

        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
            converter_input_type=converter_input_type
        )

    @staticmethod
    @pytest.mark.skipif(ct.utils._macos_version() < (13, 0), reason='Bug fixed in macOS13/iOS16')
    def test_flexible_shape_with_default_value():
        # test for bug reported in https://github.com/apple/coremltools/issues/1420
        class Network(torch.nn.Module):
            def forward(self, x):
                return torch.mean(x, dim=(2, 3), keepdim=True)

        model = Network()
        x = torch.rand(1, 3, 256, 256)
        traced_model = torch.jit.trace(model, x)
        input_x = ct.TensorType(
            shape=(1, 3, ct.RangeDim(default=256), ct.RangeDim(default=256)),
            name="input",
        )
        cml = ct.convert(traced_model,
                         inputs=[input_x],
                         outputs=[ct.TensorType(name="out")],
                         convert_to="mlprogram",
                         compute_units=ct.ComputeUnit.CPU_ONLY)

        input_dict = {"input": np.random.rand(1, 3, 112, 112)}

        if ct.utils._is_macos():
            out = cml.predict(input_dict)["out"]
            assert out.shape == (1, 3, 1, 1)


class TestAffineGrid(TorchBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "x_shape_and_target_size",
                "sampling_mode",
                "padding_mode",
                "align_corners"
            ]
        ),
        itertools.product(
            compute_units,
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
        compute_unit,
        backend,
        x_shape_and_target_size,
        sampling_mode,
        padding_mode,
        align_corners,
    ):
        if backend[0] == "neuralnetwork":
            pytest.skip("nn backend not supported")

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
        self.run_compare_torch(
            x_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestGridSample(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, data_grid_shapes, mode, padding_mode, align_corners",
        itertools.product(
            compute_units,
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
        compute_unit,
        backend,
        data_grid_shapes,
        mode,
        padding_mode,
        align_corners,
    ):
        if backend[0] == "neuralnetwork":
            pytest.skip("nn backend not supported")

        params = {
            "mode": mode,
            "padding_mode": padding_mode,
            "align_corners": align_corners,
        }
        model = ModuleWrapper(
            function=torch.nn.functional.grid_sample, kwargs=params
        )
        self.run_compare_torch(
            data_grid_shapes,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestFrac(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
        ),
    )
    def test_frac(self, compute_unit, backend, shape):
        model = ModuleWrapper(function=torch.frac)
        TorchBaseTest.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
            rand_range=(-10.0, 10.0)
        )


class TestNLLLoss(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, reduction",
        itertools.product(
            compute_units,
            backends,
            ["none", "sum", "mean"],
        ),
    )
    def test_nllloss(
        self,
        compute_unit,
        backend,
        reduction,
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
            inputs,
            model,
            expected_results,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestArgSort(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape, axis, descending",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
            [-1, 0],
            [True, False],
        )
    )
    def test_argsort(self, compute_unit, backend, shape, axis, descending):
        model = ModuleWrapper(
            function=torch.argsort, kwargs={"dim": axis, "descending": descending}
        )
        TorchBaseTest.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestSort(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape, axis, descending",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
            [-1, 0],
            [True, False],
        )
    )
    def test_sort(self, compute_unit, backend, shape, axis, descending):
        model = ModuleWrapper(
            function=torch.sort, kwargs={"dim": axis, "descending": descending}
        )
        TorchBaseTest.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestSelu(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, inplace",
        itertools.product(
            compute_units,
            backends,
            [True, False],
        )
    )
    def test_selu(self, compute_unit, backend, inplace):
        x = torch.tensor([-6., -4., -2., 0., 2., 4., 6.])
        model = torch.nn.SELU(inplace=inplace)
        TorchBaseTest.run_compare_torch(
            x,
            model,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestMv(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, matrix_shape",
        itertools.product(
            compute_units,
            backends,
            [(2, 3), (10, 12), (10, 1), (1, 5)]
        )
    )
    def test_mv(self, compute_unit, backend, matrix_shape):
        model = ModuleWrapper(function=torch.mv)

        matrix = generate_input_data(matrix_shape)
        vector_length = matrix_shape[-1]
        vector = generate_input_data((vector_length,))

        TorchBaseTest.run_compare_torch(
            (matrix, vector),
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False
        )


@pytest.mark.skip(
    reason="rdar://100332029 ([PyTorch] cos_similarity unittest is failing stochastically)"
)
class TestCosineSimilarity(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, dim, eps, shape",
        itertools.product(
            compute_units,
            backends,
            [0, -1],
            [0.1, 1e-5, 1e-8],
            COMMON_SHAPES,
        ),
    )
    def test_cosine_similarity(self, compute_unit, backend, dim, eps, shape):
        class CosineSimilarity(nn.Module):
            def __init__(self, dim, eps):
                super(CosineSimilarity, self).__init__()
                self.cossim = torch.nn.CosineSimilarity(dim=dim, eps=eps)

            def forward(self, x, y):
                out = self.cossim(x, y)
                return out

        model = CosineSimilarity(dim, eps)
        input1 = generate_input_data(shape)
        input2 = generate_input_data(shape)

        TorchBaseTest.run_compare_torch(
            [input1, input2],
            model,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )


class TestDot(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, vector_length",
        itertools.product(
            compute_units,
            backends,
            [1, 5, 11]
        )
    )
    def test_dot(self, compute_unit, backend, vector_length):
        model = ModuleWrapper(function=torch.dot)

        vector1 = generate_input_data((vector_length,))
        vector2 = generate_input_data((vector_length,))

        TorchBaseTest.run_compare_torch(
            (vector1, vector2),
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False
        )


class TestOuter(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, x_vector_length, y_vector_length",
        itertools.product(
            compute_units,
            backends,
            [1, 5],
            [1, 3],
        )
    )
    def test_outer(self, compute_unit, backend, x_vector_length, y_vector_length):
        model = ModuleWrapper(function=torch.outer)

        vector1 = generate_input_data((x_vector_length,))
        vector2 = generate_input_data((y_vector_length,))

        TorchBaseTest.run_compare_torch(
            (vector1, vector2),
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False
        )


class TestCross(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape_dim",
        itertools.product(
            compute_units,
            backends,
            [((3,), 0), ((4, 3, 2), 1)]
        )
    )
    def test_cross(self, compute_unit, backend, shape_dim):
        shape = shape_dim[0]
        dim = shape_dim[1]
        class CrossModel(nn.Module):
            def forward(self, x, y):
                return torch.cross(x, y, dim)

        x = generate_input_data(shape)
        y = generate_input_data(shape)
        model = CrossModel().eval()
        torch_out = model(x, y)
        self.run_compare_torch(
            (x, y),
            model,
            expected_results=torch_out,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )


class TestNormalize(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
        )
    )
    def test_normalize(self, compute_unit, backend, shape):
        model = ModuleWrapper(function=nn.functional.normalize)
        TorchBaseTest.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestNorms(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape, keepdim",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
            [True, False]
        )
    )
    def test_frobenius_norm(self, compute_unit, backend, shape, keepdim):
        num_dims = len(shape)
        for dim in range(-num_dims, num_dims):
            model = ModuleWrapper(
                function=torch.norm, kwargs={'keepdim': keepdim, 'dim': dim}
            )
            TorchBaseTest.run_compare_torch(
                shape,
                model,
                backend=backend,
                compute_unit=compute_unit,
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape, p, keepdim",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
            [-1, 0, 1, 2, 3, np.inf, -np.inf],
            [True, False]
        )
    )
    def test_number_norm(self, compute_unit, backend, shape, p, keepdim):
        for dim in (-1, 0, 1):
            model = ModuleWrapper(
                function=torch.norm, kwargs={'p': p, 'keepdim': keepdim, 'dim': dim}
            )
            TorchBaseTest.run_compare_torch(
                shape,
                model,
                backend=backend,
                compute_unit=compute_unit,
                atol=1e-2,
            )
            
class TestWeightNorm(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, in_out_features",
        itertools.product(
            compute_units,
            backends,
            [(1, 1), (2, 10), (20, 10)],
        )
    )
    def test_linear(self, compute_unit, backend, in_out_features):
        in_features, out_features = in_out_features

        for dim in (None, -2, -1, 0, 1):
            model = nn.utils.weight_norm(nn.Linear(in_features, out_features), dim=dim)
            TorchBaseTest.run_compare_torch(
                (in_features,),
                model,
                backend=backend,
                compute_unit=compute_unit,
                atol=1e-3
            )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        )
    )
    def test_conv2d(self, compute_unit, backend):
        x = torch.randn(20, 16, 50, 100)

        for dim in (None, ) + tuple(range(-4, 4)):
            model = nn.utils.weight_norm(nn.Conv2d(16, 33, 3), dim=dim)
            TorchBaseTest.run_compare_torch(
                x,
                model,
                input_as_shape=False,
                atol=1e-3,
                backend=backend,
                compute_unit=compute_unit
            )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        )
    )
    def test_conv3d(self, compute_unit, backend):
        x = torch.randn(20, 16, 5, 50, 100)

        for dim in (None, ) + tuple(range(-5, 5)):
            model = nn.utils.weight_norm(nn.Conv3d(16, 33, 3), dim=dim)
            TorchBaseTest.run_compare_torch(
                x,
                model,
                input_as_shape=False,
                atol=1e-3,
                backend=backend,
                compute_unit=compute_unit
            )


class TestLinAlgNorms(TorchBaseTest):
    def _is_valid_config(self, shape, order, dim):
        if isinstance(dim, tuple):
            if isinstance(order, int) and (order == 0 or order > 2):
                return False
        elif isinstance(dim, int):
            if order == "fro":
                return False
        elif dim is None:
            if order is not None:
                if len(shape) > 2:
                    return False
                elif len(shape) == 2 and not isinstance(order, str) and (order == 0 or order > 2):
                    return False
                elif len(shape) == 1 and isinstance(order, str):
                    return False
        return True

    @pytest.mark.parametrize(
        "compute_unit, backend, shape, order, keepdim, dim",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
            [-2, -1, 0, 1, 2, 3, np.inf, -np.inf, "fro", None],
            [True, False],
            [-1, 0, 1, (0, 1), (0, -1), None]
        )
    )
    def test_norm(self, compute_unit, backend, shape, order, keepdim, dim):
        if not self._is_valid_config(shape, order, dim):
            pytest.skip()
        if (
            isinstance(order, int)
            and abs(order) == 2
            and ((dim is None and len(shape) == 2) or isinstance(dim, tuple))
        ):
            pytest.xfail("Matrix norm for order 2 and -2 is not implemented")
        model = ModuleWrapper(
            function=torch.linalg.norm,
            kwargs={"ord": order, "keepdim": keepdim, "dim": dim},
        )
        TorchBaseTest.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
            atol=1e-2,
        )


class TestLinAlgMatrixNorms(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape, order, keepdim, dim",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
            [-2, -1, 1, 2, np.inf, -np.inf, "fro", "nuc"],
            [True, False],
            [(0, 1), (0, -1), (1, 2), (0, 2), (2, 3)]
        )
    )
    def test_norm(self, compute_unit, backend, shape, order, keepdim, dim):
        if dim[-1] > len(shape) - 1:
            pytest.skip()
        if order == "nuc" or (type(order) != str and abs(order) == 2):
            pytest.xfail("Matrix norm for order 2, -2 and nuc is not implemented")
        model = ModuleWrapper(
            function=torch.linalg.matrix_norm,
            kwargs={"ord": order, "keepdim": keepdim, "dim": dim},
        )
        TorchBaseTest.run_compare_torch(
            shape, model, backend=backend, compute_unit=compute_unit, atol=1e-2
        )


class TestLinAlgVectorNorms(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape, order, keepdim, dim",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
            [-2, -1, 0, 1, 2, np.inf, -np.inf],
            [True, False],
            [-1, 0, 1, (0, 1), (0, -1), None]
        )
    )
    def test_norm(self, compute_unit, backend, shape, order, keepdim, dim):
        model = ModuleWrapper(
            function=torch.linalg.vector_norm,
            kwargs={"ord": order, "keepdim": keepdim, "dim": dim},
        )
        TorchBaseTest.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
            atol=1e-2,
        )


class TestHardswish(TorchBaseTest):
    class HardswishModel(nn.Module):
        def __init__(self, inplace=False):
            super(TestHardswish.HardswishModel, self).__init__()
            self.activation = nn.Hardswish(inplace=inplace)

        def forward(self, x):
            return self.activation(x)

    def test_longer_range_input_element_values(self):
        x = torch.tensor([-6., -4., -2., 0., 2., 4., 6.])

        model = TestHardswish.HardswishModel()
        TorchBaseTest.run_compare_torch(x, model, input_as_shape=False)

        model = TestHardswish.HardswishModel(inplace=True)
        TorchBaseTest.run_compare_torch(x, model, input_as_shape=False)

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
        )
    )
    def test_additional_shapes_and_backends(self, compute_unit, backend, shape):
        model = TestHardswish.HardswishModel()
        TorchBaseTest.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestBatchNorm(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, num_features, eps, affine",
        itertools.product(
            compute_units,
            backends,
            [5, 3, 1],
            [0.1, 1e-05],
            [True, False]
        ),
    )
    def test_batchnorm(self, compute_unit, backend, num_features, eps, affine):
        model = nn.BatchNorm2d(num_features, eps, affine=affine)
        self.run_compare_torch(
            (6, num_features, 5, 5),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, affine",
        itertools.product(
            compute_units,
            backends,
            [True, False]
        ),
    )
    def test_batchnorm_2d_with_conv(self, compute_unit, backend, affine):
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
        self.run_compare_torch(
            (1, 6, 15, 30),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, num_features, eps, affine, dynamic_input",
        itertools.product(
            [ct.ComputeUnit.CPU_ONLY],
            backends,
            [5, 1],
            [0.1, 1e-05],
            [True, False],
            ["None", "Batch", "Height", "Width", "Depth", "All"],
        ),
    )
    def test_batchnorm_3d(self, compute_unit, backend, num_features, eps, affine, dynamic_input):
        model = nn.BatchNorm3d(num_features, eps, affine=affine)
        input_shape = (6, num_features, 2, 3, 4)
        if dynamic_input == "None":
            self.run_compare_torch(
                input_shape,
                model,
                backend=backend,
                compute_unit=compute_unit,
            )
        else:
            if dynamic_input == "Batch":
                converter_input_type = [
                    TensorType(shape=(6, num_features, 2, 3, 4), dtype=np.float32)
                ]
                converter_input_type = [
                    TensorType(
                        shape=(RangeDim(1, 10), num_features, 2, 3, 4), dtype=np.float32
                    )
                ]
            elif dynamic_input == "Height":
                converter_input_type = [
                    TensorType(
                        shape=(6, num_features, RangeDim(1, 10), 3, 4), dtype=np.float32
                    )
                ]
            elif dynamic_input == "Width":
                converter_input_type = [
                    TensorType(
                        shape=(6, num_features, 2, RangeDim(1, 10), 4), dtype=np.float32
                    )
                ]
            elif dynamic_input == "Depth":
                converter_input_type = [
                    TensorType(
                        shape=(6, num_features, 2, 3, RangeDim(1, 10)), dtype=np.float32
                    )
                ]
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
                compute_unit=compute_unit,
                converter_input_type=converter_input_type
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank, num_features, eps, training",
        itertools.product(
            [ct.ComputeUnit.CPU_ONLY],
            backends,
            [3, 4, 5],
            [5, 1],
            [0.1, 1e-05],
            [True, False]
        ),
    )
    def test_batchnorm_dynamic(self, compute_unit, backend, rank, num_features, eps, training):
        model = ModuleWrapper(
            nn.functional.batch_norm,
            {
                "training": training,
                "eps": eps,
            },
        )
        input_shape = [6, num_features, 3, 4, 5]
        input_shape = input_shape[:rank]
        _input = torch.randn(*input_shape)
        _mean = torch.randn(num_features)
        _var = torch.randn(num_features)

        inputs = (_input, _mean, _var)
        expected_results = model(*inputs)

        self.run_compare_torch(
            inputs,
            model,
            expected_results,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit,
        )
        
    @pytest.mark.parametrize(
        "compute_unit, backend, has_weight, has_bias, has_running_mean, has_running_var",
        itertools.product(
            compute_units,
            backends,
            [True, False],
            [True, False],
            [True, False],
            [True, False],
        ),
    )
    def test_batchnorm_dynamic_stress(
        self,
        compute_unit,
        backend,
        has_weight,
        has_bias,
        has_running_mean,
        has_running_var
    ):
        num_features = 5
        input_shape = (3, num_features, 2)
        
        weight = torch.randn(num_features) if has_weight else None
        bias = torch.randn(num_features) if has_bias else None
        running_mean = torch.randn(num_features) if has_running_mean else None
        running_var = torch.randn(num_features) if has_running_var else None
        
        class Model(torch.nn.Module):
            def forward(self, x):
                res = torch.nn.functional.batch_norm(
                        input=x,
                        running_mean=running_mean,
                        running_var=running_var,
                        weight=weight,
                        bias=bias,
                        training=True,
                        momentum=0.0,
                        eps=1e-05,
                )
                return res

        self.run_compare_torch(
            input_shape,
            Model(),
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, affine",
        itertools.product(
            compute_units,
            backends,
            [True, False]
        ),
    )
    def test_batchnorm_1d_with_conv(self, compute_unit, backend, affine):
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
        self.run_compare_torch(
            (1, 6, 15),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape, eps, affine",
        itertools.product(
            compute_units,
            backends,
            [(1, 10), (4, 6), (10, 1)],
            [0.1, 1e-05],
            [True, False]
        ),
    )
    def test_batchnorm1d_rank2(self, compute_unit, backend, shape, eps, affine):
        N, C = shape
        batchnorm = nn.BatchNorm1d(C, eps=eps, affine=affine).eval()
        self.run_compare_torch(
            (N, C),
            batchnorm,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape, eps, affine",
        itertools.product(
            compute_units,
            backends,
            [(4, 8, 2), (1, 5, 3), (5, 10, 1), (6, 1, 4)],
            [0.1, 1e-05],
            [True, False]
        ),
    )
    def test_batchnorm1d_rank3(self, compute_unit, backend, shape, eps, affine):
        N, C, L = shape
        batchnorm = nn.BatchNorm1d(C, eps=eps, affine=affine).eval()
        self.run_compare_torch(
            (N, C, L),
            batchnorm,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestInstanceNorm(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, num_features, eps",
        itertools.product(
            compute_units,
            backends,
            [5, 2, 1],
            [0.1, 1e-05]
        ),
    )
    def test_instancenorm(self, compute_unit, backend, num_features, eps):
        model = nn.InstanceNorm2d(num_features, eps)
        self.run_compare_torch(
            (6, num_features, 5, 5),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, num_features",
        itertools.product(
            compute_units,
            backends,
            [5, 2, 1]
        ),
    )
    def test_instancenorm_1d(self, compute_unit, backend, num_features):
        model = nn.InstanceNorm1d(num_features)
        self.run_compare_torch(
            (6, num_features, 10),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestGroupNorm(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, group_features, eps, affine",
        itertools.product(
            compute_units,
            backends,
            [(16, 32), (1, 1)],
            [0.1, 1e-05],
            [True, False]
        ),
    )
    def test_groupnorm(self, compute_unit, backend, group_features, eps, affine):
        model = nn.GroupNorm(
            group_features[0], group_features[1], eps=eps, affine=affine
        )
        self.run_compare_torch(
            (6, group_features[1], 5, 5),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, group_features, eps, affine",
        itertools.product(
            compute_units,
            backends,
            [(16, 32), (1, 1)],
            [0.1, 1e-05],
            [True, False]
        ),
    )
    def test_groupnorm_rank3_input(self, compute_unit, backend, group_features, eps, affine):
        model = nn.GroupNorm(
            group_features[0], group_features[1], eps=eps, affine=affine
        )
        self.run_compare_torch(
            (6, group_features[1], 5),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )
        
    @pytest.mark.parametrize(
        "compute_unit, backend, group_features, eps, affine",
        itertools.product(
            compute_units,
            backends,
            [(16, 32), (1, 1)],
            [0.1, 1e-05],
            [True, False]
        ),
    )
    def test_groupnorm_rank2_input(self, compute_unit, backend, group_features, eps, affine):
        model = nn.GroupNorm(
            group_features[0], group_features[1], eps=eps, affine=affine
        )
        self.run_compare_torch(
            (4, group_features[1]),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )
        

class TestLinear(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, in_features, out_features, bias",
        itertools.product(
            compute_units,
            backends,
            [5],
            [10],
            [True, False],
        ),
    )
    def test_linear_rank1_input(self, compute_unit, backend, in_features, out_features, bias):
        model = nn.Linear(in_features, out_features, bias=bias)
        self.run_compare_torch(
            (in_features,),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, in_features, out_features, bias",
        itertools.product(
            compute_units,
            backends,
            [10, 25],
            [3, 6],
            [True, False]
        ),
    )
    def test_linear_rank2_input(self, compute_unit, backend, in_features, out_features, bias):
        model = nn.Linear(in_features, out_features, bias=bias)
        self.run_compare_torch(
            (1, in_features),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, in_features, out_features, bias",
        itertools.product(
            compute_units,
            backends,
            [10],
            [6],
            [True, False]
        ),
    )
    def test_linear_rank3_input(self, compute_unit, backend, in_features, out_features, bias):
        model = nn.Linear(in_features, out_features, bias=bias)
        self.run_compare_torch(
            (1, 3, in_features),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, in_features, out_features, bias",
        itertools.product(
            compute_units,
            backends,
            [10],
            [6],
            [True, False]
        ),
    )
    def test_linear_rank4_input(self, compute_unit, backend, in_features, out_features, bias):
        model = nn.Linear(in_features, out_features, bias=bias)
        self.run_compare_torch((1, 5, 3, in_features), model, backend=backend)


class TestConv(TorchBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "height",
                "width",
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                [ct.ComputeUnit.CPU_ONLY],
                backends,
                [
                    (5, 3, 1, 1, 1, 2, 0, 1),
                    (3, 3, 1, 1, 1, 2, 1, 3),
                    (4, 3, 3, 3, 2, 2, 0, 1),
                    (7, 3, 3, 3, 1, 3, 0, 1),
                    (5, 5, 3, 3, 1, 3, 0, 1),
                    (3, 5, 3, 3, 1, 3, 0, 1),
                    (3, 5, 3, 3, 1, 3, 1, 3),
                    (7, 5, 3, 3, 2, 3, 1, 3),
                ],
            )
        ],
    )
    def test_convolution2d(
        self,
        compute_unit,
        backend,
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
        self.run_compare_torch(
            (1, in_channels, height, width),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend,"
                "height",
                "width",
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "dilation",
                "groups",
                "bias",
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                compute_units,
                backends,
                [
                    (7, 3, 3, 1, 2, 1, 1, 2, True),
                    (5, 3, 1, 3, 2, 3, 3, 1, False),
                    (7, 3, 1, 3, 2, 3, 1, 1, False),
                    (5, 3, 3, 3, 2, 1, 3, 2, False),
                    (5, 3, 1, 1, 1, 3, 3, 2, True),
                    (5, 5, 1, 3, 2, 2, 3, 1, True),
                    (5, 3, 1, 3, 1, 3, 3, 1, False),
                    (3, 3, 3, 1, 2, 2, 3, 1, False),
                    (5, 5, 3, 3, 1, 2, 3, 1, True),
                    (7, 3, 1, 3, 1, 2, 1, 2, True),
                    (7, 5, 3, 1, 2, 3, 1, 1, False),
                    (5, 5, 1, 3, 2, 1, 1, 2, True),
                ],
            )
        ],
    )
    def test_convolution_same_padding(
            self,
            compute_unit,
            backend,
            height,
            width,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
    ):
        model = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            stride=1,
            dilation=1,
            groups=1,
            bias=bias,
        )
        self.run_compare_torch(
            (1, in_channels, height, width),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestDynamicConv(TorchBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "width",
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                compute_units,
                backends,
                [
                    (5, 1, 1, 1, 2, 1),
                    (3, 1, 1, 1, 2, 3),
                    (4, 3, 3, 1, 2, 1),
                    (7, 3, 3, 1, 3, 1),
                    (5, 3, 3, 2, 2, 1),
                    (3, 3, 3, 1, 3, 1),
                    (3, 3, 3, 1, 3, 3),
                    (7, 3, 3, 3, 1, 3),
                ],
            )
        ],
    )
    def test_convolution1d(
        self,
        compute_unit,
        backend,
        width,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
    ):
        class DynamicConv(nn.Module):
            def forward(self, input_data, weights):
                return nn.functional.conv1d(
                    input_data,
                    weights,
                    stride=stride,
                    padding=padding
                )

        model = DynamicConv()
        input_shape = [
            (1, in_channels, width),
            (out_channels, int(in_channels / groups), kernel_size),
        ]
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "height",
                "width",
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                compute_units,
                backends,
                [
                    (5, 3, 1, 1, 1, 2, 0, 1),
                    (3, 3, 1, 1, 1, 2, 1, 3),
                    (4, 3, 3, 3, 1, 2, 0, 1),
                    (7, 3, 3, 3, 1, 3, 0, 1),
                    (5, 5, 3, 3, 2, 1, 0, 1),
                    (3, 5, 3, 3, 1, 3, 0, 1),
                    (3, 5, 3, 3, 1, 3, 1, 3),
                    (7, 5, 3, 3, 2, 3, 1, 3),
                ],
            )
        ],
    )
    def test_convolution2d(
        self,
        compute_unit,
        backend,
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
        class DynamicConv(nn.Module):
            def forward(self, input_data, weights):
                return nn.functional.conv2d(
                    input_data,
                    weights,
                    stride=stride,
                    padding=padding
                )

        model = DynamicConv()

        input_shape = [
            (1, in_channels, height, width),
            (out_channels, int(in_channels / groups), kernel_size, kernel_size),
        ]
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestConvTranspose(TorchBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "width",
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                compute_units,
                backends,
                [
                    (3, 1, 1, 1, 2, 0, 1),
                    (3, 1, 1, 1, 2, 1, 3),
                    (3, 3, 3, 1, 2, 0, 1),
                    (3, 3, 3, 1, 3, 0, 1),
                    (5, 3, 3, 1, 3, 0, 1),
                    (5, 3, 3, 1, 3, 0, 1),
                    (5, 3, 3, 1, 3, 1, 3),
                    (5, 3, 3, 1, 3, 1, 3),
                ],
            )
        ],
    )
    def test_convolution_transpose1d(
            self,
            compute_unit,
            backend,
            width,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
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
        self.run_compare_torch(
            (1, in_channels, width),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "height",
                "width",
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                compute_units,
                backends,
                [
                    (5, 5, 1, 1, 1, 2, 0, 1),
                    (5, 5, 1, 1, 1, 2, 1, 3),
                    (5, 5, 3, 3, 1, 2, 0, 1),
                    (5, 5, 3, 3, 1, 3, 0, 1),
                    (6, 5, 3, 3, 1, 3, 0, 1),
                    (6, 5, 3, 3, 1, 3, 0, 1),
                    (6, 5, 3, 3, 1, 3, 1, 3),
                    (6, 5, 3, 3, 1, 3, 1, 3),
                ],
            )
        ],
    )
    def test_convolution_transpose2d(
        self,
        compute_unit,
        backend,
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
        self.run_compare_torch(
            (1, in_channels, height, width),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, dynamic_input",
        itertools.product(
            compute_units,
            backends,
            [True, False],
        ),
    )
    def test_convolution_transpose2d_dynamic_input(
        self,
        compute_unit,
        backend,
        dynamic_input,
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
            converter_input_type = [
                TensorType(
                    shape=(1, in_channels, RangeDim(256, -1), RangeDim(256, -1)),
                    dtype=np.float32,
                )
            ]
            self.run_compare_torch(
                input_shape,
                model,
                backend=backend,
                compute_unit=compute_unit,
                converter_input_type=converter_input_type
            )
        else:
            self.run_compare_torch(
                input_shape,
                model,
                backend=backend,
                compute_unit=compute_unit,
            )

    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "height",
                "width",
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "output_padding",
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                compute_units,
                backends,
                [
                    (5, 5, 1, 1, 1, 2, 1, 1, 1),
                    (5, 5, 1, 1, 1, 2, 2, 3, 2),
                    (5, 5, 3, 3, 1, 2, 0, 1, 0),
                    (5, 5, 3, 3, 1, 3, 1, 1, 1),
                    (6, 5, 3, 3, 1, 3, 2, 1, 2),
                    (6, 5, 3, 3, 1, 3, 1, 1, 1),
                    (6, 5, 3, 3, 1, 3, 2, 3, 2),
                    (6, 5, 3, 3, 1, 3, 3, 3, 3),
                ],
            )
        ],
    )
    def test_convolution_transpose2d_output_padding(
        self,
        compute_unit,
        backend,
        height,
        width,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        output_padding,
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
        self.run_compare_torch((1, in_channels, height, width), model, backend=backend)

    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "depth",
                "height",
                "width",
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                compute_units,
                backends,
                [
                    (3, 5, 5, 1, 1, 1, 2, 0, 1),
                    (3, 5, 5, 1, 1, 1, 2, 1, 3),
                    (3, 5, 5, 3, 3, 1, 2, 0, 1),
                    (3, 5, 5, 3, 3, 1, 1, 0, 2),
                    (4, 6, 5, 3, 3, 1, 3, 0, 1),
                    (4, 6, 5, 3, 3, 1, 3, 1, 2),
                    (4, 6, 5, 3, 3, 1, 3, 1, 3),
                ],
            )
        ]
        + [
            pytest.param(
                ct.ComputeUnit.CPU_ONLY,
                "neualnetwork",
                5,
                5,
                1,
                1,
                3,
                4,
                1,
                1,
                2,
                marks=pytest.mark.xfail,
            ),
            pytest.param(
                ct.ComputeUnit.CPU_ONLY,
                "neualnetwork",
                5,
                5,
                1,
                1,
                3,
                2,
                1,
                3,
                2,
                marks=pytest.mark.xfail,
            ),
        ],
    )
    def test_convolution_transpose3d(
        self,
        compute_unit,
        backend,
        depth,
        height,
        width,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
    ):
        model = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.run_compare_torch(
            (1, in_channels, depth, height, width),
            model,
            backend=backend,
            compute_unit=compute_unit
        )


def _is_float_value(x, threshold=0.001):
    return x - np.floor(x) > threshold


class TestUpsample(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, output_size, align_corners",
        itertools.product(
            compute_units,
            backends,
            [1, 3, 10, 190],
            [True, False],
        )
    )
    def test_upsample_linear1d_with_output_size(
        self, compute_unit, backend, output_size, align_corners
    ):
        input_shape = (1, 3, 10)
        output_size = 3
        model = ModuleWrapper(
            nn.functional.interpolate,
            {
                "size": output_size,
                "mode": "linear",
                "align_corners": align_corners,
            },
        )
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, scale, align_corners, recompute_scale_factor",
        itertools.product(
            compute_units,
            backends,
            [2, 0.5, 5.3],
            [True, False],
            [True, False]
        ),
    )
    def test_upsample_linear1d_with_scales(
        self, compute_unit, backend, scale, align_corners, recompute_scale_factor
    ):
        Height = 8
        input_shape = (1, 3, Height)
        output_h = Height * scale
        is_h_float = _is_float_value(output_h)

        if is_h_float and not align_corners and not recompute_scale_factor:
            pytest.xfail("rdar://81124053 (Support recompute_scale_factor)")

        model = ModuleWrapper(
            nn.functional.interpolate,
            {
                "scale_factor": scale,
                "mode": "linear",
                "align_corners": align_corners,
                "recompute_scale_factor": recompute_scale_factor,
            },
        )
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, scales, align_corners, recompute_scale_factor",
        itertools.product(
            compute_units,
            backends,
            [2, 0.7, 3.6],
            [True, False],
            [True, False]
        ),
    )
    def test_upsample_linear1d_with_scales_dynamic(
        self, compute_unit, backend, scales, align_corners, recompute_scale_factor
    ):

        is_float = _is_float_value(scales)
        input_shape = (1, 3, 22)

        if is_float and not align_corners and not recompute_scale_factor:
            pytest.xfail("rdar://81124053 (Support recompute_scale_factor)")

        model = ModuleWrapper(
            nn.functional.interpolate,
            {
                "scale_factor": scales,
                "mode": "linear",
                "align_corners": align_corners,
                "recompute_scale_factor": recompute_scale_factor,
            },
        )
        converter_input_type = [TensorType(shape=(1, 3, RangeDim(default=22)), dtype=np.float32)]
        mlmodel = self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
            converter_input_type=converter_input_type
        )[1]

        # also check if the scale factor are integers
        if backend[0] == 'neuralnetwork' and not is_float:
            for layer in mlmodel._spec.neuralNetwork.layers:
                if layer.WhichOneof('layer') == "upsample":
                    assert len(layer.upsample.fractionalScalingFactor) == 0

    @pytest.mark.parametrize(
        "compute_unit, backend, output_size",
        itertools.product(
            compute_units,
            backends,
            [10, 170],
        )
    )
    def test_upsample_nearest1d_with_output_size(self, compute_unit, backend, output_size):
        input_shape = (1, 3, 10)
        model = ModuleWrapper(
            nn.functional.interpolate, {"size": output_size, "mode": "nearest"},
        )
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, scales",
        itertools.product(
            compute_units,
            backends,
            [2, 3, 4.5]
        ),
    )
    def test_upsample_nearest1d_with_scales(self, compute_unit, backend, scales):
        if backend[0] == "neuralnetwork":
            if isinstance(scales, float):
                return  # Skip fractional scale factors tests for neuralnetwork

        input_shape = (1, 3, 10)
        model = ModuleWrapper(
            nn.functional.interpolate,
            {"scale_factor": scales, "mode": "nearest"},
        )
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, scales",
        itertools.product(
            compute_units,
            backends,
            [2, 3]
        ),
    )
    def test_upsample_nearest1d_with_scales_dynamic(self, compute_unit, backend, scales):
        input_shape = (1, 3, 10)
        model = ModuleWrapper(
            nn.functional.interpolate,
            {
                "scale_factor": scales,
                "mode": "nearest",
                "recompute_scale_factor": True,
            },
        )
        converter_input_type = [TensorType(shape=(1, 3, RangeDim()), dtype=np.float32)]
        mlmodel = self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
            converter_input_type=converter_input_type
        )[1]

        # also check if the scale factor are integers
        if backend[0] == 'neuralnetwork':
            for layer in mlmodel._spec.neuralNetwork.layers:
                if layer.WhichOneof('layer') == "upsample":
                    assert len(layer.upsample.fractionalScalingFactor) == 0

    @pytest.mark.parametrize(
        "compute_unit, backend, output_size, align_corners",
        itertools.product(
            compute_units,
            backends,
            [(10, 10),
             # PyTorch has a bug for the following parameter:
             # (1, 1),
             # See: https://github.com/pytorch/pytorch/issues/71188
             (2, 3), (190, 170)],
            [True, False],
        )
    )
    def test_upsample_bilinear2d_with_output_size(
        self, compute_unit, backend, output_size, align_corners
    ):
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate,
            {
                "size": output_size,
                "mode": "bilinear",
                "align_corners": align_corners,
            },
        )
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, scales_h, scales_w, align_corners, recompute_scale_factor",
        itertools.product(
            compute_units,
            backends,
            [2, 0.5, 4.1],
            [3, 0.5, 5.3],
            [True, False],
            [True, False],
        )
    )
    def test_upsample_bilinear2d_with_scales(
        self,
        compute_unit,
        backend,
        scales_h,
        scales_w,
        align_corners,
        recompute_scale_factor,
    ):

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
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, output_size",
        itertools.product(
            compute_units,
            backends,
            [(10, 10), (190, 170)],
        )
    )
    def test_upsample_nearest2d_with_output_size(self, compute_unit, backend, output_size):
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate, {"size": output_size, "mode": "nearest"},
        )
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, scales_h, scales_w",
        itertools.product(
            compute_units,
            backends,
            [2, 3, 4.5], [4, 5, 5.5]
        ),
    )
    def test_upsample_nearest2d_with_scales(self, compute_unit, backend, scales_h, scales_w):
        if backend[0] == "neuralnetwork":
            if isinstance(scales_h, float) or isinstance(scales_w, float):
                return  # Skip fractional scale factors tests for neuralnetwork

        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate,
            {"scale_factor": (scales_h, scales_w), "mode": "nearest"},
        )
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, scales_h, scales_w",
        itertools.product(
            compute_units,
            backends,
            [2, 3], [4, 5]
        ),
    )
    def test_upsample_nearest2d_with_scales_dynamic(
        self, compute_unit, backend, scales_h, scales_w
    ):
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(
            nn.functional.interpolate,
            {
                "scale_factor": (scales_h, scales_w),
                "mode": "nearest",
                "recompute_scale_factor": True,
            },
        )
        converter_input_type = [TensorType(shape=(1, 3, RangeDim(), RangeDim()), dtype=np.float32)]
        mlmodel = self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
            converter_input_type=converter_input_type
        )[1]

        # also check if the scale factor are integers
        if backend[0] == 'neuralnetwork':
            for layer in mlmodel._spec.neuralNetwork.layers:
                if layer.WhichOneof('layer') == "upsample":
                    assert len(layer.upsample.fractionalScalingFactor) == 0

    @pytest.mark.parametrize(
        "compute_unit, backend, scales_h, scales_w, align_corners, recompute_scale_factor",
        itertools.product(
            compute_units,
            backends,
            [2, 3.6],
            [4, 0.7],
            [True, False],
            [True, False],
        )
    )
    def test_upsample_bilinear2d_with_scales_dynamic(
        self,
        compute_unit,
        backend,
        scales_h,
        scales_w,
        align_corners,
        recompute_scale_factor,
    ):

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
        converter_input_type = [
            TensorType(
                shape=(1, 3, RangeDim(default=9), RangeDim(default=22)),
                dtype=np.float32,
            )
        ]
        mlmodel = self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
            converter_input_type=converter_input_type
        )[1]

        # also check if the scale factor are integers
        if backend[0] == 'neuralnetwork' and not is_h_float and not is_w_float:
            for layer in mlmodel._spec.neuralNetwork.layers:
                if layer.WhichOneof('layer') == "upsample":
                    assert len(layer.upsample.fractionalScalingFactor) == 0


class TestEmpty(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
        )
    )
    def test_empty_like(self, compute_unit, backend, shape):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.empty_like(x)
                # Value of y is Nondeterministic, so return length
                return torch.Tensor([len(y)])

        self.run_compare_torch(
            shape,
            TestModel(),
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
        )
    )
    def test_new_empty(self, compute_unit, backend, shape):
        class TestModel(nn.Module):
            def forward(self, _):
                tensor = torch.ones(())
                y = tensor.new_empty(shape)
                # Value of y is Nondeterministic, so return length
                return torch.Tensor([len(y)])

        self.run_compare_torch(
            shape,
            TestModel(),
            backend=backend,
            compute_unit=compute_unit,
        )


class TestAvgPool(TorchBaseTest):

    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "input_shape",
                "kernel_size",
                "stride",
                "padding",
                "ceil_mode",
                "include_pad",
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                compute_units,
                backends,
                [
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
                ],
            )
        ],
    )
    def test_avg_pool1d(
        self,
        compute_unit,
        backend,
        input_shape,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        include_pad,
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
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "input_shape",
                "kernel_size",
                "stride",
                "padding",
                "ceil_mode",
                "include_pad",
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                compute_units,
                backends,
                [
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
                ],
            )
        ],
    )
    def test_avg_pool2d(
        self,
        compute_unit,
        backend,
        input_shape,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        include_pad,
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
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "input_shape",
                "kernel_size",
                "stride",
                "padding",
                "ceil_mode",
                "include_pad",
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                compute_units,
                backends,
                [
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
                ],
            )
        ],
    )
    def test_avg_pool3d(
        self,
        compute_unit,
        backend,
        input_shape,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        include_pad,
    ):
        if padding > kernel_size / 2:
            return

        if include_pad and ceil_mode and stride > 1:
            # skip: MIL/CoreML does not support this configuration
            pytest.xfail(
                "rdar://73723194 (Support 3D Avg pooling with ceil_mode=True and include_pad = True, in MIL)"
            )
        model = nn.AvgPool3d(
            kernel_size,
            stride,
            padding,
            ceil_mode=ceil_mode,
            count_include_pad=include_pad,
        )
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestAdaptiveMaxPool(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, output_size, magnification, delta, depth",
        itertools.product(
            compute_units,
            backends,
            [(1, 1), (3, 2)],
            [1, 2, 7],
            [0, 11],
            [1, 2, 3],
        ),
    )
    def test_adaptive_max_pool2d(
            self, compute_unit, backend, output_size, magnification, delta, depth
    ):
        # input_size = output_size * magnification + delta
        input_size = (
            delta + magnification * output_size[0],
            delta + magnification * output_size[1],
        )
        # since coremltools reproduces PyTorch's kernel sizes and
        # offsets for adaptive pooling layers only when input_size is
        # a multiple of output_size, we expect failures otherwise
        if not (input_size[0] % output_size[0] == 0 and input_size[1] % output_size[1] == 0):
            pytest.xfail("Test should fail because input_size is not a multiple of output_size")
        n = 1
        in_shape = (n, depth) + input_size
        model = nn.AdaptiveMaxPool2d(
            output_size
        )
        self.run_compare_torch(
            in_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestAdaptiveAvgPool(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, output_size, magnification, delta, depth",
        itertools.product(
            compute_units,
            backends,
            [(1, 1), (3, 2)],
            [1, 2, 7],
            [0, 11],
            [1, 2, 3],
        ),
    )
    def test_adaptive_avg_pool2d(
            self, compute_unit, backend, output_size, magnification, delta, depth
    ):
        # input_size = output_size * magnification + delta
        input_size = (
            delta + magnification * output_size[0],
            delta + magnification * output_size[1],
        )
        # since coremltools reproduces PyTorch's kernel sizes and
        # offsets for adaptive pooling layers only when input_size is
        # a multiple of output_size, we expect failures otherwise
        if not (input_size[0] % output_size[0] == 0 and input_size[1] % output_size[1] == 0):
            pytest.xfail("Test should fail because input_size is not a multiple of output_size")
        n = 1
        in_shape = (n, depth) + input_size
        model = nn.AdaptiveAvgPool2d(
            output_size
        )
        self.run_compare_torch(
            in_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestMaxPool(TorchBaseTest):

    @pytest.mark.parametrize(
        "compute_unit, backend, input_shape, kernel_size, stride, padding, ceil_mode",
        itertools.product(
            compute_units,
            backends,
            [(1, 3, 15), (1, 1, 7)],
            [1, 3],
            [1, 2],
            [0, 1],
            [True, False],
        ),
    )
    def test_max_pool1d(
        self,
        compute_unit,
        backend,
        input_shape,
        kernel_size,
        stride,
        padding,
        ceil_mode,
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
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, input_shape, kernel_size, stride, padding, ceil_mode",
        itertools.product(
            compute_units,
            backends,
            [(1, 3, 15, 15), (1, 1, 7, 7)],
            [1, 3],
            [1, 2],
            [0, 1],
            [True, False],
        ),
    )
    def test_max_pool2d(
        self,
        compute_unit,
        backend,
        input_shape,
        kernel_size,
        stride,
        padding,
        ceil_mode,
    ):
        if padding > kernel_size / 2:
            return
        if ceil_mode > 0 and padding == 0 and kernel_size == 1 and stride == 2:
            for r in range(2, 4):
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
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, input_shape, kernel_size, stride, padding, ceil_mode",
        itertools.product(
            compute_units,
            backends,
            [(1, 3, 11, 3, 11), (1, 1, 7, 4, 7)],
            [1, 3],
            [1, 2],
            [0, 1],
            [True, False],
        ),
    )
    def test_max_pool3d(
        self,
        compute_unit,
        backend,
        input_shape,
        kernel_size,
        stride,
        padding,
        ceil_mode,
    ):
        if padding > kernel_size / 2:
            return
        if ceil_mode > 0 and padding == 0 and kernel_size == 1 and stride == 2:
            for r in range(2, 5):
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
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestMaximumMinimum(TorchBaseTest):

    @pytest.mark.parametrize(
        "compute_unit, backend, input_shape, mode",
        itertools.product(
            compute_units,
            backends,
            [(2, 5, 7, 3), (3, 2, 9)],
            ["minimum", "maximum"],
        ),
    )
    def test_minimum_maximum(self, compute_unit, backend, input_shape, mode):
        class TestModel(torch.nn.Module):
            def forward(self, x, y):
                if mode == "minimum":
                    return torch.minimum(x, y)
                elif mode == "maximum":
                    return torch.maximum(x, y)
                else:
                    raise ValueError("Unsupported mode: {mode}".format(mode=mode))

        model = TestModel()
        self.run_compare_torch(
            [input_shape] * 2,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


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
        self.run_compare_torch(
            input_shape,
            model,
            backend=backends[0],
            converter_input_type=converter_input_type
        )

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
        self.run_compare_torch(
            input_shape,
            model,
            backend=backends[0],
            converter_input_type=converter_input_type
        )


class TestLSTM(TorchBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "input_size",
                "hidden_size",
                "num_layers",
                "bias",
                "batch_first",
                "dropout",
                "bidirectional"
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                compute_units,
                backends,
                [
                    (1, 1, 1, True, True, 0.3, True),
                    (1, 1, 1, False, True, 0.3, False),
                    (1, 1, 1, False, True, 0.3, True),
                    (3, 1, 5, True, False, 0.3, False),
                    (3, 1, 5, True, True, 0.3, True),
                    (3, 7, 5, True, False, 0.3, False),
                    (3, 7, 5, False, True, 0.3, True),
                    (3, 7, 5, False, True, 0.3, False),
                ],
            )
        ],
    )
    def test_lstm(
        self,
        compute_unit,
        backend,
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
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
        ",".join(
            [
                "compute_unit",
                "backend",
                "input_size",
                "hidden_size",
                "num_layers",
                "bias",
                "batch_first",
                "dropout",
                "activation",
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                compute_units,
                backends,
                [
                    (1, 1, 1, True, True, 0.3, "tanh"),
                    (1, 1, 1, False, True, 0.3, "relu"),
                    (1, 1, 1, False, True, 0.3, "tanh"),
                    (3, 1, 5, True, False, 0.3, "relu"),
                    (3, 1, 5, True, True, 0.3, "tanh"),
                    (3, 7, 5, True, False, 0.3, "relu"),
                    (3, 7, 5, False, True, 0.3, "relu"),
                    (3, 7, 5, False, True, 0.3, "tanh"),
                ],
            )
        ],
    )
    def test_rnn(
        self,
        compute_unit,
        backend,
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        activation,
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
            bidirectional=False,  # bi-directional simple RNN not supported
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
            inputs,
            model,
            expected_results,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestGRU(TorchBaseTest):
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "input_size",
                "hidden_size",
                "num_layers",
                "bias",
                "batch_first",
                "sequence_length",
                "bidirectional",
            ]
        ),
        [
            (compute_unit, backend, *param)
            for compute_unit, backend, param in itertools.product(
                compute_units,
                backends,
                [
                    (1, 1, 1, True, True, 10, True),
                    (1, 1, 1, False, True, 10, True),
                    (1, 1, 1, False, True, 1, False),
                    (3, 1, 5, True, False, 10, False),
                    (3, 1, 5, True, True, 10, True),
                    (3, 7, 5, True, True, 10, False),
                    (3, 7, 5, False, True, 10, True),
                    (3, 7, 5, False, True, 1, True),
                ],
            )
        ],
    )
    def test_gru(
        self,
        compute_unit,
        backend,
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        sequence_length,
        bidirectional,
    ):
        DROPOUT = 0.3
        BATCH_SIZE = 3
        model = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=DROPOUT,
            bidirectional=bidirectional,
        )
        model.eval()
        num_directions = int(bidirectional) + 1

        if batch_first:
            _input = torch.randn(BATCH_SIZE, sequence_length, input_size)
        else:
            _input = torch.randn(sequence_length, BATCH_SIZE, input_size)

        h0 = torch.randn(num_layers * num_directions, BATCH_SIZE, hidden_size)

        inputs = (_input, h0)
        expected_results = model(*inputs)

        self.run_compare_torch(
            inputs,
            model,
            expected_results,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestLSTMWithPackedSequence(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, pack_batch_first, pad_batch_first, LSTM_batch_first, pad_value",
        itertools.product(
            compute_units,
            backends,
            [True, False],
            [True, False],
            [True, False],
            [-1, 0],
        ),
    )
    def test_lstm(
        self,
        compute_unit,
        backend,
        pack_batch_first,
        pad_batch_first,
        LSTM_batch_first,
        pad_value,
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
                packed_input = pack_padded_sequence(
                    batch_in, seq_lengths, batch_first=pack_batch_first
                )
                output_packed, (hidden, _) = self.lstm(packed_input)
                output, _ = pad_packed_sequence(
                    output_packed, padding_value=pad_value, batch_first=pad_batch_first
                )
                return output

        SEQUENCE_LENGTH = 10
        BATCH_SIZE = 3
        model = Encoder()
        model.eval()

        if pack_batch_first:
            _input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, input_size)
        else:
            _input = torch.randn(SEQUENCE_LENGTH, BATCH_SIZE, input_size)

        seq_lengths = torch.tensor([10, 5, 1], dtype=torch.int32)

        inputs = (_input, seq_lengths)
        expected_results = model(*inputs)
        self.run_compare_torch(
            inputs,
            model,
            expected_results,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )


# Workaround for GitHub Issue #824
# i.e. the return h_n/c_n for a converted BLSTM are mangled.
# Therefore, just look at output 'y' (for now) which is correct.
class StripCellAndHidden(nn.Module):
    def __init__(self, flagReturnTuple_):
        super(StripCellAndHidden, self).__init__()
        self.flagReturnTuple = flagReturnTuple_

    def forward(self, x):
        # Pass tuple, not tensor, to avoid issue in coremltools/converters/mil/frontend/torch/test/testing_utils.py on "if not expected_results:"
        # Pass tensor when we need input for LSTM #2 as part of nn.Sequential()
        return tuple(x[0]) if self.flagReturnTuple else x[0]


# Check GitHub Issue #810, assume num_layers == 2 and bidirectional == True
class TestStackedBLSTM(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional",
        itertools.product(
            compute_units,
            backends,
            [7],
            [5],
            [2],
            [True, False],
            [True, False],
            [0.3],
            [True]
        ),
    )
    def test_lstm(
        self,
        compute_unit,
        backend,
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
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
                input_size=2 * hidden_size,
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

        self.run_compare_torch(
            _input,
            model,
            expected_results,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )


class TestConcat(TorchBaseTest):
    # This tests an edge case where the list of tensors to concatenate only
    # has one item. NN throws an error for this case, hence why we have to
    # run through the full conversion process to test it.
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_cat(self, compute_unit, backend):
        class TestNet(nn.Module):
            def forward(self, x):
                x = torch.cat((x,), axis=1)
                return x

        model = TestNet()
        self.run_compare_torch(
            (1, 3, 16, 16),
            model,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestBitwiseNot(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, input_type",
        itertools.product(
            compute_units,
            backends,
            ["int", "bool"],
        ),
    )
    def test_bitwise_not(self, compute_unit, backend, input_type):
        class TestNet(nn.Module):
            def forward(self, x):
                return torch.bitwise_not(x)

        model = TestNet()
        if input_type == "int":
            torch_in = torch.tensor([1, 2, 3, -5, 0], dtype=torch.int32)
        elif input_type == "bool":
            torch_in = torch.tensor([True, False, True, False])
        self.run_compare_torch(
            torch_in,
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False
        )


class TestFull(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            [1, 3],
        ),
    )
    def test_full_dynamic(self, compute_unit, backend, rank):
        class FullDynamicModel(nn.Module):
            def forward(self, x):
                if rank == 1:
                    h = x[0]
                    x = torch.zeros(h)
                elif rank == 3:
                    h, w, d = x[0], x[1], x[2]
                    x = torch.zeros(h, w, d)
                return torch.full(x.shape, fill_value=3.14)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        torch_in = torch.tensor(input_shape, dtype=torch.int32)
        model = FullDynamicModel().eval()
        torch_out = model(torch_in)
        self.run_compare_torch(
            torch_in,
            model,
            expected_results=torch_out,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape_val",
        itertools.product(
            compute_units,
            backends,
            [
                [(1,), 0.0],
                [(2, 3), 3.1415],
                [(1, 1, 2, 5, 1), -2.0],
            ],
        ),
    )
    def test_full_static(self, compute_unit, backend, shape_val):
        shape, val = shape_val

        class FullStaticModel(nn.Module):
            def forward(self, x):
                return torch.full(x.shape, fill_value=val)

        self.run_compare_torch(
            shape,
            FullStaticModel().eval(),
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape_val",
        itertools.product(
            compute_units,
            [
                ["neuralnetwork", "fp32", ct.target.iOS14],
                ["mlprogram", "fp16", ct.target.iOS15],
                ["mlprogram", "fp32", ct.target.iOS15],
                ["mlprogram", "fp16", ct.target.iOS16],
                ["mlprogram", "fp32", ct.target.iOS16],
            ],
            [
                [(1,), 0.0],
                [(2, 3), 3.1415],
                [(1, 1, 2, 5, 1), -2.0],
            ],
        ),
    )
    def test_full_like(self, compute_unit, backend, shape_val):
        if _macos_version() < (13, 0) and backend[2] == ct.target.iOS16:
            pytest.skip("iOS16 target not available on macOS 13")
        shape, val = shape_val

        class FullLikeModel(nn.Module):
            def forward(self, x):
                return torch.full_like(x, fill_value=val)

        self.run_compare_torch(
            shape,
            FullLikeModel().eval(),
            backend=backend[:2],
            compute_unit=compute_unit,
            minimum_deployment_target=backend[2]
        )


class TestDim(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1,),
                (2, 3),
                (1, 1, 2, 5, 1),
            ],
        ),
    )
    def test_dim(self, compute_unit, backend, shape):
        class DimModel(nn.Module):
            def forward(self, x):
                return torch.tensor([x.dim()])

        self.run_compare_torch(
            shape,
            DimModel().eval(),
            backend=backend,
            compute_unit=compute_unit
        )


class TestNewZeros(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            [1, 3],
        ),
    )
    def test_new_zeros_dynamic(self, compute_unit, backend, rank):
        class ZerosDynamicModel(nn.Module):
            def forward(self, x):
                if rank == 1:
                    h = x[0]
                    x = torch.zeros(h)
                elif rank == 3:
                    h, w, d = x[0], x[1], x[2]
                    x = torch.zeros(h, w, d)
                return x.new_zeros(x.shape)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        torch_in = torch.tensor(input_shape, dtype=torch.int32)
        model = ZerosDynamicModel().eval()
        torch_out = model(torch_in)
        self.run_compare_torch(
            torch_in,
            model,
            expected_results=torch_out,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1,),
                (2, 3),
                (1, 1, 2, 5, 1),
            ],
        ),
    )
    def test_new_zeros_static(self, compute_unit, backend, shape):
        class ZerosStaticModel(nn.Module):
            def __init__(self):
                super(ZerosStaticModel, self).__init__()

            def forward(self, x):
                return x.new_zeros(x.shape)

        self.run_compare_torch(
            shape,
            ZerosStaticModel().eval(),
            backend=backend,
            compute_unit=compute_unit
        )


class TestNewFull(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            [1, 3],
        ),
    )
    def test_new_full_dynamic(self, compute_unit, backend, rank):
        class FullDynamicModel(nn.Module):
            def forward(self, x):
                if rank == 1:
                    h = x[0]
                    x = torch.zeros(h)
                elif rank == 3:
                    h, w, d = x[0], x[1], x[2]
                    x = torch.zeros(h, w, d)
                return x.new_full(x.shape, fill_value=3.14)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        torch_in = torch.tensor(input_shape, dtype=torch.int32)
        model = FullDynamicModel().eval()
        torch_out = model(torch_in)
        self.run_compare_torch(
            torch_in,
            model,
            expected_results=torch_out,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape_val",
        itertools.product(
            compute_units,
            backends,
            [
                [(1,), 0.0],
                [(2, 3), 3.1415],
                [(1, 1, 2, 5, 1), -2.0],
            ],
        ),
    )
    def test_new_full_static(self, compute_unit, backend, shape_val):
        shape, val = shape_val

        class FullStaticModel(nn.Module):
            def forward(self, x):
                return x.new_full(x.shape, fill_value=val)

        self.run_compare_torch(
            shape,
            FullStaticModel().eval(),
            backend=backend,
            compute_unit=compute_unit
        )


class TestEye(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, eye_type",
        itertools.product(
            compute_units,
            backends,
            ["single", "double"],
        ),
    )
    def test(self, compute_unit, backend, eye_type):
        class Model(nn.Module):
            def forward(self, x):
                if eye_type == "single":
                    eye = torch.eye(3)
                    return x + eye
                elif eye_type == "double":
                    eye = torch.eye(2, 3)
                    return x + eye

        input_shape = (3, 3) if eye_type == "single" else (2, 3)
        model = Model().eval()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestOnes(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            [1, 3],
        ),
    )
    def test_ones_dynamic(self, compute_unit, backend, rank):
        class OnesDynamicModel(nn.Module):
            def forward(self, x):
                if rank == 1:
                    h = x[0]
                    x = torch.zeros(h)
                elif rank == 3:
                    h, w, d = x[0], x[1], x[2]
                    x = torch.zeros(h, w, d)
                return torch.ones(x.shape)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        torch_in = torch.tensor(input_shape, dtype=torch.int32)
        model = OnesDynamicModel().eval()
        torch_out = model(torch_in)
        self.run_compare_torch(
            torch_in,
            model,
            expected_results=torch_out,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [(1,), (2, 3), (1, 1, 2, 5, 1)],
        ),
    )
    def test_ones_static(self, compute_unit, backend, shape):
        class OnesStaticModel(nn.Module):
            def forward(self, x):
                return torch.ones(x.shape)

        self.run_compare_torch(
            shape,
            OnesStaticModel().eval(),
            backend=backend,
            compute_unit=compute_unit
        )


class TestRandint(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape, low, high",
        itertools.product(
            compute_units,
            backends,
            [(1,), (2, 3)],
            [-1, 2],
            [3, 5],
        ),
    )
    def test_randint(self, compute_unit, backend, shape, low, high):
        class TestModel(nn.Module):
            def forward(self, x):
                y = torch.randint(low, high, x.shape)
                return torch.Tensor([len(y)])

        self.run_compare_torch(
            shape,
            TestModel(),
            backend=backend,
            compute_unit=compute_unit
        )


class TestTypeAs(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, type",
        itertools.product(compute_units, backends, ["int32", "float32", "bool"]),
    )
    def test_type_as(self, compute_unit, backend, type):
        class TestNet(nn.Module):
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
            torch.Tensor([0, 1, 2, 3]).to(torch.float32),
            torch.Tensor([2, 3]).to(type_map[type]),
        ]
        self.run_compare_torch(
            input,
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False
        )


class TestReduction(TorchBaseTest):

    class TestModel(nn.Module):
        def __init__(self, mode, dim=None, keepdim=None):
            super().__init__()
            args = {"dim": dim, "keepdim": keepdim}
            self.op_args = {k:v for k, v in args.items() if v is not None}

            if mode == "min":
                self.op = torch.min
            elif mode == "max":
                self.op = torch.max
            else:
                raise ValueError("Unsupported mode: {mode}".format(mode=mode))

        def forward(self, x, y=None):
            if y is not None:
                return self.op(x, y)
            return self.op(x, **self.op_args)


    @pytest.mark.parametrize(
        "compute_unit, backend, input_shape, dim, keepdim, mode",
        itertools.product(
            compute_units,
            backends,
            [(2, 2), (1, 1)],
            [0, 1],
            [True, False],
            ["min", "max"]
        )
    )
    def test_min_max(self, compute_unit, backend, input_shape, dim, keepdim, mode):
        input_data = torch.rand(input_shape)
        model = self.TestModel(mode, dim=dim, keepdim=keepdim)
        # rdar://62681982 (Determine the output names of MLModels)
        expected_results = model(input_data)[::-1]
        self.run_compare_torch(
            input_data,
            model,
            expected_results=expected_results,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, input_shape, mode",
        itertools.product(
            compute_units,
            backends,
            [(2, 2), (1, 1)],
            ["min", "max"]
        )
    )
    def test_min_max_with_no_arguments(self, compute_unit, backend, input_shape, mode):
        self.run_compare_torch(
            input_shape,
            self.TestModel(mode),
            backend=backend,
            compute_unit=compute_unit
        )
        
    @pytest.mark.parametrize(
        "compute_unit, backend, input_shape, dim, mode",
        itertools.product(
            compute_units,
            backends,
            [(2, 2), (1, 1)], [0, 1],
            ["min", "max"]
        )
    )
    def test_min_max_no_keepdim(self, compute_unit, backend, input_shape, dim, mode):
        input_data = torch.rand(input_shape)
        model = self.TestModel(mode, dim=dim)
        print(type(model))
        expected_results = model(input_data)[::-1]
        
        self.run_compare_torch(
            input_data,
            model,
            expected_results=expected_results,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, input_shape, mode",
        itertools.product(
            compute_units,
            backends,
            [(2, 2), (1, 1)],
            ["min", "max"]
        )
    )
    def test_min_max_two_tensors(self, compute_unit, backend, input_shape, mode):
        model = self.TestModel(mode)
        self.run_compare_torch(
            [input_shape] * 2,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestLayerNorm(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, input_shape, eps",
        itertools.product(
            [ct.ComputeUnit.CPU_ONLY],
            backends,
            [(1, 3, 15, 15), (1, 1, 1, 1)],
            [1e-5, 1e-7]
        ),
    )
    def test_layer_norm(self, compute_unit, backend, input_shape, eps):
        model = nn.LayerNorm(input_shape, eps=eps)
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestPixelShuffle(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, batch_size, CHW, r",
        itertools.product(
            compute_units,
            backends,
            [1, 3],
            [(1, 4, 4), (3, 2, 3)],
            [2, 4]
        ),
    )
    def test_pixel_shuffle(self, compute_unit, backend, batch_size, CHW, r):
        C, H, W = CHW
        input_shape = (batch_size, C * r * r, H, W)
        model = nn.PixelShuffle(upscale_factor=r)
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


@pytest.mark.skipif(_macos_version() < (13, 0), reason="New functionality in macOS13/iOS16")
class TestPixelUnshuffle(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, batch_size, CHW, r",
        itertools.product(
            compute_units,
            backends,
            [1, 3],
            [(1, 4, 4), (3, 2, 3)],
            [2, 4]
        ),
    )
    def test_pixel_shuffle(self, compute_unit, backend, batch_size, CHW, r):
        if backend[0] == "neuralnetwork":
            pytest.skip("pixel_unshuffle only supported in mlprogram backend.")

        C, H, W = CHW
        input_shape = (batch_size, C, H * r, W * r)
        model = nn.PixelUnshuffle(downscale_factor=r)
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
            minimum_deployment_target=ct.target.iOS16
        )


class TestExpand(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shapes",
        itertools.product(
            compute_units,
            backends,
            [
                [(2, 1), (2, 2)],
                [(3, 1), (-1, 4)],
                [(1, 3, 4, 4), (3, 3, 4, 4)],
                [(4,), (3, 4)],
                [(3, 2), (1, 2, -1, 2)]
            ],
        ),
    )
    def test_expand(self, compute_unit, backend, shapes):
        input_shape, output_shape = shapes

        class TestModel(torch.nn.Module):
            def forward(self, x):
                return x.expand(*output_shape)

        model = TestModel()

        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )
        
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_expand_dynamic_shape0(self, compute_unit, backend):
        class TestModel(nn.Module):
            def forward(self, x):
                return x.expand(x.shape[1], x.shape[1])

        self.run_compare_torch(
            torch.arange(20).reshape((1, 20)),
            TestModel(),
            input_as_shape=False,
            converter_input_type=[TensorType(shape=[1, ct.RangeDim()])],
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_expand_dynamic_shape1(self, compute_unit, backend):
        class TestModel(nn.Module):
            def forward(self, x):
                return x.expand(x.shape[0], 1, x.shape[-1], x.shape[-1])

        self.run_compare_torch(
            torch.arange(20).reshape((1, 20)),
            TestModel(),
            input_as_shape=False,
            converter_input_type=[TensorType(shape=[ct.RangeDim(), ct.RangeDim()])],
            backend=backend,
            compute_unit=compute_unit,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_expand_dynamic_shape2(self, compute_unit, backend):
        class TestModel(nn.Module):
            def forward(self, x):
                return x.expand(x.shape[-1], 1, x.shape[-1], x.shape[-1])

        self.run_compare_torch(
            torch.arange(20).reshape((1, 20)),
            TestModel(),
            input_as_shape=False,
            converter_input_type=[TensorType(shape=[1, ct.RangeDim()])],
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_expand_dynamic_shape3(self, compute_unit, backend):
        class TestModel(nn.Module):
            def forward(self, x):
                return x.expand(x.shape[0], 10)

        self.run_compare_torch(
            torch.arange(20).reshape((20, 1)),
            TestModel(),
            input_as_shape=False,
            converter_input_type=[TensorType(shape=[ct.RangeDim(), ct.RangeDim()])],
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_expand_dynamic_shape_from_another_input(self, compute_unit, backend):
        class TestModel(nn.Module):
            def forward(self, x, y):
                return x.expand(int(y[0]), int(y[1]))

        self.run_compare_torch(
            [torch.arange(20).reshape((20, 1)), torch.Tensor([20, 20])],
            TestModel(),
            input_as_shape=False,
            converter_input_type=[TensorType(shape=[ct.RangeDim(), 1])],
            backend=backend,
            compute_unit=compute_unit
        )


    @pytest.mark.parametrize(
        "compute_unit, backend, input_shapes",
        itertools.product(
            compute_units,
            backends,
            [
                [(2, 1), (2, 2)],
                [(3, 1), (3, 4)],
                [(1, 3, 4, 4), (3, 3, 4, 4)],
                [(4,), (1, 3, 4)]
            ],
        ),
    )
    def test_expand_as(self, compute_unit, backend, input_shapes):
        class TestModel(torch.nn.Module):
            def forward(self, x, y):
                return x.expand_as(y)

        model = TestModel()

        self.run_compare_torch(
            input_shapes,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestExpandDims(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank_and_axis",
        itertools.product(
            compute_units,
            backends,
            [
                (rank, axis)
                for rank in range(1, 5)
                for axis in range(-rank - 1, rank + 1)
            ],
        ),
    )
    def test_unsqueeze(self, compute_unit, backend, rank_and_axis):
        rank, axis = rank_and_axis
        input_shape = tuple(np.random.randint(low=2, high=10, size=rank))
        model = ModuleWrapper(function=torch.unsqueeze, kwargs={"dim": axis})
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestLinspace(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, start_end, steps",
        itertools.product(
            compute_units,
            backends,
            [(-0.1, -0.7), (1, 10)],
            [1, 3],
        ),
    )
    def test_linspace_static(self, compute_unit, backend, start_end, steps):
        input_shape = tuple([steps])
        start, end = start_end

        class Model(nn.Module):
            def forward(self, x):
                return torch.linspace(start, end, steps)

        model = Model()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_linspace_static_large(self, compute_unit, backend):
        input_shape = tuple([1])

        class Model(nn.Module):
            def forward(self, x):
                return torch.linspace(1, 2_000_000, 2_000_000)

        model = Model()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, start_end, steps",
        itertools.product(
            compute_units,
            backends,
            [(-0.1, -0.7), (1, 10)],
            [1, 2, 100],
        ),
    )
    def test_linspace_dynamic(self, compute_unit, backend, start_end, steps):
        start, end = start_end

        class Model(nn.Module):
            def forward(self, x):
                return torch.linspace(x[0], x[1], steps)

        model = Model()
        inputs = [torch.Tensor([start, end])]
        self.run_compare_torch(
            inputs,
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False
        )


class TestArange(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, start_end_step",
        itertools.product(
            compute_units,
            backends,
            [
                (-0.1, -0.7, -0.07),
                (3, 10, 0.3),
                (1, 10, 100),
                (1, 300000, 1),
                (1, 10, 1e-6),
            ],
        ),
    )
    def test_arange_static(self, compute_unit, backend, start_end_step):
        if start_end_step == (1, 10, 1e-6):
            pytest.xfail(
                "rdar://88998831 (range_1d has numerical issue when the step is small)"
            )
        input_shape = tuple(
            [
                1,
            ]
        )
        start, end, step = start_end_step

        class Model(nn.Module):
            def forward(self, x):
                return torch.arange(start, end, step)

        model = Model()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, start_end_step",
        itertools.product(
            compute_units,
            backends,
            [
                (-0.1, -0.7, -0.07),
                (3, 10, 0.3),
                (1, 10, 100),
                (1, 300000, 1),
            ],
        ),
    )
    def test_arange_dynamic(self, compute_unit, backend, start_end_step):
        start, end, step = start_end_step

        class Model(nn.Module):
            def forward(self, x):
                return torch.arange(x[0], x[1], x[2])

        model = Model()
        inputs = [torch.tensor([start, end, step])]
        self.run_compare_torch(
            inputs,
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False
        )


class TestEinsum(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, equation, reverse_input_order",
        itertools.product(
            compute_units,
            backends,
            [
                "abcd,adce->abce",
                "abc,cbd->abd",
                "bnqd,bnkd->bnqk",
                "abc,cd->abd",
                "abc,cde->abde",
                "btnh,bfnh->bnft",
                "bnft,btnh->bfnh",
                "abcd,cde->abe",
                "a b c d , a d c e -> a b c e",
                "bhcq,bhck->bhqk",
            ],
            [False, True],
        ),
    )
    def test_einsum(self, compute_unit, backend, equation, reverse_input_order):
        class TestEinsum(nn.Module):
            def forward(self, x, y):
                return torch.einsum(equation, x, y)

        if equation in ["abcd,adce->abce", "a b c d , a d c e -> a b c e"]:
            input_shapes = [[3, 4, 2, 6], [3, 6, 2, 2]]
        elif equation == "abc,cbd->abd":
            input_shapes = [[4, 2, 6], [6, 2, 2]]
        elif equation == "bnqd,bnkd->bnqk":
            input_shapes = [[1, 2, 3, 4], [1, 2, 4, 4]]
        elif equation == "abc,cd->abd":
            input_shapes = [[2, 3, 4], [4, 5]]
        elif equation == "abc,cde->abde":
            input_shapes = [[2, 3, 4], [4, 5, 6]]
        elif equation == "btnh,bfnh->bnft":
            input_shapes = [[1, 2, 3, 4], [1, 5, 3, 4]]
        elif equation == "bnft,btnh->bfnh":
            input_shapes = [[1, 2, 3, 4], [1, 4, 2, 6]]
        elif equation == "abcd,cde->abe":
            input_shapes = [[1, 2, 3, 4], [3, 4, 6]]
        elif equation == "bhcq,bhck->bhqk":
            input_shapes = [[2, 3, 4, 5], [2, 3, 4, 6]]
        else:
            raise ValueError("unrecognized equation")

        if reverse_input_order:
            input_output_strings = equation.split('->')
            input_strings = input_output_strings[0].split(',')
            equation = input_strings[1] + ',' + input_strings[0] + '->' + input_output_strings[1]
            input_shapes = [input_shapes[1], input_shapes[0]]

        model = TestEinsum()
        self.run_compare_torch(
            input_shapes,
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=True
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_einsum_with_same_input(self, compute_unit, backend):
        class Einsum(nn.Module):
            def forward(self, m1, m2, m3):
                y1 = torch.einsum('bnhd,bdhm->bnhm', m1, m2)
                y2 = torch.einsum('bnhd,bdhm->bnhm', m1, m3)
                return y1, y2

        m1 = torch.rand(1, 8, 8, 64)
        m3 = torch.rand(1, 8, 128, 64).transpose(1, 3).transpose(2, 3)
        m2 = m3.clone()
        model = Einsum()
        out = model(m1, m2, m3)

        self.run_compare_torch(
            [m1, m2, m3],
            Einsum(),
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False,
            expected_results=out,
        )


class TestSqueeze(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank_and_axis",
        itertools.product(
            compute_units,
            backends,
            [
                (2, 1),
                (2, 0),
                (3, 1),
                (3, None),
                (4, None),
                (4, 2),
                (5, None),
                (5, -1),
            ],
        ),
    )
    def test_squeeze(self, compute_unit, backend, rank_and_axis):
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
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestCumSum(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, axis",
        itertools.product(
            compute_units,
            backends,
            [-1, 0, 1, 2, 3],
        ),
    )
    def test_cumsum(self, compute_unit, backend, axis):
        input_shape = list(np.random.randint(low=2, high=10, size=4))
        input_shape = tuple(input_shape)
        model = ModuleWrapper(
            function=torch.cumsum, kwargs={"dim": axis}
        )
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestReshape(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, output_shape",
        itertools.product(
            compute_units,
            backends,
            [
                (3, 2),
                (2, -1),
                (2, 1, 1, 3),
            ],
        ),
    )
    def test_reshape(self, compute_unit, backend, output_shape):
        input_shape = (2, 3)
        model = ModuleWrapper(function=torch.reshape, kwargs={"shape": output_shape})
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )
        
class TestReshapeAs(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, input_output_shape",
        itertools.product(
            compute_units,
            backends,
            [
                ((6, 1, 1), (3, 2)),
                ((8,), (2, 1, 1, 2, 2)),
            ],
        ),
    )
    def test_reshape(self, compute_unit, backend, input_output_shape):
        class Model(nn.Module):
            def forward(self, x, ref):
                return x.reshape_as(ref)
        
        model = Model()
        input_shape, output_shape = input_output_shape
        self.run_compare_torch(
            [input_shape, output_shape],
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestFlatten(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, start_dim, end_dim, is_dynamic",
        itertools.product(
            compute_units,
            backends,
            [2, -2, 0],
            [3, -1],
            [False, True]
        ),
    )
    def test_flatten(self, compute_unit, backend, start_dim, end_dim, is_dynamic):
        input_shape = (2, 3, 4, 5)
        converter_input_type = None
        if is_dynamic:
            converter_input_type = [
                TensorType(
                    shape=(2, 3, RangeDim(default=4), RangeDim(default=5)),
                    dtype=np.float32,
                )
            ]
        model = ModuleWrapper(
            function=torch.flatten, kwargs={"start_dim": start_dim, "end_dim": end_dim}
        )
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
            converter_input_type=converter_input_type
        )


class TestGather(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank_and_axis",
        itertools.product(
            compute_units,
            backends,
            [(i, j) for i in range(1, 6) for j in range(0, i)]
        ),
    )
    def test_gather_along_axis(self, compute_unit, backend, rank_and_axis):
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
        self.run_compare_torch(
            [params_shape],
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestActivation(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES_ALL
        ),
    )
    def test_relu(self, compute_unit, backend, shape):
        model = nn.ReLU().eval()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

        model = ModuleWrapper(nn.functional.relu_)
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES_ALL
        ),
    )
    def test_relu6(self, compute_unit, backend, shape):
        model = nn.ReLU6().eval()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, alpha, shape, single_alpha",
        itertools.product(
            compute_units,
            backends,
            [0.25, 2.0],
            [
                (3,),
                (2, 6),
                (2, 3, 4),
                (2, 5, 6, 7),
                (2, 3, 4, 5, 6)
            ],
            [True, False],
        ),
    )
    def test_prelu(self, compute_unit, backend, alpha, shape, single_alpha):
        if (backend[0] == "mlprogram" and backend[1] == "fp16" or (len(shape) == 5)):
            pytest.xfail(
                "rdar://92175249 ([MIL] TestActivation::test_prelu[backend=(mlprogram, fp16)] CI failure)"
            )
        input_shape = shape
        num_parameters = input_shape[1] if len(input_shape) >= 2 else 1
        if single_alpha:
            num_parameters = 1
        model = nn.PReLU(num_parameters, alpha).eval()
        mlmodel = self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )
        prog = mlmodel[1]._mil_program
        # Unfortunately since all these tests result in a prelu with a common leakage factor, the
        # prelu_to_lrelu pass optimizes them to contain leaky_relu instead.
        assert len(prog.find_ops(op_type="leaky_relu")) == 1
        assert len(prog.find_ops(op_type="prelu")) == 0

    @pytest.mark.parametrize(
        "compute_unit, backend, shape, alpha",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES_ALL,
            [0.1, 2.0, 1.4]
        )
    )
    def test_leaky_relu(self, compute_unit, backend, shape, alpha):
        model = nn.LeakyReLU(negative_slope=alpha).eval()
        self.run_compare_torch(
            shape, model, backend=backend,
        )

        model = ModuleWrapper(nn.functional.leaky_relu_, {'negative_slope': alpha})
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES_ALL,
        )
    )
    def test_randomized_leaky_relu(self, compute_unit, backend, shape):
        model = nn.RReLU(lower=0.01, upper=0.9).eval()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES_ALL
        ),
    )
    def test_softmax(self, compute_unit, backend, shape):
        model = nn.Softmax().eval()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, range_val",
        itertools.product(
            compute_units,
            backends,
            [
                (-1.0, 1.0),
                (0.0, 0.1),
                (1.0, 3.0),
                (-1.0, 6.0)
            ]
        ),
    )
    def test_hardtanh(self, compute_unit, backend, range_val):
        input_shape = (1, 10, 4, 5)
        model = nn.Hardtanh(range_val[0], range_val[1]).eval()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

        model = ModuleWrapper(
            nn.functional.hardtanh_, {"min_val": range_val[0], "max_val": range_val[1]}
        )
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape, alpha",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES_ALL,
            [0.1, 2.0, 1.4]
        )
    )
    def test_elu(self, compute_unit, backend, shape, alpha):
        model = nn.ELU(alpha).eval()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES_ALL
        )
    )
    def test_gelu(self, compute_unit, backend, shape):
        model = nn.GELU().eval()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.skipif(_python_version() < (3, 6), reason="requires python 3.6")
    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES_ALL
        ),
    )
    def test_erf(self, compute_unit, backend, shape):

        class ERFActivation(nn.Module):
            def forward(self, x):
                return torch.erf(x)

        model = ERFActivation().eval()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 10),
                (1, 3, 5),
                (1, 5, 6, 7),
                (1, 3, 4, 5, 6)
            ]
        ),
    )
    def test_sigmoid(self, compute_unit, backend, shape):
        model = nn.Sigmoid().eval()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.skipif(_python_version() < (3, 6), reason="requires python 3.6")
    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES_ALL
        )
    )
    def test_sigmoid_hard(self, compute_unit, backend, shape):
        model = nn.Hardsigmoid().eval()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, beta, threshold",
        itertools.product(
            compute_units,
            backends,
            [1, 2, 5],
            [5, 10, 20]
        ),
    )
    @pytest.mark.skipif(
        _macos_version() <= (10, 15),
        reason="Parametric SoftPlus segfaults on macOS 10.15 and below.",
    )
    def test_softplus(self, compute_unit, backend, beta, threshold):
        input_shape = (1, 10, 5, 15)
        model = nn.Softplus(beta, threshold).eval()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES_ALL
        )
    )
    def test_softsign(self, compute_unit, backend, shape):
        model = nn.Softsign().eval()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.skipif(
        condition=version_lt(torch, "1.7.0"),
        reason="torch.nn.SiLU available only in PyTorch 1.7.0+",
    )
    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 10),
                (1, 3, 4),
                (1, 4, 5, 6)
            ]),
    )
    def test_silu(self, compute_unit, backend, shape):
        model = ModuleWrapper(function=torch.nn.functional.silu)
        self.run_compare_torch([shape], model, backend=backend)

    @pytest.mark.parametrize(
        "compute_unit, backend, rounding_mode",
        itertools.product(
            compute_units,
            backends,
            [None, "floor", "trunc"]
        ),
    )
    def test_div(self, compute_unit, backend, rounding_mode):
        model = ModuleWrapper(function=torch.div,
                              kwargs={"rounding_mode": rounding_mode})
        x1 = torch.from_numpy(np.array([2.3, 2.6, -3.6, -3.2], dtype=np.float32))
        x2 = torch.from_numpy(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        out = torch.div(x1, x2, rounding_mode=rounding_mode)
        self.run_compare_torch(
            [x1, x2],
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False,
            expected_results=out,
        )


class TestElementWiseUnary(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape, op_string",
        itertools.product(
            compute_units,
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
    def test_elementwise_no_params(self, compute_unit, backend, shape, op_string):
        if not contains_op(torch, op_string):
            return
        if op_string == "sqrt" and compute_unit != ct.ComputeUnit.CPU_ONLY:
            pytest.skip("sqrt on GPU producing nan.")

        op_func = getattr(torch, op_string)
        model = ModuleWrapper(function=op_func)
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape, clamp_range",
        itertools.product(
            compute_units,
            backends,
            [(1, 3, 5, 8)],
            [
                (0.0, 1.0),
                (-1.0, 0.5),
                (0.2, 0.7),
                (None, 4.0),
                (-3.0, None)
            ],
        ),
    )
    def test_clamp(self, compute_unit, backend, shape, clamp_range):
        params_dict = {}
        if clamp_range[0] is not None:
            params_dict["min"] = clamp_range[0]
        if clamp_range[1] is not None:
            params_dict["max"] = clamp_range[1]

        model = ModuleWrapper(torch.clamp, params_dict)
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
            rand_range=(-5, 5)
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape, threshold",
        itertools.product(
            compute_units,
            backends,
            [(1, 3, 5, 8)],
            [(0.0, 0.0), (0.5, 0.5), (0.5, 10), (0.9, 0.0)]
        ),
    )
    def test_threshold(self, compute_unit, backend, shape, threshold):
        model = torch.nn.Threshold(threshold[0], threshold[1]).eval()
        input_value = torch.rand(np.prod(shape))
        # make sure the values are not too close to the threshold
        for i in range(len(input_value)):
            if abs(input_value[i] - threshold[0]) < 0.005:
                input_value[i] += 0.05
        input_value = torch.reshape(input_value, shape)
        self.run_compare_torch(
            input_value,
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape, op_string",
        itertools.product(
            compute_units,
            backends,
            [(1, 3, 5, 8)],
            [
                "log",
                "rsqrt",
                "reciprocal",
            ],
        ),
    )
    def test_elementwise_numerically_stable(self, compute_unit, backend, shape, op_string):
        op_func = getattr(torch, op_string)
        model = ModuleWrapper(function=op_func)
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
            rand_range=(20, 100)
        )


class TestTriu(TorchBaseTest):

    @pytest.mark.parametrize(
        "compute_unit, backend, shape, diagonal",
        itertools.product(
            compute_units,
            backends,
            [(5, 5), (3, 4), (5, 1)],
            [None, -1, 0, 2],
        ),
    )
    def test_triu(self, compute_unit, backend, shape, diagonal):
        params_dict = {}
        if diagonal is not None:
            params_dict["diagonal"] = diagonal
        model = ModuleWrapper(torch.triu, params_dict)
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestTril(TorchBaseTest):

    @pytest.mark.parametrize(
        "compute_unit, backend, shape, diagonal",
        itertools.product(
            compute_units,
            backends,
            [(5, 5), (3, 4), (5, 1)],
            [None, -1, 0, 2],
        ),
    )
    def test_tril(self, compute_unit, backend, shape, diagonal):
        params_dict = {}
        if diagonal is not None:
            params_dict["diagonal"] = diagonal
        model = ModuleWrapper(torch.tril, params_dict)
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit,
        )


class TestMatMul(TorchBaseTest):

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_bmm(self, compute_unit, backend):
        shape_x, shape_y = (3, 4, 5), (3, 5, 6)
        model = ModuleWrapper(function=torch.bmm)
        self.run_compare_torch(
            [shape_x, shape_y],
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestSplit(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, split_size_or_sections, dim",
        itertools.product(
            compute_units,
            backends,
            [1, 2, [1, 4]],
            [0, -2]
        ),
    )
    def test_split(self, compute_unit, backend, split_size_or_sections, dim):
        input_shape = (5, 2)
        model = ModuleWrapper(function=torch.split,
                              kwargs={"split_size_or_sections": split_size_or_sections, "dim": dim})
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, split_sizes, dim",
        itertools.product(
            compute_units,
            backends,
            [[1, 4], [3, 2]],
            [-1, -2]
        ),
    )
    def test_split_with_sizes(self, compute_unit, backend, split_sizes, dim):
        input_shape = (5, 5)
        model = ModuleWrapper(function=torch.split_with_sizes,
                              kwargs={"split_sizes": split_sizes, "dim": dim})
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestUnbind(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, dim",
        itertools.product(
            compute_units,
            backends,
            [0, 1, 2]
        ),
    )
    def test_unbind(self, compute_unit, backend, dim):
        input_shape = (3, 3, 4)
        model = ModuleWrapper(function=torch.unbind,
                              kwargs={"dim": dim})
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_unbind_one_dim_shape(self, compute_unit, backend):
        input_shape = (1,)
        dim = 0
        model = ModuleWrapper(function=torch.unbind,
                              kwargs={"dim": dim})
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestTranspose(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape, dims",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
            [(0, 1), (-2, -1), (1, 0), (-1, -2)]
        ),
    )
    def test(self, compute_unit, backend, shape, dims):
        model = ModuleWrapper(function=torch.transpose,
                              kwargs={"dim0": dims[0], "dim1": dims[1]})
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestTo(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        )
    )
    def test_cast_bug(self, compute_unit, backend):
        if _macos_version() < (13, 0) and backend[0] == "mlprogram":
            pytest.xfail("Issue fixed in iOS16/macOS13")

        class TestModel(torch.nn.Module):
            def forward(self, spans, embedding):
                spans = spans.float().relu().int()

                max1, _ = torch.max(spans, dim=1, keepdim=False)
                max1, _ = torch.max(max1, dim=1, keepdim=False)
                max2, _ = torch.max(embedding, dim=1, keepdim=False)
                max2, _ = torch.max(max2, dim=1, keepdim=False)
                sigmoided_scores = max1 + max2
                return sigmoided_scores

        if (
            platform.machine() == "arm64"
            and compute_unit != ct.ComputeUnit.CPU_ONLY
            and backend[0] == "neuralnetwork"
        ):
            pytest.xfail(
                "rdar://98015195 ([M1 native tests] Some MIL unittests are failing on M1 native)"
            )
        model = TestModel()
        self.run_compare_torch(
            [(1, 4, 2), (1, 6, 3)],
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestSlice(TorchBaseTest):
    @pytest.mark.skipif(_python_version() < (3, 6), reason="requires python 3.6")
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        )
    )
    def test_dynamic_slice(self, compute_unit, backend):
        class DynamicSlicer(torch.nn.Module):
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
                embeddings = self.dynamic_slicer(
                    embeddings, torch.squeeze(context_length)
                )

                return embeddings

        model = Model()
        batch_size = 5
        inputs = [
            TensorType(name="tokens", shape=(10, batch_size), dtype=np.int64),
            TensorType(name="context", shape=(3, batch_size), dtype=np.int64),
            TensorType(name="context_length", shape=(1,), dtype=np.int32),
        ]
        self.run_compare_torch(
            inputs,
            model,
            rand_range=(0, 8),
            backend=backend,
            compute_unit=compute_unit
        )


class TestRepeat(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            range(1, 6)
        ),
    )
    def test_repeat(self, compute_unit, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        repeats = np.random.randint(low=2, high=4, size=rank)
        input_shape = tuple(input_shape)

        model = ModuleWrapper(function=lambda x: x.repeat(*repeats))
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            (1, 2)
        ),
    )
    def test_repeats_with_extra_dimensions(self, compute_unit, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)

        for num_extra_dims in (1, 2):
            repeats = np.random.randint(low=2, high=4, size=rank + num_extra_dims)
            model = ModuleWrapper(function=lambda x: x.repeat(*repeats))
            self.run_compare_torch(
                input_shape,
                model,
                backend=backend,
                compute_unit=compute_unit
            )
            
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_repeats_with_enumerated_shape_case1(self, compute_unit, backend):
        class Model(nn.Module):
            def forward(self, x, y):
                reps = x.size(0)
                return y.repeat(reps)

        enumerated_shapes = ct.EnumeratedShapes(shapes=[(1, 1), (2, 1)])
        module = Model()
        inputs = [torch.tensor([[1]]), torch.tensor([2])]

        self.run_compare_torch(
            inputs,
            module,
            input_as_shape=False,
            converter_input_type=[
                ct.TensorType(shape=enumerated_shapes),
                ct.TensorType(shape=(1,)),
            ],
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_repeats_with_enumerated_shape_case2(self, compute_unit, backend):
        class Model(nn.Module):
            def forward(self, x, y):
                return y.repeat(x.size(0), x.size(1))

        enumerated_shapes = ct.EnumeratedShapes(shapes=[(1, 1), (2, 1)])
        module = Model()
        inputs = [torch.tensor([[1], [2]]), torch.tensor([2])]
        self.run_compare_torch(
            inputs,
            module,
            input_as_shape=False,
            converter_input_type=[
                ct.TensorType(shape=enumerated_shapes),
                ct.TensorType(shape=(1,)),
            ],
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_repeats_with_symbolic_shape(self, compute_unit, backend):
        class Model(nn.Module):
            def forward(self, x, y):
                return y.repeat([x.shape[-1], 1, x.shape[0]])

        module = Model()
        inputs = [torch.tensor([[1], [2]]), torch.tensor([2])]
        self.run_compare_torch(
            inputs,
            module,
            input_as_shape=False,
            converter_input_type=[
                ct.TensorType(shape=(ct.RangeDim(), ct.RangeDim())),
                ct.TensorType(shape=(1,)),
            ],
            backend=backend,
            compute_unit=compute_unit
        )


class TestStd(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, unbiased",
        itertools.product(
            compute_units,
            backends,
            [True, False]
        ),
    )
    def test_std_2_inputs(self, compute_unit, backend, unbiased):
        model = ModuleWrapper(function=torch.std,
                              kwargs={"unbiased": unbiased})
        x = torch.randn(1, 5, 10) * 3
        out = torch.std(x, unbiased=unbiased).unsqueeze(0)
        self.run_compare_torch(
            x,
            model,
            expected_results=out,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, unbiased, dim, keepdim",
        itertools.product(
            compute_units,
            backends,
            [True, False],
            [[0, 2], [1], [2]],
            [True, False]
        ),
    )
    def test_std_4_inputs(self, compute_unit, backend, unbiased, dim, keepdim):
        model = ModuleWrapper(
            function=torch.std,
            kwargs={"unbiased": unbiased, "dim": dim, "keepdim": keepdim},
        )
        input_shape = (2, 5, 10)
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestOnesLike(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            [1, 3],
        ),
    )
    def test_ones_like_static(self, compute_unit, backend, rank):
        class OnesLikeStaticModel(nn.Module):
            def forward(self, x):
                return torch.ones_like(x)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        input_shape = tuple(input_shape)
        model = OnesLikeStaticModel()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            [
                ["neuralnetwork", "fp32", ct.target.iOS14],
                ["mlprogram", "fp16", ct.target.iOS15],
                ["mlprogram", "fp32", ct.target.iOS15],
                ["mlprogram", "fp16", ct.target.iOS16],
                ["mlprogram", "fp32", ct.target.iOS16],
            ],
            [1, 3],
        ),
    )
    def test_ones_like_dynamic(self, compute_unit, backend, rank):
        if _macos_version() < (13, 0) and backend[2] == ct.target.iOS16:
            pytest.skip("iOS16 target not available on macOS 13")

        class OnesLikeDynamicModel(nn.Module):
            def forward(self, x):
                if rank == 1:
                    h = x[0]
                    x = torch.zeros(h)
                elif rank == 3:
                    h, w, d = x[0], x[1], x[2]
                    x = torch.zeros(h, w, d)
                return torch.ones_like(x)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        torch_in = torch.tensor(input_shape)
        model = OnesLikeDynamicModel()
        torch_out = model(torch_in)
        self.run_compare_torch(
            torch_in,
            model,
            expected_results=torch_out,
            input_as_shape=False,
            backend=backend[:2],
            compute_unit=compute_unit,
            minimum_deployment_target=backend[2],
        )


class TestZeros(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            [1, 3],
        ),
    )
    def test_zeros_like_static(self, compute_unit, backend, rank):
        class ZerosLikeStaticModel(nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        input_shape = tuple(input_shape)
        model = ZerosLikeStaticModel()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            [
                ["neuralnetwork", "fp32", ct.target.iOS14],
                ["mlprogram", "fp16", ct.target.iOS15],
                ["mlprogram", "fp32", ct.target.iOS15],
                ["mlprogram", "fp16", ct.target.iOS16],
                ["mlprogram", "fp32", ct.target.iOS16],
            ],
            [1, 3],
        ),
    )
    def test_zeros_like_dynamic(self, compute_unit, backend, rank):
        if _macos_version() < (13, 0) and backend[2] == ct.target.iOS16:
            pytest.skip("iOS16 target not available on macOS 13")

        class ZerosLikeDynamicModel(nn.Module):
            def forward(self, x):
                if rank == 1:
                    h = x[0]
                    x = torch.zeros(h)
                elif rank == 3:
                    h, w, d = x[0], x[1], x[2]
                    x = torch.zeros(h, w, d)
                return torch.zeros_like(x)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        torch_in = torch.tensor(input_shape, dtype=torch.int32)
        model = ZerosLikeDynamicModel()
        torch_out = model(torch_in)
        self.run_compare_torch(
            torch_in,
            model,
            expected_results=torch_out,
            input_as_shape=False,
            backend=backend[:2],
            compute_unit=compute_unit,
            minimum_deployment_target=backend[2],
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            [1, 3],
        ),
    )
    def test_zeros_static(self, compute_unit, backend, rank):
        class ZerosStaticModel(nn.Module):
            def forward(self, x):
                if rank == 1:
                    return torch.zeros(1)
                elif rank == 3:
                    return torch.zeros(2, 3, 5)

        input_shape = np.random.randint(low=2, high=6, size=rank)
        input_shape = tuple(input_shape)
        model = ZerosStaticModel()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            [1, 3],
        ),
    )
    def test_zeros_dynamic(self, compute_unit, backend, rank):
        class ZerosDynamicModel(nn.Module):
            def forward(self, x):
                if rank == 1:
                    h = x[0]
                    x = torch.zeros(h)
                elif rank == 3:
                    h, w, d = x[0], x[1], x[2]
                    x = torch.zeros(h, w, d)
                return x

        input_shape = np.random.randint(low=2, high=6, size=rank)
        torch_in = torch.tensor(input_shape, dtype=torch.int32)
        model = ZerosDynamicModel()
        torch_out = model(torch_in)
        self.run_compare_torch(
            torch_in,
            model,
            expected_results=torch_out,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )


class TestTopk(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, largest, sort, shape_dim_k",
        itertools.product(
            compute_units,
            backends,
            [True, False],
            [True, False],
            [
                ((4, 6, 7, 3), -1, 2),
                ((10, 3, 4), 2, 2),
                ((5,), 0, 2)
            ],
        ),
    )
    def test_topk(self, compute_unit, backend, largest, sort, shape_dim_k):
        if not sort and backend[0] == "neuralnetwork":
            pytest.xfail("iOS16 version topk needed for sort = False")
        if not sort and _macos_version() < (13, 0):
            pytest.skip("New functionality in macOS13/iOS16")

        input_shape = shape_dim_k[0]
        dim = shape_dim_k[1]
        k = shape_dim_k[2]

        class TopkModel(nn.Module):
            def forward(self, x):
                topk = torch.topk(x, k, dim=dim, largest=largest, sorted=sort)
                values, indices = topk.values, topk.indices
                if not sort:
                    values, _ = torch.sort(values, dim=dim)
                    indices, _ = torch.sort(indices, dim=dim)
                return values, indices

        input_data = torch.rand(input_shape)
        model = TopkModel()
        expected_results = model(input_data)
        self.run_compare_torch(
            input_data,
            model,
            expected_results=expected_results,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit,
            minimum_deployment_target=ct.target.iOS16 if not sort else None,
        )


class TestLog10(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            range(1, 6)
        ),
    )
    def test_log10(self, compute_unit, backend, rank):
        class Log10Model(nn.Module):
            def forward(self, x):
                return torch.log10(x)

        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = Log10Model()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestLog2(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            range(1, 6)
        ),
    )
    def test_log2(self, compute_unit, backend, rank):
        class Log2Model(nn.Module):
            def __init__(self):
                super(Log2Model, self).__init__()

            def forward(self, x):
                return torch.log2(x)

        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = Log2Model()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestFlip(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank_dim",
        itertools.product(
            compute_units,
            backends,
            [
                (1, [0]),
                (2, [0, 1]),
                (3, [1]),
                (4, [0, 1, 2, 3])
            ]
        ),
    )
    def test_flip(self, compute_unit, backend, rank_dim):
        rank, dim = rank_dim

        class FlipModel(nn.Module):
            def forward(self, x):
                return torch.flip(x, dim)

        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        model = FlipModel()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestBitWiseLogical(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, x_y, op_string",
        itertools.product(
            compute_units,
            backends,
            [
                ([True, False, True, False], [True, True, False, False]),
                ([[True, False], [True, False]], [[True, True], [False, False]]),
                ([[True, False], [True, False]], [[1, 0], [2, 1]]),
                ([-1.5, 0.0, 1.0, 0.0], [0.1, 2.5, 0.0, 0.0]),
                ([2, 0, -1, 0, 5], [1, 1, 0, 0, -5]),
            ],
            [
                "eq",
                "ne",
            ],
        ),
    )
    def test_bitwise_logical(self, compute_unit, backend, x_y, op_string):
        if not contains_op(torch, op_string):
            return
        op_func = getattr(torch, op_string)
        model = ModuleWrapper(function=op_func)
        x = torch.tensor(x_y[0])
        y = torch.tensor(x_y[1])
        self.run_compare_torch(
            [x, y],
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False
        )


class TestLogicalAnd(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, x_y",
        itertools.product(
            compute_units,
            backends,
            [
                ([True, False, True, False], [True, True, False, False]),
                ([[True, False], [True, False]], [[True, True], [False, False]]),
                ([-1.5, 0.0, 1.0, 0.0], [0.1, 2.5, 0.0, 0.0]),
                ([2, 0, -1, 0, 5], [1, 1, 0, 0, -5]),
            ],
        ),
    )
    def test_logical_and(self, compute_unit, backend, x_y):
        class TestNet(nn.Module):
            def forward(self, x, y):
                return torch.logical_and(x, y)

        model = TestNet()
        x = torch.tensor(x_y[0])
        y = torch.tensor(x_y[1])
        self.run_compare_torch(
            [x, y],
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False
        )


class TestLogicalOr(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, x_y",
        itertools.product(
            compute_units,
            backends,
            [
                ([True, False, True, False], [True, True, False, False]),
                ([[True, False], [True, False]], [[True, True], [False, False]]),
                ([-1.5, 0.0, 1.0, 0.0], [0.1, 2.5, 0.0, 0.0]),
                ([2, 0, -1, 0, 5], [1, 1, 0, 0, -5]),
            ],
        ),
    )
    def test_logical_or(self, compute_unit, backend, x_y):
        class TestNet(nn.Module):
            def forward(self, x, y):
                return torch.logical_or(x, y)

        model = TestNet()
        x = torch.tensor(x_y[0])
        y = torch.tensor(x_y[1])
        self.run_compare_torch(
            [x, y],
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False
        )


class TestLogicalXor(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, x_y",
        itertools.product(
            compute_units,
            backends,
            [
                ([True, False, True, False], [True, True, False, False]),
                ([[True, False], [True, False]], [[True, True], [False, False]]),
                ([-1.5, 0.0, 1.0, 0.0], [0.1, 2.5, 0.0, 0.0]),
                ([2, 0, -1, 0, 5], [1, 1, 0, 0, -5]),
            ],
        ),
    )
    def test_logical_xor(self, compute_unit, backend, x_y):
        class TestNet(nn.Module):
            def forward(self, x, y):
                return torch.logical_xor(x, y)

        model = TestNet()
        x = torch.tensor(x_y[0])
        y = torch.tensor(x_y[1])
        self.run_compare_torch(
            [x, y],
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False
        )


class TestWhere(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [(2, 6), (3, 4, 5)]
        ),
    )
    def test_where_test1(self, compute_unit, backend, shape):
        class WhereModel(nn.Module):
            def forward(self, x, y):
                return torch.where(x > 0.5, x, y)

        input_shape = [shape, shape]
        model = WhereModel()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [(2, 6), (3, 4, 5)]
        ),
    )
    def test_where_test2(self, compute_unit, backend, shape):
        class WhereModel(nn.Module):
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
            compute_unit=compute_unit,
            expected_results=expected_results,
            input_as_shape=False,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shapes",
        itertools.product(
            compute_units,
            backends,
            [
                [(1, 2), (1, 2), (1, 1)],
                [(1, 2, 3), (1, 1, 1), (1, 1, 3)],
            ]
        ),
    )
    def test_where_test3(self, compute_unit, backend, shapes):
        class WhereModel(nn.Module):
            def forward(self, cond, x, y):
                return torch.where(cond, x, y)

        cond_shape, x_shape, y_shape = shapes
        cond = torch.rand(*cond_shape) > 0.5
        inputs = [cond, torch.rand(*x_shape), torch.rand(*y_shape)]
        model = WhereModel()
        expected_results = model(*inputs)
        self.run_compare_torch(
            inputs,
            model,
            backend=backend,
            compute_unit=compute_unit,
            expected_results=expected_results,
            input_as_shape=False,
        )
        
    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES + [(10,)]
        )
    )
    def test_where_single_param(self, compute_unit, backend, shape):
        class WhereModelSingleParam(nn.Module):
            def forward(self, x):
                return torch.where(x)

        # Create a tensor of given shape with ~90% zero entries
        x = np.zeros(shape)
        all_indices = list(zip(*np.where(x == 0)))
        num_indices = len(all_indices)
        random_picks = np.random.choice(np.arange(num_indices), size=num_indices//10, replace=False)
        for i in random_picks:
            x[all_indices[i]] = np.random.choice([-1, 12, 100])
        x = torch.Tensor(x)

        self.run_compare_torch(
            x,
            WhereModelSingleParam(),
            backend=backend,
            input_as_shape=False,
            compute_unit=compute_unit,
        )


class TestSelect(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, dim_index",
        itertools.product(
            compute_units,
            backends,
            [
                [0, 0],
                [1, 1],
                [-1, -1],
            ]
        ),
    )
    def test_select(self, compute_unit, backend, dim_index):
        dim, index = dim_index

        class SelectModel(nn.Module):
            def forward(self, x):
                return x.select(dim, index)

        input_shape = (1, 2, 3)
        model = SelectModel()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestNonZero(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, as_tuple",
        itertools.product(
            compute_units,
            backends,
            [1, 3],
            [False, True],
        ),
    )
    def test_non_zero(self, compute_unit, backend, rank, as_tuple):

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
            {'as_tuple':as_tuple},
        )

        self.run_compare_torch(
            input,
            model,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )


class TestTorchTensor(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            [1, 2, 3, 4, 5],
        ),
    )
    def test_torch_tensor(self, compute_unit, backend, rank):
        class Model(nn.Module):
            def __init__(self, rank):
                super(Model, self).__init__()
                self.rank = rank

            def forward(self, x):
                with torch.no_grad():
                    if self.rank == 1:
                        return self.generate_tensor_rank_1(x)
                    if self.rank == 2:
                        return self.generate_tensor_rank_2(x)
                    if self.rank == 3:
                        return self.generate_tensor_rank_3(x)
                    if self.rank == 4:
                        return self.generate_tensor_rank_4(x)
                    if self.rank == 5:
                        return self.generate_tensor_rank_5(x)

            @torch.jit.script
            def generate_tensor_rank_1(x):
                _, _, h, w = x.shape
                return torch.tensor([h, w, 0, 1], dtype=torch.int32)

            @torch.jit.script
            def generate_tensor_rank_2(x):
                _, _, h, w = x.shape
                return torch.tensor([[0, h], [h, w], [w, w]], dtype=torch.float32)

            @torch.jit.script
            def generate_tensor_rank_3(x):
                _, _, h, w = x.shape
                return torch.tensor([[[h, 1]], [[3, w]]], dtype=torch.int32)

            @torch.jit.script
            def generate_tensor_rank_4(x):
                _, _, h, w = x.shape
                return torch.tensor(
                    [
                        [[[h, h], [h, w]], [[w, w], [w, 1]]],
                        [[[0, 0], [1, 1]], [[0, h], [h, w]]],
                    ],
                    dtype=torch.float32,
                )

            @torch.jit.script
            def generate_tensor_rank_5(x):
                _, _, h, w = x.shape
                return torch.tensor(
                    [[[[[h, w], [w, w]], [[1, 1], [0, h]]]]], dtype=torch.float32
                )

        shape = (1, 1, 3, 4)
        model = Model(rank)
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, torch_op",
        itertools.product(
            compute_units,
            backends,
            [
                torch.abs,
                torch.acos,
                torch.asin,
                torch.atan,
                torch.atanh,
                torch.ceil,
                torch.cos,
                torch.cosh,
                torch.exp,
                torch.exp2,
                torch.floor,
                torch.round,
                torch.rsqrt,
                torch.sign,
                torch.sin,
                torch.sinh,
                torch.sqrt,
                torch.square,
                torch.tan,
                torch.tanh
            ],
        ),
    )
    def test_torch_rank0_tensor(self, compute_unit, backend, torch_op):
        class Model(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch_op(torch.tensor(.1))

        model = Model()
        self.run_compare_torch(
            torch.tensor([1., 2., 3.]),
            model,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )


class TestTensorAssign(TorchBaseTest):

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_tensor_assign_case_1(self, compute_unit, backend):
        # single dimension assignment for a 1D tensor
        class TensorAssignModel(torch.nn.Module):
            def forward(self, x):
                x[0] = 0
                x[1] = 1
                y = x + 1
                x[1] = 2 * y[1]
                return x, y

        shape = (5,)
        model = TensorAssignModel()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_tensor_assign_case_2(self, compute_unit, backend):
        # single dimension assignment for two 1D tensors
        class TensorAssignModel(torch.nn.Module):
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
            [shape, shape],
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (5, 4),
                (5, 4, 3),
            ]
        ),
    )
    def test_tensor_assign_case_3(self, compute_unit, backend, shape):
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
            [shape, shape],
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_itensor_assign_case_4(self, compute_unit, backend):
        # single dimension assignment for two n-D tensors
        class TensorAssignModel(torch.nn.Module):
            def forward(self, x, y):
                x[0] = torch.tensor([1., 2., 3., 4.])
                x[3] = 1
                y[0] = x[0]
                return x, y

        shape = (5, 4)
        model = TensorAssignModel()
        self.run_compare_torch(
            [shape, shape],
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_tensor_assign_case_5(self, compute_unit, backend):
        # slice dimension assigment
        class TensorAssignModel(torch.nn.Module):
            def forward(self, x):
                x[:, 1] = torch.tensor([1.0, 2.0])
                return x

        shape = (2, 10)
        model = TensorAssignModel()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_tensor_assign_case_6(self, compute_unit, backend):
        # a more complicated slice dimension assigment
        class TensorAssignModel(torch.nn.Module):
            def forward(self, x):
                x[:, 1, :] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).view(2, 3)
                return x

        shape = (2, 10, 3)
        model = TensorAssignModel()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_tensor_assign_type_compatibility(self, compute_unit, backend):
        class TensorAssignModel(torch.nn.Module):
            def forward(self, x):
                x[:, 1] = torch.tensor([1, 2], dtype=torch.int32)
                return x

        shape = (2, 3)
        model = TensorAssignModel()
        self.run_compare_torch(shape, model, backend=backend, compute_unit=compute_unit)


class TestIndexPut(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_index_put_case_1(self, compute_unit, backend):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, y):
                y = x + 1
                mask = torch.tensor([True, False, False, False, True, True]).view(3, 2)
                x[mask] = y[mask]
                return x

        shape = (3, 2)
        model = IndexPutModel()
        self.run_compare_torch(
            [shape, shape],
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            [0, 1],
        ),
    )
    def test_index_put_case_2(self, compute_unit, backend, rank):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x):
                mask = torch.tensor([True, False, False, False, True, True]).view(3, 2)
                if rank == 0:
                    x[mask] = 0.
                if rank == 1:
                    x[mask] = torch.tensor([1.])
                return x

        shape = (3, 2)
        model = IndexPutModel()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_index_put_case_3(self, compute_unit, backend):
        if _macos_version() < (13, 0):
            pytest.skip("Issue fixed in iOS16/macOS13")

        class IndexPutModel(torch.nn.Module):
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
            inputs,
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank, accumulate",
        itertools.product(
            compute_units,
            backends,
            [1, 2],
            [True, False]
        ),
    )
    def test_index_put_case_4(self, compute_unit, backend, rank, accumulate):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, indices, values):
                x.index_put_(tuple(indices.t()), values, accumulate=accumulate)
                return x

        if rank == 1:
            inputs = [
                torch.Tensor([1., 2., 3., 4., 5., 6]),
                torch.LongTensor([[0], [4]]),
                torch.Tensor([3., 7.])
            ]
        elif rank == 2:
            inputs = [
                torch.ones([3, 4]),
                torch.LongTensor([[0, 1], [1, 2], [2, 2]]),
                torch.Tensor([1., 5., 8.]),
            ]

        model = IndexPutModel()
        self.run_compare_torch(
            inputs,
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False,
        )


class TestIndex(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (10,),
                (3, 4, 5, 6),
            ]
        ),
    )
    def test_index_bool_index(self, compute_unit, backend, shape):
        class IndexModel(torch.nn.Module):
            def forward(self, x):
                return x[x > 0.5]

        input_data = torch.randn(*shape, dtype=torch.float32)
        input_data[0] = 0.6  # make sure the result is not an empty tensor
        model = IndexModel()
        self.run_compare_torch(
            input_data,
            model,
            backend=backend,
            compute_unit=compute_unit,
            input_as_shape=False,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 2),
                (3, 4, 5, 6),
            ]
        ),
    )
    def test_index_int_index_case_1(self, compute_unit, backend, shape):
        # all elements are selected
        class IndexModel(torch.nn.Module):
            def forward(self, x):
                if len(shape) == 2:
                    return x[:, :]
                elif len(shape) == 4:
                    return x[:]

        model = IndexModel()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 2),
                (3, 4, 5, 6),
            ]
        ),
    )
    def test_index_int_index_case_2(self, compute_unit, backend, shape):
        # only one axis is sliced
        class IndexModel(torch.nn.Module):
            def forward(self, x):
                if len(shape) == 2:
                    index = torch.tensor([0])
                    return x[index, :]
                elif len(shape) == 4:
                    index = torch.tensor([1, 2])
                    return x[:, :, index]

        model = IndexModel()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_3(self, compute_unit, backend, shape):
        # only two axes are sliced, and connected
        class IndexModel(torch.nn.Module):
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
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_4(self, compute_unit, backend, shape):
        # only two axes are sliced, and not connected
        class IndexModel(torch.nn.Module):
            def forward(self, x):
                if len(shape) == 3:
                    index_1 = torch.tensor([0])
                    index_2 = torch.tensor([1])
                    return x[index_1, :, index_2]

                elif len(shape) == 4:
                    index_1 = torch.tensor([0, 1, 1])
                    index_2 = torch.tensor([3, 3, 4])
                    return x[index_1, :, :, index_2]

        model = IndexModel()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_5(self, compute_unit, backend, shape):
        # all axes are sliced
        class IndexModel(torch.nn.Module):
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
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 2),
                (3, 4, 5, 6),
            ]
        ),
    )
    def test_index_int_index_case_6(self, compute_unit, backend, shape):
        # only one axis is sliced + nd mode
        class IndexModel(torch.nn.Module):
            def forward(self, x):
                if len(shape) == 2:
                    index = torch.tensor([0, 0, 0, 0, 0, 0])
                    index = index.view(2, 3)
                    return x[index, :]
                elif len(shape) == 4:
                    index = torch.tensor([0, 1, 2, 3, 0, 1])
                    index = index.view(3, 2)
                    return x[:, index]

        model = IndexModel()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_7(self, compute_unit, backend, shape):
        # two axes are sliced, and connected + nd mode
        class IndexModel(torch.nn.Module):
            def forward(self, x):
                if len(shape) == 3:
                    index_1 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]).view(4, 2)
                    index_2 = torch.tensor([1, 0, 0, 0, 1, 1, 1, 1]).view(4, 2)
                    return x[index_1, index_2, :]

                elif len(shape) == 4:
                    index_1 = torch.tensor([0, 0, 2, 2, 1, 1, 2, 0]).view(2, 4)
                    index_2 = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]).view(2, 4)
                    return x[:, index_1, index_2, :]

        model = IndexModel()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_8(self, compute_unit, backend, shape):
        # two axes are sliced, and not connected + nd mode
        class IndexModel(torch.nn.Module):
            def forward(self, x):
                if len(shape) == 3:
                    index_1 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]).view(2, 4)
                    index_2 = torch.tensor([1, 0, 0, 2, 2, 1, 1, 1]).view(2, 4)
                    return x[index_1, :, index_2]

                elif len(shape) == 4:
                    index_1 = torch.tensor([0, 1, 1, 1, 1, 1, 0, 0]).view(4, 2)
                    index_2 = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]).view(4, 2)
                    return x[index_1, :, :, index_2]

        model = IndexModel()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_9(self, compute_unit, backend, shape):
        # one axis is sliced through bool mask
        class IndexModel(torch.nn.Module):
            def forward(self, x):
                if len(shape) == 3:
                    return x[:, [True, False], :]

                elif len(shape) == 4:
                    return x[[True, False], :, :, :]

        model = IndexModel()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [
                (1, 2, 3),
                (2, 3, 4, 5),
            ]
        ),
    )
    def test_index_int_index_case_10(self, compute_unit, backend, shape):
        # multiple axes are sliced through bool masks
        class IndexModel(torch.nn.Module):
            def forward(self, x):
                if len(shape) == 3:
                    return x[[True], [True, False], [False, True, False]]

                elif len(shape) == 4:
                    return x[
                        [True, True],
                        :,
                        [True, True, False, False],
                        [True, False, False, True, False],
                    ]

        model = IndexModel()
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestPad(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, mode",
        itertools.product(
            compute_units,
            backends,
            range(3, 5),
            ['reflect', 'replicate']
        )
    )
    def test_pad_reflect_replicate(self, compute_unit, backend, rank: int, mode: str):
        if rank == 3:
            pad_len = 2
            input_shape = (5, 10, 10)
        elif rank == 4:
            pad_len = 4
            input_shape = (10, 5, 5, 10)
        else:
            raise NotImplementedError(
                "Only 3D, 4D padding with non-constant padding are supported for now"
            )
        max_pad = min(input_shape[-1], input_shape[-2])
        pad = list(np.random.randint(low=0, high=max_pad,
                                     size=pad_len))
        model = ModuleWrapper(function=torch.nn.functional.pad,
                              kwargs={"pad": pad, "mode": mode})
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank",
        itertools.product(
            compute_units,
            backends,
            range(1, 6)
        )
    )
    def test_pad_constant(self, compute_unit, backend, rank: int):
        if rank > 5:
            raise NotImplementedError("Only supports < 6D constant padding")
        val = float(np.random.random(1))
        input_shape = tuple(np.random.randint(low=1, high=10, size=rank))
        pad_dims = np.random.randint(low=1, high=rank + 1)
        pad = list(np.random.randint(low=0, high=10,
                                     size=pad_dims * 2))
        model = ModuleWrapper(function=torch.nn.functional.pad,
                              kwargs={"pad": pad, "mode": "constant", "value": val})
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_constant_pad_1d(self, compute_unit, backend):
        input_shape = (3, 4, 5)
        model = torch.nn.ConstantPad1d((5, 6), 3.5).eval()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_constant_pad_2d(self, compute_unit, backend):
        input_shape = (3, 4, 5, 6)
        model = torch.nn.ConstantPad2d((5, 6, 3, 8), 3.5).eval()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_constant_pad_3d(self, compute_unit, backend):
        input_shape = (3, 4, 5, 6, 2)
        model = torch.nn.ConstantPad3d((5, 6, 3, 8, 2, 4), 3.5).eval()
        self.run_compare_torch(
            input_shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestMeshgrid(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, rows, cols, dtype, inp_mode",
        itertools.product(
            compute_units,
            backends,
            [1, 2, 3],
            [1, 2, 3],
            [torch.int, torch.float],
            ["norm", "list"]
        ),
    )
    def test_meshgrid(
        self,
        compute_unit,
        backend,
        inp_mode,
        rows,
        cols,
        dtype,
    ):
        class TestModel(nn.Module):
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
            inputs,
            model,
            expected_results,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )


class TestScatter(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shapes_dims",
        itertools.product(
            compute_units,
            backends,
            [
                [(10,), (0, -1)],
                [(2, 3), (1, -1)],
                [(2, 3, 4, 5), (0, -2)],
            ],
        ),
    )
    def test_scatter(self, compute_unit, backend, shapes_dims):
        class TestModel(nn.Module):
            def __init__(self, dim, shapes):
                super(TestModel, self).__init__()
                self.dim = dim
                self.source = torch.rand(*(shapes))
                self.index = torch.randint(0, shapes[dim], size=shapes)

            def forward(self, x):
                return x.scatter_(self.dim, self.index, self.source)

        shapes, dims = shapes_dims
        for dim in dims:
            m = TestModel(0, shapes)
            self.run_compare_torch(
                shapes,
                m,
                backend=backend,
                compute_unit=compute_unit
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, shapes_dims",
        itertools.product(
            compute_units,
            backends,
            [
                [(10,), (0, -1)],
                [(2, 3), (1, -1)],
                [(2, 3, 4, 5), (0, -2)],
            ],
        ),
    )
    def test_scatter_with_scalar_source(self, compute_unit, backend, shapes_dims):
        class TestModel(nn.Module):
            def __init__(self, dim, shapes):
                super(TestModel, self).__init__()
                self.dim = dim
                self.source = 1.0
                self.index = torch.randint(0, shapes[dim], size=shapes)
                
            def forward(self, x):
                return x.scatter_(self.dim, self.index, self.source)

        shapes, dims = shapes_dims
        for dim in dims:
            m = TestModel(0, shapes)
            self.run_compare_torch(
                shapes,
                m,
                backend=backend,
                compute_unit=compute_unit
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, shapes_dims, mode",
        itertools.product(
            compute_units,
            backends,
            [
                [(10,), (0, -1)],
                [(2, 3), (1, -1)],
                [(2, 3, 4, 5), (0, -2)],
            ],
            ['add', 'multiply'],
        ),
    )
    def test_scatter_with_reduce(self, compute_unit, backend, shapes_dims, mode):
        class TestModel(nn.Module):
            def __init__(self, dim, shapes, mode):
                super(TestModel, self).__init__()
                self.dim = dim
                self.mode = mode
                self.source = torch.rand(*(shapes))
                self.index = torch.randint(0, shapes[dim], size=shapes)

            def forward(self, x):
                return x.scatter_(self.dim, self.index, self.source, reduce=self.mode)

        shapes, dims = shapes_dims
        for dim in dims:
            m = TestModel(0, shapes, mode)
            self.run_compare_torch(
                shapes,
                m,
                backend=backend,
                compute_unit=compute_unit
            )

    @pytest.mark.parametrize(
        "compute_unit, backend, shapes_dims",
        itertools.product(
            compute_units,
            backends,
            [
                [(10,), (0, -1)],
                [(2, 3), (1, -1)],
                [(2, 3, 4, 5), (0, -2)],
            ],
        ),
    )
    def test_scatter_add(self, compute_unit, backend, shapes_dims):
        class TestModel(nn.Module):
            def __init__(self, dim, shapes):
                super(TestModel, self).__init__()
                self.dim = dim
                self.source = torch.rand(*(shapes))
                self.index = torch.randint(0, shapes[dim], size=shapes)

            def forward(self, x):
                return x.scatter_add_(self.dim, self.index, self.source)

        shapes, dims = shapes_dims
        for dim in dims:
            m = TestModel(dim, shapes)
            self.run_compare_torch(
                shapes,
                m,
                backend=backend,
                compute_unit=compute_unit
            )


class TestBroadcastTensors(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shapes",
        itertools.product(
            compute_units,
            backends,
            [(1,), (1, 2)],
        ),
    )
    def test_one_tensor(self, compute_unit, backend, shapes):
        class TestModel(nn.Module):
            def forward(self, a):
                return torch.broadcast_tensors(a)

        self.run_compare_torch(
            shapes,
            TestModel().eval(),
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shapes",
        itertools.product(
            compute_units,
            backends,
            [
                [(2, 1), (1, 3)],
                [(5, 1, 4, 1), (3, 1, 1)],
                [(1,), (3, 1, 7)],
                [(2, 1), (4, 3, 2, 1)]
            ],
        ),
    )
    def test_two_tensors(self, compute_unit, backend, shapes):
        class TestModel(nn.Module):
            def forward(self, a, b):
                return torch.broadcast_tensors(a, b)

        self.run_compare_torch(
            shapes,
            TestModel().eval(),
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shapes",
        itertools.product(
            compute_units,
            backends,
            [
                [(2, 1), (1, 3), (1,), (1, 1)],
                [(5, 1, 4, 1), (3, 1, 1), (1,), (4, 8)],
                [(1,), (2, 1), (3, 2, 1), (5, 4, 3, 2, 1)],
            ],
        ),
    )
    def test_four_tensors(self, compute_unit, backend, shapes):
        class TestModel(nn.Module):
            def forward(self, a, b, c, d):
                return torch.broadcast_tensors(a, b, c, d)

        self.run_compare_torch(
            shapes,
            TestModel().eval(),
            backend=backend,
            compute_unit=compute_unit
        )


class TestEmbedding(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, input_dtype",
        itertools.product(
            compute_units,
            backends,
            [np.int32, np.float32],
        )
    )
    def test_embedding(self, compute_unit, backend, input_dtype):
        num_embeddings = 4
        embedding_size = 10
        B = 2
        dim = 5
        converter_input_type = [TensorType(shape=(B, dim), dtype=input_dtype)]

        # input shape: (B, dim)
        # output shape : (B, dim, embedding_size)
        # shape of weights : (num_embeddings, embedding_size)
        class EmbeddingModel(nn.Module):
            def __init__(self):
                super(EmbeddingModel, self).__init__()
                self.embedding = torch.nn.Embedding(num_embeddings, embedding_size)

            def forward(self, x):
                return self.embedding(x)

        input_data = np.random.randint(low=0, high=num_embeddings, size=(B, dim))
        input_data = torch.from_numpy(input_data)
        model = EmbeddingModel()
        expected_results = model(input_data)
        self.run_compare_torch(
            input_data,
            model,
            expected_results=expected_results,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit,
            converter_input_type=converter_input_type,
        )


class TestDuplicateOutputTensors(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, input_dtype",
        itertools.product(
            compute_units,
            backends,
            [np.int32, np.float32],
        )
    )
    # Test case for rdar://100138064 (Duplicate output tensors trigger ops removal errors).
    def test_duplicate_output_not_raise_errors(self, compute_unit, backend, input_dtype):
        if backend[0] == "neuralnetwork":
            pytest.skip(
                "rdar://100243127 ([PyTorch] Duplicate Output Tensor Doesn't work for neuralnetwork)"
            )

        class DuplicateTensorsModel(torch.nn.Module):
            def forward(self, x):
                return x, x

        input_data = torch.rand(2, 2, 1, 1)
        converter_input_type = [ct.TensorType(shape=input_data.shape)]
        model = DuplicateTensorsModel()
        expected_results = model(input_data)
        self.run_compare_torch(
            input_data,
            model,
            expected_results=expected_results,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit,
            converter_input_type=converter_input_type,
        )


class TestBaddbmm(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shapes",
        itertools.product(
            compute_units,
            backends,
            [(2, 4, 6, 8), (4, 12, 6, 16)],
        )
    )
    def test_baddbmm(self, compute_unit, backend, shapes):
        B, N, M, P = shapes

        # input shape: any shape broadcastable to (B, N, P)
        # batch1 shape: (B, N, M)
        # batch2 shape: (B, M, P)
        # output shape : (B, N, P)
        class BaddbmmModel(nn.Module):
            def __init__(self):
                super(BaddbmmModel, self).__init__()
                self.batch1 = torch.randn(B, N, M)
                self.batch2 = torch.randn(B, M, P)

            def forward(self, x):
                return torch.baddbmm(x, self.batch1, self.batch2)

        model = BaddbmmModel()
        # Makes it broadcastable to (B, N, P).
        for input_shape in [(1, N, P), (B, 1, P), (1, P)]:
            self.run_compare_torch(
                input_shape,
                model,
                backend=backend,
                compute_unit=compute_unit
            )


class TestGlu(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shapes",
        itertools.product(
            compute_units,
            backends,
            [(2, 4, 6, 8), (6, 2, 10)],
        )
    )
    def test_glu(self, compute_unit, backend, shapes):
        # The dim specified for GLU shouldn't exceed the max dim in input.
        glu_dim_list = [-1] + [i for i in range(len(shapes))]
        for glu_dim in glu_dim_list:
            model = torch.nn.GLU(glu_dim)
            self.run_compare_torch(
                shapes,
                model,
                backend=backend,
                compute_unit=compute_unit
            )


class TestHstack(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shapes",
        itertools.product(
            compute_units,
            backends,
            [[(2, 4, 6), (2, 4, 6)],
             [(1, 4, 5), (1, 2, 5)],
             [(1,), (3,)]],  # Test 1-D tensors.
        )
    )
    def test_hstack(self, compute_unit, backend, shapes):
        class HstackModel(nn.Module):
            def forward(self, *tensors):
                return torch.hstack(tensors)

        self.run_compare_torch(
            shapes,
            HstackModel(),
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shapes",
        itertools.product(
            compute_units,
            backends,
            [[(2, 4, 6), (2, 4, 6)]],
        )
    )
    def test_hstack_with_parameter_out(self, compute_unit, backend, shapes):
        class HstackModel(nn.Module):
            def forward(self, *tensors):
                output_tensor = torch.tensor([])
                torch.hstack(tensors, out=output_tensor)
                return output_tensor

        self.run_compare_torch(
            shapes,
            HstackModel(),
            backend=backend,
            compute_unit=compute_unit
        )


class TestRemainder(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shapes",
        itertools.product(
            compute_units,
            backends,
            [[(2, 4, 6), (2, 4, 6)],
             [(2, 4, 6), (4, 6)],  # broadcastable tensors
             [(2, 4, 6), (2, 1, 6)]],
        )
    )
    def test_remainder(self, compute_unit, backend, shapes):
        class RemainderModel(nn.Module):
            def forward(self, dividend, divisor):
                return torch.remainder(dividend, divisor)

        self.run_compare_torch(
            shapes,
            RemainderModel(),
            backend=backend,
            compute_unit=compute_unit
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shapes",
        itertools.product(
            compute_units,
            backends,
            [[(2, 4, 6), (2, 4, 6)]],
        )
    )
    def test_remainder_with_parameter_out(self, compute_unit, backend, shapes):
        class RemainderModel(nn.Module):
            def forward(self, dividend, divisor):
                output_tensor = torch.tensor([])
                torch.remainder(dividend, divisor, out=output_tensor)
                return output_tensor

        self.run_compare_torch(
            shapes,
            RemainderModel(),
            backend=backend,
            compute_unit=compute_unit
        )


class TestSum(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, input_dtype",
        itertools.product(
            compute_units,
            backends,
            [torch.int32, torch.float32, torch.bool]
        )
    )
    def test_sum(self, compute_unit, backend, input_dtype):
        model = ModuleWrapper(function=torch.sum)

        input_data = torch.zeros(2, 3).to(input_dtype)
        expected_results = model(input_data)

        TorchBaseTest.run_compare_torch(
            input_data,
            model,
            expected_results=expected_results,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )


class TestHannWindow(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, window_length, periodic",
        itertools.product(
            compute_units,
            backends,
            [1, 3, 6, 10, 12],
            [True, False],
        ),
    )
    def test_hann_window(self, compute_unit, backend, window_length, periodic):
        class HannWindowModel(nn.Module):
            def forward(self, x):
                return torch.hann_window(window_length, periodic)

        input_shape = np.random.randint(low=1, high=10, size=(window_length,))
        torch_in = torch.tensor(input_shape, dtype=torch.int32)
        model = HannWindowModel().eval()
        torch_out = model(torch_in)
        self.run_compare_torch(
            torch_in,
            model,
            expected_results=torch_out,
            input_as_shape=False,
            backend=backend,
            compute_unit=compute_unit
        )
            

class TestTrace(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [(1, 1), (2, 4), (4, 3), (5, 5)],
        ),
    )
    def test_trace(self, compute_unit, backend, shape):
        model = ModuleWrapper(torch.trace)
        self.run_compare_torch(
            shape,
            model,
            backend=backend,
            compute_unit=compute_unit
        )


class TestArgmax(TorchBaseTest):
    @pytest.mark.parametrize(
        "compute_unit, backend, shape, axis, input_dtype",
        itertools.product(
            compute_units,
            backends,
            COMMON_SHAPES,
            [-1, 0],
            [np.float32, np.int32, np.int64],
        ),
    )
    def test_argmax(
        self,
        compute_unit,
        backend: Tuple[str, str],
        shape: Tuple[int],
        axis: int,
        input_dtype: np.dtype,
    ):
        input_data = (
            torch.rand(*shape)
            if input_dtype == np.float32
            else torch.randint(10, shape)
        )
        converter_input_type = [
            ct.TensorType(shape=input_data.shape, dtype=input_dtype)
        ]
        model = ModuleWrapper(function=torch.argmax, kwargs={"dim": axis})
        expected_results = model(input_data)
        TorchBaseTest.run_compare_torch(
            input_data,
            model,
            expected_results=expected_results,
            input_as_shape=False,
            backend=backend,
            converter_input_type=converter_input_type,
            compute_unit=compute_unit
        )

