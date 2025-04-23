#  Copyright (c) 2025, Apple Inc. All rights reserved.


from typing import Optional

import numpy as np
import pytest
import torch

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.models.ml_program.experimental.perf_utils import MLModelBenchmarker
from coremltools.models.ml_program.experimental.torch.perf_utils import (
    TorchMLModelBenchmarker,
    TorchNode,
    TorchScriptNodeInfo,
)


@pytest.mark.skipif(
    ct.utils._macos_version() < (14, 4),
    reason="MLModelBenchmarker API is available for macos versions >= 14.4.",
)
class TestMLModelBenchmarker:
    def get_test_model():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 2, 3, 4)),
            ]
        )
        def prog(x):
            y = mb.const(val=1.2, name="y")
            x = mb.add(x=x, y=y, name="add")
            perm = mb.const(val=[0, 2, 3, 1], name="perm")
            x = mb.transpose(x=x, perm=perm, name="transpose_1")
            x = mb.square(x=x, name="output_0")
            x = mb.tanh(x=x, name="output_1")
            x = mb.transpose(x=x, perm=perm, name="transpose_2")
            return x

        return prog

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("compute_units", [compute_unit for compute_unit in ct.ComputeUnit])
    async def test_benchmark_load(
        compute_units: ct.ComputeUnit,
    ):
        prog = TestMLModelBenchmarker.get_test_model()
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)
        benchmarker = MLModelBenchmarker(model=mlmodel)

        measurement = await benchmarker.benchmark_load(iterations=3)

        assert len(measurement.samples) == 3
        assert measurement.statistics is not None
        assert measurement.statistics.average > 1e-2

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("compute_units", [compute_unit for compute_unit in ct.ComputeUnit])
    @pytest.mark.parametrize("warmup", [True, False])
    async def test_benchmark_predict(
        compute_units: ct.ComputeUnit,
        warmup: bool,
    ):
        prog = TestMLModelBenchmarker.get_test_model()
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)
        benchmarker = MLModelBenchmarker(model=mlmodel)
        measurement = await benchmarker.benchmark_predict(
            iterations=3,
            warmup=warmup,
        )

        assert len(measurement.samples) == 3
        assert measurement.statistics is not None
        assert measurement.statistics.average > 1e-2

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("compute_units", [compute_unit for compute_unit in ct.ComputeUnit])
    @pytest.mark.parametrize("warmup", [True, False])
    async def test_benchmark_operation_execution(
        compute_units: ct.ComputeUnit,
        warmup: bool,
    ):
        prog = TestMLModelBenchmarker.get_test_model()
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)
        benchmarker = MLModelBenchmarker(model=mlmodel)
        execution_infos = await benchmarker.benchmark_operation_execution(
            iterations=3,
            warmup=warmup,
        )

        for execution_info in execution_infos:
            op_type = execution_info.spec.type
            if op_type == "const" or op_type == "cast":
                continue

            measurement = execution_info.measurement
            assert len(measurement.samples) == 3
            assert measurement.statistics is not None
            assert measurement.statistics.average > 1e-6


@pytest.mark.skipif(
    ct.utils._macos_version() < (14, 4),
    reason="TorchMLModelBenchmarker uses MLComputePlan API which is available for macos versions >= 14.4",
)
class TestTorchMLModelBenchmarker:
    def get_test_model():
        class Mul(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        class Sub(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        class Add(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.add = Add()
                self.sub = Sub()
                self.mul = Mul()

            def forward(self, x, y):
                a = self.add(x, y)
                b = self.sub(x, y)
                c = self.mul(x, y)
                return (a, b, c)

        model = Model()
        model.eval()
        return model

    @staticmethod
    def get_node_kind(node: TorchNode) -> Optional[str]:
        if isinstance(node, TorchScriptNodeInfo):
            return node.kind

        elif isinstance(node, torch.fx.Node):
            return str(node.target)

        else:
            return None

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("compute_units", [compute_unit for compute_unit in ct.ComputeUnit])
    @pytest.mark.parametrize("warmup", [True, False])
    @pytest.mark.parametrize("use_torch_export", [True, False])
    async def test_benchmark_node_execution(
        compute_units: ct.ComputeUnit,
        warmup: bool,
        use_torch_export: bool,
    ):
        test_model = TestTorchMLModelBenchmarker.get_test_model()
        inputs = (
            torch.full((1, 10), 1, dtype=torch.float),
            torch.full((1, 10), 2, dtype=torch.float),
        )
        torch_model = (
            torch.export.export(test_model, inputs)
            if use_torch_export
            else torch.jit.trace(test_model, inputs)
        )

        benchmarker = TorchMLModelBenchmarker(
            model=torch_model,
            inputs=[
                ct.TensorType(name="x", shape=inputs[0].shape, dtype=np.float16),
                ct.TensorType(name="y", shape=inputs[1].shape, dtype=np.float16),
            ],
            minimum_deployment_target=ct.target.iOS16,
            compute_units=compute_units,
        )

        execution_infos = await benchmarker.benchmark_node_execution(iterations=3, warmup=warmup)

        node_types = (
            ["aten::sub", "aten::add", "aten::mul"]
            if use_torch_export
            else ["aten.sub.Tensor", "aten.add.Tensor", "aten.mul.Tensor"]
        )
        for execution_info in execution_infos:
            if TestTorchMLModelBenchmarker.get_node_kind(execution_info.node) not in node_types:
                continue

            measurement = execution_info.measurement
            assert len(measurement.samples) == 3
            assert measurement.statistics is not None
            assert measurement.statistics.average > 1e-6

    @staticmethod
    @pytest.mark.asyncio
    @pytest.mark.parametrize("compute_units", [compute_unit for compute_unit in ct.ComputeUnit])
    @pytest.mark.parametrize("warmup", [True, False])
    @pytest.mark.parametrize("use_torch_export", [True, False])
    async def test_benchmark_module_execution(
        compute_units: ct.ComputeUnit,
        warmup: bool,
        use_torch_export: bool,
    ):
        test_model = TestTorchMLModelBenchmarker.get_test_model()
        inputs = (
            torch.full((1, 10), 1, dtype=torch.float),
            torch.full((1, 10), 2, dtype=torch.float),
        )
        torch_model = (
            torch.export.export(test_model, inputs)
            if use_torch_export
            else torch.jit.trace(test_model, inputs)
        )

        benchmarker = TorchMLModelBenchmarker(
            model=torch_model,
            inputs=[
                ct.TensorType(name="x", shape=inputs[0].shape, dtype=np.float16),
                ct.TensorType(name="y", shape=inputs[1].shape, dtype=np.float16),
            ],
            minimum_deployment_target=ct.target.iOS16,
            compute_units=compute_units,
        )

        execution_infos = await benchmarker.benchmark_module_execution(
            iterations=3,
            warmup=warmup,
        )

        module_names = ["", "add", "sub", "mul"]
        for execution_info in execution_infos:
            assert execution_info.name in module_names

            measurement = execution_info.measurement
            assert len(measurement.samples) == 3
            assert measurement.statistics is not None
            assert measurement.statistics.average > 1e-6
