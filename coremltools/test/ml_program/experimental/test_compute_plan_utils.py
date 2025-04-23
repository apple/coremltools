#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Optional

import pytest

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.models.compute_device import (
    MLComputeDevice,
    MLCPUComputeDevice,
    MLGPUComputeDevice,
    MLNeuralEngineComputeDevice,
)
from coremltools.models.compute_plan import (
    MLComputePlan,
    MLModelStructure,
    MLModelStructureProgramOperation,
)
from coremltools.models.ml_program.experimental.compute_plan_utils import (
    _MLComputePlanRemoteProxy,
)
from coremltools.models.ml_program.experimental.model_structure_path import (
    ModelStructurePath,
    map_model_structure_to_path,
)
from coremltools.models.ml_program.experimental.remote_device import ComputePlan


@pytest.mark.skipif(
    ct.utils._macos_version() < (14, 4),
    reason="MLComputePlan API is available for macos versions >= 14.4.",
)
class TestComputePlanUtils:
    @staticmethod
    def get_simple_program():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, 100)),
                mb.TensorSpec(shape=(1, 100)),
            ]
        )
        def prog(x, y):
            a = mb.add(x=x, y=y, name="add")
            b = mb.sub(x=x, y=y, name="sub")
            return a, b

        return prog

    @staticmethod
    def test_model_structure_path_neural_network():
        prog = TestComputePlanUtils.get_simple_program()
        model = ct.convert(prog, convert_to="neuralnetwork")
        model_structure = MLModelStructure.load_from_path(
            compiled_model_path=model.get_compiled_model_path()
        )
        structure_and_paths = map_model_structure_to_path(
            model_structure=model_structure,
        )
        for _, path in structure_and_paths:
            assert isinstance(
                path.components[0], ModelStructurePath.NeuralNetwork
            ), "The first component of the path must be a NeuralNetwork"
            assert isinstance(
                path.components[1], ModelStructurePath.NeuralNetwork.Layer
            ), "The second component of the path must be a NeuralNetwork.Layer"

    @staticmethod
    def test_model_structure_path_program():
        prog = TestComputePlanUtils.get_simple_program()
        model = ct.convert(prog, convert_to="mlprogram")
        model_structure = MLModelStructure.load_from_path(
            compiled_model_path=model.get_compiled_model_path()
        )
        structure_and_paths = map_model_structure_to_path(
            model_structure=model_structure,
        )
        for _, path in structure_and_paths:
            assert isinstance(
                path.components[0], ModelStructurePath.Program
            ), "The first component of the path must be a Program"
            assert isinstance(
                path.components[1], ModelStructurePath.Program.Function
            ), "The second component of the path must be a Program.Function"
            assert isinstance(
                path.components[2], ModelStructurePath.Program.Block
            ), "The third component of the path must be a Program.Block"
            assert isinstance(
                path.components[3], ModelStructurePath.Program.Operation
            ), "The fourth component of the path must be a Program.Operation"

    @staticmethod
    def test_model_structure_path_serde():
        prog = TestComputePlanUtils.get_simple_program()
        model = ct.convert(prog, convert_to="mlprogram")
        model_structure = MLModelStructure.load_from_path(
            compiled_model_path=model.get_compiled_model_path()
        )
        structure_and_paths = map_model_structure_to_path(
            model_structure=model_structure,
            components=[],
        )

        for _, path in structure_and_paths:
            dict = path.to_dict()
            new_path = ModelStructurePath.from_dict(dict)
            assert (
                path == new_path
            ), f"The path created from the serialized dict {dict} must be equal to the original path {path}"

    @staticmethod
    def to_device(ml_compute_device: MLComputePlan) -> ComputePlan.Device:
        if isinstance(ml_compute_device, MLCPUComputeDevice):
            return ComputePlan.CPUDevice()
        elif isinstance(ml_compute_device, MLGPUComputeDevice):
            return ComputePlan.GPUDevice()
        elif isinstance(ml_compute_device, MLNeuralEngineComputeDevice):
            return ComputePlan.NeuralEngineDevice(
                total_core_count=ml_compute_device.total_core_count,
            )
        else:
            raise ValueError(f"Unknown compute device : {ml_compute_device}")

    @staticmethod
    def to_compute_plan_info(
        operation: MLModelStructureProgramOperation,
        ml_compute_plan: MLComputePlan,
        path: ModelStructurePath,
    ) -> Optional[ComputePlan.OperationOrLayerInfo]:
        estimated_cost = ml_compute_plan.get_estimated_cost_for_mlprogram_operation(operation)
        compute_device_usage = ml_compute_plan.get_compute_device_usage_for_mlprogram_operation(
            operation
        )
        if compute_device_usage is None:
            return None

        preferred = TestComputePlanUtils.to_device(compute_device_usage.preferred_compute_device)
        supported = [
            TestComputePlanUtils.to_device(compute_device)
            for compute_device in compute_device_usage.supported_compute_devices
        ]
        device_usage = ComputePlan.DeviceUsage(
            preferred=preferred,
            supported=supported,
        )

        return ComputePlan.OperationOrLayerInfo(
            device_usage=device_usage,
            estimated_cost=estimated_cost.weight if estimated_cost is not None else None,
            path=path,
        )

    @staticmethod
    def test_remote_proxy():
        prog = TestComputePlanUtils.get_simple_program()
        model = ct.convert(prog, convert_to="mlprogram")
        ml_compute_plan = MLComputePlan.load_from_path(path=model.get_compiled_model_path())
        model_structure = ml_compute_plan.model_structure
        structure_and_paths = map_model_structure_to_path(
            model_structure=model_structure,
        )

        main_function = model_structure.program.functions["main"]
        assert (
            len(main_function.block.operations) >= 2
        ), "The main function should contain at least 2 operations"

        infos = {}
        for structure, path in structure_and_paths:
            info = TestComputePlanUtils.to_compute_plan_info(
                operation=structure,
                ml_compute_plan=ml_compute_plan,
                path=path,
            )

            if info is not None:
                infos[path] = info

        remote_proxy = _MLComputePlanRemoteProxy(
            device=None,
            compute_plan=ComputePlan(infos=infos),
            model_structure=model_structure,
        )

        for operation in main_function.block.operations:
            if operation.operator_name == "const":
                continue

            compute_device_usage = remote_proxy.get_compute_device_usage_for_mlprogram_operation(
                operation=operation
            )

            assert compute_device_usage is not None
            assert isinstance(compute_device_usage.preferred_compute_device, MLComputeDevice)
            for compute_device in compute_device_usage.supported_compute_devices:
                assert isinstance(compute_device, MLComputeDevice)
