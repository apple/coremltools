# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
from typing import Callable, List, Optional

from coremltools import ComputeUnit, _logger, proto
from coremltools._deps import _IS_MACOS

from ...compute_device import (
    MLComputeDevice,
    MLCPUComputeDevice,
    MLGPUComputeDevice,
    MLNeuralEngineComputeDevice,
)
from ...compute_plan import (
    MLComputePlan,
    MLComputePlanCost,
    MLComputePlanDeviceUsage,
    MLModelStructure,
    MLModelStructureNeuralNetworkLayer,
    MLModelStructureProgramOperation,
)
from ...model import MLModel
from .model_structure_path import map_model_spec_to_path, map_model_structure_to_path
from .remote_device import ComputePlan, Device, _RemoteMLModelService

try:
    from coremltools.libcoremlpython import _MLComputePlanProxy
except Exception as e:
    if _IS_MACOS:
        _logger.warning(f"Failed to load _MLComputePlanProxy: {e}")
    _MLComputePlanProxy = None

try:
    from coremltools.libcoremlpython import _MLCPUComputeDeviceProxy
except Exception as e:
    if _IS_MACOS:
        _logger.warning(f"Failed to load _MLCPUComputeDeviceProxy: {e}")
    _MLCPUComputeDeviceProxy = None

try:
    from coremltools.libcoremlpython import _MLGPUComputeDeviceProxy
except Exception as e:
    if _IS_MACOS:
        _logger.warning(f"Failed to load _MLGPUComputeDeviceProxy: {e}")
    _MLGPUComputeDeviceProxy = None

try:
    from coremltools.libcoremlpython import _MLNeuralEngineComputeDeviceProxy
except Exception as e:
    if _IS_MACOS:
        _logger.warning(f"Failed to load _MLNeuralEngineComputeDeviceProxy: {e}")
    _MLNeuralEngineComputeDeviceProxy = None


if _MLCPUComputeDeviceProxy is not None:
    class _MLCPUComputeDeviceRemoteProxy(_MLCPUComputeDeviceProxy):
        def __init__(
            self,
            device: Device,
        ):
            _MLCPUComputeDeviceProxy.__init__(self)
            self.device = device

else:
    if _IS_MACOS:
        _logger.warning(
            "Failed to load '_MLCPUComputeDeviceRemoteProxy'. Remote device functionality for retrieving the compute plan is unavailable."
        )
    _MLCPUComputeDeviceRemoteProxy = None


if _MLGPUComputeDeviceProxy is not None:

    class _MLGPUComputeDeviceRemoteProxy(_MLGPUComputeDeviceProxy):
        def __init__(
            self,
            device: Device,
        ):
            _MLGPUComputeDeviceProxy.__init__(self)
            self.device = device

else:
    if _IS_MACOS:
        _logger.warning(
            "Failed to load '_MLGPUComputeDeviceRemoteProxy'. Remote device functionality for retrieving the compute plan is unavailable."
        )
    _MLGPUComputeDeviceRemoteProxy = None


if _MLNeuralEngineComputeDeviceProxy is not None:

    class _MLNeuralEngineComputeDeviceRemoteProxy(_MLNeuralEngineComputeDeviceProxy):
        def __init__(
            self,
            total_core_count: int,
            device: Device,
        ):
            _MLNeuralEngineComputeDeviceProxy.__init__(self)
            self.device = device
            self._total_core_count = total_core_count

        @property
        def total_core_count(self):
            return self._total_core_count

else:
    if _IS_MACOS:
        _logger.warning(
            "Failed to load '_MLNeuralEngineComputeDeviceRemoteProxy'. Remote device functionality for retrieving the compute plan is unavailable."
        )
    _MLNeuralEngineComputeDeviceRemoteProxy = None


if _MLComputePlanProxy is not None:

    class _MLComputePlanRemoteProxy(_MLComputePlanProxy):
        def __init__(
            self,
            device: Device,
            compute_plan: ComputePlan,
            model_structure: MLModelStructure,
        ):
            _MLComputePlanProxy.__init__(self)
            self.device = device
            self._compute_plan = compute_plan
            self._model_structure = model_structure
            structure_and_paths = map_model_structure_to_path(
                model_structure=model_structure,
                components=[],
            )
            self._model_path_map = {id(structure): path for structure, path in structure_and_paths}
            self._compute_devices = {}

        @property
        def model_structure(self):
            return copy.copy(self._model_structure)

        def _get_compute_plan_info(
            self,
            identifier: int,
        ) -> Optional[ComputePlan.OperationOrLayerInfo]:
            path = self._model_path_map.get(identifier, None)
            if path is None:
                return None

            return self._compute_plan.infos.get(path, None)

        def _to_compute_device_usage(
            self,
            device_usage: ComputePlan.DeviceUsage,
        ) -> MLComputePlanDeviceUsage:
            def _to_compute_device(
                compute_plan_device: ComputePlan.Device,
            ) -> MLComputeDevice:
                compute_device = self._compute_devices.get(compute_plan_device, None)
                if compute_device is None:
                    if isinstance(compute_plan_device, ComputePlan.CPUDevice):
                        self._compute_devices[compute_plan_device] = MLCPUComputeDevice(
                            proxy=_MLCPUComputeDeviceRemoteProxy(device=self.device)
                        )
                    elif isinstance(compute_plan_device, ComputePlan.GPUDevice):
                        self._compute_devices[compute_plan_device] = MLGPUComputeDevice(
                            proxy=_MLGPUComputeDeviceRemoteProxy(device=self.device)
                        )
                    elif isinstance(compute_plan_device, ComputePlan.NeuralEngineDevice):
                        self._compute_devices[compute_plan_device] = MLNeuralEngineComputeDevice(
                            proxy=_MLNeuralEngineComputeDeviceRemoteProxy(
                                device=self.device,
                                total_core_count=compute_plan_device.total_core_count,
                            )
                        )
                    else:
                        raise ValueError(f"Unknown compute device={compute_plan_device}")

                return self._compute_devices.get(compute_plan_device)

            preferred_compute_device = _to_compute_device(device_usage.preferred)
            supported_compute_devices = [
                _to_compute_device(compute_plan_device)
                for compute_plan_device in device_usage.supported
            ]
            return MLComputePlanDeviceUsage(
                preferred_compute_device=preferred_compute_device,
                supported_compute_devices=supported_compute_devices,
            )

        def get_compute_device_usage_for_mlprogram_operation(
            self,
            operation: MLModelStructureProgramOperation,
        ) -> Optional[MLComputePlanDeviceUsage]:
            info = self._get_compute_plan_info(id(operation))
            if info is None:
                return None

            return self._to_compute_device_usage(device_usage=info.device_usage)

        def get_compute_device_usage_for_neuralnetwork_layer(
            self,
            layer: MLModelStructureNeuralNetworkLayer,
        ) -> Optional[MLComputePlanDeviceUsage]:
            info = self._get_compute_plan_info(id(layer))
            if info is None:
                return None

            return self._to_compute_device_usage(device_usage=info.device_usage)

        def get_estimated_cost_for_mlprogram_operation(
            self,
            operation: MLModelStructureProgramOperation,
        ) -> Optional[MLComputePlanCost]:
            info = self._get_compute_plan_info(id(operation))
            if info is None or info.estimated_cost is None:
                return None

            return MLComputePlanCost(weight=info.estimated_cost)

else:
    if _IS_MACOS:
        _logger.warning(
            """
            Failed to load '_MLComputePlanRemoteProxy'.
            Remote device functionality for retrieving the compute plan is unavailable.
            """
        )
    _MLComputePlanRemoteProxy = None


async def load_compute_plan_from_path_on_device(
    path: str,
    compute_units: ComputeUnit = ComputeUnit.ALL,
    device: Optional[Device] = None,
) -> MLComputePlan:
    """
    Loads the compute plan of a compiled model on a remote or local device.

    The path must be the location of the ``mlmodelc`` directory.

    Parameters
    ----------
    path : str
        The path to the compiled model.

    Returns
    -------
    The plan for executing the model.

    Examples
    --------
    .. sourcecode:: python
        # Retrieve a development device.
        devices = Device.get_connected_development_devices(device_type=DeviceType.IPHONE)
        device = devices[0]
        # Prepare device for model debugging.
        device = await device.prepare_for_model_debugging()
        compute_plan = await coremltools.models.ml_program.experimental.compute_plan_utils.load_compute_plan_from_path_on_device(
            path=model.get_compiled_path(),
            device=device,
        )

        if compute_plan.model_structure.program is None:
            raise ValueError("Unexpected model type.")

        program = compute_plan.model_structure.program
        mainFunction = program.functions["main"]
        for operation in mainFunction.block.operations:
            # Get the compute device usage for the operation.
            compute_device_usage = (
                compute_plan.get_compute_device_usage_for_mlprogram_operation(operation)
            )
            # Get the estimated cost of executing the operation.
            estimated_cost = compute_plan.get_estimated_cost_for_mlprogram_operation(operation)

    """
    if device is None:
        return MLComputePlan.load_from_path(
            path=path,
            compute_units=compute_units,
        )
    else:
        if _MLComputePlanRemoteProxy is None:
            raise ValueError(
                "Remote device functionality for retrieving the compute plan is unavailable."
            )

        service = await _RemoteMLModelService.load_on_device(
            device=device,
            compiled_model_path=path,
            compute_units=compute_units,
            function_name=None,
        )

        compute_plan = await service.retrieve_compute_plan()
        model_structure = MLModelStructure.load_from_path(
            compiled_model_path=path,
        )
        remote_proxy = _MLComputePlanRemoteProxy(
            device=device,
            compute_plan=compute_plan,
            model_structure=model_structure,
        )

        return MLComputePlan(proxy=remote_proxy)


MLComputePlan.load_from_path_on_device = load_compute_plan_from_path_on_device


def _create_list_proto_value(elements: List[str]) -> proto.MIL_pb2.Value:
    list_type = proto.MIL_pb2.ListType()
    list_type.type.tensorType.dataType = proto.MIL_pb2.DataType.STRING
    dimension = list_type.length
    dimension.constant.size = len(elements)

    str_type = proto.MIL_pb2.ValueType()
    str_type.tensorType.dataType = proto.MIL_pb2.DataType.STRING

    value = proto.MIL_pb2.Value()
    value.type.listType.CopyFrom(list_type)
    list_value = value.immediateValue.list
    for element in elements:
        element_value = list_value.values.add()
        element_value.type.CopyFrom(str_type)
        element_value.immediateValue.tensor.strings.values.append(element)

    return value


def set_intended_backends_attr(
    op: proto.MIL_pb2.Operation,
    backend_names: Optional[List[str]],
):
    if backend_names is None:
        op.attributes.pop("IntendedBackend", None)
    else:
        backed_attr = op.attributes["IntendedBackend"]
        backed_attr.CopyFrom(_create_list_proto_value(elements=backend_names))


def set_intended_backends(
    model: MLModel,
    backend_assignment_fn: Callable[[proto.MIL_pb2.Operation], Optional[List[str]]],
) -> MLModel:
    """
    Assigns intended backends to operations in the given model.

    This function creates a new MLModel with updated backend assignments for each operation.
    It traverses the entire model structure, applying the provided backend assignment function
    to determine the intended backends for each operation.

    Parameters
    ----------
    model : MLModel
        The input model to be processed.

    backend_assignment_fn : Callable[[proto.MIL_pb2.Operation], Optional[List[str]]]
        A function that takes an operation and returns a list of intended backend names.

    Returns
    -------
    MLModel
        A new model instance with updated backend assignments.
    """

    def clone_spec(
        spec: "proto.Model_pb2.Model",
    ) -> "proto.Model_pb2.Model":
        spec_class = spec.__class__
        new_spec = spec_class()
        new_spec.CopyFrom(spec)
        return new_spec

    def set_block_intended_backends(
        block_spec: proto.MIL_pb2.Block,
    ) -> None:
        for operation_spec in block_spec.operations:
            backend_names = backend_assignment_fn(operation_spec)
            set_intended_backends_attr(
                op=operation_spec,
                backend_names=backend_names,
            )
            for operation_block_spec in operation_spec.blocks:
                set_block_intended_backends(block_spec=operation_block_spec)

    model_spec = clone_spec(model.get_spec())

    if model_spec.WhichOneof("Type") != "mlProgram":
        raise ValueError("set_intended_backends only supports ML Program.")

    for function_spec in model_spec.mlProgram.functions.values():
        block_spec = function_spec.block_specializations.get(function_spec.opset, None)
        if block_spec is None:
            raise ValueError(
                f"Invalid spec, missing block specialization for opset : {function_spec.opset}"
            )

        set_block_intended_backends(block_spec=block_spec)

    return MLModel(
        model=model_spec,
        weights_dir=model.weights_dir,
        function_name=model.function_name,
        compute_units=model.compute_unit,
    )


def _get_backend_names(
    compute_device: MLComputeDevice,
) -> List[str]:
    if isinstance(compute_device, MLGPUComputeDevice):
        return ["mps_graph"]
    elif isinstance(compute_device, MLNeuralEngineComputeDevice):
        return ["ane"]
    elif isinstance(compute_device, MLCPUComputeDevice):
        # The same is set by the Core ML framework.
        return ["bnns", "classic_cpu", "e5_minimal_cpu"]
    else:
        raise ValueError(f"Unknown compute device = {compute_device}")


def _assign_ids_to_program_operations(
    program_spec: proto.MIL_pb2.Program,
    id_generator: Callable[[proto.MIL_pb2.Operation], str],
    attr_name: str,
) -> None:
    def assign_ids_to_block_operations(
        block_spec: proto.MIL_pb2.Block,
    ) -> None:
        for operation_spec in block_spec.operations:
            attr = operation_spec.attributes[attr_name]
            attr.type.tensorType.dataType = proto.MIL_pb2.DataType.STRING
            attr.immediateValue.tensor.strings.values.append(id_generator(operation_spec))
            for operation_block_spec in operation_spec.blocks:
                assign_ids_to_block_operations(block_spec=operation_block_spec)

    for function_spec in program_spec.functions.values():
        block_spec = function_spec.block_specializations.get(function_spec.opset, None)
        if block_spec is None:
            raise ValueError(
                f"Invalid spec, missing block specialization for opset : {function_spec.opset}"
            )

        assign_ids_to_block_operations(block_spec=block_spec)


def apply_compute_plan(
    model: MLModel,
    compute_plan: MLComputePlan,
    backend_assignment_fn: Optional[
        Callable[[proto.MIL_pb2.Operation, Optional[MLComputePlanDeviceUsage]], Optional[List[str]]]
    ] = None,
) -> MLModel:
    """
    This function takes an MLModel and sets the intended backend attribute
    of each operation in the model as determined by the model's compute plan.

    It updates the 'IntendedBackend' attribute of each operation,
    ensuring the same dispatch even if the model is modified.

    Parameters:
    -----------
    model : MLModel
        The model to which the compute plan will be applied.

    compute_plan : Optional[MLComputePlan]
        The model's compute plan.

    backend_assignment_fn : Callable
        A function that determines the intended backends for each operation.

    Returns:
    --------
    MLModel
        The modified model after applying the backend assignments.
    """

    debug_id = 0
    debug_id_attr_name = "debug_id"

    def get_debug_id(op: proto.MIL_pb2.Operation) -> int:
        attr = op.attributes[debug_id_attr_name]
        values = attr.immediateValue.tensor.strings.values
        if len(values) == 0:
            raise KeyError(
                f"Debug ID attribute '{debug_id_attr_name}' not found in operation: {op.name}"
            )
        value = values[0]
        return int(value)

    def id_generator(op: proto.MIL_pb2.Operation) -> str:
        nonlocal debug_id
        debug_id += 1
        return str(debug_id)

    def backend_assignment_using_preferred_compute_device(
        op: proto.MIL_pb2.Operation, compute_device_usage: Optional[MLComputePlanDeviceUsage]
    ) -> Optional[List[str]]:
        if compute_device_usage is None:
            return None

        preferred_compute_device = compute_device_usage.preferred_compute_device
        return _get_backend_names(compute_device=preferred_compute_device)

    if backend_assignment_fn is None:
        backend_assignment_fn = backend_assignment_using_preferred_compute_device

    model_spec = model.get_spec()

    if model_spec.WhichOneof("Type") != "mlProgram":
        raise ValueError("The apply_compute_plan only supports ML Program.")

    _assign_ids_to_program_operations(
        program_spec=model_spec.mlProgram,
        id_generator=id_generator,
        attr_name=debug_id_attr_name,
    )

    model = MLModel(
        model=model_spec,
        weights_dir=model.weights_dir,
        function_name=model.function_name,
        compute_units=model.compute_unit,
    )

    values = map_model_spec_to_path(
        model_spec=model_spec,
    )

    debug_id_to_path_map = {get_debug_id(op_spec): path for op_spec, path in values}

    values = map_model_structure_to_path(
        model_structure=compute_plan.model_structure,
    )
    path_to_structure_map = {path: structure for structure, path in values}

    def set_intended_backends_for_op(
        op_spec: proto.MIL_pb2.Operation,
    ) -> Optional[List[str]]:
        path = debug_id_to_path_map.get(get_debug_id(op_spec), None)
        op_structure = path_to_structure_map.get(path, None) if path is not None else None
        compute_device_usage = None
        if op_structure is not None:
            compute_device_usage = compute_plan.get_compute_device_usage_for_mlprogram_operation(
                op_structure
            )

        backend_names = backend_assignment_fn(op_spec, compute_device_usage)
        return backend_names

    return set_intended_backends(
        model=model,
        backend_assignment_fn=set_intended_backends_for_op,
    )
