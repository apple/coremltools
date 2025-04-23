# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from coremltools import proto
from coremltools._deps import _HAS_TORCH_EXPORT_API

from ..perf_utils import MLModelBenchmarker
from ..remote_device import Device

if _HAS_TORCH_EXPORT_API:
    from torch.export import ExportedProgram

import torch

from .debugging_utils import (
    TorchScriptNodeInfo,
    _convert_and_retrieve_exported_program_op_mapping,
    _convert_and_retrieve_jit_module_mapping,
)

TorchNode = Union[TorchScriptNodeInfo, torch.fx.Node]


class TorchMLModelBenchmarker(MLModelBenchmarker):
    """
    A specialized benchmarker for PyTorch models.

    This class extends the ``MLModelBenchmarker`` to provide benchmarking capabilities
    specifically tailored for PyTorch model. It inherits all the functionality
    of ``MLModelBenchmarker`` and includes methods to provide estimated execution times
    for torch nodes and submodules.
    """

    @dataclass
    class NodeExecutionInfo:
        node: TorchNode
        targets: List[MLModelBenchmarker.OperationExecutionInfo]
        measurement: Optional["MLModelBenchmarker.Measurement"]

        @staticmethod
        def from_source(node: TorchNode) -> "TorchMLModelBenchmarker.NodeExecutionInfo":
            return TorchMLModelBenchmarker.NodeExecutionInfo(
                node=node,
                targets=[],
                measurement=MLModelBenchmarker.Measurement.from_samples([]),
            )

    @dataclass
    class ModuleExecutionInfo:
        name: str
        ops: List["TorchMLModelBenchmarker.NodeExecutionInfo"]
        measurement: Optional["MLModelBenchmarker.Measurement"]

        @staticmethod
        def from_name(name: str) -> "TorchMLModelBenchmarker.ModuleExecutionInfo":
            return TorchMLModelBenchmarker.ModuleExecutionInfo(
                name=name, ops=[], measurement=MLModelBenchmarker.Measurement.from_samples([])
            )

    def __init__(
        self,
        model: Union["ExportedProgram", torch.jit.ScriptModule],
        device: Optional[Device] = None,
        **converter_kwargs,
    ) -> None:
        mlModel = None
        source_to_target_ops_mapping = None
        if isinstance(model, ExportedProgram):
            (
                mlModel,
                source_to_target_ops_mapping,
            ) = _convert_and_retrieve_exported_program_op_mapping(
                model=model,
                **converter_kwargs,
            )

        elif isinstance(model, torch.jit.ScriptModule):
            mlModel, module_mapping = _convert_and_retrieve_jit_module_mapping(
                model=model,
                **converter_kwargs,
            )
            # Get root module's source to target ops mapping
            source_to_target_ops_mapping = module_mapping[("", 0)].source_to_target_ops_mapping
        else:
            raise ValueError(
                f"Unsupported model type: {type(model)}. Expected ExportedProgram or ScriptModule"
            )

        super().__init__(
            model=mlModel,
            device=device,
        )

        self.source_model = model
        self._source_to_target_ops_mapping = source_to_target_ops_mapping

        target_output_name_to_source_ops_mapping = defaultdict(list)
        for source, target_ops in source_to_target_ops_mapping.items():
            for target_op in target_ops:
                for output in target_op.outputs:
                    target_output_name_to_source_ops_mapping[output.name].append(source)

        self._target_output_name_to_source_ops_mapping = target_output_name_to_source_ops_mapping

    @staticmethod
    def get_module_names(node: TorchNode) -> Optional[List[str]]:
        def parse_nn_module_stack(node: torch.fx.Node) -> Optional[List[str]]:
            if "nn_module_stack" not in node.meta:
                return None

            module_stack = node.meta["nn_module_stack"]
            return [qualified_name for _, (qualified_name, _) in module_stack.items()]

        if isinstance(node, TorchScriptNodeInfo):
            return [module_key[0] for module_key in node.modules]

        elif isinstance(node, torch.fx.Node):
            module_stack = parse_nn_module_stack(node=node)
            return module_stack

        else:
            raise ValueError(
                f"Unsupported node type: {type(node)}. "
                "Expected either 'TorchScriptNodeInfo' or 'torch.fx.Node'"
            )

    def find_source_nodes_for_target_op(
        self,
        target_op: proto.MIL_pb2.Operation,
    ) -> List[TorchNode]:
        result = set()
        for output in target_op.outputs:
            result.update(self._target_output_name_to_source_ops_mapping[output.name])

        return result

    async def benchmark_node_execution(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        iterations: int = 1,
        warmup: bool = False,
    ) -> List[NodeExecutionInfo]:
        """
        Measures the execution time of individual nodes in the PyTorch model.

        This method loads the converted model, runs predictions, and retrieves the execution time
        of each operation within the model.

        Parameters
        ----------
        inputs : Optional[Dict[str, Any]]
            The input data for the model prediction. If None, random input data will be generated.

        iterations: int
            The number of prediction iterations to run. Defaults to 1.

        warmup: bool
            Whether to perform a warmup iteration. Defaults to False.

        Returns
        -------
        List[TorchNodeExecutionInfo]
            A list of TorchNodeExecutionInfo objects, each containing
            details about an node's execution, sorted by execution time in descending order.

        Notes
        -----
        - The returned list is sorted by execution time, with the most time-consuming operations first.
        - Execution times are estimated based on the overall prediction time and the converted model's compute plan.
        """
        source_operation_execution_infos = {
            source: TorchMLModelBenchmarker.NodeExecutionInfo.from_source(source)
            for source in self._source_to_target_ops_mapping
        }

        target_execution_infos = await self.benchmark_operation_execution(
            inputs=inputs,
            iterations=iterations,
            warmup=warmup,
        )

        for target_execution_info in target_execution_infos:
            target_samples = target_execution_info.measurement.samples
            source_nodes = self.find_source_nodes_for_target_op(target_execution_info.spec)
            source_samples = [target_sample / len(source_nodes) for target_sample in target_samples]
            for source_node in source_nodes:
                source_execution_info = source_operation_execution_infos[source_node]
                source_execution_info.targets.append(target_execution_info)
                source_execution_info.measurement = source_execution_info.measurement.add_samples(
                    source_samples
                )

        result = list(source_operation_execution_infos.values())
        return sorted(result, key=lambda value: value.measurement.sort_key, reverse=True)

    async def benchmark_module_execution(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        iterations: int = 1,
        warmup: bool = False,
    ) -> List[ModuleExecutionInfo]:
        """
        Measures the execution time of modules in the PyTorch model.

        This method loads the converted model, runs predictions, and retrieves the execution time
        of each operation within the model.

        Parameters
        ----------
        inputs : Optional[Dict[str, Any]]
            The input data for the model prediction. If None, random input data will be generated.

        iterations: int
            The number of prediction iterations to run. Defaults to 1.

        warmup: bool
            Whether to perform a warmup iteration. Defaults to False.

        Returns
        -------
        List[TorchModuleExecutionInfo]
            A list of ``TorchModuleExecutionInfo`` objects, each containing
            details about an modules's execution, sorted by execution time in descending order.

        Notes
        -----
        - The returned list is sorted by execution time, with the most time-consuming operations first.
        - Execution times are estimated based on the overall prediction time and the converted model's compute plan.
        """
        source_operation_execution_infos = await self.benchmark_node_execution(
            inputs=inputs,
            iterations=iterations,
            warmup=warmup,
        )

        source_module_execution_infos = {}
        for source_operation_execution_info in source_operation_execution_infos:
            source_node = source_operation_execution_info.node
            module_names = TorchMLModelBenchmarker.get_module_names(source_node)
            if module_names is None:
                continue

            for module_name in module_names:
                source_module_execution_info = source_module_execution_infos.get(
                    module_name,
                    TorchMLModelBenchmarker.ModuleExecutionInfo.from_name(name=module_name),
                )

                source_module_execution_info.ops.append(source_operation_execution_info)
                source_module_execution_info.measurement = (
                    source_module_execution_info.measurement.add_samples(
                        samples=source_operation_execution_info.measurement.samples,
                    )
                )

                source_module_execution_infos[module_name] = source_module_execution_info

        result = list(source_module_execution_infos.values())
        return sorted(result, key=lambda value: value.measurement.sort_key, reverse=True)
