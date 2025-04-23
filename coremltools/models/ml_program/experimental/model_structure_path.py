# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

from coremltools import proto as _proto

from ...compute_plan import (
    MLModelStructure,
    MLModelStructureNeuralNetwork,
    MLModelStructureNeuralNetworkLayer,
    MLModelStructureProgram,
    MLModelStructureProgramBlock,
    MLModelStructureProgramOperation,
)

@_dataclass(frozen=True)
class ModelStructurePath:
    """
    This class represents a hierarchical path within a model structure,
    allowing for the representation of various components in a program,
    neural network, or pipeline.
    """

    @_dataclass(frozen=True)
    class Program:
        """
        Represents a program in the model structure.
        """

        @_dataclass(frozen=True)
        class Function:
            name: str

            @staticmethod
            def from_dict(
                dict: _Dict[str, _Any]
            ) -> _Optional["ModelStructurePath.Program.Function"]:
                value = dict.get("programFunction", None)
                if not isinstance(value, _Mapping):
                    return None

                data = value.get("data", {})
                return ModelStructurePath.Program.Function(name=data.get("name", ""))

            def to_dict(self) -> _Dict[str, _Any]:
                return {"programFunction": {"data": {"name": self.name}}}

        @_dataclass(frozen=True)
        class Block:
            index: int

            @staticmethod
            def from_dict(dict: _Dict[str, _Any]) -> _Optional["ModelStructurePath.Program.Block"]:
                value = dict.get("programBlock", None)
                if not isinstance(value, _Mapping):
                    return None

                data = value.get("data", {})
                return ModelStructurePath.Program.Block(index=data.get("index", -1))

            def to_dict(self) -> _Dict[str, _Any]:
                return {"programBlock": {"data": {"index": self.index}}}

        @_dataclass(frozen=True)
        class Operation:
            output_name: str

            @staticmethod
            def from_dict(dict: _Dict[str, _Any]) -> _Optional["ModelStructurePath.Program.Block"]:
                value = dict.get("programOperation", None)
                if not isinstance(value, _Mapping):
                    return None

                data = value.get("data", {})
                output_name = data.get("outputName", "")
                return ModelStructurePath.Program.Operation(output_name=output_name)

            def to_dict(self) -> _Dict[str, _Any]:
                return {"programOperation": {"data": {"outputName": self.output_name}}}

        @staticmethod
        def from_dict(dict: _Dict[str, _Any]) -> _Optional["ModelStructurePath.Program.Block"]:
            value = dict.get("program", None)
            if not isinstance(value, _Mapping):
                return None

            return ModelStructurePath.Program()

        def to_dict(self) -> _Dict[str, _Any]:
            return {"program": {}}

    @_dataclass(frozen=True)
    class NeuralNetwork:
        """
        Represents a neural network in the model structure.
        """

        @_dataclass(frozen=True)
        class Layer:
            name: str

            @staticmethod
            def from_dict(
                dict: _Dict[str, _Any]
            ) -> _Optional["ModelStructurePath.NeuralNetwork.Layer"]:
                value = dict.get("neuralNetworkLayer", None)
                if not isinstance(value, _Mapping):
                    return None

                data = value.get("data", {})
                return ModelStructurePath.NeuralNetwork.Layer(name=data.get("name", ""))

            def to_dict(self) -> _Dict[str, _Any]:
                return {"neuralNetworkLayer": {"data": {"name": self.name}}}

        @staticmethod
        def from_dict(dict: _Dict[str, _Any]) -> _Optional["ModelStructurePath.NeuralNetwork"]:
            value = dict.get("neuralNetwork", None)
            if not isinstance(value, _Mapping):
                return None

            return ModelStructurePath.NeuralNetwork()

        def to_dict(self) -> _Dict[str, _Any]:
            return {"neuralNetwork": {}}

    @_dataclass(frozen=True)
    class Pipeline:
        """
        Represents a pipeline in the model structure.
        """

        @_dataclass(frozen=True)
        class Model:
            name: str

            @staticmethod
            def from_dict(dict: _Dict[str, _Any]) -> _Optional["ModelStructurePath.Pipeline.Model"]:
                value = dict.get("pipelineModel", None)
                if not isinstance(value, _Mapping):
                    return None

                data = value.get("data", {})
                return ModelStructurePath.Pipeline.Model(name=data.get("name", ""))

            def to_dict(self) -> _Dict[str, _Any]:
                return {"pipelineModel": {"data": {"name": self.name}}}

        @staticmethod
        def from_dict(dict: _Dict[str, _Any]) -> _Optional["ModelStructurePath.Pipeline"]:
            value = dict.get("pipeline", None)
            if not isinstance(value, _Mapping):
                return None

            return ModelStructurePath.Pipeline()

        def to_dict(self) -> _Dict[str, _Any]:
            return {"pipeline": {}}

    Component = _Union[
        Program,
        Program.Function,
        Program.Block,
        Program.Operation,
        NeuralNetwork,
        NeuralNetwork.Layer,
        Pipeline,
        Pipeline.Model,
    ]

    components: _Tuple[Component, ...]

    @staticmethod
    def from_dict(dict: _Dict[str, _Any]) -> "ModelStructurePath":
        components = dict.get("components", [])
        if not isinstance(components, _Sequence):
            raise ValueError(
                f"Expected 'components' to be a sequence, but got {type(components).__name__}. "
            )

        component_types = [
            ModelStructurePath.Program,
            ModelStructurePath.Program.Function,
            ModelStructurePath.Program.Block,
            ModelStructurePath.Program.Operation,
            ModelStructurePath.NeuralNetwork,
            ModelStructurePath.NeuralNetwork.Layer,
            ModelStructurePath.Pipeline,
            ModelStructurePath.Pipeline.Model,
        ]

        parsed_components = []
        for component in components:
            parsed_component = None
            for component_type in component_types:
                parsed_component = component_type.from_dict(dict=component)
                if parsed_component is not None:
                    break

            if parsed_component is None:
                raise ValueError(
                    f"Failed to parse component: {component}. "
                    "The component does not match any known type in the ModelStructurePath."
                )

            parsed_components.append(parsed_component)

        return ModelStructurePath(components=tuple(parsed_components))

    def to_dict(self) -> _Dict[str, _Any]:
        return {"components": [component.to_dict() for component in self.components]}


ModelStructureLayerOrOperation = _Union[
    MLModelStructureNeuralNetworkLayer, MLModelStructureProgramOperation
]


def _map_program_structure_to_path(
    model_structure: MLModelStructureProgram,
    components: _List[ModelStructurePath.Component],
) -> _List[_Tuple[ModelStructureLayerOrOperation, ModelStructurePath]]:
    def _map_block_structure_to_path(
        block: MLModelStructureProgramBlock,
        components: _List[ModelStructurePath.Component],
        index: int,
    ):
        result = []
        block_component = components + [ModelStructurePath.Program.Block(index=index)]
        for operation in block.operations:
            for output in operation.outputs:
                operation_component = block_component + [
                    ModelStructurePath.Program.Operation(output_name=output.name)
                ]
                path = ModelStructurePath(components=tuple(operation_component))
                result.append((operation, path))
                for index, operation_block in enumerate(operation.blocks):
                    result.extend(
                        _map_block_structure_to_path(
                            block=operation_block,
                            components=operation_component,
                            index=index,
                        )
                    )

        return result

    result = []
    program_component = components + [ModelStructurePath.Program()]
    for name, function in model_structure.functions.items():
        function_component = program_component + [ModelStructurePath.Program.Function(name=name)]
        result.extend(
            _map_block_structure_to_path(
                block=function.block,
                components=function_component,
                index=-1,
            )
        )
    return result


def _map_neural_network_structure_to_path(
    model_structure: MLModelStructureNeuralNetwork,
    components: _List[ModelStructurePath.Component],
) -> _List[_Tuple[ModelStructureLayerOrOperation, ModelStructurePath]]:
    result = []
    neural_network_component = components + [ModelStructurePath.NeuralNetwork()]
    for layer in model_structure.layers:
        neural_network_layer_component = neural_network_component + [
            ModelStructurePath.NeuralNetwork.Layer(name=layer.name)
        ]
        path = ModelStructurePath(components=tuple(neural_network_layer_component))
        result.append((layer, path))

    return result


def map_model_structure_to_path(
    model_structure: MLModelStructure,
    components: _List[ModelStructurePath.Component] = [],
) -> _List[_Tuple[ModelStructureLayerOrOperation, ModelStructurePath]]:
    if model_structure.program:
        return _map_program_structure_to_path(
            model_structure=model_structure.program,
            components=components,
        )
    elif model_structure.neuralnetwork:
        return _map_neural_network_structure_to_path(
            model_structure=model_structure.neuralnetwork,
            components=components,
        )
    elif model_structure.pipeline:
        components.append(ModelStructurePath.Pipeline())
        result = []
        for submodel_name, submodel in model_structure.pipeline.sub_models:
            result.extend(
                map_model_structure_to_path(
                    model_structure=submodel,
                    components=components
                    + [
                        ModelStructurePath.Pipeline(),
                        ModelStructurePath.Pipeline.Model(name=submodel_name),
                    ],
                )
            )

            return result
    else:
        raise ValueError("Invalid model structure: no recognized components found")


def _map_program_spec_to_path(
    program_spec: _proto.MIL_pb2.Program,
    components: _List[ModelStructurePath.Component],
) -> _List[_Tuple[_proto.MIL_pb2.Operation, ModelStructurePath]]:
    def _map_block_spec_to_path(
        block: _proto.MIL_pb2.Block,
        components: _List[ModelStructurePath.Component],
        index: int,
    ):
        result = []
        block_component = components + [ModelStructurePath.Program.Block(index=index)]
        for operation in block.operations:
            for output in operation.outputs:
                operation_component = block_component + [
                    ModelStructurePath.Program.Operation(output_name=output.name)
                ]
                path = ModelStructurePath(components=tuple(operation_component))
                result.append((operation, path))
                for index, operation_block in enumerate(operation.blocks):
                    result.extend(
                        _map_block_spec_to_path(
                            block=operation_block,
                            components=operation_component,
                            index=index,
                        )
                    )

        return result

    result = []
    program_component = components + [ModelStructurePath.Program()]
    for function_name, function_spec in program_spec.functions.items():
        block_spec = function_spec.block_specializations.get(function_spec.opset, None)
        if block_spec is None:
            continue
        function_component = program_component + [
            ModelStructurePath.Program.Function(name=function_name)
        ]
        result.extend(
            _map_block_spec_to_path(
                block=block_spec,
                components=function_component,
                index=-1,
            )
        )
    return result


def map_model_spec_to_path(
    model_spec: "_proto.Model_pb2.Model",
    components: _List[ModelStructurePath.Component] = [],
) -> _List[_Tuple[_proto.MIL_pb2.Operation, ModelStructurePath]]:
    spec_type = model_spec.WhichOneof("Type")
    if spec_type == "mlProgram":
        return _map_program_spec_to_path(
            program_spec=model_spec.mlProgram,
            components=components,
        )
    elif spec_type == "pipeline":
        components.append(ModelStructurePath.Pipeline())
        result = []
        for submodel_name, submodel in zip(model_spec.names.model_spec.models):
            result.extend(
                map_model_spec_to_path(
                    model_structure=submodel,
                    components=components
                    + [
                        ModelStructurePath.Pipeline(),
                        ModelStructurePath.Pipeline.Model(name=submodel_name),
                    ],
                )
            )

            return result
    else:
        raise ValueError("Invalid model structure: no recognized components found")
