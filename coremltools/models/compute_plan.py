#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple

from coremltools import ComputeUnit as _ComputeUnit
from coremltools import _logger

from .compute_device import MLComputeDevice as _MLComputeDevice

try:
    from ..libcoremlpython import _MLModelProxy
except Exception as e:
    _logger.warning(f"Failed to load _MLModelProxy: {e}")
    _MLModelProxy = None

try:
    from ..libcoremlpython import _MLComputePlanProxy
except Exception as e:
    _logger.warning(f"Failed to load _MLComputePlanProxy: {e}")
    _MLComputePlanProxy = None

@_dataclass(frozen=True)
class MLModelStructureNeuralNetworkLayer:
    """
    Represents a layer in a neural network model structure.

    Attributes
    ----------
    name : str
        The name of the neural network layer.

    type : str
        The type of the layer (e.g., 'Dense', 'Convolutional', etc.).

    input_names : List[str]
        A list of names representing the inputs to this layer.

    output_names : List[str]
        A list of names representing the outputs from this layer.
    """
    name: str
    type: str
    input_names: _List[str]
    output_names: _List[str]
    __proxy__: _Any


@_dataclass(frozen=True)
class MLModelStructureNeuralNetwork:
    """
    Represents the structure of a neural network model.

    Attributes
    ----------
    layers : List[MLModelStructureNeuralNetworkLayer]
        The list of layers in the neural network.
    """
    layers: _List[MLModelStructureNeuralNetworkLayer]


@_dataclass(frozen=True)
class MLModelStructureProgramValue:
    """
    Represents the value of a constant in an ML Program.
    """
    pass


@_dataclass(frozen=True)
class MLModelStructureProgramBinding:
    """
    Represents a binding between a name and a program value in an ML Program.
    This is either a previously defined name of a variable or a constant value in the Program.

    Attributes
    ----------
    name : Optional[str]
        The name of the variable, it can be None.

    value : Optional[MLModelStructureProgramValue]
        The constant value, it can be None.
    """
    name: _Optional[str]
    value: _Optional[MLModelStructureProgramValue]


@_dataclass(frozen=True)
class MLModelStructureProgramArgument:
    """
    Represents an argument in an ML Program.

    Attributes
    ----------
    bindings : List[MLModelStructureProgramBinding]
        The list of bindings.
    """
    bindings: _List[MLModelStructureProgramBinding]


@_dataclass(frozen=True)
class MLModelStructureProgramValueType:
    """
    Represents the type of a value or a variable in an ML Program.
    """
    pass

@_dataclass(frozen=True)
class MLModelStructureProgramNamedValueType:
    """
    Represents a parameter's name and type in an ML Program.

    Attributes
    ----------
    name : str
        The name of the parameter.

    type : MLModelStructureProgramValueType
        The type of the parameter.
    """
    name: str
    type: MLModelStructureProgramValueType


@_dataclass(frozen=True)
class MLModelStructureProgramOperation:
    """
    Represents an operation in an ML Program.

    Attributes
    ----------
    inputs : Dict[str, MLModelStructureProgramArgument]
        The arguments to the Operation.

    operator_name : str
        The name of the operator, e.g., "conv", "pool", "softmax", etc.

    outputs : List[MLModelStructureProgramNamedValueType]
        The outputs of the Operation.

    blocks : List[MLModelStructureProgramBlock]
        The list of nested blocks for loops and conditionals, e.g., a conditional block will have two entries here.
    """
    inputs: _Dict[str, MLModelStructureProgramArgument]
    operator_name: str
    outputs: _List[MLModelStructureProgramNamedValueType]
    blocks: _List["MLModelStructureProgramBlock"]
    __proxy__: _Any


@_dataclass(frozen=True)
class MLModelStructureProgramBlock:
    """
    Represents a block in an ML Program.

    Attributes
    ----------
    inputs : List[MLModelStructureProgramNamedValueType]
        The named inputs to the block.

    operator_name : str
        The name of the operator, e.g., "conv", "pool", "softmax", etc.

    outputs : List[MLModelStructureProgramNamedValueType]
        The outputs of the Operation.

    blocks: List[MLModelStructureProgramBlock]
        The list of nested blocks for loops and conditionals, e.g., a conditional block will have two entries here.
    """
    inputs: _List[MLModelStructureProgramNamedValueType]
    operations: _List[MLModelStructureProgramOperation]
    output_names: _List[str]


@_dataclass(frozen=True)
class MLModelStructureProgramFunction:
    """
    Represents a function in an ML Program.

    Attributes
    ----------
    inputs : List[MLModelStructureProgramNamedValueType]
        The named inputs to the function.

    block : MLModelStructureProgramBlock
        The active block in the function.
    """

    inputs: _List[MLModelStructureProgramNamedValueType]
    block: MLModelStructureProgramBlock


@_dataclass(frozen=True)
class MLModelStructureProgram:
    """
    Represents the structure of an ML Program model.

    Attributes
    ----------
    functions : List[MLModelStructureProgramFunction]
        The functions in the program.
    """
    functions: _List[MLModelStructureProgramFunction]


@_dataclass(frozen=True)
class MLModelStructurePipeline:
    """
    Represents the structure of a pipeline model.

    Attributes
    ----------
    sub_models : Tuple[str, MLModelStructure]
        The list of sub-models in the pipeline.
    """
    sub_models: _Tuple[str, "MLModelStructure"]


@_dataclass(frozen=True)
class MLModelStructure:
    """
    Represents the structure of a model.

    Attributes
    ----------
    neuralnetwork : Optional[MLModelStructureNeuralNetwork]
        The structure of a NeuralNetwork model, if the model is a NeuralNetwork; otherwise None.

    program : Optional[MLModelStructureProgram]
        The structure of an ML Program model, if the model is an ML Program; otherwise, None.

    pipeline : Optional[MLModelStructurePipeline]
        The structure of a Pipeline model. if the model is a Pipeline; otherwise None.
    """

    neuralnetwork: _Optional[MLModelStructureNeuralNetwork]
    program: _Optional[MLModelStructureProgram]
    pipeline: _Optional[MLModelStructurePipeline]

    @classmethod
    def load_from_path(cls, compiled_model_path: str) -> "MLModelStructure":
        """
        Loads the structure of a compiled model.

        The path must be the location of the ``mlmodelc`` directory.

        Parameters
        ----------
        compiled_model_path (str): The path to the compiled model.

        Returns
        -------
        MLModelStructure
            An instance of MLModelStructure.

        Examples
        --------
        .. sourcecode:: python
        
            model_structure = coremltools.models.compute_plan.MLModelStructure.load_from_path(
                model.get_compiled_path()
            )

            if model_structure.neuralNetwork is not None:
                # Examine Neural network model.
                pass
            elif model_structure.program is not None:
                # Examine ML Program model.
                pass
            elif model_structure.pipeline is not None:
                # Examine Pipeline model.
                pass
            else:
                # The model type is something else.
                pass
        
        """

        if _MLModelProxy is None:
            raise ValueError("MLModelStructure is not supported.")

        return _MLModelProxy.get_model_structure(compiled_model_path)


@_dataclass(frozen=True)
class MLComputePlanDeviceUsage:
    """
    Represents the anticipated compute devices that would be used for executing a layer/operation.

    Attributes
    ----------
    preferred_compute_device : MLComputeDevice
        The compute device that the framework prefers to execute the layer/operation.

    supported_compute_devices : List[MLComputeDevice]
        The compute device that the framework prefers to execute the layer/operation.
    """

    preferred_compute_device: _MLComputeDevice
    supported_compute_devices: _List[_MLComputeDevice]


@_dataclass(frozen=True)
class MLComputePlanCost:
    """
    Represents the estimated cost of executing a layer/operation.

    Attributes
    ----------
    weight : float
        The estimated workload of executing the operation over the total model execution. The value is between [0.0, 1.0].
    """

    weight: float

class MLComputePlan:
    """
    Represents the plan for executing a model.

    The application can use the plan to estimate the necessary cost and
    resources of the model before running the predictions.
    """

    def __init__(self, proxy):
        if _MLComputePlanProxy is None or not isinstance(proxy, _MLComputePlanProxy):
            raise TypeError("The proxy parameter must be of type _MLComputePlanProxy.")
        self.__proxy__ = proxy

    @property
    def model_structure(self) -> MLModelStructure:
        """
        Returns the model structure.
        """
        return self.__proxy__.model_structure

    def get_compute_device_usage_for_mlprogram_operation(
        self,
        operation: MLModelStructureProgramOperation,
    ) -> _Optional[MLComputePlanDeviceUsage]:
        """
        Returns the estimated cost of executing an ML Program operation.

        Parameters
        ----------
        operation : MLModelStructureProgramOperation
            An ML Program operation.

        Returns
        -------
        Optional[MLComputePlanDeviceUsage]
            The anticipated compute devices that would be used for executing the operation or ``None`` if the usage couldn't be determined.
        """
        return self.__proxy__.get_compute_device_usage_for_mlprogram_operation(operation)

    def get_compute_device_usage_for_neuralnetwork_layer(
        self,
        layer: MLModelStructureNeuralNetworkLayer,
    ) -> _Optional[MLComputePlanDeviceUsage]:
        """
        Returns the estimated cost of executing a NeuralNetwork layer.

        Parameters
        ----------
        operation MLModelStructureProgramOperation:
            A NeuralNetwork layer.

        Returns
        -------
        Optional[MLComputePlanDeviceUsage]
            The anticipated compute devices that would be used for executing the layer or ``None`` if the usage couldn't be determined.
        """
        return self.__proxy__.get_compute_device_usage_for_neuralnetwork_layer(layer)

    def get_estimated_cost_for_mlprogram_operation(
        self,
        operation: MLModelStructureProgramOperation,
    ) -> _Optional[MLComputePlanCost]:
        """
        Returns the estimated cost of executing an ML Program operation.

        Parameters
        ----------
        operation : MLModelStructureProgramOperation
            An ML Program operation.

        Returns
        -------
        Optional[MLComputePlanCost]
            The estimated cost of executing the operation.
        """
        return self.__proxy__.get_estimated_cost_for_mlprogram_operation(operation)

    @classmethod
    def load_from_path(
        cls,
        path: str,
        compute_units: _ComputeUnit = _ComputeUnit.ALL,
    ) -> "MLComputePlan":
        """
        Loads the compute plan of a compiled model.

        The path must be the location of the ``mlmodelc`` directory.

        Parameters
        ----------
        compiled_model_path : str
            The path to the compiled model.

        Returns
        -------
        The plan for executing the model.

        Examples
        --------
        .. sourcecode:: python

            compute_plan = coremltools.models.compute_plan.MLComputePlan.load_from_path(
                model.get_compiled_path()
            )

            if compute_plan.model_structure.program is None:
                raise ValueError("Unexpected model type.")

            program = compute_plan.model_structure.program
            mainFunction = program["main"]
            for operation in mainFunction.block.operations:
                # Get the compute device usage for the operation.
                compute_device_usage = (
                    compute_plan.get_compute_device_usage_for_mlprogram_operation(operation)
                )
                # Get the estimated cost of executing the operation.
                estimated_cost = compute_plan.get_estimated_cost_for_mlprogram_operation(operation)

        """

        if _MLModelProxy is None:
            raise ValueError("MLComputePlan is not supported.")

        return _MLModelProxy.get_compute_plan(path, compute_units.name)
