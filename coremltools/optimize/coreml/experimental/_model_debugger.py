# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import List

import numpy as np
from tqdm import tqdm

import coremltools as ct
from coremltools import _SPECIFICATION_VERSION_IOS_16
from coremltools import _logger as logger
from coremltools.models import MLModel


# rdar://137163049 code cleanup.
class OperationInfo:
    def __init__(self, spec):
        self.dependants = []
        self.dependencies = []
        self.outputs = dict([(output.name, output) for output in spec.outputs])
        self.spec = spec


class BlockInfo:
    def __init__(self, name, operations, spec):
        self.name = name
        self.operations = operations
        self.spec = spec


class FunctionInfo:
    def __init__(self, name, blocks, spec):
        self.name = name
        self.blocks = blocks
        self.spec = spec


class ProgramInfo:
    def __init__(self, functions, spec):
        self.functions = functions
        self.spec = spec


class ModelInfo:
    def __init__(self, program_info, spec):
        self.program_info = program_info
        self.spec = spec


class ModelDebugger:
    @staticmethod
    def batch(iterable, batch_size: int = 1):
        return (iterable[idx : idx + batch_size] for idx in range(0, len(iterable), batch_size))

    @classmethod
    def unique(cls, sequence: List) -> List:
        return list(set(sequence))

    @classmethod
    def get_block_info(cls, block_name, block_spec) -> BlockInfo:
        operations = {}
        for operation_spec in block_spec.operations:
            operation = OperationInfo(operation_spec)
            dependencies = []

            for input_name in operation_spec.inputs:
                arguments = operation_spec.inputs[input_name].arguments
                input_dependencies = [
                    operations.get(argument.name, None)
                    for argument in arguments
                    if argument.name is not None
                ]
                input_dependencies = [
                    input_dependency
                    for input_dependency in input_dependencies
                    if input_dependency is not None
                ]
                dependencies.extend(input_dependencies)

            dependencies = cls.unique(dependencies)
            for dependency in dependencies:
                dependency.dependants.append(operation)
            operation.dependencies = dependencies

            output_names = [output.name for output in operation_spec.outputs]
            for output_name in output_names:
                operations[output_name] = operation

        return BlockInfo(block_name, operations, block_spec)

    @classmethod
    def get_function_info(cls, function_name, function_spec) -> FunctionInfo:
        blocks = {}
        for block_name, block_spec in function_spec.block_specializations.items():
            blocks[block_name] = cls.get_block_info(block_name, block_spec)

        return FunctionInfo(function_name, blocks, function_spec)

    @classmethod
    def get_program_info(cls, program_spec) -> ProgramInfo:
        functions = {}
        for function_name, function_spec in program_spec.functions.items():
            functions[function_name] = cls.get_function_info(function_name, function_spec)

        return ProgramInfo(functions, program_spec)

    @classmethod
    def get_model_info(cls, model: MLModel) -> ModelInfo:
        model_spec = model.get_spec()
        return ModelInfo(cls.get_program_info(model_spec.mlProgram), model_spec)

    @classmethod
    def get_all_outputs(cls, block_info: BlockInfo) -> List[ct.proto.MIL_pb2.NamedValueType]:
        result: List[ct.proto.MIL_pb2.NamedValueType] = []
        output_names = block_info.spec.outputs
        while len(output_names) > 0:
            operations = [
                block_info.operations[output_name]
                for output_name in output_names
                if output_name in block_info.operations
            ]
            result.extend(
                [output for operation in operations for output in operation.outputs.values()]
            )
            prev_output_names = [
                output_name
                for operation in operations
                for dependency in operation.dependencies
                for output_name in dependency.outputs.keys()
            ]
            output_names = cls.unique(prev_output_names)
        return result

    @classmethod
    def get_any_block(cls, model_info: ModelInfo):
        function_info: FunctionInfo = list(model_info.program_info.functions.values())[0]
        return list(function_info.blocks.values())[0]

    @classmethod
    def clone_spec(cls, spec):
        new_spec = spec.__class__()
        new_spec.CopyFrom(spec)
        return new_spec

    @classmethod
    def get_output_feature_type(cls, output_name, operations):
        operation = operations[output_name]
        data_type = operation.outputs[output_name].type.tensorType.dataType

        # Valid data type as model outputs.
        data_type_to_feature_type = {
            ct.proto.MIL_pb2.DataType.FLOAT16: ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16,
            ct.proto.MIL_pb2.DataType.FLOAT64: ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE,
            ct.proto.MIL_pb2.DataType.FLOAT32: ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32,
            ct.proto.MIL_pb2.DataType.INT32: ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32,
        }

        # Return None for invalid data type as model outputs (e.g. bool).
        if data_type not in data_type_to_feature_type:
            return None

        return data_type_to_feature_type[data_type]

    def __init__(self, model: MLModel):
        self.weights_dir = model.weights_dir
        self.model_info = self.__class__.get_model_info(model)
        self.block_info = self.__class__.get_any_block(self.model_info)

        model_outputs = [output for output in self.model_info.spec.description.output]
        output_names = set([output.name for output in model_outputs])
        all_outputs = self.__class__.get_all_outputs(self.block_info)
        intermediate_outputs = [output for output in all_outputs if output.name not in output_names]

        self.__model_outputs = model_outputs
        self.__all_outputs = all_outputs
        self.__intermediate_outputs = intermediate_outputs
        self.__intermediate_output_names = self.__class__.unique(
            [output_spec.name for output_spec in intermediate_outputs]
        )
        self.__cached_models = {}

    @property
    def output_names(self):
        return self.__class__.unique([output.name for output in self.outputs])

    def get_intermediate_output_names(
        self, op_include_fn=(lambda op: not (op.spec.type == "const"))
    ):
        all_operations = self.block_info.operations
        intermediate_output_names = list(
            filter(
                lambda name: op_include_fn(all_operations[name]), self.__intermediate_output_names
            )
        )
        intermediate_output_names.reverse()

        return self.__class__.unique(intermediate_output_names)

    def _get_concat_op_info(self) -> List[List[str]]:
        """
        Return a list of lists of input/output names of concat ops.
        """
        intermediate_output_names_list = self.get_intermediate_output_names(
            lambda op: (op.spec.type == "concat")
        )
        all_operations = self.block_info.operations
        concat_op_info_list = []

        for concat_output_name in intermediate_output_names_list:
            # Get a list of input names (to "values") of current concat op.
            arguments = all_operations[concat_output_name].spec.inputs["values"].arguments
            argument_list = [val.name for val in arguments if val.name is not None]

            # Append the output name of current concat op.
            argument_list.append(concat_output_name)

            # Append a list of input/output names of current concat op.
            concat_op_info_list.append(argument_list)

        return concat_op_info_list

    def predict_intermediate_outputs(
        self, inputs, intermediate_output_names, compute_units=ct.ComputeUnit.CPU_ONLY
    ):
        """Append all intermediate_output_names to model's output, and then use model.predict to get those outputs."""
        model_key = frozenset(intermediate_output_names)

        cloned_spec = self.__class__.clone_spec(self.model_info.spec)
        if self.model_info.spec.specificationVersion < _SPECIFICATION_VERSION_IOS_16:
            logger.warning(
                f"The model has spec version {self.model_info.spec.specificationVersion}, but the minimum "
                f"version to do activation quantization is {_SPECIFICATION_VERSION_IOS_16}. Forcely updated the spec to {_SPECIFICATION_VERSION_IOS_16} during calibration."
            )
            cloned_spec.specificationVersion = max(self.model_info.spec.specificationVersion, 7)
        cloned_model_info = ModelInfo(
            ModelDebugger.get_program_info(cloned_spec.mlProgram), cloned_spec
        )
        cloned_block_info = self.__class__.get_any_block(cloned_model_info)

        for output_name in intermediate_output_names:
            cloned_output_type = self.__class__.get_output_feature_type(
                output_name, self.block_info.operations
            )

            # Some intermediate tensors cannot be appended to outputs since their data type is not valid as an output data type.
            # For example, an intermediate tensor with bool type cannot be appended to outputs (which will cause compilation error).
            if cloned_output_type is None:
                continue

            cloned_block_info.spec.outputs.append(output_name)
            cloned_output = ct.proto.Model_pb2.FeatureDescription()
            cloned_output.name = output_name
            cloned_output.type.multiArrayType.dataType = cloned_output_type
            cloned_model_info.spec.description.output.append(cloned_output)

        model = ct.models.MLModel(
            cloned_spec,
            weights_dir=self.weights_dir,
            compute_units=compute_units,
            skip_model_load=False,  # Don't skip model load as we need model prediction to get activations range.
        )
        return model.predict(inputs)

    @staticmethod
    def record_intermediate_output(output_value, output_name, activation_stats_dict):
        tensor_min = np.min(output_value.flatten())
        tensor_max = np.max(output_value.flatten())
        activation_stats_dict[output_name]["rmin"] = tensor_min
        activation_stats_dict[output_name]["rmax"] = tensor_max
        if output_name in activation_stats_dict:
            activation_stats_dict[output_name]["rmin"] = min(
                tensor_min, activation_stats_dict[output_name]["rmin"]
            )
            activation_stats_dict[output_name]["rmax"] = max(
                tensor_max, activation_stats_dict[output_name]["rmax"]
            )
        else:
            activation_stats_dict[output_name]["rmin"] = tensor_min
            activation_stats_dict[output_name]["rmax"] = tensor_max

    def step(
        self,
        inputs,
        activation_stats_dict,
        intermediate_output_names=None,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        op_group_size=-1,
    ):
        """
        During activation quantization, to get activations, all intermediate tensors will be added to model's outputs
        to form a temperary model. Then the temp model's output will be recorded for quantization.

        inputs: Inputs to the model for running `model.predict`.
        activation_stats_dict: Dictionary to store the min/max of the activation statistics.
        intermediate_output_names: The output names of the intermediate tensors.
        compute_units: The compute units to use for temperary model's prediction.
        op_group_size: If the model is very large, it could lead to the temperary model having thousands of outputs,
            which may lead to model hanging forever during model loading. To work around this issue, intermediate
            outputs are grouped into smaller groups, where each time a temperary model will only have `op_group_size`
            outputs. By default (op_group_size = -1), op_group_size is equal to the number of valid intermediate ops.
        """
        if intermediate_output_names is None:
            intermediate_output_names = self.get_intermediate_output_names()
        if len(intermediate_output_names) == 0:
            return
        if op_group_size == -1:
            op_group_size = len(intermediate_output_names)

        model_output_names = [output.name for output in self.__model_outputs]
        model_outputs = dict()

        for output_names in tqdm(
            self.batch(intermediate_output_names, op_group_size),
            desc="Running linear_quantize_activations on intermediate tensors batch-by-batch",
            unit=" batches",
        ):
            outputs = self.predict_intermediate_outputs(inputs, output_names, compute_units)

            model_outputs.update(
                {key: value for key, value in outputs.items() if key in model_output_names}
            )
            intermediate_outputs = {
                key: value for key, value in outputs.items() if key not in model_output_names
            }

            for output_name, output_value in outputs.items():
                self.record_intermediate_output(output_value, output_name, activation_stats_dict)
