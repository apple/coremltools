# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import List

import numpy as np

import coremltools as ct


class OperationInfo:
    def __init__(self, spec):
        self.dependants = []
        self.dependencies = []
        outputs = dict([(output.name, output) for output in spec.outputs])
        self.outputs = outputs
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
    @classmethod
    def batch(cls, iterable, n=1):
        l = len(iterable)
        for index in range(0, l, n):
            yield iterable[index : min(index + n, l)]

    @classmethod
    def unique(cls, sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    @classmethod
    def split_list(cls, list):
        half = len(list) // 2
        return list[:half], list[half:]

    @classmethod
    def get_block_info(cls, block_name, block_spec):
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
    def get_function_info(cls, function_name, function_spec):
        blocks = {}
        for block_name, block_spec in function_spec.block_specializations.items():
            blocks[block_name] = cls.get_block_info(block_name, block_spec)

        return FunctionInfo(function_name, blocks, function_spec)

    @classmethod
    def get_program_info(cls, program_spec):
        functions = {}
        for function_name, function_spec in program_spec.functions.items():
            functions[function_name] = cls.get_function_info(function_name, function_spec)

        return ProgramInfo(functions, program_spec)

    @classmethod
    def get_model_info(cls, model):
        model_spec = model.get_spec()
        return ModelInfo(cls.get_program_info(model_spec.mlProgram), model_spec)

    @classmethod
    def populate_outputs(cls, output_names, all_operations, acc):
        if len(output_names) == 0:
            return
        next_output_names = []
        operations = [all_operations.get(output_name, None) for output_name in output_names]
        operations = [operation for operation in operations if operation is not None]
        acc.extend([output for operation in operations for output in operation.outputs.values()])
        prev_output_names = [
            output_name
            for operation in operations
            for dependency in operation.dependencies
            for output_name in dependency.outputs.keys()
        ]
        prev_output_names = cls.unique(prev_output_names)
        cls.populate_outputs(prev_output_names, all_operations, acc)

    @classmethod
    def get_all_outputs(cls, block_info):
        acc = []
        output_names = block_info.spec.outputs
        cls.populate_outputs(output_names, block_info.operations, acc)
        return acc

    @classmethod
    def get_any_function(cls, model_info):
        program_info = model_info.program_info
        function_name = list(program_info.functions.keys())[0]
        return program_info.functions[function_name]

    @classmethod
    def get_any_block(cls, model_info):
        function_info = cls.get_any_function(model_info)
        block_specialization_name = list(function_info.blocks.keys())[0]
        return function_info.blocks[block_specialization_name]

    @classmethod
    def clone_spec(cls, spec):
        spec_class = spec.__class__
        new_spec = spec_class()
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

    def __init__(self, model):
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

    def get_model_with_intermediate_outputs(
        self, intermediate_output_names, compute_units=ct.ComputeUnit.ALL
    ):
        model_key = frozenset(intermediate_output_names)
        model = self.__cached_models.get(model_key)
        if model is not None:
            # Found cached model.
            return model

        cloned_spec = self.__class__.clone_spec(self.model_info.spec)
        cloned_model_info = ModelInfo(
            ModelDebugger.get_program_info(cloned_spec.mlProgram), cloned_spec
        )
        cloned_spec.specificationVersion = max(self.model_info.spec.specificationVersion, 7)
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
            cloned_spec, weights_dir=self.weights_dir, compute_units=compute_units
        )

        self.__cached_models[model_key] = model

        return model

    def get_models_with_intermediate_outputs_safely(
        self, intermediate_output_names, compute_units=ct.ComputeUnit.ALL
    ):
        if len(intermediate_output_names) == 0:
            return []

        models = []
        output_names = [intermediate_output_names]
        while len(output_names) > 0:
            curr_output_names = output_names[0]
            del output_names[0]
            model = None
            try:
                # This could fail compilation
                model = self.get_model_with_intermediate_outputs(curr_output_names, compute_units)
            except ValueError as ex:
                print(
                    f"Failed to create model with intermediate outputs={intermediate_output_names}, error={ex}"
                )
                if len(curr_output_names) > 1:
                    print("Retrying")
                    # split in two and then retry
                    xs = self.__class__.split_list(curr_output_names)
                    output_names.insert(0, xs[1])
                    output_names.insert(0, xs[0])

            if model is not None:
                models.append(model)

        return models

    # Clears all cached models
    def clear_cached_models(self):
        self.__cached_models.clear()

    # The function will get called for each intermediate output, return `False` if you want to stop the enumeration otherwise `True`.
    def check_intermediate_output(output_value, output_name, operation, activation_stats_dict):
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
        return True

    def step(
        self,
        step_fn,
        inputs,
        activation_stats_dict,
        intermediate_output_names=None,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        batch_size=500,
    ):
        if intermediate_output_names is None:
            intermediate_output_names = self.get_intermediate_output_names()

        model_output_names = [output.name for output in self.__model_outputs]
        model_outputs = None

        batch_size = len(intermediate_output_names)
        for output_names in self.__class__.batch(intermediate_output_names, batch_size):
            models = self.get_models_with_intermediate_outputs_safely(output_names, compute_units)
            for model in models:
                outputs = model.predict(inputs)
                # cache model outputs
                if model_outputs is None:
                    model_outputs = {
                        key: value for key, value in outputs.items() if key in model_output_names
                    }
                # remove model outputs
                outputs = {
                    key: value for key, value in outputs.items() if key not in model_output_names
                }
                output_names = list(outputs.keys())
                for output_name in output_names:
                    output_value = outputs[output_name]
                    del outputs[output_name]
                    operation = self.block_info.operations.get(output_name, None)
                    if not step_fn(output_value, output_name, operation, activation_stats_dict):
                        return
                outputs = {}

            for (output_name, output_value) in model_outputs.items():
                operation = self.block_info.operations.get(output_name, None)
                if not step_fn(output_value, output_name, operation, activation_stats_dict):
                    return
