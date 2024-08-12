#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from typing import Dict, List, Tuple

import sympy
import torch

from coremltools import _logger as logger
from coremltools.converters.mil.input_types import RangeDim, TensorType
from coremltools.converters.mil.mil import types

from .utils import TORCH_DTYPE_TO_MIL_DTYPE


def _map_sympy_number_to_int(sympy_number: sympy.core.numbers.Number) -> int:
    MAX_DIM = 2147483647
    if sympy_number == sympy.oo or sympy_number > MAX_DIM:
        return MAX_DIM
    else:
        return int(sympy_number)


def _construct_ct_range_dim_from_torch_value_ranges(
    symbol_name: str,
    value_ranges,  # torch.utils._sympy.value_ranges.ValueRanges
) -> RangeDim:
    if value_ranges.is_bool:
        raise NotImplementedError("Only non-bool torch value range handled yet")

    lower = _map_sympy_number_to_int(value_ranges.lower)
    upper = _map_sympy_number_to_int(value_ranges.upper)
    return RangeDim(lower_bound=lower, upper_bound=upper, symbol=symbol_name)


def _construct_symbol_name_to_ct_range_dim_dict(
    exported_program,  # torch.export.ExportedProgram
) -> Dict[str, RangeDim]:
    symbol_name_to_ct_range_dim = {}
    for symbol, value_ranges in exported_program.range_constraints.items():
        symbol_name = str(symbol)
        symbol_name_to_ct_range_dim[symbol_name] = _construct_ct_range_dim_from_torch_value_ranges(
            symbol_name, value_ranges
        )
    return symbol_name_to_ct_range_dim


def _construct_ct_tensor_type_from_torch(
    name: str,
    tensor: torch.Tensor,
    symbol_name_to_ct_range_dim: Dict[str, RangeDim],
) -> TensorType:
    coreml_dtype = TORCH_DTYPE_TO_MIL_DTYPE[tensor.dtype]
    # TODO (rdar://115845792): Once we support user inputs, we can migrate this check to inputs validation
    if coreml_dtype == types.int16:
        coreml_dtype = types.int32
        logger.warning(
            f"Core ML does not support int16 input, so input {name} has been cast to int32"
        )

    shape = []
    for size in tensor.shape:
        size_str = str(size)
        if size_str in symbol_name_to_ct_range_dim:
            shape.append(symbol_name_to_ct_range_dim[size_str])
        else:
            shape.append(int(size))

    return TensorType(name=name, dtype=coreml_dtype, shape=shape)


def _construct_ct_input_types_from_torch_user_inputs(
    exported_program,  # torch.export.ExportedProgram
    torch_user_inputs: Dict[str, torch.Tensor],
) -> List[TensorType]:
    ct_input_types = []
    symbol_name_to_ct_range_dim = _construct_symbol_name_to_ct_range_dim_dict(exported_program)
    for name, tensor in torch_user_inputs.items():
        ct_input_type = _construct_ct_tensor_type_from_torch(
            name, tensor, symbol_name_to_ct_range_dim
        )
        ct_input_types.append(ct_input_type)
    return ct_input_types


def _extract_placeholders_from_exir_program(
    exported_program  # torch.export.ExportedProgram
) -> Dict[str, torch.fx.Node]:
    """
    Given:
        exported_program: torch.export.ExportedProgram
    Return:
        placeholders: dictionary mapping names to placeholder nodes
    """
    placeholders = {}
    for node in exported_program.graph_module.graph.nodes:
        if node.op == "placeholder":
            placeholders[node.name] = node
    return placeholders


def _extract_parameters_from_exir_program(
    exported_program,  # torch.export.ExportedProgram
) -> Dict[str, torch.Tensor]:
    """
    Given:
        exported_program: torch.export.ExportedProgram
    Return:
        parameters: dictionary mapping names to torch parameter tensors
    """
    parameters = {}
    for name, parameter in zip(
        exported_program.graph_signature.parameters, exported_program.parameters()
    ):
        if not isinstance(parameter, torch.Tensor):
            raise NotImplementedError(
                f"Only torch.Tensor parameter handled yet, but got {type(parameter)}"
            )
        parameters[name] = parameter
    return parameters


def _extract_buffers_from_exir_program(
    exported_program,  # torch.export.ExportedProgram
) -> Dict[str, torch.Tensor]:
    """
    Given:
        exported_program: torch.export.ExportedProgram
    Return:
        buffers: dictionary mapping names to torch buffer tensors
    """
    buffers = {}
    for name, buffer in zip(
        exported_program.graph_signature.buffers, exported_program.buffers()
    ):
        if not isinstance(buffer, torch.Tensor):
            raise NotImplementedError(
                f"Only torch.Tensor buffer handled yet, but got {type(buffer)}"
            )
        buffers[name] = buffer
    return buffers


def _extract_inputs_from_exir_program(
    exported_program,  # torch.export.ExportedProgram
) -> Tuple[List[TensorType], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, str],]:
    """
    Extract "input"s from torch.export.ExportedProgram

    For easy delegation to different backends,
    EXIR lifts constants and buffer references also as inputs,
    so we will extract all user inputs and constants and mutable buffers

    EXIR has 2 types of constants:
    1. parameters (e.g. weight of torch.nn.Linear)
    2. constants (e.g. torch.tensor([0]) inside a torch.nn.Module)

    We consider buffers that are not mutated also as constant,
    e.g. batch norm running mean and variance are constant during inference

    Given:
        exported_program: torch.export.ExportedProgram
    Return:
        user_inputs: List[ct.TensorType]
            list of coremltools input tensor specifications
        constants: Dict[str, torch.Tensor]
            dictionary mapping variable names to torch constant tensors
        buffers: Dict[str, torch.Tensor]
            dictionary mapping torch mutable buffer names to tensors
        input_name_to_source_buffer_name: Dict[str, str]
            dictionary mapping input variable names to underlying mutable buffer names,
            i.e. these input variables are "read" from mutable buffer

    PS: Here is an example of buffers and input_name_to_source_buffer_name. Consider a toy model
    ```
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.tensor([7, 5, 6], dtype=torch.float16))

        def forward(self, x):
            ...
    ```
    The EXIR program is
    ```
    class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: "f16[3]", arg1_1: "f16[3]"):
            ...

    Graph signature: ExportGraphSignature(input_specs=[InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='arg0_1'), target='state', persistent=True), ...
    ```
    We will have
    ```
    buffers = {"state": torch.tensor([7, 5, 6], dtype=torch.float16)}
    input_name_to_source_buffer_name = {"arg0_1": "state"}
    ```
    """
    # prepare placeholder nodes and parameters and buffers into convenient dicts
    placeholders = _extract_placeholders_from_exir_program(exported_program)
    parameters = _extract_parameters_from_exir_program(exported_program)
    buffers = _extract_buffers_from_exir_program(exported_program)

    # loop over input specs and populate results
    torch_user_inputs = {}
    constants = {}
    input_name_to_source_buffer_name = {}
    for input_spec in exported_program.graph_signature.input_specs:
        if input_spec.kind == torch.export.graph_signature.InputKind.USER_INPUT:
            node = placeholders[input_spec.arg.name]
            if node.meta is None or "val" not in node.meta:
                raise ValueError(
                    "Placeholder torch.fx.Node metadata val is required in Core ML conversion"
                )
            val = node.meta["val"]
            if not isinstance(val, torch.Tensor):
                raise NotImplementedError(
                    "Placeholder val must be a tensor or fake tensor, "
                    f"but got type {type(val)}, value {str(val)}"
                )
            torch_user_inputs[node.name] = val

        elif input_spec.kind == torch.export.graph_signature.InputKind.PARAMETER:
            constants[input_spec.arg.name] = parameters[input_spec.target]

        elif input_spec.kind == torch.export.graph_signature.InputKind.CONSTANT_TENSOR:
            constants[input_spec.arg.name] = exported_program.constants[input_spec.target]

        elif input_spec.kind == torch.export.graph_signature.InputKind.BUFFER:
            if input_spec.target in exported_program.graph_signature.buffers_to_mutate.values():
                input_name_to_source_buffer_name[input_spec.arg.name] = input_spec.target
            else:
                constants[input_spec.arg.name] = buffers.pop(input_spec.target)

        else:
            raise NotImplementedError(
                "Only 4 types of inputs handled yet: user input, parameter, constant, buffer. "
                f"But got {input_spec.kind}"
            )

    ct_input_types = _construct_ct_input_types_from_torch_user_inputs(
        exported_program, torch_user_inputs
    )

    return ct_input_types, constants, buffers, input_name_to_source_buffer_name


def _extract_outputs_from_exir_program(
    exported_program,  # torch.export.ExportedProgram
) -> Tuple[List[str], Dict[str, str],]:
    """
    Extract "outputs" from torch.export.ExportedProgram

    For easy delegation to different backends,
    EXIR lifts buffer mutations also as outputs,
    so we will extract all user outputs and buffer mutations

    Given:
        exported_program: torch.export.ExportedProgram
    Return:
        user_outputs: List[str]
            list of output names
        output_name_to_target_buffer_name: Dict[str, str]
            dictionary mapping output variable names to underlying mutable buffer names,
            i.e. these output variables are "written" to mutable buffer
    """
    user_outputs = []
    output_name_to_target_buffer_name = {}
    for output_spec in exported_program.graph_signature.output_specs:
        if output_spec.kind == torch.export.graph_signature.OutputKind.USER_OUTPUT:
            user_outputs.append(output_spec.arg.name)

        elif output_spec.kind == torch.export.graph_signature.OutputKind.BUFFER_MUTATION:
            output_name_to_target_buffer_name[output_spec.arg.name] = output_spec.target

        elif output_spec.kind == torch.export.graph_signature.OutputKind.USER_INPUT_MUTATION:
            raise ValueError(
                "Core ML cannot handle user input mutation, because Core ML distinguishes "
                "input (immutable) and state (mutable). You have 2 options:\n"
                "1. If mutation is intentional and necessary to carry over, then please "
                "replace input with buffer by your torch model.register_buffer then re-export\n"
                "2. If mutation is unnecessary, then please avoid it, e.g. by "
                "adding `input = input.clone()` at the 1st line of your torch model.forward method"
            )

        else:
            raise NotImplementedError(
                "Only 2 types of outputs handled yet: user output, buffer mutation. "
                f"But got {output_spec.kind}"
            )

    return user_outputs, output_name_to_target_buffer_name


def extract_io_from_exir_program(
    exported_program,  # torch.export.ExportedProgram
) -> Tuple[
    List[TensorType],
    List[str],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, str],
    Dict[str, str],
]:
    """
    Extract "input"s and "output"s from torch.export.ExportedProgram

    For easy delegation to different backends,
    EXIR lifts constants and buffer references also as inputs, buffer mutations also as outputs,
    so we will extract all user inputs / outputs, constants, buffers and their mutations

    Given:
        exported_program: torch.export.ExportedProgram
    Return:
        user_inputs: List[ct.TensorType]
            list of coremltools input tensor specifications
        user_outputs: List[str]
            list of output names
        constants: Dict[str, torch.Tensor]
            dictionary mapping variable names to torch constant tensors
        buffers: Dict[str, torch.Tensor]
            dictionary mapping torch mutable buffer names to tensors
        input_name_to_source_buffer_name: Dict[str, str]
            dictionary mapping input variable names to underlying mutable buffer names,
            i.e. these input variables are "read" from mutable buffer
        output_name_to_target_buffer_name: Dict[str, str]
            dictionary mapping output variable names to underlying mutable buffer names,
            i.e. these output variables are "written" to mutable buffer
    """
    (
        user_inputs,
        constants,
        buffers,
        input_name_to_source_buffer_name,
    ) = _extract_inputs_from_exir_program(exported_program)
    user_outputs, output_name_to_target_buffer_name = _extract_outputs_from_exir_program(
        exported_program
    )
    return (
        user_inputs,
        user_outputs,
        constants,
        buffers,
        input_name_to_source_buffer_name,
        output_name_to_target_buffer_name,
    )
