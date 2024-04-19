#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from typing import Dict, List, Tuple

import torch

import coremltools as ct

from .torchscript_utils import torch_to_mil_types


def to_coreml_tensor_type(name: str, tensor: torch.Tensor) -> "ct.TensorType":
    # TODO: rdar://115845948 ([Executorch] Handle inputs of shapes with dynamic dimensions)
    return ct.TensorType(name=name, dtype=torch_to_mil_types[tensor.dtype], shape=tensor.shape)


def extract_inputs_from_exir_program(
    exported_program  # torch.export.ExportedProgram
) -> Tuple[
    List["ct.TensorType"],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]:
    """
    Extract "input"s from torch.export.ExportedProgram

    EXIR lifts constants also as inputs to easily delegate to different backends,
    so the extracted "input"s consist of both user inputs and constants

    EXIR has 3 types of constants:
    1. parameters (e.g. weight of torch.nn.Linear)
    2. buffers
    3. constants (e.g. torch.tensor([0]) inside a torch.nn.Module)

    Given:
        exported_program: torch.export.ExportedProgram
    Return:
        user_inputs: list of coremltools input tensor specifications
        lifted_parameters: dictionary mapping names to torch parameter tensors
        lifted_buffers: dictionary mapping names to torch buffer tensors
        lifted_constants: dictionary mapping names to torch constant tensors
    """
    # prepare placeholder nodes into a convenient dict
    placeholder_nodes = {}
    for node in exported_program.graph_module.graph.nodes:
        if node.op == "placeholder":
            placeholder_nodes[node.name] = node

    # prepare parameters into a convenient dict
    parameters = {}
    for name, parameter in zip(
        exported_program.graph_signature.parameters, exported_program.parameters()
    ):
        if not isinstance(parameter, torch.Tensor):
            raise NotImplementedError(
                f"Only torch.Tensor parameter handled yet, but got {type(parameter)}"
            )
        parameters[name] = parameter

    # prepare buffers into a convenient dict
    buffers = {}
    for name, buffer in zip(
        exported_program.graph_signature.buffers, exported_program.buffers()
    ):
        if not isinstance(buffer, torch.Tensor):
            raise NotImplementedError(
                f"Only torch.Tensor buffer handled yet, but got {type(buffer)}"
            )
        buffers[name] = buffer

    # loop over input specs and populate results
    user_inputs = []
    lifted_parameters = {}
    lifted_buffers = {}
    lifted_constants = {}
    for input_spec in exported_program.graph_signature.input_specs:
        if input_spec.kind == torch.export.graph_signature.InputKind.USER_INPUT:
            node = placeholder_nodes[input_spec.arg.name]
            assert (
                node.meta is not None and "val" in node.meta
            ), "placeholder torch.fx.Node must have metadata val"
            val = node.meta["val"]
            assert isinstance(val, torch.Tensor), "placeholder val must be a tensor or fake tensor"
            user_inputs.append(to_coreml_tensor_type(node.name, val))

        elif input_spec.kind == torch.export.graph_signature.InputKind.PARAMETER:
            lifted_parameters[input_spec.arg.name] = parameters[input_spec.target]

        elif input_spec.kind == torch.export.graph_signature.InputKind.BUFFER:
            # This is a workaround on mutable buffer: Core ML does not support stateful execution,
            # so ExecuTorch will pass mutable buffers as inputs/outputs to Core ML delegation,
            # then in-place copy Core ML outputs into buffers
            # On Core ML side, we do not have to do anything special with outputs,
            # but for inputs we will need to identify ExecuTorch lifted mutable buffers
            # as Core ML user inputs
            if input_spec.target in exported_program.graph_signature.buffers_to_mutate.values():
                user_inputs.append(
                    to_coreml_tensor_type(input_spec.arg.name, buffers[input_spec.target])
                )
            else:
                lifted_buffers[input_spec.arg.name] = buffers[input_spec.target]

        elif input_spec.kind == torch.export.graph_signature.InputKind.CONSTANT_TENSOR:
            lifted_constants[input_spec.arg.name] = exported_program.constants[input_spec.target]

        else:
            raise NotImplementedError(
                "Only 4 types of inputs handled yet: user input, parameter, buffer, constant. "
                f"But got {input_spec.kind}"
            )

    return user_inputs, lifted_parameters, lifted_buffers, lifted_constants
