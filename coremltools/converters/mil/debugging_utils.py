#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
from collections import OrderedDict
from typing import List, Optional

import coremltools as ct
from coremltools.converters.mil.frontend.milproto.load import load as milproto_to_pymil
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.models import MLModel


def extract_submodel(
        model: MLModel,
        outputs: List[str],
        inputs: Optional[List[str]] = None,
        function_name: str = "main"
    ) -> MLModel:
    """
    This utility function lets you extract a submodel from a Core ML model.

    For a neural network model, the function extracts only in-memory Core ML models.
    You should always call this function for a model directly from :py:class:`~coremltools.converters._converters_entry.convert`. It is not
    allowed to load the model from disk and then call this API.

    For an ML program model, both cases (in-memory and from disk) are supported.

    Parameters
    ----------
    model: MLModel
        The Core ML model from which the submodel is extracted.

    outputs: list[str]
        A list of names of Vars, which are the outputs of the extracted submodel.

    inputs: list[str] (Optional)
        A list of names of Vars, which are the inputs of the extracted submodel.
        If not provided, the inputs from the original model are used.

    function_name: str (Optional)
        Name of the function where the subgraph is extracted. Default is ``main``.

    Examples
    --------

    Neural network:

        >>> from coremltools.converters.mil.debugging_utils import extract_submodel
        >>> mlmodel = ct.convert(model, convert_to="neuralnetwork")
        >>> outputs = ["output_0", "output_1"]
        >>> submodel = extract_submodel(mlmodel, outputs)

    ML program:

        >>> from coremltools.converters.mil.debugging_utils import extract_submodel
        >>> mlmodel = ct.convert(model, convert_to="mlprogram")
        >>> outputs = ["output_0", "output_1"]
        >>>
        >>> # Directly extract model in memory
        >>> submodel = extract_submodel(mlmodel, outputs)
        >>>
        >>> # Extract model loaded from disk
        >>> mlmodel.save("model.mlpackage")
        >>> mlmodel = coremltools.model.models.MLModel("model.mlpackage")
        >>> submodel = extract_submodel(mlmodel, outputs)

    """
    def validate_inputs(func, input_vars):
        reachable_vars = set(input_vars)
        for op in func.operations:
            if op.op_type == "const":
                reachable_vars.add(op.outputs[0])

        for op in func.operations:
            if all([x in reachable_vars for x in op.inputs.values()]):
                reachable_vars.update(op.outputs)

        for out in func.outputs:
            if out not in reachable_vars:
                raise ValueError(f"output {output} not reachable from inputs")

    @block_context_manager
    def replace_inputs(func, input_vars):
        func_inputs = {}
        for input in input_vars:
            name = input.name
            func_inputs[name] = mb.placeholder(input.shape, dtype=input.dtype)
            func.replace_uses_of_var_after_op(
                anchor_op=input.op,
                old_var=input,
                new_var=func_inputs[name].outputs[0],
            )
        func._input_dict = OrderedDict()
        for k, v in func_inputs.items():
            v.set_name(k)
            func._input_dict[k] = v.outputs[0]

    if not isinstance(outputs, (list, tuple)):
        raise ValueError(f"outputs must be of type list/tuple. Got {type(outputs)}.")

    for output in outputs:
        if not isinstance(output, str):
            raise ValueError(f"outputs must be a list of str. Got element {output} with type {type(output)}.")
        if outputs.count(output) > 1:
            raise ValueError(f"outputs must be a list of unique elements. '{output}' occurs {outputs.count(output)} times.")

    model_spec = model.get_spec()
    backend = "mlprogram" if model_spec.WhichOneof("Type") == "mlProgram" else "neuralnetwork"
    if backend == "neuralnetwork":
        if model._mil_program is None:
            raise ValueError("NeuralNetwork model loaded from the disk is not supported by the extract_submodel util.")
        program = model._mil_program
    else:
        assert backend == "mlprogram"
        if model._mil_program is None:
            program = milproto_to_pymil(
                model_spec=model_spec,
                specification_version=model_spec.specificationVersion,
                file_weights_dir=model.weights_dir,
            )
        else:
            program = model._mil_program

    # extract subgraph
    prog = copy.deepcopy(program)
    func = prog.functions[function_name]
    vars = {}
    new_outputs = []
    for op in func.operations:
        for o in op.outputs:
            if o.name in outputs:
                new_outputs.append(o)
            vars[o.name] = o

    if len(outputs) != len(new_outputs):
        new_outputs_names = [o.name for o in new_outputs]
        outputs_not_found = [name for name in outputs if name not in new_outputs_names]
        raise ValueError(f"outputs {outputs_not_found} not found in the function.")

    func.set_outputs(new_outputs)

    # Clean up the graph
    PASS_REGISTRY["common::dead_code_elimination"](prog)

    # If the inputs are provided, we subtract the subgraph starting from them
    if inputs is not None:
        if not isinstance(inputs, (list, tuple)):
            raise ValueError(f"inputs must be of type list/tuple. Got {type(inputs)}.")

        input_vars = []
        for input in inputs:
            if not isinstance(input, str):
                raise ValueError(f"inputs must be a list of str. Got element {input} with type {type(input)}.")
            if inputs.count(input) > 1:
                raise ValueError(f"inputs must be a list of unique elements. '{input}' occurs {inputs.count(input)} times.")
            if input not in vars and input not in func.inputs:
                raise ValueError(f"input {input} not found in the function.")
            if input in vars:
                input_vars.append(vars[input])
            if input in func.inputs:
                input_vars.append(func.inputs[input])

        validate_inputs(func, input_vars)
        replace_inputs(func, input_vars)
        PASS_REGISTRY["common::dead_code_elimination"](prog)

    prog.skip_all_passes = True
    submodel = ct.convert(prog, convert_to=backend, compute_units=model.compute_unit)

    return submodel
