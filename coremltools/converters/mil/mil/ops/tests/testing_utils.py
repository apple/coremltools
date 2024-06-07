#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import functools
from typing import Dict, List, Optional

import pytest

import coremltools as ct
from coremltools import _logger as logger
from coremltools.converters.mil import mil
from coremltools.converters.mil.input_types import TensorType
from coremltools.converters.mil.mil import Function, Placeholder
from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline
from coremltools.converters.mil.mil.types.symbolic import is_symbolic
from coremltools.converters.mil.testing_reqs import BackendConfig
from coremltools.converters.mil.testing_utils import (
    compare_backend,
    ct_convert,
    validate_minimum_deployment_target,
)

UNK_VARIADIC = "*s_unk"
UNK_SYM = "s_unk"


def mark_api_breaking(breaking_opset_version: ct.target):
    """
    The function is used to mark api breaking for MIL op unittests.
    For instance, if `test_op_1` is supposed to pass from iOS14 -> iOS16 and breaks starting from iOS17,
    we can use the following syntax:

    @makr_api_breaking(breaking_opsey_version=ct.target.iOS17)
    def test_op_1(self, backend, ...):
        pass

    Note that the test function must take `backend` with type of `BackendConfig` as an input.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            backend = kwargs.get("backend", None)
            if backend is None:
                raise ValueError(
                    f'Function {func} decorated with mark_api_breaking must takes "backend" as an input.'
                )
            if backend.opset_version >= breaking_opset_version:
                pytest.skip(f"The test is breaking at opset version {breaking_opset_version}.")
            return func(*args, **kwargs)
        return wrapper

    return decorator

def run_compare_builder(
    build,
    input_placeholders,
    input_values=None,
    expected_output_types=None,
    expected_outputs=None,
    compute_unit=ct.ComputeUnit.CPU_ONLY,
    frontend_only=False,
    backend: Optional[BackendConfig] = None,
    atol=1e-04,
    rtol=1e-05,
    inputs=None,
    also_compare_shapes=True,
    converter=ct.convert,
    pass_pipeline: Optional[PassPipeline] = None,
    pred_iters: Optional[int] = None,
):
    """
    Inputs:
        - build: python function taking input of Vars and returning Var or
          list[Var]. Each input argument in build must match a key in
          input_values / input_placeholders.

        - input_placeholders: str -> placeholder. It may not be an empty
                              dict as MLModel doesn't support function with
                              no input.

        - input_values: str -> np.array or PIL.Image. Keys must match those in
          input_placeholders.

        - expected_output_types: list[(shape, builtin_type)] or (shape,
          builtin_type).  None skips type inference validation.

        - compute_unit: Enum[ct.ComputeUnit]. Compute unit for the coreml model

        - expected_outputs: list[np.array] or np.array. Required iff
          frontend_only == False

        - frontend_only: True to test up to proto generation.

        - inputs: type of inputs (either None (defaults to tensor) or [ct.ImageType])

        - converter: function
            Reference to convert function to be used.
            Default: ct.convert

        - backend: A BackendConfig that specifies the compute backend, precision and minimum_deployment_target

        - pred_iters: Number of prediction to run the mlmodel. For a stateful model,
          each prediction can have different numerical results. Can only be provided when mlmodel is stateful.

    Returns:
        The converted mlmodel (MLModel), or Tuple[MLModel, MLState].
    """
    if backend is None:
        backend = BackendConfig(
            backend="neuralnetwork",
            precision="fp32",
            opset_version=ct.target.iOS14,
        )
    minimum_deployment_target = backend.opset_version
    backend = (backend.backend, backend.precision)

    validate_minimum_deployment_target(minimum_deployment_target, backend)

    if not isinstance(expected_output_types, list):
        expected_output_types = [expected_output_types]

    if expected_outputs is not None and not isinstance(expected_outputs, list):
        expected_outputs = [expected_outputs]

    prog = mil.Program()
    with Function(input_placeholders, opset_version=minimum_deployment_target) as ssa_func:
        output_vars = build(**ssa_func.inputs)
        if isinstance(output_vars, tuple):
            output_vars = list(output_vars)
        elif not isinstance(output_vars, list):
            output_vars = [output_vars]
        ssa_func.set_outputs(output_vars)
        prog.add_function("main", ssa_func)

    # get output names for output_vars
    output_names = [x.name for x in output_vars]

    # Validate type inference
    msg = (
        "Provided expected outputs types {} should match number of output"
        + " variables {}"
    )
    assert_msg = msg.format(len(expected_output_types), len(output_vars))
    assert len(output_vars) == len(expected_output_types), assert_msg

    for out_var, s in zip(output_vars, expected_output_types):
        # The output type will be casted by the `adjust_io_to_supported_types` pass, so we don't
        # check the output var dtype matching here.
        if UNK_VARIADIC in s[:-1]:
            msg = "Skip type checking for UNK_VARIADIC. Output shape: {} vs expected shape: {}"
            logger.debug(msg.format(out_var.shape, s[:-1]))
            continue
        expected_shape = s[:-1]
        msg = "Output {} shape: expect {}, got {}. Program:\n{}".format(
            out_var.name, expected_shape, out_var.shape, prog
        )
        # No more variadic here.
        if len(out_var.shape) != len(expected_shape):
            raise ValueError(msg)
        # replace UNK_SYM in out_var.shape.
        output_shape = [
            0 if es == UNK_SYM else os for os, es in zip(out_var.shape, expected_shape)
        ]
        expected_shape = [0 if es == UNK_SYM else es for es in expected_shape]
        # convert float etc to int.
        output_shape = [i if is_symbolic(i) else int(i) for i in output_shape]
        expected_shape = [i if is_symbolic(i) else int(i) for i in expected_shape]
        if output_shape != expected_shape:
            raise ValueError(msg)

    mlmodel = ct_convert(
        prog,
        converter=converter,
        source="milinternal",
        convert_to=backend,
        inputs=inputs,
        compute_units=compute_unit,
        minimum_deployment_target=minimum_deployment_target,
        pass_pipeline=pass_pipeline,
    )

    if frontend_only:
        return mlmodel

    state = mlmodel.make_state() if mlmodel._is_stateful() else None

    if pred_iters is not None:
        assert state is not None, "pred_iters can only be provided with stateful model."
    else:
        pred_iters = 1

    for i in range(pred_iters):
        # get the expected outputs from each prediction iteration
        outputs = None
        if expected_outputs is not None:
            outputs = expected_outputs if pred_iters == 1 else expected_outputs[i]
            assert len(output_vars) == len(outputs), (
                f"Provided expected_outputs {len(outputs)}"
                " should match number of output"
                f" variables {len(output_vars)}"
            )
            outputs = {name: val for name, val in zip(output_names, outputs)}

        # run the mlmodel and compare the output numerical
        compare_backend(
            mlmodel=mlmodel,
            input_key_values=input_values,
            expected_outputs=outputs,
            atol=atol,
            rtol=rtol,
            also_compare_shapes=also_compare_shapes,
            dtype=backend[1],
            state=state,
        )

    return mlmodel


def construct_inputs_from_placeholders(
    input_placeholders: Dict[str, Placeholder], upper_bound: int
) -> [List[TensorType]]:
    """Construct the `inputs` param from placeholders with upper_bound."""
    inputs: [List[TensorType]] = []
    for input_name, placeholder in input_placeholders.items():
        input_shape = [
            ct.RangeDim(upper_bound=upper_bound) if is_symbolic(shape) else shape
            for shape in placeholder.sym_shape
        ]
        input_tensor_type = TensorType(name=input_name, shape=input_shape)
        inputs.append(input_tensor_type)
    return inputs
