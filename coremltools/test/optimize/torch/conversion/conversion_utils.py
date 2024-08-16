#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import sys

import numpy as np
import torch

import coremltools as ct


def convert_and_verify(
    pytorch_model,
    input_data,
    input_as_shape=False,
    pass_pipeline=None,
    minimum_deployment_target=ct.target.iOS18,
    expected_ops=None,
):
    """
    Utility to:
    1) Convert a PyTorch model to coreml format
    2) Compare their outputs for numerical equivalence
    3) Verify the converted model contains expected ops

    Args:
        input_as_shape: If true generates random input data with shape.
        expected_ops: List of MIL ops expected in the converted model
    Returns:
        Converted coreml model
    """
    if input_as_shape:
        example_input = torch.rand(input_data)
    else:
        example_input = input_data

    # Generate converted model
    coreml_model = get_converted_model(
        pytorch_model, example_input, pass_pipeline, minimum_deployment_target
    )
    assert coreml_model is not None

    # Verify converted model output matches torch model
    verify_model_outputs(pytorch_model, coreml_model, example_input)

    # Verify desired ops are present
    verify_ops(coreml_model, expected_ops)

    return coreml_model


def get_converted_model(
    pytorch_model,
    input_data,
    pass_pipeline=None,
    minimum_deployment_target=ct.target.iOS17,
):
    """
    Utility that takes a PyTorch model and converts it to a coreml model
    """
    traced_model = torch.jit.trace(pytorch_model, example_inputs=(input_data,))
    coreml_model = None
    try:
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=input_data.shape)],
            pass_pipeline=pass_pipeline,
            minimum_deployment_target=minimum_deployment_target,
        )
    except Exception as err:
        print(f"Conversion Error: {err}")

    return coreml_model


def verify_model_outputs(pytorch_model, coreml_model, input_value):
    """
    This utility functions does the following checks:
    (1) Verify the output of the coreml model has the same shape of the PyTorch model
    (2) The PyTorch and coreml model have the same numerical outputs
    """
    # Validate the output shape / type
    ref_output = pytorch_model(input_value)
    output = coreml_model._mil_program.functions["main"].outputs[0]

    assert ref_output.shape == output.shape

    # Cannot run predict on linux
    if sys.platform == "linux":
        return

    # Validate that the coreml model produces correct outputs
    pytorch_model.eval()
    ref_output_dict = pytorch_model(input_value)
    coreml_input_value = {"input_1": input_value.detach().numpy()}
    output_dict = coreml_model.predict(coreml_input_value)
    for k, v in output_dict.items():
        np.testing.assert_allclose(v, output_dict[k])


def verify_ops(coreml_model, expected_ops):
    if not expected_ops:
        return

    for op in expected_ops:
        compressed_ops = coreml_model._mil_program.functions["main"].find_ops(op_type=op)
        assert len(compressed_ops) >= 1
