# Copyright (c) 2022, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest
import torch

import coremltools as ct
from coremltools import ComputeUnit
from coremltools.converters.mil.converter import mil_convert
from coremltools.converters.mil.frontend.milproto.load import load as milproto_to_pymil
from coremltools.converters.mil.frontend.torch.test.test_torch_ops import TestScriptedModels as _TestScriptedModels
from coremltools.converters.mil.frontend.tensorflow.test.test_ops import TestTensorArray as _TestTensorArray
from coremltools.converters.mil.frontend.tensorflow.test.testing_utils import run_compare_tf
from coremltools.converters.mil.mil.ops.tests.testing_utils import compare_backend
from coremltools.converters.mil.testing_utils import get_op_types_in_program
from coremltools.converters._converters_entry import _get_metadata_from_mlmodel


def roundtrip_and_compare_mlmodel(mlmodel, input_dict):
    model_spec = mlmodel.get_spec()
    if model_spec.WhichOneof("Type") != "mlProgram":
        raise ValueError("Only MIL proto based mlmodels can be loaded")

    program_spec = model_spec.mlProgram
    model_description = model_spec.description

    pymil_prog = milproto_to_pymil(
        program_spec=program_spec,
        specification_version=model_spec.specificationVersion,
        file_weights_dir=mlmodel.weights_dir,
    )
    roundtripped_mlmodel = mil_convert(
        pymil_prog,
        convert_to="mlprogram",
        convert_from="milinternal",
        compute_units=ComputeUnit.ALL,
        model_description=model_description,
    )

    # set MIL program attributes
    build_info = _get_metadata_from_mlmodel(mlmodel)
    roundtripped_mlmodel._set_build_info_mil_attributes(build_info)

    expected_outputs = mlmodel.predict(input_dict)
    compare_backend(roundtripped_mlmodel, input_dict, expected_outputs)


class TestLoadAPIUsage:
    def test_mil_proto_to_pymil(self):
        from coremltools.converters.mil import Builder as mb

        # Define a PyMIL program
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 100, 100)), ])
        def prog(x):
            # MIL operation takes named inputs (instead of positional inputs).
            # Here `name` argument is optional.
            x = mb.relu(x=x, name='relu')
            x = mb.conv(x=x, weight=np.random.rand(10, 3, 2, 2), name="conv")
            x = mb.transpose(x=x, perm=[0, 3, 1, 2], name='transpose')
            x = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=False, name='reduce')
            x = mb.log(x=x, name='log')
            return x

        # Convert it to MIL proto backed MLModel
        mlmodel = mil_convert(
            prog,
            convert_to="mlprogram",
            convert_from="milinternal",
            compute_units=ComputeUnit.ALL,
        )

        # Load MLModel back to PyMIL
        model_spec = mlmodel.get_spec()
        program_spec = model_spec.mlProgram
        loaded_pymil_prog = milproto_to_pymil(
            program_spec=program_spec,
            specification_version=model_spec.specificationVersion,
            file_weights_dir=mlmodel.weights_dir,
        )

        # Assert that loaded PyMIL prog matches with defined PyMIL prog
        if get_op_types_in_program(loaded_pymil_prog) != get_op_types_in_program(prog):
            raise AssertionError("Mismatch between defined PyMIL prog and loaded PyMIL prog")


@pytest.mark.skipif(ct.utils._macos_version() < (12, 0), reason="mlprogram predict available only on macOS12+")
class TestE2ENumericalCorrectness:
    def test_elu(self):
        inputs = [ct.TensorType(name="data", shape=(2, 3, 1))]
        input_data = [torch.rand(*i.shape.to_list()) for i in inputs]
        torchmodel = torch.jit.trace(torch.nn.ELU(inplace=False), input_data)

        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram",
                             compute_units=ComputeUnit.CPU_ONLY)
        input_values = {
            i.name: val.detach().numpy() for i, val in zip(inputs, input_data)
        }
        roundtrip_and_compare_mlmodel(mlmodel, input_values)

    def test_linear(self):
        inputs = [ct.TensorType(name="data", shape=(10, 2))]
        input_data = [torch.rand(*i.shape.to_list()) for i in inputs]
        torchmodel = torch.jit.trace(
            torch.nn.Linear(in_features=2, out_features=3, bias=True), input_data
        )

        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram",
                             compute_units=ComputeUnit.CPU_ONLY)
        input_values = {
            i.name: val.detach().numpy() for i, val in zip(inputs, input_data)
        }
        roundtrip_and_compare_mlmodel(mlmodel, input_values)

    def test_conv(self):
        inputs = [ct.TensorType(name="data", shape=(5, 10, 4, 4))]
        input_data = [torch.rand(*i.shape.to_list()) for i in inputs]
        torchmodel = torch.jit.trace(
            torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=4), input_data
        )

        mlmodel = ct.convert(torchmodel, inputs=inputs, convert_to="mlprogram",
                             compute_units=ComputeUnit.CPU_ONLY)
        input_values = {
            i.name: val.detach().numpy() for i, val in zip(inputs, input_data)
        }
        roundtrip_and_compare_mlmodel(mlmodel, input_values)

    def test_while_loop(self):
        model = _TestScriptedModels.get_while_loop_model()
        model_spec = torch.jit.script(model)
        mlmodel = ct.convert(model_spec,
                             inputs=[ct.TensorType(name="data", shape=model.input_size, dtype=np.float32)],
                             convert_to="mlprogram",
                             compute_units=ComputeUnit.CPU_ONLY
        )
        input_values = {"data": np.array([10.])}
        roundtrip_and_compare_mlmodel(mlmodel, input_values)

    def test_cond(self):
        model = _TestScriptedModels.get_cond_model()
        model_spec = torch.jit.script(model)
        mlmodel = ct.convert(model_spec,
                             inputs=[ct.TensorType(name="data", shape=(1,), dtype=np.float32)],
                             convert_to="mlprogram",
                             compute_units=ComputeUnit.CPU_ONLY
        )
        roundtrip_and_compare_mlmodel(mlmodel, {"data": np.array([1.])})
        roundtrip_and_compare_mlmodel(mlmodel, {"data": np.array([11.])})

    def test_list(self):
        model, inputs, outputs = _TestTensorArray.get_dynamic_elem_shape_model()
        input_values = [np.random.rand(2, 3)]
        input_dict = dict(zip(inputs, input_values))
        _, mlmodel, _, _ = run_compare_tf(
            model,
            input_dict, 
            outputs,
            use_cpu_for_conversion=True,
            backend=("mlprogram", "fp16")
        )
        roundtrip_and_compare_mlmodel(mlmodel, {"Placeholder": input_values[0]})
