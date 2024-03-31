# Copyright (c) 2022, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest

import coremltools as ct
from coremltools import ComputeUnit
from coremltools._deps import _HAS_TF_2, _HAS_TORCH
from coremltools.converters._converters_entry import _get_metadata_from_mlmodel
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.converter import mil_convert
from coremltools.converters.mil.frontend.milproto.load import \
    load as milproto_to_pymil
from coremltools.converters.mil.frontend.tensorflow.test.test_ops import \
    TestTensorArray
from coremltools.converters.mil.frontend.tensorflow.test.testing_utils import \
    run_compare_tf
from coremltools.converters.mil.mil.ops.tests.testing_utils import \
    compare_backend
from coremltools.converters.mil.testing_utils import (
    get_op_names_in_program,
    get_op_types_in_program
)

if _HAS_TORCH:
    import torch
    from coremltools.converters.mil.frontend.torch.test.test_torch_ops import \
        TestScriptedModels


def get_pymil_prog_from_mlmodel(mlmodel):
    model_spec = mlmodel.get_spec()
    return milproto_to_pymil(
        model_spec=model_spec,
        specification_version=model_spec.specificationVersion,
        file_weights_dir=mlmodel.weights_dir,
    )

def get_roundtrip_mlmodel(mlmodel):
    """
    This utility function does the following roundtrip conversion:

    mlprogram proto -> pymil program -> mlprogram model
    """
    pymil_prog = get_pymil_prog_from_mlmodel(mlmodel)

    # convert the pymil program to mlmodel
    model_spec = mlmodel.get_spec()
    roundtripped_mlmodel = mil_convert(
        pymil_prog,
        convert_to="mlprogram",
        convert_from="milinternal",
        compute_units=mlmodel.compute_unit,
        model_description=model_spec.description,
        specification_version=model_spec.specificationVersion,
    )

    # set MIL program attributes
    build_info = _get_metadata_from_mlmodel(mlmodel)
    roundtripped_mlmodel._set_build_info_mil_attributes(build_info)
    return roundtripped_mlmodel

def roundtrip_and_compare_mlmodel(mlmodel, input_dict):
    roundtripped_mlmodel =  get_roundtrip_mlmodel(mlmodel)
    expected_outputs = mlmodel.predict(input_dict)
    compare_backend(roundtripped_mlmodel, input_dict, expected_outputs)


class TestLoadAPIUsage:
    def test_mil_proto_to_pymil(self):
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
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_units=ct.ComputeUnit.CPU_ONLY)

        # Load MLModel back to PyMIL
        loaded_pymil_prog = get_pymil_prog_from_mlmodel(mlmodel)

        # Assert that loaded PyMIL prog matches with defined PyMIL prog
        if get_op_types_in_program(loaded_pymil_prog) != get_op_types_in_program(prog):
            raise AssertionError("Mismatch between defined PyMIL prog and loaded PyMIL prog")

    def test_mil_proto_to_pymil_with_version_handling(self):
        # This test makes sure the correct version of the op is picked up during mil_proto -> pymil conversion

        # iOS15 version program with iOS13 version topk
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 4, 4))], opset_version=ct.target.iOS15)
        def prog(x):
            x = mb.topk(x=x, k=1, axis=-1, ascending=True)
            return x

        iOS15_mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS15,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        iOS15_pymil_prog = get_pymil_prog_from_mlmodel(iOS15_mlmodel)
        topk_op = iOS15_pymil_prog.functions["main"].find_ops(op_type="topk")[0]
        assert not hasattr(topk_op, "sort")

        # iOS16 version program with iOS16 version topk
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 4, 4))], opset_version=ct.target.iOS16)
        def prog(x):
            x = mb.topk(x=x, k=1, axis=-1, ascending=True)
            return x

        iOS16_mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        iOS16_pymil_prog = get_pymil_prog_from_mlmodel(iOS16_mlmodel)
        topk_op = iOS16_pymil_prog.functions["main"].find_ops(op_type="topk")[0]
        assert hasattr(topk_op, "sort")

    def test_mil_proto_preserving_ops_name(self):
        # This test is checking the route source_model -> MIL -> mil_prot -> pymil is preserving the op name
        # Define a PyMIL program
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 100, 100)), ])
        def prog(x):
            # MIL operation takes named inputs (instead of positional inputs).
            # Here `name` argument is optional.
            x = mb.relu(x=x, name='i_am_relu')
            x = mb.conv(x=x, weight=np.random.rand(10, 3, 2, 2), name="i_am_conv")
            x = mb.transpose(x=x, perm=[0, 3, 1, 2], name='i_am_transpose')
            x = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=False, name='i_am_reduce_mean')
            x = mb.log(x=x, name='i_am_log')
            return x

        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_units=ct.ComputeUnit.CPU_ONLY)
        op_names = get_op_names_in_program(mlmodel._mil_program, skip_const_ops=False)

        prog = get_pymil_prog_from_mlmodel(mlmodel)
        new_op_names = get_op_names_in_program(prog, skip_const_ops=False)

        assert op_names == new_op_names

    def test_mil_uint16(self):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(2, 2, 3))],
            opset_version=ct.target.iOS17,
        )
        def prog(x):
            indices = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 0]]], dtype=np.uint16)
            res = mb.gather(x=x, indices=indices, axis=2, batch_dims=2)
            return res

        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=ct.target.iOS17,
        )
        loaded_pymil_prog = get_pymil_prog_from_mlmodel(mlmodel)
        assert get_op_types_in_program(loaded_pymil_prog) == get_op_types_in_program(prog)


@pytest.mark.skipif(ct.utils._macos_version() < (12, 0), reason="mlprogram predict available only on macOS12+")
class TestE2ENumericalCorrectness:
    @pytest.mark.skipif(not _HAS_TORCH, reason="requires torch")
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

    @pytest.mark.skipif(not _HAS_TORCH, reason="requires torch")
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

    @pytest.mark.skipif(not _HAS_TORCH, reason="requires torch")
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

    @pytest.mark.skipif(not _HAS_TORCH, reason="requires torch")
    def test_while_loop(self):
        model = TestScriptedModels.get_while_loop_model()
        model_spec = torch.jit.script(model)
        mlmodel = ct.convert(model_spec,
                             inputs=[ct.TensorType(name="data", shape=model.input_size, dtype=np.float32)],
                             convert_to="mlprogram",
                             compute_units=ComputeUnit.CPU_ONLY
        )
        input_values = {"data": np.array([10.])}
        roundtrip_and_compare_mlmodel(mlmodel, input_values)

    @pytest.mark.skipif(not _HAS_TORCH, reason="requires torch")
    def test_cond(self):
        model = TestScriptedModels.get_cond_model()
        model_spec = torch.jit.script(model)
        mlmodel = ct.convert(model_spec,
                             inputs=[ct.TensorType(name="data", shape=(1,), dtype=np.float32)],
                             convert_to="mlprogram",
                             compute_units=ComputeUnit.CPU_ONLY
        )
        roundtrip_and_compare_mlmodel(mlmodel, {"data": np.array([1.])})
        roundtrip_and_compare_mlmodel(mlmodel, {"data": np.array([11.])})

    @pytest.mark.skipif(_HAS_TF_2, reason="Fix and re-enable this test: rdar://76293949 (TF2 unit test InvalidArgumentError)")
    def test_list(self):
        model, inputs, outputs = TestTensorArray.get_dynamic_elem_shape_model()
        input_values = [np.random.rand(2, 3)]
        input_dict = dict(zip(inputs, input_values))
        _, mlmodel, _, _ = run_compare_tf(
            model,
            input_dict,
            outputs,
            compute_unit=ct.ComputeUnit.CPU_ONLY,
            backend=("mlprogram", "fp16")
        )
        roundtrip_and_compare_mlmodel(mlmodel, {"Placeholder": input_values[0]})
