# Copyright (c) 2022, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools import _SPECIFICATION_VERSION_IOS_18, ComputeUnit
from coremltools._deps import _HAS_TF_2, _HAS_TORCH
from coremltools.converters._converters_entry import _get_metadata_from_mlmodel
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.converter import mil_convert
from coremltools.converters.mil.frontend.milproto.load import load as milproto_to_pymil

if _HAS_TF_2:
    from coremltools.converters.mil.frontend.tensorflow.test.test_ops import TestTensorArray
    from coremltools.converters.mil.frontend.tensorflow.test.testing_utils import run_compare_tf

from coremltools.converters.mil.mil import Program, types
from coremltools.converters.mil.mil.ops.tests.testing_utils import compare_backend
from coremltools.converters.mil.testing_utils import (
    get_op_names_in_program,
    get_op_types_in_program,
)

if _HAS_TORCH:
    import torch

    from coremltools.converters.mil.frontend.torch.test.test_torch_ops import TestScriptedModels


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

    @pytest.mark.parametrize(
        "immediate_value, dtype",
        itertools.product(
            (True, False),
            (types.int4, types.uint4, types.int8, types.uint8),
        ),
    )
    def test_milproto_load_to_pymil_sub_byte(self, immediate_value: bool, dtype: types):
        """Test if value in milproto (especially sub-byte) could be corrected loaded into pymil."""
        dtype_range = types.type_mapping.builtin_to_range(dtype)
        data_val = [dtype_range.low, dtype_range.high]
        if immediate_value:
            # Tensors with less than 10 elements will be stored as immediate values.
            data = np.array(data_val).reshape((1, 2, 1))
        else:
            data = np.array(data_val * 20).reshape((1, 40, 1))

        offset_val = dtype_range.high if dtype.is_unsigned() else -1
        offset = np.array([offset_val]).reshape((1, 1, 1))

        np_dtype = types.nptype_from_builtin(dtype)

        @mb.program(input_specs=[], opset_version=ct.target.iOS18)
        def prog():
            return mb.constexpr_blockwise_shift_scale(
                data=data.astype(np_dtype),
                scale=np.array([4]).reshape((1, 1, 1)).astype(np.float16),
                offset=offset.astype(np_dtype),
            )

        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=ct.target.iOS18,
        )
        pymil_prog: Program = milproto_to_pymil(
            model_spec=mlmodel.get_spec(),
            specification_version=ct.target.iOS18,
            file_weights_dir=mlmodel.weights_dir,
        )
        assert get_op_types_in_program(pymil_prog) == get_op_types_in_program(prog)

        original_ops = mlmodel._mil_program.functions["main"].find_ops(
            op_type="constexpr_blockwise_shift_scale"
        )
        load_back_ops = pymil_prog.functions["main"].find_ops(
            op_type="constexpr_blockwise_shift_scale"
        )
        for (original_op, load_back_op) in zip(original_ops, load_back_ops):
            assert original_op.data.dtype == load_back_op.data.dtype
            assert original_op.offset.dtype == load_back_op.offset.dtype
            np.testing.assert_array_equal(original_op.data.val, load_back_op.data.val)
            np.testing.assert_array_equal(original_op.offset.val, load_back_op.offset.val)



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

    def test_list(self):
        pytest.xfail(
            "Fix and re-enable this test: rdar://76293949 (TF2 unit test InvalidArgumentError)"
        )
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


class TestStatefulModelLoad:
    @staticmethod
    def convert_and_load_back(prog):
        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
        )

        return milproto_to_pymil(
            mlmodel.get_spec(),
            specification_version=_SPECIFICATION_VERSION_IOS_18,
            file_weights_dir=mlmodel.weights_dir,
        )

    @staticmethod
    def check_update_prog(prog, output_name):
        # check i/o types
        assert len(prog.functions) == 1
        func = prog.functions["main"]

        assert len(func.inputs) == 2
        in_var = func.inputs["state_workaround"]
        assert types.is_state(in_var.sym_type)
        assert in_var.name == "state_workaround"
        assert in_var.shape == (2, 3)
        assert in_var.dtype == types.fp16

        in_var_2 = func.inputs["x"]
        assert in_var_2.name == "x"
        assert in_var_2.shape == (2, 3)
        assert in_var_2.dtype == types.fp16

        assert len(func.outputs) == 1
        out_var = func.outputs[0]
        assert out_var.name == output_name
        assert out_var.shape == (2, 3)
        assert out_var.dtype == types.fp16

        # check op
        get_op_types_in_program(prog) == ["coreml_update_state"]

    def test_load_read_state(self):
        @mb.program(
            input_specs=[
                mb.StateTensorSpec((2, 3), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            return mb.read_state(input=x, name="out")

        new_prog = self.convert_and_load_back(prog)

        # check i/o types
        assert len(new_prog.functions) == 1
        func = new_prog.functions["main"]

        assert len(func.inputs) == 1
        in_var = func.inputs["x"]
        assert types.is_state(in_var.sym_type)
        assert in_var.name == "x"
        assert in_var.shape == (2, 3)
        assert in_var.dtype == types.fp16

        assert len(func.outputs) == 1
        out_var = func.outputs[0]
        assert out_var.name == "out"
        assert out_var.shape == (2, 3)
        assert out_var.dtype == types.fp16

        # check op
        get_op_types_in_program(new_prog) == ["read_state"]

    def test_load_coreml_update_state(self):
        @mb.program(
            input_specs=[
                mb.StateTensorSpec((2, 3), dtype=types.fp16),
                mb.TensorSpec((2, 3), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(state, x):
            return mb.coreml_update_state(state=state, value=x, name="out")

        new_prog = self.convert_and_load_back(prog)
        self.check_update_prog(new_prog, "out")

    def test_load_coreml_update_state_singular(self):
        @mb.program(
            input_specs=[
                mb.StateTensorSpec((2, 3), dtype=types.fp16),
                mb.TensorSpec((2, 3), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(state, x):
            mb.coreml_update_state(state=state, value=x)
            return x

        new_prog = self.convert_and_load_back(prog)
        self.check_update_prog(new_prog, "x")

    def test_load_state_complex(self):
        @mb.program(
            input_specs=[
                mb.StateTensorSpec((2, 3), dtype=types.fp16),
                mb.TensorSpec((2, 3), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(state, x):
            read_state = mb.read_state(input=state)
            add = mb.add(x=read_state, y=np.float16([0.1]))
            value = mb.coreml_update_state(state=state, value=add)
            add = mb.add(x=value, y=x)
            mb.coreml_update_state(state=state, value=add)
            return add

        new_prog = self.convert_and_load_back(prog)
        assert get_op_types_in_program(new_prog) == [
            "read_state",
            "add",
            "coreml_update_state",
            "add",
            "coreml_update_state",
        ]
