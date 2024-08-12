#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools._deps import _HAS_EXECUTORCH, _HAS_TORCH_EXPORT_API
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.types.symbolic import any_symbolic
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import (
    assert_output_dtype,
    assert_prog_output_type,
    assert_spec_input_image_type,
    assert_spec_output_image_type,
    get_op_types_in_program,
    verify_prediction,
)
from coremltools.proto import FeatureTypes_pb2 as ft

torch = pytest.importorskip("torch")

from .testing_utils import TorchFrontend, export_torch_model_to_frontend

frontends = [TorchFrontend.TORCHSCRIPT]
if _HAS_TORCH_EXPORT_API or _HAS_EXECUTORCH:
    frontends.append(TorchFrontend.EXIR)

ALTER_FRONTEND = [False]
if _HAS_EXECUTORCH:
    ALTER_FRONTEND.append(True)


@pytest.fixture
def float16_buffer_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.tensor(np.array([7, 5, 6], dtype=np.float16)))

        def forward(self, x):
            x = x.type(torch.float16)
            self.state.mul_(x)
            self.state.add_(torch.tensor(np.array([1, 2, 3], dtype=np.float16)))
            return self.state * 9

    example_input = torch.randint(0, 100, (3,), dtype=torch.int32)
    return torch.jit.trace(Model().eval(), example_input)


@pytest.fixture
def float32_buffer_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.tensor(np.array([7, 5, 6], dtype=np.float32)))

        def forward(self, x):
            self.state.add_(x)
            return self.state * 5

    example_input = torch.randint(0, 100, (3,), dtype=torch.int32)
    return torch.jit.trace(Model().eval(), example_input)


@pytest.fixture
def float32_non_persistent_buffer_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "state", torch.tensor(np.array([7, 5, 6], dtype=np.float32)), persistent=False
            )

        def forward(self, x):
            self.state.add_(x)
            return self.state * 5

    example_input = torch.randint(0, 100, (3,), dtype=torch.int32)
    return torch.jit.trace(Model().eval(), example_input)


@pytest.fixture
def float32_buffer_not_returned_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state_1", torch.tensor(np.array([7, 5, 6], dtype=np.float32)))
            self.register_buffer("state_2", torch.tensor(np.array([7, 5, 6], dtype=np.float32)))

        def forward(self, x):
            self.state_1.add_(x)
            self.state_2.add_(x)
            return x

    example_input = torch.randint(0, 100, (3,), dtype=torch.int32)
    return torch.jit.trace(Model().eval(), example_input)


@pytest.fixture
def float32_buffer_not_returned_model_2():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state_1", torch.tensor(np.array([7, 5, 6], dtype=np.float32)))
            self.register_buffer("state_2", torch.tensor(np.array([7, 5, 6], dtype=np.float32)))

        def forward(self, x):
            self.state_1.add_(x)
            self.state_2.add_(x)
            self.state_1.add_(x)
            return x

    example_input = torch.randint(0, 100, (3,), dtype=torch.int32)
    return torch.jit.trace(Model().eval(), example_input)


@pytest.fixture
def float32_buffer_model_with_two_inputs():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.tensor(np.array([7, 5, 6], dtype=np.float32)))

        def forward(self, x, y):
            self.state.add_(x)
            self.state.add_(y)
            return self.state * 5

    example_input = [
        torch.randint(0, 100, (3,), dtype=torch.int32),
        torch.randint(0, 100, (3,), dtype=torch.int32),
    ]
    return torch.jit.trace(Model().eval(), example_input)


@pytest.fixture
def float32_buffer_model_two_inputs_two_states():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state_1", torch.tensor(np.array([1, 2, 3], dtype=np.float32)))
            self.register_buffer("state_2", torch.tensor(np.array([4, 5, 6], dtype=np.float32)))

        def forward(self, x, y):
            self.state_1.add_(x)
            self.state_2.add_(y)
            return self.state_1 * self.state_2

    example_input = [
        torch.randint(0, 100, (3,), dtype=torch.int32),
        torch.randint(0, 100, (3,), dtype=torch.int32),
    ]
    return torch.jit.trace(Model().eval(), example_input)


def float32_buffer_sequantial_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.tensor(np.array([7, 5, 6], dtype=np.float32)))

        def forward(self, x):
            res = self.state + 8
            self.state[0] = 9.0
            x = self.state * x
            self.state.mul_(self.state)
            self.state.sub_(x)
            return torch.relu(self.state)

    example_input = torch.randint(0, 100, (3,), dtype=torch.int32)
    return torch.jit.trace(Model().eval(), example_input)


@pytest.fixture
def float32_two_buffers_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state_1", torch.tensor(np.array([1, 2, 3], dtype=np.float32)))
            self.register_buffer("state_2", torch.tensor(np.array([4, 5, 6], dtype=np.float32)))

        def forward(self, x):
            v1 = self.state_2 - x
            self.state_2.mul_(self.state_1)
            self.state_1.mul_(v1)
            self.state_1.add_(self.state_2)
            return self.state_1 + x

    example_input = torch.randint(0, 100, (3,), dtype=torch.int32)
    return torch.jit.trace(Model().eval(), example_input)


@pytest.fixture
def rank4_input_model_with_buffer():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "state_1", torch.tensor(np.zeros((1, 3, 10, 20), dtype=np.float32))
            )

        def forward(self, x):
            x = x + 5.5
            self.state_1.add_(x)
            self.state_1[0, 0, 0, 0:1] = torch.tensor([1.0])
            return x

    example_input = torch.randint(0, 100, (1, 3, 10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)


@pytest.fixture
def rank4_grayscale_input_model_with_buffer():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "state_1", torch.tensor(np.zeros((1, 1, 10, 20), dtype=np.float32))
            )

        def forward(self, x):
            x = x + 5
            self.state_1.add_(x)
            self.state_1[0, 0, 0, 0:1] = torch.tensor([1.0])
            return x

    example_input = torch.randint(0, 100, (1, 1, 10, 20), dtype=torch.float32)
    return torch.jit.trace(Model().eval(), example_input)


@pytest.mark.skipif(
    ct.utils._macos_version() < (15, 0), reason="Tests are for deployment target iOS18/macos15"
)
class TestStateConversionAPI:
    @pytest.mark.parametrize(
        "compute_unit, frontend, alter_frontend",
        itertools.product(compute_units, frontends, ALTER_FRONTEND),
    )
    def test_state_model_api_example(self, compute_unit, frontend, alter_frontend):
        """
        Test the public API example.
        """
        if frontend == TorchFrontend.TORCHSCRIPT and alter_frontend:
            pytest.skip("Stateful conversion from torch.jit.script is not supported")

        class UpdateBufferModel(torch.nn.Module):
            def __init__(self):
                super(UpdateBufferModel, self).__init__()
                self.register_buffer("state_1", torch.tensor(np.array([0, 0, 0], dtype=np.float32)))

            def forward(self, x):
                # In place update of the model state
                self.state_1.mul_(x)
                return self.state_1 + 1.0

        source_model = UpdateBufferModel()
        source_model.eval()
        torch_model = export_torch_model_to_frontend(
            source_model,
            (torch.tensor([1, 2, 3], dtype=torch.float16),),
            frontend,
            use_scripting=alter_frontend,
            use_edge_dialect=alter_frontend,
        )

        mlmodel = ct.convert(
            torch_model,
            inputs=(None if frontend == TorchFrontend.EXIR else [ct.TensorType(shape=(3,))]),
            states=(
                None
                if frontend == TorchFrontend.EXIR
                else [ct.StateType(wrapped_type=ct.TensorType(shape=(3,)), name="state_1")]
            ),
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
            compute_units=compute_unit,
        )
        assert get_op_types_in_program(mlmodel._mil_program) == [
            "read_state",
            "mul",
            "coreml_update_state",
            "add",
        ]
        verify_prediction(mlmodel)

    @pytest.mark.parametrize(
        "compute_unit",
        compute_units,
    )
    def test_single_state_single_input(
        self, float32_buffer_model, float32_non_persistent_buffer_model, compute_unit
    ):
        """
        Tests for different combination of input dtypes.
        """

        def test_valid_prog(prog, expected_ops=None):
            block = prog.functions["main"]
            assert types.is_tensor(block.inputs["x"].sym_type)
            assert types.is_state(block.inputs["state_workaround"].sym_type)
            assert len(block.outputs) == 1
            assert types.is_tensor(block.outputs[0].sym_type)
            if expected_ops is None:
                expected_ops = [
                    "read_state",
                    "add",
                    "coreml_update_state",
                    "mul",
                ]
            assert get_op_types_in_program(prog) == expected_ops

        """
        fp32 state / input (default with compute_precision=fp32),
        with both persistent and non-persistent buffer.
        fp32 state is not supported through runtime.

        (%x: Tensor(fp32), %state: State(fp32)) -> {
            %read_state(fp32) = read_state(%state)
            %add(fp32) = add(%read_state, %x)
            %update(fp32) = coreml_update_state(%state, %add)
            %mul(fp32) = mul(%update, 5)
        } -> (%mul)
        """
        for model in [float32_buffer_model, float32_non_persistent_buffer_model]:
            prog = ct.convert(
                model,
                inputs=[
                    ct.TensorType(shape=(3,)),
                ],
                states=[
                    ct.StateType(
                        wrapped_type=ct.TensorType(
                            shape=(3,),
                        ),
                        name="state",
                    ),
                ],
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision.FLOAT32,
                convert_to="milinternal",
            )
            test_valid_prog(prog)
            block = prog.functions["main"]
            assert block.inputs["x"].sym_type.get_primitive() == types.fp32
            assert (
                block.inputs["state_workaround"].sym_type.wrapped_type().get_primitive()
                == types.fp32
            )
            assert block.outputs[0].dtype == types.fp32

        """
        fp16 state / input (user specify)

        (%x: Tensor(fp16), %state: State(fp16)) -> {
            %read_state(fp16) = read_state(%state)
            %add(fp16) = add(%read_state, %x)
            %update(fp16) = coreml_update_state(%state, %add)
            %mul(fp16) = mul(%update, 5)
        } -> (%mul)
        """
        mlmodel = ct.convert(
            float32_buffer_model,
            inputs=[
                ct.TensorType(shape=(3,), dtype=np.float16),
            ],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(3,),
                        dtype=np.float16,
                    ),
                    name="state",
                ),
            ],
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
            compute_units=compute_unit,
        )
        # check the pymil program
        prog = mlmodel._mil_program
        test_valid_prog(prog)
        block = prog.functions["main"]
        assert block.inputs["x"].sym_type.get_primitive() == types.fp16
        assert (
            block.inputs["state_workaround"].sym_type.wrapped_type().get_primitive() == types.fp16
        )
        assert block.outputs[0].dtype == types.fp16

        # check the mil proto
        mil = mlmodel.get_spec().mlProgram
        for function in mil.functions.values():
            for block in function.block_specializations.values():
                ops = list(block.operations)
                expected_ops = [
                    "read_state",
                    "add",
                    "write_state",
                    "read_state",
                    "const",
                    "mul",
                ]
                assert [val.type for val in ops] == expected_ops
                assert len(ops[2].outputs) == 0

        verify_prediction(mlmodel)

        """
        fp16 state / input (default with compute_precision=fp16)

        (%x: Tensor(fp16), %state: State(fp16)) -> {
            %read_state(fp16) = read_state(%state)
            %add(fp16) = add(%read_state, %x)
            %update(fp16) = coreml_update_state(%state, %add)
            %mul(fp16) = mul(%update, 5)
        } -> (%mul)
        """
        mlmodel = ct.convert(
            float32_buffer_model,
            inputs=[
                ct.TensorType(shape=(3,)),
            ],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(3,),
                    ),
                    name="state",
                ),
            ],
            minimum_deployment_target=ct.target.iOS18,
            compute_units=compute_unit,
        )
        prog = mlmodel._mil_program
        test_valid_prog(prog)
        block = prog.functions["main"]
        assert block.inputs["x"].sym_type.get_primitive() == types.fp16
        assert (
            block.inputs["state_workaround"].sym_type.wrapped_type().get_primitive() == types.fp16
        )
        assert block.outputs[0].dtype == types.fp16
        verify_prediction(mlmodel)


        """
        fp16 state and fp32 input

        (%x: Tensor(fp32), %state: State(fp16)) -> {
            %read_state(fp16) = read_state(%state)
            %x_cast(fp16) = cast(%x)
            %add(fp16) = add(%read_state, %x_cast)
            %update(fp16) = coreml_update_state(%state, %add)
            %mul(fp16) = mul(%update, 5)
        } -> (%mul)
        """
        mlmodel = ct.convert(
            float32_buffer_model,
            inputs=[
                ct.TensorType(shape=(3,), dtype=np.float32),
            ],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(3,), dtype=np.float16), name="state"
                ),
            ],
            minimum_deployment_target=ct.target.iOS18,
            compute_units=compute_unit,
        )
        prog = mlmodel._mil_program
        expected_ops = [
            "read_state",
            "cast",
            "add",
            "coreml_update_state",
            "mul",
        ]
        test_valid_prog(prog, expected_ops)
        block = prog.functions["main"]
        assert block.inputs["x"].sym_type.get_primitive() == types.fp32
        assert (
            block.inputs["state_workaround"].sym_type.wrapped_type().get_primitive() == types.fp16
        )
        assert prog.find_ops("cast")[0].x.op is None
        assert block.outputs[0].dtype == types.fp16
        verify_prediction(mlmodel)

        """
        fp32 state and fp16 input. This is a rare corner case that shouldn't
        happend often.
        fp32 state is not supported through runtime.

        (%x: Tensor(fp16), %state: State(fp32)) -> {
            %read_state(fp32) = read_state(%state)
            %read_state_cast(fp16) = cast(read_state)
            %add(fp16) = add(%read_state_casr, %x)
            %add_cast(fp32) = cast(%add)
            %update(fp32) = coreml_update_state(%state, %add_cast)
            %update_cast(fp16) = cast(%update)
            %mul(fp16) = mul(%update_cast, 5)
        } -> (%mul)
        """
        prog = ct.convert(
            float32_buffer_model,
            inputs=[
                ct.TensorType(shape=(3,), dtype=np.float16),
            ],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(3,), dtype=np.float32), name="state"
                ),
            ],
            minimum_deployment_target=ct.target.iOS18,
            convert_to="milinternal",
        )
        expected_ops = [
            "read_state",
            "cast",
            "add",
            "cast",
            "coreml_update_state",
            "cast",
            "mul",
        ]
        test_valid_prog(prog, expected_ops)
        block = prog.functions["main"]
        assert block.inputs["x"].sym_type.get_primitive() == types.fp16
        assert (
            block.inputs["state_workaround"].sym_type.wrapped_type().get_primitive() == types.fp32
        )
        assert prog.find_ops("cast")[0].x.op.op_type == "read_state"
        assert prog.find_ops("cast")[1].x.op.op_type == "add"
        assert prog.find_ops("cast")[2].x.op.op_type == "coreml_update_state"
        assert block.outputs[0].dtype == types.fp16

    @pytest.mark.parametrize(
        "compute_unit",
        compute_units,
    )
    def test_single_state_single_input_model_fp16(self, float16_buffer_model, compute_unit):
        """
        Tests conversion of a stateful torch model defined in fp16.
        This will be common in model with large size.
        """
        # fp16 state / input
        mlmodel = ct.convert(
            float16_buffer_model,
            inputs=[
                ct.TensorType(shape=(3,), dtype=np.float16),
            ],
            states=[
                ct.StateType(wrapped_type=ct.TensorType(shape=(3,), dtype=np.float16), name="state")
            ],
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
            compute_units=compute_unit,
        )
        prog = mlmodel._mil_program
        assert get_op_types_in_program(prog) == [
            "read_state",
            "mul",
            "coreml_update_state",
            "add",
            "coreml_update_state",
            "mul",
        ]
        verify_prediction(mlmodel)

        # force state / input to be fp32 (intented stress test)
        prog = ct.convert(
            float16_buffer_model,
            inputs=[
                ct.TensorType(shape=(3,), dtype=np.float32),
            ],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(3,), dtype=np.float32), name="state"
                ),
            ],
            minimum_deployment_target=ct.target.iOS18,
            convert_to="milinternal",
        )
        assert get_op_types_in_program(prog) == [
            "read_state",
            "cast",
            "cast",
            "mul",
            "cast",
            "coreml_update_state",
            "cast",
            "add",
            "cast",
            "coreml_update_state",
            "cast",
            "mul",
        ]


    @pytest.mark.parametrize(
        "compute_unit",
        compute_units,
    )
    def test_multiple_states_model(self, float32_two_buffers_model, compute_unit):
        """
        Tests for a model with multiple buffers.
        """
        mlmodel = ct.convert(
            float32_two_buffers_model,
            inputs=[
                ct.TensorType(shape=(3,)),
            ],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(3,),
                    ),
                    name="state_1",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(3,),
                    ),
                    name="state_2",
                ),
            ],
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
            compute_units=compute_unit,
        )
        prog = mlmodel._mil_program
        assert get_op_types_in_program(prog) == [
            "read_state",
            "sub",
            "read_state",
            "mul",
            "coreml_update_state",
            "mul",
            "coreml_update_state",
            "add",
            "coreml_update_state",
            "add",
        ]
        verify_prediction(mlmodel)

    def test_convert_buffer_model_without_state_type(self, float32_buffer_model):
        """
        If the users don't specify StateType for buffer states,
        they will be treated as const tensors.
        We should modify this unittest after we fix this radar:
        rdar://116489054 ([Infra] Have a more sophisticated handling for torch buffer state when not declared as StateType)
        """
        prog = ct.convert(
            float32_buffer_model,
            inputs=[
                ct.TensorType(shape=(3,)),
            ],
            minimum_deployment_target=ct.target.iOS17,
            convert_to="milinternal",
        )
        assert get_op_types_in_program(prog) == [
            "add",
            "mul",
        ]

    @pytest.mark.parametrize(
        "compute_unit",
        compute_units,
    )
    def test_tensor_state_inputs_interleave(
        self, float32_buffer_model_two_inputs_two_states, compute_unit
    ):
        """
        We allow the user to interleave tensor / state input types.
        """
        mlmodel = ct.convert(
            float32_buffer_model_two_inputs_two_states,
            inputs=[
                ct.TensorType(shape=(3,)),
                ct.TensorType(shape=(3,)),
            ],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(3,),
                    ),
                    name="state_1",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(3,),
                    ),
                    name="state_2",
                ),
            ],
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
            compute_units=compute_unit,
        )
        prog = mlmodel._mil_program
        assert get_op_types_in_program(prog) == [
            "read_state",
            "add",
            "coreml_update_state",
            "read_state",
            "add",
            "coreml_update_state",
            "mul",
        ]
        verify_prediction(mlmodel)

    def test_invalid_deployment_target_error_out(self, float32_buffer_model):
        """
        The conversion should error out if the user tries to convert it
        into deployment target < ioS18.
        """
        with pytest.raises(
            ValueError,
            match="State model is supported only >= iOS18. Please update the minimum_deployment_target to at least coremltools.target.iOS18",
        ):
            prog = ct.convert(
                float32_buffer_model,
                inputs=[
                    ct.TensorType(shape=(3,)),
                ],
                states=[
                    ct.StateType(
                        wrapped_type=ct.TensorType(
                            shape=(3,),
                        ),
                        name="state",
                    ),
                ],
                minimum_deployment_target=ct.target.iOS17,
            )

        with pytest.raises(
            ValueError,
            match="State model is supported only >= iOS18. Please update the minimum_deployment_target to at least coremltools.target.iOS18",
        ):
            prog = ct.convert(
                float32_buffer_model,
                inputs=[
                    ct.TensorType(shape=(3,)),
                ],
                states=[
                    ct.StateType(
                        wrapped_type=ct.TensorType(
                            shape=(3,),
                        ),
                        name="state",
                    ),
                ],
                convert_to="neuralnetwork",
            )

    def test_invalid_state_name_error_out(self, float32_buffer_model):
        """
        The conversion should error out if the user doesn't provide /
        or provides wrong name of the buffer
        """
        with pytest.raises(
            ValueError,
            match="StateType named None not provided or not found in the source torch model. Please make sure the name in 'ct.StateType\(name=..., wrapped_type=ct.TensorType\(...\)\)' match the 'named_buffers\(\)' in the source torch model.",
        ):
            prog = ct.convert(
                float32_buffer_model,
                inputs=[
                    ct.TensorType(shape=(3,)),
                ],
                states=[
                    ct.StateType(
                        wrapped_type=ct.TensorType(
                            shape=(3,),
                        )
                    ),
                ],
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision.FLOAT32,
                convert_to="milinternal",
            )

        with pytest.raises(
            ValueError,
            match="StateType named invalid not provided or not found in the source torch model. Please make sure the name in 'ct.StateType\(name=..., wrapped_type=ct.TensorType\(...\)\)' match the 'named_buffers\(\)' in the source torch model: \['state'\]",
        ):
            prog = ct.convert(
                float32_buffer_model,
                inputs=[
                    ct.TensorType(shape=(3,)),
                ],
                states=[
                    ct.StateType(wrapped_type=ct.TensorType(shape=(3,)), name="invalid"),
                ],
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision.FLOAT32,
                convert_to="milinternal",
            )

    def test_invalid_state_shape_out(self, float32_buffer_model):
        """
        The conversion should error out if the provided StateType has
        a different shape than the registered buffer.
        """
        with pytest.raises(
            ValueError,
            match="StateType shape \(2,\) must match the torch buffer shape \(3,\)",
        ):
            prog = ct.convert(
                float32_buffer_model,
                inputs=[
                    ct.TensorType(shape=(3,)),
                ],
                states=[
                    ct.StateType(
                        wrapped_type=ct.TensorType(
                            shape=(2,),
                        ),
                        name="state",
                    ),
                ],
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision.FLOAT32,
                convert_to="milinternal",
            )

    def test_invalid_input_numbers_error_out(self, float32_buffer_model_with_two_inputs):
        """
        The checking for the tensor inputs should not be affected by
        the new added StateType inputs
        """
        with pytest.raises(
            ValueError,
            match="Number of TorchScript inputs \(2\) must match the user provided inputs \(1\).",
        ):
            prog = ct.convert(
                float32_buffer_model_with_two_inputs,
                inputs=[
                    ct.TensorType(shape=(3,)),
                ],
                states=[
                    ct.StateType(
                        wrapped_type=ct.TensorType(
                            shape=(3,),
                        ),
                        name="state",
                    ),
                ],
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision.FLOAT32,
                convert_to="milinternal",
            )

    def test_invalid_inputs_contains_states_error_out(self, float32_buffer_model_with_two_inputs):
        """
        The checking for the inputs should not contain StateType.
        """
        with pytest.raises(
            ValueError,
            match="'inputs' cannot contain an instance of StateType",
        ):
            prog = ct.convert(
                float32_buffer_model_with_two_inputs,
                inputs=[
                    ct.TensorType(shape=(3,)),
                    ct.StateType(
                        wrapped_type=ct.TensorType(
                            shape=(3,),
                        ),
                        name="state",
                    ),
                ],
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision.FLOAT32,
                convert_to="milinternal",
            )

    @staticmethod
    def convert_state_model(model, backend, compute_unit=ct.ComputeUnit.CPU_ONLY):
        return ct.convert(
            model,
            inputs=[
                ct.TensorType(shape=(3,)),
            ],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(3,),
                    ),
                    name="state_1",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(3,),
                    ),
                    name="state_2",
                ),
            ],
            minimum_deployment_target=ct.target.iOS18,
            convert_to=backend,
            compute_units=compute_unit,
        )

    @staticmethod
    def check_state_model(mlmodel, expected_ops, run_prediction=True):
        mil = mlmodel.get_spec().mlProgram
        for function in mil.functions.values():
            for block in function.block_specializations.values():
                ops = list(block.operations)
                assert [val.type for val in ops] == expected_ops
        if run_prediction:
            verify_prediction(mlmodel)

    @pytest.mark.parametrize(
        "compute_unit",
        compute_units,
    )
    def test_state_ops_cannot_removed(
        self,
        float32_buffer_not_returned_model,
        float32_buffer_not_returned_model_2,
        compute_unit,
    ):
        """
        Check the coreml_update_state should not be removed by dead_code_elimination pass.
        """
        # Test case 1
        prog = self.convert_state_model(float32_buffer_not_returned_model, "milinternal")
        assert get_op_types_in_program(prog) == [
            "identity",
            "read_state",
            "add",
            "coreml_update_state",
            "read_state",
            "add",
            "coreml_update_state",
        ]
        mlmodel = self.convert_state_model(
            float32_buffer_not_returned_model, "mlprogram", compute_unit
        )
        expected_ops = [
            "identity",
            "read_state",
            "add",
            "write_state",
            "read_state",
            "add",
            "write_state",
        ]
        # This model is failing on CPU_AND_NE, which is tracked by
        # rdar://130912134 ([Bug][Stateful model][CPU_AND_NE] Stateful model fails to run with compute_units=CPU_AND_NE)
        run_prediction = compute_unit != ct.ComputeUnit.CPU_AND_NE
        self.check_state_model(mlmodel, expected_ops, run_prediction)

        # Test case 2
        prog = self.convert_state_model(float32_buffer_not_returned_model_2, "milinternal")
        assert get_op_types_in_program(prog) == [
            "identity",
            "read_state",
            "add",
            "coreml_update_state",
            "read_state",
            "add",
            "coreml_update_state",
            "add",
            "coreml_update_state",
        ]
        mlmodel = self.convert_state_model(
            float32_buffer_not_returned_model_2, "mlprogram", compute_unit
        )
        expected_ops = [
            "identity",
            "read_state",
            "add",
            "write_state",
            "read_state",
            "read_state",
            "add",
            "write_state",
            "add",
            "write_state",
        ]
        # This model is failing on CPU_AND_NE, which is tracked by
        # rdar://130912134 ([Bug][Stateful model][CPU_AND_NE] Stateful model fails to run with compute_units=CPU_AND_NE)
        run_prediction = compute_unit != ct.ComputeUnit.CPU_AND_NE
        self.check_state_model(mlmodel, expected_ops, run_prediction)

    @pytest.mark.parametrize(
        "compute_unit, dtype",
        itertools.product(
            compute_units,
            [np.float16, np.float32],
        ),
    )
    def test_single_state_single_input_sequential_model(self, compute_unit, dtype):
        """
        Tests for a model with a sequence of inplace ops.
        """

        def get_stateful_model():
            # fp32 state is not supported through runtime
            convert_to = "milinternal" if dtype == np.float32 else "mlprogram"
            compute_precision_mapping = {
                np.float16: ct.precision.FLOAT16,
                np.float32: ct.precision.FLOAT32,
            }
            model = ct.convert(
                float32_buffer_sequantial_model(),
                inputs=[
                    ct.TensorType(shape=(3,), dtype=dtype),
                ],
                states=[
                    ct.StateType(wrapped_type=ct.TensorType(shape=(3,), dtype=dtype), name="state"),
                ],
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=compute_precision_mapping[dtype],
                convert_to=convert_to,
                compute_units=compute_unit,
            )

            if dtype == np.float32:
                return None, model
            assert dtype == np.float16
            return model, model._mil_program

        mlmodel, prog = get_stateful_model()
        assert get_op_types_in_program(prog) == [
            "read_state",
            "slice_update",
            "coreml_update_state",
            "mul",
            "mul",
            "coreml_update_state",
            "sub",
            "coreml_update_state",
            "relu",
        ]

        if mlmodel is not None:
            verify_prediction(mlmodel)

    @pytest.mark.parametrize(
        "compute_unit",
        compute_units,
    )
    def test_color_input_with_buffer(self, rank4_input_model_with_buffer, compute_unit):
        mlmodel = ct.convert(
            rank4_input_model_with_buffer,
            inputs=[ct.ImageType(shape=(1, 3, 10, 20), color_layout=ct.colorlayout.RGB)],
            states=[ct.StateType(wrapped_type=ct.TensorType(shape=(1, 3, 10, 20)), name="state_1")],
            outputs=[ct.TensorType(dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS18,
            compute_units=compute_unit,
        )
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        verify_prediction(mlmodel)

    @pytest.mark.parametrize(
        "compute_unit",
        compute_units,
    )
    def test_color_output_with_buffer(self, rank4_input_model_with_buffer, compute_unit):
        # image input / image output
        mlmodel = ct.convert(
            rank4_input_model_with_buffer,
            inputs=[ct.ImageType(shape=(1, 3, 10, 20), color_layout=ct.colorlayout.BGR)],
            states=[ct.StateType(wrapped_type=ct.TensorType(shape=(1, 3, 10, 20)), name="state_1")],
            outputs=[ct.ImageType(color_layout=ct.colorlayout.RGB)],
            minimum_deployment_target=ct.target.iOS18,
            compute_units=compute_unit,
        )
        assert_spec_input_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.BGR)
        assert_spec_output_image_type(mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB)
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        verify_prediction(mlmodel)

        # tensor input / image output
        # check mlprogram can have image output, both static and dynamic case are tested
        for is_dynamic in [True, False]:
            shape = (
                ct.Shape((1, 3, ct.RangeDim(5, 10, default=10), ct.RangeDim(5, 20, default=20)))
                if is_dynamic
                else ct.Shape((1, 3, 10, 20))
            )
            mlmodel = ct.convert(
                rank4_input_model_with_buffer,
                inputs=[ct.TensorType(shape=shape, dtype=np.float32)],
                states=[
                    ct.StateType(wrapped_type=ct.TensorType(shape=(1, 3, 10, 20)), name="state_1")
                ],
                outputs=[ct.ImageType(name="output_image", color_layout=ct.colorlayout.RGB)],
                minimum_deployment_target=ct.target.iOS18,
                compute_units=compute_unit,
            )
            assert_spec_output_image_type(
                mlmodel._spec, expected_feature_type=ft.ImageFeatureType.RGB
            )
            assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
            if is_dynamic:
                assert any_symbolic(mlmodel._mil_program.functions["main"].outputs[0].shape)
            verify_prediction(mlmodel)

    @pytest.mark.parametrize(
        "compute_unit",
        compute_units,
    )
    def test_grayscale_input_with_buffer(
        self, rank4_grayscale_input_model_with_buffer, compute_unit
    ):
        # test with GRAYSCALE
        mlmodel = ct.convert(
            rank4_grayscale_input_model_with_buffer,
            inputs=[ct.ImageType(shape=(1, 1, 10, 20), color_layout=ct.colorlayout.GRAYSCALE)],
            states=[ct.StateType(wrapped_type=ct.TensorType(shape=(1, 1, 10, 20)), name="state_1")],
            outputs=[ct.TensorType(dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS18,
            compute_units=compute_unit,
        )
        assert_spec_input_image_type(
            mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE
        )
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp32")
        verify_prediction(mlmodel)

        # test with GRAYSCALE_FLOAT16
        mlmodel = ct.convert(
            rank4_grayscale_input_model_with_buffer,
            inputs=[
                ct.ImageType(shape=(1, 1, 10, 20), color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)
            ],
            states=[ct.StateType(wrapped_type=ct.TensorType(shape=(1, 1, 10, 20)), name="state_1")],
            outputs=[ct.TensorType(dtype=np.float16)],
            minimum_deployment_target=ct.target.iOS18,
            compute_units=compute_unit,
        )
        assert_spec_input_image_type(
            mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16
        )
        assert_output_dtype(mlmodel, expected_type_str="fp16")
        verify_prediction(mlmodel)

    @pytest.mark.parametrize(
        "compute_unit",
        compute_units,
    )
    def test_grayscale_output_with_buffer(
        self, rank4_grayscale_input_model_with_buffer, compute_unit
    ):
        # grayscale fp16 input and output
        mlmodel = ct.convert(
            rank4_grayscale_input_model_with_buffer,
            inputs=[
                ct.ImageType(shape=(1, 1, 10, 20), color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)
            ],
            states=[ct.StateType(wrapped_type=ct.TensorType(shape=(1, 1, 10, 20)), name="state_1")],
            outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
            minimum_deployment_target=ct.target.iOS18,
            compute_units=compute_unit,
        )
        assert_spec_input_image_type(
            mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16
        )
        assert_spec_output_image_type(
            mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16
        )
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp16")
        verify_prediction(mlmodel)

        # grayscale input and grayscale fp16 output
        mlmodel = ct.convert(
            rank4_grayscale_input_model_with_buffer,
            inputs=[ct.ImageType(shape=(1, 1, 10, 20), color_layout=ct.colorlayout.GRAYSCALE)],
            outputs=[ct.ImageType(color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
            minimum_deployment_target=ct.target.iOS18,
            compute_units=compute_unit,
        )
        assert_spec_input_image_type(
            mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE
        )
        assert_spec_output_image_type(
            mlmodel._spec, expected_feature_type=ft.ImageFeatureType.GRAYSCALE_FLOAT16
        )
        assert_prog_output_type(mlmodel._mil_program, expected_dtype_str="fp16")
        verify_prediction(mlmodel)
