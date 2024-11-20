# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import os
import platform
import shutil
import tempfile
from collections import Counter

import numpy as np
import pytest

import coremltools as ct
from coremltools._deps import _HAS_TORCH
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil import mil
from coremltools.converters.mil.mil import Function, get_new_symbol, types
from coremltools.converters.mil.testing_utils import get_op_types_in_program
from coremltools.models.compute_device import (
    MLComputeDevice,
    MLCPUComputeDevice,
    MLNeuralEngineComputeDevice,
)
from coremltools.models.compute_plan import MLComputePlan, MLModelStructure

if _HAS_TORCH:
    import torch


class TestMILExamples:
    @staticmethod
    def test_tutorial():
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 100, 100, 3))]
        )
        def prog(x):
            x = mb.relu(x=x, name="relu")
            x = mb.transpose(x=x, perm=[0, 3, 1, 2], name="transpose")
            x = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=False, name="reduce")
            x = mb.log(x=x, name="log")
            y = mb.add(x=1, y=2)
            return x

        # Convert and verify
        mlmodel = ct.convert(prog)

        # running predict() is only supported on macOS
        if ct.utils._is_macos():
            prediction = mlmodel.predict(
                {"x": np.random.rand(1, 100, 100, 3).astype(np.float32)}
            )
            assert len(prediction) == 1


@pytest.mark.skipif(ct.utils._macos_version() < (10, 15), reason='Model produces specification 4.')
class TestInputs:
    @staticmethod
    @pytest.mark.skipif(not ct.utils._is_macos(), reason="Platform is not Mac OS")
    @pytest.mark.parametrize(
        "convert_to",
        ["mlprogram", "neuralnetwork"],
    )
    def test_unsanitized_input_name_during_prediction(convert_to):
        '''
        input name : "x/0" becomes "x_0" due to name sanitization applied during conversion
        '''
        prog = mil.Program()
        func_inputs = {"x/0": mb.placeholder(shape=[2, 3]),
                       "y": mb.placeholder(shape=[2, 3])}
        with Function(func_inputs) as ssa_fun:
            x, y = ssa_fun.inputs["x/0"], ssa_fun.inputs["y"]
            x = mb.relu(x=x, name="relu")
            z = mb.add(x=x, y=y, name="out")
            ssa_fun.set_outputs([z])
        prog.add_function("main", ssa_fun)

        mlmodel = ct.convert(prog, convert_to=convert_to)

        with pytest.raises(KeyError) as error_info:
            mlmodel.predict(
                {"x/0": np.random.rand(2, 3).astype(np.float32),
                 "y": np.random.rand(2, 3).astype(np.float32)}
            )
        error_str = str(error_info.value)
        assert "does not match any of the model input" in error_str

    @staticmethod
    def _test_variant_input_type_prediction(to_tensor, convert_to):
        prog = mil.Program()
        func_inputs = {"x": mb.placeholder(shape=[2, 3]),
                       "y": mb.placeholder(shape=[2, 3])}
        with Function(func_inputs) as ssa_fun:
            x, y = ssa_fun.inputs["x"], ssa_fun.inputs["y"]
            x = mb.relu(x=x, name="relu")
            z = mb.add(x=x, y=y, name="out")
            ssa_fun.set_outputs([z])
        prog.add_function("main", ssa_fun)

        mlmodel = ct.convert(prog, convert_to=convert_to)
        x_numpy = np.random.rand(2, 3)
        y_numpy = np.random.rand(2, 3)
        out_by_numpy = mlmodel.predict(
            {"x": x_numpy,
             "y": y_numpy}
        )
        out_by_tensor = mlmodel.predict(
            {"x": to_tensor(x_numpy),
             "y": to_tensor(y_numpy)}
        )
        np.allclose(out_by_numpy["out"], out_by_tensor["out"])

    @staticmethod
    @pytest.mark.skipif(not ct.utils._is_macos(), reason="test needs predictions")
    @pytest.mark.parametrize(
        "convert_to",
        ["mlprogram", "neuralnetwork"],
    )
    def test_list_predict_input(convert_to):
        TestInputs._test_variant_input_type_prediction(lambda x: x.tolist(), convert_to)

    @staticmethod
    def test_rank0_inputs_mil():
        with pytest.raises(ValueError, match=r"Rank-0"):
            @mb.program(
                input_specs=[
                    mb.TensorSpec(shape=()),
                ]
            )
            def prog(x):
                return x


###############################################################################
# Note: all tests are examples of conversion to the Core ML format
# Each test case is expected to be runnable and self-complete.
###############################################################################

class TestMLProgramConverterExamples:
    @staticmethod
    @pytest.mark.skipif(
        ct.utils._macos_version() < (15, 0), reason="Tests are for deployment target iOS18/macos15"
    )
    def test_build_stateful_model():
        @mb.program(
            input_specs=[
                mb.TensorSpec((1,), dtype=types.fp16),
                mb.StateTensorSpec((1,), dtype=types.fp16),
            ],
        )
        def prog(x, accumulator_state):
            # Read state
            accumulator_value = mb.read_state(input=accumulator_state)
            # Update value
            y = mb.add(x=x, y=accumulator_value, name="y")
            # Write state
            mb.coreml_update_state(state=accumulator_state, value=y)

            return y

        mlmodel = ct.convert(prog, minimum_deployment_target=ct.target.iOS18)

        # try to run prediction on the stateful model
        state = mlmodel.make_state()
        assert mlmodel.predict({"x": np.array([1.0])}, state=state)["y"] == 1
        assert mlmodel.predict({"x": np.array([1.0])}, state=state)["y"] == 2

    @staticmethod
    def test_model_save(tmpdir):
        save_path_dir = str(tmpdir)

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.square(x=x)
            return x

        # save neuralnetwork model without extension and check that it is saved with
        # mlmodel extension
        mlmodel = ct.convert(prog, convert_to="neuralnetwork")
        mlmodel_path = os.path.join(save_path_dir, "model_nn")
        mlmodel.save(mlmodel_path)
        assert os.path.exists(mlmodel_path + ".mlmodel")

        # save neuralnetwork model with mlpackage extension
        mlmodel_path = os.path.join(save_path_dir, "model_nn2.mlpackage")
        mlmodel.save(mlmodel_path)
        assert os.path.exists(mlmodel_path)

        # save mlprogram model without extension and check that it is saved with
        # mlpackage extension
        mlmodel = ct.convert(prog, convert_to="mlprogram")
        mlmodel_path = os.path.join(save_path_dir, "model_mlprogram")
        mlmodel.save(mlmodel_path)
        assert os.path.exists(mlmodel_path + ".mlpackage")

        # check error if mlprogram is saved with mlmodel extension
        mlmodel_path = os.path.join(save_path_dir, "model_mlprogram.mlmodel")
        expected_pattern = "For an ML Program\, extension must be \.mlpackage \(not \.mlmodel\)\. Please see .* to see the difference between neuralnetwork and mlprogram model types\."
        with pytest.raises(Exception, match=expected_pattern):
            mlmodel.save(mlmodel_path)

    @staticmethod
    @pytest.mark.skipif(not ct.utils._is_macos(), reason="Platform is not Mac OS")
    def test_deepcopy_error_with_symbols_in_prog():
        prog = mil.Program()
        func_inputs = {"x": mb.placeholder(shape=[get_new_symbol(), 3]),
                       "y": mb.placeholder(shape=[2, 3])}
        with Function(func_inputs) as ssa_fun:
            x, y = ssa_fun.inputs["x"], ssa_fun.inputs["y"]
            x = mb.relu(x=x)
            z = mb.add(x=x, y=y)
            ssa_fun.set_outputs([z])
        prog.add_function("main", ssa_fun)
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)
        prog2 = mlmodel._get_mil_internal()  # this will invoke a deepcopy on the prog

    @pytest.mark.skipif(not ct.utils._is_macos(), reason="Platform is not Mac OS")
    @pytest.mark.parametrize("skip_model_load", [True, False])
    def test_model_load_skip_flag(self, skip_model_load):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3,)), ])
        def prog(x):
            return mb.relu(x=x, name='relu')

        if ct.utils._macos_version() < (12, 0) and not skip_model_load:
            # converting to mlprogram, on macOS < 12
            # should raise a runtime error when skip_model_load is False
            with pytest.warns(RuntimeWarning):
                model = ct.convert(prog, convert_to="mlprogram", skip_model_load=skip_model_load)
        else:
            model = ct.convert(prog, convert_to="mlprogram", skip_model_load=skip_model_load)

        assert model is not None
        if skip_model_load:
            assert model.__proxy__ is None
        model_dir = tempfile.TemporaryDirectory()
        filename = os.path.join(model_dir.name, "test.mlpackage")
        model.save(filename)
        assert os.path.exists(filename)


@pytest.mark.skipif(ct.utils._macos_version() < (12, 0), reason='Model produces specification 6.')
class TestMLProgramFP16Transform:
    @staticmethod
    def test_compute_precision_api():
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.square(x=x)
            return x

        mlmodel = ct.convert(copy.deepcopy(prog),
                             compute_precision=ct.precision.FLOAT16,
                             convert_to='mlprogram')
        mil_prog = mlmodel._get_mil_internal()
        np.testing.assert_array_equal(["cast", "square", "cast"], get_op_types_in_program(mil_prog))

        mlmodel = ct.convert(copy.deepcopy(prog),
                             compute_precision=ct.precision.FLOAT32,
                             convert_to='mlprogram')
        mil_prog = mlmodel._get_mil_internal()
        np.testing.assert_array_equal(["square"], get_op_types_in_program(mil_prog))

        mlmodel = ct.convert(
            copy.deepcopy(prog),
            compute_precision=ct.transform.FP16ComputePrecision(
                op_selector=lambda op: op.op_type != "square"
            ),
            convert_to="mlprogram",
        )
        mil_prog = mlmodel._get_mil_internal()
        np.testing.assert_array_equal(["square"], get_op_types_in_program(mil_prog))

        with pytest.raises(ValueError) as e:
            mlmodel = ct.convert(copy.deepcopy(prog),
                                 compute_precision='fp64',
                                 convert_to='mlprogram')
        expected_error = "'compute_precision' must be either coremltools.precision.FLOAT32 or " \
                         "coremltools.precision.FLOAT16 or of type coremltools.transform.FP16ComputePrecision()"
        assert expected_error == str(e.value)

        expected_pattern = "compute_precision .* supported .* mlprogram .* None .* target=='neuralnetwork'.*minimum_deployment_target.*"
        with pytest.raises(ValueError, match=expected_pattern) as e:
            mlmodel = ct.convert(
                copy.deepcopy(prog), convert_to="neuralnetwork", compute_precision="fp16"
            )

    @staticmethod
    def test_invalid_argument_nn_backend():
        '''
        Since the  compute_precision argument is only applicable when converting to "mlprogram",
        check that an error is correctly raised when conversion is targeted at the neuralnetwork backend
        '''

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.square(x=x)
            return x

        expected_err_str = "compute_precision is only supported for mlprogram target and must be None if target.*"
        with pytest.raises(ValueError, match=expected_err_str):
            mlmodel = ct.convert(
                prog, convert_to="neuralnetwork", compute_precision=ct.precision.FLOAT16
            )
        with pytest.raises(ValueError, match=expected_err_str):
            mlmodel = ct.convert(
                prog, convert_to="neuralnetwork", compute_precision=ct.precision.FLOAT32
            )


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not found")
class TestGraphPassManagement:
    @staticmethod
    def _get_test_model():
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 8, 5, padding="same")
                self.bn1 = torch.nn.BatchNorm2d(8)
                self.linear1 = torch.nn.Linear(28 * 28 * 8, 5)
                self.alpha = 0.7

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.linear1(torch.flatten(x))
                x = torch.maximum(self.alpha * x, x)
                return x

        return TestModel().eval()

    def test_default_pipeline(self):
        model = self._get_test_model()
        example_input = torch.rand(1, 1, 28, 28)
        traced_model = torch.jit.trace(model, example_input)
        model_converted = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            convert_to="mlprogram",
            pass_pipeline=ct.PassPipeline(),
        )
        assert get_op_types_in_program(model_converted._get_mil_internal()) == [
            "cast",
            "conv",
            "reshape",
            "linear",
            "leaky_relu",
            "cast",
        ]

    def test_skip_pass(self):
        model = self._get_test_model()
        example_input = torch.rand(1, 1, 28, 28)
        traced_model = torch.jit.trace(model, example_input)

        model_converted = ct.convert(
            traced_model, inputs=[ct.TensorType(shape=example_input.shape)], convert_to="mlprogram"
        )
        assert get_op_types_in_program(model_converted._get_mil_internal()) == [
            "cast",
            "conv",
            "reshape",
            "linear",
            "leaky_relu",
            "cast",
        ]

        pipeline = ct.PassPipeline()
        pipeline.remove_passes(passes_names=["common::fuse_conv_batchnorm"])
        model_converted_with_skipped_passes = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            convert_to="mlprogram",
            pass_pipeline=pipeline,
        )
        assert get_op_types_in_program(model_converted_with_skipped_passes._get_mil_internal()) == [
            "cast",
            "conv",
            "batch_norm",
            "reshape",
            "linear",
            "leaky_relu",
            "cast",
        ]

    def test_skip_two_passes(self):
        model = self._get_test_model()
        example_input = torch.rand(1, 1, 28, 28)
        traced_model = torch.jit.trace(model, example_input)

        pipeline = ct.PassPipeline()
        pipeline.remove_passes(
            passes_names=["common::fuse_conv_batchnorm", "common::fuse_leaky_relu"]
        )
        model_converted_with_skipped_passes = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            convert_to="mlprogram",
            pass_pipeline=pipeline,
        )
        assert get_op_types_in_program(model_converted_with_skipped_passes._get_mil_internal()) == [
            "cast",
            "conv",
            "batch_norm",
            "reshape",
            "linear",
            "mul",
            "maximum",
            "cast",
        ]

    def test_skip_passes_in_different_pipelines(self):
        """
        Some passes exist in different pipelines. For example, const_elimination is in both main
        and backend pipelines. If the user want to skip the const_elimination pass, we want to make
        sure both pipelines skip that pass.
        """
        model = self._get_test_model()
        example_input = torch.rand(1, 1, 28, 28)
        traced_model = torch.jit.trace(model, example_input)

        pipeline = ct.PassPipeline()
        pipeline.remove_passes(passes_names=["common::const_elimination"])
        model_converted = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            convert_to="mlprogram",
            pass_pipeline=pipeline,
        )

        op_types = get_op_types_in_program(model_converted._mil_program, skip_const_ops=False)
        expected_counts = {
            "const": 26,
            "cast": 7,
            "conv": 1,
            "matmul": 1,
            "add": 1,
            "shape": 1,
            "slice_by_index": 2,
            "concat": 1,
            "reshape": 1,
            "leaky_relu": 1,
        }
        assert Counter(op_types) == expected_counts

    def test_empty_pipeline(self):
        model = self._get_test_model()
        example_input = torch.rand(1, 1, 28, 28)
        traced_model = torch.jit.trace(model, example_input)

        pipeline = ct.PassPipeline.EMPTY

        model_converted = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            convert_to="mlprogram",
            pass_pipeline=pipeline,
            # TODO (rdar://131396853) Re-enable model load
            skip_model_load=True,
        )
        assert get_op_types_in_program(model_converted._get_mil_internal()) == [
            "conv",
            "batch_norm",
            "shape",
            "slice_by_index",
            "slice_by_index",
            "concat",
            "cast",
            "reshape",
            "linear",
            "mul",
            "maximum",
        ]

    def test_pass_option_skip_ops_by_type(self):
        model = self._get_test_model()
        example_input = torch.rand(1, 1, 28, 28)
        traced_model = torch.jit.trace(model, example_input)

        pipeline = ct.PassPipeline()
        pipeline.set_options("common::add_fp16_cast", {"skip_ops_by_type": "conv,linear"})
        model_converted = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            convert_to="mlprogram",
            pass_pipeline=pipeline,
            # TODO (rdar://131396853) Re-enable model load
            skip_model_load=True,
        )
        # The fp16 cast is skipped for conv and linear as we specified them in the pass options.
        assert get_op_types_in_program(model_converted._get_mil_internal()) == [
            "conv",
            "cast",
            "reshape",
            "cast",
            "linear",
            "cast",
            "leaky_relu",
            "cast",
        ]

    def test_pass_option_skip_const_by_size(self):
        model = self._get_test_model()
        example_input = torch.rand(1, 1, 28, 28)
        traced_model = torch.jit.trace(model, example_input)

        model_converted_without_pipeline = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            convert_to="mlprogram",
        )

        pipeline = ct.PassPipeline()
        pipeline.set_options("common::const_elimination", {"skip_const_by_size": "1e8"})
        model_converted = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            convert_to="mlprogram",
            pass_pipeline=pipeline,
        )
        # When the threshold is set to 1e8, no var is skipped in const elimination.
        assert get_op_types_in_program(
            model_converted._get_mil_internal(), skip_const_ops=False
        ).count("const") == get_op_types_in_program(
            model_converted_without_pipeline._get_mil_internal(), skip_const_ops=False
        ).count(
            "const"
        )

        pipeline.set_options(
            "common::const_elimination", {"skip_const_by_size": "-1"}
        )
        model_converted = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=example_input.shape)],
            convert_to="mlprogram",
            pass_pipeline=pipeline,
        )
        # When the threshold -1, almost all vars (except scalars) are skipped in const elimination.
        assert (
            get_op_types_in_program(
                model_converted._get_mil_internal(), skip_const_ops=False
            ).count("const")
            == 25
        )

    def test_pass_unsupported_option(self):
        model = self._get_test_model()
        example_input = torch.rand(1, 1, 28, 28)
        traced_model = torch.jit.trace(model, example_input)

        pipeline = ct.PassPipeline()
        pipeline.set_options("common::fuse_conv_batchnorm", {"skip_ops_by_type": "conv,linear"})
        with pytest.raises(
            NotImplementedError,
            match="The graph pass `fuse_conv_batchnorm` doesn't support option `skip_ops_by_type`.",
        ):
            ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=example_input.shape)],
                convert_to="mlprogram",
                pass_pipeline=pipeline,
            )

    def test_pass_option_invalid_val(self):
        model = self._get_test_model()
        example_input = torch.rand(1, 1, 28, 28)
        traced_model = torch.jit.trace(model, example_input)

        pipeline = ct.PassPipeline()
        pipeline.set_options("common::const_elimination", {"skip_const_by_size": "dummy"})
        with pytest.raises(
            ValueError,
            match="Expected to get float threshold, but got `dummy` which cannot be converted to float",
        ):
            ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=example_input.shape)],
                convert_to="mlprogram",
                pass_pipeline=pipeline,
            )


@pytest.mark.skipif(
    ct.utils._macos_version() < (14, 0),
    reason="MLComputeDevice API is available for macos versions >= 14.0.",
)
class TestMLComputeDevice:
    def test_all_compute_devices(self):
        compute_devices = MLComputeDevice.get_all_compute_devices()
        assert len(compute_devices) > 0, "Expected at least one compute device to be available."
        cpu_compute_devices = list(
            filter(
                lambda compute_device: isinstance(compute_device, MLCPUComputeDevice),
                compute_devices,
            )
        )
        assert (
            len(cpu_compute_devices) == 1
        ), "Expected exactly one MLCPUComputeDevice to be present."

    def test_available_compute_devices(self):
        compute_devices = ct.models.MLModel.get_available_compute_devices()
        assert len(compute_devices) > 0, "Expected at least one compute device to be available."

    def test_neural_engine_core_count(self):
        compute_devices = MLComputeDevice.get_all_compute_devices()
        neural_engine_compute_devices = filter(
            lambda compute_device: isinstance(compute_device, MLNeuralEngineComputeDevice),
            compute_devices,
        )
        neural_engine_compute_device: MLNeuralEngineComputeDevice = next(
            neural_engine_compute_devices, None
        )
        if neural_engine_compute_device is not None:
            assert (
                neural_engine_compute_device.total_core_count > 0
            ), "Expected at least one NeuralEngine core to be available."


@pytest.mark.skipif(
    ct.utils._macos_version() < (14, 4),
    reason="MLModelStructure API is available for macos versions >= 14.4.",
)
class TestMLModelStructure:
    @staticmethod
    def _get_test_model(type: str):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20)), mb.TensorSpec(shape=(10, 20))])
        def prog(x, y):
            return (mb.add(x=x, y=y), mb.sub(x=x, y=y))

        mlmodel = ct.convert(copy.deepcopy(prog), convert_to=type)
        return mlmodel

    def test_mlprogram_structure(self):
        model = TestMLModelStructure._get_test_model(type="mlprogram")
        model_structure = MLModelStructure.load_from_path(model.get_compiled_model_path())
        assert model_structure.program is not None, "program must not be None."
        program = model_structure.program
        assert program.functions.get("main", None) is not None, "Expected main function."
        function = program.functions["main"]
        assert len(function.inputs) == 2
        block = program.functions["main"].block
        assert len(block.output_names) == 2
        assert block is not None, "Specialization block must not be None."
        VALID_OPERATORS = {"add", "sub", "cast", "const"}
        for operation in block.operations:
            assert (
                operation.operator_name in VALID_OPERATORS
            ), f"Expected operator to be one of {', '.join(VALID_OPERATORS)}, but got '{operation.operator_name}'."

    def test_neuralnetwork_structure(self):
        model = TestMLModelStructure._get_test_model(type="neuralnetwork")
        model_structure = MLModelStructure.load_from_path(model.get_compiled_model_path())
        assert model_structure.neuralnetwork is not None, "NeuralNetwork must not be None."
        neuralnetwork = model_structure.neuralnetwork
        VALID_OPERATORS = {"elementwise", "activation"}
        for layer in neuralnetwork.layers:
            assert (
                layer.type in VALID_OPERATORS
            ), f"Expected layer type to be one of {', '.join(VALID_OPERATORS)}, but got '{layer.type}'."


@pytest.mark.skipif(
    ct.utils._macos_version() < (14, 4),
    reason="MLComputePlan API is available for macos versions >= 14.4.",
)
class TestMLComputePlan:
    @staticmethod
    def _get_test_model(type: str):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20)), mb.TensorSpec(shape=(10, 20))])
        def prog(x, y):
            return (mb.add(x=x, y=y), mb.sub(x=x, y=y))

        mlmodel = ct.convert(copy.deepcopy(prog), convert_to=type)
        return mlmodel

    def test_mlprogram_compute_plan(self):
        model = TestMLModelStructure._get_test_model(type="mlprogram")
        compute_plan = MLComputePlan.load_from_path(
            model.get_compiled_model_path(),
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        assert compute_plan is not None, "Compute Plan must not be None."
        program = compute_plan.model_structure.program
        for operation in program.functions["main"].block.operations:
            if operation.operator_name in {"const", "cast"}:
                continue

            compute_device_usage = compute_plan.get_compute_device_usage_for_mlprogram_operation(
                operation=operation,
            )

            assert compute_device_usage is not None
            assert isinstance(compute_device_usage.preferred_compute_device, MLCPUComputeDevice)
            assert len(compute_device_usage.supported_compute_devices) > 0
            for compute_device in compute_device_usage.supported_compute_devices:
                assert isinstance(compute_device, MLComputeDevice)

            if platform.machine() == "x86_64":
                pytest.xfail("rdar://140167930 ([CI] Estimated cost assert fails on x86_64)")

            estimated_cost = compute_plan.get_estimated_cost_for_mlprogram_operation(
                operation=operation,
            )

            assert estimated_cost is not None
            assert estimated_cost.weight >= 0.0 and estimated_cost.weight <= 1.0

    def test_neuralnetwork_compute_plan(self):
        model = TestMLModelStructure._get_test_model(type="neuralnetwork")
        compute_plan = MLComputePlan.load_from_path(model.get_compiled_model_path())
        assert compute_plan is not None, "Compute Plan must not be None."
        neuralnetwork = compute_plan.model_structure.neuralnetwork
        for layer in neuralnetwork.layers:
            compute_device_usage = compute_plan.get_compute_device_usage_for_neuralnetwork_layer(
                layer=layer,
            )

            assert compute_device_usage is not None
            assert isinstance(compute_device_usage.preferred_compute_device, MLComputeDevice)
            assert len(compute_device_usage.supported_compute_devices) > 0
            for compute_device in compute_device_usage.supported_compute_devices:
                assert isinstance(compute_device, MLComputeDevice)


@pytest.mark.skipif(
    ct.utils._macos_version() < (13, 0),
    reason="MLModelAsset API is available for macos versions >= 13.0.",
)
class TestMLModelAsset:
    @staticmethod
    def _get_test_model(type: str) -> ct.models.MLModel:
        @mb.program(input_specs=[mb.TensorSpec(shape=(1,)), mb.TensorSpec(shape=(1,))])
        def prog(x, y):
            return mb.add(x=mb.square(x=x), y=mb.square(x=y))

        mlmodel = ct.convert(copy.deepcopy(prog), convert_to=type)
        return mlmodel

    @staticmethod
    def _get_test_model_with_weights(type: str) -> ct.models.MLModel:
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 500))])
        def linear_prog(input):
            W = mb.const(val=np.ones((100, 500), dtype=float), name="const_W")
            out = mb.linear(x=input, weight=W, name="output")
            return out

        # convert and save model on disk
        mlmodel = ct.convert(linear_prog, convert_to=type)
        return mlmodel

    def test_inmemory_model(self):
        model = TestMLModelAsset._get_test_model(type="mlprogram")
        model_spec = model.get_spec()
        spec_data = model_spec.SerializeToString()
        asset = ct.models.model.MLModelAsset.from_memory(spec_data=spec_data)
        assert asset is not None, "Asset must not be none."
        compiled_model = ct.models.CompiledMLModel.from_asset(asset=asset)
        assert compiled_model is not None, "Compiled model must not be none."
        result = model.predict(
            {
                "x": np.array([1.0]),
                "y": np.array([2.0]),
            }
        )
        value = next(iter(result.values()))
        assert np.allclose(value, np.array([5.0])), "Value must be 5.0."

    @staticmethod
    def _remove_path(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    @pytest.mark.skipif(
        ct.utils._macos_version() < (15, 0),
        reason="MLModelAsset blob mapping API is available for macos versions >= 15.0.",
    )
    def test_inmemory_model_blob_mapping(self):
        model = TestMLModelAsset._get_test_model_with_weights(type="mlprogram")
        weight_file_path = model.weights_dir + "/" + ct.utils._WEIGHTS_FILE_NAME
        with open(weight_file_path, "rb") as file:
            weights_data = file.read()
            model_spec = model.get_spec()
            spec_data = model_spec.SerializeToString()
            asset = ct.models.model.MLModelAsset.from_memory(
                spec_data=spec_data, blob_mapping={"weights/weight.bin": weights_data}
            )
            assert asset is not None, "Asset must not be none."
            compiled_model = ct.models.CompiledMLModel.from_asset(asset=asset)
            assert compiled_model is not None, "Compiled model must not be none."
            result = model.predict(
                {
                    "input": np.ones((4, 500), dtype=float),
                }
            )
            value = next(iter(result.values()))
            assert np.allclose(
                value, np.full((4, 100), 500.0, float)
            ), "Value must be close to 500.0."

        TestMLModelAsset._remove_path(model.package_path)
