# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import os
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, get_new_symbol
from coremltools.converters.mil.testing_utils import get_op_types_in_program


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
    def test_unsanitized_input_name_during_prediction():
        '''
        input name : "x/0" becomes "x_0" due to name sanitization applied during conversion
        '''
        prog = Program()
        func_inputs = {"x/0": mb.placeholder(shape=[2, 3]),
                       "y": mb.placeholder(shape=[2, 3])}
        with Function(func_inputs) as ssa_fun:
            x, y = ssa_fun.inputs["x/0"], ssa_fun.inputs["y"]
            x = mb.relu(x=x, name="relu")
            z = mb.add(x=x, y=y, name="out")
            ssa_fun.set_outputs([z])
        prog.add_function("main", ssa_fun)

        mlmodel = ct.convert(prog)

        with pytest.raises(KeyError) as error_info:
            mlmodel.predict(
                {"x/0": np.random.rand(2, 3).astype(np.float32),
                 "y": np.random.rand(2, 3).astype(np.float32)}
            )
        error_str = str(error_info.value)
        assert "does not match any of the model input" in error_str

    @staticmethod
    def _test_variant_input_type_prediction(to_tensor):
        prog = Program()
        func_inputs = {"x": mb.placeholder(shape=[2, 3]),
                       "y": mb.placeholder(shape=[2, 3])}
        with Function(func_inputs) as ssa_fun:
            x, y = ssa_fun.inputs["x"], ssa_fun.inputs["y"]
            x = mb.relu(x=x, name="relu")
            z = mb.add(x=x, y=y, name="out")
            ssa_fun.set_outputs([z])
        prog.add_function("main", ssa_fun)

        mlmodel = ct.convert(prog)
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
    def test_list_predict_input():
        TestInputs._test_variant_input_type_prediction(lambda x: x.tolist())
        
    @staticmethod
    def test_rank0_inputs_mil():
        with pytest.raises(ValueError, match=r"Rank-0"):
            @mb.program(
                input_specs=[mb.TensorSpec(shape=()),]
            )
            def prog(x):
                return x

###############################################################################
# Note: all tests are examples of conversion to the Core ML format
# Each test case is expected to be runnable and self-complete.
###############################################################################

class TestMLProgramConverterExamples:

    @staticmethod
    def test_model_save(tmpdir):
        save_path_dir = str(tmpdir)

        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20))])
        def prog(x):
            x = mb.square(x=x)
            return x

        # save neuralnetwork model without extension and check that it is saved with
        # mlmodel extension
        mlmodel = ct.convert(prog)
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
        with pytest.raises(Exception) as e:
            mlmodel.save(mlmodel_path)
        expected_error = "For an ML Program, extension must be .mlpackage (not .mlmodel)"
        assert expected_error == str(e.value)

    @staticmethod
    @pytest.mark.skipif(not ct.utils._is_macos(), reason="Platform is not Mac OS")
    def test_deepcopy_error_with_symbols_in_prog():
        prog = Program()
        func_inputs = {"x": mb.placeholder(shape=[get_new_symbol(), 3]),
                       "y": mb.placeholder(shape=[2, 3])}
        with Function(func_inputs) as ssa_fun:
            x, y = ssa_fun.inputs["x"], ssa_fun.inputs["y"]
            x = mb.relu(x=x)
            z = mb.add(x=x, y=y)
            ssa_fun.set_outputs([z])
        prog.add_function("main", ssa_fun)
        mlmodel = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32)
        prog2 = mlmodel._get_mil_internal() # this will invoke a deepcopy on the prog

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
                model = ct.convert(prog, convert_to='mlprogram',
                                   skip_model_load=skip_model_load)
        else:
            model = ct.convert(prog, convert_to='mlprogram',
                                skip_model_load=skip_model_load)

        assert model is not None
        if skip_model_load:
            assert model.__proxy__ is None
        model_dir = mkdtemp()
        filename = os.path.join(model_dir, 'test.mlpackage')
        model.save(filename)
        assert os.path.exists(filename)
        try:
            rmtree(model_dir)
        except:
            pass


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

        mlmodel = ct.convert(copy.deepcopy(prog),
                             compute_precision=ct.transform.FP16ComputePrecision(
                                                            op_selector=lambda op: op.op_type != 'square'),
                             convert_to='mlprogram')
        mil_prog = mlmodel._get_mil_internal()
        np.testing.assert_array_equal(["square"], get_op_types_in_program(mil_prog))

        with pytest.raises(ValueError) as e:
            mlmodel = ct.convert(copy.deepcopy(prog),
                                 compute_precision='fp64',
                                 convert_to='mlprogram')
        expected_error = "'compute_precision' must be either coremltools.precision.FLOAT32 or " \
                         "coremltools.precision.FLOAT16 or of type coremltools.transform.FP16ComputePrecision()"
        assert expected_error == str(e.value)

        expected_pattern = "compute_precision .* supported .* mlprogram .* None .* target==\'neuralnetwork\'.*\n.*minimum_deployment_target.*"
        with pytest.raises(ValueError, match=expected_pattern) as e:
            mlmodel = ct.convert(copy.deepcopy(prog), compute_precision='fp16')

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
            mlmodel = ct.convert(prog, compute_precision=ct.precision.FLOAT16)
        with pytest.raises(ValueError, match=expected_err_str):
            mlmodel = ct.convert(prog, compute_precision=ct.precision.FLOAT32)
