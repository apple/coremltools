# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import tempfile
import unittest

import numpy as np
import pytest

import coremltools as ct
from coremltools._deps import _HAS_LIBSVM, _HAS_SKLEARN
from coremltools.converters.mil import mil
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program
from coremltools.models.pipeline import PipelineClassifier, PipelineRegressor
from coremltools.models.utils import _is_macos

if _HAS_SKLEARN:
    from sklearn.datasets import load_boston
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from coremltools.converters import sklearn as converter

if _HAS_LIBSVM:
    from libsvm import svmutil

    from coremltools.converters import libsvm as libsvm_converter


@unittest.skipIf(not _HAS_SKLEARN, "Missing scikit-learn. Skipping tests.")
@unittest.skipIf(not _HAS_LIBSVM, "Missing libsvm. Skipping tests.")
class LinearRegressionPipelineCreationTest(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter.
    """

    @classmethod
    def setUpClass(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """

        if not (_HAS_SKLEARN):
            return

        scikit_data = load_boston()
        feature_names = scikit_data.feature_names

        scikit_model = LinearRegression()
        scikit_model.fit(scikit_data["data"], scikit_data["target"])
        scikit_spec = converter.convert(
            scikit_model, feature_names, "target"
        ).get_spec()

        # Save the data and the model
        self.scikit_data = scikit_data
        self.scikit_model = scikit_model
        self.scikit_spec = scikit_spec

    def test_pipeline_regression_creation(self):

        input_names = self.scikit_data.feature_names
        output_name = "target"
        p_regressor = PipelineRegressor(input_names, "target")
        p_regressor.add_model(self.scikit_spec)

        self.assertIsNotNone(p_regressor.spec)
        self.assertEqual(len(p_regressor.spec.pipelineRegressor.pipeline.models), 1)

        # Test the model class of the linear regressor model
        spec = p_regressor.spec.pipelineRegressor.pipeline.models[0]
        self.assertIsNotNone(spec.description)

        # Test the interface class
        self.assertEqual(spec.description.predictedFeatureName, "target")

        # Test the inputs and outputs
        self.assertEqual(len(spec.description.output), 1)
        self.assertEqual(spec.description.output[0].name, "target")
        self.assertEqual(
            spec.description.output[0].type.WhichOneof("Type"), "doubleType"
        )
        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof("Type"), "doubleType")
        self.assertEqual(
            sorted(input_names), sorted(map(lambda x: x.name, spec.description.input))
        )


@unittest.skipIf(not _HAS_SKLEARN, "Missing scikit-learn. Skipping tests.")
@unittest.skipIf(not _HAS_LIBSVM, "Missing libsvm. Skipping tests.")
class LibSVMPipelineCreationTest(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter.
    """

    @classmethod
    def setUpClass(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        if not _HAS_SKLEARN:
            return
        if not _HAS_LIBSVM:
            return

        scikit_data = load_boston()
        prob = svmutil.svm_problem(
            scikit_data["target"] > scikit_data["target"].mean(),
            scikit_data["data"].tolist(),
        )
        param = svmutil.svm_parameter()
        param.svm_type = svmutil.C_SVC
        param.kernel_type = svmutil.LINEAR
        param.eps = 1

        libsvm_model = svmutil.svm_train(prob, param)
        libsvm_spec = libsvm_converter.convert(
            libsvm_model, scikit_data.feature_names, "target"
        ).get_spec()

        # Save the data and the model
        self.scikit_data = scikit_data
        self.libsvm_spec = libsvm_spec

    def test_pipeline_classifier_creation(self):

        input_names = self.scikit_data.feature_names
        p_classifier = PipelineClassifier(input_names, [1, 0])
        p_classifier.add_model(self.libsvm_spec)

        self.assertIsNotNone(p_classifier.spec)
        self.assertEqual(len(p_classifier.spec.pipelineClassifier.pipeline.models), 1)

        # Test the model class of the svm model
        spec = p_classifier.spec.pipelineClassifier.pipeline.models[0]
        self.assertIsNotNone(spec.description)

        # Test the interface class
        self.assertEqual(spec.description.predictedFeatureName, "target")

        # Test the inputs and outputs
        self.assertEqual(len(spec.description.output), 1)
        self.assertEqual(spec.description.output[0].name, "target")
        self.assertEqual(
            spec.description.output[0].type.WhichOneof("Type"), "int64Type"
        )

        for input_type in spec.description.input:
            self.assertEqual(input_type.type.WhichOneof("Type"), "doubleType")
        self.assertEqual(
            sorted(input_names), sorted(map(lambda x: x.name, spec.description.input))
        )


@unittest.skipIf(not _HAS_SKLEARN, "Missing scikit-learn. Skipping tests.")
class LinearRegressionPipeline(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter.
    """

    @classmethod
    def setUpClass(self):
        """
        Set up the unit test by loading the dataset and training a model.
        """
        if not _HAS_SKLEARN:
            return
        scikit_data = load_boston()
        feature_names = scikit_data.feature_names

        scikit_model = Pipeline(steps=[("linear", LinearRegression())])
        scikit_model.fit(scikit_data["data"], scikit_data["target"])

        # Save the data and the model
        self.scikit_data = scikit_data
        self.scikit_model = scikit_model

    def test_pipeline_regression_creation(self):
        input_names = self.scikit_data.feature_names
        output_name = "target"

        p_regressor_model = converter.convert(self.scikit_model, input_names, "target")

        x = dict(zip(self.scikit_data["feature_names"], self.scikit_data["data"][0]))
        y = p_regressor_model.predict(x)
        self.assertIsNotNone(y)

        with tempfile.TemporaryDirectory() as save_dir:
            p_regressor_model.save(save_dir + "/test.mlmodel")

        p_regressor = p_regressor_model.get_spec()

        self.assertIsNotNone(p_regressor)
        self.assertEqual(len(p_regressor.pipelineRegressor.pipeline.models), 2)

        # Test the model class of the linear regressor model
        spec = p_regressor.pipelineRegressor.pipeline.models[-1]
        self.assertIsNotNone(spec.description)

        # Test the interface class
        self.assertEqual(spec.description.predictedFeatureName, "target")

        # Test the inputs and outputs
        self.assertEqual(len(spec.description.output), 1)
        self.assertEqual(spec.description.output[0].name, "target")
        self.assertEqual(
            spec.description.output[0].type.WhichOneof("Type"), "doubleType"
        )

        for input_type in p_regressor.description.input:
            self.assertEqual(input_type.type.WhichOneof("Type"), "doubleType")
        self.assertEqual(
            sorted(input_names),
            sorted(map(lambda x: x.name, p_regressor.description.input)),
        )

    def test_conversion_bad_inputs(self):
        """
        Failure testing for bad conversion.
        """
        # Error on converting an untrained model
        with self.assertRaises(TypeError):
            model = OneHotEncoder()
            spec = converter.convert(model, "data", "out", "regressor")


class TestMakePipeline:
    @staticmethod
    def _make_model(input_name, input_length,
                    output_name, output_length,
                    convert_to='mlprogram', compute_units=ct.ComputeUnit.ALL):

        weight_tensor = np.arange(input_length * output_length, dtype='float32')
        weight_tensor = weight_tensor.reshape(output_length, input_length)

        prog = mil.Program()
        func_inputs = {input_name: mb.placeholder(shape=(input_length,))}
        with Function(func_inputs) as ssa_fun:
            input = ssa_fun.inputs[input_name]
            y = mb.linear(x=input, weight=weight_tensor, name=output_name)
            ssa_fun.set_outputs([y])
            prog.add_function("main", ssa_fun)

        return ct.convert(prog, convert_to=convert_to, compute_units=compute_units)


    @staticmethod
    @pytest.mark.parametrize(
        "model1_backend, model2_backend",
        itertools.product(["mlprogram", "neuralnetwork"], ["mlprogram", "neuralnetwork"]),
    )
    def test_simple(model1_backend, model2_backend):
        # Create models
        m1 = TestMakePipeline._make_model("x", 20, "y1", 10, model1_backend)
        m2 = TestMakePipeline._make_model("y1", 10, "y2", 2, model2_backend)

        # Get non-pipeline result
        x = np.random.rand(20)
        if _is_macos():
            y1 = m1.predict({"x": x})["y1"]
            y2 = m2.predict({"y1": y1})

        pipeline_model = ct.utils.make_pipeline(m1, m2)

        if _is_macos():
            y_pipeline = pipeline_model.predict({"x": x})
            assert(len(y_pipeline) == 1)
            np.testing.assert_allclose(y2["y2"], y_pipeline["y2"])

        # Check save/load
        with tempfile.TemporaryDirectory() as save_dir:
            # Save pipeline
            save_path = save_dir + "/test.mlpackage"
            pipeline_model.save(save_path)

            # Check loading from a mlpackage path
            p2 = ct.models.MLModel(save_path)
            if _is_macos():
                y_pipeline = p2.predict({"x": x})
                np.testing.assert_allclose(y2["y2"], y_pipeline["y2"])

            # Check loading from spec and weight dir
            p3 = ct.models.MLModel(p2.get_spec(), weights_dir=p2.weights_dir)
            if _is_macos():
                y_pipeline = p3.predict({"x": x})
                np.testing.assert_allclose(y2["y2"], y_pipeline["y2"])


    @staticmethod
    def test_compute_unit():
        # Case 1 - Inferring compute_unit
        m1 = TestMakePipeline._make_model("x", 20, "y1", 10,
                                          compute_units=ct.ComputeUnit.CPU_ONLY)
        m2 = TestMakePipeline._make_model("y1", 10, "y2", 2,
                                          compute_units=ct.ComputeUnit.CPU_ONLY)
        pipeline_model = ct.utils.make_pipeline(m1, m2)
        assert pipeline_model.compute_unit is ct.ComputeUnit.CPU_ONLY

        # Case 2 - Specifying compute_unit
        pipeline_model = ct.utils.make_pipeline(m1, m2, compute_units=ct.ComputeUnit.ALL)
        assert pipeline_model.compute_unit is ct.ComputeUnit.ALL

        # Case 3 (error case) - No compute_unit specified and the two models don't agree
        m2 = TestMakePipeline._make_model("y1", 10, "y2", 2,
                                          compute_units=ct.ComputeUnit.ALL)
        with pytest.raises(ValueError, match='"compute_units" parameter must be specified.'):
            pipeline_model = ct.utils.make_pipeline(m1, m2)

        # Case 4 (error case) - Garbage compute_unit input
        with pytest.raises(TypeError, match='"compute_units" parameter must'):
            pipeline_model = ct.utils.make_pipeline(m1, m2, compute_units="Garbage!")


    @staticmethod
    def test_second_model_needs_pipeline_input():
        # First model takes one parameter
        p1 = mil.Program()
        func_inputs = {'x1': mb.placeholder(shape=(2,))}
        with Function(func_inputs) as ssa_fun:
            x1 = ssa_fun.inputs['x1']
            y1 = mb.add(x=x1, y=[0., 1.], name='y1')
            ssa_fun.set_outputs([y1])
            p1.add_function("main", ssa_fun)
        m1 = ct.convert(p1)

        # Second model takes two parameters. One will be from previous model in pipeline.
        # The other as pipeline input.
        p2 = mil.Program()
        func_inputs = {
            'y1': mb.placeholder(shape=(2,)),
            'x2': mb.placeholder(shape=(2,)),
        }
        with Function(func_inputs) as ssa_fun:
            x2, y1 = ssa_fun.inputs['x2'], ssa_fun.inputs['y1']
            y2 = mb.add(x=x2, y=y1, name='y2')
            ssa_fun.set_outputs([y2])
            p2.add_function("main", ssa_fun)
        m2 = ct.convert(p2)

        # Get predictions without a pipeline
        x1 = [1.,2.]
        y1 = m1.predict({'x1': x1})['y1']
        x2 = [4., 9.]
        y2 = m2.predict({'x2': x2, 'y1': y1})['y2']

        # Make a pipeline and get predictions from it
        pipeline = ct.utils.make_pipeline(m1, m2)
        y_pipeline = pipeline.predict({'x1': x1, 'x2': x2})
        assert len(y_pipeline) == 1

        np.testing.assert_allclose(y2, y_pipeline['y2'])


    @staticmethod
    def test_pipeline_outputs_from_multiple_models():
        # Create models
        m1 = TestMakePipeline._make_model("x", 20, "y1", 10)
        m2 = TestMakePipeline._make_model("y1", 10, "y2", 2)
        m3 = TestMakePipeline._make_model("y1", 10, "y3", 4)

        # Get non-pipeline results
        x = np.random.rand(20)
        if _is_macos():
            y1 = m1.predict({"x": x})["y1"]
            y2 = m2.predict({"y1": y1})
            y3 = m3.predict({"y1": y1})

        pipeline_model = ct.utils.make_pipeline(m1, m2, m3)
        if _is_macos():
            y_pipeline = pipeline_model.predict({"x": x})
            assert(len(y_pipeline) == 2)
            np.testing.assert_allclose(y2["y2"], y_pipeline["y2"])
            np.testing.assert_allclose(y3["y3"], y_pipeline["y3"])


    @staticmethod
    def test_pipeline_input_goes_to_multiple_models():
        # Create the first two models that take the same input
        m1 = TestMakePipeline._make_model("x", 20, "y1", 10)
        m2 = TestMakePipeline._make_model("x", 20, "y2", 10)

        # Create the last models which add the output from the other two models.
        p3 = Program()
        func_inputs = {'y1': mb.placeholder(shape=(10,)),
                       'y2': mb.placeholder(shape=(10,)),}
        with Function(func_inputs) as ssa_fun:
            y1, y2 = ssa_fun.inputs['y1'], ssa_fun.inputs['y2']
            y3 = mb.add(x=y1, y=y2, name='y3')
            ssa_fun.set_outputs([y3])
            p3.add_function("main", ssa_fun)
        m3 = ct.convert(p3)

        # Get non-pipeline result
        x = np.random.rand(20)
        if _is_macos():
            y1 = m1.predict({"x": x})["y1"]
            y2 = m2.predict({"x": x})["y2"]
            y3 = m3.predict({"y1": y1, "y2": y2})

        pipeline_model = ct.utils.make_pipeline(m1, m2, m3)
        if _is_macos():
            y_pipeline = pipeline_model.predict({"x": x})
            assert(len(y_pipeline) == 1)
            np.testing.assert_allclose(y3['y3'], y_pipeline["y3"])
