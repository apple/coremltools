# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os
import platform
import shutil
import tempfile

import numpy as np
import pytest
import torch

import coremltools as ct
from coremltools import _SPECIFICATION_VERSION_IOS_18, proto
from coremltools.converters.mil import mil
from coremltools.converters.mil.converter import mil_convert as _mil_convert
from coremltools.converters.mil.mil.builder import Builder as mb
from coremltools.models.utils import MultiFunctionDescriptor, load_spec, save_multifunction


@pytest.mark.skipif(ct.utils._macos_version() < (15, 0),
                    reason="Multi-function only supported on macOS 15+")
class TestMultiFunctionDescriptor:
    @staticmethod
    def _convert_multifunction_prog(prog):
        mlmodel = _mil_convert(
            prog,
            convert_to="mlprogram",
            convert_from="milinternal",
            specification_version=_SPECIFICATION_VERSION_IOS_18,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            export_multi_functions=True,
            skip_model_load=True,
        )
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel.save(package_path)
        return package_path

    @staticmethod
    def _get_singlefunction_mlpackage(opset_version=ct.target.iOS16):
        @mb.program(
            input_specs=[mb.TensorSpec((3,))],
            opset_version=opset_version,
        )
        def prog(x):
            return mb.relu(x=x)

        mlmodel = ct.convert(
            prog,
            minimum_deployment_target=opset_version,
        )
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel.save(package_path)

        return package_path

    def _get_multifunction_mlpackage_1(self):
        @mb.function(
            input_specs=[mb.TensorSpec((3,))],
            opset_version=ct.target.iOS18,
        )
        def func(x):
            return mb.relu(x=x)

        @mb.function(
            input_specs=[mb.TensorSpec((3,))],
            opset_version=ct.target.iOS18,
        )
        def func_1(x):
            return mb.sin(x=x)

        @mb.function(
            input_specs=[mb.TensorSpec((3,))],
            opset_version=ct.target.iOS18,
        )
        def func_2(x):
            return mb.cos(x=x)

        prog = mil.Program()
        prog.add_function("relu", func)
        prog.add_function("sin", func_1)
        prog.add_function("cos", func_2)
        prog.default_function_name = "relu"

        return self._convert_multifunction_prog(prog)

    def _get_multifunction_mlpackage_2(self):
        @mb.function(
            input_specs=[mb.TensorSpec((3,))],
            opset_version=ct.target.iOS18,
        )
        def func(x):
            return mb.relu(x=x)

        @mb.function(
            input_specs=[mb.TensorSpec((3,))],
            opset_version=ct.target.iOS18,
        )
        def func_1(x):
            return mb.sin(x=x)

        prog = mil.Program()
        prog.add_function("relu", func)
        prog.add_function("sin", func_1)
        prog.default_function_name = "sin"

        return self._convert_multifunction_prog(prog)

    def _get_multifunction_mlpackage_3(self):
        @mb.function(
            input_specs=[mb.TensorSpec((3,))],
            opset_version=ct.target.iOS18,
        )
        def func(x):
            return mb.relu(x=x)

        prog = mil.Program()
        prog.add_function("relu", func)
        prog.default_function_name = "relu"

        return self._convert_multifunction_prog(prog)

    def test_initialization(self):
        # Test empty initialization
        desc = MultiFunctionDescriptor()
        assert desc._functions() == {}

        # Initialize with a single function model
        model = self._get_singlefunction_mlpackage()
        desc = MultiFunctionDescriptor(model)
        assert desc._functions() == {"main": (model, "main")}
        shutil.rmtree(model)

        # Initialize with a multifunction model with only a single function
        model = self._get_multifunction_mlpackage_3()
        desc = MultiFunctionDescriptor(model)
        assert desc._functions() == {"relu": (model, "relu")}
        shutil.rmtree(model)

        # Initialize with a multifunction model with several functions
        model = self._get_multifunction_mlpackage_1()
        desc = MultiFunctionDescriptor(model)
        assert desc._functions() == {
            "relu": (model, "relu"),
            "sin": (model, "sin"),
            "cos": (model, "cos"),
        }
        shutil.rmtree(model)

        # Initialize with invalid path
        with pytest.raises(ValueError, match="invalid model_path invalid_path with error"):
            desc = MultiFunctionDescriptor("invalid_path")

    def test_add_function(self):
        # Add function from a single function model
        desc = MultiFunctionDescriptor()
        model = self._get_singlefunction_mlpackage()
        desc.add_function(model, "main", "main_1")
        assert desc._functions() == {"main_1": (model, "main")}
        desc.add_function(model, "main", "main_2")
        assert desc._functions() == {"main_1": (model, "main"), "main_2": (model, "main")}
        with pytest.raises(ValueError, match="src_function_name invalid not found in"):
            desc.add_function(model, "invalid", "main_3")
        with pytest.raises(ValueError, match="function main_1 already exist"):
            desc.add_function(model, "main", "main_1")
        shutil.rmtree(model)

        # Add function from multifunction model
        desc = MultiFunctionDescriptor()
        model = self._get_multifunction_mlpackage_1()
        desc.add_function(model, "relu", "main_1")
        assert desc._functions() == {"main_1": (model, "relu")}
        desc.add_function(model, "sin", "main_2")
        assert desc._functions() == {"main_1": (model, "relu"), "main_2": (model, "sin")}
        shutil.rmtree(model)

        # Initialize a desc with a model and add functions to it
        model = self._get_multifunction_mlpackage_1()
        desc = MultiFunctionDescriptor(model)
        assert desc._functions() == {
            "relu": (model, "relu"),
            "sin": (model, "sin"),
            "cos": (model, "cos"),
        }
        model_2 = self._get_multifunction_mlpackage_2()
        desc.add_function(model_2, "sin", "new_sin")
        assert desc._functions() == {
            "relu": (model, "relu"),
            "sin": (model, "sin"),
            "cos": (model, "cos"),
            "new_sin": (model_2, "sin"),
        }
        with pytest.raises(ValueError, match="function relu already exist"):
            desc.add_function(model, "relu", "relu")
        shutil.rmtree(model)
        shutil.rmtree(model_2)

    def test_add_model(self):
        # Add model from a single function model
        desc = MultiFunctionDescriptor()
        model = self._get_singlefunction_mlpackage()
        desc.add_model(model)
        assert desc._functions() == {"main": (model, "main")}
        shutil.rmtree(model)

        # Add a multifunction model with only a single function
        desc = MultiFunctionDescriptor()
        model = self._get_multifunction_mlpackage_3()
        desc.add_model(model)
        assert desc._functions() == {"relu": (model, "relu")}
        shutil.rmtree(model)

        # Add a multifunction model with several functions
        desc = MultiFunctionDescriptor()
        model = self._get_multifunction_mlpackage_1()
        desc.add_model(model)
        assert desc._functions() == {
            "relu": (model, "relu"),
            "sin": (model, "sin"),
            "cos": (model, "cos"),
        }
        shutil.rmtree(model)

        # Add a model to a desc with functions
        model = self._get_singlefunction_mlpackage()
        desc = MultiFunctionDescriptor(model)
        assert desc._functions() == {"main": (model, "main")}
        model_2 = self._get_multifunction_mlpackage_1()
        desc.add_model(model_2)
        assert desc._functions() == {
            "relu": (model_2, "relu"),
            "sin": (model_2, "sin"),
            "cos": (model_2, "cos"),
            "main": (model, "main"),
        }
        shutil.rmtree(model)
        shutil.rmtree(model_2)

        # Error handling when adding model with duplicated function name
        model = self._get_multifunction_mlpackage_2()
        with pytest.raises(ValueError, match="function relu already exist"):
            desc.add_model(model)
        shutil.rmtree(model)

    def test_remove_function(self):
        model = self._get_multifunction_mlpackage_1()
        desc = MultiFunctionDescriptor(model)
        assert desc._functions() == {
            "relu": (model, "relu"),
            "sin": (model, "sin"),
            "cos": (model, "cos"),
        }
        desc.remove_function("relu")
        assert desc._functions() == {
            "sin": (model, "sin"),
            "cos": (model, "cos"),
        }
        with pytest.raises(ValueError, match="function_name relu not found"):
            desc.remove_function("relu")

        desc.remove_function("sin")
        assert desc._functions() == {
            "cos": (model, "cos"),
        }
        desc.remove_function("cos")
        assert desc._functions() == {}
        with pytest.raises(ValueError, match="function_name relu not found"):
            desc.remove_function("relu")
        shutil.rmtree(model)

    def test_convert_single_function_into_multifunction_model(self):
        """
        Convert a single function model into a multifunction model format,
        but only consists of one function.
        """
        model = self._get_singlefunction_mlpackage()
        desc = MultiFunctionDescriptor()
        desc.add_function(model, "main", "main_1")
        desc.default_function_name = "main_1"
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        save_multifunction(desc, package_path)
        shutil.rmtree(model)

        # verify the model spec
        spec = load_spec(package_path)
        model_desc = spec.description
        assert len(model_desc.functions) == 1
        assert model_desc.functions[0].name == "main_1"
        assert model_desc.defaultFunctionName == "main_1"

        # verify the model can be load / run
        new_model = ct.models.MLModel(package_path, function_name="main_1")
        new_model.predict(
            {
                "x": np.random.rand(
                    3,
                )
            }
        )
        shutil.rmtree(package_path)

    def test_merge_two_models_into_multifunction_model(self):
        """
        Merge two single function models into one multifunction model.
        """
        model_1 = self._get_singlefunction_mlpackage()
        model_2 = self._get_singlefunction_mlpackage()
        desc = MultiFunctionDescriptor()
        desc.add_function(model_1, "main", "main_1")
        desc.add_function(model_2, "main", "main_2")
        desc.default_function_name = "main_2"
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        save_multifunction(desc, package_path)
        shutil.rmtree(model_1)
        shutil.rmtree(model_2)

        # verify the model spec
        spec = load_spec(package_path)
        model_desc = spec.description
        assert len(model_desc.functions) == 2
        assert model_desc.functions[0].name == "main_1"
        assert model_desc.functions[1].name == "main_2"
        assert model_desc.defaultFunctionName == "main_2"

        # verify the model can be load / run
        new_model = ct.models.MLModel(package_path, function_name="main_1")
        new_model.predict(
            {
                "x": np.random.rand(
                    3,
                )
            }
        )
        new_model = ct.models.MLModel(package_path, function_name="main_2")
        new_model.predict(
            {
                "x": np.random.rand(
                    3,
                )
            }
        )
        shutil.rmtree(package_path)

    def test_copy_a_single_model_twice_into_multifunction_model(self):
        """
        Copy the function in a single function model twice to make a multifunction model.
        """
        model = self._get_singlefunction_mlpackage()
        desc = MultiFunctionDescriptor()
        desc.add_function(model, "main", "main_1")
        desc.add_function(model, "main", "main_2")
        desc.default_function_name = "main_2"
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        save_multifunction(desc, package_path)
        shutil.rmtree(model)

        # verify the model spec
        spec = load_spec(package_path)
        model_desc = spec.description
        assert len(model_desc.functions) == 2
        assert model_desc.functions[0].name == "main_1"
        assert model_desc.functions[1].name == "main_2"
        assert model_desc.defaultFunctionName == "main_2"

        # verify the model can be load / run
        new_model = ct.models.MLModel(package_path, function_name="main_1")
        new_model.predict(
            {
                "x": np.random.rand(
                    3,
                )
            }
        )
        new_model = ct.models.MLModel(package_path, function_name="main_2")
        new_model.predict(
            {
                "x": np.random.rand(
                    3,
                )
            }
        )
        shutil.rmtree(package_path)

    def test_combine_multifunctin_models(self):
        """
        Combine two multifunction models into one multifunction model.
        """
        model_1 = self._get_multifunction_mlpackage_1()
        desc = MultiFunctionDescriptor(model_1)
        model_2 = self._get_multifunction_mlpackage_2()
        desc.add_function(model_2, "relu", "main_1")
        desc.add_function(model_2, "sin", "main_2")
        desc.default_function_name = "main_2"
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        save_multifunction(desc, package_path)
        shutil.rmtree(model_1)
        shutil.rmtree(model_2)

        # verify the model spec
        spec = load_spec(package_path)
        model_desc = spec.description
        assert len(model_desc.functions) == 5
        assert model_desc.functions[0].name == "relu"
        assert model_desc.functions[1].name == "sin"
        assert model_desc.functions[2].name == "cos"
        assert model_desc.functions[3].name == "main_1"
        assert model_desc.functions[4].name == "main_2"
        assert model_desc.defaultFunctionName == "main_2"

        # verify the model can be load / run
        new_model = ct.models.MLModel(package_path, function_name="relu")
        new_model.predict(
            {
                "x": np.random.rand(
                    3,
                )
            }
        )
        new_model = ct.models.MLModel(package_path, function_name="sin")
        new_model.predict(
            {
                "x": np.random.rand(
                    3,
                )
            }
        )
        new_model = ct.models.MLModel(package_path, function_name="cos")
        new_model.predict(
            {
                "x": np.random.rand(
                    3,
                )
            }
        )
        new_model = ct.models.MLModel(package_path, function_name="main_1")
        new_model.predict(
            {
                "x": np.random.rand(
                    3,
                )
            }
        )
        new_model = ct.models.MLModel(package_path, function_name="main_2")
        new_model.predict(
            {
                "x": np.random.rand(
                    3,
                )
            }
        )
        shutil.rmtree(package_path)

    def test_invalid_default_function_name(self):
        # invalid type
        model = self._get_multifunction_mlpackage_1()
        desc = MultiFunctionDescriptor(model)
        with pytest.raises(ValueError, match="default_function_name must be type of str. Got 1."):
            desc.default_function_name = 1

        # default function name not found in the program
        desc.default_function_name = "invalid"
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        with pytest.raises(
            ValueError, match="default_function_name invalid not found in the program."
        ):
            save_multifunction(desc, package_path)

        # default function name not set
        desc = MultiFunctionDescriptor(model)
        with pytest.raises(
            ValueError,
            match="default_function_name must be set for the MultiFunctionDescriptor instance before calling save_multifunction.",
        ):
            save_multifunction(desc, package_path)

        # cleanup

    def test_spec_version_save_multifunction(self):
        """
        When save models to the multifunction format, the spec version are promoted to iOS18.
        """
        model_1 = self._get_singlefunction_mlpackage(opset_version=ct.target.iOS15)
        model_2 = self._get_singlefunction_mlpackage(opset_version=ct.target.iOS16)
        desc = MultiFunctionDescriptor(model_1)
        desc.add_function(model_2, "main", "main_2")
        desc.default_function_name = "main_2"
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        save_multifunction(desc, package_path)
        shutil.rmtree(model_1)
        shutil.rmtree(model_2)

        # verify the spec version of the multifunctino model is iOS18
        spec = load_spec(package_path)
        assert spec.specificationVersion == _SPECIFICATION_VERSION_IOS_18
        shutil.rmtree(package_path)

    @staticmethod
    def _multifunction_model_from_single_function(model_path: str) -> str:
        desc = MultiFunctionDescriptor()
        desc.add_function(model_path, "main", "main_1")
        desc.add_function(model_path, "main", "main_2")
        desc.default_function_name = "main_1"
        multifunction_path = tempfile.mkdtemp(suffix=".mlpackage")
        save_multifunction(desc, multifunction_path)
        return multifunction_path

    @staticmethod
    def _multifunction_model_from_multifunction_model(model_path: str) -> str:
        desc = MultiFunctionDescriptor()
        desc.add_function(model_path, "main_1", "main_3")
        desc.add_function(model_path, "main_2", "main_4")
        desc.default_function_name = "main_3"
        multifunction_path = tempfile.mkdtemp(suffix=".mlpackage")
        save_multifunction(desc, multifunction_path)
        return multifunction_path

    def test_classifier_description(self):
        """
        If the source model is a classifier, the resulting multifunction model should
        inherit the classifier description as well.
        """

        def check_classifier_spec(model_path: str) -> None:
            spec = load_spec(model_path)
            model_desc = spec.description

            assert len(model_desc.functions) == 2

            for idx in [0, 1]:
                assert model_desc.functions[idx].predictedFeatureName == "class_label"
                assert model_desc.functions[idx].predictedProbabilitiesName == "class_label_probs"
                assert model_desc.functions[idx].output[0].name == "class_label"
                assert model_desc.functions[idx].output[1].name == "class_label_probs"

        # source model with classifier config
        torch_model = torch.nn.ReLU().eval()
        traced_model = torch.jit.trace(
            torch_model,
            torch.rand(
                3,
            ),
        )
        variable_name = "var_2"
        class_label_name = "class_label"
        classifier_config = ct.ClassifierConfig(
            class_labels=["a", "b", "c"],
            predicted_feature_name=class_label_name,
            predicted_probabilities_output=variable_name,
        )

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(3,))],
            classifier_config=classifier_config,
            minimum_deployment_target=ct.target.iOS16,
        )

        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel.save(package_path)

        # multifunction model should have the same classifier description
        model_path = self._multifunction_model_from_single_function(package_path)
        check_classifier_spec(model_path)

        # construct another multifunction model with an existing multifunction model,
        # the classifier description should still be the same.
        model_path_2 = self._multifunction_model_from_multifunction_model(model_path)
        check_classifier_spec(model_path_2)

        # cleanup
        shutil.rmtree(package_path)
        shutil.rmtree(model_path)
        shutil.rmtree(model_path_2)

    def test_input_output_description(self):
        """
        When using save_multifunction to produce a model, we should respect
        the original model description in the original model.
        """

        def check_i_o_spec(model_path: str) -> None:
            spec = load_spec(model_path)
            model_desc = spec.description

            assert len(model_desc.functions) == 2

            for idx in [0, 1]:
                assert (
                    model_desc.functions[idx].input[0].type.imageType.colorSpace
                    == proto.FeatureTypes_pb2.ImageFeatureType.BGR
                )
                assert (
                    model_desc.functions[idx].output[0].type.imageType.colorSpace
                    == proto.FeatureTypes_pb2.ImageFeatureType.RGB
                )

        # source model with i/o with ImageType
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 5.0

        example_input = torch.randint(0, 100, (1, 3, 10, 20), dtype=torch.float32)
        model = torch.jit.trace(Model().eval(), example_input)
        mlmodel = ct.convert(
            model,
            inputs=[ct.ImageType(shape=(1, 3, 10, 20), color_layout=ct.colorlayout.BGR)],
            outputs=[ct.ImageType(color_layout=ct.colorlayout.RGB)],
            minimum_deployment_target=ct.target.iOS16,
        )
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel.save(package_path)

        # multifunction model should have the same i/o description
        model_path = self._multifunction_model_from_single_function(package_path)
        check_i_o_spec(model_path)

        # construct another multifunction model with an existing multifunction model,
        # the i/o description should still be the same
        model_path_2 = self._multifunction_model_from_multifunction_model(model_path)
        check_i_o_spec(model_path_2)

        # cleanup
        shutil.rmtree(package_path)
        shutil.rmtree(model_path)
        shutil.rmtree(model_path_2)


@pytest.mark.skipif(ct.utils._macos_version() < (15, 0),
                    reason="Multi-function only supported on macOS 15+")
class TestMultiFunctionModelEnd2End:
    @staticmethod
    def _get_test_model():
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 8, 5, padding="same", bias=False)
                self.bn1 = torch.nn.BatchNorm2d(8)
                self.linear1 = torch.nn.Linear(28 * 28 * 8, 5, bias=False)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.linear1(torch.flatten(x))
                return x

        model = TestModel().eval()
        example_input = torch.rand(1, 1, 28, 28)
        return torch.jit.trace(model, example_input)

    @staticmethod
    def _get_test_model_2():
        """
        Base model have the same weights, while the weights in submodule are different.
        """

        class SubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(28 * 28 * 8, 5, bias=False)

            def forward(self, x):
                return self.linear1(x)

        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 8, 5, padding="same", bias=False)
                self.bn1 = torch.nn.BatchNorm2d(8)
                self.linear1 = None

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.linear1(torch.flatten(x))
                return x

        example_input = torch.rand(1, 1, 28, 28)
        model = TestModel().eval()

        submodule_1 = SubModel().eval()
        model.linear1 = submodule_1
        trace_1 = torch.jit.trace(model, example_input)

        submodule_2 = SubModel().eval()
        model.linear1 = submodule_2
        trace_2 = torch.jit.trace(model, example_input)

        return trace_1, trace_2

    def test_two_models(self):
        """
        model_1: base + function_1
        model_2: base + function_2

        After merging model_1 with model_2, the base weights should be shared.
        """
        traced_model_1, traced_model_2 = self._get_test_model_2()
        input = np.random.rand(1, 1, 28, 28)

        mlmodel_1 = ct.convert(
            traced_model_1,
            inputs=[ct.TensorType(name="x", shape=(1, 1, 28, 28))],
            outputs=[ct.TensorType(name="out")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS17,
        )
        mlmodel_2 = ct.convert(
            traced_model_2,
            inputs=[ct.TensorType(name="x", shape=(1, 1, 28, 28))],
            outputs=[ct.TensorType(name="out")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS17,
        )

        gt_output_1 = mlmodel_1.predict({"x": input})["out"]
        gt_output_2 = mlmodel_2.predict({"x": input})["out"]

        package_path_1 = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel_1.save(package_path_1)
        package_path_2 = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel_2.save(package_path_2)

        # save multifuntion model
        desc = MultiFunctionDescriptor()
        desc.add_function(package_path_1, "main", "main_1")
        desc.add_function(package_path_2, "main", "main_2")
        desc.default_function_name = "main_1"
        saved_package_path = tempfile.mkdtemp(suffix=".mlpackage")
        save_multifunction(desc, saved_package_path)
        shutil.rmtree(package_path_1)
        shutil.rmtree(package_path_2)

        # verify the model spec
        spec = load_spec(saved_package_path)
        model_desc = spec.description
        assert len(model_desc.functions) == 2
        assert model_desc.functions[0].name == "main_1"
        assert model_desc.functions[1].name == "main_2"
        assert model_desc.defaultFunctionName == "main_1"

        # verify the model can be load / run
        # rdar://126898335 ([multifunction][bug] CoreML "maybe" is not handling the fallback for the compute units)
        if platform.machine() == "arm64":
            multifunction_mlmodel_1 = ct.models.MLModel(saved_package_path, function_name="main_1")
            output = multifunction_mlmodel_1.predict({"x": input})["out"]
            np.testing.assert_allclose(gt_output_1, output)

            multifunction_mlmodel_2 = ct.models.MLModel(saved_package_path, function_name="main_2")
            output = multifunction_mlmodel_2.predict({"x": input})["out"]
            np.testing.assert_allclose(gt_output_2, output)

        # make sure the weights are deduplicated
        with tempfile.TemporaryDirectory() as serialize_dir:
            os.system(f"coremlcompiler compile {saved_package_path} {serialize_dir}")
            model_name_with_extension = os.path.basename(saved_package_path)
            model_name_wo_extension, _ = os.path.splitext(model_name_with_extension)
            mil_file = open(
                os.path.join(serialize_dir, f"{model_name_wo_extension}.mlmodelc", "model.mil")
            )
            mil_txt = mil_file.read()
            assert (
                mil_txt.count(
                    'const()[name = string("x_weight_0_to_fp16"), val = tensor<fp16, [8, 1, 5, 5]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];'
                )
                == 2
            )
            assert (
                mil_txt.count(
                    'tensor<fp16, [5, 6272]> linear1_linear1_weight_to_fp16 = const()[name = string("linear1_linear1_weight_to_fp16"), val = tensor<fp16, [5, 6272]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(576)))];'
                )
                == 1
            )
            assert (
                mil_txt.count(
                    'tensor<fp16, [5, 6272]> linear1_linear1_weight_to_fp16 = const()[name = string("linear1_linear1_weight_to_fp16"), val = tensor<fp16, [5, 6272]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(63360)))];'
                )
                == 1
            )
        shutil.rmtree(saved_package_path)

    def test_single_model(self):
        """
        Convert a single model into a multi-functions model with only one function.
        """
        traced_model = self._get_test_model()
        input = np.random.rand(1, 1, 28, 28)
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="x", shape=(1, 1, 28, 28))],
            outputs=[ct.TensorType(name="out")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS16,
        )
        gt_output = mlmodel.predict({"x": input})["out"]
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel.save(package_path)

        # save multifuntion model
        desc = MultiFunctionDescriptor()
        desc.add_function(package_path, "main", "main_1")
        desc.default_function_name = "main_1"
        saved_package_path = tempfile.mkdtemp(suffix=".mlpackage")
        save_multifunction(desc, saved_package_path)
        shutil.rmtree(package_path)

        # verify the model spec
        spec = load_spec(saved_package_path)
        model_desc = spec.description
        assert len(model_desc.functions) == 1
        assert model_desc.functions[0].name == "main_1"
        assert model_desc.defaultFunctionName == "main_1"

        # verify the model can be load / run
        # rdar://126898335 ([multifunction][bug] CoreML "maybe" is not handling the fallback for the compute units)
        if platform.machine() == "arm64":
            multifunction_mlmodel = ct.models.MLModel(saved_package_path, function_name="main_1")
            output = multifunction_mlmodel.predict({"x": input})["out"]
            np.testing.assert_allclose(gt_output, output)
        shutil.rmtree(saved_package_path)

    def test_10_duplicated_model(self):
        """
        Copy a single model 10 times and create a multi-functions model with 10 functions.
        """
        traced_model = self._get_test_model()
        input = np.random.rand(1, 1, 28, 28)
        NUM_MODEL = 10
        saved_paths = []

        for i in range(NUM_MODEL):
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="x", shape=(1, 1, 28, 28))],
                outputs=[ct.TensorType(name="out")],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS17,
            )
            gt_output = mlmodel.predict({"x": input})["out"]
            saved_paths.append(tempfile.mkdtemp(suffix=".mlpackage"))
            mlmodel.save(saved_paths[-1])

        # save the multifunction model
        desc = MultiFunctionDescriptor()
        for i in range(NUM_MODEL):
            desc.add_function(saved_paths[i], "main", f"main_{i}")
        desc.default_function_name = "main_5"
        saved_package_path = tempfile.mkdtemp(suffix=".mlpackage")
        save_multifunction(desc, saved_package_path)

        for val in saved_paths:
            shutil.rmtree(val)

        # verify the model spec
        spec = load_spec(saved_package_path)
        model_desc = spec.description
        assert len(model_desc.functions) == NUM_MODEL
        for i in range(NUM_MODEL):
            assert model_desc.functions[i].name == f"main_{i}"
        assert model_desc.defaultFunctionName == "main_5"

        # verify the model can be load / run
        # rdar://126898335 ([multifunction][bug] CoreML "maybe" is not handling the fallback for the compute units)
        if platform.machine() == "arm64":
            for i in range(NUM_MODEL):
                multifunction_mlmodel = ct.models.MLModel(
                    saved_package_path, function_name=f"main_{i}"
                )
                output = multifunction_mlmodel.predict({"x": input})["out"]
                np.testing.assert_allclose(gt_output, output)

        # make sure the weights are deduplicated
        with tempfile.TemporaryDirectory() as serialize_dir:
            os.system(f"coremlcompiler compile {saved_package_path} {serialize_dir}")
            model_name_with_extension = os.path.basename(saved_package_path)
            model_name_wo_extension, _ = os.path.splitext(model_name_with_extension)
            mil_file = open(
                os.path.join(serialize_dir, f"{model_name_wo_extension}.mlmodelc", "model.mil")
            )
            mil_txt = mil_file.read()
            assert (
                mil_txt.count(
                    'const()[name = string("x_weight_0_to_fp16"), val = tensor<fp16, [8, 1, 5, 5]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];'
                )
                == 10
            )
            assert (
                mil_txt.count(
                    'tensor<fp16, [5, 6272]> linear1_weight_to_fp16 = const()[name = string("linear1_weight_to_fp16"), val = tensor<fp16, [5, 6272]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(576)))];'
                )
                == 10
            )
        shutil.rmtree(saved_package_path)
