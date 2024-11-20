# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import itertools
import os
import platform
import shutil
import tempfile
from typing import Dict, Tuple

import numpy as np
import pytest
import torch

import coremltools as ct
from coremltools import _SPECIFICATION_VERSION_IOS_18, proto
from coremltools.converters.mil import mil
from coremltools.converters.mil.converter import mil_convert as _mil_convert
from coremltools.converters.mil.mil import Program, types
from coremltools.converters.mil.mil.builder import Builder as mb
from coremltools.converters.mil.mil.passes.pass_pipeline import (
    PassPipeline,
    PassPipelineManager,
)
from coremltools.converters.mil.testing_utils import assert_spec_input_type, assert_spec_output_type, DTYPE_TO_FEATURE_TYPE_MAP, get_op_types_in_program
from coremltools.models.datatypes import Array
from coremltools.models.utils import bisect_model, MultiFunctionDescriptor, load_spec, save_multifunction, load_spec, change_input_output_tensor_type
from coremltools.proto.FeatureTypes_pb2 import ArrayFeatureType
import coremltools.optimize as cto


class TestMILConvertCall:
    @staticmethod
    def test_pass_pipeline():
        X_SHAPE = (2, 3, 16, 16)
        WEIGHT_SHAPE = (5, X_SHAPE[1], 3, 3)
        BIAS_SHAPE = (WEIGHT_SHAPE[0], 1, 1)
        WEIGHT = np.random.rand(*WEIGHT_SHAPE)
        BIAS = np.random.rand(*BIAS_SHAPE)

        @mb.program(input_specs=[mb.TensorSpec(shape=X_SHAPE)])
        def prog(x):
            y = mb.conv(x=x, weight=WEIGHT)
            z = mb.add(x=y, y=BIAS)
            return z

        prog1 = copy.deepcopy(prog)
        prog2 = copy.deepcopy(prog)

        common_kwargs = {
            "convert_to": "mlprogram",
            "convert_from": "milinternal",
            "compute_units": ct.ComputeUnit.CPU_ONLY,
            "skip_model_load": True,
        }
        mlmodel1 = _mil_convert(prog1, **common_kwargs)
        mlmodel2 = _mil_convert(prog2, pass_pipeline=PassPipeline.EMPTY, **common_kwargs)

        assert get_op_types_in_program(mlmodel1._mil_program) == ["conv"]
        assert get_op_types_in_program(mlmodel2._mil_program) == ["conv", "add"]


@pytest.mark.skipif(ct.utils._macos_version() < (15, 0),
                    reason="Multi-function only supported on macOS 15+")
class TestMultiFunctionDescriptor:
    @staticmethod
    def _convert_multifunction_prog(prog):
        prog.export_as_multifunction = True
        mlmodel = _mil_convert(
            prog,
            convert_to="mlprogram",
            convert_from="milinternal",
            specification_version=_SPECIFICATION_VERSION_IOS_18,
            compute_units=ct.ComputeUnit.CPU_ONLY,
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
        if platform.machine() == "x86_64":
            pytest.xfail("rdar://137217263")

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
        default_model = ct.models.MLModel(saved_package_path)
        mil_file = open(os.path.join(default_model.get_compiled_model_path(), "model.mil"))
        mil_txt = mil_file.read()
        assert (
            mil_txt.count(
                'const()[name = string("const_0_to_fp16"), val = tensor<fp16, [8, 1, 5, 5]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];'
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
        if platform.machine() == "x86_64":
            pytest.xfail("rdar://137217263")

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
        default_model = ct.models.MLModel(saved_package_path)
        mil_file = open(os.path.join(default_model.get_compiled_model_path(), "model.mil"))
        mil_txt = mil_file.read()
        assert (
            mil_txt.count(
                'const()[name = string("const_0_to_fp16"), val = tensor<fp16, [8, 1, 5, 5]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];'
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


class TestMaterializeSymbolicShapeMLModel:
    FEATURE_DIM = 1024
    NUM_HEADS = 4
    MULTI_HEAD_OUT_FEATURE_DIM = 128

    MULTI_HEAD_IN_FEATURE_DIM = int(FEATURE_DIM / NUM_HEADS)
    OUT_FEATURE_DIM = int(NUM_HEADS * MULTI_HEAD_OUT_FEATURE_DIM)

    @staticmethod
    def initialte_weight_and_bias(in_features, out_features) -> Tuple[torch.Tensor]:
        stdv = 1.0 / np.sqrt(in_features)
        weight = torch.empty((out_features, in_features), dtype=torch.float16).uniform_(-stdv, stdv)
        bias = torch.empty(out_features, dtype=torch.float16).uniform_(-stdv, stdv)
        return weight, bias

    @staticmethod
    def create_multihead_torch_model() -> torch.nn.Module:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(
                    TestMaterializeSymbolicShapeMLModel.MULTI_HEAD_IN_FEATURE_DIM,
                    TestMaterializeSymbolicShapeMLModel.MULTI_HEAD_OUT_FEATURE_DIM,
                )
                self.relu = torch.nn.ReLU()

            def forward(self, x) -> torch.Tensor:
                multi_head_x_shape = (
                    x.shape[0],
                    x.shape[1],
                    TestMaterializeSymbolicShapeMLModel.NUM_HEADS,
                    TestMaterializeSymbolicShapeMLModel.MULTI_HEAD_IN_FEATURE_DIM,
                )
                x_multi_head = torch.reshape(x, multi_head_x_shape)
                x_batched_multi_head = torch.permute(x_multi_head, (0, 2, 1, 3))

                y_linear = self.fc(x_batched_multi_head)
                y_activated = self.relu(y_linear)
                y_multi_head = torch.permute(y_activated, (0, 2, 1, 3))
                y_shape = (
                    x.shape[0],
                    x.shape[1],
                    TestMaterializeSymbolicShapeMLModel.OUT_FEATURE_DIM,
                )
                y = torch.reshape(y_multi_head, y_shape)

                return y

        torch_model = Model()
        torch_model.eval()
        return torch_model

    @staticmethod
    def create_stateful_multihead_torch_model() -> torch.nn.Module:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(
                    TestMaterializeSymbolicShapeMLModel.MULTI_HEAD_IN_FEATURE_DIM,
                    TestMaterializeSymbolicShapeMLModel.MULTI_HEAD_OUT_FEATURE_DIM,
                )
                self.relu = torch.nn.ReLU()
                self.register_buffer(
                    "cache",
                    torch.zeros(
                        TestMaterializeSymbolicShapeMLModel.OUT_FEATURE_DIM, dtype=torch.float32
                    ),
                )

            def forward(self, x) -> torch.Tensor:
                multi_head_x_shape = (
                    x.shape[0],
                    x.shape[1],
                    TestMaterializeSymbolicShapeMLModel.NUM_HEADS,
                    TestMaterializeSymbolicShapeMLModel.MULTI_HEAD_IN_FEATURE_DIM,
                )
                x_multi_head = torch.reshape(x, multi_head_x_shape)
                x_batched_multi_head = torch.permute(x_multi_head, (0, 2, 1, 3))

                y_linear = self.fc(x_batched_multi_head)
                y_activated = self.relu(y_linear)
                y_multi_head = torch.permute(y_activated, (0, 2, 1, 3))
                y_shape = (
                    x.shape[0],
                    x.shape[1],
                    TestMaterializeSymbolicShapeMLModel.OUT_FEATURE_DIM,
                )
                y = torch.reshape(y_multi_head, y_shape)

                z = y + self.cache
                z_mean = torch.mean(z, dim=(0, 1))
                self.cache *= 0.8
                self.cache += 0.2 * z_mean

                return z

        torch_model = Model()
        torch_model.eval()
        return torch_model

    @staticmethod
    def create_intermediate_state_torch_model(leading_sizes, weight, bias) -> torch.nn.Module:
        class Model(torch.nn.Module):
            def __init__(self, leading_sizes, weight, bias) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(
                    TestMaterializeSymbolicShapeMLModel.MULTI_HEAD_IN_FEATURE_DIM,
                    TestMaterializeSymbolicShapeMLModel.MULTI_HEAD_OUT_FEATURE_DIM,
                )
                with torch.no_grad():
                    self.fc.weight.copy_(weight)
                    self.fc.bias.copy_(bias)
                self.relu = torch.nn.ReLU()
                self.register_buffer(
                    "cache",
                    torch.zeros(
                        (*leading_sizes, TestMaterializeSymbolicShapeMLModel.FEATURE_DIM),
                        dtype=torch.float32,
                    ),
                )

            def forward(self, x) -> torch.Tensor:
                self.cache *= 0.2
                self.cache += 0.8 * x
                x = self.cache

                multi_head_x_shape = (
                    x.shape[0],
                    x.shape[1],
                    TestMaterializeSymbolicShapeMLModel.NUM_HEADS,
                    TestMaterializeSymbolicShapeMLModel.MULTI_HEAD_IN_FEATURE_DIM,
                )
                x_multi_head = torch.reshape(x, multi_head_x_shape)
                x_batched_multi_head = torch.permute(x_multi_head, (0, 2, 1, 3))

                y_linear = self.fc(x_batched_multi_head)
                y_activated = self.relu(y_linear)
                y_multi_head = torch.permute(y_activated, (0, 2, 1, 3))
                y_shape = (
                    x.shape[0],
                    x.shape[1],
                    TestMaterializeSymbolicShapeMLModel.OUT_FEATURE_DIM,
                )
                y = torch.reshape(y_multi_head, y_shape)

                return y

        torch_model = Model(leading_sizes, weight, bias)
        torch_model.eval()
        return torch_model


    @staticmethod
    def read_mil_text(mlpackage_path: str) -> str:
        mlmodel = ct.models.MLModel(mlpackage_path)
        mil_file = open(os.path.join(mlmodel.get_compiled_model_path(), "model.mil"))
        mil_text = mil_file.read()
        return mil_text


    @pytest.mark.parametrize(
        "symbolic_shape, override_main_function, reload_mlmodel",
        itertools.product(
            (
                ct.EnumeratedShapes(
                    shapes=[[1, 3, FEATURE_DIM], [2, 5, FEATURE_DIM], [4, 7, FEATURE_DIM]],
                    default=[1, 3, FEATURE_DIM],
                ),
                (ct.RangeDim(1, 4, 1), ct.RangeDim(3, 7, 3), FEATURE_DIM),
            ),
            (True, False),
            (True, False),
        ),
    )
    def test_multihead(self, symbolic_shape, override_main_function, reload_mlmodel):
        new_function_name = "main" if override_main_function else "materialization_2_5"

        def export_symbolic_shape_mlmodel(torch_model: torch.nn.Module) -> ct.models.MLModel:
            example_input = torch.rand((1, 3, self.FEATURE_DIM))
            traced_model = torch.jit.trace(torch_model, (example_input,))

            ct_inputs = [ct.TensorType(name="x", shape=symbolic_shape, dtype=np.float16)]
            ct_outputs = [ct.TensorType(name="y")]

            symbolic_shape_mlmodel = ct.convert(
                traced_model,
                inputs=ct_inputs,
                outputs=ct_outputs,
                minimum_deployment_target=ct.target.iOS17,
                skip_model_load=True,
            )
            return symbolic_shape_mlmodel

        def validate_mil_text(multifunction_mlpackage_path: str) -> None:
            mil_text = self.read_mil_text(multifunction_mlpackage_path)
            if override_main_function:
                assert 1 == mil_text.count(
                    '(BLOBFILE(path = tensor<string, []>("@model_path/weights/weight.bin"), offset = tensor<uint64, []>(64)))'
                )
                assert 1 == mil_text.count(
                    '(BLOBFILE(path = tensor<string, []>("@model_path/weights/weight.bin"), offset = tensor<uint64, []>(65664)))'
                )
            else:
                assert 2 == mil_text.count(
                    '(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))'
                )
                assert 2 == mil_text.count(
                    '(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(65664)))'
                )

        def validate_inference(
            torch_model: torch.nn.Module,
            symbolic_shape_mlmodel: ct.models.MLModel,
            multifunction_mlpackage_path: str,
        ) -> None:
            size_to_function_name = {(2, 5): new_function_name}
            for size, function_name in size_to_function_name.items():
                mlmodel_materialized = ct.models.MLModel(
                    multifunction_mlpackage_path,
                    function_name=None if override_main_function else function_name,
                )
                x = torch.rand(*size, self.FEATURE_DIM)
                output_torch = torch_model(x).detach().numpy()
                output_symbolic = symbolic_shape_mlmodel.predict({"x": x.numpy()})["y"]
                output_materialized = mlmodel_materialized.predict({"x": x.numpy()})["y"]
                np.testing.assert_allclose(output_symbolic, output_torch, atol=5e-3, rtol=5e-3)
                np.testing.assert_allclose(output_materialized, output_torch, atol=5e-3, rtol=5e-3)

        torch_model = self.create_multihead_torch_model()

        symbolic_shape_mlmodel = export_symbolic_shape_mlmodel(torch_model)
        symbolic_mlpackage_path = tempfile.mkdtemp(suffix=".mlpackage")
        symbolic_shape_mlmodel.save(symbolic_mlpackage_path)
        if reload_mlmodel:
            symbolic_shape_mlmodel = ct.models.MLModel(
                symbolic_mlpackage_path, skip_model_load=True
            )

        multifunction_mlpackage_path = tempfile.mkdtemp(suffix=".mlpackage")
        ct.utils.materialize_dynamic_shape_mlmodel(
            symbolic_shape_mlmodel,
            {new_function_name: {"x": (2, 5, self.FEATURE_DIM)}},
            multifunction_mlpackage_path,
        )
        if override_main_function:
            assert (
                ct.models.MLModel(multifunction_mlpackage_path)._spec.specificationVersion
                == ct.target.iOS17
            )
        else:
            assert (
                ct.models.MLModel(multifunction_mlpackage_path)._spec.specificationVersion
                == ct.target.iOS18
            )

        # coremlcompiler had bug compiling the model < macOS 14
        if ct.utils._macos_version() >= (15, 0):
            validate_mil_text(multifunction_mlpackage_path)

        if platform.machine() == "arm64" and (
            override_main_function or ct.utils._macos_version() >= (15, 0)
        ):
            # Intel machines fails to run the model.
            # rdar://132919101 ([Bug] Intel machines fails on running several multifunction unittest)
            symbolic_shape_mlmodel = ct.models.MLModel(symbolic_mlpackage_path)
            validate_inference(torch_model, symbolic_shape_mlmodel, multifunction_mlpackage_path)

        shutil.rmtree(symbolic_mlpackage_path)
        shutil.rmtree(multifunction_mlpackage_path)

    @pytest.mark.skipif(
        ct.utils._macos_version() < (15, 0), reason="State only supported on macOS 15+"
    )
    @pytest.mark.xfail(reason="rdar://138957606 ([Bug] Stateful model regression in CoreML)")
    @pytest.mark.parametrize(
        "symbolic_shape, override_main_function, reload_mlmodel",
        itertools.product(
            (
                ct.EnumeratedShapes(
                    shapes=[[3, 1, FEATURE_DIM], [5, 2, FEATURE_DIM], [7, 4, FEATURE_DIM]],
                    default=[3, 1, FEATURE_DIM],
                ),
                (ct.RangeDim(3, 7, 3), ct.RangeDim(1, 4, 1), FEATURE_DIM),
            ),
            (True, False),
            (True, False),
        ),
    )
    def test_stateful_multihead(self, symbolic_shape, override_main_function, reload_mlmodel):
        new_function_name_1 = "main" if override_main_function else "materialization_5_2"
        new_function_name_2 = "materialization_7_4"

        def export_symbolic_shape_mlmodel(torch_model: torch.nn.Module) -> ct.models.MLModel:
            example_input = torch.rand((3, 1, self.FEATURE_DIM))
            traced_model = torch.jit.trace(torch_model, (example_input,))

            ct_inputs = [ct.TensorType(name="x", shape=symbolic_shape, dtype=np.float16)]
            ct_states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(self.OUT_FEATURE_DIM,), dtype=np.float16),
                    name="cache",
                )
            ]
            ct_outputs = [ct.TensorType(name="y")]

            symbolic_shape_mlmodel = ct.convert(
                traced_model,
                inputs=ct_inputs,
                states=ct_states,
                outputs=ct_outputs,
                minimum_deployment_target=ct.target.iOS18,
                skip_model_load=True,
            )
            return symbolic_shape_mlmodel

        def validate_mil_text(multifunction_mlpackage_path: str) -> None:
            mil_text = self.read_mil_text(multifunction_mlpackage_path)
            expected_counts = 2 if override_main_function else 3
            assert expected_counts == mil_text.count(
                '(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))'
            )
            assert expected_counts == mil_text.count(
                '(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(65664)))'
            )

        def validate_inference(
            torch_model: torch.nn.Module,
            symbolic_shape_mlmodel: ct.models.MLModel,
            multifunction_mlpackage_path: str,
        ) -> None:
            size_to_function_name = {(5, 2): new_function_name_1, (7, 4): new_function_name_2}
            for size, function_name in size_to_function_name.items():
                mlmodel_materialized = ct.models.MLModel(
                    multifunction_mlpackage_path, function_name=function_name
                )
                torch_model.cache.fill_(0.0)
                cache_main = symbolic_shape_mlmodel.make_state()
                cache_materialized = mlmodel_materialized.make_state()
                for _ in range(10):
                    x = torch.rand(*size, self.FEATURE_DIM)
                    output_torch = torch_model(x).detach().numpy()
                    output_symbolic = symbolic_shape_mlmodel.predict(
                        {"x": x.numpy()}, state=cache_main
                    )["y"]
                    output_materialized = mlmodel_materialized.predict(
                        {"x": x.numpy()}, state=cache_materialized
                    )["y"]
                    np.testing.assert_allclose(output_symbolic, output_torch, atol=5e-3, rtol=5e-3)
                    np.testing.assert_allclose(
                        output_materialized, output_torch, atol=5e-3, rtol=5e-3
                    )

        torch_model = self.create_stateful_multihead_torch_model()

        symbolic_shape_mlmodel = export_symbolic_shape_mlmodel(torch_model)
        symbolic_mlpackage_path = tempfile.mkdtemp(suffix=".mlpackage")
        symbolic_shape_mlmodel.save(symbolic_mlpackage_path)
        if reload_mlmodel:
            symbolic_shape_mlmodel = ct.models.MLModel(
                symbolic_mlpackage_path, skip_model_load=True
            )

        multifunction_mlpackage_path = tempfile.mkdtemp(suffix=".mlpackage")
        ct.utils.materialize_dynamic_shape_mlmodel(
            symbolic_shape_mlmodel,
            {
                new_function_name_1: {"x": (5, 2, self.FEATURE_DIM)},
                new_function_name_2: {"x": (7, 4, self.FEATURE_DIM)},
            },
            multifunction_mlpackage_path,
        )

        validate_mil_text(multifunction_mlpackage_path)

        if platform.machine() == "arm64" and (
            override_main_function or ct.utils._macos_version() >= (15, 0)
        ):
            # Intel machines fails to run the model.
            # rdar://132919101 ([Bug] Intel machines fails on running several multifunction unittest)
            symbolic_shape_mlmodel = ct.models.MLModel(symbolic_mlpackage_path)
            validate_inference(torch_model, symbolic_shape_mlmodel, multifunction_mlpackage_path)

        shutil.rmtree(symbolic_mlpackage_path)
        shutil.rmtree(multifunction_mlpackage_path)

    @pytest.mark.skipif(
        ct.utils._macos_version() < (15, 0), reason="State only supported on macOS 15+"
    )
    def test_advanced_intermediate_state(self):
        WEIGHT, BIAS = self.initialte_weight_and_bias(
            self.MULTI_HEAD_IN_FEATURE_DIM, self.MULTI_HEAD_OUT_FEATURE_DIM
        )

        def export_symbolic_shape_program() -> Program:
            leading_sizes = (1, 3)

            torch_model = self.create_intermediate_state_torch_model(leading_sizes, WEIGHT, BIAS)

            x_shape = (*leading_sizes, self.FEATURE_DIM)
            x = torch.rand(x_shape)
            traced_model = torch.jit.trace(torch_model, x)

            x_dynamic_shape = (ct.RangeDim(1, 1024), ct.RangeDim(1, 1024), self.FEATURE_DIM)
            ct_inputs = [ct.TensorType(name="x", shape=x_dynamic_shape, dtype=np.float16)]
            ct_states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=x_dynamic_shape, dtype=np.float16),
                    name="cache",
                )
            ]
            symbolic_shape_prog = ct.convert(
                traced_model,
                inputs=ct_inputs,
                states=ct_states,
                minimum_deployment_target=ct.target.iOS18,
                convert_to="milinternal",
            )
            return symbolic_shape_prog

        def export_fixed_shape_mlmodel(leading_sizes) -> ct.models.MLModel:
            torch_model = self.create_intermediate_state_torch_model(leading_sizes, WEIGHT, BIAS)

            x_shape = (*leading_sizes, self.FEATURE_DIM)
            x = torch.rand(x_shape)
            traced_model = torch.jit.trace(torch_model, x)

            ct_inputs = [ct.TensorType(name="x", shape=x_shape, dtype=np.float16)]
            ct_states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=x_shape, dtype=np.float16), name="cache"
                )
            ]
            fixed_shape_mlmodel = ct.convert(
                traced_model,
                inputs=ct_inputs,
                states=ct_states,
                minimum_deployment_target=ct.target.iOS18,
                compute_units=ct.ComputeUnit.CPU_ONLY,
            )
            return fixed_shape_mlmodel

        def materialize_dynamic_shape_program(
            dynamic_shape_prog: Program,
            function_name_to_materialization_map: Dict[str, Dict[str, Tuple[int]]],
            destination_path: str,
        ) -> None:
            # Materialize symbolic shapes, then run all optimization passes
            pass_pipeline = ct.PassPipeline.DEFAULT
            # If dynamic shape prog is obtained from `ct.convert(convert_to="milinternal")`,
            # then names are not sanitized. What is worse, mil_backend::sanitize_name_strings
            # does not work for multifunction pymil program. As a result,
            # we explicitly add mil_backend::sanitize_name_strings before materialization
            # TODO (rdar://131726375) Have mil_backend::sanitize_name_strings work on multifunction
            pass_pipeline.insert_pass(0, "mil_backend::sanitize_name_strings")
            pass_pipeline.insert_pass(1, "common::materialize_symbolic_shape_program")
            pass_pipeline.set_options(
                "common::materialize_symbolic_shape_program",
                {
                    "function_name_to_materialization_map": function_name_to_materialization_map,
                },
            )
            PassPipelineManager.apply_pipeline(dynamic_shape_prog, pass_pipeline)

            # Source function may no longer be needed,
            # e.g. if it has intermediate symbolic-shape state
            dynamic_shape_prog.functions.pop("main")
            dynamic_shape_prog.default_function_name = list(
                function_name_to_materialization_map.keys()
            )[0]

            dynamic_shape_prog.skip_all_passes = True
            dynamic_shape_prog.export_as_multifunction = True
            materialized_mlmodel = _mil_convert(
                dynamic_shape_prog,
                convert_from="milinternal",
                convert_to="mlprogram",
                specification_version=ct.target.iOS18,
                compute_units=ct.ComputeUnit.CPU_ONLY,
                skip_model_load=True,
            )
            materialized_mlmodel.save(destination_path)

        def validate_mil_text(multifunction_mlpackage_path: str) -> None:
            mil_text = self.read_mil_text(multifunction_mlpackage_path)
            assert 2 == mil_text.count(
                '(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))'
            )
            assert 2 == mil_text.count(
                '(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(65664)))'
            )

        def validate_inference(multifunction_mlpackage_path: str) -> None:
            size_to_function_name = {(2, 5): "materialization_2_5", (4, 7): "materialization_4_7"}
            for leading_sizes, function_name in size_to_function_name.items():
                torch_model = self.create_intermediate_state_torch_model(
                    leading_sizes, WEIGHT, BIAS
                )
                mlmodel_unifunction = export_fixed_shape_mlmodel(leading_sizes)
                mlmodel_multifunction = ct.models.MLModel(
                    multifunction_mlpackage_path,
                    function_name=function_name,
                    compute_units=ct.ComputeUnit.CPU_ONLY,
                )

                torch_model.cache.fill_(0.0)
                cache_unifunction = mlmodel_unifunction.make_state()
                cache_multifunction = mlmodel_multifunction.make_state()

                for _ in range(3):
                    x = torch.rand(size=(*leading_sizes, self.FEATURE_DIM), dtype=torch.float16)
                    output_torch = torch_model(x).detach().numpy()
                    output_unifunction = list(
                        mlmodel_unifunction.predict(
                            {"x": x.numpy()}, state=cache_unifunction
                        ).values()
                    )[0]
                    output_multifunction = list(
                        mlmodel_multifunction.predict(
                            {"x": x.numpy()}, state=cache_multifunction
                        ).values()
                    )[0]
                    np.testing.assert_allclose(
                        output_unifunction, output_torch, atol=5e-3, rtol=5e-3
                    )
                    np.testing.assert_allclose(
                        output_multifunction, output_torch, atol=5e-3, rtol=5e-3
                    )

        symbolic_shape_prog = export_symbolic_shape_program()

        multifunction_mlpackage_path = tempfile.mkdtemp(suffix=".mlpackage")
        materialize_dynamic_shape_program(
            symbolic_shape_prog,
            {
                "materialization_2_5": {
                    "x": (2, 5, self.FEATURE_DIM),
                    "cache": (2, 5, self.FEATURE_DIM),
                },
                "materialization_4_7": {
                    "x": (4, 7, self.FEATURE_DIM),
                    "cache": (4, 7, self.FEATURE_DIM),
                },
            },
            multifunction_mlpackage_path,
        )

        validate_mil_text(multifunction_mlpackage_path)
        validate_inference(multifunction_mlpackage_path)

        shutil.rmtree(multifunction_mlpackage_path)


class TestBisectModel:

    @staticmethod
    def check_spec_op_type(model_path, expected_ops):
        spec = load_spec(model_path)
        mil = spec.mlProgram
        for function in mil.functions.values():
            for block in function.block_specializations.values():
                ops = list(block.operations)
                for i, op_type in enumerate(expected_ops):
                    assert ops[i].type == op_type

    @staticmethod
    def get_test_model_path(minimum_deployment_target=ct.target.iOS16, return_as_mlmodel=False):
        # pytorch model and tracing
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(6000, 6000)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(6000, 6000)

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                x = torch.sin(x)
                return x

        example_input = torch.rand(1, 6000)
        model = Model().eval()
        traced_model = torch.jit.trace(model, example_input)

        # convert to mlpackage
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(1, 6000), name="input")],
            minimum_deployment_target=minimum_deployment_target,
        )

        # return as mlmodel
        if return_as_mlmodel:
            return mlmodel

        # save on disk and return the model path
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel.save(package_path)

        return package_path

    def test_invalid_mlpackage(self):
        traced_model = TestMultiFunctionModelEnd2End._get_test_model()
        input = np.random.rand(1, 1, 28, 28)
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="x", shape=(1, 1, 28, 28))],
            outputs=[ct.TensorType(name="out")],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS16,
        )
        package_path = tempfile.mkdtemp(suffix=".mlpackage")
        mlmodel.save(package_path)

        # function name other than "main" will error out
        desc = MultiFunctionDescriptor()
        desc.add_function(package_path, "main", "main_1")
        desc.default_function_name = "main_1"
        saved_package_path = tempfile.mkdtemp(suffix=".mlpackage")
        save_multifunction(desc, saved_package_path)

        with tempfile.TemporaryDirectory() as output_dir:
            with pytest.raises(ValueError, match="only support model with a single"):
                bisect_model(
                    saved_package_path,
                    output_dir=output_dir,
                )
            shutil.rmtree(saved_package_path)

            # multi-function model is not supported
            desc = MultiFunctionDescriptor()
            desc.add_function(package_path, "main", "main")
            desc.add_function(package_path, "main", "main_1")
            desc.default_function_name = "main"
            saved_package_path = tempfile.mkdtemp(suffix=".mlpackage")
            save_multifunction(desc, saved_package_path)
            with pytest.raises(ValueError, match="only support model with a single"):
                bisect_model(
                    saved_package_path,
                    output_dir=output_dir,
                )
            shutil.rmtree(saved_package_path)
            shutil.rmtree(package_path)

    @pytest.mark.parametrize(
        "mlmodel_as_input",
        [True, False],
    )
    def test_pipeline(self, mlmodel_as_input):
        model = self.get_test_model_path(return_as_mlmodel=mlmodel_as_input)
        output_dir = str(tempfile.TemporaryDirectory())

        # The API will bisect the model into two chunks, and produces a pipeline model
        bisect_model(
            model,
            output_dir,
            merge_chunks_to_pipeline=True,
        )

        # check the file name is correct
        if mlmodel_as_input:
            name = ""
        else:
            mlpackage_name = os.path.basename(model)
            name, _ = os.path.splitext(mlpackage_name)
            name += "_"

        pipeline_path = os.path.join(output_dir, f"{name}chunked_pipeline.mlpackage")
        assert os.path.isdir(pipeline_path)

        # check the Core ML model is a pipeline model
        spec = load_spec(pipeline_path)
        assert spec.WhichOneof("Type") == "pipeline"

        # cleanup
        if not mlmodel_as_input:
            shutil.rmtree(model)
        shutil.rmtree(output_dir)

    def test_compressed_model(self):
        # use coremltools.optimizee to palettize a Core ML model
        model = self.get_test_model_path(return_as_mlmodel=True)
        op_config = cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=8)
        config = cto.coreml.OptimizationConfig(global_config=op_config)
        model = cto.coreml.palettize_weights(model, config)

        # test that the bisect API works
        output_dir = str(tempfile.TemporaryDirectory())
        bisect_model(
            model,
            output_dir,
        )

        # test the models contain correct ops
        name = ""
        chunk1_path = os.path.join(output_dir, f"{name}chunk1.mlpackage")
        chunk2_path = os.path.join(output_dir, f"{name}chunk2.mlpackage")
        assert os.path.isdir(chunk1_path)
        assert os.path.isdir(chunk2_path)

        self.check_spec_op_type(
            chunk1_path,
            [
                "constexpr_lut_to_dense",
                "const",
                "linear",
                "const",
                "cast",
            ]
        )
        self.check_spec_op_type(
            chunk2_path,
            [
                "const",
                "cast",
                "relu",
                "constexpr_lut_to_dense",
                "const",
                "linear",
                "sin",
            ]
        )

        # cleanup
        shutil.rmtree(output_dir)


    @pytest.mark.parametrize(
        "mlmodel_as_input",
        [True, False],
    )
    def test_basic(self, mlmodel_as_input):
        def check_spec_version(model_path, expected_spec_version):
            spec = load_spec(model_path)
            assert spec.specificationVersion == expected_spec_version

        def check_output_dtype(model_path, expected_output_dtype):
            spec = load_spec(model_path)
            assert_spec_output_type(spec, DTYPE_TO_FEATURE_TYPE_MAP[expected_output_dtype])

        def check_input_dtype(model_path, expected_input_dtype):
            spec = load_spec(model_path)
            assert_spec_input_type(spec, DTYPE_TO_FEATURE_TYPE_MAP[expected_input_dtype])


        model = self.get_test_model_path(ct.target.iOS17, return_as_mlmodel=mlmodel_as_input)
        output_dir = str(tempfile.TemporaryDirectory())

        # By bisecting the model into half, there will be two new mlpackages, with suffix `_chunk1.mlpackage` and `_chunk2.mlpackage`
        # in the target `output_dir`.
        bisect_model(
            model,
            output_dir,
        )

        # check the API doesn't delete the original mlpackage
        if not mlmodel_as_input:
            assert os.path.isdir(model)

        # check the file names are correct
        if mlmodel_as_input:
            name = ""
        else:
            mlpackage_name = os.path.basename(model)
            name, _ = os.path.splitext(mlpackage_name)
            name += "_"

        chunk1_path = os.path.join(output_dir, f"{name}chunk1.mlpackage")
        chunk2_path = os.path.join(output_dir, f"{name}chunk2.mlpackage")
        assert os.path.isdir(chunk1_path)
        assert os.path.isdir(chunk2_path)

        # check the model op type
        self.check_spec_op_type(
            chunk1_path,
            [
                "const",
                "const",
                "linear",
                "const",
                "cast",
            ]
        )
        self.check_spec_op_type(
            chunk2_path,
            [
                "const",
                "cast",
                "relu",
                "const",
                "const",
                "linear",
                "sin",
            ]
        )

        # check the spec has the correct version
        check_spec_version(chunk1_path, ct.target.iOS17)
        check_spec_version(chunk2_path, ct.target.iOS17)

        # the i/o dtype of the two chunk models should be:
        # 1. fp16 -> fp32
        # 2. fp32 -> fp16
        check_input_dtype(chunk1_path, "fp16")
        check_output_dtype(chunk1_path, "fp32")

        check_input_dtype(chunk2_path, "fp32")
        check_output_dtype(chunk2_path, "fp16")

        # cleanup
        if not mlmodel_as_input:
            shutil.rmtree(model)
        shutil.rmtree(output_dir)

    def test_api_example(self):
        """
        Test the API example in https://apple.github.io/coremltools/docs-guides/source/mlmodel-utilities.html
        """
        model_path = self.get_test_model_path()
        output_dir = str(tempfile.TemporaryDirectory())

        # The following code will produce two chunks models:
        # `./output/my_model_chunk1.mlpackage` and `./output/my_model_chunk2.mlpackage`
        ct.models.utils.bisect_model(
            model_path,
            output_dir,
        )

        # The following code will produce a single pipeline model `./output/my_model_chunked_pipeline.mlpackage`
        ct.models.utils.bisect_model(
            model_path,
            output_dir,
            merge_chunks_to_pipeline=True,
        )

        # You can also pass the MLModel object directly
        mlmodel = ct.models.MLModel(model_path)
        ct.models.utils.bisect_model(
            mlmodel,
            output_dir,
        )

        # clean up
        shutil.rmtree(output_dir)
        shutil.rmtree(model_path)


class TestChangeInputOutputTensorType:
    @staticmethod
    def _build_simple_model(dtype, function_names):
        weight_dtype = np.float16 if dtype == types.fp16 else np.float32

        @mb.function(input_specs=[mb.TensorSpec(shape=(1, 3), dtype=dtype)], opset_version=ct.target.iOS16)
        def func(x):
            return mb.linear(x=x, weight=np.random.rand(2, 3).astype(weight_dtype))

        prog = mil.Program()
        for function_name in function_names:
            prog.add_function(function_name, func)

        return _mil_convert(
            prog,
            convert_to="mlprogram",
            convert_from="milinternal",
            specification_version=ct.target.iOS16,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            skip_model_load=True,
        )

    @pytest.mark.parametrize(
        "dtype, from_feature_type, to_feature_type", (
                (types.fp16, ArrayFeatureType.FLOAT16, ArrayFeatureType.FLOAT32),
                (types.fp32, ArrayFeatureType.FLOAT32, ArrayFeatureType.FLOAT16),
        )
    )
    def test_change_input_type(self, dtype, from_feature_type, to_feature_type) -> None:
        model = self._build_simple_model(dtype=dtype, function_names=["main"])
        orig_input = model.get_spec().description.input[0]
        assert orig_input.type.multiArrayType.dataType == from_feature_type

        updated_model = change_input_output_tensor_type(
            ml_model=model,
            from_type=from_feature_type,
            to_type=to_feature_type,
            input_names=["x"],
            output_names=[],
        )
        updated_input = updated_model.get_spec().description.input[0]
        assert updated_input.type.multiArrayType.dataType == to_feature_type

    @pytest.mark.parametrize(
        "dtype, from_feature_type, to_feature_type", (
                (types.fp16, ArrayFeatureType.FLOAT16, ArrayFeatureType.FLOAT32),
                (types.fp32, ArrayFeatureType.FLOAT32, ArrayFeatureType.FLOAT16),
        )
    )
    def test_change_input_type_multifunc(self, dtype, from_feature_type, to_feature_type) -> None:
        function_names = ["main", "main_2"]
        model = self._build_simple_model(dtype=dtype, function_names=function_names)
        for orig_output in model.get_spec().description.output:
            assert orig_output.type.multiArrayType.dataType == from_feature_type

        updated_model = change_input_output_tensor_type(
            ml_model=model,
            from_type=from_feature_type,
            to_type=to_feature_type,
            function_names=function_names,
            input_names=["*"],
            output_names=[],
        )
        for updated_input in updated_model.get_spec().description.input:
            assert updated_input.type.multiArrayType.dataType == to_feature_type

    @pytest.mark.parametrize(
        "dtype, from_feature_type, to_feature_type", (
                (types.fp16, ArrayFeatureType.FLOAT16, ArrayFeatureType.FLOAT32),
                (types.fp32, ArrayFeatureType.FLOAT32, ArrayFeatureType.FLOAT16),
        )
    )
    def test_change_output_type(self, dtype, from_feature_type, to_feature_type) -> None:
        model = self._build_simple_model(dtype=dtype, function_names=["main"])
        orig_output = model.get_spec().description.output[0]
        assert orig_output.type.multiArrayType.dataType == from_feature_type

        updated_model = change_input_output_tensor_type(
            ml_model=model,
            from_type=from_feature_type,
            to_type=to_feature_type,
        )
        updated_output = updated_model.get_spec().description.output[0]
        assert updated_output.type.multiArrayType.dataType == to_feature_type

    @pytest.mark.parametrize(
        "dtype, from_feature_type, to_feature_type", (
                (types.fp16, ArrayFeatureType.FLOAT16, ArrayFeatureType.FLOAT32),
                (types.fp32, ArrayFeatureType.FLOAT32, ArrayFeatureType.FLOAT16),
        )
    )
    def test_change_output_type_multifunc(self, dtype, from_feature_type, to_feature_type) -> None:
        function_names = ["main", "main_2"]
        model = self._build_simple_model(dtype=dtype, function_names=function_names)
        for orig_output in model.get_spec().description.output:
            assert orig_output.type.multiArrayType.dataType == from_feature_type

        updated_model = change_input_output_tensor_type(
            ml_model=model,
            from_type=from_feature_type,
            to_type=to_feature_type,
            function_names=function_names,
        )
        for updated_output in updated_model.get_spec().description.output:
            assert updated_output.type.multiArrayType.dataType == to_feature_type
