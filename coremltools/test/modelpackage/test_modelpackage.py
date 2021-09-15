# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import coremltools
from coremltools.libmodelpackage import ModelPackage
from coremltools.proto import Model_pb2
from coremltools import utils

import numpy as np
import os
import pytest
import shutil
import tempfile
import unittest

from coremltools.models.utils import (
    rename_feature,
    save_spec,
    _macos_version,
    _convert_neural_network_spec_weights_to_fp16,
    convert_double_to_float_multiarray_type,
)
from coremltools.models import MLModel, datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder


class MLModelTest(unittest.TestCase):
    @staticmethod
    def _remove_path(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    @classmethod
    def setUpClass(self):

        spec = Model_pb2.Model()
        spec.specificationVersion = coremltools.SPECIFICATION_VERSION

        features = ["feature_1", "feature_2"]
        output = "output"
        for f in features:
            input_ = spec.description.input.add()
            input_.name = f
            input_.type.doubleType.MergeFromString(b"")

        output_ = spec.description.output.add()
        output_.name = output
        output_.type.doubleType.MergeFromString(b"")

        lr = spec.glmRegressor
        lr.offset.append(0.1)
        weights = lr.weights.add()
        coefs = [1.0, 2.0]
        for i in coefs:
            weights.value.append(i)

        spec.description.predictedFeatureName = "output"
        self.spec = spec

    def test_model_creation(self):
        model = MLModel(self.spec)
        self.assertIsNotNone(model)

        package = tempfile.TemporaryDirectory(suffix=".mlpackage")
        package.cleanup()

        save_spec(self.spec, package.name)
        model = MLModel(package.name)
        self.assertIsNotNone(model)

        # cleanup
        MLModelTest._remove_path(package.name)

    def test_model_api(self):
        model = MLModel(self.spec)
        self.assertIsNotNone(model)

        model.author = "Test author"
        self.assertEqual(model.author, "Test author")
        self.assertEqual(model.get_spec().description.metadata.author, "Test author")

        model.license = "Test license"
        self.assertEqual(model.license, "Test license")
        self.assertEqual(model.get_spec().description.metadata.license, "Test license")

        model.short_description = "Test model"
        self.assertEqual(model.short_description, "Test model")
        self.assertEqual(
            model.get_spec().description.metadata.shortDescription, "Test model"
        )

        model.version = "1.3"
        self.assertEqual(model.version, "1.3")
        self.assertEqual(model.get_spec().description.metadata.versionString, "1.3")

        model.input_description["feature_1"] = "This is feature 1"
        self.assertEqual(model.input_description["feature_1"], "This is feature 1")

        model.output_description["output"] = "This is output"
        self.assertEqual(model.output_description["output"], "This is output")

        package = tempfile.TemporaryDirectory(suffix=".mlpackage")
        package.cleanup()

        model.save(package.name)
        loaded_model = MLModel(package.name)

        self.assertEqual(model.author, "Test author")
        self.assertEqual(model.license, "Test license")
        self.assertEqual(model.short_description, "Test model")
        self.assertEqual(model.input_description["feature_1"], "This is feature 1")
        self.assertEqual(model.output_description["output"], "This is output")

        # cleanup
        MLModelTest._remove_path(package.name)

    def test_predict_api(self):
        model = MLModel(self.spec)

        package = tempfile.TemporaryDirectory(suffix=".mlpackage")
        package.cleanup()

        model.save(package.name)

        if utils._macos_version() >= (12, 0):
            for compute_units in coremltools.ComputeUnit:
                loaded_model = MLModel(package.name, compute_units=compute_units)

                preds = loaded_model.predict({"feature_1": 1.0, "feature_2": 1.0})
                self.assertIsNotNone(preds)
                self.assertEqual(preds["output"], 3.1)
                self.assertEqual(loaded_model.compute_unit, compute_units)
        else:
            # just check if we can load it
            loaded_model = MLModel(package.name)

        # cleanup
        MLModelTest._remove_path(package.name)

    def test_rename_input(self):
        rename_feature(self.spec, "feature_1", "renamed_feature", rename_inputs=True)
        model = MLModel(self.spec)

        package = tempfile.TemporaryDirectory(suffix=".mlpackage")
        package.cleanup()

        model.save(package.name)
        loaded_model = MLModel(package.name)

        if utils._macos_version() >= (12, 0):
            preds = loaded_model.predict({"renamed_feature": 1.0, "feature_2": 1.0})
            self.assertIsNotNone(preds)
            self.assertEqual(preds["output"], 3.1)

        # reset the spec for next run
        rename_feature(self.spec, "renamed_feature", "feature_1", rename_inputs=True)

        # cleanup
        MLModelTest._remove_path(package.name)

    def test_rename_input_bad(self):
        rename_feature(self.spec, "blah", "bad_name", rename_inputs=True)
        model = MLModel(self.spec)

        package = tempfile.TemporaryDirectory(suffix=".mlpackage")
        package.cleanup()

        model.save(package.name)
        loaded_model = MLModel(package.name)

        if utils._macos_version() >= (12, 0):
            preds = loaded_model.predict({"feature_1": 1.0, "feature_2": 1.0})
            self.assertIsNotNone(preds)
            self.assertEqual(preds["output"], 3.1)

        # cleanup
        MLModelTest._remove_path(package.name)

    def test_save(self):
        model = MLModel(self.spec)

        # Verify "save" can be called twice and the saved
        # model can be loaded successfully each time
        for _ in range(0, 2):
            package = tempfile.TemporaryDirectory(suffix=".mlpackage")
            package.cleanup()

            model.save(package.name)
            loaded_model = MLModel(package.name)

            if utils._macos_version() >= (12, 0):
                preds = loaded_model.predict({"feature_1": 1.0, "feature_2": 1.0})
                self.assertIsNotNone(preds)
                self.assertEqual(preds["output"], 3.1)

            MLModelTest._remove_path(package.name)

    def test_save_in_place(self):
        model = MLModel(self.spec)

        # Verify "save" can be called twice and the saved
        # model can be loaded successfully each time
        # the mlpackage remains in place after the first save
        package = tempfile.TemporaryDirectory(suffix=".mlpackage")
        package.cleanup()
        for _ in range(2):

            model.save(package.name)
            loaded_model = MLModel(package.name)

            if utils._macos_version() >= (12, 0):
                preds = loaded_model.predict({"feature_1": 1.0, "feature_2": 1.0})
                self.assertIsNotNone(preds)
                self.assertEqual(preds["output"], 3.1)

        MLModelTest._remove_path(package.name)

    def test_mil_as_package(self):
        import torch

        num_tokens = 3
        embedding_size = 5

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.embedding = torch.nn.Embedding(num_tokens, embedding_size)

            def forward(self, x):
                return self.embedding(x)

        model = TestModule()
        model.eval()

        example_input = torch.randint(high=num_tokens, size=(2,), dtype=torch.int64)
        traced_model = torch.jit.trace(model, example_input)

        temp_package_dir = tempfile.TemporaryDirectory(suffix=".mlpackage")
        for converted_package_path in [None, temp_package_dir.name]:
            mlmodel = coremltools.convert(
                traced_model,
                package_dir = converted_package_path,
                source='pytorch',
                convert_to='mlprogram',
                compute_precision=coremltools.precision.FLOAT32,
                inputs=[
                    coremltools.TensorType(
                        name="input",
                        shape=example_input.shape,
                        dtype=example_input.numpy().dtype,
                    )
                ],
            )

            assert isinstance(mlmodel, MLModel)

            package_path = tempfile.mkdtemp(suffix=".mlpackage")
            mlmodel.save(package_path)

            assert ModelPackage.isValid(package_path)
            assert os.path.exists(ModelPackage(package_path).getRootModel().path())

            # Read back the saved bundle and compile
            mlmodel2 = MLModel(package_path, useCPUOnly=True)

            if utils._macos_version() >= (12, 0):
                result = mlmodel2.predict(
                    {"input": example_input.cpu().detach().numpy().astype(np.float32)}
                )

                # Verify outputs
                expected = model(example_input)
                name = list(result.keys())[0]
                np.testing.assert_allclose(result[name], expected.cpu().detach().numpy())

            # Cleanup package
            shutil.rmtree(package_path)

            tmp_package_path = mlmodel.package_path
            assert os.path.exists(tmp_package_path)
            del mlmodel
            if converted_package_path is not None:
                # Verify we leave the provided package dir alone
                assert os.path.exists(tmp_package_path)

        temp_package_dir.cleanup()

    def test_model_save_no_extension(self):
        import torch

        num_tokens = 3
        embedding_size = 5

        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.embedding = torch.nn.Embedding(num_tokens, embedding_size)

            def forward(self, x):
                return self.embedding(x)

        model = TestModule()
        model.eval()

        example_input = torch.randint(high=num_tokens, size=(2,), dtype=torch.int64)
        traced_model = torch.jit.trace(model, example_input)

        mlmodel = coremltools.convert(
            traced_model,
            package_dir=None,
            source='pytorch',
            convert_to='mlprogram',
            inputs=[
                coremltools.TensorType(
                    name="input",
                    shape=example_input.shape,
                    dtype=example_input.numpy().dtype,
                )
            ],
        )
        assert isinstance(mlmodel, MLModel)

        package = tempfile.TemporaryDirectory(suffix="")
        package.cleanup()
        package_path = package.name

        mlmodel.save(package_path)
        assert not os.path.exists(package_path)

        package_path = package_path + ".mlpackage"
        assert os.path.exists(package_path)

        shutil.rmtree(package_path)

if __name__ == "__main__":
    unittest.main()
