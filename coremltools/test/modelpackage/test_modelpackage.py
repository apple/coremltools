# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os
import shutil
import tempfile

import numpy as np
import pytest

import coremltools
from coremltools import ComputeUnit, utils
from coremltools.converters.mil import Builder as mb
from coremltools.libmodelpackage import ModelPackage
from coremltools.models import MLModel
from coremltools.models.utils import (_MLPACKAGE_AUTHOR_NAME,
                                      _WEIGHTS_DIR_NAME)
from coremltools.proto import Model_pb2


def _remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

class TestMLModel:

    def setup_class(self):

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
        assert model is not None

        package = tempfile.TemporaryDirectory(suffix=".mlpackage")
        package.cleanup()

        utils.save_spec(self.spec, package.name)
        model = MLModel(package.name)
        assert model is not None

        # cleanup
        _remove_path(package.name)

    def test_model_api(self):
        model = MLModel(self.spec)
        assert model is not None

        model.author = "Test author"
        assert model.author == "Test author"
        assert model.get_spec().description.metadata.author == "Test author"

        model.license = "Test license"
        assert model.license == "Test license"
        assert model.get_spec().description.metadata.license == "Test license"

        model.short_description = "Test model"
        assert model.short_description == "Test model"
        assert model.get_spec().description.metadata.shortDescription == "Test model"

        model.version = "1.3"
        assert model.version == "1.3"
        assert model.get_spec().description.metadata.versionString == "1.3"

        model.input_description["feature_1"] = "This is feature 1"
        assert model.input_description["feature_1"] == "This is feature 1"

        model.output_description["output"] = "This is output"
        assert model.output_description["output"] == "This is output"

        package = tempfile.TemporaryDirectory(suffix=".mlpackage")
        package.cleanup()

        model.save(package.name)
        loaded_model = MLModel(package.name)

        assert model.author == "Test author"
        assert model.license == "Test license"
        assert model.short_description == "Test model"
        assert model.input_description["feature_1"] == "This is feature 1"
        assert model.output_description["output"] == "This is output"

        # cleanup
        _remove_path(package.name)

    def test_predict_api(self):
        model = MLModel(self.spec)

        package = tempfile.TemporaryDirectory(suffix=".mlpackage")
        package.cleanup()

        model.save(package.name)

        if utils._macos_version() >= (12, 0):
            for compute_units in coremltools.ComputeUnit:
                if (compute_units == coremltools.ComputeUnit.CPU_AND_NE
                    and utils._macos_version() < (13, 0)):
                    continue

                loaded_model = MLModel(package.name, compute_units=compute_units)

                preds = loaded_model.predict({"feature_1": 1.0, "feature_2": 1.0})
                assert preds is not None
                assert preds["output"] == 3.1
                assert loaded_model.compute_unit == compute_units
        else:
            # just check if we can load it
            loaded_model = MLModel(package.name)

        # cleanup
        _remove_path(package.name)

    def test_rename_input(self):
        utils.rename_feature(self.spec, "feature_1", "renamed_feature", rename_inputs=True)
        model = MLModel(self.spec)

        package = tempfile.TemporaryDirectory(suffix=".mlpackage")
        package.cleanup()

        model.save(package.name)
        loaded_model = MLModel(package.name)

        if utils._macos_version() >= (12, 0):
            preds = loaded_model.predict({"renamed_feature": 1.0, "feature_2": 1.0})
            assert preds is not None
            assert preds["output"] == 3.1

        # reset the spec for next run
        utils.rename_feature(self.spec, "renamed_feature", "feature_1", rename_inputs=True)

        # cleanup
        _remove_path(package.name)

    def test_rename_input_bad(self):
        utils.rename_feature(self.spec, "blah", "bad_name", rename_inputs=True)
        model = MLModel(self.spec)

        package = tempfile.TemporaryDirectory(suffix=".mlpackage")
        package.cleanup()

        model.save(package.name)
        loaded_model = MLModel(package.name)

        if utils._macos_version() >= (12, 0):
            preds = loaded_model.predict({"feature_1": 1.0, "feature_2": 1.0})
            assert preds is not None
            assert preds["output"] == 3.1

        # cleanup
        _remove_path(package.name)

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
                assert preds is not None
                assert preds["output"] == 3.1

            _remove_path(package.name)

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
                assert preds is not None
                assert preds["output"] == 3.1

        _remove_path(package.name)

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
            mlmodel2 = MLModel(package_path, compute_units=ComputeUnit.CPU_ONLY)

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

class TestSpecAndMLModelAPIs:

    def setup_class(self):
        # define an mlprogram, which has weights
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 5000))])
        def linear_prog(input):
            W = mb.const(val=np.random.rand(100, 5000), name="const_W")
            out = mb.linear(x=input, weight=W, name="output")
            return out

        # define another mlprogram, which does not have weights
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 5, 2))])
        def relu_prog(input):
            out = mb.relu(x=input, name="output")
            return out

        # convert and save model on disk
        self.mlmodel = coremltools.convert(linear_prog, convert_to="mlprogram")
        self.mlpackage_path = tempfile.mkdtemp(suffix=utils._MLPACKAGE_EXTENSION)
        self.mlmodel.save(self.mlpackage_path)
        self.mlmodel_no_weights = coremltools.convert(relu_prog, convert_to="mlprogram")

    def teardown_class(self):
        _remove_path(self.mlpackage_path)
        self.mlmodel = None
        self.mlmodel_no_weights = None

    def _test_mlmodel_correctness(self, mlmodel):
        """
        :param mlmodel: coremltools.models.MLModel
        Test the following:
        - calling .predict on mlmodel works correctly
        - calling .save on mlmodel works correctly
        """
        # construct input dictionary
        spec = mlmodel.get_spec()
        inputs = spec.description.input
        input_dict = {}
        for input in inputs:
            input_dict[input.name] = np.random.rand(*tuple(input.type.multiArrayType.shape))
        # check prediction
        preds = mlmodel.predict(input_dict)
        assert preds is not None
        # save, load and predict again to check that the saving and loading worked correctly
        with tempfile.TemporaryDirectory(suffix=utils._MLPACKAGE_EXTENSION) as temp_path:
            mlmodel.save(temp_path)
            mlmodel_reloaded = MLModel(temp_path)
            preds = mlmodel_reloaded.predict(input_dict)
            assert preds is not None

    @pytest.mark.skipif(utils._macos_version() < (12, 0), reason="prediction on mlprogram model "
                                                                    "available only on macOS12+")
    def test_mlmodel_to_spec_to_mlmodel(self):
        """
        convert mlmodel to spec, and then back to mlmodel and verify that it works
        """
        spec = self.mlmodel.get_spec()
        # reload the model from the spec and verify it
        weights_dir = self.mlmodel.weights_dir
        mlmodel_from_spec = MLModel(spec, weights_dir=weights_dir)
        self._test_mlmodel_correctness(mlmodel_from_spec)
        # check that the original model still works
        self._test_mlmodel_correctness(self.mlmodel)
        # check that an error is raised when MLModel is initialized without the weights
        with pytest.raises(Exception, match="MLModel of type mlProgram cannot be loaded just from the model "
                                             "spec object. It also needs the path to the weights file. "
                                             "Please provide that as well, using the 'weights_dir' argument."):
            MLModel(spec)

    @pytest.mark.skipif(utils._macos_version() < (12, 0), reason="prediction on mlprogram model "
                                                                    "available only on macOS12+")
    def test_path_to_mlmodel_to_spec_to_mlmodel(self):
        """
        load an mlmodel from disk, convert it to spec, and then convert the spec back to mlmodel
        """
        mlmodel_from_disk = MLModel(self.mlpackage_path)
        spec = mlmodel_from_disk.get_spec()
        mlmodel_from_spec = MLModel(spec, weights_dir=mlmodel_from_disk.weights_dir)
        self._test_mlmodel_correctness(mlmodel_from_spec)

    @pytest.mark.skipif(utils._macos_version() < (12, 0), reason="prediction on mlprogram model "
                                                                    "available only on macOS12+")
    def test_path_to_spec_to_mlmodel(self):
        """
        load a spec from disk, then convert it to mlmodel, and check that it works
        """
        spec = utils.load_spec(self.mlpackage_path)
        weights_dir = self.mlpackage_path + "/Data/" + _MLPACKAGE_AUTHOR_NAME + "/weights"
        mlmodel = MLModel(spec, weights_dir=weights_dir)
        self._test_mlmodel_correctness(mlmodel)

    @pytest.mark.skipif(utils._macos_version() < (12, 0), reason="prediction on mlprogram model "
                                                                    "available only on macOS12+")
    def test_save_spec_api(self):
        """
        save an mlpackage using the save_spec API. Reload the model from disk and verify it works
        """
        # get spec and use it to save .mlpackage
        spec = self.mlmodel.get_spec()
        with tempfile.TemporaryDirectory(suffix=utils._MLPACKAGE_EXTENSION) as model_path:
            # this should raise error:
            with pytest.raises(Exception, match="spec of type mlProgram cannot be saved without"
                                                " the weights file. Please provide the path to "
                                                "the weights file as well, using the 'weights_dir' argument."):
                utils.save_spec(spec, model_path)

            # provide weights dir path to save the spec correctly
            utils.save_spec(spec, model_path, weights_dir=self.mlmodel.weights_dir)
            # check the correctness of .mlpackage
            model = MLModel(model_path)
            self._test_mlmodel_correctness(model)

    @pytest.mark.skipif(utils._macos_version() < (12, 0), reason="prediction on mlprogram model "
                                                                    "available only on macOS12+")
    def test_save_spec_api_model_with_no_weights(self):
        """
        save an mlprogram model with no weights, using the save SPI and an empty weights directory.
        Reload the model from disk and verify it works
        """
        spec = self.mlmodel_no_weights.get_spec()
        with tempfile.TemporaryDirectory(suffix=utils._MLPACKAGE_EXTENSION) as model_path:
            with tempfile.TemporaryDirectory() as empty_weight_dir:
                utils.save_spec(spec, model_path, weights_dir=empty_weight_dir)
                model = MLModel(model_path)
                self._test_mlmodel_correctness(model)

    @pytest.mark.skipif(utils._macos_version() < (12, 0), reason="prediction on mlprogram model "
                                                                    "available only on macOS12+")
    def test_mlmodel_to_spec_to_mlmodel_with_no_weights_model(self):
        """
        convert mlmodel to spec, and then back to mlmodel and verify that it works
        """
        spec = self.mlmodel_no_weights.get_spec()
        # if no weights_dir is passed, error will be raised
        with pytest.raises(Exception, match="MLModel of type mlProgram cannot be loaded just from the model "
                                             "spec object. It also needs the path to the weights file. "
                                             "Please provide that as well, using the 'weights_dir' argument."):
            MLModel(spec)

        # weights_dir will still exist, even though the model has no weights,
        # with a weights file that only has header and no data
        weights_dir = self.mlmodel_no_weights.weights_dir
        assert weights_dir is not None
        mlmodel_from_spec = MLModel(spec, weights_dir=weights_dir)
        self._test_mlmodel_correctness(mlmodel_from_spec)

        # load mlmodel from spec using an empty weights_dir
        with tempfile.TemporaryDirectory() as empty_weight_dir:
            mlmodel_from_spec = MLModel(spec, weights_dir=weights_dir)
            self._test_mlmodel_correctness(mlmodel_from_spec)

    def test_weights_path_correctness(self):
        """
        test that after reloading an mlmodel from the spec, the weights path is updated
        """
        spec = self.mlmodel.get_spec()
        original_weight_dir_path = self.mlmodel.weights_dir
        assert os.path.exists(original_weight_dir_path)
        # load mlmodel from spec: this will create a new mlpackage in a temp location
        # and copy over the weights
        mlmodel_reloaded = MLModel(spec, weights_dir=original_weight_dir_path)
        assert os.path.exists(mlmodel_reloaded.weights_dir)
        assert mlmodel_reloaded.weights_dir != original_weight_dir_path
        assert mlmodel_reloaded.weights_dir == mlmodel_reloaded.package_path + "/Data/" \
                                                + _MLPACKAGE_AUTHOR_NAME + "/weights"

    def test_weights_dir_discovery_method(self):
        """
        Test "coremltools.libmodelpackage.ModelPackage.findItemByNameAuthor" function
        """
        mlpackage = ModelPackage(self.mlpackage_path)
        model_package_item_info = mlpackage.findItemByNameAuthor(_WEIGHTS_DIR_NAME, _MLPACKAGE_AUTHOR_NAME)
        weights_dir_path = model_package_item_info.path()
        assert weights_dir_path == self.mlpackage_path + "/Data/" + _MLPACKAGE_AUTHOR_NAME + "/weights"
        # verify that findItemByNameAuthor returns None, when item not found
        model_package_item_info = mlpackage.findItemByNameAuthor(_WEIGHTS_DIR_NAME, "inexistent_author_name")
        assert model_package_item_info is None



