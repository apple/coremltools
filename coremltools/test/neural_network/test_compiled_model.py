# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import itertools
from shutil import copytree, rmtree
from tempfile import TemporaryDirectory

import pytest

from coremltools import ComputeUnit, ReshapeFrequency, SpecializationStrategy, utils
from coremltools.models import CompiledMLModel, MLModel
from coremltools.models.utils import compile_model, load_spec, save_spec
from coremltools.proto import Model_pb2


class TestCompiledModel:

    def setup_class(self):
        spec = Model_pb2.Model()
        spec.specificationVersion = 1
        input_ = spec.description.input.add()
        input_.name = 'x'
        input_.type.doubleType.MergeFromString(b"")

        output_ = spec.description.output.add()
        output_.name = 'y'
        output_.type.doubleType.MergeFromString(b"")

        lr = spec.glmRegressor
        lr.offset.append(0.1)
        weights = lr.weights.add()
        weights.value.append(2.0)

        spec.description.predictedFeatureName = 'y'
        self.spec = spec

        self.compiled_model_path = compile_model(self.spec)


    def teardown_class(self):
        rmtree(self.compiled_model_path)


    def _test_compile_model_path(self, compiled_model_path, compute_units=ComputeUnit.ALL):
        try:
            # Load compiled model
            model = CompiledMLModel(compiled_model_path, compute_units)

            # Single prediction
            y = model.predict({'x': 2})
            assert y['y'] == 4.1

            # Batch predictions
            y = model.predict([{'x': 2}, {'x': 4}])
            assert y == [{'y': 4.1}, {'y': 8.1}]
        finally:
            rmtree(compiled_model_path)


    def test_mlmodel_file_input(self):
        with TemporaryDirectory() as save_dir:
            file_path = save_dir + '/m.mlmodel'
            MLModel(self.spec).save(file_path)

            with pytest.raises(TypeError, match=", first load the model, "):
                compiled_model_path = compile_model(file_path)


    def test_spec_input(self):
        compiled_model_path = compile_model(self.spec)
        self._test_compile_model_path(compiled_model_path)


    def test_mlmodel_input(self):
        ml_model = MLModel(self.spec)
        with pytest.raises(TypeError, match=" model has already been compiled."):
            compiled_model_path = compile_model(ml_model)


    def test_from_existing_mlmodel(self):
        ml_model = MLModel(self.spec)
        compiled_model_path = ml_model.get_compiled_model_path()

        with TemporaryDirectory() as temp_dir:
            dst_path = temp_dir + "/foo.mlmodelc"
            copytree(compiled_model_path, dst_path)
            del ml_model
            self._test_compile_model_path(dst_path)


    def test_non_default_compute_units(self):
        non_default_compute_units = (ComputeUnit.CPU_AND_GPU,
                                     ComputeUnit.CPU_AND_NE,
                                     ComputeUnit.CPU_ONLY)
        for cur_compute_unit in non_default_compute_units:
            compiled_model_path = compile_model(self.spec)
            self._test_compile_model_path(compiled_model_path, compute_units=cur_compute_unit)


    def test_destination_path_parameter(self):
        # Check correct usage
        with TemporaryDirectory() as temp_dir:
            dst_path = temp_dir + "/foo.mlmodelc"
            compiled_model_path = compile_model(self.spec, dst_path)
            self._test_compile_model_path(compiled_model_path)

        # Check bad input
        with TemporaryDirectory() as temp_dir:
            dst_path = temp_dir + "/foo.badFileExtension"
            with pytest.raises(Exception, match=" file extension."):
                compiled_model_path = compile_model(self.spec, dst_path)


    def test_save_load_spec(self):
        with TemporaryDirectory() as save_dir:
            file_path = save_dir + '/spec.mlmodel'
            save_spec(self.spec, file_path)
            my_spec = load_spec(file_path)
            compiled_model_path = compile_model(my_spec)
        self._test_compile_model_path(compiled_model_path)


    @pytest.mark.skipif(utils._macos_version() < (15, 0),
                        reason="optimization hints available only on macOS15+")
    @pytest.mark.parametrize("reshapeFrequency, specializationStrategy",
                             itertools.product(
                                 (ReshapeFrequency.Frequent, ReshapeFrequency.Infrequent, None),
                                 (SpecializationStrategy.FastPrediction, SpecializationStrategy.Default, None),
                             ))
    def test_optimization_hints(self, reshapeFrequency, specializationStrategy):
        optimization_hints={}
        if reshapeFrequency is not None:
            optimization_hints['reshapeFrequency'] = reshapeFrequency
        if specializationStrategy is not None:
            optimization_hints["specializationStrategy"] = specializationStrategy
        if len(optimization_hints) == 0:
            optimization_hints = None

        m = CompiledMLModel(self.compiled_model_path, optimization_hints=optimization_hints)
        assert isinstance(m, CompiledMLModel)
        assert(m.optimization_hints == optimization_hints)
