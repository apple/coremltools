# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from shutil import rmtree
from tempfile import TemporaryDirectory

from unittest import TestCase

from coremltools import ComputeUnit
from coremltools.models import CompiledMLModel, MLModel
from coremltools.models.utils import compile_model, save_spec
from coremltools.proto import Model_pb2


class TestCompiledModel(TestCase):

    @classmethod
    def setUpClass(self):
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


    def _test_compile_model_input(self, model_input, compute_units=ComputeUnit.ALL):
        # Compile model
        compiled_model_path = compile_model(model_input)

        try:
            # Load compiled model
            model = CompiledMLModel(compiled_model_path, compute_units)

            # Single prediction
            y = model.predict({'x': 2})
            self.assertEquals(y['y'], 4.1)

            # Batch predictions
            y = model.predict([{'x': 2}, {'x': 4}])
            self.assertEquals(y, [{'y': 4.1}, {'y': 8.1}])
        finally:
            rmtree(compiled_model_path)


    def test_file_input(self):
        with TemporaryDirectory() as save_dir:
            spec_file_path = save_dir + '/spec.mlmodel'
            save_spec(self.spec, spec_file_path)
            self._test_compile_model_input(spec_file_path)


    def test_spec_input(self):
        self._test_compile_model_input(self.spec)


    def test_mlmodel_input(self):
        ml_model = MLModel(self.spec)
        self._test_compile_model_input(ml_model)


    def test_non_default_compute_units(self):
        non_default_compute_units = (ComputeUnit.CPU_AND_GPU,
                                     ComputeUnit.CPU_AND_NE,
                                     ComputeUnit.CPU_ONLY)
        for cur_compute_unit in non_default_compute_units:
            self._test_compile_model_input(self.spec, compute_units=cur_compute_unit)
