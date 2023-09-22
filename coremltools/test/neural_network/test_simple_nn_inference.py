# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os

import numpy as np

import coremltools
import coremltools.models.datatypes as datatypes
from coremltools import ComputeUnit, utils
from coremltools.models import neural_network as neural_network


class TestNeuralNetworkPrediction:

    @staticmethod
    def test_lrn_model(tmpdir):

        input_dim = (1, 3, 3)
        input_features = [("data", datatypes.Array(*input_dim))]
        output_features = [("output", datatypes.Array(*input_dim))]

        builder = neural_network.NeuralNetworkBuilder(input_features, output_features)
        builder.add_lrn(
            name="lrn",
            input_name="data",
            output_name="output",
            alpha=2,
            beta=3,
            local_size=1,
            k=8,
        )

        input = {"data": np.ones((1, 3, 3))}
        expected = 1e-3 * np.ones((1, 3, 3))
        model_path = os.path.join(str(tmpdir), "lrn_model.mlmodel")
        coremltools.models.utils.save_spec(builder.spec, model_path)

        try:
            model = coremltools.models.MLModel(model_path, compute_units=ComputeUnit.CPU_ONLY)
            if utils._macos_version() >= (10, 13):
                out = model.predict(input)
        except RuntimeError as e:
            print(e)
            assert str(e) == "Error compiling model: \"The file couldn’t be saved.\"."
        else:
            if utils._macos_version() >= (10, 13):
                assert out['output'].shape == (1, 3, 3)
                np.testing.assert_allclose(expected, out['output'])
                print("Core ML output", out)

