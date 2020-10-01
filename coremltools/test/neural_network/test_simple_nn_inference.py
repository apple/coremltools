import coremltools
import coremltools.models.datatypes as datatypes
from coremltools.models import neural_network as neural_network
import numpy as np
import os
import pytest

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
            model = coremltools.models.MLModel(model_path)
            out = model.predict(input, useCPUOnly=True)
        except RuntimeError as e:
            print(e)
            assert str(e) == "Error compiling model: \"The file couldn’t be saved.\"."
        else:
            assert out['output'].shape == (1, 3, 3)
            np.testing.assert_allclose(expected, out['output'])
            print("Core ML output", out)

