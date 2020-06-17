import unittest
import coremltools
import coremltools.models.datatypes as datatypes
from coremltools.models import neural_network as neural_network
import numpy as np
import copy
from coremltools._deps import _HAS_TF, MSG_TF1_NOT_FOUND
if _HAS_TF:
    import tensorflow as tf
    from coremltools.converters.tensorflow.legacy.tfcoreml import optimize_nn_spec

@unittest.skipIf(not _HAS_TF, MSG_TF1_NOT_FOUND)
class CorrectnessTest(unittest.TestCase):
    def _compare_outputs(self, output, output_ref, delta=0.05):
        x = output.flatten()
        y = output_ref.flatten()
        den = np.maximum(x, np.maximum(y, np.ones(len(x))))
        x = x / den
        y = y / den
        max_error = np.amax(np.abs(x - y))
        self.assertGreater(delta, max_error)

@unittest.skipIf(not _HAS_TF, MSG_TF1_NOT_FOUND)
class OptimizerTests(CorrectnessTest):
    def test_pad_conv_fusion(self):

        Cin = 3
        Cout = 5
        K = 9
        Hin = 32
        Win = 18
        Xin = np.random.rand(Cin, Hin, Win)
        # Test for several combinations of (pad,stride)
        params = [
            (5, 2),
            (4, 3),
            (6, 3),
            (5, 1),
            (5, 2),
            (6, 2),
            (3, 2),
            (1, 1),
            (2, 3),
        ]
        for param in params:
            pad, stride = param
            input_features = [("data", datatypes.Array(*(Cin, Hin, Win)))]
            output_features = [("output", None)]
            builder = neural_network.NeuralNetworkBuilder(
                input_features, output_features
            )
            builder.add_padding(
                name="pad",
                left=pad,
                right=pad,
                top=pad,
                bottom=pad,
                input_name="data",
                output_name="pad_out",
            )
            builder.add_convolution(
                name="conv",
                kernel_channels=Cin,
                output_channels=Cout,
                height=K,
                width=K,
                stride_height=stride,
                stride_width=stride,
                border_mode="valid",
                groups=1,
                W=np.random.rand(K, K, Cin, Cout),
                b=None,
                has_bias=False,
                input_name="pad_out",
                output_name="output",
            )

            # get unoptimized model
            original_spec = builder.spec
            model = coremltools.models.utils._get_model(original_spec)
            # get optimized model
            spec_copy = copy.deepcopy(original_spec)
            optimize_nn_spec(spec_copy)
            model_opt = coremltools.models.utils._get_model(spec_copy)

            n_layers_original_model = len(model.get_spec().neuralNetwork.layers)
            n_layers_opt_model = len(model_opt.get_spec().neuralNetwork.layers)
            self.assertEqual(n_layers_original_model, 2)
            self.assertEqual(n_layers_opt_model, 1)

            original_model_out = model.predict({"data": Xin})["output"]
            opt_model_out = model_opt.predict({"data": Xin})["output"]
            self._compare_outputs(opt_model_out, original_model_out)
