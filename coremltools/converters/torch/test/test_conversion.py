import unittest

import numpy as np
import pytest
import torch
import torch.nn as nn

import coremltools
from testdata.export_demo_net import DemoNet
from testdata.export_simple_net import SimpleNet


def _convert_to_coreml_inputs(
    input_description, inputs,
):
    """Convenience function to combine a CoreML model's input description and
    set of raw inputs into the format expected by the model's predict function.
    """
    coreml_inputs = {str(x): inp.numpy() for x, inp in zip(input_description, inputs)}
    return coreml_inputs


class TorchConversionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _convert_and_compare(self, model_spec, input_size, places=5):
        inputs = torch.rand(*input_size)
        if isinstance(model_spec, str):
            torch_model = torch.jit.load(model_spec)
        else:
            torch_model = model_spec
        mlmodel = coremltools.converters.torch.convert(model_spec, [inputs])

        coreml_inputs = _convert_to_coreml_inputs(mlmodel.input_description, [inputs])
        coreml_outputs = [str(x) for x in mlmodel.output_description]
        coreml_result = mlmodel.predict(coreml_inputs)
        coreml_result = coreml_result[coreml_outputs[0]]

        torch_result = torch_model(inputs)
        torch_result = torch_result.detach().numpy()

        self.assertEqual(coreml_result.shape, torch_result.shape)
        np.testing.assert_allclose(
            coreml_result, torch_result, atol=10.0 ** -places,
        )

    def test_simplenet(self):
        model_path = "testdata/SimpleNet.pt"
        input_size = (1, 3, 256, 256)
        self._convert_and_compare(model_path, input_size)

    def test_demonet_from_file(self):
        model_path = "testdata/DemoNet_reloaded.pt"
        input_size = (1, 1, 28, 28)
        self._convert_and_compare(model_path, input_size)

    def test_demonet_from_mem(self):
        model = DemoNet()
        model.eval()
        torch_model = torch.jit.trace(model, torch.rand(model.input_shape))
        self._convert_and_compare(torch_model, model.input_shape)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TorchConversionTest())
    unittest.TextTestRunner().run(suite)
