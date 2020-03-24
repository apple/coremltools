import numpy as np
import pytest
import torch

import coremltools
from testdata.export_demo_net import DemoNet
from testdata.export_simple_net import SimpleNet

from test_utils import *


class TestTorchConversion:
    @pytest.fixture
    def set_random_seeds(self):
        torch.manual_seed(1)
        np.random.seed(1)

    def test_simplenet(self):
        model_path = "testdata/SimpleNet.pt"
        convert_and_compare(model_path, torch.rand((1, 3, 256, 256)))

    def test_demonet_from_file(self):
        model_path = "testdata/DemoNet_reloaded.pt"
        convert_and_compare(model_path, torch.rand((1, 1, 28, 28)))

    def test_demonet_from_mem(self):
        model = DemoNet()
        model.eval()
        input_data = torch.rand(model.input_shape)
        torch_model = torch.jit.trace(model, input_data)
        convert_and_compare(torch_model, input_data)
