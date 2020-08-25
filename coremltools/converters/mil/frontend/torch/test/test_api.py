import pytest
import coremltools as ct

from coremltools._deps import (
    _HAS_TORCH,
    MSG_TORCH_NOT_FOUND,
)

@pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
class TestPyTorchConverter:
    @staticmethod
    def test_no_inputs():
        import torch
        import torchvision

        model = torchvision.models.mobilenet_v2()
        model.eval()

        example_input = torch.rand(1, 3, 256, 256)

        traced_model = torch.jit.trace(model, example_input)

        with pytest.raises(ValueError) as e:
            mlmodel = ct.convert(traced_model)
        e.match(r'Expected argument for pytorch "inputs" not provided')
