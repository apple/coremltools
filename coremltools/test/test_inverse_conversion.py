import torch
import torch.nn as nn
import coremltools as ct

def test_inverse_with_int32_shape_input():
    class Model(nn.Module):
        def forward(self, x):
            return 16 / x.shape[0]  # int32 division

    model = Model()
    model.eval()  # Make sure to silence the eval warning
    inputs = (torch.randn(2, 3, 4),)
    traced = torch.jit.trace(model, inputs)

    try:
        ct.convert(traced, inputs=[ct.TensorType(shape=(2, 3, 4))], convert_to="mlprogram")
    except ValueError as e:
        error_str = str(e).lower()
        assert "inverse" in error_str
        assert any(keyword in error_str for keyword in ["int32", "integer"])
    else:
        assert False, "Expected ValueError due to int32 input to inverse op"
