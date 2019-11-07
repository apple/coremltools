import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx_coreml import convert

# Step 0 - (a) Define ML Model
class small_model(nn.Module):
    def __init__(self):
        super(small_model, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.softmax(self.fc2(y))
        return y

# Step 0 - (b) Create model or Load from dist
model = small_model()
dummy_input = torch.randn(768)

# Step 1 - PyTorch to ONNX model
torch.onnx.export(model, dummy_input, './small_model.onnx')

# Step 2 - ONNX to CoreML model
mlmodel = convert(model='./small_model.onnx', target_ios='13')
# Save converted CoreML model
mlmodel.save('small_model.mlmodel')
