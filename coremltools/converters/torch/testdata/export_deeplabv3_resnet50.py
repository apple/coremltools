import torch
import torch.nn as nn
import torchvision

class WrappedDeeplabv3Resnet50(nn.Module):

    input_shape = (1, 3, 512, 512)

    def __init__(self):
        super(WrappedDeeplabv3Resnet50, self).__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50().eval()

    def forward(self, x):
        res = self.model(x)
        x = res["out"]
        return x


if __name__ == "__main__":
    net = WrappedDeeplabv3Resnet50().eval()
    test_data = torch.ones(net.input_shape)
    trace = torch.jit.trace(net, test_data)
    trace.save("wrapped_deeplabv3_resnet50.pt")
    print("--FINISHED--")
