import torch.nn as nn
import torch
import torch.nn.functional as F

class Layer(nn.Module):
    def __init__(self, dims):
        super(Layer, self).__init__()
        self.conv = nn.Conv2d(*dims)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return x


class SimpleNet(nn.Module):
    """A simple network to test graph generating and walking.
    """

    input_shape = (1, 3, 256, 256)

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = Layer((3, 6, 5))
        self.conv2 = Layer((6, 16, 1))
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 63 * 63, 120)
        self.fc2 = nn.Linear(120, 10)

    def input(self):
        return torch.rand(*self.input_shape)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    net = SimpleNet().eval()
    test_data = torch.ones(net.input_shape)
    trace = torch.jit.trace(net, test_data)
    trace.save("SimpleNet.pt")
    print("--FINISHED--")
