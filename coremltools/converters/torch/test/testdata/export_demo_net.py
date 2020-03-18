import torch.nn as nn
import torch
import torch.nn.functional as F
import collections


class Relu(nn.Module):
    def forward(self, x):
        return F.relu(x)


class Flatten(nn.Module):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, x):

        x = torch.flatten(x, self.start, self.end)
        return x


class DemoNet(nn.Module):
    """A simple network for a demo"""

    # Input shape of Mnist images
    input_shape = (1, 1, 28, 28)

    def __init__(self):
        super().__init__()
        modules = []
        modules.append(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4))
        modules.append(Relu())

        modules.append(nn.MaxPool2d(kernel_size=5))

        modules.append(nn.Conv2d(in_channels=64, out_channels=10, kernel_size=5))
        modules.append(Relu())

        modules.append(Flatten(1, 3))

        # ------
        self.seq = nn.Sequential(*modules)

    def input(self):
        return torch.rand(*self.input_shape)

    def forward(self, x):
        for module in self.seq:
            x = module(x)
        return x


if __name__ == "__main__":
    net = DemoNet()
    state_dict = torch.load(
        "saved_models/demo_net_weights.reloadthis", map_location=torch.device("cpu")
    )["network"]
    new_state_dict = collections.OrderedDict()
    # remove 'model.' from keys
    for key in state_dict.keys():
        new_state_dict[key.split(".", 1)[1]] = state_dict[key]
    net.load_state_dict(new_state_dict)
    test_data = torch.ones(net.input_shape)
    trace = torch.jit.trace(net, test_data)
    trace.save("DemoNet_reloaded.pt")
    print("---EXPORTED---")
