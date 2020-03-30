import torch
import torch.nn as nn
import torch.nn.functional as F


_INPUT_SIZE = (1, 3, 1024, 768)


class _Layer(nn.Module):
    def __init__(self, channels):
        """A simple conv + relu layer."""
        super(_Layer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x


class _LoopModule(nn.Module):
    def __init__(self, channels):
        """Creates a module that runs the same conv repeatedly."""
        super(_LoopModule, self).__init__()
        conv = _Layer(channels)
        # Trace the part of the model that's inside the control flow.
        self.conv = torch.jit.trace(conv, torch.ones(*_INPUT_SIZE))

    def forward(self, x, count: int):
        for _ in range(count):
            x = self.conv(x)
        return x


class _ConditionalModule(nn.Module):
    def __init__(self, module):
        super(_ConditionalModule, self).__init__()
        self.module = module

    def forward(self, x):
        avg = torch.mean(x)
        if avg.item() > 0:
            x = self.module(x, 1)
        else:
            x = self.module(x, 2)
        return x


class ControlFlowNet(nn.Module):

    input_shape = _INPUT_SIZE

    def __init__(self, num_channels=3):
        """This network provides a dummy model to test conversion of loop and
        conditional control flow."""
        super(ControlFlowNet, self).__init__()
        # Create the first loop module and script it.
        looper = _LoopModule(num_channels)
        self.looper = torch.jit.script(looper)
        # Create the second loop module, script it, then create the conditional
        # module and script that.
        looper = _LoopModule(num_channels)
        cond_module = _ConditionalModule(torch.jit.script(looper))
        self.cond_module = torch.jit.script(cond_module)
        self.linear = nn.Linear(num_channels, 2)

    def forward(self, x):
        x = self.looper(x, 2)
        x = self.cond_module(x)
        x = torch.mean(x, [2, 3])
        x = self.linear(x)
        return x


if __name__ == "__main__":
    test_data = torch.ones(_INPUT_SIZE)
    control_flow_net = torch.jit.trace(ControlFlowNet(num_channels=3), test_data,)
    control_flow_net.save("ControlFlowNet.pt")
    print("---EXPORTED---")
