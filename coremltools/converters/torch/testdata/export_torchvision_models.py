import argparse
from collections import namedtuple
from datetime import datetime
import os
import sys

import torch
import torch.nn as nn
import torchvision

from exporting_utils import (
    ModelParams,
    exporter_parser,
    trace_and_save_model,
)


"""
Certain torchvision models don't conform to the input/output requirements of
torch.jit.trace. Here we define classes that wrap them to manipulate their
input/output as necessary.
"""


class WrappedDeeplabv3Resnet50(nn.Module):

    input_shape = (1, 3, 512, 512)

    def __init__(self):
        super(WrappedDeeplabv3Resnet50, self).__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50().eval()

    def forward(self, x):
        res = self.model(x)
        x = res["out"]
        return x


"""
List of torchvision models that are available for export.
"""
MODELS = {
    "alexnet": ModelParams((1, 3, 256, 256), torchvision.models.alexnet),
    "wrapped_deeplabv3_resnet50": ModelParams(
        (1, 3, 512, 512), WrappedDeeplabv3Resnet50
    ),
    "densenet161": ModelParams((1, 3, 256, 256), torchvision.models.densenet161),
    "googlenet": ModelParams((1, 3, 256, 256), torchvision.models.googlenet),
    "inception_v3": ModelParams((1, 3, 256, 256), torchvision.models.inception_v3),
    "mnasnet1_0": ModelParams((1, 3, 256, 256), torchvision.models.mnasnet1_0),
    "mobilenet_v2": ModelParams((1, 3, 256, 256), torchvision.models.mobilenet_v2),
    "resnet18": ModelParams((1, 3, 256, 256), torchvision.models.resnet18),
    "shufflenet_v2_x1_0": ModelParams(
        (1, 3, 256, 256), torchvision.models.shufflenet_v2_x1_0
    ),
    "squeezenet1_0": ModelParams((1, 3, 256, 256), torchvision.models.squeezenet1_0),
    "vgg16": ModelParams((1, 3, 256, 256), torchvision.models.vgg16),
    "vgg19_bn": ModelParams((1, 3, 256, 256), torchvision.models.vgg19_bn),
}


if __name__ == "__main__":
    parser = exporter_parser(
        description="Export torchvision end-to-end conversion test models.",
        model_names=list(MODELS.keys()),
    )
    args = parser.parse_args(sys.argv[1:])

    datestamp = datetime.now().strftime("%m-%d-%Y")

    for name in MODELS.keys():
        if args.all or getattr(args, name):
            trace_and_save_model(
                "torchvision_{}".format(name), MODELS[name], args.path, datestamp
            )

    print("--FINISHED--")
