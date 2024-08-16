#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# type: ignore
import os
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
from filelock import FileLock
from torchvision import datasets, transforms

from coremltools.test.optimize.torch.utils import test_data_path

# IMPORTANT: DO NOT import these fixtures in your tests.
# That leads pytest to run the fixtures (even session-scoped) multiple times.
# These have been imported into conftest.py, which makes them available for all
# tests within the test/ folder.


num_classes = 10


@pytest.fixture()
def mnist_example_input():
    return torch.rand(1, 1, 28, 28)


@pytest.fixture()
def mnist_example_output():
    return torch.rand(1, num_classes)


@pytest.fixture
def mnist_model():
    return nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 32, (5, 5), padding=2)),
                ("relu1", nn.ReLU()),
                ("pool1", nn.MaxPool2d(2, stride=2, padding=0)),
                ("bn1", nn.BatchNorm2d(32, eps=0.001, momentum=0.01)),
                ("conv2", nn.Conv2d(32, 64, (5, 5), padding=2)),
                ("relu2", nn.ReLU()),
                ("pool2", nn.MaxPool2d(2, stride=2, padding=0)),
                ("flatten", nn.Flatten()),
                ("dense1", nn.Linear(3136, 1024)),
                ("relu3", nn.ReLU()),
                ("dropout", nn.Dropout(p=0.4)),
                ("dense2", nn.Linear(1024, num_classes)),
                ("softmax", nn.LogSoftmax()),
            ]
        )
    )


@pytest.fixture
def mnist_model_conv_transpose():
    # this method will be removed once conv_transpose is integrated for pruning and palettization
    return nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 32, (5, 5), padding=2)),
                ("relu1", nn.ReLU()),
                ("pool1", nn.MaxPool2d(2, stride=2, padding=0)),
                ("bn1", nn.BatchNorm2d(32, eps=0.001, momentum=0.01)),
                ("conv2", nn.Conv2d(32, 64, (5, 5), padding=2)),
                ("relu2", nn.ReLU()),
                ("pool2", nn.MaxPool2d(2, stride=2, padding=0)),
                (
                    "conv_transpose1",
                    nn.ConvTranspose2d(64, 32, stride=1, kernel_size=3, padding=1),
                ),
                ("conv4", nn.Conv2d(32, 64, stride=1, kernel_size=1, padding=0)),
                ("flatten", nn.Flatten()),
                ("dense1", nn.Linear(3136, 1024)),
                ("relu3", nn.ReLU()),
                ("dropout", nn.Dropout(p=0.4)),
                ("dense2", nn.Linear(1024, 10)),
                ("softmax", nn.LogSoftmax()),
            ]
        )
    )


@pytest.fixture
def mnist_model_quantization():
    # String padding mode like "same" or "valid" is not supported
    # for quantized models: https://github.com/pytorch/pytorch/issues/76304
    return nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 32, (5, 5), padding=2)),
                ("bn1", nn.BatchNorm2d(32, eps=0.001, momentum=0.01)),
                ("relu1", nn.ReLU6()),
                ("pool1", nn.MaxPool2d(2, stride=2, padding=0)),
                ("conv2", nn.Conv2d(32, 64, (5, 5), padding=2)),
                ("relu2", nn.ReLU6()),
                ("pool2", nn.MaxPool2d(2, stride=2, padding=0)),
                ("conv_transpose1", nn.ConvTranspose2d(64, 128, 3, padding=1)),
                ("bn3", nn.BatchNorm2d(128, eps=0.001, momentum=0.01)),
                ("relu3", nn.ReLU6()),
                ("pool3", nn.MaxPool2d(1, stride=1, padding=0)),
                ("conv_transpose2", nn.ConvTranspose2d(128, 64, 3, padding=1)),
                ("relu4", nn.GELU()),
                ("flatten", nn.Flatten()),
                ("dense1", nn.Linear(3136, 1024)),
                ("relu5", nn.ReLU6()),
                ("dropout", nn.Dropout(p=0.4)),
                ("dense2", nn.Linear(1024, num_classes)),
                ("softmax", nn.LogSoftmax()),
            ]
        )
    )


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


@pytest.fixture
def residual_mnist_model():
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "conv1",
                    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                ),
                ("bn1", nn.BatchNorm2d(64)),
                ("relu1", nn.ReLU()),
                ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                (
                    "add1",
                    Residual(
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "conv2",
                                        nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
                                    ),
                                    ("bn2", nn.BatchNorm2d(64)),
                                    ("relu2", nn.ReLU()),
                                    (
                                        "conv3",
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                    ),
                                    ("bn3", nn.BatchNorm2d(64)),
                                ]
                            )
                        )
                    ),
                ),
                ("relu3", nn.ReLU()),
                ("flatten", nn.Flatten()),
                ("dense1", nn.Linear(3136, 1024)),
                ("relu4", nn.ReLU()),
                ("dropout", nn.Dropout(p=0.4)),
                ("dense2", nn.Linear(1024, num_classes)),
                ("softmax", nn.LogSoftmax()),
            ]
        )
    )


@pytest.fixture
def mnist_model_large():
    """
    MNIST model with redundant layers for testing pruning algorithm
    """
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 32, (5, 5), padding='same')),
        ('relu1', nn.ReLU()),
        ('pool1', nn.MaxPool2d(2, stride=2, padding=0)),
        ('bn1', nn.BatchNorm2d(32, eps=0.001, momentum=0.01)),
        ('conv2', nn.Conv2d(32, 64, (5, 5), padding='same')),
        ('relu2', nn.ReLU()),
        ('pool2', nn.MaxPool2d(2, stride=2, padding=0)),
        ('conv3', nn.Conv2d(64, 64, (5, 5), padding='same')),
        ('relu3', nn.ReLU()),
        ('conv4', nn.Conv2d(64, 64, (5, 5), padding='same')),
        ('relu4', nn.ReLU()),
        ('flatten', nn.Flatten()),
        ('dense1', nn.Linear(3136, 1024)),
        ('relu5', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.4)),
        ('dense2', nn.Linear(1024, num_classes)),
        ('softmax', nn.LogSoftmax())]))


def LeNet5():
    """
    Original LeNet5 model for MNIST with sigmoid activations.
    """
    return nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 6, 5, 1, 2)),
                ("sigmoid1", nn.Sigmoid()),
                ("pool1", nn.AvgPool2d(2, 2)),
                ("conv2", nn.Conv2d(6, 16, 5, 1, 0)),
                ("sigmoid2", nn.Sigmoid()),
                ("pool2", nn.AvgPool2d(2, 2)),
                ("flatten", nn.Flatten()),
                ("dense1", nn.Linear(5 * 5 * 16, 120)),
                ("sigmoid3", nn.Sigmoid()),
                ("dense2", nn.Linear(120, 84)),
                ("sigmoid4", nn.Sigmoid()),
                ("dense3", nn.Linear(84, num_classes)),
                ("softmax", nn.LogSoftmax(dim=1)),
            ]
        )
    )


@pytest.fixture(scope="session")
def mnist_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_path = os.path.join(test_data_path(), 'mnist')
    os.makedirs(data_path, exist_ok=True)
    with FileLock(os.path.join(data_path, 'data.lock')):
        train = datasets.MNIST(data_path, train=True, download=True,
                               transform=transform)
        test = datasets.MNIST(data_path, train=False, download=True,
                              transform=transform)
    return train, test
