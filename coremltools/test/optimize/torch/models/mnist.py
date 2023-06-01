#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# type: ignore
import os
from collections import OrderedDict
from coremltools.test.optimize.torch.utils import test_data_path

import pytest
import torch.nn as nn
from filelock import FileLock
from torchvision import datasets, transforms

# IMPORTANT: DO NOT import these fixtures in your tests.
# That leads pytest to run the fixtures (even session-scoped) multiple times.
# These have been imported into conftest.py, which makes them available for all
# tests within the test/ folder.


num_classes = 10


@pytest.fixture
def mnist_model():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 32, (5, 5), padding='same')),
        ('relu1', nn.ReLU()),
        ('pool1', nn.MaxPool2d(2, stride=2, padding=0)),
        ('bn1', nn.BatchNorm2d(32, eps=0.001, momentum=0.01)),
        ('conv2', nn.Conv2d(32, 64, (5, 5), padding='same')),
        ('relu2', nn.ReLU()),
        ('pool2', nn.MaxPool2d(2, stride=2, padding=0)),
        ('flatten', nn.Flatten()),
        ('dense1', nn.Linear(3136, 1024)),
        ('relu3', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.4)),
        ('dense2', nn.Linear(1024, num_classes)),
        ('softmax', nn.LogSoftmax())]))


@pytest.fixture
def mnist_model_quantization():
    # String padding mode like "same" or "valid" is not supported
    # for quantized models: https://github.com/pytorch/pytorch/issues/76304
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 32, (5, 5), padding=2)),
        ('bn1', nn.BatchNorm2d(32, eps=0.001, momentum=0.01)),
        ('relu1', nn.ReLU6()),
        ('pool1', nn.MaxPool2d(2, stride=2, padding=0)),
        ('conv2', nn.Conv2d(32, 64, (5, 5), padding=2)),
        ('relu2', nn.ReLU6()),
        ('pool2', nn.MaxPool2d(2, stride=2, padding=0)),
        ('flatten', nn.Flatten()),
        ('dense1', nn.Linear(3136, 1024)),
        ('relu3', nn.ReLU6()),
        ('dropout', nn.Dropout(p=0.4)),
        ('dense2', nn.Linear(1024, num_classes)),
        ('softmax', nn.LogSoftmax())]))


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
