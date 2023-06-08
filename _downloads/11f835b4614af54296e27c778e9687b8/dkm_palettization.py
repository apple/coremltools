# -*- coding: utf-8 -*-
"""
.. _palettization_tutorial:

Palettization Using Differentiable K-Means
==========================================

"""

########################################################################
# In this tutorial, you learn how to palettize a
# network trained on `MNIST <http://yann.lecun.com/exdb/mnist/>`_ using
# :py:class:`~.palettizer.DKMPalettizer`.
#
# Learn more about other palettization in the coremltools `Training-Time Palettization Documentation <https://coremltools.readme.io/v7.0/docs/training-time-palettization>`_.


########################################################################
# Defining the Network and Dataset
# --------------------------------
#
# First, define your network:

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def mnist_net(num_classes=10):
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 32, 5, padding='same')),
        ('relu1', nn.ReLU()),
        ('pool1', nn.MaxPool2d(2, stride=2, padding=0)),
        ('bn1', nn.BatchNorm2d(32, eps=0.001, momentum=0.01)),
        ('conv2', nn.Conv2d(32, 64, 5, padding='same')),
        ('relu2', nn.ReLU()),
        ('pool2', nn.MaxPool2d(2, stride=2, padding=0)),
        ('flatten', nn.Flatten()),
        ('dense1', nn.Linear(3136, 1024)),
        ('relu3', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.4)),
        ('dense2', nn.Linear(1024, num_classes)),
        ('softmax', nn.LogSoftmax())]))


########################################################################
# For training, use the MNIST dataset provided by
# `PyTorch <https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#mnist>`_.
# Apply a very simple transformation to the input images to normalize them.

import os

from filelock import FileLock
from torchvision import datasets, transforms


def mnist_dataset(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_path = os.path.expanduser(data_dir)
    os.makedirs(data_path, exist_ok=True)
    with FileLock(os.path.join(data_path, 'data.lock')):
        train = datasets.MNIST(data_path, train=True, download=True,
                               transform=transform)
        test = datasets.MNIST(data_path, train=False, download=True,
                              transform=transform)
    return train, test


########################################################################
# Initialize the model and the dataset.

model = mnist_net()

batch_size = 128
train_dataset, test_dataset = mnist_dataset("~/.mnist_data/")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

########################################################################
# Training the Model Without Palettization
# ----------------------------------------
#
# Train the model without applying any palettization.

optimizer = torch.optim.SGD(model.parameters(), lr=0.008)
accuracy_unpalettized = 0.0
num_epochs = 2


def train_step(model, optimizer, train_loader, data, target, batch_idx, epoch, palettizer = None):
    optimizer.zero_grad()
    if palettizer is not None:
        palettizer.step()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 100 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))


def eval_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
            test_loss, accuracy))
    return accuracy


for epoch in range(num_epochs):
    # train one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        train_step(model, optimizer, train_loader, data, target, batch_idx, epoch)

    # evaluate
    accuracy_unpalettized = eval_model(model, test_loader)

print("Accuracy of unpalettized network: {:.0f}%\n".format(accuracy_unpalettized))

########################################################################
# Configuring Palettization
# -------------------------
#
# Insert palettization layers into the trained model.
# As a rule of thumb, we recommend choosing layers with more than 10,000 weights to palettize.
# For this example, choose the second convolutional layer, ``conv2``.
# Apply a ``4-bit`` palettization to this layer. This would mean that for all the weights
# that exist in this layer, you try to map each weight element to one of :math:`2^4`,
# that is, ``16`` clusters.
#
# Note that calling :py:meth:`~.palettization.DKMPalettizer.prepare` simply inserts palettization
# layers into the model. It doesn't actually palettize the weights. You do that in the next step when
# you fine-tune the model.

from coremltools.optimize.torch.palettization import DKMPalettizer, DKMPalettizerConfig

config = DKMPalettizerConfig.from_dict(
    {"module_name_configs": {"conv2": {"n_bits": 4}}}
)
palettizer = DKMPalettizer(model, config)

prepared_model = palettizer.prepare()

########################################################################
# Fine-Tuning the Palettized Model
# --------------------------------
#
# Fine-tune the model with palettization applied. This helps the model learn the new palettized
# layers' weights in the form of a LUT and indices.

optimizer = torch.optim.SGD(prepared_model.parameters(), lr=0.008)
accuracy_palettized = 0.0
num_epochs = 2

for epoch in range(num_epochs):
    prepared_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        train_step(prepared_model, optimizer, train_loader, data, target, batch_idx, epoch, palettizer)

    # evaluate
    accuracy_palettized = eval_model(prepared_model, test_loader)

########################################################################
# The evaluation shows that you can train a palettized network without losing much accuracy
# with the final model.

print("Accuracy of unpalettized network: {:.0f}%\n".format(accuracy_unpalettized))
print("Accuracy of palettized network: {:.0f}%\n".format(accuracy_palettized))

########################################################################
# Restoring LUT and Indices as Weights
# ------------------------------------
#
# Use :py:meth:`~.palettization.Palettizer.finalize` to
# restore the LUT and indices of the palettized modules as weights in the model.

finalized_model = palettizer.finalize()

########################################################################
# Exporting the Model for On-Device Execution
# -------------------------------------------
#
# To deploy the model on device, convert it to the
# `MLPackage <https://developer.apple.com/documentation/coreml/updating_a_model_file_to_a_model_package>`_ format,
# which can then be run with the
# `CoreML <https://developer.apple.com/documentation/coreml>`_ APIs.
#
# To export the model with Core ML Tools, first trace the model with an input, and then
# use the Core ML Tools converter, as described in
# `Converting from PyTorch <https://coremltools.readme.io/docs/pytorch-conversion>`_.
# The parameter ``ct.PassPipeline.DEFAULT_PALETTIZATION`` signals to the
# converter a palettized model is being converted, and allows its weights to be
# represented using a look-up table (LUT) and indices, which have a much smaller
# footprint on disk as compared to the dense weights.

import coremltools as ct

finalized_model.eval()
example_input = torch.rand(1, 1, 28, 28)
traced_model = torch.jit.trace(finalized_model, example_input)


coreml_model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)],
    pass_pipeline=ct.PassPipeline.DEFAULT_PALETTIZATION,
    minimum_deployment_target=ct.target.iOS16,
)
