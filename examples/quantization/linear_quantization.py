# -*- coding: utf-8 -*-
"""
.. _linear_quantization_tutorial:

Linear Quantization
===================

"""

########################################################################
# In this tutorial, you learn how to train a simple convolutional neural network on
# `MNIST <http://yann.lecun.com/exdb/mnist/>`_ using :py:class:`~.quantization.LinearQuantizer`.
#
# Learn more about other quantization in the coremltools 
# `Training-Time Quantization Documentation <https://coremltools.readme.io/v7.0/docs/data-dependent-quantization>`_.
#

########################################################################
# Network and Dataset Definition
# ------------------------------
# First define your network, which consists of a single convolution layer
# followed by a dense (linear) layer.

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mnist_net(num_classes=10):
    return nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv2d(1, 12, 3, padding=1)),
                ("relu", nn.ReLU()),
                ("pool", nn.MaxPool2d(2, stride=2, padding=0)),
                ("flatten", nn.Flatten()),
                ("dense", nn.Linear(2352, num_classes)),
                ("softmax", nn.LogSoftmax()),
            ]
        )
    )


########################################################################
# Use the `MNIST dataset provided by PyTorch <https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#mnist>`_
# for training. Apply a very simple transformation to the input
# images to normalize them.

import os

from torchvision import datasets, transforms


def mnist_dataset(data_dir="~/.mnist_qat_data"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    data_path = os.path.expanduser(f"{data_dir}/mnist")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_path, train=False, transform=transform)
    return train, test


########################################################################
# Next, initialize the model and the dataset.

model = mnist_net()

batch_size = 128
train_dataset, test_dataset = mnist_dataset()
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

########################################################################
# Training the Model Without Quantization
# ---------------------------------------
# Train the model without any quantization applied.

optimizer = torch.optim.Adam(model.parameters(), eps=1e-07)
accuracy_unquantized = 0.0
num_epochs = 4


def train_step(model, optimizer, train_loader, data, target, batch_idx, epoch):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 100 == 0:
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )


def eval_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {:.1f}%\n".format(
                test_loss, accuracy
            )
        )
    return accuracy


for epoch in range(num_epochs):
    # train one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        train_step(model, optimizer, train_loader, data, target, batch_idx, epoch)

    # evaluate
    accuracy_unquantized = eval_model(model, test_loader)


print("Accuracy of unquantized network: {:.1f}%\n".format(accuracy_unquantized))

########################################################################
# Insert Quantization Layers in the Model
# ---------------------------------------
# Install :py:class:`~.quantization.LinearQuantizer` in the trained model.
#
# Create an instance of the :py:class:`~.quantization.LinearQuantizerConfig` class
# to specify quantization parameters. ``milestones=[0, 1, 2, 1]`` refers to the following:
#
# * *Index 0*: At 0th epoch, observers will start collecting statistics of values of tensors being quantized
# * *Index 1*: At 1st epoch, quantization simulation will begin
# * *Index 2*: At 2nd epoch, observers will stop collecting and quantization parameters will be frozen
# * *Index 3*: At 1st epoch, batch normalization layers will stop collecting mean and variance, and will start running in inference mode


from coremltools.optimize.torch.quantization import (
    LinearQuantizer,
    LinearQuantizerConfig,
    ModuleLinearQuantizerConfig,
)

global_config = ModuleLinearQuantizerConfig(milestones=[0, 1, 2, 1])
config = LinearQuantizerConfig(global_config=global_config)

quantizer = LinearQuantizer(model, config)

########################################################################
# Next, call :py:meth:`~.quantization.LinearQuantizer.prepare` to insert fake quantization
# layers in the model.

qmodel = quantizer.prepare(example_inputs=torch.randn(1, 1, 28, 28))

########################################################################
# Fine-Tuning the Model
# ---------------------
# The next step is to fine tune the model with quantization applied.
# Call :py:meth:`~.quantization.LinearQuantizer.step` to step through the
# quantization milestones.

optimizer = torch.optim.Adam(qmodel.parameters(), eps=1e-07)
accuracy_quantized = 0.0
num_epochs = 4

for epoch in range(num_epochs):
    # train one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        quantizer.step()
        train_step(qmodel, optimizer, train_loader, data, target, batch_idx, epoch)

    # evaluate
    accuracy_quantized = eval_model(qmodel, test_loader)

########################################################################
# The evaluation shows that you can train a quantized network without a significant loss
# in model accuracy. In practice, for more complex models,
# quantization can be lossy and lead to degradation in validation accuracy.
# In such cases, you can choose to not quantize certain layers which are
# less amenable to quantization.


print("Accuracy of quantized network: {:.1f}%\n".format(accuracy_quantized))
print("Accuracy of unquantized network: {:.1f}%\n".format(accuracy_unquantized))

np.testing.assert_allclose(accuracy_quantized, accuracy_unquantized, atol=2)

########################################################################
# Finalizing the Model for Export
# -------------------------------
#
# The example shows that you can quantize the model with a few code changes to your
# existing PyTorch training code. Now you can deploy this model on a device.
#
# To finalize the model for export, call :py:meth:`~.pruning.LinearQuantizer.finalize`
# on the quantizer. This folds the quantization parameters like scale and zero point
# into the weights.

qmodel.eval()
quantized_model = quantizer.finalize()

########################################################################
# Exporting the Model for On-Device Execution
# -------------------------------------------
#
# In order to deploy the model, convert it to a Core ML model.
#
# Follow the same steps in Core ML Tools for exporting a regular PyTorch model
# (for details, see `Converting from PyTorch <https://coremltools.readme.io/docs/pytorch-conversion>`_).
# The parameter ``ct.target.iOS17`` is necessary here because activation quantization
# ops are only supported on iOS versions >= 17.

import coremltools as ct

example_input = torch.rand(1, 1, 28, 28)
traced_model = torch.jit.trace(quantized_model, example_input)

coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    minimum_deployment_target=ct.target.iOS17,
)

coreml_model.save("~/.mnist_qat_data/quantized_model.mlpackage")
