# -*- coding: utf-8 -*-
"""
.. _magnitude_pruning_tutorial:

Magnitude Pruning
=================

"""

########################################################################
# In this tutorial, you learn how to train a simple convolutional neural network on
# `MNIST <http://yann.lecun.com/exdb/mnist/>`_ using :py:class:`~.pruning.MagnitudePruner`.
#
# Learn more about other pruners and schedulers in the coremltools 
# `Training-Time Pruning Documentation <https://coremltools.readme.io/v7.0/docs/data-dependent-pruning>`_.
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
            [('conv', nn.Conv2d(1, 12, 3, padding='same')),
             ('relu', nn.ReLU()),
             ('pool', nn.MaxPool2d(2, stride=2, padding=0)),
             ('flatten', nn.Flatten()),
             ('dense', nn.Linear(2352, num_classes)),
             ('softmax', nn.LogSoftmax())]
        )
    )


########################################################################
# Use the `MNIST dataset provided by PyTorch <https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#mnist>`_
# for training. Apply a very simple transformation to the input
# images to normalize them.

import os

from torchvision import datasets, transforms


def mnist_dataset(data_dir="~/.mnist_pruning_data"):
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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

########################################################################
# Training the Model Without Pruning
# ----------------------------------
# Train the model without any pruning applied.

optimizer = torch.optim.Adam(model.parameters(), eps=1e-07)
accuracy_unpruned = 0.0
num_epochs = 4


def train_step(model, optimizer, train_loader, data, target, batch_idx, epoch):
    optimizer.zero_grad()
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
    accuracy_unpruned = eval_model(model, test_loader)


print("Accuracy of unpruned network: {:.1f}%\n".format(accuracy_unpruned))

########################################################################
# Installing the Pruner in the Model
# ----------------------------------
# Install :py:class:`~.pruning.MagnitudePruner` in the trained model.
#
# First, construct a :py:class:`~.pruning.pruning_scheduler.PruningScheduler` class,
# which specifies how the sparsity of your pruned layers should evolve over the course of the training.
# For this tutorial, use a :py:class:`~.pruning.PolynomialDecayScheduler`,
# which is introduced in the paper `"To prune or not to prune" <https://arxiv.org/pdf/1710.01878.pdf>`_.
#
# Begin pruning from step ``0`` and prune every ``100`` steps for two epochs. As you
# step through this pruning scheduler, the sparsity of pruned modules will increase
# gradually from the initial value to the target value.

from coremltools.optimize.torch.pruning import PolynomialDecayScheduler

scheduler = PolynomialDecayScheduler(update_steps=list(range(0, 900, 100)))

#######################################################################
# Next, create an instance of the :py:class:`~.pruning.MagnitudePrunerConfig` class
# to specify how you want different submodules to be pruned.
# Set the target sparsity of the convolution layer
# to ``70 %`` and the dense layer to ``80 %``. The point of this is to demonstrate that
# different layers can be targeted at different sparsity levels. In practice, the sparsity
# level of a layer is a hyperparameter, which needs to be tuned for your requirements and
# the amenability of the layer to sparsification.


from coremltools.optimize.torch.pruning import (
    MagnitudePruner,
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig,
)

conv_config = ModuleMagnitudePrunerConfig(target_sparsity=0.7)
linear_config = ModuleMagnitudePrunerConfig(target_sparsity=0.8)

config = MagnitudePrunerConfig().set_module_type(torch.nn.Conv2d, conv_config)
config = config.set_module_type(torch.nn.Linear, linear_config)

pruner = MagnitudePruner(model, config)

########################################################################
# Next, call :py:meth:`~.pruning.MagnitudePruner.prepare` to insert pruning
# ``forward pre hooks`` on the modules configured previously.
# These forward pre hooks are called before a call to the forward
# method of the module. They multiply the parameter with a pruning mask, which
# is a tensor of the same shape as the parameter, in which each element has a value of
# either ``1`` or ``0``.

pruner.prepare(inplace=True)

########################################################################
# Fine-Tuning the Pruned Model
# ----------------------------
# The next step is to fine tune the model with pruning applied. In order to prune the model,
# call the :py:meth:`~.pruning.MagnitudePruner.step` method on the pruner
# after every call to ``optimizer.step()`` to step through the pruning schedule.

optimizer = torch.optim.Adam(model.parameters(), eps=1e-07)
accuracy_pruned = 0.0
num_epochs = 2

for epoch in range(num_epochs):
    # train one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        train_step(model, optimizer, train_loader, data, target, batch_idx, epoch)
        pruner.step()

    # evaluate
    accuracy_pruned = eval_model(model, test_loader)

########################################################################
# The evaluation shows that you can train a pruned network without losing
# accuracy with the final model. In practice, for more complex models,
# you have a trade-off between the sparsity and the validation accuracy
# that can be achieved for the model. Finding the right sweet spot on this
# trade-off curve depends on the model and task.

print("Accuracy of pruned network: {:.1f}%\n".format(accuracy_pruned))
print("Accuracy of unpruned network: {:.1f}%\n".format(accuracy_unpruned))

np.testing.assert_allclose(accuracy_pruned, accuracy_unpruned, atol=2)

########################################################################
# Finalizing the Model for Export
# -------------------------------
#
# The example shows that you can prune the model with a few code changes to your
# existing PyTorch training code. Now you can deploy this model on a device.
#
# To finalize the model for export, call :py:meth:`~.pruning.MagnitudePruner.finalize`
# on the pruner. This removes all the forward pre-hooks you had attached on the submodules.
# It also freezes the state of the pruner and multiplies the pruning mask with the corresponding
# weight matrix.

model.eval()
pruner.finalize(inplace=True)

########################################################################
# Exporting the Model for On-Device Execution
# -------------------------------------------
#
# In order to deploy the model, convert it to a Core ML model.
#
# Follow the same steps in Core ML Tools for exporting a regular PyTorch model
# (for details, see `Converting from PyTorch <https://coremltools.readme.io/docs/pytorch-conversion>`_).
# The parameter ``ct.PassPipeline.DEFAULT_PRUNING`` signals to the converter that
# the model being converted is a pruned model, and allows the model weights to be represented as
# sparse matrices, which have a smaller memory footprint than dense matrices.

import coremltools as ct

example_input = torch.rand(1, 1, 28, 28)
traced_model = torch.jit.trace(model, example_input)

coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING,
    minimum_deployment_target=ct.target.iOS16,
)

coreml_model.save("~/.mnist_pruning_data/pruned_model.mlpackage")
