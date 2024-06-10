#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os

import numpy as np
import torch

import coremltools.test.optimize.torch.utils as utils

batch_size = 128


def verify_global_pruning_amount(supported_modules, model, expected_sparsity):
    total_params = 0
    unpruned_params = 0
    for name, module in model.named_modules():
        if type(module) in supported_modules:
            total_params += module.weight.numel()
            if hasattr(module, "weight_mask"):
                unpruned_params += torch.nonzero(module.weight_mask, as_tuple=False).size(0)
            else:
                unpruned_params += torch.nonzero(module.weight, as_tuple=False).size(0)

    actual_global_sparsity = 1 - unpruned_params / total_params
    np.testing.assert_allclose(actual_global_sparsity, expected_sparsity, atol=0.02)


def train_and_eval_model(model, dataset, pruner, num_epochs, pass_loss=False):
    train_loader, test_loader = utils.setup_data_loaders(dataset, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), eps=1e-07, weight_decay=1e-4)

    # train the model
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            loss = utils.train_step(model, optimizer, train_loader, data, target, batch_idx, epoch)
            if pass_loss:
                pruner.step(epoch, loss)
            else:
                pruner.step()

    accuracy = utils.eval_model(model, test_loader)
    return accuracy

def get_compression_ratio(model, pruner):
    # export the model
    import coremltools as ct

    model.eval()
    pruner.finalize(inplace=True)
    example_input = torch.rand(1, 1, 28, 28)
    traced_model = torch.jit.trace(model, example_input)

    converted_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=example_input.shape)],
    )

    # save and get size
    converted_model.save("/tmp/converted_model_unpruned.mlpackage")
    unpruned_model_size = os.path.getsize(
        "/tmp/converted_model_unpruned.mlpackage/Data/com.apple.CoreML/weights/weight.bin")

    # compress the model
    pruned_model = ct.compression_utils.sparsify_weights(converted_model, mode="threshold_based", threshold=1e-12)

    # save and get size
    pruned_model.save("/tmp/converted_model_pruned.mlpackage")
    pruned_model_size = os.path.getsize(
        "/tmp/converted_model_pruned.mlpackage/Data/com.apple.CoreML/weights/weight.bin")

    compression_ratio = pruned_model_size/unpruned_model_size

    print(f"Compression ratio: {compression_ratio}")
    return compression_ratio


def get_model_and_pruner(mnist_model, pruner_cls, pruner_config):
    model = mnist_model
    pruner = pruner_cls(model, pruner_config)
    pruner.prepare(inplace=True)
    return model, pruner
