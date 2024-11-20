#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import torch

from coremltools.optimize.torch.palettization import DKMPalettizer, DKMPalettizerConfig
from coremltools.optimize.torch.pruning import MagnitudePruner, MagnitudePrunerConfig
from coremltools.optimize.torch.quantization import LinearQuantizer, LinearQuantizerConfig
from coremltools.test.optimize.torch.utils import count_unique_params


@pytest.mark.parametrize(
    "prune_dict, quant_dict, expectation",
    [
        (
            {"global_config": {"target_sparsity": 0.5}},
            {"global_config": {"milestones": [0, 0, 10, 10]}},
            pytest.raises(RuntimeError),
        ),
        (
            {"module_name_configs": {"conv1": {"target_sparsity": 0.5}}},
            {
                "global_config": {"milestones": [0, 0, 10, 10]},
                "module_name_configs": {"conv1": None},
            },
            does_not_raise(),
        ),
    ],
)
def test_pruner_prepare_before_quantizer_prepare(
    mnist_model, mnist_example_input, prune_dict, quant_dict, expectation
):
    prune_config = MagnitudePrunerConfig.from_dict(prune_dict)
    quant_config = LinearQuantizerConfig.from_dict(quant_dict)

    pruner = MagnitudePruner(mnist_model, prune_config)
    pruner_prepared_model = pruner.prepare(inplace=True)

    quantizer = LinearQuantizer(pruner_prepared_model, quant_config)
    with expectation:
        quantizer.prepare(example_inputs=(mnist_example_input,))


def test_pruner_prepare_inplace_with_quantizer(mnist_model, mnist_example_input):
    quant_config = LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "milestones": [0, 0, 10, 10],
            }
        }
    )
    prune_config = MagnitudePrunerConfig.from_dict({"global_config": {"target_sparsity": 0.5}})

    quantizer = LinearQuantizer(mnist_model, quant_config)
    quant_prepared_model = quantizer.prepare(example_inputs=(mnist_example_input,))

    pruner = MagnitudePruner(quant_prepared_model, prune_config)
    # Case 1: with inplace = False
    with pytest.raises(RuntimeError):
        prepared_model = pruner.prepare()
    # Case 2: with inplace = True
    prepared_model = pruner.prepare(inplace=True)
    assert prepared_model is not None


@pytest.mark.parametrize(
    "prune_dict, quant_dict, expectation",
    [
        (
            {"global_config": {"target_sparsity": 0.5}},
            {"global_config": {"milestones": [0, 0, 10, 10]}},
            pytest.raises(RuntimeError),
        ),
        (
            {"module_name_configs": {"conv1": {"target_sparsity": 0.5}}},
            {
                "global_config": {"milestones": [0, 0, 10, 10]},
                "module_name_configs": {"conv1": None},
            },
            does_not_raise(),
        ),
    ],
)
def test_pruner_finalize_before_quantizer_finalize(
    mnist_model, mnist_example_input, prune_dict, quant_dict, expectation
):
    prune_config = MagnitudePrunerConfig.from_dict(prune_dict)
    quant_config = LinearQuantizerConfig.from_dict(quant_dict)

    quantizer = LinearQuantizer(mnist_model, quant_config)
    quant_prepared_model = quantizer.prepare(example_inputs=mnist_example_input)

    pruner = MagnitudePruner(quant_prepared_model, prune_config)
    prepared_model = pruner.prepare(inplace=True)

    prepared_model(mnist_example_input)

    with expectation:
        pruner.finalize()


def test_joint_prune_quantize(mnist_model, mnist_example_input):
    quant_config = LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "milestones": [0, 0, 10, 10],
            }
        }
    )
    prune_config = MagnitudePrunerConfig.from_dict({"global_config": {"target_sparsity": 0.5}})

    quantizer = LinearQuantizer(mnist_model, quant_config)
    quant_prepared_model = quantizer.prepare(example_inputs=(mnist_example_input,))

    pruner = MagnitudePruner(quant_prepared_model, prune_config)
    joint_prepared_model = pruner.prepare(inplace=True)

    quantizer.step()
    pruner.step()
    joint_prepared_model(mnist_example_input)

    quant_finalized_model = quantizer.finalize(inplace=True)
    joint_compressed_model = pruner.finalize(quant_finalized_model)

    # Verify sparsity
    for name, param in joint_compressed_model.named_parameters():
        if "weight" in name and "bn1" not in name:
            sparsity = torch.numel(param[param == 0]) / torch.numel(param)
            assert np.isclose(sparsity, 0.5, rtol=1e-1, atol=1e-1)

    # Verify quantization
    for name, submodule in joint_compressed_model.named_modules(remove_duplicate=True):
        if isinstance(submodule, torch.nn.Conv2d):
            assert isinstance(submodule, torch.ao.nn.quantized.reference.Conv2d)
        elif isinstance(submodule, torch.nn.Linear):
            assert isinstance(submodule, torch.ao.nn.quantized.reference.Linear)


def test_joint_prune_palettize(mnist_model, mnist_example_input):
    palett_config = DKMPalettizerConfig.from_dict(
        {"module_name_configs": {"conv2": {"n_bits": 4}, "dense2": {"n_bits": 4}}}
    )
    prune_config = MagnitudePrunerConfig.from_dict({"global_config": {"target_sparsity": 0.5}})

    palettizer = DKMPalettizer(mnist_model, palett_config)
    palett_prepared_model = palettizer.prepare()

    pruner = MagnitudePruner(palett_prepared_model, prune_config)
    joint_prepared_model = pruner.prepare(inplace=True)

    palettizer.step()
    pruner.step()
    joint_prepared_model(mnist_example_input)

    palett_finalized_model = palettizer.finalize(inplace=True)
    joint_compressed_model = pruner.finalize(palett_finalized_model)

    # Verify sparsity
    for name, param in joint_compressed_model.named_parameters():
        if "weight" in name and "bn1" not in name:
            sparsity = torch.numel(param[param == 0]) / torch.numel(param)
            np.isclose(sparsity, 0.5, rtol=1e-1, atol=1e-1)

    # Verify palettization
    conv_weight = joint_compressed_model.conv2.weight
    count_unique_params(torch.unique(conv_weight)) <= 16
    dense_weight = joint_compressed_model.dense2.weight
    count_unique_params(torch.unique(dense_weight)) <= 16


def test_quantize_then_prune(mnist_model, mnist_example_input):
    quant_config = LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "milestones": [0, 0, 10, 10],
            }
        }
    )
    prune_config = MagnitudePrunerConfig.from_dict({"global_config": {"target_sparsity": 0.5}})

    quantizer = LinearQuantizer(mnist_model, quant_config)
    quant_prepared_model = quantizer.prepare(example_inputs=(mnist_example_input,))

    quantizer.step()
    quant_prepared_model(mnist_example_input)

    quantized_model = quantizer.finalize()

    pruner = MagnitudePruner(quantized_model, prune_config)
    pruner_prepared_model = pruner.prepare(inplace=True)

    pruner.step()
    pruner_prepared_model(mnist_example_input)

    quantized_pruned_model = pruner.finalize()

    # Verify sparsity
    for name, param in quantized_pruned_model.named_parameters():
        if "weight" in name and "bn1" not in name:
            sparsity = torch.numel(param[param == 0]) / torch.numel(param)
            assert np.isclose(sparsity, 0.5, rtol=1e-1, atol=1e-1)

    # Verify quantization
    for name, submodule in quantized_pruned_model.named_modules(remove_duplicate=True):
        if isinstance(submodule, torch.nn.Conv2d):
            assert isinstance(submodule, torch.ao.nn.quantized.reference.Conv2d)
        elif isinstance(submodule, torch.nn.Linear):
            assert isinstance(submodule, torch.ao.nn.quantized.reference.Linear)


def test_pruner_then_quantize(mnist_model, mnist_example_input):
    quant_config = LinearQuantizerConfig.from_dict(
        {
            "global_config": {
                "milestones": [0, 0, 10, 10],
            }
        }
    )
    prune_config = MagnitudePrunerConfig.from_dict({"global_config": {"target_sparsity": 0.5}})

    pruner = MagnitudePruner(mnist_model, prune_config)
    pruner_prepared_model = pruner.prepare(inplace=True)

    pruner.step()
    pruner_prepared_model(mnist_example_input)

    pruned_model = pruner.finalize()

    quantizer = LinearQuantizer(pruned_model, quant_config)
    quant_prepared_model = quantizer.prepare(example_inputs=(mnist_example_input,), inplace=True)

    quantizer.step()
    quant_prepared_model(mnist_example_input)

    pruned_quantized_model = quantizer.finalize()

    # Verify sparsity
    for name, param in pruned_quantized_model.named_parameters():
        if "weight" in name and "bn1" not in name:
            sparsity = torch.numel(param[param == 0]) / torch.numel(param)
            assert np.isclose(sparsity, 0.5, rtol=1e-1, atol=1e-1)

    # Verify quantization
    for name, submodule in pruned_quantized_model.named_modules(remove_duplicate=True):
        if isinstance(submodule, torch.nn.Conv2d):
            assert isinstance(submodule, torch.ao.nn.quantized.reference.Conv2d)
        elif isinstance(submodule, torch.nn.Linear):
            assert isinstance(submodule, torch.ao.nn.quantized.reference.Linear)


def test_palettize_then_prune(mnist_model, mnist_example_input):
    palett_config = DKMPalettizerConfig.from_dict(
        {"module_name_configs": {"conv2": {"n_bits": 4}, "dense2": {"n_bits": 4}}}
    )
    prune_config = MagnitudePrunerConfig.from_dict({"global_config": {"target_sparsity": 0.5}})

    palettizer = DKMPalettizer(mnist_model, palett_config)
    palett_prepared_model = palettizer.prepare()

    palettizer.step()
    palett_prepared_model(mnist_example_input)

    palettized_model = palettizer.finalize()

    pruner = MagnitudePruner(palettized_model, prune_config)
    pruner_prepared_model = pruner.prepare(inplace=True)

    pruner.step()
    pruner_prepared_model(mnist_example_input)

    palettized_pruned_model = pruner.finalize()

    # Verify sparsity
    for name, param in palettized_pruned_model.named_parameters():
        if "weight" in name and "bn1" not in name:
            sparsity = torch.numel(param[param == 0]) / torch.numel(param)
            np.isclose(sparsity, 0.5, rtol=1e-1, atol=1e-1)

    # Verify palettization
    conv_weight = palettized_pruned_model.conv2.weight
    count_unique_params(torch.unique(conv_weight)) <= 16
    dense_weight = palettized_pruned_model.dense2.weight
    count_unique_params(torch.unique(dense_weight)) <= 16


def test_prune_then_palettize(mnist_model, mnist_example_input):
    palett_config = DKMPalettizerConfig.from_dict(
        {"module_name_configs": {"conv2": {"n_bits": 4}, "dense2": {"n_bits": 4}}}
    )
    prune_config = MagnitudePrunerConfig.from_dict({"global_config": {"target_sparsity": 0.5}})

    pruner = MagnitudePruner(mnist_model, prune_config)
    pruner_prepared_model = pruner.prepare(inplace=True)

    pruner.step()
    pruner_prepared_model(mnist_example_input)

    pruned_model = pruner.finalize()

    palettizer = DKMPalettizer(pruned_model, palett_config)
    palett_prepared_model = palettizer.prepare(inplace=True)

    palettizer.step()
    palett_prepared_model(mnist_example_input)

    pruned_palettized_model = palettizer.finalize(inplace=True)

    # Verify sparsity
    for name, param in pruned_palettized_model.named_parameters():
        if "weight" in name and "bn1" not in name:
            sparsity = torch.numel(param[param == 0]) / torch.numel(param)
            np.isclose(sparsity, 0.5, rtol=1e-1, atol=1e-1)

    # Verify palettization
    conv_weight = pruned_palettized_model.conv2.weight
    count_unique_params(torch.unique(conv_weight)) <= 16
    dense_weight = pruned_palettized_model.dense2.weight
    count_unique_params(torch.unique(dense_weight)) <= 16
