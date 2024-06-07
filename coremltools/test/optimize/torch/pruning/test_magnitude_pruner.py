#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn

from coremltools.optimize.torch._utils.metadata_utils import CompressionMetadata, CompressionType
from coremltools.optimize.torch.pruning import (
    MagnitudePruner,
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig,
)
from coremltools.optimize.torch.pruning._utils import n_m_mask


def _zero_loss(x, y):
    return torch.sum(x) * 0.0


def _mock_initializer(shape, dtype):
    # Each output channel is (entirely) an integer, increaing. This makes it so
    # that we know what to expect from the LnPruner.
    output_channel_index = 0
    num_output_channels = shape[output_channel_index]
    output_channel_values = np.arange(1, num_output_channels + 1, dtype=dtype)
    broadcast_shape = tuple(-1 if i == output_channel_index else 1 for i, _ in enumerate(shape))
    output_channel_values_reshaped = np.reshape(output_channel_values, broadcast_shape)
    return torch.tensor(np.broadcast_to(output_channel_values_reshaped, shape))


def _create_module():
    conv2d = torch.nn.Conv2d(in_channels=3,
                             out_channels=4,
                             kernel_size=(3, 3),
                             bias=False,
                             groups=1)
    conv2d.weight = torch.nn.Parameter(_mock_initializer(conv2d.weight.shape, np.float32))
    activation = torch.nn.ReLU()
    return torch.nn.Sequential(OrderedDict([
        ('conv2d', conv2d),
        ('activation', activation)]))


def _create_large_module():
    def _conv2d():
        return torch.nn.Conv2d(8, 8, (3, 3), bias=False, groups=1)

    return torch.nn.Sequential(OrderedDict([
        ('conv1', _conv2d()),
        ('conv2', _conv2d()),
        ('conv3', _conv2d()),
        ('flatten', torch.nn.Flatten()),
        ('linear1', torch.nn.Linear(2592, 100)),
        ('linear2', torch.nn.Linear(100, 10))]))


@pytest.fixture
def simple_module():
    return _create_module()


@pytest.fixture
def large_module():
    return _create_large_module()


@pytest.fixture(scope="module")
def sample_data():
    X = np.asarray([np.random.uniform(0.0, 1.0, size=(3, 8, 8)).astype(np.float32) for _ in range(4)])
    Y = np.asarray([np.random.uniform(0.0, 1.0, size=(4, 6, 6)).astype(np.float32) for _ in range(4)])
    X, Y = torch.tensor(X), torch.tensor(Y)
    return X, Y


@pytest.mark.parametrize("out_channels", [17, 127])
@pytest.mark.parametrize("block_size", [2, 3, 4])
def test_magnitude_pruner_nondivisible_block_size(out_channels, block_size):
    """
    Test block sparsity when the number of channels is not divisible by block size
    """
    conv2d = torch.nn.Conv2d(in_channels=3,
                             out_channels=out_channels,
                             kernel_size=(3, 3),
                             bias=False,
                             groups=1)
    weight_shape = tuple(conv2d.weight.shape)
    weight_tensor = torch.abs(torch.randn(*weight_shape))
    weight_tensor[weight_tensor == 0] = 1.0
    conv2d.weight = torch.nn.Parameter(weight_tensor)
    config = MagnitudePrunerConfig.from_dict(
            {"global_config":
                {
                    "scheduler": {"update_steps": [1, 2]},
                    "initial_sparsity": 0.0,
                    "target_sparsity": 0.5,
                    "block_size": block_size,
                }},
        )
    pruner = MagnitudePruner(conv2d, config)
    conv2d = pruner.prepare()

    for _ in range(4):
        pruner.step()

    if block_size > 1:
        block_sparse_channels = out_channels - out_channels % block_size
        for idx in range(0, block_sparse_channels, block_size):
            for jdx in range(1, block_size):
                assert torch.all(conv2d.weight_mask[idx] == conv2d.weight_mask[idx + jdx])

    sparsity = conv2d.weight_mask.eq(0).sum() / conv2d.weight_mask.numel()
    np.testing.assert_array_almost_equal(sparsity, 0.5, decimal=2)


@pytest.mark.parametrize("out_channels", [8])
@pytest.mark.parametrize("block_size", [5, 8, 9])
def test_magnitude_pruner_morethanhalf_block_size(out_channels, block_size):
    """
    Test block sparsity when the block size is greater than half the number of channels
    """
    conv2d = torch.nn.Conv2d(
        in_channels=3,
        out_channels=out_channels,
        kernel_size=(3, 3),
        bias=False,
        groups=1,
    )

    weight_tensor = torch.rand_like(conv2d.weight)
    weight_tensor[weight_tensor == 0] = 1.0
    conv2d.weight.data = weight_tensor

    config = MagnitudePrunerConfig.from_dict(
        {
            "global_config": {
                "scheduler": {"update_steps": [1, 2]},
                "initial_sparsity": 0.0,
                "target_sparsity": 0.5,
                "block_size": block_size,
            }
        },
    )
    pruner = MagnitudePruner(conv2d, config)
    conv2d = pruner.prepare()

    for _ in range(4):
        pruner.step()

    if block_size > 1:
        block_sparse_channels = out_channels - out_channels % block_size
        for idx in range(0, block_sparse_channels, block_size):
            for jdx in range(1, block_size):
                assert torch.all(conv2d.weight_mask[idx] == conv2d.weight_mask[idx + jdx])

    sparsity = conv2d.weight_mask.eq(0).sum() / conv2d.weight_mask.numel()
    assert np.isclose(sparsity, 0.5, rtol=0.05)


@pytest.mark.parametrize(
    "options",
    [("block_size", 2), ("granularity", "per_channel")],
)
def test_magnitude_pruner_n_m_ratio_param_usage(options):
    param_name, val = options
    with pytest.raises(Exception):
        MagnitudePrunerConfig.from_dict(
            {"global_config": {
                "n_m_ratio": [3, 4],
                param_name: val}},
        )


@pytest.mark.parametrize('config_dict', [
    {"module_type_configs": {"Linear": {"block_size": 2}}},
    {"module_name_configs": {"conv2d": {"block_size": 2}}},
    {"global_config": {"block_size": 2}},
    {},
])
def test_magnitude_pruner_config_global_config_set(config_dict):
    config = MagnitudePrunerConfig.from_dict(config_dict)
    if len(config_dict) == 0:
        assert config.global_config == ModuleMagnitudePrunerConfig()
    else:
        keys = ["global_config", "module_type_configs", "module_name_configs"]
        for key in keys:
            if key not in config_dict:
                param_in_config = getattr(config, key)
                assert param_in_config is None or len(param_in_config) == 0
        if "global_config" in config_dict:
            assert config.global_config.block_size == config_dict["global_config"]["block_size"]
        if "module_name_configs" in config_dict:
            for key in config_dict["module_name_configs"]:
                assert config.module_name_configs[key].block_size == \
                       config_dict["module_name_configs"][key]["block_size"]
        if "module_type_configs" in config_dict:
            for key in config_dict["module_type_configs"]:
                assert config.module_type_configs[key].block_size == \
                       config_dict["module_type_configs"][key]["block_size"]


@pytest.mark.parametrize('out_channels', [16, 64])
@pytest.mark.parametrize('block_size', [1, 4, 8])
def test_magnitude_pruner_block_sparsity(out_channels, block_size):
    """
    Test block sparsity structure is obtained by MagnitudePruner when block_size > 1
    """
    conv2d = torch.nn.Conv2d(in_channels=3,
                             out_channels=out_channels,
                             kernel_size=(3, 3),
                             bias=False,
                             groups=1)
    weight_shape = tuple(conv2d.weight.shape)
    weight_tensor = torch.abs(torch.randn(*weight_shape))
    weight_tensor[weight_tensor == 0] = 1.0
    conv2d.weight = torch.nn.Parameter(weight_tensor)
    config = MagnitudePrunerConfig.from_dict(
            {"global_config":
                {
                    "scheduler": {"update_steps": [1, 2]},
                    "initial_sparsity": 0.0,
                    "target_sparsity": 0.5,
                    "block_size": block_size,
                }},
        )
    pruner = MagnitudePruner(conv2d, config)
    conv2d = pruner.prepare()

    for _ in range(4):
        pruner.step()

    if block_size > 1:
        for idx in range(0, out_channels, block_size):
            for jdx in range(1, block_size):
                assert torch.all(conv2d.weight_mask[idx] == conv2d.weight_mask[idx + jdx])

    assert torch.sum(conv2d.weight_mask == 0).item() == int(0.5 * torch.numel(conv2d.weight))


def test_finalize(simple_module):
    """
    Test that calling finalize on the module leads to param being replaced with
    param_orig * param_mask.
    """
    config = MagnitudePrunerConfig.from_dict(
        {"global_config":
            {
                "scheduler": {"update_steps": [1, 2]},
                "initial_sparsity": 0.0,
                "target_sparsity": 0.5,
                "granularity": "per_channel"
            }},
    )
    pruner = MagnitudePruner(simple_module, config)
    simple_module = pruner.prepare()

    for _ in range(4):
        pruner.step()

    pruner.finalize(inplace=True)

    assert torch.sum(simple_module.conv2d.weight[:2] == 0).item() == 54
    assert torch.sum(simple_module.conv2d.weight[2] == 3).item() == 27
    assert torch.sum(simple_module.conv2d.weight[3] == 4).item() == 27


def test_magnitude_pruning_correctness(simple_module):
    """
    Test correctness of magnitude pruning.

    Initialize convolution weight with 4 output channels,
    with weights associated with channel `k` having integer value k+1 (k=0,...,3).
    We test that pruning twice indeed zeros out 3 output channels.
    """
    config = MagnitudePrunerConfig.from_dict(
        {"global_config":
            {
                "scheduler": {"update_steps": [2, 3]},
                "initial_sparsity": 0.0,
                "target_sparsity": 0.75,
                "granularity": "per_channel"
            }},
    )
    pruner = MagnitudePruner(simple_module, config)
    simple_module = pruner.prepare()

    # Perform 4 iterations: pruning should happen on steps 2 and 3
    # step 1: No pruning
    pruner.step()
    np.testing.assert_equal(simple_module.conv2d.weight_mask.numpy(),
                            np.array([1, 1, 1, 1], dtype=np.int32).reshape((4, 1, 1, 1)))
    # step 2: prune once, polynomial schedule will give new sparsity as 0.0, still no pruning
    pruner.step()
    np.testing.assert_equal(simple_module.conv2d.weight_mask.numpy(),
                            np.array([1, 1, 1, 1], dtype=np.int32).reshape((4, 1, 1, 1)))
    # step 3: prune once again, polynomial schedule will give new sparsity as 1.0, 75% = 3 out of 4
    # channels with least magnitude (first three channels) will be pruned out
    pruner.step()
    np.testing.assert_equal(simple_module.conv2d.weight_mask.numpy(),
                            np.array([0, 0, 0, 1], dtype=np.int32).reshape((4, 1, 1, 1)))

    # step 4: prune once again, polynomial schedule sparsity stays at 0.75, no further pruning
    pruner.step()
    np.testing.assert_equal(simple_module.conv2d.weight_mask.numpy(),
                            np.array([0, 0, 0, 1], dtype=np.int32).reshape((4, 1, 1, 1)))


def test_magnitude_pruning_training_and_validation(simple_module, sample_data):
    """
    Tests pruned weights are used for computing forward pass
    pruned module. Also demonstrates how pruner can be combined with
    training code in PyTorch, i.e, pruning can be done at a schedule
    different from training.

    Note: No actual training happens here because loss function is a no-op.
    """
    config = MagnitudePrunerConfig.from_dict(
        {"global_config":
            {
                "scheduler": {"update_steps": [2, 3]},
                "initial_sparsity": 0.0,
                "target_sparsity": 0.75,
                "granularity": "per_channel"
            }},
    )
    pruner = MagnitudePruner(simple_module, config)
    simple_module = pruner.prepare()

    # Train the model for 4 epochs
    num_epochs = 4
    X, Y = sample_data
    simple_module.train()
    optimizer = torch.optim.Adam(params=simple_module.parameters(), lr=0.001)
    for _ in range(num_epochs):
        for inp, label in zip(X, Y):
            inp = inp.view(1, *X.shape[1:])
            label = label.view(1, *Y.shape[1:])
            output = simple_module(inp)
            loss = _zero_loss(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pruner.step()

    # Test inference
    # After 4 iterations, pruner will zero out first 3 layers of conv2d layer in simple_module
    simple_module.eval()
    with torch.no_grad():
        x_test = torch.tensor(np.random.uniform(0.0, 1.0, size=(1, 3, 8, 8)).astype(np.float32))
        y_test = simple_module(x_test).detach().numpy()

    zero_output = y_test[:, :3, :, :]
    nonzero_output = y_test[:, 3:, :, :]
    np.testing.assert_equal(zero_output, np.zeros_like(zero_output))
    assert np.any(np.abs(nonzero_output) > 0.0)


@pytest.mark.parametrize('granularity', ["per_scalar", "per_channel"])
def test_magnitude_pruning_granularity_parameter_usage(simple_module, granularity):
    """
    Tests MagnitudePruner creates mask of the correct shape
    depending on the granularity parameter.

    We set target sparsity to 1.0 so the mask should be all zeros after 4 iterations.
    """
    config = MagnitudePrunerConfig.from_dict(
        {"global_config":
            {
                "scheduler": {"update_steps": [2, 3]},
                "initial_sparsity": 0.5,
                "target_sparsity": 1.0,
                "granularity": granularity
            }},
    )
    pruner = MagnitudePruner(simple_module, config)
    simple_module = pruner.prepare()

    # Perform 4 iterations
    for _ in range(4):
        pruner.step()

    mask_data = simple_module.conv2d.weight_mask.numpy()
    # Pruning mask should be all zeros since the pruner should be at 100% sparsity.
    if granularity == "per_scalar":
        expected_mask_shape = (4, 3, 3, 3)
    else:
        assert granularity == "per_channel"
        expected_mask_shape = (4, 1, 1, 1)
    np.testing.assert_equal(mask_data, np.zeros(expected_mask_shape))


@pytest.mark.parametrize('granularity', ["per_scalar", "per_channel"])
def test_pruner_finalize(simple_module, granularity):
    config = MagnitudePrunerConfig.from_dict(
        {"global_config":
            {
                "scheduler": {"update_steps": [2, 3]},
                "initial_sparsity": 0.5,
                "target_sparsity": 1.0,
                "granularity": granularity
            }},
    )
    pruner = MagnitudePruner(simple_module, config)
    simple_module = pruner.prepare()

    assert hasattr(simple_module.conv2d, "weight_mask")
    assert hasattr(simple_module.conv2d, "weight_orig")

    # Perform 4 iterations
    for _ in range(4):
        pruner.step()

    pruner.finalize(inplace=True)

    assert not hasattr(simple_module.conv2d, "weight_mask")
    assert not hasattr(simple_module.conv2d, "weight_orig")

    weight_data = simple_module.conv2d.weight.detach().numpy()
    np.testing.assert_equal(weight_data, np.zeros_like(weight_data))

    # calling finalize again is a no-op
    pruner.finalize(inplace=True)


@pytest.mark.parametrize("block_size", [1, 2])
@pytest.mark.parametrize("granularity", ["per_scalar", "per_channel"])
def test_sparsity_report_method(large_module, block_size, granularity):
    model = large_module
    target_sparsity = 0.5
    config = MagnitudePrunerConfig.from_dict(
        {"global_config":
            {
                "scheduler": {"update_steps": [2, 3]},
                "block_size": block_size,
                "initial_sparsity": 0.0,
                "target_sparsity": target_sparsity,
                "granularity": granularity
            }},
    )
    pruner = MagnitudePruner(model, config)
    pruner.prepare(inplace=True)

    inp = torch.ones(1, 8, 24, 24)
    for _ in range(4):
        model(inp)
        pruner.step()

    report = pruner.report()

    assert len(report) == 6

    for sparsity in [val["unstructured_weight_sparsity"] for _, val in report.items()]:
        assert sparsity == pytest.approx(target_sparsity, 0.1)
    if block_size == 2:
        for sparsity in [val["block2_weight_sparsity"] for _, val in report.items()]:
            assert sparsity == pytest.approx(target_sparsity, 0.1)
    if granularity == "per_channel":
        for sparsity in [
            val["structured_weight_sparsity"] for _, val in report.items()
        ][:3]:
            # only conv layers
            assert sparsity == pytest.approx(target_sparsity, 0.1)


def test_sparsity_report_block2_sparsity_not_applicable():
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 31, 2, 1),
                                torch.nn.Conv2d(31, 21, 2, 1))
    target_sparsity = 0.5
    config = MagnitudePrunerConfig.from_dict(
        {"global_config":
            {
                "scheduler": {"begin_step": 0},
                "initial_sparsity": 0.0,
                "target_sparsity": target_sparsity,
            }},
    )
    pruner = MagnitudePruner(model, config)
    pruner.prepare(inplace=True)

    inp = torch.ones(1, 1, 28, 28)
    for _ in range(2):
        pruner.step()
        model(inp)

    report = pruner.report()

    assert len(report) == 3

    for sparsity in [val["block2_weight_sparsity"] for _, val in report.items()]:
        assert sparsity == -1


def test_magnitude_pruner_cloning(simple_module):
    model = simple_module
    config = MagnitudePrunerConfig.from_dict(
        {"global_config":
            {
                "scheduler": {"update_steps": [0, 1]},
            }},
    )
    pruner = MagnitudePruner(model, config)
    pruner.prepare(inplace=True)

    model_copy = copy.deepcopy(model)

    assert hasattr(model_copy.conv2d, "pruning_method")
    assert torch.all(model_copy.conv2d.weight_orig == model.conv2d.weight_orig)
    assert torch.all(model_copy.conv2d.weight_mask == model.conv2d.weight_mask)

    pruner.finalize(inplace=True)

    model_copy_finalize = copy.deepcopy(model)

    assert not hasattr(model_copy_finalize.conv2d, "pruning_method")
    assert torch.all(model_copy_finalize.conv2d.weight == model.conv2d.weight)


@pytest.mark.parametrize('weights_shape', [[2, 8], [2, 8, 1, 1]])
@pytest.mark.parametrize('dim', [1, 0])
def test_nm_pruner_mask_computation(weights_shape, dim):
    weights = torch.tensor(
        [
            [2, 3, 0, 4, 5, 9, 1, 1],
            [3, 6, 1, 0, 2, 3, 8, 9]
        ]
    )
    if dim == 1:
        expected_mask = torch.tensor(
            [
                [0, 1, 0, 1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0, 1, 1]

            ]
        )
        nm = (2, 4)
    else:
        expected_mask = torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0, 1, 1]

            ]
        )
        nm = (1, 2)
    if weights_shape == [2, 8, 1, 1]:
        weights = weights.reshape([2, 8, 1, 1])
        expected_mask = expected_mask.reshape([2, 8, 1, 1])
    mask = n_m_mask(weights, nm, dim=dim)
    np.testing.assert_array_equal(mask, expected_mask)


@pytest.mark.parametrize("range_str", ["range(0, 25000, 100)", "range(0)"])
def test_polynomial_scheduler_range_str(range_str):
    pruner_config = MagnitudePrunerConfig.from_dict(
        {"global_config": {"scheduler": {"update_steps": range_str}}}
    )

    update_steps_tensor = torch.tensor(list(eval(range_str)))
    assert torch.all(
        pruner_config.global_config.scheduler.update_steps == update_steps_tensor
    )


def test_nm_pruner_polynomial_scheduler():
    model = torch.nn.Linear(8, 2)
    weights = torch.tensor(
        [[2, 3, 7, 4, 5, 8, 1, 6], [4, 5, 1, 6, 2, 3, 7, 8]], dtype=torch.float
    )
    model.weight.data = weights
    data = torch.randn(1, 8)

    config = MagnitudePrunerConfig.from_dict(
        {
            "global_config": {
                "scheduler": {"update_steps": range(8), "power": 1},
                "n_m_ratio": (7, 8),
            }
        }
    )
    pruner = MagnitudePruner(model, config)
    model = pruner.prepare()

    for idx in range(7):
        pruner.step()
        model(data)
        for row in range(2):
            assert torch.count_nonzero(model.weight_mask[row]) == (7 - idx)


def test_compression_metadata():
    """
    Test that calling finalize on the module leads to compression metadata being added to the model
    """
    model = nn.Sequential(
        OrderedDict([("conv1", nn.Conv2d(3, 32, 3)), ("fc1", nn.Linear(32, 100))])
    )
    # Disable compression for Linear layer
    config = MagnitudePrunerConfig().set_module_name("fc1", None)
    pruner = MagnitudePruner(model, config)
    pruner.prepare(inplace=True)
    pruner.step()
    pruner.finalize(inplace=True)

    # Verify metadata version is added to model
    assert "_COREML_/metadata_version" in model.state_dict()

    # Verify compression metadata is added for conv1
    metadata_dict = CompressionMetadata.from_state_dict(model.conv1.state_dict())
    assert len(metadata_dict) == 1
    assert "weight" in metadata_dict

    metadata = metadata_dict["weight"]
    assert metadata.compression_type == [CompressionType.pruning.value]

    # Verify no compression metadata is added for fc1
    metadata_dict = CompressionMetadata.from_state_dict(model.fc1.state_dict())
    assert len(metadata_dict) == 0
