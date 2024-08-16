#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch

from coremltools.optimize.torch._utils.k_means import (
    KMeansConfig,
    KMeansSupportedModulesRegistry,
    ParallelKMeans,
    SequentialKMeans,
)
from coremltools.test.optimize.torch.utils import count_unique_params


@pytest.mark.parametrize(
    "config",
    [
        KMeansConfig(n_bits=2, enable_per_channel_scale=False),
        {
            "conv1": {"weight": KMeansConfig(n_bits=4, enable_per_channel_scale=False)},
            "dense1": {"weight": KMeansConfig(n_bits=2, enable_per_channel_scale=True)},
        },
    ],
)
@pytest.mark.parametrize(
    "kmeans_cls",
    [SequentialKMeans, ParallelKMeans],
)
def test_k_means_mnist_per_weight(mock_name_main, mnist_model, config, kmeans_cls):
    model = kmeans_cls.cluster_weights(mnist_model, config=config, num_workers=4)

    layers = [
        ("conv1", model.conv1),
        ("conv2", model.conv2),
        ("dense1", model.dense1),
        ("dense2", model.dense2),
    ]
    with torch.no_grad():
        for layer_name, layer in layers:
            if isinstance(config, dict):
                if layer_name in config:
                    for param_name, layer_config in config[layer_name].items():
                        param = getattr(layer, param_name)
                        if layer_config.enable_per_channel_scale:
                            per_channel_scale_key = f"_COREML_/{param_name}/palettization_scale"
                            assert per_channel_scale_key in layer.state_dict()
                            per_channel_scale = layer.state_dict()[per_channel_scale_key]
                            param = param / per_channel_scale
                        assert count_unique_params(torch.unique(param)) == 2**layer_config.n_bits
                else:
                    assert len(torch.unique(layer.weight)) > 16
            else:
                assert len(torch.unique(layer.weight)) == 2**config.n_bits


@pytest.mark.parametrize(
    "config",
    [
        KMeansConfig(n_bits=4, block_size=4, axis=0, enable_per_channel_scale=False),
        KMeansConfig(n_bits=4, block_size=4, axis=0, enable_per_channel_scale=True),
        KMeansConfig(n_bits=4, block_size=4, axis=1, enable_per_channel_scale=False),
        KMeansConfig(n_bits=4, block_size=4, axis=1, enable_per_channel_scale=True),
    ],
)
@pytest.mark.parametrize(
    "kmeans_cls",
    [SequentialKMeans, ParallelKMeans],
)
def test_k_means_block_wise(mock_name_main, config, kmeans_cls):
    model = torch.nn.Conv2d(12, 32, (2, 2))
    model = kmeans_cls.cluster_weights(model, config=config, num_workers=4)
    block_size = config.block_size

    with torch.no_grad():
        weight = model.weight

        if config.enable_per_channel_scale:
            per_channel_scale_key = "_COREML_/weight/palettization_scale"
            assert per_channel_scale_key in model.state_dict()
            per_channel_scale = model.state_dict()[per_channel_scale_key]
            weight = weight / per_channel_scale

        if config.axis == 0:
            weight_flat = weight.flatten(1)
        else:
            weight_flat = weight.transpose(0, 1).flatten(1).transpose(0, 1)

        weight_shape = weight_flat.shape[config.axis]
        if config.axis == 0:
            for block_idx in range(0, weight_shape, block_size):
                assert (
                    count_unique_params(
                        torch.unique(weight_flat[block_idx : block_idx + block_size, :])
                    )
                    == 2**config.n_bits
                )
        else:
            for block_idx in range(0, weight_shape, block_size):
                assert (
                    count_unique_params(
                        torch.unique(weight_flat[:, block_idx : block_idx + block_size])
                    )
                    == 2**config.n_bits
                )


@pytest.mark.parametrize(
    "config",
    [
        KMeansConfig(n_bits=4, cluster_dim=4, axis=0, enable_per_channel_scale=False),
        KMeansConfig(n_bits=4, cluster_dim=4, axis=1, enable_per_channel_scale=False),
        KMeansConfig(n_bits=2, cluster_dim=2, axis=0, enable_per_channel_scale=False),
        KMeansConfig(n_bits=2, cluster_dim=2, axis=1, enable_per_channel_scale=False),
    ],
)
@pytest.mark.parametrize(
    "kmeans_cls",
    [SequentialKMeans, ParallelKMeans],
)
def test_k_means_vector_wise(mock_name_main, config, kmeans_cls):
    model = torch.nn.Conv2d(16, 8, (2, 2))
    model = kmeans_cls.cluster_weights(model, config=config, num_workers=4)
    cluster_dim = config.cluster_dim

    with torch.no_grad():
        weight = model.weight
        if config.axis == 0:
            weight_reshaped = weight.flatten(1).transpose(0, 1).reshape(-1, cluster_dim)
        elif config.axis == 1:
            weight_reshaped = (
                weight.transpose(0, 1).flatten(1).transpose(0, 1).reshape(-1, cluster_dim)
            )
        else:
            raise ValueError("axis must be 0 or 1.")

        unique_vector = torch.unique(weight_reshaped, dim=0)
        assert len(unique_vector) == 2**config.n_bits


@pytest.mark.parametrize("importance", [True, False])
@pytest.mark.parametrize("config", [tuple((4, 4, 0)), tuple((4, 4, 1))])
@pytest.mark.parametrize(
    "kmeans_cls",
    [
        SequentialKMeans,
        ParallelKMeans,
    ],
)
def test_k_means_masked(mock_name_main, importance, config, kmeans_cls):
    model = torch.nn.Linear(32, 32)
    block_size = config[1]
    axis = config[2]

    weight_mask = torch.ones_like(model.weight.data, dtype=torch.bool)
    for idx in range(32):
        if axis == 0:
            weight_mask[idx, torch.randperm(32)[:4]] = False
        else:
            weight_mask[torch.randperm(32)[:4], idx] = False

    importance = torch.abs(torch.randn(model.weight.shape)) if importance else None
    config = KMeansConfig(
        n_bits=config[0],
        block_size=block_size,
        enable_per_channel_scale=False,
        axis=axis,
        mask=weight_mask,
        importance=importance,
    )

    weight_clone = model.weight.clone()

    model = kmeans_cls.cluster_weights(model, config=config, num_workers=4)

    with torch.no_grad():
        model_weight = model.weight
        weight_shape = model_weight.shape[config.axis]
        for block_idx in range(0, weight_shape, block_size):
            if config.axis == 0:
                mask_block = weight_mask[block_idx : block_idx + block_size, :]
                weight_block_masked = model_weight[block_idx : block_idx + block_size, :][
                    mask_block
                ]
                weight_unmasked = model_weight[block_idx : block_idx + block_size, :][~mask_block]
                weight_orig_unmasked = weight_clone[block_idx : block_idx + block_size, :][
                    ~mask_block
                ]
            else:
                mask_block = weight_mask[:, block_idx : block_idx + block_size]
                weight_block_masked = model_weight[:, block_idx : block_idx + block_size][
                    mask_block
                ]
                weight_unmasked = model_weight[:, block_idx : block_idx + block_size][~mask_block]
                weight_orig_unmasked = weight_clone[:, block_idx : block_idx + block_size][
                    ~mask_block
                ]
            assert len(torch.unique(weight_block_masked)) == 2**config.n_bits
            assert torch.all(weight_orig_unmasked == weight_unmasked)


# region KMeansModule Tests


@pytest.mark.parametrize(
    "layer, layer_config",
    [
        (
            torch.nn.Linear(10, 100),
            {"weight": KMeansConfig(n_bits=4, enable_per_channel_scale=True)},
        ),
    ],
)
@torch.no_grad()
def test_zero_per_channel_scale(layer, layer_config):
    k_means_module_cls = KMeansSupportedModulesRegistry.get_kmeans_module(layer)
    k_means_module = k_means_module_cls(layer, layer_config)
    # Set one output chanel to zero so its per_channel_scale is 0
    layer.weight[0] = 0
    orig_weight = layer.weight.clone()
    # Scale weights
    scaled_weight = k_means_module._scale_by_per_channel_scale("weight", layer.weight)
    # Verify no NaN values are introduced
    assert not torch.any(torch.isnan(scaled_weight))
    # Confirm layer weights for corresponding channel remain 0
    assert torch.all(scaled_weight[0] == 0)
    # Unscale weights
    unscaled_weight = k_means_module._unscale_by_per_channel_scale("weight", layer.weight)
    # Verify no NaN values are introduced
    assert not torch.any(torch.isnan(unscaled_weight))
    # Confirm unscaled weights match original weights within tolerance
    assert torch.all(torch.isclose(unscaled_weight, orig_weight))


@pytest.mark.parametrize(
    "layer, param_name, axis, expected_shape",
    [
        (torch.nn.Conv2d(16, 32, 5), "weight", 0, (32, 16 * 5 * 5)),
        (torch.nn.Conv2d(16, 32, 5), "weight", 1, (32 * 5 * 5, 16)),
        (torch.nn.Linear(1024, 10), "weight", 0, (10, 1024)),
        (torch.nn.Linear(1024, 10), "weight", 1, (10, 1024)),
        (torch.nn.Embedding(50000, 256), "weight", 0, (50000, 256)),
        (torch.nn.Embedding(50000, 256), "weight", 1, (50000, 256)),
        (torch.nn.MultiheadAttention(256, 4), "in_proj_weight", 0, (3 * 256, 256)),
        (torch.nn.MultiheadAttention(256, 4), "in_proj_weight", 1, (3 * 256, 256)),
    ],
)
def test_parameter_reshaping(layer, param_name, axis, expected_shape):
    config = {param_name: KMeansConfig(n_bits=4, block_size=8, axis=axis)}
    k_means_module_cls = KMeansSupportedModulesRegistry.get_kmeans_module(layer)
    k_means_module = k_means_module_cls(layer, config)

    # reshape for kmeans
    param = getattr(layer, param_name)
    new_param = k_means_module._reshape_for_kmeans(param_name, param)
    assert new_param.shape == expected_shape

    # reshape back to original weight shape
    reshaped_param = k_means_module._reshape_to_original(param_name, new_param)
    assert reshaped_param.shape == param.shape

# endregion
