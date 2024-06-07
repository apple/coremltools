#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict
from typing import Tuple

import pytest
import torch

from coremltools.optimize.torch.layerwise_compression import (
    LayerwiseCompressor,
    LayerwiseCompressorConfig,
)
from coremltools.optimize.torch.palettization import (
    PostTrainingPalettizer,
    PostTrainingPalettizerConfig,
)
from coremltools.optimize.torch.palettization.sensitive_k_means import (
    SKMPalettizer,
    SKMPalettizerConfig,
)
from coremltools.optimize.torch.quantization import (
    PostTrainingQuantizer,
    PostTrainingQuantizerConfig,
)


@pytest.fixture()
def model_for_compression(request) -> torch.nn.Module:
    decomposed_multihead_forward = request.param

    class ProjectionModule(torch.nn.Module):
        def __init__(self, embed_dim: int, hidden_dim: int):
            super().__init__()
            self.query = torch.nn.Linear(embed_dim, hidden_dim)
            self.key = torch.nn.Linear(embed_dim, hidden_dim)
            self.value = torch.nn.Linear(embed_dim, hidden_dim)

        def forward(self, x: torch.Tensor):
            return self.query(x), self.key(x), self.value(x)

    if decomposed_multihead_forward:

        class MultiheadWrapper(torch.nn.Module):
            def __init__(self, multihead_layer):
                super().__init__()
                self.layer = multihead_layer

            def forward(self, q, k, v):
                return self.layer(q, k, v, need_weights=False)[0]

    else:

        class MultiheadWrapper(torch.nn.Module):
            def __init__(self, multihead_layer):
                super().__init__()
                self.layer = multihead_layer

            def forward(self, x: Tuple[torch.Tensor]):
                return self.layer(x[0], x[1], x[2], need_weights=False)[0]

    class LinearWrapper(torch.nn.Module):
        def __init__(self, linear_layer):
            super().__init__()
            self.layer = linear_layer

        def forward(self, x):
            out = self.layer(x)
            return out.reshape(-1, 100, 10, 10)

    return torch.nn.Sequential(
        OrderedDict(
            [
                ("embedding", torch.nn.Embedding(100, 100)),
                ("projection", ProjectionModule(100, 100)),
                (
                    "multihead",
                    MultiheadWrapper(torch.nn.MultiheadAttention(100, 5, batch_first=True)),
                ),
                ("linear", LinearWrapper(torch.nn.Linear(100, 100))),
                ("conv", torch.nn.Conv2d(100, 100, (3, 3), padding=(1, 1))),
            ]
        )
    )


@pytest.mark.parametrize(
    "config, expected_num_columns",
    [
        (
            {
                "global_config": {"algorithm": "gptq", "weight_dtype": "uint4"},
                "module_name_configs": {"multihead.layer.out_proj": None},
                "input_cacher": "default",
                "calibration_nsamples": 128,
            },
            3,
        ),
        (
            {
                "global_config": {
                    "algorithm": "gptq",
                    "weight_dtype": "uint4",
                    "enable_normal_float": True,
                },
                "module_name_configs": {"multihead.layer.out_proj": None},
                "input_cacher": "default",
                "calibration_nsamples": 128,
            },
            3,
        ),
        (
            {
                "global_config": {"algorithm": "gptq", "weight_dtype": "uint8"},
                "module_name_configs": {
                    "projection.*": {
                        "algorithm": "sparse_gpt",
                        "weight_dtype": "uint8",
                        "target_sparsity": 0.25,
                    },
                    "multihead.layer.out_proj": None,
                },
                "input_cacher": "default",
                "calibration_nsamples": 128,
            },
            6,
        ),
    ],
)
@pytest.mark.parametrize("model_for_compression", [True], indirect=True)
def test_report_layerwise_compressor(model_for_compression, config, expected_num_columns):
    config = LayerwiseCompressorConfig.from_dict(config)
    compressor = LayerwiseCompressor(model_for_compression, config)

    def compression_loader():
        dataset = torch.utils.data.TensorDataset(torch.randint(0, high=100, size=(100, 100)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        for data in loader:
            yield data[0]

    compressor.compress(compression_loader(), device="cpu")
    report = compressor.report()
    print(report)
    assert (len(report)) == 5
    expected_params = [
        "projection.query.weight",
        "projection.key.weight",
        "projection.value.weight",
        "linear.layer.weight",
        "conv.weight",
    ]
    for param_name in expected_params:
        assert param_name in report
        param_report = report[param_name]
        assert len(param_report) == expected_num_columns
        if not config.global_config.enable_normal_float:
            assert param_report["dtype"] == f"dtype=int{config.global_config.weight_n_bits}"
        else:
            assert (
                param_report["palettization_mode"]
                == f"num_clusters={2 ** config.global_config.weight_n_bits}, cluster_dim=1"
            )


@pytest.mark.parametrize("quantization_scheme", ["symmetric", "affine"])
@pytest.mark.parametrize(
    "granularity_block_size",
    [
        ("per_channel", None),
        ("per_tensor", None),
        ("per_block", 5),
    ],
)
@pytest.mark.parametrize("weight_dtype", ["int4", "int8"])
@pytest.mark.parametrize("model_for_compression", [True], indirect=True)
def test_report_post_training_quantization(
    model_for_compression,
    quantization_scheme,
    granularity_block_size,
    weight_dtype,
):
    granularity, block_size = granularity_block_size
    config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "weight_dtype": weight_dtype,
                "granularity": granularity,
                "block_size": block_size,
                "quantization_scheme": quantization_scheme,
            }
        }
    )
    compressor = PostTrainingQuantizer(model_for_compression, config)
    model = compressor.compress()

    report = compressor.report()

    assert (len(report)) == 7
    for param_name, param in model.named_parameters():
        if "embedding" not in param_name and "bias" not in param_name:
            assert param_name in report
            param_report = report[param_name]
            assert len(param_report) == 3
            assert param_report["dtype"] == f"dtype=int{config.global_config.weight_n_bits}"


@pytest.mark.parametrize(
    "config",
    [
        {
            "global_config": {"granularity": "per_tensor", "n_bits": 4},
        },
        {
            "global_config": {
                "granularity": "per_grouped_channel",
                "n_bits": 4,
                "group_size": 1,
            },
        },
        {
            "global_config": {
                "granularity": "per_grouped_channel",
                "n_bits": 4,
                "group_size": 5,
            },
        },
        {
            "global_config": {"granularity": "per_tensor", "n_bits": 4},
            "module_name_configs": {
                "linear.layer": {
                    "n_bits": 4,
                    "granularity": "per_tensor",
                    "cluster_dim": 5,
                },
                "conv": {
                    "n_bits": 4,
                    "granularity": "per_tensor",
                    "cluster_dim": 4,
                },
            },
        },
    ],
)
@pytest.mark.parametrize("model_for_compression", [True], indirect=True)
def test_report_post_training_palettization(model_for_compression, config):
    config = PostTrainingPalettizerConfig.from_dict(config)
    compressor = PostTrainingPalettizer(model_for_compression, config)
    model = compressor.compress(num_kmeans_workers=1)

    report = compressor.report()
    assert (len(report)) == 8
    for param_name, param in model.named_parameters():
        if "bias" not in param_name:
            assert param_name in report
            param_report = report[param_name]
            assert len(param_report) == 3
            assert "num_clusters=16" in param_report["palettization_mode"]


@pytest.mark.parametrize(
    "config",
    [
        {
            "global_config": {"granularity": "per_tensor", "n_bits": 6},
        },
        {
            "global_config": {
                "granularity": "per_grouped_channel",
                "n_bits": 8,
                "group_size": 1,
            },
        },
        {
            "global_config": {
                "granularity": "per_grouped_channel",
                "n_bits": 4,
                "group_size": 5,
            },
        },
    ],
)
@pytest.mark.parametrize("model_for_compression", [False], indirect=True)
def test_report_skm_palettizer(model_for_compression, config):
    config = SKMPalettizerConfig.from_dict(config)
    compressor = SKMPalettizer(model_for_compression, config)

    def compression_loader():
        dataset = torch.utils.data.TensorDataset(torch.randint(0, high=100, size=(100, 100)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        for data in loader:
            yield data[0]

    def loss_fn(model, data):
        out = model(data)
        return torch.sum(out)

    model = compressor.compress(
        dataloader=compression_loader(),
        loss_fn=loss_fn,
    )

    report = compressor.report()
    assert (len(report)) == 8
    for param_name, param in model.named_parameters():
        if "bias" not in param_name:
            assert param_name in report
            param_report = report[param_name]
            assert len(param_report) == 3
            assert (
                param_report["palettization_mode"]
                == f"num_clusters={2 ** config.global_config.n_bits}, cluster_dim=1"
            )
