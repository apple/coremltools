#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Dict
from unittest.mock import ANY, Mock, patch

import pytest
import torch

from coremltools.optimize.torch._utils.fsdp_utils import (
    FSDPAutoWrapPolicy,
    ModuleWrapPolicy,
    SizeBasedWrapPolicy,
)
from coremltools.optimize.torch._utils.k_means import KMeansConfig
from coremltools.optimize.torch.palettization.sensitive_k_means import (
    ModuleSKMPalettizerConfig,
    SKMPalettizer,
    SKMPalettizerConfig,
)


@pytest.mark.parametrize(
    "auto_wrap_policy",
    [
        ModuleWrapPolicy(module_classes=torch.nn.Linear),
        SizeBasedWrapPolicy(min_num_params=1000),
        None,
    ],
)
@pytest.mark.parametrize("num_sensitivity_workers", [1, 8])
@pytest.mark.parametrize("num_kmeans_workers", [1, 8])
def test_fsdp_auto_wrap_policy_compress_call(
    mocker, num_kmeans_workers, num_sensitivity_workers, auto_wrap_policy
):
    """
    Test compress passes fsdp_auto_wrap_policy argument correctly to
    compute_sensitivity method.
    """
    mock_compute_sensitivity = Mock(return_value={"weight": None})

    mocker.patch.object(SKMPalettizer, "compute_sensitivity", mock_compute_sensitivity)
    mocker.patch("coremltools.optimize.torch.palettization.sensitive_k_means._ParallelKMeans")
    mocker.patch("coremltools.optimize.torch.palettization.sensitive_k_means._SequentialKMeans")

    model = torch.nn.Linear(5, 10)
    palettizer = SKMPalettizer(model)

    palettizer.compress(
        num_sensitivity_workers=num_sensitivity_workers,
        fsdp_auto_wrap_policy=auto_wrap_policy,
        num_kmeans_workers=num_kmeans_workers,
    )

    mock_compute_sensitivity.assert_called_once_with(
        None,
        None,
        None,
        num_sensitivity_workers,
        fsdp_auto_wrap_policy=auto_wrap_policy,
    )


@pytest.mark.parametrize(
    "auto_wrap_policy",
    [
        ModuleWrapPolicy(module_classes=torch.nn.Linear),
        SizeBasedWrapPolicy(min_num_params=1000),
        None,
    ],
)
@pytest.mark.parametrize("num_sensitivity_workers", [1, 8])
def test_fsdp_auto_wrap_policy_compute_sensitivity_call(
    mocker, num_sensitivity_workers, auto_wrap_policy
):
    """
    Test compute_sensitivity passes fsdp_auto_wrap_policy argument correctly to
    impl methods
    """
    model = torch.nn.Linear(5, 10)

    mocker.patch("coremltools.optimize.torch.palettization.sensitive_k_means._torch.save")
    mocker.patch(
        "coremltools.optimize.torch.palettization.sensitive_k_means._torch.load",
        Mock(return_value=model.state_dict()),
    )
    mocker.patch(
        "coremltools.optimize.torch.palettization.sensitive_k_means._torch.cuda.is_available",
        Mock(return_value=True),
    )
    mock_ctx = Mock()
    mocker.patch(
        "coremltools.optimize.torch.palettization.sensitive_k_means._mp.get_context",
        Mock(return_value=mock_ctx),
    )
    mock_compute_sen_single_worker = Mock()
    mocker.patch.object(
        SKMPalettizer,
        "_compute_sensitivity_impl_single_worker",
        mock_compute_sen_single_worker,
    )
    mock_dataset = Mock()
    mocker.patch.object(SKMPalettizer, "_get_dataset", Mock(return_value=mock_dataset))
    mocker.patch.object(SKMPalettizer, "_process_sensitivity")

    palettizer = SKMPalettizer(model)

    dataloader = []
    loss_fn = lambda mod, dat: mod(dat)

    palettizer.compute_sensitivity(
        dataloader=dataloader,
        loss_fn=loss_fn,
        sensitivity_path=None,
        num_sensitivity_workers=num_sensitivity_workers,
        fsdp_auto_wrap_policy=auto_wrap_policy,
    )

    if num_sensitivity_workers > 1:
        for rank in range(num_sensitivity_workers):
            mock_ctx.Process.assert_any_call(
                target=palettizer._compute_sensitivity_impl_multiple_workers,
                args=(
                    rank,
                    num_sensitivity_workers,
                    mock_dataset,
                    loss_fn,
                    None,
                    auto_wrap_policy,
                ),
                name=f"Process-{rank}",
            )
    else:
        mock_compute_sen_single_worker.assert_called_once_with(mock_dataset, loss_fn, None)


@pytest.mark.parametrize("auto_wrap_policy", [Mock(spec=FSDPAutoWrapPolicy), None])
def test_fsdp_auto_wrap_policy_multi_worker_compute_sensitivity_call(mocker, auto_wrap_policy):
    """
    Test _compute_sensitivity_impl_multiple_workers passes correct value of fsdp auto wrap policy
    to FSDP call
    """
    model = torch.nn.Linear(5, 10)

    mocker.patch("coremltools.optimize.torch.palettization.sensitive_k_means._torch")
    mocker.patch("coremltools.optimize.torch.palettization.sensitive_k_means._ddp_setup")
    mocker.patch(
        "coremltools.optimize.torch.palettization.sensitive_k_means._is_leader",
        Mock(return_value=True),
    )
    mocker.patch.object(
        SKMPalettizer, "_register_grad_square_hooks", Mock(return_value=nullcontext())
    )

    if auto_wrap_policy is not None:
        expected_auto_wrap_policy = Mock()
        auto_wrap_policy.get_policy.return_value = expected_auto_wrap_policy
    else:
        expected_auto_wrap_policy = None

    with patch(
        "coremltools.optimize.torch.palettization.sensitive_k_means._FSDP", autospec=True
    ) as mock_fsdp:
        mock_fsdp.state_dict_type.return_value = nullcontext()

        palettizer = SKMPalettizer(model)

        palettizer._compute_sensitivity_impl_multiple_workers(
            rank=0,
            num_workers=1,
            dataset=[None],
            loss_fn=Mock(),
            sensitivity_path=None,
            fsdp_auto_wrap_policy=auto_wrap_policy,
        )

        # test FSDP either gets None or output of get_policy method on the
        # FSDPAutoWrapPolicy object
        mock_fsdp.assert_called_with(
            module=palettizer._model,
            auto_wrap_policy=expected_auto_wrap_policy,
            sharding_strategy=ANY,
            use_orig_params=False,
            device_id=ANY,
            sync_module_states=True,
        )


@pytest.fixture()
def model_for_compression() -> torch.nn.Module:
    return torch.nn.Sequential(
        OrderedDict(
            [
                ("modconv", torch.nn.Conv2d(3, 10, (3, 3))),
                ("modlinear", torch.nn.Linear(2, 5)),
                ("multihead", torch.nn.MultiheadAttention(10, 5)),
                ("embedding", torch.nn.Embedding(100, 10)),
            ]
        )
    )


@pytest.fixture()
def sensitvity_dict_for_compression() -> Dict[str, Any]:
    return {
        "modconv.weight": Mock(),
        "modlinear.weight": Mock(),
        "multihead.in_proj_weight": Mock(),
        "multihead.out_proj.weight": Mock(),
        "embedding.weight": Mock(),
    }


@pytest.fixture()
def model_for_compression_custom_module() -> torch.nn.Module:
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(data=torch.randn(5, 10))

    return torch.nn.Sequential(
        OrderedDict(
            [
                ("modconv", torch.nn.Conv2d(3, 10, (3, 3))),
                ("modlinear", torch.nn.Linear(2, 5)),
                ("multihead", torch.nn.MultiheadAttention(10, 5)),
                ("custom", MyModule()),
            ]
        )
    )


@pytest.mark.parametrize(
    "model,sensitivity_dict,config,kmeans_keys",
    [
        (torch.nn.Linear(5, 10), {"weight": None}, None, {"": "weight"}),
        (
            "model_for_compression",
            "sensitvity_dict_for_compression",
            SKMPalettizerConfig(
                global_config=ModuleSKMPalettizerConfig(),
                module_name_configs={
                    "modconv": None,
                },
            ),
            {
                "modlinear": "weight",
                "multihead": "in_proj_weight",
                "multihead.out_proj": "weight",
                "embedding": "weight",
            },
        ),
        (
            "model_for_compression",
            "sensitvity_dict_for_compression",
            SKMPalettizerConfig(
                global_config=ModuleSKMPalettizerConfig(),
                module_name_configs={
                    "mod.*": None,
                },
            ),
            {
                "multihead": "in_proj_weight",
                "multihead.out_proj": "weight",
                "embedding": "weight",
            },
        ),
        (
            "model_for_compression",
            "sensitvity_dict_for_compression",
            SKMPalettizerConfig(
                global_config=ModuleSKMPalettizerConfig(),
                module_type_configs={torch.nn.Embedding: None},
            ),
            {
                "modconv": "weight",
                "modlinear": "weight",
                "multihead": "in_proj_weight",
                "multihead.out_proj": "weight",
            },
        ),
        (
            "model_for_compression",
            "sensitvity_dict_for_compression",
            SKMPalettizerConfig(
                global_config=ModuleSKMPalettizerConfig(),
                module_type_configs={"MultiheadAttention": None},
                module_name_configs={"multihead.out_proj": None},
            ),
            {"modconv": "weight", "modlinear": "weight", "embedding": "weight"},
        ),
        (
            "model_for_compression_custom_module",
            "sensitvity_dict_for_compression",
            None,
            {
                "modconv": "weight",
                "modlinear": "weight",
                "multihead": "in_proj_weight",
                "multihead.out_proj": "weight",
            },
        ),
    ],
)
@pytest.mark.parametrize("num_kmeans_workers", [1, 8])
def test_compress_cluster_weights_call(
    mocker, num_kmeans_workers, model, sensitivity_dict, config, kmeans_keys, request
):
    """
    Test ParallelKMeans/SequentialKMeans are called with correct arguments
    """
    if isinstance(model, str):
        model = request.getfixturevalue(model)
    if isinstance(sensitivity_dict, str):
        sensitivity_dict = request.getfixturevalue(sensitivity_dict)

    mocker.patch.object(SKMPalettizer, "compute_sensitivity", Mock(return_value=sensitivity_dict))
    mock_parallel = mocker.patch(
        "coremltools.optimize.torch.palettization.sensitive_k_means._ParallelKMeans"
    )
    mock_sequential = mocker.patch(
        "coremltools.optimize.torch.palettization.sensitive_k_means._SequentialKMeans"
    )

    palettizer = SKMPalettizer(model, config)

    palettizer.compress(
        num_sensitivity_workers=1,
        fsdp_auto_wrap_policy=None,
        num_kmeans_workers=num_kmeans_workers,
    )

    k_means_config_dict = {}
    for key, val in kmeans_keys.items():
        sensitivity_key = f"{key}.{val}" if len(key) > 0 else val
        k_means_config_dict[key] = {
            val: KMeansConfig(
                n_bits=ModuleSKMPalettizerConfig().n_bits,
                axis=0,
                block_size=None,
                cluster_dim=1,
                importance=sensitivity_dict[sensitivity_key],
                enable_per_channel_scale=ModuleSKMPalettizerConfig().enable_per_channel_scale,
            )
        }

    if num_kmeans_workers > 1:
        mock_parallel.cluster_weights.assert_called_once_with(
            palettizer._model, k_means_config_dict, num_workers=num_kmeans_workers
        )
    else:
        mock_sequential.cluster_weights.assert_called_once_with(
            palettizer._model,
            k_means_config_dict,
        )
