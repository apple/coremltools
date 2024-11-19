#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch

from coremltools.optimize.torch.palettization import (
    DKMPalettizer,
    DKMPalettizerConfig,
    FakePalettize,
    ModuleDKMPalettizerConfig,
)


@pytest.fixture
def palettizer_config():
    return DKMPalettizerConfig(
        global_config=ModuleDKMPalettizerConfig(n_bits=4, cluster_dim=1, weight_threshold=0)
    )


@pytest.mark.parametrize(
    "module",
    [
        torch.nn.Conv1d(2, 10, (1,)),
        torch.nn.Conv2d(2, 10, (2, 2)),
        torch.nn.Conv3d(2, 10, (2, 2, 2)),
        torch.nn.Linear(10, 20),
        torch.nn.LayerNorm(10),
        torch.nn.Embedding(10, 20),
    ],
)
def test_fake_palettize_insertion_weighted_modules(module, palettizer_config):
    wrapped_module = torch.nn.Sequential(module)

    palettizer = DKMPalettizer(wrapped_module, palettizer_config)
    palettized_module = palettizer.prepare()
    assert isinstance(palettized_module[0].weight_fake_quant, FakePalettize)

    assert hasattr(palettized_module[0].weight_fake_quant, "reset_parameters")
    assert hasattr(palettized_module[0].weight_fake_quant, "activation_post_process")
    assert isinstance(
        palettized_module[0].weight_fake_quant.activation_post_process,
        torch.quantization.MovingAveragePerChannelMinMaxObserver,
    )
    assert hasattr(
        palettized_module[0].weight_fake_quant.activation_post_process,
        "reset_parameters",
    )


@pytest.mark.parametrize("kdim,vdim", [(None, None), (1, 1)])
@pytest.mark.parametrize("batch_first", [True, False])
def test_fake_palettize_insertion_multihead_attention(kdim, vdim, batch_first, palettizer_config):
    attention_module = torch.nn.MultiheadAttention(
        bias=True,
        embed_dim=6,
        num_heads=3,
        add_bias_kv=True,
        kdim=kdim,
        vdim=vdim,
        batch_first=batch_first,
    )

    class WrappedModule(torch.nn.Sequential):
        def __init__(self, module):
            super().__init__(module)

        def forward(self, query, key, value):
            return self[0](query, key, value)

    wrapped_module = WrappedModule(attention_module)

    palettizer = DKMPalettizer(wrapped_module, palettizer_config)
    palettized_module = palettizer.prepare(inplace=False)
    palettizer.enable_fake_palett(True)

    query_shape = (2, 3, 6)
    assert isinstance(palettized_module[0].out_proj.weight_fake_quant, FakePalettize)
    assert palettized_module[0].out_proj.weight_fake_quant.fake_palett_enabled
    if kdim is None and vdim is None:
        assert isinstance(palettized_module[0].in_proj_weight_fake_quant, FakePalettize)
        assert palettized_module[0].in_proj_weight_fake_quant.fake_palett_enabled
        data_q = data_k = data_v = torch.randn(query_shape)
    else:
        assert isinstance(palettized_module[0].q_proj_weight_fake_quant, FakePalettize)
        assert palettized_module[0].q_proj_weight_fake_quant.fake_palett_enabled
        assert isinstance(palettized_module[0].k_proj_weight_fake_quant, FakePalettize)
        assert palettized_module[0].k_proj_weight_fake_quant.fake_palett_enabled
        assert isinstance(palettized_module[0].v_proj_weight_fake_quant, FakePalettize)
        assert palettized_module[0].v_proj_weight_fake_quant.fake_palett_enabled
        data_q = torch.randn(query_shape)
        data_k = data_v = torch.randn(2, 3, 1)

    palettizer.enable_fake_palett(False)
    output, _ = palettized_module(data_q, data_k, data_v)
    if batch_first:
        assert output.shape[0] == query_shape[0]
    else:
        assert output.shape[1] == query_shape[1]
    palettizer.finalize()
    assert torch.all(palettized_module[0].out_proj.bias == attention_module.out_proj.bias)
    assert torch.all(palettized_module[0].in_proj_bias == attention_module.in_proj_bias)
    assert torch.all(palettized_module[0].bias_k == attention_module.bias_k)
    assert torch.all(palettized_module[0].bias_v == attention_module.bias_v)
    # assert hasattr()


@pytest.mark.parametrize("module", [torch.nn.Conv1d(2, 10, (1,))])
def test_fake_palettize_train_no_grad_fwd(module, palettizer_config):
    wrapped_module = torch.nn.Sequential(module)

    palettizer = DKMPalettizer(wrapped_module, palettizer_config)
    palettized_module = palettizer.prepare()
    palettized_module.train()
    palettizer.step()
    with torch.no_grad():
        palettized_module(torch.randn(3, 2, 10))
