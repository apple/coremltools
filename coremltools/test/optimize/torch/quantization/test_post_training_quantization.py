#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import numpy as np
import pytest
import torch

ct = pytest.importorskip("coremltools")
pytest.importorskip("coremltools.optimize.coreml._utils")

from coremltools.optimize.torch.optimization_config import QuantizationGranularity
from coremltools.optimize.torch.quantization import (
    PostTrainingQuantizer,
    PostTrainingQuantizerConfig,
    QuantizationScheme,
)

np.random.seed(0)
torch.manual_seed(0)


def get_rmse(a, b):
    return torch.norm(torch.abs(a - b))


def get_atol_rtol(block_size, weight_n_bits):
    if block_size is None:
        block_size = 0
    if block_size == 1:
        # With block_size == 1, the information loss is minimum.
        atol, rtol = 1e-02, 1e-02
    elif weight_n_bits >= 4 and block_size < 3:
        # When block size is small and nbits is large, the information loss is limited.
        atol, rtol = 3e-02, 3e-02
    elif weight_n_bits <= 2 and block_size >= 2:
        atol, rtol = 0.5, 0.5
    else:
        atol, rtol = 0.4, 0.4
    return (atol, rtol)


def test_ptq_default_config():
    config = PostTrainingQuantizerConfig()
    ptq = PostTrainingQuantizer(torch.nn.Identity(), config)
    assert ptq is not None
    assert config.global_config.block_size is None
    assert config.global_config.weight_dtype == torch.int8
    assert config.global_config.quantization_scheme == QuantizationScheme.symmetric
    assert config.global_config.weight_dtype == torch.int8
    assert config.global_config.granularity == QuantizationGranularity.per_channel


@pytest.mark.parametrize(
    "dtype,n_bits",
    [
        ["int4", 4],
        ["uint4", 4],
        ["int8", 8],
        ["uint8", 8],
        [torch.int8, 8],
        [torch.uint8, 8],
    ],
)
def test_ptq_config_n_bits(dtype, n_bits):
    config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "weight_dtype": dtype,
            }
        }
    )
    assert config.global_config.weight_n_bits == n_bits


@pytest.mark.parametrize(
    "module",
    [
        torch.nn.Linear(10, 10),
        torch.nn.Conv2d(10, 10, 3, 3),
        torch.nn.ConvTranspose2d(10, 20, 3, 3),
        torch.nn.Conv2d(20, 10, 3, 3),
        torch.nn.MultiheadAttention(
            bias=True,
            embed_dim=6,
            num_heads=3,
            add_bias_kv=True,
            kdim=1,
            vdim=1,
        ),
        torch.nn.MultiheadAttention(
            bias=True,
            embed_dim=6,
            num_heads=3,
            add_bias_kv=True,
            kdim=None,
            vdim=None,
        ),
    ],
)
@pytest.mark.parametrize(
    "granularity_block_size",
    [
        ("per_channel", None),
        ("per_tensor", None),
        ("per_block", 2),
        ("per_block", 5),
        ("per_block", (2,)),
        ("per_block", (5,)),
        ("per_block", (5, 2)),
        ("per_block", (2, 5)),
    ],
)
@pytest.mark.parametrize("quantization_scheme", ["symmetric", "affine"])
@pytest.mark.parametrize("weight_dtype", ["int8", "int4", "uint8", "uint4"])
def test_ptq_compress_all_combinations(
    module,
    quantization_scheme,
    granularity_block_size,
    weight_dtype,
):
    granularity, block_size = granularity_block_size
    config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": quantization_scheme,
                "granularity": granularity,
                "weight_dtype": weight_dtype,
                "block_size": block_size,
            }
        }
    )
    ptq = PostTrainingQuantizer(module, config)
    module = ptq.compress()


@pytest.mark.parametrize("quantization_scheme", ["symmetric", "affine"])
@pytest.mark.parametrize(
    "granularity_block_size",
    [
        ("per_channel", None),
        ("per_tensor", None),
        ("per_block", 2),
        ("per_block", 5),
    ],
)
@pytest.mark.parametrize("weight_dtype", ["int4", "int8"])
@pytest.mark.parametrize(
    "module",
    [
        torch.nn.Conv1d(10, 10, 3, 3),
        torch.nn.Conv2d(10, 10, 3, 3),
        torch.nn.Conv3d(10, 10, 3, 3),
        torch.nn.Linear(10, 10),
        torch.nn.ConvTranspose1d(10, 20, 3, 3),
        torch.nn.ConvTranspose2d(10, 20, 3, 3),
        torch.nn.ConvTranspose3d(10, 20, 3, 3),
    ],
)
def test_ptq_post_compress_conv_linear(
    quantization_scheme, granularity_block_size, weight_dtype, module
):
    granularity, block_size = granularity_block_size
    orig_weight = module.weight.clone()
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
    ptq = PostTrainingQuantizer(module, config)
    module = ptq.compress()
    if isinstance(
        module,
        (
            torch.nn.ConvTranspose1d,
            torch.nn.ConvTranspose2d,
            torch.nn.ConvTranspose3d,
        ),
    ):
        ch_axis = 1
        block_axis = 0
    elif isinstance(
        module,
        (
            torch.nn.Linear,
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Conv3d,
        ),
    ):
        ch_axis = 0
        block_axis = 1
    else:
        raise NotImplementedError

    assert hasattr(module, "_COREML_/weight/quantization_scale")
    if quantization_scheme == "affine":
        assert hasattr(module, "_COREML_/weight/zero_point")

    if granularity in ["per_channel", "per_block"]:
        assert (
            getattr(module, "_COREML_/weight/quantization_scale").shape[ch_axis]
            == module.weight.shape[ch_axis]
        )
        if quantization_scheme == "affine":
            assert (
                getattr(module, "_COREML_/weight/zero_point").shape[ch_axis]
                == module.weight.shape[ch_axis]
            )
        if granularity == "per_block":
            assert (
                getattr(module, "_COREML_/weight/quantization_scale").shape[block_axis]
                == module.weight.shape[block_axis] / block_size
            )
            if quantization_scheme == "affine":
                assert (
                    getattr(module, "_COREML_/weight/zero_point").shape[block_axis]
                    == module.weight.shape[block_axis] / block_size
                )

    assert not torch.equal(orig_weight, module.weight)
    atol, rtol = get_atol_rtol(block_size, config.global_config.weight_n_bits)
    np.testing.assert_allclose(
        orig_weight.detach().numpy(),
        module.weight.detach().numpy(),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("quantization_scheme", ["symmetric", "affine"])
@pytest.mark.parametrize(
    "granularity_block_size",
    [
        ("per_channel", None),
        ("per_tensor", None),
        ("per_block", 2),
        ("per_block", 5),
    ],
)
@pytest.mark.parametrize("weight_dtype", ["int4", "int8"])
def test_ptq_post_compress_multihead(
    quantization_scheme,
    granularity_block_size,
    weight_dtype,
):
    granularity, block_size = granularity_block_size
    module = torch.nn.MultiheadAttention(
        bias=True,
        embed_dim=10,
        num_heads=10,
        add_bias_kv=True,
        kdim=None,
        vdim=None,
    )
    assert hasattr(module, "in_proj_weight")
    assert hasattr(module.out_proj, "weight")
    orig_in_proj_weight = module.in_proj_weight.clone()
    orig_out_proj_weight = module.out_proj.weight.clone()
    config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "weight_dtype": weight_dtype,
                "quantization_scheme": quantization_scheme,
                "granularity": granularity,
                "block_size": block_size,
            }
        }
    )
    ptq = PostTrainingQuantizer(module, config)
    module = ptq.compress()

    assert hasattr(module, "_COREML_/in_proj_weight/quantization_scale")
    assert hasattr(module.out_proj, "_COREML_/weight/quantization_scale")
    if quantization_scheme == "affine":
        assert hasattr(module, "_COREML_/in_proj_weight/zero_point")
        assert hasattr(module.out_proj, "_COREML_/weight/zero_point")

    assert not torch.equal(orig_in_proj_weight, module.in_proj_weight)
    assert not torch.equal(orig_out_proj_weight, module.out_proj.weight)
    atol, rtol = get_atol_rtol(block_size, config.global_config.weight_n_bits)
    np.testing.assert_allclose(
        orig_in_proj_weight.detach().numpy(),
        module.in_proj_weight.detach().numpy(),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        orig_out_proj_weight.detach().numpy(),
        module.out_proj.weight.detach().numpy(),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "weight_dtype,n_bits",
    [
        ["int4", 4],
        ["uint4", 4],
        ["int8", 8],
        ["uint8", 8],
    ],
)
@pytest.mark.parametrize("qscheme", ["symmetric", "affine"])
def test_ptq_compression_metadata(weight_dtype, n_bits, qscheme):
    config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "quantization_scheme": qscheme,
                "weight_dtype": weight_dtype,
            }
        }
    )
    ptq = PostTrainingQuantizer(torch.nn.Linear(10, 10), config)
    model = ptq.compress()

    from coremltools.optimize.torch._utils.metadata_utils import CompressionType

    assert hasattr(model, "_COREML_/weight/compression_type")
    assert torch.IntTensor([CompressionType.quantization.value]) == getattr(
        model, "_COREML_/weight/compression_type"
    )
    assert torch.IntTensor([n_bits]) == getattr(model, "_COREML_/weight/quantization_n_bits")
    scale = getattr(model, "_COREML_/weight/quantization_scale")
    quant_weight = model.weight / scale
    if hasattr(model, "_COREML_/weight/zero_point"):
        quant_weight += getattr(model, "_COREML_/weight/zero_point")
    assert (quant_weight.max() - quant_weight.min()) <= (2**n_bits - 1)
