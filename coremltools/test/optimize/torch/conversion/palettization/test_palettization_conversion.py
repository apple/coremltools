#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed, by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import pytest
import torch
import torch.nn as nn

import coremltools as ct
import coremltools.test.optimize.torch.conversion.conversion_utils as util
from coremltools.optimize.torch.palettization import (
    DKMPalettizer,
    DKMPalettizerConfig,
    PostTrainingPalettizer,
    PostTrainingPalettizerConfig,
    SKMPalettizer,
    SKMPalettizerConfig,
)
from coremltools.test.optimize.torch.utils import count_unique_params

ct = pytest.importorskip("coremltools")
cto = pytest.importorskip("coremltools.optimize")

# region per_tensor
@pytest.mark.parametrize(
    "config, lut_shape_map",
    [
        # Exclude testing for 8/6 bits since all ops in MNIST get skipped for 8/6-bit palettization.
        pytest.param(
            {
                "global_config": {"n_bits": 4, "granularity": "per_tensor"},
            },
            {
                "conv1": (1, 1, 1, 1, 16, 1),
                "conv2": (1, 1, 1, 1, 16, 1),
                "dense1": (1, 1, 16, 1),
                "dense2": (1, 1, 16, 1),
            },
            id="4bits",
        ),
        pytest.param(
            {
                "global_config": {"n_bits": 2, "granularity": "per_tensor"},
            },
            {
                "conv1": (1, 1, 1, 1, 4, 1),
                "conv2": (1, 1, 1, 1, 4, 1),
                "dense1": (1, 1, 4, 1),
                "dense2": (1, 1, 4, 1),
            },
            id="2bits",
        ),
    ],
)
@pytest.mark.parametrize("algorithm", ["SKM", "PTP", "DKM"])
def test_palettization_per_tensor(
    mnist_model,
    mnist_example_input,
    mnist_example_output,
    config,
    lut_shape_map,
    algorithm,
):
    if algorithm == "DKM":
        # Skip compressing all layers for DKM to reduce test time
        config["module_name_configs"] = {"conv1": None, "dense1": None}

    compressed_model = get_compressed_model(
        algorithm, mnist_model, mnist_example_input, mnist_example_output, config
    )

    weight_sample = compressed_model.conv2.weight.detach()  # per tensor

    # Validate on torch model.
    _n_bits = config["global_config"]["n_bits"]
    max_unique_values = 2**_n_bits
    assert count_unique_params(torch.unique(weight_sample)) <= max_unique_values

    if ct.utils._macos_version() < (15, 0):
        return

    # Convert and validate on coreml model.
    compressed_model_coreml = util.convert_and_verify(
        compressed_model,
        mnist_example_input,
        expected_ops=["constexpr_lut_to_dense"],
    )
    verify_op_constexpr_lut_to_dense(compressed_model_coreml, lut_shape_map)
# endregion


# region per_channel_scale
@pytest.mark.parametrize(
    "config, lut_shape_map",
    [
        # Exclude testing for 8/6 bits since all ops in MNIST get skipped for 8/6-bit palettization.
        pytest.param(
            {
                "global_config": {
                    "n_bits": 4,
                    "granularity": "per_grouped_channel",
                    "group_size": 1,
                    "enable_per_channel_scale": True,
                },
            },
            {
                "conv1": (32, 1, 1, 1, 16, 1),
                "conv2": (64, 1, 1, 1, 16, 1),
                "dense1": (1024, 1, 16, 1),
                "dense2": (10, 1, 16, 1),
            },
            id="4bits",
        ),
        pytest.param(
            {
                "global_config": {
                    "n_bits": 2,
                    "granularity": "per_grouped_channel",
                    "group_size": 1,
                    "enable_per_channel_scale": True,
                },
            },
            {
                "conv1": (32, 1, 1, 1, 4, 1),
                "conv2": (64, 1, 1, 1, 4, 1),
                "dense1": (1024, 1, 4, 1),
                "dense2": (10, 1, 4, 1),
            },
            id="2bits",
        ),
    ],
)
@pytest.mark.parametrize("algorithm", ["SKM", "PTP", "DKM"])
def test_palettization_per_channel_scale(
    mnist_model,
    mnist_example_input,
    mnist_example_output,
    config,
    lut_shape_map,
    algorithm,
):
    if algorithm == "DKM":
        # Skip compressing all layers for DKM to reduce test time
        config["module_name_configs"] = {"conv1": None, "dense1": None}

    compressed_model = get_compressed_model(
        algorithm, mnist_model, mnist_example_input, mnist_example_output, config
    )

    # Validate on torch model.
    for i in range(32):
        weight_sample = compressed_model.conv2.weight[i].detach()  # per channel
        _n_bits = config["global_config"]["n_bits"]
        max_unique_values = 2**_n_bits
        assert count_unique_params(torch.unique(weight_sample)) <= max_unique_values

    if ct.utils._macos_version() < (15, 0):
        return

    compressed_model_coreml = util.convert_and_verify(
        compressed_model,
        mnist_example_input,
        expected_ops=["constexpr_lut_to_dense"],
    )
    verify_op_constexpr_lut_to_dense(compressed_model_coreml, lut_shape_map)
# endregion


# region grouped_channelwise
@pytest.mark.parametrize(
    "config, lut_shape_map",
    [
        pytest.param(
            {
                "global_config": {
                    "n_bits": 4,
                    "granularity": "per_grouped_channel",
                    "group_size": 16,
                    "channel_axis": 0,
                },
            },
            {
                "conv1": (2, 1, 1, 1, 16, 1),
                "conv2": (4, 1, 1, 1, 16, 1),
                "dense1": (64, 1, 16, 1),
            },
            id="4bits_group_size_16_axis_0",
        ),
        pytest.param(
            {
                "global_config": {
                    "n_bits": 4,
                    "granularity": "per_grouped_channel",
                    "group_size": 16,
                    "channel_axis": 1,
                },
            },
            {
                "conv2": (1, 2, 1, 1, 16, 1),
                "dense1": (1, 196, 16, 1),
                "dense2": (1, 64, 16, 1),
            },
            id="4bits_group_size_16_axis_1",
        ),
    ],
)
@pytest.mark.parametrize("algorithm", ["SKM", "PTP", "DKM"])
def test_palettization_grouped_channelwise(
    mnist_model,
    mnist_example_input,
    mnist_example_output,
    config,
    lut_shape_map,
    algorithm,
):
    if algorithm == "DKM":
        # DKM API currently does not support channel_axis. by default axis is 0
        if config["global_config"]["channel_axis"] == 1:
            # skip test
            return
        else:
            # remove channel_axis key, which will default to axis 0
            del config["global_config"]["channel_axis"]
            # Skip compressing all layers for DKM to reduce test time
            config["module_name_configs"] = {"conv1": None, "dense1": None}

    compressed_model = get_compressed_model(
        algorithm, mnist_model, mnist_example_input, mnist_example_output, config
    )

    # Validate on torch model.
    _group_size = config["global_config"]["group_size"]
    _axis = config["global_config"]["channel_axis"] if algorithm != "DKM" else 0

    for i in range(0, _group_size, 32):
        if _axis == 1:
            # blocking along input channel axis
            weight_sample = compressed_model.conv2.weight[:, i : i + _group_size].detach()
        else:
            # blocking along output channel axis
            weight_sample = compressed_model.conv2.weight[i : i + _group_size].detach()
        _n_bits = config["global_config"]["n_bits"]
        max_unique_values = 2**_n_bits
        assert count_unique_params(torch.unique(weight_sample)) <= max_unique_values

    if ct.utils._macos_version() < (15, 0):
        return

    compressed_model_coreml = util.convert_and_verify(
        compressed_model,
        mnist_example_input,
        expected_ops=["constexpr_lut_to_dense"],
    )
    verify_op_constexpr_lut_to_dense(compressed_model_coreml, lut_shape_map)


# endregion


# region vector
@pytest.mark.parametrize(
    "config, lut_shape_map",
    [
        pytest.param(
            {
                "module_name_configs": {
                    "conv2": {
                        "n_bits": 4,
                        "granularity": "per_tensor",
                        "cluster_dim": 4,
                    }
                },
            },
            {
                "conv2": (1, 1, 1, 1, 16, 4),
            },
            marks=pytest.mark.xfail(
                reason="rdar://124474258 ([Compression] Support Vector Palettization in coremltools)"
            ),
            id="4bits_vector_4",
        ),
    ],
)
@pytest.mark.parametrize("algorithm", ["SKM", "PTP", "DKM"])
def test_palettization_vector(
    mnist_model,
    mnist_example_input,
    mnist_example_output,
    config,
    lut_shape_map,
    algorithm,
):
    compressed_model = get_compressed_model(
        algorithm, mnist_model, mnist_example_input, mnist_example_output, config
    )

    # Validate on torch model.
    _cluster_dim = config["module_name_configs"]["conv2"]["cluster_dim"]
    weight_sample = (
        compressed_model.conv2.weight.flatten(1).transpose(0, 1).reshape(-1, _cluster_dim)
    )

    _n_bits = config["module_name_configs"]["conv2"]["n_bits"]
    max_unique_values = 2**_n_bits
    assert len(torch.unique(weight_sample, dim=0)) <= max_unique_values

    # test compression metadata is available
    assert getattr(compressed_model.conv2, "_COREML_/weight/vector_axis") == torch.tensor(0)

    if ct.utils._macos_version() < (15, 0):
        return

    compressed_model_coreml = util.convert_and_verify(
        compressed_model,
        mnist_example_input,
        expected_ops=["constexpr_lut_to_dense"],
    )
    verify_op_constexpr_lut_to_dense(compressed_model_coreml, lut_shape_map)


# endregion


@pytest.mark.parametrize(
    "config, lut_shape_map",
    [
        pytest.param(
            {
                "global_config": {
                    "n_bits": 4,
                    "granularity": "per_tensor",
                },
            },
            {
                "conv1": (1, 1, 1, 1, 16, 1),
                "conv2": (1, 1, 1, 1, 16, 1),
                "dense1": (1, 1, 16, 1),
                "dense2": (1, 1, 16, 1),
            },
            id="4bits_per_tensor",
        ),
        pytest.param(
            {
                "global_config": {
                    "n_bits": 4,
                    "granularity": "per_grouped_channel",
                    "group_size": 16,
                },
            },
            {
                "conv1": (2, 1, 1, 1, 16, 1),
                "conv2": (4, 1, 1, 1, 16, 1),
                "dense1": (64, 1, 16, 1),
            },
            id="4bits_group_size_16_axis_0",
        ),
    ],
)
@pytest.mark.parametrize("lut_dtype", ["int8", "uint8"])
@pytest.mark.parametrize("algorithm", ["SKM", "PTP", "DKM"])
@pytest.mark.xfail(
    reason="rdar://126355261 ([Compression] Support LUT with 8bit values Model Conversion)",
)
def test_palettization_int8_lut(
    mnist_model,
    mnist_example_input,
    mnist_example_output,
    config,
    lut_shape_map,
    lut_dtype,
    algorithm,
):
    config["global_config"]["lut_dtype"] = lut_dtype
    if algorithm == "DKM":
        # Skip compressing all layers for DKM to reduce test time
        config["module_name_configs"] = {"conv1": None, "dense1": None}

    compressed_model = get_compressed_model(
        algorithm, mnist_model, mnist_example_input, mnist_example_output, config
    )

    compressed_model_coreml = util.convert_and_verify(
        compressed_model,
        mnist_example_input,
        expected_ops=["constexpr_lut_to_dense"],
    )
    verify_op_constexpr_lut_to_dense(compressed_model_coreml, lut_shape_map)


# endregion


# region HelperMethods


def get_compressed_model(algorithm, mnist_model, mnist_example_input, mnist_example_output, config):
    if algorithm == "DKM":
        return get_compressed_model_for_dkm(mnist_model, mnist_example_input, config)
    elif algorithm == "SKM":
        return get_compressed_model_for_skm(
            mnist_model, mnist_example_input, mnist_example_output, config
        )
    elif algorithm == "PTP":
        return get_compressed_model_for_ptp(mnist_model, config)
    else:
        print("Unsupported compression algorithm: ", algorithm)
        return None


# Get a compressed MNIST model with DKMPalettizer and sample data.
def get_compressed_model_for_dkm(mnist_model, mnist_example_input, config):
    palettizer_config = DKMPalettizerConfig.from_dict(config)
    palettizer = DKMPalettizer(mnist_model, palettizer_config)

    prepared_model = palettizer.prepare(inplace=True)
    palettizer.step()
    prepared_model(mnist_example_input)
    model = palettizer.finalize()
    return model


# Get a compressed MNIST model with SKMPalettizer and calibration data.
def get_compressed_model_for_skm(mnist_model, mnist_example_input, mnist_example_output, config):
    palettizer_config = SKMPalettizerConfig.from_dict(config)

    def calibration_loader():
        yield mnist_example_input

    def loss_fn(mnist_model, mnist_example_input):
        out = mnist_model(mnist_example_input)
        return nn.functional.mse_loss(out, mnist_example_output)

    compressor = SKMPalettizer(mnist_model, palettizer_config)
    compressed_model = compressor.compress(dataloader=calibration_loader(), loss_fn=loss_fn)
    return compressed_model


# Get a compressed MNIST model with PostTrainingPalettization
def get_compressed_model_for_ptp(mnist_model, config):
    palettizer_config = PostTrainingPalettizerConfig.from_dict(config)
    compressor = PostTrainingPalettizer(mnist_model, palettizer_config)
    compressed_model = compressor.compress()
    return compressed_model


def verify_op_constexpr_lut_to_dense(coreml_model, per_layer_lut_shape):
    compressed_ops = coreml_model._mil_program.functions["main"].find_ops(
        op_type="constexpr_lut_to_dense"
    )
    assert len(compressed_ops) >= 1

    # Verify if number of bits is correct.
    # For palettization, it's hidden in the shape of LUT.
    for compressed_op in compressed_ops:
        layer_name = compressed_op.name.split("_weight")[0]
        assert compressed_op.lut.shape == per_layer_lut_shape[layer_name]

# endregion
