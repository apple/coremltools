#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from coremltools.optimize.torch.palettization import (
    DKMPalettizer,
    DKMPalettizerConfig,
    ModuleDKMPalettizerConfig,
)
from coremltools.optimize.torch.palettization.palettization_config import (
    DEFAULT_PALETTIZATION_SCHEME,
)
from coremltools.test.optimize.torch.palettization.palettization_utils import (
    _assert_changes_post_attach,
    _assert_changes_post_prepare,
)
from coremltools.test.optimize.torch.utils import get_logging_capture_context_manager

REGEX_YAML = """
module_name_configs:
  conv\d+:
    - n_bits: 4
      weight_threshold: 400
      palett_tau: 0.000004
    - n_bits: 2
      weight_threshold: 1000
      palett_tau: 0.000004
"""


def _create_simple_model():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    return Net()


@pytest.fixture
def simple_model():
    return _create_simple_model()


def test_inplace_false_attach_config(simple_model):
    palettizer = DKMPalettizer(simple_model)
    prepared_model = palettizer.prepare()

    assert not hasattr(simple_model.conv1, "qconfig")
    assert not hasattr(simple_model.conv2, "qconfig")
    assert not hasattr(simple_model.fc1, "qconfig")
    assert not hasattr(simple_model.fc2, "qconfig")
    assert not hasattr(simple_model.fc3, "qconfig")

    _assert_changes_post_attach(
        prepared_model.conv2,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.conv2)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.conv2)]["cluster_dim"],
    )
    _assert_changes_post_attach(
        prepared_model.fc1,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc1)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc1)]["cluster_dim"],
    )
    _assert_changes_post_attach(
        prepared_model.fc2,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc2)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc2)]["cluster_dim"],
    )


def test_empty_dict_for_config(simple_model):
    ## This test should behave the same as that when a None config is passed to DKMPalettizer
    config = DKMPalettizerConfig.from_dict({})
    palettizer = DKMPalettizer(simple_model, config)
    prepared_model = palettizer.prepare()

    assert not hasattr(simple_model.conv1, "qconfig")
    assert not hasattr(simple_model.conv2, "qconfig")
    assert not hasattr(simple_model.fc1, "qconfig")
    assert not hasattr(simple_model.fc2, "qconfig")
    assert not hasattr(simple_model.fc3, "qconfig")

    _assert_changes_post_attach(
        prepared_model.conv2,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.conv2)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.conv2)]["cluster_dim"],
    )
    _assert_changes_post_attach(
        prepared_model.fc1,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc1)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc1)]["cluster_dim"],
    )
    _assert_changes_post_attach(
        prepared_model.fc2,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc2)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc2)]["cluster_dim"],
    )


@pytest.fixture(scope="session")
def test_empty_yaml_for_config(simple_model, tmp_path_factory):
    ## This test should behave the same as that when a None config is passed to DKMPalettizer
    fname = tmp_path_factory.mktemp("test_configs") / "empty_config.yaml"
    with open(fname, "w") as file:
        file.write("\n")
    config = DKMPalettizerConfig.from_yaml(fname)
    palettizer = DKMPalettizer(simple_model, config)
    prepared_model = palettizer.prepare()

    assert not hasattr(simple_model.conv1, "qconfig")
    assert not hasattr(simple_model.conv2, "qconfig")
    assert not hasattr(simple_model.fc1, "qconfig")
    assert not hasattr(simple_model.fc2, "qconfig")
    assert not hasattr(simple_model.fc3, "qconfig")

    _assert_changes_post_attach(
        prepared_model.conv2,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.conv2)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.conv2)]["cluster_dim"],
    )
    _assert_changes_post_attach(
        prepared_model.fc1,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc1)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc1)]["cluster_dim"],
    )
    _assert_changes_post_attach(
        prepared_model.fc2,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc2)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc2)]["cluster_dim"],
    )


@pytest.fixture(scope="session")
def test_regex_module_name_configs(simple_model, tmp_path_factory):
    fname = tmp_path_factory.mktemp("test_configs") / "regex_config.yaml"
    with open(fname, "w") as file:
        file.write(REGEX_YAML)
    config = DKMPalettizerConfig.from_yaml(fname)
    palettizer = DKMPalettizer(simple_model, config)
    palettizer.prepare(inplace=True)

    assert hasattr(simple_model.fc1, "qconfig") and simple_model.fc1.qconfig is None
    _assert_changes_post_attach(simple_model.conv1, 4, 1)
    _assert_changes_post_attach(simple_model.conv2, 2, 1)


def test_attach_config_simple_model_uniform_palettization_config(simple_model):
    config = DKMPalettizerConfig.from_dict({"global_config": {"n_bits": 4}})
    palettizer = DKMPalettizer(simple_model, config)
    palettizer.prepare(inplace=True)

    n_bits = config.global_config.n_bits
    _assert_changes_post_attach(simple_model.conv2, n_bits, 1)
    _assert_changes_post_attach(simple_model.fc1, n_bits, 1)
    _assert_changes_post_attach(simple_model.fc2, n_bits, 1)


def test_attach_config_simple_model_custom_palettization_config(simple_model):
    custom_config = {
        nn.Conv2d: {"n_bits": 2, "cluster_dim": 2},
        nn.Linear: {"n_bits": 4},
    }
    config = DKMPalettizerConfig.from_dict(
        {"module_type_configs": custom_config,
         "module_name_configs": {'conv2': {"n_bits": 3, "cluster_dim": 2}}}
    )
    palettizer = DKMPalettizer(simple_model, config)
    palettizer.prepare(inplace=True)

    _assert_changes_post_attach(simple_model.conv2, 3, 2)
    _assert_changes_post_attach(simple_model.fc1, custom_config[nn.Linear]["n_bits"], 1)
    _assert_changes_post_attach(simple_model.fc2, custom_config[nn.Linear]["n_bits"], 1)


def test_attach_config_simple_model_weight_threshold_test(simple_model):
    custom_config = {nn.Conv2d: {"n_bits": 2, "cluster_dim": 2, "weight_threshold": 1000}}
    config = DKMPalettizerConfig.from_dict(
        {"module_type_configs": custom_config}
    )
    palettizer = DKMPalettizer(simple_model, config)
    palettizer.prepare(inplace=True)

    # For the below two assertions, prepare_qat would propagate a None qconfig throughout the supported modules in
    # the model
    assert hasattr(simple_model.conv1, "qconfig") and simple_model.conv1.qconfig is None
    assert hasattr(simple_model.fc1, "qconfig") and simple_model.fc1.qconfig is None
    _assert_changes_post_attach(
        simple_model.conv2,
        custom_config[nn.Conv2d]["n_bits"],
        custom_config[nn.Conv2d]["cluster_dim"],
    )


def test_attach_config_simple_model_weight_threshold_range_test(simple_model):
    custom_config = {
        nn.Conv2d: [
            {"n_bits": 4, "cluster_dim": 1, "weight_threshold": 1000},
            {"n_bits": 2, "cluster_dim": 1, "weight_threshold": 400},
        ]
    }
    config = DKMPalettizerConfig.from_dict({"module_type_configs": custom_config})
    palettizer = DKMPalettizer(simple_model, config)
    palettizer.prepare(inplace=True)

    # For the below assertion, prepare_qat would propagate a None qconfig throughout the supported modules in
    # the model
    assert hasattr(simple_model.fc1, "qconfig") and simple_model.fc1.qconfig is None
    _assert_changes_post_attach(
        simple_model.conv1,
        custom_config[nn.Conv2d][1]["n_bits"],
        custom_config[nn.Conv2d][1]["cluster_dim"],
    )
    _assert_changes_post_attach(
        simple_model.conv2,
        custom_config[nn.Conv2d][0]["n_bits"],
        custom_config[nn.Conv2d][0]["cluster_dim"],
    )


def test_attach_config_only_on_specified_modules_conv(simple_model):
    """
    If there is a module type specified in the palettization_config, qconfigs should only be applied to modules of
    those types not to modules of other type. For eg: If palettization_config only contains Conv2d, we should
    not attach a qconfig to nn.Linear despite it being supported by palettization.
    """
    custom_config = {nn.Conv2d: {"n_bits": 2, "cluster_dim": 2}}
    config = DKMPalettizerConfig.from_dict({"module_type_configs": custom_config})
    palettizer = DKMPalettizer(simple_model, config)
    palettizer.prepare(inplace=True)

    # For the below assertion, prepare_qat would propagate a None qconfig throughout the supported modules in
    # the model
    assert hasattr(simple_model.fc1, "qconfig") and simple_model.fc1.qconfig is None
    _assert_changes_post_attach(
        simple_model.conv2,
        custom_config[nn.Conv2d]["n_bits"],
        custom_config[nn.Conv2d]["cluster_dim"],
    )


def test_attach_config_only_on_specified_modules_linear(simple_model):
    custom_config = {nn.Linear: {"n_bits": 2, "cluster_dim": 2}}
    config = DKMPalettizerConfig.from_dict(
        {"module_type_configs": custom_config}
    )
    palettizer = DKMPalettizer(simple_model, config)
    palettizer.prepare(inplace=True)

    # For the below two assertions, prepare_qat would propagate a None qconfig throughout the supported modules in
    # the model
    assert hasattr(simple_model.conv1, "qconfig") and simple_model.conv1.qconfig is None
    assert hasattr(simple_model.conv2, "qconfig") and simple_model.conv2.qconfig is None
    _assert_changes_post_attach(
        simple_model.fc1,
        custom_config[nn.Linear]["n_bits"],
        custom_config[nn.Linear]["cluster_dim"],
    )


def test_prepare_palettizer_simple_model_custom_palettization_config(simple_model):
    simple_model_copy = copy.deepcopy(simple_model)
    custom_config = {nn.Conv2d: {"n_bits": 2, "cluster_dim": 2, "kmeans_max_iter": 4},
                     nn.Linear: {"n_bits": 4, "cluster_dim": 1, "kmeans_max_iter": 5}}
    config = DKMPalettizerConfig.from_dict(
        {"module_type_configs": custom_config}
    )
    palettizer = DKMPalettizer(simple_model, config)
    prepared_model = palettizer.prepare(inplace=True)
    num_epochs = 1
    for epoch in range(num_epochs):
        palettizer.step()

    _assert_changes_post_prepare(
        simple_model_copy.conv2,
        prepared_model.conv2,
        custom_config[nn.Conv2d]["n_bits"],
        custom_config[nn.Conv2d]["cluster_dim"],
        custom_config[nn.Conv2d]["kmeans_max_iter"],
    )
    _assert_changes_post_prepare(
        simple_model_copy.fc1,
        prepared_model.fc1,
        custom_config[nn.Linear]["n_bits"],
        custom_config[nn.Linear]["cluster_dim"],
        custom_config[nn.Linear]["kmeans_max_iter"],
    )
    _assert_changes_post_prepare(
        simple_model_copy.fc2,
        prepared_model.fc2,
        custom_config[nn.Linear]["n_bits"],
        custom_config[nn.Linear]["cluster_dim"],
        custom_config[nn.Linear]["kmeans_max_iter"],
    )


@pytest.mark.parametrize(
    "cluster_dim_expected_std_outputs",
    [
        (4, None),
        (
            5,
            [
                "WARNING:coremltools.optimize.torch._utils.validation_utils:conv2.weight: The number of channels in channel axis dimension: "
                "0, 16 is not divisible by cluster_dim=5"
            ],
        ),
    ],
)
def test_prepare_palettizer_simple_model_cluster_dim_mil_check(
    simple_model, cluster_dim_expected_std_outputs
):
    cluster_dim, expected_std_outputs = cluster_dim_expected_std_outputs
    custom_config = {nn.Conv2d: {"n_bits": 2, "cluster_dim": cluster_dim}}
    config = DKMPalettizerConfig.from_dict({"module_type_configs": custom_config})
    palettizer = DKMPalettizer(simple_model, config)
    logging_context_manager = get_logging_capture_context_manager()
    with logging_context_manager(
        "coremltools.optimize.torch._utils.validation_utils"
    ) as log_capture:
        simple_model = palettizer.prepare()
    output_capture = log_capture.getvalue()

    if expected_std_outputs:
        assert not hasattr(simple_model.conv2, "weight_fake_quant")
        for expected_std_output in expected_std_outputs:
            assert expected_std_output in output_capture
    else:
        assert hasattr(simple_model.conv2, "weight_fake_quant")


@pytest.mark.parametrize(
    "block_size_expected_std_outputs",
    [
        (
            5,
            [
                "WARNING:coremltools.optimize.torch._utils.validation_utils:conv2.weight: axis_length=16 is not divisible by group_size=5",
                "INFO:coremltools.optimize.torch._utils.validation_utils:Skipping compression for conv2.weight",
            ],
        ),
        (4, None),
    ],
)
def test_prepare_palettizer_simple_model_block_size_mil_check(
    simple_model, block_size_expected_std_outputs
):
    curr_block_size, expected_std_outputs = block_size_expected_std_outputs
    custom_config = {
        nn.Conv2d: {
            "n_bits": 2,
            "cluster_dim": 4,
            "granularity": "per_grouped_channel",
            "group_size": curr_block_size,
        }
    }
    config = DKMPalettizerConfig.from_dict({"module_type_configs": custom_config})
    palettizer = DKMPalettizer(simple_model, config)
    logging_context_manager = get_logging_capture_context_manager()
    with logging_context_manager(
        "coremltools.optimize.torch._utils.validation_utils"
    ) as log_capture:
        simple_model = palettizer.prepare()
    output_capture = log_capture.getvalue()

    if expected_std_outputs:
        assert not hasattr(simple_model.conv2, "weight_fake_quant")
        for expected_std_output in expected_std_outputs:
            assert expected_std_output in output_capture
    else:
        assert hasattr(simple_model.conv2, "weight_fake_quant")


def test_inplace_true_prepare_palettizer(simple_model):
    simple_model_copy = copy.deepcopy(simple_model)
    custom_config = {
        nn.Conv2d: {
            "n_bits": 2,
            "cluster_dim": 2,
            "kmeans_max_iter": 4,
            "milestone": 1,
        },
        nn.Linear: {
            "n_bits": 4,
            "cluster_dim": 1,
            "kmeans_max_iter": 5,
            "milestone": 1,
        },
    }
    config = DKMPalettizerConfig.from_dict({"module_type_configs": custom_config})
    palettizer = DKMPalettizer(simple_model, config)

    palettizer.prepare(inplace=True)
    num_steps = 2
    for step in range(num_steps):
        palettizer.step()
        if step == 0:
            assert palettizer._model.fc1.weight_fake_quant.fake_palett_enabled[0] == 0
        else:
            assert palettizer._model.fc1.weight_fake_quant.fake_palett_enabled[0] == 1

    _assert_changes_post_prepare(
        simple_model_copy.conv2,
        simple_model.conv2,
        custom_config[nn.Conv2d]["n_bits"],
        custom_config[nn.Conv2d]["cluster_dim"],
        custom_config[nn.Conv2d]["kmeans_max_iter"],
    )
    _assert_changes_post_prepare(
        simple_model_copy.fc1,
        simple_model.fc1,
        custom_config[nn.Linear]["n_bits"],
        custom_config[nn.Linear]["cluster_dim"],
        custom_config[nn.Linear]["kmeans_max_iter"],
    )
    _assert_changes_post_prepare(
        simple_model_copy.fc2,
        simple_model.fc2,
        custom_config[nn.Linear]["n_bits"],
        custom_config[nn.Linear]["cluster_dim"],
        custom_config[nn.Linear]["kmeans_max_iter"],
    )


def test_prepare_palettizer_simple_model_custom_palettization_config_milestone_1(simple_model):
    custom_config = {nn.Conv2d: {"n_bits": 2, "cluster_dim": 2, "kmeans_max_iter": 4, "milestone": 1},
                     nn.Linear: {"n_bits": 4, "cluster_dim": 1, "kmeans_max_iter": 5, "milestone": 1}}
    config = DKMPalettizerConfig.from_dict(
        {"module_type_configs": custom_config}
    )
    palettizer = DKMPalettizer(simple_model, config)

    prepared_model = palettizer.prepare()
    num_steps = 2
    for step in range(num_steps):
        palettizer.step()
        if step == 0:
            assert palettizer._model.fc1.weight_fake_quant.fake_palett_enabled[0] == 0
        else:
            assert palettizer._model.fc1.weight_fake_quant.fake_palett_enabled[0] == 1

    _assert_changes_post_prepare(simple_model.conv2, prepared_model.conv2, custom_config[nn.Conv2d]["n_bits"],
                                 custom_config[nn.Conv2d]["cluster_dim"], custom_config[nn.Conv2d]["kmeans_max_iter"])
    _assert_changes_post_prepare(simple_model.fc1, prepared_model.fc1, custom_config[nn.Linear]["n_bits"],
                                 custom_config[nn.Linear]["cluster_dim"], custom_config[nn.Linear]["kmeans_max_iter"])
    _assert_changes_post_prepare(simple_model.fc2, prepared_model.fc2, custom_config[nn.Linear]["n_bits"],
                                 custom_config[nn.Linear]["cluster_dim"], custom_config[nn.Linear]["kmeans_max_iter"])


def test_prepare_palettizer_different_milestone_per_module_type(simple_model):
    custom_config = {
        nn.Conv2d: {
            "n_bits": 2,
            "cluster_dim": 2,
            "kmeans_max_iter": 4,
            "milestone": 1,
        },
        nn.Linear: {"n_bits": 4, "kmeans_max_iter": 5, "milestone": 2},
    }
    config = DKMPalettizerConfig.from_dict({"module_type_configs": custom_config})
    palettizer = DKMPalettizer(simple_model, config)

    orig_conv_mods = [simple_model.conv2]
    orig_fc_mods = [simple_model.fc1, simple_model.fc2]

    prepared_model = palettizer.prepare()

    prepared_conv_mods = [prepared_model.conv2]
    prepared_fc_mods = [prepared_model.fc1, prepared_model.fc2]

    num_steps = 3
    for step in range(num_steps):
        palettizer.step()
        if step == 0:
            for mod in prepared_conv_mods + prepared_fc_mods:
                assert mod.weight_fake_quant.fake_palett_enabled[0] == 0
        elif step == 1:
            for mod in prepared_conv_mods:
                assert mod.weight_fake_quant.fake_palett_enabled[0] == 1
            for mod in prepared_fc_mods:
                assert mod.weight_fake_quant.fake_palett_enabled[0] == 0
        else:
            for mod in prepared_conv_mods:
                assert mod.weight_fake_quant.fake_palett_enabled[0] == 1
            for mod in prepared_fc_mods:
                assert mod.weight_fake_quant.fake_palett_enabled[0] == 1

    for orig, prep in zip(orig_conv_mods, prepared_conv_mods):
        _assert_changes_post_prepare(orig, prep, custom_config[nn.Conv2d]["n_bits"],
                                     custom_config[nn.Conv2d]["cluster_dim"],
                                     custom_config[nn.Conv2d]["kmeans_max_iter"])
    for orig, prep in zip(orig_fc_mods, prepared_fc_mods):
        _assert_changes_post_prepare(
            orig,
            prep,
            custom_config[nn.Linear]["n_bits"],
            1,
            custom_config[nn.Linear]["kmeans_max_iter"],
        )


def test_attach_config_weight_threshold_range_different_milestone(simple_model):
    custom_config = {
        nn.Conv2d: [
            {"n_bits": 4, "cluster_dim": 2, "weight_threshold": 1000, "milestone": 2},
            {"n_bits": 2, "cluster_dim": 1, "weight_threshold": 400, "milestone": 1},
        ]
    }
    config = DKMPalettizerConfig.from_dict({"module_type_configs": custom_config})
    palettizer = DKMPalettizer(simple_model, config)
    prepared_model = palettizer.prepare()

    # configs should get sorted automatically
    assert hasattr(prepared_model.fc1, "qconfig") and prepared_model.fc1.qconfig is None

    num_steps = 3
    for step in range(num_steps):
        palettizer.step()
        if step == 0:
            assert prepared_model.conv2.weight_fake_quant.fake_palett_enabled[0] == 0
        elif step == 1:
            assert prepared_model.conv2.weight_fake_quant.fake_palett_enabled[0] == 0
        else:
            assert prepared_model.conv2.weight_fake_quant.fake_palett_enabled[0] == 1

    _assert_changes_post_attach(
        prepared_model.conv2,
        custom_config[nn.Conv2d][0]["n_bits"],
        custom_config[nn.Conv2d][0]["cluster_dim"],
    )


def test_prepare_palettizer_simple_model_custom_palettization_config_none_module(simple_model):
    custom_config = {nn.Conv2d: {"n_bits": 2, "cluster_dim": 2, "kmeans_max_iter": 4},
                     nn.Linear: {"n_bits": 4, "cluster_dim": 1, "kmeans_max_iter": 5}}
    config = DKMPalettizerConfig.from_dict(
        {"module_name_configs": {"conv1": None}, "module_type_configs": custom_config}
    )
    palettizer = DKMPalettizer(simple_model, config)

    prepared_model = palettizer.prepare()
    num_epochs = 1
    for epoch in range(num_epochs):
        palettizer.step()
    assert type(prepared_model.conv1) == nn.Conv2d  # Means that if None was provided, it wasn't prepared.

    _assert_changes_post_prepare(simple_model.conv2, prepared_model.conv2, custom_config[nn.Conv2d]["n_bits"],
                                 custom_config[nn.Conv2d]["cluster_dim"], custom_config[nn.Conv2d]["kmeans_max_iter"])
    _assert_changes_post_prepare(simple_model.fc1, prepared_model.fc1, custom_config[nn.Linear]["n_bits"],
                                 custom_config[nn.Linear]["cluster_dim"], custom_config[nn.Linear]["kmeans_max_iter"])
    _assert_changes_post_prepare(simple_model.fc2, prepared_model.fc2, custom_config[nn.Linear]["n_bits"],
                                 custom_config[nn.Linear]["cluster_dim"], custom_config[nn.Linear]["kmeans_max_iter"])


def test_prepare_palettizer_simple_model_custom_palettization_config_none_conv2d(simple_model):
    custom_config = {nn.Conv2d: None,
                     nn.Linear: {"n_bits": 4, "cluster_dim": 1, "kmeans_max_iter": 5}}
    config = DKMPalettizerConfig.from_dict(
        {"module_type_configs": custom_config}
    )
    palettizer = DKMPalettizer(simple_model, config)

    prepared_model = palettizer.prepare()
    num_epochs = 1
    for epoch in range(num_epochs):
        palettizer.step()
    assert type(prepared_model.conv1) == nn.Conv2d  # Means that if None was provided, it wasn't prepared.
    assert type(prepared_model.conv2) == nn.Conv2d
    assert not hasattr(prepared_model.conv1, "weight_fake_quant")
    assert not hasattr(prepared_model.conv2, "weight_fake_quant")

    _assert_changes_post_prepare(simple_model.fc1, prepared_model.fc1, custom_config[nn.Linear]["n_bits"],
                                 custom_config[nn.Linear]["cluster_dim"], custom_config[nn.Linear]["kmeans_max_iter"])
    _assert_changes_post_prepare(simple_model.fc2, prepared_model.fc2, custom_config[nn.Linear]["n_bits"],
                                 custom_config[nn.Linear]["cluster_dim"], custom_config[nn.Linear]["kmeans_max_iter"])


def test_prepare_palettizer_simple_model_custom_palettization_config_linear_default(simple_model):
    custom_config = {nn.Conv2d: {"n_bits": 2, "cluster_dim": 2, "kmeans_max_iter": 4},
                     nn.Linear: {"n_bits": 4, "cluster_dim": 1}}
    config = DKMPalettizerConfig.from_dict(
        {"module_type_configs": custom_config}
    )
    palettizer = DKMPalettizer(simple_model, config)

    prepared_model = palettizer.prepare()
    num_epochs = 1
    for epoch in range(num_epochs):
        palettizer.step()

    _assert_changes_post_prepare(
        simple_model.conv2,
        prepared_model.conv2,
        custom_config[nn.Conv2d]["n_bits"],
        custom_config[nn.Conv2d]["cluster_dim"],
        custom_config[nn.Conv2d]["kmeans_max_iter"],
    )
    _assert_changes_post_prepare(
        simple_model.fc1,
        prepared_model.fc1,
        custom_config[nn.Linear]["n_bits"],
        custom_config[nn.Linear]["cluster_dim"],
        DEFAULT_PALETTIZATION_SCHEME[nn.Linear]["kmeans_max_iter"],
    )
    _assert_changes_post_prepare(
        simple_model.fc2,
        prepared_model.fc2,
        custom_config[nn.Linear]["n_bits"],
        custom_config[nn.Linear]["cluster_dim"],
        DEFAULT_PALETTIZATION_SCHEME[nn.Linear]["kmeans_max_iter"],
    )


def test_inplace_true_attach_config(simple_model):
    simple_model_copy = copy.deepcopy(simple_model)
    palettizer = DKMPalettizer(simple_model)
    palettizer.prepare(inplace=True)

    _assert_changes_post_attach(
        simple_model.conv2,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model_copy.conv2)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model_copy.conv2)]["cluster_dim"],
    )
    _assert_changes_post_attach(
        simple_model.fc1,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model_copy.fc1)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model_copy.fc1)]["cluster_dim"],
    )
    _assert_changes_post_attach(
        simple_model.fc2,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model_copy.fc2)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model_copy.fc2)]["cluster_dim"],
    )


def test_inplace_false_attach_config(simple_model):
    palettizer = DKMPalettizer(simple_model)
    prepared_model = palettizer.prepare()

    assert not hasattr(simple_model.conv1, "qconfig")
    assert not hasattr(simple_model.conv2, "qconfig")
    assert not hasattr(simple_model.fc1, "qconfig")
    assert not hasattr(simple_model.fc2, "qconfig")
    assert not hasattr(simple_model.fc3, "qconfig")

    _assert_changes_post_attach(
        prepared_model.conv2,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.conv2)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.conv2)]["cluster_dim"],
    )
    _assert_changes_post_attach(
        prepared_model.fc1,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc1)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc1)]["cluster_dim"],
    )
    _assert_changes_post_attach(
        prepared_model.fc2,
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc2)]["n_bits"],
        DEFAULT_PALETTIZATION_SCHEME[type(simple_model.fc2)]["cluster_dim"],
    )


def test_inplace_true_prepare_palettizer(simple_model):
    simple_model_copy = copy.deepcopy(simple_model)
    custom_config = {
        nn.Conv2d: {
            "n_bits": 2,
            "cluster_dim": 2,
            "kmeans_max_iter": 4,
            "milestone": 1,
        },
        nn.Linear: {
            "n_bits": 4,
            "cluster_dim": 1,
            "kmeans_max_iter": 5,
            "milestone": 1,
        },
    }
    config = DKMPalettizerConfig.from_dict({"module_type_configs": custom_config})
    palettizer = DKMPalettizer(simple_model, config)

    palettizer.prepare(inplace=True)
    num_steps = 2
    for step in range(num_steps):
        palettizer.step()
        if step == 0:
            assert palettizer._model.fc1.weight_fake_quant.fake_palett_enabled[0] == 0
        else:
            assert palettizer._model.fc1.weight_fake_quant.fake_palett_enabled[0] == 1

    _assert_changes_post_prepare(
        simple_model_copy.conv2,
        simple_model.conv2,
        custom_config[nn.Conv2d]["n_bits"],
        custom_config[nn.Conv2d]["cluster_dim"],
        custom_config[nn.Conv2d]["kmeans_max_iter"],
    )
    _assert_changes_post_prepare(
        simple_model_copy.fc1,
        simple_model.fc1,
        custom_config[nn.Linear]["n_bits"],
        custom_config[nn.Linear]["cluster_dim"],
        custom_config[nn.Linear]["kmeans_max_iter"],
    )
    _assert_changes_post_prepare(
        simple_model_copy.fc2,
        simple_model.fc2,
        custom_config[nn.Linear]["n_bits"],
        custom_config[nn.Linear]["cluster_dim"],
        custom_config[nn.Linear]["kmeans_max_iter"],
    )


def test_quantize_activations_flag(simple_model):
    config = DKMPalettizerConfig.from_dict(
        {"global_config": {"n_bits": 2, "cluster_dim": 1, "quantize_activations": True}}
    )

    palettizer = DKMPalettizer(simple_model, config)

    palettizer.prepare()
    for _ in range(3):
        palettizer.step()

    assert not isinstance(palettizer._model.conv2.activation_post_process, torch.nn.Identity)


def test_finalize_without_forward(simple_model):
    config = DKMPalettizerConfig.from_dict({"global_config": {"n_bits": 2, "cluster_dim": 1}})

    palettizer = DKMPalettizer(simple_model, config)

    prepared_model = palettizer.prepare()
    palettizer.step()
    finalized_model = palettizer.finalize(prepared_model)
    assert torch.equal(simple_model.fc2.weight, finalized_model.fc2.weight)


# Added to test rdar://130017103 (DKMPalettizerConfig incorrectly sets module_type_configs when initializing with module_name_configs)
def test_config_initialization_from_module_name_configs(simple_model):
    palettization_layer_config_dict = {
        "conv1": ModuleDKMPalettizerConfig(n_bits=5, milestone=0),
        "fc2": ModuleDKMPalettizerConfig(n_bits=3, milestone=0),
    }

    config = DKMPalettizerConfig(module_name_configs=palettization_layer_config_dict)

    assert len(config.module_type_configs) == 0
    assert config.module_name_configs == palettization_layer_config_dict


def test_deprecated_api():
    with pytest.raises(DeprecationWarning):
        config = DKMPalettizerConfig.from_dict({"global_config": {"partition_size": 100}})

    config = DKMPalettizerConfig(global_config=ModuleDKMPalettizerConfig())
    with pytest.raises(DeprecationWarning):
        config.global_config.partition_size = 100
