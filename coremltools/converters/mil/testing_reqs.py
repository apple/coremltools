#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import os
from typing import List

import numpy as np
import pytest
from attrs import define, field, validators

import coremltools as ct
from coremltools._deps import _HAS_TF_1, _HAS_TF_2, _HAS_TORCH
from coremltools.converters.mil.testing_utils import macos_compatible_with_deployment_target

# Setting up backend / precision / op version
_SUPPORTED_BACKENDS = ("neuralnetwork", "mlprogram")
_SUPPORTED_PRECISIONS = ("fp32", "fp16")
_SUPPORTED_OPSET_VERSIONS_NN = (ct.target.iOS14,)
_SUPPORTED_OPSET_VERSIONS_MLPROGRAM = (
    ct.target.iOS15,
    ct.target.iOS16,
    ct.target.iOS17,
    ct.target.iOS18,
)


@define(frozen=True)
class BackendConfig:
    """
    Parameters
    ----------
    backend: str
        "neuralnetwork" or "mlprogram"
    precision: str
        "fp16" or "fp32"
    opset_version: ct.target
        minimum_deployment_target for the ct.convert function
    """
    backend: str = field(validator=validators.instance_of(str))
    precision: str = field(validator=validators.instance_of(str))
    opset_version: ct.target = field(validator=validators.instance_of(ct.target))

    @backend.validator
    def check_backend(self, attr, backend):
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend {backend} not supported. Please pass one of the following values: {_SUPPORTED_BACKENDS}"
            )

    @precision.validator
    def check_precision(self, attr, precision):
        if precision not in _SUPPORTED_PRECISIONS:
            raise ValueError(
                f"precision {precision} not supported. Please pass one of the following values: {_SUPPORTED_PRECISIONS}"
            )
        if precision == "fp16" and self.backend == "neuralnetwork":
            raise ValueError("fp16 precision is only supported in mlprogram backend.")

    @opset_version.validator
    def check_opset_version(self, attr, opset_version):
        if self.backend == "neuralnetwork" and opset_version not in _SUPPORTED_OPSET_VERSIONS_NN:
            raise ValueError(
                f"opset_version {opset_version} not supported in neuralnetwork backend. Supported opset versions are {_SUPPORTED_OPSET_VERSIONS_NN}"
            )
        if self.backend == "mlprogram" and opset_version not in _SUPPORTED_OPSET_VERSIONS_MLPROGRAM:
            raise ValueError(
                f"opset_version {opset_version} not supported in mlprogram backend. Supported opset versions are {_SUPPORTED_OPSET_VERSIONS_MLPROGRAM}"
            )

if 'PYMIL_TEST_TARGETS' in os.environ:
    targets = os.environ['PYMIL_TEST_TARGETS'].split(',')
    for i in range(len(targets)):
        targets[i] = targets[i].strip()
else:
    targets = ["mlprogram", "neuralnetwork"]

# new backends using the new infrastructure
backends_internal = []
if "mlprogram" in targets:
    for v in _SUPPORTED_OPSET_VERSIONS_MLPROGRAM:
        precisions = ["fp16"]
        if os.getenv('INCLUDE_MIL_FP32_UNIT_TESTS') == '1':
            precisions.append("fp32")
        for p in precisions:
            backends_internal.append(
                BackendConfig(backend="mlprogram", precision=p, opset_version=v)
            )

if "neuralnetwork" in targets:
    for v in _SUPPORTED_OPSET_VERSIONS_NN:
        backends_internal.append(
            BackendConfig(
                backend="neuralnetwork",
                precision="fp32",
                opset_version=v,
            )
        )

# old backends approach
backends = []
if "mlprogram" in targets:
    backends.append(("mlprogram", "fp16"))
    if os.getenv("INCLUDE_MIL_FP32_UNIT_TESTS") == "1":
        backends.append(("mlprogram", "fp32"))
if "neuralnetwork" in targets:
    backends.append(("neuralnetwork", "fp32"))

if not backends or not backends_internal:
    raise ValueError("PYMIL_TEST_TARGETS can be set to one or more of: neuralnetwork, mlprogram")


def clean_up_backends(
    backends: List[BackendConfig],
    minimum_opset_version: ct.target,
    force_include_iOS15_test: bool = False,
) -> List[BackendConfig]:
    """
    Given a list of BackendConfig objects, this utility function filters out the invalid elements.

    For instance, given a list of configs with opset_versions range from iOS14 to iOS17, with minimum_opset_version set to iOS16 and environment variable `RUN_BACKWARD_COMAPTIBILITY=1`, iOS14/iOS15 configs are removed, and iOS16/iOS17 configs are preserved.

    To be more specific, the config is removed if one of the following conditions is matched:
    1. If opset_version is not compatible with the macOS.
    2. If opset_version < minimum_opset_version
    3. For the non backward compatibility run, opset_version > minimum_opset_version

    Note a corner case that when `force_include_iOS15_test=True`, the iOS15 configs are forced to be preserved.
    """
    test_all_opset_versions = os.getenv("RUN_BACKWARD_COMPATIBILITY") == "1"
    res = []
    for config in backends:
        # First check if the macOS are able to run the test
        if not macos_compatible_with_deployment_target(config.opset_version):
            continue
        if force_include_iOS15_test and config.opset_version == ct.target.iOS15:
            res.append(config)
            continue
        if config.opset_version < minimum_opset_version:
            continue
        if not test_all_opset_versions and config.opset_version > minimum_opset_version:
            continue
        res.append(config)

    if len(res) == 0:
        pytest.skip(
            f"Tests are not runnable under {minimum_opset_version.name}.", allow_module_level=True
        )

    return res

# Setting up compute unit
compute_units = []
if "COMPUTE_UNITS" in os.environ:
    for cur_str_val in os.environ["COMPUTE_UNITS"].split(","):
        cur_str_val = cur_str_val.strip().upper()
        if cur_str_val not in ct.ComputeUnit.__members__:
            raise ValueError("Compute unit \"{}\" not supported in coremltools.".format(cur_str_val))
        compute_units.append(ct.ComputeUnit[cur_str_val])
else:
    compute_units = [ct.ComputeUnit.CPU_ONLY]

np.random.seed(1984)

if _HAS_TF_1:
    tf = pytest.importorskip("tensorflow")
    tf.compat.v1.set_random_seed(1234)

if _HAS_TF_2:
    tf = pytest.importorskip("tensorflow")
    tf.random.set_seed(1234)

if _HAS_TORCH:
    torch = pytest.importorskip("torch")
