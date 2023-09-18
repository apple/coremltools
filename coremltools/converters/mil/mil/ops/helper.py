#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Dict, Type

from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil.operation import Operation


def _get_version_of_op(
    op_variants: Dict[AvailableTarget, Type[Operation]], opset_version: AvailableTarget
) -> Type[Operation]:
    """
    A utility function that retrieves an op cls given a dictionary of op variants and target version
    """
    assert isinstance(op_variants, dict)
    opset_versions = list(op_variants.keys())
    opset_versions.sort()
    if opset_version is None:
        op_cls = op_variants[opset_versions[0]]
    elif opset_version > opset_versions[-1] and opset_version > AvailableTarget.iOS17:
        # TODO(rdar://111114658): Remove when no longer required.
        # Inherit ops from the latest opset by default.
        op_cls = op_variants[opset_versions[-1]]
    else:
        if opset_version not in op_variants:
            op_type = list(op_variants.values())[0].__name__
            msg = (
                "No available version for {} in the coremltools.target.{} opset. Please update the "
                "minimum_deployment_target to at least coremltools.target.{}"
            ).format(op_type, opset_version.name, opset_versions[0].name)
            raise ValueError(msg)
        op_cls = op_variants[opset_version]
    return op_cls
