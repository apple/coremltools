#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict

import pytest
import torch.nn as nn

from coremltools.optimize.torch._utils.metadata_utils import CompressionMetadata, CompressionType
from coremltools.optimize.torch.pruning import MagnitudePruner, MagnitudePrunerConfig


@pytest.mark.parametrize(
    "algorithm, config",
    [
        (MagnitudePruner, MagnitudePrunerConfig()),
    ],
)
def test_compression_metadata(algorithm, config):
    """
    Test that calling finalize on the module leads to compression metadata being added to the model
    """
    model = nn.Sequential(
        OrderedDict([("conv1", nn.Conv2d(3, 32, 3)), ("fc1", nn.Linear(32, 100))])
    )
    # Disable compression for Linear layer
    config = config.set_module_name("fc1", None)
    pruner = algorithm(model, config)
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
