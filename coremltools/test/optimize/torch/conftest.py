#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os
import shutil
from coremltools.test.optimize.torch.models.mnist import (
    mnist_dataset,
    mnist_model,
    mnist_model_large,
    mnist_model_quantization,
)
from coremltools.test.optimize.torch.pruning.pruning_utils import get_model_and_pruner

import pytest


# dummy function to use the imported fixtures so that linter
# does not remove them as unused imports
def _dummy(
    mnist_dataset,
    mnist_model,
    mnist_model_large,
    mnist_model_quantization,
    get_model_and_pruner,
):
    return (
        mnist_dataset,
        mnist_model,
        mnist_model_large,
        mnist_model_quantization,
        get_model_and_pruner,
    )


def _datadir(request):
    # When using this fixture with parametrized tests, we end up with '[' and ']' characters in the pathname, which TF
    # is not happy with. Thus we should substitute these characters with a more universally accepted path character.
    safe_name = request.node.name.replace("[", "___").replace("]", "___")

    dir = test_data_path() / safe_name  # noqa: F821
    shutil.rmtree(str(dir), ignore_errors=True)
    os.makedirs(str(dir))
    return dir


@pytest.fixture
def datadir(request):
    """
    Directory for storing test data for latter inspection.
    """
    return _datadir(request)
