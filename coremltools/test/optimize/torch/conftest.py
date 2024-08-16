#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import os
import shutil

import pytest

from coremltools.test.optimize.torch.models.mnist import (
    mnist_dataset,
    mnist_example_input,
    mnist_example_output,
    mnist_model,
    mnist_model_conv_transpose,
    mnist_model_large,
    mnist_model_quantization,
    residual_mnist_model,
)
from coremltools.test.optimize.torch.pruning.pruning_utils import get_model_and_pruner


# dummy function to use the imported fixtures so that linter
# does not remove them as unused imports
def _dummy(
    mnist_dataset,
    mnist_example_input,
    mnist_example_output,
    mnist_model,
    residual_mnist_model,
    mnist_model_large,
    mnist_model_quantization,
    get_model_and_pruner,
    mnist_model_conv_transpose,
):
    return (
        mnist_dataset,
        mnist_example_input,
        mnist_example_output,
        mnist_model,
        residual_mnist_model,
        mnist_model_large,
        mnist_model_quantization,
        get_model_and_pruner,
        mnist_model_conv_transpose,
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


@pytest.fixture
def mock_name_main(monkeypatch):
    monkeypatch.setattr(__import__("__main__"), "__name__", "__main__")


def pytest_addoption(parser):
    """
    Adds command line option --runopt to the pytest parser
    By default, evaluates to False.
    If command line option passed, evaluates to True
    """

    parser.addoption("--runopt", action="store_true", default=False, help="run optional tests")


def pytest_configure(config):
    """
    Adds info about optional marker to pytest config
    """
    config.addinivalue_line("markers", "optional: mark test run as optional")


def marker_names(item):
    """
    Returns set containing the name of each marker associated with
    the given test item
    """
    return set(m.name for m in item.iter_markers())


def pytest_collection_modifyitems(config, items):
    """
    Modifies the test items so that items marked optional are skipped
    when the --runopt command line option is not provided.
    Otherwise, will not perform any modifications.
    """

    # No modifications required
    if config.getoption("--runopt"):
        return

    skip_opt = pytest.mark.skip(reason="need --runopt option to run")

    for item in items:
        markers = marker_names(item)
        if "optional" in markers:
            item.add_marker(skip_opt)
