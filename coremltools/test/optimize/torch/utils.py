#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import contextlib
import io
import logging
import pathlib
import sys

import torch
import torch.nn.functional as F
import torch.utils.data
from packaging import version


# region version_utils
def _python_version():
    """
    Return python version as a tuple of integers
    """
    version = sys.version.split(" ")[0]
    version = list(map(int, list(version.split("."))))
    return tuple(version)


def _macos_version():
    """
    Returns macOS version as a tuple of integers, making it easy to do proper
    version comparisons. On non-Macs, it returns an empty tuple.
    """
    if sys.platform == "darwin":
        try:
            import subprocess

            ver_str = (
                subprocess.run(["sw_vers", "-productVersion"], stdout=subprocess.PIPE)
                .stdout.decode("utf-8")
                .strip("\n")
            )
            return tuple([int(v) for v in ver_str.split(".")])
        except:
            raise Exception("Unable to detemine the macOS version")
    return ()


def count_unique_params(tensor):
    """
    Returns number of unique parameters in the same tensor.
    Set a defaulted absolute tolerance, so that very close values can be treated as identical in palletization.
    """
    unique_set = {tensor[0]}
    for elem in tensor[1:]:
        if all(not torch.isclose(elem, uelem, atol=1e-6) for uelem in unique_set):
            unique_set.add(elem)
    return len(unique_set)


def version_ge(module, target_version):
    """
    Example usage:
    >>> import torch # v1.5.0
    >>> version_ge(torch, '1.6.0') # False
    """
    return version.parse(module.__version__) >= version.parse(target_version)


def version_lt(module, target_version):
    """See version_ge"""
    return version.parse(module.__version__) < version.parse(target_version)


# endregion


# region path_utils
def test_data_path():
    return pathlib.Path(__file__).parent.absolute() / "_test_data"


# endregion

# region train_utils


def setup_data_loaders(dataset, batch_size):
    train, test = dataset
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    return train_loader, test_loader


def train_step(model, optimizer, train_loader, data, target, batch_idx, epoch):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 100 == 0:
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )
    return loss


def eval_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    accuracy = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)

        print("\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n".format(test_loss, accuracy))
    return accuracy


def get_logging_capture_context_manager():
    @contextlib.contextmanager
    def capture_logs(logger_name):
        # Create a StringIO object to capture the log output
        log_capture = io.StringIO()

        # Get the logger
        logger = logging.getLogger(logger_name)

        # Save the current handlers
        original_handlers = logger.handlers

        # Create a custom logging handler that writes to the StringIO object
        string_io_handler = logging.StreamHandler(log_capture)
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        string_io_handler.setFormatter(formatter)

        # Clear existing handlers and add the custom handler
        logger.handlers = [string_io_handler]

        # Capture the logs
        try:
            yield log_capture
        finally:
            # Restore original handlers
            logger.handlers = original_handlers

    return capture_logs


# endregion
