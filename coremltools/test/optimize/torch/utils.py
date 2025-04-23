#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import contextlib
import importlib
import inspect
import io
import logging
import os
import pathlib
import pkgutil
import sys

import torch
import torch.nn.functional as F
import torch.utils.data
from packaging import version

import coremltools as ct

# region package_utils


def get_classes_in_module(module):
    """
    Get all classes defined in a module (excluding imported ones)
    """
    return {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__.startswith(module.__name__)
    }


def get_classes_recursively(module):
    """
    Recursively get all classes from a module and its submodules
    """
    classes = get_classes_in_module(module)

    if hasattr(module, "__path__"):  # Only packages have __path__
        for _, submodule_name, is_pkg in pkgutil.walk_packages(
            module.__path__, module.__name__ + "."
        ):
            try:
                submodule = importlib.import_module(submodule_name)
                classes.update(get_classes_recursively(submodule))
            except ModuleNotFoundError:
                pass  # Ignore any submodules that fail to import

    return classes


# endregion


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

# region model_size utils


def convert_to_coreml(model, example_input, minimum_deployment_target=ct.target.iOS18):
    # Trace model
    traced_model = torch.jit.trace(model, example_input)
    # Convert model
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape)],
        minimum_deployment_target=minimum_deployment_target,
    )
    return coreml_model


def get_total_params(model):
    total_params = 0
    for param in model.parameters():
        total_params += torch.numel(param)
    return total_params


def get_model_size(model, example_input):
    coreml_model = convert_to_coreml(model, example_input)

    model_path = "/tmp/model.mlpackage"
    coreml_model.save(model_path)

    model_size_bytes = os.path.getsize(
        os.path.join(model_path, "Data/com.apple.CoreML/weights/weight.bin")
    )
    return model_size_bytes


# endregion

# region misc utils


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    eps = 1e-5
    eps2 = 1e-10
    return 20 * torch.log10((torch.max(torch.abs(target)) + eps) / (torch.sqrt(mse) + eps2))


# endregion
