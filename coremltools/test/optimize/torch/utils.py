#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pathlib
import sys

from packaging.version import Version


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
            raise Exception("Unable to determine the macOS version")
    return ()


def version_ge(module, target_version):
    """
    Example usage:
    >>> import torch # v1.5.0
    >>> version_ge(torch, '1.6.0') # False
    """
    return Version(module.__version__) >= Version(target_version)


def version_lt(module, target_version):
    """See version_ge"""
    return Version(module.__version__) < Version(target_version)


def test_data_path():
    return pathlib.Path(__file__).parent.absolute() / "_test_data"
