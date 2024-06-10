#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import torch as _torch
from packaging import version


def version_ge(module, target_version):
    return version.parse(module.__version__) >= version.parse(target_version)


def get_torch_version():
    return _torch.__version__


def is_torch_2():
    return version_ge(_torch, "2.0.0")
