#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from enum import Enum

import numpy as np
import torch

from coremltools.converters.mil.mil import types

# Some ops will receive a dtype input as an integer
# which maps to a torch dtype. The below mapping was found by
# converting test models with different dtypes passed to ones.
NUM_TO_TORCH_DTYPE = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int32,
    5: torch.float16,
    6: torch.float32,
    7: torch.float32,
    11: torch.bool,
    12: torch.qint8,
    13: torch.quint8,
    14: torch.qint32,
}

TORCH_DTYPE_TO_NUM = {
    dtype: val for val, dtype in NUM_TO_TORCH_DTYPE.items()
}

NUMPY_DTYPE_TO_TORCH_NUM = {
    np.uint8: 0,
    np.int8: 1,
    np.int16: 2,
    np.int32: 3,
    np.int64: 4,
    np.float16: 5,
    np.float32: 6,
    np.float64: 7,
    bool: 11,
}

NUM_TO_NUMPY_DTYPE = {
    val: dtype for dtype, val in NUMPY_DTYPE_TO_TORCH_NUM.items()
}

NUM_TO_DTYPE_STRING = {
    2: "int16",
    3: "int32",
    4: "int32",
    5: "fp16",
    6: "fp32",
    7: "fp32",
    11: "bool",
}

TYPE_TO_DTYPE_STRING = {
    types.bool: "bool",
    types.fp16: "fp16",
    types.fp32: "fp32",
    types.int32: "int32",
}


class TorchFrontend(Enum):
    TORCHSCRIPT = 1
    EXIR = 2


def sanitize_op_kind(op_kind: str) -> str:
    """
    In our torch converter, we register torch ops only by its "canonical" name:
    1. No namespace prefix if it is the common aten/prim: e.g. ``aten::softmax`` -> ``softmax``
    2. No double underscore prefix and suffix: e.g. ``__add__`` -> ``add``
    3. Lower-case characters only: e.g. ``mul.Tensor`` -> ``mul.tensor``
    """
    # 1. Skip the aten/prim namespace prefix
    def skip_default_namespaces_with_deliminator(op_kind: str, deliminator: str) -> str:
        split = op_kind.split(deliminator)
        namespace = split[0].lower()
        if namespace in {"aten", "prim"}:
            op_kind = deliminator.join(split[1 : -1])
        return op_kind

    op_kind = skip_default_namespaces_with_deliminator(op_kind, "::")
    op_kind = skip_default_namespaces_with_deliminator(op_kind, ".")

    # 2. Remove underscore prefix and suffix
    if op_kind.startswith("__") and op_kind.endswith("__"):
        op_kind = op_kind[2:-2]

    # 3. Lower case
    return op_kind.lower()


def unify_inplace_and_functional(op_kind: str) -> str:
    """
    In many cases, CoreML uses only functional ops,
    so we do not have to distinguish in-place from functional,
    so we will want to remove the conventional in-place suffix ``_`` of PyTorch.
    For instance, ``sub_`` -> ``sub``
    """
    if op_kind.endswith("_"):
        op_kind = op_kind[:-1]

    return op_kind
