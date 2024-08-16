#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from enum import Enum

import numpy as np
import torch

from coremltools.converters.mil.mil import types

# NOTE [represent torch dtype by integer]
# In TorchScript, some ops will receive a dtype input as an integer which maps to a torch dtype.
# The below mapping was found by converting test models with different dtypes passed to ones.
# There is one modification to original torch mapping, though, due to Core ML lacks 64-bit dtype
# When mapping from torch dtype to integer number, we map
#     * int64 to int32's number
#     * float64 to float32's number
# When mapping from integer number back to torch dtype, we map
#     * int64's number to int32
#     * float64's number to float32
# TODO(https://github.com/apple/coremltools/issues/2153): This is confusing... we should refactor
NUM_TO_TORCH_DTYPE = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    11: torch.bool,
    12: torch.qint8,
    13: torch.quint8,
    14: torch.qint32,
}

def dtype_to_32bit(dtype):
    if dtype == torch.int64:
        return torch.int32
    elif dtype == torch.float64:
        return torch.float32
    else:
        return dtype

TORCH_DTYPE_TO_NUM = {
    dtype: val for val, dtype in NUM_TO_TORCH_DTYPE.items()
}
TORCH_DTYPE_TO_NUM[torch.int64] = TORCH_DTYPE_TO_NUM[torch.int32]
TORCH_DTYPE_TO_NUM[torch.float64] = TORCH_DTYPE_TO_NUM[torch.float32]

NUM_TO_NUMPY_DTYPE = {
    0: np.uint8,
    1: np.int8,
    2: np.int16,
    3: np.int32,
    4: np.int32,
    5: np.float16,
    6: np.float32,
    7: np.float32,
    11: bool,
}

NUMPY_DTYPE_TO_TORCH_NUM = {
    dtype: val for val, dtype in NUM_TO_NUMPY_DTYPE.items()
}
NUMPY_DTYPE_TO_TORCH_NUM[np.int64] = NUMPY_DTYPE_TO_TORCH_NUM[np.int32]
NUMPY_DTYPE_TO_TORCH_NUM[np.float64] = NUMPY_DTYPE_TO_TORCH_NUM[np.float32]

NUM_TO_DTYPE_STRING = {
    0: "uint8",
    1: "int8",
    2: "int16",
    3: "int32",
    4: "int32",
    5: "fp16",
    6: "fp32",
    7: "fp32",
    11: "bool",
}

TYPE_TO_DTYPE_STRING = {
    types.uint8: "uint8",
    types.int8: "int8",
    types.int32: "int32",
    types.fp16: "fp16",
    types.fp32: "fp32",
    types.bool: "bool",
}

TORCH_QTYPE_TO_NP_TYPE = {
    torch.int8: np.int8,
    torch.qint8: np.int8,
    torch.uint8: np.uint8,
    torch.quint8: np.uint8,
}

TORCH_QTYPE_TO_STR = {
    torch.int8: "int8",
    torch.qint8: "int8",
    torch.uint8: "uint8",
    torch.quint8: "uint8",
}

MIL_DTYPE_TO_TORCH_DTYPE = {
    types.bool: torch.bool,
    types.fp16: torch.float16,
    types.fp32: torch.float32,
    types.int16: torch.int16,
    types.int32: torch.int32,
}

TORCH_DTYPE_TO_MIL_DTYPE = {v: k for k, v in MIL_DTYPE_TO_TORCH_DTYPE.items()}
TORCH_DTYPE_TO_MIL_DTYPE[torch.int64] = types.int32
TORCH_DTYPE_TO_MIL_DTYPE[torch.float64] = types.fp32


class TorchFrontend(Enum):
    TORCHSCRIPT = 1
    EXIR = 2


def sanitize_op_kind(op_kind: str) -> str:
    """
    In our torch converter, we register torch ops only by its "canonical" name:
    1. Lower-case characters only, e.g. ``div.Tensor`` -> ``div.tensor``
    2. No double underscore prefix and suffix, e.g. ``__add__`` -> ``add``
    3. No namespace prefix if it is the common aten/prim, e.g.
           ``aten::softmax`` -> ``softmax``
           ``aten.pow`` -> ``pow``
       and no type trait suffix if it is not distinguished in Core ML, e.g.
           ``bmm.default`` -> ``bmm``
           ``slice_copy.tensor`` -> ``slice_copy``
           ``mul.scalar`` -> ``mul``
    """

    def skip_default_prefix_and_suffix_with_deliminator(
        op_kind: str,
        deliminator: str,
    ) -> str:
        split = op_kind.split(deliminator)
        start = 1 if split[0] in {"aten", "prim"} and len(split) > 1 else 0
        stop = -1 if split[-1] in {
            "default",
            "tensor",
            "tensor_mode",
            "scalar",
            "tensor_scalar",
        } and len(split) - start > 1 else len(split)
        op_kind = deliminator.join(split[start : stop])
        return op_kind

    # 1. Lower case
    op_kind = op_kind.lower()

    # 2. Remove underscore prefix and suffix
    if op_kind.startswith("__") and op_kind.endswith("__"):
        op_kind = op_kind[2:-2]

    # 3. Skip the aten/prim namespace prefix, and default/tensor/scalar suffix
    op_kind = skip_default_prefix_and_suffix_with_deliminator(op_kind, "::")
    op_kind = skip_default_prefix_and_suffix_with_deliminator(op_kind, ".")

    return op_kind


def unify_inplace_and_functional(op_kind: str) -> str:
    """
    In many cases, Core ML uses only functional ops,
    so we do not have to distinguish in-place from functional,
    so we will want to remove the conventional in-place suffix ``_`` of PyTorch.
    For instance, ``sub_`` -> ``sub``
    """
    if op_kind.endswith("_"):
        op_kind = op_kind[:-1]

    return op_kind
