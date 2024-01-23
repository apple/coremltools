#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from enum import Enum

_DEFAULT_OP_NAMESPACES = set(["aten", "prim"])


class TorchFrontend(Enum):
    TORCHSCRIPT = 1
    EXIR = 2


def sanitize_op_kind(op_kind: str) -> str:
    # We conventionally skip the aten/prim namespaces in our naming
    namespace = op_kind.split("::")[0].lower()
    if namespace in _DEFAULT_OP_NAMESPACES:
        op_kind = op_kind.split("::")[-1]

    # Some ops may have double underscore, e.g. `__and__`
    # We conventionally remove such prefix and suffix, e.g. `__add__` -> `add`
    if op_kind.startswith("__") and op_kind.endswith("__"):
        op_kind = op_kind[2:-2]

    # We conventionally use only lower-case characters
    return op_kind.lower()


# In many cases, CoreML uses only functional ops,
# so we do not have to distinguish in-place from functional,
# so we will want to remove the conventional in-place suffix `_` of PyTorch.
# For instance, `sub_` -> `sub`
def unify_inplace_and_functional(op_kind: str) -> str:
    if op_kind.endswith("_"):
        op_kind = op_kind[:-1]

    return op_kind
