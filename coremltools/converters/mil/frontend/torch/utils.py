#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from enum import Enum


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
            op_kind = deliminator.join(split[1:])
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
