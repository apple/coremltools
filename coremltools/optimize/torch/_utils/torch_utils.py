#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
import operator as _operator
import re as _re
from contextlib import contextmanager
from typing import List as _List
from typing import Tuple as _Tuple
from typing import Union as _Union

import numpy as _np
import torch as _torch

_logger = _logging.getLogger(__name__)


def list_or_str_to_tensor(alist: _Union[_List[int], str, _torch.Tensor]) -> _torch.Tensor:
    if isinstance(alist, _torch.Tensor):
        return alist
    elif isinstance(alist, str):
        # Safety check since we are calling eval
        range_str_regex = r"^(range)\(\d+(\,?\s*\d+){0,2}\)$"
        assert _re.match(range_str_regex, alist), (
            f"{alist} is invalid.",
            "Please provide a string such as 'range(...)'",
        )
        try:
            alist = eval(alist)
        except Exception:
            _logger.error(
                f"Invalid range str {alist}.",
                "Please refer to the documentation for correct usage",
            )

    return _torch.tensor(
        _np.ones(
            len(alist),
        )
        * alist,
        dtype=_torch.float32,
        requires_grad=False,
    )


def maybe_convert_str_to_dtype(dtype: _Union[str, _torch.dtype]) -> _torch.dtype:
    _str_to_dtype_map = {
        "quint8": _torch.quint8,
        "qint8": _torch.qint8,
        "float32": _torch.float32,
    }
    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype in _str_to_dtype_map:
            return _str_to_dtype_map[dtype]
        else:
            raise ValueError(f"Received unsupported dtype: {dtype}")
    elif isinstance(dtype, _torch.dtype):
        return dtype
    else:
        raise ValueError(f"Received unrecognized type for dtype: {type(dtype)}")


def maybe_convert_str_to_mod_type(mod_type: str):
    """
    Convert str to module type
    """
    if not isinstance(mod_type, str):
        return mod_type
    if _re.fullmatch(r"operator\.[a-z]+", mod_type) and hasattr(_operator, mod_type.split(".")[-1]):
        return getattr(_operator, mod_type.split(".")[-1])
    elif _re.fullmatch(r"torch\.[a-z]+", mod_type) and hasattr(_torch, mod_type.split(".")[-1]):
        return getattr(_torch, mod_type.split(".")[-1])
    elif hasattr(_torch.nn, mod_type):
        return getattr(_torch.nn, mod_type)
    elif hasattr(_torch.nn.functional, mod_type):
        return getattr(_torch.nn.functional, mod_type)
    return mod_type


@contextmanager
def get_eval_model(model):
    train_flag = model.training
    try:
        yield model.eval()
    finally:
        model.train(mode=train_flag)


def get_parent_child_name(name: str) -> _Tuple[str, str]:
    """
    Returns name of parent and child modules from a full module name.
    """
    split = name.rsplit(".", 1)
    if len(split) == 1:
        return "", split[0]
    else:
        return split[0], split[1]
