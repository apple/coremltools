#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
import operator as _operator
import re as _re
from contextlib import contextmanager
from distutils.version import StrictVersion as _StrictVersion
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List
from typing import Tuple as _Tuple
from typing import Type as _Type
from typing import Union as _Union

import numpy as _np
import torch as _torch
import torch.nn as _nn

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


def _get_dtype_info(dtype: _torch.dtype):
    if dtype.is_floating_point:
        info_fn = _torch.finfo
    else:
        info_fn = _torch.iinfo

    return info_fn(dtype)


def get_n_bits_from_dtype(dtype: _Union[str, _torch.dtype]) -> int:
    if type(dtype) is _torch.dtype:
        dtype_info = _get_dtype_info(dtype)
        return dtype_info.bits
    elif type(dtype) is str:
        return int(_re.search(r"\d+", dtype).group())
    else:
        raise TypeError(
            "dtype must either be a string or an instance of torch.dtype," f" not {type(dtype)}"
        )


def get_sign_from_dtype(dtype: _Union[str, _torch.dtype]) -> int:
    if type(dtype) is _torch.dtype:
        dtype_info = _get_dtype_info(dtype)
        return dtype_info.min < 0
    elif type(dtype) is str:
        return not dtype.startswith("u")
    else:
        raise TypeError(
            "dtype must either be a string or an instance of torch.dtype," f" not {type(dtype)}"
        )


def maybe_convert_str_to_dtype(dtype: _Union[str, _torch.dtype]) -> _torch.dtype:
    _str_to_dtype_map = {
        "quint8": _torch.quint8,
        "qint8": _torch.qint8,
        "float32": _torch.float32,
        "int8": _torch.int8,
        "uint8": _torch.uint8,
        # Torch doesn't support int4 or int3
        # but we can represent it as int8
        "int4": _torch.int8,
        "uint4": _torch.uint8,
        "qint4": _torch.qint8,
        "quint4": _torch.quint8,
        "uint3": _torch.uint8,
        "int3": _torch.int8,
        "fp8_e4m3": _torch.float8_e4m3fn,
        "fp8_e5m2": _torch.float8_e5m2,
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
def get_eval_model(model: _nn.Module):
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


def get_fully_qualified_name(model: _torch.nn.Module, module: _torch.nn.Module) -> str:
    """
    Returns fully qualified name for a module if it exists in the model. The fully qualified
    name can be used to fetch the module using ``model.get_submodule``.
    """
    for mod_name, mod in model.named_modules(remove_duplicate=True):
        if mod == module:
            return mod_name
    raise ValueError(f"Module: {module} is not a submodule of {model}.")


def get_atomic_layers(
    module: _nn.Module,
    layer_types: _Union[_List[str], _List[_Type]],
    name_prefix: str = "",
) -> _Dict[str, _nn.Module]:
    """
    Returns a dictionary of layer_name: layer for every layer in the module which
    matches the types specified in layers_to_find.
    """
    if isinstance(module, tuple(layer_types)):
        return {name_prefix: module}
    result = {}
    for name, child in module.named_children():
        result.update(
            get_atomic_layers(
                child,
                layer_types=layer_types,
                name_prefix=name_prefix + "." + name if name_prefix != "" else name,
            )
        )

    return result


def clone_tensor_object(obj: _Any):
    """
    Clone a nested list, tuple or dict of tensors.
    """
    if isinstance(obj, _torch.Tensor):
        return obj.clone()
    elif isinstance(obj, tuple):
        return tuple(clone_tensor_object(item) for item in obj)
    elif isinstance(obj, list):
        return [clone_tensor_object(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: clone_tensor_object(val) for key, val in obj.items()}
    else:
        raise ValueError(f"Cannot clone unrecognized object type: {obj}.")


def get_torch_version(version):
    """
    returns torch version given a version string. Works for versions like
    "2.1.1", "2.1.1+cpu", "2.1.1+rc" etc and would return 2.1.1 for these
    cases
    """
    version_regex = r"\d+\.\d+\.\d+"
    version = _re.search(version_regex, str(version)).group(0)
    return _StrictVersion(version)
