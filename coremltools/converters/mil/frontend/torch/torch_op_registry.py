#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Callable

from coremltools.models._deprecation import deprecated as _deprecated


class TorchOpsRegistry:
    def __init__(self):
        self.name_to_func_mapping = {}

    def get_func(self, op_lookup: str) -> Callable:
        """
        Given a op type key, return the according translation function.
        Note that the key is sanitized by removing suffix and prefix ``_`` before query.
        For instance, ``__add__`` -> ``add``, ``sub_`` -> ``sub``.
        """
        if op_lookup.startswith("__") and op_lookup.endswith("__"):
            # Some ops may have double underscore, such as `__and__`.
            op_lookup = op_lookup[2:-2]
        elif op_lookup.endswith("_"):
            # This is an "in place" op.
            # Look up the standard op instead by removing underscore.
            op_lookup = op_lookup[:-1]

        return self.name_to_func_mapping.get(op_lookup, None)

    def register_func(self, func=None, torch_alias=None, override=False):
        """
        Given an op name and its alias, put the translation function (callable)
        into the registry.
        """
        f_name = func.__name__
        all_f_names = [f_name]
        if torch_alias is not None:
            all_f_names.extend(torch_alias)

        for name in all_f_names:
            if name.endswith("_"):
                raise Exception(
                    f'Attempting to register "{name}" op. Do not register inplace ops. (inplace torch ops'
                    f' end in a "_"). Instead register the normal op version: "{name[:-1]}". The inplace'
                    f" version will be supported automatically."
                )
            if not override and name in self.name_to_func_mapping:
                raise ValueError(f"Torch op {name} already registered.")
            self.set_func_by_name(func, name)

    def set_func_by_name(self, func, name):
        self.name_to_func_mapping[name] = func

    def is_inplace_op(self, op_lookup: str):
        """
        A torch op is considered inplace if the op name endswith ``_``.
        """
        return not (op_lookup.startswith("__") and op_lookup.endswith("__")) and op_lookup.endswith(
            "_"
        )

    # The following functions will be deprecated after 7.2
    # rdar://117502178 ([Infra][Pytorch] We should deprecate the direct use of _TORCH_OPS_REGISTRY in 7.2)
    @_deprecated(
        suffix="Please use coremltools.converters.mil.frontend.torch.register_torch_op",
        version="7.2",
        obj_prefix="_TORCH_OPS_REGISTRY.",
    )
    def __contains__(self, key: str) -> bool:
        return key in self.name_to_func_mapping

    @_deprecated(
        suffix="Please use coremltools.converters.mil.frontend.torch.register_torch_op",
        version="7.2",
        obj_prefix="_TORCH_OPS_REGISTRY.",
    )
    def __setitem__(self, key: str, value: Callable) -> None:
        self.name_to_func_mapping[key] = value

    @_deprecated(
        suffix="Please use coremltools.converters.mil.frontend.torch.register_torch_op",
        version="7.2",
        obj_prefix="_TORCH_OPS_REGISTRY.",
    )
    def __delitem__(self, key: str) -> None:
        del self.name_to_func_mapping[key]

    @_deprecated(
        suffix="Please use coremltools.converters.mil.frontend.torch.register_torch_op",
        version="7.2",
        obj_prefix="_TORCH_OPS_REGISTRY.",
    )
    def __getitem__(self, key: str) -> Callable:
        return self.name_to_func_mapping[key]

_TORCH_OPS_REGISTRY = TorchOpsRegistry()


def register_torch_op(_func=None, torch_alias=None, override=False):
    """
    Registration routine for PyTorch operators
    _func: (PyTorch conversion function) [Default=None]
        PyTorch conversion function to register

    torch_alias: (List of string) [Default=None]
        All other PyTorch operators that should also be mapped to
        current conversion routine.
        e.g. Sort aliased with SortV1, SortV2
        All provided alias operators must not be registered previously.

        "In place" alias are looked up automatically and do not need to
        be registered. PyTorch uses an underscore suffix to denote the
        in place version, e.g. "sum_" is the in place version of "sum".

    override: (Boolean) [Default=False]
        If True, overrides earlier registration i.e. specified
        operator and alias will start pointing to current conversion
        function.
        Otherwise, duplicate registration will error out.
    """
    def func_wrapper(func):
        _TORCH_OPS_REGISTRY.register_func(func, torch_alias, override)
        return func

    if _func is None:
        # decorator called without argument
        return func_wrapper
    return func_wrapper(_func)
