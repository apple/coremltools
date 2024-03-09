#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
from collections import OrderedDict as _OrderedDict
from typing import Any as _Any

_logger = _logging.getLogger(__name__)


def get_str(val: _Any):
    if isinstance(val, float):
        return f"{val:.5f}"
    return str(val)


class RegistryMixin:
    REGISTRY = None

    @classmethod
    def register(cls, name: str):
        if cls.REGISTRY is None:
            cls.REGISTRY = _OrderedDict()

        def inner_wrapper(wrapped_obj):
            if name in cls.REGISTRY:
                _logger.warning(
                    f"Name: {name} is already registered with object: {cls.REGISTRY[name].__name__} "
                    f"in registry: {cls.__name__}"
                    f"Over-writing the name with new class: {wrapped_obj.__name__}"
                )
            cls.REGISTRY[name] = wrapped_obj
            return wrapped_obj

        return inner_wrapper

    @classmethod
    def _get_object(cls, name: str):
        if name in cls.REGISTRY:
            return cls.REGISTRY[name]
        raise NotImplementedError(
            f"No object is registered with name: {name} in registry {cls.__name__}."
        )


class ClassRegistryMixin(RegistryMixin):
    @classmethod
    def get_class(cls, name: str):
        return cls._get_object(name)


class FunctionRegistryMixin(RegistryMixin):
    @classmethod
    def get_function(cls, name: str):
        return cls._get_object(name)
