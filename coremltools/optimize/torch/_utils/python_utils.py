#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
from collections import OrderedDict as _OrderedDict
from typing import IO as _IO
from typing import Any as _Any
from typing import Dict as _Dict
from typing import Type as _Type
from typing import Union as _Union

import cattrs as _cattrs
import torch as _torch
import yaml as _yaml
from attr import asdict as _asdict

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


class DictableDataClass:
    """
    Utility class that provides convertors to and from Python dict
    """

    @classmethod
    def from_dict(cls, data_dict: _Dict[str, _Any]) -> "DictableDataClass":
        """
        Create class from a dictionary of string keys and values.

        Args:
            data_dict (:obj:`dict` of :obj:`str` and values): A nested dictionary of strings
                and values.
        """
        # Explicitly raise exception for unrecognized keys
        cls._validate_dict(data_dict)
        converter = _cattrs.Converter(forbid_extra_keys=True)
        converter.register_structure_hook(_torch.Tensor, lambda obj, type: obj)
        return converter.structure_attrs_fromdict(data_dict, cls)

    @classmethod
    def from_yaml(cls, yml: _Union[_IO, str]) -> "DictableDataClass":
        """
        Create class from a yaml stream.

        Args:
            yml: An :py:class:`IO` stream containing yaml or a :obj:`str`
                path to the yaml file.
        """
        if isinstance(yml, str):
            with open(yml, "r") as file:
                dict_from_yml = _yaml.safe_load(file)
        else:
            dict_from_yml = _yaml.safe_load(yml)
        if dict_from_yml is None:
            dict_from_yml = {}
        assert isinstance(dict_from_yml, dict), (
            "Invalid yaml received. yaml stream should return a dict "
            f"on parsing. Received type: {type(dict_from_yml)}."
        )
        return cls.from_dict(dict_from_yml)

    def as_dict(self) -> _Dict[str, _Any]:
        """
        Returns the config as a dictionary.
        """
        return _asdict(self)

    @classmethod
    def _validate_dict(cls: _Type, config_dict: _Dict[str, _Any]):
        for key, _ in config_dict.items():
            if not hasattr(cls, key):
                raise ValueError(f"Found unrecognized key {key} in config_dict: {config_dict}.")
