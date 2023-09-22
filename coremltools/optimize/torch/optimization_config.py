#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict as _OrderedDict
from typing import IO as _IO
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Type as _Type
from typing import Union as _Union

import cattrs as _cattrs
import torch as _torch
import yaml as _yaml
from attr import Factory as _Factory
from attr import asdict as _asdict
from attr import define as _define


@_define
class ModuleOptimizationConfig:
    @classmethod
    def from_dict(cls, config_dict: _Dict[str, _Any]) -> "ModuleOptimizationConfig":
        """
        Create class from a dictionary of string keys and values.

        Args:
            config_dict (:obj:`dict` of :obj:`str` and values): A nested dictionary of strings
                and values.
        """
        # passing forbid_extra_keys=True doesn't prevent silent failure when keys are mis-spelled
        _validate_config_dict(cls, config_dict)
        converter = _cattrs.Converter(forbid_extra_keys=True)
        return converter.structure_attrs_fromdict(config_dict, cls)

    @classmethod
    def from_yaml(cls, yml: _Union[_IO, str]) -> "ModuleOptimizationConfig":
        """
        Create class from a yaml stream.

        Args:
            yml: An :py:class:`IO` stream containing yaml or a :obj:`str`
                path to the yaml file.
        """
        return _from_yaml(cls, yml)

    def as_dict(self) -> _Dict[str, _Any]:
        """
        Returns the config as a dictionary.
        """
        return _asdict(self)


@_define
class OptimizationConfig:
    global_config: _Optional[ModuleOptimizationConfig] = None
    module_type_configs: _Dict[
        _Union[_Callable, str], _Optional[ModuleOptimizationConfig]
    ] = _Factory(_OrderedDict)
    module_name_configs: _Dict[str, _Optional[ModuleOptimizationConfig]] = _Factory(_OrderedDict)

    def set_global(
        self, global_config: _Optional[ModuleOptimizationConfig]
    ) -> "OptimizationConfig":
        """
        Set the global config.
        """
        self.global_config = global_config
        return self

    def set_module_type(
        self, object_type: _Union[_Callable, str], opt_config: _Optional[ModuleOptimizationConfig]
    ) -> "OptimizationConfig":
        """
        Set the module level optimization config for a given module type. If the module level optimization config
        for an existing module type was already set, the new config will override the old one.
        """
        self.module_type_configs[object_type] = opt_config
        return self

    def set_module_name(
        self, module_name: str, opt_config: _Optional[ModuleOptimizationConfig]
    ) -> "OptimizationConfig":
        """
        Set the module level optimization config for a given module instance. If the module level optimization config
        for an existing module was already set, the new config will override the old one.
        """
        self.module_name_configs[module_name] = opt_config
        return self

    def get_module_config(
        self, name: str, module: _torch.nn.Module
    ) -> _Optional[ModuleOptimizationConfig]:
        if name in self.module_name_configs:
            return self.module_name_configs[name]
        elif type(module) in self.module_type_configs:
            return self.module_type_configs[type(module)]
        elif module.__class__.__name__ in self.module_type_configs:
            return self.module_type_configs[module.__class__.__name__]
        else:
            return self.global_config

    @classmethod
    def from_dict(cls, config_dict: _Dict[str, _Any]) -> _Optional["OptimizationConfig"]:
        """
        Create class from a dictionary of string keys and values.

        Args:
            config_dict (:obj:`dict` of :obj:`str` and values): A nested dictionary of strings
                and values.
        """
        # passing forbid_extra_keys=True doesn't prevent silent failure when keys are mis-spelled
        _validate_config_dict(cls, config_dict)
        return

    @classmethod
    def from_yaml(cls, yml: _Union[_IO, str]) -> "OptimizationConfig":
        """
        Create class from a yaml stream.

        Args:
            yml: An :py:class:`IO` stream containing yaml or a :obj:`str`
                path to the yaml file.
        """
        return _from_yaml(cls, yml)

    def as_dict(self) -> _Dict[str, _Any]:
        """
        Returns the config as a dictionary.
        """
        return _asdict(self)

    def _validate_same_params(self, param_names: _List[str]):
        """
        This method validates that all the parameters in param_names
        have the same value across all the module level configs.
        """
        expected_values = None
        if self.global_config is not None:
            expected_values = {
                param_name: getattr(self.global_config, param_name) for param_name in param_names
            }
        for name, config in self.module_type_configs.items():
            if config is not None:
                expected_values = self._validate_expected_value(
                    expected_values, name, config, param_names
                )
        for name, config in self.module_name_configs.items():
            if config is not None:
                expected_values = self._validate_expected_value(
                    expected_values, name, config, param_names
                )

    @staticmethod
    def _validate_expected_value(
        expected_values: _Dict[str, _Any],
        name: str,
        config: ModuleOptimizationConfig,
        param_names: _List[str],
    ):
        if expected_values is None:
            expected_values = {
                param_name: getattr(config, param_name) for param_name in param_names
            }
        for param_name, expected_val in expected_values.items():
            val = getattr(config, param_name)
            if val != expected_val:
                raise ValueError(
                    f"Value of parameter {param_name} cannot "
                    f"be different between different module level configs."
                    f"Expected value: {expected_val}, received: {val} "
                    f"for config {name}."
                )
        return expected_values


def _structure_from_dict_hook_factory(conversion_cls: _Any) -> _Callable:
    def _structure_from_dict_hook(
        module_type_dict: _Dict[_Union[_Callable, str], _Any], type: _Any
    ):
        return_dict = _OrderedDict()
        for key, value in module_type_dict.items():
            if value is None:
                return_dict[key] = None
            else:
                if isinstance(value, dict):
                    return_dict[key] = conversion_cls.from_dict(value)
                else:
                    assert isinstance(value, conversion_cls), (
                        "value in module type dict should be either a dict or "
                        "a module config object."
                    )
                    return_dict[key] = value
        return return_dict
    return _structure_from_dict_hook


def _validate_config_dict(cls: _Type, config_dict: _Dict[str, _Any]):
    for key, _ in config_dict.items():
        if not hasattr(cls, key):
            raise ValueError(f"Found unrecognized key {key} in config_dict: {config_dict}.")


def _from_yaml(
    cls: _Union[_Type[OptimizationConfig], _Type[ModuleOptimizationConfig]], yml: _Union[_IO, str]
):
    if isinstance(yml, str):
        with open(yml, "r") as file:
            dict_from_yml = _yaml.safe_load(file)
    else:
        dict_from_yml = _yaml.safe_load(yml)
    assert isinstance(dict_from_yml, dict), (
        "Invalid yaml received. yaml stream should return a dict "
        f"on parsing. Received type: {type(dict_from_yml)}."
    )
    return cls.from_dict(dict_from_yml)


def _validate_module_type_keys_factory(supported_modules):
    supported_module_names = [cls.__name__ for cls in supported_modules]

    def validate_module_type_key(instance, attribute, value):
        if isinstance(value, str):
            assert value in supported_module_names, (
                f"keys for module_type_configs must be one of "
                f"{supported_module_names}. Received: {value}."
            )
        else:
            assert value in supported_modules, (
                f"keys for module_type_configs must be one of "
                f"{supported_modules}. Received: {value}."
            )

    return validate_module_type_key
