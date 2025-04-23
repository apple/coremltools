#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Type as _Type

import torch as _torch

from coremltools.optimize.torch._utils.python_utils import ClassRegistryMixin as _ClassRegistryMixin
from coremltools.optimize.torch.base_model_optimizer import (
    BaseModelOptimizer as _BaseModelOptimizer,
)
from coremltools.optimize.torch.optimization_config import (
    ModuleOptimizationConfig as _ModuleOptimizationConfig,
)
from coremltools.optimize.torch.optimization_config import OptimizationConfig as _OptimizationConfig


class _ModuleToOptConfigRegistry(_ClassRegistryMixin):
    """
    A registry mapping :py:class:`ModuleOptimizationConfig` classes to corresponding
    :py:class:`OptimizationConfig` class
    """

    @classmethod
    def register_module_cfg(cls, module_opt_cfg: _Type[_ModuleOptimizationConfig]):
        return cls.register(module_opt_cfg)

    @classmethod
    def get_opt_cfg(
        cls, module_opt_cfg: _Type[_ModuleOptimizationConfig]
    ) -> _Type[_OptimizationConfig]:
        for key in cls.REGISTRY:
            if issubclass(module_opt_cfg, key):
                return cls.get_class(key)
        raise NotImplementedError(
            f"No object is registered with key: {module_opt_cfg} in registry {cls.__name__}."
        )


class _ConfigToOptimizerRegistry(_ClassRegistryMixin):
    """
    A registry mapping :py:class:`OptimizationConfig` classes to corresponding
    :py:class:`BaseModelOptimizer` class
    """

    @classmethod
    def register_config(cls, config: _Type[_OptimizationConfig]):
        return cls.register(config)

    @classmethod
    def get_optimizer(cls, config: _Type[_OptimizationConfig]) -> _Type[_BaseModelOptimizer]:
        return cls.get_class(config)


def is_supported_module_for_config(
    module: _torch.nn.Module, module_config_cls: _Type[_ModuleOptimizationConfig]
) -> bool:
    # Retrieve optimization config class from module optimization config
    opt_config_cls = _ModuleToOptConfigRegistry.get_opt_cfg(module_config_cls)

    # Retrieve optimizer class from optimization config class
    optimizer_cls = _ConfigToOptimizerRegistry.get_optimizer(opt_config_cls)

    return type(module) in optimizer_cls._supported_modules
