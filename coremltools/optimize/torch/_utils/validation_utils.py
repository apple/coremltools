#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy as _copy
import logging as _logging
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional

import torch as _torch
import torch.nn as _nn

from coremltools.optimize.torch.optimization_config import (
    ModuleOptimizationConfig as _ModuleOptimizationConfig,
)
from coremltools.optimize.torch.optimization_config import (
    PalettizationGranularity as _PalettizationGranularity,
)
from coremltools.optimize.torch.optimization_config import (
    QuantizationGranularity as _QuantizationGranularity,
)

_logger = _logging.getLogger(__name__)


class ConfigValidator:
    def __init__(
        self,
        param_name: str,
        param: _torch.Tensor,
        module: _nn.Module,
        config: _Optional[_ModuleOptimizationConfig],
        module_level_advanced_options: _Optional[_Dict] = None,
    ):
        self.param_name = param_name
        self.param = param
        self.module = module
        self.config = _copy.deepcopy(config)
        self.module_level_advanced_options = module_level_advanced_options

    def validate(self, checks_to_run: _List[str]) -> bool:
        for check_name in checks_to_run:
            check_method = getattr(self, f"sanitize_{check_name}", None)
            assert check_method, f"Check {check_method} not found"

            result = check_method()
            if not result:
                return result

        return True

    def sanitize_quantization_block_size(self):
        """
        Validates and updates block_size attribute in quantization config for specified parameter.
        If compression should be skipped for param, returns False.
        Else, returns True and updates config inplace.
        """
        if self.config.granularity != _QuantizationGranularity.per_block:
            return True

        if len(self.config.block_size) > self.param.ndim:
            _logger.warning(
                f"{self.param_name}: Length of block_size tuple {len(self.config.block_size)} "
                f"should not exceed the number of dimensions in the parameter {self.param.ndim}"
            )
            return False

        # Verify that for non input or output channel axis, block size is either zero or equal to axis length
        for idx, bs in enumerate(self.config.block_size):
            if idx > 1:
                if bs != 0 and bs != self.param.shape[idx]:
                    _logger.warning(
                        f"{self.param_name}: Unsupported block_size={self.config.block_size}. "
                        "Blocking is currently only supported along input OR output channel axis."
                    )
                    return False

        # Determine whether it is an N-D block or a integer block size
        if len(self.config.block_size) >= 2:
            bs_output = self.config.block_size[0]
            bs_input = self.config.block_size[1]
        else:
            bs_output = None
            bs_input = self.config.block_size[0]

        should_block_output = (
            bs_output > 0 and bs_output < self.param.shape[0] if bs_output else False
        )
        should_block_input = bs_input > 0 and bs_input < self.param.shape[1]

        if should_block_input and not should_block_output:
            # By default we will always have per-channel on output-channel axis
            bs_output = 1
            should_block_output = True

        if not should_block_input and not should_block_output:
            _logger.warning(
                f"{self.param_name}: Valid block_size={self.config.block_size} not specified for any axis. "
                "Use per_channel or per_tensor granularity if blocking is not required."
            )
            return False

        # Check if the output-channel block size is divisible by the axis length
        if should_block_output and self.param.shape[0] % bs_output != 0:
            _logger.warning(
                f"{self.param_name}: block_size={bs_output} is not divisible by axis length={self.param.shape[0]}"
            )
            return False

        # Check if the input-channel block size is divisible by the axis length
        if should_block_input and self.param.shape[1] % bs_input != 0:
            _logger.warning(
                f"{self.param_name}: block_size={bs_input} is not divisible by axis length={self.param.shape[0]}"
            )
            return False

        self.config.block_size = (bs_output, bs_input)
        return True

    def sanitize_palettization_group_size(self):
        """
        Validates and updates block_size attribute in palettization config for specified parameter.
        If compression should be skipped for param, returns False.
        Else, returns True and updates config inplace.
        """
        if self.config.granularity != _PalettizationGranularity.per_grouped_channel:
            return True

        # If block size is not divisible by axis length skip palettizing this param
        if (
            self.module_level_advanced_options
            and self.module_level_advanced_options["cluster_permute"]
        ):
            axis_length = self.param.permute(
                self.module_level_advanced_options["cluster_permute"]
            ).shape[self.config.channel_axis]
        else:
            axis_length = self.param.shape[self.config.channel_axis]
        if axis_length % self.config.group_size != 0:
            _logger.warning(
                f"{self.param_name}: axis_length={axis_length} is not divisible by group_size={self.config.group_size}"
            )
            return False

        return True

    def sanitize_palettization_cluster_dim(self):
        """
        Validates and updates cluster_dim attribute in palettization config for specified parameter.
        If compression should be skipped for param, returns False.
        Else, returns True and updates config inplace.
        """
        if self.config.cluster_dim is None:
            self.config.cluster_dim = 1
            return True

        if self.config.cluster_dim > 1:
            if self.config.channel_axis == 0:
                dim_size = self.param.shape[0]
            else:
                dim_size = self.param.shape[1]
            if dim_size % self.config.cluster_dim != 0:
                _logger.warning(
                    f"{self.param_name}: The number of channels in channel axis dimension: {self.config.channel_axis},"
                    f" {dim_size} is not divisible by cluster_dim={self.config.cluster_dim}"
                )
                return False

        return True


def validate_param_config(
    param_name: str,
    param: _torch.Tensor,
    module: _nn.Module,
    config: _Optional[_ModuleOptimizationConfig],
    checks_to_run: _List[str],
    module_level_advanced_options: _Optional[_Dict] = None,
):
    validator = ConfigValidator(param_name, param, module, config, module_level_advanced_options)
    is_valid_config = validator.validate(checks_to_run)
    if not is_valid_config:
        # Skip compression for this param if config is invalid
        _logger.info(f"Skipping compression for {param_name}")
        return None

    return validator.config
