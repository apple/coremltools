#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
from typing import Tuple, Type

import torch

from coremltools.optimize.torch._utils.math_utils import rmse_error
from coremltools.optimize.torch._utils.metadata_utils import CompressionMetadata, CompressionType
from coremltools.optimize.torch.base_model_optimizer import _Report
from coremltools.optimize.torch.pruning._utils import (
    block2_sparsity,
    structured_sparsity,
    unstructured_sparsity,
)

_logger = _logging.getLogger(__name__)


def _normalize_report(report: _Report) -> _Report:
    """
    Normalizes the report by making sure all parameter reports have the same number
    """
    all_keys = set()
    for _, param_report in report.items():
        for key in param_report:
            all_keys.add(key)

    for _, param_report in report.items():
        for key in all_keys:
            if key not in param_report:
                param_report[key] = -1
    return report


def compute_post_training_report(
    uncompressed_model: torch.nn.Module,
    compressed_model: torch.nn.Module,
    supported_modules: Tuple[Type[torch.nn.Module]],
) -> _Report:
    """
    Computes rmse between compressed and uncompressed parameters
    """
    report = _Report()
    for name, module in compressed_model.named_modules():
        if not isinstance(module, supported_modules):
            continue

        compression_metadata = CompressionMetadata.from_state_dict(module.state_dict())

        for param_name in compression_metadata:
            module_summary = dict()
            param_key = f"{name}.{param_name}" if name else param_name

            with torch.no_grad():
                compression_types = [
                    CompressionType(x) for x in compression_metadata[param_name].compression_type
                ]

                uncompressed_module = uncompressed_model.get_submodule(name)
                compressed_param = module.get_parameter(param_name)
                uncompressed_param = uncompressed_module.get_parameter(param_name)

                module_summary["error"] = rmse_error(compressed_param, uncompressed_param).item()

                module_summary["#params"] = int(torch.numel(compressed_param))

                if CompressionType.pruning in compression_types:
                    sparse_summary = {
                        "structured_weight_sparsity": structured_sparsity(compressed_param),
                        "unstructured_weight_sparsity": unstructured_sparsity(compressed_param),
                    }

                    if compressed_param.size(0) % 2 == 0:
                        sparse_summary["block2_weight_sparsity"] = block2_sparsity(compressed_param)
                    else:
                        sparse_summary["block2_weight_sparsity"] = -1  # Not applicable

                    module_summary.update(sparse_summary)

                if CompressionType.quantization in compression_types:
                    quantization_n_bits = compression_metadata[param_name].quantization_n_bits
                    # FIXME: add sign of dtype here
                    module_summary["dtype"] = f"dtype=int{quantization_n_bits}"

                if CompressionType.palettization in compression_types:
                    lut_shape = compression_metadata[param_name].lut.shape

                    n_clusters = lut_shape[-2]
                    cluster_dim = lut_shape[-1]

                    module_summary[
                        "palettization_mode"
                    ] = f"num_clusters={n_clusters}, cluster_dim={cluster_dim}"

                report[param_key] = module_summary

    report = _normalize_report(report)
    return report
