# Copyright (c) 2022, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as _np

from coremltools.converters.mil.mil import Operation as _Operation
from coremltools.models._deprecation import deprecated as _deprecated
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig as _OpLinearQuantizerConfig,
    OpMagnitudePrunerConfig as _OpMagnitudePrunerConfig,
    OpPalettizerConfig as _OpPalettizerConfig,
    OpThresholdPrunerConfig as _OpThresholdPrunerConfig,
    OptimizationConfig as _OptimizationConfig,
)
from coremltools.optimize.coreml import (
    linear_quantize_weights as _linear_quantize_weights,
    decompress_weights as _decompress_weights,
    palettize_weights as _palettize_weights,
    prune_weights as _prune_weights,
)

_DEFAULT_MIN_WEIGHT_SIZE_TO_COMPRESS = 2048

def _default_op_selector(const_op):
    if not isinstance(const_op, _Operation) or const_op.op_type != "const":
        raise ValueError("Input of the op_selector must be type of const Operation, got {}.".format(type(const_op)))
    return const_op.val.val.size > _DEFAULT_MIN_WEIGHT_SIZE_TO_COMPRESS

@_deprecated(
    suffix="Please use coremltools.optimize.coreml.affine_quantize_weights",
    version="7.0",
    obj_prefix="coremltools.compression_utils.",
)
def affine_quantize_weights(mlmodel, mode="linear_symmetric", op_selector=None, dtype=_np.int8):
    """
    ``coremltools.compression_utils.affine_quantize_weights`` is deprecated and will be removed in the future.
    Please use :py:class:`coremltools.optimize.coreml.linear_quantize_weights`.
    """
    if op_selector is None:
        op_selector = _default_op_selector

    op_config = _OpLinearQuantizerConfig(mode=mode, dtype=dtype, weight_threshold=None)
    config = _OptimizationConfig(global_config=op_config, is_deprecated=True, op_selector=op_selector)
    return _linear_quantize_weights(mlmodel, config)

@_deprecated(
    suffix="Please use coremltools.optimize.coreml.palettize_weights",
    version="7.0",
    obj_prefix="coremltools.compression_utils.",
)
def palettize_weights(mlmodel, nbits=None, mode="kmeans", op_selector=None, lut_function=None):
    """
    ``coremltools.compression_utils.palettize_weights`` is deprecated and will be removed in the future.
    Please use :py:class:`coremltools.optimize.coreml.palettize_weights`.
    """
    if op_selector is None:
        op_selector = _default_op_selector

    op_config = _OpPalettizerConfig(nbits=nbits, mode=mode, lut_function=lut_function, weight_threshold=None)
    config = _OptimizationConfig(global_config=op_config, is_deprecated=True, op_selector=op_selector)
    return _palettize_weights(mlmodel, config)

@_deprecated(
    suffix="Please use coremltools.optimize.coreml.sparsify_weights",
    version="7.0",
    obj_prefix="coremltools.compression_utils.",
)
def sparsify_weights(
    mlmodel, mode="threshold_based", threshold=1e-12, target_percentile=1.0, op_selector=None
):
    """
    ``coremltools.compression_utils.sparsify_weights`` is deprecated and will be removed in the future.
    Please use :py:class:`coremltools.optimize.coreml.prune_weights`.
    """
    if op_selector is None:
        op_selector = _default_op_selector

    if mode.upper() == "THRESHOLD_BASED":
        op_config = _OpThresholdPrunerConfig(
            threshold=threshold,
            minimum_sparsity_percentile=0.0,
            weight_threshold=None,
        )

    elif mode.upper() == "PERCENTILE_BASED":
        op_config = _OpMagnitudePrunerConfig(
            target_sparsity=target_percentile,
            weight_threshold=None,
        )

    else:
        raise ValueError(
            'Only modes "THRESHOLD_BASED" and "PERCENTILE_BASED" are supported for weight sparsification.'
            f' Got mode: "{mode}".'
        )

    config = _OptimizationConfig(global_config=op_config, is_deprecated=True, op_selector=op_selector)
    return _prune_weights(mlmodel, config)

@_deprecated(
    suffix="Please use coremltools.optimize.coreml.decompress_weights",
    version="7.0",
    obj_prefix="coremltools.compression_utils.",
)
def decompress_weights(mlmodel):
    """
    ``coremltools.compression_utils.decompress_weights`` is deprecated and will be removed in the future.
    Please use :py:class:`coremltools.optimize.coreml.decompress_weights`.
    """
    return _decompress_weights(mlmodel)
