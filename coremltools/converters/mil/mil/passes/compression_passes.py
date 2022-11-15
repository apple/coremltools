#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools import _logger as logger
from coremltools.converters.mil.backend.mil.load import should_use_weight_file
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.ops.defs.iOS16 import (
    constexpr_affine_dequantize, constexpr_lut_to_dense,
    constexpr_sparse_to_dense)
from coremltools.converters.mil.mil.passes.quantization_passes import \
    AbstractQuantizationPass
from coremltools.models.neural_network.quantization_utils import \
    _get_kmeans_lookup_table_and_weight


class SparseParams:
    def __init__(self, nonzero_data=None, mask=None, shape=None):
        self.nonzero_data = nonzero_data
        self.mask = mask
        self.shape = shape


class WeightSparsifier(AbstractQuantizationPass):
    """
    This transform does the following, for each const op and if the "op_selector" return True:
    - (self.sparsity) fraction of values with the least absolute value are zeroed out.
    - If fake_compression=False,  Zeroed-Out Value is encoded via constexpr_sparse_to_dense op
    - If fake_compression=True,   Zeroed-Out Value is encoded via const op
    - Old const is replaced by a new operation with zeroed-out value.
    """
    WEIGHT_SPARSIFICATION_MODES = ["THRESHOLD_BASED", "PERCENTILE_BASED"]

    def __init__(self, mode="threshold_based", threshold=1e-3, target_percentile=1.0, fake_compression=False, op_selector=None):
        super().__init__(op_selector=op_selector)
        self.fake_compression = fake_compression
        self.mode = mode.upper()
        self.threshold = threshold
        self.target_percentile = target_percentile

        if not self.mode in WeightSparsifier.WEIGHT_SPARSIFICATION_MODES:
            msg = (
                "Only mode {} supported for weight sparsification. Got mode {}.".format(
                    WeightSparsifier.WEIGHT_SPARSIFICATION_MODES, self.mode
                )
            )
            raise ValueError(msg)

        if self.mode == "PERCENTILE_BASED" and (self.target_percentile < 0 or self.target_percentile > 1):
            raise ValueError("Invalid value of target_percentile: {}. Needs to be in [0, 1]".format(self.target_percentile))

        if self.mode == "THRESHOLD_BASED" and self.threshold < 0:
            raise ValueError("Invalid value of threshold: {}. Needs to be in [0, inf)".format(self.threshold))

    def is_valid_op(self, op):
        if op.op_type == "const" and should_use_weight_file(op.val.val):
            return True
        return False

    @staticmethod
    def compress(val, mode, target_percentile=None, threshold=None):

        mode = mode.upper()

        def sparsify_with_percentile(val, target_percentile):
            q = target_percentile * 100
            return np.where(np.abs(val) <= np.percentile(np.abs(val), q), 0, val)

        def sparsify_with_thresohld(val, threshold):
            return np.where(np.abs(val) <= threshold, 0, val)

        if not isinstance(val, (np.ndarray, np.generic)):
            raise ValueError("Only numpy arrays are supported")

        flattened_val = val.flatten()

        if mode == "PERCENTILE_BASED":
            flattened_val = sparsify_with_percentile(flattened_val, target_percentile)
        elif mode == "THRESHOLD_BASED":
            flattened_val = sparsify_with_thresohld(flattened_val, threshold)

        params = SparseParams()
        params.nonzero_data = flattened_val[np.where(flattened_val != 0)]
        params.mask = np.packbits(np.where(flattened_val != 0, 1, 0), bitorder="little")
        params.shape = val.shape
        return params

    @staticmethod
    def decompress(params):
        if not isinstance(params, SparseParams):
            raise ValueError("Invalid type of params")
        return constexpr_sparse_to_dense.decompress(params.nonzero_data, params.mask, params.shape)

    def transform_op(self, op):
        block = op.enclosing_block
        sparse_params = self.compress(op.val.val, self.mode, self.target_percentile, self.threshold)

        if not self.fake_compression:
            new_var = mb.constexpr_sparse_to_dense(
                nonzero_data=sparse_params.nonzero_data,
                mask=sparse_params.mask,
                shape=np.uint32(sparse_params.shape),
                before_op=op,
                name=op.name + "_sparsified",
            )
        else:
            decompressed_val = self.decompress(sparse_params)
            new_var = mb.const(
                val=decompressed_val,
                before_op=op,
                name=op.name + "_fake_sparsified",
            )

        op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=new_var,
            no_check_var_types=True,
        )

        block.remove_ops([op])


class LutParams:
    def __init__(self, lut=None, indices=None, shape=None):
        self.lut = lut
        self.indices = indices
        self.shape = shape


class WeightPalettizer(AbstractQuantizationPass):
    """
    This transform does the following, for each const op and if the "op_selector" return True:
    - A linear look up table with 2**(nbits) entries is created and value is represented via indexing into this look up table.
    - If fake_compression=False,  compressed value is encoded via constexpr_lut_to_dense op
    - If fake_compression=True,   compressed value is decompressed and then encoded via const op
    - Old const op is replaced by a newly created operation.
    """
    WEIGHT_PALETTIZATION_MODES = ["KMEANS", "UNIFORM", "UNIQUE", "CUSTOM"]
    def __init__(self, nbits, fake_compression=False, op_selector=None, mode="kmeans", lut_function=None):
        super().__init__(op_selector=op_selector)
        self.fake_compression = fake_compression
        self.nbits = nbits
        self.mode = mode.upper()
        self.lut_function = lut_function

        if not self.mode in WeightPalettizer.WEIGHT_PALETTIZATION_MODES:
            msg = (
                "Only mode {} supported for weight palettization. Got mode {}.".format(
                    WeightPalettizer.WEIGHT_PALETTIZATION_MODES, self.mode
                )
            )
            raise ValueError(msg)

        if nbits is None and self.mode in ("KMEANS", "UNIFORM"):
            msg = "nbits must be provided for mode {}".format(mode)
            raise ValueError(msg)

        if nbits is not None and self.mode in ("UNIQUE", "CUSTOM"):
            msg = "nbits must NOT be provided for mode {}".format(mode)
            raise ValueError(msg)

        if self.nbits is not None and self.nbits not in (1, 2, 4, 6, 8):
            raise ValueError("Invalid value of nbits ({}) for palettization. Supported bits are {{1, 2, 4, 6, 8}}".format(nbits))

        if (self.mode == "CUSTOM") ^ (lut_function is not None):
            msg = "lut_function must be None if mode is not custom, and that it cannot be None when the mode is custom."
            raise ValueError(msg)

        if self.mode == "CUSTOM" and not callable(self.lut_function):
            msg = "A function object must be provided as lut_function. Got a lut_functions as type {}".format(type(self.lut_function))
            raise ValueError(msg)

    def is_valid_op(self, op):
        if op.op_type == "const" and should_use_weight_file(op.val.val):
            return True
        return False

    @staticmethod
    def compress(val, mode, nbits=None, lut_function=None):

        mode = mode.upper()

        def compress_kmeans(val, nbits):
            lut, indices = _get_kmeans_lookup_table_and_weight(nbits, val)
            lut = lut.astype(val.dtype)
            indices = indices.astype(np.uint8)
            return lut, indices

        def compress_uniform(val, nbits):
            val = val.flatten()
            val_min = np.amin(val)
            val_max = np.amax(val)
            scale = (val_max - val_min) / ((1 << nbits) - 1)
            indices = np.round(
                ((val - val_min) / (val_max - val_min)) * ((1 << nbits) - 1)
            ).astype(np.uint8)
            lut = np.array(range(0, 1 << nbits)) * scale + val_min
            lut = lut.astype(val.dtype)
            return lut, indices

        def get_nbits_for_unique_mode(val):
            val = val.flatten()
            unique_vals = np.unique(val).tolist()
            for nbits in (1, 2, 4, 6, 8):
                if len(unique_vals) <= 1 << nbits:
                    return nbits
            msg = "weight value cannot be represented in an 8 bits palettization. Skipped."
            logger.warning(msg)
            return None

        def compress_unique(val, nbits):
            val = val.flatten()
            unique_vals = np.unique(val).tolist()
            if len(unique_vals) > 1 << nbits:
                msg = "Too many unique values {} in the weight. Couldn't represented in {} bits.".format(len(unique_vals), nbits)
                raise ValueError(msg)
            lut = [0] * (1 << nbits)
            lut[:len(unique_vals)] = unique_vals
            indices = np.zeros((len(val),))
            for i, k in enumerate(lut):
                indices += (i+1) * (val == k).astype(np.int32)
            indices = indices - 1
            assert len(np.where(indices == -1)[0]) == 0, "weight must be corresponding to one existing indice"

            lut = np.array(lut).astype(val.dtype)
            indices = indices.astype(np.uint8)
            return lut, indices

        def pack_indices_into_bytes_array(indices, nbits):
            bitarray = np.unpackbits(indices.reshape(-1, 1), bitorder="little", axis=-1)[
                :, :nbits
            ]
            return np.packbits(bitarray.flatten(), bitorder="little")

        def check_lut_parameters_are_valid(val, lut, indices):
            if not isinstance(lut, np.ndarray) or not isinstance(indices, np.ndarray):
                raise ValueError("LUT and indices must be type of numpy array.")

            if indices.size != val.size:
                msg = "Indices size ({}) mismatched with the original weight({}).".format(indices.size, val.size)
                raise ValueError(msg)

            if len(indices.shape) != 1 or indices.dtype != np.uint8:
                msg = "Indices must be a numpy vector of type uint8. Found shape {} with type {}".format(indices.shape, indices.dtype)
                raise ValueError(msg)

            if lut.dtype != val.dtype:
                msg = "Dtype mismatched between LUT ({}) and weight ({})".format(lut.dtype, val.dtype)
                raise ValueError(msg)

        if not isinstance(val, (np.ndarray, np.generic)):
            raise ValueError("Only numpy arrays are supported")

        if mode == "KMEANS":
            lut, indices = compress_kmeans(val, nbits)
        elif mode == "UNIFORM":
            lut, indices = compress_uniform(val, nbits)
        elif mode == "UNIQUE":
            nbits = get_nbits_for_unique_mode(val)
            if nbits is None:
                return None
            lut, indices = compress_unique(val, nbits)
        elif mode == "CUSTOM":
            lut, indices = lut_function(val)

        check_lut_parameters_are_valid(val, lut, indices)

        params = LutParams()
        params.lut = lut
        params.shape = val.shape
        params.indices = pack_indices_into_bytes_array(indices, int(np.log2(lut.shape[0])))

        return params

    @staticmethod
    def decompress(params):
        if not isinstance(params, LutParams):
            raise ValueError("Invalid type of params")
        return constexpr_lut_to_dense.decompress(params.lut, params.indices, params.shape)

    def transform_op(self, op):
        block = op.enclosing_block
        lut_params = self.compress(op.val.val, self.mode, self.nbits, self.lut_function)

        if lut_params is None:
            return

        if not self.fake_compression:
            new_var = mb.constexpr_lut_to_dense(
                indices=lut_params.indices,
                lut=lut_params.lut,
                shape=np.uint32(lut_params.shape),
                before_op=op,
                name=op.name + "_palettized",
            )
        else:
            decompressed_val = self.decompress(lut_params)
            new_var = mb.const(
                val=decompressed_val,
                before_op=op,
                name=op.name + "_fake_palettized",
            )

        op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=new_var,
            no_check_var_types=True,
        )

        block.remove_ops([op])


class AffineQuantParams:
    def __init__(self, quantized_data=None, zero_point=None, scale=None, axis=None):
        self.quantized_data = quantized_data
        self.zero_point = zero_point
        self.scale = scale
        self.axis = axis


class WeightAffineQuantizer(AbstractQuantizationPass):
    """
    This transform does the following, for each const op and if the "op_selector" return True:
    - Values are linearly quantized into unsigned 8-bits.
    - If fake_compression=False,  compressed value is encoded via constexpr_affine_dequantize op
    - If fake_compression=True,   compressed value is decompressed and then encoded via const op
    - Old const is replaced by a newly created operation.
    """
    WEIGHT_AFFINE_QUANTIZATION_MODES = ["LINEAR_SYMMETRIC", "LINEAR"]
    def __init__(self, fake_compression=False, op_selector=None, mode="linear"):
        super().__init__(op_selector=op_selector)
        self.fake_compression = fake_compression
        self.mode = mode.upper()

        if not self.mode in WeightAffineQuantizer.WEIGHT_AFFINE_QUANTIZATION_MODES:
            msg = "Only mode {} supported for weight affine quantization. Got mode {}.".format(
                WeightAffineQuantizer.WEIGHT_AFFINE_QUANTIZATION_MODES, self.mode
            )
            raise ValueError(msg)

    def is_valid_op(self, op):
        if op.op_type == "const" and should_use_weight_file(op.val.val):
            return True
        return False

    @staticmethod
    def _get_axis(op):
        axis = 0
        var = op.outputs[0]
        if len(var.child_ops) == 1 and var.child_ops[0].op_type == "conv_transpose":
            axis = 1
        return axis

    @staticmethod
    def compress(val, axis, mode):
        mode = mode.upper()
        if not isinstance(val, (np.ndarray, np.generic)):
            raise ValueError("Only numpy arrays are supported")

        params = AffineQuantParams()
        axes = tuple([i for i in range(len(val.shape)) if i != axis])
        val_min = np.amin(val, axis=axes, keepdims=True)
        val_max = np.amax(val, axis=axes, keepdims=True)
        val_range = 255

        if mode == "LINEAR_SYMMETRIC":
            # For the linear_symmetric mode, the range is symmetrical to 0
            max_abs = np.maximum(np.abs(val_min), np.abs(val_max))
            val_min = -max_abs
            val_max = max_abs
            val_range = 254

        params.scale = (val_max - val_min) / val_range
        params.scale = params.scale.astype(val.dtype).squeeze()
        params.quantized_data = np.round(
            ((val - val_min) / (val_max - val_min)) * val_range
        ).astype(np.uint8)
        params.zero_point = (
            np.round((-val_min / (val_max - val_min)) * val_range).astype(np.uint8).squeeze()
        )
        params.axis = axis
        return params

    @staticmethod
    def decompress(params):
        if not isinstance(params, AffineQuantParams):
            raise ValueError("Invalid type of params")
        return constexpr_affine_dequantize.decompress(params.quantized_data, params.zero_point, params.scale, params.axis)

    def transform_op(self, op):
        block = op.enclosing_block
        quant_params = self.compress(op.val.val, self._get_axis(op), self.mode)

        if not self.fake_compression:
            new_var = mb.constexpr_affine_dequantize(
                quantized_data=quant_params.quantized_data,
                zero_point=quant_params.zero_point,
                scale=quant_params.scale,
                axis=quant_params.axis,
                before_op=op,
                name=op.name + "_affine_quantized",
            )
        else:
            decompressed_val = self.decompress(quant_params)
            new_var = mb.const(
                val=decompressed_val,
                before_op=op,
                name=op.name + "_fake_affine_quantized",
            )

        op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=new_var,
            no_check_var_types=True,
        )

        block.remove_ops([op])


class WeightDecompressor(AbstractQuantizationPass):
    """
    This graph pass transforms the constexpr ops back into mb.const op.
    constexpr ops includes:
    (1) constexpr_affine_dequantize
    (2) constexpr_lut_to_dense
    (3) constexpr_sparse_to_dense
    """

    def __init__(self, op_selector):
        super().__init__(op_selector=op_selector)

    def is_valid_op(self, op):
        return op.op_type in ["constexpr_affine_dequantize", "constexpr_lut_to_dense", "constexpr_sparse_to_dense"]

    def transform_op(self, op):
        block = op.enclosing_block

        decompressed_val = op.value_inference()
        new_var = mb.const(
            val=decompressed_val,
            before_op=op,
            name=op.name,
        )

        op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=new_var,
            no_check_var_types=True,
            force_replace=True,
        )

        block.remove_ops([op])
