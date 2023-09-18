# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from tqdm import tqdm

from coremltools import _logger as logger
from coremltools.converters.mil.backend.mil.load import should_use_weight_file
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Program, types
from coremltools.converters.mil.mil.block import is_current_opset_version_compatible_with
from coremltools.converters.mil.mil.ops.defs._utils import pack_elements_into_bits
from coremltools.converters.mil.mil.ops.defs.iOS16 import (
    constexpr_affine_dequantize,
    constexpr_lut_to_dense,
    constexpr_sparse_to_dense,
)
from coremltools.converters.mil.mil.passes.defs.quantization import AbstractQuantizationPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.type_mapping import nptype_from_builtin
from coremltools.models.neural_network.quantization_utils import _get_kmeans_lookup_table_and_weight
from coremltools.optimize.coreml._config import (
    OpLinearQuantizerConfig,
    OpMagnitudePrunerConfig,
    OpPalettizerConfig,
    OpThresholdPrunerConfig,
    OptimizationConfig,
)

"""
--------------------------------
Compression parameters wrapper -
--------------------------------
"""
class SparseParams:
    def __init__(self, nonzero_data=None, mask=None, shape=None):
        self.nonzero_data = nonzero_data
        self.mask = mask
        self.shape = shape

class LutParams:
    def __init__(self, lut=None, indices=None, shape=None):
        self.lut = lut
        self.indices = indices
        self.shape = shape

class AffineQuantParams:
    def __init__(self, quantized_data=None, zero_point=None, scale=None, axis=None):
        self.quantized_data = quantized_data
        self.zero_point = zero_point
        self.scale = scale
        self.axis = axis

"""
------------------------
Compression graph pass -
------------------------
"""
class AbstractCompressionPass(AbstractQuantizationPass):
    """
    The abstract class for the compression graph passes.
    """
    _MINIMUM_OPSET_VERSION = AvailableTarget.iOS16

    def __init__(self, config: OptimizationConfig = None, fake_compression: bool = False):
        if not isinstance(config, (OptimizationConfig, type(None))):
            raise ValueError(f"config must be of type OptimizationConfig. Got {type(config)}.")

        op_selector = None if config is None else config._op_selector

        super().__init__(op_selector=op_selector)

        self.fake_compression = fake_compression
        self._config = config
        if config is not None:
            self._check_config_type(config)

    def apply(self, prog):
        if not isinstance(prog, Program):
            raise TypeError('Transform "{}" can only be applied on PyMIL programs.'.format(self))

        @block_context_manager
        def apply_block(block):
            if not is_current_opset_version_compatible_with(self._MINIMUM_OPSET_VERSION):
                logger.warning(
                    f"The program's opset is not compatible with {self._MINIMUM_OPSET_VERSION}. "
                    f"Skipped the compression pass {self.__class__}.")
                return

            valid_consts = []
            for op in list(block.operations):
                for b in op.blocks:
                    apply_block(b)

                if self.is_valid_op(op):
                    need_transform = True
                    if self.op_selector is not None:
                        need_transform = self.op_selector(op)

                    if need_transform:
                        valid_consts.append(op)

            for op in tqdm(
                valid_consts,
                desc=f"Running compression pass {self.__class__.__name__}",
                unit=" ops",
            ):
                self.transform_op(op)

        for f in prog.functions.values():
            apply_block(f)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._check_config_type(value)
        self._config = value

    @staticmethod
    def need_compress_const(op: Operation, _is_deprecated: bool, weight_threshold: float):
        """
        The utility function is checking whether a const op can be compressed.
        If ``_is_deprecated = True``, the user is using the ``ct.compression_utils``, in which the ops are already filtered by ``op_selector``.
        For the new ``ct.optimize.coreml`` API, ``op_selector`` is no longer supported, so the ``weight_threshold`` is checked explicitly instead.
        """
        val = op.outputs[0].val
        if _is_deprecated and weight_threshold != None:
            raise ValueError("weight_threshold cannot be set through the deprecated ct.compression_util API")

        if _is_deprecated:
            return should_use_weight_file(val)

        # const fed into constexpr ops cannot be compressed
        if any([child_op.op_type.startswith("constexpr") for child_op in op.outputs[0].child_ops]):
            return False

        if weight_threshold is None:
            raise ValueError("weight_threshold cannot be None")

        return should_use_weight_file(val) and val.size > weight_threshold

    def _check_config_type(self, config: OptimizationConfig):
        """
        The utility function is checking the OptimizationConfig is holding correct type of op config.
        """
        def get_supported_types_as_str(supported_type):
            if not isinstance(supported_type, (tuple, list)):
                supported_type = [supported_type]
            return ", ".join([f"{val.__name__}" for val in supported_type])

        all_configs = []
        if config.global_config is not None:
            all_configs.append(config.global_config)
        all_configs.extend(list(config.op_type_configs.values()))
        all_configs.extend(list(config.op_name_configs.values()))

        for config in all_configs:
            if not isinstance(config, self._SUPPORTED_CONFIG_TYPE) and config is not None:
                supported_type_str = get_supported_types_as_str(self._SUPPORTED_CONFIG_TYPE)
                raise ValueError(f"{self.__class__.__name__} only accept {supported_type_str} type config. Got {config.__class__.__name__}.")

@register_pass(namespace="compression")
class prune_weights(AbstractCompressionPass):
    """
    This transform works for each ``const`` op if:

    - ``_is_deprecated=True`` and the ``op_selector`` returns ``True``.
    - ``_is_deprecated=False`` and the ``const`` value size ``> weight_threshold``.

    The transform performs the following:

    - The fraction of values with the least absolute value are zeroed out (self.sparsity).
    - If ``fake_compression=False``, the zeroed-out value is encoded using the ``constexpr_sparse_to_dense`` op.
    - If ``fake_compression=True``, the zeroed-out value is encoded using the ``const`` op.
    - Old ``const`` is replaced by a new operation with zeroed-out value.
    """
    _SUPPORTED_CONFIG_TYPE = (OpMagnitudePrunerConfig, OpThresholdPrunerConfig)

    def is_valid_op(self, op: Operation):
        if op.op_type == "const" and should_use_weight_file(op.outputs[0].val):
            return True
        return False

    @staticmethod
    def _pack_val_to_sparse_param(val):
        flattened_val = val.flatten()
        params = SparseParams()
        params.nonzero_data = flattened_val[np.where(flattened_val != 0)]
        params.mask = np.packbits(np.where(flattened_val != 0, 1, 0), bitorder="little")
        params.shape = val.shape
        return params

    @staticmethod
    def compress_by_threshold(val, threshold, minimum_sparsity_percentile):
        val = np.where(np.abs(val) <= threshold, 0, val)
        sparsity_percentile = np.sum(val == 0.0) / val.size
        if sparsity_percentile < minimum_sparsity_percentile:
            msg = (f"weight value has sparsity of {sparsity_percentile} < "
                   f"minimum_sparsity_percentile {minimum_sparsity_percentile}. Skipped."
                  )
            logger.warning(msg)
            return None
        return prune_weights._pack_val_to_sparse_param(val)

    @staticmethod
    def compress_by_magnitude(val, target_sparsity, block_size=None, dim=None):
        def _apply_block_sparsity(val, block_size, dim):
            shape = val.shape
            rank = len(shape)
            assert dim in [0, 1], "bock sparsity pruning only supports dim [0, 1]."
            assert rank in [2, 3, 4, 5], "block sparsity only supports weights of rank [2, 3, 4, 5]"
            """
            Block sparsity follows these steps:

            1. Input tensor with shape of ``[C_out, Cin, *K]``.
            2. If ``dim = 1``, the tensor is transposed to ``[Cin, C_out, *K]``. The following example assumes ``dim = 0``.
            3. Pad ``C_out`` so that it can be divided by ``block_size``: ``[C_out_pad, Cin, *K]``.
            4. Divide the output channel by ``block_size`` and reshape: ``[C_out_pad // block_size, block_size, C_in, *K]``.
            5. Compute the magnitude for each block: ``[C_out_pad // block_size, 1, C_in, *K]``.
            6. Replicate the magnitude values for each block: ``[C_out_pad // block_size, block_size, C_in, *K]``.
            7. Reshape the tensor back to ``[Cout_pad, C_in, *K]``.
            8. Crop the tensor to ``[C_out, C_in, *K]``.
            9. If ``dim = 1``, tranpose the tensor back to the original layout.
            """
            if dim == 1:
                perm = [1, 0] + list(range(2, rank))
                val = np.transpose(val, axes=perm)

            channel = val.shape[0]
            if channel % block_size != 0:
                pad_size = block_size - channel % block_size
                pad_value = [(0, pad_size)] + [(0, 0)] * (rank - 1)
                val = np.pad(val, pad_value)
            shape_padded = val.shape
            assert shape_padded[0] % block_size == 0

            new_shape = list(shape_padded)
            new_shape.insert(1, block_size)
            new_shape[0] = new_shape[0] // block_size
            val = np.reshape(val, (new_shape))

            val = val * val
            val = np.sum(val, axis=1, keepdims=True)
            val = np.sqrt(val)

            reps = [1] * (rank + 1)
            reps[1] = block_size
            val = np.tile(val, reps)
            val =  np.reshape(val, shape_padded)
            val = val[:channel]

            if dim == 1:
                val = np.transpose(val, axes=perm)

            return val

        magnitude_map = np.abs(val)
        if block_size is not None:
            channel = magnitude_map.shape[dim]
            if block_size > channel / 2:
                logger.warning(
                    f"block_size > channel / 2 is not applicable for block sparsity. Got block_size = {block_size}, channel = {channel}. Skipped."
                )
                return None

            magnitude_map = _apply_block_sparsity(magnitude_map, block_size, dim)
        q = target_sparsity * 100
        if q == 100:
            val = 0 * val
        elif q != 0:
            val = np.where(magnitude_map <= np.percentile(magnitude_map, q), 0, val)
        return prune_weights._pack_val_to_sparse_param(val)

    @staticmethod
    def compress_by_nm_sparsity(val, n_m_ratio, dim):
        n, m = n_m_ratio
        assert n <= m
        shape = val.shape
        rank = len(shape)
        assert dim in [0, 1], "n:m pruning only supports dim [0, 1]."
        assert rank in [2, 3, 4, 5], "m:m pruning only supports weights of rank [2, 3, 4, 5]"
        """
        The `n-m` pruning process follows these steps:
        1. Input tensor with shape of ``[C_out, C_in, *K]``, where ``K`` is the spatial dimension from ``0`` to ``3``.
        2. If ``axis = 1``, tranpose the tensor to shape ``[*K, C_out, C_in]``; otherwise, ``(axis = 0)`` to ``[*K, C_in, C_out]``.
        3. For the case of ``axis = 1``, reshape input to a 2D tensor ``[*K*C_out, C_in]``. Similar for ``axis = 0``.
        4. Pad the last dimension with ``0`` so that it can be divided by ``m``: ``[*K*C_out, C_in_pad]``.
        5. Reshape the tensor to have the last dimension ``m``: ``[*K*C_out*C_in_pad//m, m]``.
        6. For each vector of length ``m``, we set the lowest ``n`` magnitute elements to ``0``.
        7. Reshape the tensor back to the shape of ``[*K*C_out, C_in_pad]``.
        8. Crop the last dimension to match the original shape of ``[*K*C_out, C_in]``.
        9. Reshape the tensor to shape ``[*K, C_out, C_in]``.
        10. Tranpose the tensor back to ``[C_out, C_in, K]``.
        """
        perm = list(range(2, rank)) + [0, 1]
        if dim == 0:
            perm[-2], perm[-1] = 1, 0
        weight = np.copy(np.transpose(val, axes=perm))
        shape_begin = weight.shape

        weight = np.reshape(weight, (-1, weight.shape[-1]))
        channel = weight.shape[-1]
        if m > channel / 2:
            logger.warning(
                f"m > channel / 2 is not applicable for n:m pruning. Got m = {m}, channel = {channel}. Skipped."
            )
            return None
        if channel % m != 0:
            pad_size = m - channel % m
            weight = np.pad(weight, ((0, 0), (0, pad_size)))
        shape_padded = weight.shape
        assert shape_padded[-1] % m == 0

        weight = np.reshape(weight, (-1, m))
        magnitute = np.abs(weight)
        indices = np.argsort(magnitute, axis=-1)[:, :n]

        n_m_mask = np.zeros(weight.shape).astype(val.dtype)
        np.put_along_axis(n_m_mask, indices, 1.0, axis=-1)
        n_m_mask = np.reshape(n_m_mask, shape_padded)
        n_m_mask = n_m_mask[:, :channel]

        n_m_mask = np.reshape(n_m_mask, shape_begin)
        perm_back = [perm.index(i) for i in range(rank)]
        n_m_mask = np.transpose(n_m_mask, axes=perm_back)

        val = val * (1 - n_m_mask)
        return prune_weights._pack_val_to_sparse_param(val)

    @staticmethod
    def decompress(params):
        if not isinstance(params, SparseParams):
            raise ValueError("Invalid type of params")
        return constexpr_sparse_to_dense.decompress(params.nonzero_data, params.mask, params.shape)

    def transform_op(self, op: Operation):
        op_config = self.config._get_const_op_config(op)
        if op_config is None:
            return
        if not self.need_compress_const(op, self.config._is_deprecated, op_config.weight_threshold):
            return

        if not isinstance(op.outputs[0].val, (np.ndarray, np.generic)):
            raise ValueError("Only numpy arrays are supported")

        if isinstance(op_config, OpThresholdPrunerConfig):
            sparse_params = self.compress_by_threshold(
                                val=op.outputs[0].val,
                                threshold=op_config.threshold,
                                minimum_sparsity_percentile=op_config.minimum_sparsity_percentile
                            )
        elif isinstance(op_config, OpMagnitudePrunerConfig):
            # Structural sparsity can only be applied to conv / linear weight
            # For non applicable constant, we skip the compression,
            # we do allow the user to do structural pruning for non applicable constant,
            # if it is explicitly set by set_op_name,
            if not op_config._check_const_op_is_valid(op):
                if op.name not in self.config.op_name_configs:
                    logger.warning(f"op named {op.name} not applicable for {OpMagnitudePrunerConfig} configuration. Skipped.")
                    return

            if op_config.target_sparsity is not None:
                sparse_params = self.compress_by_magnitude(
                                    val=op.outputs[0].val,
                                    target_sparsity=op_config.target_sparsity,
                                    block_size=op_config.block_size,
                                    dim=op_config.dim,
                                )
            elif op_config.n_m_ratio is not None:
                sparse_params = self.compress_by_nm_sparsity(
                                    val=op.outputs[0].val,
                                    n_m_ratio=op_config.n_m_ratio,
                                    dim=op_config.dim,
                                )

        if sparse_params is None:
            return

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

        op.enclosing_block.remove_ops([op])

@register_pass(namespace="compression")
class palettize_weights(AbstractCompressionPass):
    """
    This transform works for each ``const`` op if:

    - ``_is_deprecated=True`` and the ``op_selector`` returns ``True``.
    - ``_is_deprecated=False`` and the ``const`` value size ``> weight_threshold``.

    The transform performs the following:

    - A linear look-up table (LUT) with 2\ :sup:`nbits` entries is created with values represented by indexing into this LUT.
    - If ``fake_compression=False``, compressed value is encoded using the ``constexpr_lut_to_dense`` op.
    - If ``fake_compression=True``,  compressed value is decompressed and then encoded using the ``const`` op.
    - Old ``const`` op is replaced by a newly created operation.
    """
    _SUPPORTED_CONFIG_TYPE = OpPalettizerConfig

    def is_valid_op(self, op: Operation):
        if op.op_type == "const" and should_use_weight_file(op.outputs[0].val):
            return True
        return False

    @staticmethod
    def compress(val, mode, nbits=None, lut_function=None):

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
            indices = np.round(((val - val_min) / (val_max - val_min)) * ((1 << nbits) - 1)).astype(
                np.uint8
            )
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
                msg = "Too many unique values {} in the weight. Couldn't represented in {} bits.".format(
                    len(unique_vals), nbits
                )
                raise ValueError(msg)
            lut = [0] * (1 << nbits)
            lut[: len(unique_vals)] = unique_vals
            indices = np.zeros((len(val),))
            for i, k in enumerate(lut[:len(unique_vals)]):
                indices += (i + 1) * (val == k).astype(np.int32)
            indices = indices - 1
            assert (
                len(np.where(indices == -1)[0]) == 0
            ), "weight must be corresponding to one existing indice"

            lut = np.array(lut).astype(val.dtype)
            indices = indices.astype(np.uint8)
            return lut, indices

        def check_lut_parameters_are_valid(val, lut, indices):
            if not isinstance(lut, np.ndarray) or not isinstance(indices, np.ndarray):
                raise ValueError("LUT and indices must be type of numpy array.")

            if indices.size != val.size:
                msg = "Indices size ({}) mismatched with the original weight({}).".format(
                    indices.size, val.size
                )
                raise ValueError(msg)

            if len(indices.shape) != 1 or indices.dtype != np.uint8:
                msg = "Indices must be a numpy vector of type uint8. Found shape {} with type {}".format(
                    indices.shape, indices.dtype
                )
                raise ValueError(msg)

            if lut.dtype != val.dtype:
                msg = "Dtype mismatched between LUT ({}) and weight ({})".format(
                    lut.dtype, val.dtype
                )
                raise ValueError(msg)

        if not isinstance(val, (np.ndarray, np.generic)):
            raise ValueError(f"Only numpy arrays are supported. Got {type(val)}")

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
        params.indices = pack_elements_into_bits(indices, int(np.log2(lut.shape[0])))
        return params

    @staticmethod
    def decompress(params):
        if not isinstance(params, LutParams):
            raise ValueError("Invalid type of params")
        return constexpr_lut_to_dense.decompress(params.lut, params.indices, params.shape)

    def transform_op(self, op: Operation):
        op_config = self.config._get_const_op_config(op)
        if op_config is None:
            return
        if not self.need_compress_const(op, self.config._is_deprecated, op_config.weight_threshold):
            return

        lut_params = self.compress(
            op.outputs[0].val,
            op_config.mode,
            op_config.nbits,
            op_config.lut_function
        )

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

        op.enclosing_block.remove_ops([op])

@register_pass(namespace="compression")
class linear_quantize_weights(AbstractCompressionPass):
    """
    This transform works for each ``const`` op if:

    - ``_is_deprecated=True`` and the ``op_selector`` returns ``True``.
    - ``_is_deprecated=False`` and the ``const`` value size ``> weight_threshold``.

    The transform performs the following:

    - Values are linearly quantized into unsigned 8-bits.
    - If ``fake_compression=False``, compressed value is encoded using the ``constexpr_affine_dequantize`` op.
    - If ``fake_compression=True``, compressed value is decompressed and then encoded using the ``const`` op.
    """
    _SUPPORTED_CONFIG_TYPE = OpLinearQuantizerConfig

    def is_valid_op(self, op: Operation):
        if op.op_type == "const" and should_use_weight_file(op.outputs[0].val):
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
    def compress(val, axis, mode, dtype):
        def _ensure_numerical_range_and_cast(val, low, high, np_dtype):
            '''
            For some cases, the computed quantized data might exceed the data range.
            For instance, after rounding and addition, we might get `128` for the int8 quantization.
            This utility function ensures the val in the data range before doing the cast.
            '''
            val = np.minimum(val, high)
            val = np.maximum(val, low)
            return val.astype(np_dtype)

        mode_dtype_to_range = {
            (types.int8, "LINEAR"): (-128, 127),
            (types.int8, "LINEAR_SYMMETRIC"): (-127, 127),
            (types.uint8, "LINEAR"): (0, 255),
            (types.uint8, "LINEAR_SYMMETRIC"): (0, 254),
        }

        if not isinstance(val, (np.ndarray, np.generic)):
            raise ValueError("Only numpy arrays are supported")

        params = AffineQuantParams()
        axes = tuple([i for i in range(len(val.shape)) if i != axis])
        val_min = np.amin(val, axis=axes, keepdims=True)
        val_max = np.amax(val, axis=axes, keepdims=True)

        if mode == "LINEAR_SYMMETRIC":
            # For the linear_symmetric mode, the range is symmetrical to 0
            max_abs = np.maximum(np.abs(val_min), np.abs(val_max))
            val_min = -max_abs
            val_max = max_abs
        else:
            assert mode == "LINEAR"
            # For the linear mode, we need to make sure the data range contains `0`
            val_min = np.minimum(0.0, val_min)
            val_max = np.maximum(0.0, val_max)

        q_val_min, q_val_max = mode_dtype_to_range[(dtype, mode)]

        # Set the zero point to symmetric mode
        np_dtype = nptype_from_builtin(dtype)
        if mode == "LINEAR_SYMMETRIC":
            if dtype == types.int8:
                params.zero_point = (0 * np.ones(val_min.shape)).astype(np.int8)
            else:
                assert dtype == types.uint8
                params.zero_point = (127 * np.ones(val_min.shape)).astype(np.uint8)
        else:
            assert mode == "LINEAR"
            params.zero_point = (q_val_min * val_max - q_val_max * val_min) / (val_max - val_min)
            params.zero_point = np.round(params.zero_point)
            params.zero_point = _ensure_numerical_range_and_cast(params.zero_point, q_val_min, q_val_max, np_dtype)

        # compute the params
        params.scale = (val_max - val_min) / (q_val_max - q_val_min)
        params.scale = params.scale.astype(val.dtype).squeeze()

        params.quantized_data = np.round(
            val * (q_val_max - q_val_min) / (val_max - val_min)
        )
        params.quantized_data = (params.quantized_data + params.zero_point)
        params.quantized_data = _ensure_numerical_range_and_cast(params.quantized_data, q_val_min, q_val_max, np_dtype)

        params.zero_point = params.zero_point.squeeze()
        params.axis = axis

        return params

    @staticmethod
    def decompress(params):
        if not isinstance(params, AffineQuantParams):
            raise ValueError("Invalid type of params")
        return constexpr_affine_dequantize.decompress(
            params.quantized_data, params.zero_point, params.scale, params.axis
        )

    def transform_op(self, op: Operation):
        op_config = self.config._get_const_op_config(op)
        if op_config is None:
            return
        if not self.need_compress_const(op, self.config._is_deprecated, op_config.weight_threshold):
            return

        quant_params = self.compress(op.outputs[0].val, self._get_axis(op), op_config.mode, op_config.dtype)

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

        op.enclosing_block.remove_ops([op])

@register_pass(namespace="compression")
class WeightDecompressor(AbstractQuantizationPass):
    """
    This graph pass transforms the ``constexpr`` op back into ``mb.const`` op.
    The ``constexpr`` op includes:

    - ``constexpr_affine_dequantize``
    - ``constexpr_lut_to_dense``
    - ``constexpr_sparse_to_dense``
    """

    def __init__(self, op_selector):
        super().__init__(op_selector=op_selector)

    def is_valid_op(self, op):
        return op.op_type is not None and op.op_type.startswith("constexpr_")

    def transform_op(self, op):
        decompressed_val = op.materialized_val_inference()
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

        op.enclosing_block.remove_ops([op])
