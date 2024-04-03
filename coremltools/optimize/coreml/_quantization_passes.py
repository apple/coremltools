# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Callable, Optional, Tuple

import numpy as np
from tqdm import tqdm

import coremltools.converters.mil.frontend._utils as frontend_utils
from coremltools import _logger as logger
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.backend.mil.load import should_use_weight_file
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
from coremltools.converters.mil.mil.var import Var
from coremltools.models._deprecation import deprecated as _deprecated
from coremltools.models.neural_network.quantization_utils import _get_kmeans_lookup_table_and_weight
from coremltools.optimize.coreml import _utils
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
    def config(self) -> OptimizationConfig:
        return self._config

    @config.setter
    def config(self, value: OptimizationConfig):
        self._check_config_type(value)
        self._config = value
        if value._op_selector is not None:
            self.op_selector = value._op_selector

    def need_compress_const(
        self, op: Operation, _is_deprecated: bool, weight_threshold: float
    ) -> bool:
        """
        The utility function is checking whether a const op can be compressed.
        If ``_is_deprecated = True``, the user is using the ``ct.compression_utils``, in which the ops are already filtered by ``op_selector``.
        For the new ``ct.optimize.coreml`` API, ``op_selector`` is no longer supported, so the ``weight_threshold`` is checked explicitly instead.
        """
        val = self._get_const_value(op)
        if _is_deprecated and weight_threshold != None:
            raise ValueError("weight_threshold cannot be set through the deprecated ct.compression_util API")

        if _is_deprecated:
            return should_use_weight_file(val)

        if not self._validate_child_constexpr_for_compress(op):
            return False

        if weight_threshold is None:
            raise ValueError("weight_threshold cannot be None")

        return (
            should_use_weight_file(val) and self._get_weight_to_compress_size(op) > weight_threshold
        )

    def _validate_child_constexpr_for_compress(self, op: Operation) -> bool:
        """Check if child constexpr ops support current op to be compressed."""
        for child_op in op.outputs[0].child_ops:
            if child_op.op_type.startswith("constexpr_"):
                # Const fed into constexpr_ ops cannot be further compressed.
                return False
        return True

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

    @staticmethod
    def select_input_output_channel_axis(op: Operation) -> Tuple[int, int]:
        """
        Here are some representative ops:
        - linear: [D_out, D_in]
        - matmul's y: [..., D_in, D_out] if transpose_y is False, else [..., D_out, D_in]
        - conv: [C_out, C_in_div_group, KH, KW]
        - conv_transpose: [C_in, C_out_div_group, KH, KW]

        The input output channel axis selection criteria is:
        - For conv_transpose the output channel is 1 and input channel is 0.
        - For matmul's y:
            - When transpose_y=False, output channel is -1 and input channel is -2
            - When transpose_y=True, output channel is -2 and input channel is -1
        - For all other ops, output channel is 0 and input channel is 1.
        """
        output_channel_axis, input_channel_axis = 0, 1
        var = op.outputs[0]
        if len(var.child_ops) == 1:
            child_op = var.child_ops[0]
            if child_op.op_type == "conv_transpose":
                output_channel_axis = 1
                input_channel_axis = 0
            if child_op.op_type == "matmul" and child_op.y == var:
                if child_op.transpose_y.val:
                    output_channel_axis = -2
                    input_channel_axis = -1
                else:
                    output_channel_axis = -1
                    input_channel_axis = -2
            if child_op.op_type.startswith("constexpr_"):
                return AbstractCompressionPass.select_input_output_channel_axis(child_op)
        return input_channel_axis, output_channel_axis

    def is_valid_op(self, op: Operation):
        if op.op_type == "const" and should_use_weight_file(self._get_const_value(op)):
            return True
        return False

    def _get_const_value(self, op: Operation) -> np.ndarray:
        if op.op_type != "const":
            raise ValueError(f"The op {op} is not a const")
        return op.outputs[0].val

    def _get_weight_to_compress_size(self, op: Operation) -> int:
        if op.op_type != "const":
            raise ValueError("Only const weight can be compressed")
        return np.prod(op.outputs[0].shape)


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
            9. If ``dim = 1``, transpose the tensor back to the original layout.
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
        2. If ``axis = 1``, transpose the tensor to shape ``[*K, C_out, C_in]``; otherwise, ``(axis = 0)`` to ``[*K, C_in, C_out]``.
        3. For the case of ``axis = 1``, reshape input to a 2D tensor ``[*K*C_out, C_in]``. Similar for ``axis = 0``.
        4. Pad the last dimension with ``0`` so that it can be divided by ``m``: ``[*K*C_out, C_in_pad]``.
        5. Reshape the tensor to have the last dimension ``m``: ``[*K*C_out*C_in_pad//m, m]``.
        6. For each vector of length ``m``, we set the lowest ``n`` magnitute elements to ``0``.
        7. Reshape the tensor back to the shape of ``[*K*C_out, C_in_pad]``.
        8. Crop the last dimension to match the original shape of ``[*K*C_out, C_in]``.
        9. Reshape the tensor to shape ``[*K, C_out, C_in]``.
        10. Transpose the tensor back to ``[C_out, C_in, K]``.
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

    @staticmethod
    def _create_constexpr_var(op: Operation, sparse_params: SparseParams) -> Var:
        return mb.constexpr_sparse_to_dense(
            nonzero_data=sparse_params.nonzero_data,
            mask=sparse_params.mask,
            shape=np.uint32(sparse_params.shape),
            before_op=op,
            name=op.name + "_sparsified",
        )

    def transform_op(self, op: Operation):
        op_config = self.config._get_const_op_config(op)
        if op_config is None:
            return
        if not self.need_compress_const(op, self.config._is_deprecated, op_config.weight_threshold):
            return

        const_val = self._get_const_value(op)
        if not isinstance(const_val, (np.ndarray, np.generic)):
            raise ValueError("Only numpy arrays are supported")

        if isinstance(op_config, OpThresholdPrunerConfig):
            sparse_params = self.compress_by_threshold(
                val=const_val,
                threshold=op_config.threshold,
                minimum_sparsity_percentile=op_config.minimum_sparsity_percentile,
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
                    val=const_val,
                    target_sparsity=op_config.target_sparsity,
                    block_size=op_config.block_size,
                    dim=op_config.dim,
                )
            elif op_config.n_m_ratio is not None:
                sparse_params = self.compress_by_nm_sparsity(
                    val=const_val,
                    n_m_ratio=op_config.n_m_ratio,
                    dim=op_config.dim,
                )

        if sparse_params is None:
            return

        if not self.fake_compression:
            new_var = self._create_constexpr_var(op, sparse_params)
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
    _SUPPORTED_NBITS = (1, 2, 4, 6, 8)

    @staticmethod
    def _get_nbits_for_unique_mode(val: np.ndarray, allowed_nbits: Tuple[int, ...]) -> int:
        """
        Try each nbit in allowed_nbits to find one that can represent number of unique values in val.

        Note that the values in `allowed_nbits` need to be in ascending order.
        """
        val = val.flatten()
        unique_vals = np.unique(val).tolist()
        for nbits in allowed_nbits:
            if len(unique_vals) <= 1 << nbits:
                return nbits
        raise ValueError(
            f"Unique values in weight cannot be represented by {allowed_nbits[-1]} "
            "bits palettization."
        )

    @staticmethod
    def _get_lut_and_indices(
        val: np.ndarray, mode: str, nbits: Optional[int], lut_function: Optional[Callable]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate look-up-table (LUT) and indices."""
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

        if mode == "KMEANS":
            lut, indices = compress_kmeans(val, nbits)
        elif mode == "UNIFORM":
            lut, indices = compress_uniform(val, nbits)
        elif mode == "UNIQUE":
            if nbits is None:
                nbits = palettize_weights._get_nbits_for_unique_mode(
                    val, palettize_weights._SUPPORTED_NBITS
                )
            lut, indices = compress_unique(val, nbits)
        else:
            if mode != "CUSTOM":
                raise AssertionError(f"Invalid mode {mode}")
            lut, indices = lut_function(val)

        return lut, indices

    @staticmethod
    def compress(val, mode, nbits=None, lut_function=None) -> LutParams:
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

        lut, indices = palettize_weights._get_lut_and_indices(val, mode, nbits, lut_function)

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

    @staticmethod
    def _create_constexpr_var(op: Operation, lut_params: LutParams) -> Var:
        return mb.constexpr_lut_to_dense(
            indices=lut_params.indices,
            lut=lut_params.lut,
            shape=np.uint32(lut_params.shape),
            before_op=op,
            name=op.name + "_palettized",
        )

    def transform_op(self, op: Operation):
        op_config = self.config._get_const_op_config(op)
        if op_config is None:
            return
        if not self.need_compress_const(op, self.config._is_deprecated, op_config.weight_threshold):
            return

        if op_config.mode == "UNIQUE":
            try:
                palettize_weights._get_nbits_for_unique_mode(
                    op.outputs[0].val, self._SUPPORTED_NBITS
                )
            except ValueError as e:
                logger.warning(f"Skip op {op.name} for palettization, because {e}")
                return

        lut_params = self.compress(
            op.outputs[0].val,
            op_config.mode,
            op_config.nbits,
            op_config.lut_function
        )

        if not self.fake_compression:
            new_var = palettize_weights._create_constexpr_var(op, lut_params)
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
    _MODE_DTYPE_TO_RANGE = {
        (types.int8, "LINEAR"): (-128, 127),
        (types.int8, "LINEAR_SYMMETRIC"): (-127, 127),
        (types.uint8, "LINEAR"): (0, 255),
        (types.uint8, "LINEAR_SYMMETRIC"): (0, 254),
    }

    @classmethod
    @_deprecated(
        suffix="Please use _utils.quantize_weight",
        version="8.0",
        obj_prefix="coremltools.optimize.coreml._quantization_passes.",
    )
    def _get_quantized_data(
        cls, original_data: np.ndarray, axes: Tuple[int, ...], mode: str, dtype: type
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """[Deprecated] Get quantized data along with metadata (scale, zero_point)."""
        if not np.issubdtype(original_data.dtype, np.floating):
            raise ValueError("Only floating numpy arrays are supported.")

        val_min = np.amin(original_data, axis=axes, keepdims=True)
        val_max = np.amax(original_data, axis=axes, keepdims=True)

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

        q_val_min, q_val_max = cls._MODE_DTYPE_TO_RANGE[(dtype, mode)]
        np_dtype = nptype_from_builtin(dtype)
        zero_point = None
        if mode == "LINEAR_SYMMETRIC":
            if dtype.is_unsigned():
                zero_point_shift = q_val_max // 2
                zero_point = zero_point_shift * np.ones(val_min.shape)
        else:
            assert mode == "LINEAR"
            zero_point = (q_val_min * val_max - q_val_max * val_min) / (val_max - val_min)
            zero_point = np.round(zero_point)
            zero_point = np.clip(zero_point, q_val_min, q_val_max)

        scale = (val_max - val_min) / (q_val_max - q_val_min)
        quantized_data = np.round(original_data / scale)
        if zero_point is not None:
            quantized_data += zero_point
            zero_point = zero_point.squeeze().astype(np_dtype)
        quantized_data = np.clip(quantized_data, q_val_min, q_val_max).astype(np_dtype)
        scale = scale.astype(original_data.dtype).squeeze()

        return quantized_data, scale, zero_point

    @classmethod
    def compress(cls, val: np.ndarray, axis: int, mode: str, dtype: type) -> AffineQuantParams:
        if not isinstance(val, (np.ndarray, np.generic)):
            raise ValueError("Only numpy arrays are supported")
        if isinstance(dtype, np.dtype):
            dtype = types.numpy_type_to_builtin_type(dtype)
        if not types.is_builtin(dtype):
            raise ValueError(f"The input dtype is should be a built-in type, but got {type(dtype)}")

        axes = tuple([i for i in range(len(val.shape)) if i != axis])
        quantized_data, scale, zero_point = _utils.quantize_weight(
            val,
            axes,
            nbits=dtype.get_bitwidth(),
            signed=not dtype.is_unsigned(),
            quantization_mode=mode,
            dtype=types.nptype_from_builtin(dtype),
        )

        if zero_point is None:
            # The iOS16 constexpr_affine_dequantize op requires zero_point.
            zero_point = np.zeros_like(scale).astype(quantized_data.dtype)
        return AffineQuantParams(quantized_data, zero_point, scale, axis)

    @staticmethod
    def decompress(params: AffineQuantParams) -> np.ndarray:
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

        output_channel = self.select_input_output_channel_axis(op)[1]
        quant_params = self.compress(
            op.outputs[0].val, output_channel, op_config.mode, op_config.dtype
        )

        if not self.fake_compression:
            new_var = frontend_utils._construct_constexpr_affine_op(
                quant_params.quantized_data,
                quant_params.zero_point,
                quant_params.scale,
                quant_params.axis,
                name=op.name + "_affine_quantized",
                before_op=op,
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
    The ``constexpr`` op has op_type starts with the "constexpr_" prefix.
    """

    def __init__(self, op_selector):
        super().__init__(op_selector=op_selector)

    def is_valid_op(self, op):
        return op.op_type is not None and op.op_type.startswith("constexpr_")

    def transform_op(self, op):
        decompressed_val = op.materialized_val_inference()

        if not isinstance(decompressed_val, (list, tuple)):
            decompressed_val = [decompressed_val]

        if len(decompressed_val) != len(op.outputs):
            raise ValueError(
                "The number of decompressed value should match the number of op outputs. "
                f"But got {len(decompressed_val)} vs {len(op.outputs)}"
            )

        for decomp_val, output_var in zip(decompressed_val, op.outputs):
            new_const = mb.const(val=decomp_val, before_op=op, name=op.name)
            op.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=op,
                old_var=output_var,
                new_var=new_const,
                no_check_var_types=True,
                force_replace=True,
            )

        op.enclosing_block.remove_ops([op])
