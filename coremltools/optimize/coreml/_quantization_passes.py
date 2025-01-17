# Copyright (c) 2023, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import atexit
from itertools import repeat
from multiprocessing import Pool
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from coremltools import _logger as logger
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.backend.mil.load import should_use_weight_file
from coremltools.converters.mil.frontend import _utils as frontend_utils
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Program, types
from coremltools.converters.mil.mil.block import is_current_opset_version_compatible_with
from coremltools.converters.mil.mil.ops.defs.iOS16 import constexpr_affine_dequantize
from coremltools.converters.mil.mil.ops.defs.iOS16 import (
    constexpr_lut_to_dense as constexpr_lut_to_dense_ios16,
)
from coremltools.converters.mil.mil.ops.defs.iOS16 import (
    constexpr_sparse_to_dense as constexpr_sparse_to_dense_ios16,
)
from coremltools.converters.mil.mil.ops.defs.iOS18 import (
    constexpr_blockwise_shift_scale,
    constexpr_lut_to_dense,
    constexpr_sparse_to_dense,
)
from coremltools.converters.mil.mil.passes.defs.quantization import AbstractQuantizationPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.var import Var
from coremltools.models._deprecation import deprecated
from coremltools.models.neural_network.quantization_utils import _get_kmeans_lookup_table_and_weight
from coremltools.optimize.coreml import _utils as optimize_utils
from coremltools.optimize.coreml._config import (
    CompressionGranularity,
    OpLinearQuantizerConfig,
    OpMagnitudePrunerConfig,
    OpPalettizerConfig,
    OpThresholdPrunerConfig,
    OptimizationConfig,
)


class AbstractCompressionPass(AbstractQuantizationPass):
    """
    The abstract class for the compression graph passes.
    """
    _MINIMUM_OPSET_VERSION = AvailableTarget.iOS16

    # Graph pass option for setting compression config.
    _config: Optional[OptimizationConfig] = None

    # Graph pass option for enabling joint compressions.
    _joint_compression: bool = False

    @property
    def config(self) -> OptimizationConfig:
        return self._config

    @config.setter
    def config(self, value: OptimizationConfig):
        self._check_config_type(value)
        self._config = value
        if value._op_selector is not None:
            self.op_selector = value._op_selector

    @property
    def joint_compression(self):
        return self._joint_compression

    @joint_compression.setter
    def joint_compression(self, joint_compression: bool):
        if not isinstance(joint_compression, bool):
            raise ValueError(
                f"joint_compression only supports bool, but got {type(joint_compression)}"
            )
        self._joint_compression = joint_compression

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

            if self._joint_compression and not is_current_opset_version_compatible_with(
                AvailableTarget.iOS18
            ):
                raise ValueError(
                    "Joint compression is only supported since iOS18. Please set the "
                    "minimum deployment target to iOS18 if you want to use it."
                )

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

        # Disable 1D tensor compression due to MIL 1D Tensor bug (rdar://113860800).
        if (
            not op.outputs[0].child_ops[0].op_type.startswith("constexpr_")
            and op.outputs[0].rank <= 1
        ):
            return False

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

    def is_valid_op(self, op: Operation):
        if op.op_type == "const" and should_use_weight_file(self._get_const_value(op)):
            return True
        return False

    def _get_const_value(self, op: Operation) -> np.ndarray:
        if op.op_type != "const":
            raise ValueError(f"The op {op} is not a const")
        return op.outputs[0].val

    def _get_weight_to_compress_size(self, op: Operation) -> int:
        """
        For joint compression, the constexpr op is the intermediate compressed result, so we
        need to go along the constexpr op chain to get the op which actually is the weight need
        to be compressed.

        For example, the op could be a const feed into constexpr_lut_to_dense as indices, and the
        constexpr_lut_to_dense is fed into a conv op. In this case, we need to find the original
        weight of the conv op, instead of using the const indices to determine if we want to
        compress the op.
        """
        if not (op.op_type == "const" or op.op_type.startswith("constexpr_")):
            raise ValueError(f"Only support const or constexpr ops, but got {op.op_type}")

        if self.joint_compression:
            for op_output in op.outputs:
                # If the current const/constexpr is used in multiple ops, we do a depth-first
                # search to find the endpoint of the chained const/constexpr ops.
                for child_op in op_output.child_ops:
                    if child_op.op_type.startswith("constexpr_"):
                        return self._get_weight_to_compress_size(child_op)
                    else:
                        # The child op is not constexpr, which means the current op is the real
                        # weight (not intermediate constexpr) that need compression.
                        return np.prod(op.outputs[0].shape)

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

    When the `joint_compression` option is set, for each existing compressed constexpr op, it will
    check if the result is sparse. If the result is sparse, it will replace the constexpr op by the
    corresponding sparse version to support joint compression. More specifically:
    - For quantization, `constexpr_blockwise_shift_scale` is replaced by `constexpr_sparse_blockwise_shift_scale` +
      `constexpr_sparse_to_dense` if the dequantized result is sparse.
    - For palettization, `constexpr_lut_to_dense` is replaced by `constexpr_lut_to_sparse` +
      `constexpr_sparse_to_dense` if the depalettized result is sparse.

    .. code-block::

        Input graph:

            constexpr_blockwise_shift_scale -> downstream op

        Output graph:

            constexpr_sparse_blockwise_shift_scale -> constexpr_sparse_to_dense -> downstream op

    Support Options:

    - ``joint_compression``: Enable joint compression. Similar to blockwise_quantize_weights and
    """
    _SUPPORTED_CONFIG_TYPE = (OpMagnitudePrunerConfig, OpThresholdPrunerConfig)
    # Ops to be further pruned for joint compression.
    _JOINT_SUPPORT_OPS = {"constexpr_blockwise_shift_scale", "constexpr_lut_to_dense"}

    def is_valid_op(self, op: Operation):
        if not self.joint_compression:
            return super().is_valid_op(op)
        if op.op_type in self._JOINT_SUPPORT_OPS and should_use_weight_file(
            self._get_const_value(op)
        ):
            return True
        return False

    def _get_const_value(self, op: Operation) -> np.ndarray:
        if op.op_type == "const" or not self.joint_compression:
            return super()._get_const_value(op)
        elif op.op_type.startswith("constexpr_"):
            # The materialized_val_inference is expensive, so only do it for joint compression, as
            # we need to get the de-compressed value and prune it.
            return op.materialized_val_inference()
        else:
            raise ValueError(f"The op {op} is not a const/constexpr.")

    @staticmethod
    def _produce_sparse_param(val) -> optimize_utils.SparseParamsIos16:
        flattened_val = val.flatten()
        return optimize_utils.SparseParamsIos16(
            nonzero_data=flattened_val[np.where(flattened_val != 0)],
            mask=np.packbits(np.where(flattened_val != 0, 1, 0), bitorder="little"),
            shape=val.shape,
        )

    @staticmethod
    def compress_by_threshold(
        val, threshold, minimum_sparsity_percentile
    ) -> Optional[optimize_utils.SparseParamsIos16]:
        val = np.where(np.abs(val) <= threshold, 0, val)
        sparsity_percentile = np.sum(val == 0.0) / val.size
        if sparsity_percentile < minimum_sparsity_percentile:
            msg = (f"weight value has sparsity of {sparsity_percentile} < "
                   f"minimum_sparsity_percentile {minimum_sparsity_percentile}. Skipped."
                  )
            logger.warning(msg)
            return None
        return prune_weights._produce_sparse_param(val)

    @staticmethod
    def compress_by_magnitude(
        val, target_sparsity, block_size=None, dim=None
    ) -> Optional[optimize_utils.SparseParamsIos16]:
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
        return prune_weights._produce_sparse_param(val)

    @staticmethod
    def compress_by_nm_sparsity(val, n_m_ratio, dim) -> Optional[optimize_utils.SparseParamsIos16]:
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
        return prune_weights._produce_sparse_param(val)

    @staticmethod
    def decompress(
        params: Union[optimize_utils.SparseParamsIos16, optimize_utils.SparseParams]
    ) -> np.ndarray:
        if isinstance(params, optimize_utils.SparseParamsIos16):
            return constexpr_sparse_to_dense_ios16.decompress(
                params.nonzero_data, params.mask, params.shape
            )
        elif isinstance(params, optimize_utils.SparseParams):
            return constexpr_sparse_to_dense.decompress(params.nonzero_data, params.mask)
        else:
            raise ValueError("Invalid type of params")

    @staticmethod
    def _create_constexpr_var(
        op: Operation, sparse_params: optimize_utils.SparseParams, joint_compression: bool = False
    ) -> Var:
        if not is_current_opset_version_compatible_with(AvailableTarget.iOS18):
            sparse_params_ios16 = optimize_utils.ios18_sparse_params_to_ios16(sparse_params)
            return mb.constexpr_sparse_to_dense(
                nonzero_data=sparse_params_ios16.nonzero_data,
                mask=sparse_params_ios16.mask,
                shape=np.uint32(sparse_params_ios16.shape),
                before_op=op,
                name=op.name + "_sparsified",
            )

        mask = sparse_params.mask
        nonzero_data = sparse_params.nonzero_data

        if joint_compression:
            if op.op_type == "constexpr_blockwise_shift_scale":
                mask, nonzero_data = mb.constexpr_sparse_blockwise_shift_scale(
                    data_mask=mask,
                    nonzero_data=op.data.val[mask != 0].flatten(),
                    scale=op.scale,
                    offset=op.offset,
                    before_op=op,
                )
            elif op.op_type == "constexpr_lut_to_dense":
                mask, nonzero_data = mb.constexpr_lut_to_sparse(
                    indices_mask=mask,
                    indices_nonzero_data=op.indices.val[mask != 0].flatten(),
                    lut=op.lut,
                    vector_axis=op.vector_axis,
                    before_op=op,
                )

        return mb.constexpr_sparse_to_dense(
            nonzero_data=nonzero_data,
            mask=mask,
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

        sparse_params: Optional[optimize_utils.SparseParamsIos16] = None
        skip_msg = f"op named {op.name} not applicable for {op_config} configuration. Skipped."
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
                    logger.warning(skip_msg)
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
            logger.warning(skip_msg)
            return

        sparse_params: optimize_utils.SparseParams = optimize_utils.ios16_sparse_params_to_ios18(
            sparse_params
        )

        if not self.fake_compression:
            new_var = self._create_constexpr_var(
                op,
                sparse_params,
                joint_compression=self.joint_compression and op.op_type in self._JOINT_SUPPORT_OPS,
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
            force_replace=True,  # Need force_replace to replace the constexpr.
        )

        op.enclosing_block.remove_ops([op])


@register_pass(namespace="compression")
class palettize_weights(AbstractCompressionPass):
    """
    This transform works for each ``const`` op if:

    - ``_is_deprecated=True`` and the ``op_selector`` returns ``True``.
    - ``_is_deprecated=False`` and the ``const`` value size ``> weight_threshold``.

    The transform performs the following:

    - A linear look-up table (LUT) with 2\\ :sup:`nbits` entries is created with values represented by indexing into this LUT.
    - If ``fake_compression=False``, compressed value is encoded using the ``constexpr_lut_to_dense`` op.
    - If ``fake_compression=True``,  compressed value is decompressed and then encoded using the ``const`` op.
    - Old ``const`` op is replaced by a newly created operation.

    Here is an example for input and output graph of this graph pass:

    .. code-block::

        Input graph:

            const -> downstream op

        Output graph:

            constexpr_lut_to_dense -> downstream op


    Support Options:

    - ``joint_compression``:
        Enable joint compression by quantizing an already compressed model.
        What op could be further quantized is in `_validate_child_constexpr_for_compress`.

        Using pruning + palettization as an example, for each existing ``constexpr_sparse_to_dense``
        op, it tries to palettize the non-sparse elements in the spasified data, which could be
        represented as:


          - For each existing ``constexpr_sparse_to_dense`` op, it tries to palettize the
            non-sparse elements in the spasified data, which could be represented as:


            .. code-block::

                Input graph:

                    sparse weight(fp16) -> constexpr_sparse_to_dense -> dense weight(fp16)

                Output graph:

                    sparse lut(int8) -> constexpr_lut_to_sparse -> sparse weight(fp16) -> constexpr_sparse_to_dense -> dense weight(fp16)

    For details about different palettization schemas, see `OpPalettizerConfig` for more details.
    """
    _SUPPORTED_CONFIG_TYPE = OpPalettizerConfig
    _SUPPORTED_NBITS = (1, 2, 3, 4, 6, 8)

    _compress_pool: Optional[Pool] = None

    def __del__(self):
        if palettize_weights._compress_pool is not None:
            palettize_weights._compress_pool.close()

    def _validate_child_constexpr_for_compress(self, op: Operation) -> bool:
        """
        Determines which pattern supports joint compression.

        In iOS18 joint compression, the quantized/sparsified data could be further palettized.
        For each specific op, we only palettize the specific input:
        - constexpr_sparse_to_dense's nonzero_data
        - constexpr_blockwise_shift_scale's data
        """
        if (
            is_current_opset_version_compatible_with(AvailableTarget.iOS18)
            and self.joint_compression
        ):
            if len(op.outputs[0].child_ops) == 1:
                child_op = op.outputs[0].child_ops[0]
                if (
                    child_op.op_type == "constexpr_sparse_to_dense"
                    and child_op.nonzero_data == op.outputs[0]
                ):
                    return True
                elif (
                    child_op.op_type == "constexpr_blockwise_shift_scale"
                    and child_op.data == op.outputs[0]
                ):
                    return True

        return super()._validate_child_constexpr_for_compress(op)

    @staticmethod
    def _get_nbits_for_unique_mode(
        val: np.ndarray,
        allowed_nbits: Tuple[int, ...],
        cluster_dim: int = 1,
        vector_axis: Optional[int] = None,
    ) -> int:
        """
        Try each nbit in allowed_nbits to find one that can represent number of unique values in val.

        If cluster_dim > 1, it's for vector palettization, where the unique means vector unique.
        The vector_axis is only effective for vector palettization, which indicates on which axis
        the vector is.

        Note that the values in `allowed_nbits` need to be in ascending order.
        """
        if cluster_dim == 1:
            val = val.flatten()
            unique_vals_num = len(np.unique(val))
        else:
            # Vector palettization where each cluster_dim elements form a vector on vector_axis.
            if vector_axis is None:
                raise ValueError("The `vector_axis` must be specified when cluster_dim > 1")
            val = np.swapaxes(val, -1, vector_axis).reshape((-1, cluster_dim))
            unique_vals_num = len(np.unique(val, axis=0))

        for nbits in allowed_nbits:
            if unique_vals_num <= 1 << nbits:
                return nbits
        raise ValueError(
            f"Unique values in weight cannot be represented by {allowed_nbits[-1]} "
            "bits palettization."
        )

    @staticmethod
    def _get_lut_and_indices(
        val: np.ndarray,
        mode: str,
        nbits: Optional[int],
        lut_function: Optional[Callable],
        cluster_dim: int = 1,
        vector_axis: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate look-up-table (LUT) and indices."""

        def compress_kmeans(val, nbits, cluster_dim, vector_axis):
            lut, indices = _get_kmeans_lookup_table_and_weight(
                nbits, val, force_kmeans1d=False, cluster_dim=cluster_dim, vector_axis=vector_axis
            )
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

        def compress_unique(val, nbits, cluster_dim, vector_axis):
            if nbits is None:
                nbits = palettize_weights._get_nbits_for_unique_mode(
                    val,
                    palettize_weights._SUPPORTED_NBITS,
                    cluster_dim,
                    vector_axis,
                )

            if cluster_dim > 1:
                val = optimize_utils.reshape_weight_for_vector_lut(val, cluster_dim, vector_axis)

            val = val.reshape((-1, cluster_dim))
            unique_vals, unique_inverse = np.unique(val, axis=0, return_inverse=True)
            lut = np.zeros((1 << nbits, cluster_dim))
            lut[: len(unique_vals)] = unique_vals
            indices = unique_inverse
            indices = indices.flatten()

            if cluster_dim == 1:
                # Squeeze the last dim to make behaviors back compatible with scalar palettization.
                lut = lut.squeeze(-1)

            return lut.astype(val.dtype), indices.astype(np.uint8)

        if mode == "KMEANS":
            lut, indices = compress_kmeans(val, nbits, cluster_dim, vector_axis)
        elif mode == "UNIFORM":
            if cluster_dim > 1:
                raise NotImplementedError(
                    "Vector palettization (cluster_dim > 1) doesn't support UNIFORM mode."
                )
            lut, indices = compress_uniform(val, nbits)
        elif mode == "UNIQUE":
            lut, indices = compress_unique(val, nbits, cluster_dim, vector_axis)
        else:
            if mode != "CUSTOM":
                raise AssertionError(f"Invalid mode {mode}")
            lut, indices = lut_function(val)

        return lut, indices

    @staticmethod
    @deprecated(
        suffix="Please use coremltools.optimize.coreml.palettize_weights.blockwise_compress",
        version="8.2",
        obj_prefix="coremltools.optimize.coreml.palettize_weights.",
    )
    def compress(val, mode, nbits=None, lut_function=None) -> optimize_utils.LutParamsIos16:
        """
        [Legacy] Per-tensor palletization.

        This API is for backward compatibility only. It's no longer used inside the coremltools.
        It's recommended to use `blockwise_compress` instead, which is more general.
        """
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

        params = optimize_utils.LutParamsIos16(
            lut=lut,
            indices=optimize_utils.pack_elements_into_bits(indices, int(np.log2(lut.shape[0]))),
            shape=val.shape,
        )
        return params

    @staticmethod
    def blockwise_compress(
        original_data: np.ndarray,
        mode: str,
        nbits: Optional[int],
        block_sizes: List[int],
        lut_function: Optional[Callable] = None,
        cluster_dim: int = 1,
        channel_axis: Optional[int] = None,
        num_kmeans_workers: int = 1,
    ) -> Optional[optimize_utils.LutParams]:
        """
        Compress original_data into n-bit representation by palettization.

        Supported nbits: 1, 2, 3, 4, 6, 8
        Supported mode: KMEANS, UNIFORM, UNIQUE, CUSTOM

        block_sizes: Each element is the block size on corresponding axis for original_data.

        cluster_dim: Dimension of each cluster centroid, which is the length of each element in the
            lookup table.

        channel_axis: Only useful for vector palettization (cluster_dim > 1). If not provided, we
            will try to infer it from `block_sizes`.

        Returns None if the weight cannot be compressed (for example, the dim size on an axis is not
        divisible by the corresponding block_size).
        """
        # TODO (rdar://127342739): Support more general blockwise palettization.
        # As general blockwise palettization hasn't been supported yet, we try to infer channel axis
        # and channel group size from block_sizes, and use grouped channelwise palettization instead.
        channel_group_size = 0
        for axis, block_size in enumerate(block_sizes):
            if block_size != 0 and block_size != original_data.shape[axis]:
                if channel_axis is not None and channel_axis != axis:
                    raise NotImplementedError(
                        "General block-wise palettization is not supported. Please use "
                        "'per_grouped_channel' or 'per_tensor' for the 'granularity' in config."
                    )
                channel_axis = axis
                channel_group_size = block_size
        if channel_axis is None:
            if cluster_dim > 1:
                raise ValueError(
                    "Cannot infer channel axis, which is required for vector palettization."
                )
            # Per-tensor compression, just need to pick a dummy axis.
            channel_axis = 0

        return palettize_weights.grouped_channelwise_compress(
            original_data,
            mode,
            nbits,
            channel_axis,
            channel_group_size,
            lut_function,
            cluster_dim,
            num_kmeans_workers,
        )

    @staticmethod
    def grouped_channelwise_compress(
        original_data: np.ndarray,
        mode: str,
        nbits: Optional[int],
        channel_axis: int,
        channel_group_size: int,
        lut_function: Optional[Callable] = None,
        cluster_dim: int = 1,
        num_kmeans_workers: int = 1,
    ) -> Optional[optimize_utils.LutParams]:
        """
        Compress original_data into n-bit representation by grouped channelwise palettization.

        Supported nbits: 1, 2, 3, 4, 6, 8
        Supported mode: KMEANS, UNIFORM, UNIQUE, CUSTOM

        block_sizes: Each element is the block size on corresponding axis for original_data.

        cluster_dim: Dimension of each cluster centroid, which is the length of each element in the
            lookup table.

        Returns None if the weight cannot be compressed (for example, the dim size on an axis is not
        divisible by the corresponding channel_group_size).
        """
        if not isinstance(original_data, np.ndarray):
            raise ValueError(f"Only numpy arrays are supported, but got {type(original_data)}")
        if nbits is not None and nbits not in palettize_weights._SUPPORTED_NBITS:
            raise ValueError(
                f"Invalid nbits. Support {palettize_weights._SUPPORTED_NBITS}, but got {nbits}"
            )
        data_rank = len(original_data.shape)
        if not (-data_rank <= channel_axis < data_rank):
            raise ValueError(
                "Invalid channel_axis. Should be in range "
                f"[{-data_rank}, {data_rank}), but got {channel_axis}"
            )

        if channel_axis < 0:
            channel_axis += len(original_data.shape)

        channel_num = original_data.shape[channel_axis]
        if channel_group_size == 0:
            channel_group_size = channel_num
        if channel_num % channel_group_size != 0:
            logger.warning(
                f"Can't perform palettization: The number of channels at {channel_axis}th axis "
                f"({channel_num}) is not divisible by channel_group_size ({channel_group_size})."
            )
            return None
        channel_group_num = channel_num // channel_group_size

        if channel_group_size % cluster_dim != 0:
            logger.warning(
                f"Can't perform palettization: The channel_group_size at {channel_axis}th axis "
                f"({channel_group_size}) is not divisible by cluster_dim ({cluster_dim})."
            )
            return None

        if channel_axis != 0:
            original_data = np.swapaxes(original_data, 0, channel_axis)
        grouped_channel_data = np.split(original_data, channel_group_num, axis=0)

        # As the channel axis has been swapped to 0th axis, use 0 for vector_axis.
        vector_axis = 0

        # If mode is UNIQUE, infer nbits from the number of unique values in each group.
        if mode.upper() == "UNIQUE":
            try:
                for per_group_data in grouped_channel_data:
                    per_group_nbits = palettize_weights._get_nbits_for_unique_mode(
                        per_group_data, palettize_weights._SUPPORTED_NBITS, cluster_dim, vector_axis
                    )
                    # Pick the largest per-channel nbits to be used as the nbits for the whole op.
                    if nbits is None or per_group_nbits > nbits:
                        nbits = per_group_nbits
            except ValueError as e:
                logger.warning(f"Can't perform palettization:{e}")
                return None

        # The subprocesses have overhead, so only use it for expensive computations (k-means).
        if mode.upper() == "KMEANS" and num_kmeans_workers > 1:
            if palettize_weights._compress_pool is None:
                palettize_weights._compress_pool = Pool(processes=num_kmeans_workers)
                atexit.register(lambda: palettize_weights._compress_pool.terminate())
            lut, indices = zip(
                *palettize_weights._compress_pool.starmap(
                    palettize_weights._get_lut_and_indices,
                    zip(
                        grouped_channel_data,
                        repeat(mode),
                        repeat(nbits),
                        repeat(lut_function),
                        repeat(cluster_dim),
                        repeat(vector_axis),
                    ),
                )
            )
        else:
            lut, indices = zip(
                *[
                    palettize_weights._get_lut_and_indices(
                        per_channel_group_data, mode, nbits, lut_function, cluster_dim, vector_axis
                    )
                    for per_channel_group_data in grouped_channel_data
                ]
            )

        lut = np.stack(lut, axis=0)
        indices = np.stack(indices, axis=0)

        if mode.upper() == "CUSTOM":
            # The custom lut_function provided by users should have nbits info.
            # The current `lut` has shape [group_num, lut_entry_num, Optional[cluster_dim]].
            nbits = int(np.ceil(np.log2(lut.shape[1])))

        # The lut and indices from `_get_lut_and_indices` is flattened. The desired result should be
        # `lut` with shape [channel_group_num, palette_num], and `indices` with same shape as the
        # original_data.
        palette_num = 2**nbits
        indices_target_shape = list(original_data.shape)
        if cluster_dim > 1:
            indices_target_shape[vector_axis] //= cluster_dim
        indices = indices.reshape(indices_target_shape)
        lut_target_shape = [1] * (len(original_data.shape) + 2)
        lut_target_shape[0] = channel_group_num
        lut_target_shape[-1] = cluster_dim
        lut_target_shape[-2] = palette_num
        lut = lut.reshape(lut_target_shape)

        if channel_axis != 0:
            lut = np.swapaxes(lut, 0, channel_axis)
            indices = np.swapaxes(indices, 0, channel_axis)

        indices_np_dtype = types.nptype_from_builtin(types.string_to_builtin(f"uint{nbits}"))
        return optimize_utils.LutParams(
            indices.astype(indices_np_dtype), lut, None if cluster_dim == 1 else channel_axis
        )

    @staticmethod
    def decompress(params: Union[optimize_utils.LutParamsIos16, optimize_utils.LutParams]):
        if isinstance(params, optimize_utils.LutParamsIos16):
            return constexpr_lut_to_dense_ios16.decompress(params.lut, params.indices, params.shape)
        elif isinstance(params, optimize_utils.LutParams):
            return constexpr_lut_to_dense.decompress(params.indices, params.lut, None)
        else:
            raise ValueError("Invalid type of params")

    def transform_op(self, op: Operation):
        op_config = self.config._get_const_op_config(op)
        if op_config is None:
            return
        if not self.need_compress_const(op, self.config._is_deprecated, op_config.weight_threshold):
            return

        weight_to_compress = op.outputs[0].val
        restore_original_dtype = None
        if self.joint_compression:
            child_op = op.outputs[0].child_ops[0]
            if child_op.op_type == "constexpr_sparse_to_dense":
                # When the child op is sparse_to_dense op, the weight_to_compress is the sparse
                # representation, which need to be restored to dense representation for compression.
                weight_to_compress = constexpr_sparse_to_dense.decompress(
                    weight_to_compress, child_op.mask.val
                )

            if types.is_int(op.outputs[0].dtype) and op.outputs[0].dtype.get_bitwidth() <= 8:
                # For small range int weights (e.g. int8 weight produced by quantization), convert
                # it to int32 first to avoid overflow during palettization.
                restore_original_dtype = op.outputs[0].dtype
                weight_to_compress = weight_to_compress.astype(np.int32)

        block_sizes, channel_axis = optimize_utils.infer_block_sizes(
            op, op_config, weight_to_compress, return_channel_axis=True
        )
        if block_sizes is None:
            logger.warning(
                f"Cannot perform palettization on {op.name} as block_sizes is None. Skipped this op."
            )
            return
        if op_config.cluster_dim > 1:
            if not optimize_utils.is_cluster_dim_valid(op, op_config.cluster_dim, channel_axis):
                logger.warning(f"The `cluster_dim` is invalid for {op.name}. Skipped this op.")
                return

        if op_config.enable_per_channel_scale:
            # Normalize by per channel scales before doing palettization.
            per_channel_scale = np.max(np.abs(weight_to_compress), axis=channel_axis, keepdims=True)
            per_channel_scale[per_channel_scale == 0] = 1
            weight_to_compress /= per_channel_scale

        lut_params = self.blockwise_compress(
            weight_to_compress,
            op_config.mode,
            op_config.nbits,
            block_sizes,
            op_config.lut_function,
            op_config.cluster_dim,
            channel_axis=channel_axis,
            num_kmeans_workers=op_config.num_kmeans_workers,
        )
        if lut_params is None:
            logger.warning(f"Cannot perform palettization on {op.name}. Skipped this op.")
            return
        if restore_original_dtype is not None:
            lut_params = lut_params._replace(
                lut=lut_params.lut.astype(types.nptype_from_builtin(restore_original_dtype))
            )

        if not self.fake_compression:
            new_var: Optional[Var] = None

            # Specially handle sparse-related compression ops chaining.
            if self.joint_compression:
                child_op = op.outputs[0].child_ops[0]
                if child_op.op_type == "constexpr_sparse_to_dense":
                    mask, nonzero_data = mb.constexpr_lut_to_sparse(
                        indices_mask=child_op.mask,
                        indices_nonzero_data=lut_params.indices[child_op.mask.val != 0].flatten(),
                        lut=lut_params.lut,
                        vector_axis=lut_params.vector_axis,
                        before_op=child_op,
                        name=op.name + "_palettized",
                    )
                    # Feed the sparse lut's nonzero_data output to the child sparse op.
                    new_var = nonzero_data

                    # The mask of the child `constexpr_sparse_to_dense` op need to be the output mask from sparse op
                    # because for vector-palettization the output mask could be different from input mask.
                    # So we have to re-create the child constexpr_sparse_to_dense op and remove the old one.
                    new_sparse_to_dense_op = mb.constexpr_sparse_to_dense(
                        nonzero_data=nonzero_data,
                        mask=mask,
                        before_op=child_op,
                        name=child_op.name,
                    )
                    op.enclosing_block.replace_uses_of_var_after_op(
                        anchor_op=child_op,
                        old_var=child_op.outputs[0],
                        new_var=new_sparse_to_dense_op,
                        force_replace=True,
                    )
                    op.enclosing_block.remove_ops([child_op])

            # For other cases, the new lut var could be constructed directly from lut_params.
            if new_var is None:
                new_var = frontend_utils._construct_constexpr_lut_op(
                    lut_params.indices,
                    lut_params.lut,
                    lut_params.vector_axis,
                    name=op.name + "_palettized",
                    before_op=op,
                )

            if op_config.enable_per_channel_scale:
                if not is_current_opset_version_compatible_with(AvailableTarget.iOS18):
                    raise ValueError(
                        "Palettization with per-channel-scale is only supported since "
                        "iOS18. Please set minimum_deployment_target accordingly."
                    )
                new_var = mb.constexpr_blockwise_shift_scale(
                    data=new_var,
                    scale=per_channel_scale,
                    offset=None,
                    before_op=op,
                    name=op.name + "_palettized_pcs",
                )
        else:
            decompressed_val = self.decompress(lut_params)
            if op_config.enable_per_channel_scale:
                decompressed_val *= per_channel_scale
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

    - Values are linearly quantized into n-bit.
    - If ``fake_compression=False``, compressed value is encoded using the
      ``constexpr_affine_dequantize`` op (pre-iOS18) or the ``constexpr_blockwise_shift_scale`` op (iOS18).
    - If ``fake_compression=True``, compressed value is decompressed and then encoded using the ``const`` op.

    Here is an example for input and output graph of this graph pass:

    .. code-block::

        Input graph:

            const -> downstream op

        Output graph:

            constexpr_blockwise_shift_scale -> downstream op

    Support Options:

    - ``joint_compression``:

        Enable joint compression by quantizing an already compressed model.
        What op could be further quantized is in `_validate_child_constexpr_for_compress`.

        Using palettization + quantization as an example, for each existing ``constexpr_lut_to_dense``
        op, it tries to quantize the elements in the lut, which could be represented as:

        .. code-block::

            Input graph:

                lut(fp16) -> constexpr_lut_to_dense -> dense(fp16) -> downstream op

            Output graph:

                lut(int8) -> constexpr_blockwise_shift_scale -> lut(fp16) -> constexpr_lut_to_dense -> dense(fp16) -> downstream op

    For details about different quantization schemas, see `OpLinearQuantizerConfig` for more details.
    """
    _SUPPORTED_CONFIG_TYPE = OpLinearQuantizerConfig
    _MODE_DTYPE_TO_RANGE = {
        (types.int8, "LINEAR"): (-128, 127),
        (types.int8, "LINEAR_SYMMETRIC"): (-127, 127),
        (types.uint8, "LINEAR"): (0, 255),
        (types.uint8, "LINEAR_SYMMETRIC"): (0, 254),
    }

    def _validate_child_constexpr_for_compress(self, op: Operation) -> bool:
        """
        Overrides external method to support joint compression for iOS18+.

        In iOS18 joint compression, the palettized/sparsified data could be further quantized.
        For each specific op, we only quantize the specific input:
        - constexpr_lut_to_dense's lut
        - constexpr_lut_to_sparse's lut
        - constexpr_sparse_to_dense's nonzero_data
        """
        if (
            is_current_opset_version_compatible_with(AvailableTarget.iOS18)
            and self.joint_compression
        ):
            if len(op.outputs[0].child_ops) == 1:
                child_op = op.outputs[0].child_ops[0]
                if child_op.op_type == "constexpr_lut_to_dense" and child_op.lut == op.outputs[0]:
                    return True
                elif (
                    child_op.op_type == "constexpr_lut_to_sparse" and child_op.lut == op.outputs[0]
                ):
                    return True
                elif (
                    child_op.op_type == "constexpr_sparse_to_dense"
                    and child_op.nonzero_data == op.outputs[0]
                ):
                    return True

        return super()._validate_child_constexpr_for_compress(op)

    @classmethod
    @deprecated(
        suffix="Please use coremltools.optimize.coreml.linear_quantize_weights.blockwise_compress",
        version="8.2",
        obj_prefix="coremltools.optimize.coreml.linear_quantize_weights.",
    )
    def compress(
        cls, val: np.ndarray, axis: int, mode: str, dtype: type
    ) -> optimize_utils.QuantParamsIos16:
        """
        [Legacy] Per-channel quantization on axis.

        This API is for backward compatibility only. It's no longer used inside the coremltools.
        It's recommended to use `blockwise_compress` instead, which is more general.
        """
        if not isinstance(val, (np.ndarray, np.generic)):
            raise ValueError("Only numpy arrays are supported")
        if isinstance(dtype, np.dtype):
            dtype = types.numpy_type_to_builtin_type(dtype)
        if not types.is_builtin(dtype):
            raise ValueError(f"The input dtype is should be a built-in type, but got {type(dtype)}")

        block_sizes = [0] * len(val.shape)
        block_sizes[axis] = 1
        quant_params = cls.blockwise_compress(
            val,
            nbits=dtype.get_bitwidth(),
            mode=mode,
            signed=not dtype.is_unsigned(),
            block_sizes=block_sizes,
        )
        if quant_params is None:
            raise ValueError("Failed to quantize.")

        return optimize_utils.ios18_quant_params_to_ios16(quant_params)

    @classmethod
    def blockwise_compress(
        cls,
        original_data: np.ndarray,
        nbits: int,
        mode: str,
        signed: bool,
        block_sizes: List[int],
    ) -> Optional[optimize_utils.QuantParams]:
        """
        Compress original_data into n-bit representation by quantization.

        mode: "LINEAR_SYMMETRIC" or "LINEAR".

        block_sizes: Each element is the block size on corresponding axis for original_data.

        Returns None if the weight cannot be compressed (for example, the dim size on an axis is not
        divisible by the corresponding block_size).
        """
        if not isinstance(original_data, np.ndarray):
            raise ValueError("Only numpy arrays are supported")

        result = optimize_utils.compute_qparams(
            original_data,
            nbits,
            signed,
            mode,
            types.nptype_from_builtin(types.get_nbits_int_builtin_type(nbits, signed)),
            block_sizes,
        )

        if result is None:
            return None

        quantized_data, scale, zero_point = result
        return optimize_utils.QuantParams(
            data=quantized_data, scale=scale, offset=zero_point, nbits=np.uint8(nbits)
        )

    @staticmethod
    def decompress(params: Union[optimize_utils.QuantParamsIos16, optimize_utils.QuantParams]):
        if isinstance(params, optimize_utils.QuantParamsIos16):
            return constexpr_affine_dequantize.decompress(
                params.quantized_data, params.zero_point, params.scale, params.axis
            )
        elif isinstance(params, optimize_utils.QuantParams):
            return constexpr_blockwise_shift_scale.decompress(
                params.data,
                params.scale,
                params.offset,
            )
        else:
            raise ValueError("Invalid type of params")

    @staticmethod
    def _create_constexpr_var(op: Operation, quant_params: optimize_utils.QuantParams) -> Var:
        """Create constexpr quant op based on opset version."""
        if not is_current_opset_version_compatible_with(AvailableTarget.iOS18):
            quant_params_ios16 = optimize_utils.ios18_quant_params_to_ios16(quant_params)
            return mb.constexpr_affine_dequantize(
                quantized_data=quant_params_ios16.quantized_data,
                zero_point=quant_params_ios16.zero_point,
                scale=quant_params_ios16.scale,
                axis=quant_params_ios16.axis,
                before_op=op,
                name=op.name + "_quantized",
            )

        return mb.constexpr_blockwise_shift_scale(
            data=quant_params.data,
            scale=quant_params.scale,
            offset=quant_params.offset,
            before_op=op,
            name=op.name + "_quantized",
        )

    def transform_op(self, op: Operation):
        op_config: Optional[OpLinearQuantizerConfig] = self.config._get_const_op_config(op)
        if op_config is None:
            return
        if not self.need_compress_const(op, self.config._is_deprecated, op_config.weight_threshold):
            return

        weight_to_compress = op.outputs[0].val

        if np.any(np.isinf(weight_to_compress)):
            logger.warning(
                f"The const {op} has inf/-inf, which is not supported by quantization. Skipped."
            )
            return
        elif weight_to_compress.dtype == bool:
            # bool is already the smallest possible dtype (i.e. 1 bit), cannot further compress
            return
        elif np.issubdtype(weight_to_compress.dtype, np.integer):
            # We have a real use case (llama) where a const bool mask is indexed by input position,
            # which lowers to Core ML `cast bool to int8 -> gather int8 -> cast int8 back to bool`
            # because Core ML gather does not support bool
            # The `cast bool to int8` is then const eliminated,
            # so Core ML serializes const int8 0/1 mask instead
            # Theoretically, such int8 0/1 const can be compressed to 1-bit,
            # but for now let us simply skip its quantization since it does not occupy much space
            # TODO: Explore how the 1-bit compression for such int8 0/1 const can be implemented
            if (
                np.amax(weight_to_compress) - np.amin(weight_to_compress) < 2
                and weight_to_compress.dtype.itemsize <= 1
            ):
                return

        if self.joint_compression:
            child_op = op.outputs[0].child_ops[0]
            if child_op.op_type == "constexpr_sparse_to_dense":
                # When the child op is sparse_to_dense op, the weight_to_compress is the sparse
                # representation, which need to be restored to dense representation for compression.
                weight_to_compress = constexpr_sparse_to_dense.decompress(
                    weight_to_compress, child_op.mask.val
                )
            elif child_op.op_type.startswith("constexpr_lut_to_"):
                if not op_config.granularity == CompressionGranularity.PER_TENSOR:
                    raise NotImplementedError(
                        "When use joint compression for palettization-quantization, please make "
                        "sure to use per-tensor quantization, because the axis for the data to be"
                        "quantized (palettization's lut) is different from the original weight."
                    )

        block_sizes = optimize_utils.infer_block_sizes(op, op_config, weight_to_compress)
        if block_sizes is None:
            logger.warning(
                f"Cannot perform quantization on {op.name} as block_sizes is None. Skipped this op."
            )
            return

        quant_params = self.blockwise_compress(
            weight_to_compress,
            op_config.nbits,
            op_config.mode,
            op_config.signed,
            block_sizes,
        )

        if quant_params is None:
            logger.warning(f"Cannot perform quantization on {op.name}. Skipped this op.")
            return

        if not self.fake_compression:
            new_var: Optional[Var] = None

            # Specially handle sparse-related compression ops chaining.
            if self.joint_compression:
                child_op = op.outputs[0].child_ops[0]
                if child_op.op_type == "constexpr_sparse_to_dense":
                    mask, nonzero_data = mb.constexpr_sparse_blockwise_shift_scale(
                        data_mask=child_op.mask,
                        nonzero_data=quant_params.data[child_op.mask.val != 0].flatten(),
                        scale=quant_params.scale,
                        offset=quant_params.offset,
                        before_op=child_op,
                        name=op.name + "_quantized",
                    )
                    # Feed the sparse quantization op's nonzero_data output to the child sparse op.
                    new_var = nonzero_data

                elif child_op.op_type == "constexpr_lut_to_sparse":
                    # Here we only quantize the lut itself, which is a dense data, so we cannot use
                    # the sparse version of the quant op; instead we just use the dense version of
                    # the quant op. Will change if backends don't support it.
                    pass

            # For other cases, the new quant var could be constructed directly from quant_params.
            if new_var is None:
                new_var = self._create_constexpr_var(op, quant_params)
        else:
            decompressed_val = self.decompress(quant_params)
            new_var = mb.const(
                val=decompressed_val,
                before_op=op,
                name=op.name + "_fake_quantized",
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
            new_const = mb.const(val=decomp_val, before_op=output_var.op, name=output_var.name)
            op.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=output_var.op,
                old_var=output_var,
                new_var=new_const,
                force_replace=True,
            )

        op.enclosing_block.remove_ops([op])
