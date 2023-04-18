#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from enum import Enum as _Enum
from typing import Set, Text

import numpy as np

from coremltools import _logger as logger
from coremltools.converters.mil.backend.mil.load import should_use_weight_file
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.defs.iOS16 import (
    constexpr_affine_dequantize,
    constexpr_lut_to_dense,
    constexpr_sparse_to_dense,
)
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.program import Program
from coremltools.converters.mil.mil.types.type_mapping import (
    is_builtin,
    nptype_from_builtin,
    numpy_type_to_builtin_type,
)
from coremltools.models.neural_network.quantization_utils import _get_kmeans_lookup_table_and_weight


class ComputePrecision(_Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class AbstractQuantizationPass(AbstractGraphPass):
    """
    Base class for Post-Training Quantization transforms.

    Derived class needs to implement following two methods:
        - is_valid_op(op)
        - transform_op(op)
    """

    type_eps = {}
    type_min = {}
    type_negmin = {}

    def __init__(self, op_selector=None):
        super().__init__()
        if op_selector is not None and not callable(op_selector):
            raise TypeError(
                "Argument `op_selector` needs to be a callable function which accepts "
                "a MIL operation object and returns a boolean value."
            )
        self.op_selector = op_selector

    def apply(self, prog):
        """
        Walks over each operation in the graph and performs following two steps,
        1. Checks whether an operation is valid for that quantized transform using `is_valid_op` method.
        2. If yes, calls `transform_op` method of the derived quantized transform class.

        :param prog: MIL program
        :return: Transformed MIL program
        """
        if not isinstance(prog, Program):
            raise TypeError('Transform "{}" can only be applied on PyMIL programs.'.format(self))

        if getattr(self, "skip_ops_by_type", set()) and self.op_selector is not None:
            raise ValueError(
                "The graph pass option `skip_ops_by_type` cannot be set along with "
                "the `op_selector` in FP16ComputePrecision. Please only use one "
                "method to control which ops to operate on."
            )

        @block_context_manager
        def apply_block(block):
            for op in list(block.operations):
                for b in op.blocks:
                    apply_block(b)

                if self.is_valid_op(op):
                    need_transform: bool
                    if self.op_selector is not None:
                        need_transform = self.op_selector(op)
                    else:
                        need_transform = op.op_type not in getattr(self, "skip_ops_by_type", set())
                    if need_transform:
                        self.transform_op(op)

        for f in prog.functions.values():
            apply_block(f)

    def transform_op(self, op):
        """
        Replaces an op with a transformed op.

        :param op: MIL operation
        :return: None
        """
        raise NotImplementedError(
            'Op transformation for quantization mode "{}" not implemented.'.format(self)
        )

    def is_valid_op(self, op):
        """
        Checks whether an operation is valid for given quantized transform.

        :param op: MIL operation
        :return: true | false
        """
        raise NotImplementedError(
            'Operation Preconditions for quantization mode "{}" not implemented.'.format(self)
        )

    @classmethod
    def _close_to_zero(cls, val, np_type):
        if np_type not in cls.type_eps:
            cls.type_eps[np_type] = np.finfo(np_type).eps
            cls.type_min[np_type] = np.nextafter(0.0, 1.0, dtype=np_type)
            cls.type_negmin[np_type] = np.nextafter(0.0, -1.0, dtype=np_type)

        return np.isclose(val, 0, atol=cls.type_min[np_type], rtol=cls.type_eps[np_type])

    def __repr__(self):
        return str(self)

    def __str__(self):
        return type(self).__name__


class FP16ComputePrecision(AbstractQuantizationPass):
    """
    This transform does the following, for each valid op and if the "op_selector" return True:
    - For each input of dtype float32, inject a "cast" op to change it to float16 dtype
    - For each output of dtype float16, inject a "cast" op to change it back to float32
    """

    def __init__(self, op_selector=None):
        super(FP16ComputePrecision, self).__init__(op_selector=op_selector)
        self.target_dtype = "fp16"

        # Var that feeds into multiple ops will be casted once and cached into this dict
        # For reference: Checkout test_single_input_to_multiple_operations in `TestFP16CastTransform`.
        self.cache_vars = {}

    def fp16_overflow(self, op):
        # Constants with values more than 65504 or less than -65504 overflows in FP16
        for _, inputs in op.inputs.items():
            is_list_input = isinstance(inputs, (list, tuple))
            if not is_list_input:
                inputs = [inputs]
            for var in inputs:
                if (
                    var.op is not None
                    and var.op.op_type == "const"
                    and var.is_tensor_or_scalar_of(dtype="fp32")
                ):
                    if np.max(np.abs(var.op.val.val), initial=0.0) > 65504:
                        return True
        return False

    def is_valid_op(self, op):

        if op.op_type in ["cast", "while_loop", "cond"]:
            return False

        if op.op_type in [
            "make_list",
            "list_gather",
            "list_scatter",
            "list_read",
            "list_write",
            "list_length",
        ]:
            return False  #  rdar://74458192

        if op.op_type in ["gru", "rnn", "lstm"]:
            return False

        if self.fp16_overflow(op):
            return False

        return True

    def is_valid_parameter(self, op, param_name):
        type_domain = getattr(op.input_spec.input_types[param_name], "type_domain", None)
        if type_domain is not None:
            if len(type_domain) == 0:
                return True
            return types.fp16 in type_domain
        return True

    def _check_underflow_to_zero(self, new_var, var):
        # We check whether there are casted values that "becomes" 0 which is not ideal for eps purposes.
        # However we skip arrays with more than 400 in case we compare through a large sparse matrix.
        if (
            new_var.val is not None
            and len(var.val.flatten()) < 400
            and self._close_to_zero(new_var.val, np.float16).any()
        ):
            value_modified = False
            original_val = var.val.flatten()
            new_val = new_var.val.flatten()

            for idx in range(len(original_val)):
                if not self._close_to_zero(original_val[idx], np.float32) and self._close_to_zero(
                    new_val[idx], np.float16
                ):
                    new_val[idx] = (
                        self.type_min[np.float16]
                        if np.sign(original_val[idx]) > 0
                        else self.type_negmin[np.float16]
                    )
                    value_modified = True

            if value_modified:
                if np.isscalar(new_var.val):
                    new_var._sym_val.val = new_val[0]
                else:
                    new_var._sym_val.val = new_val.reshape(new_var.val.shape)

    def transform_op(self, op):
        block = op.enclosing_block
        casted_inputs = {}
        inputs_modified = False

        for param, inputs in op.inputs.items():
            # First loop, iterates over all the input parameters of an operation.
            if not self.is_valid_parameter(op, param):
                continue

            is_list_input = isinstance(inputs, (list, tuple))
            if not is_list_input:
                inputs = [inputs]

            casted_inputs[param] = list(inputs[:])
            for i, var in enumerate(inputs):
                # Second loop, iterates over all the vars of a python list corresponding to an input parameter.
                if not var.is_tensor_or_scalar_of(dtype="fp32"):
                    continue

                inputs_modified = True
                casted_var_name = var.name + "_to_fp16"
                if (
                    len(var._child_ops) > 1
                    and casted_var_name in self.cache_vars
                    and (block.is_var_visible_in_block(self.cache_vars[casted_var_name]))
                ):
                    casted_inputs[param][i] = self.cache_vars[casted_var_name]
                else:
                    x = mb.cast(x=var, dtype="fp16", name=casted_var_name, before_op=op)
                    self._check_underflow_to_zero(x, var)

                    casted_inputs[param][i] = x
                    if len(var._child_ops) > 1:
                        self.cache_vars[casted_var_name] = casted_inputs[param][i]

            if not is_list_input:
                casted_inputs[param] = casted_inputs[param][0]

        if inputs_modified:
            casted_inputs.update({k: v for k, v in op.inputs.items() if k not in casted_inputs})
            casted_inputs["name"] = op.name + "_cast"
            casted_inputs["before_op"] = op
            quant_output = getattr(mb, op.op_type)(**casted_inputs)

            if not isinstance(quant_output, (list, tuple)):
                quant_output = [quant_output]

            for old_output_var, new_output_var in zip(op.outputs, quant_output):
                if old_output_var.is_tensor_or_scalar_of(dtype="fp32") and (
                    not new_output_var.is_tensor_or_scalar_of(dtype="fp32")
                ):
                    x = mb.cast(
                        x=new_output_var,
                        dtype="fp32",
                        name=new_output_var.name + "_to_fp32",
                        before_op=op,
                    )
                    op.enclosing_block.replace_uses_of_var_after_op(
                        anchor_op=op,
                        old_var=old_output_var,
                        new_var=x,
                        force_replace=True,
                    )
                else:
                    op.enclosing_block.replace_uses_of_var_after_op(
                        anchor_op=op,
                        old_var=old_output_var,
                        new_var=new_output_var,
                        force_replace=True,
                    )

            block.remove_ops([op])


@register_pass(namespace="common")
class add_fp16_cast(FP16ComputePrecision):
    """
    For each input of dtype float32, inject a ``cast`` op to change it to float16 dtype.
    
    For each output of dtype float16, inject a ``cast`` op to change it back to float32.

    This pass is the registered interface for FP16ComputePrecision, which makes it consistent with
    other passes' interfaces.

    Support options:
    
    - ``skip_ops_by_type``: Skip op types specified by comma-separated string; for example, ``"mul,const"``.
    """

    _skip_ops_by_type: Set[Text] = set()

    @property
    def skip_ops_by_type(self):
        return self._skip_ops_by_type

    @skip_ops_by_type.setter
    def skip_ops_by_type(self, criteria: Text):
        self._skip_ops_by_type = set(criteria.split(","))


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

    WEIGHT_SPARSIFICATION_MODES = ("THRESHOLD_BASED", "PERCENTILE_BASED")

    def __init__(
        self,
        mode="threshold_based",
        threshold=1e-3,
        target_percentile=1.0,
        fake_compression=False,
        op_selector=None,
    ):
        super().__init__(op_selector=op_selector)
        self.fake_compression = fake_compression
        self.mode = mode.upper()
        self.threshold = threshold
        self.target_percentile = target_percentile

        if self.mode not in WeightSparsifier.WEIGHT_SPARSIFICATION_MODES:
            msg = "Only mode {} supported for weight sparsification. Got mode {}.".format(
                WeightSparsifier.WEIGHT_SPARSIFICATION_MODES, self.mode
            )
            raise ValueError(msg)

        if self.mode == "PERCENTILE_BASED" and (
            self.target_percentile < 0 or self.target_percentile > 1
        ):
            raise ValueError(
                "Invalid value of target_percentile: {}. Needs to be in [0, 1]".format(
                    self.target_percentile
                )
            )

        if self.mode == "THRESHOLD_BASED" and self.threshold < 0:
            raise ValueError(
                "Invalid value of threshold: {}. Needs to be in [0, inf)".format(self.threshold)
            )

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

    WEIGHT_PALETTIZATION_MODES = ("KMEANS", "UNIFORM", "UNIQUE", "CUSTOM")

    def __init__(
        self, nbits, fake_compression=False, op_selector=None, mode="kmeans", lut_function=None
    ):
        super().__init__(op_selector=op_selector)
        self.fake_compression = fake_compression
        self.nbits = nbits
        self.mode = mode.upper()
        self.lut_function = lut_function

        if self.mode not in WeightPalettizer.WEIGHT_PALETTIZATION_MODES:
            msg = "Only mode {} supported for weight palettization. Got mode {}.".format(
                WeightPalettizer.WEIGHT_PALETTIZATION_MODES, self.mode
            )
            raise ValueError(msg)

        if nbits is None and self.mode in ("KMEANS", "UNIFORM"):
            msg = "nbits must be provided for mode {}".format(mode)
            raise ValueError(msg)

        if nbits is not None and self.mode in ("UNIQUE", "CUSTOM"):
            msg = "nbits must NOT be provided for mode {}".format(mode)
            raise ValueError(msg)

        if self.nbits is not None and self.nbits not in (1, 2, 4, 6, 8):
            raise ValueError(
                "Invalid value of nbits ({}) for palettization. Supported bits are {{1, 2, 4, 6, 8}}".format(
                    nbits
                )
            )

        if (self.mode == "CUSTOM") ^ (lut_function is not None):
            msg = "lut_function must be None if mode is not custom, and that it cannot be None when the mode is custom."
            raise ValueError(msg)

        if self.mode == "CUSTOM" and not callable(self.lut_function):
            msg = "A function object must be provided as lut_function. Got a lut_functions as type {}".format(
                type(self.lut_function)
            )
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

        def pack_indices_into_bytes_array(indices, nbits):
            bitarray = np.unpackbits(indices.reshape(-1, 1), bitorder="little", axis=-1)[:, :nbits]
            return np.packbits(bitarray.flatten(), bitorder="little")

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

    WEIGHT_AFFINE_QUANTIZATION_MODES = ("LINEAR_SYMMETRIC", "LINEAR")
    WEIGHT_AFFINE_DTYPES = (types.int8, types.uint8)

    def __init__(self, fake_compression=False, op_selector=None, mode="linear", dtype=np.int8):
        super().__init__(op_selector=op_selector)
        self.fake_compression = fake_compression
        self.mode = mode.upper()

        # check mode
        if self.mode not in WeightAffineQuantizer.WEIGHT_AFFINE_QUANTIZATION_MODES:
            msg = "Only mode {} supported for weight affine quantization. Got mode {}.".format(
                WeightAffineQuantizer.WEIGHT_AFFINE_QUANTIZATION_MODES, self.mode
            )
            raise ValueError(msg)

        # check dtype
        msg = f"dtype={dtype} is unsupported for affine_quantize_weights."
        if is_builtin(dtype):
            self.dtype = dtype
        else:
            try:
                self.dtype = numpy_type_to_builtin_type(dtype)
            except TypeError:
                raise ValueError(msg)

        if self.dtype not in WeightAffineQuantizer.WEIGHT_AFFINE_DTYPES:
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

        mode = mode.upper()
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

    def transform_op(self, op):
        block = op.enclosing_block
        quant_params = self.compress(op.val.val, self._get_axis(op), self.mode, self.dtype)

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
        return op.op_type in (
            "constexpr_affine_dequantize",
            "constexpr_lut_to_dense",
            "constexpr_sparse_to_dense",
        )

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
