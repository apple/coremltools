# Copyright (c) 2022, Apple Inc. All rights reserved.
import numpy as np

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import (InputSpec,
                                                       TensorInputType)
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS16 import _IOS16_TARGET


@register_op(opset_version=_IOS16_TARGET)
class constexpr_affine_dequantize(Operation):
    """
    A compile-time operation that returns a constant output value upon dequantizing its constant inputs.

    This operation is used to represent constant 8-bit quantized data with affine/linear quantization.
    The quantized data is stored in the parameter ``quantized_data``.
    The other parameters -- ``scale``, ``zero_point``, and ``axis`` -- describe how
    unquantized values can be extracted from it, using the equation for affine/linear quantization:
    ::
                unquantized_data = scale * (quantized_data - zero_point)

    Although all of the parameters of this op are constants, this op is not constant folded
    to a single const op at the time of model serialization. The unquantized output will
    be decompressed later, based on the implementation detail (either at model load time or runtime).

    Parameters
    ----------
    quantized_data: const tensor<SrcT, [1..]> (Required)

    zero_point: const tensor<SrcT, [0..1]> (Required)
 	   * ``zero_point`` can be either a scalar or a vector. 
 	   * ``zero_point`` follows similar broadcasting rules and size constraints as ``scale``.

    scale: const tensor<DstT, [0..1]> (Required)
       * ``scale`` can be either a scalar or a vector. If ``scale`` is a vector,
         for implementation it is broadcast to the following shape:
           * The rank of ``scale`` becomes the same as the rank of ``quantized_data``.
           * The constraint: ``size(scale-vector) == quantized_data.shape[axis]``.
           * For ``i == axis``, ``scale.shape[i] == quantized_data.shape[i]``.
           * For ``i != axis``, ``scale.shape == 1``.
         For example, assume ``quantized_data.shape = (2, 3, 4, 5)`` and ``axis = 1``.
         If ``scale`` is a vector, then ``scale.size`` needs to be equal to
         ``quantized_data.shape[axis] i.e = 3``, which would be broadcast to ``(1, 3, 1, 1)``.

    axis: const tensor<int32, []> (Required)

    Returns
    -------
    const tensor<DstT, [1..]>

    Attributes
    ----------
    SrcT: uint8, int8
    DstT: fp16, fp32
    """

    input_spec = InputSpec(
        quantized_data=TensorInputType(const=True, type_domain="SrcT"),
        zero_point=TensorInputType(const=True, type_domain="ZeroPointT"),
        scale=TensorInputType(const=True, type_domain="DstT"),
        axis=TensorInputType(const=True, type_domain=types.int32),
    )
    
    type_domains = {
        "DstT": (types.fp16, types.fp32),
        "SrcT": (types.uint8, types.int8),
        "ZeroPointT": (types.uint8, types.int8),
    }

    def type_inference(self):
        def assert_is_scalar_or_vector(param, name):
            if param.rank not in (0, 1):
                raise ValueError(
                    "Parameter {} needs to be either a scalar or vector".format(name)
                )

        def assert_vector_size_same_as_axial_dimension(param, axis_dim_size, name):
            if param.rank == 1 and param.shape[0] != axis_dim_size:
                raise ValueError(
                    "Parameter {}, if vector, needs to have same size as the dimension size along the parameter quantized_data".format(
                        name
                    )
                )

        if self.zero_point.dtype != self.quantized_data.dtype:
            raise ValueError(
                "Parameters quantized_data and zero_point needs to be of the same dtype"
            )

        rank = self.quantized_data.rank
        if self.axis.val < -rank or self.axis.val >= rank:
            raise ValueError(
                "Parameter axis needs to be in the range -quantized_data.rank <= axis < quantized_data.rank"
            )

        assert_is_scalar_or_vector(self.scale, "scale")
        assert_is_scalar_or_vector(self.zero_point, "zero_point")

        assert_vector_size_same_as_axial_dimension(
            self.scale, self.quantized_data.shape[self.axis.val], "scale"
        )
        assert_vector_size_same_as_axial_dimension(
            self.zero_point, self.quantized_data.shape[self.axis.val], "zero_point"
        )

        dtype = self.scale.dtype
        shape = self.quantized_data.shape
        return types.tensor(dtype, shape)

    def value_inference(self):
        return self.decompress(
            self.quantized_data.val, 
            self.zero_point.val, 
            self.scale.val, 
            self.axis.val
        )

    @staticmethod
    def decompress(quantized_data, zero_point, scale, axis):

        axis = axis if axis >= 0 else axis + len(quantized_data.shape)

        def rank_promoted_to_same_as_quantized_data(param):
            if len(param.shape) == 0:
                return np.reshape(param, np.ones(len(quantized_data.shape), np.int32))
            else:
                axes = [i for i in range(len(quantized_data.shape)) if i != axis]
                return np.expand_dims(param, axis=tuple(axes))

        sc = rank_promoted_to_same_as_quantized_data(scale)
        zp = rank_promoted_to_same_as_quantized_data(zero_point)
        val = sc * (quantized_data.astype(np.float32) - zp.astype(np.float32))
        return val.astype(scale.dtype)


@register_op(opset_version=_IOS16_TARGET)
class constexpr_cast(Operation):
    """
    A compile-time operation that returns a constant output value upon casting its constant input.
    ::
                Expression: output = constexpr_cast(source_val, output_dtype="fp32")

    Parameters
    ----------
    source_val: const tensor<SrcT, [...]> (Required)

    output_dtype: const tensor<string, []> (Required)

    Returns
    -------
    const tensor<DstT, [...]>

    Attributes
    ----------
    SrcT: fp16
    DstT: fp32
    """

    input_spec = InputSpec(
        source_val=TensorInputType(const=True, type_domain=types.fp16),
        output_dtype=TensorInputType(const=True, type_domain=types.str),
    )

    def type_inference(self):

        dtype = types.string_to_builtin(self.output_dtype.val)
        if dtype != types.fp32:
            raise NotImplementedError("Only output_dtype = fp32 is supported")

        shape = self.source_val.shape
        return types.tensor(dtype, shape)

    def value_inference(self):
        return np.float32(self.source_val.val)


@register_op(opset_version=_IOS16_TARGET)
class constexpr_lut_to_dense(Operation):
    """
    A compile-time operation that returns a constant output value upon decompressing 
    a look-up table (LUT) to a dense tensor.

    This operation is used to store constant weights in a LUT format (also known as
    `palettized` weights). A LUT is a mapping from index to values.
    Weights are quantized and stored as indices (or keys) into the LUT.
    Before computation, these keys are mapped to corresponding values in the LUT.

    Parameters
    ----------
    indices: const tensor<uint8, [M]> (Required)

    lut: const tensor<T, [NUM_PALETTES]> (Required)

    shape: const tensor<uint32, [K]> (Required)

	Notes
	-----

    * Any data is packed and read in a row-major order.
    * ``NUM_PALETTES`` can be one of ``{2, 4, 16, 64 or 256}``.
    * ``n_bits = log2(NUM_PALETTES)`` can thus be one of ``{1, 2, 4, 6, 8}``.
    * Indices are packed in bytes of size ``M``, where ``M = ceil(n_bits * product(shape) / 8)``.

    The bit fields are packed one byte at a time, starting with the least significant bit (LSB) and
    moving upward to the most significant bit (MSB). It follows, naturally, that if an index is split
    across two bytes, the LSBs of that index is filled over the MSBs of current byte, and the remaining
    bits of the same index are filled in the LSBs of the next byte.

    For example:
    ::
        if n_bits = 2, shape = (5,) => M = 2 bytes

                    MSB             LSB
                     |               |
        indices =  | 01   10   11   00 | xx   xx   xx   11 |      <== packed elements
                   | i3 | i2 | i1 | i0 | -- | -- | -- | i4 |      <== tagged element ids
                   |      byte 0       |       byte 1      |      <== tagged bytes

    Returns
    -------
    const tensor<T, [...]>

    Attributes
    ----------
    T: uint8, int8, fp16, fp32
    """

    input_spec = InputSpec(
        indices=TensorInputType(const=True, type_domain=types.uint8),
        lut=TensorInputType(const=True, type_domain="T"),
        shape=TensorInputType(const=True, type_domain=types.uint32),
    )
    
    type_domains = {
        "T": (types.int8, types.uint8, types.fp16, types.fp32)
    }

    def type_inference(self):
        def assert_is_vector(param, name):
            if param.rank != 1:
                raise ValueError("Parameter {} needs to have rank == 1".format(name))

        assert_is_vector(self.indices, "indices")
        assert_is_vector(self.lut, "lut")

        if self.lut.shape[0] not in (2, 4, 16, 64, 256):
            raise ValueError(
                "Parameter lut should be a vector of size from one of {2, 4, 16, 64, 256}"
            )

        nbits = int(np.log2(self.lut.shape[0]))
        output_size = np.prod(self.shape.val)
        if self.indices.shape[0] != np.ceil(nbits * (output_size / 8.0)):
            raise AssertionError(
                "Constraint violated, M = ceil(n_bits * product(shape) / 8) where M = indices.size"
            )

        dtype = self.lut.dtype
        shape = self.shape.val
        return types.tensor(dtype, shape)

    def value_inference(self):
        return self.decompress(
                self.lut.val,
                self.indices.val,
                self.shape.val,
            )

    @staticmethod
    def decompress(lut, indices, shape):
        bitarray = np.unpackbits(indices, bitorder="little")
        nbits = np.log2(lut.size).astype(np.int32)

        pad_required = bitarray.size % nbits != 0
        if pad_required:
            bitarray = np.concatenate([bitarray, np.zeros(bitarray.size % nbits)])

        assert bitarray.size % nbits == 0

        size = np.prod(shape)
        bitarray = bitarray.reshape(-1, nbits)[:size, :]

        indices = np.packbits(bitarray, bitorder="little", axis=-1).reshape(-1)
        flatten_val = lut[indices]
        return flatten_val.reshape(shape)


@register_op(opset_version=_IOS16_TARGET)
class constexpr_sparse_to_dense(Operation):
    """
    A compile-time operation that returns a constant output value upon de-sparsification of its constant inputs.

    This operation represents unstructured sparsity and uses bit mask binary representation.
    If a bit is set, then the corresponding element in the output tensor is non-zero and the
    value is read from the ``nonzero_data`` attribute. Likewise, if the bit is not set,
    then the corresponding element in the output tensor is zero.

    Parameters
    ----------
    nonzero_data: const tensor<T, [D]> (Required)

    mask: const tensor<uint8, [M]> (Required)

    shape: const tensor<uint32, [K]> (Required)

	Notes
	-----
    * Any data is packed and read in a row-major order.
    * ``mask`` contains ``M`` bytes, where ``M = ceil( product(shape) / 8)``. That is, each bit
      field corresponds to one element in the output tensor.
    * ``D ==`` the total number of set bits in ``mask``.

    The bit fields are packed one byte at a time, starting with the least significant bit and
    moving up to the most significant bit.

    For example:
    ::
        shape = (5,) => M = 1 bytes

                   MSB                  LSB
                    |                    |
        mask    =  |x  x  x  0  1  1  0  0 |      <== packed elements
                   |--|--|--|i4|i3|i2|i1|i0|      <== tagged element ids
                   |      byte 0           |      <== tagged bytes

    Returns
    -------
    const tensor<T, [...]>

    Attributes
    ----------
    T: uint8, int8, fp16, fp32
    """

    input_spec = InputSpec(
        nonzero_data=TensorInputType(const=True, type_domain="T"),
        mask=TensorInputType(const=True, type_domain=types.uint8),
        shape=TensorInputType(const=True, type_domain=types.uint32),
    )
    
    type_domains = {
        "T": (types.int8, types.uint8, types.fp16, types.fp32)
    }

    def type_inference(self):
        def assert_is_vector(param, name):
            if param.rank != 1:
                raise ValueError("Parameter {} needs to have rank == 1".format(name))

        assert_is_vector(self.nonzero_data, "nonzero_data")
        assert_is_vector(self.mask, "mask")

        if sum(bin(x).count("1") for x in self.mask.val) != self.nonzero_data.shape[0]:
            raise AssertionError(
                "Number of set bits in mask needs to be equal to number of elements in parameter nonzero_data"
            )

        output_size = np.prod(self.shape.val)
        if self.mask.shape[0] != np.ceil(output_size / 8.0):
            raise AssertionError(
                "Constraint Violated: M = ceil( product(shape) / 8) where M = mask.size"
            )

        bitarray = np.unpackbits(self.mask.val, bitorder="little")
        if any(bitarray[i] != 0 for i in range(output_size, len(bitarray))):
            raise AssertionError("Padded bits in mask should be unset or equals to zero")

        dtype = self.nonzero_data.dtype
        shape = self.shape.val
        return types.tensor(dtype, shape)

    def value_inference(self):
        return self.decompress(self.nonzero_data.val, self.mask.val, self.shape.val)

    @staticmethod
    def decompress(nonzero_data, mask, shape):
        flattend_val = np.zeros(shape, dtype=nonzero_data.dtype).flatten()
        flattend_val[
            np.where(np.unpackbits(mask, bitorder="little") != 0)
        ] = nonzero_data
        return flattend_val.reshape(shape)
