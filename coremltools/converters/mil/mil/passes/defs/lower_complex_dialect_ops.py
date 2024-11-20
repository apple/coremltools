#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
This file contains a pass for lowering complex dialect ops into core ops.

Steps for adding a new complex dialect op:
1. Add a dialect op in complex_dialect_ops.py
2. Add a corresponding lowering function

In Step 2, notice that when implementing lower functions, we need to specify mb.set_before_op during
lowering to core ops. It's for both correctness as well as SSA graph's readability, because the
generated core ops should be placed before the ops which were placed after that dialect op.
More specifically, here is the SSA graph before lowering:
    block0() {
      %1 = complex_dialect_op(data=%input)
      %2 = core_op1(x=%1)
      %3 = core_op2(x=%2)
    } -> (%3)
During lowering `complex_dialect_op`, we want all newly generated core ops are placed before the
`core_op1`.
"""

import functools
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.ops.defs.complex_dialect_ops import (
    fft_canonicalize_length_dim,
    fft_canonicalize_shapes_dims,
)
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.var import ComplexVar, Var


class LowerComplex:
    # The map recording each complex dialect op's lowering function.
    _lower_map: Dict[str, Callable] = dict()

    @staticmethod
    def register_lower_func(op_type: str) -> Callable:
        """Register lowering function for complex dialect ops."""

        def lower_func_wrapper(func):
            @functools.wraps(func)
            def wrapper_inner(*args, **kwargs):
                return func(*args, **kwargs)

            if op_type in LowerComplex._lower_map:
                raise ValueError(f"The op {op_type} already got lowering function registered.")
            LowerComplex._lower_map[op_type] = func
            return wrapper_inner

        return lower_func_wrapper

    @staticmethod
    def has_lower_func(op_type: str) -> bool:
        """Check if the complex dialect op has corresponding lowering function."""
        return op_type in LowerComplex._lower_map

    @staticmethod
    def get_lower_func(op_type: str) -> Callable:
        """Get the complex dialect op's lowering function."""
        if not LowerComplex.has_lower_func(op_type):
            raise ValueError(f"The op {op_type} doesn't have any lowering function registered.")
        return LowerComplex._lower_map[op_type]


def _resize_data(input_data: Var, dims: Tuple[int], sizes: Tuple[int]) -> Var:
    """
    For each dim in `dims`, resize the input data size to corresponding size in `sizes`.
    If the `size` is smaller than the data's size at `dim`, trim the data to `size`.
    If the `size` is larger, pad zeros to make the data reaches `size`.
    """
    for (dim, size) in zip(dims, sizes):
        if size < input_data.shape[dim]:
            indices = mb.range_1d(start=0, end=size, step=1)
            input_data = mb.gather(x=input_data, indices=indices, axis=dim)
        elif size > input_data.shape[dim]:
            zero_shape = list(input_data.shape)
            zero_shape[dim] = size - input_data.shape[dim]
            zero_data = mb.fill(shape=zero_shape, value=0.0)
            input_data = mb.concat(values=[input_data, zero_data], axis=dim)

    return input_data


def _restore_conj(input_data: ComplexVar, n: Var, dim: Var) -> Tuple[Var, Var]:
    """
    The input is interpreted as a one-sided Hermitian signal in the Fourier domain, as produced
    by rfft(). So we need to restore it to the full matrix by following X[i] = conj(X[-i]).
    Real part's conj is itself, and imaginary part's conj is negative of the original value.
    For odd number n, the last element is also included in mirroring input.
    """
    real_data: Var = input_data.real
    imag_data: Var = input_data.imag

    size = 2 * (input_data.real.shape[dim.val] - 1)
    if n is not None and n.val is not None:
        size = n.val
        real_data = _resize_data(real_data, dims=(dim.val,), sizes=(size // 2 + 1,))
        imag_data = _resize_data(imag_data, dims=(dim.val,), sizes=(size // 2 + 1,))

    range_end = real_data.shape[dim.val] - 2 if size % 2 == 0 else real_data.shape[dim.val] - 1
    if range_end > 0:
        mirror_indices = mb.range_1d(start=range_end, end=0, step=-1)
        real_part_mirror_values = mb.gather(x=real_data, indices=mirror_indices, axis=dim.val)
        imag_part_mirror_values = mb.gather(x=imag_data, indices=mirror_indices, axis=dim.val)
        imag_part_mirror_values = mb.mul(x=imag_part_mirror_values, y=-1.0)

        real_data = mb.concat(
            values=[real_data, real_part_mirror_values],
            axis=dim.val,
        )
        imag_data = mb.concat(
            values=[imag_data, imag_part_mirror_values],
            axis=dim.val,
        )

    return real_data, imag_data

def _calculate_dft_matrix(
    n_fft: Var,
    onesided: bool = False,
) -> Tuple[Var, Var]:
    """
    The core issue is how to derive the DFT matrix. As the DFT matrix is consist of different powers
    of `w`, where w=e^(2pi/N i), we need to separate the real and imaginary part of w. To achieve
    that, we need to find a way to construct the following matrix (from the power of `w` in DFT):
        0    0    0      ...    0
        0    1    2      ...    N-1
        0    2    4      ...    2(N-1)
        ...    ....      ...
        0   N-1  2(N-1)  ...    (N-1)(N-1)
    This matrix could be derived by outer product of two range tensors.

    After getting that base matrix, we can take sin and cos to get the corresponding `sin_base` and
    `cos_base` matrix.

    If the onesided flag is passed, we can take advantage of Hermitian symmetry and return a
    weight matrix consisting of only the first (n_fft // 2 + 1) values.
    """
    n_fft = mb.cast(x=n_fft, dtype="fp32")

    if onesided:
        half = mb.floor_div(x=n_fft, y=2.0)
        half = mb.add(x=half, y=1.0)

    tmp_x = mb.range_1d(start=0.0, end=(half if onesided else n_fft), step=1.0)
    tmp_y = mb.range_1d(start=0.0, end=n_fft, step=1.0)

     # Use MIL ops to calculate base = torch.outer(tmp, tmp) * (2 * torch.pi / N).
    tmp_x = mb.reshape(x=tmp_x, shape=[-1, 1])
    tmp_y = mb.reshape(x=tmp_y, shape=[1, -1])

    base = mb.matmul(x=tmp_x, y=tmp_y)
    base = mb.mul(x=base, y=2 * np.pi)
    base = mb.real_div(x=base, y=n_fft)

    # Get real part and imaginary part separately.
    cos_base = mb.cos(x=base)
    sin_base = mb.sin(x=base)

    return cos_base, sin_base

def _fft_1d(
    input_real: Var,
    input_imag: Var,
    n: Optional[Var],
    dim: Optional[Var],
    norm: Optional[Var],
    inverse: bool = False,  # For inverse FFT.
) -> Tuple[Var, Var]:
    """
    1-D FFT by DFT Matrix Multiplication.

    Now based on some math formulas including:
        * The addition of complex numbers is: (a+bi)+(c+di)=(a+c)+(b+d)i.
        * The multiplication of complex numbers is: (a+bi)(c+di)=ac+adi+bci−bd=(ac−bd)+(ad+bc)i.
        * Euler’s formula: e^xi=cosx+isinx.
        * Cosine is an even function: cos(−x)=cosx.
        * Sine is an odd function: sin(−x)=−(sinx).
    We can get
        * The real part output is: cos_base * input_real + sin_base * input_imag
        * The imaginary part output is: - (sin_base * input_real - cos_base * input_imag)
    That's how we calculate the real and imaginary part separately for the FFT.
    """
    n, dim = fft_canonicalize_length_dim(input_real, n, dim)

    # Swaps target dim axis to the first axis.
    axes = list(range(len(input_real.shape)))
    axes[0] = dim
    axes[dim] = 0
    transposed_input_real = mb.transpose(x=input_real, perm=axes)
    transposed_input_imag = mb.transpose(x=input_imag, perm=axes)

    # Trim or pad input according to n.
    transposed_input_real = _resize_data(
        input_data=transposed_input_real,
        dims=(0,),
        sizes=(n,),
    )
    transposed_input_imag = _resize_data(
        input_data=transposed_input_imag,
        dims=(0,),
        sizes=(n,),
    )

    # Calculate DFT matrix.
    original_shape = transposed_input_real.shape
    N = transposed_input_real.shape[0]
    reshaped_input_real = mb.reshape(x=transposed_input_real, shape=[N, -1])
    reshaped_input_imag = mb.reshape(x=transposed_input_imag, shape=[N, -1])

    N = mb.cast(x=N, dtype="fp32")
    cos_base, sin_base = _calculate_dft_matrix(N, onesided=False)

    if not inverse:
        real_part = mb.add(
            x=mb.matmul(x=cos_base, y=reshaped_input_real),
            y=mb.matmul(x=sin_base, y=reshaped_input_imag),
        )
        imag_part = mb.sub(
            x=mb.matmul(x=sin_base, y=reshaped_input_real),
            y=mb.matmul(x=cos_base, y=reshaped_input_imag),
        )
        imag_part = mb.mul(x=imag_part, y=-1.0)
    else:
        real_part = mb.sub(
            x=mb.matmul(x=cos_base, y=reshaped_input_real),
            y=mb.matmul(x=sin_base, y=reshaped_input_imag),
        )
        imag_part = mb.add(
            x=mb.matmul(x=sin_base, y=reshaped_input_real),
            y=mb.matmul(x=cos_base, y=reshaped_input_imag),
        )

    real_part = mb.reshape(x=real_part, shape=original_shape)
    imag_part = mb.reshape(x=imag_part, shape=original_shape)

    # Swaps dim back.
    real_part = mb.transpose(x=real_part, perm=axes)
    imag_part = mb.transpose(x=imag_part, perm=axes)

    # Normalization if needed.
    apply_scale = False
    scale = 1
    if norm.val is not None:
        # For FFT, "forward" means normalize 1/N, while in IFFT, "backward" means normalize 1/N.
        if (not inverse) and (norm.val in ["forward", "ortho"]):
            apply_scale = True
            scale = N if norm.val == "forward" else mb.sqrt(x=N)
        if inverse and (norm.val in ["backward", "ortho"]):
            apply_scale = True
            scale = N if norm.val == "backward" else mb.sqrt(x=N)
    if apply_scale:
        real_part = mb.real_div(x=real_part, y=scale)
        imag_part = mb.real_div(x=imag_part, y=scale)

    return real_part, imag_part


def _rfft_1d(
    input_real: Var,
    n: Optional[Var],
    dim: Optional[Var],
    norm: Optional[Var],
) -> Tuple[Var, Var]:
    """
    It's similar to fft, but as the input is real data, the redundant info (the conjugate part) is
    removed in the result.
    """
    input_imag = mb.fill(
        shape=mb.shape(x=input_real),
        value=0.0,
    )
    real_data, imag_data = _fft_1d(input_real, input_imag, n, dim, norm)
    remain_len = real_data.shape[dim.val] // 2 + 1
    remain_indices = mb.range_1d(start=0, end=remain_len, step=1)
    real_data = mb.gather(x=real_data, indices=remain_indices, axis=dim.val)
    imag_data = mb.gather(x=imag_data, indices=remain_indices, axis=dim.val)

    return real_data, imag_data

def _stft(
    input_real: Var,
    input_imaginary: Optional[Var],
    n_fft: Var,
    hop_length: Optional[Var],
    win_length: Optional[Var],
    window: Optional[Var],
    normalized: Optional[Var],
    onesided: Optional[Var],
) -> Tuple[Var, Var]:
    """
    We can write STFT in terms of convolutions with a DFT kernel.
    At the end:
        * The real part output is: cos_base * input_real + sin_base * input_imag
        * The imaginary part output is: - (sin_base * input_real - cos_base * input_imag)
    Adapted from: https://github.com/adobe-research/convmelspec/blob/main/convmelspec/mil.py
    """
    hop_length = hop_length or mb.floor_div(x=n_fft, y=4)

    # input should always be 2D
    should_increase_rank = input_real.rank == 1
    if should_increase_rank:
        input_real = mb.expand_dims(x=input_real, axes=(0,))
        if input_imaginary:
            input_imaginary = mb.expand_dims(x=input_imaginary, axes=(0,))

    is_onesided = onesided and onesided.val
    cos_base, sin_base = _calculate_dft_matrix(n_fft, onesided=is_onesided)

    # create a window of centered 1s of the requested size
    if win_length:
        n_left = (n_fft.val - win_length.val) // 2
        n_right = n_fft.val - win_length.val - n_left

        left = mb.fill(shape=(n_left,), value=0.0)
        if not window:
            window = mb.fill(shape=(win_length.val,), value=1.0)
        right = mb.fill(shape=(n_right,), value=0.0)

        # concatenate
        window = mb.concat(values=(left, window, right), axis=0)

    # apply time window
    if window:
        cos_base = mb.mul(x=window, y=cos_base)
        sin_base = mb.mul(x=window, y=sin_base)

    # conv with DFT kernel across the input signal
    sin_base = mb.sub(x=0.0, y=sin_base)
    cos_base = mb.expand_dims(x=cos_base, axes=(1,))
    sin_base = mb.expand_dims(x=sin_base, axes=(1,))
    hop_size = mb.expand_dims(x=hop_length, axes=(0,))

    signal_real = mb.expand_dims(x=input_real, axes=(1,))
    cos_windows_real = mb.conv(x=signal_real, weight=cos_base, strides=hop_size, pad_type="valid")
    sin_windows_real = mb.conv(x=signal_real, weight=sin_base, strides=hop_size, pad_type="valid")

    if input_imaginary:
        signal_imaginary = mb.expand_dims(x=input_imaginary, axes=(1,))
        cos_windows_imag = mb.conv(
            x=signal_imaginary, weight=cos_base, strides=hop_size, pad_type="valid"
        )
        sin_windows_imag = mb.conv(
            x=signal_imaginary, weight=sin_base, strides=hop_size, pad_type="valid"
        )

    # add everything together
    if input_imaginary:
        # sin base is already negative so subtract
        real_result = mb.sub(x=cos_windows_real, y=sin_windows_imag)
        imag_result = mb.add(x=sin_windows_real, y=cos_windows_imag)
    else:
        real_result = cos_windows_real
        imag_result = sin_windows_real

    # reduce the rank of the output
    if should_increase_rank:
        real_result = mb.squeeze(x=real_result, axes=(0,))
        imag_result = mb.squeeze(x=imag_result, axes=(0,))

    if normalized and normalized.val:
        divisor = mb.sqrt(x=mb.cast(x=n_fft, dtype="fp32"))
        real_result = mb.real_div(x=real_result, y=divisor)
        imag_result = mb.real_div(x=imag_result, y=divisor)

    return real_result, imag_result

def _wrap_complex_output(original_output: Var, real_data: Var, imag_data: Var) -> ComplexVar:
    return ComplexVar(
        name=original_output.name + "_lowered",
        sym_type=original_output.sym_type,
        real=real_data,
        imag=imag_data,
    )


@LowerComplex.register_lower_func(op_type="complex")
def _lower_complex(op: Operation):
    return _wrap_complex_output(op.outputs[0], op.real_data, op.imag_data)


@LowerComplex.register_lower_func(op_type="complex_real")
def _lower_complex_real(op: Operation):
    complex_input: ComplexVar = op.data
    # Use an identity op to avoid the block's input name inconsistency issue. If we directly use
    # complex_input.real, the var's name could be inconsistent with the block's input name.
    result = mb.identity(x=complex_input.real)
    return result


@LowerComplex.register_lower_func(op_type="complex_imag")
def _lower_complex_imag(op: Operation):
    complex_input: ComplexVar = op.data
    # Use an identity op to avoid the block's input name inconsistency issue. If we directly use
    # complex_input.imag, the var's name could be inconsistent with the block's input name.
    result = mb.identity(x=complex_input.imag)
    return result


@LowerComplex.register_lower_func(op_type="complex_fft")
def _lower_complex_fft(op: Operation):
    if types.is_complex(op.data.dtype):
        real_data = op.data.real
        imag_data = op.data.imag
    else:
        real_data = op.data
        imag_data = mb.fill(
            shape=mb.shape(x=real_data),
            value=mb.cast(
                x=mb.const(val=0.0),
                dtype=real_data.dtype.__name__,
            ),
        )
    real_data, imag_data = _fft_1d(
        real_data,
        imag_data,
        op.n,
        op.dim,
        op.norm,
    )
    return _wrap_complex_output(op.outputs[0], real_data, imag_data)


@LowerComplex.register_lower_func(op_type="complex_fftn")
def _lower_complex_fftn(op: Operation):
    if types.is_complex(op.data.dtype):
        real_data = op.data.real
        imag_data = op.data.imag
    else:
        real_data = op.data
        imag_data = mb.fill(
            shape=mb.shape(x=real_data),
            value=mb.cast(
                x=mb.const(val=0.0),
                dtype=real_data.dtype.__name__,
            ),
        )

    shapes, dims = fft_canonicalize_shapes_dims(real_data, op.shapes, op.dims)
    for shape, dim in zip(shapes, dims):
        real_data, imag_data = _fft_1d(
            real_data,
            imag_data,
            n=mb.const(val=shape),
            dim=mb.const(val=dim),
            norm=op.norm,
        )

    return _wrap_complex_output(op.outputs[0], real_data, imag_data)


@LowerComplex.register_lower_func(op_type="complex_rfft")
def _lower_complex_rfft(op: Operation):
    real_data, imag_data = _rfft_1d(op.data, op.n, op.dim, op.norm)
    return _wrap_complex_output(op.outputs[0], real_data, imag_data)


@LowerComplex.register_lower_func(op_type="complex_rfftn")
def _lower_complex_rfftn(op: Operation):
    shapes, dims = fft_canonicalize_shapes_dims(op.data, op.shapes, op.dims)
    real_data, imag_data = _rfft_1d(
        op.data,
        mb.const(val=shapes[-1]),
        mb.const(val=dims[-1]),
        op.norm,
    )
    for shape, dim in zip(shapes[:-1], dims[:-1]):
        real_data, imag_data = _fft_1d(
            real_data,
            imag_data,
            n=mb.const(val=shape),
            dim=mb.const(val=dim),
            norm=op.norm,
        )
    return _wrap_complex_output(op.outputs[0], real_data, imag_data)


@LowerComplex.register_lower_func(op_type="complex_ifft")
def _lower_complex_ifft(op: Operation):
    real_data, imag_data = _fft_1d(op.data.real, op.data.imag, op.n, op.dim, op.norm, inverse=True)
    return _wrap_complex_output(op.outputs[0], real_data, imag_data)


@LowerComplex.register_lower_func(op_type="complex_ifftn")
def _lower_complex_ifftn(op: Operation):
    real_data = op.data.real
    imag_data = op.data.imag
    shapes, dims = fft_canonicalize_shapes_dims(real_data, op.shapes, op.dims)
    for shape, dim in zip(shapes, dims):
        real_data, imag_data = _fft_1d(
            real_data,
            imag_data,
            n=mb.const(val=shape),
            dim=mb.const(val=dim),
            norm=op.norm,
            inverse=True,
        )
    return _wrap_complex_output(op.outputs[0], real_data, imag_data)


@LowerComplex.register_lower_func(op_type="complex_irfft")
def _lower_complex_irfft(op: Operation):
    real_data, imag_data = _restore_conj(op.data, op.n, op.dim)
    n, dim = fft_canonicalize_length_dim(op.data, op.n, op.dim, c2r=True)
    real_data, imag_data = _fft_1d(
        real_data,
        imag_data,
        mb.const(val=n),
        mb.const(val=dim),
        op.norm,
        inverse=True,
    )
    return real_data


@LowerComplex.register_lower_func(op_type="complex_irfftn")
def _lower_complex_irfftn(op: Operation):
    real_data = op.data.real
    imag_data = op.data.imag
    shapes, dims = fft_canonicalize_shapes_dims(real_data, op.shapes, op.dims, c2r=True)

    # For all but last dim/shape, do N-D IFFT.
    for shape, dim in zip(shapes[:-1], dims[:-1]):
        real_data, imag_data = _fft_1d(
            real_data,
            imag_data,
            n=mb.const(val=shape),
            dim=mb.const(val=dim),
            norm=op.norm,
            inverse=True,
        )

    # For the last dim/shape, do 1-D IRFFT.
    n: Var = mb.const(val=shapes[-1])
    dim: Var = mb.const(val=dims[-1])
    real_data, imag_data = _restore_conj(
        input_data=_wrap_complex_output(op.outputs[0], real_data, imag_data),
        n=n,
        dim=dim,
    )
    real_data, imag_data = _fft_1d(real_data, imag_data, n, dim, op.norm, inverse=True)
    real_data = _resize_data(real_data, dims=(dim.val,), sizes=(n.val,))

    return real_data

@LowerComplex.register_lower_func(op_type="complex_stft")
def _lower_complex_stft(op: Operation):
    is_complex = types.is_complex(op.input.dtype)

    # check parameters for validity
    if op.win_length and op.win_length.val > op.n_fft.val:
        raise ValueError("Window length must be less than or equal to n_fft")
    if is_complex and op.onesided and op.onesided.val:
        raise ValueError("Onesided is only valid for real inputs")

    real, imag = _stft(
        op.input.real if is_complex else op.input,
        op.input.imag if is_complex else None,
        op.n_fft,
        op.hop_length,
        op.win_length,
        op.window,
        op.normalized,
        op.onesided,
    )

    return _wrap_complex_output(op.outputs[0], real, imag)


@LowerComplex.register_lower_func(op_type="complex_shape")
def _lower_complex_shape(op: Operation):
    return mb.shape(x=op.data.real)

@LowerComplex.register_lower_func(op_type="complex_abs")
def _lower_complex_abs(op: Operation):
    mag_r, mag_i = (mb.square(x=x) for x in (op.x.real, op.x.imag))
    mag = mb.add(
        x=mag_r,
        y=mag_i,
    )
    return mb.sqrt(x=mag)

def _match_and_replace_dialect_op(block, op):
    if not LowerComplex.has_lower_func(op.op_type):
        return False

    with mb.set_before_op(before_op=op):
        lower_res = LowerComplex.get_lower_func(op.op_type)(op)

    if not op.enclosing_block.try_replace_uses_of_var_after_op(
        anchor_op=op,
        old_var=op.outputs[0],
        new_var=lower_res,
    ):
        raise ValueError(f"Unable to lower complex dialect op {op}")
    block.remove_ops([op])
    return True


@block_context_manager
def _lower_complex_dialect_ops_in_block(block):
    for op in list(block.operations):
        _match_and_replace_dialect_op(block, op)

@register_pass(namespace="common")
class lower_complex_dialect_ops(AbstractGraphPass):
    """
    Identify complex data related ops and replace it by using real and imaginary parts separately.
    The goal of this pass it to lower complex dialect ops into core ops.

    This pass also checks if the output is complex. As Core ML doesn't support complex data yet,
    it errors out early when detecting complex output.

    Input graph (`complex` and `complex_real` are complex dialect ops):
        %complex_data = complex(real_data=%real_data, imag_data=%imag_data)
        %real_data = complex_real(data=%complex_data)
        return %real_data

    Output graph (only core ops, no complex dialect ops):
        %complex_data_real = identity(x=%real_data)
        %complex_data_imag = identity(x=%imag_data)
        %real_data = identity(data=%complex_data_real)
        return %real_data
    """

    def apply(self, prog):
        for block in prog.functions.values():
            # Early error out for complex data output.
            for out_var in block.outputs:
                if types.is_complex(out_var.dtype):
                    raise ValueError(
                        "MIL doesn't support complex data as model's output, please "
                        "extract real and imaginary parts explicitly."
                    )

            _lower_complex_dialect_ops_in_block(block)
