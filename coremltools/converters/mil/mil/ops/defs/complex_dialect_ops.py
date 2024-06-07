#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
This file contains the dialect ops for handling complex numbers.

For example, torch.fft.fft accepts complex input and produces complex outputs, which is not
supported by CoreML. However, we can break the calculation into the real part and imaginary part
to work around the restriction.
The dialect op provided by this file could be used by any frontend (PyTorch, Tensorflow, etc).
For example, during torch frontend translation, the torch's fft_fft op could be translated to
    def fft_fft(context, nodes):
        input_data, n, dim, norm = _get_inputs(context, node, expected=[4])
        fft_res = mb.complex_fft(data=input_data, n=n, dim=dim, norm=norm)
        context.add(fft_res, node.name)
and then the fft dialect op will be lowered into core ops by calculating the real and imaginary
part separately.

There are mainly three types of complex dialect ops:
- Ops where real and imag data has interactions (such as fft).
- Ops where real and imag data go through the non-complex version op separately (such as add).
- Ops where only one of the real/imag data go through the non-complex version (such as shape).

All dialect ops in this file will be lowered into core ops by `lower_complex_dialect_ops` pass.
For adding a new dialect op, see steps in the file docstring of `lower_complex_dialect_ops.py`.
Notice that all dialect op has `complex_` as prefix, because it's required by setting the
`namespace="complex"` in `register_op`.
"""

from typing import Optional, Tuple

import numpy as np

from coremltools.converters.mil.mil import operation, types
from coremltools.converters.mil.mil.input_type import DefaultInputs, InputSpec, TensorInputType
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry
from coremltools.converters.mil.mil.types.symbolic import any_symbolic, is_symbolic
from coremltools.converters.mil.mil.types.type_mapping import (
    infer_complex_dtype,
    infer_fp_dtype_from_complex,
)
from coremltools.converters.mil.mil.var import ComplexVar, Var

register_op = SSAOpRegistry.register_op

_FFT_VALID_NORMS = {"forward", "backward", "ortho"}


def fft_canonicalize_length_dim(
    input_data: Var, length: Optional[Var], dim: Optional[Var], c2r: bool = False
) -> Tuple[int, int]:
    """
    Canonicalize shape and dim for 1-D FFT (based on PyTorch's fft documentation):
    - length: Signal length. If given, the input will either be zero-padded or trimmed to this
      length before computing the FFT.
    - dim: The dimension along which to take the one dimensional FFT.
    - c2r: Use for "complex to real", such as irfft, which takes complex and output real data.
    """
    shapes, dims = fft_canonicalize_shapes_dims(input_data, length, dim, c2r)
    return shapes[0], dims[0]


def fft_canonicalize_shapes_dims(
    input_data: Var, shapes: Optional[Var], dims: Optional[Var], c2r: bool = False
) -> Tuple[Tuple[int], Tuple[int]]:
    """
    Canonicalize shapes and dims for N-D FFT (based on PyTorch's fftn documentation):
    - shapes: Signal size in the transformed dimensions. If given, each dimension dims[i] will
              either be zero-padded or trimmed to the length s[i] before computing the FFT. If a
              length -1 is specified, no padding is done in that dimension.
              Default: s = [input.size(d) for d in dims]
    - dims: Dimensions to be transformed. Default: all dimensions, or the last len(s) dimensions if
            s is given.
    - c2r: Use for "complex to real", such as irfftn, which takes complex and output real data.
    """
    if shapes is not None:
        shapes = shapes.val
        if isinstance(shapes, np.integer):
            shapes = (shapes,)
    if dims is not None:
        dims = dims.val
        if isinstance(dims, np.integer):
            dims = (dims,)

    # Input validation.
    input_rank = input_data.rank
    if dims is not None:
        for dim in dims:
            if dim < -input_rank or dim >= input_rank:
                raise ValueError(f"Invalid dim {dim} in `dims`.")
    if shapes is not None:
        for shape in shapes:
            if shape <= 0:
                raise ValueError(f"Invalid shape {shape} in `shapes`.")

    # Determine if the last dim specified in dims need to be expanded. For IRFFTN, the input is
    # interpreted as a one-sided Hermitian signal in the Fourier domain, as produced by rfftn(), so
    # we need to expand the dim back to the full matrix (with conjugate part not pruned).
    last_dim_expand: bool = shapes is None and c2r

    if shapes is not None:
        if dims is None:
            # Has shape, no dim.
            # Default is last len(s) dimensions.
            dims = tuple(range(input_rank - len(shapes), input_rank))
        else:
            # Has shape, has dim.
            if len(shapes) != len(dims):
                raise ValueError(
                    "shapes and dims must have the same number of elements."
                )
        shapes = tuple(
            shape if shape != -1 else input_data.shape[dim]
            for (shape, dim) in zip(shapes, dims)
        )
    elif dims is None:
        # No shape, no dim.
        dims = tuple(range(input_rank))
        shapes = tuple(input_data.shape)
    else:
        # No shape, has dim.
        shapes = tuple(input_data.shape[dim] for dim in dims)

    # In RFFTN, the output is trimmed (because FFT of real-value input is Hermitian-symmetric, the
    # conjugate part is removed) to ``original_dim // 2 + 1``, so here we do the reverse
    # ``2 * (trimmed_dim - 1)`` to restore the original shape.
    if last_dim_expand:
        target_last_dim_shape = 2 * (input_data.shape[dims[-1]] - 1)
        shapes = shapes[:-1] + (target_last_dim_shape,)

    if len(shapes) != len(dims):
        raise ValueError(
            f"shape ({len(shapes)}) and dim ({len(dims)}) should have same number of elements."
        )

    return shapes, dims


@register_op(namespace="complex")
class complex(operation.Operation):
    """
    Dialect op for constructing a complex data from real and imaginary data.
    """

    input_spec = InputSpec(
        real_data=TensorInputType(type_domain="T"),
        imag_data=TensorInputType(type_domain="T"),
    )

    type_domains = {
        "T": (types.fp32,),
    }

    def type_inference(self):
        if self.real_data.shape != self.imag_data.shape:
            raise ValueError(
                f"The shape of real_data ({self.real_data.shape}) and imag_data "
                f"({self.imag_data.shape}) must match to construct complex data."
            )
        return types.tensor(
            infer_complex_dtype(self.real_data.dtype, self.imag_data.dtype),
            self.real_data.shape,
        )


@register_op(namespace="complex")
class complex_real(operation.Operation):
    """Dialect op for extracting real part of complex data."""

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
    )

    type_domains = {
        "T": (types.complex64,),
    }

    def type_inference(self):
        return types.tensor(
            infer_fp_dtype_from_complex(self.data.dtype), self.data.shape
        )


@register_op(namespace="complex")
class complex_imag(operation.Operation):
    """Dialect op for extracting imaginary part of complex data."""

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
    )

    type_domains = {
        "T": (types.complex64,),
    }

    def type_inference(self):
        return types.tensor(
            infer_fp_dtype_from_complex(self.data.dtype), self.data.shape
        )


@register_op(namespace="complex")
class complex_fft(operation.Operation):
    """
    Dialect op for 1-D FFT. As PyTorch's FFT API has a much more fine-grained control than
    TensorFlow's, the parameters of this dialect op mainly follows `torch.fft.fft`.

    Parameters
    ----------
    data: tensor<\*D, T> (Required)
        * The input tensor.
    n: const i32 (Optional. Default=None)
        * Signal length. If given, the input will either be zero-padded or trimmed to this length
          before computing the FFT.
    dim: const i32 (Optional. Default=``-1``)
        * The dimension along which to take the one dimensional FFT.
    norm: const str (Optional. Default=``backward``)
        * Normalization mode. For the forward transform (fft()), these correspond to:
            * "forward" - normalize by 1/n
            * "backward" - no normalization
            * "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)
        * Calling the backward transform (ifft()) with the same normalization mode will apply an
          overall normalization of 1/n between the two transforms. This is required to make ifft()
          the exact inverse.
        * Default is "backward" (no normalization).

    Returns
    -------
    tensor<\*V, complex64>
        * A complex tensor where real and imag parts have the same shape.
        * If ``n`` is None, real's and imag's shapes are same as the input.
        * If ``n`` is specified, shape is ``V[dim]=n``.

    Attributes
    ----------
    T: fp32, complex64

    References
    ----------
    See `torch.fft.fft <https://pytorch.org/docs/stable/generated/torch.fft.fft.html>`_.
    """

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
        n=TensorInputType(const=True, optional=True, type_domain=types.int32),
        dim=TensorInputType(const=True, optional=True, type_domain=types.int32),
        norm=TensorInputType(const=True, optional=True, type_domain=types.str),
    )

    type_domains = {
        "T": (types.fp32, types.complex64),
    }

    def default_inputs(self):
        return DefaultInputs(
            n=None,
            dim=-1,
            norm="backward",
        )

    def type_inference(self):
        if self.norm.val not in _FFT_VALID_NORMS:
            raise ValueError(
                f"Invalid norm param. Valid options are {_FFT_VALID_NORMS}"
            )
        output_type = (
            self.data.dtype if types.is_complex(self.data.dtype) else types.complex64
        )
        # The shape of FFT output is determined by `n` and `dim`.
        output_shape = list(self.data.shape)
        n, dim = fft_canonicalize_length_dim(self.data, self.n, self.dim)
        output_shape[dim] = n
        return types.tensor(output_type, tuple(output_shape))


@register_op(namespace="complex")
class complex_fftn(operation.Operation):
    """
    Dialect op for N-D FFT. As PyTorch's FFT API has a much more fine-grained control than
    TensorFlow's, the parameters of this dialect op mainly follows `torch.fft.fftn`.

    Parameters
    ----------
    data: tensor<\*D, T> (Required)
        * The input tensor.
    shapes: const tensor<rank(data), i32> (Optional. Default=None)
        * Signal size in the transformed dimensions. If given, each dimension ``dims[i]`` will
          either be zero-padded or trimmed to the length ``shapes[i]`` before computing the FFT. If
          a length ``-1`` is specified, no padding is done in that dimension. If not specified, it's
          equivalent to ``shapes = [data.size(dim) for dim in dims]``.
    dims: const tensor<rank(data), i32> (Optional. Default=None)
        * Dimensions to be transformed. If not specified, it's equivalent to all dimensions, or the
          last ``len(shapes)`` dimensions if ``shapes`` is given.
    norm: const str (Optional. Default=``backward``)
        * Normalization mode. For the forward transform (fftn()), these correspond to:
            * "forward" - normalize by 1/n
            * "backward" - no normalization
            * "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)
          where ``n = prod(shapes)`` is the logical FFT size. Calling the backward transform
          (ifftn()) with the same normalization mode will apply an overall normalization of 1/n
          between the two transforms. This is required to make ifftn() the exact inverse.
        * Default is "backward" (no normalization).

    Returns
    -------
    tensor<\*V, complex64>
        * A complex tensor where real and imag parts have the same shape.
        * If ``shapes`` and ``dims`` are both None, real's and imag's shapes are same as the input.
        * If ``shapes`` or ``dims`` is specified, shape is ``V[dim]=shapes[dim] for dim in dims``.

    Attributes
    ----------
    T: fp32, complex64

    References
    ----------
    See `torch.fft.fftn <https://pytorch.org/docs/stable/generated/torch.fft.fftn.html>`_.
    """

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
        shapes=TensorInputType(const=True, optional=True, type_domain=types.int32),
        dims=TensorInputType(const=True, optional=True, type_domain=types.int32),
        norm=TensorInputType(const=True, optional=True, type_domain=types.str),
    )

    type_domains = {
        "T": (types.fp32, types.complex64),
    }

    def default_inputs(self):
        return DefaultInputs(
            shapes=None,
            dims=None,
            norm="backward",
        )

    def type_inference(self):
        if self.norm.val not in _FFT_VALID_NORMS:
            raise ValueError(
                f"Invalid norm param. Valid options are {_FFT_VALID_NORMS}"
            )
        output_type = (
            self.data.dtype if types.is_complex(self.data.dtype) else types.complex64
        )
        # The shape of FFT output is determined by `shapes` and `dims`.
        shapes, dims = fft_canonicalize_shapes_dims(self.data, self.shapes, self.dims)
        output_shape = list(self.data.shape)
        for shape, dim in zip(shapes, dims):
            output_shape[dim] = shape
        return types.tensor(output_type, tuple(output_shape))


@register_op(namespace="complex")
class complex_rfft(operation.Operation):
    """
    Dialect op for 1-D RFFT. It's similar to 1-D FFT, but the input is real number. The FFT of a
    real signal is Hermitian-symmetric, ``X[i] = conj(X[-i])``, so the output contains only the
    positive frequencies below the Nyquist frequency. To compute the full output, use FFT.

    Parameters
    ----------
    See the ``complex_fft`` op.

    Returns
    -------
    tensor<\*V, complex64>
        * Based on the output of FFT, further remove the redundant conjugate part, which means
          ``V[dim] = V[dim] // 2 + 1``.

    Attributes
    ----------
    T: fp32

    References
    ----------
    See `torch.fft.rfft <https://pytorch.org/docs/stable/generated/torch.fft.rfft.html>`_.
    """

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
        n=TensorInputType(const=True, optional=True, type_domain=types.int32),
        dim=TensorInputType(const=True, optional=True, type_domain=types.int32),
        norm=TensorInputType(const=True, optional=True, type_domain=types.str),
    )

    type_domains = {
        "T": (types.fp32,),
    }

    def default_inputs(self):
        return DefaultInputs(
            n=None,
            dim=-1,
            norm="backward",
        )

    def type_inference(self):
        if types.is_complex(self.data.dtype):
            raise ValueError(
                "RFFT requires real-value input. For complex input, please use FFT."
            )
        output_type = infer_complex_dtype(self.data.dtype, self.data.dtype)
        output_shape = list(self.data.shape)
        n, dim = fft_canonicalize_length_dim(self.data, self.n, self.dim)
        output_shape[dim] = n
        # The shape of RFFT output is FFT after removing redundant conjugate part.
        output_shape[self.dim.val] = output_shape[self.dim.val] // 2 + 1
        return types.tensor(output_type, tuple(output_shape))


@register_op(namespace="complex")
class complex_rfftn(operation.Operation):
    """
    Dialect op for N-D RFFT (rfftn). The FFT of a real signal is Hermitian-symmetric,
    X[i_1, ..., i_n] = conj(X[-i_1, ..., -i_n]) so the full ``complex_fftn`` output contains
    redundant information. ``complex_rfftn`` omits the negative frequencies in the last dimension.

    Parameters
    ----------
    See the ``complex_fftn`` op.

    Returns
    -------
    tensor<\*V, complex64>
        * Based on the output of N-D FFT, further remove the redundant conjugate part in last dim,
          which means ``V[dims[-1]] = V[dims[-1]] // 2 + 1``.

    Attributes
    ----------
    T: fp32

    References
    ----------
    See `torch.fft.rfftn <https://pytorch.org/docs/stable/generated/torch.fft.rfftn.html>`_.
    """

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
        shapes=TensorInputType(const=True, optional=True, type_domain=types.int32),
        dims=TensorInputType(const=True, optional=True, type_domain=types.int32),
        norm=TensorInputType(const=True, optional=True, type_domain=types.str),
    )

    type_domains = {
        "T": (types.fp32,),
    }

    def default_inputs(self):
        return DefaultInputs(
            shapes=None,
            dims=None,
            norm="backward",
        )

    def type_inference(self):
        output_type = infer_complex_dtype(self.data.dtype, self.data.dtype)
        output_shape = list(self.data.shape)
        shapes, dims = fft_canonicalize_shapes_dims(self.data, self.shapes, self.dims)
        for shape, dim in zip(shapes, dims):
            output_shape[dim] = shape
        # The last dim's shape is after removing the redundant conjugate part.
        output_shape[dims[-1]] = output_shape[dims[-1]] // 2 + 1

        return types.tensor(output_type, tuple(output_shape))


@register_op(namespace="complex")
class complex_ifft(operation.Operation):
    """
    Dialect op for IFFT. Computes the one dimensional inverse discrete Fourier transform of input.

    Parameters
    ----------
    All parameters except ``norm`` are same as the ``complex_fft`` op.
    norm: const str (Optional. Default=``backward``)
        * Normalization mode. For the backward transform (ifft()), these correspond to:
            * "forward" - no normalization
            * "backward" - normalize by 1/n
            * "ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)
        * Calling the forward transform (fft()) with the same normalization mode will apply an
          overall normalization of 1/n between the two transforms. This is required to make ifft()
          the exact inverse.
        * Default is "backward" (normalize by 1/n).

    Returns
    -------
    tensor<\*V, T>
        * A complex tensor where real and imag parts have the same shape. The shape is the same as
          the input except for the ``dim``:
            * If ``n`` is None, the shape is same as the input.
            * If ``n`` is specified, the shape at the `dim` is ``V[dim]=n``.

    Attributes
    ----------
    T: complex64

    References
    ----------
    See `torch.fft.ifft <https://pytorch.org/docs/stable/generated/torch.fft.ifft.html>`_.
    """

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
        n=TensorInputType(const=True, optional=True, type_domain=types.int32),
        dim=TensorInputType(const=True, optional=True, type_domain=types.int32),
        norm=TensorInputType(const=True, optional=True, type_domain=types.str),
    )

    type_domains = {
        "T": (types.complex64,),
    }

    def default_inputs(self):
        return DefaultInputs(
            n=None,
            dim=-1,
            norm="backward",
        )

    def type_inference(self):
        output_type = self.data.dtype
        output_shape = list(self.data.shape)
        n, dim = fft_canonicalize_length_dim(self.data, self.n, self.dim)
        output_shape[dim] = n
        return types.tensor(output_type, tuple(output_shape))


@register_op(namespace="complex")
class complex_ifftn(operation.Operation):
    """
    Dialect op for N-D IFFT (ifftn).

    Parameters
    ----------
    All parameters except ``norm`` are same as the ``complex_fftn`` op.
    norm: const str (Optional. Default=``backward``)
        * Normalization mode. For the backward transform (ifftn()), these correspond to:
            * "forward" - no normalization
            * "backward" - normalize by 1/n
            * "ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)
          where n = prod(s) is the logical IFFT size. Calling the forward transform (fftn()) with
          the same normalization mode will apply an overall normalization of 1/n between the two
          transforms. This is required to make ifftn() the exact inverse.
        * Default is "backward" (normalize by 1/n).

    Returns
    -------
    tensor<\*V, T>
        * A complex tensor where real and imag parts have the same shape. The shape is the same as
          the input except for the ``dim`` in ``dims``:
            * If ``shapes`` and ``dims`` are both None, the shape is same as the input.
            * If ``shapes`` or ``dims`` is specified, shape at ``dim`` is ``shapes[dim]``.

    Attributes
    ----------
    T: complex64

    References
    ----------
    See `torch.fft.ifftn <https://pytorch.org/docs/stable/generated/torch.fft.ifftn.html>`_.
    """

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
        shapes=TensorInputType(const=True, optional=True, type_domain=types.int32),
        dims=TensorInputType(const=True, optional=True, type_domain=types.int32),
        norm=TensorInputType(const=True, optional=True, type_domain=types.str),
    )

    type_domains = {
        "T": (types.complex64,),
    }

    def default_inputs(self):
        return DefaultInputs(
            shapes=None,
            dims=None,
            norm="backward",
        )

    def type_inference(self):
        output_type = self.data.dtype
        output_shape = list(self.data.shape)
        shapes, dims = fft_canonicalize_shapes_dims(self.data, self.shapes, self.dims)
        for shape, dim in zip(shapes, dims):
            output_shape[dim] = shape
        return types.tensor(output_type, tuple(output_shape))


@register_op(namespace="complex")
class complex_irfft(operation.Operation):
    """
    Dialect op for IRFFT. Computes the inverse of RFFT. The input is interpreted as a one-sided
    Hermitian signal in the Fourier domain, as produced by rfft(). By the Hermitian property, the
    output will be real-valued.

    Parameters
    ----------
    See the ``complex_ifft`` op for details.

    Returns
    -------
    tensor<\*V, fp32>
        * The shape is the same as the input except for the ``dim``:
            * If ``n`` is None, the shape at the `dim` is ``V[dim] = 2 * (D[dim] - 1)``.
            * If ``n`` is specified, the shape at the `dim` is ``V[dim]=n``.

    Attributes
    ----------
    T: complex64

    References
    ----------
    See `torch.fft.irfft <https://pytorch.org/docs/stable/generated/torch.fft.irfft.html>`_.
    """

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
        n=TensorInputType(const=True, optional=True, type_domain=types.int32),
        dim=TensorInputType(const=True, optional=True, type_domain=types.int32),
        norm=TensorInputType(const=True, optional=True, type_domain=types.str),
    )

    type_domains = {
        "T": (types.complex64,),
    }

    def default_inputs(self):
        return DefaultInputs(
            n=None,
            dim=-1,
            norm="backward",
        )

    def type_inference(self):
        output_type = infer_fp_dtype_from_complex(self.data.dtype)
        output_shape = list(self.data.shape)
        n, dim = fft_canonicalize_length_dim(self.data, self.n, self.dim, c2r=True)
        output_shape[dim] = n
        return types.tensor(output_type, tuple(output_shape))


@register_op(namespace="complex")
class complex_irfftn(operation.Operation):
    """
    Dialect op for N-D IRFFT (irfftn).

    Parameters
    ----------
    See the ``complex_ifftn`` op for details.

    Returns
    -------
    tensor<\*V, fp32>
        * The shape is the same as the input except for:
            * If ``shapes`` and ``dims`` are both None, shape at the last dim ``V[-1]``  is
              ``2 * (D[-1] - 1)``.
            * If ``shapes`` or ``dims`` is specified, shape at ``dim`` is ``shapes[dim]``.

    Attributes
    ----------
    T: complex64

    References
    ----------
    See `torch.fft.irfftn <https://pytorch.org/docs/stable/generated/torch.fft.irfftn.html>`_.
    """

    input_spec = InputSpec(
        data=TensorInputType(type_domain="T"),
        shapes=TensorInputType(const=True, optional=True, type_domain=types.int32),
        dims=TensorInputType(const=True, optional=True, type_domain=types.int32),
        norm=TensorInputType(const=True, optional=True, type_domain=types.str),
    )

    type_domains = {
        "T": (types.complex64,),
    }

    def default_inputs(self):
        return DefaultInputs(
            shapes=None,
            dims=None,
            norm="backward",
        )

    def type_inference(self):
        output_type = infer_fp_dtype_from_complex(self.data.dtype)
        output_shape = list(self.data.shape)
        shapes, dims = fft_canonicalize_shapes_dims(
            self.data, self.shapes, self.dims, c2r=True
        )
        for shape, dim in zip(shapes, dims):
            output_shape[dim] = shape
        return types.tensor(output_type, tuple(output_shape))


@register_op(namespace="complex")
class complex_shape(operation.Operation):
    """
    Returns a 1-dimensional tensor with the shape of the input complex tensor.

    Parameters
    ----------
    x: tensor<[*?], T> (Required)
        * Input tensor.

    Returns
    -------
    tensor<K, i32>
        * Shape of the input tensor.
        * ``K = x.real.rank``.

    Attributes
    ----------
    T: complex64
    """

    input_spec = InputSpec(x=TensorInputType(type_domain="T"))

    type_domains = {
        "T": (types.complex64,),
    }

    # If type_inference or value_inference is invoked when the graph is being constructed,
    # x.real and x.imag may not be set since the complex lowering pass hasn't yet been invoked.
    # self.x should already have the shape set, so use that instead.

    def type_inference(self):
        if not isinstance(self.x, ComplexVar):
            raise ValueError("x must be a ComplexVar.")
        input_rank = self.x.rank
        return types.tensor(types.int32, tuple([input_rank]))

    def value_inference(self):
        if any_symbolic(self.x.shape):
            # convert elements in shape to int32
            res = [x if is_symbolic(x) else np.int32(x) for x in self.x.shape]
            return np.array(res)
        else:
            return np.array(self.x.shape).astype(np.int32)

@register_op(namespace="complex")
class complex_abs(operation.Operation):
    """
    Returns the absolute value of a complex tensor.

    Parameters
    ----------
    x: tensor<[*d], T> (Required)

    Returns
    -------
    tensor<[*d], fp32>
        * A float tensor with the same shape as ``x``

    Attributes
    ----------
    T: complex64
    """

    input_spec = InputSpec(x=TensorInputType(type_domain="T"))

    type_domains = {
        "T": (types.complex64,),
    }

    def type_inference(self):
        if not isinstance(self.x, ComplexVar):
            raise ValueError("x must be a ComplexVar.")
        return types.tensor(infer_fp_dtype_from_complex(self.x.dtype), self.x.shape)

@register_op(namespace="complex")
class complex_stft(operation.Operation):
    """
    Dialect op for 1-D STFT.

    Parameters
    ----------
    input: tensor<\*D, T> (Required)
        * The input tensor.
    n_fft: const i32 (Required)
        * Size of the fourier transform.
    hop_length: const i32 (Optional)
        * Stride between window frames of the input tensor.
    win_length: const i32 (optional)
        * The size of the window frame.
    window: tensor<1, win_length> (optional)
        * The window to apply to the input signal before performing the fourier transform.
    normalized: const bool (optional, Default=``false``)
        * Whether to normalize the results of the STFT
    onesided: const bool (optional, Default=``true``)
        * For real-valued inputs, whether to return the first half of the results.

    Returns
    -------
    tensor<\*V, complex64>
        * A complex tensor where real and imag parts have the same shape.

    Attributes
    ----------
    T: fp32, complex64

    References
    ----------
    See `torch.stft <https://pytorch.org/docs/stable/generated/torch.stft.html>`_.
    """

    input_spec = InputSpec(
        input=TensorInputType(type_domain="T"),
        n_fft=TensorInputType(const=True, type_domain=types.int32),
        hop_length=TensorInputType(const=True, optional=True, type_domain=types.int32),
        win_length=TensorInputType(const=True, optional=True, type_domain=types.int32),
        window=TensorInputType(const=True, optional=True, type_domain=types.fp32),
        normalized=TensorInputType(const=True, optional=True, type_domain=types.bool),
        onesided=TensorInputType(const=True, optional=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (types.fp32, types.complex64),
    }

    def default_inputs(self):
        return DefaultInputs(
            hop_length = None,
            win_length = None,
            window = None,
            normalized = False,
            onesided = True,
        )

    def type_inference(self):
        output_type = (types.complex64)

        # STFT shape is [B x N x T], where N is the number of frequency bins
        # and T is the number of windows
        # B is 1 for a time series or 2 for a batch of time series

        window_length = self.n_fft.val
        hop = self.hop_length.val if self.hop_length else self.n_fft.val // 4

        # if onesided is true, the input is real valued
        # because of Hermitian symmetry, we only need to calculate the FFT
        # for the first half of the frequencies
        if self.onesided and self.onesided.val:
            window_length = window_length // 2 + 1

        frames = (self.input.shape[-1] - self.n_fft.val) // hop + 1
        output_shape = [window_length, frames]

        # add back rank if needed
        if self.input.rank == 2:
            output_shape = [self.input.shape[0]] + output_shape

        return types.tensor(output_type, tuple(output_shape))
