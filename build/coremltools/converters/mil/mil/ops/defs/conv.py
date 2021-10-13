#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import (
    DefaultInputs,
    FloatTensorInputType,
    IntInputType,
    IntTensorInputType,
    Operation,
    OrderedDict,
    TensorInputType,
    types,
    ScalarOrTensorInputType,
    StringInputType,
)
from coremltools.converters.mil.mil.input_type import InputSpec
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs._utils import spatial_dimensions_out_shape


@register_op(doc_str="")
class conv(Operation):
    """
    Perform convolution over input. Currently supports only 1-D and 2-D
    convolution.

    Parameters
    ----------
    x: tensor<[n, C_in, \*d_in], T> (Required)

        * ``d_in`` are (possibly runtime-determined) spatial dimensions. For example,
          ``d_in = [224, 224]`` for 2D convolution.
        * ``1 <= len(d_in) <= 2``: Only 1-D and 2-D convolution.
        * ``C_in`` is the number of input channels or depth dimensions.
        * ``n``  is the batch dimension.

    weight: tensor<[C_out, C_in/groups, \*K], T> (Required)

        * Filter weights.
        * ``C_in`` is the number of input channels.
        * ``C_in`` must be divisible by ``groups``.
        * ``K`` are kernel sizes. For example, ``K = [KH, KW]`` for 2-D conv.
        * When ``dilations`` is not all ``1``, ``weight`` has to be ``const``
          at compile time

    strides: const tensor<[S], i32> (Optional)

        * Default to one vector of length equal to the number of spatial dimensions.
        * Strides along each of the spatial dimensions.
        * ``S == len(d_in)``.

    pad_type: const str (Required)

        Must be one of the following:

            * ``valid``: No padding. This is equivalent to custom pad with
              ``pad[2*i] == pad[2*i+1] == 0, for i=0,...,len(d_in)-1``.
            * ``custom``: Specify custom padding in the parameter ``pad``.
            * ``same``: input is padded such that out spatial shapes are
              ``d_out[i] = ceil(d_in[i] / strides[i])``.

        Specifically, for ``i = 0,..,,len(d_in)-1``, the equivalent paddings are
        calculated as follows:

            * ``dilated_kernel = (K[i] - 1) * dilate[i] + 1``
            * if ``dilated_kernel`` is odd,
              ``padding[2*i] = padding[2*i+1] = floor(dilated_kernel / 2)``
            * Otherwise:
              ``padding[2*i] = ceil((dilated_kernel - 1) / 2)``,
              ``padding[2*i+1] = floor((dilated_kernel - 1) / 2)``

    pad: const tensor<[P], i32> (Optional. Default to all zeros)

        * ``len(P) = 2 * len(d_in)``
        * ``pad`` should be specified if and only if ``pad_type == custom``,
          otherwise errors occur.
        * ``pad`` represents the number of elements to pad before and after each
          dimension. Specifically, ``pad[0], pad[1]`` are the pad size before / after
          spatial dimension 0, ``pad[2], pad[3]`` are the pad size before / after
          spatial dimension 1, etc.

    dilations: const tensor<[S], i32> (Optional. Default to all 1s)

        * Dilation value along each spatial dimension in ``d_in``.
          See `visualization <https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md>`_.
        * ``S == len(d_in)``.

    groups: const tensor<[], i32> (Optional, default to 1)

        * Input and output channels are split by ``groups``.
        * ``C_in`` must be divisible by ``groups``.
        * Maximum value for group is ``C_in``, in which case it is a depthwise
          convolution.

        For examples (assuming ``C_in = 16, C_out = 32``):

            * ``groups == 1``, ``weight`` has shape ``[32, 16, KH, KW]``: All input
              channels are convolved with the ``weight`` kernel to produce all output
              channels.
            * ``groups == 2``, ``weight`` has shape ``[32, 8, KH, KW]``: Input
              channels 0~7 are convolved with half of the ``weight`` kernel to produce
              output channels 0~15. Similarly, input channels 8~15 are convolved with
              the other half of ``weight`` to product output channels 16~31.
            * ``groups == C_in``, ``weight`` has shape ``[32, 1, KH, KW]``: Each input
              channel is convolved with its own set of filters and each produce
              ``C_out / C_in = 2`` channels. This is equivalent to depthwise
              convolution.

    bias: const tensor<[C_out],T> (Optional, default to all 0)
        * Bias along output channels.

    Returns
    -------
    tensor<[n, C_out, \*d_out], T>
        * Output activation has the same rank and spatial dimension as the input.
          That is, ``len(d_out) == len(d_in)``.
        * For ``i=0,..,len(d_in)-1, d_out[i] = floor [(D_in[i] + pad[2*i] +
          pad[2*i+1] - (K[i]-1)*dilations[i] - 1) / strides[i] ] + 1``

    Attributes
    ----------
    T: fp32

    See Also
    --------
    conv_transpose
    """

    input_spec = InputSpec(
        x=FloatTensorInputType(),
        weight=FloatTensorInputType(),
        bias=FloatTensorInputType(const=True, optional=True),
        strides=IntTensorInputType(const=True, optional=True),
        pad_type=StringInputType(const=True, optional=True),
        pad=IntTensorInputType(const=True, optional=True),
        dilations=IntTensorInputType(const=True, optional=True),
        groups=IntInputType(const=True, optional=True),
    )

    def default_inputs(self):
        num_spatial_dims = self.x.rank - 2
        return DefaultInputs(
            bias=None,
            strides=[1]*num_spatial_dims,
            pad_type="valid",
            pad=[0]*num_spatial_dims*2,
            dilations=[1]*num_spatial_dims,
            groups=1,
            )

    def __init__(self, **kwargs):
        super(conv, self).__init__(**kwargs)

    def type_inference(self):
        inshape = self.x.shape
        f_shape = self.weight.shape
        kernel_shape = f_shape[2:]
        num_dims = len(inshape) - 2
        C_out = f_shape[0]
        C_in = self.x.shape[1]
        groups = self.groups.val

        if self.bias is not None and self.bias.val.shape[0] != C_out:
            msg = "# of bias values {} not equal to # output channels {}"
        if C_in % groups != 0:
            msg = "# of input channels {} not divisible by groups {}"
            raise ValueError(msg.format(C_in, groups))
        if C_in // groups != self.weight.shape[1]:
            msg = "C_in / groups = {}/{} != weight[1] ({})"
            raise ValueError(msg.format(C_in, groups, self.weight.shape[1]))

        strides = self.strides.val
        dilations = self.dilations.val
        # Ignore self.pad if pad_type != custom
        custom_pad = None if self.pad_type.val != 'custom' else self.pad.val

        if self.weight.val is None and any([True if d > 1 else False for d in dilations]):
            raise ValueError("Convolution with dynamic weights does not support dilations!")

        N = inshape[0]
        C_out = f_shape[0]
        # spatial dimensions
        d_out_shape = spatial_dimensions_out_shape(
            pad_type=self.pad_type.val,
            input_shape=inshape[2:],
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations,
            custom_pad=custom_pad,
        )
        retshape = [N, C_out] + d_out_shape
        return types.tensor(self.x.dtype, tuple(retshape))


@register_op(doc_str="")
class conv_quantized(conv):
    """
    Note: This is experimental and may change in the future.
    Supports weight quantization for parameters while performing convolution over input.
    ``W_float = W_quantized * scale + bias``.

    Parameters
    ----------
    In addition to convolutional layer parameters, the following additional parameters
    are required.

    quantization_type: const str (Required)
        * One of ``linear``, or ``lut``.

    nbits: const tensor<[], i32> (Optional. Default to 8)
        * Denotes the bit-width of the quantization. ``1 <= nbits <= 8``.

    quant_scale: tensor<*?, T> (Required)
        * Denotes the scale of quantization.

    quant_bias: tensor<*?, T> (Required)
        * Denotes the bias that is used to quantize/dequantize.

    Returns
    -------
    tensor<[n, C_out, *d_out], T>
        * Output activation has the same rank and spatial dimension as the input.
          That is, ``len(d_out) == len(d_in)``.

    Attributes
    ----------
    T: fp32
    """
    input_spec = InputSpec(
        x=TensorInputType(),
        weight=TensorInputType(),
        bias=TensorInputType(const=True, optional=True),
        quantization_type=StringInputType(const=True),
        nbits=IntInputType(const=True, optional=True),
        quant_scale=ScalarOrTensorInputType(const=True),
        quant_bias=ScalarOrTensorInputType(const=True),
        strides=IntTensorInputType(const=True, optional=True),
        pad_type=StringInputType(const=True, optional=True),
        pad=IntTensorInputType(const=True, optional=True),
        dilations=IntTensorInputType(const=True, optional=True),
        groups=IntInputType(const=True, optional=True),
        )

    def default_inputs(self):
        return super().default_inputs() + \
            DefaultInputs(
                nbits=8,
                )

    def __init__(self, **kwargs):
        super(conv_quantized, self).__init__(**kwargs)


@register_op(doc_str="")
class conv_transpose(Operation):
    """
    Perform transposed convolution (also known as deconvolution and fractionally
    stride convolution) over input. ``conv_transpose`` can also be used to compute
    the gradient of conv. Currently supports only 1-D and 2-D.

    Parameters
    ----------

    x: tensor<[n,C_in,*D_in],T> (Required)
        * Input data.
        * ``D_in`` are spatial dimensions.
        * ``1 <= len(D_in) <= 2``.
        * ``C_in`` is the number of input channels.

    weight: const tensor<[C_in,C_out/groups,*D_in], T> (Required)
        * Filter weights. ``C_in, C_out`` are the number of input and output channels
          respectively.
        * ``D_in`` are spatial dimensions. ``1 <= len(D_in) <= 2``.

    bias: const tensor<[C_out],T> (Optional, default to all 0)
        * Bias added along output channels.

    pad: const tensor<[P],i32> (Optional, default to all 0s)
        * Number of elements to pad before and after each dimension.
        * ``P == 2 * len(D_in)``.
        * ``pad[2*i], pad[2*i+1]`` are pad sizes before and after
          dimension ``i``, where ``0 <= i < len(D_in)``.

    output_shape: const tensor<[P],i32> (Optional, default None)
        * Expected output shape. The first two dim must be ``[n, C_out]``.
        * The output shape of conv_transpose is underdetermined in general,
        * because conv can map multiple input shape to a single output shape. For example, for 'same' padding mode, `conv_out = ceil(conv_in/stride)`. Hence we need `output_shape` when this occurs.

    pad_type: const tensor<[P],i32> (Optional, default valid)
        * One of ``same``, ``valid``, or ``custom``.

    strides: const tensor<[S],i32> (Optional. Default to all 1s)
        * Stride along each of the spatial dimensions.
        * ``S == len(D_in)``.

    dilations: const tensor<[S],i32> (Optional. Default to all 1s)
        * Dilation value along each spatial dimension in ``d_in``. See ``conv``.
        * ``S == len(D_in)``.

    groups: const tensor<[], i32> (Optional. Default to 1)
        * Input and output channels are separated into ``groups``.
        * ``C_in`` and ``C_out`` must be divisible by the number of groups.
          See ``conv`` for examples.

    Returns
    -------
    tensor<[n,C_out,*D_out],T>
        * ``D_out[i] = (D_in[i]-1) * strides[i] + pad[2*i] + pad[2*i+1] -
          (K[i] - 1) * dilations[i] + 1)] for i = 0, 1``.

    Attributes
    ----------
    T: fp32

    See Also
    --------
    conv
    """

    input_spec = InputSpec(
        x=FloatTensorInputType(),  # [n, C_in, spatial_dims]
        weight=FloatTensorInputType(const=True),  # [C_out, C_in, spatial_dims]
        bias=FloatTensorInputType(const=True, optional=True),
        pad=IntTensorInputType(const=True, optional=True),
        output_shape=IntTensorInputType(const=True, optional=True),
        pad_type=StringInputType(const=True, optional=True),
        strides=TensorInputType(const=True, optional=True),
        dilations=TensorInputType(const=True, optional=True),
        groups=IntInputType(const=True, optional=True),
    )

    def default_inputs(self):
        num_spatial_dims = self.x.rank - 2
        return DefaultInputs(
            bias=None,
            pad=[0]*2*num_spatial_dims,
            output_shape=None,
            pad_type="valid",
            strides=[1]*num_spatial_dims,
            dilations=[1]*num_spatial_dims,
            groups=1,
            )

    def __init__(self, **kwargs):
        super(conv_transpose, self).__init__(**kwargs)

    def type_inference(self):
        # Input shape is [n, C_in, spatial_dims]
        in_shape = self.x.shape
        # Weight shape is [C_in, C_out/group, spatial_dims]
        f_shape = self.weight.shape
        kernel_shape = f_shape[2:]
        spatial_dim_rank = len(in_shape) - 2
        N = in_shape[0]
        C_in = self.x.shape[0]
        groups = self.groups.val
        C_out = f_shape[1] * groups

        if self.bias is not None and self.bias.val.shape[0] != C_out:
            msg = "# of bias values {} not equal to # output channels {}"
            raise ValueError(msg.format(self.bias.val.shape[0], C_out))
        if C_out % groups != 0:
            msg = "# of input channels {} not divisible by groups {}"
            raise ValueError(msg.format(C_in, groups))

        # If output shape is given, return it
        if self.output_shape is not None:
            output_shape = self.output_shape.val
            assert output_shape[0] == N
            assert output_shape[1] == C_out
            return types.tensor(
                self.x.dtype, tuple(output_shape)
            )

        strides = self.strides.val
        dilations = self.dilations.val
        kernel_shape = [
            (kernel_shape[r] - 1) * dilations[r] + 1 for r in range(spatial_dim_rank)
        ]

        D_in = in_shape[2:]  # spatial dimensions

        # Deconv's output shape is non-deterministic, we follow TF shape logic here.
        if self.pad_type.val == "same":
            d_out_shape = [strides[r] * D_in[r] for r in range(spatial_dim_rank)]
        elif self.pad_type.val == "valid":
            d_out_shape = [
                strides[r] * (D_in[r]-1) + kernel_shape[r]
                for r in range(spatial_dim_rank)
            ]
        elif self.pad_type.val == "custom":
            if self.pad is None:
                raise ValueError("self.pad must exist if pad_type is custom")
            pad = self.pad.val
            d_out_shape = [
                strides[r] * (D_in[r] - 1)
                + kernel_shape[r]
                - pad[2 * r]
                - pad[2 * r + 1]
                for r in range(spatial_dim_rank)
            ]

        retshape = [N, C_out] + d_out_shape
        return types.tensor(self.x.dtype, tuple(retshape))
