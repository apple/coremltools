from ._op_reqs import *

def _aggregated_pad(pad_type, kernel_shape, dilations=None, custom_pad=None):
    """
    custom_pad: Required iff pad_type == 'custom'. custom_pad[2*i],
    custom_pad[2*i+1] are left/right custom padding for spatial dim i.

    kernel_shape: [kH, kW, ...]: spatial kernel dims (excluding channels)

    Returns:
        pad[i] = left + right padding for dim i
    """
    num_spatial_dims = len(kernel_shape)
    if dilations is None:
        dilations = [1]*num_spatial_dims
    if pad_type == 'same':
        effective_ks = [(k-1)*d+1 for k, d in zip(kernel_shape, dilations)]
        return [k - 1 for k in effective_ks]
    if pad_type == 'valid':
        return [0] * num_spatial_dims
    if pad_type == 'custom':
        if custom_pad is None or len(custom_pad) != 2*num_spatial_dims:
            raise ValueError('Invalid custom_pad.')
        return [custom_pad[2*d] + custom_pad[2*d+1] for d in
                range(num_spatial_dims)]
    raise ValueError('Invalid padding pad_type "{}"'.format(pad_type))

# rdar://58622145
@register_op(doc_str='TODO')
class conv(Operation):
    input_spec = InputSpec(
            x = TensorInputType(),
            W = TensorInputType(),
            strides = IntTensorInputType(const=True, optional=True),
            pad_type = StringInputType(const=True),
            pad = IntTensorInputType(const=True, optional=True),
            dilations = IntTensorInputType(const=True, optional=True),
            group = IntInputType(const=True, default=1),
            B = TensorInputType(const=True, optional=True),
            )

    def __init__(self, **kwargs):
        super(conv, self).__init__(**kwargs)

    def type_inference(self):
        inshape = self.x.shape
        f_shape = self.W.shape
        kernel_shape = f_shape[2:]
        num_dims = len(inshape) - 2
        C_in = self.x.shape[1]
        group = self.group.val
        if C_in % group != 0:
            msg = '# of input channels {} not divisible by group {}'
            raise ValueError(msg.format(C_in, group))
        if C_in // group != self.W.shape[1]:
            msg = 'C_in / group = {}/{} != W[1] ({})'
            raise ValueError(msg.format(C_in, group, self.W.shape[1]))

        strides = [1]*num_dims if self.strides is None else self.strides.val
        dilations = [1]*num_dims if self.dilations is None \
                else self.dilations.val
        custom_pad = None if self.pad is None else self.pad.val
        N = inshape[0]
        C_out = f_shape[0]
        D_in = inshape[2:]  # spatial dimensions
        pad_type = self.pad_type.val
        pad = _aggregated_pad(pad_type, kernel_shape, dilations, custom_pad)

        d_out_shape = [
            (D_in[r] + pad[r] - dilations[r] * (kernel_shape[r] - 1) - 1) \
                // strides[r] + 1 for r in range(num_dims) ]
        retshape = [N, C_out] + d_out_shape
        return builtins.tensor(self.x.dtype, tuple(retshape))


@register_op(doc_str='TODO')
class conv_transpose(Operation):
    input_spec = InputSpec(
        x = TensorInputType(),
        weight = TensorInputType(const=True),
        bias = TensorInputType(const=True, optional=True),
        pad = IntTensorInputType(const=True, optional=True),
        output_shape = IntTensorInputType(const=True, optional=True),
        pad_type = StringInputType(const=True, default='valid'),
        strides = TensorInputType(const=True, default=[1, 1]),
        dilations = TensorInputType(const=True, default=[1, 1]),
        group = IntInputType(const=True, default=1),
        )

    def __init__(self, **kwargs):
        super(conv_transpose, self).__init__(**kwargs)

    def type_inference(self):
        # Input shape is [N, C_in, H, W]
        in_shape = self.x.shape
        # Weight shape is [H, W, C_out, C_in]
        f_shape = self.weight.shape
        kernel_shape = f_shape[:2]
        spatial_dim_rank = len(in_shape) - 2
        N = in_shape[0]
        C_in = self.x.shape[1]
        C_out = f_shape[-2]
        group = self.group.val
        if C_out % group != 0:
            msg = '# of input channels {} not divisible by group {}'
            raise ValueError(msg.format(C_in, group))

        # If output shape is given, return it
        if self.output_shape is not None:
            output_shape = self.output_shape.val
            return builtins.tensor(self.x.dtype, tuple([N, C_out, output_shape[0], output_shape[1]]))

        strides = [1] * spatial_dim_rank if self.strides is None else self.strides.val
        dilations = [1] * spatial_dim_rank if self.dilations is None else self.dilations.val
        pad = None if self.pad is None else self.pad.val

        D_in = in_shape[2:]  # spatial dimensions
        if pad is None:
            pad = [0] * spatial_dim_rank
        d_out_shape = [ strides[r] * (D_in[r] - 1) + ((kernel_shape[r] - 1) * dilations[r]) - pad[2*r] - pad[2*r+1] + 1
                        for r in range(spatial_dim_rank) ]
        retshape = [N, C_out] + d_out_shape
        return builtins.tensor(self.x.dtype, tuple(retshape))
