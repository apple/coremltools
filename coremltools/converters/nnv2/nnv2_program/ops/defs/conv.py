from ._op_reqs import *

def _conv2d_pad(pad_type, num_dims, custom_pad, filter_dims, strides):
    # pad = [t+b, l+r]
    if pad_type == 'same':
        return [d - 1 for d in filter_dims]
    if pad_type == 'valid':
        return [0] * 2
    if pad_type == 'custom':
        if custom_pad is None or len(custom_pad) != 2*num_dims:
            raise ValueError('Invalid custom_pad.')
        return custom_pad
    raise ValueError('Invalid padding pad_type "{}"'.format(pad_type))


# rdar://58622145
@register_op(doc_str='TODO')
class conv(Operation):
    input_spec = InputSpec(
            x = TensorInputType(),
            W = TensorInputType(const=True),
            strides = TensorInputType(const=True, optional=True),
            pad_type = StringInputType(const=True),
            pad = IntTensorInputType(const=True, optional=True),
            dilations = TensorInputType(const=True, optional=True),
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

        strides = [1]*num_dims if self.strides is None else self.strides.val
        dilations = [1]*num_dims if self.dilations is None else self.dilations.val
        pad = None if self.pad is None else self.pad.val
        N = inshape[0]
        C_out = f_shape[0]
        D_in = inshape[2:]  # spatial dimensions
        pad_type = self.pad_type.val
        if pad_type == 'same':
            for k in kernel_shape:
                if k % 2 == 0:
                    msg = "Even kernel size {} is disallowed " + \
                        "under same padding. Use custom padding instead"
                    raise ValueError(msg.format(kernel_shape))
        pad = _conv2d_pad(pad_type,
                num_dims, pad, kernel_shape, strides)

        D_out_shape = [
            int((D_in[r] + pad[r] - dilations[r] * (kernel_shape[r] - 1) - 1) \
                / strides[r] + 1) for r in range(num_dims) ]
        retshape = [N, C_out] + D_out_shape
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
