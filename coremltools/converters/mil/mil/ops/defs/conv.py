from coremltools.converters.mil.mil.ops.defs._utils import spatial_dimensions_out_shape
from ._op_reqs import *


# rdar://58622145
@register_op(doc_str='TODO')
class conv(Operation):
    input_spec = InputSpec(
            x = TensorInputType(),
            weight = TensorInputType(),
            strides = IntTensorInputType(const=True, optional=True),
            pad_type = StringInputType(const=True),
            pad = IntTensorInputType(const=True, optional=True),
            dilations = IntTensorInputType(const=True, optional=True),
            group = IntInputType(const=True, default=1),
            bias = TensorInputType(const=True, optional=True),
            )

    def __init__(self, **kwargs):
        super(conv, self).__init__(**kwargs)

    def type_inference(self):
        inshape = self.x.shape
        f_shape = self.weight.shape
        kernel_shape = f_shape[2:]
        num_dims = len(inshape) - 2
        C_in = self.x.shape[1]
        group = self.group.val
        if C_in % group != 0:
            msg = '# of input channels {} not divisible by group {}'
            raise ValueError(msg.format(C_in, group))
        if C_in // group != self.weight.shape[1]:
            msg = 'C_in / group = {}/{} != weight[1] ({})'
            raise ValueError(msg.format(C_in, group, self.weight.shape[1]))

        strides = [1]*num_dims if self.strides is None else self.strides.val
        dilations = [1]*num_dims if self.dilations is None \
                else self.dilations.val
        custom_pad = None if self.pad is None else self.pad.val
        N = inshape[0]
        C_out = f_shape[0]
        # spatial dimensions
        d_out_shape = spatial_dimensions_out_shape(
            pad_type=self.pad_type.val,
            input_shape=inshape[2:],
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations,
            custom_pad=custom_pad)
        retshape = [N, C_out] + d_out_shape
        return types.tensor(self.x.dtype, tuple(retshape))


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
            return types.tensor(self.x.dtype, tuple([N, C_out, output_shape[0], output_shape[1]]))

        strides = [1] * spatial_dim_rank if self.strides is None else self.strides.val
        dilations = [1] * spatial_dim_rank if self.dilations is None else self.dilations.val
        kernel_shape = [(kernel_shape[r] -1) * dilations[r] + 1 for r in range(spatial_dim_rank)]

        D_in = in_shape[2:]  # spatial dimensions

        # Deconv's output shape is non-deterministic, we follow TF shape logic here.
        if self.pad_type.val == 'same':
            d_out_shape = [ strides[r] * D_in[r] for r in range(spatial_dim_rank) ]
        elif self.pad_type.val == 'valid':
            d_out_shape = [ strides[r] * D_in[r] + kernel_shape[r] - 1 for r in range(spatial_dim_rank) ]
        elif self.pad_type.val == 'custom':
            if self.pad is None:
                raise ValueError('self.pad must exist if pad_type is custom')
            pad = self.pad.val
            d_out_shape = [ strides[r] * (D_in[r] - 1) + kernel_shape[r] - pad[2*r] - pad[2*r+1]
                            for r in range(spatial_dim_rank) ]

        retshape = [N, C_out] + d_out_shape
        return types.tensor(self.x.dtype, tuple(retshape))
