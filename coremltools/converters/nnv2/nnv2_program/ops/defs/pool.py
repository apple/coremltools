import math
from ._op_reqs import *
from .conv import _conv2d_pad

"""
Pooling Op Superclass
"""
class Pooling(Operation):
    input_spec = InputSpec(
        x=TensorInputType(),
        kernel_sizes=IntTensorInputType(const=True),
        strides=IntTensorInputType(const=True, optional=True),
        pad_type=StringInputType(const=True),
        pad=IntTensorInputType(const=True, optional=True),
    )

    def __init__(self, **kwargs):
        super(Pooling, self).__init__(**kwargs)

    def type_inference(self):
        ksize = self.kernel_sizes.val
        x_shape = self.x.shape
        D_in_rank = len(x_shape) - 2

        strides = [1] * D_in_rank if self.strides is None else self.strides.val
        pad_type = 'valid' if self.pad_type is None else self.pad_type.val.lower()
        pad = None if self.pad is None else self.pad.val
        D_in = x_shape[2:]  # spatial dimensions

        if pad_type == 'same':
            D_out_shape = [int(math.ceil(float(d) / float(s))) for d, s in zip(D_in, strides)]
        else:
            # rdar://59740053 (Padding Calculation for Conv2D does not work for custom padding)
            pad = _conv2d_pad(pad_type, D_in_rank, pad, ksize, strides)
            D_out_shape = [
                ((D_in[r] + pad[r] - (ksize[r] - 1) - 1) // strides[r] + 1) for r in range(D_in_rank)
            ]
        ret_shape = list(x_shape[:2]) + D_out_shape
        return builtins.tensor(self.x.dtype, tuple(ret_shape))


"""
Pooling op implementations
"""

@register_op(doc_str='TODO')
class avg_pool(Pooling):
    input_spec = InputSpec(
        x=TensorInputType(),
        kernel_sizes=IntTensorInputType(const=True),
        strides=IntTensorInputType(const=True, optional=True),
        pad_type=StringInputType(const=True),
        pad=IntTensorInputType(const=True, optional=True),
        exclude_padding_from_average=BoolInputType(const=True, default=False)
    ) + Pooling.input_spec

    def __init__(self, **kwargs):
        super(avg_pool, self).__init__(**kwargs)


@register_op(doc_str='TODO')
class l2_pool(Pooling):
    input_spec = InputSpec(
        x=TensorInputType(),
        kernel_sizes=IntTensorInputType(const=True),
        strides=IntTensorInputType(const=True, optional=True),
        pad_type=StringInputType(const=True),
        pad=IntTensorInputType(const=True, optional=True),
    ) + Pooling.input_spec

    def __init__(self, **kwargs):
        super(l2_pool, self).__init__(**kwargs)


# rdar://58622145
@register_op(doc_str='TODO')
class max_pool(Pooling):
    input_spec = InputSpec(
        x=TensorInputType(),
        kernel_sizes=IntTensorInputType(const=True),
        strides=IntTensorInputType(const=True, optional=True),
        pad_type=StringInputType(const=True),
        pad=IntTensorInputType(const=True, optional=True),
    ) + Pooling.input_spec

    def __init__(self, **kwargs):
        super(max_pool, self).__init__(**kwargs)


