from ._op_reqs import *

@register_op(doc_str='TODO (rdar://58622145), https://quip-apple.com/1mNfAW4JhWR9#PDe9CARK6vT')
class upsample_nearest_neighbor(Operation):
    input_spec = InputSpec(
        x = TensorInputType(),
        upscale_factor_height = IntInputType(const=True, default=1),
        upscale_factor_width = IntInputType(const=True, default=1)
    )

    def __init__(self, **kwargs):
        super(upsample_nearest_neighbor, self).__init__(**kwargs)

    def type_inference(self):
        if len(self.x.shape) < 3:
            raise ValueError("input to the \"upsample_nearest_neighbor\" op must have rank at least 3")

        in_shape = self.x.shape
        ret_shape = list(in_shape)
        ret_shape[-1] *= self.upscale_factor_width.val
        ret_shape[-2] *= self.upscale_factor_height.val
        return builtins.tensor(self.x.dtype, ret_shape)


@register_op(doc_str='TODO (rdar://58622145), https://quip-apple.com/1mNfAW4JhWR9#PDe9CA9aGcP')
class upsample_bilinear(Operation):
    input_spec = InputSpec(
        x = TensorInputType(),
        scale_factor_height = IntOrFloatInputType(const=True, default=1),
        scale_factor_width = IntOrFloatInputType(const=True, default=1),
        align_corners = BoolInputType(const=True, default=True)
    )

    def __init__(self, **kwargs):
        super(upsample_bilinear, self).__init__(**kwargs)

    def type_inference(self):
        if len(self.x.shape) < 3:
            raise ValueError("input to the \"upsample_bilinear\" op must have rank at least 3")

        in_shape = self.x.shape
        ret_shape = list(in_shape)
        ret_shape[-1] = np.floor(self.scale_factor_width.val * ret_shape[-1])
        ret_shape[-2] = np.floor(self.scale_factor_height.val * ret_shape[-2])
        return builtins.tensor(self.x.dtype, ret_shape)


@register_op(doc_str='TODO (rdar://58622145), https://quip-apple.com/1mNfAW4JhWR9#PDe9CAaiOQP')
class resize_bilinear(Operation):
    input_spec = InputSpec(
        x = TensorInputType(),
        target_size_height = IntInputType(const=True, default=1),
        target_size_width = IntInputType(const=True, default=1),
        sampling_mode = StringInputType(const=True, default="DEFAULT")
    )

    def __init__(self, **kwargs):
        super(resize_bilinear, self).__init__(**kwargs)

    def type_inference(self):
        if len(self.x.shape) < 3:
            raise ValueError("input to the \"resize_bilinear\" op must have rank at least 3")

        if self.sampling_mode.val not in {"STRICT_ALIGN_CORNERS", "ALIGN_CORNERS", "DEFAULT", "OFFSET_CORNERS"}:
            raise ValueError("\"resize_bilinear\" op: unrecognized sampling mode \"{}\"".format(self.sampling_mode))

        in_shape = self.x.shape
        ret_shape = list(in_shape)
        ret_shape[-1] = self.target_size_width.val
        ret_shape[-2] = self.target_size_height.val
        return builtins.tensor(self.x.dtype, ret_shape)

