from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.input_types import ImageType, Shape, EnumeratedShapes

from coremltools.converters.mil.mil import Builder as mb


@register_pass(namespace='common')
def image_input_preprocess(prog):
    """
    Plug in transpose for image input that were NHWC to NCHW.

    Steps:
    1. Check whether there are any inputs that the users specify is ImageType.
    2. Check the channel's dimension for all ImageType
        a) channel_first == True
            We do not modify this input, since channel_first is the intended
            behaviour for feeding images for optimal performance
        b) channel_first == False
            We convert the input into a "channel_first" input, and plug in a 
            transpose for the input to maintain the remaining graph's dimensionality.
    """
    for f_name, f in prog.functions.items():
        if f_name == 'main':
            # We need to make sure main exist and start here.
            _image_input_preprocess(prog)

def _transform_to_channel_first(shape):
    if isinstance(shape, tuple):
        shape = list(shape)
        return tuple(shape[:-3] + [shape[-1]] + shape[-3:-1])
    else:
        return shape[:-3] + [shape[-1]] + shape[-3:-1]

def _image_input_preprocess(prog):
    main_input_types = list(prog.main_input_types)
    for idx, input_type in enumerate(main_input_types):
        if isinstance(input_type, ImageType) and not input_type.channel_first:
            name = input_type.name
            # Build new ImageType to change data layout
            if isinstance(input_type.shape, Shape):
                new_shape = _transform_to_channel_first(input_type.shape.shape)
                new_default = _transform_to_channel_first(input_type.shape.default)
                shape_type = Shape(shape=new_shape, default=new_default)
            elif isinstance(input_type.shape, EnumeratedShapes):
                shape_list = []
                for shape in input_type.shape.shapes:
                    shape_list.append(_transform_to_channel_first(shape))
                shape_type = EnumeratedShapes(shapes=shape_list,
                                              default=_transform_to_channel_first(input_type.shape.default))
            new_image_type = ImageType(name=name,
                                       shape=shape_type,
                                       bias=input_type.bias,
                                       scale=input_type.scale,
                                       color_layout=input_type.color_layout,
                                       channel_first=True)
            main_input_types[idx] = new_image_type

            # Reconstruct Placeholder of Function inputs.
            placeholder_op = prog.functions['main'].placeholder_inputs[name]
            old_var = placeholder_op.outputs[0]
            nchw_shape = _transform_to_channel_first(placeholder_op.sym_shape)
            placeholder_op.__init__(nchw_shape, dtype=placeholder_op.dtype, name=placeholder_op.name)

            # Update Function input var
            prog.functions['main']._input_dict[name] = placeholder_op.outputs[0]
            prog.functions['main'].function_inputs = tuple(prog.functions['main']._input_dict.values())

            # Add transpose into graph (Transpose from NCHW back to NHWC)
            curr_block = prog.functions['main']
            curr_var = prog.functions['main'].inputs[name]

            perm = list(range(curr_var.rank))
            perm = perm[:-3] + [perm[-2], perm[-1], perm[-3]]
            with curr_block:
                new_input = mb.transpose(x=curr_var,
                                         perm=perm,
                                         before_op=prog.functions['main'].operations[0],
                                         name=curr_var.name+"__transpose_from_nchw__")
            curr_block.replace_uses_of_var_after_op(anchor_op=None,
                                                    old_var=old_var,
                                                    new_var=new_input)
    prog.main_input_types = tuple(main_input_types)
