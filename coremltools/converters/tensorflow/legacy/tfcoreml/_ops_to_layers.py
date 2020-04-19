from tensorflow.python.util import compat
from . import _layers
from . import _layers_common

_CORE_OPS = {
    # core
    "Abs": _layers.abs,
    "Add": _layers.add,
    "ArgMax": _layers.argmax,
    "AvgPool": _layers.avgpool,
    "BatchNormWithGlobalNormalization": _layers.batchnorm,
    "BatchToSpaceND": _layers.batch_to_space,
    "BiasAdd": _layers.add,
    "Concat": _layers.concat,
    "ConcatV2": _layers.concat,
    "Const": _layers.constant,
    "Conv2D": _layers.conv2d,
    "Conv2DBackpropInput": _layers.deconv2d,
    "CropAndResize": _layers.crop_and_resize,
    "DepthToSpace": _layers.depth_to_space,
    "DepthwiseConv2dNative": _layers.conv2d,
    "Elu": _layers.elu,
    "Exp": _layers.exp,
    "ExtractImagePatches": _layers.extract_image_patches,
    "FusedBatchNorm": _layers.batchnorm,
    "Identity": _layers.identity,
    "LeakyRelu": _layers.leaky_relu,
    "Log": _layers.log,
    "LRN": _layers.lrn,
    "Max": _layers.reduce_max,  # TODO - there're unsupported configurations
    "Maximum": _layers.maximum,
    "MaxPool": _layers.maxpool,
    "MatMul": _layers.inner_product,
    "Mean": _layers.mean,  # TODO - there're unsupported configurations
    "Min": _layers.reduce_min,  # TODO - there're unsupported configurations
    "Minimum": _layers.minimum,
    "MirrorPad": _layers.mirror_pad,
    "Mul": _layers.mul,
    "Neg": _layers.neg,
    "OneHot": _layers.one_hot,
    "Pad": _layers.pad,
    "Placeholder": _layers.placeholder,
    "Pow": _layers.pow,
    "Prod": _layers.product,  # TODO - there're unsupported configurations
    "QuantizedBiasAdd": _layers.add,
    "QuantizedConv2D": _layers.conv2d,
    "QuantizedMatMul": _layers.inner_product,
    "QuantizedRelu": _layers.relu,
    "QuantizedReshape": _layers.reshape,
    "RealDiv": _layers.real_div,
    "Reciprocal": _layers.reciprocal,
    "Relu": _layers.relu,
    "Relu6": _layers.relu6,
    "ResizeBilinear": _layers.resize_bilinear,  # TODO: there're unsupported configurations
    "ResizeNearestNeighbor": _layers.resize_nearest_neighbor,
    "Reshape": _layers.reshape,
    "Rsqrt": _layers.rsqrt,
    "Sigmoid": _layers.sigmoid,
    "Softmax": _layers.softmax,
    "SpaceToBatchND": _layers.space_to_batch,
    "SpaceToDepth": _layers.space_to_depth,
    "Split": _layers.split,
    "Sqrt": _layers.sqrt,
    "Square": _layers.square,
    "SquaredDifference": _layers.squared_difference,
    "Sub": _layers.sub,
    "Sum": _layers.reduce_sum,  # TODO - there're unsupported configurations
    "Tanh": _layers.tanh,
    "Transpose": _layers.transpose,  # TODO - only works 4D tensors
}

_NON_CORE_OPS = {
    # dummy for CoreML
    "Cast": _layers.skip,
    "CheckNumerics": _layers.skip,
    "Dequantize": _layers.skip,
    "ExpandDims": _layers.skip,
    "NoOp": _layers.skip,
    "PlaceholderWithDefault": _layers.skip,
    "QuantizeV2": _layers.skip_one_to_one,
    "RequantizationRange": _layers.skip,
    "Requantize": _layers.skip,
    "Squeeze": _layers.skip,
    "StopGradient": _layers.skip,
    # partially supported
    "Slice": _layers.slice,
    "StridedSlice": _layers.strided_slice,
    # generally unsupported
    "Floor": _layers.skip,  # TODO - need to handle it better
    "RandomStandardNormal": _layers.random,  # TODO - CoreML not supporting random numbers
    "RandomUniform": _layers.random,  # TODO - CoreML not supporting random numbers
    # 'Shape': _layers.shape,
    # 'Gather': _layers.gather,  # TODO- handled in a very limited setting
    # 'GreaterEqual' : _layers.greater, # TODO - need to handle it better
    # 'LogicalAnd' : _layers.mul, # TODO - need to handle it better
    # 'Fill': _layers.fill,
    # 'Greater': _layers.greater, # TODO - only works for x > c where c is const
    # 'FloorMod': _layers.floormod, #TODO-works when this op's output does not depend on network's input values
    # 'Assert': _layers.skip,
    # 'Equal': _layers.skip,
    # 'All': _layers.skip,
    # 'Pack': _layers.skip,  # TODO - need to handle it better
}

_OP_REGISTRY = dict(_CORE_OPS, **_NON_CORE_OPS)


def _get_translator_function(op_type):
    """Get the right translator function
  """
    if op_type in _OP_REGISTRY:
        return _OP_REGISTRY[op_type]
    else:
        raise TypeError(
            "Translation function missing for op of type %s." % type(op_type)
        )


def connect_skipped_ops(context):
    nn_spec = context.builder.nn_spec
    for layer in nn_spec.layers:
        for i, inp_name in enumerate(layer.input):
            if inp_name in context.skip_map_names:
                layer.input[i] = context.skip_map_names[inp_name]

        for i, out_name in enumerate(layer.output):
            if out_name in context.skip_map_names:
                layer.output[i] = context.skip_map_names[out_name]


def check(op, context):
    for inp in op.inputs:
        inp_name = compat.as_str_any(inp.name)
        assert inp_name in context.translated, "No translation found for {}".format(
            inp_name
        )
    for out in op.outputs:
        assert (
            out.name in context.shape_dict
        ), "Shape for {} is not fully defined".format(out.name)


def translation_required(op, context):
    for out in op.outputs:
        out_name = compat.as_str_any(out.name)
        if out_name in context.translated:
            continue
        else:
            return True
    return False


def stop_translation(context):
    """ Check whether all outputs all translated. If yes return True,
  Otherwise return False
  """
    for out in context.output_names:
        if out not in context.translated:
            return False
    return True


def convert_ops_to_layers(context):
    for i, op in enumerate(context.all_ops):
        print(
            "%d/%d: Analysing op name: %s ( type:  %s )"
            % (i + 1, len(context.all_ops), op.name, op.type)
        )

        if stop_translation(context):
            connect_skipped_ops(context)
            return
        else:
            if op.name in context.unused_ops:
                continue
            elif op.name in context.effectively_constant_ops:
                translator = _layers_common.effectively_constant_op
            elif op.name in context.skip_ops:
                translator = _layers.skip
            elif op.type in _OP_REGISTRY:
                check(op, context)
                if context.add_custom_layers and (
                    op.name in context.custom_conversion_functions
                    or op.type in context.custom_conversion_functions
                ):
                    translator = _layers_common.custom_layer
                else:
                    translator = _get_translator_function(op.type)
            elif context.add_custom_layers:
                check(op, context)
                translator = _layers_common.custom_layer
            else:
                raise TypeError(
                    "Translation function missing for op of type %s." % op.type
                )

            if translation_required(op, context):
                translator(op, context)
            connect_skipped_ops(context)
