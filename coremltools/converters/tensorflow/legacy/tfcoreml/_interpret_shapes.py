import numpy as np
from copy import copy

_DEBUG_SHAPE_INTERPRETATION = False

_SHAPE_TRANSLATOR_REGISTRY = dict()


def _get_translator_function(op_name):
    """Get the right translator function
  """
    if op_name in _SHAPE_TRANSLATOR_REGISTRY:
        return _SHAPE_TRANSLATOR_REGISTRY[op_name]
    else:
        raise TypeError("Shape Translator missing for OP of type %s." % (op_name))


def _labeled_dims_to_rank_4_shape(blob_name, context):
    context.shape_dict_rank_4[blob_name] = [1, 1, 1, 1]
    labeled_shape = context.dim_labels[blob_name]
    for i, dim in enumerate(labeled_shape):
        if dim == "S":
            context.shape_dict_rank_4[blob_name][0] = context.shape_dict[blob_name][i]
        elif dim == "C":
            context.shape_dict_rank_4[blob_name][3] = context.shape_dict[blob_name][i]
        elif dim == "H":
            context.shape_dict_rank_4[blob_name][1] = context.shape_dict[blob_name][i]
        elif dim == "W":
            context.shape_dict_rank_4[blob_name][2] = context.shape_dict[blob_name][i]
        else:
            assert False, "Incorrect dim label value"


def _expand_dims(op, blob_name, output_name, context):
    axis = context.consts[op.inputs[1].name]

    output_shape = context.shape_dict[output_name]
    output_shape_label = context.dim_labels[output_name]
    input_shape = context.shape_dict[blob_name]

    if len(input_shape) != len(output_shape) - 1:
        return

    input_shape_label = copy(output_shape_label)
    del input_shape_label[axis]

    context.dim_labels[blob_name] = input_shape_label


def _reshape(op, blob_name, output_name, context):

    output_shape = context.shape_dict[output_name]
    output_shape_label = context.dim_labels[output_name]
    input_shape = context.shape_dict[blob_name]

    if len(output_shape) == len(input_shape):
        context.dim_labels[blob_name] = output_shape_label
        return
    elif len(output_shape) == 2 and len(input_shape) == 1:
        input_shape_copy = copy(input_shape)
        input_shape_copy.append(1)
        if np.array_equal(output_shape, input_shape_copy):
            context.dim_labels[blob_name] = [output_shape_label[0]]
    elif len(output_shape) == 4 and len(input_shape) == 1:
        dim = input_shape[0]
        if dim in output_shape:
            idx = output_shape.index(dim)
            context.dim_labels[blob_name] = "SHWC"[idx]
    elif len(output_shape) == 3 and len(input_shape) == 1:
        dim = input_shape[0]
        if dim in output_shape:
            idx = output_shape.index(dim)
            context.dim_labels[blob_name] = output_shape_label[idx]
    else:
        return


def _broadcast_op(op, blob_name, output_name, context):

    if len(context.shape_dict[output_name]) == len(context.shape_dict[blob_name]):
        context.dim_labels[blob_name] = context.dim_labels[output_name]
        return

    output_shape = context.shape_dict_rank_4[output_name]
    rank_4_label = ["S", "H", "W", "C"]
    input_shape = context.shape_dict[blob_name]
    dim_labels = ["" for i in input_shape]

    if not (len(input_shape) < 4 and len(input_shape) > 0 and len(output_shape) == 4):
        return

    # Handle the case when input shape is [C]
    # and output shape is [S,H,W,C]
    if len(input_shape) == 1 and input_shape[0] == output_shape[3]:
        context.dim_labels[blob_name] = ["C"]
        return

    index = 0
    for i, value in enumerate(input_shape):
        for o in range(index, 4):
            if value == output_shape[o]:
                dim_labels[i] = rank_4_label[o]
                index = o + 1
                break
    context.dim_labels[blob_name] = dim_labels


"""
Identity: Applicable to all layers that do not change the 
rank of the input tensor and that do not rearrange the axis.
"""


def _identity(op, blob_name, output_name, context):
    if len(context.shape_dict[output_name]) == len(context.shape_dict[blob_name]):
        context.dim_labels[blob_name] = context.dim_labels[output_name]
    else:
        return


# Make the interpret_shape function return False, making shape interpretation
# fall back to static mapping
def _terminate(op, blob_name, output_name, context):
    return


_SHAPE_TRANSLATOR_REGISTRY = {
    "ExpandDims": _expand_dims,
    "Reshape": _reshape,
    "Mul": _broadcast_op,
    "Add": _broadcast_op,
    "Sum": _identity,
    "Mean": _identity,
    "Rsqrt": _identity,
    "Sub": _broadcast_op,
    "BiasAdd": _broadcast_op,
    "QuantizedBiasAdd": _broadcast_op,
    "RealDiv": _broadcast_op,
    "RandomUniform": _terminate,
    "BatchToSpaceND": _identity,
    "SpaceToBatchND": _identity,
    "Dequantize": _identity,
    "QuantizedReshape": _reshape,
    "QuantizeV2": _identity,
    "ResizeNearestNeighbor": _identity,
    "Log": _identity,
    "Neg": _identity,
    "ResizeBilinear": _identity,
    "Pad": _identity,
    "NoOp": _identity,
    "Cast": _identity,
    "Squeeze": _identity,
    "StopGradient": _identity,
    "CheckNumerics": _identity,
    "Floor": _identity,
    "Assert": _identity,
    "Equal": _identity,
    "All": _identity,
    "Pack": _identity,
    "RequantizationRange": _identity,
    "Requantize": _identity,
    "PlaceholderWithDefault": _identity,
    "ConcatV2": _identity,
    "GreaterEqual": _identity,
    "LogicalAnd": _broadcast_op,
    "Fill": _terminate,
    "Maximum": _broadcast_op,
    "Sigmoid": _identity,
    "Square": _identity,
    "SquaredDifference": _broadcast_op,
    "MirrorPad": _identity,
    "Greater": _identity,
    "Const": _terminate,
    "Softmax": _identity,
    "Relu6": _identity,
    "Relu": _identity,
    "LeakyRelu": _identity,
    "QuantizedRelu": _identity,
    "DepthwiseConv2dNative": _identity,
    "MaxPool": _identity,
    "AvgPool": _identity,
    "Conv2DBackpropInput": _identity,
    "Conv2D": _identity,
    "QuantizedConv2D": _identity,
    "Concat": _identity,
    "BatchNormWithGlobalNormalization": _identity,
    "Identity": _identity,
    "Placeholder": _terminate,
    "Elu": _identity,
    "Reciprocal": _identity,
    "FusedBatchNorm": _identity,
    "LRN": _identity,
    "Tanh": _identity,
    "Minimum": _broadcast_op,
    "Exp": _identity,
    "Split": _identity,
    "Sqrt": _identity,
    "Pow": _identity,
}

# TODO. Need to figure out the correct rule for adding these ops:
# Slice, StridedSlice, ExtractImagePatches, ArgMax, Shape,
# Transpose, Prod, Max, Min, MatMul, OneHot, Gather, FloorMod,


def _interpret_and_label_shapes(blob_name, context, tracking_dict):
    """Fills in dictionaries "shape_dict_rank_4" and "dim_labels"
  shape_dict_rank_4: Tensor name to rank 4 shape (Batch/Sequence, H, W, C)
  dim_labels: Tensor name to labeled shapes (one of 'S','C','H','W').
  e.g.: 'input' tensor which has shape (1,224,224,3) --> ('S','H','W','C')
  """

    if _DEBUG_SHAPE_INTERPRETATION:
        print("Shape interpretation called in for {}".format(blob_name))

    shape = context.shape_dict[blob_name]
    if blob_name in context.dim_labels:
        return True
    elif len(shape) == 4:
        context.dim_labels[blob_name] = ["S", "H", "W", "C"]
        _labeled_dims_to_rank_4_shape(blob_name, context)
        return True
    elif len(shape) < 1 or len(shape) > 4:
        return False
    else:
        ops_list = context.blob_graph[blob_name]
        if len(ops_list) == 0:
            return False
        else:
            for ii, op in enumerate(ops_list):

                output_name = op.outputs[0].name

                # Recursion
                if output_name not in tracking_dict:
                    tracking_dict[output_name] = 1
                else:
                    return False

                if _DEBUG_SHAPE_INTERPRETATION:
                    print("Calling interpret shape for tensor: {}".format(output_name))

                status = _interpret_and_label_shapes(
                    output_name, context, tracking_dict
                )

                if not status:
                    continue
                else:
                    fun = _get_translator_function(op.type)
                    # The shape of "output_name" of "op" has been interpreted. Now we are
                    # asking to interpret the shape of the input to this op: "blob_name"
                    if _DEBUG_SHAPE_INTERPRETATION:
                        print(
                            "\nInterpreted shape of '{}' is {} , {}".format(
                                output_name,
                                str(context.shape_dict[output_name]),
                                str(context.dim_labels[output_name]),
                            )
                        )
                        print(
                            "Now interpreting shape of tensor: '{}' with raw shape: {}".format(
                                blob_name, str(context.shape_dict[blob_name])
                            )
                        )
                        print(
                            "by calling an op named: '{}', of type: '{}'".format(
                                op.name, op.type
                            )
                        )

                    fun(op, blob_name, output_name, context)
                    if blob_name in context.dim_labels:
                        if _DEBUG_SHAPE_INTERPRETATION:
                            print(
                                "interpreted shape of tensor '{}' is {}\n".format(
                                    blob_name, str(context.dim_labels[blob_name])
                                )
                            )
                        assert len(context.dim_labels[blob_name]) == len(shape), (
                            "labeled dimensions length not equal to the length its shape for Tensor %s"
                            % blob_name
                        )
                        _labeled_dims_to_rank_4_shape(blob_name, context)
                        return True
                    else:
                        continue

            return False


def _interpret_shape(blob_name, context):
    return _interpret_and_label_shapes(blob_name, context, tracking_dict={})
