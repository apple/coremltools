# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from ..basic_graph_ops import delete_node, replace_source

from coremltools.converters.mil.mil.passes.defs.optimize_conv import fuse_dilated_conv

_try_same = fuse_dilated_conv._uses_same_padding

def _pattern_match_and_rewrite(gddict, conv_op):
    node = gddict[conv_op]
    channel_first = node.attr["data_format"].startswith("NC")

    if len(node.inputs) == 0 or len(node.outputs) == 0:
        return

    prev_node = gddict[node.inputs[0]]
    next_node = gddict[node.outputs[0]]

    expand_node = None
    squeeze_node = None
    # Check for Conv1D cases
    if prev_node.op == "ExpandDims":
        # All Conv1D has ExpandDims and Squeeze as pairs.
        if next_node.op != "Squeeze":
            return

        expand_node = prev_node
        squeeze_node = next_node

        if len(prev_node.inputs) == 0 or len(next_node.outputs) == 0:
            return
        prev_node = gddict[prev_node.inputs[0]]
        next_node = gddict[next_node.outputs[0]]

    # Check if Conv1D/Conv2D is surrounded by SpaceToBatchND and BatchToSpaceND
    if prev_node.op != "SpaceToBatchND" or next_node.op != "BatchToSpaceND":
        return
    else:
        stb_node = prev_node
        bts_node = next_node

    dilation_node = gddict[stb_node.inputs[1]]
    if dilation_node.value is None:
        return
    dilation_factor = dilation_node.value.val
    if gddict[bts_node.inputs[1]].value is None or np.any(
        dilation_factor != gddict[bts_node.inputs[1]].value.val
    ):
        # If SpaceToBatchND and BatchToSpaceND doesn't match, we do not fuse.
        return

    padding_node = gddict[stb_node.inputs[2]]
    if padding_node.value is None:
        return
    padding_val = padding_node.value.val.flatten()

    crop_node = gddict[bts_node.inputs[2]]
    if crop_node.value is None:
        return
    crop_val = crop_node.value.val.flatten()

    if expand_node:
        dilation_factor = [1] + list(dilation_factor)
        padding_val = [0, 0] + list(padding_val)
        crop_val = [0, 0] + list(crop_val)
    # Trying to inverse the logic of TF generating padding/cropping values for
    # SpaceToBatchND and BatchToSpaceND with different padding values in Conv2D.
    # Logic extracted from TF's builder at:
    # tensorflow/python/ops/nn_ops.py and tensorflow/python/ops/array_ops.py
    is_same = False
    if np.any(padding_val != 0):
        input_shape = gddict[stb_node.inputs[0]].attr.get("_output_shapes", None)
        if input_shape is None:
            input_shape = gddict[stb_node.inputs[0]].attr.get("shape", None)
        else:
            input_shape = input_shape[0]
        W_node = gddict[node.inputs[1]]
        W_shape = None if W_node.op != "Const" else W_node.datatype.get_shape()
        if input_shape is None or W_shape is None:
            return
        W_h, W_w = W_shape[0], W_shape[1]
        HW = input_shape[2:] if channel_first else input_shape[1:-1]
        if expand_node:
            HW = [1] + list(HW)
        is_same = _try_same(
            HW[0], HW[1], W_h, W_w, dilation_factor, padding_val, crop_val
        )

    # Re-wiring the nodes to skip SpaceToBatchND.
    # We change BatchToSpaceND to Identity since it might be a terminate op.
    deleted_nodes = set()
    if expand_node:
        replace_source(gddict, stb_node, expand_node, stb_node.inputs[0])
    else:
        replace_source(gddict, stb_node, node, stb_node.inputs[0])

    bts_node.op = "Identity"
    bts_node.attr = {}

    deleted_nodes.update(stb_node.inputs[1:])
    deleted_nodes.update([stb_node.name])
    deleted_nodes.update(bts_node.inputs[1:])

    # Rewrite dilation attribute for (Depthwise)Conv2D
    dilation_val = (
        [1, 1] + list(dilation_factor)
        if node.attr["data_format"] == "NCHW"
        else [1] + list(dilation_factor) + [1]
    )
    node.attr["dilations"] = dilation_val
    # Rewrite padding attribute for (Depthwise)Conv2D
    # This is due to, TF always plug in VALID padding for Conv2D after
    # SpaceToBatchND. If, the original Conv2D is SAME padding, TF would
    # automatically insert padding, therefore, we set it as SAME over here.
    if is_same:
        node.attr["padding"] = "SAME"

    # Removing stale attributes for nodes.
    if expand_node and "_output_shapes" in expand_node.attr:
        del expand_node.attr["_output_shapes"]
    if squeeze_node and "_output_shapes" in squeeze_node.attr:
        del squeeze_node.attr["_output_shapes"]
    if "_output_shapes" in node.attr:
        del node.attr["_output_shapes"]
    if expand_node and "shape" in expand_node.attr:
        del expand_node.attr["shape"]
    if squeeze_node and "shape" in squeeze_node.attr:
        del squeeze_node.attr["shape"]
    if "shape" in node.attr:
        del node.attr["shape"]

    for d in deleted_nodes:
        delete_node(gddict, d)


def _fuse_dilation_conv(gddict):
    """
    A dilated convolution in older tensorflow versions might not be fused in the
    Conv2D or DepthwiseConv2D op, but represented with the following format:

        SpaceToBatchND -> (Depthwise)Conv2D -> BatchToSpaceND

    We try to fuse it back into (Depthwise)Conv2D with the dilation parameter
    set in attribute.
    There are several patterns that exist in tensorflow for breaking up dilation
    convolutions. We detect the following patterns:

      SpaceToBatchND -> ExpandDims -> Conv2D -> Squeeze -> BatchToSpaceND

      SpaceToBatchND -> Conv2D -> BatchToSpaceND

    The first case appears when Conv1D is used, TF expands/squeeze the inputs to
    conform Conv2D pattern.
    The second case is a basic Conv2D pattern.

    """
    for name in list(gddict.keys()):
        if name not in gddict:
            # Node might have been removed from graph during fusion.
            continue
        node = gddict[name]
        if node.op in {"Conv2D", "DepthwiseConv2dNative"}:
            _pattern_match_and_rewrite(gddict, name)


def fuse_dilation_conv(tfssa):
    """
    Tensorflow decomposes Depthwise Convolution with dialtion into:

    SpaceToBatchND ---> Conv2D/DepthwiseConv2D ---> BatchToSpaceND

    We identify such pattern and use Conv2D/DepthwiseConv2D to represent it.
    """
    for f in tfssa.functions.keys():
        _fuse_dilation_conv(tfssa.functions[f].graph)
