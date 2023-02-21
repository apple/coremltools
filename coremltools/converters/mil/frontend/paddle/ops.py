#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import builtins
import numbers

import numpy as _np
import paddle

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.defs._utils import (
    promote_input_dtypes,
    solve_slice_by_index_shape)
from coremltools.converters.mil.mil.types.symbolic import (
    any_symbolic,
    is_symbolic,
)
from coremltools.converters.mil.mil.var import Var

from .._utils import build_einsum_mil
from .paddle_op_registry import _PADDLE_OPS_REGISTRY, register_paddle_op


# This is a magic number in PaddlePaddle. It's used as a default value in many
# functions.
PYPADDLE_MAGIC_DEFAULT = 9223372036854775807
VALUE_CLOSE_TO_INFINITY = 1e+38

def convert_nodes(context, paddle_program):
    """
    Iterate over the nodes of a graph or block and convert to MIL.

    Arguments:
        context: A TranscriptionContext object to pull node inputs and
            assign node outputs.
        graph: An InternalPaddleIRGraph or InternalPaddleIRBlock object.
    """

    for block in paddle_program.blocks:
        for i, op in enumerate(block.ops):
            if op.type in ['feed', 'fetch']:
                continue
            else:
                op_lookup = op.type
                add_op = _PADDLE_OPS_REGISTRY.get(op_lookup, None)

                if add_op is None:
                    raise RuntimeError(
                        "PaddlePaddle convert function for op '{}' not implemented.".format(node.kind)
                    )
                add_op(context, op)

# Some ops will receive a dtype input as an integer
# which maps to a paddle dtype. The below mapping was found by
# converting test models with different dtypes passed to ones.
NUM_TO_PADDLE_DTYPE = {
    0: paddle.uint8,
    1: paddle.int8,
    2: paddle.int16,
    3: paddle.int32,
    4: paddle.int32,
    5: paddle.float16,
    6: paddle.float32,
    7: paddle.float32,
    11: paddle.bool,
}


def _construct_constant(val, name):
    # Converter cannot handle paddle tensors.
    if isinstance(val, paddle.Tensor):
        val = val.cpu().numpy()

    # MIL casts ints to int32, which can't represent the 64 bit magic number.
    # So we instead represent it with None, and any ops that might get the
    # value will check for None instead.
    if isinstance(val, int) and val == PYPADDLE_MAGIC_DEFAULT:
        val = None

    # PaddlePaddle uses inf
    if val is not None and isinstance(val, numbers.Number) and _np.isinf(val):
        if val < 0:  # neg inf
            # most negative number in fp32
            val = -3.4e+38
        else:  # positive inf
            val = 3.4e+38
    if val is None:
        return None
    else:
        return mb.const(val=val, name=name)

@register_paddle_op(paddle_alias=["conv2d"])
def _convolution(context, node):
    dilations = node.desc.attr('dilations')
    groups = node.desc.attr('groups')
    strides = node.desc.attr('strides')
    paddings = node.desc.attr('paddings')
    paddings += paddings

    x = context[node.input("Input")[0]]
    weight = context[node.input("Filter")[0]]
    if node.input("Bias"):
        bias = context[node.input("Bias")[0]]
    else:
        bias = None
    
    output_name = node.output('Output')[0]

    kwargs = {
        "x": x,
        "weight": weight,
        "strides": strides,
        "pad_type": "custom",
        "pad": paddings,
        "dilations": dilations,
        "groups": groups,
        "name": output_name,
    }
    # # Bias is optional in PaddlePaddle's convolution.
    if bias is not None:
        kwargs["bias"] = bias

    conv = mb.conv(**kwargs)
    context.add(conv)
