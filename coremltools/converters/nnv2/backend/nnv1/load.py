# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import logging

from coremltools.models import neural_network as neural_network
import coremltools.models.datatypes as datatypes
from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.builtin_types.symbolic import (
        any_symbolic, any_variadic, is_symbolic)
from .op_mapping import convert_ops
from coremltools.models.neural_network.flexible_shape_utils import set_multiarray_ndshape_range
from .passes.nnv1_passes import nnv1_backend_passes


def load(prog, **kwargs):
    if 'main' not in prog.functions:
        msg = 'main function not found in program {}'
        raise ValueError(msg.format(prog))
    if len(prog.functions) != 1:
        msg = 'SsaProgram must have exactly one `main` function to ' \
            'convert to NNv1. SsaProgram: {}'
        raise ValueError(msg.format(prog))

    nnv1_backend_passes(prog)

    v1_inputs = []
    symbolic_inputs = []
    for name, var in prog.functions['main'].inputs.items():
        if builtins.is_tensor(var.sym_type):
            shape = var.sym_type.get_shape()
            if any_variadic(shape):
                # TODO: rdar://59559656 
                raise NotImplementedError('Variadic rank is not supported')
            if any_symbolic(shape):
                # Use dummy static shape, and will set it later.
                symbolic_inputs.append((name, shape))
                # Pick an example shape (symbolic dimensions must have value
                # between lower_bound and upper_bound in
                # `set_multiarray_ndshape_range`
                shape = [1 if is_symbolic(d) else d for d in shape]
            v1_inputs.append((name, datatypes.Array(*shape)))
        elif builtins.is_scalar(var.sym_type):
            v1_inputs.append((name, datatypes.Array(1)))
        else:
            raise NotImplementedError()

    v1_outputs = []
    for var in prog.functions['main'].outputs:
        if builtins.is_tensor(var.sym_type):
            # Disregard the output types
            v1_outputs.append((var.name, None))
        else:
            raise NotImplementedError()

    builder = neural_network.NeuralNetworkBuilder(
        v1_inputs, v1_outputs,
        disable_rank5_shape_mapping=True)

    # const in V2 are added lazily to V1 by each op whenever needed.
    # `const_context` stores the const names we've added so far and avoid
    # adding a const more than once.
    const_context = set()  # set of str: const name for v1 & v2 (the same)

    # Iterate through ops and add to builder
    convert_ops(const_context, builder, prog.functions['main'].operations)

    model = builder.spec

    # Set symbolic input shapes
    for input_name, shape in symbolic_inputs:
        lb = [1 if is_symbolic(d) else d for d in shape]
        ub = [-1 if is_symbolic(d) else d for d in shape]
        set_multiarray_ndshape_range(model, input_name, lower_bounds=lb,
                upper_bounds=ub)
    return model
