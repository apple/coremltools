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

from six import string_types as _string_types

def get_image_params(kwargs):
    image_input_names = kwargs.get('image_input_names', None)
    if isinstance(image_input_names, _string_types):
        image_input_names = [image_input_names]
    is_bgr = kwargs.get('is_bgr', False)
    red_bias = kwargs.get('red_bias', 0.)
    green_bias = kwargs.get('green_bias', 0.)
    blue_bias = kwargs.get('blue_bias', 0.)
    gray_bias = kwargs.get('gray_bias', 0.)
    image_scale = kwargs.get('image_scale', 1.)
    image_format = kwargs.get('image_format', 'NHWC')

    return dict(zip(['image_input_names','is_bgr','red_bias','green_bias','blue_bias',
                     'gray_bias','image_scale','image_format'],
                    [image_input_names, is_bgr, red_bias, green_bias, blue_bias,
                     gray_bias, image_scale, image_format]))



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
        if builtins.is_tensor(var.sym_type) or builtins.is_primitive(var.sym_type):
            # Disregard the output types
            v1_outputs.append((var.name, None))
        else:
            raise NotImplementedError()

    # classifier flag
    class_labels = kwargs.get('class_labels', None)
    is_classifier = class_labels is not None
    neural_network_type = 'classifier' if is_classifier else None

    # create neural network builder
    builder = neural_network.NeuralNetworkBuilder(
        v1_inputs, v1_outputs,
        mode=neural_network_type,
        disable_rank5_shape_mapping=True)

    # const in V2 are added lazily to V1 by each op whenever needed.
    # `const_context` stores the const names we've added so far and avoid
    # adding a const more than once.
    const_context = set()  # set of str: const name for v1 & v2 (the same)

    # Iterate through ops and add to builder
    convert_ops(const_context, builder, prog.functions['main'].operations,
            prog.functions['main'].outputs)

    # Add image input
    image_params = get_image_params(kwargs)
    builder.set_pre_processing_parameters(image_input_names=image_params['image_input_names'],
                                          is_bgr=image_params['is_bgr'],
                                          red_bias=image_params['red_bias'],
                                          green_bias=image_params['green_bias'],
                                          blue_bias=image_params['blue_bias'],
                                          gray_bias=image_params['gray_bias'],
                                          image_scale=image_params['image_scale'],
                                          image_format=image_params['image_format'])

    # Replace model outputs's name with v1_outputs
    output_names = [x[0] for x in v1_outputs]
    for i, spec_layer in enumerate(builder.nn_spec.layers):
        for j, name in enumerate(spec_layer.output):
            for output_name in output_names:
                if output_name.split(':')[0] == name:
                    spec_layer.output[j] = output_name

    # Add classifier classes
    predicted_feature_name = kwargs.get('predicted_feature_name', None)
    predicted_probabilities_output = kwargs.get('predicted_probabilities_output','')
    message = 'Class labels must be a list of integers / strings or a file path'

    if is_classifier:
        classes_in = class_labels
        if isinstance(classes_in, str):
            import os
            if not os.path.isfile(classes_in):
                raise ValueError("Path to class labels (%s) does not exist." % \
                                 classes_in)
                with open(classes_in, 'r') as f:
                    classes = f.read()
                classes = classes.splitlines()
        elif isinstance(classes_in, list):  # list[int or str]
            classes = classes_in
            assert(all([isinstance(x, (int, str)) for x in classes])), message
        else:
            raise ValueError(message)

        if predicted_feature_name is not None:
            builder.set_class_labels(
                classes, predicted_feature_name=predicted_feature_name,
                prediction_blob=predicted_probabilities_output)
        else:
            builder.set_class_labels(classes)

    proto = builder.spec
    # Set symbolic input shapes
    for input_name, shape in symbolic_inputs:
        lb = [1 if is_symbolic(d) else d for d in shape]
        ub = [-1 if is_symbolic(d) else d for d in shape]
        set_multiarray_ndshape_range(
            proto, input_name, lower_bounds=lb, upper_bounds=ub)

    return proto
