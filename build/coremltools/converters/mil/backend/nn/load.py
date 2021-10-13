# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from collections import defaultdict
import coremltools as ct
from coremltools.converters.mil.input_types import (
    ClassifierConfig,
    ImageType,
    EnumeratedShapes,
    Shape,
    RangeDim,
)
from coremltools.converters.mil.backend.backend_helper import _get_probability_var_for_classifier
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.types.symbolic import (
    any_symbolic,
    any_variadic,
    is_symbolic,
)
from coremltools.converters._profile_utils import _profile
from coremltools.models import MLModel
from coremltools.models import neural_network as neural_network
from coremltools.models.datatypes import Array
from coremltools.models.neural_network import flexible_shape_utils
from coremltools.models.neural_network.flexible_shape_utils import (
    update_image_size_range,
    add_enumerated_image_sizes,
    set_multiarray_ndshape_range,
    add_multiarray_ndshape_enumeration,
)
import logging
from .op_mapping import convert_ops
from .passes.nn_passes import nn_backend_passes


def _convert_to_image_input(proto, inputs, skip_model_load=False):
    tmp_model = MLModel(proto, skip_model_load=skip_model_load)
    for input_type in inputs:
        if isinstance(input_type, ImageType):
            if input_type.color_layout == "G":
                gray_bias = input_type.bias
                red_bias, green_bias, blue_bias = 0.0, 0.0, 0.0
            elif input_type.color_layout == "RGB":
                gray_bias = 0.0
                red_bias, green_bias, blue_bias = input_type.bias
            elif input_type.color_layout == "BGR":
                gray_bias = 0.0
                blue_bias, green_bias, red_bias = input_type.bias
            tmp_model = neural_network.utils.make_image_input(
                tmp_model,
                input_type.name,
                is_bgr=input_type.color_layout == "BGR",
                image_format="NCHW" if input_type.channel_first else "NHWC",
                red_bias=red_bias,
                green_bias=green_bias,
                blue_bias=blue_bias,
                gray_bias=gray_bias,
                scale=input_type.scale,
            )
    return tmp_model.get_spec()


def _convert_to_classifier(proto, classifier_config, skip_model_load=False):
    tmp_model = MLModel(proto, skip_model_load=skip_model_load)
    tmp_model = neural_network.utils.make_nn_classifier(
        tmp_model,
        classifier_config.class_labels,
        classifier_config.predicted_feature_name,
        classifier_config.predicted_probabilities_output,
    )
    return tmp_model.get_spec()


def _set_user_inputs(proto, inputs):
    for input_type in inputs:
        shape = input_type.shape
        if isinstance(shape, EnumeratedShapes):
            if isinstance(input_type, ImageType):
                default_height , default_width = 0, 0
                for inp in proto.description.input:
                    if inp.name == input_type.name:
                        default_height = inp.type.imageType.height
                        default_width = inp.type.imageType.width
                        break
                image_sizes = []
                if input_type.channel_first:
                    for s in shape.shapes:
                        if s.shape[-2] == default_height and s.shape[-1] == default_width:
                            continue
                        image_sizes.append(
                            flexible_shape_utils.NeuralNetworkImageSize(
                                height=s.shape[-2], width=s.shape[-1]
                            )
                        )
                else:
                    for s in shape.shapes:
                        if s.shape[-3] == default_height and s.shape[-2] == default_width:
                            continue
                        image_sizes.append(
                            flexible_shape_utils.NeuralNetworkImageSize(
                                height=s.shape[-3], width=s.shape[-2]
                            )
                        )
                add_enumerated_image_sizes(
                    proto, input_type.name, sizes=image_sizes
                )
            else:
                add_multiarray_ndshape_enumeration(
                    proto, input_type.name, [tuple(s.shape) for s in shape.shapes]
                )
        elif isinstance(shape, Shape):
            shape = shape.shape  # This is shape in Shape
            if all(
                [
                    not isinstance(s, RangeDim) and not is_symbolic(s) and s > 0
                    for s in shape
                ]
            ):
                continue
            if isinstance(input_type, ImageType):
                img_range = flexible_shape_utils.NeuralNetworkImageSizeRange()
                if input_type.channel_first:
                    H = shape[-2]
                    W = shape[-1]
                else:
                    H = shape[-3]
                    W = shape[-2]

                if isinstance(H, RangeDim):
                    img_range.add_height_range((H.lower_bound, H.upper_bound))
                elif is_symbolic(H):
                    img_range.add_height_range((1, -1))
                else:
                    img_range.add_height_range((H, H))
                if isinstance(W, RangeDim):
                    img_range.add_width_range((W.lower_bound, W.upper_bound))
                elif is_symbolic(W):
                    img_range.add_width_range((1, -1))
                else:
                    img_range.add_width_range((W, W))

                flexible_shape_utils.update_image_size_range(
                    proto, input_type.name, img_range
                )
            else:
                lb = []
                ub = []
                for s in shape:
                    if isinstance(s, RangeDim):
                        lb.append(s.lower_bound)
                        ub.append(s.upper_bound)
                    elif is_symbolic(s):
                        lb.append(1)
                        ub.append(-1)
                    else:
                        lb.append(s)
                        ub.append(s)
                set_multiarray_ndshape_range(
                    proto, input_type.name, lower_bounds=lb, upper_bounds=ub
                )


def _set_symbolic_inputs(proto, symbolic_inputs):
    # Set symbolic input shapes by -1 infered from graph
    for input_name, shape in symbolic_inputs.items():
        lb = [1 if is_symbolic(d) else d for d in shape]
        ub = [-1 if is_symbolic(d) else d for d in shape]
        set_multiarray_ndshape_range(
            proto, input_name, lower_bounds=lb, upper_bounds=ub
        )

def _set_optional_inputs(proto, input_types):
    # Set default values for optional input_types
    default_map = {}
    for input_type in input_types:
        if isinstance(input_type, ImageType):
            continue
        if input_type.default_value is not None:
            default_map[input_type.name] = input_type.default_value

    for idx, input in enumerate(proto.description.input):
        name = proto.description.input[idx].name
        if name in default_map:
            default_value = default_map[name]
            proto.description.input[idx].type.isOptional = True
            array_t = proto.description.input[idx].type.multiArrayType
            default_fill_val = default_value.flatten()[0]
            array_t.floatDefaultValue = default_fill_val
            if default_fill_val != 0 or list(default_value.shape) != \
                array_t.shape:
                # promote spec version to 5 and set the default value
                proto.specificationVersion = max(proto.specificationVersion,
                    ct._SPECIFICATION_VERSION_IOS_14)
                # array_t.shape is not empty.
                array_t.ClearField('shape')
                array_t.shape.extend(list(default_value.shape))


@_profile
def load(prog, **kwargs):
    if "main" not in prog.functions:
        msg = "main function not found in program {}"
        raise ValueError(msg.format(prog))
    if len(prog.functions) != 1:
        msg = (
            "Program must have exactly one `main` function to "
            "convert to NN. Program: {}"
        )
        raise ValueError(msg.format(prog))

    nn_backend_passes(prog)
    input_types = prog.main_input_types

    v1_inputs = []
    symbolic_inputs = {}
    for name, var in prog.functions["main"].inputs.items():
        if types.is_tensor(var.sym_type):
            sym_shape = var.sym_type.get_shape()
            if any_variadic(sym_shape):
                raise NotImplementedError("Variadic rank is not supported")
            if any_symbolic(sym_shape):
                user_specified = False
                for input_type in input_types:
                    if name == input_type.name:
                        sym_shape = input_type.shape.default
                        user_specified = True
                        break
                # Use dummy static shape, and will set it later.
                shape = [1 if is_symbolic(d) else d for d in sym_shape]
                if not user_specified:
                    symbolic_inputs[name] = sym_shape
            else:
                shape = sym_shape
            v1_inputs.append((name, Array(*shape)))
        elif types.is_scalar(var.sym_type):
            v1_inputs.append((name, Array(1)))
        else:
            raise NotImplementedError()

    v1_outputs = []
    for var in prog.functions["main"].outputs:
        if types.is_tensor(var.sym_type) or types.is_primitive(var.sym_type):
            # Disregard the output types
            v1_outputs.append((var.name, None))
        else:
            raise NotImplementedError()

    # create neural network builder
    builder = neural_network.NeuralNetworkBuilder(
        v1_inputs,
        v1_outputs,
        disable_rank5_shape_mapping=True,
        use_float_arraytype=True,
    )

    # const in V2 are added lazily to V1 by each op whenever needed.
    # `const_context` stores the const names we've added so far and avoid
    # adding a const more than once.
    # const_context: list[set of str] (const name for v1 & v2
    # (the same)). Note that in NN in outer layer is visible from the inner
    # layer, so the const_context is simply a stack of set.
    const_context = []
    # Iterate through ops and add to builder
    convert_ops(
        const_context,
        builder,
        prog.functions["main"].operations,
        prog.functions["main"].outputs,
    )

    proto = builder.spec
    # image input
    has_image_input = any([isinstance(s, ImageType) for s in input_types])
    if has_image_input:
        proto = _convert_to_image_input(proto, input_types,
                                        skip_model_load=kwargs.get("skip_model_load", False))

    # classifier flag
    classifier_config = kwargs.get("classifier_config", None)
    if classifier_config is not None:
        # verify that classifier_config.predicted_probabilities_output if its exists.
        # And if its empty/None, fill it with the last non const op's output
        # this is done in "_get_probability_var_for_classifier()"
        probability_var = _get_probability_var_for_classifier(prog, classifier_config)
        if classifier_config.predicted_probabilities_output != probability_var.name:
            classifier_config.predicted_probabilities_output = probability_var.name
        # add classifier related fields to the proto spec
        proto = _convert_to_classifier(proto, classifier_config,
                                       skip_model_load=kwargs.get("skip_model_load", False))

    _set_user_inputs(proto, input_types)
    _set_symbolic_inputs(proto, symbolic_inputs)
    _set_optional_inputs(proto, input_types)

    return proto
