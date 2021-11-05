#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import logging
import numpy as np
import os

from .passes import mil_passes
from coremltools import _SPECIFICATION_VERSION_IOS_15
from coremltools.converters.mil.backend.mil.helper import (
    cast_to_framework_io_dtype,
    create_file_value,
    create_immediate_value,
    create_list_scalarvalue,
    create_scalar_value,
    types_to_proto
)
from coremltools.converters.mil.backend.backend_helper import _get_probability_var_for_classifier
from coremltools.converters.mil.mil import (
    Builder as mb,
    Function,
    mil_list,
    types
)
from coremltools.converters.mil.backend.nn.load import _set_optional_inputs
from coremltools.converters.mil.input_types import ImageType, TensorType, EnumeratedShapes, RangeDim
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry
from coremltools.converters.mil.mil.types.symbolic import (
    any_symbolic,
    any_variadic,
    is_symbolic,
)
from coremltools.converters.mil.mil.types.type_mapping import types_int64
from coremltools.libmilstoragepython import _BlobStorageWriter as BlobWriter
from coremltools.models.model import _WEIGHTS_FILE_NAME
from coremltools.models.neural_network.flexible_shape_utils import (
    add_enumerated_image_sizes,
    add_multiarray_ndshape_enumeration,
    NeuralNetworkImageSize,
    NeuralNetworkImageSizeRange,
    set_multiarray_ndshape_range,
    update_image_size_range
)
from coremltools.proto import (
    FeatureTypes_pb2 as ft,
    MIL_pb2 as pm,
    Model_pb2 as ml
)


def should_use_weight_file(val):
    return (
        val is not None
        and isinstance(val, (np.ndarray, np.generic))
        and val.size >= 10
        and val.dtype in ['float16', 'float32']
    )

def translate_const(op, blob_writer):
    output_var = op.outputs[0]

    if should_use_weight_file(output_var.val):
        value = create_file_value(output_var, blob_writer)
    else:
        value = create_immediate_value(output_var)

    return pm.Operation(
        type="const",
        attributes={"name": create_scalar_value(op.name), "val": value},
        outputs=[
            pm.NamedValueType(
                name=output_var.name, type=types_to_proto(output_var.sym_type)
            )
        ],
    )


def translate_generic_op(op, parameters, blob_writer, literal_params=[]):
    inputs = {}
    for param_name, vars in op.inputs.items():
        if param_name.startswith("_"):
            continue
        if not isinstance(vars, (list, tuple)):
            vars = [vars]

        arguments = []
        for _var in vars:
            binding = pm.Argument.Binding()
            # use const value literals if requested
            if param_name in literal_params:
                binding.value.CopyFrom(create_immediate_value(_var))
            else:
                binding.name = _var.name
            arguments.append(binding)

        args = pm.Argument()
        args.arguments.extend(arguments)
        inputs[param_name] = args

    outputs = [
        pm.NamedValueType(name=v.name, type=types_to_proto(v.sym_type))
        for v in op.outputs
    ]
    blocks = None
    if len(op.blocks) > 0:
        blocks = [create_block(b, parameters, blob_writer) \
                  for b in op.blocks]

    op_type = op.op_type
    attr_dict = {}
    if op.op_type in SSAOpRegistry.custom_ops:
        op_type = "custom_layer"
        class_name = op.bindings.get("class_name", op.name)
        input_order = op.bindings.get("input_order", [])
        parameters = op.bindings.get("parameters", [])
        weights = op.bindings.get("weights", [])
        description = op.bindings.get("description", "")

        attr_dict["name"] = create_scalar_value(op.name)
        attr_dict["class_name"] = create_scalar_value(class_name)
        attr_dict["input_order"] = create_list_scalarvalue(input_order, np.str)
        attr_dict["parameters"] = create_list_scalarvalue(parameters, np.str)
        attr_dict["weights"] = create_list_scalarvalue(weights, np.str)
        attr_dict["description"] = create_scalar_value(description)

    return pm.Operation(
        type=op_type,
        blocks=blocks,
        inputs=inputs,
        attributes=attr_dict,
        outputs=outputs,
    )


def create_block(block, parameters, blob_writer):
    proto_ops = []

    # Find the const op that generates classify's "label" / "class" string vec.
    classify_const_classes_op = None
    if len(block.operations) > 0:
        # Classify is always the last operation in the block.
        op = block.operations[-1]
        op_cls_name = type(op).__name__
        if (op_cls_name == "classify"):
            classes_var = op.inputs["classes"]
            classify_const_classes_op = classes_var.op
            if (len(classes_var.child_ops) != 1):
                raise ValueError("Classify's labels/classes should be input to only 1 op (classify).")

    for op in block.operations:
        op_cls_name = type(op).__name__
        if op_cls_name == "const":
            # Do not serialize the const op that creates the var bound to the classifier's "classes" param.
            # The variable's value will be bound directly to classify's "classes" param instead.
            if op != classify_const_classes_op:
                proto_ops.append(translate_const(op, blob_writer))
        elif op_cls_name == "classify":
            # Classify's "classes" param should be serialized as a value literal bound
            # directly to the param, rather than as a const-generated variable.
            proto_ops.append(translate_generic_op(op, parameters, blob_writer, ["classes"]))
        else:
            proto_ops.append(translate_generic_op(op, parameters, blob_writer))

    inputs = []
    if not isinstance(block, Function):
        # Function is subclass of Block, but function's block has no input,
        # and hence skipping reading the block inputs.
        for var in block.inputs:
            proto_type = types_to_proto(var.sym_type)
            inputs.append(pm.NamedValueType(name=var.name, type=proto_type))
    output_names = [v.name for v in block.outputs]
    return pm.Block(inputs=inputs, outputs=output_names, operations=proto_ops)


def convert_function(function, parameters, blob_writer):
    block = create_block(function, parameters, blob_writer)

    inputs = []
    for name, var in function.inputs.items():
        proto_type = types_to_proto(var.sym_type)
        inputs.append(pm.NamedValueType(name=name, type=proto_type))

    return pm.Function(inputs=inputs, opset="CoreML5", block_specializations={"CoreML5": block})


# Add a classify op to the output.
# Replaces the original probabilites output (in the containing MIL block)
# with the outputs of the classifier op. Returns the name of the original
# probabilities output variable.
def _add_classify_op(prog, classifier_config):
    '''
    Add a "classify" op to the program, at the end of the main block
    '''
    block = prog.functions["main"]

    message = "Class labels must be a list of integers / strings or a file path"
    classes_in = classifier_config.class_labels
    if isinstance(classes_in, str):
        import os

        if not os.path.isfile(classes_in):
            raise ValueError("Path to class labels (%s) does not exist." % classes_in)
        with open(classes_in, "r") as f:
            classes = f.read()
        classes = classes.splitlines()
    elif isinstance(classes_in, list):  # list[int or str]
        classes = classes_in
        assert all([isinstance(x, (int, str)) for x in classes]), message
    else:
        raise ValueError(message)

    probability_var = _get_probability_var_for_classifier(prog, classifier_config)

    # add the classify op now
    with block:
        # cast the int label to np.int64
        if isinstance(classes[0], int):
            classes = [np.int64(x) for x in classes]
        classes_var = mb.const(val=mil_list(classes))
        out = mb.classify(probabilities=probability_var, classes=classes_var)

        predicted_feature_name = "classLabel" if classifier_config.predicted_feature_name is None \
                                              else classifier_config.predicted_feature_name
        out[0].name = predicted_feature_name
        out[1].name = predicted_feature_name + "_probs"

        # Remove probabilities from block outputs, replace with classify's outputs
        for i in range(0, len(block.outputs)):
            if block.outputs[i] is probability_var:
                block.outputs.pop(i)
                break
        block.outputs[:0] = out
        return out[0].name, out[1].name

def load(prog, weights_dir, resume_on_errors=False, **kwargs):
    if "main" not in prog.functions:
        raise ValueError("main function not found in program")

    mil_passes.mil_backend_passes(prog)

    # if user has specified "ClassifierConfig", then add the "classify" op to the prog
    classifier_config = kwargs.get("classifier_config", None)
    predicted_feature_name = None
    predicted_probabilities_name = None
    if classifier_config is not None:
        predicted_feature_name, predicted_probabilities_name = _add_classify_op(prog, classifier_config)

    input_types = prog.main_input_types
    weight_path = os.path.join(weights_dir, _WEIGHTS_FILE_NAME)
    blob_writer = BlobWriter(weight_path)

    function_protos = {}
    for func_name, func in prog.functions.items():
        function_protos[func_name] = convert_function(func, prog.parameters,
            blob_writer)

    proto = pm.Program(
        version=1,
        functions=function_protos,
    )

    input_features = []
    output_features = []
    symbolic_inputs = []
    image_input_names = {} # these are the model inputs marked as image by the user
    input_shape_map = {}

    for input_type in input_types:
        if isinstance(input_type, ImageType):
            image_input_names[input_type.name] = input_type
            # error checking for input(s) marked as images
            if input_type.name not in list(prog.functions["main"].inputs.keys()):
                msg = "Provided image input '{}' is not one of the inputs of the MIL program"
                raise ValueError(msg.format(input_type.name))
        input_shape_map[input_type.name] = input_type

    for name, var in prog.functions["main"].inputs.items():
        input_feature_type = ft.FeatureType()

        # error checking for input(s) marked as images
        # an image input must be of type tensor in program proto
        # (since an image type does not exist in MIL program)
        if name in image_input_names and \
                not types.is_tensor(var.sym_type):
            raise ValueError("For the image input, '{}', its type in the MIL program must be tensor. "
                             "Instead it is {}.".format(name, var.sym_type.__type_info__()))

        if types.is_tensor(var.sym_type):
            shape = var.sym_type.get_shape()
            if any_variadic(shape):
                raise ValueError("Variable rank model inputs are not supported!")
            if any_symbolic(shape):
                symbolic_inputs.append(name)
                # We extract the default input shape given by user first
                if name in input_shape_map:
                    shape = input_shape_map[name].shape.default
                else:
                    logging.warning("Input shape not fully specified by enumerated shapes or range dim! 1 will be used for dimension not specified instead.")
                # If no input shape is provided (ex. auto conversion of -1 in Tensorflow)
                shape = [1 if is_symbolic(d) else d for d in shape]

            if name not in image_input_names:
                # make a feature type of Type "multiArrayType"
                array_type = ft.ArrayFeatureType(shape=shape, dataType=cast_to_framework_io_dtype(var, False))
                input_feature_type.multiArrayType.CopyFrom(array_type)
            else:
                if len(shape) < 3:
                    raise ValueError("Image input, '{}', must have rank at least 3. Instead it has rank {}".
                                     format(name, len(shape)))
                # make a feature type of Type "imageType"
                input_type = image_input_names[name]
                if not input_type.channel_first:
                    raise ValueError("Image input, '{}', must be in the channel_first format".
                                     format(name))

                if input_type.color_layout == "G":
                    clr_space = ft.ImageFeatureType.ColorSpace.GRAYSCALE
                elif input_type.color_layout == "BGR":
                    clr_space = ft.ImageFeatureType.ColorSpace.BGR
                else:
                    clr_space = ft.ImageFeatureType.ColorSpace.RGB

                image_type = ft.ImageFeatureType(width=shape[-1],
                                                 height=shape[-2],
                                                 colorSpace=clr_space)
                input_feature_type.imageType.CopyFrom(image_type)

            input_features.append(
                ml.FeatureDescription(name=name, type=input_feature_type)
            )
        elif types.is_scalar(var.sym_type):
            array_type = ft.ArrayFeatureType(shape=[1], dataType=cast_to_framework_io_dtype(var, False))
            input_feature_type.multiArrayType.CopyFrom(array_type)
            input_features.append(ml.FeatureDescription(name=var.name, type=input_feature_type))
        else:
            raise NotImplementedError()

    for var in prog.functions["main"].outputs:
        output_feature_type = ft.FeatureType()
        if types.is_tensor(var.sym_type) or types.is_primitive(var.sym_type):
            dataType = None
            if classifier_config is None or var.name != predicted_feature_name:
                # Not a classifier output, make sure model output type matches with ML Program type.
                dataType = cast_to_framework_io_dtype(var, True)
            else:
                # Classifier outputs are set up separately, so default to fp32 for now.
                dataType = ft.ArrayFeatureType.ArrayDataType.FLOAT32

            array_type = ft.ArrayFeatureType(shape=None, dataType=dataType)
            output_feature_type.multiArrayType.CopyFrom(array_type)
            output_features.append(ml.FeatureDescription(name=var.name, type=output_feature_type))
        elif (types.is_dict(var.sym_type)):
            output_feature_type.dictionaryType.MergeFromString(b"")
            keytype, valtype = var.sym_type.T
            if types.is_str(keytype):
                output_feature_type.dictionaryType.stringKeyType.MergeFromString(b"")
            elif (keytype == types.int64):
                output_feature_type.dictionaryType.int64KeyType.MergeFromString(b"")
            else:
                raise ValueError("Dictionary key type not supported.")
            output_features.append(ml.FeatureDescription(name=var.name, type=output_feature_type))
        else:
            raise NotImplementedError()

    # Model description
    desc = ml.ModelDescription(input=input_features, output=output_features)
    if classifier_config is not None:
        desc.predictedFeatureName = predicted_feature_name
        desc.predictedProbabilitiesName = predicted_probabilities_name

        # Manually edit output type of predictedFeatureName.
        # It doesn't use MLMultiArray and really uses a "primitive" type.
        for output in desc.output:
            if output.name == predicted_feature_name:
                if type(classifier_config.class_labels[0]) == int:
                    output.type.int64Type.MergeFromString(b"")
                else:
                    output.type.stringType.MergeFromString(b"")
                break

    # Create ML Model
    model = ml.Model(description=desc, specificationVersion=_SPECIFICATION_VERSION_IOS_15)
    model.mlProgram.CopyFrom(proto)

    # Set symbolic shapes
    for input_name in symbolic_inputs:
        input_type = input_shape_map.get(input_name, None)

        if isinstance(input_type, ImageType):
            if isinstance(input_type.shape, EnumeratedShapes):
                enumerated_shapes = []
                for s in input_type.shape.shapes:
                    enumerated_shapes.append(
                        NeuralNetworkImageSize(
                            height=s.shape[-2], width=s.shape[-1]
                        )
                    )
                add_enumerated_image_sizes(
                    model, input_name, sizes=enumerated_shapes
                )
            else:
                img_range = NeuralNetworkImageSizeRange()
                H = input_type.shape.shape[-2]
                W = input_type.shape.shape[-1]

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

                update_image_size_range(
                    model, input_name, img_range
                )
        elif isinstance(input_type, TensorType):
            if isinstance(input_type.shape, EnumeratedShapes):
                add_multiarray_ndshape_enumeration(
                    model, input_name, [tuple(s.shape) for s in input_type.shape.shapes]
                )
            else:
                lb = []
                ub = []
                for s in input_type.shape.shape:
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
                    model, input_name, lower_bounds=lb, upper_bounds=ub
                )
        elif input_type is None:
            sym_type = prog.functions["main"].inputs[input_name].sym_type
            lb = []
            ub = []
            for s in sym_type.get_shape():
                if is_symbolic(s):
                    lb.append(1)
                    ub.append(-1)
                else:
                    lb.append(s)
                    ub.append(s)
            set_multiarray_ndshape_range(
                model, input_name, lower_bounds=lb, upper_bounds=ub
            )

    # Set optional inputs
    _set_optional_inputs(model, input_types)

    return model
