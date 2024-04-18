#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np

from coremltools import _OPSET, _SPECIFICATION_VERSION_IOS_15, _SPECIFICATION_VERSION_IOS_17
from coremltools import _logger as logger
from coremltools import proto
from coremltools.converters.mil import mil
from coremltools.converters.mil.backend.backend_helper import _get_probability_var_for_classifier
from coremltools.converters.mil.backend.mil import helper
from coremltools.converters.mil.backend.mil.helper import (
    cast_to_framework_io_dtype,
    create_file_value_tensor,
    create_immediate_value,
    create_list_scalarvalue,
    create_scalar_value,
    create_valuetype_list,
    create_valuetype_scalar,
    create_valuetype_tensor,
    types_to_proto_primitive,
)
from coremltools.converters.mil.backend.nn.load import _set_optional_inputs
from coremltools.converters.mil.input_types import (
    ClassifierConfig,
    EnumeratedShapes,
    ImageType,
    RangeDim,
    TensorType,
)
from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Operation, Program, Var, mil_list, types
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry
from coremltools.converters.mil.mil.scope import ScopeInfo, ScopeSource
from coremltools.converters.mil.mil.types.symbolic import any_symbolic, any_variadic, is_symbolic
from coremltools.models.neural_network import flexible_shape_utils
from coremltools.models.neural_network.flexible_shape_utils import (
    NeuralNetworkImageSize,
    NeuralNetworkImageSizeRange,
)
from coremltools.models.utils import _WEIGHTS_DIR_NAME, _WEIGHTS_FILE_NAME

from ..backend_helper import _get_colorspace_enum, _validate_image_input_output_shapes

try:
    from coremltools.libmilstoragepython import _BlobStorageWriter as BlobWriter
except Exception as e:
    logger.warning(f"Fail to import BlobWriter from libmilstoragepython. {e}")
    BlobWriter = None


def should_use_weight_file(val):
    return (
        val is not None
        and isinstance(val, (np.ndarray, np.generic))
        and val.size >= 10
        and val.dtype in ['float16', 'float32', 'uint8', 'int8']
    )

class MILProtoExporter:
    """
    An utility class to export a pymil program to milproto.
    """

    def __init__(
        self,
        prog: Program,
        weights_dir: str,
    ):
        self.prog = prog
        self.weights_dir = weights_dir
        self.blob_writers = {}
        self.prog.validate(check_essential_scope=True)

    def translate_program_attributes(self) -> Dict[str, Any]:
        """
        Get the program attributes which need to be exported to mil proto.
        """
        return {}

    def get_weight_path(self, op: Operation) -> str:
        """
        Get the weight path for a constant operation.
        By default, the weight is saved in {weight_dir}/weight.bin
        """
        assert (
            op.op_type == "const"
        ), f"Expected op (op.name) be a const op. Got op_type of {op.op_type}."
        return os.path.join(self.weights_dir, _WEIGHTS_FILE_NAME)

    def get_blob_writer(self, weight_path: str) -> BlobWriter:
        """
        Get a blob writer given a weight_path.
        """
        if weight_path not in self.blob_writers:
            self.blob_writers[weight_path] = BlobWriter(weight_path)
        return self.blob_writers[weight_path]

    def create_file_value(self, var: Var) -> proto.MIL_pb2.Value:
        """
        Returns the mil proto file value of a var.
        """
        weight_path = self.get_weight_path(var.op)
        blob_writer = self.get_blob_writer(weight_path)
        offset = helper._get_offset_by_writing_data(var, blob_writer)
        weight_file_name = os.path.basename(weight_path)

        return create_file_value_tensor(
            file_name=os.path.join(
                os.path.join("@model_path", _WEIGHTS_DIR_NAME), weight_file_name
            ),
            offset=offset,
            dim=var.val.shape,
            data_type=types_to_proto_primitive(var.sym_type.get_primitive()),
        )

    def get_milproto_value(self, var: Var) -> proto.MIL_pb2.Value:
        """
        Translate a pymil Var into milproto value.
        """
        if should_use_weight_file(var.val):
            return self.create_file_value(var)
        else:
            return create_immediate_value(var)

    @staticmethod
    def _get_input_dict(op: Operation) -> Dict[str, Any]:
        """
        Given an op, returns a dict that maps the param name into the corresponding Var.
        """
        return op.inputs

    @staticmethod
    def _get_attr_dict(op: Operation) -> Dict[str, Any]:
        """
        Return the initial attribute dict for an op.
        """
        return {"name": create_scalar_value(op.name)}

    def translate_const(self, op: Operation) -> proto.MIL_pb2.Operation:
        """
        Translate constant operation.
        """
        if len(op.outputs) != 1:
            raise AssertionError(f"const {op.name} must have 1 output, but got {len(op.outputs)}")

        output_var = op.outputs[0]
        value = self.get_milproto_value(output_var)

        return proto.MIL_pb2.Operation(
            type="const",
            attributes={"name": create_scalar_value(op.name), "val": value},
            outputs=[
                proto.MIL_pb2.NamedValueType(
                    name=output_var.name, type=self.types_to_proto(output_var.sym_type)
                )
            ],
        )

    def translate_constexpr(self, op: Operation) -> proto.MIL_pb2.Operation:
        """
        Translate constexpr operation.
        """
        inputs = {}
        attributes = {"name": create_scalar_value(op.name)}

        if op.opset_version <= _SPECIFICATION_VERSION_IOS_17:
            attributes.update(
                {param_name: self.get_milproto_value(var) for param_name, var in op.inputs.items()}
            )
        else:
            for param_name, var in op.inputs.items():
                if var.op.op_type.startswith("constexpr_"):
                    arguments = [proto.MIL_pb2.Argument.Binding(name=var.name)]
                else:
                    arguments = [proto.MIL_pb2.Argument.Binding(value=self.get_milproto_value(var))]
                args = proto.MIL_pb2.Argument()
                args.arguments.extend(arguments)
                inputs[param_name] = args

        return proto.MIL_pb2.Operation(
            type=op.op_type,
            inputs=inputs,
            attributes=attributes,
            outputs=[
                proto.MIL_pb2.NamedValueType(
                    name=output_var.name, type=self.types_to_proto(output_var.sym_type)
                )
                for output_var in op.outputs
            ],
        )

    def create_valuetype_dict(self, key_type: type, value_type: type) -> proto.MIL_pb2.ValueType:
        """
        Return proto.MIL_pb2.ValueType with dict (dictionaryType) set
        """
        v_type = proto.MIL_pb2.ValueType()
        v_type.dictionaryType.keyType.CopyFrom(self.types_to_proto(key_type))
        v_type.dictionaryType.valueType.CopyFrom(self.types_to_proto(value_type))
        return v_type

    def types_to_proto(self, valuetype: type) -> proto.MIL_pb2.ValueType:
        """
        Return proto.MIL_pb2.ValueType from PyMIL types.
        """
        if types.is_tensor(valuetype):
            primitive = types_to_proto_primitive(valuetype.get_primitive())
            return create_valuetype_tensor(valuetype.get_shape(), primitive)
        elif types.is_tuple(valuetype):
            v_type = proto.MIL_pb2.ValueType()
            t_type = v_type.tupleType
            for t in valuetype.T:
                new_v_type = t_type.types.add()
                new_v_type.CopyFrom(self.types_to_proto(t))
            return v_type
        elif types.is_list(valuetype):
            elem = valuetype.T[0]
            length = valuetype.T[1]
            if types.is_tensor(elem):
                dtype = types_to_proto_primitive(elem.get_primitive())
                elem_shape = elem.get_shape()
            elif types.is_scalar(elem):
                dtype = types_to_proto_primitive(valuetype)
                elem_shape = ()
            elif types.is_str(elem):
                dtype = types_to_proto_primitive(elem)
                elem_shape = ()
            else:
                raise NotImplementedError(
                    "Only list of either tensors or scalars supported. "
                    "Got element of type {}".format(elem.__type_info__())
                )
            return create_valuetype_list(length=length, elem_shape=elem_shape, dtype=dtype)
        elif types.is_dict(valuetype):
            return self.create_valuetype_dict(valuetype.T[0], valuetype.T[1])
        else:
            return create_valuetype_scalar(types_to_proto_primitive(valuetype))

    def translate_generic_op(
        self, op: Operation, literal_params: Optional[List[str]] = None
    ) -> proto.MIL_pb2.Operation:
        """
        Translate a generic pymil Operation.
        """
        if literal_params is None:
            literal_params = []

        inputs = {}

        for param_name, vars in self._get_input_dict(op).items():
            if param_name.startswith("_"):
                continue
            if not isinstance(vars, (list, tuple)):
                vars = [vars]

            arguments = []
            for _var in vars:
                binding = proto.MIL_pb2.Argument.Binding()
                # use const value literals if requested
                if param_name in literal_params:
                    binding.value.CopyFrom(create_immediate_value(_var))
                else:
                    binding.name = _var.name
                arguments.append(binding)

            args = proto.MIL_pb2.Argument()
            args.arguments.extend(arguments)
            inputs[param_name] = args

        outputs = [
            proto.MIL_pb2.NamedValueType(name=v.name, type=self.types_to_proto(v.sym_type))
            for v in op.outputs
        ]
        blocks = None
        if len(op.blocks) > 0:
            blocks = [self.create_block(b) for b in op.blocks]

        op_type = op.op_type
        attr_dict = self._get_attr_dict(op)
        if op.op_type in SSAOpRegistry.custom_ops:
            op_type = "custom_layer"
            class_name = op.bindings.get("class_name", op.name)
            input_order = op.bindings.get("input_order", [])
            parameters = op.bindings.get("parameters", [])
            weights = op.bindings.get("weights", [])
            description = op.bindings.get("description", "")

            attr_dict["class_name"] = create_scalar_value(class_name)
            attr_dict["input_order"] = create_list_scalarvalue(input_order, str)
            attr_dict["parameters"] = create_list_scalarvalue(parameters, str)
            attr_dict["weights"] = create_list_scalarvalue(weights, str)
            attr_dict["description"] = create_scalar_value(description)

        return proto.MIL_pb2.Operation(
            type=op_type,
            blocks=blocks,
            inputs=inputs,
            attributes=attr_dict,
            outputs=outputs,
        )

    def create_block(self, block: Block) -> proto.MIL_pb2.Block:
        """
        Translate pymil Block.
        """
        def feeds_to_only_constexprs(op: Operation) -> bool:
            return (
                (op.op_type == "const")
                and len(op.outputs[0].child_ops) > 0
                and all(
                    (child_op.op_type.startswith("constexpr_"))
                    for child_op in op.outputs[0].child_ops
                )
            )

        proto_ops = []

        # Find the const op that generates classify's "label" / "class" string vec.
        classify_const_classes_op = None
        if len(block.operations) > 0:
            # Classify is always the last operation in the block.
            op = block.operations[-1]
            op_cls_name = type(op).__name__
            if op_cls_name == "classify":
                classes_var = op.inputs["classes"]
                classify_const_classes_op = classes_var.op
                if len(classes_var.child_ops) != 1:
                    raise ValueError(
                        "Classify's labels/classes should be input to only 1 op (classify)."
                    )

        for op in block.operations:
            op_cls_name = type(op).__name__
            if op_cls_name == "const":
                if feeds_to_only_constexprs(op):
                    continue
                # Do not serialize the const op that creates the var bound to the classifier's "classes" param.
                # The variable's value will be bound directly to classify's "classes" param instead.
                if op != classify_const_classes_op:
                    proto_ops.append(self.translate_const(op))
            elif op_cls_name.startswith("constexpr_"):
                proto_ops.append(self.translate_constexpr(op))
            elif op_cls_name == "classify":
                # Classify's "classes" param should be serialized as a value literal bound
                # directly to the param, rather than as a const-generated variable.
                proto_ops.append(self.translate_generic_op(op, ["classes"]))
            elif op_cls_name == "reshape_like":
                # The reshape_like should also be able to take value from a const op
                # This is a workaround solution
                # rdar://98689808 (Reshape_like should also accept const value from non literal input)
                literal_params = ["begins", "ends", "end_masks"]
                proto_ops.append(self.translate_generic_op(op, literal_params))
            else:
                # A single pymil op might be decomposed into multiple ops
                ops = self.translate_generic_op(op)
                if not isinstance(ops, list):
                    ops = [ops]
                proto_ops.extend(ops)

        inputs = []
        if not isinstance(block, Function):
            # Function is subclass of Block, but function's block has no input,
            # and hence skipping reading the block inputs.
            for var in block.inputs:
                proto_type = self.types_to_proto(var.sym_type)
                inputs.append(proto.MIL_pb2.NamedValueType(name=var.name, type=proto_type))
        output_names = [v.name for v in block.outputs]
        return proto.MIL_pb2.Block(inputs=inputs, outputs=output_names, operations=proto_ops)

    def convert_function(self, function: Function, opset: str) -> proto.MIL_pb2.Function:
        """
        Translate pymil Function.
        """
        block = self.create_block(function)

        inputs = []
        for name, var in function.inputs.items():
            proto_type = self.types_to_proto(var.sym_type)
            inputs.append(proto.MIL_pb2.NamedValueType(name=name, type=proto_type))

        return proto.MIL_pb2.Function(
            inputs=inputs, opset=opset, block_specializations={opset: block}
        )

    def export(
        self, specification_version: Optional[str] = _SPECIFICATION_VERSION_IOS_15
    ) -> proto.MIL_pb2.Program:
        """
        Export a pymil program into mil proto with the given specification version.
        """
        if BlobWriter is None:
            raise RuntimeError("BlobWriter not loaded")

        function_protos = {}
        for func_name, func in self.prog.functions.items():
            function_protos[func_name] = self.convert_function(func, _OPSET[specification_version])

        kwargs = {
            "version": 1,
            "functions": function_protos,
        }

        prog_attributes = self.translate_program_attributes()
        if len(prog_attributes) > 0:
            kwargs["attributes"] = prog_attributes

        return proto.MIL_pb2.Program(**kwargs)

# Add a classify op to the output.
# Replaces the original probabilities output (in the containing MIL block)
# with the outputs of the classifier op. Returns the name of the original
# probabilities output variable.
def _add_classify_op(prog, classifier_config):
    '''
    Add a "classify" op to the program, at the end of the main block
    '''
    def remove_output(block, prob_var):
        for i in range(len(block.outputs)):
            if block.outputs[i] is prob_var:
                block.outputs.pop(i)
                if block in prob_var.consuming_blocks:
                    prob_var.consuming_blocks.remove(block)
                break

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
    original_probability_var = probability_var

    # add the classify op now
    # we consider this step as a scope of coremltools graph pass
    with mb.scope(ScopeInfo(source=ScopeSource.COREMLTOOLS_GRAPH_PASS, data=["add_classify_op"])):
        with block:
            # cast the int label to np.int64
            if isinstance(classes[0], int):
                classes = [np.int64(x) for x in classes]
            classes_var = mb.const(val=mil_list(classes))
            if probability_var.dtype != types.fp32:
                remove_output(block, probability_var)
                probability_var = mb.cast(
                    x=probability_var, dtype="fp32", name=probability_var.name + "_cast_to_fp32"
                )
            out = mb.classify(probabilities=probability_var, classes=classes_var)

            predicted_feature_name = (
                "classLabel"
                if classifier_config.predicted_feature_name is None
                else classifier_config.predicted_feature_name
            )
            out[0].name = predicted_feature_name
            out[1].name = predicted_feature_name + "_probs"

            # Remove probabilities from block outputs, replace with classify's outputs
            remove_output(block, probability_var)
            block.outputs[:0] = out
            out[0].consuming_blocks.append(block)
            out[1].consuming_blocks.append(block)

            # The new classifier op should have scope information
            Block._copy_scope_info(original_probability_var, out[0])

            return out[0].name, out[1].name


class CoreMLProtoExporter:
    """
    An utility class to export a pymil program to coreml model.
    """

    _DEFAULT_FUNCTION_NAME = "main"

    def __init__(
        self,
        prog: mil.Program,
        mil_proto: proto.MIL_pb2.Program,
        predicted_feature_name: str,
        predicted_probabilities_name: str,
        classifier_config: ClassifierConfig,
        convert_to: str,
        convert_from: str,
    ):
        self.prog = prog
        self.mil_proto = mil_proto
        self.predicted_feature_name = predicted_feature_name
        self.predicted_probabilities_name = predicted_probabilities_name
        self.classifier_config = classifier_config
        self.convert_to = convert_to
        self.convert_from = convert_from
        self.prog.validate(check_essential_scope=True)

    @staticmethod
    def get_additional_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get additional coreml proto related kwargs.
        """
        return {}

    @staticmethod
    def _try_convert_other_input_type(
        input_var: Var, input_features: List[proto.Model_pb2.FeatureDescription]
    ) -> bool:
        """
        Try to convert an input var with additional type.
        """
        return False

    def get_func_input(self, func: mil.Function) -> List[proto.Model_pb2.FeatureDescription]:
        """
        Utils to get function input feature description.
        """
        input_types = func.input_types

        input_features = []
        image_input_names = {}  # these are the model inputs marked as image by the user
        input_shape_map = {}

        for input_type in input_types:
            if isinstance(input_type, ImageType):
                image_input_names[input_type.name] = input_type
                # error checking for input(s) marked as images
                if input_type.name not in list(func.inputs.keys()):
                    raise ValueError(
                        f"Provided image input '{input_type.name}' is not one of the inputs of the MIL program"
                    )
            if input_type.name is None:
                raise ValueError(
                    'Fail to auto-determine the input name. Please specify the "name" '
                    'parameter when use "inputs" in ct.convert().'
                )
            input_shape_map[input_type.name] = input_type

        for name, var in func.inputs.items():
            input_feature_type = proto.FeatureTypes_pb2.FeatureType()
            is_input_shape_symbolic = False

            # error checking for input(s) marked as images
            # an image input must be of type tensor in program proto
            # (since an image type does not exist in MIL program)
            if name in image_input_names and not types.is_tensor(var.sym_type):
                raise ValueError(
                    "For the image input, '{}', its type in the MIL program must be tensor. "
                    "Instead it is {}.".format(name, var.sym_type.__type_info__())
                )

            if types.is_tensor(var.sym_type):
                shape = var.sym_type.get_shape()
                if any_variadic(shape):
                    raise ValueError("Variable rank model inputs are not supported!")
                if any_symbolic(shape):
                    is_input_shape_symbolic = True
                    # We extract the default input shape given by user first
                    if name in input_shape_map:
                        shape = input_shape_map[name].shape.default
                    else:
                        logger.warning(
                            "Input shape not fully specified by enumerated shapes or range dim! 1 will be used for dimension not specified instead."
                        )
                    # If no input shape is provided (ex. auto conversion of -1 in Tensorflow)
                    shape = [1 if is_symbolic(d) else d for d in shape]

                if name not in image_input_names:
                    # make a feature type of Type "multiArrayType"
                    array_type = proto.FeatureTypes_pb2.ArrayFeatureType(
                        shape=shape, dataType=cast_to_framework_io_dtype(var, False)
                    )
                    input_feature_type.multiArrayType.CopyFrom(array_type)
                else:
                    # make a feature type of Type "imageType"
                    input_type = image_input_names[name]
                    _validate_image_input_output_shapes(
                        input_type.color_layout, shape, name, is_input=True
                    )
                    if not input_type.channel_first:
                        raise ValueError(
                            "Image input, '{}', must be in the channel_first format".format(name)
                        )
                    clr_space = _get_colorspace_enum(input_type.color_layout)
                    image_type = proto.FeatureTypes_pb2.ImageFeatureType(
                        width=shape[-1], height=shape[-2], colorSpace=clr_space
                    )
                    input_feature_type.imageType.CopyFrom(image_type)

                input_features.append(
                    proto.Model_pb2.FeatureDescription(name=name, type=input_feature_type)
                )
            elif types.is_scalar(var.sym_type):
                array_type = proto.FeatureTypes_pb2.ArrayFeatureType(
                    shape=[1], dataType=cast_to_framework_io_dtype(var, False)
                )
                input_feature_type.multiArrayType.CopyFrom(array_type)
                input_features.append(
                    proto.Model_pb2.FeatureDescription(name=var.name, type=input_feature_type)
                )
            elif not self._try_convert_other_input_type(var, input_features):
                raise NotImplementedError(f"Unsupported input type {var.sym_type}.")

            if not is_input_shape_symbolic:
                continue

            # Set symbolic shapes
            default_lower_bound = 1
            default_upper_bound = default_lower_bound + 1 if self.convert_to == "mlprogram" else -1
            default_bound_used = False
            input_type = input_shape_map.get(name, None)

            if isinstance(input_type, ImageType):
                if isinstance(input_type.shape, EnumeratedShapes):
                    enumerated_shapes = []
                    for s in input_type.shape.shapes:
                        enumerated_shapes.append(
                            NeuralNetworkImageSize(height=s.shape[-2], width=s.shape[-1])
                        )
                    flexible_shape_utils._add_enumerated_image_sizes_for_feature(
                        input_features[-1], sizes=enumerated_shapes
                    )
                else:
                    img_range = NeuralNetworkImageSizeRange()
                    H = input_type.shape.shape[-2]
                    W = input_type.shape.shape[-1]

                    if isinstance(H, RangeDim):
                        img_range.add_height_range((H.lower_bound, H.upper_bound))
                    elif is_symbolic(H):
                        img_range.add_height_range((default_lower_bound, default_upper_bound))
                        default_bound_used = True
                    else:
                        img_range.add_height_range((H, H))
                    if isinstance(W, RangeDim):
                        img_range.add_width_range((W.lower_bound, W.upper_bound))
                    elif is_symbolic(W):
                        img_range.add_width_range((default_lower_bound, default_upper_bound))
                        default_bound_used = True
                    else:
                        img_range.add_width_range((W, W))

                    flexible_shape_utils._update_image_size_range_for_feature(
                        input_features[-1], img_range
                    )
            elif isinstance(input_type, TensorType):
                if isinstance(input_type.shape, EnumeratedShapes):
                    flexible_shape_utils._add_multiarray_ndshape_enumeration_for_feature(
                        input_features[-1], [tuple(s.shape) for s in input_type.shape.shapes]
                    )
                else:
                    lb = []
                    ub = []
                    for s in input_type.shape.shape:
                        if isinstance(s, RangeDim):
                            lb.append(s.lower_bound)
                            ub.append(s.upper_bound)
                        elif is_symbolic(s):
                            lb.append(default_lower_bound)
                            ub.append(default_upper_bound)
                            default_bound_used = True
                        else:
                            lb.append(s)
                            ub.append(s)
                    flexible_shape_utils._set_multiarray_ndshape_range_for_feature(
                        input_features[-1], lower_bounds=lb, upper_bounds=ub
                    )
            elif input_type is None:
                sym_type = func.inputs[name].sym_type
                lb = []
                ub = []
                for s in sym_type.get_shape():
                    if is_symbolic(s):
                        lb.append(default_lower_bound)
                        ub.append(default_upper_bound)
                        default_bound_used = True
                    else:
                        lb.append(s)
                        ub.append(s)
                flexible_shape_utils._set_multiarray_ndshape_range_for_feature(
                    input_features[-1], lower_bounds=lb, upper_bounds=ub
                )

            if default_bound_used and self.convert_to == "mlprogram":
                warnings.warn(
                    "Some dimensions in the input shape are unknown, hence they are set to flexible ranges "
                    f"with lower bound and default value = {default_lower_bound}, and upper bound = "
                    f"{default_upper_bound}. To set different values for the default shape and upper bound, "
                    "please use the ct.RangeDim() method as described here: "
                    "https://coremltools.readme.io/docs/flexible-inputs#set-the-range-for-each-dimension.",
                    UserWarning,
                )
                convert_from = self.convert_from
                if convert_from is not None and convert_from.startswith("tensorflow"):
                    warnings.warn(
                        'There is "None" dim in TF input placeholder. Please consider specifying '
                        'input shapes by using the "inputs" param in ct.convert().'
                    )

        return input_features

    def get_func_output(self, func: mil.Function) -> List[proto.Model_pb2.FeatureDescription]:
        """
        Utils to get function output feature description.
        """

        output_types = func.output_types
        output_features = []

        if output_types is not None and self.classifier_config is None:
            assert len(output_types) == len(
                func.outputs
            ), "number of mil program outputs do not match the number of outputs provided by the user"

        for i, var in enumerate(func.outputs):
            output_feature_type = proto.FeatureTypes_pb2.FeatureType()
            if types.is_tensor(var.sym_type) or types.is_primitive(var.sym_type):
                if output_types is not None and isinstance(output_types[i], ImageType):
                    if not types.is_tensor(var.sym_type):
                        raise ValueError(
                            "Image output, '{}', is a scalar, but it should be a tensor of rank 4".format(
                                var.name
                            )
                        )

                    clr_space = _get_colorspace_enum(output_types[i].color_layout)

                    shape = var.sym_type.get_shape()
                    if any_variadic(shape):
                        raise ValueError(
                            "Variable rank model outputs, that are ImageTypes, are not supported"
                        )
                    if any_symbolic(shape):
                        # For flexible shape output, we set the imageSizeRange to [1, -1],
                        # util this radar is fixed in CoreML: rdar://122895892 ([Bug] CoreML produce empty dictionary with image output with dynamic shape)
                        image_type = proto.FeatureTypes_pb2.ImageFeatureType(
                            width=1, height=1, colorSpace=clr_space
                        )
                        image_type.imageSizeRange.widthRange.lowerBound = 1
                        image_type.imageSizeRange.widthRange.upperBound = -1
                        image_type.imageSizeRange.heightRange.lowerBound = 1
                        image_type.imageSizeRange.heightRange.upperBound = -1
                    else:
                        image_type = proto.FeatureTypes_pb2.ImageFeatureType(
                            width=shape[-1], height=shape[-2], colorSpace=clr_space
                        )
                    _validate_image_input_output_shapes(
                        output_types[i].color_layout, shape, var.name, is_input=False
                    )

                    output_feature_type.imageType.CopyFrom(image_type)
                    output_features.append(
                        proto.Model_pb2.FeatureDescription(name=var.name, type=output_feature_type)
                    )
                else:
                    dataType = None
                    if self.classifier_config is None or var.name != self.predicted_feature_name:
                        # Not a classifier output, make sure model output type matches with ML Program type.
                        dataType = cast_to_framework_io_dtype(var, True)
                    else:
                        # Classifier outputs are set up separately, so default to fp32 for now.
                        dataType = proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.FLOAT32

                    output_shape = (
                        None
                        if any_symbolic(var.shape) or types.is_primitive(var.sym_type)
                        else var.shape
                    )
                    array_type = proto.FeatureTypes_pb2.ArrayFeatureType(
                        shape=output_shape, dataType=dataType
                    )
                    output_feature_type.multiArrayType.CopyFrom(array_type)
                    output_features.append(
                        proto.Model_pb2.FeatureDescription(name=var.name, type=output_feature_type)
                    )
            elif types.is_dict(var.sym_type):
                output_feature_type.dictionaryType.MergeFromString(b"")
                keytype, valtype = var.sym_type.T
                if types.is_str(keytype):
                    output_feature_type.dictionaryType.stringKeyType.MergeFromString(b"")
                elif keytype == types.int64:
                    output_feature_type.dictionaryType.int64KeyType.MergeFromString(b"")
                else:
                    raise ValueError("Dictionary key type not supported.")
                output_features.append(
                    proto.Model_pb2.FeatureDescription(name=var.name, type=output_feature_type)
                )
            else:
                raise NotImplementedError(f"Unsupported output type {var.sym_type}.")

        return output_features

    def create_model_description(
        self,
        input_features: List[proto.Model_pb2.FeatureDescription],
        output_features: List[proto.Model_pb2.FeatureDescription],
    ) -> proto.Model_pb2.ModelDescription:
        """
        Create model description from input and output features
        """
        return proto.Model_pb2.ModelDescription(input=input_features, output=output_features)

    def get_coreml_model(
        self,
        input: Dict[str, List[proto.Model_pb2.FeatureDescription]],
        output: Dict[str, List[proto.Model_pb2.FeatureDescription]],
        specification_version: int,
    ) -> proto.Model_pb2.Model:
        """
        Utils to get a coreml model description.
        """
        # Model description
        input_features = input[self._DEFAULT_FUNCTION_NAME]
        output_features = output[self._DEFAULT_FUNCTION_NAME]
        desc = self.create_model_description(input_features, output_features)

        if self.classifier_config is not None:
            desc.predictedFeatureName = self.predicted_feature_name
            desc.predictedProbabilitiesName = self.predicted_probabilities_name

            # Manually edit output type of predictedFeatureName.
            # It doesn't use MLMultiArray and really uses a "primitive" type.
            for output in desc.output:
                if output.name == self.predicted_feature_name:
                    if type(self.classifier_config.class_labels[0]) == int:
                        output.type.int64Type.MergeFromString(b"")
                    else:
                        output.type.stringType.MergeFromString(b"")
                    break

        # Create ML Model
        model = proto.Model_pb2.Model(description=desc, specificationVersion=specification_version)
        model.mlProgram.CopyFrom(self.mil_proto)

        return model

    def export(
        self, specification_version: Optional[int] = _SPECIFICATION_VERSION_IOS_15
    ) -> proto.Model_pb2.Model:

        # get functions input / output description
        func_to_input = OrderedDict()
        func_to_output = OrderedDict()

        for name, func in self.prog.functions.items():
            func_to_input[name] = self.get_func_input(func)
            func_to_output[name] = self.get_func_output(func)

        # create a coreml model with I/O description and mil proto
        model = self.get_coreml_model(
            func_to_input,
            func_to_output,
            specification_version,
        )

        # Set optional inputs for main function
        _set_optional_inputs(model, self.prog.functions["main"].input_types)

        return model


def load(
    prog: Program,
    weights_dir: str,
    resume_on_errors: Optional[bool] = False,
    specification_version: Optional[int] = _SPECIFICATION_VERSION_IOS_15,
    **kwargs,
) -> proto.Model_pb2.Model:
    if "main" not in prog.functions:
        raise ValueError("main function not found in program")

    # if user has specified "ClassifierConfig", then add the "classify" op to the prog
    classifier_config = kwargs.get("classifier_config", None)
    predicted_feature_name, predicted_probabilities_name = None, None
    if classifier_config is not None:
        predicted_feature_name, predicted_probabilities_name = _add_classify_op(
            prog, classifier_config
        )

    # convert pymil program into mil proto
    mil_proto_exporter = MILProtoExporter(
        prog,
        weights_dir,
    )
    mil_proto = mil_proto_exporter.export(specification_version)

    # return the model provided by users
    desc = kwargs.get("model_description", None)
    if desc and not isinstance(desc, proto.Model_pb2.ModelDescription):
        raise ValueError("Invalid model descriptor")

    if desc:
        if classifier_config is not None:
            raise AssertionError("Both model_description and classifier_config can't be provided")
        model = proto.Model_pb2.Model(description=desc, specificationVersion=specification_version)
        model.mlProgram.CopyFrom(mil_proto)
        return model

    # create a CoreML model protobuf
    exporter_kwargs = CoreMLProtoExporter.get_additional_kwargs(kwargs)
    coreml_proto_exporter = CoreMLProtoExporter(
        prog,
        mil_proto,
        predicted_feature_name,
        predicted_probabilities_name,
        classifier_config=kwargs.get("classifier_config", None),
        convert_to=kwargs.get("convert_to", None),
        convert_from=kwargs.get("convert_from", None),
        **exporter_kwargs,
    )
    return coreml_proto_exporter.export(specification_version)
