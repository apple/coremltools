# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import logging
import os

import coremltools.proto.Program_pb2 as pm
import coremltools.proto.Model_pb2 as ml
import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools.models.neural_network.flexible_shape_utils import set_multiarray_ndshape_range
from coremltools.models.program.builder import NeuralNetBuffer as NetBuffer
from coremltools.converters.nnv2.builtin_types import builtins
import tempfile
from coremltools.converters.nnv2.builtin_types.symbolic import (
        any_symbolic, is_symbolic)

from coremltools.converters.nnv2.backend.nnv2.helper import *

default_weight_path = tempfile.mkstemp(suffix="weights.wt", prefix="tmp")[1]

def translate_const(op, net_buffer, weight_path):
    output_var = op.outputs[0]
    if op.mode.val == "file_value":
        if not builtins.is_tensor(output_var.sym_type):
            raise NotImplementedError("Only support tensor type for file value.")
        offset = net_buffer.add_buffer(output_var.val.flatten())
        value = create_file_value_tensor(file_name=weight_path,
                offset=offset, dim=output_var.val.shape,
                scalar_type=builtins_to_proto_primitive(
                    output_var.sym_type.get_primitive()))
    elif op.mode.val == "immediate_value":
        if builtins.is_tensor(output_var.sym_type):
            value = create_tensor_value(output_var.val)
        else:
            value = create_scalar_value(output_var.val)
    elif op.mode.val == "parameter":
        raise NotImplementedError("Parameter is not supported yet.")
    else:
        raise RuntimeError("Unrecognized constant mode")

    return pm.Operation(name=op.name, type='const',
                      attributes={'val': value},
                      outputs=[pm.NamedValueType(name=output_var.name,
                          type=builtins_to_proto(output_var.sym_type))])

def translate_generic_op(op, parameters, net_buffer):
    inputs = {arg_name: v.name for arg_name, v in op.inputs.items() \
                    if arg_name[0] != "_"}
    outputs = [pm.NamedValueType(name=v.name,
                type=builtins_to_proto(v.sym_type)) for v in op.outputs]
    blocks = None
    if len(op.blocks) > 0:
        blocks = [create_block(b, parameters, net_buffer) for b in op.blocks]
    return pm.Operation(name=op.name,
            type=op.op_type,
            blocks=blocks,
            inputs=inputs,
            outputs=outputs)

def create_block(block, parameters, net_buffer):
    proto_ops = []
    for op in block.operations:
        op_cls_name = type(op).__name__
        if op_cls_name == 'const':
            proto_ops.append(translate_const(op, net_buffer,
                default_weight_path))
        else:
            proto_ops.append(translate_generic_op(op, parameters, net_buffer))

    input_map = {}#{v: v for v in block.inputs} // TODO <rdar://problem/58883380> Handle block inputs neatly and correctly.
    output_names = [v.name for v in block.outputs]
    return pm.Block(inputs=input_map,
            outputs=output_names, operations=proto_ops)

def convert_function(function, parameters, net_buffer):
    block = create_block(function, parameters, net_buffer)

    inputs = []
    outputs = []
    for name, var in function.inputs.items():
        proto_type = builtins_to_proto(var.sym_type)
        inputs.append(pm.NamedValueType(name=name, type=proto_type))

    for var in function.outputs:
        outputs.append(builtins_to_proto(var.sym_type))

    return pm.Function(inputs=inputs, outputs=outputs, block=block)

def load(prog, resume_on_errors=False, **kwargs):
    if 'main' not in prog.functions:
        raise ValueError('main function not found in program')

    if os.path.exists(default_weight_path):
        os.remove(default_weight_path)

    net_buffer = NetBuffer(default_weight_path)

    function_protos = {}
    parameter_protos = {}
    for func_name, func in prog.functions.items():
        function_protos[func_name] = convert_function(func, prog.parameters,
                net_buffer)

    proto = pm.Program(version=1, functions=function_protos,
            parameters=parameter_protos)

    input_features = []
    output_features = []
    symbolic_inputs = []

    for name, var in prog.functions['main'].inputs.items():
        input_feature_type = ft.FeatureType()
        if builtins.is_tensor(var.sym_type):
            shape = var.sym_type.get_shape()
            if any_symbolic(shape):
                # Use dummy static shape, and will set it later.
                symbolic_inputs.append((name, shape))
                # Pick an example shape (symbolic dimensions must have value
                # between lower_bound and upper_bound in
                # `set_multiarray_ndshape_range`
                shape = [1 if is_symbolic(d) else d for d in shape]
            array_type = ft.ArrayFeatureType(shape=shape,
                    dataType=ft.ArrayFeatureType.ArrayDataType.FLOAT32)
            input_feature_type.multiArrayType.CopyFrom(array_type)
            input_features.append(ml.FeatureDescription(name=name,
                type=input_feature_type))
        else:
            raise NotImplementedError()

    for var in prog.functions['main'].outputs:
        output_feature_type = ft.FeatureType()
        if builtins.is_tensor(var.sym_type):
            # TODO: what should we do for non-deterministic shape?
            #       dataType needs to be mapped to such type.
            #       <rdar://problem/57402427> FeatureType for MLModel unsupported
            shape = var.sym_type.get_shape()
            if any_symbolic(shape):
                # Use . MLmodel doesn't care output shape.
                shape = [-1 if is_symbolic(d) else d for d in shape]
            array_type = ft.ArrayFeatureType(shape=shape,
                    dataType=ft.ArrayFeatureType.ArrayDataType.FLOAT32)
            output_feature_type.multiArrayType.CopyFrom(array_type)
            output_features.append(ml.FeatureDescription(name=var.name,
                type=output_feature_type))
        else:
            raise NotImplementedError()

    # Model description
    desc = ml.ModelDescription(input=input_features,
                               output=output_features)
    # Create ML Model
    # TODO: <rdar://problem/57402360> specificationVersion invalidated
    model = ml.Model(description=desc, specificationVersion=5)
    model.serializedModel.identifier = "program"
    model.serializedModel.model = proto.SerializeToString()

    # Set symbolic input shapes
    for input_name, shape in symbolic_inputs:
        lb = [1 if is_symbolic(d) else d for d in shape]
        ub = [-1 if is_symbolic(d) else d for d in shape]
        set_multiarray_ndshape_range(model, input_name, lower_bounds=lb,
                upper_bounds=ub)
    return model
