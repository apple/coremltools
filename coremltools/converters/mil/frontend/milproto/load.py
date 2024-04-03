#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os
from typing import Tuple

import numpy as np

from coremltools import _logger as logger
from coremltools import proto
from coremltools.converters.mil import mil
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as _target
from coremltools.converters.mil.backend.mil import helper
from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import (
    Function,
    ListVar,
    Placeholder,
    TupleInputType,
    Var,
    mil_list,
    types,
)
from coremltools.converters.mil.mil.block import curr_block
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry as _SSAOpRegistry

from .helper import proto_to_types

try:
    from coremltools.libmilstoragepython import _BlobStorageReader as BlobReader
except Exception as e:
    logger.warning(f"Fail to import BlobReader from libmilstoragepython. {e}")
    BlobReader = None


class TranscriptionContext:
    """
    Holds shared variables needed for transcription.
    """

    def __init__(self, weights_dir=""):
        self.name_to_var = {} # mapping from name -> var object
        self.blob_reader_from_filename = (
            {}
        )  # mapping from filename -> BlobReader object
        self.weights_dir = weights_dir

    def register_var_with_name(self, name, var):
        var.name = name
        if name in self.name_to_var:
            # Overriding allow us to translate control flow blocks
            msg = "Var %s is added again. Overriding previous value"
            logger.info(msg % name)
        self.name_to_var[name] = var

    def get_var_from_name(self, name):
        if name not in self.name_to_var:
            raise KeyError("Var {} not found".format(name))
        return self.name_to_var[name]


def _load_tensorvalue(tensorvalue_spec):
    if not isinstance(tensorvalue_spec, proto.MIL_pb2.TensorValue):
        raise TypeError("Invalid TensorValue spec object")

    if tensorvalue_spec.WhichOneof("value") == "floats":
        return tensorvalue_spec.floats.values
    elif tensorvalue_spec.WhichOneof("value") == "ints":
        return tensorvalue_spec.ints.values
    elif tensorvalue_spec.WhichOneof("value") == "bools":
        return tensorvalue_spec.bools.values
    elif tensorvalue_spec.WhichOneof("value") == "strings":
        return tensorvalue_spec.strings.values
    elif tensorvalue_spec.WhichOneof("value") == "longInts":
        return tensorvalue_spec.longInts.values
    elif tensorvalue_spec.WhichOneof("value") == "doubles":
        return tensorvalue_spec.doubles.values
    elif tensorvalue_spec.WhichOneof("value") == "bytes":
        return tensorvalue_spec.bytes.values
    else:
        raise ValueError("Invalid dtype for TensorValue type")


def _load_immediate_value(immediatevalue_spec):
    if not isinstance(immediatevalue_spec, proto.MIL_pb2.Value.ImmediateValue):
        raise TypeError("Invalid ImmedidateValue spec object")

    if immediatevalue_spec.WhichOneof("value") == "tensor":
        return _load_tensorvalue(immediatevalue_spec.tensor)
    elif immediatevalue_spec.WhichOneof("value") == "list":
        return immediatevalue_spec.list.values
    else:
        raise NotImplementedError(
            "Immediate value type not supported yet."
        )


def _load_file_value(context, filevalue_spec, dtype):
    if BlobReader is None:
        raise RuntimeError("BlobReader not loaded")
    if not isinstance(filevalue_spec, proto.MIL_pb2.Value.BlobFileValue):
        raise TypeError("Invalid BlobFileValue spec object")

    filename = os.path.join(context.weights_dir, filevalue_spec.fileName.split("/")[-1])
    offset = filevalue_spec.offset

    if filename in context.blob_reader_from_filename:
        blob_reader = context.blob_reader_from_filename[filename]
    else:
        blob_reader = BlobReader(filename)
        context.blob_reader_from_filename[filename] = blob_reader

    if dtype == types.uint8:
        np_value = blob_reader.read_uint8_data(offset)
    elif dtype == types.int8:
        np_value = blob_reader.read_int8_data(offset)
    elif dtype == types.uint16:
        np_value = blob_reader.read_uint16_data(offset)
    elif dtype == types.int16:
        np_value = blob_reader.read_int16_data(offset)
    elif dtype == types.fp16:
        np_value_uint16 = blob_reader.read_fp16_data(offset)
        np_value = np.frombuffer(np_value_uint16.tobytes(), np.float16)
    elif dtype == types.fp32:
        np_value = blob_reader.read_float_data(offset)
    else:
        raise ValueError("Invalid dtype for blob file value type")

    return np_value


def _restore_np_from_bytes_value(value: bytes, dtype: types, shape: Tuple[int]) -> np.ndarray:
    return np.frombuffer(value, types.nptype_from_builtin(dtype)).reshape(shape)


def _load_value(context, value_spec):
    if not isinstance(value_spec, proto.MIL_pb2.Value):
        raise TypeError("Invalid Value spec object")

    if value_spec.docString:
        raise ValueError("Docstring would get lost in the process.")

    value_spec_type = value_spec.type.WhichOneof("type")
    if value_spec.type.WhichOneof("type") == "tensorType":
        valuetype = proto_to_types(value_spec.type)

        is_tensor = types.is_tensor(valuetype)

        dtype = valuetype if not is_tensor else valuetype.get_primitive()
        shape = () if not is_tensor else valuetype.get_shape()

        if value_spec.WhichOneof("value") == "immediateValue":
            value = _load_immediate_value(value_spec.immediateValue)
        else:
            value = _load_file_value(context, value_spec.blobFileValue, dtype)

        target_np_dtype = types.nptype_from_builtin(dtype)
        if dtype in helper.IMMEDIATE_VALUE_TYPES_IN_BYTES:
            value = _restore_np_from_bytes_value(value, dtype, shape).astype(target_np_dtype)
        elif dtype == types.str and shape == ():
            value = str(value[0])
        elif dtype in (
            types.fp32,
            types.str,
            types.bool,
            types.int16,
            types.uint16,
            types.int32,
            types.int64,
        ):
            value = np.array(value).astype(target_np_dtype).reshape(shape)
        else:
            raise ValueError("Invalid dtype for tensor value")
    else:
        raise NotImplementedError("Only value of tensorType implemented yet")

    if not is_tensor and not isinstance(value, str):
        value = types.nptype_from_builtin(dtype)(value.item())

    return value


def _create_var_from_spec(spec):
    """
    This helper function is used for creating PyMIL Var/ListVar from the proto spec.
    Mainly used for the construction of the control flow ops.
    """
    assert isinstance(spec, proto.MIL_pb2.NamedValueType)
    sym_type = proto_to_types(spec.type)
    name = spec.name
    if types.is_list(sym_type):
        var = ListVar(
            name,
            elem_type=sym_type.T[0],
            init_length=sym_type.T[1],
            dynamic_length=sym_type.T[2])
    else:
        var = Var(name, sym_type, None, op=None, op_output_idx=None)
    return var

def _set_outer_op_for_nested_blocks(blocks, op):
    """
    An ultility function that sets the outer_op of the blocks for control flow ops.
    """
    for block in blocks:
        block.outer_op = op

def _create_nested_blocks(context, op_spec):
    """
    An utility function that creates nested blocks for control flow ops.
    """
    if not op_spec.blocks:
        return []

    blocks = []

    for block_spec in op_spec.blocks:
        input_vars = [_create_var_from_spec(input) for input in block_spec.inputs]

        # add block input vars to the context
        for v in input_vars:
            context.register_var_with_name(v.name, v)

        # In pymil, the outer_op for a block can only be None if the block is a Function.
        # As the result, we use a dummy outer_op here for block creation, and set it to
        # the legit op later on in _set_outer_op_for_nested_blocks
        dummy = mb.const(val=0.)
        with Block(block_inputs=input_vars, outer_op=dummy._op,
                   name=Block._get_new_name()) as block:
            _load_block(context, block_spec)

        blocks.append(block)

    return blocks

def _set_inputs_for_control_flow_op(inputs, blocks, op_type):
    """
    An utility function that set the dummy functional inputs and blocks inputs for
    control flow ops.
    """
    if op_type == "while_loop":
        def _dummy_cond(*loop_vars):
            return None

        def _dummy_body(*loop_vars):
            return None

        inputs["_existing_blocks"] = blocks
        inputs["_cond"] = _dummy_cond
        inputs["_body"] = _dummy_body

    elif op_type == "cond":
        def _dummy_true_fn(*loop_vars):
            return None
        def _dummy_false_fn(*loop_vars):
            return None

        inputs["_existing_blocks"] = blocks
        inputs["_true_fn"] = _dummy_true_fn
        inputs["_false_fn"] = _dummy_false_fn


def _load_const_op(context, op_spec):
    inputs = {k: _load_value(context, v) for k, v in op_spec.attributes.items()}
    if len(op_spec.inputs) > 0:
        for param_name, argument in op_spec.inputs.items():
            vars = []
            for binding in argument.arguments:
                binding_type = binding.WhichOneof("binding")
                if binding_type == "name":
                    vars.append(context.get_var_from_name(binding.name))
                elif binding_type == "value":
                    vars.append(_load_value(context, binding.value))
                else:
                    raise ValueError(f"Invalid binding_type {binding_type}")
            if len(vars) == 1:
                inputs[param_name] = vars[0]
            else:
                inputs[param_name] = vars

    output_var = getattr(mb, op_spec.type)(**inputs)

    if not isinstance(output_var, (tuple, list)):
        output_var = [output_var]
    if len(output_var) != len(op_spec.outputs):
        raise AssertionError(
            "Mismatch between number of outputs in operation specification vs PyMIL outputs"
        )
    for spec, var in zip(op_spec.outputs, output_var):
        context.register_var_with_name(spec.name, var)


def _load_operation(context: TranscriptionContext, op_spec: proto.MIL_pb2.Operation):
    if not isinstance(op_spec, proto.MIL_pb2.Operation):
        raise TypeError("Invalid Operation spec object")

    op_type = op_spec.type
    if op_type == "const" or "constexpr_" in op_type:
        if op_spec.blocks:
            raise ValueError("const / constexpr operation can't have any block")
        if op_type == "const" and op_spec.inputs:
            raise ValueError("const operation can't have any input")
        _load_const_op(context, op_spec)

    else:
        if op_type == "custom_layer":
            raise NotImplementedError(
                "Loading Custom Layer operation not yet implemented"
            )

        # The conversion steps of an operation proto -> PyMIL operation are as following:

        # (i)   Convert the input arguments:
        #       In most of the cases, the input variable is already created beforehand, hence we can
        #       directly access and get them through the TranscriptionContext.
        #       There are cases, though, the inputs are literal value. This could happens in the classify op spec.
        #       For that case, we directly create a constant variable.

        # (ii)  Create nested blocks for control flow operations:
        #       The Python functional input arguments for control flow ops cannot be recovered from milproto -> pymil conversion,
        #       for instance, the _body, _cond for mb.while_loop and _true_fn, _false_fn for mb.cond are not invertible
        #       Hence, here we directly create the nested blocks from the proto, and set them to mb.while_loop.blocks / mb.cond.blocks.
        #       Note that, when creating a block, PyMIL required an outer_op, which should be the control flow operation itself. However,
        #       in this approach we take, the outer_op hasn't been created at the time when the blocks produced. Here, we make a "dummy outer_op",
        #       which could pass the check in PyMIL, also it could provide enough information (such as visible variables in the blocks etc.)
        #       for the creation of the block.

        # (iii) Create PyMIL operation using inputs / blocks
        #       Note that for the control flow cases, we create dummy functional inputs, and use the existing block to create the op.

        # (iv)  Set the outer_op for control flow
        #       Once the operation is created, we replace the dummy outer_op with the legit one, to make it a valid PyMIL program

        attrs = list(op_spec.attributes.items())
        if len(attrs) > 0:
            if len(attrs) != 1 or attrs[0][0] != "name":
                raise ValueError("\"name\" is the only supported attribute for operation")
        inputs = {k: _load_value(context, v) for k, v in op_spec.attributes.items()}

        for param_name, argument in op_spec.inputs.items():
            vars = []
            for binding in argument.arguments:
                binding_type = binding.WhichOneof("binding")
                if binding_type == "name":
                    vars.append(context.get_var_from_name(binding.name))
                elif binding_type == "value":
                    # We only support the list value for now (for the classifier use case)
                    value_spec = binding.value
                    assert value_spec.WhichOneof("value") == "immediateValue"
                    assert value_spec.immediateValue.WhichOneof("value") == "list"
                    list_value = _load_immediate_value(value_spec.immediateValue)
                    values = []
                    for value_spec in list_value:
                        values.append(_load_value(context, value_spec))
                    var = mb.const(val=mil_list(values))
                    vars.append(var)
                else:
                    raise NotImplementedError("Binding {} not yet implemented".format(binding_type))
            op_cls = _SSAOpRegistry._get_core_op_cls(op_type)
            if len(vars) == 1 and not isinstance(
                op_cls.input_spec.input_types[param_name], TupleInputType
            ):
                inputs[param_name] = vars[0]
            else:
                inputs[param_name] = vars

        blocks = _create_nested_blocks(context, op_spec)
        _set_inputs_for_control_flow_op(inputs, blocks, op_type)

        output_var = getattr(mb, op_type)(**inputs)
        if not isinstance(output_var, (tuple, list)):
            output_var = [output_var]

        if len(output_var) != len(op_spec.outputs):
            raise AssertionError(
                "Mismatch between number of outputs in operation specification vs PyMIL outputs"
            )

        for spec, var in zip(op_spec.outputs, output_var):
            context.register_var_with_name(spec.name, var)

            pymil_type = var.sym_type
            proto_type = proto_to_types(spec.type)
            if not types.is_compatible_type(pymil_type, proto_type):
                # We allow a corner case where the pymil has an 0 rank tensor and the spec produces a scalar
                if types.is_tensor(pymil_type) and types.is_scalar(proto_type):
                    if pymil_type.get_primitive() == proto_type:
                        continue
                raise AssertionError(
                    "Mismatch between var types in specification vs PyMIL"
                )

        _set_outer_op_for_nested_blocks(blocks, output_var[0].op)


def _load_block(context, block_spec):
    if not isinstance(block_spec, proto.MIL_pb2.Block):
        raise TypeError("Invalid Block spec object")

    if block_spec.attributes:
        raise ValueError("Attributes on block not supported")

    block_outputs = block_spec.outputs
    output_vars = []
    for op_spec in block_spec.operations:
        _load_operation(context, op_spec)

    for proto_output_name in block_outputs:
        output_vars.append(context.get_var_from_name(proto_output_name))

    pymil_block = curr_block()
    pymil_block.set_outputs(output_vars)
    return pymil_block


def _load_function(context, func_spec, spec_version):
    if not isinstance(func_spec, proto.MIL_pb2.Function):
        raise TypeError("Invalid Function spec object")

    if func_spec.attributes:
        raise ValueError("Attributes on functions not supported")

    func_inputs = {}
    for named_value_type in func_spec.inputs:
        name = named_value_type.name
        valuetype = proto_to_types(named_value_type.type)

        if not types.is_tensor(valuetype):
            raise ValueError("Functions inputs can only be tensors")
        func_inputs[name] = Placeholder(
            sym_shape=valuetype.get_shape(), dtype=valuetype.get_primitive(), name=name
        )
        context.register_var_with_name(name, func_inputs[name].outputs[0])

    opset = func_spec.opset
    if opset not in func_spec.block_specializations:
        raise ValueError("Missing block specialization for opset {}".format(opset))

    with Function(func_inputs, opset_version=_target(spec_version)) as pymil_func:
        _load_block(context, func_spec.block_specializations[opset])

    return pymil_func


def load_mil_proto(program_spec, specification_version, file_weights_dir=""):
    """
    Load in-memory Proto specification of MILSpec.Program(.Proto) object to PyMIL
    """
    if not isinstance(program_spec, proto.MIL_pb2.Program):
        raise TypeError("Invalid Program spec object")

    if program_spec.docString:
        raise NotImplementedError("Docstring would be lost in the process")

    if program_spec.version != 1:
        raise ValueError("Invalid program version")

    context = TranscriptionContext(file_weights_dir)
    pymil_program = mil.Program()
    for func_name, func_spec in program_spec.functions.items():
        pymil_program.add_function(
            func_name, _load_function(context, func_spec, specification_version)
        )

    for attr_name, attr_spec in program_spec.attributes.items():
        if attr_name not in ("buildInfo",):
            raise ValueError(f"Invalid attribute {attr_name} for program")

    return pymil_program


def load(model_spec, specification_version, file_weights_dir="", **kwargs):
    """
    Load in-memory Proto specification of Model(.Proto) object to PyMIL

    Set force_spec_version to force override the spec version.
    """
    if not isinstance(model_spec, proto.Model_pb2.Model):
        raise TypeError("Invalid Model sepc object")

    if specification_version < model_spec.specificationVersion:
        if not kwargs.get("force_spec_version", False):
            raise ValueError(
                "specification_version must be greater or equal to the input model spec version"
            )

    if model_spec.WhichOneof("Type") != "mlProgram":
        raise ValueError("Only MIL proto based mlmodels can be loaded")

    return load_mil_proto(model_spec.mlProgram, specification_version, file_weights_dir)
