#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict

import numpy as _np
import paddle as _paddle

from coremltools import _logger as logger
from coremltools.converters.mil._deployment_compatibility import \
    AvailableTarget as _target
from coremltools.converters.mil.input_types import InputType, TensorType
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types

from .ops import convert_nodes
from .ssa_passes.paddle_passes import paddle_passes
from .paddle_op_registry import _PADDLE_OPS_REGISTRY
from .paddleir_passes import (fuse_conv_bias)

paddle_to_mil_types = {
    _paddle.bool: types.bool,
    _paddle.float16: types.fp16,
    _paddle.float32: types.fp32,
    _paddle.float64: types.fp32,
    _paddle.int32: types.int32,
    _paddle.int64: types.int32,
}


mil_to_paddle_types = {v: k for k, v in paddle_to_mil_types.items()}


class TranscriptionContext:
    """
    Maintains a map from paddle operations to their MIL values
    while building the graph. Can be used to process subgraphs recursively
    by pushing new context when stepping into a subgraph and popping that
    context when stepping out.
    """

    def __init__(self, name=None):
        self.name = name if name else ""
        self._current_graph = [{}]

    def add(self, ssa_var, paddle_name=None):
        """
        Arguments:
            ssa_var: Variable to add to the graph being constructed.
            paddle_name: Optional unique string identifier of the operation. If
                omitted, it will use @ssa_var.name.
        """
        if paddle_name is None:
            paddle_name = ssa_var.name
        if paddle_name in self._current_graph[-1]:
            print("Paddle var {} is added again.".format(paddle_name))
            return
        self._current_graph[-1][paddle_name] = ssa_var

    def __getitem__(self, paddle_name):
        """
        Lookup a name in the context. Note that since nested blocks must be
        able to access anything that was defined before them, we have to
        search all contexts for a name, starting with the most local scope.
        """
        for idx in reversed(range(len(self._current_graph))):
            current_graph = self._current_graph[idx]
            if paddle_name in current_graph:
                return self._current_graph[idx][paddle_name]
        raise ValueError(
            "Paddle var {} not found in context {}".format(paddle_name, self.name)
        )

    def __contains__(self, paddle_name):
        """Returns whether or not the paddle var exist in context."""
        return paddle_name in self._current_graph[-1]

    def push(self, inputs=None):
        """
        Add another frame to the context. Optionally provide a tuple of
        (name list, Var list) to populate the new context frame.
        """
        self._current_graph.append({})

        if inputs is not None:
            if len(inputs[0]) != len(inputs[1]):
                raise ValueError("name list and Var list must be the same length")
            for name, var in zip(inputs[0], inputs[1]):
                self.add(var, paddle_name=name)

    def pop(self):
        """
        Remove and discard the top context frame.
        """
        self._current_graph = self._current_graph[:-1]

    def __str__(self):
        _str = ""
        for current_graph in reversed(self._current_graph):
            __str = ""
            for k, v in current_graph.items():
                if hasattr(v, "shape_str"):
                    shape_str = v.shape_str()
                elif hasattr(v, "sym_shape"):
                    shape_str = v.sym_shape()
                else:
                    shape_str = "None"
                __str += "%{} : {}\n".format(k, shape_str)
            _str += __str + "\n"
        return _str

    def __repr__(self):
        return str(self)


class PaddleConverter:
    """
    Class that handles conversion of paddle models represented in PaddlePaddle 'Program'
    format to the MIL format.

    Models passed to the @PaddleConverter go from:
    PaddleScript -> Expanded/Optimized Paddle IR -> Internal Graph -> CoreML SSA
    The internal graph representation was added to make testing easier.
    """

    def __init__(
        self, paddle_program, inputs, outputs=None, cut_at_symbols=None, opset_version=None
    ):
        """
        Arguments:
            paddle_program: paddle.static.Program object representing the model to convert.
            inputs: Input values and optional names. See kwarg in load.py for full description.
            outputs: List of outputs as ct.InputType. See kwarg in load.py for full description.
            cut_at_symbols: A list of internal symbol name strings. Graph conversion will
                terminate once these symbols have been generated. For debugging use
                only. See kwarg in load.py.
            opset_version: An int represents the Core ML opset version
        """
        assert isinstance(paddle_program, _paddle.static.Program)

        self.paddle_program = paddle_program
        
        self.input_names = [k.name  for k in inputs]
        self.inputs = inputs

        self.outputs = outputs
        self.output_names = [var.name for var in self.outputs] 

        self.opset_version = _target(opset_version) if opset_version is not None else None
        self.context = TranscriptionContext()

        # Apply Paddle IR passes
        passes = [
            fuse_conv_bias
        ]
        for p in passes:
            p(self.paddle_program)

        self.paddle_passes = paddle_passes
        self._prog = Program()

    @staticmethod
    def _create_placeholder(_input):
        """
        Converts an InputType into a Placeholder.

        _input: TensorType
        """
        shape = _input.shape.symbolic_shape
        dtype = _input.dtype
        return mb.placeholder(shape, dtype=dtype)
    
    def _convert_to_inputtype(self, inputs):
        input_type = []
        for _input in inputs:
            if isinstance(_input, (list, tuple)):
                input_type.append(self._convert_to_inputtype(_input))
            elif isinstance(_input, InputType):
                if _input.shape is None:
                    raise ValueError("'shape' must be provided in the 'inputs' argument for paddle conversion")
                input_type.append(_input)
            elif isinstance(_input, _paddle.static.Variable):
                input_type.append(
                    TensorType(
                        shape=_input.shape, dtype=paddle_to_mil_types[_input.dtype]
                    )
                )
            else:
                raise ValueError(
                    "Unknown type {} for conversion to InputType.".format(type(_input))
                )
        return input_type

    def convert_const(self):
        scope = _paddle.fluid.global_scope()
        vars = self.paddle_program.global_block().vars
        
        for name in vars:
            var = self.paddle_program.global_block().var(name)
            if name.endswith('feed') or name.endswith('fetch'):
                continue
            if not var.persistable:
                continue
            val = _np.array(scope.var(name).get_tensor())
            if val.dtype == _np.uint8 or val.dtype == _np.int64:
                val = val.astype(_np.int32)          
            const = mb.const(val=val, name=name)
            self.context.add(const)

    def convert(self):
        logger.info("Converting graph.")

        # This will hold the converted model.
        prog = self._prog

        # Construct placeholder for input to ssa function
        # This is where input renaming occurs
        ssa_func_inputs = OrderedDict()

        for index, (name, tensor) in enumerate(zip(self.input_names, self.inputs)):
            placeholder = self._create_placeholder(tensor)
            ssa_func_inputs[name] = placeholder
        prog.set_main_input_types(tuple(self.inputs))

        # Initialize the SSA for conversion
        with Function(ssa_func_inputs, opset_version=self.opset_version) as ssa_func:
            # Map internal @self.graph.inputs to user specified @ssa_func_inputs
            # If @self.graph.inputs == @ssa_func_inputs this just adds the inputs
            # to the context.

            for internal_name, users_name in zip(
                self.input_names, ssa_func_inputs.keys()
            ):
                input_var = ssa_func.inputs[users_name]
                if (types.is_tensor(input_var.sym_type) or types.is_scalar(input_var.sym_type)) \
                    and (input_var.dtype == types.fp16 or input_var.dtype == types.fp64):
                    # cast the input var to float32
                    # We need to do this because the type inference is very buggy when started from
                    # float16/float64 typed inputs. Until that is fixed in the following radar
                    # we cast all inputs of type float16/float64 to float32 as the first step.
                    # These casts will later get removed, if compute_precision=Float16 is
                    # provided, which will cause the FP16ComputePrecision pass to run.
                    # TODO: remove this when this radar is fixed: rdar://93731970
                    input_var = mb.cast(x=input_var, dtype="fp32")
                self.context.add(input_var, paddle_name=internal_name)

            self.convert_const()

            # Add the rest of the operations
            convert_nodes(self.context, self.paddle_program)

            graph_outputs = [self.context[name] for name in self.output_names]

            # An output can be None when it's a None constant, which happens
            # in Fairseq MT.
            for g in graph_outputs:
                if g is None:
                    msg = "Droping output {} which is None"
                    logger.warning(msg.format(g))
            graph_outputs = [g for g in graph_outputs if g is not None]

            # Output renaming occurs
            if self.outputs is not None:
                if len(self.outputs) != len(graph_outputs):
                    msg = "Number of outputs provided, {}, do not match the number of outputs detected in the model, {}."
                    raise ValueError(msg.format(
                        len(self.outputs),
                        len(graph_outputs),
                    ))
            if self.output_names:
                for index, var in enumerate(graph_outputs):
                    if self.output_names[index] is not None:
                        output_rename = self.output_names[index]
                        var.name = output_rename

            ssa_func.set_outputs(graph_outputs)
            prog.add_function("main", ssa_func)
            if self.outputs is not None:
                prog.set_main_output_types(self.outputs)
        self.paddle_passes(prog)
        return prog

