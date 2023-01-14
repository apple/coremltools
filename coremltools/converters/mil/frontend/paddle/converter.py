#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict

import numpy as _np
import paddle as _paddle

from coremltools import _logger as logger
from coremltools._deps import version_lt
from coremltools.converters.mil._deployment_compatibility import \
    AvailableTarget as _target
from coremltools.converters.mil.input_types import InputType, TensorType
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types

from .._utils import get_output_names
from .internal_graph import InternalPaddleIRGraph
from .ops import convert_nodes
from .ssa_passes.paddle_passes import paddle_passes
from .paddle_op_registry import _PADDLE_OPS_REGISTRY
from .paddleir_passes import (fuse_conv_bias,
                             flatten_graph_input_values,
                             flatten_graph_output_values,
                             generate_tensor_assignment_ops,
                             remove_getattr_nodes, transform_inplace_ops)

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
    Class that handles conversion of pypaddle models represented in PaddleScript
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
            paddle_program: paddle.jit.ScriptModule object representing the model to convert.
            inputs: Input values and optional names. See kwarg in load.py for full description.
            outputs: List of outputs as ct.InputType. See kwarg in load.py for full description.
            cut_at_symbols: A list of internal symbol name strings. Graph conversion will
                terminate once these symbols have been generated. For debugging use
                only. See kwarg in load.py.
            opset_version: An int represents the Core ML opset version
        """
        # assert isinstance(paddle_program, _paddle.jit.TranslatedLayer)

       
        # for idx, inp in enumerate(self.inputs):
        #     if isinstance(inp, ImageType) and self.inputs[idx].channel_first is None:
        #         self.inputs[idx].channel_first = True


        # input_shape_dict = {k.name: k.shape.symbolic_shape for k in inputs}
        self.paddle_program = paddle_program
        # self.infer_shape(input_shape_dict)
        
        self.input_names = [k.name  for k in inputs]
        # input_vars = []
        # for name in self.input_names:
        #     input_vars.append(self.paddle_program.global_block().var(name))
        # self.inputs = self._convert_to_inputtype(input_vars)
        self.inputs = inputs

        self.outputs = outputs
        self.output_names = [var.name for var in self.outputs] 

        self.opset_version = _target(opset_version) if opset_version is not None else None
        self.context = TranscriptionContext()

        # raw_graph, params_dict = self._expand_and_optimize_ir(self.paddle_program)
        # self.params_dict = params_dict
        # self.graph = InternalPaddleIRGraph(
        #     raw_graph, params_dict, self.inputs, cut_at_symbols
        # )
        # self.graph = self.paddle_program.program()

        # Apply Paddle IR passes
        passes = [
            fuse_conv_bias
        ]
        for p in passes:
            p(self.paddle_program)

        # self.inputs = list(self.graph.inputs.values())
        self.paddle_passes = paddle_passes
        self._prog = Program()

    @staticmethod
    def _check_ops(graph):
        """
        Returns the set of ops in @graph that are implemented, and the set
        for which no conversion function is registered. @graph can be
        either InternalPaddleIRGraph or InternalPaddleIRBlock.
        """
        implemented_ops = set()
        missing_ops = set()
        for node in graph.nodes:
            _add_op = _PADDLE_OPS_REGISTRY.get(node.kind, None)
            if _add_op is None:
                missing_ops.add(node.kind)
            else:
                implemented_ops.add(node.kind)
            for block in node.blocks:
                _impl, _miss = PaddleConverter._check_ops(block)
                implemented_ops.update(_impl)
                missing_ops.update(_miss)
        return implemented_ops, missing_ops

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
                    raise ValueError("'shape' must be provided in the 'inputs' argument for pypaddle conversion")
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
    
    def infer_shape(self, input_shape_dict):
        import paddle
        paddle.enable_static()
        import paddle.fluid as fluid

        OP_WITHOUT_KERNEL_SET = {
            'feed', 'fetch', 'recurrent', 'go', 'rnn_memory_helper_grad',
            'conditional_block', 'while', 'send', 'recv', 'listen_and_serv',
            'fl_listen_and_serv', 'ncclInit', 'select', 'checkpoint_notify',
            'gen_bkcl_id', 'c_gen_bkcl_id', 'gen_nccl_id', 'c_gen_nccl_id',
            'c_comm_init', 'c_sync_calc_stream', 'c_sync_comm_stream',
            'queue_generator', 'dequeue', 'enqueue', 'heter_listen_and_serv',
            'c_wait_comm', 'c_wait_compute', 'c_gen_hccl_id', 'c_comm_init_hccl',
            'copy_cross_scope'
        }
        model_version = self.paddle_program.desc._version()
        paddle_version = paddle.__version__
        major_ver = model_version // 1000000
        minor_ver = (model_version - major_ver * 1000000) // 1000
        patch_ver = model_version - major_ver * 1000000 - minor_ver * 1000
        model_version = "{}.{}.{}".format(major_ver, minor_ver, patch_ver)
        if model_version != paddle_version:
            print(
                "[WARNING] The model is saved by paddlepaddle v{}, but now your paddlepaddle is version of {}, this difference may cause error, it is recommend you reinstall a same version of paddlepaddle for this model".
                format(model_version, paddle_version))
        for k, v in input_shape_dict.items():
            self.paddle_program.blocks[0].var(k).desc.set_shape(v)
        for i in range(len(self.paddle_program.blocks)):
            for j in range(len(self.paddle_program.blocks[0].ops)):
                if self.paddle_program.blocks[i].ops[j].type in OP_WITHOUT_KERNEL_SET:
                    continue
                self.paddle_program.blocks[i].ops[j].desc.infer_shape(self.paddle_program.blocks[i].desc)

    def check_ops(self):
        """
        Returns the set of ops in @self.graph that are implemented, and
        the set for which no conversion function is registered.
        """
        return PaddleConverter._check_ops(self.graph)

    def convert_const(self):
        vars = self.paddle_program.global_block().vars
       
        scope = _paddle.fluid.global_scope()
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

        # for index, (name, spec) in enumerate(self.graph.inputs.items()):
        #     placeholder = self._create_placeholder(spec)
        #     # Set ssa function input name to user defined name if provided.
        #     if spec.name is not None:
        #         name = spec.name
        #     self.inputs[index].name = name
        #     ssa_func_inputs[name] = placeholder
        # prog.set_main_input_types(tuple(self.inputs))

        # Initialize the SSA for conversion
        with Function(ssa_func_inputs, opset_version=self.opset_version) as ssa_func:

            # Map internal @self.graph.inputs to user specified @ssa_func_inputs
            # If @self.graph.inputs == @ssa_func_inputs this just adds the inputs
            # to the context.

            # for internal_name, users_name in zip(
            #     self.graph.inputs.keys(), ssa_func_inputs.keys()
            # ):
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

    def _jit_pass_lower_graph(graph, paddle_program):
        """
        This graph pass does a similar thing as _paddle._C._jit_pass_lower_graph does.
        It does two things:
        1. Rename getattr nodes which produce a paddle tensor to match the keys in paddle model's state_dict
        2. Construct the params_dict, with the keys similar to state_dict

        To be more specific, this graph pass traces down series of GetAttr ops, and rename the final node to match the paddle model state_dict.
        It also replaces the node inputs by the first created tensor node with the same name.

        Example:
        Input graph:
        graph(%self.1 : __paddle__.paddle.nn.modules.Sequential, %input.1 : Tensor):
        %2 : prim::GetAttr[name="linear"](%self.1)
        %3 : prim::GetAttr[name="weight"](%2)
        %4 : prim::GetAttr[name="bias"](%2)
        %5 : prim::GetAttr[name="bias"](%2) # duplicated node
        %6 : conv(%input.1, %3, %4)
        %7 : add(%input.1, %5)
        return (%6, %7)

        Output graph:
        graph(%self.1 : __paddle__.paddle.nn.modules.Sequential, %input.1 : Tensor):
        %2 : prim::GetAttr[name="linear"](%self.1)
        %linear.weight : prim::GetAttr[name="weight"](%2)
        %linear.bias : prim::GetAttr[name="bias"](%2)
        %5 : prim::GetAttr[name="bias"](%2) # duplicated node, it is not used now
        %6 : conv(%input.1, %linear.weight, %linear.bias)
        %7 : add(%input.1, %linear.bias) # the second input is replaced
        return (%6, %7)

        And a dictionary {"linear.weight": ..., "linear.bias": ...} is returned, to record the parameters values.
        Note that, those GetAttr nodes are still in the paddle ir graph, but they would be removed in a latter
        graph pass in the coremltools paddle internal graph

        """

        """
        Each getattr node corresponds to a paddle object in the paddle IR,
        it could be either:
        1. paddle.nn.modules: submodule in a paddle model. For instance, a linear layer in a MLP network.
        2. paddle.Tensor: paddle model parameters. For instance, weight for a conv layer.
        3. paddle._C.ScriptObject: quantized paddle model parameters.
        For example, in the graph above, %2 is pointing to the __paddle__.paddle.nn.modules.Sequential.linear paddle submodule.
        node_to_module_map tracks these mapping.

        node_to_prefic_map track the name for each module,
        for example, %2 has the prefix name linear and %3 is linear.weight.
        These names are also keys in the state_dict
        """
        node_to_module_map = {}
        node_to_prefix_map = {}
        first_node_with_prefix = {}
        replace_input = {}

        base_module_node = list(graph.inputs())[0]
        node_to_module_map[base_module_node] = paddle_program
        node_to_prefix_map[base_module_node] = ""

        """
        params_dict will be contructed in this graph pass. It contains all const tensors needed for the graph computation.
        And the value is validated against the state_dict if the key is presented in both dictionaries.
        In some rare cases, state_dict lacks parameters / buffers, so we still need to go through the while graph ourselves.
        """
        params_dict = {}
        state_dict = paddle_program.state_dict(keep_vars=True)

        def _check_is_tensor(node, module):
            if not isinstance(module, _paddle.Tensor):
                return False
            if str(node.output().type()) not in ("Tensor", "Optional[Tensor]"):
                raise TypeError("Type \"{}\" not supported".format(node.output().type()))
            return True

        def _check_is_quantized_tensor(node, module):
            if not isinstance(module, _paddle._C.ScriptObject):
                return False
            # There are three quantized parameters currently supported in Paddle:
            # ref: https://github.com/pypaddle/pypaddle/blob/master/paddle/csrc/jit/passes/lower_graph.cpp
            supported_quantizes_types = ["LinearPackedParamsBase", "Conv2dPackedParamsBase", "Conv3dPackedParamsBase"]
            assert node.output().type().name() in supported_quantizes_types
            return True

        def _get_tensor(module):
            return module

        def _get_quantized_tensor(module):
            return tuple(list(module.__getstate__())[:-1])

        def _lower_graph_block(graph):
            for node in list(graph.nodes()):

                for block in node.blocks():
                    _lower_graph_block(block)

                for idx, _input in enumerate(list(node.inputs())):
                    if _input in replace_input:
                        node.replaceInput(idx, replace_input[_input])

                kind = node.kind().split("::")[1].lower()
                if kind != "getattr":
                    continue

                _input = node.input()
                _output = node.output()
                attr_name = getattr(node, node.kindOf("name"))("name")

                module = getattr(node_to_module_map[_input], attr_name)
                node_to_module_map[_output] = module

                input_prefix = node_to_prefix_map[_input]
                prefix = input_prefix + '.' + attr_name if input_prefix != "" else attr_name
                node_to_prefix_map[_output] = prefix

                is_tensor = _check_is_tensor(node, module)
                is_quantized_tensor = _check_is_quantized_tensor(node, module)

                if is_tensor or is_quantized_tensor:
                    if is_tensor and prefix in state_dict:
                        assert _paddle.equal(module, state_dict[prefix]), "tensor value not consistent between paddle ir and state_dict"
                    if prefix in params_dict:
                        assert _paddle.equal(module, params_dict[prefix])
                        replace_input[_output] = first_node_with_prefix[prefix]
                    else:
                        params_dict[prefix] = _get_tensor(module) if is_tensor else _get_quantized_tensor(module)
                        first_node_with_prefix[prefix] = _output
                        _output.setDebugName(prefix)

        _lower_graph_block(graph)

        return graph, params_dict

    @staticmethod
    def _expand_and_optimize_ir(paddle_program):
        """
        Given a paddle.jit.ScriptModule, convert it to a optimized
        paddle._C.Graph and dict of model parameter's names to tensors.
        """
        graph = paddle_program.forward.graph

        # From PyPaddle code: Inline function and method calls.
        _paddle._C._jit_pass_inline(graph)
        # From PyPaddle code: This inlines the forked section in the fork()
        # callsite and replaces uses of the result of wait() calls with the
        # values produced from the (now-inlined) forked section.
        _paddle._C._jit_pass_inline_fork_wait(graph)
        # Starting from the return node, marks all nodes that feed into the
        # output, as well as nodes with side effects. Any nodes not marked are
        # eliminated.
        _paddle._C._jit_pass_dce(graph)
        # From PyPaddle code: checks well-formedness and invariants of graph.
        _paddle._C._jit_pass_lint(graph)
        # Replaces a couple specific ops patterns (add, sub, mul, div, chunk).
        if version_lt(_paddle, '1.6.0'):
            _paddle._C._jit_pass_canonicalize_ops(graph)
            _paddle._C._jit_pass_lint(graph)

            # From PyPaddle code: This pass catches all of the small, easy to catch
            # peephole optimizations you might be interested in doing.
            #     Eliminate no-op 'expand' nodes
            #     Simplify x.t().t() to x
            # pass disabled for v1.6.0 and onwards, wrongly captures the shape of dummy inputs during tracing.
            _paddle._C._jit_pass_peephole(graph, addmm_fusion_enabled=False)
        else:
            # v1.6.0 pass renamed
            _paddle._C._jit_pass_canonicalize_graph_fuser_ops(graph)
        _paddle._C._jit_pass_lint(graph)

        # From PyPaddle docs: Renumber the graph so that all structurally
        # equivalent graphs have same numbers.
        graph = _paddle._C._jit_pass_canonicalize(graph)
        _paddle._C._jit_pass_lint(graph)
        if version_lt(_paddle, '1.6.0'):
            # v1.6.0 JIT changes disallows pulling list values out of
            # prim::Constant. We can only pull scalar values. constant
            # propagation removes `listConstruct` and results in list values.
            # We disallow constant prop pass to keep them as scalars, and rely
            # on our own constant prop to interpret `listConstruct`.
            _paddle._C._jit_pass_constant_propagation(graph)
        # NOTE: Don't need another DCE, it's included in constant propagation.
        _paddle._C._jit_pass_lint(graph)

        # Get the params_dict and rename the getattr nodes in the graph
        graph, params_dict = PaddleConverter._jit_pass_lower_graph(graph, paddle_program)

        return graph, params_dict
