from __future__ import print_function

import logging

import torch

from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.program import (
    Placeholder,
    SsaFunction,
    SsaProgram,
)
from coremltools.converters.nnv2.nnv2_program.program.var import Var

from .internal_graph import *
from .ops import *
from .torch_op_registry import _TORCH_OPS_REGISTRY

torch_to_proto_types = {
    torch.float32: builtins.fp32,
    torch.int32: builtins.int32,
    torch.int64: builtins.int64,
}


class TranscriptionContext:
    """Maintains a map from torch operations to their NNV2 values
        while building the graph. Can be used to process subgraphs recursively
        by pushing new context when stepping into a subgraph and popping that
        context when stepping out."""

    def __init__(self, name=None):
        self.name = name if name else ""
        self._current_graph = [{}]

    def add(self, ssa_var, torch_name=None):
        """
        Arguments:
            ssa_var: Varable to add to the graph being constructed.
            torch_name: Optional unique string identifier of the operation. If
                ommitted, it will use @ssa_var.name.
        """
        if torch_name is None:
            torch_name = ssa_var.name
        if torch_name in self._current_graph[-1]:
            print("Torch var {} is added again.".format(torch_name))
            return
        self._current_graph[-1][torch_name] = ssa_var

    def __getitem__(self, torch_name):
        """ Lookup a name in the context. Note that since nested blocks must be
            able to access anything that was defined before them, we have to
            search all contexts for a name, starting with the most local scope.
        """
        for idx in reversed(range(len(self._current_graph))):
            current_graph = self._current_graph[idx]
            if torch_name in current_graph:
                return self._current_graph[idx][torch_name]
        raise ValueError(
            "Torch var {} not found in context {}".format(torch_name, self.name)
        )

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
                self.add(var, torch_name=name)

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


class TorchConverter:
    """Class that handles conversion of pytorch models represented in TorchScript
    format to the NNv2 format.

    Models passed to the @TorchConverter go from: 
    TorchScript -> Expanded/Optimized Torch IR -> Internal Graph -> CoreML SSA
    The internal graph representation was added to make testing easier. 

    Arguments:
        torchscript: torch.jit.ScriptModule object representing the model to convert.
        input_values: A single torch.Tensor or list of torch.Tensor objects
            representing model inputs.
    """

    def __init__(
        self, torchscript, input_values,
    ):
        assert isinstance(torchscript, torch.jit.ScriptModule)
        if isinstance(input_values, torch.Tensor):
            input_values = [input_values]

        self.torchscript = torchscript
        self.input_values = input_values
        self.context = TranscriptionContext()
        raw_graph, params_dict = self._expand_and_optimize_ir(self.torchscript)
        self.graph = InternalTorchIRGraph(raw_graph, params_dict, self.input_values)

    @staticmethod
    def _create_placeholder(_input):
        """Converts a torch.Tensor into a Placeholder.
        """
        shape = _input.shape
        dtype = torch_to_proto_types[_input.dtype]
        return cb.placeholder(shape, dtype=dtype)

    def check_ops(self):
        """Returns the set of ops in @self.graph that are implemented, and
            the set for which no conversion function is registered."""
        implemented_ops = set()
        missing_ops = set()
        for node in self.graph.nodes:
            _add_op = _TORCH_OPS_REGISTRY.get(node.kind, None)
            if _add_op is None:
                missing_ops.add(node.kind)
            else:
                implemented_ops.add(node.kind)
        return implemented_ops, missing_ops

    def convert(self):

        logging.info("Converting graph.")

        # This will hold the converted model.
        prog = SsaProgram()

        graph_inputs = {
            name: TorchConverter._create_placeholder(value)
            for (name, value) in self.graph.inputs.items()
        }
        # Initialize the SSA for conversion
        with SsaFunction(graph_inputs) as ssa_func:

            # Add inputs and params to the context
            for name in self.graph.inputs.keys():
                self.context.add(ssa_func.inputs[name])
            for name, val in self.graph.params.items():
                mode = decide_immediate_or_file(val)
                const = cb.const(val=val, mode=mode, name=name)
                self.context.add(const)

            # Add the rest of the operations
            convert_nodes(self.context, self.graph)

            graph_outputs = [self.context[name] for name in self.graph.outputs]
            ssa_func.set_outputs(graph_outputs)
            prog.add_function("main", ssa_func)

        # TODO (sberardi): graph cleanup passes
        # rdar://60177439
        return prog

    @staticmethod
    def _expand_and_optimize_ir(torchscript):
        """Given a torch.jit.ScriptModule, convert it to a optimized
        torch._C.Graph and dict of model parameter's names to tensors.        
        """

        graph, params = torch._C._jit_pass_lower_graph(
            torchscript.forward.graph, torchscript._c
        )

        torch._C._jit_pass_inline(graph)
        torch._C._jit_pass_inline_fork_wait(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_remove_inplace_ops(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_canonicalize_ops(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_peephole(graph, True)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
        torch._C._jit_pass_lint(graph)
        graph = torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_constant_propagation(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)

        input_and_param_names = [val.debugName() for val in graph.inputs()]
        param_names = input_and_param_names[len(input_and_param_names) - len(params) :]
        params_dict = dict(zip(param_names, params))

        return graph, params_dict
