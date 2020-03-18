from __future__ import print_function

import logging

import torch

from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.nnv2_program.program import (Placeholder,
                                                              SsaFunction,
                                                              SsaProgram)
from coremltools.converters.nnv2.nnv2_program.program.var import Var

from .internal_graph import *
from .ops import *
from .torch_op_registry import _TORCH_OPS_REGISTRY

torch_to_proto_types = {torch.float32: builtins.float}


class TranscriptionContext:
    """Mantains a map from torch operations to their NNV2 values 
    while building the graph"""

    def __init__(self, name=None):
        self.name = name if name else ""
        self._current_graph = {}

    def add(self, ssa_var, torch_name=None):
        """
        Arguments:
            ssa_var: Varable to add to the graph being constructed.
            torch_name: Optional unique string identifier of the operation. If
                ommitted, it will use @ssa_var.name.
        """
        if torch_name is None:
            torch_name = ssa_var.name
        if torch_name in self._current_graph:
            print("Torch var {} is added again.".format(torch_name))
            return
        self._current_graph[torch_name] = ssa_var

    def __getitem__(self, torch_name):
        if torch_name not in self._current_graph:
            raise ValueError(
                "Torch var {} not found in context {}".format(torch_name, self.name)
            )
        return self._current_graph[torch_name]

    def __str__(self):
        _str = ""
        for k, v in self._current_graph.items():
            _str += "%{} : {}\n".format(k, v.shape_str())
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
        torchir: torch.jit.ScriptModule object representing the model to convert.
        input_values: A single torch.Tensor or list of torch.Tensor objects
            representing model inputs.
    """

    def __init__(
        self, torchir, input_values,
    ):
        assert isinstance(torchir, torch.jit.ScriptModule)
        if isinstance(input_values, torch.Tensor):
            input_values = [input_values]

        self.torchir = torchir
        self.input_values = input_values
        self.context = TranscriptionContext()
        raw_graph, params_dict = self._expand_and_optimize_ir(self.torchir)
        self.graph = InternalTorchIRGraph(raw_graph, params_dict, self.input_values)

    @staticmethod
    def _create_placeholder(_input):
        """Converts a torch.Tensor into a Placeholder.
        """
        shape = _input.shape
        dtype = torch_to_proto_types[_input.dtype]
        return cb.placeholder(shape, dtype=dtype)

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
            for node in self.graph.nodes:
                _add_op = _TORCH_OPS_REGISTRY.get(node.kind, None)
                logging.debug("Converting op {}".format(node.kind))
                if _add_op is None:
                    raise RuntimeError(
                        "Pytorch convert function for op {} not implemented".format(
                            node.kind
                        )
                    )
                else:
                    _add_op(self.context, node)

            graph_outputs = [self.context[name] for name in self.graph.outputs]
            ssa_func.set_outputs(graph_outputs)
            prog.add_function("main", ssa_func)

        # TODO (sberardi): graph cleanup passes
        # rdar://60177439
        return prog

    @staticmethod
    def _expand_and_optimize_ir(torchir):
        """Given a torch.jit.ScriptModule, convert it to a optimized
        torch._C.Graph and dict of model parameter's names to tensors.        
        """

        graph, params = torch._C._jit_pass_lower_graph(
            torchir.forward.graph, torchir._c
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
