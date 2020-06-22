#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import print_function as _

from six import string_types as _string_types
import logging as _logging
import torch as _torch

from coremltools.converters.mil.input_types import InputType, ImageType
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import (
    Placeholder,
    Function,
    Program,
    get_new_symbol,
)
from coremltools.converters.mil.mil import Var

from .internal_graph import *
from .ops import *
from .torch_op_registry import _TORCH_OPS_REGISTRY

torch_to_mil_types = {
    _torch.float32: types.fp32,
    _torch.float64: types.fp64,
    _torch.int32: types.int32,
    _torch.int64: types.int64,
}

mil_to_torch_types = {v: k for k, v in torch_to_mil_types.items()}


class TranscriptionContext:
    """Maintains a map from torch operations to their MIL values
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
    format to the MIL format.

    Models passed to the @TorchConverter go from:
    TorchScript -> Expanded/Optimized Torch IR -> Internal Graph -> CoreML SSA
    The internal graph representation was added to make testing easier.

    Arguments:
        torchscript: torch.jit.ScriptModule object representing the model to convert.
        inputs: Input values and optional names. See kwarg in load.py for full description.
        outputs: Names of the graph's outputs. See kwarg in load.py for full description.
        cut_at_symbols: A list of internal symbol name strings. Graph conversion will
            terminate once these symbols have been generated. For debugging use
            only. See kwarg in load.py.
    """

    def __init__(
        self, torchscript, inputs, outputs=None, cut_at_symbols=None,
    ):
        assert isinstance(torchscript, _torch.jit.ScriptModule)
        self.inputs = inputs
        for idx, inp in enumerate(self.inputs):
            if isinstance(inp, ImageType) and self.inputs[idx].channel_first is None:
                self.inputs[idx].channel_first = True
        self.torchscript = torchscript
        self.output_names = outputs
        self.context = TranscriptionContext()
        raw_graph, params_dict = self._expand_and_optimize_ir(self.torchscript)
        self.graph = InternalTorchIRGraph(
            raw_graph, params_dict, self.inputs, cut_at_symbols
        )
        self._flatten_graph_input_values()
        self._flatten_graph_output_values()

    def _flatten_graph_input_values(self):
        """ CoreML can't handle nested iterables of tensors, so we flatten the
            inputs of any graph that expects them.
        """
        new_graph_inputs = self.graph.inputs
        all_new_nodes = []
        changed = True
        notified = False

        while changed:
            old_graph_inputs = new_graph_inputs
            new_graph_inputs = OrderedDict()
            new_nodes = []
            changed = False
            for _input_name, _input_val in old_graph_inputs.items():
                if isinstance(_input_val, (tuple, list)):
                    changed = True
                    if not notified:
                        notified = True
                        _logging.warning(
                            "Tuple detected at graph input. This will be flattened in the converted model."
                        )
                    # If this input to the graph is a tuple, we want to replace it
                    # with a flattened version and add an op to construct the tuple.
                    node_inputs = []
                    for idx, item in enumerate(_input_val):
                        name = _input_name + "_{}".format(idx)
                        new_graph_inputs[name] = item
                        node_inputs.append(name)
                    new_nodes.append(
                        InternalTorchIRNode(
                            inputs=node_inputs,
                            outputs=[_input_name],
                            kind="tupleconstruct",
                        )
                    )
                else:
                    # This input isn't a tuple, keep it as is.
                    new_graph_inputs[_input_name] = _input_val
            all_new_nodes = new_nodes + all_new_nodes
        self.graph.inputs = new_graph_inputs
        self.graph.nodes = all_new_nodes + self.graph.nodes
        self.inputs = [v for v in self.graph.inputs.values()]

    def _flatten_graph_output_values(self):
        """ CoreML can't handle nested iterables of tensors, so we flatten the
            outputs of any graph that produces them.
        """
        node_names = [node.name for node in self.graph.nodes]
        new_graph_outputs = self.graph.outputs
        changed = True
        notified = False

        while changed:
            old_graph_outputs = new_graph_outputs
            new_graph_outputs = []
            changed = False
            for outp in old_graph_outputs:
                # Find the node that generates this output var.
                # It is possible to not find the output var in the list of node
                # names since nodes are named after their first output. In that
                # case, it means the output var comes from a node that returns
                # multiple outputs, which means that node cannot be a construct op.
                try:
                    node_idx = node_names.index(outp)
                except:
                    # @outp doesn't come from a construct op
                    new_graph_outputs.append(outp)
                    continue
                if self.graph.nodes[node_idx].kind in [
                    "tupleconstruct",
                    "listconstruct",
                ]:
                    # Since this output came from a construct op, we can replace it
                    # with the inputs to the op.
                    new_graph_outputs.extend(self.graph.nodes[node_idx].inputs)
                    changed = True
                    if not notified:
                        notified = True
                        _logging.warning(
                            "Tuple detected at graph output. This will be flattened in the converted model."
                        )
                else:
                    new_graph_outputs.append(outp)
        # Note: if we flattened outputs, there are likely to be construct ops
        # that are no longer needed. These will be removed in a later DCE pass.
        self.graph.outputs = new_graph_outputs

    @staticmethod
    def _check_ops(graph):
        """ Returns the set of ops in @graph that are implemented, and the set
            for which no conversion function is registered. @graph can be
            either InternalTorchIRGraph or InternalTorchIRBlock."""
        implemented_ops = set()
        missing_ops = set()
        for node in graph.nodes:
            _add_op = _TORCH_OPS_REGISTRY.get(node.kind, None)
            if _add_op is None:
                missing_ops.add(node.kind)
            else:
                implemented_ops.add(node.kind)
            for block in node.blocks:
                _impl, _miss = TorchConverter._check_ops(block)
                implemented_ops.update(_impl)
                missing_ops.update(_miss)
        return implemented_ops, missing_ops

    @staticmethod
    def _create_placeholder(_input):
        """Converts an InputType torch.Tensor into a Placeholder.
        """
        shape = _input.shape.shape
        dtype = _input.dtype
        return mb.placeholder(shape, dtype=dtype)

    def check_ops(self):
        """ Returns the set of ops in @self.graph that are implemented, and
            the set for which no conversion function is registered."""
        return TorchConverter._check_ops(self.graph)

    def convert(self):

        _logging.info("Converting graph.")

        # This will hold the converted model.
        prog = Program()

        # Construct placeholder for input to ssa function
        # This is where input renaming occurs
        ssa_func_inputs = OrderedDict()
        for index, (name, spec) in enumerate(self.graph.inputs.items()):
            placeholder = self._create_placeholder(spec)
            # Set ssa function input name to user defined name if provided.
            if spec.name is not None:
                name = spec.name
            self.inputs[index].name = name
            ssa_func_inputs[name] = placeholder
        prog.set_main_input_types(tuple(self.inputs))

        # Initialize the SSA for conversion
        with Function(ssa_func_inputs) as ssa_func:

            # Map internal @self.graph.inputs to user specified @ssa_func_inputs
            # If @self.graph.inputs == @ssa_func_inputs this just adds the inputs
            # to the context.
            for internal_name, users_name in zip(
                self.graph.inputs.keys(), ssa_func_inputs.keys()
            ):
                self.context.add(ssa_func.inputs[users_name], torch_name=internal_name)
            for name, val in self.graph.params.items():
                mode = decide_immediate_or_file(val)
                const = mb.const(val=val, mode=mode, name=name)
                self.context.add(const)

            # Add the rest of the operations
            convert_nodes(self.context, self.graph)

            graph_outputs = [self.context[name] for name in self.graph.outputs]
            # Output renaming occurs
            if self.output_names:
                for index, var in enumerate(graph_outputs):
                    output_rename = self.output_names[index]
                    var.name = output_rename

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

        # Recursively replaces all attribute accesses with the sub-graphs of
        # those modules. The resulting graph will be self-contained and will
        # not reference into other modules. Params will contain the "trainable"
        # inputs to the graph.
        graph, params = _torch._C._jit_pass_lower_graph(
            torchscript.forward.graph, torchscript._c
        )

        # From PyTorch code: Inline function and method calls.
        _torch._C._jit_pass_inline(graph)
        # From PyTorch code: This inlines the forked section in the fork()
        # callsite and replaces uses of the result of wait() calls with the
        # values produced from the (now-inlined) forked section.
        _torch._C._jit_pass_inline_fork_wait(graph)
        # Starting from the return node, marks all nodes that feed into the
        # output, as well as nodes with side effects. Any nodes not marked are
        # eliminated.
        _torch._C._jit_pass_dce(graph)
        # From PyTorch code: checks well-formedness and invariants of graph.
        _torch._C._jit_pass_lint(graph)
        # From PyTorch code: remove all in-place ops and replace them with
        # out-of-place equivalents.
        # e.g.
        #   %foo = aten::add_(%foo, %n)
        # becomes
        #   %foo.2 = aten::add(%foo, %n)
        _torch._C._jit_pass_remove_inplace_ops(graph)
        _torch._C._jit_pass_dce(graph)
        _torch._C._jit_pass_lint(graph)
        # Replaces a couple specific ops patterns (add, sub, mul, div, chunk).
        _torch._C._jit_pass_canonicalize_ops(graph)
        _torch._C._jit_pass_lint(graph)
        # From PyTorch code: This pass catches all of the small, easy to catch
        # peephole optimizations you might be interested in doing.
        #     Eliminate no-op 'expand' nodes
        #     Simplify x.t().t() to x
        _torch._C._jit_pass_peephole(graph, addmm_fusion_enabled=False)
        _torch._C._jit_pass_lint(graph)
        # From PyTorch docs: Renumber the graph so that all structurally
        # equivalent graphs have same numbers.
        graph = _torch._C._jit_pass_canonicalize(graph)
        _torch._C._jit_pass_lint(graph)
        _torch._C._jit_pass_constant_propagation(graph)
        # NOTE: Don't need another DCE, it's included in constant propagation.
        _torch._C._jit_pass_lint(graph)

        input_and_param_names = [val.debugName() for val in graph.inputs()]
        param_names = input_and_param_names[len(input_and_param_names) - len(params) :]
        params_dict = dict(zip(param_names, params))

        return graph, params_dict
