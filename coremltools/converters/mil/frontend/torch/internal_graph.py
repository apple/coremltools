#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict
from itertools import islice

import torch


def _make_ssa_name(name):
    """Converts a symbol name (string) into an SSA name, by prepending '%'.
    Only used for pretty printing the graph.
    """
    return "%" + name


def _ssa_name_list(names):
    """Take a list of symbol names (strings) and return them as SSA names. Only
    used for pretty printing the graph.
    """
    return [_make_ssa_name(x) for x in names]


class InternalTorchIRBlock:
    """CoreML internal representation of a torch IR block.

        Arguments:
            raw_block: The torch._C.Block to convert, or None.
            nodes: If @raw_block is None, the list of InternalTorchIRNodes in the block
            inputs: If @raw_block is None, the list of input symbols.
            outputs: If @raw_block is None, the list of output symbols.
    """

    def __init__(self, raw_block=None, nodes=None, inputs=None, outputs=None):
        self.nodes = []
        self.inputs = []
        self.outputs = []

        if raw_block:
            # Add nodes
            for raw_node in raw_block.nodes():
                self.nodes.append(InternalTorchIRNode(raw_node))

            # Add inputs
            for inp in raw_block.inputs():
                self.inputs.append(inp.debugName())

            # Add outputs
            for outp in raw_block.outputs():
                self.outputs.append(outp.debugName())
        else:
            self.nodes = nodes
            self.inputs = inputs
            self.outputs = outputs

    def __str__(self, indent=2):
        indent_str = " " * indent
        graph_str = "{}block({}):\n".format(
            indent_str, ", ".join(_ssa_name_list(self.inputs))
        )
        graph_str += "{}\n".format(indent_str).join(
            [x.__str__(indent=indent + 2) for x in self.nodes]
        )
        graph_str += "\n{}return ({})".format(
            indent_str, ", ".join(_ssa_name_list(self.outputs))
        )
        return graph_str

    def __repr__(self):
        return str(self)


class InternalTorchIRNode:
    """CoreML internal representation of a torch IR node.
    Can construct itself from a provided torchIR node or manually constructed with
    args for testing.

    See InternalTorchIRGraph for the motivation behind this structure.

        Arguments:
            node: The torch._C.Node to convert, or None.
            attr: If @node is not specified, the dict of named attributes.
            inputs: If @node is not specified, the list of input symbols.
            outputs: If @node is not specified, the list of output symbols.
            kind: If @node is not specified, the kind (op) of the node.
            blocks: If @node is not specified, the list of InternalTorchIRBlock.
    """

    def __init__(
        self, node=None, attr=None, inputs=None, outputs=None, kind=None, blocks=None,
    ):
        if node:
            self.inputs = [_input.debugName() for _input in node.inputs()]
            self.outputs = [output.debugName() for output in node.outputs()]
            self.name = self.outputs[0]
            self.kind = node.kind().split("::")[-1].lower()
            self.blocks = [InternalTorchIRBlock(raw_block=b) for b in node.blocks()]
            self.attr = {
                name: getattr(node, node.kindOf(name))(name)
                for name in node.attributeNames()
            }
            if "value" not in self.attr:
                self.attr["value"] = None
            # If the output is boolean, explicitly cast it so type inference
            # will work correctly.
            if len(self.outputs) == 1 and next(node.outputs()).type().str() == "bool":
                self.attr["value"] = bool(self.attr["value"])
        else:
            self.inputs = inputs
            self.outputs = outputs
            self.name = self.outputs[0]
            self.kind = kind
            self.blocks = blocks if blocks is not None else []
            self.attr = attr if attr is not None else {"value": None}

    def __str__(self, indent=2):
        node_str = " " * indent + "{} = {}".format(
            ", ".join(_ssa_name_list(self.outputs)), self.kind
        )
        node_str += "[{}]".format(
            ", ".join(
                ["{}={}".format(n, v) for n, v in self.attr.items() if v is not None]
            )
        )
        node_str += "({})".format(", ".join(_ssa_name_list(self.inputs)))
        for b in self.blocks:
            node_str += "\n" + b.__str__(indent=indent + 2)
        return node_str

    def __repr__(self):
        return str(self)


class InternalTorchIRGraph:
    """CoreML internal representation of a torch IR graph. A torch._C.Graph
    object is not an ideal structure to use in converting to CoreML. Conversion
    to an InternalTorchIRGraph is inserted between the original graph and the
    final CoreML model to address several issues:
        1. A torch._C.graph is hard to work with. For example, its .inputs()
          and .outputs() functions return iterators, so the only way to
          determine the number of inputs/outputs is by counting to the end.
          There are other examples of why the torch structure is hard to work
          with, and this structure alleviates those isses.
        2. torch._C.graph is an internal API and so we can't count on its
          stability. By inserting a layer in between, we can handle any changes
          to torch._C.graph here and isolate the ops code that processes the
          graph.
        3. torch._C.graph does not expose a Python constructor. This makes
          it impossible to write unit tests that isolate specific ops since
          they have to come from actually converting a PyTorch graph. With an
          internal structure, we can directly build the test cases we need for
          unit testing.

        Arguments:
            raw_graph: The torch._C.Graph to convert.
            params_dict: A dictionary mapping graph parameter names to tensors.
            input_spec: A list of InputType objects, describing the name,
                shape, and dtype of graph inputs.
            cut_at_symbols: The list of desired outputs from the graph. Must
                be present in the graph. For debugging use only.
                See kwarg in load.py for more information.
    """

    def __init__(
        self, raw_graph, params_dict, input_spec, cut_at_symbols=None,
    ):
        self.nodes = []
        self.params = {}
        self.inputs = OrderedDict()
        self.outputs = []

        # Add nodes
        for raw_node in raw_graph.nodes():
            self.nodes.append(InternalTorchIRNode(raw_node))

        # Add params
        for name, param in params_dict.items():
            value = param.detach().numpy()
            self.params[name] = value

        # Add inputs
        for index, _input in enumerate(islice(raw_graph.inputs(), len(input_spec))):
            name = _input.debugName()
            spec = input_spec[index]
            self.inputs[name] = spec

        # Add outputs, cutting if @cut_at_symbols is set
        output_names = cut_at_symbols
        if output_names is None:
            output_names = [x.debugName() for x in raw_graph.outputs()]
        for output in output_names:
            self.outputs.append(output)

    def __str__(self):
        graph_str = "graph(\n"
        graph_str += self._format_inputs(self.inputs, unpack=True)
        graph_str += self._format_inputs(self.params)
        graph_str += "):\n"
        graph_str += "\n".join([str(x) for x in self.nodes]) + "\n"
        graph_str += "return ({})".format(", ".join(_ssa_name_list(self.outputs)))
        return graph_str

    def _format_inputs(self, inputs, unpack=False):
        def tensor_str(x):
            return "Tensor{}".format(
                tuple(list(x.shape.shape if unpack else x.shape) + [str(x.dtype)])
            )

        inp_str = ""
        for k, v in inputs.items():
            if isinstance(v, (tuple, list)):
                shape_str = "({})".format(", ".join([tensor_str(x) for x in v]))
            else:
                shape_str = tensor_str(v)
            inp_str += "    {} : {},\n".format(_make_ssa_name(k), shape_str)
        return inp_str

    def __repr__(self):
        return str(self)
