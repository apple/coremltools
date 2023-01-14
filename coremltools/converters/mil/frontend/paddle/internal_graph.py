#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict
from itertools import islice


def _make_ssa_name(name):
    """
    Converts a symbol name (string) into an SSA name, by prepending '%'.
    Only used for pretty printing the graph.
    """
    if name is None:
        return "None"
    return "%" + name


def _ssa_name_list(names):
    """
    Take a list of symbol names (strings) and return them as SSA names. Only
    used for pretty printing the graph.
    """
    return [_make_ssa_name(x) for x in names]


def _find_new_name(old_name, node_names):
    """
    Disambiguate a node's name from a list of existing node names by adding
    successively larger integers.
    """
    count = 0
    new_name = old_name + "." + str(count) if count != 0 else old_name
    while new_name in node_names:
        count += 1
        new_name = old_name + "." + str(count)
    return new_name


def _replace_in_list(ls, old_val, new_val):
    """Helper function to replace a value in a list."""
    try:
        idx = ls.index(old_val)
    except ValueError:
        pass
    else:
        ls[idx] = new_val


class InternalPaddleIRBlock:
    """
    coremltools internal representation of a paddle IR block.
    """

    def __init__(self, raw_block=None, parent=None, nodes=None, inputs=None, outputs=None):
        """"
        Arguments:
            raw_block: The paddle._C.Block to convert, or None.
            parent: The InternalPaddleIRNode this block belongs to.
            nodes: If @raw_block is None, the list of InternalPaddleIRNodes in the block
            inputs: If @raw_block is None, the list of input symbols.
            outputs: If @raw_block is None, the list of output symbols.
        """

        self.nodes = []
        node_names = set()
        self.inputs = []
        self.outputs = []
        self.parent = parent

        if raw_block:
            # Add nodes
            for raw_node in raw_block.nodes():
                new_node = InternalPaddleIRNode(raw_node, parent=self)
                if new_node.name == new_node.kind:
                    new_node.name = _find_new_name(new_node.name, node_names)
                self.nodes.append(new_node)
                node_names.add(new_node.name)

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

    def replace_name(self, old_name, new_name):
        """Replaces all instances of @old_name with @new_name in @self."""

        # Replace graph inputs/outputs
        _replace_in_list(self.inputs, old_name, new_name)
        _replace_in_list(self.outputs, old_name, new_name)

        for node in self.nodes:
            node.replace_name(old_name, new_name)


class InternalPaddleIRNode:
    """
    coremltools internal representation of a paddle IR node.
    Can construct itself from a provided paddleIR node or manually constructed with
    args for testing.

    See InternalPaddleIRGraph for the motivation behind this structure.
    """

    def __init__(
        self, node=None, parent=None, attr=None, inputs=None, outputs=None, kind=None, blocks=None,
    ):
        """
        Arguments:
            node: The paddle._C.Node to convert, or None.
            parent: The InternalPaddleIRGraph/Block this node belongs to.
            attr: If @node is not specified, the dict of named attributes.
            inputs: If @node is not specified, the list of input symbols.
            outputs: If @node is not specified, the list of output symbols.
            kind: If @node is not specified, the kind (op) of the node.
            blocks: If @node is not specified, the list of InternalPaddleIRBlock.
        """

        self.parent = parent
        if node is not None:
            self.inputs = [_input.debugName() for _input in node.inputs()]
            self.outputs = [output.debugName() for output in node.outputs()]
            self.kind = node.kind().split("::")[-1].lower()
            self.blocks = [InternalPaddleIRBlock(raw_block=b, parent=self) for b in node.blocks()]
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
            self.kind = kind
            self.blocks = blocks if blocks is not None else []
            self.attr = attr if attr is not None else {"value": None}
        # On rare occassions, a node has no outputs. In that case, the node's
        # name will be its kind. However, this no longer guarantees the node's
        # name is unique. It will be up to the graph constructing the node to
        # make sure names are unique.
        self.name = self.outputs[0] if len(self.outputs) > 0 else self.kind

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

    def replace_name(self, old_name, new_name):
        """Replaces all instances of @old_name with @new_name in @self."""

        _replace_in_list(self.inputs, old_name, new_name)
        _replace_in_list(self.outputs, old_name, new_name)

        if self.name == old_name:
            self.name = new_name
        for block in self.blocks:
            block.replace_name(old_name, new_name)


class InternalPaddleIRGraph:
    """
    CoreML internal representation of a paddle IR graph. A paddle._C.Graph
    object is not an ideal structure to use in converting to CoreML. Conversion
    to an InternalPaddleIRGraph is inserted between the original graph and the
    final CoreML model to address several issues:
        1. A paddle._C.graph is hard to work with. For example, its .inputs()
          and .outputs() functions return iterators, so the only way to
          determine the number of inputs/outputs is by counting to the end.
          There are other examples of why the paddle structure is hard to work
          with, and this structure alleviates those isses.
        2. paddle._C.graph is an internal API and so we can't count on its
          stability. By inserting a layer in between, we can handle any changes
          to paddle._C.graph here and isolate the ops code that processes the
          graph.
        3. paddle._C.graph does not expose a Python constructor. This makes
          it impossible to write unit tests that isolate specific ops since
          they have to come from actually converting a PyPaddle graph. With an
          internal structure, we can directly build the test cases we need for
          unit testing.
    """

    def __init__(
            self, raw_graph=None, params_dict=None, input_values=None, cut_at_symbols=None,
            nodes=None, params=None, inputs=None, outputs=None,
    ):
        """
        Arguments:
            raw_graph: raw_graph: The paddle._C.Graph to convert, or None.
            params_dict: A dictionary mapping graph parameter names to tensors.
                Must be given if @raw_graph is not None.
            input_values: A list of inputs to the graph. Must be given is
                @raw_graph if not None.
            cut_at_symbols: The list of desired outputs from the graph. Symbols
                must be present in the graph. For debugging use only. Can only
                be given if @raw_graph is not None.
            nodes: If @raw_graph is None, the list of InternalPaddleIRNodes in
                the graph.
            params: If @raw_graph is None, the dict mapping parameter names to
                their numpy value.
            inputs: If @raw_graph is None, the OrderedDict mapping input names
                to their example values.
            outputs: list[str], If @raw_graph is None, the list of outputs from the graph.
        """

        self.nodes = []
        node_names = set()
        self.params = {}
        self.inputs = OrderedDict()
        self.outputs = []

        if raw_graph is not None:
            # Add nodes
            for raw_node in raw_graph.nodes():
                new_node = InternalPaddleIRNode(raw_node, parent=self)
                if new_node.name == new_node.kind:
                    new_node.name = _find_new_name(new_node.name, node_names)
                self.nodes.append(new_node)
                node_names.add(new_node.name)

            # Add params
            for name, param in params_dict.items():
                value = param.detach().cpu().numpy()
                self.params[name] = value

            # Add inputs
            # The first element of the raw_graph.inputs() is the 'self' of the module, which is not used.
            graph_inputs = list(raw_graph.inputs())[1:]
            for index, _input in enumerate(islice(graph_inputs, len(input_values))):
                name = _input.debugName()
                value = input_values[index]
                self.inputs[name] = value

            # Add outputs, cutting if @cut_at_symbols is set
            output_names = cut_at_symbols
            if output_names is None:
                output_names = [x.debugName() for x in raw_graph.outputs()]
            for output in output_names:
                self.outputs.append(output)
        else:
            self.nodes = nodes
            self.params = params
            self.inputs = inputs
            self.outputs = outputs

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

    def replace_name(self, old_name, new_name):
        """Replaces all instances of @old_name with @new_name in @self."""

        # Replace graph inputs/outputs
        _replace_in_list(self.inputs, old_name, new_name)
        _replace_in_list(self.outputs, old_name, new_name)

        for node in self.nodes:
            node.replace_name(old_name, new_name)
