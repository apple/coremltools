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


class InternalTorchIRNode:
    """CoreML internal representation of a torch IR node. Can construct itself from a provided torchIR node or manually constructed with args for testing. See InternalTorchIRGraph for the motivation behind this structure.

        TODO: Support control flow by adding blocks
        rdar://60177478

        Arguments:
            node: The torch._C.Node to convert, or None.
            val: If @node is not specified, the value of the node.
            inputs: If @node is not specified, the list of input symbols.
            outputs: If @node is not specified, the list of output symbols.
            kind: If @node is not specified, the kind (op) of the node.
    """

    def __init__(
        self, node=None, val=None, inputs=None, outputs=None, kind=None,
    ):
        if node:
            try:
                const_type = node.kindOf("value")
                self.val = getattr(node, const_type)("value")
            except:
                self.val = None
            self.inputs = [_input.debugName() for _input in node.inputs()]
            self.outputs = [output.debugName() for output in node.outputs()]
            self.name = self.outputs[0]
            self.kind = node.kind().split("::")[1].lower()
            # If the output is boolean, explicitly cast it so type inference
            # will work correctly.
            if len(self.outputs) == 1 and next(node.outputs()).type().str() == "bool":
                self.val = bool(self.val)
        else:
            self.val = val
            self.inputs = inputs
            self.outputs = outputs
            self.name = self.outputs[0]
            self.kind = kind

    def __str__(self):
        node_str = "{} = {}".format(", ".join(_ssa_name_list(self.outputs)), self.kind)
        node_str += "[value={}]".format(self.val) if self.val else ""
        node_str += "({})".format(", ".join(_ssa_name_list(self.inputs)))
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
            input_values: A list of inputs to the graph.
    """

    def __init__(
        self, raw_graph, params_dict, input_values,
    ):
        self.nodes = []
        self.params = {}
        self.inputs = {}
        self.outputs = []

        # Add nodes
        for raw_node in raw_graph.nodes():
            self.nodes.append(InternalTorchIRNode(raw_node))

        # Add params
        for name, param in params_dict.items():
            value = param.detach().numpy()
            self.params[name] = value

        # Add inputs
        for index, _input in enumerate(islice(raw_graph.inputs(), len(input_values))):
            name = _input.debugName()
            value = input_values[index]
            self.inputs[name] = value

        # Add outputs
        for output in raw_graph.outputs():
            self.outputs.append(output.debugName())

    def __str__(self):
        graph_str = "graph(\n"
        graph_str += self._format_inputs(self.inputs)
        graph_str += self._format_inputs(self.params)
        graph_str += "):\n"
        graph_str += "\n".join(["  {}".format(x) for x in self.nodes]) + "\n"
        graph_str += "return ({})".format(", ".join(_ssa_name_list(self.outputs)))
        return graph_str

    def _format_inputs(self, inputs):
        inp_str = ""
        for k, v in inputs.items():
            inp_str += "    {} : {},\n".format(_make_ssa_name(k), list(v.shape))
        return inp_str

    def __repr__(self):
        return str(self)
