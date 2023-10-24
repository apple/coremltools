#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict

import torch
from torch.fx.node import Node

from coremltools import _logger as logger

from .edgeir_utils import extract_inputs_from_edge_program
from .torchscript_utils import _expand_and_optimize_ir

_DEFAULT_OP_NAMESPACES = set(["aten", "prim"])


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


class InternalTorchIRBlock:
    """
    coremltools internal representation of a torch IR block.
    """

    def __init__(self, parent=None, nodes=None, inputs=None, outputs=None):
        """
        Arguments:
            parent: The InternalTorchIRNode this block belongs to.
            nodes: list of InternalTorchIRNodes in the block
            inputs: list of input symbols.
            outputs: list of output symbols.
        """

        self.nodes = nodes
        self.inputs = inputs
        self.outputs = outputs
        self.parent = parent

    @classmethod
    def from_edgeir_block(cls, block, parent):
        raise NotImplementedError(
            "EdgeIR: Support for Ops containing blocks not implemented yet"
        )  # TODO: rdar://115846569 ([Executorch] Handle control flow ops from edge ir)

    @classmethod
    def from_torchscript_block(cls, block, parent):

        node_names = set()
        nodes = []
        inputs = []
        outputs = []

        # Add inputs
        for inp in block.inputs():
            inputs.append(inp.debugName())

        # Add outputs
        for outp in block.outputs():
            outputs.append(outp.debugName())

        internal_block = cls(parent=parent, inputs=inputs, outputs=outputs, nodes=nodes)

        # Add nodes
        for raw_node in block.nodes():
            new_node = InternalTorchIRNode.from_torchscript_node(
                node=raw_node, parent=internal_block
            )
            if new_node.name == new_node.kind:
                new_node.name = _find_new_name(new_node.name, node_names)
            internal_block.nodes.append(new_node)
            node_names.add(new_node.name)

        return internal_block


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


class InternalTorchIRNode:
    """
    coremltools internal representation of a torch IR node.
    Can construct itself from a provided torchIR node or manually constructed with
    args for testing.

    See InternalTorchIRGraph for the motivation behind this structure.
    """

    def __init__(
        self,
        kind,
        inputs,
        outputs,
        name=None,
        parent=None,
        attr=None,
        blocks=None,
    ):
        """
        Arguments:
            name: Name of the node.
            kind: the kind (op) of the node.
            inputs: list of input symbols.
            outputs: list of output symbols.
            parent: The InternalTorchIRGraph/Block this node belongs to.
            attr:  dict of named attributes.
            blocks: list of InternalTorchIRBlock.
        """
        if not name and not outputs:
            self.name = ""
        else:
            self.name = name if name else outputs[0]

        self.kind = kind
        self.inputs = inputs
        self.outputs = outputs
        self.parent = parent
        self.attr = attr if attr is not None else {"value": None}
        self.blocks = blocks if blocks is not None else []

    @classmethod
    def from_torchscript_node(cls, node, parent):
        inputs = [_input.debugName() for _input in node.inputs()]
        outputs = [output.debugName() for output in node.outputs()]
        namespace = node.kind().split("::")[0].lower()
        if namespace in _DEFAULT_OP_NAMESPACES:
            # We conventionally skip the aten/prim namespaces in our naming.
            kind = node.kind().split("::")[-1].lower()
        else:
            kind = node.kind().lower()

        attr = {name: getattr(node, node.kindOf(name))(name) for name in node.attributeNames()}
        if "value" not in attr:
            attr["value"] = None
        # If the output is boolean, explicitly cast it so type inference
        # will work correctly.
        if len(outputs) == 1 and next(node.outputs()).type().str() == "bool":
            attr["value"] = bool(attr["value"])

        # On rare occassions, a node has no outputs. In that case, the node's
        # name will be its kind. However, this no longer guarantees the node's
        # name is unique. It will be up to the graph constructing the node to
        # make sure names are unique.
        name = outputs[0] if len(outputs) > 0 else kind

        internal_node = cls(
            name=name,
            kind=kind,
            parent=parent,
            inputs=inputs,
            outputs=outputs,
            attr=attr,
            blocks=None,
        )
        internal_node.blocks = [
            InternalTorchIRBlock.from_torchscript_block(block=b, parent=internal_node)
            for b in node.blocks()
        ]
        return internal_node

    @classmethod
    def from_edgeir_node(cls, node):
        def get_arguments(alist):
            args = []
            for i in alist:
                if isinstance(i, Node):
                    args.append(i.name)
                elif isinstance(i, torch.fx.immutable_collections.immutable_list):
                    args.append(get_arguments(i))
                elif isinstance(i, (int, float)):
                    args.append(i)
                elif i is None:
                    args.append(None)
                else:
                    raise AssertionError(f"Unhandled type of the node: {type(i)}")
            return tuple(args)

        inputs = get_arguments(node.args)
        outputs = [
            node.name
        ]  # TODO: rdar://115846125 ([Executorch] Handle Models/Layers with Multiple outputs)

        try:
            kind = node.target.name()
        except:
            if callable(node.target):
                kind = node.target.__name__
            else:
                kind = str(node.target)

        namespace = kind.split("::")[0].lower()
        if namespace in _DEFAULT_OP_NAMESPACES:
            # We conventionally skip the aten/prim namespaces in our naming.
            kind = kind.split("::")[-1].lower()
        else:
            kind = kind.lower()

        name = node.name
        return cls(
            name=name,
            kind=kind,
            inputs=inputs,
            outputs=outputs,
            parent=None,
            attr=None,
            blocks=None,
        )

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


class InternalTorchIRGraph:
    """
    CoreML internal representation of a torch IR graph. A torch._C.Graph
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
    """

    def __init__(
        self,
        params,
        inputs,
        outputs,
        nodes=None,
    ):
        """
        Arguments:
            params: dict mapping parameter names to their numpy value.
            inputs: OrderedDict mapping input names to their example values.
            outputs: list[str], list of outputs from the graph.
            nodes: list of InternalTorchIRNodes in the graph.
        """
        self.nodes = nodes
        self.params = params
        self.inputs = inputs
        self.outputs = outputs

    @classmethod
    def from_torchscript(cls, torchscript, input_values=None, cut_at_symbols=None):
        """
        Arguments:
            torchscript: TorchScript object representing the model to convert.
            input_values: A list of inputs to the graph. Must be given is
                @raw_graph if not None.
            cut_at_symbols: The list of desired outputs from the graph. Symbols
                must be present in the graph. For debugging use only. Can only
                be given if @raw_graph is not None.
        """
        if not isinstance(torchscript, torch.jit.ScriptModule):
            raise AssertionError(
                f"Input should be an object of type torch.jit.ScriptModule. Provide: {type(torchscript)}"
            )

        if hasattr(torchscript, "training") and torchscript.training:
            logger.warning(
                "Model is not in eval mode. "
                "Consider calling '.eval()' on your model prior to conversion"
            )
        if type(torchscript) == torch.jit._script.RecursiveScriptModule:
            logger.warning(
                "Support for converting Torch Script Models is experimental. "
                "If possible you should use a traced model for conversion."
            )

        nodes = []
        params = {}
        inputs = OrderedDict()
        outputs = []

        raw_graph, params_dict, buffer_dict = _expand_and_optimize_ir(torchscript)

        # Add params
        for name, param in params_dict.items():
            if isinstance(param, torch.Tensor):
                if param.is_quantized:
                    value = param
                else:
                    value = param.detach().cpu().numpy()
            else:
                value = param
            params[name] = value

        # Add inputs
        # The first element of the raw_graph.inputs() is the 'self' of the module, which is not used.
        graph_inputs = list(raw_graph.inputs())[1:]
        if len(graph_inputs) != len(input_values):
                raise ValueError(
                    f"Number of TorchScript inputs ({len(graph_inputs)}) must match the user provided inputs ({len(input_values)})."
                )
        for index, _input in enumerate(graph_inputs):
            name = _input.debugName()
            value = input_values[index]
            inputs[name] = value

        # Add outputs, cutting if @cut_at_symbols is set
        output_names = cut_at_symbols
        if output_names is None:
            output_names = [x.debugName() for x in raw_graph.outputs()]
        for output in output_names:
            outputs.append(output)

        internal_graph = cls(nodes=nodes, params=params, inputs=inputs, outputs=outputs)

        node_names = set()
        # Add nodes
        for raw_node in raw_graph.nodes():
            new_node = InternalTorchIRNode.from_torchscript_node(
                node=raw_node, parent=internal_graph
            )
            if new_node.name == new_node.kind:
                new_node.name = _find_new_name(new_node.name, node_names)
            internal_graph.nodes.append(new_node)
            node_names.add(new_node.name)

        return internal_graph, params_dict, buffer_dict

    @classmethod
    def from_edgeir(cls, edgeir):
        exported_program = edgeir

        nodes = []
        params = {}
        outputs = []
        inputs = OrderedDict(
            [
                (i.name, i)
                for i in extract_inputs_from_edge_program(exported_program=exported_program)
            ]
        )

        inputs_to_parameters = exported_program.graph_signature.inputs_to_parameters
        inputs_to_buffers = exported_program.graph_signature.inputs_to_buffers

        inputs_to_consts = {**inputs_to_parameters, **inputs_to_buffers}

        parameters_to_inputs = {
            v: k if not k.startswith("%") else k[1:] for k, v in inputs_to_consts.items()
        }

        # Add params
        for name, param in exported_program.state_dict.items():
            if isinstance(param, torch.Tensor):
                value = param.detach().cpu().numpy()
            else:
                raise NotImplementedError("Only torch.Tensor handled yet")

            params[name if name not in parameters_to_inputs else parameters_to_inputs[name]] = value

        graph = exported_program.graph

        outputs = []
        for node in graph.nodes:
            if node.op == "call_function":
                nodes.append(InternalTorchIRNode.from_edgeir_node(node=node))
            elif node.op == "placeholder":
                continue
            elif node.op == "output":
                outputs = [
                    node.name for node in node.args[0]
                ]  # TODO: rdar://115846125 ([Executorch] Handle Models/Layers with Multiple outputs)
            else:
                raise NotImplementedError(f"Nodes of type {node.op} not yet implemented")

        return cls(nodes=nodes, params=params, inputs=inputs, outputs=outputs)

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
            try:
                return "Tensor{}".format(
                    tuple(list(x.shape.shape if unpack else x.shape) + [str(x.dtype)])
                )
            except:

                return "Custom Params({})".format(type(x))

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
