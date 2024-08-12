#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from coremltools import _logger as logger
from coremltools.converters.mil.input_types import TensorType

from .exir_utils import extract_io_from_exir_program
from .torchscript_utils import _expand_and_optimize_ir
from .utils import TORCH_DTYPE_TO_NUM, sanitize_op_kind


def _make_ssa_name(name: str) -> str:
    """
    Converts a symbol name (string) into an SSA name, by prepending '%'.
    Only used for pretty printing the graph.
    """
    if name is None:
        return "None"
    return "%" + name


def _ssa_name_list(names: List[str]) -> List[str]:
    """
    Take a list of symbol names (strings) and return them as SSA names. Only
    used for pretty printing the graph.
    """
    return [_make_ssa_name(x) for x in names]


def _find_new_name(old_name: str, node_names: List[str]) -> str:
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


def _replace_in_list(ls: List[Any], old_val: Any, new_val: Any) -> None:
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

    def __init__(
        self,
        parent: Optional["InternalTorchIRNode"] = None,
        nodes: Optional[List["InternalTorchIRNode"]] = None,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
    ):
        """
        Arguments:
            parent: The InternalTorchIRNode this block belongs to.
            nodes: list of InternalTorchIRNode in the block
            inputs: list of input symbols.
            outputs: list of output symbols.
        """

        self.nodes = nodes
        self.inputs = inputs
        self.outputs = outputs
        self.parent = parent

    @classmethod
    def from_exir_block(cls, block, parent):
        raise NotImplementedError(
            "EXIR: Support for Ops containing blocks not implemented yet"
        )  # TODO: rdar://115846569 ([Executorch] Handle control flow ops from EXIR)

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
        kind: str,
        inputs: List[str],
        outputs: List[str],
        name: Optional[str] = None,
        parent: Optional[Union["InternalTorchIRGraph", "InternalTorchIRBlock"]] = None,
        attr: Optional[Dict[str, Any]] = None,
        blocks: Optional[List["InternalTorchIRBlock"]] = None,
        model_hierarchy: Optional[str] = None,
        meta: Optional[Dict] = None,
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
            model_hierarchy: str represents TorchScript node's model hierarchy.
            meta: A dictionary of torch fx node metadata inherited from torch.fx.Node.meta
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
        self.model_hierarchy = model_hierarchy
        self.meta = meta

    @classmethod
    def from_torchscript_node(cls, node, parent):
        inputs = [_input.debugName() for _input in node.inputs()]
        outputs = [output.debugName() for output in node.outputs()]
        kind = sanitize_op_kind(node.kind())

        attr = {name: getattr(node, node.kindOf(name))(name) for name in node.attributeNames()}
        if "value" not in attr:
            attr["value"] = None
        # If the output is boolean, explicitly cast it so type inference
        # will work correctly.
        if len(outputs) == 1 and next(node.outputs()).type().str() == "bool":
            attr["value"] = bool(attr["value"])

        # On rare occassions, a node has no outputs. In that case, the node's
        # On rare occasions, a node has no outputs. In that case, the node's
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
            model_hierarchy=node.getModuleHierarchy(),
        )
        internal_node.blocks = [
            InternalTorchIRBlock.from_torchscript_block(block=b, parent=internal_node)
            for b in node.blocks()
        ]
        return internal_node

    @classmethod
    def from_exir_node(cls, node):
        def get_arguments(alist):
            args = []
            for i in alist:
                if isinstance(i, torch.fx.Node):
                    args.append(i.name)
                elif isinstance(i, torch.fx.immutable_collections.immutable_list):
                    args.append(get_arguments(i))
                elif isinstance(i, (int, float)):
                    args.append(i)
                # This is necessitated by backward compatibility:
                # * TorchScript used to store dtype as integers/enums
                # * Subsequently, we built our PyTorch converter based on numbered dtypes
                # * Now EXIR uses dtype directly...
                # * Until refactoring EXIR converter to be independent from TorchScript converter,
                #   we have to map dtype to number ourselves
                #   to leverage the existing TorchScript converter infra
                elif isinstance(i, torch.dtype):
                    args.append(TORCH_DTYPE_TO_NUM[i])
                elif i is None:
                    args.append(None)
                else:
                    raise AssertionError(f"Unhandled type of the node: {type(i)}")
            return tuple(args)

        # TODO (rdar://128768037) handle kwargs
        inputs = get_arguments(node.args)
        # TODO: rdar://115846125 ([Executorch] Handle Models/Layers with Multiple outputs)
        outputs = [node.name]

        try:
            kind = node.target.name()
        except:
            if callable(node.target):
                kind = node.target.__name__
            else:
                kind = str(node.target)
        kind = sanitize_op_kind(kind)

        name = node.name
        return cls(
            name=name,
            kind=kind,
            inputs=inputs,
            outputs=outputs,
            parent=None,
            attr=None,
            blocks=None,
            meta=node.meta,
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

    def get_scope_info(self) -> Tuple[List[str], List[str]]:
        """
        Get the scope information (``scope_name``, ``scope_type``) of a TorchScript node.
        In a TorchScript node, a model hierarchy is represented in a string of format:
            ``scope_name_1(scope_type_1).scope_name_2(scope_type_1).<...>.scope_name_n(scope_type_n)``
        For instance, given a torch model:

            class SubModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear_1 = torch.nn.Linear(2, 3)

                def forward(self, x):
                    x_1 = self.linear(x)
                    x_2 = torch.relu(x_1)
                    return x_2

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.submodule_1 = SubModule()

                def forward(self, x):
                    return self.submodule_1(x)

        The model hierarchy of ``x_1`` is ``submodule_1(SubModule).linear_1(Linear)``,
        and ``x_2`` has ``submodule_1(SubModule)``.

        We consider the ``node.name`` as the most inner ``scope_name``, and
        ``node.kind`` (aten op type) as the most inner ``scope_type``.

        ``x_1`` results in:
            {
                "scope_name": ["submodule_1", "linear_1", "x_1"],
                "scope_type": ["SubModule", "Linear", "linear"],
            },
        and ``x_2`` gets:
            {
                "scope_name": ["submodule_1", "x_2"],
                "scope_type": ["SubModule", "relu"],
            }.

        Note that, for the model weight const ops, the names are in the following format:
        "submodule_1.linear_1.weight", which would result in a long ``scope_name``:
        ``["submodule_1", "linear_1", "submodule_1.linear_1.weight"]``.
        This function does a special handling to trim it to:
        ``["submodule_1", "linear_1", "weight"]``
        """

        def _trim_scopename_for_weight(scope_names: List[str]) -> List[str]:
            weight_name = scope_names[-1]
            if scope_names[:-1] != weight_name.split(".")[:-1]:
                return scope_names
            scope_names[-1] = weight_name.split(".")[-1]
            return scope_names

        if self.model_hierarchy == "" or self.model_hierarchy is None:
            scopes = []
        else:
            scopes = self.model_hierarchy.split(".")
        scope_names, scope_types = [], []
        for val in scopes:
            if val == "":
                scope_names.append("UNKNOWN_SCOPE_NAME")
                scope_types.append("UNKNOWN_SCOPE_TYPE")
                continue
            if val.count("(") != 1 or val.count(")") != 1:
                raise ValueError(f"{val} is not a valid model hierarchy string.")
            lower_idx, upper_idx = val.index("("), val.index(")")
            scope_names.append(val[:lower_idx])
            scope_types.append(val[lower_idx + 1 : upper_idx])
        scope_names.append(self.name)
        scope_types.append(self.kind)
        if self.kind == "getattr":
            scope_names = _trim_scopename_for_weight(scope_names)
        return scope_names, scope_types


class InternalTorchIRGraph:
    """
    Core ML internal representation of a torch IR graph. A torch._C.Graph
    object is not an ideal structure to use in converting to CoreML. Conversion
    to an InternalTorchIRGraph is inserted between the original graph and the
    final Core ML model to address several issues:
        1. A torch._C.graph is hard to work with. For example, its .inputs()
          and .outputs() functions return iterators, so the only way to
          determine the number of inputs/outputs is by counting to the end.
          There are other examples of why the torch structure is hard to work
          with, and this structure alleviates those issues.
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
        params: Dict[str, np.ndarray],
        inputs: Dict[str, TensorType],
        outputs: List[str],
        nodes: Optional[List["InternalTorchIRNode"]] = None,
        buffers: Optional[Dict[str, torch.Tensor]] = None,
        input_name_to_source_buffer_name: Optional[Dict[str, str]] = None,
        output_name_to_target_buffer_name: Optional[Dict[str, str]] = None,
    ):
        """
        Arguments:
            params: dict mapping parameter names to their numpy value.
            inputs: OrderedDict mapping input names to their input types.
            outputs: list[str], list of outputs from the graph.
            nodes: list of InternalTorchIRNode in the graph.
            buffers: Dict mapping torch model buffers to their names.
            input_name_to_source_buffer_name: Dict[str, str] (EXIR only)
                dictionary mapping input variable names to underlying mutable buffer names,
                i.e. these input variables are "read" from mutable buffer
            output_name_to_target_buffer_name: Dict[str, str] (EXIR only)
                dictionary mapping output variable names to underlying mutable buffer names,
                i.e. these output variables are "written" to mutable buffer
        """
        self.nodes = nodes
        self.params = params
        self.inputs = inputs
        self.outputs = outputs
        self.buffers = buffers
        self.input_name_to_source_buffer_name = input_name_to_source_buffer_name
        self.output_name_to_target_buffer_name = output_name_to_target_buffer_name

        self.params_scope = {}

    @classmethod
    def from_torchscript(cls, torchscript, inputs=None, cut_at_symbols=None):
        """
        Arguments:
            torchscript: TorchScript object representing the model to convert.
            inputs: A list of input types to the graph.
            cut_at_symbols: The list of desired outputs from the graph. Symbols
                must be present in the graph. For debugging use only.
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
        inputs_name_to_type = OrderedDict()
        outputs = []

        raw_graph, params, buffers = _expand_and_optimize_ir(torchscript)

        # Add inputs
        # The first element of the raw_graph.inputs() is the 'self' of the module, which is not used.
        graph_inputs = list(raw_graph.inputs())[1:]
        if len(graph_inputs) != len(inputs):
            raise ValueError(
                f"Number of TorchScript inputs ({len(graph_inputs)}) must match the user provided inputs ({len(inputs)})."
            )
        for index, _input in enumerate(graph_inputs):
            name = _input.debugName()
            inputs_name_to_type[name] = inputs[index]

        # Add outputs, cutting if @cut_at_symbols is set
        output_names = cut_at_symbols
        if output_names is None:
            output_names = [x.debugName() for x in raw_graph.outputs()]
        for output in output_names:
            outputs.append(output)

        internal_graph = cls(
            nodes=nodes,
            params=params,
            inputs=inputs_name_to_type,
            outputs=outputs,
            buffers=buffers,
        )

        # Add nodes
        node_names = set()
        for raw_node in raw_graph.nodes():
            new_node = InternalTorchIRNode.from_torchscript_node(
                node=raw_node, parent=internal_graph
            )
            if new_node.name == new_node.kind:
                new_node.name = _find_new_name(new_node.name, node_names)
            internal_graph.nodes.append(new_node)
            node_names.add(new_node.name)

        internal_graph._cache_model_hierarchy_for_params()

        return internal_graph

    def _cache_model_hierarchy_for_params(self) -> None:
        # We cache the model hierarchy information for model weights in self.params_scope,
        # since self.params doesn't contain the information.
        def cache_model_hierarchy_block(block):
            for node in block.nodes:
                for b in node.blocks:
                    cache_model_hierarchy_block(b)
                if node.name in self.params:
                    self.params_scope[node.name] = node.get_scope_info()
        cache_model_hierarchy_block(self)

    @classmethod
    def from_exir(cls, exir):
        exported_program: torch.export.ExportedProgram = exir
        (
            user_inputs,
            user_outputs,
            params,
            buffers,
            input_name_to_source_buffer_name,
            output_name_to_target_buffer_name,
        ) = extract_io_from_exir_program(exported_program)

        inputs = OrderedDict([(i.name, i) for i in user_inputs])

        nodes = []
        for node in exported_program.graph_module.graph.nodes:
            if node.op == "call_function":
                nodes.append(InternalTorchIRNode.from_exir_node(node=node))
            elif node.op == "get_attr":
                name = node.target
                attr = exported_program.graph_module.__getattr__(name)
                # Only handle simple tensor attribute for now
                # There may be unconvertible advanced attributes,
                # e.g. higher-level callables such as "call_delegate"
                if not isinstance(attr, torch.Tensor):
                    raise NotImplementedError("Only torch.Tensor attr handled yet")
                params[name] = attr
            elif node.op == "placeholder":
                continue
            elif node.op == "output":
                continue
            else:
                raise NotImplementedError(f"Nodes of type {node.op} not yet implemented")

        return cls(
            nodes=nodes,
            params=params,
            inputs=inputs,
            outputs=user_outputs,
            buffers=buffers,
            input_name_to_source_buffer_name=input_name_to_source_buffer_name,
            output_name_to_target_buffer_name=output_name_to_target_buffer_name,
        )

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
