# Copyright (c) 2025, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# from coremltools.converters import convert
from collections import OrderedDict, deque
from dataclasses import dataclass
from logging import getLogger as _getLogger
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.jit._script import ScriptModule

from coremltools import proto
from coremltools._deps import _HAS_TORCH_EXPORT_API
from coremltools.converters.mil.mil.scope import ScopeSource

from ....model import MLModel
from ..debugging_utils import MLModelInspector, _retrieve_dependency_names, _Status
from ..remote_device import Device

if _HAS_TORCH_EXPORT_API:
    from torch.export import ExportedProgram

_logger = _getLogger(__name__)

import re

import tqdm

@dataclass(frozen=True)
class TorchScriptNodeInfo:
    """
    Represents information about a node in a torch graph.

    Attributes
    ----------
    source_range : str
    The node's source range.

    modules : Iterable["TorchScriptModuleInfo.Key"]
    The modules to which this node belongs.

    input_names : List[str]
    The input names.

    output_names : List[str]
    The output names.
    """

    source_range: str
    modules: Tuple["TorchScriptModuleInfo.Key"]
    desc: str
    kind: str
    input_names: Tuple[str]
    output_names: Tuple[str]

    @property
    def owning_module(
        self,
    ) -> Optional["TorchScriptModuleInfo.Key"]:
        return self.modules[0] if len(self.modules) > 0 else None

@dataclass(frozen=True)
class TorchScriptModuleInfo:
    """
    Represents information about a torch module.

    Attributes
    ----------
    name : str
    The name of the module.

    call_sequence :
    The sequence number of this module call.

    hierarchy : List[str]
    The hierarchical path of the module.

    output_names : List[str]
    Names of the module's outputs.
    """

    Key = Tuple[str, int]

    name: str
    call_sequence: int
    hierarchy: Tuple[str]
    input_names: Tuple[str]
    output_names: Tuple[str]
    submodules: Tuple[Key]
    code: str

    @property
    def hierarchy_name(
        self,
    ) -> str:
        return ".".join(filter(bool, self.hierarchy))

    @property
    def key(self) -> "TorchScriptModuleInfo.Key":
        return (self.hierarchy_name, self.call_sequence)


def _topological_sort_graph(
    graph: Dict[TorchScriptNodeInfo, List[TorchScriptNodeInfo]]
) -> List[TorchScriptNodeInfo]:
    in_degrees = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degrees[neighbor] += 1

    # Create a queue and enqueue all vertices with in-degree 0
    queue = deque([node for node in graph if in_degrees[node] == 0])

    result = []
    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                queue.append(neighbor)

    return result


def _build_graph(
    nodes: Dict[str, TorchScriptNodeInfo],
):
    result = {node: [] for node in nodes.values()}
    for node in nodes.values():
        for input_name in node.input_names:
            dep_node = nodes.get(input_name, None)
            if dep_node is None:
                continue

            neighbors = result[dep_node]
            neighbors.append(node)

    return result


def _topological_sort_nodes(
    nodes: Dict[str, TorchScriptNodeInfo],
) -> "OrderedDict[str, TorchScriptNodeInfo]":
    graph = _build_graph(nodes=nodes)
    result = OrderedDict()
    for node in _topological_sort_graph(graph=graph):
        for output_name in node.output_names:
            result[output_name] = node

    return result


class TorchScriptModuleAnnotator:
    """
    A class for annotating the module graph.

    This class provides methods to annotate torch modules, track their call sequences,
    and analyze node dependencies within the graph.
    """

    class ContextManager:
        def __init__(
            self,
            annotator: "TorchScriptModuleAnnotator",
            module: ScriptModule,
            module_hierarchy: List[str],
            module_name: str,
            graph: torch._C.Graph,
        ) -> None:
            self.module = module
            self.annotator = annotator
            self.module_hierarchy = module_hierarchy
            self.module_name = module_name
            self.graph = graph

        def __enter__(
            self,
        ) -> None:
            self.annotator.enter_module(
                module=self.module,
                hierarchy=self.module_hierarchy,
                name=self.module_name,
                graph=self.graph,
            )

        def __exit__(
            self,
            *args,
        ) -> None:
            self.annotator.exit_module()

    def __init__(
        self,
        name_prefix: str = "var",
    ):
        self.name_prefix = name_prefix
        self._module = None
        self._graphs = []
        self._module_call_sequence = {}
        self._module_infos = []
        self._node_infos = OrderedDict()
        self._module_call_stack = OrderedDict()
        self._submodules = {}
        self._handle = 1

    def annotate_module(
        self,
        module: ScriptModule,
        hierarchy: Tuple[str],
        name: str,
        graph: torch._C.Graph,
    ) -> "ContextManager":
        """
        Creates a context manager for annotating a module.

        Parameters
        ----------
        hierarchy : Tuple[str]
            The hierarchical path of the module.

        name : str
            The name of the module.

        graph : _TorchGraph
            The module graph.

        Returns
        -------
        ContextManager
            A context manager for module annotation.
        """
        return TorchScriptModuleAnnotator.ContextManager(
            annotator=self,
            module=module,
            module_hierarchy=hierarchy,
            module_name=name,
            graph=graph,
        )

    @property
    def node_infos(self) -> "OrderedDict[str, TorchScriptNodeInfo]":
        """
        Returns a copy of the node information dictionary.
        """
        return self._node_infos.copy()

    @property
    def module_call_stack(self) -> "OrderedDict[TorchScriptModuleInfo.Key, TorchScriptModuleInfo]":
        """
        Returns a copy of the module call stack.
        """
        return self._module_call_stack.copy()

    def _get_module_call_sequence(
        self,
        hierarchy: List[str],
    ) -> int:
        key = frozenset(hierarchy)
        value = self._module_call_sequence.get(key, 0)
        self._module_call_sequence[key] = value + 1
        return value

    def _annotate_graph(
        self,
        graph: torch._C.Graph,
    ):
        for node in graph.nodes():
            for output in node.outputs():
                name = f"{self.name_prefix}_{self._handle}"
                output.setDebugName(name)
                self._handle += 1

    def _create_node_infos(
        self,
        module_key: TorchScriptModuleInfo.Key,
        graph: torch._C.Graph,
    ):
        for node in graph.nodes():
            node_info = TorchScriptNodeInfo(
                source_range=node.sourceRange(),
                modules=(module_key,),
                desc="",
                input_names=(),
                output_names=(),
                kind=node.kind(),
            )
            for output in node.outputs():
                self._node_infos[output.debugName()] = node_info

    def _update_node_infos(
        self,
        module_key: TorchScriptModuleInfo.Key,
        graph: torch._C.Graph,
    ):
        for node in graph.nodes():
            for output in node.outputs():
                name = output.debugName()
                node_info = self._node_infos.get(name, None)
                if node_info is None:
                    continue

                input_names = [input.debugName() for input in node.inputs()]
                output_names = [output.debugName() for output in node.outputs()]
                modules = node_info.modules
                if module_key not in modules:
                    modules = modules + (module_key,)

                self._node_infos[name] = TorchScriptNodeInfo(
                    source_range=node_info.source_range,
                    modules=modules,
                    input_names=tuple(input_names),
                    output_names=tuple(output_names),
                    kind=node_info.kind,
                    desc=str(node),
                )

    def enter_module(
        self,
        module: ScriptModule,
        hierarchy: List[str],
        name: str,
        graph: torch._C.Graph,
    ):
        """
        Enters a module context and annotates its graph.

        Parameters
        ----------
        hierarchy : List[str]
            The hierarchical path of the module.

        name : str
            The name of the module.

        graph : _TorchGraph
            The module graph.
        """

        input_names = [input.debugName() for input in graph.inputs()]

        call_sequence = self._get_module_call_sequence(
            hierarchy=hierarchy,
        )

        self._annotate_graph(graph=graph)

        module_info = TorchScriptModuleInfo(
            name=name,
            hierarchy=tuple(hierarchy),
            output_names=(),
            input_names=tuple(input_names),
            call_sequence=call_sequence,
            code=module.code,
            submodules=(),
        )

        self._create_node_infos(
            module_key=module_info.key,
            graph=graph,
        )

        self._module_infos.append(module_info)
        self._graphs.append(graph)

    def _cleanup_dead_nodes(
        self,
        graph: torch._C.Graph,
    ):
        output_names = set(
            [output.debugName() for node in graph.nodes() for output in node.outputs()]
        )
        keys_to_remove = set()
        for output_name in self._node_infos:
            if output_name not in output_names:
                keys_to_remove.add(output_name)

        for key in keys_to_remove:
            self._node_infos.pop(key)

    def _update_submodules(
        self,
    ):
        for (module_key, submodule_keys) in self._submodules.items():
            module_info = self._module_call_stack.get(module_key, None)
            if module_info is None:
                continue

            new_module_info = TorchScriptModuleInfo(
                name=module_info.name,
                call_sequence=module_info.call_sequence,
                hierarchy=module_info.hierarchy,
                output_names=module_info.output_names,
                input_names=module_info.input_names,
                code=module_info.code,
                submodules=tuple(submodule_keys),
            )

            self._module_call_stack[module_key] = new_module_info

    def exit_module(
        self,
    ) -> None:
        """
        Exits a module context and finalizes annotations.
        """
        graph = self._graphs.pop()
        module_info = self._module_infos.pop()
        output_names = [output.debugName() for output in graph.outputs()]
        new_module_info = TorchScriptModuleInfo(
            name=module_info.name,
            call_sequence=module_info.call_sequence,
            hierarchy=module_info.hierarchy,
            output_names=tuple(output_names),
            input_names=module_info.input_names,
            code=module_info.code,
            submodules=module_info.submodules,
        )

        self._module_call_stack[new_module_info.key] = new_module_info
        parent_module_info = self._module_infos[-1] if len(self._module_infos) > 0 else None

        if parent_module_info is not None:
            deps = self._submodules.get(parent_module_info.key, [])
            deps.append(new_module_info.key)
            self._submodules[parent_module_info.key] = deps

        self._update_node_infos(
            graph=graph,
            module_key=new_module_info.key,
        )

        if len(self._module_infos) == 0:
            self._cleanup_dead_nodes(graph=graph)
            self._node_infos = _topological_sort_nodes(nodes=self._node_infos)
            self._update_submodules()


class _TorchScriptModuleInliner:
    """
    A class for inlining a TorchScript module.

    This class provides methods to inline submodules within a TorchScript module,
    effectively flattening the module hierarchy.
    """

    def __init__(
        self,
        module: ScriptModule,
        annotator: Optional[TorchScriptModuleAnnotator] = None,
    ):
        """
        Initializes the _TorchModuleInliner.

        Parameters
        ----------
        module : ScriptModule
            The root torch module to be inlined.

        annotator : Optional[_TorchScriptModuleAnnotator]
            An optional annotator for annotating and tracking the nodes in the graph.
        """
        self.module = module
        self.annotator = annotator

    @staticmethod
    def _clone_block(
        src: torch._C.Block,
        dst: torch._C.Block,
        dst_graph: torch._C.Graph,
        find_value: Callable[[torch._C.Value], torch._C.Value],
    ) -> torch._C.Block:
        local_value_map = {}

        def block_find_value(old_value: torch._C.Value):
            nonlocal local_value_map
            value = local_value_map.get(old_value, None)
            value = value if value is not None else find_value(old_value)
            if value is None:
                raise ValueError(
                    f"Unable to find corresponding value for {old_value}. "
                    f"This error occurs when a value from the original graph cannot be mapped "
                    f"to a value in the new graph. "
                )
            return value

        for input in src.inputs():
            dst_input = dst.paramNode().addInput()
            dst_input.copyMetadata(input)
            local_value_map[input] = dst_input

        for node in src.nodes():
            new_node = _TorchScriptModuleInliner._clone_node(
                node=node,
                graph=dst_graph,
                find_value=find_value,
            )

            dst.addNode(new_node)
            for new_output, output in zip(new_node.outputs(), node.outputs()):
                local_value_map[output] = new_output
                new_output.copyMetadata(output)

            for output in src.outputs():
                dst.registerOutput(block_find_value(output))

        return dst

    @staticmethod
    def _clone_node(
        node: torch._C.Node,
        graph: torch._C.Graph,
        find_value: Callable[[torch._C.Value], torch._C.Value],
    ) -> torch._C.Node:
        new_node_inputs = [find_value(input) for input in node.inputs()]

        new_node = graph.create(node.kind(), new_node_inputs, node.outputsSize())
        new_node.copyMetadata(node)
        new_node.copyAttributes(node)

        for new_output, output in zip(new_node.outputs(), node.outputs()):
            new_output.copyMetadata(output)

        for src_block in node.blocks():
            dst_block = new_node.addBlock()
            _TorchScriptModuleInliner._clone_block(
                src=src_block,
                dst=dst_block,
                dst_graph=graph,
                find_value=find_value,
            )

        return new_node

    def _insert_graph(
        graph: torch._C.Graph,
        callee: torch._C.Graph,
        inputs: Iterable[torch._C.Value],
        value_map: Dict[torch._C.Value, torch._C.Value],
    ) -> List[torch._C.Value]:
        def find_value(old_value: torch._C.Value):
            nonlocal value_map
            value = value_map.get(old_value, None)
            if value is None:
                raise ValueError(f"Missing value for {old_value}")

            return value

        for callee_input, input in zip(callee.inputs(), inputs):
            value_map[callee_input] = input

        for node in callee.nodes():
            cloned_node = _TorchScriptModuleInliner._clone_node(
                node=node,
                graph=graph,
                find_value=find_value,
            )

            graph.insertNode(cloned_node)
            for output, cloned_output in zip(node.outputs(), cloned_node.outputs()):
                cloned_output.setDebugName(output.debugName())
                value_map[output] = cloned_output

        outputs = [find_value(output) for output in callee.outputs()]
        return outputs

    @staticmethod
    def _retrieve_submodule_from_node(
        node: torch._C.Node,
        module: ScriptModule,
    ) -> Optional[Tuple[ScriptModule, str]]:
        def retrieve_submodule(
            node: torch._C.Node,
        ) -> Optional[ScriptModule]:
            stack = []
            curr_node = node
            while curr_node is not None and curr_node.kind() == "prim::GetAttr":
                stack.insert(0, curr_node)
                node_input = next(curr_node.inputs(), None)
                curr_node = node_input.node()

            if len(stack) == 0:
                return None

            curr_module = module
            module_names = []
            for attr_node in stack:
                module_name = attr_node.s("name")
                module_names.append(module_name)
                curr_module = getattr(curr_module, module_name, None)
                if curr_module is None:
                    return None

            return (curr_module, ".".join(module_names))

        if node.kind() != "prim::CallMethod":
            return None

        node_input = next(node.inputs(), None)
        if node_input is None:
            return None

        return retrieve_submodule(node=node_input.node())

    @staticmethod
    def _inline_call_node(
        node: torch._C.Node,
        graph: torch._C.Graph,
        callee: torch._C.Graph,
    ) -> List[torch._C.Value]:
        if node.kind() != "prim::CallMethod":
            raise ValueError(
                f"Invalid node kind encountered. Expected 'prim::CallMethod', but got '{node.kind()}'."
            )

        value_map = {}
        new_outputs = []
        with graph.insert_point_guard(node):
            new_outputs = _TorchScriptModuleInliner._insert_graph(
                graph=graph,
                callee=callee,
                inputs=node.inputs(),
                value_map=value_map,
            )

            for new_output, output in zip(new_outputs, node.outputs()):
                output.replaceAllUsesWith(new_output)

        node.destroy()
        return new_outputs

    def _inline_graph(
        self,
        graph: torch._C.Graph,
        module: ScriptModule,
        hierarchy: List[str],
    ):
        # Create a copy of nodes for iteration as the graph is mutated when inlining.
        for node in list(graph.nodes()):
            submodule_and_name = _TorchScriptModuleInliner._retrieve_submodule_from_node(
                node,
                module=module,
            )

            if submodule_and_name is not None:
                submodule, submodule_name = submodule_and_name
                submodule_graph = submodule.graph.copy()

                call_node_inputs = iter(node.inputs())
                submodule_graph_inputs = iter(submodule_graph.inputs())
                # Ignore the first input
                _ = next(call_node_inputs)
                _ = next(submodule_graph_inputs)
                for node_input, submodule_graph_input in zip(
                    call_node_inputs, submodule_graph_inputs
                ):
                    submodule_graph_input.setDebugName(node_input.debugName())

                self._inline_module(
                    module=submodule,
                    name=submodule_name,
                    hierarchy=hierarchy,
                    graph=submodule_graph,
                )

                _TorchScriptModuleInliner._inline_call_node(
                    node=node,
                    graph=graph,
                    callee=submodule_graph,
                )

        torch._C._jit_pass_dce(graph)

    def _inline_module(
        self,
        module: ScriptModule,
        graph: torch._C.Graph,
        name: str,
        hierarchy: List[str],
    ) -> None:
        curr_hierarchy = hierarchy + [name]
        if self.annotator is not None:
            with self.annotator.annotate_module(
                module=module,
                name=name,
                graph=graph,
                hierarchy=curr_hierarchy,
            ):
                self._inline_graph(
                    graph=graph,
                    module=module,
                    hierarchy=curr_hierarchy,
                )
        else:
            self._inline_graph(
                graph=graph,
                module=module,
                hierarchy=curr_hierarchy,
            )

    def inline(self) -> None:
        """
        Performs the inlining process on the root module.

        This method initiates the inlining process, starting from the root module
        and recursively inlining all submodules.
        """
        self._inline_module(
            module=self.module,
            name="",
            hierarchy=[],
            graph=self.module.graph,
        )


class _TorchModuleInspector:
    """
    A class for inspecting PyTorch modules during forward pass.

    This class provides methods to inspect specific modules within a PyTorch model,
    collecting their output tensors during a forward pass.
    """

    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:
        """
        Initializes the TorchModuleInspector.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to be inspected.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError('The "module" parameter must be of type "torch.nn.Module"')
        self._model = model

    def inspect(
        self,
        inputs: Tuple[torch.Tensor],
        module_info_keys: List[TorchScriptModuleInfo.Key],
    ) -> List[Tuple[(TorchScriptModuleInfo.Key, Optional[Any])]]:
        """
        Inspects specified modules during a forward pass of the model.

        This method registers forward hooks on the specified modules, performs a forward pass
        with the given inputs, and collects the output tensors for each specified module.

        Parameters
        ----------
        inputs : Tuple[torch.Tensor]
            The input tensors for the model's forward pass.

        module_infos : List[ModuleInfo]
            A list of ModuleInfo objects specifying which modules to inspect.

        Returns
        -------
        List[Tuple[ModuleInfo, Optional[torch.Tensor]]]
            A list of tuples, each containing a ModuleInfo object and its corresponding output tensor (if available).

        Note
        ----
        The method uses forward hooks to capture module outputs. These hooks are removed after the inspection process.
        """

        module_call_sequences = {}
        result = OrderedDict()

        for module_info_key in module_info_keys:
            result[module_info_key] = (module_info_key, None)

        def register_forward_hook(module: torch.nn.Module, name: str) -> Any:
            def set_module_output(
                curr_module: torch.nn.Module,
                inputs: Tuple[torch.Tensor],
                output: torch.Tensor,
            ) -> None:
                nonlocal module_call_sequences
                nonlocal result

                sequence = module_call_sequences.get(name, 0)
                module_call_sequences[name] = sequence + 1
                value = result.get((name, sequence), None)
                if value is not None:
                    result[(name, sequence)] = (value[0], output)

                return output

            return module.register_forward_hook(set_module_output, always_call=False)

        hooks = []
        interested_module_names = [module_info_key[0] for module_info_key in module_info_keys]
        for (name, module) in self._model.named_modules():
            if name in interested_module_names:
                hook = register_forward_hook(module=module, name=name)
                hooks.append(hook)

        self._model(*inputs)
        for hook in hooks:
            hook.remove()

        return result.values()


@dataclass
class TorchScriptModuleMappingInfo:
    """
    A class that holds mapping information for a TorchScript module
    and its corresponding operations in the converted Core ML model.

    Attributes
    ----------
    source : TorchScriptModuleInfo
        An instance representing the source module that is being mapped.

    source_to_target_mapping : OrderedDict[TorchScriptNodeInfo, List[proto.MIL_pb2.Operation]]
        An ordered mapping from nodes in the source TorchScript module to
        a list of corresponding operations in Core ML. Each key represents
        a node in the module, while each value is a list of
        operations that correspond to that node in the target model.

    deps : Dict[TorchScriptModuleInfo.Key, Iterable[TorchScriptModuleInfo.Key]]:
        A dictionary mapping each module to its immediate dependencies.
    """

    source: TorchScriptModuleInfo
    source_to_target_ops_mapping: "OrderedDict[TorchScriptNodeInfo : List[proto.MIL_pb2.Operation]]"
    deps: Dict[TorchScriptModuleInfo.Key, Iterable["TorchScriptModuleMappingInfo"]]
    outputs: List[proto.MIL_pb2.Operation]
    submodules: List["TorchScriptModuleMappingInfo"]

    @property
    def key(self) -> TorchScriptModuleInfo.Key:
        return self.source.key

    def get_source_ops_for_target_op(
        self,
        target_op: proto.MIL_pb2.Operation,
    ) -> List[TorchScriptNodeInfo]:
        result = []
        for source_op, target_ops in self.source_to_target_ops_mapping:
            if any(op == target_op for op in target_ops):
                result.append(source_op)

        return result


def _create_output_name_to_op_map_from_spec(
    spec: "proto.Model_pb2.Model",
    function_name: str,
) -> "OrderedDict[str, proto.MIL_pb2.Operation]":
    func_spec = spec.mlProgram.functions.get(
        function_name if function_name is not None else "main", None
    )
    if func_spec is None:
        raise ValueError(f"Missing function for name : {function_name}")

    block_spec = func_spec.block_specializations.get(func_spec.opset, None)
    if block_spec is None:
        raise ValueError(f"Missing block specialization for opset : {func_spec.opset}")

    result = OrderedDict()
    for op in block_spec.operations:
        for output in op.outputs:
            result[output.name] = op

    return result


def _invert_deps(
    modules: Dict[TorchScriptModuleInfo.Key, TorchScriptModuleMappingInfo],
    deps: Dict[TorchScriptModuleInfo.Key, Iterable[TorchScriptModuleMappingInfo]],
) -> Dict[TorchScriptModuleInfo.Key, Iterable[TorchScriptModuleMappingInfo]]:
    result = {}

    for module_key, module_map_infos in deps.items():
        dep = modules[module_key]
        for module_map_info in module_map_infos:
            values = result.get(module_map_info.key, [])
            values.append(dep)
            result[module_map_info.key] = values

    return result


def _get_module_deps(
    parent_module: TorchScriptModuleInfo,
    submodules: Iterable[TorchScriptModuleInfo],
    nodes: Dict[str, TorchScriptNodeInfo],
    graph: Dict[TorchScriptNodeInfo, List[TorchScriptNodeInfo]],
) -> Dict[TorchScriptModuleInfo.Key, Iterable[TorchScriptModuleInfo]]:
    input_name_to_modules_map = {}
    for module in submodules:
        for input_name in module.input_names:
            values = input_name_to_modules_map.get(input_name, [])
            values.append(module)
            input_name_to_modules_map[input_name] = values

    for output_name in parent_module.output_names:
        input_name_to_modules_map[output_name] = [parent_module]

    modules = {submodule.key: submodule for submodule in submodules}
    modules[parent_module.key] = parent_module
    result = {module.key: set() for module in modules.values()}

    for module in submodules:
        module_deps = result[module.key]
        start_nodes = [nodes.get(output_name, None) for output_name in module.output_names]
        start_nodes = filter(lambda node: node is not None, start_nodes)
        queue = deque(start_nodes)
        while len(queue) > 0:
            node = queue.popleft()
            deps = []
            for output_name in node.output_names:
                deps = input_name_to_modules_map.get(output_name, [])
                for module in deps:
                    module_deps.add(module)

            if len(deps) == 0:
                for neighbor in graph.get(node, []):
                    queue.append(neighbor)

    return _invert_deps(
        modules=modules,
        deps=result,
    )


@dataclass(frozen=True)
class _OpTranslationMap:
    output_name_to_target_ops: Dict[str, List[proto.MIL_pb2.Operation]]


def _retrieve_source_to_target_op_map(
    target_model: MLModel,
    function_name: str,
) -> _OpTranslationMap:
    output_name_to_op_map = _create_output_name_to_op_map_from_spec(
        spec=target_model.get_spec(),
        function_name=function_name,
    )

    output_name_to_target_ops = {}
    program = target_model._mil_program
    block = program.functions.get(function_name, None)
    if block is None:
        raise ValueError(f"Program is missing function for name={function_name}")

    for op in block.operations:
        module_names = op.scopes[ScopeSource.TORCHSCRIPT_MODULE_NAME]
        if len(module_names) == 0:
            continue

        source_name = module_names[-1]
        output_names = [output.name for output in op.outputs]
        target_ops = [output_name_to_op_map.get(output_name, None) for output_name in output_names]
        target_ops = list(filter(lambda op: op is not None, target_ops))
        ops = output_name_to_target_ops.get(source_name, [])
        ops.extend(target_ops)
        output_name_to_target_ops[source_name] = ops

    return _OpTranslationMap(
        output_name_to_target_ops=output_name_to_target_ops,
    )


def inline_and_annotate_module(
    model: ScriptModule,
    name_prefix: str = "var",
) -> TorchScriptModuleAnnotator:
    """
    Inlines and annotates a TorchScript module.

    This function takes a TorchScript module and performs two operations:
    1. Inlining: It inlines the module, which means it replaces calls to submodules
       with the actual operations performed by those submodules.
    2. Annotation: It adds annotations to the module, providing additional
       information about the module's structure and operations.

    Parameters:
    -----------
    model : ScriptModule
        The TorchScript module to be inlined and annotated. This should be an
        instance of ScriptModule, which is a subclass of torch.nn.Module
        that has been scripted using torch.jit.script().

    Returns:
    --------
    TorchScriptModuleAnnotator
        An annotator object that contains the inlined and annotated version of
        the input module.
    """
    annotator = TorchScriptModuleAnnotator(name_prefix=name_prefix)
    inliner = _TorchScriptModuleInliner(
        module=model,
        annotator=annotator,
    )
    inliner.inline()

    return annotator

def _convert_and_retrieve_jit_module_mapping(
    model: ScriptModule,
    annotator: Optional[TorchScriptModuleAnnotator] = None,
    **converter_kwargs,
) -> Tuple[MLModel, Dict[TorchScriptModuleInfo.Key, TorchScriptModuleMappingInfo]]:
    """
    Converts a TorchScript model to a Core ML model and returns the
    mapping information mapping between the source TorchScript operations and their
    corresponding operations in the converted Core ML model.

    This function takes a TorchScript model, performs  inlining and annotation,
    converts it to a Core ML model using the specified conversion
    parameters, and constructs a mapping that associates nodes in the source
    module with their corresponding operations in the Core ML model.

    Parameters
    ----------
    model : ScriptModule
        The source model.

    annotator : Optional[TorchScriptModuleAnnotator]
        The TorchScript module annotator, if the module has been annotated.

    **converter_kwargs:
        Additional keyword arguments to be passed to the Core ML conversion process.
        These can include options for optimization, input/output specifications, etc.

    Returns
    -------
    Tuple[MLModel, Dict[TorchScriptModuleInfo.Key, TorchScriptModuleMappingInfo]]
        A tuple containing:
            - The converted Core ML model.
            -  A dictionary mapping keys from `TorchScriptModuleInfo` to their corresponding
               ``TorchModuleMLModelMappingInfo`` instances. This mapping provides detailed
               information about how each node in the source module corresponds to operations
               in the target Core ML model.

    Examples
    --------
    .. sourcecode:: python
        (
            target_model,
            module_mapping,
        ) = coremltools.models.ml_program.experimental.torch._convert_and_retrieve_jit_module_mapping(
            model=traced_model,
            inputs=[
                coremltools.TensorType(
                    name="x", shape=example_inputs[0].shape, dtype=np.float16
                ),
                coremltools.TensorType(
                    name="y", shape=example_inputs[1].shape, dtype=np.float16
                ),
            ],
            minimum_deployment_target=coremltools.target.iOS16,
            compute_units=coremltools.ComputeUnit.CPU_ONLY,
            skip_model_load=True,
        )
    """
    # Importing here to avoid circular import issues
    from .....converters import convert

    if annotator is None:
        annotator = inline_and_annotate_module(model=model)

    target_model = convert(
        model=model,
        **converter_kwargs,
    )

    op_translation_map = _retrieve_source_to_target_op_map(
        target_model=target_model,
        function_name="main",
    )

    module_call_stack = annotator.module_call_stack
    module_map_infos = OrderedDict()

    def is_module_output(
        op: proto.MIL_pb2.Operation, output_names: List[str]
    ) -> List[proto.MIL_pb2.Operation]:
        for output in op.outputs:
            name = output.name
            for output_name in output_names:
                if name == output_name or name.startswith(f"{output_name}_cast"):
                    return True

        return False

    def _get_module_output_names(
        module_info: TorchScriptModuleInfo,
    ) -> List[str]:
        result = []
        node_infos = annotator.node_infos
        for output_name in module_info.output_names:
            node_info = node_infos.get(output_name, None)
            if node_info.kind == "prim::TupleConstruct":
                result.extend(node_info.input_names)
            else:
                result.append(output_name)

        return result

    for node_info in annotator.node_infos.values():
        module_ops = [
            op_translation_map.output_name_to_target_ops.get(name, [])
            for name in node_info.output_names
        ]
        module_ops = sum(module_ops, [])
        for module_key in node_info.modules:
            module_info = module_call_stack[module_key]
            module_map_info = module_map_infos.setdefault(
                module_key,
                TorchScriptModuleMappingInfo(
                    source=module_info,
                    source_to_target_ops_mapping=OrderedDict(),
                    deps=[],
                    outputs=[],
                    submodules=[],
                ),
            )

            module_output_names = _get_module_output_names(module_info)
            if len(module_map_info.outputs) < len(module_output_names):
                for op in module_ops:
                    if is_module_output(op, output_names=module_output_names):
                        module_map_info.outputs.append(op)

            ops = module_map_info.source_to_target_ops_mapping.get(node_info, [])
            ops.extend(module_ops)
            module_map_info.source_to_target_ops_mapping[node_info] = ops

    # Set module deps and submodules
    graph = _build_graph(nodes=annotator.node_infos)
    for module_key, module_map_info in module_map_infos.items():
        submodules = [module_call_stack[key] for key in module_map_info.source.submodules]
        module_map_info.submodules = submodules
        deps = _get_module_deps(
            parent_module=module_map_info.source,
            submodules=submodules,
            nodes=annotator.node_infos,
            graph=graph,
        )
        module_map_info.deps = {
            key: [module_map_infos[value.key] for value in values] for key, values in deps.items()
        }

    return (target_model, module_map_infos)


def _convert_and_retrieve_exported_program_op_mapping(
    model: "ExportedProgram",
    **converter_kwargs,
) -> Tuple[MLModel, Dict[torch.fx.Node, List[proto.MIL_pb2.Operation]]]:
    """
    Converts an ``ExportedProgram`` to a Core ML model and returns the
    mapping information mapping between `torch.fx.Node` and their
    corresponding operations in the converted Core ML model.

    Parameters
    ----------
    model : ExportedProgram
        The source model.

    **converter_kwargs:
        Additional keyword arguments to be passed to the Core ML conversion process.
        These can include options for optimization, input/output specifications, etc.

    Returns
    -------
    Tuple[MLModel, Dict[torch.fx.Node, List[proto.MIL_pb2.Operation]]]
        A tuple containing:
            - The converted Core ML model.
            - A dictionary-like object mapping each ``torch.fx.Node`` from the source model
              to a list of corresponding MIL operations in the Core ML model.
              This mapping helps trace the origin of operations in the converted
              model back to the source model for debugging and analysis.

    Examples
    --------
    .. sourcecode:: python
        (
            target_model,
            op_mapping,
        ) = coremltools.models.ml_program.experimental.torch._convert_and_retrieve_exported_program_op_mapping(
            model=program,
            inputs=[
                coremltools.TensorType(
                    name="x", shape=example_inputs[0].shape, dtype=np.float16
                ),
                coremltools.TensorType(
                    name="y", shape=example_inputs[1].shape, dtype=np.float16
                ),
            ],
            minimum_deployment_target=coremltools.target.iOS16,
            compute_units=coremltools.ComputeUnit.CPU_ONLY,
            skip_model_load=True,
        )
    """
    # Importing here to avoid circular import issues
    from .....converters import convert

    debug_handle = 1
    source_nodes = {}
    graph_module = model.graph_module
    for node in graph_module.graph.nodes:
        node.meta["debug_handle"] = debug_handle
        source_nodes[debug_handle] = node
        debug_handle += 1

    target_model = convert(
        model=model,
        **converter_kwargs,
    )

    output_name_to_op_map = _create_output_name_to_op_map_from_spec(
        spec=target_model.get_spec(),
        function_name="main",
    )

    program = target_model._mil_program
    block = program.functions["main"]
    source_to_target_ops = {source_node: [] for source_node in source_nodes.values()}
    for op in block.operations:
        scopes = op.scopes[ScopeSource.EXIR_DEBUG_HANDLE]
        debug_handle = scopes[-1] if isinstance(scopes, (list, tuple)) else None
        if debug_handle is None:
            continue

        source_op = source_nodes.get(debug_handle, None)
        if source_op is not None:
            output_names = [output.name for output in op.outputs]
            target_ops = [
                output_name_to_op_map.get(output_name, None) for output_name in output_names
            ]
            target_ops = list(filter(lambda op: op is not None, target_ops))
            source_to_target_ops[source_op].extend(target_ops)

    return (target_model, source_to_target_ops)


TorchNode = Union[TorchScriptNodeInfo, torch.fx.Node]


@dataclass(frozen=True)
class FrameInfo:
    filename: str
    lineno: int


def get_stack_frame_infos(node: TorchNode) -> Optional[List[FrameInfo]]:
    """
    Extracts frame infos from the node's source range.

    This method parses the stack trace of the node, attempts to extract the filename and line number
    using a regular expression.
    """

    source_range = None
    pattern = None
    if isinstance(node, TorchScriptNodeInfo):
        source_range = node.source_range if len(node.source_range) > 0 else None
        pattern = r"([^ ]+\.py)\((\d+)\)"
    elif isinstance(node, torch.fx.Node):
        source_range = node.meta.get("stack_trace", None)
        pattern = r'File "(.*?)", line (\d+), in (.*?)\n'
    else:
        raise ValueError(
            f"Expected node of type 'TorchScriptNodeInfo' or 'torch.fx.Node' but got {type(node).__name__}"
        )

    if source_range is None:
        return None

    matches = re.findall(pattern=pattern, string=source_range)
    frame_infos = [FrameInfo(filename=match[0], lineno=int(match[1])) for match in matches]
    return frame_infos

@dataclass
class TorchNodeToMILOperationMapping:
    """
    A class that represents the mapping between PyTorch nodes and MIL operations.
    """
    node_to_operations_map: Dict[TorchNode, List[proto.MIL_pb2.Operation]]
    operation_output_name_to_node_map: Dict[str, TorchNode]

    def get_source_nodes_for_output_name(
        self,
        output_name: str,
    ) -> Optional[TorchNode]:
        """
        Retrieves the source node for a given MIL operation's output name.

        Parameters
        ----------
        output_name : str
            The name of the output.

        Returns
        -------
        Optional[TorchNode]
            The corresponding TorchNode if found, None otherwise.
        """
        return self.operation_output_name_to_node_map.get(output_name, None)

    def get_source_nodes_for_operation(
        self,
        operation: proto.MIL_pb2.Operation,
    ) -> List[TorchNode]:
        """
        Retrieves the source node for a given MIL operation.

        Parameters
        ----------
        operation : proto.MIL_pb2.Operation
            The MIL operation.

        Returns
        -------
        Optional[TorchNode]
            The corresponding TorchNode if found, None otherwise.
        """
        result = []
        for output in operation.outputs:
            node = self.get_source_nodes_for_output_name(output.name)
            if node is not None:
                result.append(node)

        return result

def convert_and_retrieve_op_mapping(
    model: Union["ExportedProgram", ScriptModule],
    **converter_kwargs,
) -> Tuple[MLModel, TorchNodeToMILOperationMapping]:
    """
    Converts a TorchScript model to a Core ML model and returns the
    mapping information mapping between the source TorchScript operations and their
    corresponding operations in the converted Core ML model.

    Parameters
    ----------
    model : ScriptModule or ExportedProgram
        The source model.

    **converter_kwargs:
        Additional keyword arguments to be passed to the Core ML conversion process.
        These can include options for optimization, input/output specifications, etc.

    Returns
    -------
    Tuple[MLModel, TorchNodeToMILOperationMapping]
        A tuple containing:
            - The converted Core ML model.
            - A dictionary-like object mapping each TorchScript node from the original model
              to a list of corresponding MIL operations in the Core ML model.
              This mapping helps trace the origin of operations in the converted
              model back to the source model for debugging and analysis

    Examples
    --------
    .. sourcecode:: python
        (
            target_model,
            mapping_info,
        ) = coremltools.models.ml_program.experimental.torch.convert_and_retrieve_op_mapping(
            model=traced_model,  # ScriptModule or ExportedProgram
            inputs=[
                coremltools.TensorType(
                    name="x", shape=example_inputs[0].shape, dtype=np.float16
                ),
                coremltools.TensorType(
                    name="y", shape=example_inputs[1].shape, dtype=np.float16
                ),
            ],
            minimum_deployment_target=coremltools.target.iOS16,
            compute_units=coremltools.ComputeUnit.CPU_ONLY,
            skip_model_load=True,
        )
    """

    def get_output_name_to_node_mapping(
        op_mapping: Dict[TorchNode, List[proto.MIL_pb2.Operation]]
    ) -> Dict[str, TorchNode]:
        result = {}
        for source_node, target_operations in op_mapping.items():
            for target_operation in target_operations:
                for output in target_operation.outputs:
                    result[output.name] = source_node

        return result

    converted_model = None
    node_to_operations_map = None
    if isinstance(model, ScriptModule):
        converted_model, module_mapping = _convert_and_retrieve_jit_module_mapping(
            model=model,
            **converter_kwargs,
        )
        node_to_operations_map = module_mapping[("", 0)].source_to_target_ops_mapping

    elif isinstance(model, ExportedProgram):
        converted_model, node_to_operations_map = _convert_and_retrieve_exported_program_op_mapping(
            model=model, **converter_kwargs
        )
    else:
        raise ValueError(
            f"Unsupported model type: {type(model)}. Expected either ScriptModule or ExportedProgram."
        )

    mapping = TorchNodeToMILOperationMapping(
        node_to_operations_map=node_to_operations_map,
        operation_output_name_to_node_map=get_output_name_to_node_mapping(node_to_operations_map),
    )

    return (converted_model, mapping)


def _get_model_input_names(
    model: MLModel,
    function_name: str,
) -> List[str]:
    spec = model.get_spec()
    function = spec.mlProgram.functions.get(function_name, None)
    if function is None:
        raise ValueError(f"Program is missing function for name={function_name}")

    return [input.name for input in function.inputs]


def _flatten(values: Iterable[Any]) -> List[Any]:
    result = []
    for value in values:
        if isinstance(value, (list, tuple)):
            result.extend(_flatten(value))
        else:
            result.append(value)

    return result


class TorchScriptMLModelComparator:
    """
    A class for comparing the the intermediate outputs of a torch model with its converted Core ML model.

    This class provides functionality to compare the intermediate outputs of a torch model and
    the converted Core ML model, helping to identify the torch modules that produce different results.
    """

    CompareOutputs = Callable[
        [
            proto.MIL_pb2.Operation,
            np.array,
            np.array,
        ],
        bool,
    ]

    def _is_failure_source(
        self,
        module_map_info: TorchScriptModuleMappingInfo,
        statuses: Dict[TorchScriptModuleInfo.Key, _Status],
        deps: Dict[TorchScriptModuleInfo.Key, Iterable[TorchScriptModuleMappingInfo]],
    ) -> bool:
        if statuses.get(module_map_info.key, None) != _Status.FAIL:
            return False

        queue = deque()

        def add_deps(curr: TorchScriptModuleMappingInfo):
            for dep in deps.get(curr.key, []):
                queue.insert(0, dep)

        add_deps(module_map_info)

        while len(queue) > 0:
            dep = queue.popleft()
            status = statuses.get(dep.key, None)
            if status is None:
                return False
            elif status == _Status.FAIL:
                return False
            elif status == _Status.UNKNOWN:
                # If the dep status is unknown then we check its direct dependencies.
                add_deps(dep)

        # The module itself is likely the source of the failure, as its dependencies have passed
        # the validation check.
        return True

    def __init__(
        self,
        model: torch.nn.Module,
        example_inputs: Tuple[torch.tensor],
        num_predict_intermediate_outputs: int = 20,
        target_device: Optional[Device] = None,
        **converter_kwargs,
    ):
        """
        Initialize the TorchScriptMLModelComparator.

        This constructor sets up the comparator by preparing both the PyTorch model and its Core ML counterpart
        for comparison. It traces the PyTorch model, converts it to a Core ML model, and initializes necessary attributes
        for the comparison process.

        Parameters:
        -----------
        model: torch.nn.Module
            The PyTorch model to be compared. This should be a valid PyTorch module that can be traced
            using `torch.jit.trace` and converted to Core ML.

        example_inputs: Tuple[torch.tensor]
            A tuple of example input tensors that will be used to trace the PyTorch model. These inputs
            should be representative of the expected input format and shapes for the model.

        num_predict_intermediate_outputs: int
            The number of intermediate outputs to retrieve in each ``MLModel`` prediction call. Defaults to 20.

        target_device: Optional[Device]
            The target device on which to run the Core ML model. If None, the current default device will be used.

        **converter_kwargs : dict
            Additional keyword arguments to be passed to the Core ML converter. These can include
            options like 'minimum_deployment_target', 'compute_units', etc., allowing for fine-tuning
            of the conversion process.
        """
        model.eval()
        self.source_model = model
        self._source_model_inspector = _TorchModuleInspector(
            model=model,
        )

        traced_model = torch.jit.trace(model, example_inputs)
        target_model, module_map_infos = _convert_and_retrieve_jit_module_mapping(
            model=traced_model,
            **converter_kwargs,
        )

        self._source_root_module = next(
            filter(
                lambda module_map: module_map.source.hierarchy == ("",), module_map_infos.values()
            ),
            None,
        )
        self._module_map_infos = module_map_infos
        self.target_model = target_model

        self._target_model_inspector = MLModelInspector(
            model=self.target_model,
            function_name="main",
            device=target_device,
        )

        self._target_output_name_to_op_map = self._target_model_inspector.output_name_to_op_map
        self._target_model_input_names = _get_model_input_names(
            model=self.target_model,
            function_name="main",
        )

        self._source_outputs = {}
        self._target_outputs = {}
        self._num_predict_intermediate_outputs = num_predict_intermediate_outputs

    @property
    def module_map_infos(
        self,
    ) -> List[TorchScriptModuleMappingInfo]:
        return list(self._module_map_infos.values())

    def get_source_ops_for_targe_ops(
        self,
        target_op: proto.MIL_pb2.Operation,
    ) -> List[TorchScriptNodeInfo]:
        result = set()
        for module_map_info in self.module_map_infos:
            result.update(
                module_map_info.get_source_ops_for_target_op(
                    target_op=target_op,
                )
            )

        return result

    def _set_target_model(
        self,
        target_model: MLModel,
    ):
        compute_units = self.target_model.compute_unit
        self.target_model = target_model
        self._target_model_inspector = MLModelInspector(
            model=self.target_model,
            compute_units=compute_units,
            function_name=self._target_model_inspector._function_name,
            device=self._target_model_inspector.device,
        )

    async def _retrieve_outputs(
        self,
        op: proto.MIL_pb2.Operation,
        inputs: Dict[str, np.array],
    ) -> Iterable[Tuple[str, np.array]]:
        processed_output_names = set(self._target_outputs.keys())
        dep_names = _retrieve_dependency_names(
            op=op,
            max_dependencies=self._num_predict_intermediate_outputs,
            processed_output_names=processed_output_names,
            output_name_to_op_map=self._target_output_name_to_op_map,
        )

        op_output_names = [output.name for output in op.outputs]
        output_names = list(op_output_names) + dep_names
        # Retrieve the operation outputs along with selected dependencies.
        # This will reduce the number of prediction calls.
        outputs = await self._target_model_inspector.retrieve_outputs(
            output_names=output_names,
            inputs=inputs,
        )
        self._target_outputs.update(outputs)
        self._target_model_inspector.clear_cached_models()

        result = [
            (output.name, self._target_outputs.get(output.name, None)) for output in op.outputs
        ]
        return result

    async def _process_module(
        self,
        source_module_outputs: Iterable[Any],
        module_map_info: TorchScriptModuleMappingInfo,
        inputs: Dict[str, np.array],
        compare_outputs: CompareOutputs,
        queue: deque,
        deps: Dict[TorchScriptModuleInfo.Key, Iterable["TorchScriptModuleMappingInfo"]],
    ) -> _Status:
        def add_module_deps():
            for dep in deps.get(module_map_info.key, []):
                queue.insert(0, dep)

        if len(source_module_outputs) == 0:
            add_module_deps()
            return _Status.PASS

        def to_numpy(value: torch.Tensor, shape: Iterable[int]) -> Optional[torch.Tensor]:
            try:
                value = value.detach().cpu().numpy()
                return value.reshape(shape)
            except Exception as e:
                _logger.error(
                    f"Failed to reshape output for module {module_map_info.source}: {str(e)}"
                )
                return None

        source_module_outputs = _flatten(source_module_outputs)
        target_outputs = []
        for target_op in module_map_info.outputs:
            outputs = await self._retrieve_outputs(
                target_op,
                inputs=inputs,
            )

            target_outputs.extend(list(outputs))

        if len(source_module_outputs) != len(target_outputs):
            _logger.warning(
                f"Output mismatch detected for module: {module_map_info.source}\n"
                f"Number of source outputs: {len(source_module_outputs)}\n"
                f"Number of target outputs: {len(target_outputs)}\n"
            )
            # We will skip this module but process it's dependencies
            add_module_deps()
            return _Status.UNKNOWN

        status = _Status.PASS
        for source_output, (target_name, target_output) in zip(
            source_module_outputs, target_outputs
        ):
            if target_output is None:
                _logger.warning(
                    f"Failed to retrieve target output for '{target_name}' in module: {module_map_info.source}"
                )
                status = _Status.UNKNOWN
                break

            source_output = to_numpy(source_output, target_output.shape)
            if source_output is None:
                _logger.warning(
                    f"Failed to retrieve source output for '{target_name}' in module: {module_map_info.source}"
                )
                status = _Status.UNKNOWN
                break

            target_op = self._target_output_name_to_op_map.get(target_name, None)
            if not compare_outputs(target_op, source_output, target_output):
                status = _Status.FAIL
                break

        if status != _Status.PASS:
            add_module_deps()

        return status

    async def _find_failing_modules(
        self,
        source_outputs: Iterable[Any],
        module_map_info: TorchScriptModuleMappingInfo,
        inputs: Dict[str, np.array],
        compare_outputs: CompareOutputs,
        statuses: Dict[TorchScriptModuleInfo.Key, _Status],
        pbar: tqdm.tqdm,
    ) -> List[TorchScriptModuleMappingInfo]:
        queue = deque()
        failing_module_map_infos = []
        deps = module_map_info.deps

        queue.append(module_map_info)
        if statuses.get(module_map_info.key, None) == _Status.FAIL:
            for dep in deps.get(module_map_info.key, []):
                queue.insert(0, dep)

        while len(queue) > 0:
            curr_module_map_info = queue[0]
            status = statuses.get(curr_module_map_info.key, None)
            if status is not None:
                if self._is_failure_source(
                    module_map_info=curr_module_map_info,
                    statuses=statuses,
                    deps=deps,
                ):
                    """
                    The module did not pass comparison check, but its dependencies did.
                    This suggests that the failure likely comes from this module.
                    """
                    failing_module_map_infos.append(curr_module_map_info)

                queue.popleft()
            else:
                source_module_outputs = source_outputs.get(curr_module_map_info.key, [])
                status = await self._process_module(
                    module_map_info=curr_module_map_info,
                    source_module_outputs=source_module_outputs,
                    inputs=inputs,
                    compare_outputs=compare_outputs,
                    queue=queue,
                    deps=deps,
                )

                pbar.set_description(
                    desc=f"\033[1mAnalyzed module: {curr_module_map_info.key}\033[0m"
                )
                pbar.update(n=1)

            statuses[curr_module_map_info.key] = status

        result = []
        for curr_module_map_info in failing_module_map_infos:
            if curr_module_map_info is module_map_info or len(curr_module_map_info.submodules) == 0:
                result.append(curr_module_map_info)
            else:
                # Check submodules
                values = await self._find_failing_modules(
                    source_outputs=source_outputs,
                    module_map_info=curr_module_map_info,
                    inputs=inputs,
                    compare_outputs=compare_outputs,
                )
                result.extend(values)

        return result

    async def find_failing_modules(
        self,
        inputs: Tuple[torch.tensor],
        compare_outputs: CompareOutputs,
        source_module_keys: Optional[List[TorchScriptModuleInfo.Key]] = None,
    ) -> List[TorchScriptModuleMappingInfo]:
        """
        Asynchronously finds failing operations in the converted model.

        This method compares the outputs of the source PyTorch model with the
        converted Core ML model to identify operations that produce different results.

        Parameters
        ----------
        inputs : Tuple[torch.tensor]
            Input data for the models.

        compare_outputs : CompareOutputs
            Function to compare outputs between models.

        source_module_keys : Optional[List[TorchScriptModuleInfo.Key]]
            Specific module keys to check. Defaults to None (checks all modules).

        Returns
        -------
        List[TorchModuleMLModelMappingInfo]
            A list of failing modules.

        Examples
        --------
        .. sourcecode:: python
            class Model(torch.nn.Module):
                def forward(self, x, y):
                    return x + y


            model = Model()
            input1 = torch.full((1, 10), 1, dtype=torch.float)
            input2 = torch.full((1, 10), 2, dtype=torch.float)
            example_inputs = (input1, input2)
            comparator = (
                coremltools.models.ml_program.experimental.torch.TorchScriptMLModelComparator(
                    model=model,
                    example_inputs=example_inputs,
                    inputs=[
                        coremltools.TensorType(name="x", shape=inputs[0].shape, dtype=np.float32),
                        coremltools.TensorType(name="y", shape=inputs[1].shape, dtype=np.float32),
                    ],
                    minimum_deployment_target=coremltools.target.iOS16,
                    compute_units=coremltools.ComputeUnit.ALL,
                )
            )


            def compare_outputs(module, reference_output, target_output):
                return np.allclose(reference_output, target_output, atol=0.01)


            modules = await comparator.find_failing_modules(
                inputs=example_inputs, compare_outputs=compare_outputs
            )
        """

        statuses = {}

        if source_module_keys is None:
            source_module_keys = [self._source_root_module.source.key]

        source_module_outputs = self._source_model_inspector.inspect(
            inputs=inputs,
            module_info_keys=[key for key in self._module_map_infos],
        )

        target_inputs = {}
        for name, input in zip(self._target_model_input_names, inputs):
            target_inputs[name] = input.detach().cpu().numpy()

        source_outputs = {}
        for key, output in source_module_outputs:
            if isinstance(output, (list, tuple)):
                source_outputs[key] = output
            else:
                source_outputs[key] = [output]

        module_map_infos = []
        for module_key in source_module_keys:
            module_map_info = self._module_map_infos.get(module_key, None)
            if module_map_info is None:
                raise ValueError(
                    f"Module key '{module_key}' not found in the module map. "
                    f"Available keys are: {list(self._module_map_infos.keys())}"
                )

            module_map_infos.append(module_map_info)

        result = []
        with tqdm.tqdm(
            total=len(self._module_map_infos), desc="\033[1mAnalyzing modules...\033[0m"
        ) as pbar:
            for module_map_info in module_map_infos:
                values = await self._find_failing_modules(
                    source_outputs=source_outputs,
                    module_map_info=module_map_info,
                    inputs=target_inputs,
                    compare_outputs=compare_outputs,
                    statuses=statuses,
                    pbar=pbar,
                )

                result.extend(values)

        self._source_outputs = {}
        self._target_outputs = {}

        return result


class _TorchFXNodeValueInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        module: torch.nn.Module,
        garbage_collect_values=True,
    ) -> None:
        super().__init__(module, garbage_collect_values)
        self._intermediates = {}
        self._interested_nodes = None

    def run_node(
        self,
        node: torch.fx.Node,
    ) -> Any:
        result = super().run_node(node)
        if self._interested_nodes is None or node in self._interested_nodes:
            self._intermediates[node] = result
        return result

    def run(
        self,
        inputs: Tuple[torch.tensor],
        nodes: Optional[Iterable[torch.fx.Node]] = None,
    ) -> Dict[torch.fx.Node, Any]:
        self._interested_nodes = None
        if nodes is not None:
            self._interested_nodes = set(nodes)

        super().run(*inputs)
        result = self._intermediates.copy()
        self._intermediates.clear()
        return result

def _find_terminal_ops(
    ops: List[proto.MIL_pb2.Operation],
) -> List[proto.MIL_pb2.Operation]:
    if len(ops) < 2:
        return ops

    bindings = [input for op in ops for input in op.inputs.values()]
    arguments = [argument for binding in bindings for argument in binding.arguments]
    input_names = set([argument.name for argument in arguments if argument.value is not None])
    result = []
    for op in ops:
        output_names = [output.name for output in op.outputs]
        if all(output_name not in input_names for output_name in output_names):
            result.append(op)

    return result


class TorchExportMLModelComparator:
    """
    Compares intermediate outputs between a PyTorch ExportedProgram and its converted Core ML model.

    This class facilitates the comparison of intermediate outputs from a PyTorch model
    and its corresponding Core ML conversion. It helps identify specific PyTorch operations
    that may produce divergent results in the converted model.
    """

    CompareOutputs = Callable[
        [
            proto.MIL_pb2.Operation,
            np.array,
            np.array,
        ],
        bool,
    ]

    def __init__(
        self,
        model: "ExportedProgram",
        num_predict_intermediate_outputs: int = 20,
        target_device: Optional[Device] = None,
        **converter_kwargs,
    ):
        """
        Initialize the TorchExportMLModelComparator.

        Parameters
        ----------
        model : torch.export.ExportedProgram
            The ExportedProgram to be compared.

        num_predict_intermediate_outputs : int
            The number of intermediate outputs to retrieve in each ``MLModel`` prediction call. Defaults to 20.

        target_device : Optional[Device]
            The target device on which to run the Core ML model. If None, the current default device will be used.

        **converter_kwargs : Any
            Additional keyword arguments to be passed to the Core ML converter. These can include
            options like 'minimum_deployment_target', 'compute_units', etc., allowing for fine-tuning
            of the conversion process.

        """
        self.source_model = model
        self._source_model_inspector = _TorchFXNodeValueInterpreter(
            module=model.graph_module,
        )

        (
            target_model,
            source_to_target_ops_mapping,
        ) = _convert_and_retrieve_exported_program_op_mapping(
            model=model,
            **converter_kwargs,
        )

        self._source_to_target_ops_mapping = source_to_target_ops_mapping
        self.target_model = target_model

        self._target_model_inspector = MLModelInspector(
            model=self.target_model,
            function_name="main",
            device=target_device,
        )

        self._target_output_name_to_op_map = self._target_model_inspector.output_name_to_op_map
        self._target_model_input_names = _get_model_input_names(
            model=self.target_model,
            function_name="main",
        )

        self._source_outputs = {}
        self._target_outputs = {}
        self._num_predict_intermediate_outputs = num_predict_intermediate_outputs
        self._visited = set()

    @property
    def source_to_target_ops_mapping(
        self,
    ) -> Dict[torch.fx.Node, List[proto.MIL_pb2.Operation]]:
        return self._source_to_target_ops_mapping.copy()

    def get_source_ops_for_target_op(
        self,
        target_op: proto.MIL_pb2.Operation,
    ) -> List[torch.fx.Node]:
        result = []
        for source_op, target_ops in self._source_to_target_ops_mapping:
            if any(item is target_op for item in target_ops):
                result.append(source_op)

        return result

    def _set_target_model(
        self,
        target_model: MLModel,
    ):
        compute_units = self.target_model.compute_unit
        self.target_model = target_model
        self._target_model_inspector = MLModelInspector(
            model=target_model,
            compute_units=compute_units,
            function_name=self._target_model_inspector._function_name,
            device=self._target_model_inspector.device,
        )

    async def _retrieve_outputs(
        self,
        op: proto.MIL_pb2.Operation,
        inputs: Dict[str, np.array],
    ) -> Iterable[Tuple[str, np.array]]:
        processed_output_names = set(self._target_outputs.keys())
        dep_names = _retrieve_dependency_names(
            op=op,
            max_dependencies=self._num_predict_intermediate_outputs,
            processed_output_names=processed_output_names,
            output_name_to_op_map=self._target_output_name_to_op_map,
        )

        op_output_names = [output.name for output in op.outputs]
        output_names = list(op_output_names) + dep_names
        # Retrieve the operation outputs along with selected dependencies.
        # This will reduce the number of prediction calls.
        outputs = await self._target_model_inspector.retrieve_outputs(
            output_names=output_names,
            inputs=inputs,
        )
        self._target_outputs.update(outputs)
        self._target_model_inspector.clear_cached_models()

        result = [
            (output.name, self._target_outputs.get(output.name, None)) for output in op.outputs
        ]
        return result

    @staticmethod
    def _is_failure_source(
        node: torch.fx.Node,
        statuses: Dict[torch.fx.Node, _Status],
    ) -> bool:
        if statuses.get(node, None) != _Status.FAIL:
            return False

        queue = deque()

        def add_deps(curr: TorchScriptModuleMappingInfo):
            op_type_to_skip = ["placeholder", "get_attr", "call_module"]
            for dep in curr.all_input_nodes:
                if dep.op not in op_type_to_skip:
                    queue.insert(0, dep)

        add_deps(node)

        while len(queue) > 0:
            dep = queue.popleft()
            status = statuses.get(dep, None)
            if status is None:
                return False
            elif status == _Status.FAIL:
                return False
            elif status == _Status.UNKNOWN:
                # If the dep status is unknown then we check its direct dependencies.
                add_deps(dep)

        return True

    async def _process_node(
        self,
        node: torch.fx.Node,
        target_inputs: Dict[str, np.array],
        queue: deque,
        compare_outputs: Callable[[proto.MIL_pb2.Operation, np.array, np.array], bool],
    ) -> _Status:
        def is_output_op(op: proto.MIL_pb2.Operation) -> bool:
            return any(
                output.name == node.name or output.name.startswith(f"{node.name}_cast")
                for output in op.outputs
            )

        def add_deps(node: torch.fx.Node):
            for dep in node.all_input_nodes:
                if dep not in self._visited:
                    queue.insert(0, dep)

        def to_numpy(value: torch.Tensor, shape: Iterable[int]) -> Optional[torch.Tensor]:
            try:
                value = value.detach().cpu().numpy()
                return value.reshape(shape)
            except Exception as e:
                _logger.error(f"Failed to reshape tensor for {node}: {str(e)}")
                return None

        if node.op == "get_attr" or node.op == "placeholder":
            return _Status.PASS

        self._visited.add(node)
        source_outputs = self._source_outputs.get(node, [])
        source_outputs = _flatten(source_outputs)

        if len(source_outputs) == 0:
            _logger.warning(f"Unable to retrieve outputs for node: {node}\n")
            add_deps(node=node)
            return _Status.UNKNOWN

        # Find ops that produce the output
        target_ops = _find_terminal_ops(self._source_to_target_ops_mapping.get(node, []))
        target_ops = list(filter(lambda op: op.type != "const" and is_output_op(op), target_ops))
        if len(target_ops) == 0:
            _logger.warning(f"Unable to retrieve target ops for node: {node}\n")
            add_deps(node=node)
            return _Status.UNKNOWN

        target_outputs = []
        for target_op in target_ops:
            outputs = await self._retrieve_outputs(
                target_op,
                inputs=target_inputs,
            )

            target_outputs.extend(list(outputs))

        if len(source_outputs) != len(target_outputs):
            _logger.warning(
                f"Output mismatch detected for node: {node}\n"
                f"Number of source outputs: {len(source_outputs)}\n"
                f"Number of target outputs: {len(target_outputs)}\n"
            )
            add_deps(node=node)
            return _Status.UNKNOWN

        status = _Status.PASS
        for source_output, (target_name, target_output) in zip(source_outputs, target_outputs):
            if target_output is None:
                _logger.warning(
                    f"Failed to retrieve target output for '{target_name}' for node: {node}"
                )
                status = _Status.UNKNOWN
                break

            source_output = to_numpy(source_output, target_output.shape)
            if source_output is None:
                _logger.warning(
                    f"Failed to retrieve source output for '{target_name}' for node: {node}"
                )
                status = _Status.UNKNOWN
                break

            target_op = self._target_output_name_to_op_map.get(target_name, None)
            if not compare_outputs(target_op, source_output, target_output):
                status = _Status.FAIL
                break

        if status != _Status.PASS:
            add_deps(node=node)

        return status

    async def find_failing_ops(
        self,
        inputs: Tuple[torch.tensor],
        compare_outputs: Callable[[proto.MIL_pb2.Operation, np.array, np.array], bool],
        output_nodes: Optional[List[torch.fx.Node]] = None,
    ) -> Iterable[torch.fx.Node]:
        """
        Identifies operations that produce different outputs in the reference and target models.

        This method compares the outputs of the source and target models for specified operations,
        identifying those that fail the comparison criteria.

        Parameters
        ----------
        inputs: Tuple[torch.tensor]
            Input data for the model.

        compare_outputs : Callable[[proto.MIL_pb2.Operation, np.array, np.array], bool])
            A function to compare outputs of an operation between the two models.

        outputs: Optional[List[torch.fx.Node]]
            Specific outputs to compare. If None, all model outputs are compared. Defaults to None.

        Notes
        -----
        - The method uses a breadth-first search strategy to traverse the operation graph.
        - An operation is considered a failure source if it fails comparison while its direct inputs do not.

        Returns
        -------
        List[proto.MIL_pb2.Operation]
            A list of operations that failed the comparison.

        Examples
        --------

        .. sourcecode:: python
            class Model(torch.nn.Module):
                def forward(self, x, y):
                    return x + y


            model = Model()
            input1 = torch.full((1, 10), 1, dtype=torch.float)
            input2 = torch.full((1, 10), 2, dtype=torch.float)
            inputs = (input1, input2)
            exported_program = torch.export.export(model, inputs)

            comparator = (
                coremltools.models.ml_program.experimental.torch.TorchExportMLModelComparator(
                    model=exported_program,
                    inputs=[
                        coremltools.TensorType(name="x", shape=inputs[0].shape, dtype=np.float16),
                        coremltools.TensorType(name="y", shape=inputs[1].shape, dtype=np.float16),
                    ],
                    minimum_deployment_target=coremltools.target.iOS16,
                    compute_units=coremltools.ComputeUnit.ALL,
                )
            )


            def compare_outputs(op, reference_output, target_output):
                return np.allclose(reference_output, target_output, atol=0.01)


            ops = await comparator.find_failing_ops(inputs=inputs, compare_outputs=compare_outputs)
        """
        if output_nodes is None:
            output_nodes = []
            for node in self.source_model.graph.nodes:
                if node.op == "output":
                    output_nodes.append(node)

        source_outputs = self._source_model_inspector.run(inputs=inputs)
        self._source_outputs = {
            node: value if isinstance(value, (list, tuple)) else [value]
            for node, value in source_outputs.items()
        }

        target_inputs = {}
        input_names = _get_model_input_names(self.target_model, "main")
        for name, input in zip(input_names, inputs):
            target_inputs[name] = input.detach().cpu().numpy()

        queue = deque(output_nodes)
        result = set()
        statuses = {}

        with tqdm.tqdm(
            total=len(self.source_model.graph.nodes), desc="\033[1mAnalyzing nodes...\033[0m"
        ) as pbar:
            while len(queue) > 0:
                node = queue[0]
                status = statuses.get(node, None)
                if status is not None:
                    if self._is_failure_source(
                        node=node,
                        statuses=statuses,
                    ):
                        """
                        The node did not pass comparison check, but its dependencies did.
                        This suggests that the failure likely comes from this node.
                        """
                        result.add(node)

                    queue.popleft()
                else:
                    status = await self._process_node(
                        node=node,
                        target_inputs=target_inputs,
                        queue=queue,
                        compare_outputs=compare_outputs,
                    )

                    pbar.set_description(
                        desc=f"\033[1mAnalyzed node: {node.name}, type: {node.op}\033[0m"
                    )
                    pbar.update(n=1)
                statuses[node] = status

            self._visited = set()
            self._reference_outputs = {}
            self._target_outputs = {}

        return list(result)
