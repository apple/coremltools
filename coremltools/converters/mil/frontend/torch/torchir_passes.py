#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict, defaultdict
from typing import Dict, Optional

from coremltools import _logger as logger

from .internal_graph import InternalTorchIRGraph, InternalTorchIRNode


def generate_tensor_assignment_ops(graph: InternalTorchIRGraph) -> None:
    """
    This graph pass handles inplace tensor assignments, specifically it handles:
    `torch.Tensor.copy_` and `torch.Tensor.fill_`. There are many other inplace tensor
    assignments which are currently not handled.

    for instance:

        def forward(self, x):    # x a tensor with shape [4,10]
            x[:2, 4] = [[1],[3]]
            return x

    In Pytorch, this is represented by a sequence of slice / select ops followed by a copy op:

        input -> %x
        %1 = slice(%x, dim=0, begin=0, end=2, stride=1) # the slice for dimension 0
        %2 = select(%1, dim=1, index=4) # the select for dimension 1
        %3 = copy_(%2, value=[[1], [3]])
        output -> %x

    This graph pass fuses the sequences into a single InternalTorchIRNode of a new kind, which is defined as `_internal_op_tensor_inplace_copy_`.

        input -> %x
        %nodes_to_fuse = [slice(%x, begin=0, end=2, stride=1), select(%1, dim=1, index=4)]
        %x_internal_tensor_assign_1 = _internal_op_tensor_inplace_copy_(%x, value=[[1],[3]], nodes_to_fuse=nodes_to_fuse)
        output -> x_internal_tensor_assign_1

    The _internal_tensor_value_assign op takes an additional internal data member nodes_to_fuse,
    which is a list of select / slice InternalTorchIRNodes that need to be fused.
    Here is a more complicated example:

        def forward(self, x):    # x a tensor with shape [4,10]
            x[0, 0] = 1
            x[1:2, 1:2] = [[0]]
            return x

    Input graph:
        input -> %x
        %1 = select(%x, dim=0, index=0)
        %2 = select(%1, dim=0, index=0)
        %3 = copy_(%2, value=1)
        %4 = slice(%x, dim=0, begin=1, end=2, stride=1)
        %5 = slice(%4, dim=1, begin=1, end=2, stride=1)
        %6 = copy_(%5, value=[[0]])
        output -> %x

    Output graph:
        input -> %x
        %nodes_to_fuse_1 = [select(%x, dim=0, index=0), select(%1, dim=0, index=0)]
        %x_internal_tensor_assign_1 = _internal_op_tensor_inplace_copy_(%x, value=1, nodes_to_fuse=nodes_to_fuse_1)
        %nodes_to_fuse_2 = [slice(%x, dim=0, begin=1, end=2, stride=1), slice(%4, dim=1, begin=1, end=2, stride=1)]
        %x_internal_tensor_assign_2 = _internal_op_tensor_inplace_copy_(%x_internal_tensor_assign_1, value=[[0]], nodes_to_fuse=nodes_to_fuse_2)
        output -> x_internal_tensor_assign_2

    torch.Tensor.fill_ works in a similar way, except the InternalTorchIRNodes is defined by `_internal_op_tensor_inplace_fill_`.

    A fill_ operator is generated from the following forward pass:

        def forward(self, x):    # x a tensor with shape [5, 4]
            x[2] = 9
            return x

    In case:

        def forward(self, x):    # x a tensor with shape [4,10]
            y = torch.empty(*x.shape)
            y.copy_(0)
            return y

    Input graph:

        input -> %x
        %y = empty[](x.shape)
        %1 = copy_[](%y, %x)
        return (%1)
        output -> %1

    In result of fuse

        input -> %x
        %y = [empty[](x.shape)]
        %x_internal_tensor_assign_1 = _internal_op_tensor_inplace_copy_(%y, %x)
        output -> %x_internal_tensor_assign_1

    As a result of side effects of fusing, output of `_internal_op_tensor_inplace_copy_` will be renamed to `x_internal_tensor_assign_1`.
    If `%1` should be renamed to `x_internal_tensor_assign_1` too, the graph will be invalid.
    In this purpose out_alias was introduced.
    """

    TENSOR_ASSIGMENT_PREFIX = "_internal_tensor_assign_"

    def _get_updated_name(name, updated_tensor_count, out_alias):
        if name in updated_tensor_count:
            return name + TENSOR_ASSIGMENT_PREFIX + str(updated_tensor_count[name])
        if name in out_alias:
            return out_alias[name]
        return name

    def _construct_nodes_to_fuse_inputs(nodes_to_fuse):
        inputs = []
        for node in nodes_to_fuse:
            if node.kind == "select":
                inputs += [node.inputs[2], None, None]
            if node.kind == "slice":
                inputs += [node.inputs[2], node.inputs[3], node.inputs[4]]
        return inputs

    tensor_to_node_sequence_mapping = {}
    updated_tensor_count = defaultdict(lambda: 0)
    out_alias = {}

    for i in range(len(graph.nodes)):
        node = graph.nodes[i]

        for idx in range(len(node.inputs)):
            input_name = node.inputs[idx]
            node.inputs[idx] = _get_updated_name(input_name, updated_tensor_count, out_alias)

        if node.kind in ("empty", "select", "slice"):
            node_input = node.inputs[0]
            node_output = node.outputs[0]
            node_sequence = tensor_to_node_sequence_mapping.get(node_input, [])
            if len(node_sequence) > 0:
                tensor_to_node_sequence_mapping.pop(node_input)
            node_sequence.append(node)
            tensor_to_node_sequence_mapping[node_output] = node_sequence

        if node.kind == "to":
            node_input = node.inputs[0]
            if node_input in tensor_to_node_sequence_mapping:
                # update the mapping
                node_output = node.outputs[0]
                val = tensor_to_node_sequence_mapping[node_input]
                del tensor_to_node_sequence_mapping[node_input]
                tensor_to_node_sequence_mapping[node_output] = val

        if node.kind in ("copy_", "fill_"):
            node_input = node.inputs[0]
            if node_input not in tensor_to_node_sequence_mapping:
                raise ValueError("No matching select or slice.")

            if node.kind == "copy_":
                kind = "_internal_op_tensor_inplace_copy_"
            else:
                kind = "_internal_op_tensor_inplace_fill_"

            nodes_to_fuse = tensor_to_node_sequence_mapping[node_input]
            if nodes_to_fuse[0].kind in ["select", "slice"]:
                source_tensor = nodes_to_fuse[0].inputs[0]
            else:
                source_tensor = nodes_to_fuse[0].outputs[0]

            origin_name = source_tensor.split(TENSOR_ASSIGMENT_PREFIX)[0]
            updated_tensor_count[origin_name] += 1
            outputs = [_get_updated_name(origin_name, updated_tensor_count, out_alias)]
            out_alias[node.outputs[0]] = outputs[0]

            update_value = node.inputs[1]
            nodes_to_fuse_inputs = _construct_nodes_to_fuse_inputs(nodes_to_fuse)
            tensor_assign_node = InternalTorchIRNode(
                name=outputs[0],
                inputs=[source_tensor, update_value] + nodes_to_fuse_inputs,
                outputs=outputs,
                kind=kind,
                blocks=[],
                model_hierarchy=node.model_hierarchy,
            )
            graph.nodes[i] = tensor_assign_node

    # modify the graph outputs if it is effected by this graph pass
    for idx in range(len(graph.outputs)):
        output = graph.outputs[idx]
        graph.outputs[idx] = _get_updated_name(output, updated_tensor_count, out_alias)


def populate_native_const_model_hierarchy(graph: InternalTorchIRGraph) -> None:
    """
    Torchscript doesn't capture the model hierarchy of those python native consts.
    For instance:

    class Submodule(torch.nn.Module):
        def forward(self, x):
            x = x + 0.9
            x = x * 0.9
            return torch.relu(x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.submodule_1 = Submodule()

        def forward(self, x):
            return self.submodule_1(x)

    The two ``0.9`` constants don't have the scope of Submodule.
    In this graph pass, we make the model hierarchy of such constants inherited from
    their child ops.
    """

    cached_model_hierarchy = {}
    child_ops = defaultdict(list)

    for node in graph.nodes:
        for b in node.blocks:
            populate_native_const_model_hierarchy(b)

    for node in graph.nodes:
        cached_model_hierarchy[node.name] = node.model_hierarchy
        for val in node.inputs:
            child_ops[val].append(node.name)

    for node in graph.nodes:
        if node.kind != "constant":
            continue
        if node.model_hierarchy == "" and len(child_ops[node.name]) == 1:
            node.model_hierarchy = cached_model_hierarchy[child_ops[node.name][0]]


def remove_getattr_nodes(graph: InternalTorchIRGraph) -> None:
    """
    Remove the getattr nodes in the graph
    """

    getattr_nodes = []
    new_nodes = []

    for node in graph.nodes:

        for block in node.blocks:
            remove_getattr_nodes(block)

        if node.kind == "getattr":
            getattr_nodes.append(node)
        else:
            new_nodes.append(node)

    # check the getattr nodes not in the outputs
    for node in getattr_nodes:
        if node.name in graph.outputs:
            raise RuntimeError("{} should not be in the graph outputs.".format(node.name))

    # remove the getattr nodes
    graph.nodes = new_nodes


def transform_inplace_ops(
    graph: InternalTorchIRGraph, name_remap_dict: Optional[Dict[str, str]] = None
) -> None:

    # As we modify ops, we'll need to remap symbols.
    if name_remap_dict is None:
        name_remap_dict = {}

    for node in graph.nodes:
        for k, v in name_remap_dict.items():
            node.replace_name(k, v)

        if node.kind == "append":
            if isinstance(node.parent, InternalTorchIRGraph):
                # If append appears in a graph (outer block), replace
                # subsequent uses of its input symbol with its output symbol.
                name_remap_dict[node.inputs[0]] = node.outputs[0]
            elif node.parent.parent.kind == "loop":
                # If append appears in a loop block, add its inputs to the block
                # inputs and loop inputs, and its outputs to the block outputs
                # and loop outputs.

                # This is the global input to append. We need to add it to the
                # loop's input list, and replace any uses after the node with
                # @global_output below.
                global_input = node.inputs[0]
                # This will be the name of the input to append within the
                # block. We need to add it to the block inputs.
                local_input = node.parent.parent.name + ".0"
                # This is the output of append. We need to add it to the list
                # of block outputs.
                local_output = node.outputs[0]
                # This is the name of the new output from the loop. It should
                # replace any uses of @global_input after the loop op.
                global_output = local_output + ".out"
                name_remap_dict[global_input] = global_output

                node.parent.parent.inputs.append(global_input)
                node.parent.inputs.append(local_input)
                node.replace_name(global_input, local_input)
                node.parent.outputs.append(local_output)
                node.parent.parent.outputs.append(global_output)
                node.parent.parent.name = node.parent.parent.outputs[0]
            elif node.parent.parent.kind == "if":
                # If append appears in an if/else block, add its outputs to the
                # block outputs and loop outputs.
                # Note that we can't assume the append appears in both blocks.
                raise NotImplementedError(
                    "inplace_ops pass doesn't yet support append op inside conditional"
                )

        for block in node.blocks:
            transform_inplace_ops(block, name_remap_dict)

    # Replace names in graph outputs
    for k, v in name_remap_dict.items():
        try:
            idx = graph.outputs.index(k)
        except ValueError:
            pass
        else:
            graph.outputs[idx] = v


def flatten_graph_input_values(graph: InternalTorchIRGraph) -> None:
    """CoreML can't handle nested iterables of tensors, so we flatten the
    inputs of any graph that expects them.
    """
    new_graph_inputs = graph.inputs
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
                    logger.warning(
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
                        name=_input_name,
                    )
                )
            else:
                # This input isn't a tuple, keep it as is.
                new_graph_inputs[_input_name] = _input_val
        all_new_nodes = new_nodes + all_new_nodes
    graph.inputs = new_graph_inputs
    graph.nodes = all_new_nodes + graph.nodes


def flatten_graph_output_values(graph: InternalTorchIRGraph) -> None:
    """
    CoreML can't handle nested iterables of tensors, so we flatten the
    outputs of any graph that produces them.
    """
    node_names = [node.name for node in graph.nodes]
    new_graph_outputs = graph.outputs
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
            if graph.nodes[node_idx].kind in [
                "tupleconstruct",
                "listconstruct",
            ]:
                # Since this output came from a construct op, we can replace it
                # with the inputs to the op.
                new_graph_outputs.extend(graph.nodes[node_idx].inputs)
                changed = True
                if not notified:
                    notified = True
                    logger.warning(
                        "Tuple detected at graph output. This will be flattened in the converted model."
                    )
            else:
                new_graph_outputs.append(outp)
    # Note: if we flattened outputs, there are likely to be construct ops
    # that are no longer needed. These will be removed in a later DCE pass.
    graph.outputs = new_graph_outputs
