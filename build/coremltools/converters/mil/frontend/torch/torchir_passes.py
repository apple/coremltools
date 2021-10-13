from collections import OrderedDict
import logging as _logging

from .internal_graph import *

def generate_tensor_assignment_ops(graph):
    """
    This graph pass handles the tensor value assignement,
    for instance:

        def forward(self, x): # x a tensor with shape [4,10]
            x[:2, 4] = [[1],[3]]
            return x

    In Pytorch, this is represented by a sequence of slice / select ops followed by a copy op:

        input -> %x
        %1 = slice(%x, dim=0, begin=0, end=2) # the slice for dimension 0
        %2 = select(%1, dim=1, index=4) # the select for dimension 1
        %3 = copy(%2, value=[[1], [3]])
        output -> %x

    This graph pass fuses the sequences into a single InternalTorchIRNode of a new kind, which is defined as _internal_tensor_value_assign.

        input -> %x
        %nodes_to_fuse = [slice(%x, begin=0, end=2), select(%1, dim=1, index=4)]
        %x_internal_tensor_assign_1 = _internal_tensor_value_assign(%x, value=[[1],[3]], nodes_to_fuse=nodes_to_fuse)
        output -> x_internal_tensor_assign_1

    The _internal_tensor_value_assign op takes an additional internal data member nodes_to_fuse,
    which is a list of select / slice InternalTorchIRNodes that need to be fused.
    Here is a more complicated example:

        def forward(self, x): # x a tensor with shape [4,10]
            x[0, 0] = 1
            x[1:2, 1:2] = [[0]]
            return x

    Input graph:
        input -> %x
        %1 = select(%x, dim=0, index=0)
        %2 = select(%1, dim=0, index=0)
        %3 = copy(%2, value=1)
        %4 = slice(%x, dim=0, begin=1, end=2)
        %5 = slice(%4, dim=1, begin=1, end=2)
        %6 = copy(%5, value=[[0]])
        output -> %x

    Output graph:
        input -> %x
        %nodes_to_fuse_1 = [select(%x, dim=0, index=0), select(%1, dim=0, index=0)]
        %x_internal_tensor_assign_1 = _internal_tensor_value_assign(%x, value=1, nodes_to_fuse=nodes_to_fuse_1)
        %nodes_to_fuse_2 = [slice(%x, dim=0, begin=1, end=2), slice(%4, dim=1, begin=1, end=2)]
        %x_internal_tensor_assign_2 = _internal_tensor_value_assign(%x_internal_tensor_assign_1, value=[[0]], nodes_to_fuse=nodes_to_fuse_2)
        output -> x_internal_tensor_assign_2
    """

    TENSOR_ASSIGMENT_PREFIX = "_internal_tensor_assign_"

    def _get_updated_name(name, updated_tensor_count):
        if name in updated_tensor_count:
            return name + TENSOR_ASSIGMENT_PREFIX + str(updated_tensor_count[name])
        return name

    def _construct_nodes_to_fuse_inputs(nodes_to_fuse):
        inputs = []
        for node in nodes_to_fuse:
            if node.kind == "select":
                inputs += [node.inputs[2], None]
            if node.kind == "slice":
                inputs += [node.inputs[2], node.inputs[3]]
        return inputs

    tensor_to_node_sequence_mapping = {}
    updated_tensor_count = {}

    for i in range(len(graph.nodes)):
        node = graph.nodes[i]

        for idx in range(len(node.inputs)):
            input_name = node.inputs[idx]
            node.inputs[idx] = _get_updated_name(input_name, updated_tensor_count)

        if node.kind in ["select", "slice"]:
            node_input = node.inputs[0]
            node_output = node.outputs[0]
            node_sequence = tensor_to_node_sequence_mapping.get(node_input, [])
            if len(node_sequence) > 0:
                tensor_to_node_sequence_mapping.pop(node_input)
            node_sequence.append(node)
            tensor_to_node_sequence_mapping[node_output] = node_sequence

        if node.kind == "copy_":
            node_input = node.inputs[0]
            if node_input in tensor_to_node_sequence_mapping:
                nodes_to_fuse = tensor_to_node_sequence_mapping[node_input]
                source_tensor = nodes_to_fuse[0].inputs[0]
                origin_name = source_tensor.split(TENSOR_ASSIGMENT_PREFIX)[0]

                if origin_name not in updated_tensor_count:
                    updated_tensor_count[origin_name] = 1
                else:
                    updated_tensor_count[origin_name] += 1

                outputs = [_get_updated_name(origin_name, updated_tensor_count)]

                update_value = node.inputs[1]
                nodes_to_fuse_inputs = _construct_nodes_to_fuse_inputs(nodes_to_fuse)
                tensor_assign_node = InternalTorchIRNode(
                                    node=None,
                                    inputs=[source_tensor, update_value] + nodes_to_fuse_inputs,
                                    outputs=outputs,
                                    kind="_internal_tensor_value_assign",
                                    blocks=[],
                                )
                graph.nodes[i] = tensor_assign_node

    # modify the graph outputs if it is effected by this graph pass
    for idx in range(len(graph.outputs)):
        output = graph.outputs[idx]
        if output in updated_tensor_count:
            graph.outputs[idx] = _get_updated_name(output, updated_tensor_count)


def transform_inplace_ops(graph, name_remap_dict=None):

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


def flatten_graph_input_values(graph):
    """ CoreML can't handle nested iterables of tensors, so we flatten the
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
    graph.inputs = new_graph_inputs
    graph.nodes = all_new_nodes + graph.nodes


def flatten_graph_output_values(graph):
    """ CoreML can't handle nested iterables of tensors, so we flatten the
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
                    _logging.warning(
                        "Tuple detected at graph output. This will be flattened in the converted model."
                    )
            else:
                new_graph_outputs.append(outp)
    # Note: if we flattened outputs, there are likely to be construct ops
    # that are no longer needed. These will be removed in a later DCE pass.
    graph.outputs = new_graph_outputs
