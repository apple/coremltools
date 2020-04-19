from . import _ops_to_layers
from tensorflow.python.util import compat
import numpy as np

MARK_CONST_OPS_BASED_ON_OUTPUT_VALUES = True


def _create_graph(ops):
    """
  Creates an adjacency representation of the directed graph formed from the
  list of TF ops.
  Nodes are the ops. Directed edge from an op "A" to an op "B" implies that at
  least one of the outputs of A is feeding as an input to B.

  :param: ops: list of ops. List of size N.
  :return: G: list of lists. Outer list is of size N.

  Let the number of ops be N.
  Then the adjacency representation is a list of lists: G. G[i] is the list of
  all ops that have directed edges impingent from op "i",
  i.e. G[i] is the fan-out list of op "i"
  """
    n = len(ops)
    G = [[] for i in range(n)]
    op_name_to_index = dict()
    # First pass to assign op name to its index
    for i, op in enumerate(ops):
        op_name_to_index[op.name] = i
    # Second pass to construct the graph
    for i, op in enumerate(ops):
        for inp in op.inputs:
            G[op_name_to_index[inp.op.name]].append(i)

    return G


def _push_stack(stack, node, in_stack):
    stack.append(node)
    if node in in_stack:
        raise ValueError("Graph has cycles.")
    else:
        in_stack[node] = True


def _get_unvisited_child(G, node, not_visited):
    for child in G[node]:
        if child in not_visited:
            return child
    return -1


def _find_unused_ops(ops, sess, output_names, feed_dict1, feed_dict2):
    """
  :param ops: list of TF graph ops
  :param sess: TF session for evaluating graph
  :param output_names: [str]: list of output names
  :param feed_dict1, feed_dict2: two input feed dictionaries with different valued inputs
  
  :return: [str], [str]: list of op names 
           - ununsed_op_names: they do not connect to the output
           - effectively_constant_op_names: their outputs do not change with feeding different valued Graph inputs
  """
    effective_const_ops_ids = []  # List[int]
    effective_const_op_names = []  # List[str]

    if MARK_CONST_OPS_BASED_ON_OUTPUT_VALUES:

        # first find the ops whose outputs do no change with different valued inputs: these can be skipped

        network_out_ids = []  # [int]
        network_out_ops = (
            []
        )  # [str], list of op names that connect to network outputs, these cannot be skipped
        tensors_to_evaluate = []  # [(str, tensor)]
        op_name_to_out_ids = (
            {}
        )  # {str: ([ints], int)}, skippable op name to (a) list of ids to index into list of tensors returned by session run (b) the id of the op
        ctr = 0

        for op in ops:
            for out in op.outputs:
                if out.name in output_names:
                    tensors_to_evaluate.append((compat.as_str_any(out.name), out))
                    network_out_ids.append(ctr)
                    ctr += 1
                    if op.name not in network_out_ops:
                        network_out_ops.append(op.name)

        for i, op in enumerate(ops):
            if (op.type not in _ops_to_layers._CORE_OPS) and (
                op.name not in network_out_ops
            ):
                ids = []
                for out in op.outputs:
                    tensors_to_evaluate.append((compat.as_str_any(out.name), out))
                    ids.append(ctr)
                    ctr += 1
                op_name_to_out_ids[op.name] = (ids, i)

        if len(tensors_to_evaluate) > 0:
            tensor_names, tensors = zip(*tensors_to_evaluate)
            tensors_evaluated1 = sess.run(tensors, feed_dict=feed_dict1)
            tensors_evaluated2 = sess.run(tensors, feed_dict=feed_dict2)
            networks_out_dont_match = True
            for idx in network_out_ids:
                out1 = tensors_evaluated1[idx].flatten().astype(np.float32)
                out2 = tensors_evaluated2[idx].flatten().astype(np.float32)
                if np.amax(np.abs(out1 - out2)) < 1e-4:
                    networks_out_dont_match = False
                    break
            if networks_out_dont_match:
                for op_name in list(op_name_to_out_ids.keys()):
                    is_skippable = True
                    for idx in op_name_to_out_ids[op_name][0]:
                        out1 = tensors_evaluated1[idx].flatten().astype(np.float32)
                        out2 = tensors_evaluated2[idx].flatten().astype(np.float32)
                        if out1.size == 0 and out2.size == 0:
                            continue
                        if np.amax(np.abs(out1 - out2)) > 1e-6:
                            is_skippable = False
                            break
                    if is_skippable:
                        effective_const_ops_ids.append(op_name_to_out_ids[op_name][1])

    # find ops that do not connect to the output
    G = _create_graph(ops)

    # first reverse the graph
    n = len(ops)
    reverse_G = [[] for i in range(n)]
    for i, child_list in enumerate(G):
        for j in child_list:
            reverse_G[j].append(i)

    # ids of all unvisited ops: initially all the ops are unvisited
    unvisited_op_ids = set(range(n))

    # get ids of ops that produce the network output nodes:
    # these will be the start nodes for our graph traversal
    start_ids = []
    for i, op in enumerate(ops):
        for out in op.outputs:
            if out.name in output_names:
                start_ids.append(i)

    if len(start_ids) == 0:
        raise ValueError(
            "No op found in the TF graph that produces the given output name(s)"
        )

    # Lets do BFS Graph traversal
    # (on the reverse TF graph starting from output producing ops)
    from collections import deque

    list_queue = deque()
    for idx in start_ids:
        # Mark idx as visited and put idx in queue
        # only visited nodes are put in the queue
        if idx in unvisited_op_ids:
            unvisited_op_ids.remove(idx)
            list_queue.append(idx)

        while len(list_queue) > 0:
            op_id = list_queue.popleft()
            for child_op in reverse_G[op_id]:
                if child_op in unvisited_op_ids:
                    unvisited_op_ids.remove(child_op)  # now child op is visited
                    if child_op in effective_const_ops_ids:
                        effective_const_op_names.append(ops[child_op].name)
                    else:
                        list_queue.append(
                            child_op
                        )  # add it to the queue, so that later we can look at its children

    # Collect all unvisited ops
    unused_op_names = []
    for i in unvisited_op_ids:
        unused_op_names.append(ops[i].name)

    return unused_op_names, effective_const_op_names


def _topological_sort_ops(ops):
    """
  :param ops: list of TF ops
  :return: list of TF ops, in topological sort order such that an op is
  encountered only after all the ops that generated its inputs have been
  visited.
  And also return a set of op names that can be skipped during conversion,
  as they are not connected to the output

  As a by product, also checks if the graph has cycles. Raises an error if
  it does.
  """

    G = _create_graph(ops)
    n = len(ops)
    # Topological label for each op. Highest will be for the sink nodes.
    topological_label = [-1 for i in range(n)]
    stack = []
    in_stack = dict()
    not_visited = dict.fromkeys([i for i in range(n)])
    label_counter = n - 1

    while len(not_visited) > 0:
        node = list(not_visited.keys())[0]
        _push_stack(stack, node, in_stack)
        while len(stack) > 0:
            node = _get_unvisited_child(G, stack[-1], not_visited)
            if node != -1:
                _push_stack(stack, node, in_stack)
            else:
                node = stack.pop()
                in_stack.pop(node)
                not_visited.pop(node)
                topological_label[node] = label_counter
                label_counter -= 1

    return [x for _, x in sorted(zip(topological_label, ops))]
