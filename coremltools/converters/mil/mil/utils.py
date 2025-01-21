# Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .operation import Operation


class OpNode:
    """
    A helper node class for the doubly linked list.
    It contains an Operation data and pointers to the previous and the next node.
    """

    def __init__(self, op: Operation):
        self.op = op
        self.next: Optional[OpNode] = None
        self.prev: Optional[OpNode] = None

    def __deepcopy__(
        self,
        memo: Dict[int, Any],
    ) -> "OpNode":
        # The implementation overrides the default ``__deepcopy__`` to prevent infinite recursion
        # when dealing with a large number of linked nodes that exceed the Python
        # call stack limit. It implements a non-recursive deep copy algorithm.
        if id(self) in memo:
            return memo[id(self)]

        def copy_node(node: "OpNode") -> "OpNode":
            if id(node) in memo:
                return memo[id(node)]

            op_copy = deepcopy(node.op, memo=memo)
            node_copy = OpNode(op=op_copy)
            memo[id(node)] = node_copy

            return node_copy

        def copy_linked_nodes(node: OpNode) -> List[Tuple[OpNode, OpNode]]:
            nodes = []
            visited = set()
            queue = [node]
            while len(queue) > 0:
                curr_node = queue.pop(0)
                if id(curr_node) in visited:
                    continue

                visited.add(id(curr_node))
                copied_node = copy_node(node=curr_node)
                nodes.append((curr_node, copied_node))

                if curr_node.next is not None:
                    queue.append(curr_node.next)

                if curr_node.prev is not None:
                    queue.append(curr_node.prev)

            return nodes

        # Copy all nodes that are reachable from ``self``
        all_nodes = copy_linked_nodes(self)
        for node, node_copy in all_nodes:
            if node.next is not None:
                node_copy.next = memo[id(node.next)]

            if node.prev is not None:
                node_copy.prev = memo[id(node.prev)]

        return memo[id(self)]

class CacheDoublyLinkedList:
    """
    This array-like data structure is useful to implement pymil's
    core program transformations, including:

    1. Insert an op at a target location (before a target op)
    2. Remove an op from the program

    Given the fact that each op in the list must be unique, a hash table
    is maintained in this data structure, and hence the insert / pop can both be performed in O(1).
    """

    INVALID_NODE = OpNode(None)

    def __init__(self, array: Optional[List[Operation]] = None):
        self.start: OpNode = None
        self.end: OpNode = None
        self.op_to_node: Dict[Operation, OpNode] = {}

        if array is not None:
            for op in array:
                self.insert_op_before(op)

    def insert_op_before(self, new_op: Operation, before_op: Optional[Operation] = None):
        """
        Insert an op right before before_op. If before_op is None,
         then the new op is appended in the end.
        """
        if new_op in self.op_to_node:
            raise ValueError(f"{new_op} already exisits.")

        new_node = OpNode(new_op)

        if before_op is None:
            # If before op is None, the new node is appended in the end.
            if self.start is None:
                self.start = self.end = new_node
            else:
                self.end.next = new_node
                new_node.prev = self.end
                self.end = new_node
        else:
            anchor_node = self.op_to_node[before_op]
            prev_node = anchor_node.prev

            if prev_node is None:
                self.start = new_node
            else:
                prev_node.next = new_node

            new_node.prev = prev_node
            new_node.next = anchor_node
            anchor_node.prev = new_node

        self.op_to_node[new_op] = new_node

    def remove(self, op: Operation):
        """
        Remove an op from the data structure.
        """
        node = self.op_to_node[op]
        prev_node, next_node = node.prev, node.next

        # reconnect the linked list
        if prev_node is None:
            self.start = next_node
        else:
            prev_node.next = next_node

        if next_node is None:
            self.end = prev_node
        else:
            next_node.prev = prev_node

        node.prev = node.next = self.INVALID_NODE

        # remove op from the cache
        del self.op_to_node[op]

    def __getitem__(self, idx: int) -> Operation:
        """
        The indexing is expensive in doubly linked list, we should prevent direct access besides [0] and [-1].
        """
        if self.start is None:
            raise ValueError("Cannot index an empty list.")
        if idx >= len(self):
            raise ValueError("Index out of range")
        if idx == 0:
            return self.start.op
        elif idx == -1:
            return self.end.op
        raise ValueError("Doubly linked list does not support indexing other than 0, -1.")

    def _get_node_from_op(self, op: Operation) -> OpNode:
        return self.op_to_node[op]

    def __iter__(self):
        cursor = self.start
        while cursor is not None:
            if cursor is self.INVALID_NODE:
                raise ValueError("Invalid iterator on CacheDoublyLinkedList.")
            yield cursor.op
            cursor = cursor.next

    def __reversed__(self):
        cursor = self.end
        while cursor is not None:
            if cursor is self.INVALID_NODE:
                raise ValueError("Invalid iterator on CacheDoublyLinkedList.")
            yield cursor.op
            cursor = cursor.prev

    def __len__(self) -> int:
        return len(self.op_to_node)
