# -*- coding: utf-8 -*-
# Copyright (c) 2017, Apple Inc. All rights reserved.
import os
import six
import unittest

from coremltools.converters.nnssa.commons.features import Features
from coremltools.converters.nnssa.nnssa import ParsedNode as TestNode
from coremltools.converters.nnssa.commons.basic_graph_ops import *

class TestBasicGraphOps(unittest.TestCase):

    def _compare_items(self, item1, item2):
        if six.PY2:
            self.assertItemsEqual(item1, item2)
        else:
            self.assertCountEqual(item1, item2)

    def _get_test_nodes_with_name(self, *args):
        if len(args) == 1:
            # Python doesn't dispatch list with single item.
            node = TestNode()
            node.name = args[0]
            return node
        else:
            nodes = []
            for s in args:
                node = TestNode()
                node.name = s
                nodes.append(node)
            return nodes

    def _get_test_nodes_with_counts(self, count):
        if count == 1:
            # Python doesn't dispatch list with single item.
            node = TestNode()
            node.name = 'node'
            return node
        else:
            nodes = []
            for idx in range(count):
                node = TestNode()
                node.name = 'node_' + str(idx)
                nodes.append(node)
            return nodes

    def _get_gd(self, *args):
        gd = {}
        for node in args:
            gd[node.name] = node
        return gd

    def _check_node(self, gd, node, inputs, outputs, control_inputs, control_outputs):
        """
        Check if node matches the inputs/outputs/control_inputs/control_outputs
        """
        if Features.new_ssa():
            inputs = [gd[i] if isinstance(i, six.string_types) else i for i in inputs]
            control_inputs = [gd[i] if isinstance(i, six.string_types) else i for i in control_inputs]
        else:
            inputs = [i.name if not isinstance(i, six.string_types) else i for i in inputs]
            control_inputs = [i.name if not isinstance(i, six.string_types) else i for i in control_inputs]
        outputs = [i.name if not isinstance(i, six.string_types) else i for i in outputs]
        control_outputs = [i.name if not isinstance(i, six.string_types) else i for i in control_outputs]

        self.assertEqual(node.inputs, inputs)
        self.assertEqual(node.control_inputs, control_inputs)
        self.assertEqual(node.outputs, outputs)
        self.assertEqual(node.control_outputs, control_outputs)

    def _add_inputs(self, *args):
        inputs = []
        for inp in args:
            if Features.new_ssa():
                inputs.append(inp)
            else:
                inputs.append(inp.name)
        return inputs

    def test_build_connection(self):

        if Features.new_ssa():
            # No GetTuple, one-to-one
            P = Placeholder(dtype=DataType.FLOAT32, shape=[], name='placeholder')()
            I = Identity(name='identity')(P)
            gd = self._get_gd(P, I)
            build_connections(gd)
            self._compare_items(P.outputs, ['identity'])
            self._compare_items(I.outputs, [])

            # No GetTuple, one-to-many
            P = Placeholder(dtype=DataType.FLOAT32, shape=[], name='placeholder')()
            I = Identity(name='identity')(P)
            I_1 = Identity(name='identity_1')(P)
            gd = self._get_gd(P, I, I_1)
            build_connections(gd)
            self._compare_items(P.outputs, ['identity', 'identity_1'])
            self._compare_items(I.outputs, [])
            self._compare_items(I_1.outputs, [])

            # No GetTuple, many-to-one
            P = Placeholder(dtype=DataType.FLOAT32, shape=[], name='placeholder')()
            P_1 = Placeholder(dtype=DataType.FLOAT32, shape=[], name='placeholder_1')()
            A = Add(name='add')(P, P_1)
            gd = self._get_gd(P, P_1, A)
            build_connections(gd)
            self._compare_items(P.outputs, ['add'])
            self._compare_items(P_1.outputs, ['add'])
            self._compare_items(A.outputs, [])

            # No GetTuple, many-to-many
            P = Placeholder(dtype=DataType.FLOAT32, shape=[], name='placeholder')()
            P_1 = Placeholder(dtype=DataType.FLOAT32, shape=[], name='placeholder_1')()
            A = Add(name='add')(P, P_1)
            S = Sub(name='sub')(P_1, P)
            gd = self._get_gd(P, P_1, A, S)
            build_connections(gd)
            self._compare_items(P.outputs, ['add', 'sub'])
            self._compare_items(P_1.outputs, ['add', 'sub'])
            self._compare_items(A.outputs, [])
            self._compare_items(S.outputs, [])

            # With GetTuple, output uses all indices
            P = Placeholder(dtype=DataType.FLOAT32, shape=[2, 3], name='placeholder')()
            S_0, S_1, S_2 = UniformSplit(n_splits=3, name="split")(P)
            S = S_0.inputs[0] # Manually capture the "Split" node.
            gd = self._get_gd(P, S, S_0, S_1, S_2)
            build_connections(gd)
            self._compare_items(P.outputs, ['split'])
            self._compare_items(S.outputs, [S_0.name, S_1.name, S_2.name])
            self._compare_items(S_0.outputs, [])
            self._compare_items(S_1.outputs, [])
            self._compare_items(S_2.outputs, [])

            # With GetTuple, output doesn't use all indices

    def test_connect_edge(self):
        # connect_edge with node inputs
        src, dst = self._get_test_nodes_with_name('src', 'dst')

        gd = self._get_gd(src, dst)
        check_connections(gd)
        connect_edge(gd, src, dst)

        self._check_node(gd, src, [], [dst], [], [])
        self._check_node(gd, dst, [src], [], [], [])

        # connect_edge with str inputs
        src, dst = self._get_test_nodes_with_name('src', 'dst')

        gd = self._get_gd(src, dst)
        check_connections(gd)
        connect_edge(gd, src.name, dst.name)

        self._check_node(gd, src, [], [dst], [], [])
        self._check_node(gd, dst, [src], [], [], [])

    def test_replace_source(self):
        # Test node version
        src, dst, new_src = self._get_test_nodes_with_name('src', 'dst', 'new_src')

        gd = self._get_gd(src, dst, new_src)
        check_connections(gd)
        src.outputs = [dst.name]
        dst.inputs = self._add_inputs(src)
        replace_source(gd, src, dst, new_src)

        self._check_node(gd, src, [], [], [], [])
        self._check_node(gd, dst, [new_src], [], [], [])
        self._check_node(gd, new_src, [], [dst], [], [])

        # Test string version
        src, dst, new_src = self._get_test_nodes_with_name('src', 'dst', 'new_src')

        gd = self._get_gd(src, dst, new_src)
        check_connections(gd)
        src.outputs = [dst.name]
        dst.inputs = self._add_inputs(src)
        replace_source(gd, src.name, dst.name, new_src.name)

        self._check_node(gd, src, [], [], [], [])
        self._check_node(gd, dst, [new_src], [], [], [])
        self._check_node(gd, new_src, [], [dst], [], [])

    def test_replace_control_source(self):
        # Test node version
        src, dst, new_src = self._get_test_nodes_with_name('src', 'dst', 'new_src')

        gd = self._get_gd(src, dst, new_src)
        check_connections(gd)
        src.control_outputs = [dst.name]
        dst.control_inputs = self._add_inputs(src)
        replace_control_source(gd, src, dst, new_src)

        self._check_node(gd, src, [], [], [], [])
        self._check_node(gd, dst, [], [], [new_src], [])
        self._check_node(gd, new_src, [], [], [], [dst])

        # Test string version
        src, dst, new_src = self._get_test_nodes_with_name('src', 'dst', 'new_src')

        gd = self._get_gd(src, dst, new_src)
        check_connections(gd)
        src.control_outputs = [dst.name]
        dst.control_inputs = self._add_inputs(src)
        replace_control_source(gd, src.name, dst.name, new_src.name)

        self._check_node(gd, src, [], [], [], [])
        self._check_node(gd, dst, [], [], [new_src], [])
        self._check_node(gd, new_src, [], [], [], [dst])

    def test_replace_dest(self):
        # Test node version
        src, dst, new_dst = self._get_test_nodes_with_name('src', 'dst', 'new_dst')

        gd = self._get_gd(src, dst, new_dst)
        check_connections(gd)
        src.outputs = [dst.name]
        dst.inputs = self._add_inputs(src)
        replace_dest(gd, src, dst, new_dst)

        self._check_node(gd, src, [], [new_dst], [], [])
        self._check_node(gd, dst, [], [], [], [])
        self._check_node(gd, new_dst, [src], [], [], [])

        # Test string version
        src, dst, new_dst = self._get_test_nodes_with_name('src', 'dst', 'new_dst')

        gd = self._get_gd(src, dst, new_dst)
        check_connections(gd)
        src.outputs = [dst.name]
        dst.inputs = self._add_inputs(src)
        replace_dest(gd, src.name, dst.name, new_dst.name)

        self._check_node(gd, src, [], [new_dst], [], [])
        self._check_node(gd, dst, [], [], [], [])
        self._check_node(gd, new_dst, [src], [], [], [])

    def test_replace_control_dest(self):
        # Test node version
        src, dst, new_dst = self._get_test_nodes_with_name('src', 'dst', 'new_dst')

        gd = self._get_gd(src, dst, new_dst)
        check_connections(gd)
        src.control_outputs = [dst.name]
        dst.control_inputs = self._add_inputs(src)
        replace_control_dest(gd, src, dst, new_dst)

        self._check_node(gd, src, [], [], [], [new_dst])
        self._check_node(gd, dst, [], [], [], [])
        self._check_node(gd, new_dst, [], [], [src], [])

        # Test string version
        src, dst, new_dst = self._get_test_nodes_with_name('src', 'dst', 'new_dst')

        gd = self._get_gd(src, dst, new_dst)
        check_connections(gd)
        src.control_outputs = [dst.name]
        dst.control_inputs = self._add_inputs(src)
        replace_control_dest(gd, src.name, dst.name, new_dst.name)

        self._check_node(gd, src, [], [], [], [new_dst])
        self._check_node(gd, dst, [], [], [], [])
        self._check_node(gd, new_dst, [], [], [src], [])

    def test_connect_dests(self):
        # Test node version
        src, dst_1, dst_2, dst_3 = self._get_test_nodes_with_name('src', 'dst_1', 'dst_2', 'dst_3')

        gd = self._get_gd(src, dst_1, dst_2, dst_3)
        check_connections(gd)
        connect_dests(gd, src, [dst_1, dst_2, dst_3])

        self._check_node(gd, src, [], [dst_1, dst_2, dst_3], [], [])
        self._check_node(gd, dst_1, [src], [], [], [])
        self._check_node(gd, dst_2, [src], [], [], [])
        self._check_node(gd, dst_3, [src], [], [], [])

        # Test string version
        src, dst_1, dst_2, dst_3 = self._get_test_nodes_with_name('src', 'dst_1', 'dst_2', 'dst_3')

        gd = self._get_gd(src, dst_1, dst_2, dst_3)
        check_connections(gd)
        connect_dests(gd, src.name, [dst_1.name, dst_2.name, dst_3.name])

        self._check_node(gd, src, [], [dst_1, dst_2, dst_3], [], [])
        self._check_node(gd, dst_1, [src], [], [], [])
        self._check_node(gd, dst_2, [src], [], [], [])
        self._check_node(gd, dst_3, [src], [], [], [])

    def test_connect_sources(self):
        # Test node version
        dst, src_1, src_2, src_3 = self._get_test_nodes_with_name('dst', 'src_1', 'src_2', 'src_3')

        gd = self._get_gd(dst, src_1, src_2, src_3)
        check_connections(gd)
        connect_sources(gd, [src_1, src_2, src_3], dst)

        self._check_node(gd, src_1, [], [dst], [], [])
        self._check_node(gd, src_2, [], [dst], [], [])
        self._check_node(gd, src_3, [], [dst], [], [])
        self._check_node(gd, dst, [src_1, src_2, src_3], [], [], [])

        # Test string version
        dst, src_1, src_2, src_3 = self._get_test_nodes_with_name('dst', 'src_1', 'src_2', 'src_3')

        gd = self._get_gd(dst, src_1, src_2, src_3)
        check_connections(gd)
        connect_sources(gd, [src_1.name, src_2.name, src_3.name], dst.name)

        self._check_node(gd, src_1, [], [dst], [], [])
        self._check_node(gd, src_2, [], [dst], [], [])
        self._check_node(gd, src_3, [], [dst], [], [])
        self._check_node(gd, dst, [src_1, src_2, src_3], [], [], [])

    def test_disconnect_edge(self):
        # Test node version
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)
        dst_1.inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.inputs = self._add_inputs(src_2, src_3)
        dst_3.inputs = self._add_inputs(src_3)
        src_1.outputs = [dst_1.name]
        src_2.outputs = [dst_1.name, dst_2.name]
        src_3.outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)
        disconnect_edge(gd, src_1, dst_3) # No-op

        self._check_node(gd, src_1, [], [dst_1], [], [])
        self._check_node(gd, src_2, [], [dst_1, dst_2], [], [])
        self._check_node(gd, src_3, [], [dst_1, dst_2, dst_3], [], [])
        self._check_node(gd, dst_1, [src_1, src_2, src_3], [], [], [])
        self._check_node(gd, dst_2, [src_2, src_3], [], [], [])
        self._check_node(gd, dst_3, [src_3], [], [], [])

        disconnect_edge(gd, src_1, dst_1)
        disconnect_edge(gd, src_2, dst_3) # No-op
        disconnect_edge(gd, src_3, dst_3)

        self._check_node(gd, src_1, [], [], [], [])
        self._check_node(gd, src_2, [], [dst_1, dst_2], [], [])
        self._check_node(gd, src_3, [], [dst_1, dst_2], [], [])
        self._check_node(gd, dst_1, [src_2, src_3], [], [], [])
        self._check_node(gd, dst_2, [src_2, src_3], [], [], [])
        self._check_node(gd, dst_3, [], [], [], [])

        # Test string version
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)
        dst_1.inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.inputs = self._add_inputs(src_2, src_3)
        dst_3.inputs = self._add_inputs(src_3)
        src_1.outputs = [dst_1.name]
        src_2.outputs = [dst_1.name, dst_2.name]
        src_3.outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)
        disconnect_edge(gd, src_1.name, dst_3.name) # No-op

        self._check_node(gd, src_1, [], [dst_1], [], [])
        self._check_node(gd, src_2, [], [dst_1, dst_2], [], [])
        self._check_node(gd, src_3, [], [dst_1, dst_2, dst_3], [], [])
        self._check_node(gd, dst_1, [src_1, src_2, src_3], [], [], [])
        self._check_node(gd, dst_2, [src_2, src_3], [], [], [])
        self._check_node(gd, dst_3, [src_3], [], [], [])

        disconnect_edge(gd, src_1.name, dst_1.name)
        disconnect_edge(gd, src_2.name, dst_3.name) # No-op
        disconnect_edge(gd, src_3.name, dst_3.name)

        self._check_node(gd, src_1, [], [], [], [])
        self._check_node(gd, src_2, [], [dst_1, dst_2], [], [])
        self._check_node(gd, src_3, [], [dst_1, dst_2], [], [])
        self._check_node(gd, dst_1, [src_2, src_3], [], [], [])
        self._check_node(gd, dst_2, [src_2, src_3], [], [], [])
        self._check_node(gd, dst_3, [], [], [], [])

    def test_disconnect_control_edge(self):
        # Test node mode
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)
        dst_1.control_inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.control_inputs = self._add_inputs(src_2, src_3)
        dst_3.control_inputs = self._add_inputs(src_3)
        src_1.control_outputs = [dst_1.name]
        src_2.control_outputs = [dst_1.name, dst_2.name]
        src_3.control_outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)
        disconnect_control_edge(gd, src_1, dst_3) # No-op

        self._check_node(gd, src_1, [], [], [], [dst_1])
        self._check_node(gd, src_2, [], [], [], [dst_1, dst_2])
        self._check_node(gd, src_3, [], [], [], [dst_1, dst_2, dst_3])
        self._check_node(gd, dst_1, [], [], [src_1, src_2, src_3], [])
        self._check_node(gd, dst_2, [], [], [src_2, src_3], [])
        self._check_node(gd, dst_3, [], [], [src_3], [])

        disconnect_control_edge(gd, src_1, dst_1)
        disconnect_control_edge(gd, src_2, dst_3) # No-op
        disconnect_control_edge(gd, src_3, dst_3)

        self._check_node(gd, src_1, [], [], [], [])
        self._check_node(gd, src_2, [], [], [], [dst_1, dst_2])
        self._check_node(gd, src_3, [], [], [], [dst_1, dst_2])
        self._check_node(gd, dst_1, [], [], [src_2, src_3], [])
        self._check_node(gd, dst_2, [], [], [src_2, src_3], [])
        self._check_node(gd, dst_3, [], [], [], [])

        # Test string mode
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)
        dst_1.control_inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.control_inputs = self._add_inputs(src_2, src_3)
        dst_3.control_inputs = self._add_inputs(src_3)
        src_1.control_outputs = [dst_1.name]
        src_2.control_outputs = [dst_1.name, dst_2.name]
        src_3.control_outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)
        disconnect_control_edge(gd, src_1.name, dst_3.name) # No-op

        self._check_node(gd, src_1, [], [], [], [dst_1])
        self._check_node(gd, src_2, [], [], [], [dst_1, dst_2])
        self._check_node(gd, src_3, [], [], [], [dst_1, dst_2, dst_3])
        self._check_node(gd, dst_1, [], [], [src_1, src_2, src_3], [])
        self._check_node(gd, dst_2, [], [], [src_2, src_3], [])
        self._check_node(gd, dst_3, [], [], [src_3], [])

        disconnect_control_edge(gd, src_1.name, dst_1.name)
        disconnect_control_edge(gd, src_2.name, dst_3.name) # No-op
        disconnect_control_edge(gd, src_3.name, dst_3.name)

        self._check_node(gd, src_1, [], [], [], [])
        self._check_node(gd, src_2, [], [], [], [dst_1, dst_2])
        self._check_node(gd, src_3, [], [], [], [dst_1, dst_2])
        self._check_node(gd, dst_1, [], [], [src_2, src_3], [])
        self._check_node(gd, dst_2, [], [], [src_2, src_3], [])
        self._check_node(gd, dst_3, [], [], [], [])

    def test_disconnect_vertex_outs(self):
        # Test node version
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)
        dst_1.inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.inputs = self._add_inputs(src_2, src_3)
        dst_3.inputs = self._add_inputs(src_3)
        src_1.outputs = [dst_1.name]
        src_2.outputs = [dst_1.name, dst_2.name]
        src_3.outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)

        disconnect_vertex_outs(gd, src_3)

        self._check_node(gd, src_1, [], [dst_1], [], [])
        self._check_node(gd, src_2, [], [dst_1, dst_2], [], [])
        self._check_node(gd, src_3, [], [], [], [])
        self._check_node(gd, dst_1, [src_1, src_2], [], [], [])
        self._check_node(gd, dst_2, [src_2], [], [], [])
        self._check_node(gd, dst_3, [], [], [], [])

        # Test string version
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)
        dst_1.inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.inputs = self._add_inputs(src_2, src_3)
        dst_3.inputs = self._add_inputs(src_3)
        src_1.outputs = [dst_1.name]
        src_2.outputs = [dst_1.name, dst_2.name]
        src_3.outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)

        disconnect_vertex_outs(gd, src_3.name)

        self._check_node(gd, src_1, [], [dst_1], [], [])
        self._check_node(gd, src_2, [], [dst_1, dst_2], [], [])
        self._check_node(gd, src_3, [], [], [], [])
        self._check_node(gd, dst_1, [src_1, src_2], [], [], [])
        self._check_node(gd, dst_2, [src_2], [], [], [])
        self._check_node(gd, dst_3, [], [], [], [])

    def test_disconnect_vertex_ins(self):
        # Test node version
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)
        dst_1.inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.inputs = self._add_inputs(src_2, src_3)
        dst_3.inputs = self._add_inputs(src_3)
        src_1.outputs = [dst_1.name]
        src_2.outputs = [dst_1.name, dst_2.name]
        src_3.outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)

        disconnect_vertex_ins(gd, dst_1)

        self._check_node(gd, src_1, [], [], [], [])
        self._check_node(gd, src_2, [], [dst_2], [], [])
        self._check_node(gd, src_3, [], [dst_2, dst_3], [], [])
        self._check_node(gd, dst_1, [], [], [], [])
        self._check_node(gd, dst_2, [src_2, src_3], [], [], [])
        self._check_node(gd, dst_3, [src_3], [], [], [])

        # Test string version
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)
        dst_1.inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.inputs = self._add_inputs(src_2, src_3)
        dst_3.inputs = self._add_inputs(src_3)
        src_1.outputs = [dst_1.name]
        src_2.outputs = [dst_1.name, dst_2.name]
        src_3.outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)

        disconnect_vertex_ins(gd, dst_1.name)

        self._check_node(gd, src_1, [], [], [], [])
        self._check_node(gd, src_2, [], [dst_2], [], [])
        self._check_node(gd, src_3, [], [dst_2, dst_3], [], [])
        self._check_node(gd, dst_1, [], [], [], [])
        self._check_node(gd, dst_2, [src_2, src_3], [], [], [])
        self._check_node(gd, dst_3, [src_3], [], [], [])

    def test_disconnect_vertex_control_ins(self):
        # Test node version
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)
        dst_1.control_inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.control_inputs = self._add_inputs(src_2, src_3)
        dst_3.control_inputs = self._add_inputs(src_3)
        src_1.control_outputs = [dst_1.name]
        src_2.control_outputs = [dst_1.name, dst_2.name]
        src_3.control_outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)

        disconnect_vertex_control_ins(gd, dst_1)

        self._check_node(gd, src_1, [], [], [], [])
        self._check_node(gd, src_2, [], [], [], [dst_2])
        self._check_node(gd, src_3, [], [], [], [dst_2, dst_3])
        self._check_node(gd, dst_1, [], [], [], [])
        self._check_node(gd, dst_2, [], [], [src_2, src_3], [])
        self._check_node(gd, dst_3, [], [], [src_3], [])

        # Test string version
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)
        dst_1.control_inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.control_inputs = self._add_inputs(src_2, src_3)
        dst_3.control_inputs = self._add_inputs(src_3)
        src_1.control_outputs = [dst_1.name]
        src_2.control_outputs = [dst_1.name, dst_2.name]
        src_3.control_outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)

        disconnect_vertex_control_ins(gd, dst_1.name)

        self._check_node(gd, src_1, [], [], [], [])
        self._check_node(gd, src_2, [], [], [], [dst_2])
        self._check_node(gd, src_3, [], [], [], [dst_2, dst_3])
        self._check_node(gd, dst_1, [], [], [], [])
        self._check_node(gd, dst_2, [], [], [src_2, src_3], [])
        self._check_node(gd, dst_3, [], [], [src_3], [])

    def test_disconnect_vertex_control_outs(self):
        # Test node version
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)
        dst_1.control_inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.control_inputs = self._add_inputs(src_2, src_3)
        dst_3.control_inputs = self._add_inputs(src_3)
        src_1.control_outputs = [dst_1.name]
        src_2.control_outputs = [dst_1.name, dst_2.name]
        src_3.control_outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)

        disconnect_vertex_control_outs(gd, src_2)

        self._check_node(gd, src_1, [], [], [], [dst_1])
        self._check_node(gd, src_2, [], [], [], [])
        self._check_node(gd, src_3, [], [], [], [dst_1, dst_2, dst_3])
        self._check_node(gd, dst_1, [], [], [src_1, src_3], [])
        self._check_node(gd, dst_2, [], [], [src_3], [])
        self._check_node(gd, dst_3, [], [], [src_3], [])

        # Test string version
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)
        dst_1.control_inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.control_inputs = self._add_inputs(src_2, src_3)
        dst_3.control_inputs = self._add_inputs(src_3)
        src_1.control_outputs = [dst_1.name]
        src_2.control_outputs = [dst_1.name, dst_2.name]
        src_3.control_outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)

        disconnect_vertex_control_outs(gd, src_2.name)

        self._check_node(gd, src_1, [], [], [], [dst_1])
        self._check_node(gd, src_2, [], [], [], [])
        self._check_node(gd, src_3, [], [], [], [dst_1, dst_2, dst_3])
        self._check_node(gd, dst_1, [], [], [src_1, src_3], [])
        self._check_node(gd, dst_2, [], [], [src_3], [])
        self._check_node(gd, dst_3, [], [], [src_3], [])

    def test_delete_node(self):
        # Test node version
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)

        dst_1.inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.inputs = self._add_inputs(src_2, src_3)
        dst_3.inputs = self._add_inputs(src_3)
        src_1.outputs = [dst_1.name]
        src_2.outputs = [dst_1.name, dst_2.name]
        src_3.outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)

        delete_node(gd, dst_1)

        self._compare_items(gd, self._get_gd(src_1, src_2, src_3, dst_2, dst_3))
        self._check_node(gd, src_1, [], [], [], [])
        self._check_node(gd, src_2, [], [dst_2], [], [])
        self._check_node(gd, src_3, [], [dst_2, dst_3], [], [])
        self._check_node(gd, dst_2, [src_2, src_3], [], [], [])
        self._check_node(gd, dst_3, [src_3], [], [], [])

        # Test string version
        src_1, src_2, src_3, dst_1, dst_2, dst_3 = self._get_test_nodes_with_counts(6)

        dst_1.inputs = self._add_inputs(src_1, src_2, src_3)
        dst_2.inputs = self._add_inputs(src_2, src_3)
        dst_3.inputs = self._add_inputs(src_3)
        src_1.outputs = [dst_1.name]
        src_2.outputs = [dst_1.name, dst_2.name]
        src_3.outputs = [dst_1.name, dst_2.name, dst_3.name]

        gd = self._get_gd(src_1, src_2, src_3, dst_1, dst_2, dst_3)
        check_connections(gd)

        delete_node(gd, dst_1.name)

        self._compare_items(gd, self._get_gd(src_1, src_2, src_3, dst_2, dst_3))
        self._check_node(gd, src_1, [], [], [], [])
        self._check_node(gd, src_2, [], [dst_2], [], [])
        self._check_node(gd, src_3, [], [dst_2, dst_3], [], [])
        self._check_node(gd, dst_2, [src_2, src_3], [], [], [])
        self._check_node(gd, dst_3, [src_3], [], [], [])

    def test_replace_node(self):
        # Test node version
        node_1, node_2, node_3, node_4, node_5, replace = self._get_test_nodes_with_counts(6)

        node_1.outputs = [node_3.name]
        node_2.outputs = [node_3.name]
        node_3.inputs = self._add_inputs(node_1, node_2)
        node_3.outputs = [node_4.name, node_5.name]
        node_4.inputs = self._add_inputs(node_3)
        node_4.outputs = [node_5.name]
        node_5.inputs = self._add_inputs(node_3, node_4)

        gd = self._get_gd(node_1, node_2, node_3, node_4, node_5, replace)
        check_connections(gd)

        self._check_node(gd, node_1, [], [node_3], [], [])
        self._check_node(gd, node_2, [], [node_3], [], [])
        self._check_node(gd, node_3, [node_1, node_2], [node_4, node_5], [], [])
        self._check_node(gd, node_4, [node_3], [node_5], [], [])
        self._check_node(gd, node_5, [node_3, node_4], [], [], [])
        self._check_node(gd, replace, [], [], [], [])

        replace_node(gd, node_3, replace)

        self._check_node(gd, node_1, [], [replace], [], [])
        self._check_node(gd, node_2, [], [replace], [], [])
        self._check_node(gd, replace, [node_1, node_2], [node_4, node_5], [], [])
        self._check_node(gd, node_4, [replace], [node_5], [], [])
        self._check_node(gd, node_5, [replace, node_4], [], [], [])
        self._check_node(gd, node_3, [], [], [], [])

        # Test string version
        node_1, node_2, node_3, node_4, node_5, replace = self._get_test_nodes_with_counts(6)

        node_1.outputs = [node_3.name]
        node_2.outputs = [node_3.name]
        node_3.inputs = self._add_inputs(node_1, node_2)
        node_3.outputs = [node_4.name, node_5.name]
        node_4.inputs = self._add_inputs(node_3)
        node_4.outputs = [node_5.name]
        node_5.inputs = self._add_inputs(node_3, node_4)

        gd = self._get_gd(node_1, node_2, node_3, node_4, node_5, replace)
        check_connections(gd)

        replace_node(gd, node_3.name, replace.name)

        self._check_node(gd, node_1, [], [replace], [], [])
        self._check_node(gd, node_2, [], [replace], [], [])
        self._check_node(gd, replace, [node_1, node_2], [node_4, node_5], [], [])
        self._check_node(gd, node_4, [replace], [node_5], [], [])
        self._check_node(gd, node_5, [replace, node_4], [], [], [])
        self._check_node(gd, node_3, [], [], [], [])

if __name__ == '__main__':
    unittest.main()
