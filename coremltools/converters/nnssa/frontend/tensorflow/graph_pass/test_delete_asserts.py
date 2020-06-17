# -*- coding: utf-8 -*-
from coremltools._deps import _HAS_TF, MSG_TF1_NOT_FOUND
import coremltools.converters.nnssa.commons.builtins as builtins
import coremltools.converters.nnssa.builder as builder
import numpy as np
import unittest
if _HAS_TF:
    from coremltools.converters.nnssa.frontend.tensorflow.graph_pass.delete_asserts import delete_asserts

@unittest.skipUnless(_HAS_TF, MSG_TF1_NOT_FOUND)
class TestDeleteAsserts(unittest.TestCase):
    def _add_assert(self, builder, input, name='assert', op='Assert'):
        node = builder.add_identity(input, name=name)
        builder.graph[node].op = op
        return node

    def _check(self, ssa, expected_num_deleted, expected_remaining=None):
        actual_num_deleted = delete_asserts(ssa)
        graph = ssa.functions['main'].graph

        self.assertEqual(expected_num_deleted, actual_num_deleted)
        if expected_remaining is not None:
            self.assertEqual(len(expected_remaining), len(graph))
            for n in expected_remaining:
                self.assertIn(n, graph)

    def test_deep(self):
        """Very deep graph: we should not blow the stack"""
        b = builder.GraphBuilder()
        b.add_placeholder(name='placeholder', datatype=builtins.tensor(builtins.fp32, [3]))
        b.add_const(np.array([1, 2, 3], dtype=np.float), 'const')
        b.add_elementwise('add', ['placeholder', 'const'], 'add')
        b.add_make_tuple(['placeholder', 'const'], 'make_tuple')

        depth = 1000
        b.add_identity('make_tuple', 'identity_0')
        for i in range(1, depth):
            b.add_identity('identity_{}'.format(i - 1), name='identity_{}'.format(i))
        self._add_assert(b, 'identity_{}'.format(depth - 1))

        expected_num_deleted = depth + 2  # all identities + make_tuple + assert
        self._check(b.get_ssa(), expected_num_deleted, ['placeholder', 'const', 'add'])

    def test_loop(self):
        """Check that delete_asserts terminates with a loop in the input"""
        main = builder.GraphBuilder(prefix='main_')
        i = main.add_const(np.int32(0), name="i")
        target = main.add_placeholder(init_from=np.int32(5), name="target")
        mt = main.add_make_tuple([target, i], name="make_tuple_0")
        loop = main.add_while(mt, "body_function_0", "cond_function_0", name="loop")
        loop_out = main.add_get_tuple(loop, index=1, name="out")
        main.add_identity(loop_out)
        self._add_assert(main, loop_out)

        body = builder.GraphBuilder(prefix='body_')
        b_entry = body.add_function_entry(name="body_function_0")
        add_one = body.add_const(np.int32(1), name="one")
        to_add = body.add_get_tuple(b_entry, index=1)
        target = body.add_get_tuple(b_entry, index=0)
        added = body.add_elementwise("Add", [to_add, add_one])
        ret = body.add_make_tuple([target, added])
        body.add_return(ret)

        cond = builder.GraphBuilder(prefix='cond_')
        c_entry = cond.add_function_entry(name="cond_function_0")
        now = cond.add_get_tuple(c_entry, index=1)
        target = cond.add_get_tuple(c_entry, index=0)
        stop = cond.add_elementwise("Less", [now, target], name="cond")
        cond.add_return(stop)

        ssa_builder = builder.SSABuilder()
        ssa_builder.add_graph(main.get_graph())
        ssa_builder.add_function(body.get_function(), name="body_function_0")
        ssa_builder.add_function(cond.get_function(), name="cond_function_0")
        ssa = ssa_builder.get_ssa()

        self._check(ssa, 1)

    def test_simple(self):
        """Happy path: () -> make_tuple -> assert deletes make_tuple and assert"""
        for assert_op in ('Assert', 'CheckNumerics'):
            b = builder.GraphBuilder()
            b.add_placeholder(name='placeholder', datatype=builtins.tensor(builtins.fp32, [3]))
            b.add_const(np.array([1, 2, 3], dtype=np.float), 'const')
            b.add_elementwise('add', ['placeholder', 'const'], 'add')
            b.add_make_tuple(['placeholder', 'const'], 'make_tuple')
            self._add_assert(b, 'make_tuple', op=assert_op)
            self._check(b.get_ssa(), 2, ['placeholder', 'const', 'add'])

    def test_trivial(self):
        b = builder.GraphBuilder()
        n = b.add_placeholder(name='assert', datatype=builtins.tensor(builtins.fp32, [3]))
        b.graph[n].op = 'Assert'
        self._check(b.get_ssa(), 1, [])


if __name__ == '__main__':
    unittest.main()
