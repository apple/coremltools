# -*- coding: utf-8 -*-

from coremltools.converters.nnssa.commons import builtins, symbolic
from coremltools.converters.nnssa.commons.parse import numpy_val_to_builtin_val
from coremltools.converters.nnssa.frontend.graph_pass.type_inference import TypeInferenceVisitor, graph_make_symbolic_values, make_symbol
from coremltools.converters.nnssa.frontend.tensorflow.parsed_tf_node import ParsedTFNode
import numpy as np
import sympy as sm
import unittest


class TestTypeInference(unittest.TestCase):
    def setUp(self):
        self._graph_dict = {}

    def _visit(self, node):
        visitor = TypeInferenceVisitor(self._graph_dict, None, pedantic=True)
        graph_make_symbolic_values(self._graph_dict)
        return visitor.visit(node)

    def _builtin_value(self, builtin_type, x):
        value = builtin_type()
        value.val = x
        return value

    def _builtin_bool_value(self, x):
        return self._builtin_value(builtins.bool, x)

    def _builtin_int32_value(self, x):
        return self._builtin_value(builtins.int32, x)

    def _const_node(self, name, type, value):
        node = self._node(name, type=type, op='Const')
        node.value, _ = numpy_val_to_builtin_val(value)
        return node

    def _node(self, name, type=builtins.int32, value=None, op='Const'):
        node = ParsedTFNode()
        node.name = name
        node.op = op
        node.datatype = type
        node.value = value
        self._graph_dict[name] = node
        return node

    def _node_with_inputs(self, name, inputs, op='Identity'):
        node = self._node(name, type=None, op=op)
        node.inputs = inputs
        return node

    def _node_with_size(self, name, input_type=builtins.int32, input_size=None, op='Const'):
        if input_size is None:
            input_size = [5000, 24, 24, 32]
        type = builtins.tensor(input_type, input_size)
        node = self._node(name, type=type, op=op)
        return node

    def _pooling_node(
            self,
            op,
            input_type=builtins.int32,
            input_size=None,
            data_format="NHWC",
            ksize=None,
            strides=None,
            padding="SAME"):
        if input_size is None:
            input_size = [5000, 24, 24, 32]
        if ksize is None:
            ksize = [1, 2, 2, 1]
        if strides is None:
            strides = [1, 1, 1, 1]
        input_node = self._node_with_size('x', input_type=input_type, input_size=input_size)

        node = self._node_with_inputs('node', ['x'], op=op)
        node.attr = {
            'data_format': data_format,
            'ksize': ksize,
            'strides': strides,
            'padding': padding,
        }

        return node

    def _symbolic_value(self, node):
        value = node.attr.get('symbolic_value', None)
        if value is None:
            return value
        return value.val

    def _test_batch_matmul(self, op):
        def make_node(x_size, y_size, x_type=builtins.int32, y_type=builtins.int32):
            x = self._node_with_size('x', input_size=x_size, input_type=x_type)
            y = self._node_with_size('y', input_size=y_size, input_type=y_type)
            node = self._node_with_inputs('node', inputs=['x', 'y'], op=op)
            return node

        # (*) Rank == 2
        node = make_node([2, 3], [3, 4])
        expected = builtins.tensor(builtins.int32, [2, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Rank > 2
        node = make_node([8, 9, 2, 3], [8, 9, 3, 4])
        expected = builtins.tensor(builtins.int32, [8, 9, 2, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Rank > 2 with broadcasting
        node = make_node([1, 2, 3], [8, 9, 3, 4])
        expected = builtins.tensor(builtins.int32, [8, 9, 2, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Transpose A
        node = make_node([3, 2], [3, 4])
        node.attr['transpose_a'] = True
        expected = builtins.tensor(builtins.int32, [2, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Transpose B
        node = make_node([2, 3], [4, 3])
        node.attr['transpose_b'] = True
        expected = builtins.tensor(builtins.int32, [2, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Adjunct A
        node = make_node([3, 2], [3, 4])
        node.attr['adj_x'] = True
        expected = builtins.tensor(builtins.int32, [2, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Adjunct B
        node.attr['adj_y'] = True
        expected = builtins.tensor(builtins.int32, [2, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Primitive promotion
        node = make_node([2, 3], [3, 4], x_type=builtins.int8, y_type=builtins.float)
        expected = builtins.tensor(builtins.float, [2, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Rank < 2
        node = make_node([1], [2, 3])
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Incompatible shapes
        node = make_node([2, 3], [2, 4])
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Both adj and transpose specified for A
        node = make_node([3, 2], [3, 4])
        node.attr['transpose_a'] = True
        node.attr['adj_x'] = True
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Both adj and transpose specified for B
        node = make_node([3, 2], [3, 4])
        node.attr['transpose_b'] = True
        node.attr['adj_y'] = True
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too few inputs
        node = make_node([2, 3], [3, 4])
        del node.inputs[1]
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too many inputs
        node = make_node([2, 3], [3, 4])
        node.inputs = node.inputs + [node.inputs[0]]
        with self.assertRaises(ValueError):
            self._visit(node)

    def _test_broadcast_op(self, op, is_predicate=False):
        def _get_primitive(if_not_predicate):
            return builtins.bool if is_predicate else if_not_predicate

        def _get_tensor(if_not_predicate):
            if is_predicate:
                return builtins.tensor(builtins.bool, if_not_predicate.get_shape())
            return if_not_predicate

        # tensor vs tensor (same size)
        x = self._node_with_size('x', input_size=[5, 1, 4])
        y = self._node_with_size('y', input_size=x.datatype.get_shape())
        node = self._node_with_inputs('node', ['x', 'y'], op=op)
        expected = _get_tensor(x.datatype)
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # tensor vs tensor with broadcasting
        # (https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics)
        x = self._node_with_size('x', input_size=[5, 1, 3, 1])
        y = self._node_with_size('y', input_size=[2, 3, 1])
        node = self._node_with_inputs('node', ['x', 'y'], op=op)
        expected = builtins.tensor(_get_primitive(x.datatype.get_primitive()), [5, 2, 3, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        x = self._node_with_size('x', input_size=[2, 3, 1])
        y = self._node_with_size('y', input_size=[5, 1, 3, 1])
        node = self._node_with_inputs('node', ['x', 'y'], op=op)
        expected = builtins.tensor(_get_primitive(x.datatype.get_primitive()), [5, 2, 3, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) tensor vs scalar
        x = self._node_with_size('x', input_size=[5, 1, 3, 1], input_type=builtins.int8)
        y = self._node('y', type=builtins.float, op=op)
        node = self._node_with_inputs('node', ['x', 'y'], op=op)
        expected = builtins.tensor(_get_primitive(y.datatype), x.datatype.get_shape())
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) scalar vs tensor
        x = self._node('x', type=builtins.int8, op=op)
        y = self._node_with_size('y', input_size=[5, 1, 3, 1], input_type=builtins.int64)
        node = self._node_with_inputs('node', ['x', 'y'], op=op)
        expected = _get_tensor(y.datatype)
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) scalar vs scalar
        x = self._node('x', type=builtins.int8, op=op)
        y = self._node('y', type=builtins.float, op=op)
        node = self._node_with_inputs('node', ['x', 'y'], op=op)
        expected = _get_primitive(y.datatype)
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Primitive type promotion
        x = self._node_with_size('x', input_type=builtins.int8, input_size=[5, 1, 4])
        y = self._node_with_size('y', input_type=builtins.int32, input_size=x.datatype.get_shape())
        node = self._node_with_inputs('node', ['x', 'y'], op=op)
        expected = _get_tensor(y.datatype)  # int8 --> int32
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Incompatible shapes (look at broadcasting examples for inspiration)
        x = self._node_with_size('x', input_size=[5, 2, 4, 1])
        y = self._node_with_size('y', input_size=[2, 3, 1])  # 3 vs 4 --> cannot broadcast
        node = self._node_with_inputs('node', ['x', 'y'], op=op)
        with self.assertRaises(ValueError):
            self._visit(node)

    def _test_reduce_op(self, op):
        # Scalar axis, keep_dims=False
        xs = self._node_with_size('xs', input_type=builtins.int32, input_size=[5, 3, 4])
        axis = self._const_node('axis', builtins.int32, 1)
        node = self._node_with_inputs('node', ['xs', 'axis'], op)
        expected = builtins.tensor(axis.datatype, [5, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # Tensor axis, keep_dims=False
        xs = self._node_with_size('xs', input_type=builtins.int32, input_size=[5, 3, 4])
        axis = self._const_node(
            'axis', builtins.tensor(builtins.int32, [1]), np.array([2, 1], dtype=np.int32))
        node = self._node_with_inputs('node', ['xs', 'axis'], op)
        expected = builtins.tensor(axis.datatype.get_primitive(), [5])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # Scalar axis, keep_dims=True
        xs = self._node_with_size('xs', input_type=builtins.int32, input_size=[5, 3, 4])
        axis = self._const_node('axis', builtins.int32, 1)
        node = self._node_with_inputs('node', ['xs', 'axis'], op)
        node.attr['keep_dims'] = True
        expected = builtins.tensor(axis.datatype, [5, 1, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # Tensor axis, keep_dims=True
        xs = self._node_with_size('xs', input_type=builtins.int32, input_size=[5, 3, 4])
        axis = self._const_node(
            'axis', builtins.tensor(builtins.int32, [1]), np.array([2, 1], dtype=np.int32))
        node = self._node_with_inputs('node', ['xs', 'axis'], op)
        node.attr['keep_dims'] = True
        expected = builtins.tensor(axis.datatype.get_primitive(), [5, 1, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # Reduce to scalar
        xs = self._node_with_size('xs', input_type=builtins.int32, input_size=[5])
        axis = self._const_node('axis', builtins.int32, 0)
        node = self._node_with_inputs('node', ['xs', 'axis'], op)
        expected = axis.datatype
        actual = self._visit(node)
        self.assertEqual(expected, actual)

    def _test_unary_op(self, op):
        # Tensor
        x = self._node_with_size('x', input_size=[5, 1, 4])
        node = self._node_with_inputs('node', ['x'], op=op)
        expected = x.datatype
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # Scalar
        x = self._node('x', type=builtins.int8, op=op)
        node = self._node_with_inputs('node', ['x'], op=op)
        expected = x.datatype
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # Too few inputs
        node = self._node_with_inputs('node', [], op=op)
        with self.assertRaises(ValueError):
            self._visit(node)

        # Too many inputs
        x = self._node('x', type=builtins.int8, op=op)
        y = self._node('y', type=builtins.int8, op=op)
        node = self._node_with_inputs('node', ['x', 'y'], op=op)
        with self.assertRaises(ValueError):
            self._visit(node)

    def test_Add(self):
        self._test_broadcast_op('Add')

    def test_ArgMax(self):
        self._test_reduce_op('ArgMax')

    def test_BatchMatMul(self):
        self._test_batch_matmul('BatchMatMul')

    def test_BatchMatMulV2(self):
        self._test_batch_matmul('BatchMatMulV2')

    def test_BiasAdd(self):
        self._test_broadcast_op('BiasAdd')

    def test_Cast(self):
        def _make_cast(dst_type):
            node = self._node_with_inputs('node', 'x', op='Cast')
            node.attr['DstT'] = dst_type
            return node

        # (*) change primitive type of tensor
        x = self._node_with_size('x', input_type=builtins.int8, input_size=[5, 1, 4])
        node = _make_cast(builtins.int16)
        expected = builtins.tensor(builtins.int16, x.datatype.get_shape())
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) cast primitive to another primitive
        x = self._node('x', type=builtins.int32)
        node = _make_cast(builtins.bool)
        expected = builtins.bool
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Bogus destination type (tensor)
        x = self._node_with_size('x', input_type=builtins.int8, input_size=[5, 1, 4])
        node = _make_cast(builtins.tensor(builtins.int8, [2, 4, 8]))
        with self.assertRaises(ValueError):
            self._visit(node)

    def test_ConcatV2(self):
        def _make_concatv2(size_0, size_1, axis):
            x = self._node_with_size('x', input_size=size_0)
            y = self._node_with_size('y', input_size=size_1)
            axis = self._node('axis', op='Const', value=self._builtin_int32_value(axis))
            node = self._node_with_inputs('node', ['x', 'y', 'axis'], op='ConcatV2')
            return x, y, node

        # (*) Rank 2 along axis 0
        x, _, node = _make_concatv2([1, 3], [2, 3], 0)
        expected = builtins.tensor(x.datatype.get_primitive(), [3, 3])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Rank 2 along axis 1
        x, _, node = _make_concatv2([2, 1], [2, 3], 1)
        expected = builtins.tensor(x.datatype.get_primitive(), [2, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Rank 2 along axis -1
        x, _, node = _make_concatv2([2, 1], [2, 3], -1)
        expected = builtins.tensor(x.datatype.get_primitive(), [2, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Primitive type mismatch
        x, _, node = _make_concatv2([1, 3], [2, 3], 0)
        x.datatype = builtins.tensor(builtins.bool, [1, 3])
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Tensors of different rank
        _, _, node = _make_concatv2([3], [2, 3], 0)
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Tensors with same rank, but incompatible dimensions
        _, _, node = _make_concatv2([1, 3], [2, 3], 1)
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) No tensor arguments
        axis = self._node('axis', op='Const', value=self._builtin_int32_value(0))
        node = self._node_with_inputs('node', ['axis'], op='ConcatV2')
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) No axis argument
        x = self._node_with_size('x', input_size=[1, 3])
        y = self._node_with_size('y', input_size=[1, 3])
        node = self._node_with_inputs('node', ['x', 'y'], op='ConcatV2')
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Multiple axis arguments
        x = self._node_with_size('x', input_size=[1, 3])
        y = self._node_with_size('y', input_size=[1, 3])
        axis1 = self._node('axis1', op='Const', value=self._builtin_int32_value(axis))
        axis2 = self._node('axis2', op='Const', value=self._builtin_int32_value(axis))
        node = self._node_with_inputs('node', ['x', 'y', 'axis1', 'axis2'], op='ConcatV2')
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Allowing tensor with same symbolic shape to be concat-ed
        sym_shape = make_symbol('sym_shape')
        x = self._node_with_size('x', input_size=[1, sym_shape, 3])
        y = self._node_with_size('y', input_size=[1, sym_shape, 3])
        axis = self._node('axis', op='Const', value=self._builtin_int32_value(-1))
        node = self._node_with_inputs('node', ['x', 'y', 'axis'], op='ConcatV2')
        actual = self._visit(node)
        expected = builtins.tensor(x.datatype.get_primitive(), [1, sym_shape, 6])
        self.assertEqual(expected, actual)

        # (*) Allowing tensor with different symbolic shape to be concat-ed
        sym_shape1 = make_symbol('sym_shape1')
        sym_shape2 = make_symbol('sym_shape2')
        x = self._node_with_size('x', input_size=[1, sym_shape1, 3])
        y = self._node_with_size('y', input_size=[1, sym_shape2, 3])
        axis = self._node('axis', op='Const', value=self._builtin_int32_value(-1))
        node = self._node_with_inputs('node', ['x', 'y', 'axis'], op='ConcatV2')
        actual = self._visit(node)
        expected = builtins.tensor(x.datatype.get_primitive(), [1, sym_shape1, 6])
        self.assertEqual(expected, actual)

    def test_Const(self):
        node = self._const_node('x', builtins.int32, 4)
        expected = node.datatype
        actual = self._visit(node)
        self.assertEqual(expected, actual)

    def test_Conv2D(self):
        def make_node(
                input_size=None,
                window_size=None,
                data_format='NHWC',
                padding='VALID',
                strides=None,
                dilations=None,
                pad=None):
            if input_size is None:
                input_size = [1, 32, 32, 1]
            if window_size is None:
                window_size = [1, 4, 4, 1]
            x = self._node_with_size('x', input_size=input_size)
            f = self._node_with_size('f', input_size=window_size)
            node = self._node_with_inputs('node', ['x', 'f'], op='Conv2D')
            node.attr['data_format'] = data_format
            node.attr['padding'] = padding
            node.attr['strides'] = strides
            node.attr['dilations'] = dilations
            node.attr['pad'] = pad
            return node

        # (*) VALID padding
        node = make_node(padding='VALID', input_size=[1, 6, 6, 1], window_size=[3, 3, 1, 1])
        expected = builtins.tensor(builtins.int32, [1, 4, 4, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        node = make_node(padding='VALID', input_size=[1, 6, 6, 1], window_size=[3, 5, 1, 1])
        expected = builtins.tensor(builtins.int32, [1, 4, 2, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) SAME padding
        node = make_node(padding='SAME', input_size=[1, 6, 6, 1], window_size=[3, 3, 1, 1])
        expected = builtins.tensor(builtins.int32, [1, 6, 6, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        node = make_node(padding='SAME', input_size=[1, 6, 6, 1], window_size=[3, 5, 1, 1])
        expected = builtins.tensor(builtins.int32, [1, 6, 6, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) CUSTOM padding
        node = make_node(
            padding='CUSTOM', input_size=[1, 6, 6, 1], window_size=[3, 3, 1, 1], pad=[0, 1, 0, 2])
        expected = builtins.tensor(builtins.int32, [1, 5, 6, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) NCHW
        node = make_node(
            padding='VALID', input_size=[1, 1, 6, 6], window_size=[3, 5, 1, 1], data_format='NCHW')
        expected = builtins.tensor(builtins.int32, [1, 1, 4, 2])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) strides == 1
        node = make_node(
            padding='VALID', input_size=[1, 6, 6, 1], window_size=[3, 3, 1, 1], strides=1)
        expected = builtins.tensor(builtins.int32, [1, 4, 4, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        node = make_node(
            padding='VALID', input_size=[1, 6, 6, 1], window_size=[3, 3, 1, 1], strides=[1, 1])
        expected = builtins.tensor(builtins.int32, [1, 4, 4, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        node = make_node(
            padding='VALID',
            input_size=[1, 6, 6, 1],
            window_size=[3, 3, 1, 1],
            strides=[1, 1, 1, 1])
        expected = builtins.tensor(builtins.int32, [1, 4, 4, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) strides != 1
        node = make_node(
            padding='VALID', input_size=[1, 6, 6, 1], window_size=[3, 3, 1, 1], strides=2)
        expected = builtins.tensor(builtins.int32, [1, 2, 2, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        node = make_node(
            padding='VALID', input_size=[1, 6, 6, 1], window_size=[3, 3, 1, 1], strides=[2, 2])
        expected = builtins.tensor(builtins.int32, [1, 2, 2, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        node = make_node(
            padding='VALID',
            input_size=[1, 6, 6, 1],
            window_size=[3, 3, 1, 1],
            strides=[1, 2, 2, 1])
        expected = builtins.tensor(builtins.int32, [1, 2, 2, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) dilations != 1
        node = make_node(dilations=[1, 2, 3, 2])
        with self.assertRaises(ValueError):
            self._visit(node)

        node = make_node(dilations=[1, 2, 3, 1])
        expected = builtins.tensor(builtins.int32, [1, 32, 23, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Wrong number of inputs
        node = make_node()
        node.inputs = ['x']
        with self.assertRaises(ValueError):
            self._visit(node)

        node = make_node()
        node.inputs = ['x', 'x', 'x']
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Input not a 4D tensor
        node = make_node(input_size=[1, 2, 3])
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Window not a 4D tensor
        node = make_node(window_size=[1, 2, 3])
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Invalid data format
        node = make_node(data_format='PANTS')
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Invalid padding
        node = make_node(padding='PANTS')
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Missing padding
        node = make_node(padding=None)
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) strides not a 1-, 2-, or 4-element list of non-negative ints
        node = make_node(strides=[1, 2, 3])
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) dilations not a 1-, 2-, or 4-element list of non-negative ints
        node = make_node(dilations=[1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) CUSTOM, but pad != 4 element list of non-negative ints
        node = make_node(padding='CUSTOM')
        with self.assertRaises(ValueError):
            self._visit(node)

        node = make_node(padding='CUSTOM', pad=[1, 2, 3, 4, 5])
        with self.assertRaises(ValueError):
            self._visit(node)

    def test_Cos(self):
        self._test_unary_op('Cos')

    def test_Elu(self):
        self._test_unary_op('Elu')

    def test_Equal(self):
        self._test_broadcast_op('Equal', is_predicate=True)

    def test_Exp(self):
        self._test_unary_op('Exp')

    def test_ExpandDims(self):
        def make_node(input_size, axis):
            x = self._node_with_size('x', input_size=input_size)
            axis = self._node('axis', value=axis)
            node = self._node_with_inputs('node', ['x', 'axis'], 'ExpandDims')
            return node

        # (*) Tensor input
        node = make_node([2, 4, 6], self._builtin_int32_value(1))
        expected = builtins.tensor(builtins.int32, [2, 1, 4, 6])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        node = make_node([2, 4, 6], self._builtin_int32_value(0))
        expected = builtins.tensor(builtins.int32, [1, 2, 4, 6])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        node = make_node([2, 4, 6], self._builtin_int32_value(3))
        expected = builtins.tensor(builtins.int32, [2, 4, 6, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Scalar input
        x = self._node('x')
        axis = self._node('axis', value=self._builtin_int32_value(1))
        node = self._node_with_inputs('node', ['x', 'axis'], 'ExpandDims')
        expected = builtins.tensor(builtins.int32, [1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Negative axis
        node = make_node([2, 4, 6], self._builtin_int32_value(-1))
        expected = builtins.tensor(builtins.int32, [2, 4, 6, 1])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        node = make_node([2, 4, 6], self._builtin_int32_value(-4))
        expected = builtins.tensor(builtins.int32, [1, 2, 4, 6])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Too few arguments
        node = make_node([2, 4, 6], self._builtin_int32_value(0))
        node.inputs = node.inputs[0]
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too many arguments
        node = make_node([2, 4, 6], self._builtin_int32_value(0))
        node.inputs = node.inputs + [node.inputs[0]]
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Axis not an int
        x = self._node_with_size('x', input_size=[2, 2])
        axis = self._node('axis', type=builtins.bool, value=self._builtin_bool_value(False))
        node = self._node_with_inputs('node', ['x', 'axis'], 'ExpandDims')
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Axis out of bounds
        node = make_node([2, 4, 6], self._builtin_int32_value(4))
        with self.assertRaises(IndexError):
            self._visit(node)

        node = make_node([2, 4, 6], self._builtin_int32_value(-5))
        with self.assertRaises(IndexError):
            self._visit(node)

    def test_Fill(self):
        def make_node(shape, value):
            shape_node = self._node_with_size('shape', input_size=[len(shape)])
            shape_dtype = np.int32 if all([isinstance(s, int) for s in shape]) else None
            shape_node.value = self._builtin_value(
                shape_node.datatype, np.array(shape, dtype=shape_dtype))
            value_node = self._node('value', value=value)
            node = self._node_with_inputs('node', ['shape', 'value'], op='Fill')
            expected_type = builtins.tensor(builtins.int32, shape)
            return node, expected_type

        # (*) Shape and value are compile time constants
        node, expected_type = make_node([2, 3], self._builtin_int32_value(55))
        actual_type = self._visit(node)
        self.assertEqual(expected_type, actual_type)
        self.assertEqual((2, 3), self._symbolic_value(node).shape)
        self.assertEqual(55, self._symbolic_value(node)[0][0])

        # (*) Value is symbolic
        value = sm.Symbol('aValue')
        node, expected_type = make_node([2, 3], self._builtin_int32_value(value))
        actual_type = self._visit(node)
        self.assertEqual(expected_type, actual_type)
        self.assertEqual((2, 3), self._symbolic_value(node).shape)
        self.assertEqual(value, self._symbolic_value(node)[0][0])

        # (*) Shape has a symbolic value (we don't infer a symbolic value for node)
        dim0 = sm.Symbol('dim0')
        node, expected_type = make_node([dim0, 3], self._builtin_int32_value(55))
        actual_type = self._visit(node)
        self.assertEqual(expected_type, actual_type)

        # (*) Too few inputs
        node, _ = make_node([2, 3], self._builtin_int32_value(55))
        del node.inputs[1]
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too many inputs
        node, _ = make_node([2, 3], self._builtin_int32_value(55))
        node.inputs = node.inputs + [node.inputs[0]]
        with self.assertRaises(ValueError):
            self._visit(node)

    def test_Floor(self):
        self._test_unary_op('Floor')

    def test_FloorDiv(self):
        self._test_broadcast_op('FloorDiv')

    def test_FloorMod(self):
        self._test_broadcast_op('FloorMod')

    def test_Greater(self):
        self._test_broadcast_op('Greater', is_predicate=True)

    def test_GreaterEqual(self):
        self._test_broadcast_op('GreaterEqual', is_predicate=True)

    def test_Identity(self):
        self._test_unary_op('Identity')

    def test_LeakyRelu(self):
        self._test_unary_op('LeakyRelu')

    def test_Less(self):
        self._test_broadcast_op('Less', is_predicate=True)

    def test_LessEqual(self):
        self._test_broadcast_op('LessEqual', is_predicate=True)

    def test_Log(self):
        self._test_unary_op('Log')

    def test_LogicalAnd(self):
        self._test_broadcast_op('LogicalAnd')

    def test_LogicalOr(self):
        self._test_broadcast_op('LogicalOr')

    def test_LogSoftmax(self):
        self._test_unary_op('LogSoftmax')

    def test_MatMul(self):
        self._test_batch_matmul('MatMul')

    def test_MaxPool(self):
        # NHWC, SAME, strides = 1...
        node = self._pooling_node("MaxPool", padding='SAME', data_format='NHWC', ksize=2)
        actual = self._visit(node)
        self.assertEqual(self._graph_dict[node.inputs[0]].datatype, actual)

        # NCHW, SAME, strides = 1...
        node = self._pooling_node(
            "MaxPool",
            padding='SAME',
            data_format='NCHW',
            input_size=[5000, 32, 24, 24],
            ksize=[2, 2])
        actual = self._visit(node)
        self.assertEqual(self._graph_dict[node.inputs[0]].datatype, actual)

        # NHWC, SAME, stride = window size, input is multiple of window size
        node = self._pooling_node(
            "MaxPool", padding='SAME', data_format='NHWC', ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1])
        expected = builtins.tensor(builtins.int32, [5000, 4, 4, 32])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # NCHW, SAME, stride = window size, input is multiple of window size
        node = self._pooling_node(
            "MaxPool",
            padding="SAME",
            data_format='NCHW',
            input_size=[5000, 32, 24, 24],
            ksize=[1, 1, 6, 6],
            strides=[1, 1, 6, 6])
        expected = builtins.tensor(builtins.int32, [5000, 32, 4, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # NHWC, SAME, stride = window size, input is not a multiple of window size; padding
        # will be added to avoid discarding elements in the input
        node = self._pooling_node(
            "MaxPool", padding="SAME", data_format='NHWC', ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1])
        expected = builtins.tensor(builtins.int32, [5000, 4, 4, 32])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # NHWC, SAME, stride != window size, input is not multiple of window size, but stride
        # is such that it'll cover the entire input
        node = self._pooling_node(
            "MaxPool",
            padding="SAME",
            data_format='NHWC',
            input_size=[5000, 10, 10, 32],
            ksize=[1, 4, 4, 1],
            strides=[1, 3, 3, 1])
        expected = builtins.tensor(builtins.int32, [5000, 4, 4, 32])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # NHWC, VALID, strides = 1...
        node = self._pooling_node("MaxPool", padding='VALID', data_format='NHWC')
        expected = builtins.tensor(builtins.int32, [5000, 23, 23, 32])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # NCHW, VALID, strides = 1...
        node = self._pooling_node(
            "MaxPool",
            padding='VALID',
            data_format='NCHW',
            input_size=[5000, 32, 24, 24],
            ksize=[1, 1, 2, 2])
        expected = builtins.tensor(builtins.int32, [5000, 32, 23, 23])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # NHWC, VALID, stride = window size, input is multiple of window size
        node = self._pooling_node(
            "MaxPool",
            padding="VALID",
            data_format='NHWC',
            ksize=[1, 6, 6, 1],
            strides=[1, 6, 6, 1])
        expected = builtins.tensor(builtins.int32, [5000, 4, 4, 32])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # NCHW, VALID, stride = window size, input is multiple of window size
        node = self._pooling_node(
            "MaxPool",
            padding="VALID",
            data_format='NCHW',
            input_size=[5000, 32, 24, 24],
            ksize=[1, 1, 6, 6],
            strides=[1, 1, 6, 6])
        expected = builtins.tensor(builtins.int32, [5000, 32, 4, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # NHWC, VALID, stride = window size, input is not multiple of window size; without
        # additional padding, elements will be discarded.
        node = self._pooling_node(
            "MaxPool",
            padding="VALID",
            data_format='NHWC',
            ksize=[1, 7, 7, 1],
            strides=[1, 7, 7, 1])
        expected = builtins.tensor(builtins.int32, [5000, 3, 3, 32])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # NHWC, VALID, stride != window size, input is not multiple of window size, but stride
        # is such that it'll cover the entire input
        node = self._pooling_node(
            "MaxPool",
            padding="VALID",
            data_format='NHWC',
            input_size=[5000, 10, 10, 32],
            ksize=[1, 4, 4, 1],
            strides=[1, 3, 3, 1])
        expected = builtins.tensor(builtins.int32, [5000, 3, 3, 32])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Too few arguments
        node = self._pooling_node('MaxPool')
        node.inputs = []
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too many arguments
        node = self._pooling_node('MaxPool')
        node.inputs = [node.inputs] * 2
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Invalid data_format
        node = self._pooling_node('MaxPool', data_format='PANTS')
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Invalid padding
        node = self._pooling_node('MaxPool', padding='PANTS')
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Invalid ksize
        node = self._pooling_node('MaxPool', ksize=[1, 2, 3])
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Missing/invalid strides
        node = self._pooling_node('MaxPool', strides=[1, 2, 3])
        with self.assertRaises(ValueError):
            self._visit(node)

        node = self._pooling_node('MaxPool', strides='yes, please')
        with self.assertRaises(ValueError):
            self._visit(node)

    def test_Max(self):
        self._test_reduce_op('Max')

    def test_Maximum(self):
        self._test_broadcast_op('Maximum')

    def test_Mean(self):
        self._test_reduce_op('Mean')

    def test_Minimum(self):
        self._test_broadcast_op('Minimum')

    def test_Min(self):
        self._test_reduce_op('Min')

    def test_Mul(self):
        self._test_broadcast_op('Mul')

    def test_Neg(self):
        self._test_unary_op('Neg')

    def test_NotEqual(self):
        self._test_broadcast_op('NotEqual', is_predicate=True)

    def test_OneHot(self):
        def make_node(
                indices, depth, axis, dtype=None, depth_type=None, on_type=None, off_type=None):
            if depth_type is None:
                depth_type = builtins.int32
            if on_type is None:
                on_type = builtins.int32
            if off_type is None:
                off_type = builtins.int32
            indices = self._node_with_size('indices', input_size=indices)
            depth = self._node('depth', type=depth_type, value=depth)
            on_type = self._node('on_type', type=on_type)
            off_type = self._node('off_type', type=off_type)
            node = self._node_with_inputs(
                'node', ['indices', 'depth', 'on_type', 'off_type'], op='OneHot')
            node.attr['axis'] = axis
            if dtype is not None:
                node.attr['dtype'] = True
                node.attr['T'] = dtype
            return node, indices

        node, _ = make_node([2, 2], self._builtin_int32_value(3), -1)
        expected = builtins.tensor(builtins.int32, [2, 2, 3])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        node, _ = make_node([2, 2], self._builtin_int32_value(3), 0)
        expected = builtins.tensor(builtins.int32, [3, 2, 2])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        node, _ = make_node([2, 2], self._builtin_int32_value(3), 1)
        expected = builtins.tensor(builtins.int32, [2, 3, 2])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        node, _ = make_node([0], self._builtin_int32_value(3), 1)
        expected = builtins.tensor(builtins.int32, [0, 3])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Zero rank indices
        node, _ = make_node([], self._builtin_int32_value(3), 1)
        expected = builtins.tensor(builtins.int32, [])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Manually specified dtype
        node, _ = make_node([2, 2], self._builtin_int32_value(3), 1, dtype=builtins.int64)
        expected = builtins.tensor(builtins.int64, [2, 3, 2])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Scalar indices
        # This behavior differs from TensorFlow
        # (*) TF: "If indices is a scalar the output shape will be a vector of length depth"
        # (*) Nitro: Output shape is [1, len(depth)]
        node, indices = make_node([1], self._builtin_int32_value(3), -1)
        indices.datatype = builtins.int32
        expected = builtins.tensor(builtins.int32, [1, 3])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Non-integral depth
        node, _ = make_node([2, 2], self._builtin_int32_value(3), 1, depth_type=builtins.float)
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Depth < 0
        node, _ = make_node([2, 2], self._builtin_int32_value(-1), -1)
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) axis < -1
        node, _ = make_node([2, 2], self._builtin_int32_value(3), -2)
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Non-matching on and off value types
        node, _ = make_node([2, 2],
                            self._builtin_int32_value(3),
                            -1,
                            on_type=builtins.int32,
                            off_type=builtins.int64)
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Non-primitive on/off value types
        on_off_type = builtins.tensor(builtins.int32, [1])
        node, _ = make_node([2, 2],
                            self._builtin_int32_value(3),
                            -1,
                            on_type=on_off_type,
                            off_type=on_off_type)
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Missing axis attribute
        node, _ = make_node([2, 2], self._builtin_int32_value(3), None)
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) dtype attribute with missing T attribute
        node, _ = make_node([2, 2], self._builtin_int32_value(3), 1, dtype=builtins.int64)
        del node.attr['T']
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) dtype attribute with non-primitive T attribute
        node, _ = make_node([2, 2],
                            self._builtin_int32_value(3),
                            1,
                            dtype=builtins.tensor(builtins.int64, [1]))
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too few inputs
        node, _ = make_node([2, 2], self._builtin_int32_value(3), -1)
        del node.inputs[3]
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too many inputs
        node, _ = make_node([2, 2], self._builtin_int32_value(3), -1)
        node.inputs += [node.inputs[0]]
        with self.assertRaises(ValueError):
            self._visit(node)

    def test_PRelu(self):
        # We rely on type inference to evaluate PRelu's alpha parameter when it's specified as node

        # (*) alpha node as scalar
        alpha = self._node('alpha', type=builtins.double)
        alpha.value = builtins.double()
        alpha.value.val = 0.2
        x = self._node_with_size('x', input_size=[2, 3, 1])
        node = self._node_with_inputs('node', inputs=['x'], op='PRelu')
        node.attr['alpha'] = 'alpha'
        expected = x.datatype
        actual = self._visit(node)
        self.assertEqual(expected, actual)
        self.assertEqual(node.attr['alpha'], 0.2)

        # (*) alpha node as ndarray
        alpha = self._node_with_size('alpha', input_type=builtins.double, input_size=[1])
        alpha.value = builtins.tensor(builtins.double, [1])
        alpha.value.val = np.array(0.2)
        x = self._node_with_size('x', input_size=[2, 3, 1])
        node = self._node_with_inputs('node', inputs=['x'], op='PRelu')
        node.attr['alpha'] = 'alpha'
        expected = x.datatype
        actual = self._visit(node)
        self.assertEqual(expected, actual)
        self.assertEqual(node.attr['alpha'], 0.2)

        # (*) Fail: alpha node as ndarray with more than one element
        alpha = self._node_with_size('alpha', input_type=builtins.double, input_size=[1])
        alpha.value = builtins.tensor(builtins.double, [2])
        alpha.value.val = np.array([0.2, 0.3])
        x = self._node_with_size('x', input_size=[2, 3, 1])
        node = self._node_with_inputs('node', inputs=['x'], op='PRelu')
        node.attr['alpha'] = 'alpha'
        with self.assertRaises(ValueError):
            self._visit(node)

        # Generic unary op tests
        self._test_unary_op('PRelu')

    def test_Pow(self):
        self._test_broadcast_op('Pow')

    def test_Prod(self):
        self._test_reduce_op('Prod')

    def test_RandomUniform(self):
        def make_node(shape, dtype=builtins.float, value=None):
            if value is None:
                x = self._node('x', builtins.tensor(builtins.int32, shape))
            else:
                x = self._const_node('x', builtins.tensor(builtins.int32, shape), value)
            node = self._node_with_inputs('node', ['x'], op='RandomUniform')
            if dtype is not None:
                node.attr['dtype'] = dtype
            return node

        # (*) Input with shape known at compile time
        node = make_node([2], value=np.array([2, 3]))
        expected = builtins.tensor(builtins.float, [2, 3])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Input without known shape
        node = make_node([2])
        actual = self._visit(node)
        self.assertTrue(builtins.is_tensor(actual))
        self.assertEqual(builtins.float, actual.get_primitive())
        self.assertEqual(1, len(actual.get_shape()))
        self.assertTrue(symbolic.is_symbolic(actual.get_shape()[0]))

        # (*) Wrong shape input
        node = make_node([2, 2])
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Non-tensor input
        x = self._const_node('x', builtins.int32, value=np.int(4))
        node = self._node_with_inputs('node', ['x'], op='RandomUniform')
        node.attr['dtype'] = builtins.float
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Missing dtype
        node = make_node([2], dtype=None)
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Non-scalar dtype
        node = make_node([2], dtype=builtins.str)
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too few inputs
        node = make_node([2])
        del node.inputs[0]
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too many inputs
        node = make_node([2])
        node.inputs = node.inputs + [node.inputs[0]]
        with self.assertRaises(ValueError):
            self._visit(node)

    def test_Range(self):
        def make_node(start=None, limit=5, delta=1):
            inputs = []
            if start is not None:
                self._node('start', value=self._builtin_int32_value(start))
                inputs = ['start']
            limit = self._node('limit', value=self._builtin_int32_value(limit))
            delta = self._node('delta', value=self._builtin_int32_value(delta))
            inputs += ['limit', 'delta']
            node = self._node_with_inputs('node', inputs, op='Range')
            return node

        # (*) limit, delta
        node = make_node(limit=5, delta=1)
        expected_type = builtins.tensor(builtins.int32, [5])
        expected_value = [0, 1, 2, 3, 4]
        actual_type = self._visit(node)
        actual_value = self._symbolic_value(node)
        self.assertEqual(expected_type, actual_type)
        self.assertSequenceEqual(expected_value, list(actual_value))

        # (*) start, limit, delta
        node = make_node(start=2, limit=5, delta=2)
        expected_type = builtins.tensor(builtins.int32, [2])
        expected_value = [2, 4]
        actual_type = self._visit(node)
        actual_value = self._symbolic_value(node)
        self.assertEqual(expected_type, actual_type)
        self.assertSequenceEqual(expected_value, list(actual_value))

        # (*) Single value tensor
        limit = self._node('limit', value=self._builtin_int32_value(5))
        delta_type = builtins.tensor(builtins.int32, [1])
        delta_value = self._builtin_value(delta_type, np.array([2]))
        delta = self._node('delta', type=delta_type, value=delta_value)
        node = self._node_with_inputs('node', ['limit', 'delta'], op='Range')
        expected_type = builtins.tensor(builtins.int32, [3])
        expected_value = [0, 2, 4]
        actual_type = self._visit(node)
        actual_value = self._symbolic_value(node)
        self.assertEqual(expected_type, actual_type)
        self.assertSequenceEqual(expected_value, list(actual_value))

        # (*) Non-single value tensor
        limit = self._node('limit', value=self._builtin_int32_value(5))
        delta_type = builtins.tensor(builtins.int32, [2])
        delta_value = self._builtin_value(delta_type, np.array([2, 3]))
        delta = self._node('delta', type=delta_type, value=delta_value)
        node = self._node_with_inputs('node', ['limit', 'delta'], op='Range')
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Datatype with mismatched input types
        limit_type = builtins.int64
        limit = self._node('limit', type=limit_type, value=self._builtin_value(limit_type, 4))
        delta_type = builtins.fp32
        delta = self._node('delta', type=delta_type, value=self._builtin_value(delta_type, 1))
        node = self._node_with_inputs('node', ['limit', 'delta'], op='Range')
        expected_type = builtins.tensor(builtins.fp32, [4])
        expected_value = [0, 1, 2, 3]
        actual_type = self._visit(node)
        actual_value = self._symbolic_value(node)
        self.assertEqual(expected_type, actual_type)
        self.assertEqual(np.float32, actual_value.dtype)
        self.assertSequenceEqual(expected_value, list(actual_value))

        # (*) Too few args
        node = make_node()
        del node.inputs[-1]
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too many args
        node = make_node()
        node.inputs = node.inputs * 2
        with self.assertRaises(ValueError):
            self._visit(node)

    def test_RealDiv(self):
        self._test_broadcast_op('RealDiv')

    def test_Relu(self):
        self._test_unary_op('Relu')

    def test_return(self):
        self._test_unary_op('return')

    def test_Rsqrt(self):
        self._test_unary_op('Rsqrt')

    def test_Selu(self):
        self._test_unary_op('Selu')

    def test_Sigmoid(self):
        self._test_unary_op('Sigmoid')

    def test_Sin(self):
        self._test_unary_op('Sin')

    def test_Softmax(self):
        self._test_unary_op('Softmax')

    def test_Sqare(self):
        self._test_unary_op('Square')

    def test_Sqrt(self):
        self._test_unary_op('Sqrt')

    def test_SquaredDifference(self):
        self._test_broadcast_op('SquaredDifference')

    def test_StopGradient(self):
        self._test_unary_op('StopGradient')

    def test_Sub(self):
        self._test_broadcast_op('Sub')

    def test_Sum(self):
        self._test_reduce_op('Sum')

    def test_Tile(self):
        def make_node(input_size, tile_value):
            x = self._node_with_size('x', input_size=input_size)
            tile = self._node_with_size('tile', input_size=[len(input_size)])
            if tile_value is not None:
                tile_value_dtype = np.int32 if all([isinstance(tv, int)
                                                    for tv in tile_value]) else None
                tile.value = self._builtin_value(
                    tile.datatype, np.array(tile_value, dtype=tile_value_dtype))
            node = self._node_with_inputs('node', ['x', 'tile'], op='Tile')
            return node

        # (*) 1-D input
        node = make_node([4], [2])
        expected = builtins.tensor(builtins.int32, [8])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) 3-D input
        node = make_node([1, 4, 2], [4, 1, 2])
        expected = builtins.tensor(builtins.int32, [4, 4, 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Tile has no symbolic value
        node = make_node([1, 4, 2], None)
        expected = builtins.tensor(
            builtins.int32, [make_symbol('node_0'),
                             make_symbol('node_1'),
                             make_symbol('node_2')])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Symbolic tile
        node = make_node([1, 4, 2], [4, sm.Symbol('tile1'), 2])
        expected = builtins.tensor(builtins.int32, [4, 4 * sm.Symbol('tile1'), 4])
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Tiles not the same length as rank(input)
        # With symbolic value for tile
        node = make_node([4], [2, 1])
        with self.assertRaises(ValueError):
            self._visit(node)

        # Without symbolic value for tile
        node = make_node([4], None)
        tile = self._node_with_size('tile', input_size=[2])
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Tiles not a 1-D tensor
        x = self._node_with_size('x', input_size=[2, 2, 2])
        tile = self._node_with_size('tile', input_size=[2, 2, 2])
        node = self._node_with_inputs('node', ['x', 'tile'], op='Tile')
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too few inputs
        node = make_node([1, 4, 2], [4, 1, 2])
        del node.inputs[1]
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too many inputs
        node = make_node([1, 4, 2], [4, 1, 2])
        node.inputs = node.inputs + [node.inputs[0]]
        with self.assertRaises(ValueError):
            self._visit(node)

    def test_TopKV2(self):
        def make_node(x_size, k, k_type=builtins.int32):
            k = self._builtin_int32_value(k)
            x = self._node_with_size('x', input_size=x_size, input_type=builtins.float)
            k = self._node('k', type=k_type, value=k)
            node = self._node_with_inputs('node', inputs=['x', 'k'], op='TopKV2')
            return node

        # (*) 1-D tensor
        node = make_node([20], 5)
        expected = builtins.tuple(
            (builtins.tensor(builtins.float, [5]), builtins.tensor(builtins.int32, [5])))
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) 3-D tensor
        node = make_node([2, 4, 20], 5)
        expected = builtins.tuple((
            builtins.tensor(builtins.float, [2, 4, 5]), builtins.tensor(builtins.int32, [2, 4, 5])))
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) Symbolic K
        node = make_node([2, 4, 20], sm.Symbol('k'))
        expected = builtins.tuple((
            builtins.tensor(builtins.float, [2, 4, sm.Symbol('k')]),
            builtins.tensor(builtins.int32, [2, 4, sm.Symbol('k')])))
        actual = self._visit(node)
        self.assertEqual(expected, actual)

        # (*) K > len(x)
        node = make_node([20], 21)
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Tensor input not a tensor
        x = self._node('x', value=self._builtin_int32_value(2))
        k = self._node('k', value=self._builtin_int32_value(2))
        node = self._node_with_inputs('node', inputs=['x', 'k'], op='TopKV2')
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) K input not an int
        node = make_node([20], 21, k_type=builtins.float)
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too few inputs
        node = make_node([20], 5)
        del node.inputs[1]
        with self.assertRaises(ValueError):
            self._visit(node)

        # (*) Too many inputs
        node = make_node([20], 5)
        node.inputs = node.inputs + [node.inputs[0]]
        with self.assertRaises(ValueError):
            self._visit(node)


if __name__ == '__main__':
    unittest.main()
