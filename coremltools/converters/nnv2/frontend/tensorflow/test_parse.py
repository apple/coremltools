# -*- coding: utf-8 -*-
import unittest

from tensorflow.core.framework import attr_value_pb2 as attr_value
from tensorflow.core.framework import tensor_shape_pb2 as tensor_shape
from tensorflow.core.framework import types_pb2 as types

from coremltools.converters.nnv2.builtin_types import builtins
import coremltools.converters.nnv2.frontend.tensorflow.parse as parse


class TestParse(unittest.TestCase):
    def test_parse_list(self):
        def compare(expected, lst, field_name):
            attr = attr_value.AttrValue()
            field = getattr(attr.list, field_name)
            field.extend(lst)

            actual = parse.parse_attr(attr)
            self.assertEqual(expected, actual)

        compare([1, 2, 3], [1, 2, 3], "i")
        compare(["foo", "bar"], [b"foo", b"bar"], "s")

    def test_parse_scalar(self):
        def compare(expected, val, field_name):
            a = attr_value.AttrValue()
            setattr(a, field_name, val)
            actual = parse.parse_attr(a)
            self.assertEqual(expected, actual)

        compare("a String", b"a String", "s")
        compare(55, 55, "i")
        compare(True, True, "b")

        attr = attr_value.AttrValue()
        attr.f = 12.3
        self.assertAlmostEqual(12.3, parse.parse_attr(attr), places=2)

    @staticmethod
    def _attr_with_shape(dims, unknown_rank=0):
        attr = attr_value.AttrValue()
        for (dim_size, dim_name) in dims:
            tf_dim = tensor_shape.TensorShapeProto.Dim()
            tf_dim.size = dim_size
            tf_dim.name = dim_name
            attr.shape.dim.append(tf_dim)
        attr.shape.unknown_rank = unknown_rank
        return attr

    def test_parse_shape(self):
        def compare(expected, dims, unknown_rank=0):
            attr = self._attr_with_shape(dims, unknown_rank)
            actual = parse.parse_attr(attr)
            self.assertEqual(expected, actual)

        compare(None, [], 5)
        compare([100], [(100, "outer")])
        compare([1, 2, 3], [(1, "outer"), (2, "middle"), (3, "inner")])

    def test_parse_tensor(self):
        # Zero-rank tensor
        attr = attr_value.AttrValue()
        attr.tensor.version_number = 1
        attr.tensor.dtype = types.DataType.DT_INT32
        t = parse.parse_attr(attr)
        self.assertTrue(isinstance(t, builtins.int32))
        self.assertEqual(0, t.val)

        # Non-zero rank
        attr = attr_value.AttrValue()
        attr.tensor.version_number = 1
        attr.tensor.dtype = types.DataType.DT_INT32
        shaped_attr = self._attr_with_shape([(1, "outer"), (2, "middle"), (3, "inner")])
        attr.tensor.tensor_shape.dim.extend(shaped_attr.shape.dim)
        attr.tensor.int_val.extend([55, 56, 57])

        t = parse.parse_attr(attr)
        self.assertEqual([55, 56, 57], t.val)
        self.assertEqual("tensor", builtins.get_type_info(t).name)

        # Note that the result of t.get_primitive() is a function that returns a type
        # rather than an instance of that type as it is when the tensor has rank zero.
        self.assertTrue(isinstance(t.get_primitive()(), builtins.int32))
        self.assertEqual((1, 2, 3), t.get_shape())

    def test_parse_type(self):
        def compare(expected, tf_type):
            attr = attr_value.AttrValue()
            attr.type = tf_type
            self.assertEqual(expected, parse.parse_attr(attr))

        compare(None, types.DataType.DT_INVALID)
        compare(builtins.float, types.DataType.DT_FLOAT)
        compare(builtins.double, types.DataType.DT_DOUBLE)
        compare(builtins.int32, types.DataType.DT_INT32)
        compare(builtins.uint8, types.DataType.DT_UINT8)
        compare(builtins.int16, types.DataType.DT_INT16)
        compare(builtins.int8, types.DataType.DT_INT8)
        compare(builtins.int8, types.DataType.DT_INT8)
        compare(builtins.str, types.DataType.DT_STRING)
        compare(None, types.DataType.DT_COMPLEX64)
        compare(builtins.int64, types.DataType.DT_INT64)
        compare(builtins.bool, types.DataType.DT_BOOL)
        compare(None, types.DataType.DT_QINT8)
        compare(None, types.DataType.DT_QUINT8)
        compare(None, types.DataType.DT_QINT32)
        compare(None, types.DataType.DT_BFLOAT16)
        compare(None, types.DataType.DT_QINT16)
        compare(None, types.DataType.DT_QUINT16)
        compare(builtins.uint16, types.DataType.DT_UINT16)
        compare(None, types.DataType.DT_COMPLEX128)
        compare(None, types.DataType.DT_HALF)
        compare(None, types.DataType.DT_RESOURCE)
        compare(None, types.DataType.DT_VARIANT)
        compare(builtins.uint32, types.DataType.DT_UINT32)
        compare(builtins.uint64, types.DataType.DT_UINT64)


if __name__ == '__main__':
    unittest.main()