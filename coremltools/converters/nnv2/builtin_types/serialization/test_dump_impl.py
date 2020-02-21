# -*- coding: utf-8 -*-
from coremltools.converters.nnv2.builtin_types.serialization import dump_impl as di
from coremltools.converters.nnv2.builtin_types.serialization import typedefs as t
from collections import deque
import numpy as np
import unittest


class WriterEvent:
    __slots__ = ['type', 'value']

    WRITE_BOOL = 1
    WRITE_BYTE = 2
    WRITE_INT = 3
    WRITE_DOUBLE = 4
    WRITE_STR = 5

    def __init__(self, type, value):
        self.type = type
        self.value = value


class MockWriter:
    def __init__(self):
        self.events = deque()

    def write_bool(self, i):
        self.events.append(WriterEvent(WriterEvent.WRITE_BOOL, i))

    def write_byte(self, i):
        self.events.append(WriterEvent(WriterEvent.WRITE_BYTE, i))

    def write_int(self, i):
        self.events.append(WriterEvent(WriterEvent.WRITE_INT, i))

    def write_double(self, i):
        self.events.append(WriterEvent(WriterEvent.WRITE_DOUBLE, i))

    def write_str(self, i):
        self.events.append(WriterEvent(WriterEvent.WRITE_STR, i))


class TestDumpImpl(unittest.TestCase):
    def setUp(self):
        self.writer = MockWriter()

    def tearDown(self):
        # Assert all events were consumed
        self.assertTrue(len(self.writer.events) == 0)
        self.writer = None

    def _assert_event(self, expected_type, expected_value):
        actual_event = self.writer.events.popleft()
        self.assertEqual(expected_type, actual_event.type)
        if expected_value is not None:
            self.assertEqual(expected_value, actual_event.value)

    def _assert_scalar(self, expected_event_type, expected_scalar_type,
                       expected_scalar_value):
        self._assert_event(WriterEvent.WRITE_BYTE, expected_scalar_type)
        self._assert_event(expected_event_type, expected_scalar_value)

    def _test_scalar_type(self, cases, expected_event_type,
                          expected_value_type):
        for case in cases:
            di.dump_obj(case, self.writer)
        self.assertEqual(2 * len(cases), len(self.writer.events))
        for case in cases:
            self._assert_scalar(expected_event_type, expected_value_type, case)

    def test_dump_bool(self):
        cases = [bool(True), np.bool(False), np.bool_(True)]
        self._test_scalar_type(cases, WriterEvent.WRITE_INT,
                               t.py_types.int.value)

    def test_dump_dict(self):
        # type (byte), len(int), (key, value)*
        # hash ordering is non-deterministic and the serializer does not
        # impose its own order so we only dump one key-value-pair.
        d = {True: np.float32(5.0)}
        di.dump_obj(d, self.writer)
        self.assertEqual(6, len(self.writer.events))
        self._assert_event(WriterEvent.WRITE_BYTE, t.py_types.dict.value)
        self._assert_event(WriterEvent.WRITE_INT, 1)
        self._assert_scalar(WriterEvent.WRITE_INT, t.py_types.int.value, True)
        self._assert_scalar(WriterEvent.WRITE_DOUBLE, t.py_types.double.value,
                            5.0)

    def test_dump_double(self):
        cases = [float(0.1), np.float32(0.2), np.float64(0.3)]
        self._test_scalar_type(cases, WriterEvent.WRITE_DOUBLE,
                               t.py_types.double.value)

    def test_dump_int(self):
        cases = [int(123), np.int8(456), np.uint64(789)]
        self._test_scalar_type(cases, WriterEvent.WRITE_INT,
                               t.py_types.int.value)

    def test_dump_list(self):
        # type (byte), len (int), values
        lst = [55, float(0.1), bool(False)]
        di.dump_obj(lst, self.writer)
        self.assertEqual(8, len(self.writer.events))
        self._assert_event(WriterEvent.WRITE_BYTE, t.py_types.list.value)
        self._assert_event(WriterEvent.WRITE_INT, 3)
        self._assert_scalar(WriterEvent.WRITE_INT, t.py_types.int.value, 55)
        self._assert_scalar(WriterEvent.WRITE_DOUBLE, t.py_types.double.value,
                            0.1)
        self._assert_scalar(WriterEvent.WRITE_INT, t.py_types.int.value, False)

    def test_dump_ndarray(self):
        # type (byte), element type (byte), len of shape (int), shape (int*), data (bytes)
        # Note that the array is unconditionally converted to element type np.float32.
        xs = np.array([[1, 2, 3], [4, 5, 6]], np.float64)
        di.dump_obj(xs, self.writer)
        self.assertEqual(6, len(self.writer.events))
        self._assert_event(WriterEvent.WRITE_BYTE, t.py_types.ndarray.value)
        self._assert_event(WriterEvent.WRITE_BYTE, t.dump_np_types(np.float32))
        self._assert_event(WriterEvent.WRITE_INT, 2)
        self._assert_event(WriterEvent.WRITE_INT, 2)
        self._assert_event(WriterEvent.WRITE_INT, 3)
        # value if string is numpy's concern; just validate some bytes were written
        self._assert_event(WriterEvent.WRITE_STR, None)

    def test_dump_str(self):
        # type (byte), strlen (int), string (latin-1 bytes)
        di.dump_obj("pants", self.writer)
        self.assertEqual(3, len(self.writer.events))
        self._assert_event(WriterEvent.WRITE_BYTE, t.py_types.str.value)
        self._assert_event(WriterEvent.WRITE_INT, 5)
        self._assert_event(WriterEvent.WRITE_STR, "pants".encode("latin-1"))


if __name__ == '__main__':
    unittest.main()
