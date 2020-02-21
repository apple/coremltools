# -*- coding: utf-8 -*-
from __future__ import absolute_import as _
from __future__ import division as _
from __future__ import print_function as _

import os
import unittest

import numpy as np

from coremltools.converters.nnv2.builtin_types import serialization


class TestSerialize(unittest.TestCase):
    def setUp(self):
        pass

    def test_int(self):
        a = 1234567
        filename = os.path.join('', 'test_serial.meta')
        serialization.dump(a, filename)
        with open(filename, 'rb') as fp:
            reader = serialization.file_reader(fp)
            b = reader.read_value()
        os.remove(filename)
        np.testing.assert_array_almost_equal(a, b, 4)

    def test_float(self):
        a = 12345.67
        filename = os.path.join('', 'test_serial.meta')
        serialization.dump(a, filename)
        with open(filename, 'rb') as fp:
            reader = serialization.file_reader(fp)
            b = reader.read_value()
        os.remove(filename)
        np.testing.assert_array_almost_equal(a, b, 4)

    def test_str(self):
        a = 'what are you talking about'
        filename = os.path.join('', 'test_serial.meta')
        serialization.dump(a, filename)
        with open(filename, 'rb') as fp:
            reader = serialization.file_reader(fp)
            b = reader.read_value()
        os.remove(filename)
        assert a == b

    def test_bool(self):
        a = True
        filename = os.path.join('', 'test_serial.meta')
        serialization.dump(a, filename)
        with open(filename, 'rb') as fp:
            reader = serialization.file_reader(fp)
            b = reader.read_value()
        os.remove(filename)
        np.testing.assert_array_almost_equal(a, b, 4)

    def test_1darray(self):
        a = np.random.rand(3)
        filename = os.path.join('', 'test_serial.meta')
        serialization.dump(a, filename)
        with open(filename, 'rb') as fp:
            reader = serialization.file_reader(fp)
            b = reader.read_value()
        os.remove(filename)
        np.testing.assert_array_almost_equal(a, b, 4)

    def test_2darray(self):
        a = np.random.rand(4, 5)
        filename = os.path.join('', 'test_serial.meta')
        serialization.dump(a, filename)
        with open(filename, 'rb') as fp:
            reader = serialization.file_reader(fp)
            b = reader.read_value()
        os.remove(filename)
        np.testing.assert_array_almost_equal(a, b, 4)

    def test_3darray(self):
        a = np.random.rand(4, 5, 7)
        filename = os.path.join('', 'test_serial.meta')
        serialization.dump(a, filename)
        with open(filename, 'rb') as fp:
            reader = serialization.file_reader(fp)
            b = reader.read_value()
        os.remove(filename)
        np.testing.assert_array_almost_equal(a, b, 4)

    def test_int_list(self):
        a = np.random.randint(100, size=8)
        a = list(a)
        filename = os.path.join('', 'test_serial.meta')
        serialization.dump(a, filename)
        with open(filename, 'rb') as fp:
            reader = serialization.file_reader(fp)
            b = reader.read_value()
        os.remove(filename)
        np.testing.assert_array_almost_equal(a, b, 4)

    def test_double_list(self):
        a = np.random.rand(8)
        a = list(a)
        filename = os.path.join('', 'test_serial.meta')
        serialization.dump(a, filename)
        with open(filename, 'rb') as fp:
            reader = serialization.file_reader(fp)
            b = reader.read_value()
        os.remove(filename)
        np.testing.assert_array_almost_equal(a, b, 4)

    def test_str_list(self):
        a = ['apple', 'banana', 'monkey', 'mango']
        filename = os.path.join('', 'test_serial.meta')
        serialization.dump(a, filename)
        with open(filename, 'rb') as fp:
            reader = serialization.file_reader(fp)
            b = reader.read_value()
        os.remove(filename)
        for i, j in zip(a, b):
            assert i == j
