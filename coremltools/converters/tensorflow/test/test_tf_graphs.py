# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

import os, sys
import tensorflow as tf
import numpy as np
import unittest

from test_base import TFNetworkTest


class TFSimpleNetworkTest(TFNetworkTest):

    # Allows you to override common test entry for this class
    # Backend - set use_cpu_only to be True when working on Intel GPU macs
    def _test_tf_model(
            self,
            graph,
            input_shapes,
            output_node_names,
            data_mode='random',
            input_refs=None,
            delta=1e-2,
            use_cpu_only=True,
            use_freeze=True,
            quantize_tf_model=False):

        super(TFSimpleNetworkTest, self)._test_tf_model(
            graph,
            input_shapes,
            output_node_names,
            data_mode=data_mode,
            input_refs=input_refs,
            delta=delta,
            use_cpu_only=use_cpu_only,
            use_freeze=use_freeze,
            quantize_tf_model=quantize_tf_model)

    def test_simple_matmul(self):
        graph = tf.Graph()
        with graph.as_default() as g:
            matrix1 = tf.placeholder(tf.float32, shape=[1, 2], name='input')
            matrix2 = tf.Variable(tf.truncated_normal([2, 3]))
            product = tf.matmul(matrix1, matrix2, name='product')
        self._test_tf_model(graph, {'input': [1, 2]}, ['product'], delta=1e-2)

    def test_matmul_transposed_weight(self):
        graph = tf.Graph()
        with graph.as_default() as g:
            matrix1 = tf.placeholder(tf.float32, shape=[1, 2], name='input')
            matrix2 = tf.Variable(tf.truncated_normal([3, 2]))
            product = tf.matmul(matrix1, matrix2, transpose_b=True, name='product')
            bias = tf.Variable(tf.truncated_normal([3]))
            y = tf.nn.bias_add(product, bias, name='y')

        self._test_tf_model(graph, {'input': [1, 2]}, ['y'], delta=1e-2)

    def test_matmul_biasadd_sub(self):
        graph = tf.Graph()
        with graph.as_default() as g:
            x = tf.placeholder(tf.float32, shape=[None, 2], name='input')
            weight = tf.Variable(tf.truncated_normal([2, 3]))
            y = tf.matmul(x, weight)
            bias = tf.Variable(tf.truncated_normal([3]))
            z0 = tf.nn.bias_add(y, bias)
            c = tf.Variable(tf.truncated_normal([3]))
            z = tf.subtract(z0, c, name='output')
        self._test_tf_model(graph, {'input': [1, 2]}, ['output'], delta=1e-2)

    def test_matmul_transpose(self):
        graph = tf.Graph()
        with graph.as_default() as g:
            matrix1 = tf.placeholder(tf.float32, shape=[1, 5], name='input')
            matrix2 = tf.Variable(tf.truncated_normal([5, 3]))
            product = tf.matmul(matrix1, matrix2, name='product')
            tp = tf.transpose(product, [0, 1], name='tp')
        self._test_tf_model(graph, {'input': [1, 5]}, ['tp'], delta=1e-2)

    def test_matmul_unstack(self):
        graph = tf.Graph()
        with graph.as_default() as g:
            matrix1 = tf.placeholder(tf.float32, shape=[2, 5], name='input')
            matrix2 = tf.Variable(tf.truncated_normal([5, 3]))
            product = tf.matmul(matrix1, matrix2, name='product')
            y1, y2 = tf.unstack(product)
            y1 = tf.identity(y1, name='output_1')
            y2 = tf.identity(y2, name='output_2')
        self._test_tf_model(graph, {'input': [2, 5]}, ['output_1', 'output_2'], delta=1e-2)

    def test_dense_activations(self):
        # TODO - Add other activations
        for act_type in ['sigmoid', 'tanh']:
            graph = tf.Graph()
            with graph.as_default() as g:
                matrix1 = tf.placeholder(tf.float32, shape=[1, 8], name='input')
                matrix2 = tf.Variable(tf.truncated_normal([8, 2]))
                product = tf.matmul(matrix1, matrix2, name='product')
                if act_type == 'sigmoid':
                    act = tf.sigmoid(product, name='act')
                elif act_type == 'tanh':
                    act = tf.tanh(product, name='act')
            self._test_tf_model(graph, {'input': [1, 8]}, ['act'], delta=1e-2)

    def test_extract_shape_1d(self):
        shape = [None, 2]
        graph = tf.Graph()
        with graph.as_default() as g:
            x = tf.placeholder(tf.float32, shape=shape, name='input')
            m = tf.Variable(tf.truncated_normal([1, 2]))
            y = tf.shape(x + m, name='output')
        self._test_tf_model(graph, {'input': [1, 2]}, ['output'], delta=1e-2)

    def test_extract_shape(self):
        dims = [2, 3, 4]
        for rank in range(1, len(dims) + 1):
            shape = [None] + dims[:rank]
            batched_shape = [1] + dims[:rank]
            graph = tf.Graph()
            with graph.as_default() as g:
                x = tf.placeholder(tf.float32, shape=shape, name='input')
                m = tf.Variable(tf.truncated_normal(batched_shape))
                y = tf.shape(x + m, name='output')
            self._test_tf_model(graph, {'input': batched_shape}, ['output'], delta=1e-2)

    @unittest.skip
    def test_shape_slice(self):
        seqlen = 2
        graph = tf.Graph()
        with graph.as_default() as g:
            data = tf.placeholder(
                tf.float32, [1, None, 1], name='input')  # (batch_size, seq_len, input_dim)
            m = tf.Variable(tf.truncated_normal([1, 1, 1]))
            data_t = tf.transpose(data + m, [1, 0, 2], name='tp')
            data_shape = tf.shape(data_t)
            output = tf.identity(data_shape[0], name='output')  # What is the slice here?
        self._test_tf_model(graph, {'input': [1, seqlen, 1]}, ['output'], delta=1e-2)

    @unittest.skip
    # "Backend exception: \"Invalid blob shape\": scatter_kernel_cpu: Invalid shape of input blob"
    def test_array_scatter(self):
        batch_size = 2
        graph = tf.Graph()
        with graph.as_default() as g:
            data = tf.placeholder(
                tf.float32, shape=[batch_size, 3], name='input')  # (batch_size, input_dim)
            m = tf.Variable(tf.truncated_normal([batch_size, 3]))
            arr = tf.TensorArray(size=2, element_shape=[batch_size, 3], dtype=tf.float32)
            arr = arr.write(0, data)
            arr = arr.write(1, m)
            output = arr.gather([0, 1], name='output')
        self._test_tf_model(graph, {'input': [batch_size, 3]}, ['output'], delta=1e-2)

    def test_range(self):
        graph = tf.Graph()
        with graph.as_default() as g:
            data = tf.placeholder(tf.int32, shape=(), name='input')  # input is a scalar
            m = tf.Variable(1)
            output = tf.range(0, data + m, 1, name='output')
        self._test_tf_model(graph, {'input': []}, ['output'], input_refs={'input': 1}, delta=1e-2)

    def test_simple_loop(self):
        graph = tf.Graph()
        with graph.as_default() as g:
            data = tf.placeholder(tf.float32, shape=[None, 2], name='data')
            i = tf.constant(0)
            # When providing placeholder directly into while loop structures,
            # placeholder must be the first one.
            c = lambda x, i, v: tf.less(i, 10)
            b = lambda x, i, v: (tf.add(x, v), i + 1, v)  # Dummy
            w = tf.Variable(2.0, dtype=tf.float32, name='weight')
            r = tf.while_loop(c, b, [data, i, w], name='output')

        self._test_tf_model(graph, {"data": [1, 2]}, ["output/Exit"], delta=1e-2)

    def test_onehot_matmul_encoding(self):
        seq_len = 6
        embedding_dim = 10  # depth
        out_channels = 4
        graph = tf.Graph()
        with graph.as_default() as g:
            indices = tf.placeholder(tf.int32, shape=[None, seq_len], name='indices')
            onehot = tf.one_hot(indices, depth=embedding_dim)  # (batch_size, seq_len, embedding_dim)
            weight = tf.Variable(tf.truncated_normal([1, embedding_dim, out_channels]))
            y = tf.matmul(onehot, weight, name='output')

        self._test_tf_model(graph, {"indices": [1, seq_len]}, ["output"], data_mode='linear', delta=1e-2)

    def test_two_input_batch_matmul(self):
        test_cases = [
            {'r_x': 6, 'c_x': 10, 'r_y': 10, 'c_y': 4, 'transpose_x': False, 'transpose_y': False},
            {'r_x': 6, 'c_x': 10, 'r_y': 4, 'c_y': 10, 'transpose_x': False, 'transpose_y': True}
        ]
        # r_o, c_o = 6, 4
        for tc in test_cases:
            graph = tf.Graph()
            with graph.as_default() as g:
                r_x, c_x, r_y, c_y, tp_x, tp_y = tc['r_x'], tc['c_x'], tc['r_y'], tc['c_y'], tc['transpose_x'], tc['transpose_y']
                data_shape = [1, r_x, c_x]
                weight_shape = [1, r_y, c_y]
                input_data = tf.placeholder(tf.float32, shape=data_shape, name='input_data')
                input_weight = tf.placeholder(tf.float32, shape=weight_shape, name='input_weight')
                y = tf.matmul(input_data, input_weight, name='output', transpose_a=tp_x, transpose_b=tp_y)
            self._test_tf_model(graph, {"input_data": data_shape, "input_weight": weight_shape}, ["output"], delta=1e-2,
                                use_freeze=False)


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TFSimpleNetworkTest("test_simple_loop"))
    # unittest.TextTestRunner().run(suite)
