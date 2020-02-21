import os
import itertools
import numpy as np
import pytest
import scipy
import sys

import coremltools.proto.Program_pb2 as pm
from coremltools.converters.nnv2.nnv2_program.ops import CoremlBuilder as cb
from coremltools.converters.nnv2.builtin_types import builtins
from coremltools.converters.nnv2._deps import HAS_TF, HAS_PYTORCH
from coremltools.converters.nnv2.nnv2_program.program import (
        get_new_symbol,
        get_new_variadic_symbol,
        )
from coremltools.converters.nnv2.nnv2_program.ops.testing_ops_utils import (
        run_compare_builder,
        run_compare_tf,
        ssa_fn, is_close,
        get_core_ml_prediction,
        _random_gen,
        UNK_SYM,
        UNK_VARIADIC,
        )


if HAS_TF:
    import tensorflow.compat.v1 as tf

if HAS_PYTORCH:
    import torch

np.random.seed(123)


class FrontendTesterSample:
    def test_builder_to_backend_smoke(self, use_cpu_only):
        """
        Construct op from builder and test output numerical parity.

        Comment: Tests here are like simple demonstration of op (similar to tf
        op's examples). Note that we should hard code values inline as much as
        possible here, which helps readers to know the op's behavior
        explicitly and also minimizes programmatic errors in generating
        expected outputs.

        Comment: Focus on fp32 for most ops for now.
        """
        raise NotImplementedError("Smoke test should be implemented")

    def test_builder_eval(self):
        """
        We supply const for all inputs of the op, and check value from
        eval(). Only needed if eval() is implemented.

        Comment: By using @ssa_fn from util, we can places this function inside
        a SsaFunction, so we can directly call CoremlBuilder (e.g. cb.expand_dims()).
        """
        pass

    def test_builder_to_backend_symbolic(self):
        """
        Test inputs with symbolic shape. (Need some infra)
        """
        # TODO (rdar://58818028): Test input with symbolic shapes
        pass

    def test_builder_to_backend_programmatic(self, use_cpu_only, **kwargs):
        """
        More programatic tests to cover more corner cases.
        """
        pass

    def test_tf(self, use_cpu_only, **kwargs):
        """
        TF graph --> backend numerical parity.

        Comment: The tests here may loosely mirror test_builder_to_backend_*
        """
        pass

class TestAbs:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.abs(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.random.rand(1, 2, 4, 3)
        v = cb.abs(x=x_val)
        assert is_close(np.abs(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.abs(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-1, rand_max=1)}, # Generate (-1, 1] random numbers
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestAcos:

    @pytest.mark.parametrize("use_cpu_only, backend",
        itertools.product(
            [True, False],
            ['nnv2'],
            ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, -0.5, 0],[0.4, 0.5, 0.8]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.acos(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[3.14159265, 2.0943951 , 1.57079633],
                                    [1.15927948, 1.04719755, 0.64350111]],
                                    dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-1, rand_max=1)
        v = cb.acos(x=x_val)
        assert is_close(np.arccos(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.acos(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-1, rand_max=1)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestAsin:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, -0.5, 0],[0.4, 0.5, 0.8]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.asin(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-1.57079633, -0.52359878, 0.],
                                     [0.41151685, 0.52359878, 0.92729522]],
                                     dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-1, rand_max=1)
        v = cb.asin(x=x_val)
        assert is_close(np.arcsin(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                             ))
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.asin(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-1, rand_max=1)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestCumsum():

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):

        #Need to be removed
        frontend_only = True

        t = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.cumsum(x=x, axis=0, reverse=True, exclusive=False)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[5,7,9],[4,5,6]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only,
                            frontend_only=frontend_only, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        v = cb.cumsum(x=x_val)
        assert is_close(np.cumsum(x_val, axis=0), v.val)

    @pytest.mark.skiptf(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank, reverse, exclusive",
                             itertools.product(
                                [True, False],
                                ['nnv2'],
                                [rank for rank in range(1,6)],
                                [True, False],
                                [True, False]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank, reverse, exclusive):
        #Need to be removed
        frontend_only = True

        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            for axis in range(-1, rank):
                res = tf.math.cumsum(x, axis=axis, reverse=reverse, exclusive=exclusive)
                run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-100, rand_max=100)},
                               res, use_cpu_only=use_cpu_only,
                               frontend_only=frontend_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_empty_input_tf(self, use_cpu_only, backend):
        #Corner cases
        #Need to be removed
        frontend_only = True
        empty_inputs = [[], [[]], [[[]]], [[],[]], [[[]],[[]]]]
        for input_x in empty_inputs:
            input_x = np.array(input_x)
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_x.shape)
                for axis in range(-1, len(input_x.shape)):
                    res = tf.math.cumsum(x, axis=axis)
                    run_compare_tf(graph, {x: input_x},
                                   res, use_cpu_only=use_cpu_only,
                                   frontend_only=frontend_only,
                                   backend=backend)

    @ssa_fn
    def test_invalid_axis(self):
        x_val = _random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            pred = cb.cumsum(x=x_val, axis=-2)
        with pytest.raises(ValueError):
            pred = cb.cumsum(x=x_val, axis=len(x_val.shape))
        with pytest.raises(ValueError):
            pred = cb.cumsum(x=x_val, axis='')

    @ssa_fn
    def test_invalid_reverse(self):
        x_val = _random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            pred = cb.cumsum(x=x_val, reverse='')
        with pytest.raises(ValueError):
            pred = cb.cumsum(x=x_val, reverse=0)
        with pytest.raises(ValueError):
            pred = cb.cumsum(x=x_val, reverse=1)

    @ssa_fn
    def test_invalid_exclusive(self):
        x_val = _random_gen(shape=(1, 2, 3, 4, 5), rand_min=-100, rand_max=100)
        with pytest.raises(ValueError):
            pred = cb.cumsum(x=x_val, exclusive='')
        with pytest.raises(ValueError):
            pred = cb.cumsum(x=x_val, exclusive=0)
        with pytest.raises(ValueError):
            pred = cb.cumsum(x=x_val, exclusive=1)

    @ssa_fn
    def test_invalid_input(self):
        x_val = 1
        with pytest.raises(ValueError):
            pred = cb.cumsum(x=x_val)
        x_val = ['1']
        with pytest.raises(TypeError):
            pred = cb.cumsum(x=x_val)

class TestAtan:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.atan(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-0.78539816, 1.10714872, -1.24904577],
                                     [1.32581766, -1.37340077, 1.40564765]],
                                     dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-100, rand_max=100)
        v = cb.atan(x=x_val)
        assert is_close(np.arctan(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                             ))
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.atan(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-100, rand_max=100)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestAvgPool:

    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([
            [[[-10.80291205, -6.42076184],
              [-7.07910997, 9.1913279]],
             [[-3.18181497, 0.9132147],
              [11.9785544, 7.92449539]]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.avg_pool(x=x, kernel_sizes=[1, 2], strides=[2, 1], pad_type='valid'),
            ]

        expected_output_types = [(1, 2, 1, 1, builtins.fp32)]
        expected_outputs = [np.array([[[[-8.611837]], [[-1.1343001]]]], dtype=np.float32)]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, kernel_sizes, strides, pad_type',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [(1,)],
                                 [(1,), (2,)],
                                 ['same', 'valid']))
    def test_avg_pool_1d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        input_shape = np.random.randint(low=2, high=6, size=3)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.nn.avg_pool1d(x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper())
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-100, rand_max=100)},
                           res, use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, kernel_sizes, strides, pad_type',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [(1,), (2,), (1, 1), (1, 2), (2, 2)],
                                 [(1,), (2,), (1, 1), (1, 2), (2, 2)],
                                 ['same', 'valid']))
    def test_avg_pool_2d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        shape = np.random.randint(low=2, high=6, size=4)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            res = tf.nn.avg_pool(x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper())
            run_compare_tf(graph, {x: _random_gen(shape, rand_min=-100, rand_max=100)},
                           res, use_cpu_only=use_cpu_only, backend=backend)


class TestCeil:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1.2, 2, -3.4],[4.5, -5, 6.7]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.ceil(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = [np.array([[-1, 2, -3], [5, -5, 7]], dtype=np.float32)]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-100, rand_max=100)
        v = cb.ceil(x=x_val)
        assert is_close(np.ceil(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        eps_from_int = 0.0
        if not use_cpu_only:
            eps_from_int = 0.1
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.ceil(x)
            run_compare_tf(graph,
                    {x: _random_gen(input_shape, rand_min=-100,
                        rand_max=100, eps_from_int=eps_from_int)},
                    res, use_cpu_only=use_cpu_only,
                    frontend_only=False, backend=backend)

class TestClip:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1.2, 2, -3.4],[4.5, -5, 6.7]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.clip(x=x, alpha=0.0, beta=5.0)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = [np.array([[0, 2, 0], [4.5, 0, 5]], dtype=np.float32)]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-5, rand_max=10)
        v = cb.clip(x=x_val, alpha=0.0, beta=5.0)
        assert is_close(np.clip(x_val, 0.0, 5.0), v.val)

    @pytest.mark.skip("TF decomposes clip_by_value into min and max <rdar://problem/59047747>")
    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=5.0)
            run_compare_tf(graph, {x: _random_gen(input_shape,
                rand_min=-5, rand_max=10)},
                res, use_cpu_only=use_cpu_only,
                frontend_only=False, backend=backend)

class TestCos:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.cos(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[0.54030231, -0.41614684, -0.9899925 ],
                                     [-0.65364362, 0.28366219, 0.96017029]],
                                     dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-1000, rand_max=1000)
        v = cb.cos(x=x_val)
        assert is_close(np.cos(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        rand_range = 1000
        if not use_cpu_only:
            rand_range = 10
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.cos(x)
            run_compare_tf(graph,
                    {x: _random_gen(input_shape, rand_min=-rand_range,
                        rand_max=rand_range)},
                    res, use_cpu_only=use_cpu_only,
                    frontend_only=False, backend=backend)


class TestCosh:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, -2, -3],[1, 2, 3]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.cosh(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[1.54308063, 3.76219569, 10.067662],
                                     [1.54308063, 3.76219569, 10.067662]],
                                     dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-4, rand_max=4)
        v = cb.cosh(x=x_val)
        assert is_close(np.cosh(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.cosh(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-4, rand_max=4)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestDepthToSpace:

    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 ['nnv1'],
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        # original input type is (1, 4, 1, 1, fp32)
        val = np.array([[[[9.]], [[5.]], [[1.]], [[3.]]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=val.shape)}
        input_values = {'x': val}

        def build(x):
            return [cb.depth_to_space(x=x, block_size=2)]

        expected_output_types = (1, 1, 2, 2, builtins.fp32)
        expected_outputs = np.array([[[9., 5.], [1., 3.]]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, shape, block_size',
                             itertools.product(
                                 [True, False],
                                 ['nnv1'],
                                 [(1, 1, 1, 16), (1, 1, 1, 32), (1, 3, 3, 16)],
                                 [2, 4]
                             ))
    def test_tf(self, use_cpu_only, backend, shape, block_size):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.depth_to_space(x, block_size)
            run_compare_tf(graph, {x: np.random.rand(*shape)}, ref,
                           use_cpu_only=use_cpu_only, backend=backend)


class TestEqual:

    @pytest.mark.skip("Output does not support bool values rdar://problem/59216804")
    @pytest.mark.parametrize("use_cpu_only", [True, False])
    def test_builder_to_backend_smoke(self, use_cpu_only):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.equal(x=x, y=y)
        expected_output_types = (2, 3, builtins.bool)
        expected_outputs = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.bool)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.bool)
        v = cb.equal(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skip("Output does not support bool values rdar://problem/59216804")
    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, rank",
                             itertools.product(
                                 [True, False],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.equal(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only, frontend_only=False)

class TestExp:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.exp(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[0.36787944, 7.3890561 , 0.04978707],
                                     [54.5981500, 0.0067379, 403.428793]],
                                     dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-4, rand_max=20)
        v = cb.exp(x=x_val)
        assert is_close(np.exp(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True], # Exp(1) in numpy already differs from Espresso.
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.exp(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-4, rand_max=20)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestExp2:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.exp2(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[0.5, 4., 0.125], [16, 0.03125, 64]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-4, rand_max=20)
        v = cb.exp2(x=x_val)
        assert is_close(np.exp2(x_val), v.val)

class TestConv:
    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend",
                             itertools.product(
                                 [True, False],
                                 ['nnv1'],
                             ))
    def test_tf(self, use_cpu_only, backend):
        N, H, W, C_in = 1, 3, 4, 2
        kH, kW, C_out = 3, 3, 5
        input_shape = (N, H, W, C_in)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            W = tf.constant(np.random.rand(kH, kW, C_in, C_out), tf.float32)
            conv = tf.nn.conv2d(x, W, strides=1, padding='SAME',
                    data_format='NHWC')
            run_compare_tf(graph, {x: np.random.rand(*input_shape)},
                           conv, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestExpandDims:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2', 'nnv1'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return [cb.expand_dims(x=x, axis=0),
                   cb.expand_dims(x=x, axis=1),
                   cb.expand_dims(x=x, axis=2),
                   cb.expand_dims(x=x, axis=-1)]
        expected_output_types = [
                (1, 2, 3, builtins.fp32),
                (2, 1, 3, builtins.fp32),
                (2, 3, 1, builtins.fp32),
                (2, 3, 1, builtins.fp32),
                ]
        expected_outputs = [
                np.array([[[1, 2, 3],
                           [4, 5, 6]]], dtype=np.float32),
                np.array([[[1, 2, 3]],
                          [[4, 5, 6]]], dtype=np.float32),
                np.array([[[1],
                           [2],
                           [3]],
                          [[4],
                           [5],
                           [6]]], dtype=np.float32),
                np.array([[[1],
                           [2],
                           [3]],
                          [[4],
                           [5],
                           [6]]], dtype=np.float32),
                          ]


        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.random.rand(1, 1, 6, 6)
        v = cb.expand_dims(x=x_val, axis=2)
        assert is_close(np.expand_dims(x_val, 2), v.val)

    @pytest.mark.parametrize("use_cpu_only, backend, rank_and_axis",
                             itertools.product(
                                 [True, False],
                                 ['nnv2', 'nnv1'],
                                 [(rank, axis) for rank in range(1, 5) for
                                     axis in range(-rank - 1, rank + 1)],
                             )
                             )
    def test_builder_to_backend_programmatic(self, use_cpu_only, backend,
            rank_and_axis):
        rank, axis = rank_and_axis
        x_shape = np.random.randint(low=2, high=6, size=rank)
        input_placeholders = {"x": cb.placeholder(shape=x_shape)}
        input_values = {"x": np.random.sample(x_shape).astype(np.float32)}
        def build(x): return cb.expand_dims(x=x, axis=axis)
        adjusted_axis = axis if axis >= 0 else rank + axis + 1
        x_shape = list(x_shape)
        out_shape = x_shape[:adjusted_axis] + [1] + x_shape[adjusted_axis:]
        expected_output_types = tuple(out_shape[:]) + (builtins.fp32,)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types,
                            np.expand_dims(input_values['x'], axis),
                            use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank_and_axis",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [(rank, axis) for rank in range(1, 5) for
                                     axis in range(-rank - 1, rank + 1)],
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank_and_axis):
        rank, axis = rank_and_axis
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.expand_dims(x, axis=axis)
            run_compare_tf(graph, {x: np.random.rand(*input_shape)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestFill:

    @pytest.mark.skipif(True, reason='TODO')
    @pytest.mark.parametrize('use_cpu_only, value',
                             itertools.product(
                                 [True, False],
                                 [0.]))
    def test_builder_to_backend_smoke(self, use_cpu_only, value):
        shape = (2, 1, 3)
        x_val = np.array([1.], dtype=np.float32)
        input_placeholders = {
            'x': cb.placeholder(shape=x_val.shape),
            'y': cb.placeholder(shape=x_val.shape)}

        input_values = {'x': x_val, 'y': x_val}

        def build(x, y):
            return [
                cb.add(x=x, y=y),
                cb.fill(shape=shape, value=value)
            ]

        expected_output_types = [
            (1, builtins.fp32),
            (2, 1, 3, builtins.fp32)
        ]

        expected_outputs = [
            np.array(np.ones(shape=(1,)) * 2., np.float32),
            np.full(shape=shape, fill_value=value)
        ]

        run_compare_builder(build, input_placeholders, input_values, expected_output_types,
                            expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only)

    @ssa_fn
    def test_builder_eval(self):
        shape = np.random.randint(low=1, high=3, size=5).astype(np.int32)
        v = cb.fill(shape=shape, value=1991.)
        assert is_close(np.full(shape, fill_value=1991.), v.val)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank, value',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                                 [-19., 0., 37.]))
    def test_tf(self, use_cpu_only, backend, rank, value):
        shape = np.random.randint(low=1, high=3, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.add(x, tf.fill(dims=np.array(shape, dtype=np.float32), value=value))
            run_compare_tf(graph, {x: np.random.rand(*shape)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)


class TestFloor:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1.2, 2, -3.4],[4.5, -5, 6.7]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.floor(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-2, 2, -4], [4, -5, 6]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-100, rand_max=100)
        v = cb.floor(x=x_val)
        assert is_close(np.floor(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        eps_from_int = 0.0
        if not use_cpu_only:
            eps_from_int = 0.1
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.floor(x)
            run_compare_tf(graph, {x: _random_gen(input_shape,
                rand_min=-100, rand_max=100, eps_from_int=eps_from_int)},
                res, use_cpu_only=use_cpu_only,
                frontend_only=False, backend=backend)

class TestFloorDiv:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[10, 20, 30],[40, 50, 60]], dtype=np.float32)
        y = np.array([[11, 12, 13],[14, 15, 16]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.floor_div(x=x, y=y)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[0, 1, 2], [2, 3, 3]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[10, 20, 30],[40, 50, 60]], dtype=np.float32)
        y_val = np.array([[11, 12, 13],[14, 15, 16]], dtype=np.float32)
        expected_outputs = np.array([[0, 1, 2], [2, 3, 3]], dtype=np.float32)
        v = cb.floor_div(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.floor_div(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=0, high=100, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=1, high=20, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestGreater:

    @pytest.mark.skip("Output does not support bool values rdar://problem/59216804")
    @pytest.mark.parametrize("use_cpu_only", [True, False])
    def test_builder_to_backend_smoke(self, use_cpu_only):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.greater(x=x, y=y)
        expected_output_types = (2, 3, builtins.bool)
        expected_outputs = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.bool)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.bool)
        v = cb.greater(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skip("Output does not support bool values rdar://problem/59216804")
    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, rank",
                             itertools.product(
                                 [True, False],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.greater(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only, frontend_only=False)

class TestGreaterEqual:

    @pytest.mark.skip("Output does not support bool values rdar://problem/59216804")
    @pytest.mark.parametrize("use_cpu_only", [True, False])
    def test_builder_to_backend_smoke(self, use_cpu_only):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.greater_equal(x=x, y=y)
        expected_output_types = (2, 3, builtins.bool)
        expected_outputs = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.bool)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.bool)
        v = cb.greater_equal(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skip("Output does not support bool values rdar://problem/59216804")
    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, rank",
                             itertools.product(
                                 [True, False],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.greater_equal(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only, frontend_only=False)

class TestLess:

    @pytest.mark.skip("Output does not support bool values rdar://problem/59216804")
    @pytest.mark.parametrize("use_cpu_only", [True, False])
    def test_builder_to_backend_smoke(self, use_cpu_only):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.less(x=x, y=y)
        expected_output_types = (2, 3, builtins.bool)
        expected_outputs = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.bool)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.bool)
        v = cb.less(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skip("Output does not support bool values rdar://problem/59216804")
    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, rank",
                             itertools.product(
                                 [True, False],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.less(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only, frontend_only=False)

class TestLessEqual:

    @pytest.mark.skip("Output does not support bool values rdar://problem/59216804")
    @pytest.mark.parametrize("use_cpu_only", [True, False])
    def test_builder_to_backend_smoke(self, use_cpu_only):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.less_equal(x=x, y=y)
        expected_output_types = (2, 3, builtins.bool)
        expected_outputs = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.bool)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.bool)
        v = cb.less_equal(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skip("Output does not support bool values rdar://problem/59216804")
    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, rank",
                             itertools.product(
                                 [True, False],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.less_equal(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only, frontend_only=False)

class TestLog:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.log(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[0., 0.69314718, 1.09861229],
                                     [1.38629436, 1.60943791, 1.79175947]],
                                     dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=0.2, rand_max=1000)
        v = cb.log(x=x_val)
        assert is_close(np.log(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.log(x)
            run_compare_tf(graph,
                    {x: _random_gen(input_shape, rand_min=0.2,
                        rand_max=1000)},
                    res, use_cpu_only=use_cpu_only, frontend_only=False,
                    backend=backend)

class TestMaximum:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.maximum(x=x, y=y)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        v = cb.maximum(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.maximum(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestMinimum:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.minimum(x=x, y=y)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.minimum(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.minimum(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestMod:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[10, 20, 30],[40, 50, 60]], dtype=np.float32)
        y = np.array([[11, 12, 13],[14, 15, 16]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.mod(x=x, y=y)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[10, 8, 4], [12, 5, 12]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[10, 20, 30],[40, 50, 60]], dtype=np.float32)
        y_val = np.array([[11, 12, 13],[14, 15, 16]], dtype=np.float32)
        expected_outputs = np.array([[10, 8, 4], [12, 5, 12]], dtype=np.float32)
        v = cb.mod(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.mod(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=0, high=100, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=1, high=20, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestMul:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.mul(x=x, y=y)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-1, 4, -9], [16, -25, 36]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[-1, 4, -9], [16, -25, 36]], dtype=np.float32)
        v = cb.mul(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.multiply(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestNotEqual:

    @pytest.mark.skip("Output does not support bool values rdar://problem/59216804")
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.not_equal(x=x, y=y)
        expected_output_types = (2, 3, builtins.bool)
        expected_outputs = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.bool)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.bool)
        v = cb.not_equal(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skip("Output does not support bool values rdar://problem/59216804")
    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.not_equal(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestPow:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.pow(x=x, y=y)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[1, 4, 0.037], [256, 0.00032, 46656]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[1, 4, 0.037], [256, 0.00032, 46656]], dtype=np.float32)
        v = cb.pow(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.pow(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=5, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=5, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestRealDiv:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[10, 20, 30],[40, 50, 60]], dtype=np.float32)
        y = np.array([[11, 12, 13],[14, 15, 16]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.real_div(x=x, y=y)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[0.90909091, 1.66666667, 2.30769231],
                                     [2.85714286, 3.33333333, 3.75]],
                                     dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[10, 20, 30],[40, 50, 60]], dtype=np.float32)
        y_val = np.array([[11, 12, 13],[14, 15, 16]], dtype=np.float32)
        expected_outputs = np.array([[0.90909091, 1.66666667, 2.30769231],
                                     [2.85714286, 3.33333333, 3.75]],
                                     dtype=np.float32)
        v = cb.real_div(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.truediv(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=0, high=100, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=1, high=20, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestMaxPool:

    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([
            [[[-10.80291205, -6.42076184],
              [-7.07910997, 9.1913279]],
             [[-3.18181497, 0.9132147],
              [11.9785544, 7.92449539]]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [cb.max_pool(x=x, kernel_sizes=[1, 2], strides=[2, 1], pad_type='valid')]

        expected_output_types = [(1, 2, 1, 1, builtins.fp32)]
        expected_outputs = [np.array([[[[-6.42076184]], [[0.9132147]]]], dtype=np.float32)]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not installed.')
    @pytest.mark.parametrize('use_cpu_only, backend, kernel_sizes, strides, pad_type',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [(1,)],
                                 [(1,), (2,)],
                                 ['same', 'valid']))
    def test_tf_max_pool_1d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        input_shape = np.random.randint(low=2, high=6, size=3)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.nn.max_pool1d(x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper())
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-100, rand_max=100)},
                           res, use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not installed.')
    @pytest.mark.parametrize('use_cpu_only, backend, kernel_sizes, strides, pad_type',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [(1,), (2,), (1, 1), (1, 2), (2, 2)],
                                 [(1,), (2,), (1, 1), (1, 2), (2, 2)],
                                 ['same', 'valid']))
    def test_tf_max_pool_2d(self, use_cpu_only, backend, kernel_sizes, strides, pad_type):
        shape = np.random.randint(low=2, high=6, size=4)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            res = tf.nn.max_pool(x, ksize=kernel_sizes[:], strides=strides[:], padding=pad_type.upper())
            run_compare_tf(graph, {x: _random_gen(shape, rand_min=-100, rand_max=100)},
                           res, use_cpu_only=use_cpu_only, backend=backend)


class TestMatMul:

    @pytest.mark.skipif(not sys.version_info.major == 3 and sys.version_info.minor >= 6,
                        reason='input map order not guaranteed.')
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([[-4.71823726, 11.94002459],
                          [-3.39397839, 9.21668793]], dtype=np.float32)
        y_val = np.array([[1.23134601, -0.09504865],
                          [-1.40759034, -0.88166538]], dtype=np.float32)
        input_placeholders = {
            'x': cb.placeholder(shape=x_val.shape),
            'y': cb.placeholder(shape=y_val.shape)}
        input_values = {'x': x_val, 'y': y_val}

        def build(x, y):
            return [
                cb.matmul(x=x, y=y)
            ]

        expected_output_types = [(2, 2, builtins.fp32)]
        expected_outputs = [
            np.array([[-22.61644592, -10.07864419],
                      [-17.15248267, -7.80344157]], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(2, 2, 4), rand_min=-37, rand_max=64)
        y_val = _random_gen(shape=(2, 4, 2), rand_min=-91, rand_max=84)
        v = cb.matmul(x=x_val, y=y_val)
        assert is_close(np.matmul(x_val, y_val), v.val)

    # TODO: rdar://59460970 (NNv2: More tests for MatMul op)
    @pytest.mark.skipif(not sys.version_info.major == 3 and sys.version_info.minor >= 6,
                        reason='input map order not guaranteed.')
    @pytest.mark.parametrize('use_cpu_only, backend, dim, transpose_x, transpose_y',
                             itertools.product(
                                 [True],
                                 ['nnv2'],
                                 [2, 4, 8],
                                 [False],
                                 [True, False]))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, dim, transpose_x, transpose_y):
        shape_x = np.array([dim, dim])
        shape_y = shape_x if transpose_y else np.flip(shape_x, axis=-1)
        x_val = np.random.rand(*shape_x)
        y_val = np.random.rand(*shape_y)
        input_placeholders = {
            'x': cb.placeholder(shape=x_val.shape),
            'y': cb.placeholder(shape=y_val.shape)
        }
        input_values = {'x': x_val, 'y': y_val}

        def build(x, y):
            return [
                cb.matmul(x=x, y=y, transpose_x=transpose_x, transpose_y=transpose_y)
            ]

        expected_outputs = [
            np.matmul(x_val, np.transpose(y_val) if transpose_y else y_val)
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values, expected_output_types,
                            expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, dim, transpose_x, transpose_y',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [2, 4, 8],
                                 [True, False],
                                 [True, False]))
    def test_tf(self, use_cpu_only, backend, dim, transpose_x, transpose_y):
        shape_x = np.array([dim, dim])
        shape_y = shape_x if transpose_y else np.flip(shape_x, axis=-1)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape_x)
            y = tf.placeholder(tf.float32, shape=shape_y)
            res = tf.linalg.matmul(x, y, transpose_a=transpose_x, transpose_b=transpose_y)
            run_compare_tf(graph, {
                x: _random_gen(shape_x, rand_min=-100, rand_max=100),
                y: _random_gen(shape_y, rand_min=-1., rand_max=1.)},
                           res, use_cpu_only=use_cpu_only, backend=backend)


class TestReshape:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv1'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}
        def build(x):
            return [cb.reshape(x=x, shape=[3, 2]),
                    cb.reshape(x=x, shape=[2, -1]),
                    cb.reshape(x=x, shape=[2, 1, 1, 3])]

        expected_output_types = [
                (3, 2, builtins.fp32),
                (2, 3, builtins.fp32),
                (2, 1, 1, 3, builtins.fp32),
                ]
        expected_outputs = [
                np.array([[1, 2],
                          [3, 4],
                          [5, 6]], dtype=np.float32),
                np.array([[1, 2, 3],
                          [4, 5, 6]], dtype=np.float32),
                np.array([[[[1., 2., 3.]]],
                          [[[4., 5., 6.]]]], dtype=np.float32)
                ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        r = cb.reshape(x=t, shape=[3, 2])
        expected_r = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        assert is_close(expected_r, r.val)
        r2 = cb.reshape(x=t, shape=[2, -1])
        expected_r2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        assert is_close(expected_r2, r2.val)

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv1'],
                ))
    def test_builder_to_backend_symbolic(self, use_cpu_only, backend):
        s0 = get_new_symbol()
        s_len = get_new_symbol()
        s1 = get_new_variadic_symbol()

        # Test variadic (rdar://59559656)
        input_placeholders = {
                "x": cb.placeholder(shape=(2, s0)),
                # TODO: variadic (rdar://59559656)
                #"x2": cb.placeholder(shape=(s1, 2)),
                "shape": cb.placeholder(shape=(3,), dtype=pm.INT32),
                "shape2": cb.placeholder(shape=(s_len,), dtype=pm.INT32),
                }

        def build(x, shape, shape2):
            return [cb.reshape(x=x, shape=[2, -1]),
                    cb.reshape(x=x, shape=[1, -1]),
                    cb.reshape(x=x, shape=[2, 1, 1, -1]),
                    # TODO: variadic (rdar://59559656)
                    #cb.reshape(x=x2, shape=[2, 1, 1]),
                    cb.reshape(x=x, shape=shape),
                    cb.reshape(x=x, shape=shape2),
                    ]

        expected_output_types = [
                (2, s0, builtins.fp32),
                (1, 2*s0, builtins.fp32),
                (2, 1, 1, s0, builtins.fp32),
                # TODO: variadic (rdar://59559656)
                #(2, 1, 1, builtins.fp32),
                (UNK_SYM, UNK_SYM, UNK_SYM, builtins.fp32),
                (UNK_VARIADIC, builtins.fp32),
                ]
        expected_outputs = [
                np.array([[1, 2, 3],
                          [4, 5, 6]], dtype=np.float32),
                np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32),
                np.array([[[[1., 2., 3.]]],
                          [[[4., 5., 6.]]]], dtype=np.float32),
                # TODO: variadic (rdar://59559656)
                #np.array([[1, 2, 3],
                #          [4, 5, 6]], dtype=np.float32),
                np.array([[[1, 2, 3]],
                          [[4, 5, 6]]], dtype=np.float32),
                np.array([[[1, 2, 3]],
                          [[4, 5, 6]]], dtype=np.float32),
                ]

        input_values = {
                "x": np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32),
                # TODO: variadic (rdar://59559656)
                #"x2": np.array([[[1, 2, 3],[4, 5, 6]]], dtype=np.float32),
                "shape": np.array([2, 1, 3], dtype=np.float32),
                "shape2": np.array([2, 1, 3], dtype=np.float32),
                }
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend",
                             itertools.product(
                                 [False],
                                 ['nnv1'],
                             ))
    def test_tf(self, use_cpu_only, backend):
        def test_tf_static():
            # (input_shape, output_shape_as_a_parameter)
            shapes = [([10, 10], [5, 20]),
                      ([3, 4, 5, 6], [4, 5, 3, 6]),
                      ([4, 4, 5, 6], [2, 2, -1])]

            for _shape in shapes:
                with tf.Graph().as_default() as graph:
                    x = tf.placeholder(tf.float32, shape=_shape[0])
                    res = tf.reshape(x, shape=_shape[1])
                    run_compare_tf(graph, {x: np.random.rand(*_shape[0])},
                                   res, use_cpu_only=use_cpu_only,
                                   frontend_only=False, backend=backend)
        def test_tf_dynamic():
            shapes = [([10, 10], [5, 20]),
                      ([3, 4, 5, 6], [4, 5, 3, 6]),
                      ([4, 4, 5, 6], [2, 2, -1]),
                      ([2, 3, 5, 3], [2, -1])]

            for _shape in shapes:
                with tf.Graph().as_default() as graph:
                    x = tf.placeholder(tf.float32, shape=_shape[0])
                    y = tf.placeholder(tf.int32, shape=[len(_shape[1])])
                    res = tf.reshape(x, shape=y)
                    run_compare_tf(graph, {x: np.random.rand(*_shape[0]),
                        # numpy
                        y:np.array(_shape[1], dtype=np.float32)},
                                   res, use_cpu_only=use_cpu_only,
                                   frontend_only=False, backend=backend)
        test_tf_static()
        test_tf_dynamic()


class TestRandomBernoulli:

    @pytest.mark.parametrize('use_cpu_only, backend',
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        """
        Construct op from builder and test output numerical parity.
        """

        x_val = np.array([1.], dtype=np.int32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.relu(x=x),
                cb.random_bernoulli(shape=np.array([2, 1, 3], np.int32), prob=1.0),
                cb.random_bernoulli(shape=np.array([3, 1, 2], np.int32), prob=0.0),
            ]

        expected_outputs = [
            np.array(np.ones(shape=(1,)), np.float32),
            np.array(np.ones(shape=(2, 1, 3)), np.float32),
            np.array(np.zeros(shape=(3, 1, 2)), np.float32),
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values,
                expected_output_types,
                expected_outputs=expected_outputs,
                use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.parametrize('use_cpu_only, backend, rank, prob',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                                 [1.0, 0.0]))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, rank, prob):
        low_factor = np.random.randint(low=2, high=4)
        lo = int(np.power(1000, 1. / rank)) * low_factor
        hi = int(np.power(2000, 1. / rank)) * np.random.randint(low=low_factor, high=4)
        shape = np.random.randint(low=lo, high=hi, size=rank, dtype=np.int32)
        x_val = np.array([1], dtype=np.int32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.relu(x=x),
                cb.random_bernoulli(shape=shape, prob=prob)
            ]

        expected_outputs = [
            np.array(np.ones(shape=(1,)), np.float32),
            np.random.binomial(1, prob, shape)
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values, expected_output_types,
                            expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skip("TODO: rdar://59071295 (Add dynamic input / TF conversion in frontend for Random Distribution ops)")
    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, size',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [size for size in range(1, 10)]))
    def test_tf(self, use_cpu_only, backend, size):
        input_shape = np.random.randint(low=2, high=6, size=2)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            ref = tf.keras.backend.random_binomial(shape=(1, 2, 3), p=1.0)
            run_compare_tf(graph, {x: np.random.rand(*input_shape)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)


class TestRandomCategorical:

    def softmax(self, data):
        e_data = np.exp(data - np.max(data))
        return e_data / e_data.sum()

    @pytest.mark.parametrize('use_cpu_only, backend',
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        """
        Construct op from builder and test output numerical parity.
        """
        x_val = np.array([1], dtype=np.int32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.random_categorical(x=x, seed=1),
                cb.random_categorical(x=x, seed=1, size=4),
            ]

        expected_outputs = [
            np.array(np.zeros(shape=(1,)), dtype=np.float32),
            np.array(np.zeros(shape=(4,)), dtype=np.float32),
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.parametrize('use_cpu_only, backend, n_sample, n_class',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [50000],
                                 [2, 10, 20]))
    def test_builder_to_backend_stress(self, use_cpu_only, backend,
            n_sample, n_class):
        output_name = 'random_categorical'
        logits = np.random.rand(2, n_class)
        probs = [self.softmax(logits[0]), self.softmax(logits[1])]

        # Test logits input
        input_placeholders = {'x': cb.placeholder(shape=(2, n_class))}
        input_values = {'x': logits}

        def build(x):
            return [cb.random_categorical(
                x=x, size=n_sample, mode='logits', name=output_name)
            ]

        prediction = get_core_ml_prediction(build, input_placeholders,
                input_values, backend=backend)

        ref0 = np.random.multinomial(n_sample, probs[0])
        ref1 = np.random.multinomial(n_sample, probs[1])

        pred0 = prediction[output_name].reshape(2, n_sample)[0]
        pred1 = prediction[output_name].reshape(2, n_sample)[1]

        # convert to bincount and validate probabilities
        pred0 = np.bincount(np.array(pred0).astype(np.int), minlength=n_class)
        pred1 = np.bincount(np.array(pred1).astype(np.int), minlength=n_class)

        assert np.allclose(np.true_divide(pred0, n_sample), probs[0], atol=1e-2)
        assert np.allclose(np.true_divide(pred0, n_sample),
                           np.true_divide(ref0, n_sample), atol=1e-2)

        assert np.allclose(np.true_divide(pred1, n_sample), probs[1], atol=1e-2)
        assert np.allclose(np.true_divide(pred1, n_sample),
                           np.true_divide(ref1, n_sample), atol=1e-2)

        # Test probs input
        input_placeholders = {'x': cb.placeholder(shape=(2, n_class))}
        input_values = {'x': np.array(probs)}

        def build(x):
            return [cb.random_categorical(
                x=x, size=n_sample, mode='probs', name=output_name)
            ]

        prediction = get_core_ml_prediction(build, input_placeholders,
                input_values, backend=backend)

        pred0 = prediction[output_name].reshape(2, n_sample)[0]
        pred1 = prediction[output_name].reshape(2, n_sample)[1]

        # convert to bincount and validate probabilities
        pred0 = np.bincount(np.array(pred0).astype(np.int), minlength=n_class)
        pred1 = np.bincount(np.array(pred1).astype(np.int), minlength=n_class)

        assert np.allclose(np.true_divide(pred0, n_sample), probs[0], atol=1e-2)
        assert np.allclose(np.true_divide(pred0, n_sample),
                           np.true_divide(ref0, n_sample), atol=1e-2)

        assert np.allclose(np.true_divide(pred1, n_sample), probs[1], atol=1e-2)
        assert np.allclose(np.true_divide(pred1, n_sample),
                           np.true_divide(ref1, n_sample), atol=1e-2)

    @pytest.mark.skip("TODO: rdar://59071295 (Add dynamic input / TF conversion in frontend for Random Distribution ops)")
    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, size',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [size for size in range(1, 10)]))
    def test_tf(self, use_cpu_only, backend, size):
        # TensorFlow's input is 2-D tensor with shape [batch_size, num_classes].
        input_shape = np.random.randint(low=2, high=6, size=2)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            ref = tf.random.categorical(x, size)
            run_compare_tf(graph, {x: np.random.rand(*input_shape)},
                           ref, use_cpu_only=use_cpu_only,
                           validate_shapes_only=True, backend=backend)


class TestRandomNormal:

    @pytest.mark.parametrize('use_cpu_only, backend',
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        """
        Construct op from builder and test output numerical parity.
        """

        x_val = np.array([1], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.relu(x=x),
                cb.random_normal(shape=np.array([2, 1, 3], np.int32), mean=1., stddev=0.),
                cb.random_normal(shape=np.array([3, 1, 2], np.int32), mean=0., stddev=0.),
            ]

        expected_outputs = [
            np.array(np.ones(shape=(1,)), np.float32),
            np.array(np.ones(shape=(2, 1, 3)), np.float32),
            np.array(np.zeros(shape=(3, 1, 2)), np.float32),
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values, expected_output_types,
                            expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.parametrize('use_cpu_only, backend, rank, mean',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                                 [1.0, 0.0]))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, rank, mean):
        shape = np.random.randint(low=1, high=4, size=rank).astype(np.int32)
        x_val = np.array([1], dtype=np.int32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.relu(x=x),
                cb.random_normal(shape=shape, mean=mean, stddev=0.)
            ]

        expected_outputs = [
            np.array(np.ones(shape=(1,)), np.float32),
            np.random.normal(loc=mean, scale=0., size=shape)
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values, expected_output_types,
                            expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skip("TODO: rdar://59071295 (Add dynamic input / TF conversion in frontend for Random Distribution ops)")
    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, mean',
                             itertools.product(
                                 [True],
                                 ['nnv2'],
                                 [0.]))
    def test_tf(self, use_cpu_only, backend, mean):
        input_shape = np.random.randint(low=2, high=6, size=2)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            ref = tf.keras.backend.random_normal(shape=(1, 2, 3), mean=mean, stddev=0.)
            run_compare_tf(graph, {x: np.random.rand(*input_shape)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)


class TestRandomUniform:

    @pytest.mark.parametrize('use_cpu_only, backend',
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        """
        Construct op from builder and test output numerical parity.
        """

        x_val = np.array([1], dtype=np.int32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.relu(x=x),
                cb.random_uniform(shape=np.array([2, 1, 3], np.int32), low=0., high=0.),
                cb.random_uniform(shape=np.array([3, 1, 2], np.int32), low=1., high=1.),
            ]

        expected_outputs = [
            np.array(np.ones(shape=(1,)), np.float32),
            np.array(np.zeros(shape=(2, 1, 3)), np.float32),
            np.array(np.ones(shape=(3, 1, 2)), np.float32),
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values, expected_output_types,
                            expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.parametrize('use_cpu_only, backend, rank, low, high',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                                 [0.0], [0.0]))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, rank, low, high):
        low_factor = np.random.randint(low=2, high=4)
        lo = int(np.power(1000, 1. / rank)) * low_factor
        hi = int(np.power(2000, 1. / rank)) * np.random.randint(low=low_factor, high=4)
        shape = np.random.randint(low=lo, high=hi, size=rank, dtype=np.int32)
        x_val = np.array([1], dtype=np.int32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.relu(x=x),
                cb.random_uniform(shape=shape, low=low, high=high)
            ]

        expected_outputs = [
            np.array(np.ones(shape=(1,)), np.float32),
            np.random.uniform(low=low, high=high, size=shape)
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values, expected_output_types,
                            expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @pytest.mark.skip("TODO: rdar://59071295 (Add dynamic input / TF conversion in frontend for Random Distribution ops)")
    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, low, high',
                             itertools.product(
                                 [True],
                                 ['nnv2'],
                                 [0.], [0.]))
    def test_tf(self, use_cpu_only, backend, low, high):
        input_shape = np.random.randint(low=2, high=6, size=2)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            ref = tf.keras.backend.random_uniform(shape=(1, 2, 3), minval=low, maxval=high)
            run_compare_tf(graph, {x: np.random.rand(*input_shape)},
                           ref, use_cpu_only=use_cpu_only, backend=backend)


class TestReduction:

    @pytest.mark.parametrize('use_cpu_only, backend, mode',
                             itertools.product(
                                 [True, False],
                                 ['nnv1'],
                                 ['argmax', 'argmin', 'l1_norm', 'l2_norm',
                                  'log_sum', 'log_sum_exp', 'max', 'mean',
                                  'min', 'prod', 'sum', 'sum_square'],
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend, mode):
        val = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=val.shape)}
        input_values = {'x': val}

        expected_output_types = (2, builtins.fp32)

        if mode == 'argmax':
            build = lambda x: cb.reduce_argmax(x=x, axis=1, keep_dims=False)
            expected_outputs = np.array([2., 2.], dtype=np.float32)
        elif mode == 'argmin':
            build = lambda x: cb.reduce_argmin(x=x, axis=1, keep_dims=False)
            expected_outputs = np.array([0., 0.], dtype=np.float32)
        elif mode == 'l1_norm':
            build = lambda x: cb.reduce_l1_norm(x=x, axes=[1], keep_dims=False)
            expected_outputs = np.array([6., 15.], dtype=np.float32)
        elif mode == 'l2_norm':
            build = lambda x: cb.reduce_l2_norm(x=x, axes=[1], keep_dims=False)
            expected_outputs = np.array([3.74165738, 8.77496438], dtype=np.float32)
        elif mode == 'log_sum':
            build = lambda x: cb.reduce_log_sum(x=x, axes=[1], keep_dims=False)
            expected_outputs = np.array([1.7917595, 2.70805025], dtype=np.float32)
        elif mode == 'log_sum_exp':
            build = lambda x: cb.reduce_log_sum_exp(x=x, axes=[1], keep_dims=False)
            expected_outputs = np.array([3.40760589, 6.40760612], dtype=np.float32)
        elif mode == 'max':
            build = lambda x: cb.reduce_max(x=x, axes=[1], keep_dims=False)
            expected_outputs = np.array([3., 6.], dtype=np.float32)
        elif mode == 'mean':
            build = lambda x: cb.reduce_mean(x=x, axes=[1], keep_dims=False)
            expected_outputs = np.array([2., 5.], dtype=np.float32)
        elif mode == 'min':
            build = lambda x: cb.reduce_min(x=x, axes=[1], keep_dims=False)
            expected_outputs = np.array([1., 4.], dtype=np.float32)
        elif mode == 'prod':
            build = lambda x: cb.reduce_prod(x=x, axes=[1], keep_dims=False)
            expected_outputs = np.array([6., 120.], dtype=np.float32)
        elif mode == 'sum':
            build = lambda x: cb.reduce_sum(x=x, axes=[1], keep_dims=False)
            expected_outputs = np.array([6., 15.], dtype=np.float32)
        elif mode == 'sum_square':
            build = lambda x: cb.reduce_sum_square(x=x, axes=[1], keep_dims=False)
            expected_outputs = np.array([14., 77.], dtype=np.float32)
        else:
            raise NotImplementedError()

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.parametrize(['axis', 'keep_dims'],
                             itertools.product([1, -3], [True, False]))
    def test_builder_eval(self, axis, keep_dims):
        x_val = _random_gen(shape=(1, 3, 4, 4), rand_min=-100., rand_max=100.)

        @ssa_fn
        def test_reduce_argmax():
            res = cb.reduce_argmax(x=x_val, axis=axis, keep_dims=keep_dims).val
            ref = np.argmax(x_val, axis=axis)
            assert is_close(ref, res)

        @ssa_fn
        def test_reduce_argmin():
            res = cb.reduce_argmin(x=x_val, axis=axis, keep_dims=keep_dims).val
            ref = np.argmin(x_val, axis=axis)
            assert is_close(ref, res)

        @ssa_fn
        def test_reduce_l1_norm():
            res = cb.reduce_l1_norm(x=x_val, axes=[axis], keep_dims=keep_dims).val
            ref = np.sum(np.abs(x_val), axis=axis, keepdims=keep_dims)
            assert is_close(ref, res)

        @ssa_fn
        def test_reduce_l2_norm():
            res = cb.reduce_l2_norm(x=x_val, axes=[axis], keep_dims=keep_dims).val
            ref = np.sqrt(np.sum(np.square(x_val), axis=axis, keepdims=keep_dims))
            assert is_close(ref, res)

        @ssa_fn
        def test_reduce_log_sum():
            x_val = _random_gen(shape=(1, 3, 4, 4), rand_min=0., rand_max=100.)
            res = cb.reduce_log_sum(x=x_val, axes=[axis], keep_dims=keep_dims).val
            ref = np.log(np.sum(x_val, axis=axis, keepdims=keep_dims))
            assert is_close(ref, res)

        @ssa_fn
        def test_reduce_log_sum_exp():
            res = cb.reduce_log_sum_exp(x=x_val, axes=[axis], keep_dims=keep_dims).val
            ref = scipy.special.logsumexp(x_val, axis=axis, keepdims=keep_dims)
            assert is_close(ref, res)

        @ssa_fn
        def test_reduce_max():
            res = cb.reduce_max(x=x_val, axes=[axis], keep_dims=keep_dims).val
            ref = np.max(x_val, axis=axis, keepdims=keep_dims)
            assert is_close(ref, res)

        @ssa_fn
        def test_reduce_mean():
            res = cb.reduce_mean(x=x_val, axes=[axis], keep_dims=keep_dims).val
            ref = np.mean(x_val, axis=axis, keepdims=keep_dims)
            assert is_close(ref, res)

        @ssa_fn
        def test_reduce_min():
            res = cb.reduce_min(x=x_val, axes=[axis], keep_dims=keep_dims).val
            ref = np.min(x_val, axis=axis, keepdims=keep_dims)
            assert is_close(ref, res)

        @ssa_fn
        def test_reduce_prod():
            res = cb.reduce_prod(x=x_val, axes=[axis], keep_dims=keep_dims).val
            ref = np.prod(x_val, axis=axis, keepdims=keep_dims)
            assert is_close(ref, res)

        @ssa_fn
        def test_reduce_sum():
            res = cb.reduce_sum(x=x_val, axes=[axis], keep_dims=keep_dims).val
            ref = np.sum(x_val, axis=axis, keepdims=keep_dims)
            assert is_close(ref, res)

        @ssa_fn
        def test_reduce_sum_square():
            res = cb.reduce_sum_square(x=x_val, axes=[axis], keep_dims=keep_dims).val
            ref = np.sum(np.square(x_val), axis=axis, keepdims=keep_dims)
            assert is_close(ref, res)

        test_reduce_argmax()
        test_reduce_argmin()
        test_reduce_l1_norm()
        test_reduce_l2_norm()
        test_reduce_log_sum()
        test_reduce_log_sum_exp()
        test_reduce_max()
        test_reduce_mean()
        test_reduce_min()
        test_reduce_prod()
        test_reduce_sum()
        test_reduce_sum_square()

    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 ['nnv1'],
                             ))
    def test_builder_to_backend_symbolic(self, use_cpu_only, backend):
        # TODO: variadic (rdar://59559656)

        s0 = get_new_symbol()

        val = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=(s0, 3))}
        input_values = {'x': val}

        def build(x):
            return [
                cb.reduce_argmax(x=x, axis=1, keep_dims=True),
                cb.reduce_argmin(x=x, axis=0, keep_dims=True)
            ]

        expected_output_types = [
            (s0, 1, builtins.fp32),
            (1, 3, builtins.fp32)
        ]
        expected_outputs = [
            np.array([2., 2.], dtype=np.float32),
            np.array([[0.], [0.], [0.]], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only,
                            frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank_and_axes, keep_dims, mode',
                             itertools.product(
                                 [True, False],
                                 ['nnv1'],
                                 [(1, (-1,)), (2, (0,)), (2, (-1, 0)), (3, (1, -3)), (3, (-2,)),
                                  (4, (0, 1, 2)), (4, (-2, -1, 0)), (4, (1, -2)), (5, (-3, -1)), (5, (0, -1, 1, -2))],
                                 [True, False],
                                 ['l2', 'max', 'mean', 'min', 'prod', 'sum']
                                 # TODO: add 'log_sum_exp' tests which requires IsFinate op conversion.
                                 # rdar://59563732 (Add support for TensorFlow IsFinate op conversion.)
                             ))
    def test_tf(self, use_cpu_only, backend, rank_and_axes, keep_dims, mode):
        rank, axes = rank_and_axes
        shape = np.random.randint(low=2, high=4, size=rank)

        def test_tf_argmax():
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=shape)
                res = tf.math.argmax(x, axis=axes[0])
                run_compare_tf(graph, {x: np.random.rand(*shape)}, res,
                               use_cpu_only=use_cpu_only, frontend_only=False,
                               backend=backend)

        def test_tf_argmin():
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=shape)
                res = tf.math.argmin(x, axis=axes[0])
                run_compare_tf(graph, {x: np.random.rand(*shape)}, res,
                               use_cpu_only=use_cpu_only, frontend_only=False,
                               backend=backend)

        def test_tf_reduction():
            if len(axes) == rank and not keep_dims:
                return  # TODO <rdar://problem/59152311> NNV2: Add rank 0 and dim size 0 related tests for every op

            if mode == 'l2':
                tf_op = tf.math.reduce_euclidean_norm
            elif mode == 'log_sum_exp':
                tf_op = tf.math.reduce_logsumexp
            elif mode == 'max':
                tf_op = tf.reduce_max
            elif mode == 'mean':
                tf_op = tf.reduce_mean
            elif mode == 'min':
                tf_op = tf.reduce_min
            elif mode == 'prod':
                tf_op = tf.reduce_prod
            elif mode == 'sum':
                tf_op = tf.reduce_sum
            else:
                raise NotImplementedError()

            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=shape)
                res = tf_op(x, axis=axes, keepdims=keep_dims)
                run_compare_tf(graph, {x: np.random.rand(*shape)}, res,
                               use_cpu_only=use_cpu_only, frontend_only=False,
                               backend=backend)

        test_tf_argmax()
        test_tf_argmin()
        test_tf_reduction()


class TestRound:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1.2, 2, -3.4],[4.6, -5, 6.7]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.round(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-1, 2, -3], [5, -5, 7]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-1000, rand_max=1000)
        v = cb.round(x=x_val)
        assert is_close(np.round(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.round(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-1000, rand_max=1000)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestRsqrt:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.rsqrt(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[1., 0.70710678, 0.57735027],
                                     [0.5, 0.4472136, 0.40824829]],
                                     dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=0.5, rand_max=1000)
        v = cb.rsqrt(x=x_val)
        assert is_close(1. / np.sqrt(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.rsqrt(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=0.5, rand_max=1000)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestSin:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.sin(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-0.84147098, 0.90929743, -0.14112001],
                                     [-0.7568025 , 0.95892427, -0.2794155]],
                                     dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-1000, rand_max=1000)
        v = cb.sin(x=x_val)
        assert is_close(np.sin(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)],
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        rand_range = 1000
        if not use_cpu_only:
            rand_range = 10
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.sin(x)
            run_compare_tf(graph, {x: _random_gen(input_shape,
                rand_min=-rand_range, rand_max=rand_range)},
                res, use_cpu_only=use_cpu_only, frontend_only=False,
                backend=backend)

class TestSinh:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.sinh(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-1.1752, 3.62686, -10.017874],
                                     [27.289917 , -74.20321, 201.71315]],
                                     dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-10, rand_max=10)
        v = cb.sinh(x=x_val)
        assert is_close(np.sinh(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.sinh(x)
            run_compare_tf(graph, {x: _random_gen(input_shape,
                rand_min=-10, rand_max=10)},
                res, use_cpu_only=use_cpu_only,
                frontend_only=False, backend=backend)

class TestSqrt:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.sqrt(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[1., 1.41421356, 1.73205081],
                                     [2., 2.23606798, 2.44948974]],
                                     dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=0.5, rand_max=1000)
        v = cb.sqrt(x=x_val)
        assert is_close(np.sqrt(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.sqrt(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=0.5, rand_max=1000)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestSub:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=x.shape), "y": cb.placeholder(shape=y.shape)}
        input_values = {"x": x, "y": y}

        def build(x, y):
            return cb.sub(x=x, y=y)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[2, 0, 6], [0, 10, 0]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        y_val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        expected_outputs = np.array([[2, 0, 6], [0, 10, 0]], dtype=np.float32)
        v = cb.sub(x=x_val, y=y_val)
        assert is_close(expected_outputs, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 4)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        x_shape = list(np.random.randint(low=2, high=6, size=rank))
        y_shape = x_shape[:]
        for i in range(rank):
            if np.random.randint(4) == 0:
                y_shape[i] = 1
        if np.random.randint(2) == 0:
            y_shape = [1] + y_shape

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=x_shape)
            y = tf.placeholder(tf.float32, shape=y_shape)
            res = tf.subtract(x, y)
            run_compare_tf(graph, {x: np.random.randint(low=-5, high=3, size=x_shape).astype(np.float32),
                                   y: np.random.randint(low=-5, high=3, size=y_shape).astype(np.float32)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)

class TestTan:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.tan(x=x)
        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-1.5574, -2.185, 0.1425],
                                     [1.15782, 3.3805, -0.291]],
                                     dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(1, 2, 4, 3), rand_min=-1000, rand_max=1000)
        v = cb.tan(x=x_val)
        assert is_close(np.tan(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        input_shape = np.random.randint(low=2, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.tan(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-1000, rand_max=1000)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestReLU:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.relu(x=x)

        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[0, 2, 0], [4, 0, 6]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.relu(x=x_val)
        assert is_close(np.maximum(x_val, 0), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]))
    def test_tf(self, use_cpu_only, backend, rank, **kwargs):
        input_shape = np.random.randint(low=1, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.nn.relu(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-1, rand_max=1)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestERF:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.erf(x=x)

        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-0.8427007929497148, 0.9953222650189527, -0.9999779095030014],
                                     [0.9999999845827421, -0.9999999999984626, 1.0]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.erf(x=x_val)
        assert is_close(scipy.special.erf(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]))
    def test_tf(self, use_cpu_only, backend, rank, **kwargs):
        input_shape = np.random.randint(low=1, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.math.erf(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-1, rand_max=1)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestSigmoid:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.sigmoid(x=x)

        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[0.2689414213699951, 0.8807970779778823, 0.04742587],
                                     [0.98201376, 0.00669285, 0.9975274]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.sigmoid(x=x_val)
        assert is_close(1/(1 + np.exp(-x_val)), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]))
    def test_tf(self, use_cpu_only, backend, rank, **kwargs):
        input_shape = np.random.randint(low=1, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.math.sigmoid(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-1, rand_max=1)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestHardSigmoid:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.sigmoid_hard(x=x, alpha=1.0, beta=2.0)

        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[1., 1., 0.],
                                     [1., 0., 1.]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        alpha = 1.0
        beta = 2.0
        v = cb.sigmoid_hard(x=x_val, alpha=alpha, beta=beta)
        assert is_close(np.minimum(np.maximum((alpha * x_val) + beta, 0), 1), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]))
    def test_tf(self, use_cpu_only, backend, rank, **kwargs):
        pass


class TestSoftplus:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.softplus(x=x)

        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[0.31326166, 2.126928  , 0.04858733],
                                     [4.01815   , 0.00671535, 6.0024757 ]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.softplus(x=x_val)
        assert is_close(np.log(1 + np.exp(-np.abs(x_val))) + np.maximum(x_val, 0), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]))
    def test_tf(self, use_cpu_only, backend, rank, **kwargs):
        input_shape = np.random.randint(low=1, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.math.softplus(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-1, rand_max=1)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestSoftsign:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.softsign(x=x)

        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-0.5       ,  0.66666667, -0.75      ],
                                     [ 0.8       , -0.83333333,  0.85714286]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.softsign(x=x_val)
        assert is_close(x_val / (1 + np.abs(x_val)), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]))
    def test_tf(self, use_cpu_only, backend, rank, **kwargs):
        input_shape = np.random.randint(low=1, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.math.softsign(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-1, rand_max=1)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestTanh:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.tanh(x=x)

        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-0.7615942,  0.9640276, -0.9950548],
                                     [ 0.9993293, -0.9999092,  0.9999877]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.tanh(x=x_val)
        assert is_close(np.tanh(x_val), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]))
    def test_tf(self, use_cpu_only, backend, rank, **kwargs):
        input_shape = np.random.randint(low=1, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.math.tanh(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-1, rand_max=1)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestClampedRelu:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {
            "x": cb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            return cb.clamped_relu(x=x, alpha=1.0, beta=2.0)

        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[1, 2, 1], [2, 1, 2]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.clamped_relu(x=x_val, alpha=1.0, beta=2.0)
        assert is_close(np.minimum(np.maximum(x_val, 1.0), 2.0), v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]))
    def test_tf(self, use_cpu_only, backend, rank, **kwargs):
        pass


class TestELU:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {
            "x": cb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            return cb.elu(x=x, alpha=2.0)

        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-1.2642411, 2.0       , -1.9004259],
                                     [ 4.0      , -1.9865241,  6.0      ]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.elu(x=x_val, alpha=2.0)

        b = np.copy(x_val)
        b[b < 0] = 2.0 * (np.exp(b[b < 0]) - 1)

        assert is_close(b, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]))
    def test_tf(self, use_cpu_only, backend, rank, **kwargs):
        input_shape = np.random.randint(low=1, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.nn.elu(x)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-1, rand_max=1)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)
        pass


class TestLeakyReLU:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape),}
        input_values = {"x": t}

        def build(x):
            return cb.leaky_relu(x=x, alpha=2.0)

        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[-2, 2, -6], [4, -10, 6]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.leaky_relu(x=x_val, alpha=2.0)

        b = np.copy(x_val)
        b[b < 0] *= 2.0
        assert is_close(b, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]))
    def test_tf(self, use_cpu_only, backend, rank, **kwargs):
        input_shape = np.random.randint(low=1, high=6, size=rank)
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=input_shape)
            res = tf.nn.leaky_relu(x, 0.2)
            run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=-1, rand_max=1)},
                           res, use_cpu_only=use_cpu_only,
                           frontend_only=False, backend=backend)


class TestLinear:

    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array([[-4.7182, 11.94],
                          [-3.3939, 9.2166]], dtype=np.float32)
        weight_val = np.array([[1.2313, -0.095],
                               [-1.4075, -0.8816]], dtype=np.float32)
        bias_val = np.array([1., 2.], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape)}
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.linear(x=x, weight=weight_val, bias=bias_val)
            ]

        expected_output_types = [(2, 2, builtins.fp32)]
        expected_outputs = [
            np.array([[-5.9438195, -1.8854373],
                      [-4.054486, -1.3484411]], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = _random_gen(shape=(2, 2), rand_min=-37, rand_max=64)
        weight_val = _random_gen(shape=(2, 2), rand_min=-91, rand_max=84)
        bias_val = _random_gen(shape=(2,), rand_min=0., rand_max=9.)
        v = cb.linear(x=x_val, weight=weight_val, bias=bias_val)
        assert is_close(np.matmul(x_val, weight_val.T) + bias_val, v.val)

    @pytest.mark.parametrize('use_cpu_only, backend, dim',
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [2, 4, 8]))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, dim):
        shape = np.array([dim, dim])
        x_val = np.random.rand(*shape)
        weight_val = np.random.rand(*shape).astype(np.float32)
        bias_val = np.random.rand(dim).astype(np.float32)
        input_placeholders = {
            'x': cb.placeholder(shape=x_val.shape),
        }
        input_values = {'x': x_val}

        def build(x):
            return [
                cb.linear(x=x, weight=weight_val, bias=bias_val)
            ]

        expected_outputs = [
            np.matmul(x_val, np.transpose(weight_val)) + bias_val
        ]

        expected_output_types = [o.shape[:] + (builtins.fp32,) for o in expected_outputs]

        run_compare_builder(build, input_placeholders, input_values, expected_output_types,
                            expected_outputs=expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)


class TestLinearActivation:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        input_placeholders = {
            "x": cb.placeholder(shape=t.shape)
        }
        input_values = {"x": t}

        def build(x):
            return cb.linear_activation(x=x, alpha=2.0, beta=3.0)

        expected_output_types = (2, 3, builtins.fp32)
        expected_outputs = np.array([[1, 7, -3], [11, -7, 15]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.linear_activation(x=x_val, alpha=2.0, beta=3.0)
        assert is_close(x_val * 2.0 + 3.0, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(1, 6)]))
    def test_tf(self, use_cpu_only, backend, rank, **kwargs):
        pass


class TestPReLU:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[[-1, 3, 6]], [[-1, 2, -3]], [[4, -5, 6]]], dtype=np.float32)
        input_placeholders = {"x": cb.placeholder(shape=t.shape)}
        input_values = {"x": t}

        def build(x):
            return cb.prelu(x=x, alpha=np.array([1, 2, 3], dtype=np.float32))

        expected_output_types = (3, 1, 3, builtins.fp32)
        expected_outputs = np.array([[[-1, 3, 6]], [[-2, 2, -6]], [[4, -15, 6]]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[[-1, 3, 6]], [[-1, 2, -3]], [[4, -5, 6]]], dtype=np.float32)
        alpha = np.array([1, 2, 3], dtype=np.float32)
        v = cb.prelu(x=x_val, alpha=alpha)

        alpha_br = alpha

        for i in range(1, x_val.shape[-3]):
            alpha_br = np.expand_dims(alpha_br, i)

        x_pos = np.maximum(x_val, 0)
        b = np.minimum(x_val, 0)

        assert is_close(x_pos + b * alpha_br, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(3, 5)]))
    def test_tf(self, use_cpu_only, backend, rank, **kwargs):
        pass


class TestParametricSoftplus:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        t = np.array([[[-1, 3, 6]], [[-1, 2, -3]], [[4, -5, 6]]], dtype=np.float32)
        input_placeholders = {
            "x": cb.placeholder(shape=t.shape)
        }
        input_values = {"x": t}

        def build(x):
            return cb.softplus_parametric(x=x,
                                          alpha=np.array([1, 2, 3], dtype=np.float32),
                                          beta=np.array([4, 5, 6], dtype=np.float32))

        expected_output_types = (3, 1, 3, builtins.fp32)
        expected_outputs = np.array([[[1.8142700e-02, 1.2000000e+01, 2.4000000e+01]],
                                     [[1.3427734e-02, 2.0000000e+01, 7.1525574e-07]],
                                     [[7.2000000e+01, 0.0000000e+00, 1.0800000e+02]]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array([[[-1, 3, 6]], [[-1, 2, -3]], [[4, -5, 6]]], dtype=np.float32)
        v = cb.softplus_parametric(x=x_val,
                                   alpha=np.array([1, 2, 3], dtype=np.float32),
                                   beta=np.array([4, 5, 6], dtype=np.float32))

        alpha_br = np.array([1, 2, 3], dtype=np.float32)
        beta_br = np.array([4, 5, 6], dtype=np.float32)
        for i in range(1, x_val.shape[-3]):
            alpha_br = np.expand_dims(alpha_br, i)
            beta_br = np.expand_dims(beta_br, i)
        out = alpha_br * np.exp(x_val * beta_br)

        assert is_close(out, v.val)

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(3, 5)]))
    def test_tf(self, use_cpu_only, backend, rank, **kwargs):
        pass

class TestPad():
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                ['nnv2'],
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        def test_constant_mode():
            t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            pad = np.array([[1, 1], [2, 2]], dtype=np.int32)
            input_placeholders = {"x": cb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return cb.pad(x=x, pad=pad, mode="constant", constant_val=0.0)
            expected_output_types = (4, 7, builtins.fp32)
            expected_outputs = np.array([[0., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 1., 2., 3., 0., 0.],
                                         [0., 0., 4., 5., 6., 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 0.]],
                                         dtype=np.float32)

            run_compare_builder(build, input_placeholders, input_values,
                                expected_output_types, expected_outputs,
                                use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)

        def test_constant_mode_constant_val():
            t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            pad = np.array([[1, 1], [2, 2]], dtype=np.int32)
            input_placeholders = {"x": cb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return cb.pad(x=x, pad=pad, mode="constant", constant_val=0.5)
            expected_output_types = (4, 7, builtins.fp32)
            expected_outputs = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                         [0.5, 0.5, 1.,  2.,  3.,  0.5, 0.5],
                                         [0.5, 0.5, 4.,  5.,  6.,  0.5, 0.5],
                                         [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
                                         dtype=np.float32)

            run_compare_builder(build, input_placeholders, input_values,
                                expected_output_types, expected_outputs,
                                use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)

        def test_reflect_mode():
            t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            pad = np.array([[1, 1], [2, 2]], dtype=np.int32)
            input_placeholders = {"x": cb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return cb.pad(x=x, pad=pad, mode="reflect")
            expected_output_types = (4, 7, builtins.fp32)
            expected_outputs = np.array([[6., 5., 4., 5., 6., 5., 4.],
                                         [3., 2., 1., 2., 3., 2., 1.],
                                         [6., 5., 4., 5., 6., 5., 4.],
                                         [3., 2., 1., 2., 3., 2., 1.]],
                                         dtype=np.float32)

            run_compare_builder(build, input_placeholders, input_values,
                                expected_output_types, expected_outputs,
                                use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)

        def test_replicate_mode():
            t = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            pad = np.array([[1, 1], [2, 2]], dtype=np.int32)
            input_placeholders = {"x": cb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return cb.pad(x=x, pad=pad, mode="replicate")
            expected_output_types = (4, 7, builtins.fp32)
            expected_outputs = np.array([[1., 1., 1., 2., 3., 3., 3.],
                                         [1., 1., 1., 2., 3., 3., 3.],
                                         [4., 4., 4., 5., 6., 6., 6.],
                                         [4., 4., 4., 5., 6., 6., 6.]],
                                         dtype=np.float32)

            run_compare_builder(build, input_placeholders, input_values,
                                expected_output_types, expected_outputs,
                                use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)

        def test_constant_general():
            t = np.arange(12, dtype=np.float32).reshape([2, 2, 3])
            pad = np.array([[1, 1], [2, 2], [1, 1]], dtype=np.int32)
            input_placeholders = {"x": cb.placeholder(shape=t.shape)}
            input_values = {"x": t}

            def build(x):
                return cb.pad(x=x, pad=pad, mode="constant", constant_val=0.0)
            expected_output_types = (4, 6, 5, builtins.fp32)
            expected_outputs = np.pad(t, pad, mode="constant")

            run_compare_builder(build, input_placeholders, input_values,
                                expected_output_types, expected_outputs,
                                use_cpu_only=use_cpu_only,
                                frontend_only=False, backend=backend)
        # Test different modes
        test_constant_mode()
        test_constant_mode_constant_val()
        test_reflect_mode()
        test_replicate_mode()
        test_constant_general()

    @ssa_fn
    def test_builder_eval(self):
        def test_constant_mode():
            x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            v = cb.pad(x=x_val, pad=np.array([[1, 1], [2, 2]], dtype=np.int32), mode="constant", constant_val=0.0)
            expected_outputs = np.array([[0., 0., 0., 0., 0., 0., 0.],
                                         [0., 0., 1., 2., 3., 0., 0.],
                                         [0., 0., 4., 5., 6., 0., 0.],
                                         [0., 0., 0., 0., 0., 0., 0.]],
                                         dtype=np.float32)
            assert is_close(expected_outputs, v.val)

        def test_reflect_mode():
            x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            v = cb.pad(x=x_val, pad=np.array([[1, 1], [2, 2]], dtype=np.int32), mode="reflect")
            expected_outputs = np.array([[6., 5., 4., 5., 6., 5., 4.],
                                         [3., 2., 1., 2., 3., 2., 1.],
                                         [6., 5., 4., 5., 6., 5., 4.],
                                         [3., 2., 1., 2., 3., 2., 1.]],
                                         dtype=np.float32)
            assert is_close(expected_outputs, v.val)

        def test_replicate_mode():
            x_val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            v = cb.pad(x=x_val, pad=np.array([[1, 1], [2, 2]], dtype=np.int32), mode="replicate")
            expected_outputs = np.array([[1., 1., 1., 2., 3., 3., 3.],
                                         [1., 1., 1., 2., 3., 3., 3.],
                                         [4., 4., 4., 5., 6., 6., 6.],
                                         [4., 4., 4., 5., 6., 6., 6.]],
                                         dtype=np.float32)
            assert is_close(expected_outputs, v.val)

        def test_constant_general():
            x_val = np.arange(12, dtype=np.float32).reshape([2, 2, 3])
            pad = np.array([[1, 1], [2, 2], [1, 1]], dtype=np.int32)
            v = cb.pad(x=x_val, pad=pad, mode="constant", constant_val=0.0)
            expected_outputs = np.pad(x_val, pad, mode="constant")
            assert is_close(expected_outputs, v.val)

        # Test different modes
        test_constant_mode()
        test_reflect_mode()
        test_replicate_mode()
        test_constant_general()

    @pytest.mark.skipif(not HAS_TF, reason="Tensorflow not installed.")
    @pytest.mark.parametrize("use_cpu_only, backend, rank",
                             itertools.product(
                                 [True, False],
                                 ['nnv2'],
                                 [rank for rank in range(2, 3)]
                             )
                             )
    def test_tf(self, use_cpu_only, backend, rank):
        def test_constant_mode():
            input_shape = np.random.randint(low=2, high=10, size=rank)
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                paddings = tf.constant([[1, 1,], [2, 2]])
                res = tf.pad(x, paddings=paddings)
                run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=0.2, rand_max=1000)},
                            res, use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)

        def test_reflect_mode():
            input_shape = np.random.randint(low=3, high=10, size=rank)
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                paddings = tf.constant([[1, 1], [2, 2]])
                res = tf.pad(x, paddings=paddings, mode="REFLECT")
                run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=0.2, rand_max=1000)},
                            res, use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)

        def test_constant_general():
            input_shape = np.random.randint(low=2, high=10, size=rank)
            padding_val = np.random.randint(low=0, high=rank, size=(rank, 2), dtype=np.int32)
            with tf.Graph().as_default() as graph:
                x = tf.placeholder(tf.float32, shape=input_shape)
                paddings = tf.constant(padding_val)
                res = tf.pad(x, paddings=paddings, mode="CONSTANT")
                run_compare_tf(graph, {x: _random_gen(input_shape, rand_min=0.2, rand_max=1000)},
                            res, use_cpu_only=use_cpu_only,
                            frontend_only=False, backend=backend)

        # Test different modes
        test_constant_mode()
        test_reflect_mode()
        test_constant_general()
        # Tensorflow does not support replicate mode, hence skipping replicate test on tf

class TestSpaceToDepth:

    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 ['nnv1'],
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        # original input type is (1, 1, 2, 2, fp32)
        val = np.array([[[[7., 9.], [4., 6.]]]], dtype=np.float32)
        input_placeholders = {'x': cb.placeholder(shape=val.shape)}
        input_values = {'x': val}

        def build(x):
            return [cb.space_to_depth(x=x, block_size=2)]

        expected_output_types = (1, 4, 1, 1, builtins.fp32)
        expected_outputs = np.array([[[[7.]], [[9.]], [[4.]], [[6.]]]], dtype=np.float32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, shape, block_size',
                             itertools.product(
                                 [True, False],
                                 ['nnv1'],
                                 [(1, 6, 6, 1), (1, 12, 12, 1), (1, 6, 6, 3)],
                                 [2, 3]
                             ))
    def test_tf(self, use_cpu_only, backend, shape, block_size):
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            ref = tf.space_to_depth(x, block_size)
            run_compare_tf(graph, {x: np.random.rand(*shape)}, ref,
                           use_cpu_only=use_cpu_only, backend=backend)

class TestLSTM:
    @pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed.")
    @pytest.mark.parametrize(argnames=["use_cpu_only", "backend", "seq_len",
                             "batch_size", "input_size", "hidden_size", "has_bias",
                             "output_sequence", "direction", "symbolic"],
                             argvalues=
                             itertools.product(
                                [True, False],
                                ['nnv1'],
                                [1, 8],
                                [1, 32],
                                [1, 64],
                                [1, 16],
                                [True, False],
                                [True, False],
                                ["forward", "reverse"],
                                [True, False]
                                )
                            )
    def test_builder_to_backend_smoke_unilstm(self, use_cpu_only, backend, seq_len, batch_size, input_size,
                                              hidden_size, has_bias, output_sequence, direction, symbolic):
        # TODO: <rdar://problem/59540160> [NNv2] LSTM layer- Implement eval and tf register routine
        # Testing 1. peephole values
        #         2. clip values

        torch.manual_seed(50)
        rnn = torch.nn.LSTM(input_size, hidden_size, 1, bias=has_bias)
        state_dict = rnn.state_dict()

        ih_wt = state_dict['weight_ih_l0'].detach().numpy()
        hh_wt = state_dict['weight_hh_l0'].detach().numpy()

        # Make weight compatible to CoreML format
        def ifzo_to_ifoz(x):
            i, f, z, o = np.split(x, 4)
            return np.concatenate([i, f, o, z], axis=0)

        w = np.concatenate([ih_wt, hh_wt], axis=1)
        w = ifzo_to_ifoz(w).transpose()
        # ih_wt = ifzo_to_ifoz(ih_wt).transpose()
        # hh_wt = ifzo_to_ifoz(hh_wt).transpose()
        # w = np.concatenate([ih_wt, hh_wt], axis=0)

        b = None
        if has_bias:
            ih_b = state_dict['bias_ih_l0'].detach().numpy()
            hh_b = state_dict['bias_hh_l0'].detach().numpy()
            ih_b = ifzo_to_ifoz(ih_b).transpose()
            hh_b = ifzo_to_ifoz(hh_b).transpose()
            b = np.stack([ih_b, hh_b], axis=0)

        t = torch.randn(seq_len, batch_size, input_size)
        h0 = torch.randn(1, batch_size, hidden_size)
        c0 = torch.randn(1, batch_size, hidden_size)

        n_t = t
        if direction == "reverse":
            n_t = torch.flip(n_t, [0])

        output, (hn, cn) = rnn(n_t, (h0, c0))
        if output_sequence == False:
            output = output[-1].unsqueeze(0)

        output = output.detach().numpy()
        hn = hn.detach().numpy()
        cn = cn.detach().numpy()

        t = np.reshape(t.detach().numpy(), [seq_len, batch_size, input_size])
        h = np.reshape(h0.detach().numpy().squeeze(0), [batch_size, hidden_size])
        c = np.reshape(c0.detach().numpy().squeeze(0), [batch_size, hidden_size])

        if symbolic:
            batch_size = get_new_symbol()
            seq_len = get_new_symbol()

        input_shape = [seq_len, batch_size, input_size]
        h_shape = [batch_size, hidden_size]
        c_shape = [batch_size, hidden_size]

        expected_output_types = [(seq_len if output_sequence else 1, batch_size, hidden_size, builtins.fp32),
                                 (batch_size, hidden_size, builtins.fp32),
                                 (batch_size, hidden_size, builtins.fp32)]
        expected_outputs = [output, hn, cn]

        input_placeholders = {"x": cb.placeholder(shape=input_shape),
                              "initial_h": cb.placeholder(shape=h_shape),
                              "initial_c": cb.placeholder(shape=c_shape),}
        input_values = {"x": t, "initial_h": h, "initial_c": c}

        def build(x, initial_h, initial_c):
            arguments = {
                        "x":x,
                        "initial_h":initial_h,
                        "initial_c":initial_c,
                        "weight":w,
                        "direction":direction,
                        "output_sequence":output_sequence,
                        }
            # If bias is provided, add in arguments
            if b is not None:
                arguments["bias"] = b
            return cb.lstm(**arguments)

    @pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed.")
    @pytest.mark.parametrize(argnames=["use_cpu_only", "backend", "seq_len", "batch_size", "input_size",
                                       "hidden_size", "has_bias", "output_sequence", "symbolic"],
                             argvalues=
                             itertools.product(
                                [True],
                                ['nnv1'],
                                [1, 8],
                                [1, 32],
                                [1, 64],
                                [2, 16],
                                [True, False],
                                [True, False],
                                [True, False]
                                )
                            )
    def test_builder_to_backend_smoke_bidirlstm(self, use_cpu_only, backend, seq_len, batch_size, input_size,
                                                hidden_size, has_bias, output_sequence, symbolic):
        def _pytorch_hidden_to_coreml(x):
            x = x.detach().numpy()
            # Split of Direction axis
            f, b = np.split(x, 2, axis=0)
            # Concat on Hidden Size axis
            x    = np.concatenate([f, b], axis=2)
            x    = np.squeeze(x, axis=0)
            return x
        direction = "bidirectional"
        torch.manual_seed(20)
        rnn = torch.nn.LSTM(input_size, hidden_size, 1, bidirectional=True, bias=has_bias)
        state_dict = rnn.state_dict()

        ih_wt = state_dict['weight_ih_l0'].detach().numpy()
        hh_wt = state_dict['weight_hh_l0'].detach().numpy()
        ih_wt_r = state_dict['weight_ih_l0_reverse'].detach().numpy()
        hh_wt_r = state_dict['weight_hh_l0_reverse'].detach().numpy()

        f_wt = np.concatenate([ih_wt, hh_wt], axis=1)
        r_wt = np.concatenate([ih_wt_r, hh_wt_r], axis=1)

        def ifzo_to_ifoz(x):
            i, f, z, o = np.split(x, 4)
            return np.concatenate([i, f, o, z], axis=0)

        f_wt = ifzo_to_ifoz(f_wt).transpose()
        r_wt = ifzo_to_ifoz(r_wt).transpose()
        w = np.concatenate([f_wt, r_wt], axis=1)

        b = None
        if has_bias:
            ih_b = state_dict['bias_ih_l0'].detach().numpy()
            hh_b = state_dict['bias_hh_l0'].detach().numpy()
            ih_b_r = state_dict['bias_ih_l0_reverse'].detach().numpy()
            hh_b_r = state_dict['bias_hh_l0_reverse'].detach().numpy()
            # Convert forward bias into [2, 4*H]
            ih_b = ifzo_to_ifoz(ih_b)
            hh_b = ifzo_to_ifoz(hh_b)
            f_b = np.stack([ih_b, hh_b], axis=0)
            # Convert reverse bias into [2, 4*H]
            ih_b_r = ifzo_to_ifoz(ih_b_r)
            hh_b_r = ifzo_to_ifoz(hh_b_r)
            r_b = np.stack([ih_b_r, hh_b_r], axis=0)
            # Final bias of [2, 2*4*H]
            b = np.concatenate([f_b, r_b], axis=1)

        t = torch.randn(seq_len, batch_size, input_size)
        h0 = torch.randn(2, batch_size, hidden_size)
        c0 = torch.randn(2, batch_size, hidden_size)

        output, (hn, cn) = rnn(t, (h0, c0))
        if output_sequence == False:
            output_f = output[-1].unsqueeze(0)[:,:,:hidden_size]
            output_r = output[0].unsqueeze(0)[:,:,hidden_size:]
            output = torch.cat([output_f, output_r], axis=2)

        output = output.detach().numpy()
        hn = _pytorch_hidden_to_coreml(hn)
        cn = _pytorch_hidden_to_coreml(cn)

        if symbolic:
            batch_size = get_new_symbol()
            seq_len = get_new_symbol()

        input_shape = [seq_len, batch_size, input_size]
        h_shape = [batch_size, 2*hidden_size]
        c_shape = [batch_size, 2*hidden_size]

        expected_output_types = [(seq_len if output_sequence else 1, batch_size, 2*hidden_size, builtins.fp32),
                                 (batch_size, 2*hidden_size, builtins.fp32),
                                 (batch_size, 2*hidden_size, builtins.fp32)]
        expected_outputs = [output, hn, cn]

        t = t.detach().numpy()
        h = _pytorch_hidden_to_coreml(h0)
        c = _pytorch_hidden_to_coreml(c0)

        input_placeholders = {"x": cb.placeholder(shape=input_shape),
                              "initial_h": cb.placeholder(shape=h_shape),
                              "initial_c": cb.placeholder(shape=c_shape)}
        input_values = {"x": t, "initial_h": h, "initial_c": c}

        def build(x, initial_h, initial_c):
            arguments = {
                        "x":x,
                        "initial_h":initial_h,
                        "initial_c":initial_c,
                        "weight":w,
                        "direction":direction,
                        "output_sequence":output_sequence,
                        }
            # If bias is provided, add in arguments
            if b is not None:
                arguments["bias"] = b
            return cb.lstm(**arguments)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)
