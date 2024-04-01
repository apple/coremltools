#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.ops.tests.iOS14 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import (
    UNK_SYM,
    construct_inputs_from_placeholders,
    run_compare_builder,
)
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import random_gen, ssa_fn


class TestSelect:
    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        cond_val = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]], dtype=np.float32)
        a_val = np.array([[3, 1, 1], [1, 4, 1], [5, 6, 1]], dtype=np.float32)
        b_val = np.array([[3, 2, 2], [2, 4, 2], [5, 6, 2]], dtype=np.float32)
        input_placeholders = {
            "cond": mb.placeholder(shape=cond_val.shape),
            "a": mb.placeholder(shape=a_val.shape),
            "b": mb.placeholder(shape=b_val.shape),
        }
        input_values = {"cond": cond_val, "a": a_val, "b": b_val}

        def build(cond, a, b):
            if not types.is_bool(cond.dtype):
                cond = mb.cast(x=cond, dtype="bool")
            return [mb.select(cond=cond, a=a, b=b)]

        expected_output_types = [(3, 3, types.fp32)]
        expected_outputs = [
            np.array([[3.0, 2.0, 2.0], [2.0, 4.0, 2.0], [5.0, 6.0, 2.0]], dtype=np.float32)
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_builder_to_backend_smoke_broadcast(self, compute_unit, backend):
        cond_val = np.array([[1], [0], [2]], dtype=np.float32)
        a_val = np.array([1, 7, 8], dtype=np.float32)
        b_val = np.array([[3, 2, 2], [2, 4, 2], [5, 6, 2]], dtype=np.float32)
        input_placeholders = {
            "cond": mb.placeholder(shape=cond_val.shape),
            "a": mb.placeholder(shape=a_val.shape),
            "b": mb.placeholder(shape=b_val.shape),
        }
        input_values = {"cond": cond_val, "a": a_val, "b": b_val}

        def build(cond, a, b):
            if not types.is_bool(cond.dtype):
                cond = mb.cast(x=cond, dtype="bool")
            return [mb.select(cond=cond, a=a, b=b)]

        expected_output_types = [(3, 3, types.fp32)]
        expected_outputs = [
            np.array([[1.0, 7.0, 8.0], [2.0, 4.0, 2.0], [1.0, 7.0, 8.0]], dtype=np.float32)
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_builder_to_backend_smoke_scalar_and_tensor(self, compute_unit, backend):
        cond_val = np.array([[1], [0], [2]], dtype=np.float32)
        a_val = np.float32(1.0)
        b_val = np.array([[3, 2, 2], [2, 4, 2], [5, 6, 2]], dtype=np.float32)
        input_placeholders = {
            "cond": mb.placeholder(shape=cond_val.shape),
            "b": mb.placeholder(shape=b_val.shape),
        }
        input_values = {"cond": cond_val, "b": b_val}

        def build(cond, b):
            if not types.is_bool(cond.dtype):
                cond = mb.cast(x=cond, dtype="bool")
            return [mb.select(cond=cond, a=a_val, b=b)]

        expected_output_types = [(3, 3, types.fp32)]
        expected_outputs = [
            np.array([[1.0, 1.0, 1.0], [2.0, 4.0, 2.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(compute_units, backends),
    )
    def test_builder_to_backend_smoke_symbolic(self, compute_unit, backend):
        SYMBOLIC_SHAPE = tuple([get_new_symbol() for _ in range(5)])
        VALUE = 100.0

        input_placeholders = {"a": mb.placeholder(shape=SYMBOLIC_SHAPE)}

        def build(a):
            return [mb.select(cond=False, a=a, b=np.float32(VALUE))]

        shape = tuple(np.random.randint(1, 5, size=len(SYMBOLIC_SHAPE)))
        a = np.random.rand(*shape)
        input_values = {"a": a}
        expected_outputs = [
            VALUE * np.ones(shape),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types=[SYMBOLIC_SHAPE + (types.fp32,)],
            expected_outputs=expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, upper_bound=10),
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval(self):
        cond = np.random.randint(low=0, high=2, size=(6, 1, 7)).astype(bool)
        a = random_gen(shape=(6, 1, 7), rand_min=-1962.0, rand_max=0.0)
        b = random_gen(shape=(6, 1, 7), rand_min=0.0, rand_max=1964.0)
        res = mb.select(cond=cond, a=a, b=b)
        np.testing.assert_allclose(np.where(cond, a, b), res.val, atol=1e-04, rtol=1e-05)

    @ssa_fn
    def test_builder_eval_broadcast(self):
        cond = np.array([[True], [False], [True]])
        a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        b = np.array([7, 8], dtype=np.float32)
        res = mb.select(cond=cond, a=a, b=b)
        np.testing.assert_allclose(
            np.array([[1, 2], [7, 8], [5, 6]], dtype=np.float32), res.val, atol=1e-04, rtol=1e-05
        )

    @ssa_fn
    def test_builder_eval_scalar(self):
        res = mb.select(cond=True, a=np.float32(1), b=np.float32(2))
        assert isinstance(res.val, np.float32)
        np.testing.assert_allclose(np.float32(1), res.val)


class TestCond:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        input_placeholders = {
            "a": mb.placeholder(shape=(1,), dtype=types.bool),
            "b": mb.placeholder(shape=(1,)),
        }

        def build(a, b):
            def true_fn():
                return mb.add(x=b, y=1.0), mb.mul(x=b, y=2.0)

            def false_fn():
                return mb.add(x=b, y=-1.0), mb.mul(x=b, y=-2.0)

            pred = mb.squeeze(x=a)
            return mb.cond(pred=pred, _true_fn=true_fn, _false_fn=false_fn)

        input_values = {
            "a": np.array([0], dtype=np.float32),
            "b": np.array([2], dtype=np.float32),
        }

        expected_output_types = [
            (1, types.fp32),
            (1, types.fp32),
        ]

        expected_outputs = [
            np.array([1], dtype=np.float32),
            np.array([-4], dtype=np.float32),
        ]
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestWhileLoop:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        def body(a, b):
            return mb.add(x=a, y=np.float32(1)), b

        def cond(a, b):
            return mb.less(x=a, y=b)

        input_placeholders = {
            "a": mb.placeholder(shape=(1,)),
            "b": mb.placeholder(shape=(1,)),
        }

        def build(a, b):
            return mb.while_loop(_cond=cond, _body=body, loop_vars=(a, b))

        input_values = {
            "a": np.array([1], dtype=np.float32),
            "b": np.array([2], dtype=np.float32),
        }

        expected_output_types = [
            (1, types.fp32),
            (1, types.fp32),
        ]

        expected_outputs = [
            np.array([2], dtype=np.float32),
            np.array([2], dtype=np.float32),
        ]
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_power(self, compute_unit, backend):

        input_placeholders = {
            "a": mb.placeholder(shape=(1,)),
            "b": mb.placeholder(shape=(1,)),
        }

        def build(a, b):
            # Compute a^b
            def body(res, bx):
                return mb.mul(x=res, y=a), mb.add(x=bx, y=np.float32(1))

            def cond(res, bx):
                return mb.less(x=bx, y=b)

            res, ignored = mb.while_loop(_cond=cond, _body=body, loop_vars=([1.0], [0.0]))
            return res

        input_values = {
            "a": np.array([2], dtype=np.float32),
            "b": np.array([4], dtype=np.float32),
        }

        expected_output_types = [
            (1, types.fp32),
        ]

        expected_outputs = [
            np.array([16], dtype=np.float32),
        ]
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_nested(self, compute_unit, backend):
        if backend.backend == "neuralnetwork":
            pytest.xfail(
                "rdar://96862073 (test_control_flow::TestWhileLoop::test_builder_to_backend_nested failing on nnv1)"
            )

        input_placeholders = {
            "x": mb.placeholder(shape=(1,)),
            "y": mb.placeholder(shape=(1,)),
        }

        def build(x, y):
            # i, j = x, y
            # while i < j:
            #   while 2*i < i+2:
            #     i += 1
            #   i += 2
            # return i, j

            # Create const outside of while loop for testing purpose
            two = mb.const(val=[2.0], name="const_two")
            one = mb.const(val=[1.0], name="const_one")

            def cond2(i):
                return mb.less(x=mb.mul(x=two, y=i), y=mb.add(x=i, y=two))

            def body2(i):
                return mb.add(x=i, y=one)

            def cond1(i, j):
                return mb.less(x=i, y=j)

            def body1(i, j):
                new_i = mb.while_loop(_cond=cond2, _body=body2, loop_vars=(i,))
                return mb.add(x=new_i, y=two), j

            return mb.while_loop(_cond=cond1, _body=body1, loop_vars=(x, y))

        input_values = {
            "x": np.array([0], dtype=np.float32),
            "y": np.array([10], dtype=np.float32),
        }

        expected_output_types = [
            (1, types.fp32),
            (1, types.fp32),
        ]

        expected_outputs = [
            np.array([10], dtype=np.float32),
            np.array([10], dtype=np.float32),
        ]
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestList:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        elem_shape = (2,)
        input_placeholders = {
            "a": mb.placeholder(shape=elem_shape),
            "b": mb.placeholder(shape=elem_shape),
        }

        def build(a, b):
            ls = mb.make_list(init_length=2, elem_shape=elem_shape)
            # list is initially all 0
            init_t = mb.list_read(ls=ls, index=0)
            ls = mb.list_write(ls=ls, index=0, value=a)
            # this write is out of bound
            ls = mb.list_write(ls=ls, index=4, value=b)
            ls = mb.list_scatter(
                ls=ls,
                indices=[2, 1],
                value=np.array([[-1, -2], [-4, -5]], dtype=np.float32),
            )
            return (
                init_t,
                mb.list_read(ls=ls, index=0),
                mb.list_gather(ls=ls, indices=[4, 2, 3]),
            )

        input_values = {
            "a": np.array([1, 3], dtype=np.float32),
            "b": np.array([2, 4], dtype=np.float32),
        }

        expected_output_types = [
            (2, types.fp32),
            (2, types.fp32),
            (3, 2, types.fp32),
        ]

        expected_outputs = [
            np.array([0, 0], dtype=np.float32),
            np.array([1, 3], dtype=np.float32),
            np.array([[2, 4], [-1, -2], [0, 0]], dtype=np.float32),
        ]
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_while(self, compute_unit, backend):
        # The while_loop appends [1, 2]*i to `ls` for each iteration
        # i = 0, ... num_iters-1.
        def body(i, num_iters, ls, update):
            y = mb.cast(x=i, dtype="fp32")
            new_elem = mb.mul(x=update, y=y)
            return (
                mb.add(x=i, y=1),
                num_iters,
                mb.list_write(ls=ls, index=i, value=new_elem),
                update,
            )

        def cond(i, num_iters, ls, update):
            i = mb.cast(x=i, dtype="fp32")
            return mb.less(x=i, y=num_iters)

        elem_shape = (2,)
        input_placeholders = {
            "num_iters": mb.placeholder(shape=(1,)),
            "update": mb.placeholder(shape=elem_shape),
        }

        def build(num_iters, update):
            i = 0
            ls = mb.make_list(init_length=1, elem_shape=elem_shape)
            _, _, final_tensor_list, _ = mb.while_loop(
                _cond=cond, _body=body, loop_vars=(i, num_iters, ls, update)
            )
            list_len = mb.list_length(ls=final_tensor_list)
            indices = mb.range_1d(start=0, end=list_len, step=1)
            return mb.list_gather(ls=final_tensor_list, indices=indices)

        input_values = {
            "num_iters": np.array([3], dtype=np.float32),
            "update": np.array([1, 2], dtype=np.float32),
        }

        expected_output_types = [
            # Type inference does not unroll loop
            (UNK_SYM, 2, types.fp32),
        ]

        expected_outputs = [
            np.array([[0, 0], [1, 2], [2, 4]], dtype=np.float32),
        ]
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestConst:
    @pytest.mark.parametrize(
        "compute_unit, backend, dtype",
        itertools.product(
            compute_units,
            backends,
            [
                np.int32,
                np.int64,
                np.float16,
                np.float32,
                np.float64,
            ],
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, dtype):
        t = np.random.randint(0, 5, (4, 2)).astype(np.float32)
        constant = np.random.randint(0, 5, (4, 2)).astype(dtype)
        input_placeholders = {
            "x": mb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            y = mb.const(val=constant)
            y = mb.cast(x=y, dtype="fp32")
            return mb.add(x=x, y=y)

        expected_output_types = (4, 2, types.fp32)
        expected_outputs = t + constant.astype(np.float32)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, dtype",
        itertools.product(
            compute_units,
            backends,
            (
                np.int8,
                np.uint8,
                np.int16,
                np.uint16,
                np.int32,
                np.int64,
                np.float16,
                np.float32,
                np.float64,
            ),
        ),
    )
    def test_const_type(self, compute_unit, backend, dtype):
        """Makes sure the ndarray in const has the correct type."""
        @mb.program(input_specs=[], opset_version=backend.opset_version)
        def prog():
            return mb.const(val=np.random.randint(0, 5, (4, 2)).astype(dtype))

        const_op = prog.functions["main"].find_ops(op_type="const")[0]

        if dtype == np.int64:
            target_dtype = np.int32
        elif dtype == np.float64:
            target_dtype = np.float32
        else:
            target_dtype = dtype
        assert const_op.outputs[0].dtype == types.numpy_type_to_builtin_type(target_dtype)
