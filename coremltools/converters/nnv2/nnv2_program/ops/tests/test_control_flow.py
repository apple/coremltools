from coremltools.converters.nnv2 import testing_reqs
from coremltools.converters.nnv2.testing_reqs import *
from .testing_utils import (run_compare_builder, UNK_SYM)

backends = testing_reqs.backends


class TestSelect:
    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        cond_val = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]], dtype=np.float32)
        a_val = np.array([[3, 1, 1], [1, 4, 1], [5, 6, 1]], dtype=np.float32)
        b_val = np.array([[3, 2, 2], [2, 4, 2], [5, 6, 2]], dtype=np.float32)
        input_placeholders = {
            'cond': cb.placeholder(shape=cond_val.shape),
            'a': cb.placeholder(shape=a_val.shape),
            'b': cb.placeholder(shape=b_val.shape),
        }
        input_values = {'cond': cond_val, 'a': a_val, 'b': b_val}

        def build(cond, a, b):
            return [cb.select(cond=cond, a=a, b=b)]

        expected_output_types = [(3, 3, builtins.fp32)]
        expected_outputs = [
            np.array([[3., 2., 2.], [2., 4., 2.], [5., 6., 2.]], dtype=np.float32)
        ]

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        cond = np.random.randint(low=0, high=2, size=(6, 1, 7))
        a = random_gen(shape=(6, 1, 7), rand_min=-1962., rand_max=0.)
        b = random_gen(shape=(6, 1, 7), rand_min=0., rand_max=1964.)
        res = cb.select(cond=cond, a=a, b=b)
        assert is_close(np.where(cond, a, b), res.val)


class TestCond:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):

        input_placeholders = {
                "a": cb.placeholder(shape=(1,), dtype=builtins.bool),
                "b": cb.placeholder(shape=(1,)),
                }
        def build(a, b):
            def true_fn():
                return cb.add(x=b, y=1), cb.mul(x=b, y=2)

            def false_fn():
                return cb.add(x=b, y=-1), cb.mul(x=b, y=-2)

            pred = cb.squeeze(x=a)
            return cb.cond(pred=pred, _true_fn=true_fn, _false_fn=false_fn)

        input_values = {
                "a": np.array([0], dtype=np.float32),
                "b": np.array([2], dtype=np.float32),
                }

        expected_output_types = [
                (1, builtins.fp32),
                (1, builtins.fp32),
                ]

        expected_outputs = [
                np.array([1], dtype=np.float32),
                np.array([-4], dtype=np.float32),
                ]
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)


class TestWhileLoop:

    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        def body(a, b):
            return cb.add(x=a, y=np.float32(1)), b

        def cond(a, b):
            return cb.less(x=a, y=b)

        input_placeholders = {
                "a": cb.placeholder(shape=(1,)),
                "b": cb.placeholder(shape=(1,)),
                }
        def build(a, b):
            return cb.while_loop(_cond=cond, _body=body, loop_vars=(a, b))

        input_values = {
                "a": np.array([1], dtype=np.float32),
                "b": np.array([2], dtype=np.float32),
                }

        expected_output_types = [
                (1, builtins.fp32),
                (1, builtins.fp32),
                ]

        expected_outputs = [
                np.array([2], dtype=np.float32),
                np.array([2], dtype=np.float32),
                ]
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)


class TestList:
    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        elem_shape = (2,)
        input_placeholders = {
                "a": cb.placeholder(shape=elem_shape),
                "b": cb.placeholder(shape=elem_shape),
                }
        def build(a, b):
            ls = cb.make_list(init_length=2, elem_shape=elem_shape)
            # list is initially all 0
            init_t = cb.list_read(ls=ls, index=0)
            ls = cb.list_write(ls=ls, index=0,
                    value=a)
            # this write is out of bound
            ls = cb.list_write(ls=ls, index=4,
                    value=b)
            ls = cb.list_scatter(ls=ls,
                    indices=[2, 1],
                    value=np.array([[-1, -2],
                                    [-4, -5]], dtype=np.float32))
            return init_t, cb.list_read(ls=ls, index=0), \
                    cb.list_gather(ls=ls, indices=[4, 2, 3])


        input_values = {
                "a": np.array([1, 3], dtype=np.float32),
                "b": np.array([2, 4], dtype=np.float32),
                }

        expected_output_types = [
                (2, builtins.fp32),
                (2, builtins.fp32),
                (3, 2, builtins.fp32),
                ]

        expected_outputs = [
                np.array([0, 0], dtype=np.float32),
                np.array([1, 3], dtype=np.float32),
                np.array([[2, 4],
                          [-1, -2],
                          [0, 0]], dtype=np.float32),
                ]
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)


    @pytest.mark.parametrize("use_cpu_only, backend",
            itertools.product(
                [True, False],
                backends,
                ))
    def test_builder_to_backend_while(self, use_cpu_only, backend):
        # The while_loop appends [1, 2]*i to `ls` for each iteration
        # i = 0, ... num_iters-1.
        def body(i, num_iters, ls, update):
            new_elem = cb.mul(x=update, y=i)
            return cb.add(x=i, y=1), num_iters, \
                    cb.list_write(ls=ls, index=i,
                            value=new_elem), update

        def cond(i, num_iters, ls, update):
            return cb.less(x=i, y=num_iters)

        elem_shape = (2,)
        input_placeholders = {
                "num_iters": cb.placeholder(shape=(1,)),
                "update": cb.placeholder(shape=elem_shape),
                }

        def build(num_iters, update):
            i = 0
            ls = cb.make_list(init_length=1, elem_shape=elem_shape)
            _, _, final_tensor_list, _ = cb.while_loop(_cond=cond, _body=body,
                    loop_vars=(i, num_iters, ls, update))
            list_len = cb.list_length(ls=final_tensor_list)
            indices = cb.range_1d(start=0, end=list_len, step=1)
            return cb.list_gather(ls=final_tensor_list,
                    indices=indices)

        input_values = {
                "num_iters": np.array([3], dtype=np.float32),
                "update": np.array([1, 2], dtype=np.float32),
                }

        expected_output_types = [
                # Type inference does not unroll loop
                (UNK_SYM, 2, builtins.fp32),
                ]

        expected_outputs = [
                np.array([[0, 0],
                          [1, 2],
                          [2, 4]], dtype=np.float32),
                ]
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

