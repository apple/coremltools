from ._test_utils import UNK_SYM
from . import _test_reqs
from ._test_reqs import *
backends = _test_reqs.backends

class TestSliceByIndex:

    @pytest.mark.parametrize('use_cpu_only, backend',
                             itertools.product(
                                 [True, False],
                                 backends,
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend):
        x_val = np.array(list(range(24))).reshape((2, 3, 4)).astype(np.float32)
        begin_val = np.array([1, 1, 1], dtype=np.int32)
        end_val = np.array([2, 3, 3], dtype=np.int32)
        input_placeholders = {'x': cb.placeholder(shape=x_val.shape),
                              'begin': cb.placeholder(shape=begin_val.shape, dtype=builtins.int32),
                              'end': cb.placeholder(shape=end_val.shape, dtype=builtins.int32)}
        input_values = {'x': x_val, 'begin': begin_val, 'end': end_val}

        def build(x, begin, end):
            return [
                cb.slice_by_index(x=x, begin=begin, end=end),
            ]

        expected_output_types = [(UNK_SYM, UNK_SYM, UNK_SYM, builtins.fp32)]
        expected_outputs = [np.array([[[17, 18], [21, 22]]], dtype=np.float32)]
        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False, backend=backend)

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array(list(range(24))).reshape((2,3,4))
        v = [cb.slice_by_index(x=x_val, begin=[1, 1, 1], end=[2, 2, 2]),
             cb.slice_by_index(x=x_val, begin=[1, 1, 1], end=[2, 3, 4], stride=[1, 1, 2]),
             cb.slice_by_index(x=x_val, begin=[1, 1, 1], end=[2, 3, 4], stride=[1, 1, 2], begin_mask=[True, False, True]),
             cb.slice_by_index(x=x_val, begin=[1, 1, 1], end=[2, 3, 4], stride=[1, 1, 2], begin_mask=[True, False, True], end_mask=[True, True, False]),
             cb.slice_by_index(x=x_val, begin=[1, 1, 1], end=[2, 3, 4], stride=[1, 1, 2], begin_mask=[False, False, True], end_mask=[True, False, False], squeeze_mask=[False, True, False]),
            ]
        ans = [x_val[1:2, 1:2, 1:2],
               x_val[1:2, 1:3, 1:4:2],
               x_val[:2, 1:3, :4:2],
               x_val[:, 1:, :4:2],
               x_val[1::1, 1, :3:2],
              ]
        for idx in range(len(v)):
            assert is_close(ans[idx], v[idx].val)

    @pytest.mark.skipif(not HAS_TF, reason='TensorFlow not found.')
    @pytest.mark.parametrize('use_cpu_only, backend, rank, masking',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 [rank for rank in range(1, 5)],
                                 [True, False]
                                 ))
    def test_tf(self, use_cpu_only, backend, rank, masking):
        shape = np.random.randint(low=2, high=6, size=rank)
        begin_val = np.array([np.random.randint(low=-shape[i], high=shape[i]) for i in range(rank)]).astype(np.int32)
        end_val = np.array([np.random.randint(low=-shape[i], high=shape[i]) for i in range(rank)]).astype(np.int32)
        stride_val = np.array([np.random.randint(low=-shape[i], high=shape[i]) for i in range(rank)]).astype(np.int32)
        if not masking:
            begin_mask = [False] * rank
            end_mask = [False] * rank
            squeeze_mask = [False] * rank
        else:
            begin_mask = np.array([np.random.choice([True, False, False]) for i in range(rank)]).astype(np.bool)
            end_mask = np.array([np.random.choice([True, False, False]) for i in range(rank)]).astype(np.bool)
            squeeze_flag = True
            # We do not squeeze to scalar in nnv1
            while squeeze_flag and backend == 'nnv1_proto':
                squeeze_mask = np.array([np.random.choice([True, False]) for i in range(rank)]).astype(np.bool)
                for i in range(rank):
                    if begin_mask[i] or end_mask[i]:
                        squeeze_mask[i] = False
                for s in squeeze_mask:
                    if not s:
                        squeeze_flag = False

        for i in range(rank):
            if begin_mask[i] or end_mask[i]:
                stride = 0
                while stride == 0:
                    stride = np.random.randint(low=-shape[i], high=shape[i])
                stride_val[i] = stride

                if not end_mask[i]:
                    while True:
                        end = np.random.randint(low=-shape[i], high=shape[i])
                        normalized_end = shape[i] + end if end < 0 else end
                        if normalized_end == 0 and stride_val[i] > 0:
                            continue
                        elif normalized_end == shape[i]-1 and stride_val[i] < 0:
                            continue
                        else:
                            end_val[i] = end
                            break
                continue
            if squeeze_mask[i]:
                stride_val[i] = 1
            while True:
                end = np.random.randint(low=-shape[i], high=shape[i])
                normalized_end = shape[i] + end if end < 0 else end
                normalized_begin = shape[i] + begin_val[i] if begin_val[i] < 0 else begin_val[i]
                if normalized_end == normalized_begin:
                    continue
                if begin_mask[i] or end_mask[i] or squeeze_mask[i]:
                    stride = 1
                elif normalized_end < normalized_begin:
                    stride = -np.random.randint(low=1, high=shape[i])
                else:
                    stride = np.random.randint(low=1, high=shape[i])
                end_val[i] = end
                stride_val[i] = stride
                break

        def _mask_to_bit(mask):
            ret = 0
            for x in mask[::-1]:
                ret <<= 1
                if x:
                    ret += 1
            return ret

        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=shape)
            begin = tf.placeholder(tf.int32, shape=begin_val.shape)
            end = tf.placeholder(tf.int32, shape=end_val.shape)

            x_val = np.array(list(range(np.prod(shape)))).reshape(shape).astype(np.float32)
            res = tf.strided_slice(x, begin, end, stride_val,
                                   begin_mask=_mask_to_bit(begin_mask),
                                   end_mask=_mask_to_bit(end_mask),
                                   shrink_axis_mask=_mask_to_bit(squeeze_mask))
            run_compare_tf(graph, {x: x_val, begin: begin_val, end: end_val},
                           res, use_cpu_only=use_cpu_only, backend=backend)


