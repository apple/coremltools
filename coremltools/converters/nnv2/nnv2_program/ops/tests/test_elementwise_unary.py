import scipy
from coremltools.converters.nnv2 import testing_reqs
from coremltools.converters.nnv2.testing_reqs import *

from .testing_utils import run_compare_builder

backends = testing_reqs.backends


class TestElementwiseUnary:
    # All ops in this test share the same backends
    @pytest.mark.parametrize('use_cpu_only, backend, mode',
                             itertools.product(
                                 [True, False],
                                 backends,
                                 ['abs', 'acos', 'asin', 'atan', 'atanh',
                                  'exp2', 'clip', 'cos', 'cosh', 'erf', 'exp',
                                  'erf', 'floor', 'inverse', 'log', 'round', 'rsqrt',
                                  'sign', 'sin', 'sinh', 'sqrt', 'tan', 'tanh',
                                  'threshold', 'cast']
                             ))
    def test_builder_to_backend_smoke(self, use_cpu_only, backend, mode):
        if mode == 'abs':
            val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)

            build = lambda x: cb.abs(x=x)
        elif mode == 'acos':
            val = np.array([[-1, -0.5, 0],[0.4, 0.5, 0.8]], dtype=np.float32)
            expected_outputs = np.array([[3.14159265, 2.0943951 , 1.57079633],
                                         [1.15927948, 1.04719755, 0.64350111]],
                                         dtype=np.float32)

            build = lambda x: cb.acos(x=x)
        elif mode == 'asin':
            val = np.array([[-1, -0.5, 0],[0.4, 0.5, 0.8]], dtype=np.float32)
            expected_outputs = np.array([[-1.57079633, -0.52359878, 0.],
                                         [0.41151685, 0.52359878, 0.92729522]],
                                         dtype=np.float32)

            build = lambda x: cb.asin(x=x)
        elif mode == 'atan':
            val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[-0.78539816, 1.10714872, -1.24904577],
                                         [1.32581766, -1.37340077, 1.40564765]],
                                         dtype=np.float32)
            build = lambda x: cb.atan(x=x)
        elif mode == 'atanh':
            if backend == 'nnv2_proto':
                #TODO
                return
            val = np.array([[-0.8, -0.5, 0],[0.4, 0.5, 0.8]], dtype=np.float32)
            expected_outputs = np.array([[-1.09861229, -0.54930614, 0.],
                                         [0.42364893, 0.54930614, 1.09861229]],
                                         dtype=np.float32)

            build = lambda x: cb.atanh(x=x)
        elif mode == 'cast':
            if backend == 'nnv2_proto':
                #TODO <rdar://problem/61400566> [NNV2] Add cast operation in NNV2 backend and enable tests
                return
            val = np.array([[-1.2, 2, -3.6], [4.5, -5, 6.7]], dtype=np.float32)
            expected_outputs = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int32)

            build = lambda x: cb.cast(x=x, dtype="int32")
        elif mode == 'ceil':
            val = np.array([[-1.2, 2, -3.4],[4.5, -5, 6.7]], dtype=np.float32)
            expected_outputs = np.array([[-1, 2, -3], [5, -5, 7]], dtype=np.float32)

            build = lambda x: cb.ceil(x=x)
        elif mode == 'clip':
            val = np.array([[-1.2, 2, -3.4],[4.5, -5, 6.7]], dtype=np.float32)
            expected_outputs = np.array([[0, 2, 0], [4.5, 0, 5]], dtype=np.float32)

            build = lambda x: cb.clip(x=x, alpha=0.0, beta=5.0)
        elif mode == 'cos':
            val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[0.54030231, -0.41614684, -0.9899925 ],
                                         [-0.65364362, 0.28366219, 0.96017029]],
                                         dtype=np.float32)

            build = lambda x: cb.cos(x=x)
        elif mode == 'cosh':
            val = np.array([[-1, -2, -3],[1, 2, 3]], dtype=np.float32)
            expected_outputs = np.array([[1.54308063, 3.76219569, 10.067662],
                                         [1.54308063, 3.76219569, 10.067662]],
                                         dtype=np.float32)

            build = lambda x: cb.cosh(x=x)
        elif mode == 'erf':
            val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[-0.8427007929497148, 0.9953222650189527, -0.9999779095030014],
                                         [0.9999999845827421, -0.9999999999984626, 1.0]],
                                         dtype=np.float32)

            build = lambda x: cb.erf(x=x)
        elif mode == 'exp':
            val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[0.36787944, 7.3890561 , 0.04978707],
                                         [54.5981500, 0.0067379, 403.428793]],
                                         dtype=np.float32)

            build = lambda x: cb.exp(x=x)
        elif mode == 'exp2':
            val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[0.5, 4., 0.125], [16, 0.03125, 64]], dtype=np.float32)

            build = lambda x: cb.exp2(x=x)
        elif mode == 'floor':
            val = np.array([[-1.2, 2, -3.4],[4.5, -5, 6.7]], dtype=np.float32)
            expected_outputs = np.array([[-2, 2, -4], [4, -5, 6]], dtype=np.float32)

            build = lambda x: cb.floor(x=x)
        elif mode == 'inverse':
            if backend == 'nnv2_proto':  # TODO
                return
            val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[-1.   ,  0.5 , -0.33333334],
                                      [ 0.25 , -0.2 ,  0.16666667]],
                                      dtype=np.float32)
            build = lambda x: cb.inverse(x=x)
        elif mode == 'log':
            val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            expected_outputs = np.array([[0., 0.69314718, 1.09861229],
                                         [1.38629436, 1.60943791, 1.79175947]],
                                         dtype=np.float32)

            build = lambda x: cb.log(x=x)
        elif mode == 'round':
            val = np.array([[-1.2, 2, -3.4],[4.6, -5, 6.7]], dtype=np.float32)
            expected_outputs = np.array([[-1, 2, -3], [5, -5, 7]], dtype=np.float32)

            build = lambda x: cb.round(x=x)
        elif mode == 'rsqrt':
            val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            expected_outputs = np.array([[1., 0.70710678, 0.57735027],
                                         [0.5, 0.4472136, 0.40824829]],
                                         dtype=np.float32)

            build = lambda x: cb.rsqrt(x=x)
        elif mode == 'sign':
            val = np.array([[-1, 2, 0],[0, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[-1, 1, 0], [0, -1, 1]], dtype=np.float32)

            build = lambda x: cb.sign(x=x)
        elif mode == 'sin':
            val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[-0.84147098, 0.90929743, -0.14112001],
                                         [-0.7568025 , 0.95892427, -0.2794155]],
                                         dtype=np.float32)

            build = lambda x: cb.sin(x=x)
        elif mode == 'sinh':
            val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[-1.1752, 3.62686, -10.017874],
                                         [27.289917 , -74.20321, 201.71315]],
                                         dtype=np.float32)

            build = lambda x: cb.sinh(x=x)
        elif mode == 'sqrt':
            val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
            expected_outputs = np.array([[1., 1.41421356, 1.73205081],
                                         [2., 2.23606798, 2.44948974]],
                                         dtype=np.float32)

            build = lambda x: cb.sqrt(x=x)
        elif mode == 'tan':
            val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[-1.5574, -2.185, 0.1425],
                                         [1.15782, 3.3805, -0.291]],
                                         dtype=np.float32)

            build = lambda x: cb.tan(x=x)
        elif mode == 'tanh':
            val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
            expected_outputs = np.array([[-0.7615942,  0.9640276, -0.9950548],
                                         [ 0.9993293, -0.9999092,  0.9999877]], dtype=np.float32)

            build = lambda x: cb.tanh(x=x)
        elif mode == 'threshold':
            val = np.array([[-1.2, 2, -3.4],[4.5, -5, 6.7]], dtype=np.float32)
            expected_outputs = np.array([[1.0, 2, 1.0],[4.5, 1.0, 6.7]], dtype=np.float32)

            build = lambda x: cb.threshold(x=x, alpha=1.0)

        input_placeholders = {'x': cb.placeholder(shape=val.shape)}
        input_values = {'x': val}
        expected_output_types = (2,3,builtins.int32) if mode == "cast" else (2, 3, builtins.fp32)

        run_compare_builder(build, input_placeholders, input_values,
                            expected_output_types, expected_outputs,
                            use_cpu_only=use_cpu_only, frontend_only=False,
                            backend=backend)

    @ssa_fn
    def test_builder_abs_eval(self):
        val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        v = cb.abs(x=val)
        expected_outputs = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_acos_eval(self):
        val = np.array([[-1, -0.5, 0],[0.4, 0.5, 0.8]], dtype=np.float32)
        v = cb.acos(x=val)
        expected_outputs = np.array([[3.14159265, 2.0943951 , 1.57079633],
                                     [1.15927948, 1.04719755, 0.64350111]],
                                     dtype=np.float32)
        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_asin_eval(self):
        val = np.array([[-1, -0.5, 0],[0.4, 0.5, 0.8]], dtype=np.float32)
        v = cb.asin(x=val)
        expected_outputs = np.array([[-1.57079633, -0.52359878, 0.],
                                     [0.41151685, 0.52359878, 0.92729522]],
                                     dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_atan_eval(self):
        val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        v = cb.atan(x=val)
        expected_outputs = np.array([[-0.78539816, 1.10714872, -1.24904577],
                                     [1.32581766, -1.37340077, 1.40564765]],
                                     dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_atanh_eval(self):
        val = np.array([[-0.8, -0.5, 0],[0.4, 0.5, 0.8]], dtype=np.float32)
        v = cb.atanh(x=val)
        expected_outputs = np.array([[-1.09861229, -0.54930614, 0.],
                                     [0.42364893, 0.54930614, 1.09861229]],
                                     dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_cast_eval(self):
        val = np.array([[-1.2, 2, -3.4],[4.5, -5, 6.7]], dtype=np.float32)
        expected_outputs = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int32)

        v = cb.cast(x=val, dtype= "int32")

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_ceil_eval(self):
        val = np.array([[-1.2, 2, -3.4],[4.5, -5, 6.7]], dtype=np.float32)
        v = cb.ceil(x=val)
        expected_outputs = np.array([[-1, 2, -3], [5, -5, 7]], dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_clip_eval(self):
        val = np.array([[-1.2, 2, -3.4],[4.5, -5, 6.7]], dtype=np.float32)
        v = cb.clip(x=val, alpha=0.0, beta=5.0)
        expected_outputs = np.array([[0, 2, 0], [4.5, 0, 5]], dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_cos_eval(self):
        val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        v = cb.cos(x=val)
        expected_outputs = np.array([[0.54030231, -0.41614684, -0.9899925 ],
                                     [-0.65364362, 0.28366219, 0.96017029]],
                                     dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_cosh_eval(self):
        val = np.array([[-1, -2, -3],[1, 2, 3]], dtype=np.float32)
        v = cb.cosh(x=val)
        expected_outputs = np.array([[1.54308063, 3.76219569, 10.067662],
                                     [1.54308063, 3.76219569, 10.067662]],
                                     dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_erf_eval(self):
        x_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.erf(x=x_val)
        assert is_close(scipy.special.erf(x_val), v.val)


    @ssa_fn
    def test_builder_exp_eval(self):
        val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        v = cb.exp(x=val)
        expected_outputs = np.array([[0.36787944, 7.3890561 , 0.04978707],
                                     [54.5981500, 0.0067379, 403.428793]],
                                     dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_exp2_eval(self):
        val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        v = cb.exp2(x=val)
        expected_outputs = np.array([[0.5, 4., 0.125], [16, 0.03125, 64]], dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_floor_eval(self):
        val = np.array([[-1.2, 2, -3.4],[4.5, -5, 6.7]], dtype=np.float32)
        v = cb.floor(x=val)
        expected_outputs = np.array([[-2, 2, -4], [4, -5, 6]], dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_inverse_eval(self):
        val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        v = cb.inverse(x=val)
        expected_outputs = np.array([[-1.   ,  0.5 , -0.33333334],
                                     [ 0.25 , -0.2 ,  0.16666667]], dtype=np.float32)
        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_log_eval(self):
        val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        v = cb.log(x=val)
        expected_outputs = np.array([[0., 0.69314718, 1.09861229],
                                     [1.38629436, 1.60943791, 1.79175947]],
                                     dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_round_eval(self):
        val = np.array([[-1.2, 2, -3.4],[4.6, -5, 6.7]], dtype=np.float32)
        v = cb.round(x=val)
        expected_outputs = np.array([[-1, 2, -3], [5, -5, 7]], dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_rsqrt_eval(self):
        val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        v = cb.rsqrt(x=val)
        expected_outputs = np.array([[1., 0.70710678, 0.57735027],
                                     [0.5, 0.4472136, 0.40824829]],
                                     dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_sign_eval(self):
        val = np.array([[-1, 2, 0],[0, -5, 6]], dtype=np.float32)
        v = cb.sign(x=val)
        expected_outputs = np.array([[-1, 1, 0], [0, -1, 1]], dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_sin_eval(self):
        val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        v = cb.sin(x=val)
        expected_outputs = np.array([[-0.84147098, 0.90929743, -0.14112001],
                                     [-0.7568025 , 0.95892427, -0.2794155]],
                                     dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_sinh_eval(self):
        val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        v = cb.sinh(x=val)
        expected_outputs = np.array([[-1.1752, 3.62686, -10.017874],
                                     [27.289917 , -74.20321, 201.71315]],
                                     dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_sqrt_eval(self):
        val = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
        v = cb.sqrt(x=val)
        expected_outputs = np.array([[1., 1.41421356, 1.73205081],
                                     [2., 2.23606798, 2.44948974]],
                                     dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_tan_eval(self):
        val = np.array([[-1, 2, -3],[4, -5, 6]], dtype=np.float32)
        v = cb.tan(x=val)
        expected_outputs = np.array([[-1.5574, -2.185, 0.1425],
                                     [1.15782, 3.3805, -0.291]],
                                     dtype=np.float32)

        assert is_close(expected_outputs, v.val)

    @ssa_fn
    def test_builder_tanh_eval(self):
        x_val = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.float32)
        v = cb.tanh(x=x_val)
        assert is_close(np.tanh(x_val), v.val)

    @ssa_fn
    def test_builder_threshold_eval(self):
        val = np.array([[-1.2, 2, -3.4],[4.5, -5, 6.7]], dtype=np.float32)
        v = cb.threshold(x=val, alpha=1.0)
        expected_outputs = np.array([[1.0, 2, 1.0], [4.5, 1.0, 6.7]], dtype=np.float32)

        assert is_close(expected_outputs, v.val)