from coremltools.converters.nnv2 import testing_reqs
from coremltools.converters.nnv2.nnv2_program.program import get_new_symbol
from coremltools.converters.nnv2.testing_reqs import *

from .testing_utils import run_compare_builder

backends = testing_reqs.backends


class TestConvTranspose:
    @pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed.")
    @pytest.mark.parametrize(
            ','.join([
                'use_cpu_only',
                'backend',
                'conv_dim', # 1d or 2d conv
                'padding',
                'HWK',
                'stride',
                'dilation',
                'has_bias',
                'groups',
                'symbolic']),
             itertools.product(
                 [True, False],
                 backends,
                 ['conv1d', 'conv2d'],
                 [(1, 1), (2, 3)],
                 [(12, 12, 2), (5, 5, 4)],
                 [1, 2],
                 [1, 2],
                 [True, False],
                 [1, 2],
                 [True, False]
             ))
    def test_builder_to_backend_stress(self, use_cpu_only, backend, conv_dim, padding,
                                       HWK, stride, dilation, has_bias, groups, symbolic):
        H, W, K = HWK
        N, C_in, C_out = 1, 1, 2

        import torch
        import torch.nn as nn

        isConv1d = True if conv_dim == 'conv1d' else False
        if isConv1d:
            m = nn.ConvTranspose1d(C_in, C_out, K, stride=stride, dilation=dilation, bias=has_bias, padding=padding[0])
            input_shape = [N, C_in, H]
            padding = [padding[0], padding[0]]
            strides = [stride]
            dilations = [dilation]
        else:
            m = nn.ConvTranspose2d(C_in, C_out, K, stride=stride, dilation=dilation, bias=has_bias, padding=padding)
            input_shape = [N, C_in, H, W]
            padding = [padding[0], padding[0], padding[1], padding[1]]
            strides = [stride] * 2
            dilations = [dilation] * 2

        wts = m.state_dict()
        weight = wts['weight'].detach().numpy()
        bias = wts['bias'].detach().numpy() if has_bias else None

        # Reshape to CoreML format
        # PyTorch weight format: Cin, Cout, H, W
        # CoreML weight format: H, W, Cout, Cin
        if isConv1d:
            weight = np.transpose(weight, [2, 1, 0])
        else:
            weight = np.transpose(weight, [2, 3, 1, 0])

        input = torch.randn(*input_shape)
        output = m(input)
        output = output.detach().numpy()
        input = input.detach().numpy()

        output_shape = list(output.shape)
        if symbolic:
            # For symbolic input test
            # Make Batch Size and input channel as symbolic
            symbolic_batch_size = get_new_symbol()
            input_shape[0] = symbolic_batch_size
            output_shape[0] = symbolic_batch_size

        expected_output_types = tuple(output_shape[:]) + (builtins.fp32,)
        expected_outputs = [output]

        input_placeholders = {"x": cb.placeholder(shape=input_shape)}
        input_values = {"x": input}

        def build(x):
            arguments = {
                        "x":x,
                        "weight":weight,
                        "pad":padding,
                        "pad_type":"custom",
                        "strides":strides,
                        "dilations":dilations,
                        }
            if has_bias:
                arguments["bias"] = bias
            return cb.conv_transpose(**arguments)
        run_compare_builder(build, input_placeholders, input_values,
                    expected_output_types, expected_outputs,
                    use_cpu_only=use_cpu_only, frontend_only=False,
                    backend=backend)
