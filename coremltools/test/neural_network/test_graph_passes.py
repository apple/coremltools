import numpy as np
import unittest
import coremltools.models.datatypes as datatypes
from coremltools.models import neural_network as neural_network
from coremltools.converters.nnssa.coreml.graph_pass.mlmodel_passes import \
        remove_disconnected_constants, transform_conv_crop


class MLModelPassesTest(unittest.TestCase):

    def test_load_constant_remove(self):
        input_features = [('data', datatypes.Array(*(3, 4)))]
        output_features = [('out', None)]
        builder = neural_network.NeuralNetworkBuilder(input_features, output_features, disable_rank5_shape_mapping=True)
        builder.add_activation('relu1', 'RELU', 'data', 'relu1')
        builder.add_load_constant_nd('const1', 'c1', constant_value=np.ones((5,)), shape=(5,))
        builder.add_activation('relu2', 'RELU', 'relu1', 'out')
        builder.add_load_constant_nd('const2', 'c2', constant_value=np.ones((5,)), shape=(5,))
        builder.add_load_constant_nd('const3', 'c3', constant_value=np.ones((5,)), shape=(5,))
        spec = builder.spec
        np.testing.assert_equal(5, len(spec.neuralNetwork.layers))
        remove_disconnected_constants(spec)
        np.testing.assert_equal(2, len(spec.neuralNetwork.layers))

    def test_conv_crop_bn_to_conv_bn_crop(self):
        input_features = [('data', datatypes.Array(1, 10, 10))]
        output_features = [('out', None)]
        builder = neural_network.NeuralNetworkBuilder(input_features, output_features)
        W = np.ones((2,10,1,10), dtype=np.float32)
        builder.add_convolution(name='conv',
                                kernel_channels=1,
                                output_channels=2,
                                height=2, width=2,
                                stride_height=1, stride_width=1,
                                border_mode='valid', groups=1,
                                W=W,
                                b=None, has_bias=False,
                                input_name='data', output_name='conv_out')
        builder.add_crop(name='crop',
                        left=1, right=1, top=1, bottom=1, offset=0,
                        input_names=['conv_out'],
                        output_name='crop_out')
        builder.add_batchnorm(name='bn',
                              channels=2,
                              gamma=np.ones(2,).astype(np.float32),
                              beta=np.ones(2,).astype(np.float32),
                              mean=np.ones(2,).astype(np.float32),
                              variance=np.ones(2,).astype(np.float32),
                              input_name='crop_out',
                              output_name='out')
        # Conv -> Crop -> BN
        spec = builder.spec.neuralNetwork
        np.testing.assert_equal('crop', spec.layers[1].WhichOneof('layer'))
        np.testing.assert_equal('batchnorm', spec.layers[2].WhichOneof('layer'))

        # transform the pattern
        transform_conv_crop(builder.spec)
        # Conv -> BN -> Crop
        np.testing.assert_equal('batchnorm', spec.layers[1].WhichOneof('layer'))
        np.testing.assert_equal('crop', spec.layers[2].WhichOneof('layer'))

    def test_conv_crop_bn_relu_to_conv_bn_relu_crop(self):
        input_features = [('data', datatypes.Array(1, 10, 10))]
        output_features = [('out', None)]
        builder = neural_network.NeuralNetworkBuilder(input_features, output_features)
        W = np.ones((2,10,1,10), dtype=np.float32)
        builder.add_convolution(name='conv',
                                kernel_channels=1,
                                output_channels=2,
                                height=2, width=2,
                                stride_height=1, stride_width=1,
                                border_mode='valid', groups=1,
                                W=W,
                                b=None, has_bias=False,
                                input_name='data', output_name='conv_out')
        builder.add_crop(name='crop',
                        left=1, right=1, top=1, bottom=1, offset=0,
                        input_names=['conv_out'],
                        output_name='crop_out')
        builder.add_batchnorm(name='bn',
                              channels=2,
                              gamma=np.ones(2,).astype(np.float32),
                              beta=np.ones(2,).astype(np.float32),
                              mean=np.ones(2,).astype(np.float32),
                              variance=np.ones(2,).astype(np.float32),
                              input_name='crop_out',
                              output_name='bn_out')
        builder.add_activation(name='relu',
                               non_linearity='RELU',
                               input_name='bn_out',
                               output_name='out')
        # Conv -> Crop -> BN -> ReLU
        spec = builder.spec.neuralNetwork
        np.testing.assert_equal('crop', spec.layers[1].WhichOneof('layer'))
        np.testing.assert_equal('batchnorm', spec.layers[2].WhichOneof('layer'))
        np.testing.assert_equal('activation', spec.layers[3].WhichOneof('layer'))

        # transform the pattern
        transform_conv_crop(builder.spec)
        # Conv -> BN -> ReLU -> Crop
        np.testing.assert_equal('batchnorm', spec.layers[1].WhichOneof('layer'))
        np.testing.assert_equal('activation', spec.layers[2].WhichOneof('layer'))
        np.testing.assert_equal('crop', spec.layers[3].WhichOneof('layer'))


if __name__ == '__main__':
    RUN_ALL_TESTS = True
    if RUN_ALL_TESTS:
        unittest.main()
    else:
        suite = unittest.TestSuite()
        suite.addTest(MLModelPassesTest('test_load_constant_remove'))
        unittest.TextTestRunner().run(suite)