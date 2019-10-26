import numpy as np
import unittest
import coremltools.models.datatypes as datatypes
from coremltools.models import MLModel
from coremltools.models import neural_network as neural_network
from coremltools.converters.nnssa.coreml.graph_pass.mlmodel_passes import \
        remove_disconnected_constants, transform_conv_crop_bn_to_conv_bn_crop


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
        W = np.ones((2, 1, 2, 2), dtype=np.float32)
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
                        left=1, right=0, top=1, bottom=0, offset=0,
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

        # Predict
        mlmodel = MLModel(builder.spec)
        data = np.random.rand(1, 10, 10)
        data_dict = {'data': data}
        before_pass_out = mlmodel.predict(data_dict)['out']

        # transform the pattern
        transform_conv_crop_bn_to_conv_bn_crop(builder.spec)

        # Conv -> BN -> Crop
        np.testing.assert_equal('batchnorm', spec.layers[1].WhichOneof('layer'))
        np.testing.assert_equal('crop', spec.layers[2].WhichOneof('layer'))

        # Predict
        mlmodel = MLModel(builder.spec)
        after_pass_out = mlmodel.predict(data_dict)['out']
        np.testing.assert_equal(before_pass_out, after_pass_out)

if __name__ == '__main__':
    RUN_ALL_TESTS = True
    if RUN_ALL_TESTS:
        unittest.main()
    else:
        suite = unittest.TestSuite()
        suite.addTest(MLModelPassesTest('test_load_constant_remove'))
        unittest.TextTestRunner().run(suite)
