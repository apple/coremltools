import numpy as np
import unittest
import coremltools.models.datatypes as datatypes
from coremltools.models import neural_network as neural_network
from coremltools.models import MLModel
from coremltools.models.neural_network.printer import print_network_spec
from coremltools.converters.nnssa.coreml.graph_pass.mlmodel_passes import \
        remove_disconnected_layers, transform_conv_crop, remove_redundant_transposes

import copy
import pytest

DEBUG = False
np.random.seed(10)


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
        remove_disconnected_layers(spec)
        np.testing.assert_equal(2, len(spec.neuralNetwork.layers))

    def test_dead_layer_remove(self):
        input_features = [('data', datatypes.Array(*(3, 4)))]
        output_features = [('out', None)]
        builder = neural_network.NeuralNetworkBuilder(input_features, output_features, disable_rank5_shape_mapping=True)
        builder.add_activation('relu1', 'RELU', 'data', 'relu1')
        builder.add_load_constant_nd('const1', 'c1', constant_value=np.ones((5,)), shape=(5,))
        builder.add_load_constant_nd('const2', 'c2', constant_value=np.ones((5,)), shape=(5,))
        builder.add_split_nd('splitnd1', 'const2', ['s1', 's2', 's3'], axis=0, num_splits=3)
        builder.add_squeeze('squeeze', 's1', 'squeeze_out')
        builder.add_activation('relu4', 'RELU', 's2', 'relu4')
        builder.add_activation('relu5', 'RELU', 'relu4', 'relu5')
        builder.add_load_constant_nd('const3', 'c3', constant_value=np.ones((5,)), shape=(5,))
        builder.add_activation('relu2', 'RELU', 'relu1', 'out')
        spec = builder.spec
        np.testing.assert_equal(9, len(spec.neuralNetwork.layers))
        remove_disconnected_layers(spec)
        np.testing.assert_equal(2, len(spec.neuralNetwork.layers))

    @pytest.mark.xfail
    def test_dead_layer_remove_branch(self):
        convergence_tolerance = 1e-8

        input_features = [('input', datatypes.Array(*(2,)))]
        output_features = [('out', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features, output_features, disable_rank5_shape_mapping=True)
        # add condition to break from the loop, if convergence criterion is met
        builder.add_less_than('cond', ['input'], 'cond', alpha=convergence_tolerance)
        branch_layer = builder.add_branch('branch_layer', 'cond')
        builder_ifbranch = neural_network.NeuralNetworkBuilder(nn_spec=branch_layer.branch.ifBranch)
        builder_ifbranch.add_activation('relu1', 'RELU', 'input', 'relu1_out')
        builder_ifbranch.add_activation('relu2_out', 'RELU', 'relu1_out', 'relu2_out')
        builder_elsebranch = neural_network.NeuralNetworkBuilder(nn_spec=branch_layer.branch.elseBranch)
        builder_elsebranch.add_activation('linear1', 'LINEAR', 'input', 'linear1_out')
        builder_elsebranch.add_activation('linear2', 'LINEAR', 'linear1_out', 'relu2_out')
        builder.add_squeeze('out', 'input', 'out', squeeze_all=True)

        mlmodel = MLModel(builder.spec)
        data = np.random.rand(2,)
        data_dict = {'input': data}
        before_pass_out = mlmodel.predict(data_dict)['out']
        if DEBUG:
            print('\n mlmodel description before remove disconnected layers pass: \n')
            print_network_spec(builder.spec, style='coding')
        remove_disconnected_layers(builder.spec)
        if DEBUG:
            print('\n mlmodel description after remove disconnected layers pass: \n')
            print_network_spec(builder.spec, style='coding')
        mlmodel = MLModel(builder.spec)
        after_pass_out = mlmodel.predict(data_dict)['out']

        np.testing.assert_almost_equal(before_pass_out, after_pass_out, decimal=2)
        np.testing.assert_equal(len(builder.spec.neuralNetwork.layers), 1)

    @pytest.mark.xfail
    def test_dead_layer_partial_branch(self):
        convergence_tolerance = 1e-8

        input_features = [('input', datatypes.Array(*(2,)))]
        output_features = [('out', None)]

        builder = neural_network.NeuralNetworkBuilder(input_features, output_features, disable_rank5_shape_mapping=True)
        # add condition to break from the loop, if convergence criterion is met
        builder.add_less_than('cond', ['input'], 'cond', alpha=convergence_tolerance)
        branch_layer = builder.add_branch('branch_layer', 'cond')
        builder_ifbranch = neural_network.NeuralNetworkBuilder(nn_spec=branch_layer.branch.ifBranch)
        builder_ifbranch.add_activation('relu1', 'RELU', 'input', 'relu1_out')
        builder_ifbranch.add_activation('relu2_out', 'RELU', 'relu1_out', 'relu2_out')
        builder_elsebranch = neural_network.NeuralNetworkBuilder(nn_spec=branch_layer.branch.elseBranch)
        builder_elsebranch.add_activation('linear1', 'LINEAR', 'input', 'linear1_out')
        builder_elsebranch.add_activation('linear_red_1', 'LINEAR', 'input', 'linear_red1_out')
        builder_elsebranch.add_activation('linear_red_2', 'LINEAR', 'linear_red1_out', 'linear_red2_out')
        builder_elsebranch.add_activation('linear2', 'LINEAR', 'linear1_out', 'relu2_out')
        builder.add_squeeze('out', 'relu2_out', 'out', squeeze_all=True)

        mlmodel = MLModel(builder.spec)
        data = np.random.rand(2,)
        data_dict = {'input': data}
        before_pass_out = mlmodel.predict(data_dict)['out']
        if DEBUG:
            print('\n mlmodel description before remove disconnected layers pass: \n')
            print_network_spec(builder.spec, style='coding')
        old_spec = copy.copy(builder.spec)
        remove_disconnected_layers(builder.spec)
        if DEBUG:
            print('\n mlmodel description after remove disconnected layers pass: \n')
            print_network_spec(builder.spec, style='coding')
        mlmodel = MLModel(builder.spec)
        after_pass_out = mlmodel.predict(data_dict)['out']

        np.testing.assert_almost_equal(before_pass_out, after_pass_out, decimal=2)
        np.testing.assert_equal(len(old_spec.neuralNetwork.layers[1].branch.ifBranch.layers),
                                len(builder.spec.neuralNetwork.layers[1].branch.ifBranch.layers))
        np.testing.assert_equal(len(builder.spec.neuralNetwork.layers[1].branch.elseBranch.layers), 2)

    def test_conv_crop_bn_to_conv_bn_crop(self):
        input_features = [('data', datatypes.Array(1, 10, 10))]
        output_features = [('out', None)]
        builder = neural_network.NeuralNetworkBuilder(input_features, output_features)
        W = np.ones((1, 2, 2, 2), dtype=np.float32)
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

        # Predict
        mlmodel = MLModel(builder.spec)
        data = np.random.rand(1, 10, 10)
        data_dict = {'data': data}
        before_pass_out = mlmodel.predict(data_dict, useCPUOnly=True)['out']

        # transform the pattern
        transform_conv_crop(builder.spec)
        # Conv -> BN -> Crop
        np.testing.assert_equal('batchnorm', spec.layers[1].WhichOneof('layer'))
        np.testing.assert_equal('crop', spec.layers[2].WhichOneof('layer'))

        # Predict
        mlmodel = MLModel(builder.spec)
        after_pass_out = mlmodel.predict(data_dict, useCPUOnly=True)['out']
        np.testing.assert_almost_equal(before_pass_out, after_pass_out, decimal=3)

    def test_conv_crop_bn_relu_to_conv_bn_relu_crop(self):
        input_features = [('data', datatypes.Array(1, 10, 10))]
        output_features = [('out', None)]
        builder = neural_network.NeuralNetworkBuilder(input_features, output_features)

        W = np.ones((1, 2, 2, 2), dtype=np.float32)
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

        # Predict
        mlmodel = MLModel(builder.spec)
        data = np.random.rand(1, 10, 10)
        data_dict = {'data': data}
        before_pass_out = mlmodel.predict(data_dict, useCPUOnly=True)['out']

        # transform the pattern
        transform_conv_crop(builder.spec)
        # Conv -> BN -> ReLU -> Crop
        np.testing.assert_equal('batchnorm', spec.layers[1].WhichOneof('layer'))
        np.testing.assert_equal('activation', spec.layers[2].WhichOneof('layer'))
        np.testing.assert_equal('crop', spec.layers[3].WhichOneof('layer'))

        # Predict
        mlmodel = MLModel(builder.spec)
        after_pass_out = mlmodel.predict(data_dict, useCPUOnly=True)['out']
        np.testing.assert_almost_equal(before_pass_out, after_pass_out, decimal=3)

    def test_redundant_transposes(self):

        def _build_and_test_network(input_size, transpose_layers, expected_layers):
            """
            Helper function for testing transpose removal.

            Args:
                input_size: Size of the input network tensor.
                transpose_layers: Array of transpose axes definitions.
                expected_layers: Array of indices into transpose_layers indicating
                    which of the transpose layers should be present after the
                    graph pass.
            """
            input_features = [('data', datatypes.Array(*input_size))]
            output_features = [('out', None)]
            builder = neural_network.NeuralNetworkBuilder(input_features, output_features)
            last_layer = 'data'
            for idx, axes in enumerate(transpose_layers):
                name = 't{}'.format(idx)
                if idx == len(transpose_layers) - 1:
                    output_name = 'out'
                else:
                    output_name = name + '_out'
                builder.add_transpose(name=name,
                                      axes=axes,
                                      input_name=last_layer,
                                      output_name=output_name)
                last_layer = output_name

            spec = builder.spec.neuralNetwork
            # Check the network before the graph pass.
            for idx in range(len(transpose_layers)):
                np.testing.assert_equal('transpose', spec.layers[idx].WhichOneof('layer'))
            # Run the removal pass.
            remove_redundant_transposes(builder.spec)
            # Verify only the expected layers remain.
            np.testing.assert_equal(len(spec.layers), len(expected_layers))
            for output_layer_idx, input_layer_idx in enumerate(expected_layers):
                np.testing.assert_equal(
                    'transpose',
                    spec.layers[output_layer_idx].WhichOneof('layer')
                )
                np.testing.assert_array_equal(
                    transpose_layers[input_layer_idx],
                    spec.layers[output_layer_idx].transpose.axes
                )

        _build_and_test_network(
            input_size=[1, 10, 10],
            # These transposes together are the identity.
            transpose_layers=[[2, 0, 1], [1, 2, 0]],
            expected_layers=[],
        )

        _build_and_test_network(
            input_size=[1, 10, 10],
            # These transposes are not inverses.
            transpose_layers=[[2, 0, 1], [2, 0, 1]],
            expected_layers=[0, 1],
        )

        _build_and_test_network(
            input_size=[1, 1, 10, 10, 3],
            # First two are the identity, then an extra.
            transpose_layers=[[2, 4, 1, 0, 3], [3, 2, 0, 4, 1], [1, 0, 2, 3, 4]],
            expected_layers=[2],
        )

        _build_and_test_network(
            input_size=[1, 1, 10, 10, 3],
            # First is okay, next two are the identity.
            transpose_layers=[[1, 0, 2, 3, 4], [2, 4, 1, 0, 3], [3, 2, 0, 4, 1]],
            expected_layers=[0],
        )

        # A slightly more complicated test case where there are two transposes
        # in topological order, but are actually in parallel in the graph.
        builder = neural_network.NeuralNetworkBuilder(
            [('data', datatypes.Array(2, 4, 8))],
            [('out', None)]
        )
        last_layer = 'data'
        builder.add_transpose(name='t1',
                              axes=[0, 2, 1],
                              input_name='data',
                              output_name='t1')
        builder.add_transpose(name='t2',
                              axes=[0, 2, 1],
                              input_name='data',
                              output_name='t2')
        builder.add_stack(name='stack',
                          input_names=['t1', 't2'],
                          output_name='out')
        spec = builder.spec.neuralNetwork
        # Run the removal pass.
        remove_redundant_transposes(builder.spec)
        # Verify nothing was removed.
        np.testing.assert_equal(len(spec.layers), 3)


if __name__ == '__main__':
    RUN_ALL_TESTS = True
    if RUN_ALL_TESTS:
        unittest.main()
    else:
        suite = unittest.TestSuite()
        suite.addTest(MLModelPassesTest('test_load_constant_remove'))
        unittest.TextTestRunner().run(suite)
