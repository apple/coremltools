import unittest
import tempfile
import numpy as np
import coremltools
import os
import shutil

import tensorflow as tf
from tensorflow.keras import backend as _keras
from tensorflow.keras import layers
from coremltools._deps import HAS_TF_2
from test_utils import generate_data, tf_transpose


class TensorFlowKerasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.keras.backend.set_learning_phase(False)

    def setUp(self):
        self.saved_model_dir = tempfile.mkdtemp()
        _, self.model_file = tempfile.mkstemp(suffix='.h5', prefix=self.saved_model_dir)

    def tearDown(self):
        if os.path.exists(self.saved_model_dir):
            shutil.rmtree(self.saved_model_dir)

    def _get_tf_tensor_name(self, graph, name):
        return graph.get_operation_by_name(name).outputs[0].name

    def _test_model(self, model, data_mode='random_zero_mean', decimal=4, use_cpu_only=False, has_variables=True, verbose=False):
        if not HAS_TF_2:
            self._test_keras_model_tf1(model, data_mode, decimal, use_cpu_only, has_variables, verbose)
        else:
            self._test_keras_model_tf2(model, data_mode, decimal, use_cpu_only, has_variables, verbose)

    def _test_keras_model_tf1(self, model, data_mode, decimal, use_cpu_only, has_variables, verbose):

        graph_def_file = os.path.join(self.saved_model_dir, 'graph.pb')
        frozen_model_file = os.path.join(self.saved_model_dir, 'frozen.pb')
        core_ml_model_file = os.path.join(self.saved_model_dir, 'model.mlmodel')

        input_shapes = {inp.op.name: inp.shape.as_list() for inp in model.inputs}
        for name, shape in input_shapes.items():
            input_shapes[name] = [dim if dim is not None else 1 for dim in shape]

        output_node_names = [output.op.name for output in model.outputs]

        tf_graph = _keras.get_session().graph
        tf.reset_default_graph()
        if has_variables:
            with tf_graph.as_default():
                saver = tf.train.Saver()

        # note: if Keras backend has_variable is False, we're not making variables constant
        with tf.Session(graph=tf_graph) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {}
            for name, shape in input_shapes.items():
                tensor_name = tf_graph.get_operation_by_name(name).outputs[0].name
                feed_dict[tensor_name] = generate_data(shape, data_mode)
            # run the result
            fetches = [
                tf_graph.get_operation_by_name(name).outputs[0] for name in output_node_names
            ]
            result = sess.run(fetches, feed_dict=feed_dict)
            # save graph definition somewhere
            tf.train.write_graph(sess.graph, self.saved_model_dir, graph_def_file, as_text=False)

            # freeze_graph() has been raising error with tf.keras models since no
            # later than TensorFlow 1.6, so we're not using freeze_graph() here.
            # See: https://github.com/tensorflow/models/issues/5387
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                tf_graph.as_graph_def(),  # The graph_def is used to retrieve the nodes
                output_node_names  # The output node names are used to select the useful nodes
            )
            with tf.gfile.GFile(frozen_model_file, 'wb') as f:
                f.write(output_graph_def.SerializeToString())

        _keras.clear_session()

        # convert to Core ML model format
        core_ml_model = coremltools.converters.tensorflow.convert(
            frozen_model_file,
            inputs=input_shapes,
            outputs=output_node_names,
            use_cpu_only=use_cpu_only)

        if verbose:
            print('\nFrozen model saved at {}'.format(frozen_model_file))
            print('\nCore ML model description:')
            from coremltools.models.neural_network.printer import print_network_spec
            print_network_spec(core_ml_model.get_spec(), style='coding')
            core_ml_model.save(core_ml_model_file)
            print('\nCore ML model saved at {}'.format(core_ml_model_file))

        # transpose input data as Core ML requires
        core_ml_inputs = {
            name: tf_transpose(feed_dict[self._get_tf_tensor_name(tf_graph, name)])
            for name in input_shapes
        }

        # run prediction in Core ML
        core_ml_output = core_ml_model.predict(core_ml_inputs, useCPUOnly=use_cpu_only)

        for idx, out_name in enumerate(output_node_names):
            tf_out = result[idx]
            if len(tf_out.shape) == 0:
                tf_out = np.array([tf_out])
            tp = tf_out.flatten()
            coreml_out = core_ml_output[out_name]
            cp = coreml_out.flatten()
            self.assertTrue(tf_out.shape == coreml_out.shape)
            for i in range(len(tp)):
                max_den = max(1.0, tp[i], cp[i])
                self.assertAlmostEqual(tp[i] / max_den, cp[i] / max_den, delta=10 ** -decimal)

    def _test_keras_model_tf2(self, model, data_mode, decimal, use_cpu_only, has_variables, verbose):

        core_ml_model_file = self.model_file.rsplit('.')[0] + '.mlmodel'

        input_dict = {inp.op.name: inp.shape.as_list() for inp in model.inputs}
        for name, shape in input_dict.items():
            input_dict[name] = [dim if dim is not None else 1 for dim in shape]
        output_list = ['Identity']
        model.save(self.model_file)

        # convert Keras model into Core ML model format
        core_ml_model = coremltools.converters.tensorflow.convert(
            filename=self.model_file,
            inputs=input_dict,
            outputs=output_list,
            use_cpu_only=use_cpu_only)

        if verbose:
            print('\nKeras model saved at {}'.format(self.model_file))
            print('\nCore ML model description:')
            from coremltools.models.neural_network.printer import print_network_spec
            print_network_spec(core_ml_model.get_spec(), style='coding')
            core_ml_model.save(core_ml_model_file)
            print('\nCore ML model saved at {}'.format(core_ml_model_file))

        core_ml_inputs = {
            name: generate_data(shape, data_mode) for name, shape in input_dict.items()
        }

        # run prediction and compare results
        keras_output = model.predict(list(core_ml_inputs.values())[0])
        core_ml_output = core_ml_model.predict(
            core_ml_inputs, useCPUOnly=use_cpu_only)[output_list[0]]

        if verbose:
            print('\nPredictions', keras_output.shape, ' vs.', core_ml_output.shape)
            print(keras_output.flatten()[:6])
            print(core_ml_output.flatten()[:6])

        np.testing.assert_array_equal(
            keras_output.shape, core_ml_output.shape)
        np.testing.assert_almost_equal(
            keras_output.flatten(), core_ml_output.flatten(), decimal=decimal)


class SimpleLayerTests(TensorFlowKerasTests):

    def test_dense_softmax(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(16, input_shape=(16,), activation=tf.nn.softmax))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model)

    def test_dense_elu(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(16, input_shape=(16,), activation=tf.nn.elu))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model, decimal=2)

    def test_dense_tanh(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(16, input_shape=(16,), activation=tf.nn.tanh))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model)

    def test_housenet_random(self):
        num_hidden = 2
        num_features = 3
        model = tf.keras.Sequential()
        model.add(layers.Dense(num_hidden, input_dim=num_features))
        model.add(layers.Activation(tf.nn.relu))
        model.add(layers.Dense(1, input_dim=num_features))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model)

    def test_tiny_conv2d_random(self):
        input_dim = 10
        input_shape = (input_dim, input_dim, 1)
        num_kernels, kernel_height, kernel_width = 3, 5, 5
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(
            input_shape=input_shape,
            filters=num_kernels, kernel_size=(kernel_height, kernel_width)))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model)

    def test_tiny_conv2d_dilated_random(self):
        input_dim = 10
        input_shape = (input_dim, input_dim, 1)
        num_kernels, kernel_height, kernel_width = 3, 5, 5
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(
            input_shape=input_shape, dilation_rate=(2, 2),
            filters=num_kernels, kernel_size=(kernel_height, kernel_width)))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model)

    def test_tiny_conv1d_same_random(self):
        input_dim = 2
        input_length = 10
        filter_length = 3
        nb_filters = 4
        model = tf.keras.Sequential()
        model.add(layers.Conv1D(
            nb_filters, kernel_size=filter_length, padding='same',
            input_shape=(input_length, input_dim)))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model)

    def test_tiny_conv1d_valid_random(self):
        input_dim = 2
        input_length = 10
        filter_length = 3
        nb_filters = 4
        model = tf.keras.Sequential()
        model.add(layers.Conv1D(
            nb_filters, kernel_size=filter_length, padding='valid',
            input_shape=(input_length, input_dim)))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model)

    @unittest.skip('non-equal block shape is not yet supported')
    def test_tiny_conv1d_dilated_random(self):
        input_shape = (20, 1)
        num_kernels = 2
        filter_length = 3
        model = tf.keras.Sequential()
        model.add(layers.Conv1D(
            num_kernels, kernel_size=filter_length, padding='valid',
            input_shape=input_shape, dilation_rate=3))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model)

    def test_flatten(self):
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(2, 2, 2)))
        self._test_model(model, data_mode='linear', has_variables=False)

    def test_conv_dense(self):
        input_shape = (48, 48, 3)
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=input_shape))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation=tf.nn.softmax))
        self._test_model(model)

    def test_conv_batchnorm_random(self):
        input_dim = 10
        input_shape = (input_dim, input_dim, 3)
        num_kernels = 3
        kernel_height = 5
        kernel_width = 5
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(
            input_shape=input_shape,
            filters=num_kernels,
            kernel_size=(kernel_height, kernel_width)))
        model.add(layers.BatchNormalization(epsilon=1e-5))
        model.add(layers.Dense(10, activation=tf.nn.softmax))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model, decimal=2, has_variables=True)

    @unittest.skip('list index out of range')
    def test_tiny_deconv_random(self):
        input_dim = 13
        input_shape = (input_dim, input_dim, 5)
        num_kernels = 16
        kernel_height = 3
        kernel_width = 3
        model = tf.keras.Sequential()
        model.add(layers.Conv2DTranspose(
            filters=num_kernels,
            kernel_size=(kernel_height, kernel_width),
            input_shape=input_shape, padding='valid', strides=(2, 2)))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model)

    @unittest.skip('Deconvolution layer has weight matrix of size 432 to encode a 3 x 4 x 3 x 3 convolution.')
    def test_tiny_deconv_random_same_padding(self):
        input_dim = 14
        input_shape = (input_dim, input_dim, 3)
        num_kernels = 16
        kernel_height = 3
        kernel_width = 3
        model = tf.keras.Sequential()
        model.add(layers.Conv2DTranspose(
            filters=num_kernels,
            kernel_size=(kernel_height, kernel_width),
            input_shape=input_shape, padding='same', strides=(2, 2)))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model)

    def test_tiny_depthwise_conv_same_pad_depth_multiplier(self):
        input_dim = 16
        input_shape = (input_dim, input_dim, 3)
        depth_multiplier = 4
        kernel_height = 3
        kernel_width = 3
        model = tf.keras.Sequential()
        model.add(layers.DepthwiseConv2D(
            depth_multiplier=depth_multiplier,
            kernel_size=(kernel_height, kernel_width),
            input_shape=input_shape, padding='same', strides=(1, 1)))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model)

    def test_tiny_depthwise_conv_valid_pad_depth_multiplier(self):
        input_dim = 16
        input_shape = (input_dim, input_dim, 3)
        depth_multiplier = 2
        kernel_height = 3
        kernel_width = 3
        model = tf.keras.Sequential()
        model.add(layers.DepthwiseConv2D(
            depth_multiplier=depth_multiplier,
            kernel_size=(kernel_height, kernel_width),
            input_shape=input_shape, padding='valid', strides=(1, 1)))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model)

    def test_tiny_separable_conv_valid_depth_multiplier(self):
        input_dim = 16
        input_shape = (input_dim, input_dim, 3)
        depth_multiplier = 5
        kernel_height = 3
        kernel_width = 3
        num_kernels = 40
        model = tf.keras.Sequential()
        model.add(layers.SeparableConv2D(
            filters=num_kernels, kernel_size=(kernel_height, kernel_width),
            padding='valid', strides=(1, 1), depth_multiplier=depth_multiplier,
            input_shape=input_shape))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model, decimal=2)

    def test_tiny_separable_conv_same_fancy_depth_multiplier(self):
        input_dim = 16
        input_shape = (input_dim, input_dim, 3)
        depth_multiplier = 2
        kernel_height = 3
        kernel_width = 3
        num_kernels = 40
        model = tf.keras.Sequential()
        model.add(layers.SeparableConv2D(
            filters=num_kernels, kernel_size=(kernel_height, kernel_width),
            padding='same', strides=(2, 2), activation='relu', depth_multiplier=depth_multiplier,
            input_shape=input_shape))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_model(model, decimal=2)

    def test_max_pooling_no_overlap(self):
        # no_overlap: pool_size = strides
        model = tf.keras.Sequential()
        model.add(layers.MaxPooling2D(
            input_shape=(16, 16, 3), pool_size=(2, 2),
            strides=None, padding='valid'))
        self._test_model(model, has_variables=False)

    def test_max_pooling_overlap_multiple(self):
        # input shape is multiple of pool_size, strides != pool_size
        model = tf.keras.Sequential()
        model.add(layers.MaxPooling2D(
            input_shape=(18, 18, 3), pool_size=(3, 3),
            strides=(2, 2), padding='valid'))
        self._test_model(model, has_variables=False)

    def test_max_pooling_overlap_odd(self):
        model = tf.keras.Sequential()
        model.add(layers.MaxPooling2D(
            input_shape=(16, 16, 3), pool_size=(3, 3),
            strides=(2, 2), padding='valid'))
        self._test_model(model, has_variables=False)

    def test_max_pooling_overlap_same(self):
        model = tf.keras.Sequential()
        model.add(layers.MaxPooling2D(
            input_shape=(16, 16, 3), pool_size=(3, 3),
            strides=(2, 2), padding='same'))
        self._test_model(model, has_variables=False)

    def test_global_max_pooling_2d(self):
        model = tf.keras.Sequential()
        model.add(layers.GlobalMaxPooling2D(input_shape=(16, 16, 3)))
        self._test_model(model, has_variables=False)

    def test_global_avg_pooling_2d(self):
        model = tf.keras.Sequential()
        model.add(layers.GlobalAveragePooling2D(input_shape=(16, 16, 3)))
        self._test_model(model, has_variables=False)

    def test_max_pooling_1d(self):
        model = tf.keras.Sequential()
        model.add(layers.MaxPooling1D(input_shape=(16, 3), pool_size=2))
        self._test_model(model, has_variables=False)


if __name__ == '__main__':
    np.random.seed(1984)
    unittest.main()
