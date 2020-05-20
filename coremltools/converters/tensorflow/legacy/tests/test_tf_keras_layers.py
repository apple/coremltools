import unittest
import sys, os, shutil, tempfile
import numpy as np
import coremltools
from os.path import dirname
from coremltools._deps import HAS_TF_1, HAS_KERAS2_TF, MSG_TF1_NOT_FOUND, MSG_KERAS2_NOT_FOUND

if HAS_TF_1:
    import tensorflow as tf
    import tensorflow.contrib.slim as slim
    from tensorflow.python.tools.freeze_graph import freeze_graph
    import coremltools.converters.tensorflow as tf_converter

if HAS_KERAS2_TF:
    from keras import backend as K
    from keras.models import Sequential, Model
    from keras.layers import (
        Dense,
        Activation,
        Conv2D,
        Conv1D,
        Flatten,
        BatchNormalization,
        Conv2DTranspose,
        SeparableConv2D,
    )
    from keras.layers import (
        MaxPooling2D,
        AveragePooling2D,
        GlobalAveragePooling2D,
        GlobalMaxPooling2D,
    )
    from keras.layers import (
        MaxPooling1D,
        AveragePooling1D,
        GlobalAveragePooling1D,
        GlobalMaxPooling1D,
    )
    from keras.layers import Embedding, Input, Permute, Reshape, RepeatVector, Dropout
    from keras.layers import Add, Multiply, Concatenate, Dot, Maximum, Average
    from keras.layers import add, multiply, concatenate, dot, maximum, average
    from keras.layers import ZeroPadding2D, UpSampling2D, Cropping2D
    from keras.layers import ZeroPadding1D, UpSampling1D, Cropping1D
    from keras.layers.core import SpatialDropout1D, SpatialDropout2D
    from keras.layers import DepthwiseConv2D


def _tf_transpose(x, is_sequence=False):
    if not hasattr(x, "shape"):
        return x
    if len(x.shape) == 4:
        # [Batch, Height, Width, Channels] --> [Batch, Channels, Height, Width]
        x = np.transpose(x, [0, 3, 1, 2])
        return np.expand_dims(x, axis=0)
    elif len(x.shape) == 3:
        # We only deal with non-recurrent networks for now
        # [Batch, (Sequence) Length, Channels] --> [1,B, Channels, 1, Seq]
        # [0,1,2] [0,2,1]
        return np.transpose(x, [0, 2, 1])[None, :, :, None, :]
    elif len(x.shape) == 2:
        if is_sequence:  # (N,S) --> (S,N,1,)
            return x.reshape(x.shape[::-1] + (1,))
        else:  # (N,C) --> (N,C,1,1)
            return x.reshape((1,) + x.shape)  # Dense
    elif len(x.shape) == 1:
        if is_sequence:  # (S) --> (S,N,1,1,1)
            return x.reshape((x.shape[0], 1, 1))
        else:
            return x
    else:
        return x


def _convert_to_coreml(
    tf_model_path, mlmodel_path, input_name_shape_dict, output_names
):
    """ Convert and return the coreml model from the Tensorflow
  """
    model = tf_converter.convert(
        filename=tf_model_path,
        mlmodel_path=mlmodel_path,
        outputs=output_names,
        inputs=input_name_shape_dict,
    )
    return model


def _generate_data(input_shape, mode="random"):
    """
  Generate some random data according to a shape.
  """
    if input_shape is None or len(input_shape) == 0:
        return 0.5
    if mode == "zeros":
        X = np.zeros(input_shape)
    elif mode == "ones":
        X = np.ones(input_shape)
    elif mode == "linear":
        X = np.array(range(np.product(input_shape))).reshape(input_shape) * 1.0
    elif mode == "random":
        X = np.random.rand(*input_shape)
    elif mode == "random_zero_mean":
        X = np.random.rand(*input_shape) - 0.5
    return X


class TFNetworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """ Set up the unit test by loading common utilities.
    """
        K.set_learning_phase(0)

    def _simple_freeze(
        self, input_graph, input_checkpoint, output_graph, output_node_names
    ):
        # output_node_names is a string of names separated by comma
        freeze_graph(
            input_graph=input_graph,
            input_saver="",
            input_binary=True,
            input_checkpoint=input_checkpoint,
            output_node_names=output_node_names,
            restore_op_name="save/restore_all",
            filename_tensor_name="save/Const:0",
            output_graph=output_graph,
            clear_devices=True,
            initializer_nodes="",
        )

    def _test_keras_model(
        self,
        model,
        data_mode="random",
        delta=1e-2,
        use_cpu_only=False,
        one_dim_seq_flags=None,
        has_variables=True,
    ):

        """
    Saves out the backend TF graph from the Keras model and tests it  
    """

        # Some file processing
        model_dir = tempfile.mkdtemp()
        graph_def_file = os.path.join(model_dir, "tf_graph.pb")
        checkpoint_file = os.path.join(model_dir, "tf_model.ckpt")
        frozen_model_file = os.path.join(model_dir, "tf_frozen.pb")
        coreml_model_file = os.path.join(model_dir, "coreml_model.mlmodel")

        input_shape = [i for i in model.input_shape]
        for i, d in enumerate(input_shape):
            if d is None:
                input_shape[i] = 1

        input_tensor_shapes = {model.input.name: input_shape}
        output_node_names = [model.output.name[:-2]]

        tf_graph = K.get_session().graph
        tf.reset_default_graph()
        if has_variables:
            with tf_graph.as_default() as g:
                saver = tf.train.Saver()

        with tf.Session(graph=tf_graph) as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {}
            for in_tensor_name in input_tensor_shapes:
                in_tensor_shape = input_tensor_shapes[in_tensor_name]
                feed_dict[in_tensor_name] = _generate_data(in_tensor_shape, data_mode)
            # run the result
            fetches = [
                tf_graph.get_operation_by_name(name).outputs[0]
                for name in output_node_names
            ]
            result = sess.run(fetches, feed_dict=feed_dict)
            # save graph definition somewhere
            tf.train.write_graph(sess.graph, model_dir, graph_def_file, as_text=False)
            # save the weights
            if has_variables:
                saver.save(sess, checkpoint_file)

        K.clear_session()

        # freeze the graph
        if has_variables:
            self._simple_freeze(
                input_graph=graph_def_file,
                input_checkpoint=checkpoint_file,
                output_graph=frozen_model_file,
                output_node_names=",".join(output_node_names),
            )
        else:
            frozen_model_file = graph_def_file

        # convert the tensorflow model
        output_tensor_names = [name + ":0" for name in output_node_names]
        coreml_model = _convert_to_coreml(
            tf_model_path=frozen_model_file,
            mlmodel_path=coreml_model_file,
            input_name_shape_dict=input_tensor_shapes,
            output_names=output_tensor_names,
        )

        # evaluate coreml
        coreml_inputs = {}
        for idx, in_tensor_name in enumerate(input_tensor_shapes):
            in_shape = input_tensor_shapes[in_tensor_name]
            coreml_in_name = in_tensor_name.replace(":", "__").replace("/", "__")
            if one_dim_seq_flags is None:
                coreml_inputs[coreml_in_name] = _tf_transpose(
                    feed_dict[in_tensor_name]
                ).copy()
            else:
                coreml_inputs[coreml_in_name] = _tf_transpose(
                    feed_dict[in_tensor_name], one_dim_seq_flags[idx]
                ).copy()

        coreml_output = coreml_model.predict(coreml_inputs, useCPUOnly=use_cpu_only)

        for idx, out_name in enumerate(output_node_names):
            tp = _tf_transpose(result[idx]).flatten()
            out_tensor_name = out_name.replace("/", "__") + "__0"
            cp = coreml_output[out_tensor_name].flatten()
            self.assertEqual(len(tp), len(cp))
            for i in range(len(tp)):
                max_den = max(1.0, tp[i], cp[i])
                self.assertAlmostEqual(tp[i] / max_den, cp[i] / max_den, delta=delta)

        # Cleanup files - models on disk no longer useful
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

@unittest.skipIf(not HAS_TF_1, MSG_TF1_NOT_FOUND)
@unittest.skipIf(not HAS_KERAS2_TF, MSG_KERAS2_NOT_FOUND)
class KerasBasicNumericCorrectnessTest(TFNetworkTest):
    def test_dense_softmax(self):
        np.random.seed(1987)
        # Define a model
        model = Sequential()
        model.add(Dense(32, input_shape=(32,), activation="softmax"))
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test it
        self._test_keras_model(model)

    def test_dense_elu(self):
        np.random.seed(1988)
        # Define a model
        model = Sequential()
        model.add(Dense(32, input_shape=(32,), activation="elu"))
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    def test_dense_tanh(self):
        np.random.seed(1988)
        # Define a model
        model = Sequential()
        model.add(Dense(32, input_shape=(32,), activation="tanh"))
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    def test_housenet_random(self):
        np.random.seed(1988)
        num_hidden = 2
        num_features = 3
        # Define a model
        model = Sequential()
        model.add(Dense(num_hidden, input_dim=num_features))
        model.add(Activation("relu"))
        model.add(Dense(1, input_dim=num_features))
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_conv_random(self):
        np.random.seed(1988)
        input_dim = 10
        input_shape = (input_dim, input_dim, 1)
        num_kernels, kernel_height, kernel_width = 3, 5, 5
        # Define a model
        model = Sequential()
        model.add(
            Conv2D(
                input_shape=input_shape,
                filters=num_kernels,
                kernel_size=(kernel_height, kernel_width),
            )
        )
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_conv_dilated(self):
        np.random.seed(1988)
        input_dim = 10
        input_shape = (input_dim, input_dim, 1)
        num_kernels, kernel_height, kernel_width = 3, 5, 5
        # Define a model
        model = Sequential()
        model.add(
            Conv2D(
                input_shape=input_shape,
                dilation_rate=(2, 2),
                filters=num_kernels,
                kernel_size=(kernel_height, kernel_width),
            )
        )
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_conv1d_same_random(self):
        np.random.seed(1988)
        input_dim = 2
        input_length = 10
        filter_length = 3
        nb_filters = 4
        model = Sequential()
        model.add(
            Conv1D(
                nb_filters,
                kernel_size=filter_length,
                padding="same",
                input_shape=(input_length, input_dim),
            )
        )
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_conv1d_valid_random(self):
        np.random.seed(1988)
        input_dim = 2
        input_length = 10
        filter_length = 3
        nb_filters = 4
        model = Sequential()
        model.add(
            Conv1D(
                nb_filters,
                kernel_size=filter_length,
                padding="valid",
                input_shape=(input_length, input_dim),
            )
        )
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_conv1d_dilated_random(self):
        np.random.seed(1988)
        input_shape = (20, 1)
        num_kernels = 2
        filter_length = 3
        # Define a model
        model = Sequential()
        model.add(
            Conv1D(
                num_kernels,
                kernel_size=filter_length,
                padding="valid",
                input_shape=input_shape,
                dilation_rate=3,
            )
        )
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    @unittest.skip("Failing: tensorflow.python.framework.errors_impl.NotFoundError")
    def test_flatten(self):
        model = Sequential()
        model.add(Flatten(input_shape=(2, 2, 2)))
        self._test_keras_model(model, data_mode="linear", has_variables=False)

    def test_conv_dense(self):
        input_shape = (48, 48, 3)
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(10, activation="softmax"))
        # Get the coreml model
        self._test_keras_model(model)

    def test_conv_batchnorm_random(self):
        np.random.seed(1988)
        input_dim = 10
        input_shape = (input_dim, input_dim, 3)
        num_kernels = 3
        kernel_height = 5
        kernel_width = 5
        # Define a model
        # import ipdb
        # ipdb.set_trace()
        model = Sequential()
        model.add(
            Conv2D(
                input_shape=input_shape,
                filters=num_kernels,
                kernel_size=(kernel_height, kernel_width),
            )
        )
        model.add(BatchNormalization(epsilon=1e-5))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Get the coreml model
        self._test_keras_model(model)

    def test_tiny_deconv_random(self):
        np.random.seed(1988)
        input_dim = 13
        input_shape = (input_dim, input_dim, 5)
        num_kernels = 16
        kernel_height = 3
        kernel_width = 3
        # Define a model
        model = Sequential()
        model.add(
            Conv2DTranspose(
                filters=num_kernels,
                kernel_size=(kernel_height, kernel_width),
                input_shape=input_shape,
                padding="valid",
                strides=(2, 2),
            )
        )
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_deconv_random_same_padding(self):
        np.random.seed(1988)
        input_dim = 14
        input_shape = (input_dim, input_dim, 3)
        num_kernels = 16
        kernel_height = 3
        kernel_width = 3
        # Define a model
        model = Sequential()
        model.add(
            Conv2DTranspose(
                filters=num_kernels,
                kernel_size=(kernel_height, kernel_width),
                input_shape=input_shape,
                padding="same",
                strides=(2, 2),
            )
        )
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_depthwise_conv_same_pad_depth_multiplier(self):
        np.random.seed(1988)
        input_dim = 16
        input_shape = (input_dim, input_dim, 3)
        depth_multiplier = 4
        kernel_height = 3
        kernel_width = 3
        # Define a model
        model = Sequential()
        model.add(
            DepthwiseConv2D(
                depth_multiplier=depth_multiplier,
                kernel_size=(kernel_height, kernel_width),
                input_shape=input_shape,
                padding="same",
                strides=(1, 1),
            )
        )
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_depthwise_conv_valid_pad_depth_multiplier(self):
        np.random.seed(1988)
        input_dim = 16
        input_shape = (input_dim, input_dim, 3)
        depth_multiplier = 2
        kernel_height = 3
        kernel_width = 3
        # Define a model
        model = Sequential()
        model.add(
            DepthwiseConv2D(
                depth_multiplier=depth_multiplier,
                kernel_size=(kernel_height, kernel_width),
                input_shape=input_shape,
                padding="valid",
                strides=(1, 1),
            )
        )
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_separable_conv_valid_depth_multiplier(self):
        np.random.seed(1988)
        input_dim = 16
        input_shape = (input_dim, input_dim, 3)
        depth_multiplier = 5
        kernel_height = 3
        kernel_width = 3
        num_kernels = 40
        # Define a model
        model = Sequential()
        model.add(
            SeparableConv2D(
                filters=num_kernels,
                kernel_size=(kernel_height, kernel_width),
                padding="valid",
                strides=(1, 1),
                depth_multiplier=depth_multiplier,
                input_shape=input_shape,
            )
        )
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_separable_conv_same_fancy_depth_multiplier(self):
        np.random.seed(1988)
        input_dim = 16
        input_shape = (input_dim, input_dim, 3)
        depth_multiplier = 2
        kernel_height = 3
        kernel_width = 3
        num_kernels = 40
        # Define a model
        model = Sequential()
        model.add(
            SeparableConv2D(
                filters=num_kernels,
                kernel_size=(kernel_height, kernel_width),
                padding="same",
                strides=(2, 2),
                activation="relu",
                depth_multiplier=depth_multiplier,
                input_shape=input_shape,
            )
        )
        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        # Test the keras model
        self._test_keras_model(model)

    @unittest.skip("Failing: tensorflow.python.framework.errors_impl.NotFoundError")
    def test_max_pooling_no_overlap(self):
        # no_overlap: pool_size = strides
        model = Sequential()
        model.add(
            MaxPooling2D(
                input_shape=(16, 16, 3), pool_size=(2, 2), strides=None, padding="valid"
            )
        )
        self._test_keras_model(model, has_variables=False)

    def test_max_pooling_overlap_multiple(self):
        # input shape is multiple of pool_size, strides != pool_size
        model = Sequential()
        model.add(
            MaxPooling2D(
                input_shape=(18, 18, 3),
                pool_size=(3, 3),
                strides=(2, 2),
                padding="valid",
            )
        )
        self._test_keras_model(model, has_variables=False)

    def test_max_pooling_overlap_odd(self):
        model = Sequential()
        model.add(
            MaxPooling2D(
                input_shape=(16, 16, 3),
                pool_size=(3, 3),
                strides=(2, 2),
                padding="valid",
            )
        )
        self._test_keras_model(model, has_variables=False)

    def test_max_pooling_overlap_same(self):
        model = Sequential()
        model.add(
            MaxPooling2D(
                input_shape=(16, 16, 3),
                pool_size=(3, 3),
                strides=(2, 2),
                padding="same",
            )
        )
        self._test_keras_model(model, has_variables=False)

    def test_global_max_pooling(self):
        model = Sequential()
        model.add(GlobalMaxPooling2D(input_shape=(16, 16, 3)))
        self._test_keras_model(model, has_variables=False)

    @unittest.skip("Failing: https://github.com/tf-coreml/tf-coreml/issues/33")
    def test_max_pooling_1d(self):
        model = Sequential()
        model.add(MaxPooling1D(input_shape=(16, 3), pool_size=4))
        self._test_keras_model(model, has_variables=False)
