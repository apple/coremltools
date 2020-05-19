import unittest

import numpy as np
import tempfile
import pytest
import shutil
import os

import coremltools
from coremltools._deps import HAS_KERAS_TF
from coremltools._deps import HAS_TF
from coremltools.models.utils import _get_custom_layer_names, \
    _replace_custom_layer_name, macos_version, is_macos
from coremltools.proto import Model_pb2

if HAS_KERAS_TF:
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    from coremltools.converters import keras as keras_converter

if HAS_TF:
    import tensorflow as tf
    from tensorflow.python.platform import gfile
    from tensorflow.python.tools import freeze_graph

    tf.compat.v1.disable_eager_execution()


@unittest.skipIf(not HAS_KERAS_TF, 'Missing keras. Skipping tests.')
@pytest.mark.keras1
class KerasBasicNumericCorrectnessTest(unittest.TestCase):

    def test_classifier(self):
        np.random.seed(1988)

        print('running test classifier')

        input_dim = 5
        num_hidden = 12
        num_classes = 6
        input_length = 3

        model = Sequential()
        model.add(LSTM(num_hidden, input_dim=input_dim, input_length=input_length, return_sequences=False))
        model.add(Dense(num_classes, activation='softmax'))

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        input_names = ['input']
        output_names = ['zzzz']
        class_labels = ['a', 'b', 'c', 'd', 'e', 'f']
        predicted_feature_name = 'pf'
        coremlmodel = keras_converter.convert(model, input_names, output_names, class_labels=class_labels, predicted_feature_name=predicted_feature_name, predicted_probabilities_output=output_names[0])

        if is_macos() and macos_version() >= (10, 13):
            inputs = np.random.rand(input_dim)
            outputs = coremlmodel.predict({'input': inputs})
            # this checks that the dictionary got the right name and type
            self.assertEquals(type(outputs[output_names[0]]), type({'a': 0.5}))

    def test_classifier_no_name(self):
        np.random.seed(1988)

        input_dim = 5
        num_hidden = 12
        num_classes = 6
        input_length = 3

        model = Sequential()
        model.add(LSTM(num_hidden, input_dim=input_dim, input_length=input_length, return_sequences=False))
        model.add(Dense(num_classes, activation='softmax'))

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        input_names = ['input']
        output_names = ['zzzz']
        class_labels = ['a', 'b', 'c', 'd', 'e', 'f']
        predicted_feature_name = 'pf'
        coremlmodel = keras_converter.convert(model, input_names, output_names, class_labels=class_labels, predicted_feature_name=predicted_feature_name)

        if is_macos() and macos_version() >= (10, 13):
            inputs = np.random.rand(input_dim)
            outputs = coremlmodel.predict({'input': inputs})
            # this checks that the dictionary got the right name and type
            self.assertEquals(type(outputs[output_names[0]]), type({'a': 0.5}))

    def test_internal_layer(self):

        np.random.seed(1988)

        input_dim = 5
        num_channels1 = 10
        num_channels2 = 7
        num_channels3 = 5

        w1 = (np.random.rand(input_dim, num_channels1) - 0.5) / 5.0;
        w2 = (np.random.rand(num_channels1, num_channels2) - 0.5) / 5.0;
        w3 = (np.random.rand(num_channels2, num_channels3) - 0.5) / 5.0;

        b1 = (np.random.rand(num_channels1, ) - 0.5) / 5.0;
        b2 = (np.random.rand(num_channels2, ) - 0.5) / 5.0;
        b3 = (np.random.rand(num_channels3, ) - 0.5) / 5.0;

        model = Sequential()
        model.add(Dense(num_channels1, input_dim=input_dim))
        model.add(Dense(num_channels2, name='middle_layer'))
        model.add(Dense(num_channels3))

        model.set_weights([w1, b1, w2, b2, w3, b3])

        input_names = ['input']
        output_names = ['output']
        coreml1 = keras_converter.convert(model, input_names, output_names)

        # adjust the output parameters of coreml1 to include the intermediate layer
        spec = coreml1.get_spec()
        coremlNewOutputs = spec.description.output.add()
        coremlNewOutputs.name = 'middle_layer_output'
        coremlNewParams = coremlNewOutputs.type.multiArrayType
        coremlNewParams.dataType = coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Value('DOUBLE')
        coremlNewParams.shape.extend([num_channels2])

        coremlfinal = coremltools.models.MLModel(spec)

        # generate a second model which
        model2 = Sequential()
        model2.add(Dense(num_channels1, input_dim=input_dim))
        model2.add(Dense(num_channels2))
        model2.set_weights([w1, b1, w2, b2])

        coreml2 = keras_converter.convert(model2, input_names, ['output2'])

        if is_macos() and macos_version() >= (10, 13):
            # generate input data
            inputs = np.random.rand(input_dim)

            fullOutputs = coremlfinal.predict({'input': inputs})

            partialOutput = coreml2.predict({'input': inputs})

            for i in range(0, num_channels2):
                self.assertAlmostEquals(fullOutputs['middle_layer_output'][i], partialOutput['output2'][i], 2)

# Base class for basic TF conversions
@unittest.skipIf(not HAS_TF, 'Missing TF. Skipping tests.')
class TfConversionTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        _, self.graph_file = tempfile.mkstemp(suffix='.pb', prefix=self.tmp_dir)
        _, self.checkpoint_file = tempfile.mkstemp(suffix='.ckpt', prefix=self.tmp_dir)
        _, self.class_label_file = tempfile.mkstemp(suffix='.txt', prefix=self.tmp_dir)
        _, self.frozen_graph_file = tempfile.mkstemp(suffix='.pb', prefix=self.tmp_dir)
        self.image_size = 224
        self._setup_tf_model()

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def _setup_tf_model(self):
        with open(self.class_label_file, 'w+') as labels_file:
            for a in range(10):
                labels_file.write(str(a + 1) + "\n")

        with tf.Graph().as_default():
            images = tf.random.uniform(self._get_input_shape(), maxval=1)

            # Create the model.
            (i_placeholder, probabilities) = self._get_network()

            saver = tf.train.Saver()
            init_op = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init_op)
                probabilities = sess.run(probabilities, {i_placeholder: images.eval()})
                saver.save(sess, self.checkpoint_file)

                with gfile.GFile(self.graph_file, 'wb') as f:
                    f.write(sess.graph_def.SerializeToString())
                freeze_graph.freeze_graph(self.graph_file,
                                          '',
                                          True,
                                          self.checkpoint_file,
                                          'Softmax',
                                          '',
                                          '',
                                          self.frozen_graph_file,
                                          False,
                                          '')

    # Returns (input_layer, output_layer)
    def _get_network(self):
        raise NotImplementedError

    # Returns something like [batch_size, height, width, channels] or
    # [batch_size, channels, height, width]
    def _get_input_shape(self):
        raise NotImplementedError

# Converting TF models with convolution layers
@unittest.skipIf(not HAS_TF, 'Missing TF. Skipping tests.')
class TFBasicConversionTest(TfConversionTestBase):
    # Basic NN using convolutions
    def _get_network(self):
        i_placeholder = tf.placeholder(name='input', dtype=tf.float32, shape=self._get_input_shape())
        net = self.my_conv_2d(i_placeholder, [1, 3, 3, 1], 1, 1, 'first')
        net = tf.nn.avg_pool2d(net, 224, strides=1, padding='VALID', name='AvgPool_1a')
        net = self.my_conv_2d(net, [1, 1, 1, 10], 10, 1, 'fc', activation_fn=None, with_bias_add=False)
        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        probabilities = tf.nn.softmax(net, name='Softmax')
        return (i_placeholder, probabilities)

    def _get_input_shape(self):
        return [1, self.image_size, self.image_size, 3]

    def my_conv_2d(self, input, weight_shape, num_filters, strides, name, activation_fn=tf.nn.relu, with_bias_add=True):
        my_weights = tf.get_variable(name=name + 'weights', shape=weight_shape)
        if with_bias_add:
            my_bias = tf.get_variable(name=name + 'bias', shape=num_filters)
        my_conv = tf.nn.conv2d(input, my_weights, strides=strides, padding='SAME', name=name)
        if with_bias_add:
            my_conv = tf.nn.bias_add(my_conv, my_bias)
        if (activation_fn != None):
            conv_layer_out = activation_fn(my_conv)
        else:
            conv_layer_out = my_conv
        return conv_layer_out

    def test_classifier_with_label_file(self):
        coremltools.converters.tensorflow.convert(
            self.frozen_graph_file,
            inputs={'input': [1, 224, 224, 3]},
            image_input_names=['input'],
            outputs=['Softmax'],
            predicted_feature_name='classLabel',
            class_labels=self.class_label_file)

    def test_classifier_with_int_label_list(self):
        coremltools.converters.tensorflow.convert(
            self.frozen_graph_file,
            inputs={'input': [1, 224, 224, 3]},
            image_input_names=['input'],
            outputs=['Softmax'],
            predicted_feature_name='classLabel',
            class_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_classifier_with_string_label_list(self):
        coremltools.converters.tensorflow.convert(
            self.frozen_graph_file,
            inputs={'input': [1, 224, 224, 3]},
            image_input_names=['input'],
            outputs=['Softmax'],
            predicted_feature_name='classLabel',
            class_labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

    def test_classifier_without_class_labels(self):
        coremltools.converters.tensorflow.convert(
            self.frozen_graph_file,
            inputs={'input': [1, 224, 224, 3]},
            image_input_names=['input'],
            outputs=['Softmax'])

    def test_classifier_nhwc(self):
        # Test manually specifying the image format. The converter would have
        # detected NHWC.
        coremltools.converters.tensorflow.convert(
            self.frozen_graph_file,
            inputs={'input': [1, 224, 224, 3]},
            image_input_names=['input'],
            outputs=['Softmax'],
            tf_image_format='NHWC')

    def test_classifier_nchw(self):
        # Expect failure - input dimensions are incompatible with NCHW
        with self.assertRaises(ValueError) as e:
            coremltools.converters.tensorflow.convert(
                self.frozen_graph_file,
                inputs={'input': [1, 224, 224, 3]},
                image_input_names=['input'],
                outputs=['Softmax'],
                tf_image_format='NCHW')

class TFConversionTestWithSimpleModelBase(TfConversionTestBase):
    # Create a basic network with no convolution layers; converter is not given hints about the image format
    def _get_network(self):
        i_placeholder = tf.placeholder(name='input', dtype=tf.float32, shape=self._get_input_shape())
        net = tf.layers.Flatten(name='flatten')(i_placeholder)
        net = tf.contrib.slim.fully_connected(net, 256)
        net = tf.contrib.slim.dropout(net)
        net = tf.contrib.slim.fully_connected(net, 10, activation_fn=None)
        probabilities = tf.nn.softmax(net, name='Softmax')
        return (i_placeholder, probabilities)

@unittest.skipIf(not HAS_TF, 'Missing TF. Skipping tests.')
class TFConversionTestWithSimpleNHWCModel(TFConversionTestWithSimpleModelBase):
    # Use NHWC format
    def _get_input_shape(self):
        return [1, self.image_size, self.image_size, 3]

    def test_classifier_no_tf_image_format_selected(self):
        # Expect to succeed; model has no convolutions but NHWC should have been
        # default
        coremltools.converters.tensorflow.convert(
            self.frozen_graph_file,
            inputs={'input': [1, 224, 224, 3]},
            image_input_names=['input'],
            outputs=['Softmax'])

    def test_classifier_nhwc(self):
        # Manually using the correct format should succeed
        coremltools.converters.tensorflow.convert(
            self.frozen_graph_file,
            inputs={'input': [1, 224, 224, 3]},
            image_input_names=['input'],
            outputs=['Softmax'],
            tf_image_format='NHWC')

    def test_classifier_nchw(self):
        # Expect failure - input dimensions are incompatible with NCHW
        with self.assertRaises(ValueError) as e:
            coremltools.converters.tensorflow.convert(
                self.frozen_graph_file,
                inputs={'input': [1, 224, 224, 3]},
                image_input_names=['input'],
                outputs=['Softmax'],
                tf_image_format='NCHW')

@unittest.skipIf(not HAS_TF, 'Missing TF. Skipping tests.')
class TFConversionTestWithSimpleNCHWModel(TFConversionTestWithSimpleModelBase):
    # Use NHWC format
    def _get_input_shape(self):
        return [1, 3, self.image_size, self.image_size]

    def test_classifier_no_tf_image_format_selected(self):
        # Expect to fail. Could not find image format in convolution layers and no parameter was given,
        # so fall back to NHWC which is incompatible
        with self.assertRaises(ValueError) as e:
            coremltools.converters.tensorflow.convert(
                self.frozen_graph_file,
                inputs={'input': [1, 3, 224, 224]},
                image_input_names=['input'],
                outputs=['Softmax'])

    def test_classifier_nhwc(self):
        # Expect to fail, NHWC is incorrect format
        with self.assertRaises(ValueError) as e:
            coremltools.converters.tensorflow.convert(
                self.frozen_graph_file,
                inputs={'input': [1, 3, 224, 224]},
                image_input_names=['input'],
                outputs=['Softmax'],
                tf_image_format='NHWC')

    def test_classifier_nchw(self):
        # Expect success - user selected the correct format
        coremltools.converters.tensorflow.convert(
            self.frozen_graph_file,
            inputs={'input': [1, 3, 224, 224]},
            image_input_names=['input'],
            outputs=['Softmax'],
            tf_image_format='NCHW')

class CustomLayerUtilsTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        spec = Model_pb2.Model()
        spec.specificationVersion = coremltools.SPECIFICATION_VERSION

        features = ['feature_1', 'feature_2']
        output = 'output'
        for f in features:
            input_ = spec.description.input.add()
            input_.name = f
            input_.type.doubleType.MergeFromString(b'')

        output_ = spec.description.output.add()
        output_.name = output
        output_.type.doubleType.MergeFromString(b'')

        layer = spec.neuralNetwork.layers.add()
        layer.name = 'custom1'
        layer.input.append('input')
        layer.output.append('temp1')
        layer.custom.className = 'name1'

        layer2 = spec.neuralNetwork.layers.add()
        layer2.name = 'custom2'
        layer2.input.append('temp1')
        layer2.output.append('temp2')
        layer2.custom.className = 'name2'

        layer3 = spec.neuralNetwork.layers.add()
        layer3.name = 'custom3'
        layer3.input.append('temp2')
        layer3.output.append('output')
        layer3.custom.className = 'name1'

        self.spec = spec

    def test_get_custom_names(self):
        names = _get_custom_layer_names(self.spec)
        self.assertEqual(names, {'name1', 'name2'})

    def test_change_custom_name(self):
        _replace_custom_layer_name(self.spec, 'name1', 'notname1')
        names = _get_custom_layer_names(self.spec)
        self.assertEqual(names, {'notname1', 'name2'})
        # set it back for future tests
        _replace_custom_layer_name(self.spec, 'notname1', 'name1')
