import unittest

import numpy as np
import tempfile
import pytest
import shutil
import os

import coremltools
from coremltools._deps import HAS_KERAS_TF
from coremltools._deps import HAS_TF
from coremltools.models.utils import get_custom_layer_names, \
    replace_custom_layer_name, macos_version, is_macos
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


@unittest.skipIf(not HAS_TF, 'Missing TF. Skipping tests.')
class TFBasicConversionTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tmp_dir = tempfile.mkdtemp()
        _, self.graph_file = tempfile.mkstemp(suffix='.pb', prefix=self.tmp_dir)
        _, self.checkpoint_file = tempfile.mkstemp(suffix='.ckpt', prefix=self.tmp_dir)
        _, self.class_label_file = tempfile.mkstemp(suffix='.txt', prefix=self.tmp_dir)
        _, self.frozen_graph_file = tempfile.mkstemp(suffix='.pb', prefix=self.tmp_dir)
        _, self.converted_coreml_file = tempfile.mkstemp(suffix='.mlmodel', prefix=self.tmp_dir)

    @classmethod
    def tearDownClass(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

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

    def test_1(self):
        with open(self.class_label_file, 'w+') as labels_file:
            for a in range(10):
                labels_file.write(str(a + 1) + "\n")

        image_size = 224

        with tf.Graph().as_default():
            batch_size, height, width, channels = 1, image_size, image_size, 3
            images = tf.random.uniform([batch_size, height, width, channels], maxval=1)

            # Create the model.
            i_placeholder = tf.placeholder(name='input', dtype=tf.float32, shape=[1, image_size, image_size, 3])
            net = self.my_conv_2d(i_placeholder, [1, 3, 3, 1], 1, 1, 'first')
            net = tf.nn.avg_pool2d(net, 224, strides=1, padding='VALID', name='AvgPool_1a')
            net = self.my_conv_2d(net, [1, 1, 1, 10], 10, 1, 'fc', activation_fn=None, with_bias_add=False)
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            probabilities = tf.nn.softmax(net, name='Softmax')

            saver = tf.train.Saver()
            init_op = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init_op)
                probabilities = sess.run(probabilities, {i_placeholder: images.eval()})
                save_path = saver.save(sess, self.checkpoint_file)

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

    def test_classifier_with_label_file(self):
        coremltools.converters.tensorflow.convert(
            self.frozen_graph_file,
            mlmodel_path=self.converted_coreml_file,
            input_name_shape_dict={'input': [1, 224, 224, 3]},
            image_input_names=['input'],
            output_feature_names=['Softmax'],
            predicted_feature_name='classLabel',
            class_labels=self.class_label_file)

    def test_classifier_with_int_label_list(self):
        coremltools.converters.tensorflow.convert(
            self.frozen_graph_file,
            mlmodel_path=self.converted_coreml_file,
            input_name_shape_dict={'input': [1, 224, 224, 3]},
            image_input_names=['input'],
            output_feature_names=['Softmax'],
            predicted_feature_name='classLabel',
            class_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_classifier_with_string_label_list(self):
        coremltools.converters.tensorflow.convert(
            self.frozen_graph_file,
            mlmodel_path=self.converted_coreml_file,
            input_name_shape_dict={'input': [1, 224, 224, 3]},
            image_input_names=['input'],
            output_feature_names=['Softmax'],
            predicted_feature_name='classLabel',
            class_labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

    def test_classifier_without_class_labels(self):
        coremltools.converters.tensorflow.convert(
            self.frozen_graph_file,
            mlmodel_path=self.converted_coreml_file,
            input_name_shape_dict={'input': [1, 224, 224, 3]},
            image_input_names=['input'],
            output_feature_names=['Softmax'])

    def test_classifier_nhwc(self):
        # Test manually specifying the image format. The converter would have
        # detected NHWC.
        mlmodel = coremltools.converters.tensorflow.convert(
            self.frozen_graph_file,
            mlmodel_path=self.converted_coreml_file,
            input_name_shape_dict={'input': [1, 224, 224, 3]},
            image_input_names=['input'],
            output_feature_names=['Softmax'],
            image_format='NHWC')
        spec = mlmodel.get_spec()
        self.assertEqual(spec.neuralNetwork.layers[0].name, "input_to_nhwc",)

    def test_classifier_nchw(self):
        # Expect failure - input dimensions are incompatible with NCHW
        with self.assertRaises(ValueError) as e:
            coremltools.converters.tensorflow.convert(
                self.frozen_graph_file,
                mlmodel_path=self.converted_coreml_file,
                input_name_shape_dict={'input': [1, 224, 224, 3]},
                image_input_names=['input'],
                output_feature_names=['Softmax'],
                image_format='NCHW')

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
        names = get_custom_layer_names(self.spec)
        self.assertEqual(names, {'name1', 'name2'})

    def test_change_custom_name(self):
        replace_custom_layer_name(self.spec, 'name1', 'notname1')
        names = get_custom_layer_names(self.spec)
        self.assertEqual(names, {'notname1', 'name2'})
        # set it back for future tests
        replace_custom_layer_name(self.spec, 'notname1', 'name1')
