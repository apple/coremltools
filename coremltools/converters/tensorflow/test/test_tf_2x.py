import unittest
import tempfile
import numpy as np
import tensorflow as tf
import coremltools
import os
import shutil
from test_utils import generate_data
from coremltools._deps import HAS_TF_2
import math
import pytest


@unittest.skipUnless(HAS_TF_2, 'missing TensorFlow 2+.')
class TestSingleOp(unittest.TestCase):
    # In this class we test tensorflow 2.x op without using Keras API

    def setUp(self):
        self.saved_model_dir = tempfile.mkdtemp()

    def _test_coreml(self, model, input_dic=None, output_names=None):

        # Get concrete function
        tf.saved_model.save(model, self.saved_model_dir)
        tf_model = tf.saved_model.load(self.saved_model_dir)
        concrete_func = tf_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        # Get function input
        inputs = []
        if input_dic == None:
            for input in concrete_func.inputs:
                name = input.name.split(':')[0]
                shape = input.shape
                if shape == None or any([x is None for x in input.shape.as_list()]):
                    raise ValueError("Please specify 'input_dic' for dynamic shape input.")
                shape = input.shape.as_list()
                inputs.append((name, np.random.rand(*shape).astype(np.float32), shape))
        else:
            if not isinstance(input_dic, list):
                raise TypeError("'input_dic' should be [(str, tensor)] type.")
            inputs = input_dic

        # Get output names
        if output_names == None:
            output_names = [output.name.split(':')[0] for output in concrete_func.outputs]

        # Tensorflow predict
        tf_inputs = [tf.convert_to_tensor(value) for name, value, shape in inputs]
        tf_outputs = tf_model(*tf_inputs)

        # Coreml model predict
        coreml_inputs = {name: shape for name, value, shape in inputs}
        model = coremltools.converters.tensorflow.convert(
            [concrete_func],
            inputs=coreml_inputs,
            outputs=output_names
        )
        coreml_outputs = model.predict({name: value for name, value, shape in inputs})

        # Compare Tensorflow and Coreml
        if not isinstance(tf_outputs, tuple):
            tf_outputs = tuple([tf_outputs])
        self.assertTrue(len(tf_outputs), len(coreml_outputs))
        self.assertTrue(len(tf_outputs), len(output_names))
        for i, output_name in enumerate(output_names):
            np.testing.assert_almost_equal(tf_outputs[i].numpy(), coreml_outputs[output_name], decimal=3)

    def test_single_output_example(self):

        class model(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec(shape=[3,3], dtype=tf.float32),
                                          tf.TensorSpec(shape=[3,3], dtype=tf.float32)])
            def __call__(self, x, y):
                return x+y
        self._test_coreml(model())

    def test_multiple_outputs_example(self):

        class model(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec(shape=[3,3], dtype=tf.float32),
                                          tf.TensorSpec(shape=[3,3], dtype=tf.float32)])
            def __call__(self, x, y):
                return x+y, x-y, x*y
        self._test_coreml(model())


@unittest.skipUnless(HAS_TF_2, 'missing TensorFlow 2+.')
class TestKerasFashionMnist(unittest.TestCase):

    def setUp(self):
        self.input_shape = (1, 28, 28)
        self.saved_model_dir = tempfile.mkdtemp()
        _, self.model_path = tempfile.mkstemp(suffix='.h5', prefix=self.saved_model_dir)

    def tearDown(self):
        if os.path.exists(self.saved_model_dir):
            shutil.rmtree(self.saved_model_dir)

    @staticmethod
    def _build_model_sequential():
        keras_model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        return keras_model

    @staticmethod
    def _build_model_functional():
        inputs = tf.keras.Input(shape=(28, 28), name='data')
        x = tf.keras.layers.Flatten(input_shape=(28, 28))(inputs)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return keras_model

    def _test_conversion_prediction(self, keras_model, model_path, inputs, outputs):
        # convert and validate
        model = coremltools.converters.tensorflow.convert(
            model_path,
            inputs=inputs,
            outputs=outputs
        )
        self.assertTrue(isinstance(model, coremltools.models.MLModel))

        # verify numeric correctness of predictions
        inputs = generate_data(shape=self.input_shape)
        keras_prediction = keras_model.predict(inputs)
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        prediction = model.predict({keras_model.inputs[0].name.split(':')[0]: inputs})[output_name]
        np.testing.assert_array_equal(keras_prediction.shape, prediction.shape)
        np.testing.assert_almost_equal(keras_prediction.flatten(), prediction.flatten(), decimal=4)

    def test_sequential_builder_keras_model_format(self):
        keras_model = self._build_model_sequential()
        # save model as Keras hdf5 .h5 model file
        keras_model.save(self.model_path)
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]

        self._test_conversion_prediction(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: self.input_shape},
            outputs=[output_name]
        )

    def test_sequential_builder_saved_model_format(self):
        keras_model = self._build_model_sequential()
        # save model as SavedModel directory
        keras_model.save(self.saved_model_dir, save_format='tf')
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_conversion_prediction(
            keras_model=keras_model,
            model_path=self.saved_model_dir,
            inputs={input_name: self.input_shape},
            outputs=[output_name]
        )

    def test_functional_builder(self):
        keras_model = self._build_model_functional()
        # save model as Keras hdf5 .h5 model file
        keras_model.save(self.model_path)
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_conversion_prediction(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: self.input_shape},
            outputs=[output_name]
        )


@unittest.skipUnless(HAS_TF_2, 'missing TensorFlow 2+.')
class TestModelFormats(unittest.TestCase):

    def setUp(self):
        self.saved_model_dir = tempfile.mkdtemp()
        _, self.model_path = tempfile.mkstemp(suffix='.h5', prefix=self.saved_model_dir)

    def tearDown(self):
        if os.path.exists(self.saved_model_dir):
            shutil.rmtree(self.saved_model_dir)

    @staticmethod
    def _test_prediction(keras_model, core_ml_model, inputs, decimal=4):
        keras_model.predict(inputs)
        keras_prediction = keras_model.predict(inputs)
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        prediction = core_ml_model.predict({input_name: inputs})[output_name]
        np.testing.assert_array_equal(keras_prediction.shape, prediction.shape)
        np.testing.assert_almost_equal(keras_prediction.flatten(), prediction.flatten(), decimal=decimal)

    def test_concrete_function(self):
        # construct a toy model
        root = tf.train.Checkpoint()
        root.v1 = tf.Variable(3.)
        root.v2 = tf.Variable(2.)
        root.f = tf.function(lambda x: root.v1 * root.v2 * x)

        # save the model
        input_data = tf.constant(1., shape=[1, 1])
        to_save = root.f.get_concrete_function(input_data)
        tf.saved_model.save(root, self.saved_model_dir, to_save)

        tf_model = tf.saved_model.load(self.saved_model_dir)
        concrete_func = tf_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        model = coremltools.converters.tensorflow.convert(
            [concrete_func],
            inputs={'x': (1, 1)},
            outputs=['Identity']
        )

        self.assertTrue(isinstance(model, coremltools.models.MLModel))

    def test_control_flow(self):
        @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
        def control_flow(x):
            if x <= 0:
                return 0.
            else:
                return x * 3.

        to_save = tf.Module()
        to_save.control_flow = control_flow

        saved_model_dir = tempfile.mkdtemp()
        tf.saved_model.save(to_save, saved_model_dir)
        tf_model = tf.saved_model.load(saved_model_dir)
        concrete_func = tf_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        model = coremltools.converters.tensorflow.convert(
            [concrete_func],
            inputs={'x': (1,)},
            outputs=['Identity']
        )

        self.assertTrue(isinstance(model, coremltools.models.MLModel))
        input_data = generate_data(shape=[20])
        for data in input_data:
            tf_prediction = to_save.control_flow(data).numpy().flatten()
            cm_prediction = model.predict({'x': np.array([data])})['Identity'].flatten()
            np.testing.assert_array_almost_equal(tf_prediction, cm_prediction, decimal=2)

    def test_subclassed_keras_model(self):
        class MyModel(tf.keras.Model):
            def __init__(self):
                super(MyModel, self).__init__()
                self.dense1 = tf.keras.layers.Dense(4)
                self.dense2 = tf.keras.layers.Dense(5)

            @tf.function
            def call(self, input_data):
                return self.dense2(self.dense1(input_data))

        keras_model = MyModel()
        inputs = generate_data(shape=(4, 4))

        # subclassed model can only be saved as SavedModel format
        keras_model._set_inputs(inputs)
        keras_model.save(self.saved_model_dir, save_format='tf')
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        # convert and validate
        model = coremltools.converters.tensorflow.convert(
            self.saved_model_dir,
            inputs={input_name: (4, 4)},
            outputs=[output_name])
        self.assertTrue(isinstance(model, coremltools.models.MLModel))
        self._test_prediction(keras_model=keras_model, core_ml_model=model, inputs=inputs)


@unittest.skipIf(False, 'skipping slow full model conversion tests.')
@unittest.skipUnless(HAS_TF_2, 'missing TensorFlow 2+.')
class TestKerasApplications(unittest.TestCase):

    def setUp(self):
        self.saved_model_dir = tempfile.mkdtemp()
        _, self.model_path = tempfile.mkstemp(suffix='.h5', prefix=self.saved_model_dir)

    def tearDown(self):
        if os.path.exists(self.saved_model_dir):
            shutil.rmtree(self.saved_model_dir)

    def _predict_keras_intermediate_layer(self, data, layer_name):
        """
        Helper function to print intermediate layer for debugging.
        """
        partial_keras_model = tf.keras.models.Model(
            inputs=self.keras_model.input,
            outputs=self.keras_model.get_layer(layer_name).output)
        return partial_keras_model.predict(data)

    def _test_model(self, keras_model, model_path, inputs, outputs, decimal=4, verbose=False):
        keras_model.save(model_path)

        # convert and validate
        model = coremltools.converters.tensorflow.convert(
            model_path,
            inputs=inputs,
            outputs=outputs
        )
        self.assertTrue(isinstance(model, coremltools.models.MLModel))

        if verbose:
            print('TensorFlow Keras model saved at {}'.format(model_path))
            tmp_model_path = self.model_path.rsplit('.')[0] + '.mlmodel'
            model.save(tmp_model_path)
            print('Core ML model saved at {}'.format(tmp_model_path))

        # verify numeric correctness of predictions
        # assume one input one output for now
        name, shape = list(inputs.items())[0]
        data = generate_data(shape=shape)

        # self._predict_keras_intermediate_layer(data, 'conv1')
        keras_prediction = keras_model.predict(data)
        prediction = model.predict({name: data})[outputs[0]]

        if verbose:
            print('Shape Keras:', keras_prediction.shape, ' vs. Core ML:', prediction.shape)
            print('Input  :', data.flatten()[:16])
            print('Keras  :', keras_prediction.flatten()[:16])
            print('Core ML:', prediction.flatten()[:16])

        np.testing.assert_array_equal(
            keras_prediction.shape, prediction.shape)
        np.testing.assert_almost_equal(
            keras_prediction.flatten(), prediction.flatten(), decimal=decimal)

    @pytest.mark.slow
    def test_vgg16_keras_model(self):
        # load the tf.keras model
        keras_model = tf.keras.applications.VGG16(
            weights=None, input_shape=(32, 32, 3))
        # save model as Keras hdf5 .h5 model file
        keras_model.save(self.model_path)
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=[output_name])

    @pytest.mark.slow
    def test_vgg19_saved_model(self):
        # load the tf.keras model
        keras_model = tf.keras.applications.VGG19(
            weights=None, input_shape=(32, 32, 3))
        # save model as SavedModel directory
        keras_model.save(self.saved_model_dir, save_format='tf')
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_model(
            keras_model=keras_model,
            model_path=self.saved_model_dir,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=[output_name])

    @pytest.mark.slow
    def test_densenet121(self):
        keras_model = tf.keras.applications.DenseNet121(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=[output_name])

    @pytest.mark.slow
    def test_inception_resnet_v2(self):
        keras_model = tf.keras.applications.InceptionResNetV2(
            weights=None, input_shape=(75, 75, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 75, 75, 3)},
            outputs=[output_name])

    @pytest.mark.slow
    def test_inception_v3(self):
        keras_model = tf.keras.applications.InceptionV3(
            weights=None, input_shape=(75, 75, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 75, 75, 3)},
            outputs=[output_name])

    def test_mobilenet(self):
        keras_model = tf.keras.applications.MobileNet(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=[output_name])

    def test_mobilenet_v2(self):
        keras_model = tf.keras.applications.MobileNetV2(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=[output_name])

    @unittest.skip('shape mismatch')
    @pytest.mark.slow
    def test_nasnet_mobile(self):
        keras_model = tf.keras.applications.NASNetMobile(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=[output_name], decimal=3)

    @unittest.skip('shape mismatch')
    @pytest.mark.slow
    def test_nasnet_large(self):
        keras_model = tf.keras.applications.NASNetLarge(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=[output_name], decimal=3)

    @pytest.mark.slow
    def test_resnet50(self):
        keras_model = tf.keras.applications.ResNet50(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=[output_name])

    @pytest.mark.slow
    def test_resnet50_v2(self):
        keras_model = tf.keras.applications.ResNet50V2(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=[output_name])

    @pytest.mark.slow
    def test_xception(self):
        keras_model = tf.keras.applications.Xception(
            weights=None, input_shape=(71, 71, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 71, 71, 3)},
            outputs=[output_name])


@unittest.skipUnless(HAS_TF_2, 'missing TensorFlow 2+.')
class TestCornerCases(unittest.TestCase):

    def setUp(self):
        self.saved_model_dir = tempfile.mkdtemp()
        _, self.model_path = tempfile.mkstemp(suffix='.h5', prefix=self.saved_model_dir)

    def tearDown(self):
        if os.path.exists(self.saved_model_dir):
            shutil.rmtree(self.saved_model_dir)

    def _test_model(
        self,
        keras_model,
        model_path,
        inputs,
        outputs=None,
        decimal=4,
        use_cpu_only=False,
        verbose=False
    ):
        keras_model.save(model_path)

        # convert and validate
        model = coremltools.converters.tensorflow.convert(
            model_path,
            inputs=inputs,
            outputs=outputs
        )
        self.assertTrue(isinstance(model, coremltools.models.MLModel))

        if verbose:
            print('TensorFlow Keras model saved at {}'.format(model_path))
            tmp_model_path = self.model_path.rsplit('.')[0] + '.mlmodel'
            model.save(tmp_model_path)
            print('Core ML model saved at {}'.format(tmp_model_path))

        # verify numeric correctness of predictions
        # assume one input one output for now
        name, shape = list(inputs.items())[0]
        data = generate_data(shape=shape)

        keras_prediction = keras_model.predict(data)

        # If outputs are not supplied, get the output name
        # from the keras model.
        if not outputs:
            output_name = keras_model.outputs[0].name
            outputs = [output_name.split('/')[1].split(':')[0]]

        prediction = model.predict({name: data}, use_cpu_only=use_cpu_only)[outputs[0]]

        if verbose:
            print('Shape Keras:', keras_prediction.shape, ' vs. Core ML:', prediction.shape)
            print('Input  :', data.flatten()[:16])
            print('Keras  :', keras_prediction.flatten()[:16])
            print('Core ML:', prediction.flatten()[:16])

        np.testing.assert_array_equal(
            keras_prediction.shape, prediction.shape)
        np.testing.assert_almost_equal(
            keras_prediction.flatten(), prediction.flatten(), decimal=decimal)

        return model

    def test_output_identity_node_removal(self):
        inpt = tf.keras.layers.Input(shape=[32, 32, 3], batch_size=1)
        out = tf.keras.layers.SeparableConv2D(
            filters=5,
            kernel_size=(3, 3),
        )(inpt)
        out = tf.keras.layers.Conv2D(
            filters=5,
            kernel_size=1,
        )(out)
        keras_model = tf.keras.Model(inpt, out)
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_model(keras_model=keras_model,
                         model_path=self.model_path,
                         inputs={input_name: (1, 32, 32, 3)},
                         decimal=2)

    def test_batch_norm_node_fusion(self):
        x = tf.keras.layers.Input(shape=[32, 32, 3], batch_size=1)
        conv = tf.keras.layers.Conv2D(filters=3, kernel_size=1)(x)
        bn = tf.keras.layers.BatchNormalization(axis=-1)(conv)
        out = tf.keras.layers.Activation('relu')(bn)
        keras_model = tf.keras.Model(x, out)
        input_name = keras_model.inputs[0].name.split(':')[0]
        model = self._test_model(keras_model=keras_model,
                                 model_path=self.model_path,
                                 inputs={input_name: (1, 32, 32, 3)})
        num_batch_norm = 0
        for layer in model.get_spec().neuralNetwork.layers:
            if layer.WhichOneof('layer') == 'batchnorm':
                num_batch_norm += 1
        self.assertEqual(num_batch_norm, 1)

    def test_conv_bias_fusion(self):
        x = tf.keras.layers.Input(shape=[32, 32, 3], batch_size=1)
        conv = tf.keras.layers.Conv2D(filters=3, kernel_size=1)(x)
        conv = tf.keras.layers.DepthwiseConv2D(kernel_size=1)(conv)
        keras_model = tf.keras.Model(x, conv)
        input_name = keras_model.inputs[0].name.split(':')[0]
        model = self._test_model(keras_model=keras_model,
                                 model_path=self.model_path,
                                 decimal=3,
                                 inputs={input_name: (1, 32, 32, 3)})
        add_broadcastables = 0
        load_constants = 0
        for layer in model.get_spec().neuralNetwork.layers:
            if layer.WhichOneof('layer') == 'addBroadcastable':
                add_broadcastables += 1
            if layer.WhichOneof('layer') == 'loadConstantND':
                load_constants += 1

        self.assertEqual(add_broadcastables, 0)
        self.assertEqual(load_constants, 0)

    def test_conv2d_with_activation(self):
        inputs = tf.keras.layers.Input(shape=[256, 256, 3], batch_size=1)
        out = tf.keras.layers.Conv2D(
            filters=5,
            kernel_size=1,
            padding='same',
            activation='softmax')(inputs)
        keras_model = tf.keras.Model(inputs, out)
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        self._test_model(keras_model=keras_model,
                         model_path=self.model_path,
                         inputs={input_name: (1, 256, 256, 3)},
                         outputs=[output_name])

    def test_extra_transposes_1(self):
        # this model generates an extra transpose layer
        keras_model = tf.keras.Sequential()
        keras_model.add(tf.keras.layers.Reshape((75, 6), input_shape=(6 * 75,)))
        keras_model.add(tf.keras.layers.Dense(100, activation='relu'))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        model = self._test_model(keras_model=keras_model,
                                 model_path=self.model_path,
                                 inputs={input_name: (1, 6 * 75)},
                                 outputs=[output_name], verbose=True)
        num_reshapes = 0
        num_transposes = 0
        for layer in model.get_spec().neuralNetwork.layers:
            if layer.WhichOneof('layer') == 'reshapeStatic':
                num_reshapes += 1
            if layer.WhichOneof('layer') == 'transpose':
                num_transposes += 1
        self.assertEqual(num_reshapes, 2)
        self.assertEqual(num_transposes, 0)

    def test_extra_transposes_2(self):
        keras_model = tf.keras.Sequential()
        keras_model.add(tf.keras.layers.Reshape((75, 6, 1), input_shape=(6 * 75,)))
        keras_model.add(tf.keras.layers.Permute((2, 3, 1)))
        keras_model.add(tf.keras.layers.Permute((2, 3, 1)))
        # inserting several unnecessary extra transpose layers
        keras_model.add(tf.keras.layers.Permute((1, 2, 3)))
        keras_model.add(tf.keras.layers.Permute((1, 2, 3)))
        keras_model.add(tf.keras.layers.Permute((1, 2, 3)))
        keras_model.add(tf.keras.layers.Activation(tf.nn.relu))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        model = self._test_model(keras_model=keras_model,
                                 model_path=self.model_path,
                                 inputs={input_name: (1, 6 * 75)},
                                 outputs=[output_name])
        num_transposes = 0
        for layer in model.get_spec().neuralNetwork.layers:
            if layer.WhichOneof('layer') == 'transpose':
                num_transposes += 1
        self.assertEqual(num_transposes, 2)

    def test_extra_reshapes(self):
        keras_model = tf.keras.Sequential()
        # inserting several unnecessary extra reshape layers
        keras_model.add(tf.keras.layers.Reshape((1, 75, 6, 1), input_shape=(6 * 75,)))
        keras_model.add(tf.keras.layers.Reshape((75, 6, 1)))
        keras_model.add(tf.keras.layers.Reshape((75, 1, 6, 1)))
        keras_model.add(tf.keras.layers.Reshape((75, 6, 1)))
        keras_model.add(tf.keras.layers.Reshape((75, 1, 6, 1)))
        keras_model.add(tf.keras.layers.Activation(tf.nn.relu))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        model = self._test_model(keras_model=keras_model,
                                 model_path=self.model_path,
                                 inputs={input_name: (1, 6 * 75)},
                                 outputs=[output_name])

        num_reshapes = 0
        for layer in model.get_spec().neuralNetwork.layers:
            if layer.WhichOneof('layer') == 'reshapeStatic':
                num_reshapes += 1
        self.assertEqual(num_reshapes, 1)

    def test_gelu_tanh_approx_fusion(self):

        @tf.function(input_signature=[tf.TensorSpec(shape=(6,), dtype=tf.float32)])
        def gelu_tanh(x):
            y = 0.5 * (1.0 + tf.tanh((math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
            return x * y

        conc_func = gelu_tanh.get_concrete_function()
        mlmodel = coremltools.converters.tensorflow.convert(
            [conc_func],
            inputs={conc_func.inputs[0].name[:-2]: conc_func.inputs[0].shape},
            outputs=[conc_func.outputs[0].name[:-2]]
        )

        spec = mlmodel.get_spec()
        nn_spec = spec.neuralNetwork
        number_gelu_layers = 0
        for layer in nn_spec.layers:
            if layer.WhichOneof('layer') == 'gelu':
                number_gelu_layers += 1
        self.assertEqual(number_gelu_layers, 1)

    def disable_test_layer_norm_fusion(self):
        keras_model = tf.keras.Sequential()
        keras_model.add(tf.keras.layers.LayerNormalization(axis=-1, input_shape=(3, 4, 5)))
        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        model = self._test_model(keras_model=keras_model,
                                 model_path=self.model_path,
                                 inputs={input_name: (3, 4, 5)},
                                 outputs=[output_name])

    def test_wrong_out_name_error(self):

        @tf.function(input_signature=[tf.TensorSpec(shape=(1,), dtype=tf.float32)])
        def sin(x):
            y = tf.sin(x)
            return y

        conc_func = sin.get_concrete_function()
        with self.assertRaises(Exception) as cm:
            coremltools.converters.tensorflow.convert(
                [conc_func],
                inputs={conc_func.inputs[0].name[:-2]: conc_func.inputs[0].shape},
                outputs=['output_not_present'])

        the_exception = str(cm.exception)
        self.assertTrue("is not an output node in the source graph" in the_exception)

    def test_softplus(self):
        keras_model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='softplus')
        ])

        input_name = keras_model.inputs[0].name.split(':')[0]
        output_name = keras_model.outputs[0].name.split(':')[0].split('/')[-1]
        model = self._test_model(keras_model=keras_model,
                                 model_path=self.model_path,
                                 inputs={input_name: (1, 28, 28)},
                                 outputs=[output_name], decimal=3)

    def test_redundant_transpose(self):
        H = 224
        W = 224
        C = 3
        inputs = tf.keras.layers.Input(shape=(H, W, C), batch_size=1)
        out = tf.keras.layers.Conv2D(
            filters=4,
            kernel_size=3,
        )(inputs)
        model = tf.keras.Model(inputs, out)
        input_name = model.inputs[0].name.split(":")[0]
        input_shape = (1, H, W, C)
        output_name = model.outputs[0].name.split(':')[0].split('/')[-1]

        model.save(self.model_path, include_optimizer=False, save_format="h5")

        mlmodel = coremltools.converters.tensorflow.convert(
            self.model_path,
            inputs={input_name: input_shape},
            image_input_names=input_name,
            outputs=[output_name],
        )

        spec = mlmodel.get_spec()
        output_types = [layer.WhichOneof('layer') for layer in spec.neuralNetwork.layers]
        expected_types = ['convolution', 'transpose']
        np.testing.assert_array_equal(output_types, expected_types)


if __name__ == '__main__':
    np.random.seed(1984)
    RUN_ALL_TESTS = True
    if RUN_ALL_TESTS:
        unittest.main()
    else:
        suite = unittest.TestSuite()
        suite.addTest(TestCornerCases('test_wrong_out_name_error'))
        unittest.TextTestRunner().run(suite)
