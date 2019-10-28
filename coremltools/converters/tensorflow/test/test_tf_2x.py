import unittest
import tempfile
import numpy as np
import tensorflow as tf
import coremltools
import os
import shutil
from test_utils import generate_data
from coremltools._deps import HAS_TF_2


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
            outputs=outputs,
            target_ios='13'
        )
        assert isinstance(model, coremltools.models.MLModel)

        # verify numeric correctness of predictions
        inputs = generate_data(shape=self.input_shape)
        keras_prediction = keras_model.predict(inputs)
        prediction = model.predict({keras_model.inputs[0].name.split(':')[0]: inputs})['Identity']
        np.testing.assert_array_equal(keras_prediction.shape, prediction.shape)
        np.testing.assert_almost_equal(keras_prediction.flatten(), prediction.flatten(), decimal=4)

    def test_sequential_builder_keras_model_format(self):
        keras_model = self._build_model_sequential()
        # save model as Keras hdf5 .h5 model file
        keras_model.save(self.model_path)
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_conversion_prediction(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: self.input_shape},
            outputs=['Identity']
        )

    def test_sequential_builder_saved_model_format(self):
        keras_model = self._build_model_sequential()
        # save model as SavedModel directory
        keras_model.save(self.saved_model_dir, save_format='tf')
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_conversion_prediction(
            keras_model=keras_model,
            model_path=self.saved_model_dir,
            inputs={input_name: self.input_shape},
            outputs=['Identity']
        )

    def test_functional_builder(self):
        keras_model = self._build_model_functional()
        # save model as Keras hdf5 .h5 model file
        keras_model.save(self.model_path)
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_conversion_prediction(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: self.input_shape},
            outputs=['Identity']
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
        prediction = core_ml_model.predict({input_name: inputs})['Identity']
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
            outputs=['Identity'],
            target_ios='13'
        )

        assert isinstance(model, coremltools.models.MLModel)

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
            outputs=['Identity'],
            target_ios='13'
        )

        assert isinstance(model, coremltools.models.MLModel)
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
        # convert and validate
        model = coremltools.converters.tensorflow.convert(
            self.saved_model_dir,
            inputs={input_name: (4, 4)},
            outputs=['Identity'],
            target_ios='13'
        )
        assert isinstance(model, coremltools.models.MLModel)
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
            outputs=outputs,
            target_ios='13'
        )
        assert isinstance(model, coremltools.models.MLModel)

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

    def test_vgg16_keras_model(self):
        # load the tf.keras model
        keras_model = tf.keras.applications.VGG16(
            weights=None, input_shape=(32, 32, 3))
        # save model as Keras hdf5 .h5 model file
        keras_model.save(self.model_path)
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=['Identity'])

    def test_vgg19_saved_model(self):
        # load the tf.keras model
        keras_model = tf.keras.applications.VGG19(
            weights=None, input_shape=(32, 32, 3))
        # save model as SavedModel directory
        keras_model.save(self.saved_model_dir, save_format='tf')
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_model(
            keras_model=keras_model,
            model_path=self.saved_model_dir,
            inputs={'input_1': (1, 32, 32, 3)},
            outputs=['Identity'])

    def test_densenet121(self):
        keras_model = tf.keras.applications.DenseNet121(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={'input_1': (1, 32, 32, 3)},
            outputs=['Identity'])

    def test_inception_resnet_v2(self):
        keras_model = tf.keras.applications.InceptionResNetV2(
            weights=None, input_shape=(75, 75, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 75, 75, 3)},
            outputs=['Identity'])

    def test_inception_v3(self):
        keras_model = tf.keras.applications.InceptionV3(
            weights=None, input_shape=(75, 75, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 75, 75, 3)},
            outputs=['Identity'])

    def test_mobilenet(self):
        keras_model = tf.keras.applications.MobileNet(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=['Identity'])

    def test_mobilenet_v2(self):
        keras_model = tf.keras.applications.MobileNetV2(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=['Identity'])

    @unittest.skip('shape mismatch')
    def test_nasnet_mobile(self):
        keras_model = tf.keras.applications.NASNetMobile(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=['Identity'], decimal=3)

    @unittest.skip('shape mismatch')
    def test_nasnet_large(self):
        keras_model = tf.keras.applications.NASNetLarge(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=['Identity'], decimal=3)

    def test_resnet50(self):
        keras_model = tf.keras.applications.ResNet50(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=['Identity'])

    def test_resnet50_v2(self):
        keras_model = tf.keras.applications.ResNet50V2(
            weights=None, input_shape=(32, 32, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 32, 32, 3)},
            outputs=['Identity'])

    def test_xception(self):
        keras_model = tf.keras.applications.Xception(
            weights=None, input_shape=(71, 71, 3))
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_model(
            keras_model=keras_model,
            model_path=self.model_path,
            inputs={input_name: (1, 71, 71, 3)},
            outputs=['Identity'])


if __name__ == '__main__':
    np.random.seed(1984)
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestKerasApplications('test_vgg16_keras_model'))
    # unittest.TextTestRunner().run(suite)
