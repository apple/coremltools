import unittest
import tempfile
import numpy as np
import tensorflow as tf
import coremltools
import os
import shutil
from packaging import version
from test_utils import generate_data

_tf_version = tf.__version__


class DummyTest(unittest.TestCase):

    def test_blah(self):
        pass


@unittest.skipIf(version.parse(_tf_version).release[0] < 2, 'missing TensorFlow 2+.')
class TestKerasFashionMnist(unittest.TestCase):

    def setUp(self) -> None:
        self.input_shape = (1, 28, 28)
        self.saved_model_dir = tempfile.mkdtemp()
        _, self.model_file = tempfile.mkstemp(suffix='.h5', prefix=self.saved_model_dir)

    def tearDown(self) -> None:
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
        keras_model.save(self.model_file)
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_conversion_prediction(
            keras_model=keras_model,
            model_path=self.model_file,
            inputs={input_name: self.input_shape},
            outputs=['Identity'])

    def test_sequential_builder_saved_model_format(self):
        keras_model = self._build_model_sequential()
        # save model as SavedModel directory
        keras_model.save(self.saved_model_dir, save_format='tf')
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_conversion_prediction(
            keras_model=keras_model,
            model_path=self.saved_model_dir,
            inputs={input_name: self.input_shape},
            outputs=['Identity'])

    def test_functional_builder(self):
        keras_model = self._build_model_functional()
        # save model as Keras hdf5 .h5 model file
        keras_model.save(self.model_file)
        input_name = keras_model.inputs[0].name.split(':')[0]
        self._test_conversion_prediction(
            keras_model=keras_model,
            model_path=self.model_file,
            inputs={input_name: self.input_shape},
            outputs=['Identity'])


@unittest.skipIf(version.parse(_tf_version).release[0] < 2, 'missing TensorFlow 2+.')
class TestModelFormats(unittest.TestCase):

    def setUp(self) -> None:
        self.saved_model_dir = tempfile.mkdtemp()
        _, self.model_file = tempfile.mkstemp(suffix='.h5', prefix=self.saved_model_dir)

    def tearDown(self) -> None:
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
            target_ios='13')

        assert isinstance(model, coremltools.models.MLModel)


@unittest.skipIf(False, 'skipping slow full model conversion tests.')
@unittest.skipIf(version.parse(_tf_version).release[0] < 2, 'missing TensorFlow 2+.')
class TestKerasApplications(unittest.TestCase):

    def setUp(self) -> None:
        self.saved_model_dir = tempfile.mkdtemp()
        _, self.model_file = tempfile.mkstemp(suffix='.h5', prefix=self.saved_model_dir)

    def tearDown(self) -> None:
        if os.path.exists(self.saved_model_dir):
            shutil.rmtree(self.saved_model_dir)

    @staticmethod
    def _test_model(keras_model, model_path, inputs, outputs):
        # convert and validate
        model = coremltools.converters.tensorflow.convert(
            model_path,
            inputs=inputs,
            outputs=outputs,
            target_ios='13'
        )
        assert isinstance(model, coremltools.models.MLModel)

        # verify numeric correctness of predictions
        # assume one input one output for now
        name, shape = list(inputs.items())[0]
        data = generate_data(shape=shape)
        keras_prediction = keras_model.predict(data)
        prediction = model.predict({name: data})[outputs[0]]

        np.testing.assert_array_equal(keras_prediction.shape, prediction.shape)
        np.testing.assert_almost_equal(keras_prediction.flatten(), prediction.flatten(), decimal=4)

    def test_vgg16_keras_model(self):
        # load the tf.keras model
        keras_model = tf.keras.applications.VGG16(
            weights=None, input_shape=(32, 32, 3))
        # save model as Keras hdf5 .h5 model file
        keras_model.save(self.model_file)

        self._test_model(
            keras_model=keras_model,
            model_path=self.model_file,
            inputs={'input_1': (1, 32, 32, 3)},
            outputs=['Identity'])

    def test_vgg19_saved_model(self):
        # load the tf.keras model
        keras_model = tf.keras.applications.VGG19(
            weights=None, input_shape=(32, 32, 3))
        # save model as SavedModel directory
        keras_model.save(self.saved_model_dir, save_format='tf')

        self._test_model(
            keras_model=keras_model,
            model_path=self.saved_model_dir,
            inputs={'input_1': (1, 32, 32, 3)},
            outputs=['Identity'])


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestKerasApplications('test_vgg16_keras_model'))
    # unittest.TextTestRunner().run(suite)
