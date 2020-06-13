from os import getcwd, chdir
from shutil import rmtree
from os.path import exists
from tempfile import mkdtemp
import pytest

from coremltools._deps import (
    HAS_TF_1,
    HAS_TF_2,
    HAS_TORCH,
    MSG_TF1_NOT_FOUND,
    MSG_TF2_NOT_FOUND,
    MSG_TORCH_NOT_FOUND,
)


###############################################################################
# Note: all tests are also used as examples such as in readme.md as a reference
# Whenever any of the following test fails, we should update API documentations
# Each test case is expected to be runnable and self-complete, then sync to the
# documentation pages as API example code snippet.
###############################################################################


@pytest.mark.skipif(not HAS_TF_1, reason=MSG_TF1_NOT_FOUND)
class TestTensorFlow1ConverterExamples:
    def setup_class(self):
        self._cwd = getcwd()
        self._temp_dir = mkdtemp()

        # step into temp directory as working directory
        # to make the user-facing examples cleaner
        chdir(self._temp_dir)

        # create toy models for conversion examples
        import tensorflow as tf

        # write a toy frozen graph
        # Note that we usually needs to run freeze_graph() on tf.Graph()
        # skipping here as this toy model does not contain any variables
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(3, 4, 5), name="input")
            tf.nn.relu(x, name="output")
        tf.io.write_graph(graph, "./", "./frozen_graph.pb", as_text=False)

        # write a toy SavedModel directory
        with tf.compat.v1.Session() as sess:
            x = tf.placeholder(shape=(1, 3, 5), dtype=tf.float32)
            y = tf.nn.relu(x)
            inputs = {"x": x}
            outputs = {"y": y}
        tf.compat.v1.saved_model.simple_save(sess, "./saved_model", inputs, outputs)

    def teardown_class(self):
        chdir(self._cwd)
        if exists(self._temp_dir):
            rmtree(self._temp_dir)

    @staticmethod
    def test_convert_from_frozen_graph_file():
        import coremltools as ct

        # The input `.pb` file is a frozen graph format that usually
        # generated by TensorFlow's utility function `freeze_graph()`
        mlmodel = ct.convert(
            "./frozen_graph.pb",
            inputs=[ct.TensorType("input", (3, 4, 5))],
            outputs=["output"],
        )
        mlmodel.save("./model.mlmodel")

        # `inputs` is optional when there's only 1 input, it is
        # required when the model contains multiple inputs.
        mlmodel = ct.convert("./frozen_graph.pb", outputs=["output"])
        mlmodel.save("./model.mlmodel")

    @staticmethod
    def test_convert_from_saved_model_dir():
        import coremltools as ct

        # SavedModel directory generated by TensorFlow 1.x
        # when converting from SavedModel dir, inputs / outputs are optional
        mlmodel = ct.convert("./saved_model")
        mlmodel.save("./model.mlmodel")


@pytest.mark.skipif(not HAS_TF_2, reason=MSG_TF2_NOT_FOUND)
class TestTensorFlow2ConverterExamples:
    def setup_class(self):
        self._cwd = getcwd()
        self._temp_dir = mkdtemp()
        # step into temp directory as working directory
        # to make the user-facing examples cleaner
        chdir(self._temp_dir)

        # create toy models for conversion examples
        import tensorflow as tf

        # write a toy tf.keras HDF5 model
        tf_keras_model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation=tf.nn.relu),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax),
            ]
        )
        tf_keras_model.save("./tf_keras_model.h5")

        # write a toy SavedModel directory
        tf_keras_model.save("./saved_model", save_format="tf")

    def teardown_class(self):
        chdir(self._cwd)
        if exists(self._temp_dir):
            rmtree(self._temp_dir)

    @staticmethod
    def test_convert_from_tf_keras_h5_file():
        import coremltools as ct

        mlmodel = ct.convert("./tf_keras_model.h5")
        mlmodel.save("./model.mlmodel")

    @staticmethod
    def test_convert_from_tf_keras_model():
        import tensorflow as tf
        import coremltools as ct

        tf_keras_model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation=tf.nn.relu),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax),
            ]
        )

        # Alternatively, we can just pass in `tf.keras.Model`
        mlmodel = ct.convert(tf_keras_model)
        mlmodel.save("./model.mlmodel")

    @staticmethod
    def test_convert_from_tf_keras_applications_model():
        import tensorflow as tf
        import coremltools as ct

        tf_keras_model = tf.keras.applications.MobileNet(
            weights="imagenet", input_shape=(224, 224, 3)
        )

        # inputs / outputs are optional, we can get from tf.keras model
        # this can be extremely helpful when we want to extract sub-graphs
        input_name = tf_keras_model.inputs[0].name.split(":")[0]
        # note that the `convert()` requires tf.Graph's outputs instead of
        # tf.keras.Model's outputs, to access that, we can do the following
        output_name = tf_keras_model.outputs[0].name.split(":")[0]
        tf_graph_output_name = output_name.split("/")[-1]

        mlmodel = ct.convert(
            tf_keras_model,
            inputs=[ct.TensorType(name=input_name, shape=(1, 224, 224, 3))],
            outputs=[tf_graph_output_name],
        )
        mlmodel.save("./mobilenet.mlmodel")

    @staticmethod
    def test_convert_from_saved_model_dir():
        import coremltools as ct

        # SavedModel directory generated by TensorFlow 2.x
        mlmodel = ct.convert("./saved_model")
        mlmodel.save("./model.mlmodel")


@pytest.mark.skipif(not HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
class TestPyTorchConverterExamples:
    def setup_class(self):
        self._cwd = getcwd()
        self._temp_dir = mkdtemp()
        # step into temp directory as working directory
        # to make the user-facing examples cleaner
        chdir(self._temp_dir)

    def teardown_class(self):
        chdir(self._cwd)
        if exists(self._temp_dir):
            rmtree(self._temp_dir)

    @staticmethod
    def test_convert_from_torch_vision_mobilenet_v2():
        import torch
        import torchvision
        import coremltools as ct

        """
        In this example, we'll instantiate a PyTorch classification model and convert
        it to Core ML.
        """

        """
        Here we instantiate our model. In a real use case this would be your trained
        model.
        """
        model = torchvision.models.mobilenet_v2()

        """
        The next thing we need to do is generate TorchScript for the model. The easiest
        way to do this is by tracing it.
        """

        """
        It's important that a model be in evaluation mode (not training mode) when it's
        traced. This makes sure things like dropout are disabled.
        """
        model.eval()

        """
        Tracing takes an example input and traces its flow through the model. Here we
        are creating an example image input.

        The rank and shape of the tensor will depend on your model use case. If your
        model expects a fixed size input, use that size here. If it can accept a
        variety of input sizes, it's generally best to keep the example input small to
        shorten how long it takes to run a forward pass of your model. In all cases,
        the rank of the tensor must be fixed.
        """
        example_input = torch.rand(1, 3, 256, 256)

        """
        Now we actually trace the model. This will produce the TorchScript that the
        CoreML converter needs.
        """
        traced_model = torch.jit.trace(model, example_input)

        """
        Now with a TorchScript representation of the model, we can call the CoreML
        converter. The converter also needs a description of the input to the model, 
        where we can give it a convenient name.
        """
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
        )

        """
        Now with a conversion complete, we can save the MLModel and run inference.
        """
        mlmodel.save("./mobilenet_v2.mlmodel")
        result = mlmodel.predict({"input": example_input.numpy()})
