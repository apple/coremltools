# Supported Source Formats

Formats supported by the [Unified Conversion API](convert-learning-models) include the following:

## TensorFlow versions 1.x Formats

- Frozen [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph)
- Frozen graph (`.pb`) file path
- [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras)
- [HDF5 file path](https://keras.io/api/models/model_saving_apis/) (`.h5`)
- [SavedModel](https://www.tensorflow.org/guide/saved_model) directory path

## TensorFlow versions 2.x Formats

- [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras)
- [HDF5 file path](https://keras.io/api/models/model_saving_apis/) (`.h5`)
- [SavedModel](https://www.tensorflow.org/guide/saved_model) directory path
- [Concrete function](https://www.tensorflow.org/guide/concrete_function).

## PyTorch Formats

- [TorchScript](https://pytorch.org/docs/stable/jit.html) object
- TorchScript object saved as a `.pt` file


