import itertools
import unittest
import numpy as np
import os, shutil
import tempfile
from nose.tools import raises
from nose.plugins.attrib import attr

from coremltools._deps import HAS_KERAS2_TF


if HAS_KERAS2_TF:
    import keras.backend
    from keras.models import Sequential, Model
    from keras.layers import Dense, Activation, Conv2D, Conv1D, Flatten, BatchNormalization, Conv2DTranspose
    from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
    from keras.layers import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D 
    from keras.layers import Embedding, Input, Permute, Reshape, RepeatVector, Dropout
    from keras.layers import Add, Multiply, Concatenate, Dot, Maximum, Average
    from keras.layers import add, multiply, concatenate, dot, maximum, average
    from keras.layers import ZeroPadding2D, UpSampling2D, Cropping2D
    from keras.layers import ZeroPadding1D, UpSampling1D, Cropping1D
    from keras.layers import SimpleRNN, LSTM, GRU
    from keras.layers.wrappers import Bidirectional, TimeDistributed
    from coremltools.converters import keras as kerasConverter


def _keras_transpose(x, is_sequence=False):
    if len(x.shape) == 4:
        # Keras input shape = [Batch, Height, Width, Channels]
        x = np.transpose(x, [0,3,1,2])
        return np.expand_dims(x, axis=0)
    elif len(x.shape) == 3:
        # Keras input shape = [Batch, (Sequence) Length, Channels]
        return np.transpose(x, [1,0,2])
    elif len(x.shape) == 2:
        if is_sequence:  # (N,S) --> (S,N,1,)
            return x.reshape(x.shape[::-1] + (1,))
        else:  # (N,C) --> (N,C,1,1)
            return x.reshape((1, ) + x.shape) # Dense
    elif len(x.shape) == 1:
        if is_sequence: # (S) --> (S,N,1,1,1)
            return x.reshape((x.shape[0], 1, 1))
        else: 
            return x
    else:
        return x

def _get_coreml_model(model, model_path, input_names, output_names):
    """
    Get the coreml model from the Keras model.
    """
    # Convert the model
    from coremltools.converters import keras as keras_converter
    model = keras_converter.convert(model, input_names, output_names)
    return model


def _generate_data(input_shape, mode = 'random'):
    """
    Generate some random data according to a shape.
    """
    if mode == 'zeros':
        X = np.zeros(input_shape)
    elif mode == 'ones':
        X = np.ones(input_shape)
    elif mode == 'linear':
        X = np.array(range(np.product(input_shape))).reshape(input_shape)
    elif mode == 'random':
        X = np.random.rand(*input_shape)
    elif mode == 'random_zero_mean':
        X = np.random.rand(*input_shape)-0.5
    return X

@unittest.skipIf(not HAS_KERAS2_TF, 'Missing keras. Skipping tests.')
@attr('keras2')
class KerasNumericCorrectnessTest(unittest.TestCase):
    """
    Unit test class for testing the Keras converter.
    """
    def _test_keras_model(self, model, num_samples = 1, mode = 'random',
            input_blob = 'data', output_blob = 'output', delta = 1e-2,
            model_dir = None, transpose_keras_result = True, 
            one_dim_seq_flags = None ):
        # transpose_keras_result: if true, compare the transposed Keras result
        # one_dim_seq_flags: a list of same length as the number of inputs in 
        # the model; if None, treat all 1D input (if any) as non-sequence
        # if one_dim_seq_flags[i] is True, it means the ith input, with shape
        # (X,) is in fact a sequence of length X. 
        
        # Get the CoreML model
        use_tmp_folder = False
        if model_dir is None:
            use_tmp_folder = True
            model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, 'keras.mlmodel')

        # Generate data
        nb_inputs = len(model.inputs)
        if nb_inputs > 1:
            input_names = []; input_data = []; coreml_input = {}
            for i in range(nb_inputs):
                input_shape = [1 if a is None else a for a in \
                        model.input_shape[i]]
                X = _generate_data(input_shape, mode)
                feature_name = "data_%s" % i
                input_names.append(feature_name)
                input_data.append(X)
                if one_dim_seq_flags is None: 
                    coreml_input[feature_name] = _keras_transpose(X
                            ).astype('f').copy()
                else: 
                    coreml_input[feature_name] = _keras_transpose(X, 
                            one_dim_seq_flags[i]).astype('f').copy()
        else:
            input_shape = [1 if a is None else a for a in model.input_shape]
            input_names = ['data']
            input_data = _generate_data(input_shape, mode)
            if one_dim_seq_flags is None: 
                coreml_input = {'data': _keras_transpose(input_data).astype(
                        'f').copy()}
            else: 
                coreml_input = {'data': _keras_transpose(input_data, 
                        one_dim_seq_flags[0]).astype('f').copy()}

        # Compile the model
        output_names = ['output'+str(i) for i in xrange(len(model.outputs))]
        coreml_model = _get_coreml_model(model, model_path, input_names, 
                output_names)
        
        # Assuming coreml model output names are in the same order as Keras 
        # Output list, put predictions into a list, sorted by output name
        coreml_preds = coreml_model.predict(coreml_input)
        c_preds = [coreml_preds[name] for name in output_names]

        # Cleanup files - models on disk no longer useful
        if use_tmp_folder and os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        
        # Run Keras predictions
        keras_preds = model.predict(input_data)
        k_preds = keras_preds if type(keras_preds) is list else [keras_preds]
        
        # Compare each output blob
        for idx, k_pred in enumerate(k_preds):
            if transpose_keras_result:
                kp = _keras_transpose(k_pred).flatten()
            else:
                kp = k_pred.flatten()
            cp = c_preds[idx].flatten()
            # Compare predictions
            self.assertEquals(len(kp), len(cp))
            for i in xrange(len(kp)):
                max_den = max(1.0, kp[i], cp[i])
                self.assertAlmostEquals(kp[i]/max_den, 
                                        cp[i]/max_den, 
                                        delta=delta)


@unittest.skipIf(not HAS_KERAS2_TF, 'Missing keras. Skipping tests.')
@attr('keras2')
class KerasBasicNumericCorrectnessTest(KerasNumericCorrectnessTest):

    def test_tiny_inner_product(self):
        np.random.seed(1988)

        # Define a model
        model = Sequential()
        model.add(Dense(2, input_shape=(2,)))

        # Test all zeros
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_keras_model(model, mode = 'zeros')

        # Test all ones
        model.set_weights([np.ones(w.shape) for w in model.get_weights()])
        self._test_keras_model(model, mode = 'ones')
        
        # Test random
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_keras_model(model)

    def test_inner_product_random(self):
        np.random.seed(1988)

        # Define a model
        model = Sequential()
        model.add(Dense(100, input_shape=(100,)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)

    def test_dense_softmax(self):
        np.random.seed(1988)

        # Define a model
        model = Sequential()
        model.add(Dense(32, input_shape=(32,), activation='softmax'))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)

    def test_dense_elu(self):
        np.random.seed(1988)

        # Define a model
        model = Sequential()
        model.add(Dense(32, input_shape=(32,), activation='elu'))

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
        model.add(Dense(num_hidden, input_dim = num_features))
        model.add(Activation('relu'))
        model.add(Dense(1, input_dim = num_features))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_conv_ones(self):
        np.random.seed(1988)
        input_dim = 10
        input_shape = (input_dim, input_dim, 1)
        num_kernels, kernel_height, kernel_width = 3, 5, 5

        # Define a model
        model = Sequential()
        model.add(Conv2D(input_shape = input_shape,
            filters = num_kernels, kernel_size = (kernel_height, kernel_width)))

        # Set some random weights
        model.set_weights([np.ones(w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_conv_random(self):
        np.random.seed(1988)
        input_dim = 10
        input_shape = (input_dim, input_dim, 1)
        num_kernels, kernel_height, kernel_width = 3, 5, 5

        # Define a model
        model = Sequential()
        model.add(Conv2D(input_shape = input_shape,
            filters = num_kernels, kernel_size = (kernel_height, kernel_width)))

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
        model.add(Conv2D(input_shape = input_shape, dilation_rate=(2,2),
            filters = num_kernels, kernel_size = (kernel_height, kernel_width)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model) 

    def test_tiny_conv_dilated_rect_random(self):
        np.random.seed(1988)
        input_shape = (32, 20, 3)
        num_kernels = 2
        kernel_height = 3
        kernel_width = 3

        # Define a model
        model = Sequential()
        model.add(Conv2D(input_shape = input_shape, dilation_rate=(2,2),
            filters = num_kernels, kernel_size = (kernel_height, kernel_width)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)
        
    def test_tiny_conv_pseudo_1d_x(self):
        np.random.seed(1988)
        input_dim = 2
        input_length = 5
        filter_length = 1 # 3
        nb_filters = 1
        # Define a model
        model = Sequential()
        model.add(Conv2D(nb_filters, kernel_size = (1,filter_length), input_shape=(1,input_length,input_dim), padding = 'valid'))
        # Set some random weights
        model.set_weights([np.ones(w.shape) for w in model.get_weights()])
        self._test_keras_model(model, mode='linear')

    def test_tiny_conv1d_same_random(self):
        np.random.seed(1988)
        input_dim = 2
        input_length = 10
        filter_length = 3
        nb_filters = 4
        model = Sequential()
        model.add(Conv1D(nb_filters, kernel_size = filter_length, padding = 'same',
            input_shape=(input_length, input_dim)))

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
        model.add(Conv1D(nb_filters, kernel_size = filter_length, padding = 'valid',
            input_shape=(input_length, input_dim)))

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
        model.add(Conv1D(num_kernels, kernel_size = filter_length, padding = 'valid',
            input_shape = input_shape, dilation_rate = 3))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_conv_rect_kernel_x(self):
        np.random.seed(1988)
        input_dim = 10
        input_shape = (input_dim, input_dim, 1)
        num_kernels = 3
        kernel_height = 1
        kernel_width = 5

        # Define a model
        model = Sequential()
        model.add(Conv2D(input_shape = input_shape,
            filters = num_kernels, kernel_size = (kernel_height, kernel_width),
            padding = 'same'))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_conv_rect_kernel_y(self):
        np.random.seed(1988)
        input_dim = 10
        input_shape = (input_dim, input_dim, 1)
        num_kernels = 3
        kernel_height = 5
        kernel_width = 1

        # Define a model
        model = Sequential()
        model.add(Conv2D(input_shape = input_shape,
            filters = num_kernels, kernel_size = (kernel_height, kernel_width),
            padding = 'valid'))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_conv_rect_kernel_xy(self):
        np.random.seed(1988)
        input_dim = 10
        input_shape = (input_dim, input_dim, 1)
        num_kernels = 3
        kernel_height = 5
        kernel_width = 3

        # Define a model
        model = Sequential()
        model.add(Conv2D(input_shape = input_shape,
            filters = num_kernels, kernel_size = (kernel_height, kernel_width),
            padding = 'valid'))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)

    def test_flatten(self):
        model = Sequential()
        model.add(Flatten(input_shape=(2,2,2)))
        self._test_keras_model(model, mode='linear')

    def test_conv_dense(self):
        input_shape = (48, 48, 3)
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape = input_shape))
        model.add(Flatten())
        model.add(Dense(10, activation = 'softmax'))

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
        model = Sequential()
        model.add(Conv2D(input_shape = input_shape,
            filters = num_kernels, kernel_size = (kernel_height, kernel_width)))
        model.add(BatchNormalization(epsilon=1e-5))

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_conv_batchnorm_no_gamma_no_beta(self):
        np.random.seed(1988)
        input_dim = 10
        input_shape = (input_dim, input_dim, 3)
        num_kernels = 3
        kernel_height = 5
        kernel_width = 5

        # Define a model
        model = Sequential()
        model.add(Conv2D(input_shape = input_shape, 
            filters = num_kernels, kernel_size = (kernel_height, kernel_width)))
        model.add(BatchNormalization(center=False, scale=False, epsilon=1e-5))

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_tiny_deconv_random(self):
        # In Keras 2, deconvolution auto computes the output shape.
        np.random.seed(1988)
        input_dim = 13
        input_shape = (input_dim, input_dim, 5)
        num_kernels = 16
        kernel_height = 3
        kernel_width = 3

        # Define a model
        model = Sequential()
        model.add(Conv2DTranspose(filters = num_kernels, kernel_size=(kernel_height, kernel_width), 
            input_shape = input_shape, padding = 'valid', strides = (2,2)))

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
        model.add(Conv2DTranspose(filters = num_kernels, kernel_size=(kernel_height, kernel_width), 
            input_shape = input_shape, padding = 'same', strides = (2,2)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)


    def test_max_pooling_no_overlap(self):
        # no_overlap: pool_size = strides
        model = Sequential()
        model.add(MaxPooling2D(input_shape=(16,16,3), pool_size=(2, 2),
                               strides=None, padding='valid'))
        self._test_keras_model(model)

    def test_max_pooling_overlap_multiple(self):
        # input shape is multiple of pool_size, strides != pool_size
        model = Sequential()
        model.add(MaxPooling2D(input_shape=(18,18,3), pool_size=(3, 3),
                               strides=(2,2), padding='valid'))
        self._test_keras_model(model)

    def test_max_pooling_overlap_odd(self):
        model = Sequential()
        model.add(MaxPooling2D(input_shape=(16,16,3), pool_size=(3, 3),
                               strides=(2,2), padding='valid'))
        self._test_keras_model(model)

    def test_max_pooling_overlap_same(self):
        model = Sequential()
        model.add(MaxPooling2D(input_shape=(16,16,3), pool_size=(3, 3),
                               strides=(2,2), padding='same'))
        self._test_keras_model(model)

    def test_global_max_pooling(self):
        model = Sequential()
        model.add(GlobalMaxPooling2D(input_shape=(16,16,3)))
        self._test_keras_model(model)

    def test_average_pooling_no_overlap(self):
        # no_overlap: pool_size = strides
        model = Sequential()
        model.add(AveragePooling2D(input_shape=(16,16,3), pool_size=(2, 2),
                               strides=None, padding='valid'))
        self._test_keras_model(model, delta=1e-2)

    def test_average_pooling_inception_config_1(self):
        # no_overlap: pool_size = strides
        model = Sequential()
        model.add(AveragePooling2D(input_shape=(16,16,3), pool_size=(3,3),
                               strides=(1,1), padding='same'))
        self._test_keras_model(model, delta=1e-2)

    def test_global_average_pooling(self):
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=(16,16,3)))
        self._test_keras_model(model)

    def test_max_pooling_1d(self):
        model = Sequential()
        model.add(MaxPooling1D(input_shape=(16,3), pool_size=4))
        self._test_keras_model(model)

    def test_global_max_pooling_1d(self):
        np.random.seed(1988)
        input_dim = 2
        input_length = 10
        filter_length = 3
        nb_filters = 4
        model = Sequential()
        model.add(Conv1D(nb_filters, kernel_size = filter_length, padding='same',
            input_shape=(input_length, input_dim)))
        model.add(GlobalMaxPooling1D())
        self._test_keras_model(model)

    def test_average_pooling_1d(self):
        np.random.seed(1988)
        input_dim = 2
        input_length = 10
        filter_length = 3
        nb_filters = 4
        model = Sequential()
        model.add(Conv1D(nb_filters, kernel_size = filter_length, padding='same',
            input_shape=(input_length, input_dim)))
        model.add(AveragePooling1D(pool_size=2))
        self._test_keras_model(model)

    def test_global_average_pooling_1d(self):
        np.random.seed(1988)
        input_dim = 2
        input_length = 10
        filter_length = 3
        nb_filters = 4
        model = Sequential()
        model.add(Conv1D(nb_filters, kernel_size = filter_length, padding='same',
            input_shape=(input_length, input_dim)))
        model.add(GlobalAveragePooling1D())
        self._test_keras_model(model)

    def test_tiny_conv_upsample_random(self):
        np.random.seed(1988)
        input_dim = 10
        input_shape = (input_dim, input_dim, 1)
        num_kernels = 3
        kernel_height = 5
        kernel_width = 5

        # Define a model
        model = Sequential()
        model.add(Conv2D(input_shape = input_shape, 
            filters = num_kernels, kernel_size = (kernel_height, kernel_width)))
        model.add(UpSampling2D(size = 2))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_conv_upsample_1d_random(self):
        np.random.seed(1988)
        input_dim = 2
        input_length = 10
        filter_length = 3
        nb_filters = 4
        model = Sequential()
        model.add(Conv1D(nb_filters, kernel_size = filter_length, padding='same',
            input_shape=(input_length, input_dim)))
        model.add(UpSampling1D(size = 2))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_conv_crop_1d_random(self):
        np.random.seed(1988)
        input_dim = 2
        input_length = 10
        filter_length = 3
        nb_filters = 4
        model = Sequential()
        model.add(Conv1D(nb_filters, kernel_size = filter_length, padding='same',
            input_shape=(input_length, input_dim)))
        model.add(Cropping1D(cropping = 2))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)

    def test_tiny_conv_pad_1d_random(self):
        np.random.seed(1988)
        input_dim = 2
        input_length = 10
        filter_length = 3
        nb_filters = 4
        model = Sequential()
        model.add(Conv1D(nb_filters, kernel_size = filter_length, padding='same',
            input_shape=(input_length, input_dim)))
        model.add(ZeroPadding1D(padding = 2))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model)

    def test_embedding(self):
        model = Sequential()
        num_inputs = 10
        num_outputs = 3
        model.add(Embedding(num_inputs, num_outputs))

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_keras_model(model)

    def test_embedding_seq(self):
        model = Sequential()
        num_inputs = 10
        num_outputs = 3
        model.add(Embedding(num_inputs, num_outputs, input_length=7))

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_keras_model(model, one_dim_seq_flags=[True])

    def test_tiny_no_sequence_simple_rnn_random(self):
        np.random.seed(1988)
        input_dim = 10
        input_length = 1
        num_channels = 1

        # Define a model
        model = Sequential()
        model.add(SimpleRNN(num_channels, input_shape=(input_length, input_dim)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_tiny_sequence_simple_rnn_random(self):
        np.random.seed(1988)
        input_dim = 2
        input_length = 4
        num_channels = 3

        # Define a model
        model = Sequential()
        model.add(SimpleRNN(num_channels, input_shape=(input_length, input_dim)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2 - 0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_tiny_seq2seq_rnn_random(self):
        np.random.seed(1988)
        input_dim = 2
        input_length = 4
        num_channels = 3

        # Define a model
        model = Sequential()
        model.add(SimpleRNN(num_channels, input_shape=(input_length, input_dim), return_sequences=True))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2 - 0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_rnn_seq(self):
        np.random.seed(1988)
        input_dim = 11
        input_length = 5

        # Define a model
        model = Sequential()
        model.add(SimpleRNN(20, input_shape=(input_length, input_dim), return_sequences=False))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2 - 0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_rnn_seq_backwards(self):
        np.random.seed(1988)
        input_dim = 11
        input_length = 5

        # Define a model
        model = Sequential()
        model.add(SimpleRNN(20, input_shape=(input_length, input_dim), return_sequences=False, go_backwards=True))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2 - 0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_medium_no_sequence_simple_rnn_random(self):
        np.random.seed(1988)
        input_dim = 10
        input_length = 1
        num_channels = 10

        # Define a model
        model = Sequential()
        model.add(SimpleRNN(num_channels, input_shape=(input_length, input_dim)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2 - 0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_tiny_no_sequence_lstm_zeros(self):
        np.random.seed(1988)
        input_dim = 1
        input_length = 1
        num_channels = 1

        model = Sequential()
        model.add(LSTM(num_channels, input_shape = (input_length, input_dim),
            implementation = 0, recurrent_activation = 'sigmoid'))

        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])
        self._test_keras_model(model, mode = 'zeros', input_blob = 'data', output_blob = 'output')

    def test_tiny_no_sequence_lstm_ones(self):
        np.random.seed(1988)
        input_dim = 1
        input_length = 1
        num_channels = 1

        model = Sequential()
        model.add(LSTM(num_channels, input_shape = (input_length, input_dim),
            implementation = 0, recurrent_activation = 'sigmoid'))

        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])
        self._test_keras_model(model, mode = 'ones', input_blob = 'data', output_blob = 'output')

    def test_small_no_sequence_lstm_zeros(self):
        np.random.seed(1988)
        input_dim = 10
        input_length = 1
        num_channels = 1

        model = Sequential()
        model.add(LSTM(num_channels, input_shape = (input_length, input_dim),
            implementation = 2, recurrent_activation = 'sigmoid'))

        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])
        self._test_keras_model(model, mode = 'zeros', input_blob = 'data', output_blob = 'output')

    def test_small_no_sequence_lstm_ones(self):
        np.random.seed(1988)
        input_dim = 10
        input_length = 1
        num_channels = 1

        model = Sequential()
        model.add(LSTM(num_channels, input_shape = (input_length, input_dim),
            implementation = 2, recurrent_activation = 'sigmoid'))

        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])
        self._test_keras_model(model, mode = 'ones', input_blob = 'data', output_blob = 'output')

    def test_lstm_seq(self):
        np.random.seed(1988)
        input_dim = 11
        input_length = 5

        model = Sequential()
        model.add(LSTM(20, input_shape = (input_length, input_dim), return_sequences=False))

        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_lstm_seq_backwards(self):
        np.random.seed(1988)
        input_dim = 11
        input_length = 5

        model = Sequential()
        model.add(LSTM(20, input_shape = (input_length, input_dim), return_sequences=False, go_backwards=True))

        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_medium_no_sequence_lstm_random(self):
        np.random.seed(1988)
        input_dim = 10
        input_length = 1
        num_channels = 10

        # Define a model
        model = Sequential()
        model.add(LSTM(num_channels, input_shape = (input_length, input_dim),
                       recurrent_activation = 'sigmoid'))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_tiny_no_sequence_lstm_zeros_gpu(self):
        np.random.seed(1988)
        input_dim = 1
        input_length = 1
        num_channels = 1

        # Define a model
        model = Sequential()
        model.add(LSTM(num_channels, input_shape = (input_length, input_dim),
                       implementation = 2, recurrent_activation = 'sigmoid'))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, mode = 'zeros', input_blob = 'data', output_blob = 'output')

    def test_small_no_sequence_lstm_random(self):
        np.random.seed(1988)
        input_dim = 10
        input_length = 1
        num_channels = 1

        # Define a model
        model = Sequential()
        model.add(LSTM(num_channels, input_shape = (input_length, input_dim),
                       implementation = 2, recurrent_activation = 'sigmoid'))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_tiny_no_sequence_gru_random(self):
        np.random.seed(1988)
        input_dim = 1
        input_length = 1
        num_channels = 1
        num_samples = 1

        # Define a model
        model = Sequential()
        model.add(GRU(num_channels, input_shape = (input_length, input_dim),
                      recurrent_activation = 'sigmoid'))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_small_no_sequence_gru_random(self):
        np.random.seed(1988)
        input_dim = 10
        input_length = 1
        num_channels = 1

        # Define a model
        model = Sequential()
        model.add(GRU(num_channels, input_shape = (input_length, input_dim),
               recurrent_activation = 'sigmoid'))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_medium_no_sequence_gru_random(self):
        np.random.seed(1988)
        input_dim = 10
        input_length = 1
        num_channels = 10

        # Define a model
        model = Sequential()
        model.add(GRU(num_channels, input_shape = (input_length, input_dim), recurrent_activation = 'sigmoid'))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_gru_seq(self):
        np.random.seed(1988)
        input_dim = 11
        input_length = 5

        # Define a model
        model = Sequential()
        model.add(GRU(20, input_shape = (input_length, input_dim), return_sequences=False))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_gru_seq_backwards(self):
        np.random.seed(1988)
        input_dim = 11
        input_length = 5

        # Define a model
        model = Sequential()
        model.add(GRU(20, input_shape = (input_length, input_dim), return_sequences=False, go_backwards=True))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_tiny_no_sequence_bidir_random(self):
        np.random.seed(1988)
        input_dim = 1
        input_length = 1
        num_channels = 1
        num_samples = 1

        # Define a model
        model = Sequential()
        model.add(Bidirectional(LSTM(num_channels, 
            implementation = 0, recurrent_activation = 'sigmoid'),
            input_shape=(input_length, input_dim)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_tiny_no_sequence_bidir_random_gpu(self):
        np.random.seed(1988)
        input_dim = 1
        input_length = 1
        num_channels = 1
        num_samples = 1

        # Define a model
        model = Sequential()
        model.add(Bidirectional(LSTM(num_channels, 
                                     implementation = 2, recurrent_activation = 'sigmoid'),
                                input_shape=(input_length, input_dim)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])

                                # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_small_no_sequence_bidir_random(self):
        np.random.seed(1988)
        input_dim = 10
        input_length = 1
        num_channels = 1

        # Define a model
        model = Sequential()
        model.add(Bidirectional(LSTM(num_channels, 
            implementation = 2, recurrent_activation = 'sigmoid'),
            input_shape=(input_length, input_dim)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_medium_no_sequence_bidir_random(self):
        np.random.seed(1988)
        input_dim = 10
        input_length = 1
        num_channels = 10

        # Define a model
        model = Sequential()
        model.add(Bidirectional(LSTM(num_channels, 
            implementation = 2, recurrent_activation = 'sigmoid'),
            input_shape=(input_length, input_dim)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')

    def test_medium_bidir_random_return_seq_false(self):
        np.random.seed(1988)
        input_dim = 7
        input_length = 5
        num_channels = 10

        # Define a model
        model = Sequential()
        model.add(Bidirectional(LSTM(num_channels, 
                                return_sequences=False, implementation=2, recurrent_activation='sigmoid'),
                                input_shape=(input_length, input_dim)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob='data', output_blob='output')

    def test_medium_bidir_random_return_seq_true(self):
        np.random.seed(1988)
        input_dim = 7
        input_length = 5
        num_channels = 10

        # Define a model
        model = Sequential()
        model.add(Bidirectional(LSTM(num_channels, 
                                return_sequences = True, implementation = 2, recurrent_activation = 'sigmoid'),
                                input_shape=(input_length, input_dim)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape)*0.2-0.1 for w in model.get_weights()])

                                # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output')


    def test_tiny_conv_elu_random(self):
        np.random.seed(1988)

        # Define a model
        from keras.layers.advanced_activations import ELU
        model = Sequential()
        model.add(Conv2D(input_shape = (10, 10, 3),
            filters = 3, kernel_size = (5,5)))
        model.add(ELU(alpha=0.8))

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_tiny_conv_prelu_random(self):
        np.random.seed(1988)

        # Define a model
        from keras.layers.advanced_activations import PReLU
        model = Sequential()
        model.add(Conv2D(input_shape = (10, 10, 3),
            filters = 3, kernel_size = (5,5), padding = 'same'))
        model.add(PReLU(shared_axes=[1, 2]))

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_tiny_conv_leaky_relu_random(self):
        np.random.seed(1988)

        # Define a model
        from keras.layers.advanced_activations import LeakyReLU
        model = Sequential()
        model.add(Conv2D(input_shape = (10, 10, 3),
            filters = 3, kernel_size=(5,5), padding = 'same'))
        model.add(LeakyReLU(alpha=0.3))

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_tiny_conv_thresholded_relu_random(self):
        np.random.seed(1988)

        # Define a model
        from keras.layers.advanced_activations import ThresholdedReLU
        model = Sequential()
        model.add(Conv2D(input_shape = (10, 10, 3),
            filters = 3, kernel_size=(5,5), padding = 'same'))
        model.add(ThresholdedReLU(theta=0.8))

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_tiny_concat_random(self):
        np.random.seed(1988)
        input_dim = 10
        num_channels = 6

        # Define a model
        input_tensor = Input(shape = (input_dim, ))
        x1 = Dense(num_channels)(input_tensor)
        x2 = Dense(num_channels)(x1)
        x3 = Dense(num_channels)(x1)
        x4 = concatenate([x2, x3])
        x5 = Dense(num_channels)(x4)

        model = Model(inputs=[input_tensor], outputs=[x5])

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_tiny_concat_seq_random(self):
        np.random.seed(1988)
        max_features = 10
        embedding_dims = 4
        seq_len = 5
        num_channels = 6

        # Define a model
        input_tensor = Input(shape = (seq_len, ))
        x1 = Embedding(max_features, embedding_dims)(input_tensor)
        x2 = Embedding(max_features, embedding_dims)(input_tensor)
        x3 = concatenate([x1, x2], axis=1)

        model = Model(inputs=[input_tensor], outputs=[x3])

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model, one_dim_seq_flags=[True])

    def test_tiny_add_random(self):
        np.random.seed(1988)
        input_dim = 10
        num_channels = 6

        # Define a model
        input_tensor = Input(shape = (input_dim, ))
        x1 = Dense(num_channels)(input_tensor)
        x2 = Dense(num_channels)(x1)
        x3 = Dense(num_channels)(x1)
        x4 = add([x2, x3])
        x5 = Dense(num_channels)(x4)

        model = Model(inputs=[input_tensor], outputs=[x5])

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_tiny_mul_random(self):
        np.random.seed(1988)
        input_dim = 10
        num_channels = 6

        # Define a model
        input_tensor = Input(shape = (input_dim, ))
        x1 = Dense(num_channels)(input_tensor)
        x2 = Dense(num_channels)(x1)
        x3 = Dense(num_channels)(x1)
        x4 = multiply([x2, x3])
        x5 = Dense(num_channels)(x4)

        model = Model(inputs=[input_tensor], outputs=[x5])

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_tiny_cos_random(self):
        np.random.seed(1988)
        input_dim = 10
        num_channels = 6

        # Define a model
        input_tensor = Input(shape = (input_dim, ))
        x1 = Dense(num_channels)(input_tensor)
        x2 = Dense(num_channels)(x1)
        x3 = Dense(num_channels)(x1)
        x4 = dot([x2, x3], axes=-1, normalize=True)
        x5 = Dense(num_channels)(x4)

        model = Model(inputs=[input_tensor], outputs=[x5])

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_zeropad_simple(self):

        input_shape = (48, 48, 3)
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=input_shape))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_zeropad_fancy(self):

        input_shape = (48, 48, 3)
        model = Sequential()
        model.add(ZeroPadding2D(((2,5),(3,4)),input_shape=input_shape))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_crop_simple(self):

        input_shape = (48, 48, 3)
        model = Sequential()
        model.add(Cropping2D(cropping=((2,5),(2,5)),input_shape=input_shape))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_tiny_permute(self):
        model = Sequential()
        model.add(Permute((3, 2, 1), input_shape=(4, 3, 2)))

        # When input blob is 3D array (D1, D2, D3), Keras assumes the axes' meaning is
        # (D1=H,D2=W,D3=C), while CoreML assumes (D1=C,D2=H,D3=W). However,
        # it's unclear after permutation, what the axes' meaning is for the output blob.
        # Since permutation done on (H,W,C) blobs usually is usually followed by
        # recurrent layers / Dense, we choose that the ouput axis order of CoreML is
        # the same as Keras after permutation.
        self._test_keras_model(model, transpose_keras_result=False)

    def test_reshape_3d(self):
        model = Sequential()
        model.add(Reshape((10,1,6), input_shape=(5,4,3)))
        self._test_keras_model(model,  mode = 'linear')

    def test_tiny_conv_dense_random(self):
        np.random.seed(1988)
        num_samples = 1
        input_dim = 8
        input_shape = (input_dim, input_dim, 3)
        num_kernels = 2
        kernel_height = 5
        kernel_width = 5
        hidden_dim = 4

        # Define a model
        model = Sequential()
        model.add(Conv2D(input_shape = input_shape,
            filters = num_kernels, kernel_size=(kernel_height, kernel_width)))
        model.add(Flatten())
        model.add(Dense(hidden_dim))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_tiny_dense_tanh_fused_random(self):
        np.random.seed(1988)
        num_samples = 1
        input_dim = 3
        hidden_dim = 4

        # Define a model
        model = Sequential()
        model.add(Dense(hidden_dim, input_shape=(input_dim,), activation='tanh'))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_tiny_conv_relu_fused_random(self):
        np.random.seed(1988)
        num_samples = 1
        input_dim = 8
        input_shape = (input_dim, input_dim, 3)
        num_kernels = 2
        kernel_height = 5
        kernel_width = 5
        hidden_dim = 4

        # Define a model
        model = Sequential()
        model.add(Conv2D(input_shape = input_shape, activation='relu',
            filters = num_kernels, kernel_size=(kernel_height, kernel_width)))

        # Set some random weights
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)


    def test_tiny_time_distrbuted(self):

        # as the first layer in a model
        model = Sequential()
        model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        self._test_keras_model(model)

    def test_tiny_sequence_lstm(self):

        np.random.seed(1988)
        input_dim = 1
        input_length = 2
        num_channels = 1

        # Define a model
        model = Sequential()
        model.add(LSTM(num_channels, input_shape = (input_length, input_dim), 
            implementation = 0, recurrent_activation = 'sigmoid'))

        # Set some random weights
        model.set_weights([(np.random.rand(*w.shape)-0.5)*0.2 for w in model.get_weights()])

        # Test the keras model
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output', delta=1e-4)

    def test_tiny_spatial_bn(self):
        np.random.seed(1988)
        x_in = Input(shape=(7,7,2))
        x = ZeroPadding2D(padding=(1, 1))(x_in)
        x = BatchNormalization(axis=2)(x)
        model = Model(x_in, x)

        self._test_keras_model(model, input_blob = 'data', output_blob = 'output', delta=1e-2)
    
    def test_dense_fused_act_in_td(self):
        np.random.seed(1988)
        x_in = Input(shape=(10,2))
        x = TimeDistributed(Dense(6, activation = 'softmax'))(x_in)
        model = Model(inputs=[x_in], outputs=[x])
    
        self._test_keras_model(model, input_blob = 'data', output_blob = 'output', delta=1e-4)

    def test_conv_batch_1d(self):
        vocabulary_size = 4
        embedding_dimension = 6
        input_length = 10

        model = Sequential()
        model.add(Embedding(vocabulary_size, embedding_dimension, input_length=input_length, trainable=True))

        model.add(Conv1D(5, 2))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(MaxPooling1D(2))

        self._test_keras_model(model, one_dim_seq_flags=[True])

    # Making sure that giant channel sizes get handled correctly
    def test_large_channel_gpu(self):

        input_shape = (20, 20, 3)
        num_channels = 2049
        kernel_size = 3

        model = Sequential()
        model.add(Conv2D(input_shape = input_shape,
                         filters = num_channels, kernel_size = (kernel_size, kernel_size)))

        model.set_weights([(np.random.rand(*w.shape)-0.5)*0.2 for w in model.get_weights()])

        self._test_keras_model(model, input_blob='data', output_blob='output', delta=1e-2)

    @raises(Exception)
    def test_large_batch_gpu(self):

        batch_size = 2049
        num_channels = 4
        kernel_size = 3

        model = Sequential()
        model.add(TimeDistributed(Dense(num_channels), input_shape=(batch_size, kernel_size)))

        model.set_weights([(np.random.rand(*w.shape)-0.5)*0.2 for w in model.get_weights()])

        self._test_keras_model(model, input_blob='data', output_blob='output', delta=1e-2)


@unittest.skipIf(not HAS_KERAS2_TF, 'Missing keras. Skipping tests.')
@attr('keras2')
class KerasTopologyCorrectnessTest(KerasNumericCorrectnessTest):

    def test_dangling_merge_left(self):

        x1 = Input(shape=(4,), name = 'input1')
        x2 = Input(shape=(5,), name = 'input2')
        y1 = Dense(6, name = 'dense')(x2)
        z = concatenate([x1, y1])
        model = Model(inputs = [x1,x2], outputs = [z])

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        self._test_keras_model(model)

    def test_dangling_merge_right(self):

        x1 = Input(shape=(4,), name = 'input1')
        x2 = Input(shape=(5,), name = 'input2')
        y1 = Dense(6, name = 'dense')(x2)
        z = concatenate([y1, x1])
        model = Model(inputs = [x1,x2], outputs = [z])

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])

        self._test_keras_model(model)

    def test_shared_vision(self):
        digit_input = Input(shape=(27, 27,1))
        x = Conv2D(64, (3, 3))(digit_input)
        x = Conv2D(64, (3, 3))(x)
        out = Flatten()(x)

        vision_model = Model(inputs=[digit_input], outputs=[out])

        # then define the tell-digits-apart model
        digit_a = Input(shape=(27,27,1))
        digit_b = Input(shape=(27,27,1))

        # the vision model will be shared, weights and all
        out_a = vision_model(digit_a)
        out_b = vision_model(digit_b)

        concatenated = concatenate([out_a, out_b])
        out = Dense(1, activation='sigmoid')(concatenated)
        model = Model(inputs=[digit_a, digit_b], outputs=out)
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_keras_model(model)

    def test_tiny_weight_sharing(self):
        #     - Dense1 -----------
        # x - |                   |- Merge
        #     - Dense1 - Dense2 --

        x = Input(shape=(3,))
        dense = Dense(4)
        y1 = dense(x)
        y2 = dense(x)
        y3 = Dense(4)(y2)
        z = concatenate([y1, y3])
        model = Model(inputs=[x], outputs=[z])

        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_keras_model(model, mode = 'random', delta=1e-4)
    
    def test_tiny_multiple_outputs(self):
        x = Input(shape=(3,))
        y1 = Dense(4)(x)
        y2 = Dense(5)(x)
        model = Model([x], [y1,y2])
        
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_keras_model(model, mode = 'random', delta=1e-4)
        

@attr('slow')
@attr('keras2')
@unittest.skipIf(not HAS_KERAS2_TF, 'Missing keras. Skipping tests.')
class KerasNumericCorrectnessStressTest(KerasNumericCorrectnessTest):
    """
    Unit test class for testing all combinations of a particular
    layer.
    """
    def _run_test(self, model, param, model_dir = None, delta=1e-2, transpose_keras_result=True, one_dim_seq_flags = None):
        """ Run a test on a particular model
        """
        use_tmp_folder = False
        if model_dir is None:
            use_tmp_folder = True
            model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, 'keras.mlmodel')

        # Generate some random data
        nb_inputs = len(model.inputs)
        if nb_inputs > 1:
            input_names = []; input_data = []; coreml_input = {}
            for i in range(nb_inputs):
                input_shape = [1 if a is None else a for a in model.input_shape[i]]
                X = _generate_data(input_shape)
                feature_name = "data_%s" % i
                input_names.append(feature_name)
                input_data.append(X)
                if one_dim_seq_flags is None:
                    coreml_input[feature_name] = _keras_transpose(X).astype('f')
                else:
                    coreml_input[feature_name] = _keras_transpose(X, one_dim_seq_flags[i]).astype('f')
        else:
            input_shape = [1 if a is None else a for a in model.input_shape]
            input_names = ['data']
            input_data = _generate_data(input_shape)
            if one_dim_seq_flags is None:
                coreml_input = {'data': _keras_transpose(input_data).astype('f')}
            else:
                coreml_input = {'data': _keras_transpose(input_data, one_dim_seq_flags[0]).astype('f')}

        # Make predictions
        if transpose_keras_result:
            keras_preds = _keras_transpose(model.predict(input_data)).flatten()
        else:
            keras_preds = model.predict(input_data).flatten()

        # Get the model
        coreml_model = _get_coreml_model(model, model_path, input_names, ['output'])
        # get prediction
        coreml_preds = coreml_model.predict(coreml_input)['output'].flatten()

        if use_tmp_folder:
            shutil.rmtree(model_dir)
        self.assertEquals(len(coreml_preds), len(keras_preds),
                msg = 'Failed test case %s. Lengths wrong (%s vs %s)' % (param, len(coreml_preds), len(keras_preds)))
        for i in range(len(keras_preds)):
            max_den = max(1.0, keras_preds[i], coreml_preds[i])
            self.assertAlmostEquals(keras_preds[i]/max_den, coreml_preds[i]/max_den, delta = delta,
                msg = 'Failed test case %s. Predictions wrong (%s vs %s)' % (param, coreml_preds[i], keras_preds[i]))

    @attr('slow')
    def test_activation_layer_params(self):
        options = dict(
            activation = ['tanh', 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'hard_sigmoid', 'elu']
        )

        # Define a function that tests a model
        num_channels = 10
        input_dim = 10
        def build_model(x):
            model = Sequential()
            model.add(Dense(num_channels, input_dim = input_dim))
            model.add(Activation(**dict(zip(options.keys(), x))))
            return x, model

        # Iterate through all combinations
        product = itertools.product(*options.values())
        args = [build_model(p) for p in product]

        # Test the cases
        print("Testing a total of %s cases. This could take a while" % len(args))
        for param, model in args:
            model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
            self._run_test(model, param)

    @attr('slow')
    def test_dense_layer_params(self):
        options = dict(
            # TODO - hard-sigmoid result mismatch rdar://problem/32363330
            # activation = ['relu', 'softmax', 'tanh', 'sigmoid', 'hard_sigmoid', 'softplus', 'softsign', 'elu'],
            activation = ['relu', 'softmax', 'tanh', 'sigmoid', 'softplus', 'softsign', 'elu'],
            use_bias = [True, False],
        )
        # Define a function that tests a model
        input_shape = (10,)
        num_channels = 10
        def build_model(x):
            kwargs = dict(zip(options.keys(), x))
            model = Sequential()
            model.add(Dense(num_channels, input_shape = input_shape, **kwargs))
            return x, model

        # Iterate through all combinations
        product = itertools.product(*options.values())
        args = [build_model(p) for p in product]

        # Test the cases
        print("Testing a total of %s cases. This could take a while" % len(args))
        for param, model in args:
            self._run_test(model, param)

    @attr('slow')
    def test_upsample_layer_params(self):
        options = dict(
            size= [(2,2), (3,3), (4,4), (5,5)]
        )

        np.random.seed(1988)
        input_dim = 10
        input_shape = (input_dim, input_dim, 1)
        X = np.random.rand(1, *input_shape)

        # Define a function that tests a model
        def build_model(x):
            kwargs = dict(zip(options.keys(), x))
            model = Sequential()
            model.add(Conv2D(input_shape = input_shape, nb_row = 7,
                nb_col = 7, nb_filter = 5))
            model.add(UpSampling2D(**kwargs))
            return x, model

        # Iterate through all combinations
        product = itertools.product(*options.values())
        args = [build_model(p) for p in product]

        # Test the cases
        print("Testing a total of %s cases. This could take a while" % len(args))
        for param, model in args:
            self._run_test(model, param)

    @attr('slow')
    def test_conv_layer_params(self):
        options = dict(
            activation = ['relu', 'tanh', 'sigmoid'], # keras does not support softmax on 4-D
            use_bias = [True, False],
            padding = ['same', 'valid'],
            filters = [1, 3, 5],
            kernel_size = [[5,5]], # fails when sizes are different
        )

        # Define a function that tests a model
        input_shape = (10, 10, 1)
        def build_model(x):
            kwargs = dict(zip(options.keys(), x))
            model = Sequential()
            model.add(Conv2D(input_shape = input_shape, **kwargs))
            return x, model

        # Iterate through all combinations
        product = itertools.product(*options.values())
        args = [build_model(p) for p in product]

        # Test the cases
        print("Testing a total of %s cases. This could take a while" % len(args))
        for param, model in args:
            self._run_test(model, param)

    @attr('slow')
    def test_dense_elementwise_params(self):
        options = dict(
            modes = [add, multiply, concatenate, average, maximum]
        )
        def build_model(mode):
            x1 = Input(shape=(3,))
            x2 = Input(shape=(3,))
            y1 = Dense(4)(x1)
            y2 = Dense(4)(x2)
            z = mode([y1, y2])
            model = Model([x1,x2], z)
            return mode, model

        product = itertools.product(*options.values())
        args = [build_model(p[0]) for p in product]
        print("Testing a total of %s cases. This could take a while" % len(args))
        for param, model in args:
            self._run_test(model, param)

    def test_vgg_16_tiny(self):

        input_shape = (48, 48, 3)
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=input_shape))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000)) # activation='softmax'))

        # Set some random weights
        model.set_weights([(np.random.rand(*w.shape)-0.5)*0.2 for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_vgg_16_tiny_no_pooling(self):

        input_shape = (48, 48, 3)
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=input_shape))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(1000)) # activation='softmax'))

        # Set some random weights
        model.set_weights([(np.random.rand(*w.shape)-0.5)*0.2 for w in model.get_weights()])

        # Get the coreml model
        self._test_keras_model(model)

    def test_vgg_16_tiny_no_pooling_no_padding(self):

        input_shape = (48, 48, 3)
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', input_shape = input_shape))
        model.add(Conv2D(32, (3,3), activation='relu'))

        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(Conv2D(32, (3,3), activation='relu'))

        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(Conv2D(32, (3,3), activation='relu'))

        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(Conv2D(32, (3,3), activation='relu'))

        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(Conv2D(32, (3,3), activation='relu'))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))

        # Get the coreml model
        self._test_keras_model(model)

    def test_imdb_fasttext_first_2(self):

        max_features = 10
        max_len = 6
        embedding_dims = 4
        pool_length = 2

        model = Sequential()
        model.add(Embedding(max_features, embedding_dims, input_length=max_len))
        # we add a AveragePooling1D, which will average the embeddings
        # of all words in the document
        model.add(AveragePooling1D(pool_size=pool_length))

        self._test_keras_model(model, one_dim_seq_flags=[True])

    def test_tiny_mcrnn_td(self):

        model = Sequential()
        model.add(Conv2D(3,(1,1), input_shape=(2,4,4), padding='same'))
        model.add(AveragePooling2D(pool_size=(2,2)))
        model.add(Reshape((2,3)))
        model.add(TimeDistributed(Dense(5)))

        self._test_keras_model(model)

    def test_tiny_mcrnn_recurrent(self):

        model = Sequential()
        model.add(Conv2D(3,(1,1), input_shape=(2,4,4), padding='same'))
        model.add(AveragePooling2D(pool_size=(2,2)))
        model.add(Reshape((2,3)))
        model.add(LSTM(5, recurrent_activation = 'sigmoid'))

        self._test_keras_model(model)

    def test_tiny_mcrnn_music_tagger(self):

        x_in = Input(shape=(4,6,1))
        x = ZeroPadding2D(padding=(0, 1))(x_in)
        x = BatchNormalization(axis=2, name='bn_0_freq')(x)
        # Conv block 1
        x = Conv2D(2, (3, 3), padding='same', name='conv1')(x)
        x = BatchNormalization(axis=3, name='bn1')(x)
        x = Activation('elu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
        # Conv block 2
        x = Conv2D(4, (3, 3), padding='same', name='conv2')(x)
        x = BatchNormalization(axis=3, name='bn2')(x)
        x = Activation('elu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)

        # Should get you (1,1,2,4)
        x = Reshape((2, 4))(x)
        x = GRU(32, return_sequences=True, name='gru1')(x)
        x = GRU(32, return_sequences=False, name='gru2')(x)

        # Create model.
        model = Model(x_in, x)
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        self._test_keras_model(model, mode='random_zero_mean', delta=1e-2)

    def test_tiny_apple_manual(self):
        model = Sequential()
        model.add(LSTM(3, input_shape=(4, 5), recurrent_activation='sigmoid'))
        model.add(Dense(5))
        model.add(Activation('softmax'))

        self._test_keras_model(model)

    def test_tiny_image_captioning_image_branch(self):
        img_input_1 = Input(shape=(16,16,3))
        x = Conv2D(2,(3,3))(img_input_1)
        x = Flatten()(x)
        img_model = Model(inputs=[img_input_1], outputs=[x])

        img_input = Input(shape=(16,16,3))
        x = img_model(img_input)
        x = Dense(8, name = 'cap_dense')(x)
        x = Reshape((1,8), name = 'cap_reshape')(x)
        image_branch = Model(inputs=[img_input],outputs=[x])
        self._test_keras_model(image_branch)

    def test_tiny_image_captioning_feature_merge(self):

        img_input_1 = Input(shape=(16,16,3))
        x = Conv2D(2,(3,3))(img_input_1)
        x = Flatten()(x)
        img_model = Model([img_input_1], [x])

        img_input = Input(shape=(16,16,3))
        x = img_model(img_input)
        x = Dense(8, name = 'cap_dense')(x)
        x = Reshape((1,8), name = 'cap_reshape')(x)

        sentence_input = Input(shape=(5,)) # max_length = 5
        y = Embedding(8, 8, name = 'cap_embedding')(sentence_input)
        z = concatenate([x,y], axis = 1, name = 'cap_merge')

        combined_model = Model(inputs=[img_input, sentence_input], outputs=[z])
        self._test_keras_model(combined_model, one_dim_seq_flags=[False, True])

    def test_tiny_image_captioning(self):
        # use a conv layer as a image feature branch
        img_input_1 = Input(shape=(16,16,3))
        x = Conv2D(2,(3,3))(img_input_1)
        x = Flatten()(x)
        img_model = Model(inputs=[img_input_1], outputs=[x])

        img_input = Input(shape=(16,16,3))
        x = img_model(img_input)
        x = Dense(8, name = 'cap_dense')(x)
        x = Reshape((1,8), name = 'cap_reshape')(x)

        sentence_input = Input(shape=(5,)) # max_length = 5
        y = Embedding(8, 8, name = 'cap_embedding')(sentence_input)
        z = concatenate([x,y], axis = 1, name = 'cap_merge')
        z = LSTM(4, return_sequences = True, name = 'cap_lstm')(z)
        z = TimeDistributed(Dense(8), name = 'cap_timedistributed')(z)

        combined_model = Model(inputs=[img_input, sentence_input], outputs=[z])
        self._test_keras_model(combined_model, one_dim_seq_flags=[False, True])

    def test_tiny_babi_rnn(self):
        vocab_size = 10
        embed_hidden_size = 8
        story_maxlen = 5
        query_maxlen = 5

        input_tensor_1 = Input(shape=(story_maxlen,))
        x1 = Embedding(vocab_size, embed_hidden_size)(input_tensor_1)
        x1 = Dropout(0.3)(x1)
        
        input_tensor_2 = Input(shape=(query_maxlen,))
        x2 = Embedding(vocab_size, embed_hidden_size)(input_tensor_2)
        x2 = Dropout(0.3)(x2)
        x2 = LSTM(embed_hidden_size, return_sequences=False)(x2)
        x2 = RepeatVector(story_maxlen)(x2)
        
        x3 = add([x1, x2])
        x3 = LSTM(embed_hidden_size, return_sequences=False)(x3)
        x3 = Dropout(0.3)(x3)
        x3 = Dense(vocab_size, activation='softmax')(x3)
        
        model = Model(inputs=[input_tensor_1,input_tensor_2], outputs=[x3])

        self._test_keras_model(model, one_dim_seq_flags=[True, True])

    def test_clickbait_cnn(self):
        # from: https://github.com/saurabhmathur96/clickbait-detector
        vocabulary_size = 500
        embedding_dimension = 30
        input_length = 20
        
        model = Sequential()
        model.add(Embedding(vocabulary_size, embedding_dimension, input_length=input_length, trainable=True))

        model.add(Conv1D(32, 2))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv1D(32, 2))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(Conv1D(32, 2))
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(MaxPooling1D(17))
        model.add(Flatten())

        model.add(Dense(1, bias=True))
        model.add(BatchNormalization())
        model.add(Activation("sigmoid"))

        self._test_keras_model(model, one_dim_seq_flags=[True])
