import unittest
import numpy as np
import itertools
from coremltools._deps import HAS_KERAS2_TF, HAS_KERAS_TF
from nose.plugins.attrib import attr
np.random.seed(1377)

if HAS_KERAS2_TF or HAS_KERAS_TF:
    import keras
    from keras.models import Sequential, Model
    from keras.layers import LSTM, GRU, SimpleRNN, RepeatVector
    from keras.layers.wrappers import Bidirectional, TimeDistributed
    import keras.backend as K
    from coremltools.converters import keras as keras_converter


'''
=============================
Utility Functions
=============================
'''
    
def get_recurrent_activation_name_from_keras(activation):
    if activation == keras.activations.sigmoid:
        activation_str = 'SIGMOID'
    elif activation == keras.activations.hard_sigmoid:
        activation_str = 'SIGMOID_HARD'
    elif activation == keras.activations.tanh:
        activation_str = 'TANH'
    elif activation == keras.activations.relu:
        activation_str = 'RELU'
    elif activation == keras.activations.linear:
        activation_str = 'LINEAR'
    else:
        raise NotImplementedError('activation %s not supported for Recurrent layer.' % activation)
    return activation_str    

def linear(x,alpha=1,beta=0):
    return alpha*x + beta

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def hard_sigmoid(x,alpha=0.2,beta=0.5):
    return np.minimum(np.maximum(alpha * x + beta, 0), 1)

def tanh(x):
    return np.tanh(x)

def apply_act(x, option):
    if option == 'TANH':
        return tanh(x)
    elif option == 'RELU':
        return relu(x)
    elif option == 'SIGMOID':
        return sigmoid(x)
    elif option == 'SIGMOID_HARD':
        return hard_sigmoid(x)
    elif option == 'LINEAR':
        return linear(x)
        
def clip(x, threshold = 50.0):
    return np.maximum(np.minimum(x,threshold),-threshold)     
    
'''
=============================
Numpy implementations
=============================
'''           

def get_numpy_prediction_gru(model, X):
    X = X[0,:,:]
    seq_len, input_size = X.shape
    keras_layer = model.layers[0]    
    return_seq = keras_layer.return_sequences
    if keras_layer.go_backwards:
        X = X[::-1,:]
    
    if HAS_KERAS2_TF:
        hidden_size = keras_layer.units
        
        keras_W_h = keras_layer.get_weights()[1].T
        R_z = keras_W_h[0 * hidden_size:][:hidden_size]
        R_r = keras_W_h[1 * hidden_size:][:hidden_size]
        R_o = keras_W_h[2 * hidden_size:][:hidden_size]

        keras_W_x = keras_layer.get_weights()[0].T
        W_z = keras_W_x[0 * hidden_size:][:hidden_size]
        W_r = keras_W_x[1 * hidden_size:][:hidden_size]
        W_o = keras_W_x[2 * hidden_size:][:hidden_size]
        
        keras_b = keras_layer.get_weights()[2]
        b_z = keras_b[0 * hidden_size:][:hidden_size]
        b_r = keras_b[1 * hidden_size:][:hidden_size]
        b_o = keras_b[2 * hidden_size:][:hidden_size]
        
        inner_activation_str = get_recurrent_activation_name_from_keras(keras_layer.recurrent_activation)
        activation_str = get_recurrent_activation_name_from_keras(keras_layer.activation)
        
    else:
        hidden_size = keras_layer.output_dim
        
        W_z = keras_layer.get_weights()[0].T
        W_r = keras_layer.get_weights()[3].T
        W_o = keras_layer.get_weights()[6].T
        
        R_z = keras_layer.get_weights()[1].T
        R_r = keras_layer.get_weights()[4].T
        R_o = keras_layer.get_weights()[7].T

        b_z = keras_layer.get_weights()[2]
        b_r = keras_layer.get_weights()[5]
        b_o = keras_layer.get_weights()[8]
        
        inner_activation_str = get_recurrent_activation_name_from_keras(keras_layer.inner_activation)
        activation_str = get_recurrent_activation_name_from_keras(keras_layer.activation)
        
        
    h = np.zeros((hidden_size))
    c = np.zeros((hidden_size))
    np_out = np.zeros((seq_len,hidden_size))
    for k in range(seq_len):
        x = X[k,:]
        z = apply_act(clip(np.dot(W_z,x) + np.dot(R_z,h) + b_z), inner_activation_str)
        r = apply_act(clip(np.dot(W_r, x) + np.dot(R_r, h) + b_r), inner_activation_str)
        c = clip(h * r)
        o = apply_act(clip(np.dot(W_o, x) + np.dot(R_o, c) + b_o), activation_str)
        h = (1 - z) * o + z * h
        np_out[k,:] = h

    if return_seq:
        np_out_final = np_out
    else:
        np_out_final = np_out[-1,:]           
    return np_out_final

def get_numpy_prediction_unilstm(model, X):
    X = X[0,:,:]
    seq_len, input_size = X.shape
    keras_layer = model.layers[0]    
    return_seq = keras_layer.return_sequences
    if keras_layer.go_backwards:
        X = X[::-1,:]
    
    if HAS_KERAS2_TF:
        hidden_size = keras_layer.units
        
        keras_W_h = keras_layer.get_weights()[1].T
        R_i = keras_W_h[0 * hidden_size:][:hidden_size]
        R_f = keras_W_h[1 * hidden_size:][:hidden_size]
        R_o = keras_W_h[3 * hidden_size:][:hidden_size]
        R_g = keras_W_h[2 * hidden_size:][:hidden_size]

        keras_W_x = keras_layer.get_weights()[0].T
        W_i = keras_W_x[0 * hidden_size:][:hidden_size]
        W_f = keras_W_x[1 * hidden_size:][:hidden_size]
        W_o = keras_W_x[3 * hidden_size:][:hidden_size]
        W_g = keras_W_x[2 * hidden_size:][:hidden_size]
        
        keras_b = keras_layer.get_weights()[2]
        b_i = keras_b[0 * hidden_size:][:hidden_size]
        b_f = keras_b[1 * hidden_size:][:hidden_size]
        b_o = keras_b[3 * hidden_size:][:hidden_size]
        b_g = keras_b[2 * hidden_size:][:hidden_size]
        
        inner_activation_str = get_recurrent_activation_name_from_keras(keras_layer.recurrent_activation)
        activation_str = get_recurrent_activation_name_from_keras(keras_layer.activation)
        
    else:
        hidden_size = keras_layer.output_dim
        
        R_i = keras_layer.get_weights()[1].T
        R_f = keras_layer.get_weights()[7].T
        R_o = keras_layer.get_weights()[10].T
        R_g = keras_layer.get_weights()[4].T

        W_i = keras_layer.get_weights()[0].T
        W_f = keras_layer.get_weights()[6].T
        W_o = keras_layer.get_weights()[9].T
        W_g = keras_layer.get_weights()[3].T

        b_i = keras_layer.get_weights()[2]
        b_f = keras_layer.get_weights()[8]
        b_o = keras_layer.get_weights()[11]
        b_g = keras_layer.get_weights()[5]
        
        inner_activation_str = get_recurrent_activation_name_from_keras(keras_layer.inner_activation)
        activation_str = get_recurrent_activation_name_from_keras(keras_layer.activation)
        
    h = np.zeros((hidden_size))
    c = np.zeros((hidden_size))
    np_out = np.zeros((seq_len,hidden_size))
    for k in range(seq_len):
        x = X[k,:]
        i = apply_act(clip(np.dot(W_i,x) + np.dot(R_i,h) + b_i), inner_activation_str)
        f = apply_act(clip(np.dot(W_f,x) + np.dot(R_f,h) + b_f), inner_activation_str)
        g = apply_act(clip(np.dot(W_g,x) + np.dot(R_g,h) + b_g), activation_str)
        c = c * f + i * g
        c = clip(c, 50000.0)
        o = apply_act(clip(np.dot(W_o, x) + np.dot(R_o, h) + b_o), inner_activation_str)
        h = o * apply_act(c, activation_str)
        np_out[k,:] = h
    if return_seq:
        np_out_final = np_out
    else:
        np_out_final = np_out[-1,:]           
    return np_out_final

def get_numpy_prediction_bilstm(model, X):
    X = X[0,:,:]
    seq_len, input_size = X.shape
    keras_layer = model.layers[0] 
    return_seq = keras_layer.return_sequences
    
    if HAS_KERAS2_TF:
        hidden_size = keras_layer.forward_layer.units
        
        keras_W_h = keras_layer.forward_layer.get_weights()[1].T
        R_i = keras_W_h[0 * hidden_size:][:hidden_size]
        R_f = keras_W_h[1 * hidden_size:][:hidden_size]
        R_o = keras_W_h[3 * hidden_size:][:hidden_size]
        R_g = keras_W_h[2 * hidden_size:][:hidden_size]

        keras_W_x = keras_layer.forward_layer.get_weights()[0].T
        W_i = keras_W_x[0 * hidden_size:][:hidden_size]
        W_f = keras_W_x[1 * hidden_size:][:hidden_size]
        W_o = keras_W_x[3 * hidden_size:][:hidden_size]
        W_g = keras_W_x[2 * hidden_size:][:hidden_size]
        
        keras_b = keras_layer.forward_layer.get_weights()[2]
        b_i = keras_b[0 * hidden_size:][:hidden_size]
        b_f = keras_b[1 * hidden_size:][:hidden_size]
        b_o = keras_b[3 * hidden_size:][:hidden_size]
        b_g = keras_b[2 * hidden_size:][:hidden_size]
        
        keras_W_h = keras_layer.backward_layer.get_weights()[1].T
        R_i_back = keras_W_h[0 * hidden_size:][:hidden_size]
        R_f_back = keras_W_h[1 * hidden_size:][:hidden_size]
        R_o_back = keras_W_h[3 * hidden_size:][:hidden_size]
        R_g_back = keras_W_h[2 * hidden_size:][:hidden_size]

        keras_W_x = keras_layer.backward_layer.get_weights()[0].T
        W_i_back = keras_W_x[0 * hidden_size:][:hidden_size]
        W_f_back = keras_W_x[1 * hidden_size:][:hidden_size]
        W_o_back = keras_W_x[3 * hidden_size:][:hidden_size]
        W_g_back = keras_W_x[2 * hidden_size:][:hidden_size]
        
        keras_b = keras_layer.backward_layer.get_weights()[2]
        b_i_back = keras_b[0 * hidden_size:][:hidden_size]
        b_f_back = keras_b[1 * hidden_size:][:hidden_size]
        b_o_back = keras_b[3 * hidden_size:][:hidden_size]
        b_g_back = keras_b[2 * hidden_size:][:hidden_size]
        
        inner_activation_str = get_recurrent_activation_name_from_keras(keras_layer.forward_layer.recurrent_activation)
        activation_str = get_recurrent_activation_name_from_keras(keras_layer.forward_layer.activation)
        
    else:
        hidden_size = keras_layer.forward_layer.output_dim
        
        R_i = keras_layer.get_weights()[1].T
        R_f = keras_layer.get_weights()[7].T
        R_o = keras_layer.get_weights()[10].T
        R_g = keras_layer.get_weights()[4].T

        W_i = keras_layer.get_weights()[0].T
        W_f = keras_layer.get_weights()[6].T
        W_o = keras_layer.get_weights()[9].T
        W_g = keras_layer.get_weights()[3].T

        b_i = keras_layer.get_weights()[2]
        b_f = keras_layer.get_weights()[8]
        b_o = keras_layer.get_weights()[11]
        b_g = keras_layer.get_weights()[5]
        
        R_i_back = keras_layer.backward_layer.get_weights()[1].T
        R_f_back = keras_layer.backward_layer.get_weights()[7].T
        R_o_back = keras_layer.backward_layer.get_weights()[10].T
        R_g_back = keras_layer.backward_layer.get_weights()[4].T

        W_i_back = keras_layer.backward_layer.get_weights()[0].T
        W_f_back = keras_layer.backward_layer.get_weights()[6].T
        W_o_back = keras_layer.backward_layer.get_weights()[9].T
        W_g_back = keras_layer.backward_layer.get_weights()[3].T

        b_i_back = keras_layer.backward_layer.get_weights()[2]
        b_f_back = keras_layer.backward_layer.get_weights()[8]
        b_o_back = keras_layer.backward_layer.get_weights()[11]
        b_g_back = keras_layer.backward_layer.get_weights()[5]
        
        inner_activation_str = get_recurrent_activation_name_from_keras(keras_layer.forward_layer.inner_activation)
        activation_str = get_recurrent_activation_name_from_keras(keras_layer.forward_layer.activation)
        
    h = np.zeros((hidden_size))
    c = np.zeros((hidden_size))
    np_out_forward = np.zeros((seq_len,hidden_size))
    for k in range(seq_len):
        x = X[k,:]
        i = apply_act(clip(np.dot(W_i,x) + np.dot(R_i,h) + b_i), inner_activation_str)
        f = apply_act(clip(np.dot(W_f,x) + np.dot(R_f,h) + b_f), inner_activation_str)
        g = apply_act(clip(np.dot(W_g,x) + np.dot(R_g,h) + b_g), activation_str)
        c = c * f + i * g
        c = clip(c, 50000.0)
        o = apply_act(clip(np.dot(W_o, x) + np.dot(R_o, h) + b_o), inner_activation_str)
        h = o * apply_act(c, activation_str)
        np_out_forward[k,:] = h
        
    h = np.zeros((hidden_size))
    c = np.zeros((hidden_size))
    np_out_backward = np.zeros((seq_len,hidden_size))
    for k in range(seq_len):
        x = X[seq_len - k - 1,:]
        i = apply_act(clip(np.dot(W_i_back,x) + np.dot(R_i_back,h) + b_i_back), inner_activation_str)
        f = apply_act(clip(np.dot(W_f_back,x) + np.dot(R_f_back,h) + b_f_back), inner_activation_str)
        g = apply_act(clip(np.dot(W_g_back,x) + np.dot(R_g_back,h) + b_g_back), activation_str)
        c = c * f + i * g
        c = clip(c, 50000.0)
        o = apply_act(clip(np.dot(W_o_back, x) + np.dot(R_o_back, h) + b_o_back), inner_activation_str)
        h = o * apply_act(c, activation_str)
        np_out_backward[k,:] = h    
          
    if return_seq:
        np_out_final = np.zeros((seq_len,2*hidden_size))
        for k in range(seq_len):
            np_out_final[k, :hidden_size] = np_out_forward[k, :]
            np_out_final[k, hidden_size:] = np_out_backward[seq_len - k - 1, :]
    else:
        np_out_final = np.zeros((1, 2 * hidden_size))
        np_out_final[0,:hidden_size] = np_out_forward[-1,:]
        np_out_final[0,hidden_size:] = np_out_backward[-1,:]          
    return np_out_final

'''
=============================
Nosetest Functions
=============================
'''  

def _get_mlkit_model_from_path(model):
    from coremltools.converters import keras as keras_converter
    model = keras_converter.convert(model, ['data'], ['output'])
    return model


def generate_input(dim0, dim1, dim2):
    input_data = np.random.rand(dim0, dim1, dim2).astype('f') #astype() should be removed after radar://31569743
    return input_data


def simple_model_eval(params, model):
    mlkitmodel = _get_mlkit_model_from_path(model)
    # New test case takes in 2D input as opposed to uniform 3d input across all other tests
    if len(params[0]['input_dims']) == 3:
        input_data = generate_input(params[0]['input_dims'][0], params[0]['input_dims'][1],
                                    params[0]['input_dims'][2])
        keras_preds = model.predict(input_data).flatten()
    elif len(params[0]['input_dims']) == 2:
        input_data = np.squeeze(np.random.rand(params[0]['input_dims'][0], params[0]['input_dims'][1]))
        keras_preds = model.predict(input_data.reshape((params[0]['input_dims'][0], params[0]['input_dims'][1]))).flatten()
    if len(params[0]['input_dims']) == 3:
        input_data = np.transpose(input_data, [1, 0, 2])
    coreml_preds = mlkitmodel.predict({'data': input_data})['output'].flatten()
    if K.tensorflow_backend._SESSION:
        import tensorflow as tf
        tf.reset_default_graph()
        K.tensorflow_backend._SESSION.close()
        K.tensorflow_backend._SESSION = None
        
    max_denominator = np.maximum(np.maximum(np.abs(coreml_preds), np.abs(keras_preds)),1.0)
    relative_error = coreml_preds/max_denominator - keras_preds/max_denominator   
    return relative_error, keras_preds, coreml_preds


class SimpleTestCase(unittest.TestCase):
    """
    Test Simple test cases to make sure layers work under basic params. Also, template for testing
    different failing test cases from stress tests
    """
    def test_SimpleRNN(self):
        params = dict(
            input_dims=[1, 2, 100], go_backwards=False, activation='tanh',
            stateful=False, unroll=False, return_sequences=True, output_dim=4  # Passes for < 3
        ),
        model = Sequential()
        if keras.__version__[:2] == '2.':
            model.add(SimpleRNN(units=params[0]['output_dim'],
                                input_shape=(params[0]['input_dims'][1],params[0]['input_dims'][2]),
                                activation=params[0]['activation'],
                                return_sequences=params[0]['return_sequences'],
                                go_backwards=params[0]['go_backwards'],
                                unroll=True,
                                ))            
        else:
            model.add(SimpleRNN(output_dim=params[0]['output_dim'],
                                input_length=params[0]['input_dims'][1],
                                input_dim=params[0]['input_dims'][2],
                                activation=params[0]['activation'],
                                return_sequences=params[0]['return_sequences'],
                                go_backwards=params[0]['go_backwards'],
                                unroll=True,
                                ))
        relative_error, keras_preds, coreml_preds = simple_model_eval(params, model)
        for i in range(len(relative_error)):
            self.assertLessEqual(relative_error[i], 0.01)

    def test_SimpleLSTM(self):
        params = dict(
            input_dims=[1, 3, 5], go_backwards=True, activation='linear',
            stateful=False, unroll=False, return_sequences=False, output_dim=3,
            inner_activation='linear'
        ),
        model = Sequential()
        if keras.__version__[:2] == '2.':
            model.add(LSTM(units=params[0]['output_dim'],
                           input_shape=(params[0]['input_dims'][1],params[0]['input_dims'][2]),
                           activation=params[0]['activation'],
                           return_sequences=params[0]['return_sequences'],
                           go_backwards=params[0]['go_backwards'],
                           unroll=True,
                           recurrent_activation='linear'
                           ))
        else:
            model.add(LSTM(output_dim=params[0]['output_dim'],
                           input_length=params[0]['input_dims'][1],
                           input_dim=params[0]['input_dims'][2],
                           activation=params[0]['activation'],
                           return_sequences=params[0]['return_sequences'],
                           go_backwards=params[0]['go_backwards'],
                           unroll=True,
                           inner_activation='linear'
                           ))
        relative_error, keras_preds, coreml_preds = simple_model_eval(params, model)
        for i in range(len(relative_error)):
            self.assertLessEqual(relative_error[i], 0.01)

    def test_SimpleGRU(self):
        params = dict(
            input_dims=[1, 4, 8], go_backwards=False, activation='tanh',  
            stateful=False, unroll=False, return_sequences=False, output_dim=4
        ),
        model = Sequential()
        if keras.__version__[:2] == '2.':
             model.add(GRU(units=params[0]['output_dim'],
                           input_shape=(params[0]['input_dims'][1],params[0]['input_dims'][2]),
                           activation=params[0]['activation'],
                           recurrent_activation='sigmoid',
                           return_sequences=params[0]['return_sequences'],
                           go_backwards=params[0]['go_backwards'],
                           unroll=True,
                           ))
        else:
            model.add(GRU(output_dim=params[0]['output_dim'],
                          input_length=params[0]['input_dims'][1],
                          input_dim=params[0]['input_dims'][2],
                          activation=params[0]['activation'],
                          inner_activation='sigmoid',
                          return_sequences=params[0]['return_sequences'],
                          go_backwards=params[0]['go_backwards'],
                          unroll=True,
                          ))
        model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
        relative_error, keras_preds, coreml_preds = simple_model_eval(params, model)
        for i in range(len(relative_error)):
            self.assertLessEqual(relative_error[i], 0.01)

    @attr('keras2')
    def test_keras2_SimpleRNN(self):
        self.test_SimpleRNN()

    @attr('keras2')
    def test_keras2_SimpleLSTM(self):
        self.test_SimpleLSTM()

    @attr('keras2')
    def test_keras2_SimpleGRU(self):
        self.test_SimpleGRU()


class RecurrentLayerTest(unittest.TestCase):
    """
    Base class for recurrent layer tests. Masking param not included here
    """

    def setUp(self):

        self.params_dict = dict(
            input_dims=[[1, 5, 10], [1, 1, 1], [1, 2, 5]],
            output_dim=[1, 5, 10],
            stateful = [False],
            go_backwards=[False, True],
            unroll=[True],
            return_sequences=[False, True],
            activation=['tanh', 'linear', 'sigmoid', 'hard_sigmoid', 'relu'],
        )
        self.base_layer_params = list(itertools.product(*self.params_dict.values()))

@attr('slow')
class RNNLayer(RecurrentLayerTest):
    """
    Class for testing single RNN layer
    """
    def setUp(self):
        super(RNNLayer, self).setUp()
        self.simple_rnn_params_dict = dict()
        self.rnn_layer_params = list(itertools.product(self.simple_rnn_params_dict.values()))

    def test_rnn_layer(self):
        i = 0
        numerical_err_models = []
        shape_err_models = []
        numerical_failiure = 0
        for base_params in self.base_layer_params:
            base_params = dict(zip(self.params_dict.keys(), base_params))
            for rnn_params in self.rnn_layer_params:
                rnn_params = dict(zip(self.simple_rnn_params_dict.keys(), rnn_params))
                model = Sequential()
                model.add(
                    SimpleRNN(
                        base_params['output_dim'],
                        input_length=base_params['input_dims'][1],
                        input_dim=base_params['input_dims'][2],
                        activation=base_params['activation'],
                        return_sequences=base_params['return_sequences'],
                        go_backwards=base_params['go_backwards'],
                        unroll=base_params['unroll'],
                    )
                )
                mlkitmodel = _get_mlkit_model_from_path(model)
                input_data = generate_input(base_params['input_dims'][0], base_params['input_dims'][1],
                                            base_params['input_dims'][2])
                keras_preds = model.predict(input_data).flatten()
                if K.tensorflow_backend._SESSION:
                    import tensorflow as tf
                    tf.reset_default_graph()
                    K.tensorflow_backend._SESSION.close()
                    K.tensorflow_backend._SESSION = None
                input_data = np.transpose(input_data, [1, 0, 2])
                coreml_preds = mlkitmodel.predict({'data': input_data})['output'].flatten()
                try:
                    self.assertEquals(coreml_preds.shape, keras_preds.shape)
                except AssertionError:
                    print("Shape error:\nbase_params: {}\nkeras_preds.shape: {}\ncoreml_preds.shape: {}".format(
                        base_params, keras_preds.shape, coreml_preds.shape))
                    shape_err_models.append(base_params)
                    i += 1
                    continue
                try:
                    max_denominator = np.maximum(np.maximum(np.abs(coreml_preds), np.abs(keras_preds)),1.0)
                    relative_error = coreml_preds/max_denominator - keras_preds/max_denominator
                    for i in range(len(relative_error)):
                        self.assertLessEqual(relative_error[i], 0.01)
                except AssertionError:
                    print("Assertion error:\nbase_params: {}\nkeras_preds: {}\ncoreml_preds: {}".format(base_params, keras_preds, coreml_preds))
                    numerical_failiure += 1
                    numerical_err_models.append(base_params)
                i += 1

        self.assertEquals(shape_err_models, [], msg='Shape error models {}'.format(shape_err_models))
        self.assertEquals(numerical_err_models, [], msg='Numerical error models {}\n'
                                                        'Total numerical failiures: {}/{}\n'.format(
            numerical_err_models,
            numerical_failiure, i)
                          )
    @attr('keras2')
    def test_keras2_rnn_layer(self):
        self.test_rnn_layer()

@attr('slow')
class LSTMLayer(RecurrentLayerTest):
    """
    Class for testing single RNN layer
    """
    def setUp(self):
        super(LSTMLayer, self).setUp()
        self.lstm_params_dict = dict(
            inner_activation=['tanh', 'linear', 'sigmoid', 'hard_sigmoid', 'relu'],
            bidirectional=[True, False],
        )
        self.lstm_layer_params = list(itertools.product(*self.lstm_params_dict.values()))

    def test_lstm_layer(self):
        i = 0
        numerical_err_models = []
        shape_err_models = []
        numerical_failiure = 0
        for base_params in self.base_layer_params:
            base_params = dict(zip(self.params_dict.keys(), base_params))
            for lstm_params in self.lstm_layer_params:
                lstm_params = dict(zip(self.lstm_params_dict.keys(), lstm_params))
                model = Sequential()
                if lstm_params['bidirectional'] is True:
                    if keras.__version__[:2] == '2.':
                        model.add(
                            Bidirectional(
                                LSTM(
                                    base_params['output_dim'],
                                    activation=base_params['activation'],
                                    recurrent_activation=lstm_params['inner_activation'],
                                    return_sequences=base_params['return_sequences'],
                                    go_backwards=False,
                                    unroll=base_params['unroll'],
                                ),
                                input_shape=(base_params['input_dims'][1], base_params['input_dims'][2]),

                            )
                        )
                    else:
                        model.add(
                            Bidirectional(
                                LSTM(
                                    base_params['output_dim'],
                                    activation=base_params['activation'],
                                    inner_activation=lstm_params['inner_activation'],
                                    return_sequences=base_params['return_sequences'],
                                    go_backwards=False,
                                    unroll=base_params['unroll'],
                                ),
                                input_shape=(base_params['input_dims'][1], base_params['input_dims'][2]),

                            )
                        )
                else:
                    if keras.__version__[:2] == '2.':
                        model.add(
                            LSTM(
                                base_params['output_dim'],
                                input_shape=(base_params['input_dims'][1], base_params['input_dims'][2]),
                                activation=base_params['activation'],
                                recurrent_activation=lstm_params['inner_activation'],
                                return_sequences=base_params['return_sequences'],
                                go_backwards=base_params['go_backwards'],
                                unroll=base_params['unroll'],
                            )
                        )
                    else:
                        model.add(
                            LSTM(
                                base_params['output_dim'],
                                input_shape=(base_params['input_dims'][1], base_params['input_dims'][2]),
                                activation=base_params['activation'],
                                inner_activation=lstm_params['inner_activation'],
                                return_sequences=base_params['return_sequences'],
                                go_backwards=base_params['go_backwards'],
                                unroll=base_params['unroll'],
                            )
                        )
                mlkitmodel = _get_mlkit_model_from_path(model)
                input_data = generate_input(base_params['input_dims'][0], base_params['input_dims'][1],
                                            base_params['input_dims'][2])
                
                activations_to_test_with_numpy = {'linear', 'relu'}
                if base_params['activation'] in activations_to_test_with_numpy or lstm_params['inner_activation'] in activations_to_test_with_numpy:
                    if lstm_params['bidirectional']:
                        keras_preds = get_numpy_prediction_bilstm(model, input_data).flatten()
                    else:
                        keras_preds = get_numpy_prediction_unilstm(model, input_data).flatten()                    
                else:
                    keras_preds = model.predict(input_data).flatten()
                    
                input_data = np.transpose(input_data, [1, 0, 2])
                coreml_preds = mlkitmodel.predict({'data': input_data})['output'].flatten()

                if K.tensorflow_backend._SESSION:
                    import tensorflow as tf
                    tf.reset_default_graph()
                    K.tensorflow_backend._SESSION.close()
                    K.tensorflow_backend._SESSION = None
                    
                try:
                    self.assertEquals(coreml_preds.shape, keras_preds.shape)
                except AssertionError:
                    print("Shape error:\n base_params: {}\n\n lstm_params: {}\n\n keras_preds.shape: {}\n\n coreml_preds.shape: {}".format(
                        base_params, lstm_params, keras_preds.shape, coreml_preds.shape))
                    shape_err_models.append(base_params)
                    i += 1
                    continue
                try:
                    max_denominator = np.maximum(np.maximum(np.abs(coreml_preds), np.abs(keras_preds)),1.0)
                    relative_error = coreml_preds/max_denominator - keras_preds/max_denominator
                    for i in range(len(relative_error)):
                        self.assertLessEqual(relative_error[i], 0.01)
                except AssertionError:
                    print("Assertion error:\n base_params: {}\n lstm_params: {}\n\n keras_preds: {}\n\n coreml_preds: {}\n\n\n keras_preds: {}\n\n\n coreml_preds: {}\n".format(base_params,
                                                                                                        lstm_params,    
                                                                                                        keras_preds/max_denominator, 
                                                                                                        coreml_preds/max_denominator,
                                                                                                        keras_preds,
                                                                                                        coreml_preds))
                    numerical_failiure += 1
                    numerical_err_models.append(base_params)        
                i += 1

        self.assertEquals(shape_err_models, [], msg='Shape error models {}'.format(shape_err_models))
        self.assertEquals(numerical_err_models, [], msg='Numerical error models {}'.format(numerical_err_models))
        
    @attr('keras2')
    def test_keras2_lstm_layer(self):
        self.test_lstm_layer()

@attr('slow')
class GRULayer(RecurrentLayerTest):
    """
    Class for testing GRU layer
    """
    def setUp(self):
        super(GRULayer, self).setUp()
        self.gru_params_dict = dict(
            inner_activation=['tanh', 'linear', 'sigmoid', 'hard_sigmoid', 'relu']
        )
        self.gru_layer_params = list(itertools.product(*self.gru_params_dict.values()))

    def test_gru_layer(self):
        i = 0
        numerical_err_models = []
        shape_err_models = []
        numerical_failiure = 0
        for base_params in self.base_layer_params:
            base_params = dict(zip(self.params_dict.keys(), base_params))
            for gru_params in self.gru_layer_params:
                gru_params = dict(zip(self.gru_params_dict.keys(), gru_params))
                model = Sequential()
                if keras.__version__[:2] == '2.':
                    model.add(
                        GRU(
                            base_params['output_dim'],
                            input_shape=(base_params['input_dims'][1],base_params['input_dims'][2]),
                            activation=base_params['activation'],
                            recurrent_activation=gru_params['inner_activation'],
                            return_sequences=base_params['return_sequences'],
                            go_backwards=base_params['go_backwards'],
                            unroll=base_params['unroll'],
                        )
                    )
                else:
                    model.add(
                        GRU(
                            base_params['output_dim'],
                            input_length=base_params['input_dims'][1],
                            input_dim=base_params['input_dims'][2],
                            activation=base_params['activation'],
                            inner_activation=gru_params['inner_activation'],
                            return_sequences=base_params['return_sequences'],
                            go_backwards=base_params['go_backwards'],
                            unroll=base_params['unroll'],
                        )
                    )
                model.set_weights([np.random.rand(*w.shape) for w in model.get_weights()])
                mlkitmodel = _get_mlkit_model_from_path(model)
                input_data = generate_input(base_params['input_dims'][0], base_params['input_dims'][1],
                                            base_params['input_dims'][2])
                
                activations_to_test_with_numpy = {'linear', 'relu'}
                if base_params['activation'] in activations_to_test_with_numpy or gru_params['inner_activation'] in activations_to_test_with_numpy:
                    keras_preds = get_numpy_prediction_gru(model, input_data).flatten()                    
                else:
                    keras_preds = model.predict(input_data).flatten()
                
                input_data = np.transpose(input_data, [1, 0, 2])
                coreml_preds = mlkitmodel.predict({'data': input_data})['output'].flatten()
                if K.tensorflow_backend._SESSION:
                    import tensorflow as tf
                    tf.reset_default_graph()
                    K.tensorflow_backend._SESSION.close()
                    K.tensorflow_backend._SESSION = None
                try:
                    self.assertEquals(coreml_preds.shape, keras_preds.shape)
                except AssertionError:
                    print("Shape error:\nbase_params: {}\n gru_params: {}\nkeras_preds.shape: {}\ncoreml_preds.shape: {}".format(
                        base_params, gru_params, keras_preds.shape, coreml_preds.shape))
                    shape_err_models.append(base_params)
                    i += 1
                    continue
                try:
                    max_denominator = np.maximum(np.maximum(np.abs(coreml_preds), np.abs(keras_preds)), 1.0)
                    relative_error = coreml_preds/max_denominator - keras_preds/max_denominator
                    for i in range(len(relative_error)):
                        self.assertLessEqual(relative_error[i], 0.01)
                except AssertionError:
                    print("===============Assertion error:\n base_params: {}\n gru_params: {}\n\n keras_preds: {}\n\n coreml_preds: {}\n\n\n keras_preds: {}\n\n\n coreml_preds: {}\n".format(base_params,
                                                                                                        gru_params,    
                                                                                                        keras_preds/max_denominator, 
                                                                                                        coreml_preds/max_denominator,
                                                                                                        keras_preds,
                                                                                                        coreml_preds))
                    numerical_failiure += 1
                    numerical_err_models.append(base_params)
                i += 1

        self.assertEquals(shape_err_models, [], msg='Shape error models {}'.format(shape_err_models))
        self.assertEquals(numerical_err_models, [], msg='Numerical error models {}'.format(numerical_err_models))

    @attr('keras2')
    def test_keras2_test_gru_layer(self):
        self.test_gru_layer()


@attr('slow')
class LSTMStacked(unittest.TestCase):
    """
    Class for testing LSTMStacked
    """
    def setUp(self):
        self.params_dict = dict(
            input_dims=[[1, 1, 1], [1, 2, 5], [1, 5, 10]],
            output_dim=[1, 5, 10, 20],
            stateful=[False],
            go_backwards=[False],
            unroll=[True],
            return_sequences=[True],
            top_return_sequences=[True, False],
            activation=['tanh','sigmoid', 'hard_sigmoid'],
            number_of_layers=[1, 2, 3]
        )
        self.base_layer_params = list(itertools.product(*self.params_dict.values()))

    def test_SimpleLSTMStacked(self):
        params = dict(
            input_dims=[1, 1, 1], go_backwards=False, activation='tanh',  
            stateful=False, unroll=False, return_sequences=False, output_dim=1
        ),
        model = Sequential()
        model.add(LSTM(output_dim=params[0]['output_dim'],
                            input_length=params[0]['input_dims'][1],
                            input_dim=params[0]['input_dims'][2],
                            activation=params[0]['activation'],
                            inner_activation = 'sigmoid',
                            return_sequences=True,
                            go_backwards=params[0]['go_backwards'],
                            unroll=params[0]['unroll'],
                            ))
        model.add(LSTM(output_dim=1,
                       activation='tanh',
                       inner_activation = 'sigmoid',
                       ))
        relative_error, keras_preds, coreml_preds = simple_model_eval(params, model)
        for i in range(len(relative_error)):
            self.assertLessEqual(relative_error[i], 0.01)

    def test_StressLSTMStacked(self):
        numerical_err_models = []
        shape_err_models = []
        numerical_failiure = 0
        i= 0
        for base_params in self.base_layer_params:
            base_params = dict(zip(self.params_dict.keys(), base_params))
            model = Sequential()
            model.add(LSTM(output_dim=base_params['output_dim'],
                           input_length=base_params['input_dims'][1],
                           input_dim=base_params['input_dims'][2],
                           activation=base_params['activation'],
                           inner_activation = 'sigmoid',
                           return_sequences=True,
                           go_backwards=base_params['go_backwards'],
                           unroll=base_params['unroll'],
                           ))
            for idx in range(0, base_params['number_of_layers']):
                try:
                    model.add(LSTM(output_dim=base_params["output_dim"],
                                   return_sequences=True,
                                   activation='tanh',
                                   inner_activation = 'sigmoid',
                               ))
                except ValueError as e:
                    print("error ocurred with {}".format(e))
            model.add(LSTM(output_dim=10, return_sequences=base_params['top_return_sequences'], activation='sigmoid'))
            mlkitmodel = _get_mlkit_model_from_path(model)
            input_data = generate_input(base_params['input_dims'][0], base_params['input_dims'][1],
                                        base_params['input_dims'][2])
            keras_preds = model.predict(input_data).flatten()
            input_data = np.transpose(input_data, [1, 0, 2])
            coreml_preds = mlkitmodel.predict({'data': input_data})['output'].flatten()
            import tensorflow as tf
            tf.reset_default_graph()
            K.tensorflow_backend._SESSION.close()
            K.tensorflow_backend._SESSION = None
            try:
                self.assertEquals(coreml_preds.shape, keras_preds.shape)
            except AssertionError:
                print("Shape error:\nbase_params: {}\nkeras_preds.shape: {}\ncoreml_preds.shape: {}".format(
                    base_params, keras_preds.shape, coreml_preds.shape))
                shape_err_models.append(base_params)
                i += 1
                continue
            try:
                max_denominator = np.maximum(np.maximum(np.abs(coreml_preds), np.abs(keras_preds)),1.0)
                relative_error = coreml_preds/max_denominator - keras_preds/max_denominator
                for i in range(len(relative_error)):
                    self.assertLessEqual(relative_error[i], 0.01)
            except AssertionError:
                print("Assertion error:\nbase_params: {}\nkeras_preds: {}\ncoreml_preds: {}".format(base_params,
                                                                                                    keras_preds,
                                                                                                    coreml_preds))
                numerical_failiure += 1
                numerical_err_models.append(base_params)
            i += 1
        self.assertEquals(shape_err_models, [], msg='Shape error models {}'.format(shape_err_models))
        self.assertEquals(numerical_err_models, [], msg='Numerical error models {}'.format(numerical_err_models))

    @attr('keras2')
    def test_keras2_SimpleLSTMStacked(self):
        self.test_SimpleLSTMStacked()

    @attr('keras2')
    def test_keras2_StressLSTMStacked(self):
        self.test_StressLSTMStacked()


class DifferentIOModelsTypes(unittest.TestCase):
    """
    Class for testing different I/O combinations for LSTMS
    """
    def test_one_to_many(self):
        params = dict(
            input_dims=[1, 10], activation='tanh', 
            return_sequences=False, output_dim=3
        ),
        number_of_times = 4
        model = Sequential()
        model.add(RepeatVector(number_of_times, input_shape=(10,)))

        model.add(LSTM(output_dim=params[0]['output_dim'],
                       activation=params[0]['activation'],
                       inner_activation = 'sigmoid',
                       return_sequences=True,
                       ))
        relative_error, keras_preds, coreml_preds = simple_model_eval(params, model)
        #print relative_error, '\n', keras_preds, '\n', coreml_preds, '\n' 
        for i in range(len(relative_error)):
            self.assertLessEqual(relative_error[i], 0.01)

    def test_many_to_one(self):
        params = dict(
            input_dims=[1, 10, 5], go_backwards=False, activation='tanh',  # fails with hard_sigmoid
            stateful=False, unroll=False, return_sequences=False, output_dim=1
        ),
        model = Sequential()
        model.add(LSTM(output_dim=params[0]['output_dim'],
                       input_shape=(10, 5),
                       activation=params[0]['activation'],
                       inner_activation = 'sigmoid',
                       ))
        relative_error, keras_preds, coreml_preds = simple_model_eval(params, model)
        #print relative_error, '\n', keras_preds, '\n', coreml_preds, '\n' 
        for i in range(len(relative_error)):
            self.assertLessEqual(relative_error[i], 0.01)

    def test_many_to_many(self):
        params = dict(
            input_dims=[1, 10, 5], go_backwards=False, activation='tanh',  # fails with hard_sigmoid
            stateful=False, unroll=False, return_sequences=True, output_dim=1
        ),
        model = Sequential()
        model.add(LSTM(output_dim=params[0]['output_dim'],
                       input_shape=(10, 5),
                       activation=params[0]['activation'],
                       inner_activation = 'sigmoid',
                       return_sequences=True,
                       ))
        relative_error, keras_preds, coreml_preds = simple_model_eval(params, model)
        #print relative_error, '\n', keras_preds, '\n', coreml_preds, '\n' 
        for i in range(len(relative_error)):
            self.assertLessEqual(relative_error[i], 0.01)

    @attr('keras2')
    def test_keras2_test_one_to_many(self):
        self.test_one_to_many()

    @attr('keras2')
    def test_keras2_test_many_to_one(self):
        self.test_many_to_one()

    @attr('keras2')
    def test_keras2_many_to_many(self):
        self.test_many_to_many()
