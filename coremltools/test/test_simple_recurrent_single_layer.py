# Copyright (c) 2017, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import unittest
import numpy as np
import os
import shutil
import tempfile
import itertools
import coremltools
from coremltools._deps import HAS_KERAS_TF
from nose.plugins.attrib import attr


if HAS_KERAS_TF:
    from keras.models import Sequential
    from keras.layers import LSTM, GRU, SimpleRNN
    from coremltools.converters import keras as keras_converter


def _get_mlkit_model_from_path(model, model_path):
    from coremltools.converters import keras as keras_converter
    print('converting')
    model = keras_converter.convert(model, ['data'], ['output'])
    return model


def generate_input(dim0, dim1, dim2):
    input_data = np.random.rand(dim0, dim1, dim2).astype('f') #astype() should be removed after radar://31569743
    return input_data


@unittest.skipIf(not HAS_KERAS_TF, 'Missing keras. Skipping tests.')
class RecurrentLayerTest(unittest.TestCase):
    """
    Base class for recurrent layer tests. Masking param not included here
    """

    def setUp(self):
        self.params_dict = dict(
            input_dims=[[1, 1, 1], [1, 1, 5], [1, 1, 10]],  # [1, x > 1, y] not added due to radars: 31741361[batch_size, time_steps, dimensions]
            output_dim=[1, 5, 10, 20],
            stateful = [False, True],
            go_backwards=[False], #True],
            unroll=[False, True],
            return_sequences=[False, True],
            activation=['sigmoid', 'tanh', 'hard_sigmoid', 'linear']
        )
        self.base_layer_params = list(itertools.product(*self.params_dict.values()))

@unittest.skipIf(not HAS_KERAS_TF, 'Missing keras. Skipping tests.')
class SimpleRNNLayer(RecurrentLayerTest):
    """
    Class for testing single RNN layer
    """
    def setUp(self):
        super(SimpleRNNLayer, self).setUp()
        self.simple_rnn_params_dict = dict(
            dropout=[{'dropout_W': 0.,'dropout_U':0. }],
            regularizer=[{'W_regularizer': None, 'U_regularizer': None, 'b_regularizer': None}],
        )
        self.rnn_layer_params = list(itertools.product(*self.simple_rnn_params_dict.values()))

    @attr('slow')
    def test_rnn_layer_stress(self):
        self._test_rnn_layer(True)

    def test_rnn_layer(self):
        self._test_rnn_layer(False)

    def _test_rnn_layer(self, allow_slow):
        i = 0
        layer_name = str(SimpleRNN).split('.')[3].split("'>")[0]
        numerical_err_models = []
        shape_err_models = []
        for base_params in self.base_layer_params:
            base_params = dict(zip(self.params_dict.keys(), base_params))
            input_data = generate_input(base_params['input_dims'][0], base_params['input_dims'][1],
                                        base_params['input_dims'][2])
            for rnn_params in self.rnn_layer_params:
                rnn_params = dict(zip(self.simple_rnn_params_dict.keys(), rnn_params))
                model = Sequential()
                model.add(
                    SimpleRNN(
                        base_params['output_dim'],
                        input_length=base_params['input_dims'][1],
                        input_dim=base_params['input_dims'][2],
                        # init=base_params['init'],
                        # inner_init=base_params['inner_init'],
                        activation=base_params['activation'],
                        return_sequences=base_params['return_sequences'],
                        go_backwards=base_params['go_backwards'],
                        unroll=base_params['unroll'],
                        dropout_U=rnn_params['dropout']['dropout_U'],
                        dropout_W=rnn_params['dropout']['dropout_W'],
                        W_regularizer=rnn_params['regularizer']['W_regularizer'],
                        U_regularizer=rnn_params['regularizer']['U_regularizer'],
                        b_regularizer=rnn_params['regularizer']['b_regularizer'],
                    )
                )
                model_dir = tempfile.mkdtemp()
                keras_model_path = os.path.join(model_dir, 'keras.h5')
                coreml_model_path = os.path.join(model_dir, 'keras.mlmodel')
                model.save_weights(keras_model_path)
                mlkitmodel = _get_mlkit_model_from_path(model, coreml_model_path)
                keras_preds = model.predict(input_data).flatten()
                input_data = np.transpose(input_data, [1, 0, 2])
                coreml_preds = mlkitmodel.predict({'data': input_data})['output'].flatten()
                try:
                    self.assertEquals(coreml_preds.shape, keras_preds.shape)
                except AssertionError:
                    print("Shape error:\nbase_params: {}\nkeras_preds.shape: {}\ncoreml_preds.shape: {}".format(
                        base_params, keras_preds.shape, coreml_preds.shape))
                    shape_err_models.append(base_params)
                    shutil.rmtree(model_dir)
                    i += 1
                    continue
                try:
                    for idx in range(0, len(coreml_preds)):
                        relative_error = (coreml_preds[idx] - keras_preds[idx]) / coreml_preds[idx]
                        self.assertAlmostEqual(relative_error, 0, places=2)
                except AssertionError:
                    print("Assertion error:\nbase_params: {}\nkeras_preds: {}\ncoreml_preds: {}".format(base_params, keras_preds, coreml_preds))
                    numerical_err_models.append(base_params)
                shutil.rmtree(model_dir)
                i += 1

                if not allow_slow:
                    break

            if not allow_slow:
                break

        self.assertEquals(shape_err_models, [], msg='Shape error models {}'.format(shape_err_models))
        self.assertEquals(numerical_err_models, [], msg='Numerical error models {}'.format(numerical_err_models))

@unittest.skipIf(not HAS_KERAS_TF, 'Missing keras. Skipping tests.')
class LSTMLayer(RecurrentLayerTest):
    """
    Class for testing single RNN layer
    """
    def setUp(self):
        super(LSTMLayer, self).setUp()
        self.lstm_params_dict = dict(
            dropout=[{'dropout_W': 0.,'dropout_U':0. }],
            regularizer=[{'W_regularizer': None, 'U_regularizer': None, 'b_regularizer': None}],
        )
        self.lstm_layer_params = list(itertools.product(*self.lstm_params_dict.values()))

    @attr('slow')
    def test_lstm_layer_stress(self):
        self._test_lstm_layer(True)

    def test_lstm_layer(self):
        self._test_lstm_layer(False)

    def _test_lstm_layer(self, allow_slow):
        i = 0
        numerical_err_models = []
        shape_err_models = []
        for base_params in self.base_layer_params:
            base_params = dict(zip(self.params_dict.keys(), base_params))
            input_data = generate_input(base_params['input_dims'][0], base_params['input_dims'][1],
                                        base_params['input_dims'][2])
            for lstm_params in self.lstm_layer_params:
                lstm_params = dict(zip(self.lstm_params_dict.keys(), lstm_params))
                model = Sequential()
                model.add(
                    LSTM(
                        base_params['output_dim'],
                        input_length=base_params['input_dims'][1],
                        input_dim=base_params['input_dims'][2],
                        # init=base_params['init'],
                        # inner_init=base_params['inner_init'],
                        activation=base_params['activation'],
                        return_sequences=base_params['return_sequences'],
                        go_backwards=base_params['go_backwards'],
                        unroll=base_params['unroll'],
                        dropout_U=lstm_params['dropout']['dropout_U'],
                        dropout_W=lstm_params['dropout']['dropout_W'],
                        W_regularizer=lstm_params['regularizer']['W_regularizer'],
                        U_regularizer=lstm_params['regularizer']['U_regularizer'],
                        b_regularizer=lstm_params['regularizer']['b_regularizer'],
                    )
                )
                model_dir = tempfile.mkdtemp()
                keras_model_path = os.path.join(model_dir, 'keras.h5')
                coreml_model_path = os.path.join(model_dir, 'keras.mlmodel')
                model.save_weights(keras_model_path)
                mlkitmodel = _get_mlkit_model_from_path(model, coreml_model_path)
                keras_preds = model.predict(input_data).flatten()
                input_data = np.transpose(input_data, [1, 0, 2])
                coreml_preds = mlkitmodel.predict({'data': input_data})['output'].flatten()
                try:
                    self.assertEquals(coreml_preds.shape, keras_preds.shape)
                except AssertionError:
                    print("Shape error:\nbase_params: {}\nkeras_preds.shape: {}\ncoreml_preds.shape: {}".format(
                        base_params, keras_preds.shape, coreml_preds.shape))
                    shape_err_models.append(base_params)
                    shutil.rmtree(model_dir)
                    i += 1
                    continue
                try:
                    for idx in range(0, len(coreml_preds)):
                        relative_error = (coreml_preds[idx] - keras_preds[idx]) / coreml_preds[idx]
                        self.assertAlmostEqual(relative_error, 0, places=2)
                except AssertionError:
                    print("Assertion error:\nbase_params: {}\nkeras_preds: {}\ncoreml_preds: {}".format(base_params,
                                                                                                        keras_preds,
                                                                                                        coreml_preds))
                    numerical_err_models.append(base_params)
                shutil.rmtree(model_dir)
                i += 1

                if not allow_slow:
                    break

            if not allow_slow:
                break
            

        self.assertEquals(shape_err_models, [], msg='Shape error models {}'.format(shape_err_models))
        self.assertEquals(numerical_err_models, [], msg='Numerical error models {}'.format(numerical_err_models))

@unittest.skipIf(not HAS_KERAS_TF, 'Missing keras. Skipping tests.')
class GRULayer(RecurrentLayerTest):
    """
    Class for testing GRU layer
    """
    def setUp(self):
        super(GRULayer, self).setUp()
        self.gru_params_dict = dict(
            dropout=[{'dropout_W': 0.,'dropout_U':0. }],
            regularizer=[{'W_regularizer': None, 'U_regularizer': None, 'b_regularizer': None}],
        )
        self.gru_layer_params = list(itertools.product(*self.gru_params_dict.values()))

    def test_gru_layer(self):
        i = 0
        numerical_err_models = []
        shape_err_models = []
        for base_params in self.base_layer_params:
            base_params = dict(zip(self.params_dict.keys(), base_params))
            input_data = generate_input(base_params['input_dims'][0], base_params['input_dims'][1],
                                        base_params['input_dims'][2])
            for gru_params in self.gru_layer_params:
                gru_params = dict(zip(self.gru_params_dict.keys(), gru_params))
                model = Sequential()
                model.add(
                    GRU(
                        base_params['output_dim'],
                        input_length=base_params['input_dims'][1],
                        input_dim=base_params['input_dims'][2],
                        # init=base_params['init'],
                        # inner_init=base_params['inner_init'],
                        activation=base_params['activation'],
                        return_sequences=base_params['return_sequences'],
                        go_backwards=base_params['go_backwards'],
                        unroll=base_params['unroll'],
                        dropout_U=gru_params['dropout']['dropout_U'],
                        dropout_W=gru_params['dropout']['dropout_W'],
                        W_regularizer=gru_params['regularizer']['W_regularizer'],
                        U_regularizer=gru_params['regularizer']['U_regularizer'],
                        b_regularizer=gru_params['regularizer']['b_regularizer'],
                    )
                )
                model_dir = tempfile.mkdtemp()
                keras_model_path = os.path.join(model_dir, 'keras.h5')
                coreml_model_path = os.path.join(model_dir, 'keras.mlmodel')
                model.save_weights(keras_model_path)
                mlkitmodel = _get_mlkit_model_from_path(model, coreml_model_path)
                keras_preds = model.predict(input_data).flatten()
                input_data = np.transpose(input_data, [1, 0, 2])
                coreml_preds = mlkitmodel.predict({'data': input_data})['output'].flatten()
                try:
                    self.assertEquals(coreml_preds.shape, keras_preds.shape)
                except AssertionError:
                    print("Shape error:\nbase_params: {}\nkeras_preds.shape: {}\ncoreml_preds.shape: {}".format(
                        base_params, keras_preds.shape, coreml_preds.shape))
                    shape_err_models.append(base_params)
                    shutil.rmtree(model_dir)
                    i += 1
                    continue
                try:
                    for idx in range(0, len(coreml_preds)):
                        relative_error = (coreml_preds[idx] - keras_preds[idx]) / coreml_preds[idx]
                        self.assertAlmostEqual(relative_error, 0, places=2)
                except AssertionError:
                    print("Assertion error:\nbase_params: {}\nkeras_preds: {}\ncoreml_preds: {}".format(base_params,
                                                                                                        keras_preds,
                                                                                                        coreml_preds))
                    numerical_err_models.append(base_params)
                shutil.rmtree(model_dir)
                i += 1

        self.assertEquals(shape_err_models, [], msg='Shape error models {}'.format(shape_err_models))
        self.assertEquals(numerical_err_models, [], msg='Numerical error models {}'.format(numerical_err_models))
