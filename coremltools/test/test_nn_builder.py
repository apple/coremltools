import unittest
import numpy as np
import os, shutil
import tempfile
from nose.tools import raises
from nose.plugins.attrib import attr
import coremltools
from coremltools.models import datatypes, MLModel
from coremltools.models.neural_network import NeuralNetworkBuilder

class BasicNumericCorrectnessTest(unittest.TestCase):
            
    def test_undefined_shape_single_output(self):
        W = np.ones((3,3))
        input_features = [('data', datatypes.Array(3))]
        output_features = [('probs', None)]
        builder = NeuralNetworkBuilder(input_features, output_features)
        builder.add_inner_product(name = 'ip1', 
                                  W = W, 
                                  b = None, 
                                  input_channels = 3, 
                                  output_channels = 3,
                                  has_bias = False, 
                                  input_name = 'data', 
                                  output_name = 'probs')
        mlmodel = MLModel(builder.spec)
        data = np.ones((3,))
        data_dict = {'data': data}
        probs = mlmodel.predict(data_dict)['probs']
        self.assertTrue(np.allclose(probs, np.ones(3) * 3))

