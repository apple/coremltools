import unittest
import numpy as np
import os, shutil
import tempfile
import coremltools.models.datatypes as datatypes
from coremltools.models import neural_network as neural_network
import coremltools
from nose.plugins.attrib import attr
import tensorflow as tf
import itertools

np.random.seed(10)

class CorrectnessTest(unittest.TestCase):
    
    def _compare_shapes(self, tf_preds, coreml_preds):
        if np.squeeze(tf_preds).shape != np.squeeze(coreml_preds).shape:
            return False
        else: 
            return True    
    
    def _compare_predictions(self, tf_preds, coreml_preds, delta = .01):
        tf_preds = tf_preds.flatten()
        coreml_preds = coreml_preds.flatten()
        for i in range(len(tf_preds)):
            max_den = max(1.0, tf_preds[i], coreml_preds[i])
            if np.abs(tf_preds[i] / max_den - coreml_preds[i] / max_den) > delta:
                return False
        return True
        
def get_tf_predictions_reorganize(X, params):
    Hin = params["H"]
    Win = params["W"]
    Cin = params["C"]
    with tf.Graph().as_default(), tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=(1,Hin,Win,Cin))
        if params["mode"] == 'SPACE_TO_DEPTH': 
            y = tf.space_to_depth(x, params["block_size"])
        else:
            y = tf.depth_to_space(x, params["block_size"])

    return sess.run(y,feed_dict={x: X})    

def get_coreml_predictions_reorganize(X, params):
    
    coreml_preds = []
    eval = True
    
    try:
        input_dim = X.shape[2:]
        output_dim = (1, 1, 1) #some random dimensions here: we are going to remove this information later
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', datatypes.Array(*output_dim))]
        builder = neural_network.NeuralNetworkBuilder(input_features, output_features)
        builder.add_reorganize_data('reorg', 'data', 'output', mode = params["mode"], block_size = params["block_size"])
        #Remove output shape by deleting and adding an output
        del builder.spec.description.output[-1]                            
        output = builder.spec.description.output.add()
        output.name = 'output' 
        output.type.multiArrayType.dataType = coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Value('DOUBLE')
        #save the model                        
        model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, 'test_layer.mlmodel')                        
        coremltools.utils.save_spec(builder.spec, model_path)
        #preprare input and get predictions
        coreml_model = coremltools.models.MLModel(model_path)
        coreml_input = {'data': X}
        coreml_preds = coreml_model.predict(coreml_input)['output']
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
    except RuntimeError as e:
        print(e)
        eval = False        
        
    return coreml_preds, eval    
    

def get_tf_predictions_depthwise(X, params, w):
    Hin = Win = params["H"]
    Cin = params["C"]
    Kh = Kw = params["kernel_size"]
    channel_multiplier = params["multiplier"]
    with tf.Graph().as_default(), tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=(1,Hin,Win,Cin))
        W = tf.constant(w,dtype= tf.float32, shape=[Kh,Kw,Cin,channel_multiplier])
        y = tf.nn.depthwise_conv2d(x,W,strides=[1, params["stride"], params["stride"], 1], padding = params["padding"])
        
    return sess.run(y,feed_dict={x: X})
    
def get_coreml_predictions_depthwise(X, params, w):    
    coreml_preds = []
    eval = True
    
    try:
        input_dim = X.shape[2:]
        output_dim = (1, 1, 1) #some random dimensions here: we are going to remove this information later
        input_features = [('data', datatypes.Array(*input_dim))]
        output_features = [('output', datatypes.Array(*output_dim))]
        builder = neural_network.NeuralNetworkBuilder(input_features, output_features)
        
        #tranlate weights : (Kh, Kw, kernel_channels, output_channels) == (Kh, Kw, Cin/g, Cout) == (Kh, Kw, 1, channel_multiplier * Cin)
        w_e = np.reshape(w, (params["kernel_size"],params["kernel_size"],params["multiplier"]*params["C"],1))
        w_e = np.transpose(w_e, [0,1,3,2])
        
        if params["padding"] == 'SAME':
            pad_mode = 'same'
        else:
            pad_mode = 'valid'
        
        builder.add_convolution('conv', 
                                kernel_channels = 1, 
                                output_channels = params["multiplier"]*params["C"], 
                                height = params["kernel_size"], width = params["kernel_size"], 
                                stride_height = params["stride"], stride_width = params["stride"], 
                                border_mode = pad_mode, 
                                groups = params["C"], 
                                W = w_e, b = None,
                                has_bias = 0, is_deconv = 0, output_shape = None, 
                                input_name = 'data', output_name = 'output')
        
        
        #Remove output shape by deleting and adding an output
        del builder.spec.description.output[-1]                            
        output = builder.spec.description.output.add()
        output.name = 'output' 
        output.type.multiArrayType.dataType = coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.ArrayDataType.Value('DOUBLE')
        #save the model                        
        model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, 'test_layer.mlmodel')                        
        coremltools.utils.save_spec(builder.spec, model_path)
        #preprare input and get predictions
        coreml_model = coremltools.models.MLModel(model_path)
        coreml_input = {'data': X}
        coreml_preds = coreml_model.predict(coreml_input)['output']
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
    except RuntimeError as e:
        print(e)
        eval = False        
        
    return coreml_preds, eval
    
    
            
class StressTest(CorrectnessTest):
    
    def test_data_reorganize(self):
        
        '''
        Define Params
        '''
        params_dict = dict( C = [1,2,8,16,15,27],
                            H = [2,4,6,8,10,15,21,16],
                            W = [2,4,6,8,10,15,21,16],
                            block_size = [2,3,4,5],
                            mode = ['SPACE_TO_DEPTH','DEPTH_TO_SPACE']
                            )
        params = [x for x in apply(itertools.product, params_dict.values())] 
        all_candidates = [dict(zip(params_dict.keys(), x)) for x in params]     
        valid_params = []               
        for pr in all_candidates:
            if pr["mode"] == 'SPACE_TO_DEPTH': 
                if pr["H"] % pr["block_size"] == 0 and pr["W"] % pr["block_size"] == 0:
                    valid_params.append(pr)  
            else:
                if pr["C"] % (pr["block_size"] ** 2) == 0:
                    valid_params.append(pr)        
        print "Total params to be tested: ", len(valid_params), "out of canditates: ", len(all_candidates)
        '''
        Test
        '''
        failed_tests_compile = []
        failed_tests_shape = []
        failed_tests_numerical = []
        for i in range(len(valid_params)):
            params = valid_params[i]
            #print "=========: ", params
            #if i % 10 == 0: print "======== Testing {}/{}".format(str(i), str(len(valid_params)))
            X = np.random.rand(1,params["C"],params["H"],params["W"])
            tf_preds = get_tf_predictions_reorganize(np.transpose(X,[0,2,3,1]), params)
            tf_preds = np.transpose(tf_preds, [0,3,1,2])
            coreml_preds, eval = get_coreml_predictions_reorganize(np.expand_dims(X, axis=0), params)
            if eval is False:
                failed_tests_compile.append(params)
            else:
                if not self._compare_shapes(tf_preds, coreml_preds):    
                    failed_tests_shape.append(params)
                elif not self._compare_predictions(tf_preds, coreml_preds):
                    failed_tests_numerical.append(params)
                    
        self.assertEqual(failed_tests_compile,[])
        self.assertEqual(failed_tests_shape, [])
        self.assertEqual(failed_tests_numerical,[])
        
        
    def test_depthwise_conv(self):
        
        '''
        Define Params
        '''
        params_dict = dict( C = [1,4,7],
                           H = [11,16],
                           stride = [1,2,3],
                           kernel_size = [1,2,3,5],
                           multiplier = [1,2,3,4],
                           padding = ['SAME', 'VALID']
                           )
        params = [x for x in apply(itertools.product, params_dict.values())] 
        all_candidates = [dict(zip(params_dict.keys(), x)) for x in params]     
        valid_params = []               
        for pr in all_candidates:
            if pr["padding"] == 'VALID':
                if np.floor((pr["H"]-pr["kernel_size"])/pr["stride"]) + 1 <= 0:
                    continue
            valid_params.append(pr)       
        print "Total params to be tested: ", len(valid_params), "out of canditates: ", len(all_candidates)
        '''
        Test
        '''
        failed_tests_compile = []
        failed_tests_shape = []
        failed_tests_numerical = []
        for i in range(len(valid_params)):
            params = valid_params[i]
            #print "=========: ", params
            #if i % 10 == 0: print "======== Testing {}/{}".format(str(i), str(len(valid_params)))
            X = np.random.rand(1,params["C"],params["H"],params["H"])
            w = np.random.rand(params["kernel_size"], params["kernel_size"], params["C"], params["multiplier"])
            tf_preds = get_tf_predictions_depthwise(np.transpose(X,[0,2,3,1]), params, w)
            tf_preds = np.transpose(tf_preds, [0,3,1,2])
            coreml_preds, eval = get_coreml_predictions_depthwise(np.expand_dims(X, axis=0), params, w)
            if eval is False:
                failed_tests_compile.append(params)
            else:
                if not self._compare_shapes(tf_preds, coreml_preds):    
                    failed_tests_shape.append(params)
                elif not self._compare_predictions(tf_preds, coreml_preds):
                    failed_tests_numerical.append(params)
                    
        self.assertEqual(failed_tests_compile,[])
        self.assertEqual(failed_tests_shape, [])
        self.assertEqual(failed_tests_numerical,[])    
        
                    
                        
            
            




            