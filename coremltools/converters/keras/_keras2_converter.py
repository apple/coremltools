from six import string_types as _string_types

from ...models.neural_network import NeuralNetworkBuilder as _NeuralNetworkBuilder
from ...proto import FeatureTypes_pb2 as _FeatureTypes_pb2
from collections import OrderedDict as _OrderedDict
from ...models import datatypes
from ...models import MLModel as _MLModel
from ...models.utils import save_spec as _save_spec

from ..._deps import HAS_KERAS2_TF as _HAS_KERAS2_TF

if _HAS_KERAS2_TF:
    import keras as _keras
    from . import _layers2
    from . import _topology2
    _KERAS_LAYER_REGISTRY  = {
        _keras.layers.core.Dense: _layers2.convert_dense,
        _keras.layers.core.Activation: _layers2.convert_activation,
        _keras.layers.advanced_activations.LeakyReLU: _layers2.convert_activation,
        _keras.layers.advanced_activations.PReLU: _layers2.convert_activation,
        _keras.layers.advanced_activations.ELU: _layers2.convert_activation,
        _keras.layers.advanced_activations.ThresholdedReLU: _layers2.convert_activation,

        _keras.layers.convolutional.Conv2D: _layers2.convert_convolution,
        _keras.layers.convolutional.Conv2DTranspose: _layers2.convert_convolution,
        _keras.layers.convolutional.SeparableConv2D: _layers2.convert_separable_convolution, 
        _keras.layers.pooling.AveragePooling2D: _layers2.convert_pooling,
        _keras.layers.pooling.MaxPooling2D: _layers2.convert_pooling,
        _keras.layers.pooling.GlobalAveragePooling2D: _layers2.convert_pooling,
        _keras.layers.pooling.GlobalMaxPooling2D: _layers2.convert_pooling,
        _keras.layers.convolutional.ZeroPadding2D: _layers2.convert_padding,
        _keras.layers.convolutional.Cropping2D: _layers2.convert_cropping,
        _keras.layers.convolutional.UpSampling2D: _layers2.convert_upsample,

        _keras.layers.convolutional.Conv1D: _layers2.convert_convolution1d,
        _keras.layers.pooling.AveragePooling1D: _layers2.convert_pooling,
        _keras.layers.pooling.MaxPooling1D: _layers2.convert_pooling,
        _keras.layers.pooling.GlobalAveragePooling1D: _layers2.convert_pooling,
        _keras.layers.pooling.GlobalMaxPooling1D: _layers2.convert_pooling,
        _keras.layers.convolutional.ZeroPadding1D: _layers2.convert_padding,
        _keras.layers.convolutional.Cropping1D: _layers2.convert_cropping,
        _keras.layers.convolutional.UpSampling1D: _layers2.convert_upsample,

        _keras.layers.recurrent.LSTM: _layers2.convert_lstm,
        _keras.layers.recurrent.SimpleRNN: _layers2.convert_simple_rnn,
        _keras.layers.recurrent.GRU: _layers2.convert_gru,
        _keras.layers.wrappers.Bidirectional: _layers2.convert_bidirectional,

        _keras.layers.normalization.BatchNormalization: _layers2.convert_batchnorm,

        _keras.layers.Add: _layers2.convert_merge,
        _keras.layers.Multiply: _layers2.convert_merge,
        _keras.layers.Average: _layers2.convert_merge,
        _keras.layers.Maximum: _layers2.convert_merge,
        _keras.layers.Concatenate: _layers2.convert_merge,
        _keras.layers.Dot: _layers2.convert_merge,
    
        _keras.layers.core.Flatten: _layers2.convert_flatten,
        _keras.layers.core.Permute:_layers2.convert_permute,
        _keras.layers.core.Reshape:_layers2.convert_reshape,
        _keras.layers.embeddings.Embedding:_layers2.convert_embedding,
        _keras.layers.core.RepeatVector:_layers2.convert_repeat_vector,

        _keras.engine.topology.InputLayer:_layers2.default_skip,
        _keras.layers.core.Dropout:_layers2.default_skip,
        _keras.layers.wrappers.TimeDistributed:_layers2.default_skip,
        
        _keras.applications.mobilenet.DepthwiseConv2D:_layers2.convert_convolution,

    }

    _KERAS_SKIP_LAYERS = [
        _keras.layers.core.Dropout,
    ]

def _check_unsupported_layers(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, _keras.models.Sequential) or isinstance(layer, _keras.models.Model):
            _check_unsupported_layers(layer)
        else:
            if type(layer) not in _KERAS_LAYER_REGISTRY:
                 raise ValueError(
                     "Keras layer '%s' not supported. " % str(type(layer)))
            if isinstance(layer, _keras.layers.wrappers.TimeDistributed):
                if type(layer.layer) not in _KERAS_LAYER_REGISTRY:
                     raise ValueError(
                         "Keras layer '%s' not supported. " % str(type(layer.layer)))
            if isinstance(layer, _keras.layers.wrappers.Bidirectional):
                if not isinstance(layer.layer,  _keras.layers.recurrent.LSTM):
                    raise ValueError(
                        "Keras bi-directional wrapper conversion supports only LSTM layer at this time. ")

def _get_layer_converter_fn(layer):
    """Get the right converter function for Keras
    """
    layer_type = type(layer)
    if layer_type in _KERAS_LAYER_REGISTRY:
        return _KERAS_LAYER_REGISTRY[layer_type]
    else:
        raise TypeError("Keras layer of type %s is not supported." % type(layer))

def _load_keras_model(model_network_path, model_weight_path):
    """Load a keras model from disk

    Parameters
    ----------
    model_network_path: str
        Path where the model network path is (json file)

    model_weight_path: str
        Path where the model network weights are (hd5 file)

    Returns
    -------
    model: A keras model
    """
    from keras.models import model_from_json
    import json

    # Load the model network
    json_file = open(model_network_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # Load the model weights
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weight_path)

    return loaded_model

def _convert_to_coreml_shape(dim, is_seq = False):
    """
    Keras -> Core ML input dimension dictionary
    (D) -> [D] or [D,1,1,1,1] for RNN
    (None, None) -> [1, 1, 1, 1, 1] (unknown sequence length or batch)
    (S, D) -> [S, 1, D, 1, 1] for RNN or [1,1,D,1,S] for 1D conv
    (None, D) -> [D] or [D, 1, 1, 1, 1] for Embedding layer of RNN
    (None, S, D) -> [S, 1, D, 1, 1] for RNN or [1, 1, D, 1, S] for 1D conv
    (None, H, W, C) -> [C, H, W]
    (B, S, D) -> [D]
    For protobuf we're ignoring (S,B)
    """
    if len(dim) == 1:
        # (D) -> [D] or [D,1,1,1,1] for RNN
        return (1,) if is_seq else dim
    elif len(dim) == 2:
        if dim[0] == None and dim[1] == None:
            # (None,None) -> [1,1,1,1,1]
            return (1,)
        elif dim[0] != None and dim[1] != None: 
            # (S,D) -> [S,1,D,1,1] for RNN or [1,1,D,1,S] for 1D conv
            return (dim[1],1,1) if is_seq else (dim[1],1,dim[0])
        elif dim[0] == None and dim[1] != None: 
            # (None,D) -> [D] or [D,1,1,1,1]
            return (1,) if is_seq else (dim[1],)
    elif len(dim) == 3:
        if dim[0] == None: #(None,S,D)
            return (dim[2],) if is_seq else (dim[2],1,dim[1])
        else: # (B,S,D)
            return (dim[2],)
    elif len(dim) == 4: #(None,H,W,C)
        return (dim[3],dim[1],dim[2])
    else:
        raise ValueError('Unrecognizable shape ' + str(dim))

def _convert(model, 
            input_names = None, 
            output_names = None, 
            image_input_names = None, 
            is_bgr = False, 
            red_bias = 0.0, 
            green_bias = 0.0, 
            blue_bias = 0.0, 
            gray_bias = 0.0, 
            image_scale = 1.0, 
            class_labels = None, 
            predicted_feature_name = None,
            predicted_probabilities_output = ''):

    if isinstance(model, _string_types):
        model = _keras.models.load_model(model)
    elif isinstance(model, tuple):
        model = _load_keras_model(model[0], model[1])
    
    # Check valid versions
    _check_unsupported_layers(model)
    
    # Build network graph to represent Keras model
    graph = _topology2.NetGraph(model)
    graph.build()
    graph.remove_skip_layers(_KERAS_SKIP_LAYERS)
    graph.insert_1d_sequence_permute_layers()
    graph.insert_permute_for_spatial_bn()
    graph.defuse_activation()
    graph.remove_internal_input_layers()
    graph.make_output_layers()

    # The graph should be finalized before executing this
    graph.generate_blob_names()
    graph.add_recurrent_optionals()
    
    inputs = graph.get_input_layers()
    outputs = graph.get_output_layers()
    
    # standardize input / output names format
    def to_name_list(names):
        return [names] if isinstance(names, _string_types) else names
    
    input_names = ['input' + str(i+1) for i in range(len(inputs))] \
            if input_names is None else to_name_list(input_names)
    output_names = ['output' + str(i+1) for i in range(len(outputs))] \
            if output_names is None else to_name_list(output_names)
    
    graph.reset_model_input_names(input_names)
    graph.reset_model_output_names(output_names)

    # Retrieve input shapes from model
    input_dims = model.input_shape if type(model.input_shape) is list \
            else [model.input_shape]
    for idx, dim in enumerate(input_dims):
        is_seq = False
        s = graph.get_successors(inputs[idx])[0]
        ks = graph.get_keras_layer(s)
        if len(dim) == 2 and isinstance(ks,_keras.layers.embeddings.Embedding):
            is_seq = True
        if len(dim) == 3 and (_topology2.is_recurrent_layer(ks) or 
                isinstance(ks, _keras.layers.wrappers.TimeDistributed)):
            is_seq = True
        input_dims[idx] = _convert_to_coreml_shape(dim, is_seq)

    # Retrieve output shapes from model
    output_dims = model.output_shape if type(model.output_shape) is list \
            else [model.output_shape]
    for idx, dim in enumerate(output_dims):
        is_seq = False
        out_layer = outputs[idx]
        if _topology2.is_activation_layer(graph.get_keras_layer(out_layer)):
            out_layer = graph.backtrace_activation_layers(out_layer)
        kl = graph.get_keras_layer(out_layer)
        if len(dim) == 3 and (_topology2.is_recurrent_layer(kl) or 
                _topology2.is_seq_merge_layer(kl) or 
                isinstance(kl, _keras.layers.embeddings.Embedding) or
                isinstance(kl, _keras.layers.wrappers.TimeDistributed)):
            is_seq = True
        output_dims[idx] =  _convert_to_coreml_shape(dim, is_seq)

    # from nose.tools import set_trace
    # set_trace()
    input_types = [datatypes.Array(*dim) for dim in input_dims]
    output_types = [datatypes.Array(*dim) for dim in output_dims]

    # Some of the feature handling is sensitive about string vs. unicode
    input_names = map(str, input_names)
    output_names = map(str, output_names)
    
    mode = 'classifier' if class_labels is not None else None
    
    is_classifier = class_labels is not None
    mode = 'classifier' if is_classifier else None

    # assuming these match
    input_features = list(zip(input_names, input_types))
    output_features = list(zip(output_names, output_types))

    builder = _NeuralNetworkBuilder(input_features, output_features, 
            mode = mode)

    for iter, layer in enumerate(graph.layer_list):
        keras_layer = graph.keras_layer_map[layer]
        print("%d : %s, %s" % (iter, layer, keras_layer))
        if isinstance(keras_layer, _keras.layers.wrappers.TimeDistributed):
            keras_layer = keras_layer.layer
        converter_func = _get_layer_converter_fn(keras_layer)
        input_names, output_names = graph.get_layer_blobs(layer)
        converter_func(builder, layer, input_names, output_names, keras_layer)

    # Since we aren't mangling anything the user gave us, we only need to update
    # the model interface here
    builder.add_optionals(graph.optional_inputs, graph.optional_outputs)

    # Add image input identifier
    if image_input_names is not None and isinstance(image_input_names, 
            _string_types):
        image_input_names = [image_input_names]

    # Add classifier classes (if applicable)
    if is_classifier:
        classes_in = class_labels
        if isinstance(classes_in, _string_types):
            import os
            if not os.path.isfile(classes_in):
                raise ValueError("Path to class labels (%s) does not exist." % classes_in)
            with open(classes_in, 'r') as f:
                classes = f.read()
            classes = classes.splitlines()
        elif type(classes_in) is list: # list[int or str]
            classes = classes_in
        else:
            raise ValueError('Class labels must be a list of integers / strings, or a file path')

        if predicted_feature_name is not None:
            builder.set_class_labels(classes, predicted_feature_name = predicted_feature_name,
                                     prediction_blob = predicted_probabilities_output)
        else:
            builder.set_class_labels(classes)

    # Set pre-processing paramsters
    builder.set_pre_processing_parameters(image_input_names = image_input_names, 
                                          is_bgr = is_bgr, 
                                          red_bias = red_bias, 
                                          green_bias = green_bias, 
                                          blue_bias = blue_bias, 
                                          gray_bias = gray_bias, 
                                          image_scale = image_scale)

    # Return the protobuf spec
    spec = builder.spec
    return _MLModel(spec)
