# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _


def _get_nn_spec(spec):
    if spec.WhichOneof('Type') == 'neuralNetwork':
        nn_spec = spec.neuralNetwork
    elif spec.WhichOneof('Type') == 'neuralNetworkClassifier':
        nn_spec = spec.neuralNetworkClassifier
    elif spec.WhichOneof('Type') == 'neuralNetworkRegressor':
        nn_spec = spec.neuralNetworkRegressor
    else:
        raise ValueError('Specification must contain a neural network')
    return nn_spec

def _get_blob_use_count(spec):
    """
    Computes use count of every tensor/node in NN graph
    i.e. How many layers are using it as an input

    :param nn_spec : NeuralNetworkSpecification
    :returns use_count_dict : str -> int, a dictionary with node name as a key and it's use count as a value
    """
    def _get_blob_use_count_rec(nn_spec, use_count):
        nn_layers = nn_spec.layers
        for layer in nn_layers:
            layer_type = layer.WhichOneof('layer')
            if layer_type == 'loop':
                _get_blob_use_count_rec(layer.loop.conditionNetwork, use_count)
                _get_blob_use_count_rec(layer.loop.bodyNetwork, use_count)
            elif layer_type == 'branch':
                _get_blob_use_count_rec(layer.loop.ifBranch, use_count)
                _get_blob_use_count_rec(layer.loop.elseBranch, use_count)
            else:
                for inp in layer.input:
                    use_count[inp] = use_count.get(inp, 0) + 1

    use_count_dict = {}
    # Collect variable use count recursively
    nn_spec = _get_nn_spec(spec)
    _get_blob_use_count_rec(nn_spec, use_count_dict)

    # Network outputs are variable use
    network_outputs = _get_network_output(spec)
    for _output in network_outputs:
        use_count_dict[_output] = use_count_dict.get(_output, 0) + 1
    return use_count_dict

def _is_layer(nn_layer, layer_type):
    """
    :param nn_layer : NN layer proto message
    :param layer_type : str Layer type to check against
    :returns True if nn_layer is of type `layer_type` otherwise False
    """
    return nn_layer.WhichOneof('layer') == layer_type

def _get_input(layer, index=0):
    """
    :param layer : NN Layer Proto message
    :param index : Layer input index (Default 0)
    :returns name of input at provided index if present, otherwise None
    """
    if len(layer.input) <= index:
        return None
    return layer.input[index]

def _get_output(layer, index=0):
    """
    :param layer : NN Layer Proto message
    :param index : Layer output index (Default 0)
    :returns name of output at provided index if present, otherwise None
    """
    if len(layer.output) <= index:
        return None
    return layer.output[index]

def _get_network_output(spec):
    """
    :param spec : CoreML Specification
    :returns network output names
    """
    network_output_names = []
    for _out in spec.description.output:
        network_output_names.append(_out.name)
    return network_output_names

def _find_disconnected_load_constants(nn_spec, disconnected_load_constants):
    nn_layers = nn_spec.layers
    for layer in nn_layers:
        layer_type = layer.WhichOneof('layer')
        if layer_type == 'loadConstant' or layer_type == 'loadConstantND':
            disconnected_load_constants[layer.output[0]] = layer

        for inp in layer.input:
            if inp in disconnected_load_constants:
                disconnected_load_constants.pop(inp)

        if layer_type == 'loop':
            _find_disconnected_load_constants(
                layer.loop.conditionNetwork, disconnected_load_constants)
            _find_disconnected_load_constants(layer.loop.bodyNetwork, disconnected_load_constants)
            if layer.loop.conditionVar in disconnected_load_constants:
                disconnected_load_constants.pop(layer.loop.conditionVar)

        if layer_type == 'branch':
            _find_disconnected_load_constants(layer.branch.ifBranch, disconnected_load_constants)
            _find_disconnected_load_constants(layer.branch.elseBranch, disconnected_load_constants)


def _delete_disconnected_load_constants(nn_spec, disconnected_load_constants):
    nn_layers = nn_spec.layers
    N = len(nn_layers)
    for i in range(N-1, -1, -1):
        layer = nn_layers[i]
        layer_type = layer.WhichOneof('layer')
        if layer_type == 'loadConstant' or layer_type == 'loadConstantND':
            if layer.output[0] in disconnected_load_constants:
                nn_layers.remove(layer)

        if layer_type == 'loop':
            _delete_disconnected_load_constants(layer.loop.conditionNetwork, disconnected_load_constants)
            _delete_disconnected_load_constants(layer.loop.bodyNetwork, disconnected_load_constants)

        if layer_type == 'branch':
            _delete_disconnected_load_constants(layer.branch.ifBranch, disconnected_load_constants)
            _delete_disconnected_load_constants(layer.branch.elseBranch, disconnected_load_constants)


def remove_disconnected_constants(spec):
    """
    remove constant layers whose outputs are not connected to any other layer
    """
    nn_spec = _get_nn_spec(spec)
    disconnected_load_constants = dict()  # output_name -> layer reference
    _find_disconnected_load_constants(nn_spec, disconnected_load_constants)
    if len(disconnected_load_constants) > 0:
        _delete_disconnected_load_constants(nn_spec, disconnected_load_constants)
        print('[Core ML Pass] {} disconnected constants nodes deleted'.format(
            len(disconnected_load_constants)))


def transform_conv_crop(spec):
    """
    Transforms Conv -> Crop -> BN (if present) -> Activation (if present) into
               Conv -> BN (if present) -> Activation (if present) -> Crop
    This transformation will allow Conv -> BN -> Activation fusion by changing
    the position of the crop layer, which does not affect the computation
    """
    # Collect metadata
    use_count = _get_blob_use_count(spec)
    network_output_names = _get_network_output(spec)

    nn_spec = _get_nn_spec(spec)
    nn_layers = nn_spec.layers
    for i in range(0, len(nn_layers)-2):

        # If Convolution output is being using as a network output or more than one layers
        # that's acceptable
        if not _is_layer(nn_layers[i], 'convolution'):
            continue

        # Output of Crop layer must not be network output or used by more than one layer
        if not (_is_layer(nn_layers[i+1], 'crop') \
                and _get_input(nn_layers[i+1]) not in network_output_names \
                and use_count[_get_output(nn_layers[i+1])] == 1):
            continue

        layer_to_shuffle_with = -1

        # Output of Batchnorm layer must not be network output or used by more than one layer
        if _is_layer(nn_layers[i+2], 'batchnorm') \
            and use_count[_get_output(nn_layers[i+2])] == 1:
            layer_to_shuffle_with = i+2

        # Output of Activation layer must not be network output or used by more than one layer
        if i+3 < len(nn_layers) and _is_layer(nn_layers[i+3], 'activation') \
            and use_count[_get_output(nn_layers[i+3])] == 1:
            layer_to_shuffle_with = i+3

        if layer_to_shuffle_with == -1:
            continue
        # restructure crop layer
        # Conv --->  Crop  ---> BN ---> Activation ---> Layer1
        # In following three steps
        # 1. Conv --------------> BN ---> Activation ---> Layer1
        #        \            /  
        #         ---> Crop --
        nn_layers[i].output[0] = nn_layers[i+1].output[0]
        # 2. Conv ---> BN ---> Activation ---> Layer1
        #      \                           /
        #        -----------------Crop ----
        nn_layers[i+1].output[0] = nn_layers[layer_to_shuffle_with].output[0]
        # 3. Conv ---> BN ---> Activation ---> Crop ---> Layer1
        nn_layers[layer_to_shuffle_with].output[0] = nn_layers[i+1].input[0]

        # Add Crop layer at new position and remove from current position
        crop_layer = nn_layers[i+1]
        nn_layers.remove(crop_layer)
        nn_layers.insert(layer_to_shuffle_with, crop_layer)
