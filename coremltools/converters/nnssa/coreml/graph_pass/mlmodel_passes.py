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


def transform_conv_crop_bn_to_conv_bn_crop(spec):
    """
    Convert Conv -> Crop -> BatchNorm to
            Conv -> BatchNorm -> Crop
    Currently, this pass works only if Conv -> Crop -> BatchNorm appears consecutively.
    Right now, Conv Tranpose adds Crop immediately after and hence generalized pass is not needed.
    Current Implementation does not handle cases when few other ops are present in between Conv -> Crop or Crop -> BatchNorm
    """
    nn_spec = _get_nn_spec(spec)
    use_dict = {}

    for _input in spec.description.input:
        use_dict[_input.name] = 1

    for _layer in nn_spec.layers:
        for _input in _layer.input:
            use_dict[_input] = use_dict.get(_input, 0) + 1

    nn_layers = nn_spec.layers
    for i in range(2, len(nn_layers)):
        layer = nn_layers[i]
        layer_type = layer.WhichOneof('layer')

        if layer_type == 'batchnorm':
            bn_layer = layer
            bn_index = i

            crop_index = bn_index - 1
            crop_layer = nn_layers[crop_index]
            if crop_layer.WhichOneof('layer') != 'crop':
                continue

            convt_index = crop_index - 1
            convt_layer = nn_layers[convt_index]
            if convt_layer.WhichOneof('layer') != 'convolution':
                continue

            # If Conv2d Transpose or CropLayer is being used by other OPs then don't re-order
            if use_dict.get(crop_layer.output[0]) > 1 or use_dict.get(convt_layer.output[0]) > 1:
                continue

            bn_output = bn_layer.output[0]
            crop_output = crop_layer.output[0]

            # Pass Conv2d Transpose to Batch Norm
            bn_layer.input[0] = convt_layer.output[0]
            # Swap input and output Batch Norm and Crop Layer
            bn_layer.output[0] = crop_output
            crop_layer.input[0] = crop_output
            crop_layer.output[0] = bn_output

            # Remove old Crop Layer and insert after BatchNorm layer
            nn_layers.remove(crop_layer)
            nn_layers.insert(bn_index, crop_layer)
