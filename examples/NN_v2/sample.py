# This example will be removed once
# concrete example demontrating conversion through this proto is added

import sys
sys.path.insert(0, '../../coremltools/proto')
import Program_pb2 as pm
from coremltools.models.program.builder import NeuralNetBuffer as NetBuffer

import google.protobuf.json_format as json_format
from helper import *

pad_vals = create_tensor_value(np.array([2, 2, 3, 3], dtype=np.int32))

in_channels = 2
out_channels = 3
H = 3
W = 3
param_type = create_valuetype_tensor([in_channels, out_channels, H, W], pm.ScalarType.FLOAT32)
tp_value = create_tuple_value((3, np.arange(5).astype(np.int32)))
nn_buffer = NetBuffer('./sample.wt')

messages = [
    # Inputs
    pm.Operation(name="load_conv_weight", type="load_param",
                 # load_params will use use BufferLocator to get the param
                 attributes={"val" : pm.Value(type=param_type,
                             fileValue=pm.Value.FileValue(fileName="/path/weight.bin", offset=nn_buffer.add_buffer(np.arange(5).astype(np.float32))))}),
    pm.Operation(name='input', type='TODO'), # type=...

    pm.Operation(name='conv2d/filters.1', type='const',
                 attributes={"pad":pad_vals}),
    pm.Operation(name='conv2d.1', type='conv2d',
                 inputs={'input':'input', 'filters':'conv2d/filters.1'}),

    pm.Operation(name='max_pool2d/pool_size', type='const'), # type=..., attr=...
    pm.Operation(name='max_pool2d.1', type='max_pool',
                 inputs={'input':'conv2d.1', 'pool_size':'maxpool2d/pool_size'}),

    pm.Operation(name='conv2d/filters.2', type='const'), # type=..., attr=...
    pm.Operation(name='conv2d.2', type='conv2d',
                 inputs={'input':'max_pool2d.1', 'filters':'conv2d/filters.2'}),

    pm.Operation(name='max_pool2d.2', type='max_pool',
                 inputs={'input':'conv2d.2', 'pool_size':'maxpool2d/pool_size'}),

    pm.Operation(name='flatten', type='flatten',
                 inputs={'input':'max_pool2d.2'}),

    pm.Operation(name='dense/units', type='const'), # type=..., attr=...
    pm.Operation(name='dense/activation', type='const'), # type=..., attr=...
    pm.Operation(name='dense', type='dense',
                 inputs={'input':'flatten', 'units':'dense/units', 'activation':'dense/activation'}),

    pm.Operation(name='dropout/rate', type='const'), # type=..., attr=...
    pm.Operation(name='dropout', type='drOperationout', 
                 inputs={'input':'dense', 'rate':'dropout/rate'}),

    pm.Operation(name='output/units', type='const'), # type=..., attr=...
    pm.Operation(name='output/activation', type='const'), # type=..., attr=...
    pm.Operation(name='output', type='dense',
                 inputs={'input':'dropout', 'units':'output/units', 'activation':'dense/activation'}),
]

if __name__ == "__main__":
    for message in messages:
        print(json_format.MessageToJson(message))