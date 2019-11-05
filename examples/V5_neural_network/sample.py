# This example will be removed once
# concrete example demontrating conversion through this proto is added

import sys
sys.path.insert(0, '../../coremltools/proto')
import Program_pb2 as pm

import google.protobuf.json_format as json_format
from helper import *

pad_type = create_valuetype_tensor([2, 2, 3, 3], pm.ScalarType.INT32)
pad_vals = create_tensor_value(np.array([2, 2, 3, 3], dtype=np.int32))

in_channels = 2
out_channels = 3
H = 3
W = 3
param_type = create_valuetype_tensor([in_channels, out_channels, H, W], pm.ScalarType.FLOAT32)
tp_value = create_tuple_value((3, np.arange(5).astype(np.int32)))

messages = [
    # Inputs
    pm.Operator(name="load_conv_weight", opType="load_param",
                # load_params will use use BufferLocator to get the param
                attributes={"val" : pm.Value(type=param_type,
                            fileValue=pm.Value.FileValue(fileName="/path/weight.bin", offset=19847,
                            length=4*2*3*3*3))}),
    pm.Operator(name='input', opType='TODO'), # type=...

    pm.Operator(name='conv2d/filters.1', opType='const',
                attributes={"pad":pm.Value(type=pad_type,
                            immediateValue=pm.Value.ImmediateValue(tensor=pad_vals))}),
    pm.Operator(name='conv2d.1', opType='conv2d',
                inputs={'input':'input', 'filters':'conv2d/filters.1'}),

    pm.Operator(name='max_pool2d/pool_size', opType='const'), # type=..., attr=...
    pm.Operator(name='max_pool2d.1', opType='max_pool',
                inputs={'input':'conv2d.1', 'pool_size':'maxpool2d/pool_size'}),

    pm.Operator(name='conv2d/filters.2', opType='const'), # type=..., attr=...
    pm.Operator(name='conv2d.2', opType='conv2d',
                inputs={'input':'max_pool2d.1', 'filters':'conv2d/filters.2'}),

    pm.Operator(name='max_pool2d.2', opType='max_pool',
                inputs={'input':'conv2d.2', 'pool_size':'maxpool2d/pool_size'}),

    pm.Operator(name='flatten', opType='flatten',
                inputs={'input':'max_pool2d.2'}),

    pm.Operator(name='dense/units', opType='const'), # type=..., attr=...
    pm.Operator(name='dense/activation', opType='const'), # type=..., attr=...
    pm.Operator(name='dense', opType='dense',
                inputs={'input':'flatten', 'units':'dense/units', 'activation':'dense/activation'}),

    pm.Operator(name='dropout/rate', opType='const'), # type=..., attr=...
    pm.Operator(name='dropout', opType='drOperatorout', 
                inputs={'input':'dense', 'rate':'dropout/rate'}),

    pm.Operator(name='output/units', opType='const'), # type=..., attr=...
    pm.Operator(name='output/activation', opType='const'), # type=..., attr=...
    pm.Operator(name='output', opType='dense',
                inputs={'input':'dropout', 'units':'output/units', 'activation':'dense/activation'}),
]

if __name__ == "__main__":
    for message in messages:
        print(json_format.MessageToJson(message))