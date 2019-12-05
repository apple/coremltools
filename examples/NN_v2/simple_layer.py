import sys
import coremltools.proto.Program_pb2 as pm
import coremltools.proto.Model_pb2 as ml
import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools.models.utils import save_spec
from coremltools.models.program.builder import NeuralNetBuffer as NetBuffer

import google.protobuf.json_format as json_format
from helper import *

"""
Following is an example for using Program proto to create ML model
We create a simple two layer model (Dense + Softmax)
"""

# input0: fp32[2, 4]       \
#                           |--> matmul --> dense1_out: fp32[2, 2]
# dense1/bias: fp32[4, 2]  / 
#
# dense1_out              \
#                          | --> add --> output0: fp32[2, 2]
# dense1/bias: fp32[1, 2] /

#
# Expected output0:
# [ [3.9, 5.7],
#   [2.7, 4.45] ]
#


# Define parameters
parameters = {}
import numpy as np
dense1_wt = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dense1_bias = [-0.5,  0.5]

nn_buffer = NetBuffer('./simple_layer.wt')
parameters['dense1/weight'] = create_load_from_file_value(file_name='./simple_layer.wt', offset=nn_buffer.add_buffer(dense1_wt), dim=[1, 4, 2], scalar_type=pm.FLOAT32)
parameters['dense1/bias'] = create_load_from_file_value(file_name='./simple_layer.wt', offset=nn_buffer.add_buffer(dense1_bias), dim=[2], scalar_type=pm.FLOAT32)

# Matmul layer
dense1 = pm.Operation(name='dense1', type='matmul',
                      inputs={'x':'input0', 'y':'dense1/weight'},
                      outputs=[pm.NamedValueType(name='dense1_out', type=create_valuetype_tensor([1, 2, 2], pm.FLOAT32))])

# Add bias
bias1 = pm.Operation(name='bias1', type='add',
                     inputs={'x': 'dense1_out', 'y': 'dense1/bias'},
                     outputs=[pm.NamedValueType(name='output0', type=create_valuetype_tensor([1, 2, 2], pm.FLOAT32))])

# Create a block
main_block = pm.Block(inputs={'input0':'input0'}, outputs=['output0'], operations=[dense1, bias1])

# Create function
main_function = pm.Function(inputs=[pm.NamedValueType(name='input0', type=create_valuetype_tensor([4], pm.FLOAT32))],
                            outputs=[create_valuetype_tensor([1, 2, 2], pm.FLOAT32)],
                            block=main_block)
# Create program
simple_net = pm.Program(version=1, functions={'main': main_function}, parameters=parameters)

# Define input and output features for Model
input_feature_type = ft.FeatureType()
array_type = ft.ArrayFeatureType(shape=[1, 2, 4], dataType=ft.ArrayFeatureType.ArrayDataType.FLOAT32)
input_feature_type.multiArrayType.CopyFrom(array_type)

output_feature_type = ft.FeatureType()
array_type = ft.ArrayFeatureType(shape=[1, 2, 2], dataType=ft.ArrayFeatureType.ArrayDataType.FLOAT32)
output_feature_type.multiArrayType.CopyFrom(array_type)

# Model description
desc = ml.ModelDescription(input=[ml.FeatureDescription(name='input0', type=input_feature_type)],
                           output=[ml.FeatureDescription(name='output0', type=output_feature_type)])
# Create ML Model
model = ml.Model(description=desc, program=simple_net, specificationVersion=5)
# Save spec
save_spec(model, 'simple_layer.mlmodel')
print(model)
