import sys
import coremltools.proto.Program_pb2 as pm
import coremltools.proto.Model_pb2 as ml
import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools.models.utils import save_spec

import google.protobuf.json_format as json_format
from helper import *

"""
Following is an example for using Program proto to create ML model
We create a simple two layer model (Dense + Softmax)
"""
# Define parameters
parameters = {}
parameters['dense1/weight'] = create_load_from_file_value(file_name='./simple_layer.wt', offset=0, size=64*10, dim=[64, 10], scalar_type=pm.FLOAT32)
parameters['dense1/bias'] = create_load_from_file_value(file_name='./simple_layer.wt', offset=2560, size=10, dim=[10], scalar_type=pm.FLOAT32)

# Constant op
softmax_axis = pm.Operation(name='softmax_axis', type='constant',
                            attributes={'val':create_scalar_value(1)},
                            outputs=[pm.NamedValueType(name='softmax_axis', type=create_scalartype(pm.INT32))])

# Load parameters
dense1_wt = pm.Operation(name='dense1/weight', type='loadParam',
                         inputs={'input':'dense1/weight'},
                         outputs=[pm.NamedValueType(name='dense1/weight', type=create_valuetype_tensor([64, 10], pm.FLOAT32))])
dense1_b = pm.Operation(name='dense1/bias', type='loadParam',
                        inputs={'input':'dense1/bias'},
                        outputs=[pm.NamedValueType(name='dense1/bias', type=create_valuetype_tensor([10], pm.FLOAT32))])

# Dense layer
dense1 = pm.Operation(name='dense1', type='dense',
                      inputs={'input':'input0', 'weight':'dense1/weight', 'bias':'dense1/bias'},
                      outputs=[pm.NamedValueType(name='dense1_out', type=create_valuetype_tensor([10], pm.FLOAT32))])

# Softmax layer
out = pm.Operation(name='softmax', type='softmax',
                   inputs={'input':'dense1_out', 'axis':'softmax_axis'},
                   outputs=[pm.NamedValueType(name='output0', type=create_valuetype_tensor([10], pm.FLOAT32))])

# Create a block
main_block = pm.Block(inputs={'input0':'input0'}, outputs=['output0'], operations=[dense1_wt, dense1_b, softmax_axis, dense1, out])
# Create function
main_function = pm.Function(inputs=[pm.NamedValueType(name='input0', type=create_valuetype_tensor([64], pm.FLOAT32))],
                            outputs=[create_valuetype_tensor([10], pm.FLOAT32)],
                            block=main_block)
# Create program
simple_net = pm.Program(version=1, functions={'main': main_function}, parameters=parameters)

# Define input and output features for Model
input_feature_type = ft.FeatureType()
array_type = ft.ArrayFeatureType(shape=[64], dataType=ft.ArrayFeatureType.ArrayDataType.FLOAT32)
input_feature_type.multiArrayType.CopyFrom(array_type)

output_feature_type = ft.FeatureType()
array_type = ft.ArrayFeatureType(shape=[10], dataType=ft.ArrayFeatureType.ArrayDataType.FLOAT32)
output_feature_type.multiArrayType.CopyFrom(array_type)

# Model description
desc = ml.ModelDescription(input=[ml.FeatureDescription(name='input0', type=input_feature_type)],
                           output=[ml.FeatureDescription(name='output0', type=output_feature_type)])
# Create ML Model
model = ml.Model(description=desc, program=simple_net)
# Save spec
save_spec(model, 'simple_layer.mlmodel')
print(model)
