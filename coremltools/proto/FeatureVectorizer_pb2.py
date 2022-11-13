# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: FeatureVectorizer.proto

import sys

_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pb2
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='FeatureVectorizer.proto',
  package='CoreML.Specification',
  syntax='proto3',
  serialized_pb=_b('\n\x17\x46\x65\x61tureVectorizer.proto\x12\x14\x43oreML.Specification\"\x98\x01\n\x11\x46\x65\x61tureVectorizer\x12\x46\n\tinputList\x18\x01 \x03(\x0b\x32\x33.CoreML.Specification.FeatureVectorizer.InputColumn\x1a;\n\x0bInputColumn\x12\x13\n\x0binputColumn\x18\x01 \x01(\t\x12\x17\n\x0finputDimensions\x18\x02 \x01(\x04\x42\x02H\x03\x62\x06proto3')
)




_FEATUREVECTORIZER_INPUTCOLUMN = _descriptor.Descriptor(
  name='InputColumn',
  full_name='CoreML.Specification.FeatureVectorizer.InputColumn',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='inputColumn', full_name='CoreML.Specification.FeatureVectorizer.InputColumn.inputColumn', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='inputDimensions', full_name='CoreML.Specification.FeatureVectorizer.InputColumn.inputDimensions', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=143,
  serialized_end=202,
)

_FEATUREVECTORIZER = _descriptor.Descriptor(
  name='FeatureVectorizer',
  full_name='CoreML.Specification.FeatureVectorizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='inputList', full_name='CoreML.Specification.FeatureVectorizer.inputList', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_FEATUREVECTORIZER_INPUTCOLUMN, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=50,
  serialized_end=202,
)

_FEATUREVECTORIZER_INPUTCOLUMN.containing_type = _FEATUREVECTORIZER
_FEATUREVECTORIZER.fields_by_name['inputList'].message_type = _FEATUREVECTORIZER_INPUTCOLUMN
DESCRIPTOR.message_types_by_name['FeatureVectorizer'] = _FEATUREVECTORIZER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FeatureVectorizer = _reflection.GeneratedProtocolMessageType('FeatureVectorizer', (_message.Message,), dict(

  InputColumn = _reflection.GeneratedProtocolMessageType('InputColumn', (_message.Message,), dict(
    DESCRIPTOR = _FEATUREVECTORIZER_INPUTCOLUMN,
    __module__ = 'FeatureVectorizer_pb2'
    # @@protoc_insertion_point(class_scope:CoreML.Specification.FeatureVectorizer.InputColumn)
    ))
  ,
  DESCRIPTOR = _FEATUREVECTORIZER,
  __module__ = 'FeatureVectorizer_pb2'
  # @@protoc_insertion_point(class_scope:CoreML.Specification.FeatureVectorizer)
  ))
_sym_db.RegisterMessage(FeatureVectorizer)
_sym_db.RegisterMessage(FeatureVectorizer.InputColumn)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('H\003'))
# @@protoc_insertion_point(module_scope)
