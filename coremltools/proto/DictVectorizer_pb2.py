# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: DictVectorizer.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import DataStructures_pb2 as DataStructures__pb2
try:
  FeatureTypes__pb2 = DataStructures__pb2.FeatureTypes__pb2
except AttributeError:
  FeatureTypes__pb2 = DataStructures__pb2.FeatureTypes_pb2

from .DataStructures_pb2 import *

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14\x44ictVectorizer.proto\x12\x14\x43oreML.Specification\x1a\x14\x44\x61taStructures.proto\"\x8f\x01\n\x0e\x44ictVectorizer\x12;\n\rstringToIndex\x18\x01 \x01(\x0b\x32\".CoreML.Specification.StringVectorH\x00\x12\x39\n\x0cint64ToIndex\x18\x02 \x01(\x0b\x32!.CoreML.Specification.Int64VectorH\x00\x42\x05\n\x03MapB\x02H\x03P\x00\x62\x06proto3')



_DICTVECTORIZER = DESCRIPTOR.message_types_by_name['DictVectorizer']
DictVectorizer = _reflection.GeneratedProtocolMessageType('DictVectorizer', (_message.Message,), {
  'DESCRIPTOR' : _DICTVECTORIZER,
  '__module__' : 'DictVectorizer_pb2'
  # @@protoc_insertion_point(class_scope:CoreML.Specification.DictVectorizer)
  })
_sym_db.RegisterMessage(DictVectorizer)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'H\003'
  _DICTVECTORIZER._serialized_start=69
  _DICTVECTORIZER._serialized_end=212
# @@protoc_insertion_point(module_scope)
