import coremltools.proto.Program_pb2 as pm
from coremltools.converters.nnv2.builtin_types import builtins

builtin_to_proto_types = {
    builtins.float: pm.FLOAT32,
    builtins.double: pm.FLOAT64,
    builtins.int32: pm.INT32,
    builtins.uint8: pm.UINT8,
    builtins.int16: pm.INT16,
    builtins.int8: pm.INT8,
    builtins.str: pm.STRING,
    builtins.int64: pm.INT64,
    builtins.bool: pm.BOOL,
    builtins.uint16: pm.UINT16,
    builtins.uint32: pm.UINT32,
    builtins.uint64: pm.UINT64
}

proto_to_builtin_types = {v: k for k, v in builtin_to_proto_types.items()}

builtin_to_str = {
    builtins.float: 'fp32',
    builtins.double: 'fp64',
    builtins.int32: 'i32',
    builtins.uint8: 'u8',
    builtins.int16: 'i16',
    builtins.int8: 'i8',
    builtins.str: 'str',
    builtins.int64: 'i64',
    builtins.bool: 'bool',
    builtins.uint16: 'u16',
    builtins.uint32: 'u16',
    builtins.uint64: 'u64',
}
