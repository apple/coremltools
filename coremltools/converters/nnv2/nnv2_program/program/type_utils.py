from coremltools.converters.nnv2.builtin_types import builtins

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
