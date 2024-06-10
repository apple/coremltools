#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from .annotate import annotate, apply_delayed_types, class_annotate, delay_type
from .get_type_info import get_type_info
from .global_methods import global_remap
from .type_bool import bool, is_bool
from .type_complex import complex, complex64, complex128, is_complex
from .type_dict import dict, empty_dict
from .type_double import double, float, fp16, fp32, fp64, is_float
from .type_globals_pseudo_type import globals_pseudo_type
from .type_int import (
    _SUB_BYTE_TYPES,
    SUB_BYTE_DTYPE_METADATA_KEY,
    int4,
    int8,
    int16,
    int32,
    int64,
    is_int,
    is_sub_byte,
    np_int4_dtype,
    np_uint1_dtype,
    np_uint2_dtype,
    np_uint3_dtype,
    np_uint4_dtype,
    np_uint6_dtype,
    uint,
    uint1,
    uint2,
    uint3,
    uint4,
    uint6,
    uint8,
    uint16,
    uint32,
    uint64,
)
from .type_list import empty_list, is_list, list
from .type_mapping import (
    BUILTIN_TO_PROTO_TYPES,
    PROTO_TO_BUILTIN_TYPE,
    builtin_to_string,
    get_nbits_int_builtin_type,
    is_builtin,
    is_dict,
    is_primitive,
    is_scalar,
    is_str,
    is_subtype,
    is_tensor,
    is_tuple,
    np_dtype_to_py_type,
    nptype_from_builtin,
    numpy_type_to_builtin_type,
    numpy_val_to_builtin_val,
    promote_dtypes,
    promote_types,
    string_to_builtin,
    type_to_builtin_type,
)
from .type_state import is_state, state
from .type_str import str
from .type_tensor import (
    is_compatible_type,
    is_tensor_and_is_compatible,
    tensor,
    tensor_has_complete_shape,
)
from .type_tuple import tuple
from .type_unknown import unknown
from .type_void import void

apply_delayed_types()
