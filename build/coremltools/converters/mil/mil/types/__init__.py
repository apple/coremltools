# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
from .type_double import fp16, fp32, fp64, float, double, is_float
from .type_int import (
    int8,
    int16,
    int32,
    int64,
    int,
    uint8,
    uint16,
    uint32,
    uint64,
    uint,
    is_int,
)
from .type_str import str
from .type_bool import bool, is_bool
from .type_list import list, empty_list, is_list
from .type_tensor import (
    tensor,
    tensor_has_complete_shape,
    is_tensor_and_is_compatible,
    is_tensor_and_is_compatible_general_shape,
    is_compatible_type,
)
from .type_dict import dict, empty_dict
from .type_void import void
from .type_globals_pseudo_type import globals_pseudo_type
from .type_unknown import unknown
from .type_tuple import tuple
from .type_mapping import (
    is_primitive,
    is_scalar,
    is_tensor,
    is_tuple,
    is_dict,
    is_str,
    is_builtin,
    promote_types,
    promote_dtypes,
    numpy_val_to_builtin_val,
    builtin_to_string,
    numpy_type_to_builtin_type,
    type_to_builtin_type,
    is_subtype,
    string_to_builtin,
    nptype_from_builtin,
    np_dtype_to_py_type,
    proto_to_builtin_types,
    builtin_to_proto_types
)
from .annotate import annotate
from .annotate import class_annotate
from .annotate import apply_delayed_types
from .annotate import delay_type
from .get_type_info import get_type_info
from .global_methods import global_remap
from math import log, exp

apply_delayed_types()
