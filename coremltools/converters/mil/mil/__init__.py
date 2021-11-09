#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
SPACES = "  "

from .block import curr_block, Block, Function
from .input_type import (
    BoolInputType,
    BoolTensorInputType,
    DefaultInputs,
    FloatInputType,
    FloatTensorInputType,
    InputSpec,
    IntInputType,
    IntOrFloatInputType,
    IntOrFloatOrBoolInputType,
    IntTensorInputType,
    InternalInputType,
    InternalScalarOrTensorInputType,
    InternalStringInputType,
    InternalVar,
    ListInputType,
    ListOrScalarOrTensorInputType,
    PyFunctionInputType,
    SUPPORT_FLOAT_TYPES,
    SUPPORT_INT_TYPES,
    ScalarOrTensorInputType,
    StringInputType,
    TensorInputType,
    TupleInputType
)
from .operation import mil_list, precondition, Operation
from .program import (
    get_existing_symbol,
    get_new_symbol,
    get_new_variadic_symbol,
    InputType,
    Placeholder,
    Program,
    Symbol,
)
from .var import ListVar, Var

from .builder import Builder
from .ops.defs._op_reqs import register_op
