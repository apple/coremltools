#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

SPACES = "  "

from .block import Block, Function, curr_block
from .builder import Builder
from .input_type import (SUPPORT_FLOAT_TYPES, SUPPORT_INT_TYPES, DefaultInputs,
                         InputSpec, InternalVar, ListInputType,
                         PyFunctionInputType, TensorInputType, TupleInputType)
from .operation import Operation, mil_list, precondition
from .program import (InputType, Placeholder, Program, Symbol,
                      get_existing_symbol, get_new_symbol,
                      get_new_variadic_symbol)
from .var import ListVar, Var
from .ops.defs._op_reqs import register_op

