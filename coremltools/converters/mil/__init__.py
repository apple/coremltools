#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# This import should be pruned rdar://84519338
from .mil import (
    block,
    Block,
    BoolInputType,
    BoolTensorInputType,
    builder,
    Builder,
    curr_block,
    DefaultInputs,
    FloatInputType,
    FloatTensorInputType,
    Function,
    get_existing_symbol,
    get_new_symbol,
    get_new_variadic_symbol,
    input_type,
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
    ListVar,
    mil_list,
    operation,
    Operation,
    ops,
    Placeholder,
    precondition,
    program,
    Program,
    PyFunctionInputType,
    register_op,
    SPACES,
    SUPPORT_FLOAT_TYPES,
    SUPPORT_INT_TYPES,
    ScalarOrTensorInputType,
    StringInputType,
    Symbol,
    TensorInputType,
    TupleInputType,
    types,
    var,
    Var,
    visitors
)

from .frontend.torch import register_torch_op

from .input_types import (
    ClassifierConfig,
    InputType,
    TensorType,
    ImageType,
    RangeDim,
    Shape,
    EnumeratedShapes,
)

from coremltools.converters.mil.frontend.tensorflow.tf_op_registry import register_tf_op
