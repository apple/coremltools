#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, StateInputType
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil.ops.defs.iOS18 import _IOS18_TARGET


@register_op(opset_version=_IOS18_TARGET)
class read_state(Operation):
    """
    Read a state, copy its content into a new variable, and return the variable.
    The type of the output variable depends on the type that is wrapped inside the state,
    which could be ``types.tensor``.

    Parameters
    ----------
    input: state<ST> (Required)

    Returns
    -------
    ST

    Attributes
    ----------
    ST: tensor
    """

    input_spec = InputSpec(
        input=StateInputType(),
    )

    def type_inference(self):
        sym_type = self.input.sym_type.wrapped_type()
        if not types.is_tensor(sym_type):
            raise ValueError(
                f"State only supports wrapped type of types.tensor. Got {sym_type.__type_info__()}."
            )
        return sym_type
