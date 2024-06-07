#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.input_type import InputSpec, StateInputType, TensorInputType
from coremltools.converters.mil.mil.operation import Operation
from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op


@register_op(namespace="coreml")
class coreml_update_state(Operation):
    """
    Copy the content of a variable into a state and return the copy of the variable.
    The type of the variable must match the type that is wrapped inside the state.
    This is a coreml dialect op to simplify the program. When
    loading into MIL, the following transformation is done:

    .. code-block::

        %x = coreml_update_state(state=%state, value=%value)

        -->

        write_state(state=%state, value=%value)
        %x = read_state(input=%state)

    Parameters
    ----------
    state: state<ST> (Required)
    value: ST (Required)

    Returns
    -------
    ST

    Attributes
    ----------
    ST: tensor
    """

    input_spec = InputSpec(
        state=StateInputType(),
        value=TensorInputType(type_domain="T"),
    )

    type_domains = {
        "T": (
            types.fp16,
            types.fp32,
            types.int8,
            types.int16,
            types.int32,
            types.uint8,
            types.uint16,
            types.bool,
        ),
    }

    def type_inference(self):
        state_wrapped_type = self.state._sym_type.wrapped_type()
        if not state_wrapped_type == self.value.sym_type:
            raise ValueError(
                f"State wrapped type {state_wrapped_type.__type_info__()} not matched with the value's sym_type {self.value.sym_type.__type_info__()}."
            )
        return self.value.sym_type
