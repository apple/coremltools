#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.builder import Builder as mb
from coremltools.converters.mil.mil.ops.defs.iOS18 import _IOS18_TARGET
from coremltools.converters.mil.testing_utils import get_op_types_in_program


class TestCoreMLUpdateState:
    @staticmethod
    def test_update_tensor_state_builder():
        @mb.program(
            input_specs=[mb.StateTensorSpec((2, 3)), mb.TensorSpec((2, 3))],
            opset_version=_IOS18_TARGET,
        )
        def prog(x, value):
            return mb.coreml_update_state(state=x, value=value)

        update_state_op = prog.find_ops("coreml_update_state")[0]
        assert types.is_state(update_state_op.state._sym_type)
        assert types.is_tensor(update_state_op.outputs[0]._sym_type)

    @staticmethod
    def test_update_tensor_state_builder_invalid():
        # Update state with value of different shape
        with pytest.raises(
            ValueError,
            match="State wrapped type tensor\[2,3,fp32\] not matched with the value's sym_type tensor\[3,2,fp32\]",
        ):

            @mb.program(
                input_specs=[mb.StateTensorSpec((2, 3)), mb.TensorSpec((3, 2))],
                opset_version=_IOS18_TARGET,
            )
            def prog(x, value):
                return mb.coreml_update_state(state=x, value=value)

        # Update state with value of different dtype
        with pytest.raises(
            ValueError,
            match="State wrapped type tensor\[2,3,fp32\] not matched with the value's sym_type tensor\[2,3,fp16\]",
        ):

            @mb.program(
                input_specs=[mb.StateTensorSpec((2, 3)), mb.TensorSpec((2, 3), dtype=types.fp16)],
                opset_version=_IOS18_TARGET,
            )
            def prog(x, value):
                return mb.coreml_update_state(state=x, value=value)

        @staticmethod
        def test_simple_stateful_model_builder():
            @mb.program(
                input_specs=[mb.StateTensorSpec((2, 3)), mb.TensorSpec((2, 3))],
                opset_version=_IOS18_TARGET,
            )
            def prog(x, value):
                read_val = mb.read_state(input=x)
                add = mb.add(x=read_val, y=value)
                return mb.coreml_update_state(state=x, value=add)

            assert get_op_types_in_program(prog) == ["read_state", "add", "coreml_update_state"]
