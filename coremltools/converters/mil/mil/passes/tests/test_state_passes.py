#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import numpy as np

from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    get_op_types_in_program,
)


class TestCanonicalizeInplacePattern:
    @staticmethod
    def test_simple():
        """
        Given:

            mul = mul(state, x)
            add = add(mul, 1.0)
            update = coreml_update_state(state, mul)

        Return:

            mul = mul(state, x)
            update = coreml_update_state(state, mul)
            add = add(mul, 1.0)
        """
        SHAPE = (2,)

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=SHAPE, dtype=types.fp16),
                mb.StateTensorSpec(shape=SHAPE, dtype=types.fp16),
            ],
            opset_version=AvailableTarget.iOS18,
        )
        def prog(x, state):
            read = mb.read_state(input=state)
            mul = mb.mul(x=read, y=x)
            add = mb.add(x=mul, y=np.float16(1.0))
            update = mb.coreml_update_state(state=state, value=mul)
            return add

        assert get_op_types_in_program(prog) == ["read_state", "mul", "add", "coreml_update_state"]

        apply_pass_and_basic_check(prog, "common::canonicalize_inplace_pattern")
        assert get_op_types_in_program(prog) == ["read_state", "mul", "coreml_update_state", "add"]

    @staticmethod
    def test_irrelevant_ops_jam():
        """
        Given:

            relu = relu(x)
            mul = mul(state, x)
            tanh = tanh(x)
            add = add(mul, 1.0)
            update = coreml_update_state(state, mul)
            softmax = softmax(x)

        Where ``relu``, ``tanh``, and ``softmax`` are irrelevant to state

        Return:

            relu = relu(x)
            mul = mul(state, x)
            update = coreml_update_state(state, mul)
            tanh = tanh(x)
            add = add(mul, 1.0)
            softmax = softmax(x)
        """
        SHAPE = (2, 3)

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=SHAPE, dtype=types.fp16),
                mb.StateTensorSpec(shape=SHAPE, dtype=types.fp16),
            ],
            opset_version=AvailableTarget.iOS18,
        )
        def prog(x, state):
            read = mb.read_state(input=state)
            relu = mb.relu(x=x)
            mul = mb.mul(x=read, y=x)
            tanh = mb.tanh(x=x)
            add = mb.add(x=mul, y=np.float16(1.0))
            update = mb.coreml_update_state(state=state, value=mul)
            softmax = mb.softmax(x=x)
            return add, relu, tanh, softmax

        assert get_op_types_in_program(prog) == [
            "read_state",
            "relu",
            "mul",
            "tanh",
            "add",
            "coreml_update_state",
            "softmax",
        ]

        apply_pass_and_basic_check(prog, "common::canonicalize_inplace_pattern")
        assert get_op_types_in_program(prog) == [
            "read_state",
            "relu",
            "mul",
            "coreml_update_state",
            "tanh",
            "add",
            "softmax",
        ]


class TestPreferStateInDownstream:
    @staticmethod
    def test_simple():
        """
        Given:

            mul = mul(state, x)
            update = coreml_update_state(state, mul)
            add = add(mul, y)

        Return:

            mul = mul(state, x)
            update = coreml_update_state(state, mul)
            add = add(update, y)
        """
        SHAPE = (2, 3, 5)

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=SHAPE, dtype=types.fp16),
                mb.StateTensorSpec(shape=SHAPE, dtype=types.fp16),
            ],
            opset_version=AvailableTarget.iOS18,
        )
        def prog(x, state):
            read = mb.read_state(input=state)
            mul = mb.mul(x=x, y=read)
            update = mb.coreml_update_state(state=state, value=mul)
            add = mb.add(x=x, y=mul)
            return add

        mul_op = prog.find_ops(op_type="mul")[0]
        add_op = prog.find_ops(op_type="add")[0]
        assert add_op.y is mul_op.outputs[0]

        apply_pass_and_basic_check(prog, "common::prefer_state_in_downstream")
        coreml_update_state_op = prog.find_ops(op_type="coreml_update_state")[0]
        add_op = prog.find_ops(op_type="add")[0]
        assert add_op.y is coreml_update_state_op.outputs[0]

    @staticmethod
    def test_no_affect_if_var_is_input_and_output():
        """
        If the val of the coreml_update_state op is both block input and output,
        the graph pass should have no affects on it.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1,), dtype=types.fp16),
                mb.StateTensorSpec(shape=(1,), dtype=types.fp16),
            ],
            opset_version=AvailableTarget.iOS18,
        )
        def prog(x, state):
            mb.coreml_update_state(state=state, value=x)
            return x

        apply_pass_and_basic_check(prog, "common::prefer_state_in_downstream")
        block = prog.functions["main"]
        assert block.outputs[0] == list(block.inputs.values())[0]

    @staticmethod
    def test_no_other_child_op():
        """
        If the val of the coreml_update_state doesn't feed into any other op, and only serves as a block output,
        the graph pass has no affects.
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1,), dtype=types.fp16),
                mb.StateTensorSpec(shape=(1,), dtype=types.fp16),
            ],
            opset_version=AvailableTarget.iOS18,
        )
        def prog(x, state):
            x = mb.sin(x=x)
            mb.coreml_update_state(state=state, value=x)
            return x

        apply_pass_and_basic_check(prog, "common::prefer_state_in_downstream")
        block = prog.functions["main"]
        sin_op = prog.find_ops(op_type="sin")[0]
        assert block.outputs[0] == sin_op.outputs[0]

    @staticmethod
    def test_output_with_affect():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1,), dtype=types.fp16),
                mb.StateTensorSpec(shape=(1,), dtype=types.fp16),
            ],
            opset_version=AvailableTarget.iOS18,
        )
        def prog(x, state):
            x = mb.sin(x=x)
            mb.coreml_update_state(state=state, value=x)
            cos = mb.cos(x=x)
            return x, cos

        apply_pass_and_basic_check(prog, "common::prefer_state_in_downstream")
        block = prog.functions["main"]

        update_state_op = prog.find_ops(op_type="coreml_update_state")[0]
        assert block.outputs[0] == update_state_op.outputs[0]

        cos_op = prog.find_ops(op_type="cos")[0]
        assert update_state_op.outputs[0] == cos_op.x

    @staticmethod
    def test_only_feeds_in_update_state():
        """
        If value only feeds into multiple coreml_update_state ops, the graph pass has no affects
        """

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1,), dtype=types.fp16),
                mb.StateTensorSpec(shape=(1,), dtype=types.fp16),
            ],
            opset_version=AvailableTarget.iOS18,
        )
        def prog(x, state):
            x = mb.sin(x=x)
            mb.coreml_update_state(state=state, value=x)
            mb.coreml_update_state(state=state, value=x)
            mb.coreml_update_state(state=state, value=x)
            return x

        apply_pass_and_basic_check(prog, "common::prefer_state_in_downstream")
        block = prog.functions["main"]

        sin_op = prog.find_ops(op_type="sin")[0]
        update_state_ops = prog.find_ops(op_type="coreml_update_state")
        for op in update_state_ops:
            assert op.value == sin_op.outputs[0]

        assert block.outputs[0] == sin_op.outputs[0]

    @staticmethod
    def test_feeds_in_update_state_and_other_op():
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1,), dtype=types.fp16),
                mb.StateTensorSpec(shape=(1,), dtype=types.fp16),
            ],
            opset_version=AvailableTarget.iOS18,
        )
        def prog(x, state):
            x = mb.sin(x=x)
            mb.coreml_update_state(state=state, value=x)
            mb.coreml_update_state(state=state, value=x)
            return x, mb.identity(x=x)

        apply_pass_and_basic_check(prog, "common::prefer_state_in_downstream")
        block = prog.functions["main"]

        sin_op = prog.find_ops(op_type="sin")[0]
        update_state_ops = prog.find_ops(op_type="coreml_update_state")
        assert update_state_ops[1].value == update_state_ops[0].outputs[0]
        assert block.outputs[0] == update_state_ops[1].outputs[0]

        identity_op = prog.find_ops(op_type="identity")[0]
        assert identity_op.x == update_state_ops[1].outputs[0]

    @staticmethod
    def test_invalid_if_not_canonical():
        """
        Since the inplace op is not in canonical pattern, there is nothing this graph pass can do
        """
        SHAPE = (2, 3, 5, 7)

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=SHAPE, dtype=types.fp16),
                mb.StateTensorSpec(shape=SHAPE, dtype=types.fp16),
            ],
            opset_version=AvailableTarget.iOS18,
        )
        def prog(x, state):
            read = mb.read_state(input=state)
            mul = mb.mul(x=x, y=read)
            add = mb.add(x=x, y=mul)
            update = mb.coreml_update_state(state=state, value=mul)
            return add

        mul_op = prog.find_ops(op_type="mul")[0]
        add_op = prog.find_ops(op_type="add")[0]
        assert add_op.y is mul_op.outputs[0]

        apply_pass_and_basic_check(prog, "common::prefer_state_in_downstream")
        mul_op = prog.find_ops(op_type="mul")[0]
        coreml_update_state_op = prog.find_ops(op_type="coreml_update_state")[0]
        add_op = prog.find_ops(op_type="add")[0]
        assert add_op.y is mul_op.outputs[0]
        assert add_op.y is not coreml_update_state_op.outputs[0]
