#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import itertools

import numpy as np
import pytest

from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    assert_same_output_names,
    get_op_types_in_program,
)

backends = testing_reqs.backends


class TestConv1dDeompositionPasses:
    @pytest.mark.parametrize(
        "backend, has_strides, pad_type, has_pad, has_dilations, has_bias",
        itertools.product(
            backends,
            (True, False),
            ("valid", "custom", "same"),
            (True, False),
            (True, False),
            (True, False),
        ),
    )
    def test_conv1d_decomposition(
        self, backend, has_strides, pad_type, has_pad, has_dilations, has_bias
    ):
        """
        Input graph:
        input -> expand_dims -> conv2d -> squeeze -> out

        Output graph:
        input -> conv1d -> out
        """
        N, L = 2, 8
        C_in, C_out = 3, 4
        K = 3

        conv_kwargs = {"weight": np.random.rand(C_out, C_in, K), "pad_type": pad_type}
        if has_strides:
            conv_kwargs["strides"] = (2,)
        if has_pad:
            conv_kwargs["pad"] = (1, 1)
        if has_dilations:
            conv_kwargs["dilations"] = (2,)
        if has_bias:
            conv_kwargs["bias"] = np.random.rand(C_out)

        @mb.program(input_specs=[mb.TensorSpec(shape=(N, C_in, L))])
        def prog(x):
            y = mb.conv(x=x, **conv_kwargs)
            return y

        assert get_op_types_in_program(prog) == ["conv"]

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "nn_backend::decompose_conv1d"
        )
        assert get_op_types_in_program(prog) == ["expand_dims", "expand_dims", "conv", "squeeze"]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::const_elimination")
        assert get_op_types_in_program(prog) == ["expand_dims", "conv", "squeeze"]

        # infer output shape
        strides = conv_kwargs["strides"] if has_strides else (1,)
        pad = conv_kwargs["pad"] if has_pad else (0, 0)
        dilations = conv_kwargs["dilations"] if has_dilations else (1,)
        L_out = None
        if pad_type == "valid":
            L_out = (L - dilations[-1] * (K - 1) - 1) // strides[-1] + 1
        elif pad_type == "custom":
            L_out = (L + pad[-2] + pad[-1] - dilations[-1] * (K - 1) - 1) // strides[-1] + 1
        elif pad_type == "same":
            L_out = np.ceil(L / strides[-1])
        else:
            raise Exception("unsupported pad type")
        output_shape = (N, C_out, L_out)

        assert_model_is_valid(
            prog,
            {"x": (N, C_in, L)},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )

    @pytest.mark.parametrize("backend", backends)
    def test_conv1d_decomposition_dynamic_weight(self, backend):
        """
        Input graph:
        input -> expand_dims -> conv2d -> squeeze -> out

        Output graph:
        input -> conv1d -> out
        """
        N, L = 2, 9
        C_in, C_out = 4, 3
        K = 4

        strides = (2,)
        pad = (1, 1)
        # MIL convolution with dynamic weights does not support dilations != 1
        # see coremltools/coremltools/converters/mil/mil/ops/defs/iOS15/conv.py
        dilations = (1,)

        # infer L_out with pad_type fixed to custom
        L_out = (L + pad[-2] + pad[-1] - dilations[-1] * (K - 1) - 1) // strides[-1] + 1

        conv_kwargs = {
            "strides": strides,
            "pad_type": "custom",
            "pad": pad,
            "dilations": dilations,
        }

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(N, C_in, L)),
                mb.TensorSpec(shape=(C_out, C_in, K)),
            ]
        )
        def prog(x, weight):
            y = mb.conv(x=x, weight=weight, **conv_kwargs)
            return y

        assert get_op_types_in_program(prog) == ["conv"]

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "nn_backend::decompose_conv1d"
        )
        assert get_op_types_in_program(prog) == ["expand_dims", "expand_dims", "conv", "squeeze"]

        prev_prog, prev_block, block = apply_pass_and_basic_check(prog, "common::const_elimination")
        assert get_op_types_in_program(prog) == ["expand_dims", "expand_dims", "conv", "squeeze"]

        output_shape = (N, C_out, L_out)
        assert_model_is_valid(
            prog,
            {"x": (N, C_in, L), "weight": (C_out, C_in, K)},
            expected_output_shapes={block.outputs[0].name: output_shape},
            backend=backend,
        )


def test_commingle_loop_vars():
    def body(a, b):
        # b is a loop invariant
        return mb.add(x=a, y=b), b

    def cond(a, b):
        a_mean = mb.reduce_mean(x=a, axes=[0, 1])
        b_mean = mb.reduce_mean(x=b, axes=[0, 1])
        return mb.less(x=a_mean, y=b_mean)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 2)), mb.TensorSpec(shape=(1, 2)),]
    )
    def prog(a, b):
        return mb.while_loop(_cond=cond, _body=body, loop_vars=(a, b))

    while_op = prog.find_ops(op_type="while_loop", exactly_one=True)[0]
    assert while_op.blocks[0].inputs[0].name == "a_x0"
    assert while_op.blocks[0].inputs[1].name == "b_x0"

    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY["nn_backend::commingle_loop_vars"](prog)
    assert_same_output_names(prev_prog, prog)

    while_op = prog.find_ops(op_type="while_loop", exactly_one=True)[0]
    assert while_op.blocks[0].inputs[0].name == while_op.outputs[0].name
    assert while_op.blocks[0].inputs[1].name == while_op.outputs[1].name

    prog.validate()

    # The program is not ssa and thus cannot be converted


def test_handle_return_inputs_as_outputs():
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 2)), mb.TensorSpec(shape=(1, 2)),]
    )
    def prog(a, b):
        return mb.mul(x=a, y=2.), b

    prev_main_output_names = [o.name for o in prog["main"].outputs]
    assert prog["main"].outputs[1].op is None  # output comes from input

    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY["nn_backend::handle_return_inputs_as_outputs"](prog)
    assert_same_output_names(prev_prog, prog)

    assert prog["main"].outputs[1].op is not None  # output comes from an op
    assert prog["main"].outputs[1].op.op_type == "identity"

    with pytest.raises(ValueError, match='used both as function\'s input and output'):
        # prog has input and output names 'b' that refer to different vars
        # This program can pass if we disable 'dedup_op_and_var_names' pass
        assert_model_is_valid(prog, {"a": (1, 2), "b": (1, 2)})


def test_handle_unused_inputs():
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 2)),]
    )
    def prog(unused_input):
        return mb.const(val=[3, 2])

    prev_prog = copy.deepcopy(prog)
    PASS_REGISTRY["nn_backend::handle_unused_inputs"](prog)
    assert_same_output_names(prev_prog, prog)

    id_op = prog.find_ops(op_type="identity", exactly_one=True)[0]
    # Assert that input var is consumed by an identity op.
    assert id_op in prog["main"].inputs["unused_input"].child_ops

    assert_model_is_valid(prog, {"unused_input": (1, 2)})
