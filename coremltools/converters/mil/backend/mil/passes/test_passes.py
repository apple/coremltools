#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import itertools
import numpy as np
import copy

# import mil internal ops to add it to the builder
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil import types
from coremltools.converters.mil.mil.types import string_to_builtin, builtin_to_string, promote_types

# Set the testing backend
import coremltools.converters.mil.testing_reqs as testing_reqs

from coremltools.converters.mil.testing_utils import (
    get_op_types_in_program,
    apply_pass_and_basic_check,
    assert_model_is_valid,
)


class TestAdjustToSupportedTypes:

    def test_basic(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 1, 1), dtype=types.bool),
                                 mb.TensorSpec(shape=(1, 1, 1, 1), dtype=types.int32),
                                 mb.TensorSpec(shape=(1, 1, 1, 1), dtype=types.fp32)])
        def prog(x, y, z):
            out = mb.logical_not(x=x)
            return (out, y, z)
        prog.functions['not_main'] = copy.deepcopy(prog.functions['main'])

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::adjust_io_to_supported_types"
        )

        """
        Input graph:

        func main(bool x, int32 y, fp32 z) {
            bool  out = logical_not(x)
        } -> (out, y, z)

        becomes

        func main(fp32 x, int32 y, fp32 z) {
           bool  x_casted = cast(x)
           bool  out__pre__output__fp32__cast = logical_not(x_casted)
           fp32  out = cast(out__pre__output__fp32__cast)
        } -> (out, y, z)
        """
        assert get_op_types_in_program(prev_prog) == ['logical_not']
        assert get_op_types_in_program(prog) == ['cast', 'logical_not', 'cast']

        prev_inputs = list(prev_prog.functions['main'].inputs.items())
        inputs = list(prog.functions['main'].inputs.items())
        assert prev_inputs[0][1].name == inputs[0][1].name
        assert inputs[0][1].dtype == types.fp32
        for i in range(1, len(inputs)):
            assert prev_inputs[i][1].name == inputs[i][1].name
            assert prev_inputs[i][1].dtype == inputs[i][1].dtype

        prev_outputs = prev_prog.functions['main'].outputs
        outputs = prog.functions['main'].outputs
        assert prev_outputs[0].name == outputs[0].name
        assert outputs[0].dtype == types.fp32
        for i in range(1, len(outputs)):
            assert prev_outputs[i].name == outputs[i].name
            assert prev_outputs[i].dtype == outputs[i].dtype

        """
        Input graph:

        func not_main(bool x, int32 y, fp32 z) {
            bool  out = logical_not(x)
        } -> (out, y, z)

        is identical after the pass.
        """
        assert get_op_types_in_program(prev_prog, 'not_main') == ['logical_not']
        assert get_op_types_in_program(prog, 'not_main') == ['logical_not']

        prev_inputs = list(prev_prog.functions['not_main'].inputs.items())
        inputs = list(prog.functions['not_main'].inputs.items())
        for i in range(0, len(inputs)):
            assert prev_inputs[i][1].name == inputs[i][1].name
            assert prev_inputs[i][1].dtype == inputs[i][1].dtype

        prev_outputs = prev_prog.functions['not_main'].outputs
        outputs = prog.functions['not_main'].outputs
        for i in range(0, len(outputs)):
            assert prev_outputs[i].name == outputs[i].name
            assert prev_outputs[i].dtype == outputs[i].dtype

    def test_int64_input(self):
        """
        Input graph:

        func main(int64 x) {
        } -> (x)

        becomes

        func main(int32 x) {
        } -> (x)
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 1, 1), dtype=types.int64)])
        def prog(x):
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::adjust_io_to_supported_types"
        )

        prev_inputs = list(prev_prog.functions['main'].inputs.items())
        inputs = list(prog.functions['main'].inputs.items())
        assert prev_inputs[0][1].name == inputs[0][1].name
        assert inputs[0][1].dtype == types.int32

    def test_float64_input(self):
        """
        Input graph:

        func main(float64 x) {
        } -> (x)

        becomes

        func main(float32 x) {
        } -> (x)
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 1, 1), dtype=types.fp64)])
        def prog(x):
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::adjust_io_to_supported_types"
        )

        prev_inputs = list(prev_prog.functions['main'].inputs.items())
        inputs = list(prog.functions['main'].inputs.items())
        assert prev_inputs[0][1].name == inputs[0][1].name
        assert inputs[0][1].dtype == types.fp32

    def test_int8_input(self):
        """
        Input graph:

        func main(int8 x) {
        } -> (x)

        becomes

        func main(int32 x) {
        } -> (x)
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 1, 1), dtype=types.int8)])
        def prog(x):
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::adjust_io_to_supported_types"
        )

        prev_inputs = list(prev_prog.functions['main'].inputs.items())
        inputs = list(prog.functions['main'].inputs.items())
        assert prev_inputs[0][1].name == inputs[0][1].name
        assert inputs[0][1].dtype == types.int32

    def test_subblock(self):
        """
        Input graph:

        func main(float64 a, float32 b) {
            float64 out_0, float32 out_1 = while_loop(a, b,
                (float64 a, float32 b) {
                    bool cond = less(a, b)
                } -> (cond)
                (float64 a, float32 b) {
                    float64 temp = const(1)
                    float64 out = add(a, b)
                } -> (out, b)
            );
        } -> (out_0, out_1)

        becomes

        func main(float32 a, float32 b) {
            float32 out_0, float32 out_1 = while_loop(a, b,
                (float32 a, float32 b) {
                    bool cond = less(a, b)
                } -> (cond)
                (float32 a, float32 b) {
                    float32 temp = const(1)
                    float32 out = add(a, b)
                } -> (out, b)
            );
        } -> (out_0, out_1)
        """
        def body(a, b):
            return mb.add(x=a, y=np.float64(1)), b

        def cond(a, b):
            return mb.less(x=a, y=b)

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp64),
                                 mb.TensorSpec(shape=(1,), dtype=types.fp32)])
        def prog(a, b):
            return mb.while_loop(_cond=cond, _body=body, loop_vars=(a, b))

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::adjust_io_to_supported_types"
        )

        prev_inputs = list(prev_prog.functions['main'].inputs.items())
        inputs = list(prog.functions['main'].inputs.items())
        for i in range(0, len(prev_inputs)):
            assert prev_inputs[i][1].name == inputs[i][1].name
            assert inputs[i][1].dtype == types.fp32

        assert get_op_types_in_program(prev_prog) == ['while_loop']
        assert get_op_types_in_program(prog) == ['while_loop']

        def assert_block_inputs(prev_inputs, inputs):
            for i in range(0, len(prev_inputs)):
                assert prev_inputs[i].name == inputs[i].name
                assert inputs[i].dtype == types.fp32

        subblocks = prog.functions['main'].operations[0].blocks
        prev_subblocks = prev_prog.functions['main'].operations[0].blocks
        for i in range(0, len(subblocks)):
            assert_block_inputs(prev_subblocks[i].inputs, subblocks[i].inputs)

    def test_adjust_cast(self):
        """
        Input graph:

        func main(int32 x) {
            fp64 y = cast(x=x, dtype="fp64")
        } -> (y)

        becomes

        func main(int32 x) {
            fp32 y = cast(x=x, dtype="fp32")
        } -> (y)
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 1, 1), dtype=types.int32)])
        def prog(x):
            y = mb.cast(x=x, dtype="fp64")
            return y

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::adjust_io_to_supported_types"
        )

        assert get_op_types_in_program(prev_prog) == ['cast']
        assert get_op_types_in_program(prog) == ['cast']

        prev_cast = prev_prog.functions['main'].operations[1]
        cast = prog.functions['main'].operations[2]

        assert prev_cast.dtype.val == "fp64"
        assert prev_cast.outputs[0].dtype == types.fp64

        assert cast.dtype.val == "fp32"
        assert cast.outputs[0].dtype == types.fp32

    def test_adjust_redundant_cast(self):
        """
        Input graph:

        func main(int32 x) {
            int64 y = cast(x=x, dtype="int64")
        } -> (y)

        becomes

        func main(int32 x) {
        } -> (x)
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 1, 1), dtype=types.int32)])
        def prog(x):
            y = mb.cast(x=x, dtype="int64")
            return y

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::adjust_io_to_supported_types"
        )

        assert get_op_types_in_program(prev_prog) == ['cast']
        assert get_op_types_in_program(prog) == []

class TestImagePreprocessingPass:

    def test_program_grayscale(self):
        """
        Input graph:

        main(x: ImageType(color_layout="G", channel_first=True)) {
            y1 = relu(x)
            y2 = relu(x)
            output = add(y1, y2)
        } [output]

        Output graph:

        main(x: ImageType(channel_first=True)) {
            y1 = relu(x)
            y2 = relu(x)
            output = add(y1, y2)
        } [output]
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 20, 20))])
        def prog(x):
            y1 = mb.relu(x=x)
            y2 = mb.relu(x=x)
            z = mb.add(x=y1, y=y2)
            return z

        prog.main_input_types = (ct.ImageType(name='x',
                                              shape=[1, 1, 20, 20],
                                              color_layout="G",
                                              channel_first=True),)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::insert_image_preprocessing_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "relu", "add"]
        assert get_op_types_in_program(prog) == ["relu", "relu", "add"]

    def test_program_grayscale_with_scale(self):
        """
        Input graph:

        main(x: ImageType(scale=2.0, color_layout="G", channel_first=True)) {
            y1 = relu(x)
            y2 = relu(x)
            output = add(y1, y2)
        } [output]

        Output graph:

        main(x: ImageType(channel_first=True)) {
            y = mul(x, 2)
            y1 = relu(y)
            y2 = relu(y)
            output = add(y1, y2)
        } [output]
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 20, 20))])
        def prog(x):
            y1 = mb.relu(x=x)
            y2 = mb.relu(x=x)
            z = mb.add(x=y1, y=y2)
            return z

        prog.main_input_types = (ct.ImageType(name='x',
                                              shape=[1, 1, 20, 20],
                                              scale=2.0,
                                              color_layout="G",
                                              channel_first=True),)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::insert_image_preprocessing_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "relu", "add"]
        assert get_op_types_in_program(prog) == ["mul", "relu", "relu", "add"]
        scale_op = prog.find_ops(op_type="mul", exactly_one=True)[0]
        assert scale_op.y.val == 2.0

    def test_program_grayscale_with_bias(self):
        """
        Input graph:

        main(x: ImageType(bias=2.0, color_layout="G", channel_first=True)) {
            y1 = relu(x)
            y2 = relu(x)
            output = add(y1, y2)
        } [output]

        Output graph:

        main(x: ImageType(channel_first=True)) {
            y = add(x, 2)
            y1 = relu(y)
            y2 = relu(y)
            output = add(y1, y2)
        } [output]
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 20, 20))])
        def prog(x):
            y1 = mb.relu(x=x)
            y2 = mb.relu(x=x)
            z = mb.add(x=y1, y=y2)
            return z

        prog.main_input_types = (ct.ImageType(name='x',
                                             shape=[1, 1, 20, 20],
                                             bias=2.0,
                                             color_layout="G",
                                             channel_first=True),)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::insert_image_preprocessing_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "relu", "add"]
        assert get_op_types_in_program(prog) == ["add", "relu", "relu", "add"]
        add_op = prog.find_ops(op_type="add", exactly_one=False)[0]
        assert add_op.y.val == 2.0

    def test_program_grayscale_with_scale_bias(self):
        """
        Input graph:

        main(x: ImageType(scale=2.0, bias=2.0, color_layout="G", channel_first=True)) {
            y1 = relu(x)
            y2 = relu(x)
            output = add(y1, y2)
        } [output]

        Output graph:

        main(x: ImageType(channel_first=True)) {
            y_scaled = mul(x, 2)
            y = add(y_scaled, 2)
            y1 = relu(y)
            y2 = relu(y)
            output = add(y1, y2)
        } [output]
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 20, 20))])
        def prog(x):
            y1 = mb.relu(x=x)
            y2 = mb.relu(x=x)
            z = mb.add(x=y1, y=y2)
            return z

        prog.main_input_types = (ct.ImageType(name='x',
                                              shape=[1, 1, 20, 20],
                                              scale=2.0,
                                              bias=2.0,
                                              color_layout="G",
                                              channel_first=True),)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::insert_image_preprocessing_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "relu", "add"]
        assert get_op_types_in_program(prog) == ["mul", "add", "relu", "relu", "add"]
        scale_op = prog.find_ops(op_type="mul", exactly_one=True)[0]
        assert scale_op.y.val == 2.0
        add_op = prog.find_ops(op_type="add", exactly_one=False)[0]
        assert add_op.y.val == 2.0

    def test_program_rgb(self):
        """
        Input graph:

        main(x: ImageType(color_layout="RGB", channel_first=True)) {
            y1 = relu(x)
            y2 = relu(x)
            output = add(y1, y2)
        } [output]

        Output graph:

        main(x: ImageType(channel_first=True)) {
            y1 = relu(x)
            y2 = relu(x)
            output = add(y1, y2)
        } [output]
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 20, 20))])
        def prog(x):
            y1 = mb.relu(x=x)
            y2 = mb.relu(x=x)
            z = mb.add(x=y1, y=y2)
            return z

        prog.main_input_types = (ct.ImageType(name='x',
                                              shape=[1, 3, 20, 20],
                                              color_layout="RGB",
                                              channel_first=True),)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::insert_image_preprocessing_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "relu", "add"]
        assert get_op_types_in_program(prog) == ["relu", "relu", "add"]

    def test_program_rgb_scale_bias(self):
        """
        Input graph:

        main(x: ImageType(color_layout="RGB", scale=2.0, bias=[1.0, 2.0, 3.0], channel_first=True)) {
            y1 = relu(x)
            y2 = relu(x)
            output = add(y1, y2)
        } [output]

        Output graph:

        main(x: ImageType(channel_first=True)) {
            y = mul(x, scale)
            y_bias = add(y, bias)
            y1 = relu(y_bias)
            y2 = relu(y_bias)
            output = add(y1, y2)
        } [output]
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 20, 20))])
        def prog(x):
            y1 = mb.relu(x=x)
            y2 = mb.relu(x=x)
            z = mb.add(x=y1, y=y2)
            return z

        prog.main_input_types = (ct.ImageType(name='x',
                                              shape=[1, 3, 20, 20],
                                              scale=2.0,
                                              bias=[1.0, 2.0, 3.0],
                                              color_layout="RGB",
                                              channel_first=True),)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::insert_image_preprocessing_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "relu", "add"]
        assert get_op_types_in_program(prog) == ["mul", "add", "relu", "relu", "add"]
        scale_op = prog.find_ops(op_type="mul", exactly_one=True)[0]
        assert scale_op.y.val == 2.0
        add_op = prog.find_ops(op_type="add", exactly_one=False)[0]
        assert np.all(add_op.y.val == np.array([1.0, 2.0, 3.0]).reshape([1, 3, 1, 1]))

    def test_program_bgr(self):
        """
        Input graph:

        main(x: ImageType(color_layout="BGR", channel_first=True)) {
            y1 = relu(x)
            y2 = relu(x)
            output = add(y1, y2)
        } [output]

        Output graph:

        main(x: ImageType(channel_first=True)) {
            y1 = relu(x)
            y2 = relu(x)
            output = add(y1, y2)
        } [output]
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 20, 20))])
        def prog(x):
            y1 = mb.relu(x=x)
            y2 = mb.relu(x=x)
            z = mb.add(x=y1, y=y2)
            return z

        prog.main_input_types = (ct.ImageType(name='x',
                                              shape=[1, 3, 20, 20],
                                              color_layout="BGR",
                                              channel_first=True),)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::insert_image_preprocessing_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "relu", "add"]
        assert get_op_types_in_program(prog) == ["relu", "relu", "add"]

    def test_program_bgr_scale_bias(self):
        """
        Input graph:

        main(x: ImageType(color_layout="BGR", scale=2.0, bias=[1.0, 2.0, 3.0], channel_first=True)) {
            y1 = relu(x)
            y2 = relu(x)
            output = add(y1, y2)
        } [output]

        Output graph:

        main(x: ImageType(channel_first=True)) {
            y = mul(x, scale)
            y_bias = add(y, bias)
            y1 = relu(y_bias)
            y2 = relu(y_bias)
            output = add(y1, y2)
        } [output]
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 20, 20))])
        def prog(x):
            y1 = mb.relu(x=x)
            y2 = mb.relu(x=x)
            z = mb.add(x=y1, y=y2)
            return z

        prog.main_input_types = (ct.ImageType(name='x',
                                              shape=[1, 3, 20, 20],
                                              scale=2.0,
                                              bias=[1.0, 2.0, 3.0],
                                              color_layout="BGR",
                                              channel_first=True),)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::insert_image_preprocessing_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "relu", "add"]
        assert get_op_types_in_program(prog) == ["mul", "add", "relu", "relu", "add"]
        scale_op = prog.find_ops(op_type="mul", exactly_one=True)[0]
        assert scale_op.y.val == 2.0
        add_op = prog.find_ops(op_type="add", exactly_one=False)[0]
        assert np.all(add_op.y.val == np.array([1.0, 2.0, 3.0]).reshape([1, 3, 1, 1]))

    @pytest.mark.parametrize(
        "scale_type, bias_type", itertools.product([np.float, np.int32], [np.float, np.int32])
    )
    def test_scale_bias_types(self, scale_type, bias_type):
        """
        Input graph:

        main(x: ImageType(color_layout="RGB", scale=2.0, bias=[1.0, 2.0, 3.0], channel_first=True)) {
            y1 = relu(x)
            y2 = relu(x)
            output = add(y1, y2)
        } [output]

        Output graph:

        main(x: ImageType(channel_first=True)) {
            y = mul(x, scale)
            y_bias = add(y, bias)
            y1 = relu(y_bias)
            y2 = relu(y_bias)
            output = add(y1, y2)
        } [output]
        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 20, 20))])
        def prog(x):
            y1 = mb.relu(x=x)
            y2 = mb.relu(x=x)
            z = mb.add(x=y1, y=y2)
            return z

        prog.main_input_types = (ct.ImageType(name='x',
                                              shape=[1, 3, 20, 20],
                                              scale=scale_type(2.0),
                                              bias=np.array([1, 2, 3]).astype(bias_type),
                                              color_layout="RGB",
                                              channel_first=True),)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::insert_image_preprocessing_ops"
        )
        assert get_op_types_in_program(prev_prog) == ["relu", "relu", "add"]
        assert get_op_types_in_program(prog) == ["mul", "add", "relu", "relu", "add"]
        scale_op = prog.find_ops(op_type="mul", exactly_one=True)[0]
        assert scale_op.y.dtype() == prog.functions["main"].inputs["x"].dtype()
        add_op = prog.find_ops(op_type="add", exactly_one=False)[0]
        assert add_op.y.dtype() == prog.functions["main"].inputs["x"].dtype()

class TestSanitizerPass:

    def test_sanitize_numeric_var_names(self):
        """
        Input:
            main(%x: (1, 3, 20, fp32)(Tensor)) {
              block0() {
                %var_1!: (1, 3, 20, fp32)(Tensor) = relu(x=%x, name="var_1!")
                %1: (1, 3, 20, fp32)(Tensor) = relu(x=%x, name="1")
                %3: (1, 3, 20, fp32)(Tensor) = add(x=%Var_1!, y=%1, name="3")
              } -> (%3)
            }

        Output:
            main(%x: (1, 3, 20, fp32)(Tensor)) {
              block0() {
                %var_1_: (1, 3, 20, fp32)(Tensor) = relu(x=%x, name="var_1_")
                %var_1: (1, 3, 20, fp32)(Tensor) = relu(x=%x, name="op_1")
                %var_3: (1, 3, 20, fp32)(Tensor) = add(x=%var_1_, y=%var_1, name="op_3")
              } -> (%var_3)
            }

        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 20))])
        def prog(x):
            y1 = mb.relu(x=x, name = "var_1!")
            y2 = mb.relu(x=x, name = "1")
            z = mb.add(x=y1, y=y2, name = "3")
            return z

        PASS_REGISTRY["mil_backend::sanitize_name_strings"](prog)
        block = prog.functions["main"]
        assert block.find_ops(op_type="relu")[0].outputs[0].name == "var_1_"
        assert block.find_ops(op_type="relu")[1].outputs[0].name == "var_1"
        assert prog["main"].outputs[0].name == "var_3"
        assert block.find_ops(op_type="relu")[0].name == "var_1_"
        assert block.find_ops(op_type="relu")[1].name == "op_1"
        assert block.find_ops(op_type="add")[0].name == "op_3"


class TestPassFuseActivationSiLU:
    """
    Input graph:
    input --> sigmoid --> mul --> output
    Output graph:
    input --> silu --> output
    """

    @pytest.mark.skipif(ct.utils._macos_version() < (12, 0), reason="mlprogram predict available only on macOS12+")
    @pytest.mark.parametrize(
        "reverse_order", itertools.product([True, False]),
    )
    def test_0(self, reverse_order):
        x_shape = tuple(np.random.randint(low=1, high=4, size=5))

        @mb.program(input_specs=[mb.TensorSpec(shape=x_shape)])
        def program(x):
            sigmoid_x = mb.sigmoid(x=x)
            if not reverse_order:
                x = mb.mul(x=x, y=sigmoid_x)
            else:
                x = mb.mul(x=sigmoid_x, y=x)
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            program, "mil_backend::fuse_activation_silu"
        )

        assert get_op_types_in_program(prev_prog) == ["sigmoid", "mul"]
        assert get_op_types_in_program(program) == ["silu"]

        assert_model_is_valid(
            program=program,
            inputs={"x": x_shape},
            backend=("mlprogram", "fp32"),
            expected_output_shapes={block.outputs[0].name: tuple(x_shape)},
        )


class TestHomogenizeInputDtypes:

    @pytest.mark.parametrize(
        ["op", "x_dtype", "y_dtype"],
        [
            ["add", "int32", "fp32"],
            ["mul", "fp32", "int32"],
            ["minimum", "int64", "int32"],
            ["add", "int32", "fp16"],
            ["add", "fp16", "int32"],
            ["equal", "bool", "int32"],
            ["mod", "int64", "fp16"],
            ["not_equal", "fp32", "bool"],
            ["pow", "fp16", "fp32"],
            ["greater", "fp16", "fp32"],
            ["matmul", "fp16", "int32"],
        ]
    )
    def test_mixed_input_dtypes(self, op, x_dtype, y_dtype):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 10), dtype=string_to_builtin(x_dtype)),
                                 mb.TensorSpec(shape=(10, 10), dtype=string_to_builtin(y_dtype))])
        def prog(x, y):
            x = getattr(mb, op)(x=x, y=y)
            return x

        assert get_op_types_in_program(prog) == [op]

        _, _, block = apply_pass_and_basic_check(prog, "mil_backend::homogenize_input_dtypes")

        assert get_op_types_in_program(prog) == ["cast", op]

        promoted_dtype = promote_types(string_to_builtin(x_dtype), string_to_builtin(y_dtype))

        # Asserting cast configuration
        cast = block.find_ops(op_type="cast")[0]
        assert cast.dtype.val == builtin_to_string(promoted_dtype)
        assert len(cast.outputs) == 1
        assert len(cast.outputs[0].child_ops) == 1
        assert cast.outputs[0].child_ops[0].op_type == op
