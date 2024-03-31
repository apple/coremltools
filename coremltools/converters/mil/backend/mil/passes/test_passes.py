#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil._deployment_compatibility import \
    AvailableTarget as target
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check, assert_model_is_valid, get_op_types_in_program)


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
            prog,
            "mil_backend::adjust_io_to_supported_types",
            skip_output_type_check=True,
            skip_input_type_check=True,
        )  # output dtype is modified

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
            prog,
            "mil_backend::adjust_io_to_supported_types",
            skip_output_type_check=True,
            skip_input_type_check=True,
        )  # output dtype is modified

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
            prog,
            "mil_backend::adjust_io_to_supported_types",
            skip_output_type_check=True,
            skip_input_type_check=True,
        )  # output dtype is modified

        prev_inputs = list(prev_prog.functions['main'].inputs.items())
        inputs = list(prog.functions['main'].inputs.items())
        assert prev_inputs[0][1].name == inputs[0][1].name
        assert inputs[0][1].dtype == types.fp32


    @pytest.mark.parametrize(
        "opset_version",
        [None, target.iOS13, target.iOS16],
    )
    def test_float16_input_output(self, opset_version):
        """
        Input graph:

        main(%x: (1, 1, 1, 1, fp16)(Tensor)) {
            block0() {
                %relu_0: (1, 1, 1, 1, fp16)(Tensor) = relu(x=%x, name="relu_0")
            } -> (%relu_0)
        }

        Output graph (if opset_version < ios16):

        main(%x: (1, 1, 1, 1, fp32)(Tensor)) {
            block0() {
                %cast_0: (1, 1, 1, 1, fp16)(Tensor) = cast(x=%x, dtype="fp16", name="cast_0")
                %relu_0__pre__output__fp32__cast: (1, 1, 1, 1, fp16)(Tensor) = relu(x=%cast_0, name="relu_0")
                %relu_0: (1, 1, 1, 1, fp32)(Tensor) = cast(x=%relu_0__pre__output__fp32__cast, dtype="fp32", name="cast_1")
            } -> (%relu_0)
        }

        Output graph (if opset_version >= ios16): same as the input graph
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 1, 1), dtype=types.fp16)], opset_version=opset_version)
        def prog(x):
            return mb.relu(x=x)

        skip_type_check = opset_version in [None, ct.target.iOS13]
        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog,
            "mil_backend::adjust_io_to_supported_types",
            skip_output_type_check=skip_type_check,
            skip_input_type_check=skip_type_check,
        )

        prev_inputs = list(prev_block.inputs.items())
        inputs = list(block.inputs.items())
        prev_outputs = prev_block.outputs
        outputs = block.outputs
        assert prev_inputs[0][1].name == inputs[0][1].name
        assert outputs[0].name == prev_outputs[0].name
        if opset_version is None or opset_version < target.iOS16:
            assert get_op_types_in_program(prog) == ['cast', 'relu', 'cast']
            assert inputs[0][1].dtype == types.fp32
            assert outputs[0].dtype == types.fp32
        else:
            assert get_op_types_in_program(prog) == ['relu']
            assert inputs[0][1].dtype == types.fp16
            assert block.outputs[0].dtype == types.fp16

    def test_float16_input_output_with_opset_version_inference(self):
        """
        Input graph:

        main(%x: (1, 1, 4, 4, fp16)(Tensor)) {
          block0() {
            %pixel_unshuffle_0: (1, 4, 2, 2, fp16)(Tensor) = pixel_unshuffle(x=%x, downscale_factor=2, name="pixel_unshuffle_0")
          } -> (%pixel_unshuffle_0)
        }

        This function would be inferred as an iOS16 function, and the graph pass should behave properly
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 4, 4), dtype=types.fp16)])
        def prog(x):
            x = mb.pixel_unshuffle(x=x, downscale_factor=np.uint32(2))
            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "mil_backend::adjust_io_to_supported_types"
        )

        prev_inputs = list(prev_block.inputs.items())
        inputs = list(block.inputs.items())
        prev_outputs = prev_block.outputs
        outputs = block.outputs
        assert prev_inputs[0][1].name == inputs[0][1].name
        assert outputs[0].name == prev_outputs[0].name
        assert get_op_types_in_program(prog) == ['pixel_unshuffle']
        assert inputs[0][1].dtype == types.fp16
        assert block.outputs[0].dtype == types.fp16

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
            prog,
            "mil_backend::adjust_io_to_supported_types",
            skip_output_type_check=True,
            skip_input_type_check=True,
        )  # output dtype is modified

        prev_inputs = list(prev_prog.functions['main'].inputs.items())
        inputs = list(prog.functions['main'].inputs.items())
        assert prev_inputs[0][1].name == inputs[0][1].name
        assert inputs[0][1].dtype == types.int32

    @pytest.mark.parametrize(
        "opset_version",
        [None, target.iOS17],
    )
    def test_int16_input(self, opset_version):
        """
        Input graph:
            func main(int16 x) {
            ....
            } -> (x)

        Before IOS17, it becomes
            func main(int32 x) {
            ....
            } -> (x)

        In IOS17+, it becomes
            func main(int32 x) {
                %cast_0: (1, 1, 1, 1, int16)(Tensor) = cast(x=%x, dtype="int16", name="cast_0")
                ....
                %cast_1: (1, 1, 1, 1, int32)(Tensor) = cast(x=%x, dtype="int32", name="cast_1")
            } -> (cast_1)
        because IOS17+ supports int16 in Runtime (but doesn't support int16 for I/O).
        """

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, 1, 1, 1), dtype=types.int16)],
            opset_version=opset_version,
        )
        def prog(x):
            return x

        skip_type_check = opset_version is None
        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog,
            "mil_backend::adjust_io_to_supported_types",
            skip_output_type_check=True,
            skip_input_type_check=True,
        )  # output dtype id modified

        prev_inputs = list(prev_block.inputs.items())
        inputs = list(block.inputs.items())
        prev_outputs = prev_block.outputs
        outputs = block.outputs
        assert prev_inputs[0][1].dtype == types.int16
        assert prev_outputs[0].dtype == types.int16
        assert inputs[0][1].dtype == types.int32
        assert outputs[0].dtype == types.int32
        assert prev_inputs[0][1].name == inputs[0][1].name
        assert outputs[0].name == prev_outputs[0].name
        if opset_version and opset_version >= target.iOS17:
            assert get_op_types_in_program(prog) == ["cast", "cast"]
            cast_ops = [op for op in prog["main"].operations if op.op_type != "const"]
            # The first cast is for int32 to int16.
            assert cast_ops[0].x.dtype == types.int32
            assert cast_ops[0].outputs[0].dtype == types.int16
            # The second cast is for int16 to int32.
            assert cast_ops[1].x.dtype == types.int16
            assert cast_ops[1].outputs[0].dtype == types.int32
        else:
            # Before IOS17, the int16 is not supported in Runtime, so there is no cast inserted.
            assert get_op_types_in_program(prog) == []

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
        pytest.xfail("fp64 dtype not supported in MIL")
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

        subblocks = prog.functions["main"].operations[0].blocks
        prev_subblocks = prev_prog.functions["main"].operations[0].blocks
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
        pytest.xfail("cast operation does not support casting to fp64")
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
        pytest.xfail("cast not supports dtype=`int64`")
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

        prog.functions["main"].input_types = (
            ct.ImageType(name="x", shape=[1, 1, 20, 20], color_layout="G", channel_first=True),
        )

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

        prog.functions["main"].input_types = (
            ct.ImageType(
                name="x", shape=[1, 1, 20, 20], scale=2.0, color_layout="G", channel_first=True
            ),
        )

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

        prog.functions["main"].input_types = (
            ct.ImageType(
                name="x", shape=[1, 1, 20, 20], bias=2.0, color_layout="G", channel_first=True
            ),
        )

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

        prog.functions["main"].input_types = (
            ct.ImageType(
                name="x",
                shape=[1, 1, 20, 20],
                scale=2.0,
                bias=2.0,
                color_layout="G",
                channel_first=True,
            ),
        )

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

        prog.functions["main"].input_types = (
            ct.ImageType(name="x", shape=[1, 3, 20, 20], color_layout="RGB", channel_first=True),
        )

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

        prog.functions["main"].input_types = (
            ct.ImageType(
                name="x",
                shape=[1, 3, 20, 20],
                scale=2.0,
                bias=[1.0, 2.0, 3.0],
                color_layout="RGB",
                channel_first=True,
            ),
        )

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

        prog.functions["main"].input_types = (
            ct.ImageType(name="x", shape=[1, 3, 20, 20], color_layout="BGR", channel_first=True),
        )

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

        prog.functions["main"].input_types = (
            ct.ImageType(
                name="x",
                shape=[1, 3, 20, 20],
                scale=2.0,
                bias=[1.0, 2.0, 3.0],
                color_layout="BGR",
                channel_first=True,
            ),
        )

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
        "scale_type, bias_type", itertools.product([np.float32, np.int32], [np.float32, np.int32])
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

        prog.functions["main"].input_types = (
            ct.ImageType(
                name="x",
                shape=[1, 3, 20, 20],
                scale=scale_type(2.0),
                bias=np.array([1, 2, 3]).astype(bias_type),
                color_layout="RGB",
                channel_first=True,
            ),
        )

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

    def test_sanitize_var_names_with_two_functions(self):
        """
        Input:
            main(%x: (1, 3, 20, fp32)(Tensor)) {
              block0() {
                %var_1!: (1, 3, 20, fp32)(Tensor) = relu(x=%x, name="var_1!")
              } -> (%var_1!)
            }

            main_2(%x: (1, 3, 20, fp32)(Tensor)) {
              block0() {
                %var_1!: (1, 3, 20, fp32)(Tensor) = relu(x=%x, name="var_1!")
              } -> (%var_1!)
            }


        Output:
            main(%x: (1, 3, 20, fp32)(Tensor)) {
              block0() {
                %var_1!: (1, 3, 20, fp32)(Tensor) = relu(x=%x, name="var_1_")
              } -> (%var_1_)
            }

            main_2(%x: (1, 3, 20, fp32)(Tensor)) {
              block0() {
                %var_1!: (1, 3, 20, fp32)(Tensor) = relu(x=%x, name="var_1_")
              } -> (%var_1_)
            }

        """

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 20))])
        def prog(x):
            z = mb.relu(x=x, name = "var_1!")
            return z

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 20))])
        def prog_2(x):
            z = mb.relu(x=x, name = "var_1!")
            return z

        prog.add_function("main_2", prog_2.functions["main"])
        PASS_REGISTRY["mil_backend::sanitize_name_strings"](prog)
        block = prog.functions["main"]
        assert block.find_ops(op_type="relu")[0].outputs[0].name == "var_1_"
        assert prog["main"].outputs[0].name == "var_1_"
        assert block.find_ops(op_type="relu")[0].name == "var_1_"
        block = prog.functions["main_2"]
        assert block.find_ops(op_type="relu")[0].outputs[0].name == "var_1_"
        assert prog["main"].outputs[0].name == "var_1_"
        assert block.find_ops(op_type="relu")[0].name == "var_1_"


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


class TestPassFusePow2Sqrt:
    """
    Input graph:
    input --> pow(2) --> sqrt --> output
    Output graph:
    input --> output
    """

    @pytest.mark.skipif(ct.utils._macos_version() < (12, 0), reason="mlprogram predict available only on macOS12+")
    @pytest.mark.parametrize(
        "reverse_order", itertools.product([True, False]),
    )
    def test_fuse(self, reverse_order):
        x_shape = tuple(np.random.randint(low=1, high=4, size=5))

        @mb.program(input_specs=[mb.TensorSpec(shape=x_shape)])
        def program(x):
            if not reverse_order:
                x = mb.sqrt(x=mb.pow(x=x, y=2.0))
            else:
                x = mb.pow(x=mb.sqrt(x=x), y=2.0)
            return x

        prev_prog, _, block = apply_pass_and_basic_check(
            program, "mil_backend::fuse_pow2_sqrt"
        )

        assert set(get_op_types_in_program(prev_prog)) == set(("pow", "sqrt"))
        assert get_op_types_in_program(program) == ["identity"]

        assert_model_is_valid(
            program=program,
            inputs={"x": x_shape},
            backend=("mlprogram", "fp32"),
            expected_output_shapes={block.outputs[0].name: tuple(x_shape)},
        )

    @pytest.mark.skipif(ct.utils._macos_version() < (12, 0), reason="mlprogram predict available only on macOS12+")
    @pytest.mark.parametrize(
        "reverse_order", itertools.product([True, False]),
    )
    def test_illegal_pow(self, reverse_order):
        x_shape = tuple(np.random.randint(low=1, high=4, size=5))

        @mb.program(input_specs=[mb.TensorSpec(shape=x_shape)])
        def program(x):
            if not reverse_order:
                x = mb.sqrt(x=mb.pow(x=x, y=3.0))
            else:
                x = mb.pow(x=mb.sqrt(x=x), y=3.0)
            return x

        prev_prog, _, block = apply_pass_and_basic_check(
            program, "mil_backend::fuse_pow2_sqrt"
        )

        assert set(get_op_types_in_program(prev_prog)) == set(("pow", "sqrt"))
        assert set(get_op_types_in_program(program)) == set(("pow", "sqrt"))

        assert_model_is_valid(
            program=program,
            inputs={"x": x_shape},
            backend=("mlprogram", "fp32"),
            expected_output_shapes={block.outputs[0].name: tuple(x_shape)},
        )

    @pytest.mark.skipif(ct.utils._macos_version() < (12, 0), reason="mlprogram predict available only on macOS12+")
    def test_no_pow(self):
        x_shape = tuple(np.random.randint(low=1, high=4, size=5))

        @mb.program(input_specs=[mb.TensorSpec(shape=x_shape)])
        def program(x):
            return mb.sqrt(x=x)

        prev_prog, _, block = apply_pass_and_basic_check(
            program, "mil_backend::fuse_pow2_sqrt"
        )

        assert get_op_types_in_program(prev_prog) == ["sqrt"]
        assert get_op_types_in_program(program) == ["sqrt"]

        assert_model_is_valid(
            program=program,
            inputs={"x": x_shape},
            backend=("mlprogram", "fp32"),
            expected_output_shapes={block.outputs[0].name: tuple(x_shape)},
        )

    @pytest.mark.skipif(ct.utils._macos_version() < (12, 0), reason="mlprogram predict available only on macOS12+")
    def test_no_sqrt(self):
        x_shape = tuple(np.random.randint(low=1, high=4, size=5))

        @mb.program(input_specs=[mb.TensorSpec(shape=x_shape)])
        def program(x):
            return mb.pow(x=x, y=2.0)

        prev_prog, _, block = apply_pass_and_basic_check(
            program, "mil_backend::fuse_pow2_sqrt"
        )

        assert get_op_types_in_program(prev_prog) == ["pow"]
        assert get_op_types_in_program(program) == ["pow"]

        assert_model_is_valid(
            program=program,
            inputs={"x": x_shape},
            backend=("mlprogram", "fp32"),
            expected_output_shapes={block.outputs[0].name: tuple(x_shape)},
        )

    @pytest.mark.skipif(ct.utils._macos_version() < (12, 0), reason="mlprogram predict available only on macOS12+")
    @pytest.mark.parametrize(
        "reverse_order", itertools.product([True, False]),
    )
    def test_multiple_nodes(self, reverse_order):
        x_shape = tuple(np.random.randint(low=1, high=4, size=5))

        @mb.program(input_specs=[mb.TensorSpec(shape=x_shape)])
        def program(x):
            if not reverse_order:
                x = mb.mul(x=x, y=x)
                x = mb.pow(x=x, y=2.0)
                x = mb.sqrt(x=x)
                x = mb.reduce_argmax(x=x)
                x = mb.reshape(x=x, shape=[*x_shape[:-1]])
            else:
                x = mb.mul(x=x, y=x)
                x = mb.sqrt(x=x)
                x = mb.pow(x=x, y=2.0)
                x = mb.reduce_argmax(x=x)
                x = mb.reshape(x=x, shape=[*x_shape[:-1]])
            return x

        prev_prog, _, block = apply_pass_and_basic_check(
            program, "mil_backend::fuse_pow2_sqrt"
        )

        assert set(get_op_types_in_program(prev_prog)) == set(("mul", "pow", "sqrt", "reduce_argmax", "reshape"))
        assert get_op_types_in_program(program) == ["mul", "identity", "reduce_argmax", "reshape"]

        assert_model_is_valid(
            program=program,
            inputs={"x": x_shape},
            backend=("mlprogram", "fp32"),
            expected_output_shapes={block.outputs[0].name: tuple(x_shape[:-1])},
        )
