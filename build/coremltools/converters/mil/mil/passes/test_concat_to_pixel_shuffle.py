#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools._deps import _IS_MACOS
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    assert_model_is_valid,
    get_op_types_in_program,
    apply_pass_and_basic_check,
)
from coremltools.converters.mil.testing_reqs import ct

import unittest

import numpy as np

np.random.seed(1984)


class ConcatToPixelShuffleTests(unittest.TestCase):
    
    def test_success(self):
        """
        Input graph:
        input1(1, 2, 3, 4) -----> concat(axis=2, interleave=True) -----> concat(axis=3, interleave=True) ---> out(1, 2, 6, 8)
                                             ^                                           ^
                                             |                                           |
        input2(1, 2, 3, 4) -------------------                                           |
                                                                                         |
        input3(1, 2, 3, 4) -----> concat(axis=2, interleave=True) -----------------------|
                                             ^
                                             |
        input4(1, 2, 3, 4) ------------------|

        Output graph:
        input1(1, 2, 3, 4) -----> concat(axis=1) ---> pixel_shuffle(upsample_factor=2) ----> out(1, 2, 6, 8)
                                     ^
        input2(1, 2, 3, 4) ----------|
                                     | 
        input3(1, 2, 3, 4) ----------|
                                     | 
        input4(1, 2, 3, 4) ----------|
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4)), 
                                 mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["concat", "concat", "concat"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat", "pixel_shuffle"])

        inputs = {"x1": (1, 2, 3, 4), "x2": (1, 2, 3, 4), "x3": (1, 2, 3, 4), "x4": (1, 2, 3, 4)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 2, 6, 8)},
        )

        mlmodel = ct.convert(prog, source="milinternal", convert_to="neuralnetwork")

        if not _IS_MACOS:
            # Can not get predictions unless on macOS.
            return

        input_dict = dict()
        input_dict["x1"] = np.ones(inputs["x1"])
        input_dict["x2"] = np.ones(inputs["x2"]) * 2
        input_dict["x3"] = np.ones(inputs["x3"]) * 3
        input_dict["x4"] = np.ones(inputs["x4"]) * 4

        output_name = block.outputs[0].name

        ab = np.reshape(np.stack((input_dict["x1"], input_dict["x2"]), axis=3), newshape=[1, 2, 6, 4])
        cd = np.reshape(np.stack((input_dict["x3"], input_dict["x4"]), axis=3), newshape=[1, 2, 6, 4])
        old_prediction = np.reshape(np.stack((ab, cd), axis=4), newshape=[1, 2, 6, 8])        

        prediction = mlmodel.predict(input_dict, useCPUOnly=True)
        np.testing.assert_allclose(old_prediction, prediction[output_name], atol=1e-04, rtol=1e-05)

    def test_nested(self):
        """
        Two nested blocks that will each be transformed.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4)), 
                                 mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4)), 
                                 mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4)), 
                                 mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x1, x2, x3, x4, x5, x6, x7, x8):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            ef = mb.concat(values=[x5, x6], axis=2, interleave=True)
            gh = mb.concat(values=[x7, x8], axis=2, interleave=True)
            y = mb.concat(values=[ef, gh], axis=3, interleave=True)

            z = mb.concat(values=[x, y], axis=1)

            return z

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["concat", "concat", "concat", "concat", "concat", "concat", "concat"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat", "pixel_shuffle", "concat", "pixel_shuffle", "concat"])

        inputs = {"x1": (1, 2, 3, 4), "x2": (1, 2, 3, 4), "x3": (1, 2, 3, 4), "x4": (1, 2, 3, 4),
                  "x5": (1, 2, 3, 4), "x6": (1, 2, 3, 4), "x7": (1, 2, 3, 4), "x8": (1, 2, 3, 4)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={block.outputs[0].name: (1, 4, 6, 8)},
        )

        input_dict = dict()
        for name, shape in inputs.items():
            input_dict[name] = np.random.rand(*shape)

        output_name = block.outputs[0].name

        ab = np.reshape(np.stack((input_dict["x1"], input_dict["x2"]), axis=3), newshape=[1, 2, 6, 4])
        cd = np.reshape(np.stack((input_dict["x3"], input_dict["x4"]), axis=3), newshape=[1, 2, 6, 4])
        x = np.reshape(np.stack((ab, cd), axis=4), newshape=[1, 2, 6, 8])        

        ef = np.reshape(np.stack((input_dict["x5"], input_dict["x6"]), axis=3), newshape=[1, 2, 6, 4])
        gh = np.reshape(np.stack((input_dict["x7"], input_dict["x8"]), axis=3), newshape=[1, 2, 6, 4])
        y = np.reshape(np.stack((ef, gh), axis=4), newshape=[1, 2, 6, 8])        

        old_prediction = np.concatenate((x, y), axis=1)

        mlmodel = ct.convert(prog, source="milinternal", convert_to="neuralnetwork")

        if _IS_MACOS:
            prediction = mlmodel.predict(input_dict, useCPUOnly=True)
            np.testing.assert_allclose(old_prediction, prediction[output_name], atol=1e-04, rtol=1e-05)

    def test_failure_0(self):
        """
        The h_concat has three inputs, so the pattern won't match.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4)), 
                                 mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2, x3], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4, x1], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["concat", "concat", "concat"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_1(self):
        """
        The first concat is on the wrong axis, so the pattern won't match.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4)), 
                                 mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=3, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=3, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["concat", "concat", "concat"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_2(self):
        """
        The last concat is on the wrong axis, so the pattern won't match.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4)), 
                                 mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=2, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["concat", "concat", "concat"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_3(self):
        """
        The first concat is not interleaved, so the pattern won't match.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4)), 
                                 mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=False)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["concat", "concat", "concat"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_4(self):
        """
        The second concat is not interleaved, so the pattern won't match.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4)), 
                                 mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=False)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["concat", "concat", "concat"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_5(self):
        """
        The last concat is not interleaved, so the pattern won't match.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4)), 
                                 mb.TensorSpec(shape=(1, 2, 3, 4)), mb.TensorSpec(shape=(1, 2, 3, 4))])
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=False)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["concat", "concat", "concat"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_5(self):
        """
        The inputs are the wrong rank, so the pattern won't match.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 3, 4, 5)), mb.TensorSpec(shape=(1, 2, 3, 4, 5)), 
                                 mb.TensorSpec(shape=(1, 2, 3, 4, 5)), mb.TensorSpec(shape=(1, 2, 3, 4, 5))])
        def prog(x1, x2, x3, x4):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["concat", "concat", "concat"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])

    def test_failure_6(self):
        """ 
        Extra input to the w_concats means the pattern won't match.
        """
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2, 4, 4)), mb.TensorSpec(shape=(1, 2, 4, 4)), 
                                 mb.TensorSpec(shape=(1, 2, 4, 4)), mb.TensorSpec(shape=(1, 2, 4, 4)), 
                                 mb.TensorSpec(shape=(1, 2, 8, 4))])
        def prog(x1, x2, x3, x4, x5):
            ab = mb.concat(values=[x1, x2], axis=2, interleave=True)
            cd = mb.concat(values=[x3, x4], axis=2, interleave=True)
            x = mb.concat(values=[ab, cd, x5], axis=3, interleave=True)

            return x

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::concat_to_pixel_shuffle"
        )
        self.assertEqual(
            get_op_types_in_program(prev_prog), ["concat", "concat", "concat"]
        )
        self.assertEqual(get_op_types_in_program(prog), ["concat", "concat", "concat"])
