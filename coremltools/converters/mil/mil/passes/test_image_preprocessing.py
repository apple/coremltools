#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools import ImageType, models
from coremltools.converters.mil.testing_reqs import ct
from coremltools.converters.mil.testing_utils import (
    assert_op_count_match,
    assert_model_is_valid,
    get_op_types_in_program,
    apply_pass_and_basic_check,
)
from coremltools.converters.mil.mil import Builder as mb
import unittest

class ImagePreprocessingPass(unittest.TestCase):
    """
    Input graph:
    input (format=NHWC) ------> transpose(axis=[0, 3, 1, 2]) ---------> add ----> relu ---> out
                           |                                             ^
                           |                                             |
                           ---> relu ---> transpose(axis=[0, 3, 1, 2]) ---

    Intermediate graph:
    input (format=NCHW) -----> transpose(axis=[0, 2, 3, 1]) ----> transpose(axis=[0, 3, 1, 2]) ---------> add ----> relu ---> out
                                                              |                                             ^
                                                              |                                             |
                                                              ---> relu ---> transpose(axis=[0, 3, 1, 2]) ---


    Output graph:
    input (format=NCHW) -----> relu -----> add -----> relu -----> out
                          |                 ^
                          |                 |
                          -------------------
    """

    def test_fusion_with_image_intermediate_graph(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20, 30, 3))])
        def prog(x):
            x1 = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x2 = mb.relu(x=x)
            x3 = mb.transpose(x=x2, perm=[0, 3, 1, 2])
            x4 = mb.add(x=x1, y=x3)
            return mb.relu(x=x4)

        prog.main_input_types = [ImageType(name="x", shape=(10, 20, 30, 3), channel_first=False)]
        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::image_input_preprocess"
        )
        self.assertEqual(get_op_types_in_program(prev_prog), ["transpose", "relu", "transpose", "add", "relu"])
        self.assertEqual(get_op_types_in_program(prog), ["transpose", "transpose", "relu", "transpose", "add", "relu"])

    def test_fusion_with_image_full(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(10, 20, 30, 3))])
        def prog(x):
            x1 = mb.transpose(x=x, perm=[0, 3, 1, 2])
            x2 = mb.relu(x=x)
            x3 = mb.transpose(x=x2, perm=[0, 3, 1, 2])
            x4 = mb.add(x=x1, y=x3)
            return mb.relu(x=x4)

        mlmodel = ct.convert(prog,
            inputs=[ImageType(name="x", shape=(10, 20, 30, 3),
              channel_first=False)],
            source="mil", convert_to="nn_proto")
        assert mlmodel is not None
        assert len(mlmodel.get_spec().neuralNetwork.layers) == 3
