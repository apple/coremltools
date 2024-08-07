#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline, PassPipelineManager
from coremltools.converters.mil.testing_utils import assert_model_is_valid, get_op_types_in_program

np.random.seed(1984)


class TestPassPipeline:
    def test_add_pass(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3))])
        def prog(x):
            x = mb.relu(x=x)
            x = mb.relu(x=x)
            x = mb.add(x=x, y=1.0)
            return x

        assert get_op_types_in_program(prog) == ["relu", "relu", "add"]
        pipeline = PassPipeline.EMPTY
        pipeline.append_pass("common::merge_consecutive_relus")
        assert pipeline.passes == ["common::merge_consecutive_relus"]
        PassPipelineManager.apply_pipeline(prog, pipeline)
        assert get_op_types_in_program(prog) == ["relu", "add"]

        inputs = {"x": (2, 3)}
        assert_model_is_valid(
            prog,
            inputs,
            expected_output_shapes={prog.functions["main"].outputs[0].name: (2, 3)},
        )

    def test_insert_pass_at_index(self):
        pipeline = PassPipeline.EMPTY
        pipeline.insert_pass(index=0, pass_name="common::merge_consecutive_relus")
        pipeline.insert_pass(index=0, pass_name="common::noop_elimination")
        pipeline.insert_pass(index=1, pass_name="common::noop_elimination")
        pipeline.insert_pass(index=1, pass_name="common::merge_consecutive_reshapes")
        assert pipeline.passes == [
            "common::noop_elimination",
            "common::merge_consecutive_reshapes",
            "common::noop_elimination",
            "common::merge_consecutive_relus",
        ]

    def test_insert_invalid_pass(self):
        pipeline = PassPipeline.EMPTY
        with pytest.raises(ValueError, match="The pass test_pass is not registered."):
            pipeline.append_pass("test_pass")
        with pytest.raises(ValueError, match="The pass test_pass is not registered."):
            pipeline.insert_pass(0, "test_pass")
        with pytest.raises(ValueError, match="The pass invalid_pass is not registered."):
            pipeline.passes = ["invalid_pass"]

    def test_remove_passes(self):
        pipeline = PassPipeline.EMPTY
        pipeline.passes = [
            "common::noop_elimination",
            "common::merge_consecutive_reshapes",
            "common::noop_elimination",
            "common::merge_consecutive_relus",
        ]
        pipeline.remove_passes(passes_names=["common::noop_elimination"])
        assert pipeline.passes == [
            "common::merge_consecutive_reshapes",
            "common::merge_consecutive_relus",
        ]
        pipeline.remove_pass(index=1)
        assert pipeline.passes == ["common::merge_consecutive_reshapes"]

    def test_set_pass_options(self):
        pipeline = PassPipeline.EMPTY
        pipeline.append_pass("common::add_fp16_cast")
        assert pipeline.get_options("common::add_fp16_cast") is None
        pipeline.set_options("common::add_fp16_cast", {"skip_ops_by_type": "matmul,const"})
        assert len(pipeline.get_options("common::add_fp16_cast")) == 1
        assert pipeline.get_options("common::add_fp16_cast")[0].option_name == "skip_ops_by_type"
        assert pipeline.get_options("common::add_fp16_cast")[0].option_val == "matmul,const"

    def test_set_pass_options_already_exist(self):
        pipeline = PassPipeline()
        pipeline.set_options("common::add_fp16_cast", {"skip_ops_by_type": "matmul,const"})
        with pytest.raises(
            ValueError, match="The pass common::add_fp16_cast already has associated options."
        ):
            pipeline.set_options("common::add_fp16_cast", {"skip_ops_by_type": "concat"}, override=False)
        # Override the options.
        pipeline.set_options("common::add_fp16_cast", {"skip_ops_by_type": "concat"})
        assert pipeline.get_options("common::add_fp16_cast")[0].option_name == "skip_ops_by_type"
        assert pipeline.get_options("common::add_fp16_cast")[0].option_val == "concat"

    def test_set_pass_options_for_pass_not_in_pipeline(self):
        pipeline = PassPipeline.EMPTY
        pipeline.set_options("common::add_fp16_cast", {"skip_ops_by_type": "matmul,const"})
        with pytest.raises(
            ValueError,
            match="This pass pipeline is not valid. The pass common::add_fp16_cast "
            "has associated options but it's not in the passes.",
        ):
            pipeline.validate()

    def test_get_invalid_pipeline(self):
        with pytest.raises(
            ValueError,
            match="There is no pipeline for `invalid`.",
        ):
            PassPipeline.get_pipeline("invalid")

    def test_list_available_pipelines(self):
        available_pipelines = PassPipeline.list_available_pipelines()
        assert len(available_pipelines) == 12
        assert "default" in available_pipelines
        assert "default_palettization" in available_pipelines

    def test_merge_reshape_in_nested_block(self):
        INPUT_SHAPE = (6, 7)
        OUTPUT_SHAPE = (7, 6)

        @mb.program(input_specs=[mb.TensorSpec(shape=INPUT_SHAPE)])
        def prog(x):
            loop_var = np.int32(2)
            def while_cond(loop_Var, _x):
                return mb.equal(x=loop_Var, y=np.int32(0))

            def while_body(loop_var, x):
                # Do reshapes of the input
                y1 = mb.reshape(x=x, shape=(3, 2, 7))
                y2 = mb.reshape(x=y1, shape=(7, 2, 3))
                y3 = mb.reshape(x=y2, shape=(14, 3))
                y4 = mb.reshape(x=y3, shape=OUTPUT_SHAPE)
                return mb.add(x=loop_var, y=np.int(-1)), y4

            while_results = mb.while_loop(_cond=while_cond, _body=while_body, loop_vars=(loop_var, x))
            return while_results[1]

        pipeline = PassPipeline.EMPTY
        pipeline.append_pass("common::merge_consecutive_reshapes")
        PassPipelineManager.apply_pipeline(prog, pipeline)

    @staticmethod
    def test_get_pipeline_should_use_copy():
        pipeline = PassPipeline.DEFAULT_PRUNING
        pipeline.append_pass("compression::palettize_weights")
        pipeline_2 = PassPipeline.DEFAULT_PRUNING
        assert "compression::palettize_weights" not in pipeline_2.passes
