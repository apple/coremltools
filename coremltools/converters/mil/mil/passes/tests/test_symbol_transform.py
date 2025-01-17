#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import itertools

import numpy as np
import pytest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    get_op_types_in_program,
)


class TestMaterializeSymbolicShapeProgram:
    @staticmethod
    def test_weight_id_pass_down():
        """
        When materializing dynamic shape of a program, the weight_id need to be passed down to the created function
        """
        symbolic_shape = (get_new_symbol(), get_new_symbol())
        fixed_shape = (2, 3)

        @mb.program(input_specs=[mb.TensorSpec(shape=symbolic_shape)])
        def prog(x):
            y = mb.const(val=np.random.rand(*fixed_shape), name="y")
            return mb.add(x=x, y=y)

        graph_pass = PASS_REGISTRY["common::materialize_symbolic_shape_program"]
        graph_pass.function_name_to_materialization_map = {
            "main": {"x": fixed_shape},
            "main_2": {"x": fixed_shape},
        }
        apply_pass_and_basic_check(
            prog, graph_pass, skip_output_shape_check=True, skip_function_name_check=True
        )

        # check that the weight_id is passed down
        for block in prog.functions.values():
            const_ops = block.find_ops(op_type="const")
            assert len(const_ops) == 1
            assert const_ops[0].weight_id == "const_y_weight_id"

    @pytest.mark.parametrize("override_main_function", (True, False))
    def test_simple(self, override_main_function):
        """
        Input graph:

            x -> shape -> add

        Output graph:

            const
        """
        symbolic_shape = (get_new_symbol(), get_new_symbol())
        fixed_shape = (2, 3)
        new_function_name = "main" if override_main_function else "materialization"

        @mb.program(input_specs=[mb.TensorSpec(shape=symbolic_shape)])
        def prog(x):
            shape = mb.shape(x=x)
            return mb.add(x=shape, y=1)

        graph_pass = PASS_REGISTRY["common::materialize_symbolic_shape_program"]
        graph_pass.function_name_to_materialization_map = {new_function_name: {"x": fixed_shape}}
        prev_prog, _, _ = apply_pass_and_basic_check(
            prog, graph_pass, skip_output_shape_check=True, skip_function_name_check=True
        )
        apply_pass_and_basic_check(prog, "common::const_elimination")

        if override_main_function:
            assert set(prog.functions.keys()) == {"main"}
        else:
            assert set(prog.functions.keys()) == {"main", "materialization"}
            assert prog.functions["main"].inputs["x"].shape == symbolic_shape
            assert get_op_types_in_program(prev_prog, "main") == get_op_types_in_program(
                prog, "main"
            )
        assert prog.functions[new_function_name].inputs["x"].shape == fixed_shape
        assert len(get_op_types_in_program(prog, new_function_name)) == 0

    @pytest.mark.parametrize(
        "source_function_name, override_source_function",
        itertools.product(
            ("main", "func2"),
            (True, False),
        ),
    )
    def test_multifunction_source_program(self, source_function_name, override_source_function):
        """
        Input graph:

            x -> shape -> sub

        Output graph:

            const
        """
        symbolic_shape = (get_new_symbol(),)
        fixed_shape = (5,)
        new_function_name = source_function_name if override_source_function else "materialization"

        @mb.program(input_specs=[mb.TensorSpec(shape=symbolic_shape)])
        def prog(x):
            shape = mb.shape(x=x)
            return mb.sub(x=shape, y=1)

        @mb.function(input_specs=[mb.TensorSpec(shape=symbolic_shape)])
        def func2(x):
            shape = mb.shape(x=x)
            return mb.sub(x=shape, y=2)

        prog.add_function("func2", func2)

        graph_pass = PASS_REGISTRY["common::materialize_symbolic_shape_program"]
        graph_pass.function_name_to_materialization_map = {new_function_name: {"x": fixed_shape}}
        if source_function_name != "main":
            graph_pass.souce_function_name = source_function_name
        prev_prog, _, _ = apply_pass_and_basic_check(
            prog,
            graph_pass,
            skip_output_name_check=True,
            skip_output_shape_check=True,
            skip_function_name_check=True,
        )
        apply_pass_and_basic_check(prog, "common::const_elimination")

        if override_source_function:
            assert set(prog.functions.keys()) == {"main", "func2"}
        else:
            assert set(prog.functions.keys()) == {"main", "func2", "materialization"}
        assert prog.functions[new_function_name].inputs["x"].shape == fixed_shape
        assert len(get_op_types_in_program(prog, new_function_name)) == 0
        for function_name in ("main", "func2"):
            if function_name == source_function_name and override_source_function:
                continue
            else:
                assert prog.functions[function_name].inputs["x"].shape == symbolic_shape
                assert get_op_types_in_program(prev_prog, function_name) == get_op_types_in_program(
                    prog, function_name
                )

    @pytest.mark.parametrize(
        "inconsistency_in_single_input",
        (True, False),
    )
    def test_inconsistent_materialization(self, inconsistency_in_single_input):
        symbol = get_new_symbol()
        graph_pass = PASS_REGISTRY["common::materialize_symbolic_shape_program"]
        if inconsistency_in_single_input:

            @mb.program(input_specs=[mb.TensorSpec(shape=(symbol, symbol))])
            def prog(x):
                return mb.add(x=x, y=1.0)

            graph_pass.function_name_to_materialization_map = {"materialization": {"x": (2, 3)}}
        else:

            @mb.program(
                input_specs=[mb.TensorSpec(shape=(2, symbol)), mb.TensorSpec(shape=(symbol, 4))]
            )
            def prog(x, y):
                return mb.matmul(x=x, y=y)

            graph_pass.function_name_to_materialization_map = {
                "materialization": {"x": (2, 3), "y": (5, 4)}
            }

        with pytest.raises(
            ValueError,
            match=(
                r"Inconsistent symbol materialization in new function .*: "
                r"symbol [a-zA-Z]+[0-9]+ is to be materialized into [0-9]+ and [0-9]+\. "
                r"Please make sure input (.*) has compatible shape with others"
            ),
        ):
            apply_pass_and_basic_check(
                prog, graph_pass, skip_output_shape_check=True, skip_function_name_check=True
            )
