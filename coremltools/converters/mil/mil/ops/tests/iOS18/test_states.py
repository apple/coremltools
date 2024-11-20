#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.builder import Builder as mb
from coremltools.converters.mil.mil.ops.defs.iOS18 import _IOS18_TARGET
from coremltools.converters.mil.mil.ops.tests.iOS18 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import random_gen


class TestCoreMLUpdateState:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_coreml_update_state_smoke(self, compute_unit, backend):
        def build(state, value):
            return mb.coreml_update_state(
                state=state,
                value=value,
            )

        input_placeholders = {
            "state": mb.state_tensor_placeholder(
                shape=(2,),
                dtype=types.fp16,
            ),
            "value": mb.placeholder(shape=(2,), dtype=types.fp16),
        }
        value = random_gen((2,))
        input_values = {"value": value}

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types=[(2, types.fp16)],
            expected_outputs=[value],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(
            compute_units,
            backends,
            [(1,), (2, 3), (4, 5, 6)],
        ),
    )
    def test_coreml_update_stress(self, compute_unit, backend, shape):
        def build(x_in, y_in, z_in):
            def increase_val_by_one(state, input):
                v = mb.add(x=input, y=np.float16(1))
                return mb.coreml_update_state(state=state, value=v)

            x = mb.read_state(input=x_in)
            y = mb.read_state(input=y_in)
            z = mb.read_state(input=z_in)

            for i in range(10):
                x = increase_val_by_one(x_in, x)
                y = increase_val_by_one(y_in, y)
                z = increase_val_by_one(z_in, z)

            return mb.read_state(input=x_in), mb.read_state(input=y_in), mb.read_state(input=z_in)

        input_placeholders = {
            "x_in": mb.state_tensor_placeholder(
                shape=shape,
                dtype=types.fp16,
            ),
            "y_in": mb.state_tensor_placeholder(
                shape=shape,
                dtype=types.fp16,
            ),
            "z_in": mb.state_tensor_placeholder(
                shape=shape,
                dtype=types.fp16,
            ),
        }
        input_values = {}

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types=[
                (
                    *shape,
                    types.fp16,
                )
            ]
            * 3,
            expected_outputs=[
                [
                    10
                    * np.ones(
                        shape,
                    )
                ]
                * 3,
                [
                    20
                    * np.ones(
                        shape,
                    )
                ]
                * 3,
            ],
            compute_unit=compute_unit,
            backend=backend,
            pred_iters=2,
        )


class TestReadState:
    @staticmethod
    def test_read_tensor_state_builder():
        @mb.program(input_specs=[mb.StateTensorSpec((2, 3))], opset_version=_IOS18_TARGET)
        def prog(x):
            return mb.read_state(input=x)

        read_state_op = prog.find_ops("read_state")[0]
        assert types.is_state(read_state_op.input._sym_type)
        assert types.is_tensor(read_state_op.outputs[0]._sym_type)

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_read_state_smoke(self, compute_unit, backend):
        def build(state):
            return mb.read_state(
                input=state,
            )

        input_placeholders = {
            "state": mb.state_tensor_placeholder(
                shape=(2,),
                dtype=types.fp16,
            ),
        }
        input_values = {}

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types=[(2, types.fp16)],
            expected_outputs=[
                np.zeros(
                    2,
                )
            ],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, shape",
        itertools.product(compute_units, backends, [(1,), (2, 3), (4, 5, 6)]),
    )
    def test_read_state_stress(self, compute_unit, backend, shape):
        def build(x, y, z):
            return (
                mb.read_state(
                    input=x,
                ),
                mb.read_state(
                    input=y,
                ),
                mb.read_state(
                    input=z,
                ),
            )

        input_placeholders = {
            "x": mb.state_tensor_placeholder(
                shape=shape,
                dtype=types.fp16,
            ),
            "y": mb.state_tensor_placeholder(
                shape=shape,
                dtype=types.fp16,
            ),
            "z": mb.state_tensor_placeholder(
                shape=shape,
                dtype=types.fp16,
            ),
        }
        input_values = {}

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types=[
                (
                    *shape,
                    types.fp16,
                )
            ]
            * 3,
            expected_outputs=[
                np.zeros(
                    shape,
                )
            ]
            * 3,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestStatefulModel:
    @pytest.mark.xfail(reason="rdar://138957299 ([Bug] Stateful model slice update regression)")
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_state_model_with_slice_update(self, compute_unit, backend):
        def build(x_in, y_in, z_in, update_1, update_2):
            def single_slice_update(state, input):
                v = mb.slice_update(
                    x=input,
                    update=update_1,
                    begin=[0, 0],
                    end=[1, 2],
                )
                return mb.coreml_update_state(state=state, value=v)

            def double_slice_update(state, input):
                v = mb.slice_update(
                    x=input,
                    update=update_1,
                    begin=[0, 0],
                    end=[1, 2],
                )
                v = mb.slice_update(
                    x=input,
                    update=update_2,
                    begin=[1, 1],
                    end=[3, 3],
                )
                return mb.coreml_update_state(state=state, value=v)

            x = mb.read_state(input=x_in)
            y = mb.read_state(input=y_in)
            z = mb.read_state(input=z_in)

            for i in range(10):
                # single slice update
                x = single_slice_update(x_in, x)
                y = single_slice_update(y_in, y)
                z = single_slice_update(z_in, z)

                # double slice update
                x = double_slice_update(x_in, x)
                y = double_slice_update(y_in, y)
                z = double_slice_update(z_in, z)

            return mb.read_state(input=x_in), mb.read_state(input=y_in), mb.read_state(input=z_in)

        shape = (8, 9)

        input_placeholders = {
            "x_in": mb.state_tensor_placeholder(
                shape=shape,
                dtype=types.fp16,
            ),
            "y_in": mb.state_tensor_placeholder(
                shape=shape,
                dtype=types.fp16,
            ),
            "z_in": mb.state_tensor_placeholder(
                shape=shape,
                dtype=types.fp16,
            ),
            "update_1": mb.placeholder(
                shape=(1, 2),
                dtype=types.fp16,
            ),
            "update_2": mb.placeholder(
                shape=(2, 2),
                dtype=types.fp16,
            ),
        }

        update_1_val = np.array([[1, 2]], dtype=np.float16)
        update_2_val = np.array([[1, 2], [3, 4]], dtype=np.float16)
        input_values = {
            "update_1": update_1_val,
            "update_2": update_2_val,
        }

        output = np.zeros(shape, dtype=np.float16)
        output[:1, :2] = update_1_val
        output[1:3, 1:3] = update_2_val

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types=[
                (
                    *shape,
                    types.fp16,
                )
            ]
            * 3,
            expected_outputs=[
                [output] * 3,
                [output] * 3,
            ],
            compute_unit=compute_unit,
            backend=backend,
            pred_iters=2,
        )
