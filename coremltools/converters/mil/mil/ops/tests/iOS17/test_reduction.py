#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.tests.iOS17 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.testing_reqs import compute_units


class TestReduction:
    @pytest.mark.parametrize(
        "compute_unit, backend, op_name, output_dtype",
        itertools.product(
            compute_units, backends, ["reduce_argmax", "reduce_argmin"], ["int32", "uint16", None]
        ),
    )
    def test_reduce_arg_ios17_output_dtype(self, compute_unit, backend, op_name, output_dtype):
        def build(x):
            return getattr(mb, op_name)(x=x, axis=1, keep_dims=False, output_dtype=output_dtype)

        val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}
        output_np_type = np.uint16 if output_dtype == "uint16" else np.int32
        output_type = types.uint16 if output_dtype == "uint16" else types.int32
        expected_output_types = (2, output_type)
        expected_outputs = np.array(
            [2, 2] if op_name == "reduce_argmax" else [0, 0], dtype=output_np_type
        )

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "backend, op_name",
        itertools.product(
            backends,
            ["reduce_argmax", "reduce_argmin"],
        ),
    )
    def test_reduce_arg_ios17_output_dtype_invalid(self, backend, op_name):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

        def prog():
            return getattr(mb, op_name)(x=x, axis=1, keep_dims=False, output_dtype="dummy")

        with pytest.raises(ValueError, match='Invalid "output_dtype" dummy'):
            mb.program(input_specs=[], opset_version=backend.opset_version)(prog)
