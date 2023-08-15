#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools._deps import _HAS_TORCH, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.tests.iOS16 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.mil.types.type_mapping import numpy_type_to_builtin_type

compute_units = testing_reqs.compute_units


if _HAS_TORCH:
    import torch

class TestPixelUnshuffle:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        val = np.array([[[[9.0, 5.0], [1.0, 3.0]]]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return [mb.pixel_unshuffle(x=x, downscale_factor=np.uint32(2))]

        expected_output_types = (1, 4, 1, 1, types.fp32)
        expected_outputs = np.array([[[[9.0]], [[5.0]], [[1.0]], [[3.0]]]], dtype=np.float32)

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.skipif(not testing_reqs._HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        "compute_unit, backend, shape, downscale_factor",
        itertools.product(
            compute_units,
            backends,
            [(1, 2, 4, 4), (2, 1, 8, 4)],
            [2, 4],
        ),
    )
    def test_builder_to_backend_stress(
        self,
        compute_unit,
        backend,
        shape,
        downscale_factor,
    ):
        val = np.random.rand(*shape)
        input_placeholders = {"x": mb.placeholder(shape=val.shape)}
        input_values = {"x": val}

        def build(x):
            return [mb.pixel_unshuffle(x=x, downscale_factor=np.uint32(downscale_factor))]

        torch_pixel_unshuffle = torch.nn.PixelUnshuffle(downscale_factor)
        expected_outputs = [torch_pixel_unshuffle(torch.Tensor(val)).numpy()]
        expected_output_types = [o.shape[:] + (types.fp32,) for o in expected_outputs]
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )


class TestReshapeLike:
    @pytest.mark.parametrize(
        "compute_unit, backend, InputShape_RefShapes_Begins_Ends_EndMasks, x_dtype, ref_dtype",
        itertools.product(
            compute_units,
            backends,
            [
                [(4, 3), ((2, 2, 3), (1, 3)), (0, 1), (2, 2), (False, False)],
                [(32,), ((1, 2, 2, 2), (3, 2, 2)), (1, 1), (0, 0), (True, True)],
                [(72, 1), ((1, 2, 3, 4, 1), (3,)), (1, 0), (0, 1), (True, False)],
            ],
            [np.float16, np.float32, np.int32, bool],
            [np.float16, np.float32, np.int32, bool],
        ),
    )
    def test_builder_to_backend_smoke(
        self,
        compute_unit,
        backend,
        InputShape_RefShapes_Begins_Ends_EndMasks,
        x_dtype,
        ref_dtype,
    ):
        input_shape, ref_shapes, begins, ends, end_masks = InputShape_RefShapes_Begins_Ends_EndMasks
        ref_shape_1, ref_shape_2 = ref_shapes
        x_builtin_dtype = numpy_type_to_builtin_type(x_dtype)
        ref_builtin_dtype = numpy_type_to_builtin_type(ref_dtype)

        x_val = np.random.randint(low=0, high=6, size=input_shape).astype(x_dtype)
        ref_tensor_1 = np.random.randint(low=0, high=6, size=ref_shape_1).astype(ref_dtype)
        ref_tensor_2 = np.random.randint(low=0, high=6, size=ref_shape_2).astype(ref_dtype)

        input_placeholders = {
            "x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype),
            "ref_tensor_1": mb.placeholder(shape=ref_shape_1, dtype=ref_builtin_dtype),
            "ref_tensor_2": mb.placeholder(shape=ref_shape_2, dtype=ref_builtin_dtype),
        }
        input_values = {
            "x": x_val,
            "ref_tensor_1": ref_tensor_1,
            "ref_tensor_2": ref_tensor_2,
        }

        def build(x, ref_tensor_1, ref_tensor_2):
            return mb.reshape_like(
                x=x,
                ref_tensors=(ref_tensor_1, ref_tensor_2),
                begins=begins,
                ends=ends,
                end_masks=end_masks,
            )

        output_shape = ()
        for ref_shape, begin, end, end_mask in zip(
            (ref_shape_1, ref_shape_2), begins, ends, end_masks
        ):
            if end_mask:
                output_shape += tuple(ref_shape[begin:])
            else:
                output_shape += tuple(ref_shape[begin:end])

        expected_output_types = [output_shape + (x_builtin_dtype,)]
        expected_outputs = [np.reshape(x_val, output_shape).astype(x_dtype)]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )
