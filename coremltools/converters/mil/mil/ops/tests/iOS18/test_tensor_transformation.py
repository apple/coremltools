#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.tests.iOS18 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.testing_reqs import compute_units


def _test_eval(
    x,
    update,
    begin,
    end,
    stride=None,
    begin_mask=None,
    end_mask=None,
    squeeze_mask=None,
    ans=None,
    compute_unit=None,
    backend=None,
    x_builtin_dtype=None,
    run_conversion_test=True,
):
    # Test the value inference in pymil
    @mb.program(input_specs=[], opset_version=target.iOS18)
    def prog():
        res = mb.slice_update(
            x=x,
            update=update,
            begin=begin,
            end=end,
            stride=stride,
            begin_mask=begin_mask,
            end_mask=end_mask,
            squeeze_mask=squeeze_mask,
        )
        assert res.shape == ans.shape
        np.testing.assert_allclose(ans, res.val, atol=1e-04, rtol=1e-05)
        return res

    if not run_conversion_test:
        return

    # pymil to backend test
    x_val = np.array(x, dtype=np.float32)
    update_val = np.array(update, dtype=np.float32)
    begin_val = np.array(begin, dtype=np.int32)
    end_val = np.array(end, dtype=np.int32)

    input_placeholders = {
        "x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype),
        "update": mb.placeholder(shape=update_val.shape, dtype=x_builtin_dtype),
        "begin": mb.placeholder(shape=begin_val.shape, dtype=types.int32),
        "end": mb.placeholder(shape=end_val.shape, dtype=types.int32),
    }

    input_values = {"x": x_val, "update": update_val, "begin": begin_val, "end": end_val}

    expected_output_shape = list(ans.shape)
    expected_output_types = [expected_output_shape + [types.fp32]]
    expected_outputs = [ans]

    def build(x, update, begin, end):
        return mb.slice_update(
            x=x,
            update=update,
            begin=begin,
            end=end,
            begin_mask=begin_mask,
            end_mask=end_mask,
            squeeze_mask=squeeze_mask,
            stride=stride,
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


class TestSliceUpdate:
    @pytest.mark.parametrize(
        "compute_unit, backend, x_dtype, idx_dtype",
        itertools.product(
            compute_units,
            backends,
            (np.float16, np.float32, np.int32),
            (np.int16, np.int32, np.int8),
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, x_dtype, idx_dtype):
        x_builtin_dtype = types.numpy_type_to_builtin_type(x_dtype)
        idx_builtin_dtype = types.numpy_type_to_builtin_type(idx_dtype)

        x_val = np.array(list(range(24))).reshape((2, 3, 4)).astype(x_dtype)
        update_val = np.array([[[-1, -2], [-3, -4]]]).astype(x_dtype)
        begin_val = np.array([1, 1, 1], dtype=idx_dtype)
        end_val = np.array([2, 3, 3], dtype=idx_dtype)

        input_placeholders = {
            "x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype),
            "update": mb.placeholder(shape=update_val.shape, dtype=x_builtin_dtype),
            "begin": mb.placeholder(shape=begin_val.shape, dtype=idx_builtin_dtype),
            "end": mb.placeholder(shape=end_val.shape, dtype=idx_builtin_dtype),
        }

        input_values = {"x": x_val, "update": update_val, "begin": begin_val, "end": end_val}

        expected_output_types = [(2, 3, 4, x_builtin_dtype)] * 2
        copy_x_val = np.array(x_val, dtype=x_dtype)
        copy_x_val[1:2, 1:3, 1:3] = update_val
        expected_outputs = [copy_x_val, copy_x_val]

        def build(x, update, begin, end):
            begin_c = mb.const(val=begin_val)
            end_c = mb.const(val=end_val)
            update_c = mb.const(val=update_val)
            return [
                mb.slice_update(x=x, update=update, begin=begin, end=end),
                mb.slice_update(x=x, update=update_c, begin=begin_c, end=end_c),
            ]

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
        "compute_unit, backend, x_dtype, idx_dtype",
        itertools.product(
            compute_units,
            backends,
            (np.float16, np.float32, np.int32),
            (np.int16, np.int32, np.int8),
        ),
    )
    def test_stress(self, compute_unit, backend, x_dtype, idx_dtype):
        x_val = np.array(list(range(24))).reshape((2, 3, 4)).astype(x_dtype)
        x_builtin_dtype = types.numpy_type_to_builtin_type(x_dtype)

        update = np.random.rand(1, 1, 1).astype(x_dtype)
        ans = np.copy(x_val)
        ans[1:2, 1:2, 1:2] = update
        _test_eval(
            x=x_val,
            begin=[1, 1, 1],
            end=[2, 2, 2],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(1, 2, 2).astype(x_dtype)
        ans = np.copy(x_val)
        ans[1:2, 1:3, 1:4:2] = update
        _test_eval(
            x=x_val,
            begin=[1, 1, 1],
            end=[2, 3, 4],
            stride=[1, 1, 2],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(1, 2, 2).astype(x_dtype)
        ans = np.copy(x_val)
        ans[-3:-1, -3:-1, -3:-1] = update
        _test_eval(
            x=x_val,
            begin=[-3, -3, -3],
            end=[-1, -1, -1],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
            # rdar://128037672 ([Bug][iOS18][Classic CPU] slice_update fails on classic CPU on an unittest)
            run_conversion_test=False,
        )

        update = np.random.rand(1, 1, 1).astype(x_dtype)
        ans = np.copy(x_val)
        ans[0:-1, 0:-2, -3:-2] = update
        _test_eval(
            x=x_val,
            begin=[0, 0, -3],
            end=[-1, -2, -2],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(1, 1, 1).astype(x_dtype)
        ans = np.copy(x_val)
        ans[-1:0:-2, -1:1:-1, -1:-3:-3] = update
        _test_eval(
            x=x_val,
            begin=[-1, -1, -1],
            end=[0, 1, -3],
            stride=[-2, -1, -3],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(2, 2, 2).astype(x_dtype)
        ans = np.copy(x_val)
        ans[:2, 1:3, :4:2] = update
        _test_eval(
            x=x_val,
            begin=[1, 1, 1],
            end=[2, 3, 4],
            stride=[1, 1, 2],
            begin_mask=[True, False, True],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(2, 2, 2).astype(x_dtype)
        ans = np.copy(x_val)
        ans[:, 1:, :4:2] = update
        _test_eval(
            x=x_val,
            begin=[1, 1, 1],
            end=[2, 3, 4],
            stride=[1, 1, 2],
            begin_mask=[True, False, True],
            end_mask=[True, True, False],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(1, 2).astype(x_dtype)
        ans = np.copy(x_val)
        ans[1::1, 1, :3:2] = update
        _test_eval(
            x=x_val,
            begin=[1, 1, 1],
            end=[2, 3, 3],
            stride=[1, 1, 2],
            begin_mask=[False, False, True],
            end_mask=[True, False, False],
            squeeze_mask=[False, True, False],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(2, 3, 4).astype(x_dtype)
        ans = np.copy(x_val)
        ans[:, :, :] = update
        _test_eval(
            x=x_val,
            begin=[0, 0, 0],
            end=[0, 0, 0],
            stride=[1, 1, 1],
            begin_mask=[True, True, True],
            end_mask=[True, True, True],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(1, 1).astype(x_dtype)
        ans = np.copy(x_val)
        ans[1:2, 1:2, 1] = update
        _test_eval(
            x=x_val,
            begin=[1, 1, 1],
            end=[2, 2, 0],
            stride=[1, 1, 1],
            squeeze_mask=[False, False, True],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(1, 3, 4).astype(x_dtype)
        ans = np.copy(x_val)
        ans[1:2, ...] = update
        _test_eval(
            x=x_val,
            begin=[1, 0, 0],
            end=[2, 0, 0],
            stride=[1, 1, 1],
            begin_mask=[False, True, True],
            end_mask=[False, True, True],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(2, 3, 4).astype(x_dtype)
        ans = np.copy(x_val)
        ans[...] = update
        _test_eval(
            x=x_val,
            begin=[0, 0, 0],
            end=[0, 0, 0],
            stride=[1, 1, 1],
            begin_mask=[True, True, True],
            end_mask=[True, True, True],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(1, 3, 1).astype(x_dtype)
        ans = np.copy(x_val)
        ans[1:2, ..., 1:2] = update
        _test_eval(
            x=x_val,
            begin=[1, 0, 1],
            end=[2, 0, 2],
            stride=[1, 1, 1],
            begin_mask=[False, True, False],
            end_mask=[False, True, False],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(2, 3).astype(x_dtype)
        ans = np.copy(x_val)
        ans[..., 1] = update
        _test_eval(
            x=x_val,
            begin=[0, 0, 1],
            end=[0, 0, 0],
            stride=[1, 1, 1],
            begin_mask=[True, True, False],
            end_mask=[True, True, False],
            squeeze_mask=[False, False, True],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(
            4,
        ).astype(x_dtype)
        ans = np.copy(x_val)
        ans[0, 0, :] = update
        _test_eval(
            x=x_val,
            begin=[0, 0, 0],
            end=[0, 0, 0],
            stride=[1, 1, 1],
            begin_mask=[False, False, True],
            end_mask=[False, False, True],
            squeeze_mask=[True, True, False],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(1, 3, 4).astype(x_dtype)
        ans = np.copy(x_val)
        ans[1:2] = update
        _test_eval(
            x=x_val,
            begin=[1, 0, 0],
            end=[2, 0, 0],
            stride=[1, 1, 1],
            begin_mask=[False, True, True],
            end_mask=[False, True, True],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(1, 1, 4).astype(x_dtype)
        ans = np.copy(x_val)
        ans[1:2, 1:2] = update
        _test_eval(
            x=x_val,
            begin=[1, 1, 0],
            end=[2, 2, 0],
            stride=[1, 1, 1],
            begin_mask=[False, False, True],
            end_mask=[False, False, True],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(3, 4).astype(x_dtype)
        ans = np.copy(x_val)
        ans[1] = update
        _test_eval(
            x=x_val,
            begin=[1, 0, 0],
            end=[0, 0, 0],
            stride=[1, 1, 1],
            begin_mask=[False, True, True],
            end_mask=[False, True, True],
            squeeze_mask=[True, False, False],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(2, 3, 4).astype(x_dtype)
        ans = np.copy(x_val)
        ans[:] = update
        _test_eval(
            x=x_val,
            begin=[0, 0, 0],
            end=[0, 0, 0],
            begin_mask=[True, True, True],
            end_mask=[True, True, True],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

        update = np.random.rand(2, 3, 4).astype(x_dtype)
        ans = np.copy(x_val)
        ans[..., ::-1] = update
        _test_eval(
            x=x_val,
            begin=[0, 0, 0],
            end=[0, 0, 0],
            stride=[1, 1, -1],
            begin_mask=[True, True, True],
            end_mask=[True, True, True],
            update=update,
            ans=ans,
            compute_unit=compute_unit,
            backend=backend,
            x_builtin_dtype=x_builtin_dtype,
        )

    def test_builder_eval_scalar_corner_cases(self):
        pytest.xfail(
            "rdar://128221986 ([Feature][Slice_update] The backend is not supporting scalar update for the slice_update op)"
        )
        # two corner cases
        x_val = np.array([2.0])
        update = np.float32(3.14)
        ans = np.copy(x_val)
        ans[0] = update
        _test_eval(
            x=x_val,
            begin=[0],
            end=[0],
            squeeze_mask=[True],
            update=update,
            ans=ans,
            run_conversion_test=False,  # rank 0 input is not supported
        )

        x_val = np.array([[[[1.0], [3.0]]]])
        update = np.float32(7.78)
        ans = np.copy(x_val)
        ans[0, 0, 0, 0] = update
        _test_eval(
            x=x_val,
            begin=[0, 0, 0, 0],
            end=[0, 0, 0, 0],
            squeeze_mask=[True, True, True, True],
            update=update,
            ans=ans,
            run_conversion_test=False,  # rank 0 input is not supported
        )

    @staticmethod
    def test_rank_0_update_early_error_out():
        """
        Backend does not support rank-0 update for the slice_update op.
        coremltools should early error out until this radar is fixed:
        rdar://128221986 ([Feature][Slice_update] The backends is not supporting scalar update for the slice_update op)
        """
        with pytest.raises(
            ValueError, match="rank-0 'update' is not supported in 'slice_update' op"
        ):

            @mb.program(input_specs=[], opset_version=target.iOS18)
            def prog():
                return mb.slice_update(
                    x=[0.0, 0.0],
                    update=0.0,
                    begin=[0],
                    end=[1],
                    squeeze_mask=[True],
                )
