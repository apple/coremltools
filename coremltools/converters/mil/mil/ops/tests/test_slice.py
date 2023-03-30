#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.testing_reqs import backends, compute_units
from coremltools.converters.mil.testing_utils import ssa_fn

from .testing_utils import UNK_SYM, run_compare_builder


class TestSliceByIndex:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x_val = np.array(list(range(24))).reshape((2, 3, 4)).astype(np.float32)
        begin_val = np.array([1, 1, 1], dtype=np.int32)
        end_val = np.array([2, 3, 3], dtype=np.int32)
        input_placeholders = {
            "x": mb.placeholder(shape=x_val.shape),
            "begin": mb.placeholder(shape=begin_val.shape, dtype=types.int32),
            "end": mb.placeholder(shape=end_val.shape, dtype=types.int32),
        }
        input_values = {"x": x_val, "begin": begin_val, "end": end_val}

        def build(x, begin, end):
            begin_c = mb.const(val=begin_val)
            end_c = mb.const(val=end_val)
            return [
                mb.slice_by_index(x=x, begin=begin, end=end),
                mb.slice_by_index(x=x, begin=begin_c, end=end_c)
            ]

        expected_output_types = [(UNK_SYM, UNK_SYM, UNK_SYM, types.fp32)] * 2
        expected_outputs = [np.array([[[17, 18], [21, 22]]], dtype=np.float32)] * 2
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    def test_type_inference(self):
        s0 = get_new_symbol()
        s1 = get_new_symbol()
        s2 = get_new_symbol()

        input_placeholders = {
            "x": mb.placeholder(shape=(10, s0, s1, s2)),
        }

        def build(x):
            return [
                mb.slice_by_index(
                    x=x, begin=[2, 5, 6, 12], end=[6, 9, 20, -9], stride=[2, 1, 2, 1]
                ),
                mb.slice_by_index(
                    x=x,
                    begin=[-2, -5, -3, 9],
                    end=[-6, -9, -6, -7],
                    stride=[-2, -1, -2, 1],
                ),
                mb.slice_by_index(
                    x=x,
                    begin=[0, 0, 0, 0],
                    end=[-6, -9, 3, -2],
                    stride=[-2, -3, 1, 2],
                    begin_mask=[True, True, True, True],
                    end_mask=[False, False, False, False],
                ),
                mb.slice_by_index(
                    x=x,
                    begin=[-2, 5, -1, -7],
                    end=[0, 0, 0, 0],
                    stride=[-2, -3, 1, -2],
                    begin_mask=[False, False, False, False],
                    end_mask=[True, True, True, True],
                ),
                mb.slice_by_index(
                    x=x, begin=[4, -1, 0, -5], end=[4, -1, 0, -5], stride=[1, -1, 2, -2]
                ),
            ]

        expected_output_types = [
            (2, 4, 7, UNK_SYM, types.fp32),
            (2, 4, 2, UNK_SYM, types.fp32),
            (3, 3, 3, UNK_SYM, types.fp32),
            (5, 2, 1, UNK_SYM, types.fp32),
            (0, 0, 0, 0, types.fp32),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            expected_output_types=expected_output_types,
            frontend_only=True,
        )


    @pytest.mark.xfail(reason="rdar://99664032")
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends,)
    )
    def test_single_element_edge_case(self, compute_unit, backend):
        x_val = np.array(list(range(6))).reshape((1, 3, 2)).astype(np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=x_val.shape),
        }
        input_values = {"x": x_val}

        def build(x):
            return mb.slice_by_index(
                x=x,
                begin=[-1, 0, 0],
                end=[-2, 0, 0],
                stride=[-1, 1, 1],
                begin_mask=[False, True, True],
                end_mask=[False, True, True]
            )

        expected_output_types = [(1, 3, 2, types.fp32)]
        expected_outputs = [np.array([[[0, 1], [2, 3], [4, 5]]], dtype=np.float32)]
        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
        )

    @ssa_fn
    def test_builder_eval_scalar_output_corner_cases(self):
        x1 = np.array([2.])
        x2 = np.array([[[[1.],[3.]]]])
        v = [
            mb.slice_by_index(
                x=x1, begin=[0,], end=[0], squeeze_mask=[True],
            ), 
            mb.slice_by_index(
                x=x2, begin=[0, 0, 0, 0], end=[0, 0, 0, 0], squeeze_mask=[True, True, True, True],
            ), 
        ]
        assert v[0].val.shape == ()
        assert v[0].val == 2
        assert v[1].val.shape == ()
        assert v[1].val == 1

    @ssa_fn
    def test_builder_eval(self):
        x_val = np.array(list(range(24))).reshape((2, 3, 4))
        v = [
            mb.slice_by_index(
                x=x_val, begin=[1, 1, 1], end=[2, 2, 2]
            ),  # x_val[1:2, 1:2, 1:2]
            mb.slice_by_index(
                x=x_val, begin=[1, 1, 1], end=[2, 3, 4], stride=[1, 1, 2]
            ),  #  x_val[1:2, 1:3, 1:4:2]
            mb.slice_by_index(
                x=x_val, begin=[-3, -3, -3], end=[-1, -1, -1]
            ),  # x_val[-3:-1, -3:-1, -3:-1]
            mb.slice_by_index(
                x=x_val, begin=[0, 0, -3], end=[-1, -2, -2]
            ),  # x_val[0:-1, 0:-2, -3:-2]
            mb.slice_by_index(
                x=x_val, begin=[-1, -1, -1], end=[0, 1, -3], stride=[-2, -1, -3]
            ),  # x_val[-1:0:-2, -1:1:-1, -1:-3:-3]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 1, 1],
                end=[2, 3, 4],
                stride=[1, 1, 2],
                begin_mask=[True, False, True],
            ),  # x_val[:2, 1:3, :4:2]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 1, 1],
                end=[2, 3, 4],
                stride=[1, 1, 2],
                begin_mask=[True, False, True],
                end_mask=[True, True, False],
            ),  # x_val[:, 1:, :4:2]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 1, 1],
                end=[2, 3, 4],
                stride=[1, 1, 2],
                begin_mask=[False, False, True],
                end_mask=[True, False, False],
                squeeze_mask=[False, True, False],
            ),  # x_val[1::1, 1, :3:2]
            mb.slice_by_index(
                x=x_val,
                begin=[0, 0, 0],
                end=[0, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[True, True, True],
                end_mask=[True, True, True],
            ),  # x_val[:, :, :]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 1, 1],
                end=[2, 2, 0],
                stride=[1, 1, 1],
                squeeze_mask=[False, False, True],
            ),  # x_val[1:2, 1:2, 1]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 0, 0],
                end=[2, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[False, True, True],
                end_mask=[False, True, True],
            ),  # x_val[1:2, ...]
            mb.slice_by_index(
                x=x_val,
                begin=[0, 0, 0],
                end=[0, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[True, True, True],
                end_mask=[True, True, True],
            ),  # x_val[...]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 0, 1],
                end=[2, 0, 2],
                stride=[1, 1, 1],
                begin_mask=[False, True, False],
                end_mask=[False, True, False],
            ),  # x_val[1:2, ..., 1:2]
            mb.slice_by_index(
                x=x_val,
                begin=[0, 0, 1],
                end=[0, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[True, True, False],
                end_mask=[True, True, False],
                squeeze_mask=[False, False, True],
            ),  # x_val[..., 1]
            mb.slice_by_index(
                x=x_val,
                begin=[0, 0, 0],
                end=[0, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[False, False, True],
                end_mask=[False, False, True],
                squeeze_mask=[True, True, False],
            ),  # x_val[0, 0, :]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 0, 0],
                end=[2, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[False, True, True],
                end_mask=[False, True, True],
            ),  # x_val[1:2]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 1, 0],
                end=[2, 2, 0],
                stride=[1, 1, 1],
                begin_mask=[False, False, True],
                end_mask=[False, False, True],
            ),  # x_val[1:2, 1:2]
            mb.slice_by_index(
                x=x_val,
                begin=[1, 0, 0],
                end=[0, 0, 0],
                stride=[1, 1, 1],
                begin_mask=[False, True, True],
                end_mask=[False, True, True],
                squeeze_mask=[True, False, False],
            ),  # x_val[1]
            mb.slice_by_index(
                x=x_val,
                begin=[0, 0, 0],
                end=[0, 0, 0],
                begin_mask=[True, True, True],
                end_mask=[True, True, True],
            ),  # x_val[:]
            mb.slice_by_index(
                x=x_val,
                begin=[0, 0, 0],
                end=[0, 0, 0],
                stride=[1, 1, -1],
                begin_mask=[True, True, True],
                end_mask=[True, True, True],
            ),  # x_val[..., ::-1]
        ]
        ans = [
            x_val[1:2, 1:2, 1:2],
            x_val[1:2, 1:3, 1:4:2],
            x_val[-3:-1, -3:-1, -3:-1],
            x_val[0:-1, 0:-2, -3:-2],
            x_val[-1:0:-2, -1:1:-1, -1:-3:-3],
            x_val[:2, 1:3, :4:2],
            x_val[:, 1:, :4:2],
            x_val[1::1, 1, :3:2],
            x_val[:, :, :],
            x_val[1:2, 1:2, 1],
            x_val[1:2, ...],
            x_val[...],
            x_val[1:2, ..., 1:2],
            x_val[..., 1],
            x_val[0, 0, :],
            x_val[1:2],
            x_val[1:2, 1:2],
            x_val[1],
            x_val[:],
            x_val[..., ::-1],
        ]
        for idx in range(len(v)):
            assert ans[idx].shape == v[idx].shape
            np.testing.assert_allclose(ans[idx], v[idx].val, atol=1e-04, rtol=1e-05)


    @staticmethod
    def test_slice_by_index():
        INPUT_SHAPE = (1, 2, 8, 16)

        @mb.program(input_specs=[mb.TensorSpec(shape=INPUT_SHAPE)])
        def prog(x):
            x = mb.slice_by_index(
                x=x,
                begin=[0, 0, 0, 0],
                end=[1, 2, 8, 12],
                stride=[1, 1, 2, 2],
                begin_mask=None,
                end_mask=None,
                squeeze_mask=None,
            )
            return x

        x = np.random.rand(*INPUT_SHAPE)

        # slice by index is x[begin[0]: end[0]: stride[0], begin[1]: end[1]: stride[1], ...]
        y_numpy = x[0:1:1, 0:2:1, 0:8:2, 0:12:2]

        model = ct.convert(prog, source="milinternal", convert_to="neuralnetwork")
        y_neuralnetwork = list(model.predict({'x': x}).values())[0]
        np.testing.assert_allclose(y_numpy, y_neuralnetwork)

        model = ct.convert(prog, source="milinternal", convert_to="mlprogram")
        y_mlprogram = list(model.predict({'x': x}).values())[0]
        # rdar://102217935 needs to be fixed before mlprogram will pass
        # np.testing.assert_allclose(y_numpy, y_mlprogram)

    @staticmethod
    def test_slice_by_index_slice_squeeze_separate():
        INPUT_SHAPE = (1, 2, 8, 16)

        @mb.program(input_specs=[mb.TensorSpec(shape=INPUT_SHAPE)])
        def prog(x):
            x = mb.slice_by_index(
                x=x,
                begin=[0, 0, 0, 0],
                end=[1, 2, 8, 12],
                stride=[1, 1, 1, 2],
                begin_mask=None,
                end_mask=None,
                squeeze_mask=[True, False, False, False],
            )
            return x

        x = np.random.rand(*INPUT_SHAPE)

        # slice by index is x[begin[0]: end[0]: stride[0], begin[1]: end[1]: stride[1], ...]
        # and squeeze dim 0
        y_numpy = x[0:1:1, 0:2:1, 0:8:1, 0:12:2]
        y_numpy = np.squeeze(y_numpy, axis=0)

        model = ct.convert(prog, source="milinternal", convert_to="neuralnetwork")
        y_neuralnetwork = list(model.predict({'x': x}).values())[0]

        assert y_numpy.shape == y_neuralnetwork.shape
        np.testing.assert_allclose(y_numpy, y_neuralnetwork)

        model = ct.convert(prog, source="milinternal", convert_to="mlprogram")
        y_mlprogram = list(model.predict({'x': x}).values())[0]
        # TODO: rdar://103365766 MLProgram does not apply squeeze_mask.
        # np.testing.assert_allclose(y_numpy, y_mlprogram)
