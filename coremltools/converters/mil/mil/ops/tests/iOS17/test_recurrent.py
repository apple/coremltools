#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

from coremltools._deps import _HAS_TORCH, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil.mil.ops.tests.iOS14.test_recurrent import TestGRU as _TestGRU_iOS14
from coremltools.converters.mil.mil.ops.tests.iOS14.test_recurrent import (
    TestLSTM as _TestLSTM_iOS14,
)
from coremltools.converters.mil.mil.ops.tests.iOS14.test_recurrent import TestRNN as _TestRNN_iOS14
from coremltools.converters.mil.mil.ops.tests.iOS17 import backends
from coremltools.converters.mil.testing_reqs import compute_units


class TestGRU(_TestGRU_iOS14):
    @pytest.mark.parametrize(
        argnames=[
            "compute_unit",
            "backend",
            "seq_len",
            "batch_size",
            "input_size",
            "hidden_size",
            "has_bias",
            "output_sequence",
            "direction",
            "activation_functions",
            "symbolic",
            "dtype",
        ],
        argvalues=itertools.product(
            compute_units,
            backends,
            [1, 3],
            [1],  # <rdar://problem/59644603> [MIL] GRU with batch size 1 produces incorrect
            # output(always 0) for second batch onwards
            [1, 2],
            [1, 2],
            [True, False],
            [True, False],
            ["forward", "reverse"],
            [
                ["tanh", "sigmoid"],
                ["sigmoid", "tanh"],
            ],
            [True, False],
            [np.float16, np.float32],
        ),
    )
    def test_builder_to_backend_smoke(
        self,
        compute_unit,
        backend,
        seq_len,
        batch_size,
        input_size,
        hidden_size,
        has_bias,
        output_sequence,
        direction,
        activation_functions,
        symbolic,
        dtype,
    ):
        super().test_builder_to_backend_smoke(
            compute_unit,
            backend,
            seq_len,
            batch_size,
            input_size,
            hidden_size,
            has_bias,
            output_sequence,
            direction,
            activation_functions,
            symbolic,
            dtype,
        )


class TestLSTM(_TestLSTM_iOS14):
    @pytest.mark.parametrize(
        ",".join(
            [
                "compute_unit",
                "backend",
                "input_dims",
                "output_dim",
                "activation",
                "inner_activation",
                "outer_activation",
                "return_seq",
                "has_bias",
                "forget_bias",
                "has_peephole",
                "coupled_input_forget",
                "clip",
                "dtype",
            ]
        ),
        itertools.product(
            compute_units,
            backends,
            [[8, 32, 32]],
            [4],
            ["sigmoid"],
            ["tanh"],
            ["relu"],
            [False, True],
            [False, True],
            [False, True],
            [True, False],
            [False],
            [50.0, 0.01],
            [np.float16, np.float32],
        ),
    )
    def test_numpy_numerical(
        self,
        compute_unit,
        backend,
        input_dims,
        output_dim,
        activation,
        inner_activation,
        outer_activation,
        return_seq,
        has_bias,
        forget_bias,
        has_peephole,
        coupled_input_forget,
        clip,
        dtype,
    ):
        super().test_numpy_numerical(
            compute_unit,
            backend,
            input_dims,
            output_dim,
            activation,
            inner_activation,
            outer_activation,
            return_seq,
            has_bias,
            forget_bias,
            has_peephole,
            coupled_input_forget,
            clip,
            dtype,
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        argnames=[
            "compute_unit",
            "backend",
            "seq_len",
            "batch_size",
            "input_size",
            "hidden_size",
            "has_bias",
            "output_sequence",
            "direction",
            "symbolic",
            "dtype",
        ],
        argvalues=itertools.product(
            compute_units,
            backends,
            [1, 8],
            [1, 32],
            [1, 64],
            [1, 16],
            [True, False],
            [True, False],
            ["forward", "reverse"],
            [True, False],
            [np.float16, np.float32],
        ),
    )
    def test_builder_to_backend_smoke_unilstm(
        self,
        compute_unit,
        backend,
        seq_len,
        batch_size,
        input_size,
        hidden_size,
        has_bias,
        output_sequence,
        direction,
        symbolic,
        dtype,
    ):
        super().test_builder_to_backend_smoke_unilstm(
            compute_unit,
            backend,
            seq_len,
            batch_size,
            input_size,
            hidden_size,
            has_bias,
            output_sequence,
            direction,
            symbolic,
            dtype,
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        argnames=[
            "compute_unit",
            "backend",
            "seq_len",
            "batch_size",
            "input_size",
            "hidden_size",
            "has_bias",
            "output_sequence",
            "symbolic",
            "dtype",
        ],
        argvalues=itertools.product(
            compute_units,
            backends,
            [1, 8],
            [1, 32],
            [1, 64],
            [2, 16],
            [True, False],
            [True, False],
            [True, False],
            [np.float16, np.float32],
        ),
    )
    def test_builder_to_backend_smoke_bidirlstm(
        self,
        compute_unit,
        backend,
        seq_len,
        batch_size,
        input_size,
        hidden_size,
        has_bias,
        output_sequence,
        symbolic,
        dtype,
    ):
        super().test_builder_to_backend_smoke_bidirlstm(
            compute_unit,
            backend,
            seq_len,
            batch_size,
            input_size,
            hidden_size,
            has_bias,
            output_sequence,
            symbolic,
            dtype,
        )


class TestRNN(_TestRNN_iOS14):
    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        argnames=[
            "compute_unit",
            "backend",
            "seq_len",
            "batch_size",
            "input_size",
            "hidden_size",
            "has_bias",
            "output_sequence",
            "direction",
            "symbolic",
            "dtype",
        ],
        argvalues=itertools.product(
            compute_units,
            backends,
            [2, 8],
            [1, 32],
            [1, 64],
            [1, 16],
            [True, False],
            [True, False],
            ["forward", "reverse"],
            [True, False],
            [np.float16, np.float32],
        ),
    )
    def test_builder_to_backend_smoke(
        self,
        compute_unit,
        backend,
        seq_len,
        batch_size,
        input_size,
        hidden_size,
        has_bias,
        output_sequence,
        direction,
        symbolic,
        dtype,
    ):
        super().test_builder_to_backend_smoke(
            compute_unit,
            backend,
            seq_len,
            batch_size,
            input_size,
            hidden_size,
            has_bias,
            output_sequence,
            direction,
            symbolic,
            dtype,
        )
