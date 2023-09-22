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


class TestInputWeightDifferentDtypes:
    """
    Starting from IOS17 the alpha/beta can have different dtypes from the input/output, so this
    test class is mainly to verify the behaviour of those alpha/beta related activations.
    """

    @pytest.mark.parametrize(
        "backend, different_dtype, op_name",
        itertools.product(
            backends,
            [True, False],
            ["elu", "leaky_relu", "prelu", "thresholded_relu"],
        ),
    )
    def test_builder_eval_alpha(self, backend, different_dtype, op_name):
        x = np.array([[[-1, 2, -3], [4, -5, 6]]], dtype=np.float32)
        alpha = np.float16(2.0) if different_dtype else np.float32(2.0)
        if op_name == "prelu":
            alpha = np.array([2.0, 2.0], dtype=alpha.dtype)  # prelu requires alpha to be rank 1.

        def prog():
            return getattr(mb, op_name)(x=x, alpha=alpha)

        mb.program(input_specs=[], opset_version=backend.opset_version)(prog)

    @pytest.mark.parametrize(
        "backend, different_dtype, op_name",
        itertools.product(
            backends,
            [True, False],
            [
                "clamped_relu",
                "linear_activation",
                "scaled_tanh",
                "sigmoid_hard",
                "softplus_parametric",
            ],
        ),
    )
    def test_builder_eval_alpha_beta(self, backend, different_dtype, op_name):
        x = np.array([[[-1, 2, -3], [4, -5, 6]]], dtype=np.float32)
        alpha = np.float16(2.0) if different_dtype else np.float32(2.0)
        beta = np.float16(1.0) if different_dtype else np.float32(1.0)
        if op_name == "softplus_parametric":
            alpha = np.array([2.0, 2.0], dtype=alpha.dtype)
            beta = np.array([1.0, 1.0], dtype=beta.dtype)

        def prog():
            return getattr(mb, op_name)(x=x, alpha=alpha, beta=beta)

        mb.program(input_specs=[], opset_version=backend.opset_version)(prog)

    @pytest.mark.parametrize(
        "compute_unit, backend, different_dtype, op_name",
        itertools.product(
            compute_units,
            backends,
            [True, False],
            ["elu", "leaky_relu", "prelu", "thresholded_relu"],
        ),
    )
    def test_builder_to_backend_numerical_alpha(
        self, compute_unit, backend, different_dtype, op_name
    ):
        x = np.array([[[-1, 2, -3], [4, -5, 6]]], dtype=np.float32)
        alpha = np.float16(2.0) if different_dtype else np.float32(2.0)
        if op_name == "prelu":
            alpha = np.array([2.0, 2.0], dtype=alpha.dtype)

        def calculate_by_np():
            if op_name == "elu":
                res = np.copy(x)
                res[res < 0] = alpha * (np.exp(res[res < 0]) - 1)
                return res
            elif op_name == "leaky_relu":
                res = np.copy(x)
                res[res < 0] *= 2.0
                return res
            elif op_name == "prelu":
                alpha_br = np.copy(alpha)
                for i in range(len(x.shape)):
                    if i != 1:
                        alpha_br = np.expand_dims(alpha_br, i)
                res = np.maximum(x, 0) + np.minimum(x, 0) * alpha_br
                return res
            elif op_name == "thresholded_relu":
                res = np.copy(x)
                res[res < alpha] = 0.0
                return res
            else:
                raise ValueError(f"Invalid op_name: {op_name}")

        def build(x):
            return getattr(mb, op_name)(x=x, alpha=alpha)

        run_compare_builder(
            build,
            input_placeholders={"x": mb.placeholder(shape=x.shape)},
            input_values={"x": x},
            expected_output_types=x.shape + (types.fp32,),
            expected_outputs=calculate_by_np(),
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, different_dtype, op_name",
        itertools.product(
            compute_units,
            backends,
            [True, False],
            [
                "clamped_relu",
                "linear_activation",
                "scaled_tanh",
                "sigmoid_hard",
                "softplus_parametric",
            ],
        ),
    )
    def test_builder_to_backend_numerical_alpha_beta(
        self, compute_unit, backend, different_dtype, op_name
    ):
        x = np.array([[[-1, 2, -3], [4, -5, 6]]], dtype=np.float32)
        alpha = np.float16(2.0) if different_dtype else np.float32(2.0)
        beta = np.float16(1.0) if different_dtype else np.float32(1.0)
        if op_name == "softplus_parametric":
            alpha = np.array([2.0, 2.0], dtype=alpha.dtype)
            beta = np.array([1.0, 1.0], dtype=beta.dtype)

        def calculate_by_np():
            if op_name == "clamped_relu":
                return np.minimum(np.maximum(x, 0), beta) + np.minimum(
                    np.minimum(x, 0) * alpha, beta
                )
            elif op_name == "linear_activation":
                return x * alpha + beta
            elif op_name == "scaled_tanh":
                return alpha * np.tanh(x * beta)
            elif op_name == "sigmoid_hard":
                return np.minimum(np.maximum((alpha * x) + beta, 0), 1)
            elif op_name == "softplus_parametric":
                alpha_br = alpha
                beta_br = beta
                for i in range(len(x.shape)):
                    if i != 1:
                        alpha_br = np.expand_dims(alpha_br, i)
                        beta_br = np.expand_dims(beta_br, i)
                res = alpha_br * np.log(np.exp(x * beta_br) + 1)
                return res
            else:
                raise ValueError(f"Invalid op_name: {op_name}")

        def build(x):
            return getattr(mb, op_name)(x=x, alpha=alpha, beta=beta)

        run_compare_builder(
            build,
            input_placeholders={"x": mb.placeholder(shape=x.shape)},
            input_values={"x": x},
            expected_output_types=x.shape + (types.fp32,),
            expected_outputs=calculate_by_np(),
            compute_unit=compute_unit,
            backend=backend,
        )
