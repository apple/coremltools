#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import platform

import numpy as np
import pytest

import coremltools as ct
from coremltools._deps import _HAS_TF_2, _HAS_TORCH, MSG_TF2_NOT_FOUND, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, get_new_symbol, types
from coremltools.converters.mil.mil.ops.tests.iOS14 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import (
    UNK_SYM,
    construct_inputs_from_placeholders,
    run_compare_builder,
)
from coremltools.converters.mil.testing_reqs import compute_units
from coremltools.converters.mil.testing_utils import random_gen

if _HAS_TORCH:
    import torch

if _HAS_TF_2:
    import tensorflow as tf


class TestNormalizationBatchNorm:
    @pytest.mark.parametrize(
        "compute_unit, backend, x_param_dtype",
        itertools.product(
            compute_units,
            backends,
            [(np.float16, np.float16), (np.float32, np.float32)],
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, x_param_dtype):
        x_dtype, param_dtype = x_param_dtype
        x_builtin_dtype = types.numpy_type_to_builtin_type(x_dtype)

        if x_dtype == np.float16 and backend.backend == "neuralnetwork":
            pytest.skip("No need to test fp16 for neuralnetwork backend.")

        x_val = np.array(
            [
                [
                    [[-16.0, 13.0], [11.0, -16.0]],
                    [[13.0, -15.0], [13.0, 9.0]],
                    [[-9.0, -4.0], [-6.0, 3.0]],
                ]
            ],
            dtype=x_dtype,
        )
        mean_val = np.array([9.0, 6.0, 3.0], dtype=param_dtype)
        variance_val = np.array([6.0, 1.0, 7.0], dtype=param_dtype)
        gamma_val = np.array([1.0, 1.0, 1.0], dtype=param_dtype)
        beta_val = np.array([1.0, 3.0, 0.0], dtype=param_dtype)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype)}
        input_values = {"x": x_val}

        def build(x):
            return [
                mb.batch_norm(x=x, mean=mean_val, variance=variance_val),
                mb.batch_norm(
                    x=x,
                    mean=mean_val,
                    variance=variance_val,
                    gamma=gamma_val,
                    beta=beta_val,
                    epsilon=param_dtype(1e-4),
                ),
            ]

        expected_output_types = [
            (1, 3, 2, 2, x_builtin_dtype),
            (1, 3, 2, 2, x_builtin_dtype),
        ]
        expected_outputs = [
            np.array(
                [
                    [
                        [[-10.206199, 1.6329918], [0.8164959, -10.206199]],
                        [[6.999965, -20.999895], [6.999965, 2.9999852]],
                        [[-4.53557, -2.6457493], [-3.4016776, 0.0]],
                    ]
                ],
                dtype=x_dtype,
            ),
            np.array(
                [
                    [
                        [[-9.206122, 2.6329796], [1.8164899, -9.206122]],
                        [[9.99965, -17.998951], [9.99965, 5.9998503]],
                        [[-4.535541, -2.6457324], [-3.4016557, 0.0]],
                    ]
                ],
                dtype=x_dtype,
            ),
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


class TestNormalizationInstanceNorm:
    @pytest.mark.parametrize(
        "compute_unit, backend, x_param_dtype",
        itertools.product(
            compute_units,
            backends,
            [(np.float16, np.float16), (np.float32, np.float32)],
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, x_param_dtype):
        x_dtype, param_dtype = x_param_dtype
        x_builtin_dtype = types.numpy_type_to_builtin_type(x_dtype)

        if x_dtype == np.float16 and backend.backend == "neuralnetwork":
            pytest.skip("No need to test fp16 for neuralnetwork backend.")

        x_val = np.array(
            [
                [
                    [[-16.0, 13.0], [11.0, 16.0]],
                    [[13.0, 15.0], [13.0, 9.0]],
                    [[-9.0, 4.0], [-6.0, 3.0]],
                ],
                [
                    [[-5.0, 1.0], [12.0, 3.0]],
                    [[0.0, 9.0], [2.0, -8.0]],
                    [[2.0, 5.0], [10.0, 0.0]],
                ],
            ],
            dtype=x_dtype,
        )
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype)}
        input_values = {"x": x_val}

        def build(x):
            return mb.instance_norm(x=x, epsilon=param_dtype(1e-2))

        expected_output_types = [(2, 3, 2, 2, x_builtin_dtype)]
        expected_outputs = [
            np.array(
                [
                    [
                        [[-1.71524656, 0.54576027], [0.38982874, 0.77965748]],
                        [[0.22917463, 1.14587319], [0.22917463, -1.60422242]],
                        [[-1.2470212, 1.06887531], [-0.71258354, 0.89072943]],
                    ],
                    [
                        [[-1.27070526, -0.28693344], [1.51664821, 0.04099049]],
                        [[-0.12380638, 1.36187018], [0.20634397, -1.44440776]],
                        [[-0.59714057, 0.19904686], [1.5260259, -1.12793219]],
                    ],
                ],
                dtype=np.float32,
            )
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
        "compute_unit, backend, x_param_dtype",
        itertools.product(
            compute_units,
            backends,
            [(np.float16, np.float16), (np.float32, np.float32)],
        ),
    )
    def test_builder_to_backend_smoke_with_gamma_and_beta(
        self, compute_unit, backend, x_param_dtype
    ):
        x_dtype, param_dtype = x_param_dtype
        x_builtin_dtype = types.numpy_type_to_builtin_type(x_dtype)

        if x_dtype == np.float16 and backend.backend == "neuralnetwork":
            pytest.skip("No need to test fp16 for neuralnetwork backend.")

        x_val = np.array(
            [
                [
                    [[-16.0, 13.0], [11.0, 16.0]],
                    [[13.0, 15.0], [13.0, 9.0]],
                    [[-9.0, 4.0], [-6.0, 3.0]],
                ],
                [
                    [[-5.0, 1.0], [12.0, 3.0]],
                    [[0.0, 9.0], [2.0, -8.0]],
                    [[2.0, 5.0], [10.0, 0.0]],
                ],
            ],
            dtype=x_dtype,
        )
        gamma_val = np.array([-9.0, 3.2, 1.3], dtype=param_dtype)
        beta_val = np.array([-0.8, 3.4, 1.2], dtype=param_dtype)

        input_placeholders = {"x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype)}
        input_values = {"x": x_val}

        def build(x):
            return mb.instance_norm(x=x, gamma=gamma_val, beta=beta_val, epsilon=param_dtype(1e-2))

        expected_output_types = [(2, 3, 2, 2, x_builtin_dtype)]
        expected_outputs = [
            np.array(
                [
                    [
                        [[14.63721807, -5.71184211], [-4.30845865, -7.8169173]],
                        [[4.1333588, 7.06679399], [4.1333588, -1.73351158]],
                        [[-0.42112757, 2.58953791], [0.27364139, 2.35794826]],
                    ],
                    [
                        [[10.6363473, 1.782401], [-14.44983388, -1.16891443]],
                        [[3.00381959, 7.75798456], [4.06030069, -1.22210484]],
                        [[0.42371726, 1.45876091], [3.18383368, -0.26631185]],
                    ],
                ],
                dtype=np.float32,
            )
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

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        "rank, compute_unit, backend, epsilon, x_param_dtype",
        itertools.product(
            [3, 4],
            compute_units,
            backends,
            [1e-3, 1e-5, 1e-10],
            [(np.float16, np.float16), (np.float32, np.float32)],
        ),
    )
    def test_builder_to_backend_stress(self, rank, compute_unit, backend, epsilon, x_param_dtype):
        x_dtype, param_dtype = x_param_dtype
        x_builtin_dtype = types.numpy_type_to_builtin_type(x_dtype)

        if x_dtype == np.float16 and backend.backend == "neuralnetwork":
            pytest.skip("No need to test fp16 for neuralnetwork backend.")

        shape = np.random.randint(low=2, high=6, size=rank)
        x_val = random_gen(shape=shape, rand_min=-100.0, rand_max=100.0).astype(x_dtype)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype)}
        input_values = {"x": x_val}

        def build(x):
            return mb.instance_norm(x=x, epsilon=param_dtype(epsilon))

        layer = torch.nn.InstanceNorm2d if rank == 4 else torch.nn.InstanceNorm1d
        torch_op = layer(num_features=shape[1], eps=epsilon)
        # PyTorch's batch_norm op is not implemented for fp16, so need to cast to fp32 first.
        expected_outputs = [torch_op(torch.as_tensor(x_val.astype(np.float32))).numpy()]
        expected_output_types = [o.shape[:] + (x_builtin_dtype,) for o in expected_outputs]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-3,
            rtol=1e-4,
            also_compare_shapes=True,
        )


class TestNormalizationL2Norm:
    @staticmethod
    def _compute_l2_norm(val, eps):
        shape = val.shape
        rank = len(shape)
        batch_dims = rank - 3
        if batch_dims == 0:
            square_sum = np.sum(val**2)
            output = val / np.power(square_sum + eps, 0.5)
        else:
            batch_dim_prod = np.prod(shape[:batch_dims])
            reshape_val = np.reshape(val, (batch_dim_prod, -1))
            square_sum = np.sum(reshape_val * reshape_val, axis=1, keepdims=True) + eps
            output = reshape_val / np.power(square_sum, 0.5)
            output = np.reshape(output, shape)
        return output

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x_val = np.array([[[1.0, -7.0], [5.0, -6.0], [-3.0, -5.0]]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        input_values = {"x": x_val}

        def build(x):
            return [mb.l2_norm(x=x, epsilon=1e-10)]

        expected_output_types = [(1, 3, 2, types.fp32)]
        expected_outputs = [
            np.array(
                [
                    [
                        [0.08304548, -0.58131838],
                        [0.41522741, -0.4982729],
                        [-0.24913645, -0.41522741],
                    ]
                ],
                dtype=np.float32,
            )
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
        "compute_unit, backend, rank, epsilon, x_param_dtype",
        itertools.product(
            compute_units,
            backends,
            [3, 4, 5],
            [1e-4, 5.7],
            [(np.float16, np.float16), (np.float32, np.float32)],
        ),
    )
    def test_builder_to_backend_stress(self, compute_unit, backend, rank, epsilon, x_param_dtype):
        x_dtype, param_dtype = x_param_dtype
        x_builtin_dtype = types.numpy_type_to_builtin_type(x_dtype)

        if x_dtype == np.float16 and backend.backend == "neuralnetwork":
            pytest.skip("No need to test fp16 for neuralnetwork backend.")

        shape = np.random.randint(low=2, high=6, size=rank)
        x_val = random_gen(shape=shape, rand_min=-1.0, rand_max=1.0).astype(x_dtype)
        input_placeholders = {"x": mb.placeholder(shape=shape, dtype=x_builtin_dtype)}
        input_values = {"x": x_val}

        def build(x):
            return [mb.l2_norm(x=x, epsilon=param_dtype(epsilon))]

        output = TestNormalizationL2Norm._compute_l2_norm(x_val, epsilon)
        expected_output_types = [list(output.shape) + [x_builtin_dtype]]
        expected_outputs = [output]

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
        "rank, epsilon",
        itertools.product(
            [3, 4, 5],
            [1e-4, 11.2],
        ),
    )
    def test_builder_eval_stress(self, rank, epsilon):
        shape = np.random.randint(low=2, high=6, size=rank)
        x_val = random_gen(shape=shape, rand_min=-1, rand_max=1)
        with Function({}):
            res = mb.l2_norm(x=x_val, epsilon=epsilon)
            ref = TestNormalizationL2Norm._compute_l2_norm(x_val, epsilon)
            np.testing.assert_allclose(ref, res.val, atol=1e-6, rtol=1e-5)


class TestNormalizationLayerNorm:
    @staticmethod
    def _keras_layer_norm(x, axes, epsilon):
        layer = tf.keras.layers.LayerNormalization(axis=axes, epsilon=epsilon)
        data = tf.constant(x, dtype=tf.float32)
        output = layer(data)
        return output.numpy()

    @staticmethod
    def _np_layer_norm(x, axes, gamma=None, beta=None, epsilon=1e-5):
        rank = len(x.shape)
        axes = [axis + rank if axis < 0 else axis for axis in axes]
        normalized_shape = [x.shape[i] if i in axes else 1 for i in range(rank)]
        gamma = (
            np.ones(shape=normalized_shape)
            if gamma is None
            else np.reshape(gamma, normalized_shape)
        )
        beta = (
            np.zeros(shape=normalized_shape) if beta is None else np.reshape(beta, normalized_shape)
        )
        num = x - np.mean(x, axis=tuple(axes), keepdims=True)
        dem = np.sqrt(
            np.sum(np.square(num), axis=tuple(axes), keepdims=True) / np.prod(normalized_shape)
            + epsilon
        )
        return num / dem * gamma + beta

    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x_val = np.array([[[1.0, -7.0], [5.0, -6.0], [-3.0, -5.0]]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        input_values = {"x": x_val}
        gamma_val = np.array([1.0, 1.0], dtype=np.float32)
        beta_val = np.array([1.0, 0.0], dtype=np.float32)

        def build(x):
            return [
                # V2->V1 lowering (op_mappings.py): if branch
                mb.layer_norm(x=x, axes=[2], epsilon=1e-4),
                # V2->V1 lowering (op_mappings.py): else branch
                mb.layer_norm(x=x, axes=[-2, -1], epsilon=1e-4),
                # V2->V1 lowering (op_mappings.py): if branch with scale
                mb.layer_norm(x=x, axes=[2], epsilon=1e-4, gamma=gamma_val, beta=beta_val),
            ]

        expected_output_types = [
            (1, 3, 2, types.fp32),
            (1, 3, 2, types.fp32),
            (1, 3, 2, types.fp32),
        ]
        expected_outputs = [
            np.array(
                [
                    [
                        [0.9999969, -0.9999969],
                        [0.99999833, -0.99999833],
                        [0.99995005, -0.99995005],
                    ]
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [
                        [0.82687193, -1.06312108],
                        [1.77186835, -0.82687193],
                        [-0.11812456, -0.59062278],
                    ]
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [
                        [1.9999969, -0.9999969],
                        [1.99999833, -0.99999833],
                        [1.99995005, -0.99995005],
                    ]
                ],
                dtype=np.float32,
            ),
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
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke_rank_2(self, compute_unit, backend):
        x_val = np.array([[1.0, -7.0], [5.0, -6.0], [-3.0, -5.0]], dtype=np.float32)
        gamma_val = np.array([1.0, 1.0], dtype=np.float32)
        beta_val = np.array([1.0, 0.0], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        input_values = {"x": x_val}

        def build(x):
            return [
                # V2->V1 lowering (op_mappings.py): if branch
                mb.layer_norm(x=x, axes=[1], epsilon=1e-4),
                mb.layer_norm(x=x, axes=[1], epsilon=1e-4, gamma=gamma_val, beta=beta_val),
            ]

        expected_output_types = [(3, 2, types.fp32), (3, 2, types.fp32)]
        expected_outputs = [
            np.array(
                [
                    [0.9999969, -0.9999969],
                    [0.99999833, -0.99999833],
                    [0.99995005, -0.99995005],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [1.9999969, -0.9999969],
                    [1.99999833, -0.99999833],
                    [1.99995005, -0.99995005],
                ],
                dtype=np.float32,
            ),
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
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke_with_dynamic_shape(self, compute_unit, backend):
        x_val = np.array([[[1.0, -7.0], [5.0, -6.0], [-3.0, -5.0]]], dtype=np.float32)
        shape = (get_new_symbol(), get_new_symbol(), 2)
        input_placeholders = {"x": mb.placeholder(shape=shape)}
        input_values = {"x": x_val}

        def build(x):
            return [
                mb.layer_norm(x=x, axes=[2], epsilon=1e-4),
            ]

        expected_output_types = [(UNK_SYM, UNK_SYM, 2, types.fp32)]
        expected_outputs = [
            np.array(
                [
                    [
                        [0.9999969, -0.9999969],
                        [0.99999833, -0.99999833],
                        [0.99995005, -0.99995005],
                    ]
                ],
                dtype=np.float32,
            ),
        ]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            inputs=construct_inputs_from_placeholders(input_placeholders, 10)
            if backend.backend == "mlprogram"
            else None,
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, rank_and_axes, epsilon, provides_gamma_beta",
        itertools.product(
            compute_units,
            backends,
            [[3, [0, 2]], [3, [-2]], [4, [0, 1, 3]], [5, [0, 4]], [5, [-5, -4, -3, -2, -1]]],
            [0.0001, 0.01],
            [True, False],
        ),
    )
    def test_builder_to_backend_stress_numpy(
        self, compute_unit, backend, rank_and_axes, epsilon, provides_gamma_beta
    ):

        if (
            backend.backend == "mlprogram"
            and backend.precision == "fp16"
            and compute_unit != ct.ComputeUnit.CPU_ONLY
        ):
            pytest.xfail(
                "rdar://80662357 ([GPU failures] LayerNorm FP16 tests failing on GPU with numerical errors)"
            )

        if (
            backend.backend == "neuralnetwork"
            and compute_unit != ct.ComputeUnit.CPU_ONLY
            and platform.machine() == "arm64"
        ):
            pytest.xfail(
                "rdar://98015195 ([M1 native tests] Some MIL unittests are failing on M1 native)"
            )

        rank, axes = rank_and_axes
        shape = np.random.randint(low=2, high=6, size=rank)
        x_val = random_gen(shape=shape, rand_min=-100.0, rand_max=100.0)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        input_values = {"x": x_val}

        gamma, beta = None, None

        if provides_gamma_beta:
            positive_axes = [axis + rank if axis < 0 else axis for axis in axes]
            normalized_shape = [shape[i] for i in range(rank) if i in positive_axes]
            gamma = random_gen(shape=normalized_shape, rand_min=-100, rand_max=100)
            beta = random_gen(shape=normalized_shape, rand_min=-100, rand_max=100)

        def build(x):
            return [mb.layer_norm(x=x, axes=axes, epsilon=epsilon, gamma=gamma, beta=beta)]

        output = TestNormalizationLayerNorm._np_layer_norm(
            x=x_val, axes=axes, epsilon=epsilon, gamma=gamma, beta=beta
        )
        expected_output_types = [tuple(output.shape) + (types.fp32,)]
        expected_outputs = [output]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-3,
            rtol=1e-4,
        )

    @pytest.mark.skipif(not _HAS_TF_2, reason=MSG_TF2_NOT_FOUND)
    @pytest.mark.parametrize(
        "compute_unit, backend, rank_and_axes, epsilon, x_param_dtype",
        itertools.product(
            compute_units,
            backends,
            [[3, [0, 2]], [3, [-2]], [4, [0, 1, 3]], [5, [0, 4]], [5, [-5, -4, -3, -2, -1]]],
            [0.0001, 0.01],
            [(np.float16, np.float16), (np.float32, np.float32)],
        ),
    )
    def test_builder_to_backend_stress_keras(
        self, compute_unit, backend, rank_and_axes, epsilon, x_param_dtype
    ):
        x_dtype, param_dtype = x_param_dtype
        x_builtin_dtype = types.numpy_type_to_builtin_type(x_dtype)

        if x_dtype == np.float16 and backend.backend == "neuralnetwork":
            pytest.skip("No need to test fp16 for neuralnetwork backend.")

        rank, axes = rank_and_axes
        shape = np.random.randint(low=2, high=6, size=rank)
        x_val = random_gen(shape=shape, rand_min=-100.0, rand_max=100.0).astype(x_dtype)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype)}
        input_values = {"x": x_val}

        def build(x):
            return [mb.layer_norm(x=x, axes=axes, epsilon=param_dtype(epsilon))]

        output = TestNormalizationLayerNorm._keras_layer_norm(x=x_val, axes=axes, epsilon=epsilon)
        expected_output_types = [tuple(output.shape) + (x_builtin_dtype,)]
        expected_outputs = [output]

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
        "rank_and_axes, epsilon",
        itertools.product(
            [
                [3, [0, 2]],
                [3, [-2, -1]],
                [4, [0, 1, 2, 3]],
                [5, [0, 2, -1]],
                [5, [-5, -4, -3, -2, -1]],
            ],
            [0.0001, 0.01],
        ),
    )
    def test_builder_eval_stress(self, rank_and_axes, epsilon):
        rank, axes = rank_and_axes
        shape = np.random.randint(low=2, high=6, size=rank)
        x_val = random_gen(shape=shape, rand_min=-100.0, rand_max=100.0)
        positive_axes = [axis + rank if axis < 0 else axis for axis in axes]
        normalized_shape = [shape[i] for i in range(rank) if i in positive_axes]
        gamma_val = random_gen(shape=normalized_shape, rand_min=-100, rand_max=100)
        beta_val = random_gen(shape=normalized_shape, rand_min=-100, rand_max=100)
        with Function({}):
            res = mb.layer_norm(x=x_val, axes=axes, epsilon=epsilon, gamma=gamma_val, beta=beta_val)
            ref = TestNormalizationLayerNorm._np_layer_norm(
                x=x_val, axes=axes, epsilon=epsilon, gamma=gamma_val, beta=beta_val
            )
            np.testing.assert_allclose(ref, res.val, atol=1e-04, rtol=1e-05)


class TestNormalizationLocalResponseNorm:
    @pytest.mark.parametrize(
        "compute_unit, backend",
        itertools.product(
            compute_units,
            backends,
        ),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):
        x_val = np.array([[[1.0, -7.0], [5.0, -6.0], [-3.0, -5.0]]], dtype=np.float32)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}
        input_values = {"x": x_val}

        def build(x):
            return [
                mb.local_response_norm(x=x, size=2),
                mb.local_response_norm(x=x, size=3, alpha=0.0001, beta=0.75, k=1.0),
            ]

        expected_output_types = [(1, 3, 2, types.fp32), (1, 3, 2, types.fp32)]
        expected_outputs = [
            np.array(
                [
                    [
                        [0.99996257, -6.98716545],
                        [4.99531746, -5.99191284],
                        [-2.99898791, -4.99531746],
                    ]
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [
                        [0.99997497, -6.99143696],
                        [4.99687672, -5.99460602],
                        [-2.99932504, -4.99687672],
                    ]
                ],
                dtype=np.float32,
            ),
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

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        "compute_unit, backend, rank, size, alpha, beta, k, x_param_dtype",
        itertools.product(
            compute_units,
            backends,
            [rank for rank in range(3, 6)],
            [2, 3, 5],
            [0.0001, 0.01],
            [0.75, 1.0],
            [1.0, 2.0],
            [(np.float16, np.float16), (np.float32, np.float32)],
        ),
    )
    def test_builder_to_backend_stress(
        self, compute_unit, backend, rank, size, alpha, beta, k, x_param_dtype
    ):
        x_dtype, param_dtype = x_param_dtype
        x_builtin_dtype = types.numpy_type_to_builtin_type(x_dtype)

        if x_dtype == np.float16 and backend.backend == "neuralnetwork":
            pytest.skip("No need to test fp16 for neuralnetwork backend.")

        shape = np.random.randint(low=2, high=5, size=rank)
        x_val = random_gen(shape=shape).astype(x_dtype)
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape, dtype=x_builtin_dtype)}
        input_values = {"x": x_val}

        def build(x):
            return mb.local_response_norm(
                x=x, size=size, alpha=param_dtype(alpha), beta=param_dtype(beta), k=param_dtype(k)
            )

        torch_lrn = torch.nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
        # PyTorch doesn't support LocalResponseNorm with fp16, so need to cast to float32 first.
        expected_outputs = [torch_lrn(torch.as_tensor(x_val.astype(np.float32))).numpy()]
        expected_output_types = [o.shape[:] + (x_builtin_dtype,) for o in expected_outputs]

        run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-2,
            rtol=1e-3,
        )
