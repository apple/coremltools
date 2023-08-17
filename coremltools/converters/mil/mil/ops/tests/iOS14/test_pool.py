#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.tests.iOS14 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.testing_reqs import compute_units


class TestAvgPool:
    @pytest.mark.parametrize(
        "compute_unit, backend, inputshape_kernelshape",
        itertools.product(
            compute_units,
            backends,
            [
                [(1, 1, 2), (2,)],
                [(1, 1, 2, 2), (2, 2)],
                [(1, 1, 2, 2, 2), (2, 2, 2)],
            ],
        ),
    )
    def test_avgpool_builder_to_backend_smoke_samelower_padtype(
        self, compute_unit, backend, inputshape_kernelshape
    ):
        input_shape, kernel_shape = inputshape_kernelshape
        rank = len(input_shape) - 2

        if backend.backend == "neuralnetwork" and rank == 3:
            pytest.skip(
                "pad_type `same_lower` not supported for 3d pooling in neuralnetwork backend"
            )
        if backend.backend == "mlprogram" and rank == 1:
            pytest.xfail(
                "rdar://98852008 (MIL backend producing wrong result for 1d pooling with pad_type "
                "same_lower)"
            )
        if backend.opset_version == ct.target.iOS15:
            pytest.skip("same_lower pad_type not supported in iOS15 opset.")

        x_val = np.arange(1, np.prod(input_shape) + 1).reshape(*input_shape).astype(np.float32)

        if rank == 1:
            expected_output_val = [0.5, 1.5]
        elif rank == 2:
            expected_output_val = [0.25, 0.75, 1, 2.5]
        else:
            expected_output_val = [0.125, 0.375, 0.5, 1.25, 0.75, 1.75, 2, 4.5]

        expected_output_types = [input_shape + (types.fp32,)]
        expected_outputs = [np.array(expected_output_val).reshape(*input_shape).astype(np.float32)]
        input_values = {"x": x_val}
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}

        def build(x):
            return mb.avg_pool(
                x=x,
                kernel_sizes=kernel_shape,
                pad_type="same_lower",
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
        "compute_unit, backend, num_dims",
        itertools.product(compute_units, backends, [1, 2, 3]),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, num_dims):
        kernel_sizes = [1, 2, 3]
        strides = [2, 1, 3]

        if num_dims == 1:
            x_val = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]], dtype=np.float32)
            expected_output_types = [(1, 1, 4, types.fp32), (1, 1, 3, types.fp32)]
            expected_outputs = [
                np.array([[[1.0, 3.0, 5.0, 7.0]]], dtype=np.float32),
                np.array([[[1.5, 4.0, 6.5]]], dtype=np.float32),
            ]
        elif num_dims == 2:
            x_val = np.array(
                [
                    [
                        [[-10.80291205, -6.42076184], [-7.07910997, 9.1913279]],
                        [[-3.18181497, 0.9132147], [11.9785544, 7.92449539]],
                    ]
                ],
                dtype=np.float32,
            )
            expected_output_types = [(1, 2, 1, 1, types.fp32), (1, 2, 2, 1, types.fp32)]
            expected_outputs = [
                np.array([[[[-8.611837]], [[-1.1343001]]]], dtype=np.float32),
                np.array(
                    [[[[-3.7778642], [1.056109]], [[4.4086123], [9.951525]]]],
                    dtype=np.float32,
                ),
            ]
        else:  # num_dims == 3
            x_val = np.array(
                [
                    [
                        [
                            [[-1, -5, -1], [-3, -3, 8], [2, 6, 2]],
                            [[-4, 7, -4], [4, 6, 7], [4, 4, 8]],
                            [[5, -3, 5], [0, -5, 8], [1, 7, 2]],
                        ]
                    ],
                    [
                        [
                            [[7, -3, -5], [5, 4, 7], [-2, -4, -3]],
                            [[-4, 3, -1], [6, -4, 4], [3, 6, 2]],
                            [[-1, 4, -4], [-2, -1, -2], [3, 2, 8]],
                        ]
                    ],
                ],
                dtype=np.float32,
            )
            expected_output_types = [
                (2, 1, 2, 2, 1, types.fp32),
                (2, 1, 2, 3, 1, types.fp32),
            ]
            expected_outputs = [
                np.array(
                    [
                        [[[[-0.8333334], [2.0]], [[1.6666667], [2.1666667]]]],
                        [[[[2.5], [1.1666667]], [[-1.0], [1.3333334]]]],
                    ],
                    dtype=np.float32,
                ),
                np.array(
                    [
                        [
                            [
                                [[-0.8333334], [2.0], [3.3333335]],
                                [[1.6666667], [2.1666667], [3.3333335]],
                            ]
                        ],
                        [
                            [
                                [[2.5], [1.1666667], [-3.0]],
                                [[-1.0], [1.3333334], [4.3333335]],
                            ]
                        ],
                    ],
                    dtype=np.float32,
                ),
            ]

        input_values = {"x": x_val}
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}

        def build(x):
            return [
                mb.avg_pool(
                    x=x,
                    kernel_sizes=kernel_sizes[:num_dims],
                    strides=strides[:num_dims],
                    pad_type="valid",
                ),
                mb.avg_pool(
                    x=x,
                    kernel_sizes=kernel_sizes[-num_dims:],
                    strides=strides[-num_dims:],
                    pad_type="same",
                    exclude_padding_from_average=True,
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


class TestMaxPool:
    @pytest.mark.parametrize(
        "compute_unit, backend, inputshape_kernelshape",
        itertools.product(
            compute_units,
            backends,
            [
                [(1, 1, 2), (2,)],
                [(1, 1, 2, 2), (2, 2)],
                [(1, 1, 2, 2, 2), (2, 2, 2)],
            ],
        ),
    )
    def test_maxpool_builder_to_backend_smoke_samelower_padtype(
        self, compute_unit, backend, inputshape_kernelshape
    ):
        input_shape, kernel_shape = inputshape_kernelshape
        rank = len(input_shape) - 2

        if backend.backend == "neuralnetwork" and rank == 3:
            pytest.skip(
                "pad_type `same_lower` not supported for 3d pooling in neuralnetwork backend"
            )
        if backend.backend == "mlprogram" and rank == 1:
            pytest.xfail(
                "rdar://98852008 (MIL backend producing wrong result for 1d pooling with pad_type "
                "same_lower)"
            )
        if backend.opset_version == ct.target.iOS15:
            pytest.skip("same_lower pad_type not supported in iOS15 opset.")

        x_val = np.arange(1, np.prod(input_shape) + 1).reshape(*input_shape).astype(np.float32)

        if rank == 1:
            expected_output_val = [1, 2]
        elif rank == 2:
            expected_output_val = [1, 2, 3, 4]
        else:
            expected_output_val = [1, 2, 3, 4, 5, 6, 7, 8]

        expected_output_types = [input_shape + (types.fp32,)]
        expected_outputs = [np.array(expected_output_val).reshape(*input_shape).astype(np.float32)]
        input_values = {"x": x_val}
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}

        def build(x):
            return mb.max_pool(
                x=x,
                kernel_sizes=kernel_shape,
                pad_type="same_lower",
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
        "compute_unit, backend, num_dims",
        itertools.product(compute_units, backends, [1, 2, 3]),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, num_dims):
        kernel_sizes = [1, 2, 3]
        strides = [2, 1, 3]

        if num_dims == 1:
            x_val = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]], dtype=np.float32)
            expected_output_types = [(1, 1, 4, types.fp32), (1, 1, 3, types.fp32)]
            expected_outputs = [
                np.array([[[1.0, 3.0, 5.0, 7.0]]], dtype=np.float32),
                np.array([[[2.0, 5.0, 7.0]]], dtype=np.float32),
            ]
        elif num_dims == 2:
            x_val = np.array(
                [
                    [
                        [[-10.80291205, -6.42076184], [-7.07910997, 9.1913279]],
                        [[-3.18181497, 0.9132147], [11.9785544, 7.92449539]],
                    ]
                ],
                dtype=np.float32,
            )
            expected_output_types = [(1, 2, 1, 1, types.fp32), (1, 2, 2, 1, types.fp32)]
            expected_outputs = [
                np.array([[[[-6.42076184]], [[0.9132147]]]], dtype=np.float32),
                np.array(
                    [[[[9.191328], [9.191328]], [[11.978555], [11.978555]]]],
                    dtype=np.float32,
                ),
            ]
        else:  # num_dims == 3
            x_val = np.array(
                [
                    [
                        [
                            [[-1, -5, -1], [-3, -3, 8], [2, 6, 2]],
                            [[-4, 7, -4], [4, 6, 7], [4, 4, 8]],
                            [[5, -3, 5], [0, -5, 8], [1, 7, 2]],
                        ]
                    ],
                    [
                        [
                            [[7, -3, -5], [5, 4, 7], [-2, -4, -3]],
                            [[-4, 3, -1], [6, -4, 4], [3, 6, 2]],
                            [[-1, 4, -4], [-2, -1, -2], [3, 2, 8]],
                        ]
                    ],
                ],
                dtype=np.float32,
            )
            expected_output_types = [
                (2, 1, 2, 2, 1, types.fp32),
                (2, 1, 2, 3, 1, types.fp32),
            ]
            expected_outputs = [
                np.array(
                    [
                        [[[[8.0], [8.0]], [[8.0], [8.0]]]],
                        [[[[7.0], [7.0]], [[4.0], [8.0]]]],
                    ],
                    dtype=np.float32,
                ),
                np.array(
                    [
                        [[[[8.0], [8.0], [6.0]], [[8.0], [8.0], [7.0]]]],
                        [[[[7.0], [7.0], [-2.0]], [[4.0], [8.0], [8.0]]]],
                    ],
                    dtype=np.float32,
                ),
            ]

        input_values = {"x": x_val}
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}

        def build(x):
            return [
                mb.max_pool(
                    x=x,
                    kernel_sizes=kernel_sizes[:num_dims],
                    strides=strides[:num_dims],
                    pad_type="valid",
                ),
                mb.max_pool(
                    x=x,
                    kernel_sizes=kernel_sizes[-num_dims:],
                    strides=strides[-num_dims:],
                    pad_type="same",
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


class TestL2Pool:
    @pytest.mark.parametrize(
        "compute_unit, backend, inputshape_kernelshape",
        itertools.product(
            compute_units,
            backends,
            [
                [(1, 1, 2), (2,)],
                [(1, 1, 2, 2), (2, 2)],
            ],
        ),
    )
    def test_l2pool_builder_to_backend_smoke_samelower_padtype(
        self, compute_unit, backend, inputshape_kernelshape
    ):
        input_shape, kernel_shape = inputshape_kernelshape
        rank = len(input_shape) - 2

        if backend.backend == "mlprogram" and rank == 1:
            pytest.xfail(
                "rdar://98852008 (MIL backend producing wrong result for 1d pooling with pad_type "
                "same_lower)"
            )
        if backend.opset_version == ct.target.iOS15:
            pytest.skip("same_lower pad_type not supported in iOS15 opset.")

        x_val = np.arange(1, np.prod(input_shape) + 1).reshape(*input_shape).astype(np.float32)

        if rank == 1:
            expected_output_val = [1, 2.236068]
        else:
            expected_output_val = [1, 2.236068, 3.162278, 5.477226]

        expected_output_types = [input_shape + (types.fp32,)]
        expected_outputs = [np.array(expected_output_val).reshape(*input_shape).astype(np.float32)]
        input_values = {"x": x_val}
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}

        def build(x):
            return mb.l2_pool(
                x=x,
                kernel_sizes=kernel_shape,
                pad_type="same_lower",
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
        "compute_unit, backend, num_dims",
        itertools.product(compute_units, backends, [1, 2]),
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend, num_dims):
        kernel_sizes = [1, 2, 3]
        strides = [2, 1, 3]

        if num_dims == 1:
            x_val = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]], dtype=np.float32)
            expected_output_types = [(1, 1, 4, types.fp32), (1, 1, 3, types.fp32)]
            expected_outputs = [
                np.array([[[1.0, 3.0, 5.0, 7.0]]], dtype=np.float32),
                np.array([[[2.236068, 7.071068, 9.219544]]], dtype=np.float32),
            ]
        elif num_dims == 2:
            x_val = np.array(
                [[[[-10.0, -6.0], [-7.0, 9.0]], [[-3.0, 0.0], [11.0, 7.0]]]],
                dtype=np.float32,
            )
            expected_output_types = [(1, 2, 1, 1, types.fp32), (1, 2, 2, 1, types.fp32)]
            expected_outputs = [
                np.array([[[[11.66190338]], [[3.0]]]], dtype=np.float32),
                np.array(
                    [[[[16.309507], [11.401754]], [[13.379088], [13.038404]]]],
                    dtype=np.float32,
                ),
            ]
        else:  # num_dims == 3
            pass  # Enum PoolingType3D has no value defined for name L2

        input_values = {"x": x_val}
        input_placeholders = {"x": mb.placeholder(shape=x_val.shape)}

        def build(x):
            return [
                mb.l2_pool(
                    x=x,
                    kernel_sizes=kernel_sizes[:num_dims],
                    strides=strides[:num_dims],
                    pad_type="valid",
                ),
                mb.l2_pool(
                    x=x,
                    kernel_sizes=kernel_sizes[-num_dims:],
                    strides=strides[-num_dims:],
                    pad_type="same",
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
