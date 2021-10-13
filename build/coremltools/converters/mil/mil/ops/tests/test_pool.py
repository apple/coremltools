#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.testing_reqs import *

from .testing_utils import run_compare_builder

backends = testing_reqs.backends


class TestAvgPool:
    @pytest.mark.parametrize(
        "use_cpu_only, backend, num_dims",
        itertools.product([True, False], backends, [1, 2, 3]),
    )
    def test_builder_to_backend_smoke(self, use_cpu_only, backend, num_dims):
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
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestMaxPool:
    @pytest.mark.parametrize(
        "use_cpu_only, backend, num_dims",
        itertools.product([True, False], backends, [1, 2, 3]),
    )
    def test_builder_to_backend_smoke(self, use_cpu_only, backend, num_dims):
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
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )


class TestL2Pool:
    @pytest.mark.parametrize(
        "use_cpu_only, backend, num_dims",
        itertools.product([True, False], backends, [1, 2]),
    )
    def test_builder_to_backend_smoke(self, use_cpu_only, backend, num_dims):
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
            use_cpu_only=use_cpu_only,
            frontend_only=False,
            backend=backend,
        )
