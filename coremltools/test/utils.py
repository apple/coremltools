# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import numpy as np

_NUM_SAMPLES = 506
_FEATURE_NAMES = np.array(["feature_%d" % i for i in range(13)])


def load_test_data():
    rng = np.random.RandomState(42)
    n = _NUM_SAMPLES

    skewed = np.round(np.clip(rng.lognormal(-1.0, 2.0, n), 0.006, 89.0), 5)
    mostly_zero = np.where(
        rng.rand(n) < 0.735, 0.0, np.round(rng.uniform(12.5, 100.0, n) / 2.5) * 2.5
    )
    wide = np.round(rng.uniform(0.46, 27.74, n), 2)
    binary = (rng.rand(n) < 0.07).astype(np.float64)
    narrow = np.round(rng.uniform(0.385, 0.871, n), 3)
    gaussian = np.round(np.clip(rng.normal(6.28, 0.7, n), 3.56, 8.78), 3)
    percent = np.round(rng.uniform(2.9, 100.0, n), 1)
    positive = np.round(rng.uniform(1.13, 12.13, n), 4)
    discrete = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 24.0], n)
    large = np.round(rng.uniform(187.0, 711.0, n))
    ratio = np.round(rng.uniform(12.6, 22.0, n), 1)
    clustered = np.round(np.clip(396.9 - rng.exponential(40.0, n), 0.32, 396.9), 2)
    fraction = np.round(rng.uniform(1.73, 37.97, n), 2)

    data = np.array(
        np.stack(
            [skewed, mostly_zero, wide, binary, narrow, gaussian, percent, positive,
             discrete, large, ratio, clustered, fraction],
            axis=1,
        ),
        order="C",
    )

    signal = (
        5.0 * gaussian
        - 0.55 * fraction
        - 0.9 * ratio
        - 0.06 * skewed
        + 3.0 * binary
        + 0.02 * mostly_zero
        - 8.0 * narrow
        - 0.4 * positive
        - 0.01 * large
        - 0.02 * percent
        + rng.normal(0.0, 3.0, n)
    )
    signal = (signal - signal.mean()) / signal.std()
    target = np.round(np.clip(5.0 + 17.5 * np.exp(0.5 * signal), 5.0, 50.0), 1)

    return {"data": data, "target": target, "feature_names": _FEATURE_NAMES}
