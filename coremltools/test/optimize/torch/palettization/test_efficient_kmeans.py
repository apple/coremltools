#  Copyright (c) 2026, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import torch

from coremltools.optimize.torch.palettization._efficient_kmeans import _EfficientKMeans


def test_efficient_kmeans_reduces_cluster_when_error_bnd_satisfied():
    # Regression test for issue #2698.
    #
    # On all-negative data, the previous implementation computed `reduce_inertia`
    # from the sum of the *cluster centers* (which can be negative), making the
    # subsequent sqrt produce NaN and the `rmse_error < error_bnd` check fail
    # silently. With the fix, `reduce_inertia` is the sum of per-point min
    # distance errors, so the comparison behaves as documented and the cluster
    # count is reduced from 3 to 2.
    X = torch.tensor(
        [[-1.0], [-1.1], [-0.9], [-10.0], [-10.1], [-9.9]], dtype=torch.float32
    )
    init_centers = torch.tensor(
        [[-1.0], [-10.0], [-50.0]], dtype=torch.float32
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)

    kmeans = _EfficientKMeans(
        n_clusters=3,
        init=init_centers,
        labels=labels,
        max_iter=5,
        tol=1e-4,
        error_bnd=2.0,
    )
    kmeans.fit(X)

    assert kmeans.n_clusters == 2
