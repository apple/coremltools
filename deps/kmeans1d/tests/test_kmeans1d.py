# MIT License
#
# Copyright (c) 2019 Daniel Steinberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Copyright © 2023 Apple Inc.

import unittest

from kmeans1d import cluster

def round_list(L, ndigits=0) -> list:
    """ Round all values in a list """
    return [round(v, ndigits=ndigits) for v in L]

def compute_inertial(points, weights, clusters, centroids) -> float:
    """ Compute the inertia (k-means loss, i.e. weighted sum of squared differences) """
    assert len(points) == len(weights) == len(clusters)
    assert all(0 <= k < len(centroids) for k in clusters)
    return sum(
        w * (x - centroids[k]) ** 2
        for x, w, k in zip(points, weights, clusters)
    )

class TestKmeans1D(unittest.TestCase):
    """kmeans1d tests"""
    def test_cluster(self):
        x = [4.0, 4.1, 4.2, -50, 200.2, 200.4, 200.9, 80, 100, 102]
        k = 4
        clusters, centroids = cluster(x, k)
        self.assertEqual(clusters, [1, 1, 1, 0, 3, 3, 3, 2, 2, 2])
        self.assertEqual(centroids, [-50.0, 4.1, 94.0, 200.5])

    def test_cluster_with_weights(self):
        x = [4.0, 4.1, 4.2, -50, 200.2, 200.4, 200.9, 80, 100, 102]
        w = [1, 1, 1, 0.125, 4, 1, 1, 3, 2, 2]
        k = 4
        clusters, centroids = cluster(x, k, weights=w)
        centroids = round_list(centroids, ndigits=9)  # because of numerical inaccuracy
        self.assertEqual(clusters, [0, 0, 0, 0, 3, 3, 3, 1, 2, 2])
        self.assertEqual(centroids, [1.936, 80.0, 101.0, 200.35])

    def test_weights_vs_repetition(self):
        x = [10, 24, 16, 12, 20]
        w = [3, 1, 4, 2, 3]
        k = 2

        # Unweighted
        u_clusters, _ = cluster(x, k)
        self.assertEqual(u_clusters, [0, 1, 0, 0, 1])

        # Weighted: different than unweighted
        w_clusters, w_centroids = cluster(x, k, weights=w)
        w_ssd = compute_inertial(x, w, w_clusters, w_centroids)
        self.assertEqual(w_clusters, [0, 1, 1, 0, 1])
        self.assertEqual(w_centroids, [10.8, 18.5])
        self.assertEqual(w_ssd, 66.8)

        # Repeated values: same as weighted
        r_x = [xi for xi, n in zip(x, w) for _ in range(n)]
        self.assertEqual(len(r_x), sum(w))
        r_clusters, r_centroids = cluster(r_x, k)
        r_ssd = compute_inertial(r_x, [1] * len(r_x), r_clusters, r_centroids)
        self.assertEqual(r_centroids, w_centroids)
        self.assertEqual(r_ssd, w_ssd)

if __name__ == '__main__':
    unittest.main()
