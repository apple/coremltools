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

from collections import namedtuple
import ctypes
from typing import Optional, Sequence

from . import _core  # type: ignore


Clustered = namedtuple('Clustered', 'clusters centroids')

_DLL = ctypes.cdll.LoadLibrary(_core.__file__)


def cluster(
        array: Sequence[float],
        k: int,
        *,
        weights: Optional[Sequence[float]] = None) -> Clustered:
    """
    :param array: A sequence of floats
    :param k: Number of clusters (int)
    :param weights: Sequence of weights (if provided, must have same length as `array`)
    :return: A tuple with (clusters, centroids)
    """
    assert k > 0, f'Invalid k: {k}'
    n = len(array)
    assert n > 0, f'Invalid len(array): {n}'
    k = min(k, n)

    if weights is not None:
        assert len(weights) == n, f'len(weights)={len(weights)} != len(array)={n}'

    c_array = (ctypes.c_double * n)(*array)
    c_n = ctypes.c_ulong(n)
    c_k = ctypes.c_ulong(k)
    c_clusters = (ctypes.c_ulong * n)()
    c_centroids = (ctypes.c_double * k)()

    if weights is None:
        _DLL.cluster(c_array, c_n, c_k, c_clusters, c_centroids)
    else:
        c_weights = (ctypes.c_double * n)(*weights)
        _DLL.cluster_with_weights(c_array, c_weights, c_n, c_k, c_clusters, c_centroids)


    clusters = list(c_clusters)
    centroids = list(c_centroids)

    output = Clustered(clusters=clusters, centroids=centroids)

    return output
