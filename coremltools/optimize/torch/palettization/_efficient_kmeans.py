#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as _np
import torch as _torch
import torch.distributed as _dist


class _EfficientKMeans:
    """
    _EfficientKMeans is primarily used by palettization to perform a k-means operation. This class also has an in-house
    implementation of k-means, called ``kmeans_pp`` which runs entirely on GPU and is ~10x faster than sklearn's API.
    """

    @staticmethod
    def get_cluster_avg(n_clusters: int, indices, vals):
        v_sum = (
            _torch.zeros([n_clusters] + list(vals[0].size()), dtype=vals.dtype)
            .to(vals.device)
            .index_add_(0, indices, vals)
        )
        v_numel = (
            _torch.zeros(n_clusters, dtype=_torch.int)
            .to(vals.device)
            .index_add_(0, indices, _torch.ones(len(vals), dtype=_torch.int).to(vals.device))
        )
        v_avg = v_sum / v_numel.reshape(-1, 1)

        return v_avg

    @staticmethod
    def x_c_dist(weights: _torch.Tensor, centroids: _torch.Tensor):
        """
        Method to calculate distance between weights and centroids.
        """
        return _torch.cdist(weights, centroids).square()

    def __init__(
        self,
        n_clusters: int,
        init: str,
        n_init: int = 0,
        labels=None,
        verbose: int = 0,
        max_iter: int = 100,
        tol: float = 0.0001,
        error_bnd: int = 0,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.labels_ = labels
        self.inertia_ = None
        self.cluster_centers_ = init
        self.error_bnd = error_bnd

        assert self.max_iter > 0
        assert self.n_clusters > 0

    def kmeans_pp(self, n_init: str, X: _torch.Tensor, random_state: int, offset: int = 0):
        """
        In-house implementation of kmeans that runs entirely on GPU and is ~10x faster.
        """
        assert (
            len(X) >= self.n_clusters
        ), f"Weight fewer points than the number of clusters: {len(X)} vs. {self.n_clusters}"

        S = X[offset:]

        self.inertia_ = None

        width = (len(S) - 1) // (random_state + 1)

        for i in range(n_init):
            idx = int(i / n_init * width)
            C = S[idx].unsqueeze(0)

            for j in range(len(C), self.n_clusters):
                min_error, labels = self.__class__.x_c_dist(S, C).min(dim=-1)

                while True:
                    max_dist_idx = _torch.argmax(min_error)
                    assert min_error[max_dist_idx] >= 0, "Cannot find a next candidate"

                    candidate_C = S[max_dist_idx]
                    if candidate_C in set(C):
                        _dist[max_dist_idx] = -1
                    else:
                        C = _torch.vstack((C, candidate_C))
                        break

            if len(set(C)) != self.n_clusters:
                return self.kmeans_pp(n_init, X, random_state, offset + 1)

            min_error, labels = self.__class__.x_c_dist(X, C).min(dim=-1)
            cur_cost = min_error.sum()

            if self.inertia_ is None or self.inertia_ > cur_cost:
                self.inertia_ = cur_cost
                self.cluster_centers_ = C
                self.labels_ = labels

    def cost(self, i: int, j: int, new_cluster_cost: float):
        if i > j:
            cur_cost = 0
        else:
            size = j - i + 1
            sum_i_j = self.prefix_x[j] - (self.prefix_x[i - 1] if i >= 1 else 0)
            sum2_i_j = self.prefix_x2[j] - (self.prefix_x2[i - 1] if i >= 1 else 0)
            mean_i_j = sum_i_j / size
            cc_i_j = -mean_i_j * mean_i_j * size + sum2_i_j

            if cc_i_j < 0:
                cc_i_j = 0

            cur_cost = cc_i_j * (1 - self.tol) + new_cluster_cost * self.tol

        return cur_cost

    def backtrace(self, P, T, i, m):
        if m >= 0:
            P = [m] + P

        if m == 0:
            return P

        return self.backtrace(P, T, i - 1, T[i - 1][m - 1])

    def fit(self, X: _torch.Tensor):
        """
        Method to run kmeans operation.
        """
        N = len(X)
        if isinstance(self.cluster_centers_, str):
            if "kmeans++" in self.cluster_centers_:

                if _dist.is_available() and _dist.is_initialized():
                    world_size = _dist.get_world_size()
                    rank = _dist.get_rank()
                else:
                    world_size = 1
                    rank = 0

                if "cpu" in self.cluster_centers_:
                    import sklearn.cluster

                    kmeans = sklearn.cluster.KMeans(
                        n_init=max(10, self.n_init // world_size),
                        n_clusters=self.n_clusters,
                        max_iter=self.max_iter,
                        random_state=rank + 1,
                        verbose=0,
                        tol=self.tol,
                    ).fit(X.cpu().numpy())
                    self.inertia_ = _torch.Tensor([kmeans.inertia_]).to(X.device)
                    self.labels_ = _torch.from_numpy(kmeans.labels_).to(_torch.int).to(X.device)
                    self.cluster_centers_ = None
                else:
                    self.kmeans_pp(self.n_init, X, rank + 1)

                self.fit(X)

                bcast_rank = self.get_best_rank(self.inertia_, _torch.argmin)
                if bcast_rank is not None:
                    _dist.broadcast(self.cluster_centers_, bcast_rank)
                    _dist.broadcast(self.labels_, bcast_rank)

                return self

            elif self.cluster_centers_ == "opt1d":
                nX, sort_order = _torch.sort(X, dim=0)
                nX = nX.cpu().numpy()
                rN = range(N)

                self.prefix_x = _np.cumsum(nX)
                self.prefix_x2 = _np.cumsum(_np.square(nX))

                new_cluster_cost = 0  # 2 * self.cost(0, N - 1, 0)

                num_D = self.n_clusters if self.verbose >= 2 else 2

                D = _np.full((num_D, N), _np.inf)
                D[0] = [self.cost(0, m, new_cluster_cost) for m in rN]
                T = _np.full((self.n_clusters, N), -1, dtype=int)
                T[0] = [0 for m in rN]

                opt_t_cost = D[0][-1]
                opt_n_clusters = 0
                for c in range(1, self.n_clusters):
                    if True:

                        def lookup(m, j):
                            return -(
                                D[(c - 1) % num_D][min(j - 1, m)]
                                + self.cost(j, m, new_cluster_cost)
                            )

                        R = self.smawk(rN, rN, lookup)

                        for k, v in R.items():
                            D[c % num_D][k] = -lookup(k, v)
                            T[c][k] = v
                    else:
                        for m in range(1, N):
                            for j in range(m):
                                cur_cost = D[(c - 1) % num_D][j] + self.cost(
                                    j + 1, m, new_cluster_cost
                                )
                                if cur_cost < D[c % num_D][m]:
                                    D[c % num_D][m] = cur_cost
                                    T[c][m] = j + 1

                    if opt_t_cost > D[c % num_D][-1]:
                        opt_t_cost = D[c % num_D][-1]
                        opt_n_clusters = c

                P = []
                P = self.backtrace(P, T, opt_n_clusters, T[opt_n_clusters][-1])
                P.append(N)

                self.labels_ = []
                self.cluster_centers_ = []
                for i in range(len(P) - 1):
                    v = nX[P[i] : P[i + 1]]
                    if len(v):
                        self.labels_ += [len(self.cluster_centers_)] * len(v)
                        self.cluster_centers_.append([_np.mean(v)])

                self.n_clusters = len(self.cluster_centers_)
                self.cluster_centers_ = _torch.from_numpy(_np.array(self.cluster_centers_)).to(
                    device=X.device, dtype=X.dtype
                )
                min_error, self.labels_ = self.__class__.x_c_dist(X, self.cluster_centers_).min(
                    dim=-1
                )
                self.inertia_ = min_error.sum()

        else:
            self.inertia_ = None

            for i in range(self.max_iter):

                self.cluster_centers_ = self.__class__.get_cluster_avg(
                    self.n_clusters, self.labels_, X
                )

                nan_centers = self.cluster_centers_.isnan()
                if nan_centers.any():
                    self.kmeans_pp(self.n_init, X, i)
                    continue

                self.x_c_dist = self.__class__.x_c_dist(X, self.cluster_centers_)
                min_error, self.labels_ = self.x_c_dist.min(dim=-1)
                cur_inertia = min_error.sum()

                if self.error_bnd and _torch.sqrt(cur_inertia / N) < self.error_bnd:
                    unique, counts = _torch.unique(self.labels_, return_counts=True)
                    idx = unique[counts.argmin()]

                    reduce_cluster_centers_ = self.cluster_centers_.clone()
                    reduce_cluster_centers_[idx] = _np.nan

                    reduce_cluster_centers_ = reduce_cluster_centers_[
                        ~_torch.isnan(reduce_cluster_centers_)
                    ].view(-1, 1)
                    reduce_min_error, reduce_labels_ = self.__class__.x_c_dist(
                        X, reduce_cluster_centers_
                    ).min(dim=-1)
                    reduce_inertia = reduce_cluster_centers_.sum()
                    self.rmse_error = _torch.sqrt(reduce_inertia / N)

                    if self.rmse_error < self.error_bnd:
                        self.cluster_centers_ = reduce_cluster_centers_
                        self.labels_ = reduce_labels_
                        self.n_clusters = len(self.cluster_centers_)
                        continue

                if self.inertia_ is None or abs(self.inertia_ - cur_inertia) > self.tol:
                    self.inertia_ = cur_inertia
                else:
                    self.inertia_ = cur_inertia
                    break

        return self

    def get_best_rank(self, metric, func=_torch.argmin):
        if _dist.is_available() and _dist.is_initialized():
            world_size = _dist.get_world_size()
            if world_size > 1:
                tensor_list = [_torch.zeros_like(metric) for _ in range(world_size)]
                _dist.all_gather(tensor_list, metric)
                bcast_rank = func(_torch.Tensor(tensor_list))

                return bcast_rank

        return None

    def rmse_error(self, a, b):
        return _torch.sqrt(_torch.mean(_torch.square(a - b)))

    def smawk(self, rows, cols, lookup):
        """Search for row-maxima in a 2d totally monotone matrix M[i,j].
        The input is specified by a list of row indices, a list of column
        indices, and a function "lookup" satisfying lookup(i,j) = M[i,j].
        The matrix must satisfy the totally monotone ordering property:
        if i occurs before i' in rows, j occurs before j' in cols, and
        M[i,j] < M[i,j'], then also M[i',j] < M[i',j'].  The result is
        returned as a dictionary mapping row i to the column j containing
        the largest value M[i,j].  Ties are broken in favor of earlier
        columns.  The number of calls to lookup is O(len(rows)+len(cols))."""

        # base case of recursion
        if not rows:
            return {}

        # reduce phase: make number of columns at most equal to number of rows
        stack = []
        for c in cols:
            while len(stack) >= 1 and lookup(rows[len(stack) - 1], stack[-1]) < lookup(
                rows[len(stack) - 1], c
            ):
                stack.pop()
            if len(stack) != len(rows):
                stack.append(c)

        cols = stack

        # recursive call to search for every odd row
        result = self.smawk([rows[i] for i in range(1, len(rows), 2)], cols, lookup)

        # go back and fill in the even rows
        c = 0
        for r in range(0, len(rows), 2):
            row = rows[r]
            if r == len(rows) - 1:
                cc = len(cols) - 1  # if r is last row, search through last col
            else:
                cc = c  # otherwise only until pos of max in row r+1
                target = result[rows[r + 1]]
                while cols[cc] != target:
                    cc += 1
            result[row] = max([(lookup(row, cols[x]), -x, cols[x]) for x in range(c, cc + 1)])[2]
            c = cc

        return result
