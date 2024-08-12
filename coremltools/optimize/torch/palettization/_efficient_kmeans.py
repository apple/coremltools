#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Optional as _Optional
from typing import Union as _Union

import numpy as _np
import torch as _torch
import torch.distributed as _dist


class _EfficientKMeans:
    """
    An implementation of k-means which runs entirely on GPU.
    """

    def __init__(
        self,
        n_clusters: int,
        init: _Union[str, _torch.Tensor],
        n_init: int = 0,
        labels=None,
        max_iter: int = 100,
        tol: float = 0.0001,
        error_bnd: float = 0.0,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = labels
        self.inertia_ = None
        self.cluster_centers_ = init
        self.error_bnd = error_bnd

        assert self.max_iter > 0
        assert self.n_clusters > 0

    @staticmethod
    def _get_cluster_avg(
        n_clusters: int,
        indices: _torch.Tensor,
        vals: _torch.Tensor,
        sample_weight: _Optional[_torch.Tensor] = None,
    ) -> _torch.Tensor:
        agg_vals = (
            vals.float() * sample_weight.float() if sample_weight is not None else vals.float()
        )
        v_sum = (
            _torch.zeros([n_clusters] + list(vals[0].size()))
            .to(vals.device)
            .index_add_(0, indices, agg_vals)
        )
        weight = (
            _torch.ones(len(vals), dtype=_torch.int).to(vals.device)
            if sample_weight is None
            else sample_weight.squeeze(1).to(vals.device)
        )
        v_numel = (
            _torch.zeros(n_clusters, dtype=weight.dtype)
            .to(vals.device)
            .index_add_(0, indices, weight)
        )
        v_numel[v_numel == 0] = 1

        v_avg = v_sum / v_numel.reshape(-1, 1)

        return v_avg.to(vals.dtype)

    @staticmethod
    def x_c_dist(params: _torch.Tensor, clusters: _torch.Tensor) -> _torch.Tensor:
        """
        Calculate the distance between weights and clusters.
        """
        clusters = clusters.contiguous()

        if _torch.finfo(params.dtype).bits > _torch.finfo(clusters.dtype).bits:
            return _torch.cdist(params.to(clusters.dtype), clusters).square()
        else:
            return _torch.cdist(params, clusters.to(params.dtype)).square()

    def _kmeans_pp(
        self, parameters: _torch.Tensor, sample_weight: _Optional[_torch.Tensor] = None
    ) -> "_EfficientKMeans":
        assert len(parameters) >= self.n_clusters

        self.inertia_ = int(1e9)

        for n in range(self.n_init):
            centroids = _torch.zeros(
                (self.n_clusters, parameters.size(-1)),
                device=parameters.device,
                dtype=parameters.dtype,
            )
            for i in range(self.n_clusters):
                if i == 0:
                    centroids[i] = parameters[_torch.randint(0, len(parameters), [1])]
                    d_ij_curr = _torch.cdist(centroids[:i], parameters)
                else:
                    d_ij_prev = _torch.cdist(centroids[i - 1 : i], parameters)
                    d_ij_prev[d_ij_prev == 0] = -int(1e9)

                    d_ij_curr = _torch.cat((d_ij_curr, d_ij_prev), dim=0)

                    c_to_x = _torch.min(d_ij_curr, dim=0)
                    centroids[i] = parameters[c_to_x[0].argmax()]

            for i in range(self.max_iter):
                min_error, labels = _torch.cdist(parameters, centroids).min(dim=-1)

                # if W is None:
                centroids.zero_()
                agg_params = parameters * sample_weight if sample_weight is not None else parameters
                weights = sample_weight.squeeze(1) if sample_weight is not None else None
                centroids.scatter_add_(
                    0,
                    labels.view(-1, 1).expand([-1, parameters.size(-1)]),
                    agg_params,
                )
                n_centroids = _torch.bincount(
                    labels, weights=weights, minlength=self.n_clusters
                ).view(-1, 1)

                centroids /= n_centroids
                cur_inertia = min_error.square().sum()

                if cur_inertia < self.inertia_:
                    exit = self.inertia_ <= cur_inertia * (1 + self.tol)
                    self.inertia_ = cur_inertia
                    self.labels_ = labels
                    self.cluster_centers_ = centroids
                    if exit:
                        break

        return self

    def fit(
        self, X: _torch.Tensor, sample_weight: _Optional[_torch.Tensor] = None
    ) -> "_EfficientKMeans":
        """
        Compute k-means clustering.
        """
        N = len(X)

        assert N >= self.n_clusters, f"too many clusters {self.n_clusters} for {N} samples"

        if isinstance(self.cluster_centers_, str):
            if "kmeans++" in self.cluster_centers_:
                if _dist.is_available() and _dist.is_initialized():
                    rank = _dist.get_rank()
                else:
                    rank = 0

                if "cpu" in self.cluster_centers_:
                    import sklearn.cluster

                    if "minibatch" in self.cluster_centers_:
                        clustering_method = sklearn.cluster.MiniBatchKMeans
                    else:
                        clustering_method = sklearn.cluster.KMeans

                    kmeans = clustering_method(
                        n_init=self.n_init,
                        n_clusters=self.n_clusters,
                        max_iter=self.max_iter,
                        random_state=rank + 1,
                        tol=self.tol,
                    ).fit(X.float().cpu().numpy(), sample_weight=sample_weight)
                    self.inertia_ = _torch.Tensor([kmeans.inertia_]).to(X.device)
                    self.labels_ = _torch.from_numpy(kmeans.labels_).int().to(X.device)
                    self.cluster_centers_ = None
                else:
                    self._kmeans_pp(X.float(), sample_weight=sample_weight)

                self.cluster_centers_ = _EfficientKMeans._get_cluster_avg(
                    self.n_clusters, self.labels_, X, sample_weight=sample_weight
                )

            elif self.cluster_centers_ == "opt1d":
                from coremltools._deps import _kmeans1d

                self.labels_, self.cluster_centers_ = _kmeans1d.cluster(
                    X, self.n_clusters, weights=sample_weight
                )

                self.n_clusters = len(self.cluster_centers_)
                self.cluster_centers_ = (
                    _torch.Tensor(self.cluster_centers_)
                    .to(device=X.device, dtype=X.dtype)
                    .view(-1, 1)
                )
                self.labels_ = _torch.Tensor(self.labels_).int().to(X.device)

                min_error, _ = _EfficientKMeans.x_c_dist(X, self.cluster_centers_).min(dim=-1)
                self.inertia_ = min_error.sum()
        else:
            self.inertia_ = None

            for i in range(self.max_iter):
                self.cluster_centers_ = _EfficientKMeans._get_cluster_avg(
                    self.n_clusters, self.labels_, X, sample_weight=sample_weight
                )

                # remove empty clusters perhaps due to pruning
                nan_centers = self.cluster_centers_.isnan()
                if nan_centers.any():
                    self._kmeans_pp(X, sample_weight=sample_weight)
                    continue

                x_c_dist = _EfficientKMeans.x_c_dist(X, self.cluster_centers_)
                min_error, self.labels_ = x_c_dist.min(dim=-1)
                cur_inertia = min_error.sum()

                if self.error_bnd and _torch.sqrt(cur_inertia / N) < self.error_bnd:
                    unique, counts = _torch.unique(self.labels_, return_counts=True)
                    idx = unique[counts.argmin()]

                    reduce_cluster_centers_ = self.cluster_centers_.clone()
                    reduce_cluster_centers_[idx] = _np.nan

                    reduce_cluster_centers_ = reduce_cluster_centers_[
                        ~_torch.isnan(reduce_cluster_centers_)
                    ].view(-1, 1)
                    reduce_min_error, reduce_labels_ = _EfficientKMeans.x_c_dist(
                        X, reduce_cluster_centers_
                    ).min(dim=-1)
                    reduce_inertia = reduce_cluster_centers_.sum()
                    rmse_error = _torch.sqrt(reduce_inertia / N)

                    if rmse_error < self.error_bnd:
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
