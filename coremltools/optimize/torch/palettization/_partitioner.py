#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import math as _math
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import torch as _torch
import torch.distributed as _dist

from ._efficient_kmeans import _EfficientKMeans
from ._utils import get_shard_list as _get_shard_list
from ._utils import vectorize as _vectorize

# NF Cluster sizes for which partitioning has been verified.
NF_CLUSTER_SIZES = [8, 9, 16, 17]
class _Partitioner:
    """
    Internal class that manages partitioning. The ``FakePalettize`` class base classes the ``_Partitioner`` class
    and all the partitioning logic is controlled by this class.
    """

    def __init__(
        self,
        n_bits: int,
        enforce_zero: bool,
        prune_threshold: float,
        cluster_dim: int,
        cluster_permute: _Tuple,
        group_size: _Optional[int],
        palett_tau: float,
        kmeans_init: str,
        percentage_palett_enable: float,
        kmeans_opt1d_threshold: int,
        kmeans_batch_threshold: int,
        kmeans_n_init: int,
        kmeans_error_bnd: float,
    ):
        self.centroids = [kmeans_init]
        self.n_clusters = 2 ** int(n_bits)
        self.labels = [None]
        self.enforce_zero = [enforce_zero]
        self.enable_partition = []
        self.proj_factor = None
        self.partitions = []
        self.cum_inertia = []
        self.cluster_dim = cluster_dim
        self.cluster_permute = cluster_permute
        self.prune_threshold = prune_threshold
        self.palett_tau = palett_tau  # rename to palett_tau
        self.group_size = group_size
        self.percentage_palett_enable = percentage_palett_enable
        self.kmeans_opt1d_threshold = kmeans_opt1d_threshold
        self.kmeans_batch_threshold = kmeans_batch_threshold
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_error_bnd = kmeans_error_bnd

    def create_partitions(self, weights) -> None:
        """
        Method to create partitions in the weights. These partitions can be used to run channel level palettization.
        """
        with _torch.no_grad():
            num_channels = len(weights)
            usr_num_channels_per_partition = (
                int(self.group_size) if self.group_size else num_channels
            )
            self.partitions = [
                list(range(i, min(num_channels, i + usr_num_channels_per_partition)))
                for i in range(0, num_channels, usr_num_channels_per_partition)
            ]
            num_partitions = len(self.partitions)
            self.centroids = self.centroids * num_partitions
            self.labels = self.labels * num_partitions
            self.enforce_zero = self.enforce_zero * num_partitions
            self.cum_inertia = [1e9] * num_partitions
            self.partition_numel = _torch.tensor(
                [_torch.numel(weights[p]) for p in self.partitions]
            )
            self.enable_partition = [True] * max(
                1, int(self.percentage_palett_enable * num_partitions)
            )
            self.enable_partition += [False] * (num_partitions - len(self.enable_partition))
            numel_per_partition = max(self.partition_numel)
            assert numel_per_partition
            assert (
                numel_per_partition >= self.n_clusters * self.cluster_dim
            ), f"The number of clusters ({self.n_clusters}) and/or the cluster dim ({self.cluster_dim}) is TOO big"

    def get_partition_kmeans(
        self,
        X,
        partition,
        n_clusters,
        labels,
        enforce_zero,
        max_iter,
        init,
        n_init=10,
    ) -> _EfficientKMeans:
        """
        Method to get kmeans for a particular partition.
        """
        cY, pad = _vectorize(X[partition], self.cluster_dim)

        kmeans = _EfficientKMeans(
            n_clusters=n_clusters,
            init=init,
            labels=labels,
            n_init=n_init,
            max_iter=max_iter,
            error_bnd=self.kmeans_error_bnd,
        ).fit(cY)

        if enforce_zero:
            # fix zero
            zero_point = _torch.zeros_like(kmeans.cluster_centers_[0]).unsqueeze(0)
            zero_idx = _torch.argmin(
                _torch.cdist(kmeans.cluster_centers_.float(), zero_point.float())
            )

            # always put zero in the first
            temp = kmeans.cluster_centers_[0]
            kmeans.cluster_centers_[zero_idx] = temp
            kmeans.cluster_centers_[0] = zero_point

        return kmeans

    def init_partitions(self, parameters) -> None:
        """
        Method to initialize the partitions and set the k-means. Called during first iteration of palettization in the
        forward method of ``FakePalettize``.
        """
        if isinstance(self.centroids[0], _torch.Tensor):
            return
        with _torch.no_grad():
            num_partitions = len(self.partitions)
            numel_per_partition = max(self.partition_numel)
            if "nf" in self.centroids[0]:
                if self.n_clusters in NF_CLUSTER_SIZES and self.cluster_dim == 1:
                    nf_fit = "fit" in self.centroids[0]
                    for i, partition in enumerate(self.partitions):
                        bit = int(_math.log2(self.n_clusters))
                        sparse = bool(_math.log2(self.n_clusters) - bit)

                        self.centroids[i] = (
                            _generate_natural_float(bit=bit, sparse=sparse)
                            .to(parameters.device)
                            .to(parameters.dtype)
                            .view(-1, 1)
                        )

                        if nf_fit:
                            best_err = _torch.finfo(_torch.float).max
                            best_lambd = 1
                            best_retry = 0
                            best_thold = 10
                            up_down_hill = 0
                            lambd_list = [[1 + x / 100, 1 - x / 100] for x in range(99)]
                            lambd_list = [1] + [v for sublist in lambd_list for v in sublist]

                            cur_X = parameters[self.partitions[i]].view(-1, 1)

                            for cur_lambd in lambd_list:
                                if up_down_hill > best_thold and cur_lambd < 1:
                                    continue

                                if up_down_hill < -best_thold and cur_lambd > 1:
                                    continue

                                cur_lut = _torch.stack(
                                    [x.sign() * x.abs() ** (cur_lambd) for x in self.centroids[i]]
                                )
                                x_c_dist = _torch.cdist(cur_X, cur_lut.to(cur_X.dtype)).square()
                                cur_err = x_c_dist.min(-1).values.float().sum()

                                if best_err > cur_err:
                                    best_retry = 0
                                    best_err = cur_err
                                    best_lambd = cur_lambd
                                    if best_lambd > 1:
                                        up_down_hill += 1
                                    else:
                                        up_down_hill -= 1

                                elif best_retry > best_thold:
                                    break
                                else:
                                    best_retry += 1

                            self.centroids[i] = _torch.stack(
                                [x.sign() * x.abs() ** (best_lambd) for x in self.centroids[i]]
                            )
                    return

                self.centroids = ["auto"] * num_partitions

            for i in range(num_partitions):
                if self.centroids[i] == "auto":
                    # if auto then pick either init method
                    self.centroids[i] = (
                        "opt1d"
                        if (
                            numel_per_partition <= self.n_clusters
                            or numel_per_partition <= self.kmeans_opt1d_threshold
                        )
                        and self.cluster_dim == 1
                        else "kmeans++"
                    )

            if _dist.is_available() and _dist.is_initialized():
                distributed_world_size = _dist.get_world_size()
            else:
                distributed_world_size = 1
            if max(num_partitions, distributed_world_size) < self.kmeans_batch_threshold:
                for i, partition in enumerate(self.partitions):
                    kmeans = self.get_partition_kmeans(
                        parameters,
                        partition,
                        self.n_clusters,
                        self.labels[i],
                        self.enforce_zero[i],
                        max_iter=100,
                        init=self.centroids[i],
                        n_init=max(1, self.kmeans_n_init // distributed_world_size),
                    )
                    bcast_rank = _get_best_rank(kmeans.inertia_, _torch.argmin)
                    if bcast_rank:
                        _dist.broadcast(kmeans.cluster_centers_, bcast_rank)

                    self.centroids[i] = kmeans.cluster_centers_
                    self.labels[i] = None
            else:
                shard_list = _get_shard_list(num_partitions)
                centroids_list = [None] * distributed_world_size

                for i in range(distributed_world_size):
                    begin, end = shard_list[i], shard_list[i + 1]
                    current_rank = (
                        _dist.get_rank() if _dist.is_available() and _dist.is_initialized() else 0
                    )
                    if i == current_rank and begin < end:
                        for p in range(begin, end):
                            kmeans = self.get_partition_kmeans(
                                parameters,
                                self.partitions[p],
                                self.n_clusters,
                                self.labels[p],
                                self.enforce_zero[p],
                                max_iter=100,
                                init=self.centroids[p],
                                n_init=self.kmeans_n_init,
                            )
                            self.centroids[p] = kmeans.cluster_centers_

                        centroids_list[i] = _torch.stack(self.centroids[begin:end])
                    else:
                        centroids_list[i] = _torch.full(
                            [end - begin, self.n_clusters, self.cluster_dim],
                            float("nan"),
                            dtype=parameters.dtype,
                            device=parameters.device,
                        )

                if _dist.is_available() and _dist.is_initialized():
                    _dist.all_gather(centroids_list, centroids_list[_dist.get_rank()])
                centroids_list = [v for sublist in centroids_list for v in sublist]

                assert len(centroids_list) == num_partitions
                for p in range(num_partitions):
                    self.labels[p] = None
                    self.centroids[p] = centroids_list[p]

    def _load_from_state_dict_(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.cluster_permute = state_dict.pop(prefix + "permute")
        self.partitions = state_dict.pop(prefix + "partitions")
        self.centroids = state_dict.pop(prefix + "centroids")
        self.labels = state_dict.pop(prefix + "labels")
        self.proj_factor = state_dict.pop(prefix + "proj_factor")

    def _save_to_state_dict_(self, destination, prefix, keep_vars):
        destination[prefix + "centroids"] = self.centroids
        destination[prefix + "labels"] = self.labels
        destination[prefix + "permute"] = self.cluster_permute
        destination[prefix + "partitions"] = self.partitions


def _get_best_rank(metric, func=_torch.argmin) -> int:
    """
    Get best rank of a particular metric according to a specified function.
    """
    if _dist.is_available() and _dist.is_initialized():
        distributed_world_size = _dist.get_world_size()
        if distributed_world_size > 1:
            tensor_list = [_torch.zeros_like(metric) for _ in range(distributed_world_size)]
            _dist.all_gather(tensor_list, metric)
            bcast_rank = func(_torch.Tensor(tensor_list))

            return bcast_rank

    return None


def _generate_natural_float(bit=4, sparse=False, offset=0.9677083) -> _torch.Tensor:
    """
    Function to generate NF4 values.
    """
    from scipy.stats import norm

    space = (2**bit) // 2
    # one more positive value, this is an asymmetric type
    v1 = norm.ppf(_torch.linspace(offset, 0.5, space + 1)[:-1]).tolist()

    if sparse:
        v3 = [-x for x in v1]
    else:
        v3 = (-norm.ppf(_torch.linspace(offset, 0.5, space)[:-1])).tolist()

    v = [0] + v3 + list(reversed(v1))

    values = _torch.Tensor(v)
    values /= values.max()

    return values
