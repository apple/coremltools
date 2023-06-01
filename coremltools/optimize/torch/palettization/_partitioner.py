#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Tuple as _Tuple

import torch as _torch

from ._efficient_kmeans import _EfficientKMeans


class _Partitioner:
    """
    Internal class that manages partitioning. The ``FakePalettize`` class base classes the ``_Partitioner`` class
    and all the partitioning logic is controlled by this class.
    """

    def __init__(
        self,
        n_bits: int,
        enforce_zero: bool,
        partition_size: int,
        cluster_dim: int,
        cluster_permute: _Tuple,
        palett_tau: float,
        kmeans_init: str,
        prune_threshold: float,
        kmeans_opt1d_threshold: int,
        add_extra_centroid: bool,
    ):
        self.centroids_init = [kmeans_init]
        if add_extra_centroid:
            self.n_clusters = [2 ** int(n_bits) + 1]
        else:
            self.n_clusters = [2 ** int(n_bits)]
        self.labels_init = [None]
        self.enforce_zero = [enforce_zero]
        self.partitions = []
        self.partition_size = partition_size
        self.cluster_dim = cluster_dim
        self.cluster_permute = cluster_permute
        self.prune_threshold = float(prune_threshold)

        self.kmeans_init = kmeans_init
        self.kmeans_opt1d_threshold = kmeans_opt1d_threshold
        self.palett_tau = palett_tau

    def create_partitions(self, weights: _torch.Tensor):
        """
        Method to create partitions in the weights. These partitions can be used to run channel level palettization.
        """
        num_channels = len(weights)
        numel_per_channel = _torch.numel(weights[0])
        num_channels_per_partition = min(
            num_channels, max(1, int(self.partition_size / numel_per_channel))
        )

        self.partitions = [
            (n, min(n + num_channels_per_partition, num_channels))
            for n in range(0, num_channels, num_channels_per_partition)
        ]
        num_partitions = len(self.partitions)

        if self.centroids_init[0] == "auto":
            # if auto then pick either init method
            numel_per_partition = numel_per_channel * num_channels_per_partition
            self.centroids_init[0] = (
                "opt1d"
                if (
                    numel_per_partition <= self.n_clusters[0]
                    or numel_per_partition <= self.kmeans_opt1d_threshold
                )
                and self.cluster_dim == 1
                else "cpu.kmeans++"
            )

        self.centroids_init = self.centroids_init * num_partitions
        self.n_clusters = self.n_clusters * num_partitions
        self.labels_init = self.labels_init * num_partitions
        self.enforce_zero = self.enforce_zero * num_partitions

        assert (
            num_channels_per_partition * numel_per_channel
            >= min(self.n_clusters) * self.cluster_dim
        ), f"The number of clusters ({self.n_clusters}) and/or the cluster dim ({self.cluster_dim}) is TOO big"

    def get_partition_kmeans(
        self, weights: _torch.Tensor, partition_index: int, partition: int, max_iter: int, init: str
    ):
        """
        Method to get kmeans for a particular partition.
        """
        Y = weights[partition[0] : partition[1]].detach()
        cY, pad = self.flatten(Y)

        kmeans = _EfficientKMeans(
            n_clusters=self.n_clusters[partition_index],
            init=init,
            labels=self.labels_init[partition_index],
            n_init=10,
            max_iter=max_iter,
        ).fit(cY)

        if self.enforce_zero[partition_index]:
            zero_point = (
                _torch.zeros(kmeans.cluster_centers_[0].size())
                .to(kmeans.cluster_centers_.device)
                .unsqueeze(0)
            )
            zero_idx = _torch.argmin(_torch.cdist(kmeans.cluster_centers_, zero_point))
            kmeans.cluster_centers_[zero_idx] = zero_point

        weights[partition[0] : partition[1]] = self.deflatten(
            kmeans.cluster_centers_[kmeans.labels_], Y.size(), pad
        )

        return kmeans

    def init_partitions(self, weights: _torch.Tensor):
        """
        Method to initialize the partitions and set the k-means. Called during first iteration of palettization in the
        forward method of ``FakePalettize``.
        """
        with _torch.no_grad():
            self.create_partitions(weights)
            for i, partition in enumerate(self.partitions):
                kmeans = self.get_partition_kmeans(
                    weights.clone(), i, partition, max_iter=100, init=self.centroids_init[i]
                )

                self.centroids_init[i] = kmeans.cluster_centers_
                self.labels_init[i] = kmeans.labels_
                self.n_clusters[i] = kmeans.n_clusters

    def flatten(self, weight_partition: _torch.Tensor):
        """
        Method to flatten a particular weight partition.
        """
        permute = self.cluster_permute
        dim = self.cluster_dim

        if permute and len(permute) == len(weight_partition.size()):
            weight_partition = weight_partition.permute(permute)

        num_misalignment = _torch.numel(weight_partition) % dim

        pad = None
        if num_misalignment:
            weight_partition = weight_partition.flatten()
            pad = weight_partition[-num_misalignment:]
            weight_partition = weight_partition[:-num_misalignment]

        return weight_partition.reshape(-1, dim), pad

    def deflatten(self, weight_partition: _torch.Tensor, target_size: _Tuple, pad: _torch.Tensor):
        """
        Method to deflatten a particular weight partition.
        """
        permute = self.cluster_permute

        if pad is not None:
            weight_partition = _torch.cat([weight_partition.flatten(), pad])

        if permute and len(permute) == len(target_size):
            cur_shape = [target_size[i] for i in permute]

            weight_partition = weight_partition.reshape(cur_shape)
            weight_partition = weight_partition.permute(
                _torch.argsort(_torch.Tensor(permute)).tolist()
            )
            assert weight_partition.size() == target_size

        return weight_partition.reshape(target_size)

    # Do not use _load_from_state_dict as this class doesn't call super
    # So it makes multiple inheritance easier to apprehend in child classes
    def _load_from_state_dict_(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        self.cluster_permute = state_dict.pop(prefix + "permute")
        self.partitions = state_dict.pop(prefix + "partitions")

    # Do not use _save_to_state_dict as this class doesn't call super
    # So it makes multiple inheritance easier to apprehend in child classes
    def _save_to_state_dict_(self, destination, prefix, keep_vars):
        destination[prefix + "permute"] = self.cluster_permute
        destination[prefix + "partitions"] = self.partitions
