#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import contextlib
import logging as _logging
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

import torch as _torch
import torch.distributed as _dist
import torch.nn.functional as _F
from torch.ao.quantization.observer import ObserverBase as _ObserverBase
from torch.quantization import FakeQuantize as _FakeQuantize

from ._efficient_kmeans import _EfficientKMeans
from ._fake_palettizer_tensor_hook import _FakePalettizerTensorHook
from ._partitioner import _Partitioner
from ._utils import devectorize as _devectorize
from ._utils import get_shard_list as _get_shard_list
from ._utils import vectorize as _vectorize
from .palettization_config import DEFAULT_PALETTIZATION_ADVANCED_OPTIONS


_logger = _logging.getLogger(__name__)

class FakePalettize(_FakeQuantize, _Partitioner):
    """
    A class that implements palettization algorithm described in
    `DKM: Differentiable K-Means Clustering Layer for Neural Network Compression
    <https://arxiv.org/abs/2108.12659>`_. It clusters the weights
    using a differentiable version of ``k-means``, allowing the look-up-table (LUT)
    and indices of palettized weights to be learnt using a gradient-based optimization
    algorithm such as SGD.

    Extends :py:class:`torch.quantization.FakeQuantize` to add support for
    palettization.

    Example:
            .. code-block:: python

                from collections import OrderedDict
                import torch
                import torch.nn as nn
                import coremltools.optimize.torch.palettization as palett

                model = nn.Sequential(
                    OrderedDict(
                        [
                            ("linear1", nn.Linear(4, 5)),
                            ("sigmoid1", nn.Sigmoid()),
                            ("linear2", nn.Linear(5, 4)),
                            ("sigmoid2", nn.Sigmoid),
                        ]
                    )
                )

                fq_activation = nn.Identity
                fq_weight = palett.FakePalettize.with_args(
                    observer=torch.quantization.MovingAveragePerChannelMinMaxObserver.with_args(
                        quant_min=-128, quant_max=127, dtype=torch.qint8
                    ),
                    n_bits=2,
                    cluster_dim=1,
                    module_parameter_shape=torch.Size([5, 4]),
                )
                model.linear2.qconfig = torch.quantization.QConfig(
                    activation=fq_activation, weight=fq_weight
                )

                palettized_model = palett.prepare_palettizer(model)

                train_model(palettized_model)

                palettized_converted_model = palett.finalize(palettized_model)


    Args:
        observer (:obj:`torch.ao.quantization.observer.ObserverBase`): Observer for quantizing the ``LUT``.
        n_bits (:obj:`int`): Number of palettization bits. There would be :math:`2^{n\_bits}` unique weights in the ``LUT``.
        cluster_dim (:obj:`int`): Dimensionality of centroids to use for clustering.
        enable_per_channel_scale (:obj:`bool`): When set to ``True``, per channel scaling is used along the channel dimension.
        group_size (:obj:`int`): Each group of ``group_size`` number of channels are palettized using
            different look up tables.
        quant_min (:obj:`int`): The minimum allowable quantized value.
        quant_max (:obj:`int`): The maximum allowable quantized value.
        lut_dtype (:obj:`str`): String that decides whether to quantize the ``LUT`` or not. The following are the ``str``
            LUT quantization combinations: (``u8``, ``uint8``), (``i8``, ``int8``), and (``f16``, ``float16``).
        advanced_options (:obj:`dict`): Advanced options to configure the palettization algorithm.
        observer_kwargs (optional): Arguments for the observer module.

    .. note::
        Allowed keys for ``advanced_options`` are the parameters listed as ``optional`` in
        :py:class:`ModuleDKMPalettizerConfig`, besides the ones already covered by other parameters in this class.
    """

    fake_palett_enabled: _torch.Tensor

    def __init__(
        self,
        observer: _ObserverBase,
        n_bits: int,
        cluster_dim: int,
        enable_per_channel_scale: bool = False,
        group_size: _Optional[int] = None,
        quant_min: int = -128,
        quant_max: int = 127,
        lut_dtype: str = "f32",
        advanced_options: dict = {},
        **observer_kwargs,
    ):
        cluster_permute = advanced_options.get(
            "cluster_permute", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["cluster_permute"]
        )
        palett_max_mem = advanced_options.get(
            "palett_max_mem", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_max_mem"]
        )

        palett_shard = advanced_options.get(
            "palett_shard", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_shard"]
        )
        palett_unique = advanced_options.get(
            "palett_unique", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_unique"]
        )
        palett_min_tsize = advanced_options.get(
            "palett_min_tsize",
            DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_min_tsize"],
        )
        kmeans_max_iter = advanced_options.get(
            "kmeans_max_iter", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["kmeans_max_iter"]
        )
        prune_threshold = advanced_options.get(
            "prune_threshold", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["prune_threshold"]
        )
        kmeans_init = advanced_options.get(
            "kmeans_init", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["kmeans_init"]
        )
        kmeans_opt1d_threshold = advanced_options.get(
            "kmeans_opt1d_threshold",
            DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["kmeans_opt1d_threshold"],
        )
        enforce_zero = advanced_options.get(
            "enforce_zero", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["enforce_zero"]
        )
        palett_mode = advanced_options.get(
            "palett_mode", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_mode"]
        )
        palett_cluster_tol = advanced_options.get(
            "palett_cluster_tol",
            DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_cluster_tol"],
        )
        palett_tau = advanced_options.get(
            "palett_tau", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_tau"]
        )
        palett_epsilon = advanced_options.get(
            "palett_epsilon", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_epsilon"]
        )
        palett_lambda = advanced_options.get(
            "palett_lambda", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_lambda"]
        )
        add_extra_centroid = advanced_options.get(
            "add_extra_centroid",
            DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["add_extra_centroid"],
        )
        per_channel_scaling_factor_scheme = advanced_options.get(
            "per_channel_scaling_factor_scheme",
            DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["per_channel_scaling_factor_scheme"],
        )
        percentage_palett_enable = advanced_options.get(
            "percentage_palett_enable",
            DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["percentage_palett_enable"],
        )
        kmeans_batch_threshold = advanced_options.get(
            "kmeans_batch_threshold",
            DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["kmeans_batch_threshold"],
        )
        kmeans_n_init = advanced_options.get(
            "kmeans_n_init", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["kmeans_n_init"]
        )
        zero_threshold = advanced_options.get(
            "zero_threshold", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["zero_threshold"]
        )
        palett_batch_mode = advanced_options.get(
            "palett_batch_mode",
            DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_batch_mode"],
        )
        palett_dist = advanced_options.get(
            "palett_dist",
            DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_dist"],
        )
        kmeans_error_bnd = advanced_options.get(
            "kmeans_error_bnd",
            DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["kmeans_error_bnd"],
        )
        vector_ch_axis = advanced_options.get(
            "channel_axis",
            DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["channel_axis"],
        )

        self._target_module_level_sparsity = 0.0

        _FakeQuantize.__init__(self, observer, quant_min, quant_max, **observer_kwargs)
        _Partitioner.__init__(
            self,
            n_bits,
            enforce_zero,
            prune_threshold,
            cluster_dim,
            cluster_permute,
            group_size,
            palett_tau,
            kmeans_init,
            percentage_palett_enable,
            kmeans_opt1d_threshold,
            kmeans_batch_threshold,
            kmeans_n_init,
            kmeans_error_bnd,
            vector_ch_axis,
        )

        self.cluster_permute = cluster_permute
        self.enable_per_channel_scale = enable_per_channel_scale
        self.per_channel_scaling_factor_scheme = per_channel_scaling_factor_scheme
        self.per_channel_scaling_factor = None
        self.partitions = []
        self.group_size = group_size
        self.lut_dtype = lut_dtype
        self.add_extra_centroid = add_extra_centroid
        self.need_to_quantize = self.lut_dtype in ["i8", "u8", "f16"]
        self.autograd_graph = hasattr(_torch.autograd, "graph") and palett_max_mem < 1.0
        self.palett_max_mem = palett_max_mem
        self.palett_min_tsize = palett_min_tsize
        self.palett_unique = palett_unique
        self.palett_shard = palett_shard
        self.palett_dist = palett_dist and _dist.is_available() and _dist.is_initialized()
        self.zero_threshold = zero_threshold
        self.prune_threshold = prune_threshold
        self.palett_batch_mode = palett_batch_mode
        self.palett_cluster_tol = palett_cluster_tol
        self.kmeans_max_iter = kmeans_max_iter
        self.palett_mode = palett_mode
        self.palett_tau = palett_tau
        self.palett_epsilon = palett_epsilon
        self.palett_lambda = palett_lambda
        self.n_bits = n_bits
        self.cluster_dim = cluster_dim
        self.kmeans_init = kmeans_init
        self.vector_ch_axis = vector_ch_axis
        self.register_buffer("fake_palett_enabled", _torch.tensor([0], dtype=_torch.uint8))
        self.disable_fake_quant()
        self.disable_observer()

    def reset_parameters(self) -> None:
        """
        FSDP expects reset_parameters method to initialize parameters/buffers in submodules
        FakePalettize has no nn.Parameter/nn.Buffer, so we are creating an empty method
        """
        pass

    def enable_fake_palett(self, enabled: bool = True) -> None:
        self.fake_palett_enabled[0] = 1 if enabled else 0

    def disable_fake_palett(self):
        self.enable_fake_palett(False)

    def diff_palettize(self, X) -> _torch.Tensor:
        cX, pad = list(
            zip(
                *[
                    _vectorize(X[partition], self.cluster_dim, self.vector_ch_axis)
                    for i, partition in enumerate(self.partitions)
                ]
            )
        )

        if self.training:
            with _torch.no_grad():
                if self.palett_tau > 0:
                    new_centroid_list = []
                    new_cur_n_clusters = self.n_clusters
                    for i, partition in enumerate(self.partitions):
                        if not self.enable_partition[i]:
                            continue

                        cur_clusters, cur_inverse, cur_counts = _torch.unique(
                            self.centroids[i].float(),
                            dim=0,
                            return_inverse=True,
                            return_counts=True,
                        )
                        cur_n_clusters = len(cur_clusters)
                        new_cur_n_clusters = min(new_cur_n_clusters, cur_n_clusters)

                        if cur_n_clusters < self.n_clusters * (1 - self.palett_cluster_tol):
                            for j, count in enumerate(cur_counts):
                                if count > 1:
                                    new_centroid = 0.5 * (
                                        cur_clusters[j] + cur_clusters[(j + 1) % cur_n_clusters]
                                    )
                                    self.centroids[i][cur_inverse.tolist().index(j)] = new_centroid
                                    new_centroid_list.append(new_centroid)

            batch_partitions = []
            seq_partitions = []
            disabled_partitions = []
            most_common_numel = None

            for i, numel in enumerate(self.partition_numel):
                if self.enable_partition[i]:
                    if most_common_numel is None:
                        most_common_numel = self.partition_numel[self.enable_partition].mode()[0]
                    if numel == most_common_numel:
                        batch_partitions.append(i)
                    else:
                        seq_partitions.append(i)
                elif isinstance(self.centroids[i], _torch.Tensor):
                    disabled_partitions.append(i)

            if len(batch_partitions) == 1 or not self.palett_batch_mode:
                seq_partitions += batch_partitions
                batch_partitions = []

            if batch_partitions:
                X, mean_inertia = self.diff_palettize_batch(X, cX, pad, batch_partitions)

            if seq_partitions:
                X, mean_inertia = self.diff_palettize_seq(X, cX, pad, seq_partitions)

            if disabled_partitions:
                X = self.palettize(X, cX, pad, disabled_partitions)
        else:
            X = self.palettize(X, cX, pad, partitions=range(len(self.partitions)))

        return X

    def diff_palettize_seq(
        self, X, cX, pad, partitions
    ) -> _Tuple[_torch.Tensor, _Union[_torch.Tensor, int]]:
        cur_inertia = []
        for p in partitions:
            partition = self.partitions[p]
            centroids = self.centroids[p].clone()
            if _torch.is_grad_enabled():
                assert not centroids.requires_grad

            cX_p = cX[p]
            cX_pt = cX_p.T

            last_inertia = None
            keep_sparsity = self.prune_threshold == 0 and self.enforce_zero[p]

            for j in range(self.kmeans_max_iter):
                x_c_dist = _EfficientKMeans.x_c_dist(cX_p, centroids)

                if keep_sparsity:
                    # need to be keep pruning exact, no additional weight to be pruned by being assigned to the zero
                    # centroid. the zero centroid is always centroids[0]
                    if _torch.is_nonzero(centroids[0]):
                        centroids[0] = _torch.zeros_like(centroids[0]).unsqueeze(0)

                    cX_nonzero_indices = cX_p.nonzero(as_tuple=True)[0]
                    x_c_dist[cX_nonzero_indices, :1] = 1 / self.zero_threshold

                if self.prune_threshold > 0:
                    x_c_dist[:, :1] -= self.prune_threshold

                if "dkm" in self.palett_mode:
                    attention = _F.softmax(-x_c_dist / self.palett_tau, dim=-1).clamp(
                        min=self.zero_threshold
                    )
                elif "topk" in self.palett_mode:
                    values, indices = _torch.topk(x_c_dist, k=2, dim=-1, largest=False)
                    attention_topk = _F.softmax(-values / self.palett_tau, dim=-1)
                    attention = _torch.zeros_like(x_c_dist)
                    attention[:, indices] = attention_topk
                elif "hard" in self.palett_mode:
                    col_idx = x_c_dist.min(dim=-1).indices
                    row_idx = _torch.arange(start=0, end=len(col_idx), dtype=_torch.int32).to(
                        cX_p.device
                    )
                    attention = _torch.sparse_coo_tensor(
                        _torch.vstack([row_idx, col_idx]),
                        _torch.ones_like(row_idx).to(cX_p.device),
                        x_c_dist.size(),
                        dtype=x_c_dist.dtype,
                        requires_grad=True,
                    ).to_dense()
                elif "gsm" in self.palett_mode:
                    attention = _F.gumbel_softmax(-x_c_dist / self.palett_tau, dim=-1)
                else:
                    raise ValueError(f"palett_mode: {self.palett_mode} currently not supported.")

                # attention_sum can overflow with fp16
                attention_sum = attention.sum(dim=0).view(-1, 1)
                assert not (attention_sum == 0).any()

                # matmul can overflow with fp16
                centroids = _torch.matmul(cX_pt, attention).T / attention_sum

                if self.need_to_quantize:
                    centroids = super().forward(centroids)

                if _torch.is_grad_enabled():
                    assert centroids.requires_grad

                if self.enforce_zero[p]:
                    # fix zero
                    zero_point = _torch.zeros_like(centroids[0]).unsqueeze(0)
                    centroids[0] = zero_point

                min_error, _ = x_c_dist.min(dim=-1)
                cur_inertia.append(min_error.sum())

                if last_inertia and abs(last_inertia - cur_inertia[-1]) <= self.palett_epsilon:
                    break

                last_inertia = cur_inertia[-1]

            X[partition] = _devectorize(
                _torch.matmul(attention, centroids),
                pad[p],
                X[partition].size(),
                self.cluster_dim,
                self.vector_ch_axis,
            ).to(X.dtype)

            self.labels[p] = None
            self.centroids[p] = centroids.detach().to(X.dtype)
            self.cum_inertia[p] += float(cur_inertia[-1].detach())

        return X, (_torch.stack(cur_inertia).mean() if cur_inertia else -1)

    def diff_palettize_batch(self, X, cX, pad, partitions) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        num_partitions = len(partitions)
        centroids = _torch.stack([self.centroids[i] for i in partitions]).clone()
        cX = _torch.stack([cX[i] for i in partitions])
        cXt = cX.mT
        last_inertia = None

        for j in range(self.kmeans_max_iter):
            if self.palett_dist:
                x_c_dist = dist_batch_cdist_square.apply(cX, centroids)
            else:
                x_c_dist = _EfficientKMeans.x_c_dist(cX, centroids)

            attention = _F.softmax(-x_c_dist / self.palett_tau, -1).clamp(min=self.zero_threshold)

            # attention_sum can overflow with fp16
            if _torch.is_grad_enabled():
                assert attention.requires_grad
            attention_sum = attention.sum(dim=1).view(num_partitions, -1, 1)

            centroids = _torch.matmul(cXt, attention).mT / attention_sum

            if self.need_to_quantize:
                centroids = super().forward(centroids)

            if _torch.is_grad_enabled():
                assert centroids.requires_grad
            if self.enforce_zero[0]:
                zero_point = _torch.zeros_like(centroids[0][0]).unsqueeze(0)

                for k in range(centroids.size(0)):
                    centroids[k][0] = zero_point

            if self.kmeans_max_iter <= 1 and self.percentage_palett_enable >= 1:
                cur_inertia = _torch.zeros([num_partitions], device=X.device, dtype=X.dtype)
                break
            else:
                min_error, _ = x_c_dist.min(dim=-1)
                cur_inertia = min_error.sum(dim=1)
                avg_inertia = cur_inertia.mean()

                if last_inertia and abs(last_inertia - avg_inertia) <= self.palett_epsilon:
                    break

                last_inertia = avg_inertia

        tX = _torch.matmul(attention, centroids)

        for i, p in enumerate(partitions):
            partition = self.partitions[p]
            X[partition] = _devectorize(
                tX[i],
                pad[p],
                X[partition].size(),
                self.cluster_dim,
                self.vector_ch_axis,
            ).to(X.dtype)
            self.labels[p] = None
            self.centroids[p] = centroids[i].detach().to(X.dtype)
            self.cum_inertia[p] += float(cur_inertia[i].detach())

        return X, cur_inertia

    def palettize(self, X, cX, pad, partitions) -> _torch.Tensor:
        """
        This method is run during inference time by the forward method of the ``FakePalettize`` class. It calculates the
        weight from the ``LUT`` and ``indices`` across all partitions and returns them.
        """
        batch_partitions = []
        seq_partitions = []
        most_common_numel = self.partition_numel[partitions].mode()[0]

        for p in partitions:
            if self.partition_numel[p] == most_common_numel and self.labels[p] is None:
                batch_partitions.append(p)
            else:
                seq_partitions.append(p)

        if len(batch_partitions) == 1 or not self.palett_batch_mode:
            seq_partitions += batch_partitions
            batch_partitions = []

        if seq_partitions:
            X = self.palettize_seq(X, cX, pad, seq_partitions)

        if batch_partitions:
            X = self.palettize_batch(X, cX, pad, batch_partitions)

        return X

    def palettize_seq(self, X, cX, pad, partitions) -> _torch.Tensor:
        for p in partitions:
            partition = self.partitions[p]
            labels = self.labels[p]
            centroids = self.centroids[p]
            if labels is None:
                cX_p = cX[p]

                x_c_dist = _EfficientKMeans.x_c_dist(cX_p, centroids)

                if self.prune_threshold > 0:
                    x_c_dist[:, :1] -= self.prune_threshold

                min_error, labels = x_c_dist.min(dim=-1)
                self.labels[p] = labels.cpu()

            self.labels[p] = self.labels[p].to(_torch.int)
            if X is not None:
                X[partition] = _devectorize(
                    centroids[self.labels[p]],
                    pad[p],
                    X[partition].size(),
                    self.cluster_dim,
                    self.vector_ch_axis,
                ).to(X.dtype)

        return X

    def palettize_batch(self, X, cX, pad, partitions) -> _torch.Tensor:
        # intentionally use cat instead of stack to make the backward graph distinguishable from diff_palettize_batch
        cX = _torch.cat([cX[i] for i in partitions]).view(len(partitions), -1, self.cluster_dim)
        centroids = _torch.stack([self.centroids[i] for i in partitions])
        x_c_dist = _EfficientKMeans.x_c_dist(cX, centroids)

        if self.prune_threshold > 0:
            x_c_dist[:, :, :1] -= self.prune_threshold

        min_error, labels = x_c_dist.min(dim=-1)

        for i, p in enumerate(partitions):
            partition = self.partitions[p]
            centroids = self.centroids[p]
            self.labels[p] = labels[i].to(_torch.int).cpu()

            X[partition] = _devectorize(
                centroids[self.labels[p]],
                pad[p],
                X[partition].size(),
                self.cluster_dim,
                self.vector_ch_axis,
            ).to(X.dtype)

        return X

    def forward(self, weights: _torch.Tensor) -> _torch.Tensor:
        if self.cluster_permute and len(self.cluster_permute) == len(weights.size()):
            weights = weights.permute(self.cluster_permute)
        if self.enable_per_channel_scale:
            if not isinstance(self.per_channel_scaling_factor, _torch.Tensor):
                self.per_channel_scaling_factor = _torch.zeros((weights.flatten(1).shape[0], 1))
            with _torch.no_grad():
                if not self.per_channel_scaling_factor[0][0]:
                    permuted_weights_proj = weights.flatten(1)
                    if self.per_channel_scaling_factor_scheme == "min_max":
                        self.per_channel_scaling_factor = 0.5 * (
                            permuted_weights_proj.max(1)[0].view(-1, 1)
                            - permuted_weights_proj.min(1)[0].view(-1, 1)
                        )
                    elif self.per_channel_scaling_factor_scheme == "abs":
                        self.per_channel_scaling_factor = (
                            permuted_weights_proj.abs().max(1)[0].view(-1, 1)
                        )
                    else:
                        raise ValueError(
                            f"Unsupported per_channel_scaling_factor_scheme:{self.per_channel_scaling_factor_scheme}"
                        )

            weights = (weights.flatten(1) / self.per_channel_scaling_factor).view(
                weights.size()
            )  # scale the weights using projection factors

        if self.fake_palett_enabled[0] == 1:
            if not self.partitions:
                self.create_partitions(weights.detach())
            tensor_hook = None
            if self.training and self.palett_max_mem < 1.0:
                tensor_hook = _FakePalettizerTensorHook(
                    zero_threshold=self.zero_threshold,
                    device=weights.device,
                    min_size=self.palett_min_tsize,
                    max_mem=self.palett_max_mem,
                    use_unique=self.palett_unique
                    and self.cluster_dim == 1
                    and weights.dtype in [_torch.bfloat16, _torch.float16],
                    use_shard=self.palett_shard,
                )

            with _torch.autograd.graph.saved_tensors_hooks(
                tensor_hook.pack, tensor_hook.unpack
            ) if tensor_hook else contextlib.nullcontext():
                cloned_weights = weights.clone()
                self.init_partitions(cloned_weights.detach())
                palettized_weights = self.diff_palettize(cloned_weights)
        else:
            palettized_weights = super().forward(weights)

        if self.enable_per_channel_scale:
            palettized_weights = (
                palettized_weights.flatten(1) * self.per_channel_scaling_factor
            ).view(palettized_weights.size())

        if self.cluster_permute:
            palettized_weights = palettized_weights.permute(
                _torch.argsort(_torch.Tensor(self.cluster_permute)).tolist()
            )

        if self.lut_dtype == "f16":
            palettized_weights = palettized_weights.to(_torch.float16).to(weights.dtype)
        elif self.lut_dtype == "b16":
            palettized_weights = palettized_weights.to(_torch.bfloat16).to(weights.dtype)

        return palettized_weights

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.lut_dtype = local_metadata["lut_dtype"]
        if "per_channel_scaling_factor" in local_metadata:
            self.per_channel_scaling_factor = local_metadata["per_channel_scaling_factor"]
        self.fake_palett_enabled = _torch.empty(
            state_dict[prefix + "fake_palett_enabled"].size(),
            device=self.fake_palett_enabled.device,
        )
        _Partitioner._load_from_state_dict_(
            self,
            state_dict,
            prefix + "palett.",
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        if self.need_to_quantize:
            # We will go through FakeQuantize._load_from_state_dict and then nn.Module._load_from_state_dict
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
        else:
            # Jump FakeQuantize and go to nn.Module directly
            super(_FakeQuantize, self)._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

    def _save_to_state_dict(self, destination, prefix, keep_vars):

        if self.need_to_quantize:
            # Use normal inheritance, go through FakeQuantize._save_to_state_dict
            super()._save_to_state_dict(destination, prefix, keep_vars)
            self.centroids = super().forward(self.centroids)
        else:
            # Skip FakeQuantize._save_to_state_dict and go directly to nn.Module._save_to_state_dict
            super(_FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)

        # State dicts can only contain tensors (for DDP), so store infos in the metatadata dict (in particular str)
        destination._metadata[prefix[:-1]]["lut_dtype"] = self.lut_dtype
        destination._metadata[
            prefix + "per_channel_scaling_factor"
        ] = self.per_channel_scaling_factor
        _Partitioner._save_to_state_dict_(self, destination, prefix + "palett.", keep_vars)

    def __repr__(self):
        rep = super().__repr__()
        rep += f"lut_dtype: {self.lut_dtype}, "
        rep += f"n_bits: {self.n_bits}, "
        rep += f"cluster_dim: {self.cluster_dim}, "
        rep += f"palett_tau: {self.palett_tau}, "
        rep += f"palett_mode: {self.palett_mode}"
        return rep


class dist_batch_cdist_square(_torch.autograd.Function):
    def forward_2d(X, C):
        _C = C.reshape(-1)
        _X = X.repeat(1, C.size(0))
        _T = _X - _C
        _T = _T.square()
        T = _T.view(X.size(0), C.size(0), C.size(1)).sum(dim=-1)
        return T

    def forward_3d(X, C):
        T = [None] * X.size(0)

        for i in range(X.size(0)):
            T[i] = dist_batch_cdist_square.forward_2d(X[i], C[i])

        return _torch.stack(T)

    def backward_2d(X, C, grad_output):
        _C = C.reshape(-1)
        _X = X.repeat(1, C.size(0))
        _T = _X - _C
        _T = _T.view(-1, C.size(0), C.size(1))
        _T = _T * grad_output.unsqueeze(-1).expand(
            grad_output.size(0), grad_output.size(1), C.size(1)
        )

        grad_X = _T.sum(dim=1)
        grad_C = _T.sum(dim=0)

        return 2 * grad_X, -2 * grad_C

    def backward_3d(X, C, grad_output):
        grad_X = [None] * X.size(0)
        grad_C = [None] * X.size(0)

        for i in range(X.size(0)):
            grad_X[i], grad_C[i] = dist_batch_cdist_square.backward_2d(X[i], C[i], grad_output[i])

        return _torch.stack(grad_X), _torch.stack(grad_C)

    @staticmethod
    def forward(ctx, X, C):
        shard_list = _get_shard_list(X.size(0))
        T = [None] * _dist.world_size

        for i in range(_dist.world_size):
            cur_X = X[shard_list[i] : shard_list[i + 1]]
            cur_C = C[shard_list[i] : shard_list[i + 1]]

            if i == _dist.rank:
                T[i] = _torch.cdist(cur_X, cur_C).square()
            else:
                T[i] = _torch.zeros(
                    [cur_X.size(0), cur_X.size(1), cur_C.size(1)],
                    device=X.device,
                    dtype=X.dtype,
                )

        _dist.all_gather(T, T[_dist.rank])
        T = _torch.cat(T)

        M = _torch.Tensor([])
        ctx.save_for_backward(X, C, M)

        return T

    @staticmethod
    def backward(ctx, grad_output):
        X, C, _ = ctx.saved_tensors

        # gradient is data-dependent, so it CANNOT be sharded
        if X.dim() == 3:
            grad_X, grad_C = dist_batch_cdist_square.backward_3d(X, C, grad_output)
        else:
            grad_X, grad_C = dist_batch_cdist_square.backward_2d(X, C, grad_output)

        return grad_X, grad_C
