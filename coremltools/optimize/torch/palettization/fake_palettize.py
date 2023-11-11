#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import contextlib
import gc

import torch as _torch
import torch.nn.functional as _F
from torch.ao.quantization.observer import ObserverBase as _ObserverBase
from torch.quantization import FakeQuantize as _FakeQuantize

from ._efficient_kmeans import _EfficientKMeans
from ._fake_palettizer_tensor_hook import _FakePalettizationTensorHook
from ._partitioner import _Partitioner
from .palettization_config import DEFAULT_PALETTIZATION_ADVANCED_OPTIONS


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
        quant_min (:obj:`int`): The minimum allowable quantized value.
        quant_max (:obj:`int`): The maximum allowable quantized value.
        cluster_dtype (:obj:`str`): String that decides whether to quantize the ``LUT`` or not. The following are the ``str``
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
        quant_min: int = -128,
        quant_max: int = 127,
        cluster_dtype: str = "f32",
        advanced_options: dict = {},
        **observer_kwargs,
    ):
        partition_size = advanced_options.get(
            "partition_size", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["partition_size"]
        )
        cluster_permute = advanced_options.get(
            "cluster_permute", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["cluster_permute"]
        )
        palett_max_mem = advanced_options.get(
            "palett_max_mem", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_max_mem"]
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
            "palett_cluster_tol", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["palett_cluster_tol"]
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
            "add_extra_centroid", DEFAULT_PALETTIZATION_ADVANCED_OPTIONS["add_extra_centroid"]
        )

        self._target_module_level_sparsity = 0.0

        _FakeQuantize.__init__(self, observer, quant_min, quant_max, **observer_kwargs)
        _Partitioner.__init__(
            self,
            n_bits,
            enforce_zero,
            partition_size,
            cluster_dim,
            cluster_permute,
            palett_tau,
            kmeans_init,
            prune_threshold,
            kmeans_opt1d_threshold,
            add_extra_centroid,
        )

        self.cluster_dtype = cluster_dtype
        self.add_extra_centroid = add_extra_centroid
        self.need_to_quantize = self.cluster_dtype in ["i8", "u8", "f16"]
        self.autograd_graph = hasattr(_torch.autograd, "graph") and palett_max_mem < 1.0
        self.palett_max_mem = palett_max_mem
        self.palett_cluster_tol = palett_cluster_tol
        self.kmeans_max_iter = kmeans_max_iter
        self.palett_mode = palett_mode
        self.palett_tau = palett_tau
        self.palett_epsilon = palett_epsilon
        self.palett_lambda = palett_lambda
        self.n_bits = n_bits
        self.cluster_dim = cluster_dim
        self.kmeans_init = kmeans_init
        # Temporary create placeholder buffers that will get replaced with proper centroids on the first forward,
        # or when we reload a checkpoint. Having placeholder values is useful to maintain the structure of the state
        # dict constant.
        self.register_buffer("centroids", _torch.rand([1]))
        self.register_buffer("labels", _torch.rand([1]))
        # During init, we would want the fake_palett_enabled flag to be False, i.e. to be at a state of 0. Also, we
        # would have set the fake_quant_enabled and observer_enabled to be 0 as well so that palettizer does nothing
        # until the first milestone.
        self.register_buffer("fake_palett_enabled", _torch.tensor([0], dtype=_torch.uint8))
        self.disable_fake_quant()
        self.disable_observer()
        self.buffers_are_placeholders = True

    def enable_fake_palett(self, enabled: bool = True) -> None:
        self.fake_palett_enabled[0] = 1 if enabled else 0

    def disable_fake_palett(self):
        self.enable_fake_palett(False)

    def diff_palettize(self, weights: _torch.Tensor):
        """
        Method called to run the differentiable k-means operation.
        """
        use_cpu_if_cuda_available = False
        if _torch.cuda.is_available():
            t = _torch.cuda.get_device_properties(weights.device).total_memory
            a = _torch.cuda.memory_allocated(weights.device)
            use_cpu_if_cuda_available = (a / t) > self.palett_max_mem and self.autograd_graph
            if use_cpu_if_cuda_available:
                if _FakePalettizationTensorHook.gc_trigger is None:
                    _FakePalettizationTensorHook.gc_trigger = True

        if _FakePalettizationTensorHook.gc_trigger:
            gc.collect()

        auto_grad_graph_on_cpu = (
            _torch.autograd.graph.save_on_cpu(pin_memory=True)
            if use_cpu_if_cuda_available
            else contextlib.nullcontext()
        )

        for i, partition in enumerate(self.partitions):

            current_partition_clone = weights[partition[0] : partition[1]].clone()
            cX, pad = self.flatten(current_partition_clone)

            with _torch.no_grad():
                palett_table = _torch.unique(self.centroids[i], dim=0)
                if len(palett_table) < self.n_clusters[i] * self.palett_cluster_tol:
                    # We use n_init as 3 so as to not spend a lot of time running this operation
                    kmeans = _EfficientKMeans(
                        n_clusters=self.n_clusters[i],
                        init="kmeans++",
                        labels=self.labels[i],
                        n_init=3,
                        max_iter=1,
                    )
                    kmeans.kmeans_pp(3, cX, 0)
                    self.centroids[i] = kmeans.cluster_centers_

            centroids = self.centroids[i].clone()

            assert not centroids.requires_grad
            last_inertia = None

            for j in range(self.kmeans_max_iter):
                if self.autograd_graph:
                    tensor_hook = _FakePalettizationTensorHook(
                        [_torch.Size([cX.size()[0], centroids.size()[0]])],
                        use_cpu_if_cuda_available,
                        f"FakePalettizationTensorHook.{i}.{j}",
                        self.palett_tau,
                    )
                    auto_grad_graph_hook_init = _torch.autograd.graph.saved_tensors_hooks(
                        tensor_hook.init_pack, tensor_hook.init_unpack
                    )
                    auto_grad_graph_hook_reuse = _torch.autograd.graph.saved_tensors_hooks(
                        tensor_hook.reuse_pack, tensor_hook.reuse_unpack
                    )
                else:
                    auto_grad_graph_hook_init = contextlib.nullcontext()
                    auto_grad_graph_hook_reuse = contextlib.nullcontext()

                with auto_grad_graph_hook_init:
                    x_c_dist = _EfficientKMeans.x_c_dist(cX, centroids)
                    min_error, _ = x_c_dist.min(dim=-1)

                with auto_grad_graph_hook_reuse:
                    if "dkm" in self.palett_mode:
                        attention = _F.softmax(-x_c_dist / self.palett_tau, dim=1)
                    elif "gsm" in self.palett_mode:
                        attention = _F.gumbel_softmax(-x_c_dist / self.palett_tau, dim=1)
                    elif "hard" in self.palett_mode:
                        col_idx = x_c_dist.min(dim=1).indices
                        row_idx = _torch.arange(start=0, end=len(col_idx), dtype=_torch.int32).to(
                            cX.device
                        )
                        attention = _torch.sparse_coo_tensor(
                            _torch.vstack([row_idx, col_idx]),
                            _torch.ones_like(row_idx).to(cX.device),
                            x_c_dist.size(),
                            dtype=x_c_dist.dtype,
                            requires_grad=True,
                        ).to_dense()

                assert attention.requires_grad
                attention_sum = attention.sum(dim=0).view(-1, 1)
                attention_sum[attention_sum == 0] = 1e-6

                with auto_grad_graph_hook_reuse:
                    centroids = _torch.matmul(cX.T, attention).T / attention_sum

                with auto_grad_graph_on_cpu:
                    if self.need_to_quantize:
                        centroids = super().forward(centroids)

                    assert centroids.requires_grad

                    if self.prune_threshold > 0:
                        centroids = _torch.nn.Hardshrink(self.prune_threshold.item())(centroids)

                    if self.enforce_zero[i]:
                        zero_point = (
                            _torch.zeros(centroids[0].size()).to(centroids.device).unsqueeze(0)
                        )
                        zero_idx = _torch.argmin(_torch.cdist(centroids, zero_point))
                        centroids[zero_idx] = zero_point

                cur_inertia = min_error.sum()

                if last_inertia and abs(last_inertia - cur_inertia) <= self.palett_epsilon:
                    break

                last_inertia = cur_inertia

            with auto_grad_graph_hook_reuse:
                weights[partition[0] : partition[1]] = self.deflatten(
                    _torch.matmul(attention, centroids), current_partition_clone.size(), pad
                )

                self.centroids[i] = (
                    self.palett_lambda * self.centroids[i] + (1 - self.palett_lambda) * centroids
                ).detach()
                self.labels[i] = attention.detach().max(dim=1)[1].data

        return weights

    def palettize(self, weights: _torch.Tensor):
        """
        This method is run during inference time by the forward method of the ``FakePalettize`` class. It calculates the
        weight from the ``LUT`` and ``indices`` across all partitions and returns them.
        """
        for i, partition in enumerate(self.partitions):
            labels = self.labels[i]
            if labels is not None:
                current_weight_partition = weights[partition[0] : partition[1]].detach()
                _, pad = self.flatten(current_weight_partition)

                weights[partition[0] : partition[1]] = self.deflatten(
                    self.centroids[i][labels.long()], current_weight_partition.size(), pad
                )

        return weights

    def forward(self, weights: _torch.Tensor):
        if self.partition_size == 0:
            forwarded_weights = super().forward(weights)
            if self.fake_palett_enabled[0] == 1:
                with _torch.no_grad():
                    quant_centroids, quant_labels = forwarded_weights.unique(return_inverse=True)
                    self.centroids = _torch.stack([quant_centroids.view(-1, self.cluster_dim)])
                    self.labels = _torch.stack([quant_labels])
        else:
            forwarded_weights = weights.clone()

            if self.fake_palett_enabled[0] == 1:
                if not self.partitions:
                    self.init_partitions(weights.detach())
                    self.centroids = _torch.stack(self.centroids_init)
                    self.labels = _torch.stack(self.labels_init)
                    self.buffers_are_placeholders = False

                if self.training:
                    forwarded_weights = self.diff_palettize(forwarded_weights)
                else:
                    forwarded_weights = self.palettize(forwarded_weights)
            else:
                forwarded_weights = super().forward(weights)

        if self.cluster_dtype == "f16":
            forwarded_weights = forwarded_weights.to(_torch.float16).to(weights.dtype)
        elif self.cluster_dtype == "b16":
            forwarded_weights = forwarded_weights.to(_torch.bfloat16).to(weights.dtype)

        return forwarded_weights

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):

        self.cluster_dtype = local_metadata["cluster_dtype"]
        state_dict_buffers_are_placeholders = local_metadata["buffers_are_placeholders"]

        if not self.buffers_are_placeholders and state_dict_buffers_are_placeholders:
            raise ValueError(
                f"Trying to reload an uninitialized state dict onto an initialized module: {prefix[:-1]}"
            )

        if self.buffers_are_placeholders and not state_dict_buffers_are_placeholders:
            # We only change the size of the placeholders if we intend to reload a proper checkpoint
            # onto an uninitialized module. In the other cases, we expect the state dict and the module to be compatible.
            self.centroids = _torch.empty(
                state_dict[prefix + "centroids"].size(), device=self.centroids.device
            )
            self.labels = _torch.empty(
                state_dict[prefix + "labels"].size(), device=self.labels.device
            )
            self.fake_palett_enabled = _torch.empty(
                state_dict[prefix + "fake_palett_enabled"].size(), device=self.labels.device
            )

        self.buffers_are_placeholders = state_dict_buffers_are_placeholders

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

        # State dicts can only contain tensors (for DDP), so store infos in the metadata dict (in particular str)
        destination._metadata[prefix[:-1]]["cluster_dtype"] = self.cluster_dtype
        destination._metadata[prefix[:-1]][
            "buffers_are_placeholders"
        ] = self.buffers_are_placeholders
        _Partitioner._save_to_state_dict_(self, destination, prefix + "palett.", keep_vars)

    def __repr__(self):
        rep = super().__repr__()
        if self.centroids.shape[0] != self.n_clusters:
            rep += " ===> centroids: uninitialised buffer, "
            rep += "labels: uninitialised buffer, "
        else:
            rep += f" ===> centroids: {self.centroids}, "
            rep += f"labels: {self.labels}, "
        rep += f"cluster_dtype: {self.cluster_dtype}, "
        rep += f"n_bits: {self.n_bits}, "
        rep += f"cluster_dim: {self.cluster_dim}, "
        rep += f"palett_tau: {self.palett_tau}, "
        rep += f"palett_mode: {self.palett_mode}"
        return rep
