#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.optimize.torch.palettization._supported_modules import DKMPalettizerModulesRegistry


def _assert_changes_post_attach(module, n_bits, cluster_dim):
    assert hasattr(module, 'qconfig')
    assert module.qconfig.weight.p.keywords["n_bits"] == n_bits
    assert module.qconfig.weight.p.keywords["cluster_dim"] == cluster_dim


def _assert_changes_post_prepare(
    original_module, palettized_module, n_bits, cluster_dim, kmeans_max_iter
):
    assert type(palettized_module) == DKMPalettizerModulesRegistry.get_palettizer_module(
        original_module
    )
    assert palettized_module.weight_fake_quant.n_clusters == 2**n_bits
    assert palettized_module.weight_fake_quant.cluster_dim == cluster_dim
    assert palettized_module.weight_fake_quant.kmeans_max_iter == kmeans_max_iter


def _get_max_unique_weights_in_module_post_conversion(config, module):
    return (2 ** config[type(module)]["n_bits"]) \
           * config[type(module)]["cluster_dim"]
