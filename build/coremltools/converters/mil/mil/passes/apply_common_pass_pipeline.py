#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.quantization_passes import AbstractQuantizationPass
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
import logging as _logging
from coremltools.converters._profile_utils import _profile
from tqdm import tqdm as _tqdm
from coremltools.converters.mil.experimental.passes.generic_pass_infrastructure import  PassContainer


@_profile
def apply_common_pass_pipeline(prog, passes):

    def _apply(passes, name="common"):

        if len(passes) == 0:
            return

        _logging.debug("Program before {} passes:\n{}".format(name, prog))

        prog.validate()
        s = 'passes' if len(passes) > 1 else 'pass'
        for p in _tqdm(passes, desc="Running MIL {} {}".format(name, s), unit=" passes"):
            _logging.info('Performing pass: "{}"'.format(p))

            PASS_REGISTRY[p](prog) if not isinstance(p, AbstractQuantizationPass) else p.apply(prog)
            if isinstance(p, AbstractQuantizationPass) or not isinstance(PASS_REGISTRY[p], PassContainer):
                prog.validate()

        _logging.debug("Program after {} passes:\n{}".format(name, prog))

        return

    common_passes = [
        "common::cast_optimization",
        "common::const_elimination",
        "common::sanitize_input_output_names",
        "common::divide_to_multiply",
        "common::add_conv_transpose_output_shape",
        "common::const_elimination",
        "common::loop_invariant_elimination",
        "common::remove_symbolic_reshape",
        'common::noop_elimination',
        "common::fuse_matmul_weight_bias",
        "common::fuse_linear_bias",
        "common::fuse_gelu_tanh_approximation",
        "common::fuse_gelu_exact",
        "common::fuse_leaky_relu",
        "common::rank0_expand_dims_swap",
        "common::use_reflection_padding",
        "common::merge_consecutive_paddings", # Should come after use_reflection_padding, which will introduce new padding layers
        "common::pad_conv_connect", # Should come after merge_consecutive_paddings
        'common::image_input_preprocess',
        "common::replace_stack_reshape", # should come before detect_concat_interleave since it may add concat
        "common::reduce_transposes",
        "common::fuse_conv_scale",
        "common::fuse_conv_bias",
        "common::fuse_onehot_matmul_to_gather",
        "common::fuse_layernorm_or_instancenorm",  # should come after reduce_transposes, to detect instance_norm
        "common::fuse_elementwise_to_batchnorm",  # should come after fuse_layernorm_or_instancenorm
        "common::fuse_reduce_mean", # should come after fuse_layernorm_or_instancenorm
        "common::fuse_conv_batchnorm", # should come after fuse_elementwise_to_batchnorm
        "common::fuse_conv_scale", # Re-run the fuse conv scale pass after the conv and batch_norm are fused
        "common::fuse_conv_bias", # Re-run the fuse conv bias pass after the conv and batch_norm are fused
        "common::detect_concat_interleave",
        "common::concat_to_pixel_shuffle", # should come after detect_concat_interleave and after replace_stack_reshape
        "common::dead_code_elimination",  # always end with dce
    ]

    _apply(common_passes, name="Common")

    for p in passes:
        if isinstance(p, AbstractQuantizationPass):
            _apply([p], type(p).__name__)

    cleanup_passes = [
        "common::cast_optimization",
        "common::const_elimination",
        "common::loop_invariant_elimination",
        "common::noop_elimination",
        "common::dedup_op_and_var_names",
        "common::reduce_transposes",  # fuse_layernorm_or_instancenorm can potentially adding transposes
        "common::topological_reorder",
        "common::dead_code_elimination",  # always end with dce
    ]

    _apply(cleanup_passes, name="Clean up")
