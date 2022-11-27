#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from tqdm import tqdm as _tqdm

from coremltools import _logger as logger
from coremltools.converters._profile_utils import _profile
from coremltools.converters.mil.experimental.passes.generic_pass_infrastructure import \
    PassContainer
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
from coremltools.converters.mil.mil.passes.quantization_passes import \
    AbstractQuantizationPass


@_profile
def apply_common_pass_pipeline(prog, passes):

    def _apply(passes, name="common"):

        if len(passes) == 0:
            return

        logger.debug("Program before {} passes:\n{}".format(name, prog))

        prog.validate()
        s = 'passes' if len(passes) > 1 else 'pass'
        for p in _tqdm(passes, desc="Running MIL {} {}".format(name, s), unit=" passes"):
            logger.info('Performing pass: "{}"'.format(p))
            graph_pass = PASS_REGISTRY[p] if not isinstance(p, AbstractQuantizationPass) else p
            graph_pass(prog)
            if isinstance(p, AbstractQuantizationPass) or not isinstance(PASS_REGISTRY[p], PassContainer):
                prog.validate()

        logger.debug("Program after {} passes:\n{}".format(name, prog))

        return

    common_passes = [
        "common::lower_complex_dialect_ops",
        "common::update_output_dtypes",
        "common::cast_optimization",
        "common::const_elimination",
        "common::sanitize_input_output_names",
        "common::divide_to_multiply",
        "common::add_conv_transpose_output_shape",
        "common::const_elimination",
        "common::loop_invariant_elimination",
        "common::remove_symbolic_reshape",
        "common::noop_elimination",
        "common::fuse_matmul_weight_bias",
        "common::fuse_linear_bias",
        # "common::fuse_gelu_tanh_approximation",
        "common::fuse_gelu_exact",
        "common::fuse_leaky_relu",
        "common::rank0_expand_dims_swap",
        "common::use_reflection_padding",
        "common::merge_consecutive_paddings", # Should come after use_reflection_padding, which will introduce new padding layers
        "common::pad_conv_connect", # Should come after merge_consecutive_paddings
        "common::image_input_preprocess",
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
        "common::fuse_conv_batchnorm", # In some cases, we need to run conv / batch_norm fusion again after the fuse_conv_scale and fuse_conv_bias passes
        "common::detect_concat_interleave",
        "common::concat_to_pixel_shuffle", # should come after detect_concat_interleave and after replace_stack_reshape
        # "common::fuse_prelu", # reduce_transpose pass should run before and after this pass (the one after will be run during the cleanup passes stage)
        "common::prelu_to_lrelu",
        "common::merge_consecutive_relus",
        #  "remove_redundant_ops" pass should be applied towards the end, once other graph passes have done their optimizations.
        # For instance, it should come after passes such as "reduce_transpose" that can introduce redundant transposes
        # in the network (while reducing the total number of transposes), and after passes such as "fuse_layernorm_or_instancenorm"
        # which detects patterns that involve redundant ops ("sub") etc.
        "common::remove_redundant_ops",
        "common::dead_code_elimination",  # always end with dce
    ]

    _apply(common_passes, name="Common")

    for p in passes:
        if isinstance(p, AbstractQuantizationPass):
            _apply([p], type(p).__name__)

    cleanup_passes = [
        "common::dead_code_elimination",
        "common::const_elimination",
        "common::cast_optimization",
        "common::const_elimination",
        "common::loop_invariant_elimination",
        "common::noop_elimination",
        "common::dedup_op_and_var_names",
        "common::reduce_transposes",  # fuse_layernorm_or_instancenorm can potentially add transposes
        "common::remove_redundant_ops",
        "common::topological_reorder",
        "common::dead_code_elimination",  # always end with dce
    ]

    _apply(cleanup_passes, name="Clean up")
