#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from . import (
    add_conv_transpose_output_shape,
    cast_optimization,
    concat_to_pixel_shuffle,
    const_elimination,
    conv_batchnorm_fusion,
    conv_bias_fusion,
    conv_scale_fusion,
    dead_code_elimination,
    dedup_op_and_var_names,
    detect_concat_interleave,
    divide_to_multiply,
    elementwise_batchnorm_fusion,
    gelu_exact_fusion,
    gelu_tanh_approximation_fusion,
    graph_pass,
    helper,
    image_input_preprocessing,
    layernorm_instancenorm_pattern_fusion,
    leaky_relu_fusion,
    linear_bias_fusion,
    loop_invariant_elimination,
    matmul_weight_bias_fusion,
    merge_consecutive_paddings,
    name_sanitization_utils,
    noop_elimination,
    onehot_matmul_to_gather,
    pad_conv_connect,
    quantization_passes,
    rank0_expand_dims_swap,
    reduce_mean_fusion,
    reduce_transposes,
    remove_redundant_ops,
    remove_symbolic_reshape,
    replace_stack_reshape,
    sanitize_input_output_names,
    topological_reorder,
    use_reflection_padding
)

from coremltools.converters.mil.experimental.passes import (
    generic_gelu_tanh_approximation_fusion,
    generic_layernorm_instancenorm_pattern_fusion,
    generic_linear_bias_fusion,
    generic_conv_batchnorm_fusion,
    generic_conv_scale_fusion,
    generic_conv_bias_fusion
)
