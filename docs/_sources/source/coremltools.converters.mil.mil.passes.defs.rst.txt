MIL Graph Passes
===============================================

Graph Passes supported by the Model Intermediate Language (MIL):

cleanup
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.passes.defs.cleanup

    .. autoclass:: const_deduplication
    .. autoclass:: const_elimination
    .. autoclass:: dead_code_elimination
    .. autoclass:: dedup_op_and_var_names
    .. autoclass:: expand_dynamic_linear
    .. autoclass:: fuse_reduce_mean
    .. autoclass:: loop_invariant_elimination
    .. autoclass:: noop_elimination
    .. autoclass:: remove_redundant_ops
    .. autoclass:: remove_symbolic_reshape
    .. autoclass:: topological_reorder


optimize_activation
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.passes.defs.optimize_activation

    .. autoclass:: fuse_gelu_exact
    .. autoclass:: fuse_gelu_tanh_approximation
    .. autoclass:: fuse_leaky_relu
    .. autoclass:: fuse_prelu
    .. autoclass:: prelu_to_lrelu


optimize_conv
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.passes.defs.optimize_conv

    .. autoclass:: add_conv_transpose_output_shape
    .. autoclass:: compose_conv1d
    .. autoclass:: fuse_conv_batchnorm
    .. autoclass:: fuse_conv_bias
    .. autoclass:: fuse_conv_scale
    .. autoclass:: fuse_pad_conv


optimize_elementwise_binary
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.passes.defs.optimize_elementwise_binary

    .. autoclass:: divide_to_multiply
    .. autoclass:: select_optimization
    .. autoclass:: fuse_elementwise_to_batchnorm
    .. autoclass:: rank0_expand_dims_swap


optimize_linear
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.passes.defs.optimize_linear

    .. autoclass:: fuse_linear_bias
    .. autoclass:: fuse_matmul_weight_bias
    .. autoclass:: fuse_transpose_matmul


optimize_normalization
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.passes.defs.optimize_normalization

    .. autoclass:: fuse_layernorm_or_instancenorm


optimize_quantization
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.passes.defs.optimize_quantization

    .. autoclass:: merge_affine_dequantize_with_consecutive_ops
    .. autoclass:: int_op_canonicalization
    .. autoclass:: nullify_redundant_quantization_zero_point
    .. autoclass:: dequantize_quantize_pair_elimination
    .. autoclass:: distributive_quantized_binary_op_scale_normalization
    .. autoclass:: dequantize_to_constexpr


optimize_repeat_ops
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.passes.defs.optimize_repeat_ops

    .. autoclass:: cast_optimization
    .. autoclass:: merge_consecutive_paddings
    .. autoclass:: merge_consecutive_relus
    .. autoclass:: merge_consecutive_reshapes
    .. autoclass:: merge_consecutive_transposes
    .. autoclass:: reduce_transposes


optimize_state
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.passes.defs.optimize_state

    .. autoclass:: canonicalize_inplace_pattern
    .. autoclass:: prefer_state_in_downstream


optimize_tensor_operation
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.passes.defs.optimize_tensor_operation

    .. autoclass:: concat_to_pixel_shuffle
    .. autoclass:: detect_concat_interleave
    .. autoclass:: expand_high_rank_reshape_and_transpose
    .. autoclass:: fuse_onehot_matmul_to_gather
    .. autoclass:: replace_stack_reshape
    .. autoclass:: use_reflection_padding


preprocess
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.passes.defs.preprocess

    .. autoclass:: image_input_preprocess
    .. autoclass:: sanitize_input_output_names
    .. autoclass:: update_output_dtypes


quantization
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.passes.defs.quantization

    .. autoclass:: add_fp16_cast


symbol_transform
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.passes.defs.symbol_transform

    .. autoclass:: materialize_symbolic_shape_program
