MIL Ops
===============================================

Operators supported by the Model Intermediate Language (MIL):

activation (iOS 15+)
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.activation

   .. autoclass:: clamped_relu
   .. autoclass:: elu
   .. autoclass:: gelu
   .. autoclass:: leaky_relu
   .. autoclass:: linear_activation
   .. autoclass:: prelu
   .. autoclass:: relu
   .. autoclass:: relu6
   .. autoclass:: scaled_tanh
   .. autoclass:: sigmoid
   .. autoclass:: sigmoid_hard
   .. autoclass:: silu
   .. autoclass:: softplus
   .. autoclass:: softplus_parametric
   .. autoclass:: softmax
   .. autoclass:: softsign
   .. autoclass:: thresholded_relu

activation (iOS 17+)
---------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS17.activation

   .. autoclass:: clamped_relu
   .. autoclass:: elu
   .. autoclass:: leaky_relu
   .. autoclass:: linear_activation
   .. autoclass:: prelu
   .. autoclass:: scaled_tanh
   .. autoclass:: sigmoid_hard
   .. autoclass:: softplus_parametric
   .. autoclass:: thresholded_relu

classify
---------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.classify

   .. autoclass:: classify

constexpr_ops (iOS 16+)
---------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS16.constexpr_ops

   .. autoclass:: constexpr_affine_dequantize
   .. autoclass:: constexpr_cast
   .. autoclass:: constexpr_lut_to_dense
   .. autoclass:: constexpr_sparse_to_dense

constexpr_ops (iOS 18+)
---------------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS18.compression

   .. autoclass:: constexpr_blockwise_shift_scale
   .. autoclass:: constexpr_lut_to_dense
   .. autoclass:: constexpr_sparse_to_dense
   .. autoclass:: constexpr_lut_to_sparse
   .. autoclass:: constexpr_sparse_blockwise_shift_scale
   .. autoclass:: constexpr_cast

control\_flow
------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.control_flow

   .. autoclass:: cond
   .. autoclass:: Const
   .. autoclass:: select
   .. autoclass:: while_loop
   .. autoclass:: make_list
   .. autoclass:: list_length
   .. autoclass:: list_write
   .. autoclass:: list_read
   .. autoclass:: list_gather
   .. autoclass:: list_scatter

conv (iOS 15+)
---------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.conv

   .. autoclass:: conv
   .. autoclass:: conv_transpose

conv (iOS 17+)
---------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS17.conv

   .. autoclass:: conv
   .. autoclass:: conv_transpose

coreml_update_state
---------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.coreml_dialect.ops

   .. autoclass:: coreml_update_state

elementwise\_binary
------------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_binary

   .. autoclass:: add
   .. autoclass:: equal
   .. autoclass:: floor_div
   .. autoclass:: greater
   .. autoclass:: greater_equal
   .. autoclass:: less
   .. autoclass:: less_equal
   .. autoclass:: logical_and
   .. autoclass:: logical_or
   .. autoclass:: logical_xor
   .. autoclass:: maximum
   .. autoclass:: minimum
   .. autoclass:: mod
   .. autoclass:: mul
   .. autoclass:: not_equal
   .. autoclass:: real_div
   .. autoclass:: pow
   .. autoclass:: sub

elementwise\_unary (iOS 15+)
-----------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_unary

   .. autoclass:: abs
   .. autoclass:: acos
   .. autoclass:: asin
   .. autoclass:: atan
   .. autoclass:: atanh
   .. autoclass:: cast
   .. autoclass:: ceil
   .. autoclass:: clip
   .. autoclass:: cos
   .. autoclass:: cosh
   .. autoclass:: erf
   .. autoclass:: exp
   .. autoclass:: exp2
   .. autoclass:: floor
   .. autoclass:: inverse
   .. autoclass:: log
   .. autoclass:: logical_not
   .. autoclass:: round
   .. autoclass:: rsqrt
   .. autoclass:: sign
   .. autoclass:: sin
   .. autoclass:: sinh
   .. autoclass:: sqrt
   .. autoclass:: square
   .. autoclass:: tan
   .. autoclass:: tanh
   .. autoclass:: threshold

elementwise\_unary (iOS 17+)
-----------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS17.elementwise_unary

   .. autoclass:: cast
   .. autoclass:: clip
   .. autoclass:: inverse
   .. autoclass:: log
   .. autoclass:: rsqrt

image\_resizing (iOS 15+)
--------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.image_resizing

   .. autoclass:: affine
   .. autoclass:: crop
   .. autoclass:: crop_resize
   .. autoclass:: resample
   .. autoclass:: resize_bilinear
   .. autoclass:: resize_nearest_neighbor
   .. autoclass:: upsample_bilinear
   .. autoclass:: upsample_nearest_neighbor

image\_resizing (iOS 16+)
--------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS16.image_resizing

   .. autoclass:: crop_resize
   .. autoclass:: resample
   .. autoclass:: upsample_bilinear

image\_resizing (iOS 17+)
--------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS17.image_resizing

   .. autoclass:: crop_resize
   .. autoclass:: resample
   .. autoclass:: resize

linear (iOS 15+)
-----------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.linear

   .. autoclass:: einsum
   .. autoclass:: linear
   .. autoclass:: matmul

linear (iOS 17+)
-----------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS17.linear

   .. autoclass:: linear
   .. autoclass:: matmul

normalization (iOS 15+)
------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.normalization

   .. autoclass:: batch_norm
   .. autoclass:: instance_norm
   .. autoclass:: l2_norm
   .. autoclass:: layer_norm
   .. autoclass:: local_response_norm

normalization (iOS 17+)
------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS17.normalization

   .. autoclass:: batch_norm
   .. autoclass:: instance_norm
   .. autoclass:: l2_norm
   .. autoclass:: layer_norm
   .. autoclass:: local_response_norm

pool
---------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.pool

   .. autoclass:: avg_pool
   .. autoclass:: l2_pool
   .. autoclass:: max_pool

quantization
------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS17.quantization_ops

   .. autoclass:: quantize
   .. autoclass:: dequantize

random
-----------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.random

   .. autoclass:: random_bernoulli
   .. autoclass:: random_categorical
   .. autoclass:: random_normal
   .. autoclass:: random_uniform

recurrent (iOS 15+)
--------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.recurrent

   .. autoclass:: gru
   .. autoclass:: lstm
   .. autoclass:: rnn

recurrent (iOS 17+)
--------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS17.recurrent

   .. autoclass:: gru
   .. autoclass:: lstm
   .. autoclass:: rnn

recurrent (iOS 18+)
--------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS18.recurrent

   .. autoclass:: gru


reduction (iOS 15+)
--------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.reduction

   .. autoclass:: reduce_argmax
   .. autoclass:: reduce_argmin
   .. autoclass:: reduce_l1_norm
   .. autoclass:: reduce_l2_norm
   .. autoclass:: reduce_log_sum
   .. autoclass:: reduce_log_sum_exp
   .. autoclass:: reduce_max
   .. autoclass:: reduce_mean
   .. autoclass:: reduce_min
   .. autoclass:: reduce_prod
   .. autoclass:: reduce_sum
   .. autoclass:: reduce_sum_square

reduction (iOS 17+)
--------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS17.reduction

   .. autoclass:: reduce_argmax
   .. autoclass:: reduce_argmin

scatter\_gather (iOS 15+)
--------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.scatter_gather

   .. autoclass:: gather
   .. autoclass:: gather_along_axis
   .. autoclass:: gather_nd
   .. autoclass:: scatter
   .. autoclass:: scatter_along_axis
   .. autoclass:: scatter_nd

scatter\_gather (iOS 16+)
--------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS16.scatter_gather

   .. autoclass:: gather
   .. autoclass:: gather_nd

scatter\_gather (iOS 17+)
--------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather

   .. autoclass:: gather
   .. autoclass:: gather_along_axis
   .. autoclass:: gather_nd
   .. autoclass:: scatter
   .. autoclass:: scatter_along_axis
   .. autoclass:: scatter_nd

states (iOS 18+)
--------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS18.states

   .. autoclass:: read_state

tensor\_operation (iOS 15+)
----------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.tensor_operation

   .. autoclass:: argsort
   .. autoclass:: band_part
   .. autoclass:: concat
   .. autoclass:: cumsum
   .. autoclass:: fill
   .. autoclass:: flatten2d
   .. autoclass:: identity
   .. autoclass:: non_maximum_suppression
   .. autoclass:: non_zero
   .. autoclass:: one_hot
   .. autoclass:: pad
   .. autoclass:: range_1d
   .. autoclass:: shape
   .. autoclass:: split
   .. autoclass:: stack
   .. autoclass:: tile
   .. autoclass:: topk

tensor\_operation (iOS 16+)
----------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS16.tensor_operation

   .. autoclass:: fill_like
   .. autoclass:: topk

tensor\_operation (iOS 17+)
----------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS17.tensor_operation

   .. autoclass:: non_maximum_suppression
   .. autoclass:: topk

tensor\_transformation (iOS 15)
---------------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation

   .. autoclass:: batch_to_space
   .. autoclass:: depth_to_space
   .. autoclass:: expand_dims
   .. autoclass:: pixel_shuffle
   .. autoclass:: reshape
   .. autoclass:: reverse
   .. autoclass:: reverse_sequence
   .. autoclass:: slice_by_index
   .. autoclass:: slice_by_size
   .. autoclass:: sliding_windows
   .. autoclass:: space_to_batch
   .. autoclass:: space_to_depth
   .. autoclass:: squeeze
   .. autoclass:: transpose

tensor\_transformation (iOS 16+)
---------------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS16.tensor_transformation

   .. autoclass:: pixel_unshuffle
   .. autoclass:: reshape_like

tensor\_transformation (iOS 17+)
---------------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS17.tensor_transformation

   .. autoclass:: expand_dims
   .. autoclass:: reshape
   .. autoclass:: reshape_like
   .. autoclass:: reverse
   .. autoclass:: reverse_sequence
   .. autoclass:: slice_by_index
   .. autoclass:: slice_by_size
   .. autoclass:: sliding_windows
   .. autoclass:: squeeze
   .. autoclass:: transpose

tensor\_transformation (iOS 18+)
---------------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS18.tensor_transformation

   .. autoclass:: slice_update

transformers (iOS 18+)
---------------------------------------------------------------------

.. automodule:: coremltools.converters.mil.mil.ops.defs.iOS18.transformers

   .. autoclass:: scaled_dot_product_attention

