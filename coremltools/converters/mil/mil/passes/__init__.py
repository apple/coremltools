#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# Import all frontend/backend passes to make sure they got registered.
from coremltools.converters.mil.backend.mil.passes import (
    adjust_io_to_supported_types,
    fuse_activation_silu,
    fuse_pow2_sqrt,
    insert_image_preprocessing_op,
    sanitize_name_strings,
)
from coremltools.converters.mil.backend.nn.passes import (
    alert_return_type_cast,
    commingle_loop_vars,
    conv1d_decomposition,
    handle_return_inputs_as_outputs,
    handle_return_unused_inputs,
    handle_unused_inputs,
    mlmodel_passes,
)
from coremltools.converters.mil.frontend.tensorflow2.ssa_passes import remove_vacuous_cond
from coremltools.converters.mil.frontend.tensorflow.ssa_passes import (
    backfill_make_list_elem_type,
    expand_tf_lstm,
    tf_lstm_to_core_lstm,
)
from coremltools.converters.mil.frontend.torch.ssa_passes import (
    torch_tensor_assign_to_core,
    torch_upsample_to_core_upsample,
)
from coremltools.converters.mil.mil.passes.defs import (
    cleanup,
    lower_complex_dialect_ops,
    optimize_activation,
    optimize_activation_quantization,
    optimize_conv,
    optimize_elementwise_binary,
    optimize_linear,
    optimize_normalization,
    optimize_quantization,
    optimize_repeat_ops,
    optimize_state,
    optimize_tensor_operation,
    preprocess,
    symbol_transform,
)
