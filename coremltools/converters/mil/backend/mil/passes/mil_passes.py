#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.backend.nn.passes.nn_passes import nn_backend_passes
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
import logging as _logging


def mil_backend_passes(prog):
    passes = [
        "common::const_elimination",
        "mil_backend::adjust_io_to_supported_types",
        "mil_backend::insert_image_preprocessing_ops",
        "mil_backend::fuse_activation_silu",
        "common::const_elimination", # rank0_expand_dims_swap might introduce some new const tensor
        # TODO: Right now, "const elimination" pass CANNOT be done after the "homogenize_input_dtypes" pass.
        # Remove this requirement in rdar://76032946.
        # Right now due to a bug in the PYMIL const op, which is that it can only produce FP32 and INT32 types tensors (e.g. it can't produce int64),
        # therefore invoking const elimination after the var type promotion that happens in "homogenize_input_dtypes" will lead to issues if a
        # const var (on const propagation through cast op) has to be promoted to int64 dtype.
        "mil_backend::homogenize_input_dtypes",
        "common::cast_optimization",  # Need to run after homogenize_input_dtypes
        "common::dead_code_elimination",
        "mil_backend::sanitize_name_strings",
        "common::dedup_op_and_var_names",
        "nn_backend::handle_unused_inputs",  # must come after dce.
        "nn_backend::alert_return_type_cast",  # must be at the end.
    ]

    _logging.debug("Program before common passes:\n{}".format(prog))

    prog.validate()
    for p in passes:
        _logging.info('Performing passes for mil backend: "{}"'.format(p))
        PASS_REGISTRY[p](prog)
        # No more validation from this point on as prog is not SSA anymore.

    _logging.debug("Program after mil backend passes:\n{}".format(prog))
