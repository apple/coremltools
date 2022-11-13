#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools import _logger as logger
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY


def mil_backend_passes(prog):
    passes = [
        "common::const_elimination",
        "mil_backend::adjust_io_to_supported_types",
        "mil_backend::insert_image_preprocessing_ops",
        "mil_backend::fuse_activation_silu",
        "common::const_elimination", # rank0_expand_dims_swap might introduce some new const tensor
        "common::cast_optimization",
        "common::dead_code_elimination",
        "mil_backend::sanitize_name_strings",
        "common::dedup_op_and_var_names",
        "nn_backend::handle_unused_inputs",  # must come after dce.
    ]

    logger.debug("Program before common passes:\n{}".format(prog))

    prog.validate()
    for p in passes:
        logger.info('Performing passes for mil backend: "{}"'.format(p))
        PASS_REGISTRY[p](prog)
        # No more validation from this point on as prog is not SSA anymore.

    logger.debug("Program after mil backend passes:\n{}".format(prog))
