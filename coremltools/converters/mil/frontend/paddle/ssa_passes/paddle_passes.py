#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools import _logger as logger
from coremltools.converters._profile_utils import _profile
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY


@_profile
def paddle_passes(prog):
    passes = [
        "common::dead_code_elimination",
        "common::loop_invariant_elimination",
        "common::dead_code_elimination",
        "paddle::paddle_upsample_to_core_upsample",
        "paddle::paddle_tensor_assign_to_core",
    ]

    prog.validate()
    for p in passes:
        logger.info('Performing passes for paddle frontend: "{}"'.format(p))
        PASS_REGISTRY[p](prog)
        prog.validate()

    logger.debug("Program after paddle frontend passes:\n{}".format(prog))
