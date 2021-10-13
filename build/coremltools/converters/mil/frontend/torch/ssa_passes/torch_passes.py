#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
import logging
from coremltools.converters._profile_utils import _profile


@_profile
def torch_passes(prog):
    passes = [
        "common::dead_code_elimination",
        "common::loop_invariant_elimination",
        "common::dead_code_elimination",
        "torch::torch_upsample_to_core_upsample",
        "torch::torch_tensor_assign_to_core",
    ]

    prog.validate()
    for p in passes:
        logging.info('Performing passes for torch frontend: "{}"'.format(p))
        PASS_REGISTRY[p](prog)
        prog.validate()

    logging.debug("Program after torch frontend passes:\n{}".format(prog))
