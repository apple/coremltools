# Copyright (c) 2019, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


import os.path
from ...models import MLModel


def convert(filename, inputs=None, outputs=None, **kwargs):
    if not filename or not isinstance(filename, str) or not os.path.exists(filename) or not os.path.isfile(filename):
        raise ValueError('invalid input tf_model_path: {}.'.format(filename))

    if not filename.endswith('.pb'):
        raise ValueError('invalid input tf_model_path format, expecting TensorFlow frozen graph (.pb) model.')

    # convert from TensorFlow to SSA
    try:
        from ..nnssa.frontend.tensorflow import load as frontend_load
        ssa = frontend_load(filename, resume_on_errors=False, **kwargs)
    except ImportError as err:
        raise ImportError("Frontend converter not found! Error message:\n%s" % err)

    # convert from SSA to Core ML
    try:
        from ..nnssa.coreml.ssa_converter import ssa_convert
        mlmodelspec = ssa_convert(ssa, top_func='main', inputs=inputs, outputs=outputs)
    except ImportError as err:
        raise ImportError("Backend converter not found! Error message:\n%s" % err)

    use_cpu_only = kwargs.get('use_cpu_only')
    use_cpu_only = use_cpu_only if use_cpu_only is not None else False
    return MLModel(mlmodelspec, useCPUOnly=use_cpu_only)
