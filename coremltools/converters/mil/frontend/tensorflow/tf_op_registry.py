#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from coremltools import _logger as logger

_TF_OPS_REGISTRY = {}


def register_tf_op(_func=None, tf_alias=None, override=False, strict=True):
    """
    Registration routine for TensorFlow operators
    _func: (TF conversion function) [Default=None]
        TF conversion function to register

    tf_alias: (List of string) [Default=None]
        All other TF operators that should also be mapped to
        current conversion routine.
        e.g. Sort aliased with SortV1, SortV2
        All provided alias operators must not be registered previously.

    override: (Boolean) [Default=False]
        If True, overrides earlier registration i.e. specified
        operator and alias will start pointing to current conversion
        function.

    strict: (Boolean) [Default=True]
        If True, duplicate registration will error out when override is not set.
        It's mainly for the case where the tf op is already registered for TF2, and we
        can ignore it in TF1.
    """

    def func_wrapper(func):
        funcs_to_register = [func.__name__]
        if tf_alias is not None:
            # If tf_alias is provided, all the functions mentioned as aliased
            # are mapped to current function.
            funcs_to_register.extend(tf_alias)

        for f_name in funcs_to_register:
            if not override and f_name in _TF_OPS_REGISTRY:
                msg = f"TF op {f_name} already registered."
                if strict:
                    raise ValueError(msg)
                else:
                    logger.debug(msg + " Ignored.")
                    return func
            else:
                _TF_OPS_REGISTRY[f_name] = func

        return func

    if _func is None:
        # decorator called without argument
        return func_wrapper
    return func_wrapper(_func)
