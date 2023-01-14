#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

_PADDLE_OPS_REGISTRY = {}


def register_paddle_op(_func=None, paddle_alias=None, override=False):
    """
    Registration routine for PyPaddle operators
    _func: (PyPaddle conversion function) [Default=None]
        PyPaddle conversion function to register

    paddle_alias: (List of string) [Default=None]
        All other PyPaddle operators that should also be mapped to
        current conversion routine.
        e.g. Sort aliased with SortV1, SortV2
        All provided alias operators must not be registered previously.

        "In place" alias are looked up automatically and do not need to
        be registered. PyPaddle uses an underscore suffix to denote the
        in place version, e.g. "sum_" is the in place version of "sum".

    override: (Boolean) [Default=False]
        If True, overrides earlier registration i.e. specified
        operator and alias will start pointing to current conversion
        function.
        Otherwise, duplicate registration will error out.
    """

    def func_wrapper(func):
        f_name = func.__name__

        if f_name.endswith("_"):
            raise Exception(
                "Attempting to register \"{}\" op. Do not register inplace ops. (inplace paddle ops"
                " end in a \"_\"). Instead register the normal op version: \"{}\". The inplace"
                " version will be supported automatically.".format(f_name, f_name[:-1])
            )
        if not override and f_name in _PADDLE_OPS_REGISTRY:
            raise ValueError("Paddle op {} already registered.".format(f_name))

        _PADDLE_OPS_REGISTRY[f_name] = func

        if paddle_alias is not None:
            for name in paddle_alias:
                if not override and name in _PADDLE_OPS_REGISTRY:
                    msg = "Paddle op alias {} already registered."
                    raise ValueError(msg.format(name))
                _PADDLE_OPS_REGISTRY[name] = func

        return func

    if _func is None:
        # decorator called without argument
        return func_wrapper
    return func_wrapper(_func)
