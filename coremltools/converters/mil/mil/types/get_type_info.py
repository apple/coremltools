# -*- coding: utf-8 -*-

#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from .type_spec import FunctionType, Type
from .type_void import void


def get_python_method_type(py_function):
    # given a python class method, parse the annotations to figure out the type
    function_inputs = []
    function_output = get_type_info(void)
    annotations = {}
    if hasattr(py_function, "type_annotations"):
        annotations = {
            k: get_type_info(v) for k, v in py_function.type_annotations.items()
        }
    if hasattr(py_function, "return_type"):
        function_output = get_type_info(py_function.return_type)
    try:
        if hasattr(py_function, "__func__"):
            argcount = py_function.__func__.__code__.co_argcount
            argnames = py_function.__func__.__code__.co_varnames[:argcount]
        else:
            argcount = py_function.__code__.co_argcount
            argnames = py_function.__code__.co_varnames[:argcount]
    except:
        raise TypeError(
            "Unable to derive type information from method %s. "
            "You might have a misspecified type. Ex: use compyler.int and not int"
            % py_function
        )

    for arg in argnames:
        if arg in annotations:
            function_inputs.append(annotations[arg])
        elif arg != "self":
            raise TypeError(
                "Function "
                + str(py_function)
                + " insufficient annotations. "
                + arg
                + " needs a type"
            )
    typeinfo = FunctionType(function_inputs, function_output, py_function)
    return typeinfo


def get_type_info(t):
    if hasattr(t, "__type_info__"):
        ret = t.__type_info__()
        assert ret.python_class is not None
        return ret
    elif isinstance(t, type):
        return Type(t.__name__, python_class=t)
    elif hasattr(t, "__call__"):
        return get_python_method_type(t)
    raise TypeError("Unsupported type %s" % t)
