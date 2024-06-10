# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import functools
import warnings


def deprecated(obj=None, suffix="", version="", obj_prefix=""):
    """
    Decorator to mark a function or a class as deprecated
    """

    def decorator_deprecation_warning(obj):
        @functools.wraps(obj)
        def wrapped(*args, **kwargs):
            if isinstance(obj, type):
                msg = (
                    f"Class {obj_prefix}{obj.__name__} is deprecated and will be removed in {version}."
                )
            else:
                msg = (
                    f"Function {obj_prefix}{obj.__name__} is deprecated and will be removed in {version}."
                )
            if suffix:
                msg += f"; {suffix}"
            warnings.warn(msg, category=DeprecationWarning)
            return obj(*args, **kwargs)

        return wrapped

    if obj is None:
        return decorator_deprecation_warning

    return decorator_deprecation_warning(obj)
