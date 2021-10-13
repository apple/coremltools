# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
import warnings
import functools


def deprecated(obj=None, suffix=""):
    """
    Decorator to mark a function or a class as deprecated
    """

    def decorator_deprecation_warning(obj):
        @functools.wraps(obj)
        def wrapped(*args, **kwargs):
            if isinstance(obj, type):
                msg = (
                    'Class "%s" is deprecated and will be removed in 6.0.'
                    % obj.__name__
                )
            else:
                msg = (
                    'Function "%s" is deprecated and will be removed in 6.0.'
                    % obj.__name__
                )
            if suffix:
                msg += "; %s" % suffix
            warnings.warn(msg, category=FutureWarning)
            return obj(*args, **kwargs)

        return wrapped

    if obj is None:
        return decorator_deprecation_warning

    return decorator_deprecation_warning(obj)
