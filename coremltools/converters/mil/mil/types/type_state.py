#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.types.get_type_info import get_type_info
from coremltools.converters.mil.mil.types.type_spec import Type


def memoize(f):
    memo = {}

    def helper(state_type):
        if state_type not in memo:
            memo[state_type] = f(state_type)
        return memo[state_type]

    return helper


@memoize
def state(state_type):
    class state:
        T = [state_type]

        def __init__(self):
            self.val = []

        @property
        def val(self):
            return self._val

        @classmethod
        def wrapped_type(cls):
            return state_type

        @classmethod
        def __type_info__(cls):
            return Type("state", [get_type_info(state_type)], python_class=cls)

    state.__template_name__ = f"state[{get_type_info(state_type).name}]"
    return state


def is_state(t):
    if t is None:
        return False
    return get_type_info(t).name == "state"
