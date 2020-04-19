# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .annotate import annotate
from .type_void import void
from . import type_int
from .type_spec import Type
from .get_type_info import get_type_info


def memoize(f):
    memo = {}

    def helper(x, init_length=None):
        if x not in memo:
            memo[(x, init_length)] = f(x, init_length)
        return memo[(x, init_length)]

    return helper


class empty_list:
    @classmethod
    def __type_info__(cls):
        return Type("empty_list", python_class=cls)


@memoize
def list(arg, init_length=None):
    class list:
        T = [arg, init_length]

        def __init__(self):
            self.val = []

        @classmethod
        def __type_info__(cls):
            return Type("list", [get_type_info(arg)], python_class=cls)

        @annotate(void, other=T[0])
        def append(self, other):
            assert (isinstance(other, self.T[0]))
            self.val.append(other)

        @annotate(T[0], index=type_int.int)
        def __getitem__(self, index):
            assert (isinstance(index, type_int.int))
            return self.val[index.val]

        @annotate(void, index=type_int.int, newval=T[0])
        def __setitem__(self, index, newval):
            assert (isinstance(index, type_int.int))
            assert (isinstance(newval, self.T[0]))
            self.val[index.val] = newval

        @annotate(type_int.int)
        def __len__(self):
            return type_int.int(len(self.val)) if T[1] is None else T[1]

    list.__template_name__ = "list[" + arg.__name__ + "]"
    return list


def is_list(t):
    if t is None:
        return False
    return get_type_info(t).name == 'list'
