# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .annotate import class_annotate, annotate, delay_type
from .type_spec import *


@class_annotate()
class bool:
    def __init__(self, v=False):
        self.val = v

    @classmethod
    def __type_info__(cls):
        return Type("bool", python_class=cls)

    @annotate(delay_type.bool, other=delay_type.bool)
    def __eq__(self, other):
        return bool(self.val == other.val)

    @annotate(delay_type.bool, other=delay_type.bool)
    def __ne__(self, other):
        return bool(self.val != other.val)

    @annotate(delay_type.bool)
    def __not__(self):
        return bool(not other.val)

    @annotate(delay_type.bool)
    def __bool__(self):
        return self.val

    @annotate(delay_type.int)
    def __int__(self):
        return int(self)

    @annotate(delay_type.double)
    def __double__(self):
        return float(self.val)

    @annotate(delay_type.str)
    def __str__(self):
        return str(self.val)

def is_bool(t):
    return t is bool or isinstance(t,bool)