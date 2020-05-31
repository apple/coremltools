# -*- coding: utf-8 -*-
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .type_spec import Type


class unknown:
    """
    unknown is basically Any type.
    """
    @classmethod
    def __type_info__(cls):
        return Type("unknown", python_class=cls)

    def __init__(self, val=None):
        self.val = val
