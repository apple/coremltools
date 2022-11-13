#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from .annotate import annotate, class_annotate, delay_type
from .type_spec import Type


@class_annotate()
class str:
    def __init__(self, v=""):
        self.val = v

    @classmethod
    def __type_info__(cls):
        return Type("str", python_class=cls)

    @annotate(delay_type.str, other=delay_type.str)
    def __add__(self, other):
        assert isinstance(other, str)
        return str(self.val + other.val)
