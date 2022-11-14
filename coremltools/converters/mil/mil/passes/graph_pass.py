#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod



class AbstractGraphPass(ABC):

    def __call__(self, prog):
        if not prog.skip_all_passes:
            self.apply(prog)

    def __str__(self):
        return type(self).__name__

    @abstractmethod
    def apply(self, prog):
        pass
