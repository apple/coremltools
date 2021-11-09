#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

class AbstractGraphPass():

    def __call__(self, prog):
        self.apply(prog)

    def apply(self, prog):
        raise NotImplementedError(
            'Graph pass transformation not implemented for "{}".'.format(self)
        )

