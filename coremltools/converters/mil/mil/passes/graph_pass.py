#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as target

class AbstractGraphPass(ABC):

    def __init__(self, minimun_deployment_target=target.iOS13):
        self._minimum_deployment_target = minimun_deployment_target

    def __call__(self, prog):
        if not prog.skip_all_passes:
            self.apply(prog)

    def __str__(self):
        return type(self).__name__

    @property
    def minimun_deployment_target(self):
        return self._minimum_deployment_target

    @minimun_deployment_target.setter
    def minimun_deployment_target(self, t):
        if not isinstance(t, target):
            raise TypeError("minimun_deployment_target must be an enumeration from Enum class AvailableTarget")
        self._minimum_deployment_target = t


    @abstractmethod
    def apply(self, prog):
        pass
