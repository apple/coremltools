# Copyright (c) 2019, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Program builder class to construct Core ML models.
"""

from ...proto import Model_pb2 as _Model_pb2
from ...proto import Program_pb2 as _Program_pb2
from ...libcoremlpython import _NeuralNetworkBuffer as NetBuff

import numpy as np

class NeuralNetBuffer():
    """
    This class defines an interface to Parameter buffer reader-writer
    """
    # NOTE:
    # This class can maintain map of <variable_name> to <offset>
    # Abstracting out offset details from user for quering the information
    def __init__(self, file_path, mode='write'):
        bufferMode = NetBuff.mode.write
        if mode == 'read':
            bufferMode = NetBuff.mode.read
        elif mode == 'append':
            bufferMode = NetBuff.mode.append
        else:
            assert mode == 'write' and "mode must be one of 'read', 'write' or 'append'"
        self.net_buffer = NetBuff(file_path, bufferMode)

    def add_buffer(self, data):
        out = self.net_buffer.add_buffer(data)
        return out

    def get_buffer(self, offset):
        return self.net_buffer.get_buffer(offset)
