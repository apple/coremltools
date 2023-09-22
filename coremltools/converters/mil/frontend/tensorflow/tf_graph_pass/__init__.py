#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


from .cond_to_where import cond_to_where
from .constant_propagation import constant_propagation
# graph passes
from .delete_asserts import delete_asserts
from .delete_constant import delete_unnecessary_constant_nodes
# graphdef to tfssa
from .delete_disconnected_nodes import delete_disconnected_nodes
from .functionalize_loops import functionalize_loops
from .fuse_dilation_conv import fuse_dilation_conv
from .insert_get_tuple import insert_get_tuple
from .quantization_pass import quantization_pass
from .tensor_array_transform import tensor_array_resource_removal
from .variable_node_transform import remove_variable_nodes
