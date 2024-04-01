#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from .const_deduplication import const_deduplication
from .const_elimination import const_elimination
from .dead_code_elimination import dead_code_elimination
from .dedup_op_and_var_names import dedup_op_and_var_names
from .expand_dynamic_linear import expand_dynamic_linear
from .fuse_reduce_mean import fuse_reduce_mean
from .loop_invariant_elimination import loop_invariant_elimination
from .noop_elimination import noop_elimination
from .remove_redundant_ops import remove_redundant_ops
from .remove_symbolic_reshape import remove_symbolic_reshape
from .topological_reorder import topological_reorder
