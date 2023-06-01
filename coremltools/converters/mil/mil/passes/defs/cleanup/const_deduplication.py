#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import hashlib
from typing import Dict, List, Tuple

import numpy as np

from coremltools.converters.mil.mil import Block, Var, types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common")
class const_deduplication(AbstractGraphPass):
    """
    Remove duplicated large constants (tensor with 100+ elements)

    For example

    .. code-block::

        Input graph (where weight and bias are large constants):
            weight_q = const(weight)
            weight_k = const(weight)
            bias_q = const(bias)
            bias_k = const(bias)
            q_embedding = linear(x=q, weight=weight_q, bias=bias_q)
            k_embedding = linear(x=k, weight=weight_k, bias=bias_k)

        Output graph:
            weight_q = const(weight)
            bias_q = const(bias)
            q_embedding = linear(x=q, weight=weight_q, bias=bias_q)
            k_embedding = linear(x=k, weight=weight_q, bias=bias_q)

    Concretely, we consider a constant as duplicated if there exists such a previous constant that:

    1. has same dtype and value

    2. comes from same type of op

    The reason why op type is considered is, there are 2 types of constants in Core ML:

    1. The usual constant, i.e., the output of ``const`` op

    2. The result of const expression, i.e., the output of ``constexpr_*`` ops
    """

    NUMEL_THRESH = 100
    CONSTEXPR_OPS = {
        "constexpr_affine_dequantize",
        "constexpr_cast",
        "constexpr_lut_to_dense",
        "constexpr_sparse_to_dense",
    }
    DTYPE2ATOL = {
        types.fp16: 6e-8,
        types.fp32: 1e-12,
    }

    def apply(self, prog) -> None:
        for f in prog.functions.values():
            self._constant_deduplication_block(f)

    @block_context_manager
    def _constant_deduplication_block(self, block: Block) -> None:
        for op in list(block.operations):
            for b in op.blocks:
                self._constant_deduplication_block(b)

        unique2duplicates = self.find_constants(block)
        for unique in unique2duplicates:
            for duplicate in unique2duplicates[unique]:
                if duplicate in block.outputs:
                    continue
                op = duplicate.op
                block.replace_uses_of_var_after_op(
                    anchor_op=op,
                    old_var=duplicate,
                    new_var=unique,
                    force_replace=True if op.op_type in self.CONSTEXPR_OPS else False,
                )
                block.remove_ops([op])

    def find_constants(self, block: Block) -> Dict[Var, List[Var]]:
        """
        Given a block, return all constants in the block in such a format:
            {unique_var_0: [duplicated_var_0_0, duplicated_var_0_1, ...],
             unique_var_1: [duplicated_var_1_0, duplicated_var_1_1, ...],
             ...
            }
        """
        unique2duplicates: Dict[Var, List[Var]] = {}

        # instead of brute-force C_N^2 comparison, use a hash map to be O(N)
        constant_dict: Dict[Tuple[str, types.type, Tuple[int], str], List[Var]] = {}
        for op in list(block.operations):
            op_type = op.op_type
            if op_type == "const" or op_type in self.CONSTEXPR_OPS:
                constant_var = op.outputs[0]
                shape = constant_var.shape

                numel = np.prod(shape)
                if numel < self.NUMEL_THRESH:
                    continue

                dtype = constant_var.dtype
                value = constant_var.val
                hash = hashlib.sha1(
                    np.ascontiguousarray(value.reshape(-1)[: self.NUMEL_THRESH])
                ).hexdigest()
                key = (op_type, dtype, shape, hash)

                if key not in constant_dict:
                    constant_dict[key] = [constant_var]
                    unique2duplicates[constant_var] = []
                else:
                    hash_collisions = constant_dict[key]

                    existing_constant_var = None
                    for var in hash_collisions:
                        if np.allclose(
                            value,
                            var.val,
                            rtol=0.0,
                            atol=self.DTYPE2ATOL.get(dtype, 1e-12),
                        ):
                            existing_constant_var = var
                            break

                    if existing_constant_var is None:
                        hash_collisions.append(constant_var)
                        unique2duplicates[constant_var] = []
                    else:
                        unique2duplicates[existing_constant_var].append(constant_var)

        return unique2duplicates
