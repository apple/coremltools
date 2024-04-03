#  Copyright (c) 2023, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import hashlib
from typing import Dict, List, Tuple

import numpy as np

from coremltools.converters.mil.mil import Block, Var, ListVar, types
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

    Concretely, this graph pass consists of two stages:

    (1) Deduplication of ``const`` op:

        We consider a ``const`` as duplicated if there exists such a previous ``const`` that has same dtype and value

    (2) Deduplication of ``constexpr_*`` op:

        We consider a ``constexpr_*`` as duplicated if there exists such a previous ``constexpr_*`` that has the same ``op_type`` and input attributes.
    """

    NUMEL_THRESH = 100
    DTYPE2ATOL = {
        types.fp16: 6e-8,
        types.fp32: 1e-12,
    }

    def apply(self, prog) -> None:
        for f in prog.functions.values():
            self._constant_deduplication_block(f)

    def remove_duplicate_ops(
        self, block: Block, unique2duplicates: Dict[Var, List[Var]], force_replace: bool
    ) -> None:
        for unique in unique2duplicates:
            for duplicate in unique2duplicates[unique]:
                if duplicate in block.outputs:
                    continue
                block.replace_uses_of_var_after_op(
                    anchor_op=duplicate.op,
                    old_var=duplicate,
                    new_var=unique,
                    force_replace=force_replace,
                )
                block.remove_ops([duplicate.op])

    @block_context_manager
    def _constant_deduplication_block(self, block: Block) -> None:
        for op in list(block.operations):
            for b in op.blocks:
                self._constant_deduplication_block(b)

        # Deduplication of ``const`` op
        unique2duplicates_const = self.find_constants(block)
        self.remove_duplicate_ops(block, unique2duplicates_const, force_replace=False)

        # Deduplication of ``constexpr_*`` op
        # Note that, the ``find_constexpr`` must go after ``find_constants`` + ``remove_duplicate_ops`` for ``const`` ops.
        # Since after the above two functions, ``const`` ops with identical values are
        # deduplicated into a single ``Var`` object, which allows ``find_constexpr`` to
        # directly compare the ``const`` input attr pointers instead of the actual values.
        unique2duplicates_constexpr = self.find_constexprs(block)
        self.remove_duplicate_ops(block, unique2duplicates_constexpr, force_replace=True)

    def find_constexprs(self, block: Block) -> Dict[Var, List[Var]]:
        """
        Given a block, return all constexpr in the block in such a format:
            {unique_var_0: [duplicated_var_0_0, duplicated_var_0_1, ...],
             unique_var_1: [duplicated_var_1_0, duplicated_var_1_1, ...],
             ...
            }
        """
        hashkey_2_duplicates: Dict[Tuple, List[Var]] = {}
        for op in list(block.operations):
            if "constexpr" in op.op_type:
                hash_key = [op.op_type]
                for v in op.inputs.values():
                    hash_key.append(v.dtype)
                    if np.prod(v.shape) < self.NUMEL_THRESH:
                        hash_key.append(str(v.val))
                    else:
                        hash_key.append(v)
                hash_key = tuple(hash_key)
                if hash_key not in hashkey_2_duplicates:
                    hashkey_2_duplicates[hash_key] = [op.outputs[0]]
                else:
                    hashkey_2_duplicates[hash_key].append(op.outputs[0])

        return {v[0]: v[1:] for v in hashkey_2_duplicates.values()}

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
            if op.op_type == "const":
                constant_var = op.outputs[0]
                if isinstance(constant_var, ListVar):
                    continue
                shape = constant_var.shape

                numel = np.prod(shape)
                if numel < self.NUMEL_THRESH:
                    continue

                dtype = constant_var.dtype
                value = constant_var.val
                hash = hashlib.sha1(
                    np.ascontiguousarray(value.reshape(-1)[: self.NUMEL_THRESH])
                ).hexdigest()
                key = (dtype, shape, hash)

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
