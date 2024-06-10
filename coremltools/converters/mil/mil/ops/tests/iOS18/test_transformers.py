#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest
import torch

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.ops.tests.iOS18 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.testing_reqs import compute_units


class TestScaledDotProductAttention:
    @staticmethod
    def _mb_eval_scaled_dot_product_attention(
        query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray = None
    ) -> np.ndarray:
        @mb.program(opset_version=ct.target.iOS18)
        def prog():
            return mb.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=mask,
            )

        return (
            prog.functions["main"]
            .find_ops(op_type="scaled_dot_product_attention")[0]
            .outputs[0]
            .val
        )

    @staticmethod
    def _torch_scaled_dot_product_attention(
        query: np.ndarray, key: np.ndarray, value: np.ndarray, mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Two things:
        1. torch cannot consume np.ndarray, so need to convert to torch.Tensor
        2. torch cpu kernel has no half-precision support, so need to cast to float
        """
        query_torch = torch.tensor(query).to(torch.float32)
        key_torch = torch.tensor(key).to(torch.float32)
        value_torch = torch.tensor(value).to(torch.float32)

        mask_torch = None
        if mask is not None:
            mask_torch = torch.tensor(mask)
            if mask.dtype != bool:
                mask_torch = mask_torch.to(torch.float32)

        return (
            torch.nn.functional.scaled_dot_product_attention(
                query_torch, key_torch, value_torch, mask_torch
            )
            .numpy()
            .astype(query.dtype)
        )

    @pytest.mark.parametrize(
        "batches, float_dtype, mask_dtype",
        itertools.product(
            ([3], [3, 2], [3, 2, 4]),
            (np.float16, np.float32),
            (None, bool, np.float16, np.float32),
        ),
    )
    def test_builder_eval_stress(self, batches, float_dtype, mask_dtype):
        S = 5
        L = 7
        E = 16
        EV = 32

        query_shape = batches + [L, E]
        key_shape = batches + [S, E]
        value_shape = batches + [S, EV]

        query = np.random.rand(*query_shape).astype(float_dtype)
        key = np.random.rand(*key_shape).astype(float_dtype)
        value = np.random.rand(*value_shape).astype(float_dtype)
        mask = None
        if mask_dtype is not None:
            mask = np.zeros((1, 1, S), dtype=mask_dtype)
            mask[:, :, S // 2 :] = False if mask_dtype is bool else -np.inf

        attention_coreml = self._mb_eval_scaled_dot_product_attention(query, key, value, mask)
        attention_torch = self._torch_scaled_dot_product_attention(query, key, value, mask)
        np.testing.assert_allclose(
            attention_coreml,
            attention_torch,
            atol=1e-6 if float_dtype == np.float32 else 1e-3,
            rtol=1e-6 if float_dtype == np.float32 else 1e-3,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, batches, float_dtype, mask_dtype",
        itertools.product(
            compute_units,
            backends,
            ([3], [3, 2], [3, 2, 4]),
            (np.float16, np.float32),
            (None, bool, np.float16, np.float32),
        ),
    )
    def test_builder_to_backend_stress(
        self, compute_unit, backend, batches, float_dtype, mask_dtype
    ):
        def build(query, key, value):
            return mb.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
            )

        def build_with_mask(query, key, value, mask):
            return mb.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=mask,
            )

        S = 5
        L = 7
        E = 16
        EV = 32

        query_shape = batches + [L, E]
        key_shape = batches + [S, E]
        value_shape = batches + [S, EV]

        query = np.random.rand(*query_shape).astype(float_dtype)
        key = np.random.rand(*key_shape).astype(float_dtype)
        value = np.random.rand(*value_shape).astype(float_dtype)

        input_placeholders = {
            "query": mb.placeholder(
                shape=query.shape, dtype=types.numpy_type_to_builtin_type(float_dtype)
            ),
            "key": mb.placeholder(
                shape=key.shape, dtype=types.numpy_type_to_builtin_type(float_dtype)
            ),
            "value": mb.placeholder(
                shape=value.shape, dtype=types.numpy_type_to_builtin_type(float_dtype)
            ),
        }
        input_values = {
            "query": query,
            "key": key,
            "value": value,
        }

        mask = None
        if mask_dtype is not None:
            mask = np.zeros((1, 1, S), dtype=mask_dtype)
            mask[:, :, S - 1 :] = False if mask_dtype is bool else -np.inf

            input_placeholders["mask"] = mb.placeholder(
                shape=mask.shape, dtype=types.numpy_type_to_builtin_type(mask_dtype)
            )
            input_values["mask"] = mask

        attention_torch = self._torch_scaled_dot_product_attention(query, key, value, mask)
        run_compare_builder(
            build if mask_dtype is None else build_with_mask,
            input_placeholders,
            input_values,
            expected_output_types=[attention_torch.shape + (types.fp32,)],
            expected_outputs=[attention_torch],
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-6 if backend.precision == "fp32" and float_dtype == np.float32 else 1e-3,
            rtol=1e-6 if backend.precision == "fp32" and float_dtype == np.float32 else 1e-3,
        )

    @pytest.mark.parametrize(
        "compute_unit, backend, batches, float_dtype, mask_dtype",
        itertools.product(
            compute_units,
            backends,
            ([2], [2, 3], [2, 3, 4]),
            (np.float16, np.float32),
            (None, bool, np.float16, np.float32),
        ),
    )
    def test_builder_to_backend_dynamic_stress(
        self, compute_unit, backend, batches, float_dtype, mask_dtype
    ):
        def build(query, key, value):
            return mb.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
            )

        def build_with_mask(query, key, value, mask):
            return mb.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=mask,
            )

        S = 2
        L = 2
        E = 4
        EV = 32

        query_shape = batches + [L, E]
        key_shape = batches + [S, E]
        value_shape = batches + [S, EV]

        query = np.random.rand(*query_shape).astype(float_dtype)
        key = np.random.rand(*key_shape).astype(float_dtype)
        value = np.random.rand(*value_shape).astype(float_dtype)

        dynamic_query_shape = query_shape
        dynamic_query_shape[0] = get_new_symbol()
        dynamic_query_shape[-2] = get_new_symbol()
        dynamic_key_shape = key_shape
        dynamic_key_shape[-2] = get_new_symbol()
        dynamic_value_shape = value_shape
        dynamic_value_shape[-2] = get_new_symbol()

        input_placeholders = {
            "query": mb.placeholder(
                shape=tuple(dynamic_query_shape),
                dtype=types.numpy_type_to_builtin_type(float_dtype),
            ),
            "key": mb.placeholder(
                shape=tuple(dynamic_key_shape), dtype=types.numpy_type_to_builtin_type(float_dtype)
            ),
            "value": mb.placeholder(
                shape=tuple(dynamic_value_shape),
                dtype=types.numpy_type_to_builtin_type(float_dtype),
            ),
        }
        input_values = {
            "query": query,
            "key": key,
            "value": value,
        }

        mask = None
        if mask_dtype is not None:
            mask = np.zeros((1, S), dtype=mask_dtype)
            mask[:, S - 1 :] = False if mask_dtype is bool else -np.inf

            dynamic_mask_shape = []
            for i in range(len(mask.shape)):
                dynamic_mask_shape.append(get_new_symbol())

            input_placeholders["mask"] = mb.placeholder(
                shape=tuple(dynamic_mask_shape), dtype=types.numpy_type_to_builtin_type(mask_dtype)
            )
            input_values["mask"] = mask

        attention_torch = self._torch_scaled_dot_product_attention(query, key, value, mask)
        output_shape = list(attention_torch.shape)
        output_shape[0] = query_shape[0]
        output_shape[-2] = query_shape[-2]
        run_compare_builder(
            build if mask_dtype is None else build_with_mask,
            input_placeholders,
            input_values,
            expected_output_types=[tuple(output_shape) + (types.fp32,)],
            expected_outputs=[attention_torch],
            compute_unit=compute_unit,
            backend=backend,
            atol=1e-6 if backend.precision == "fp32" and float_dtype == np.float32 else 1e-3,
            rtol=1e-6 if backend.precision == "fp32" and float_dtype == np.float32 else 1e-3,
        )

    def test_builder_invalid_shape(self):
        B = 3
        S = 5
        L = 7
        E = 16
        EV = 32

        with pytest.raises(
            ValueError,
            match=(
                r"query, key, value must have a same rank, got\n"
                r"\* query rank = [0-9]+\n"
                r"\* key rank = [0-9]+\n"
                r"\* value rank = [0-9]+"
            ),
        ):
            query_shape = [B, L, E]
            key_shape = [S, E]
            value_shape = [S, EV]

            query = np.random.rand(*query_shape)
            key = np.random.rand(*key_shape)
            value = np.random.rand(*value_shape)

            self._mb_eval_scaled_dot_product_attention(query, key, value)

        with pytest.raises(
            ValueError,
            match=(
                r"query, key, value must have at lease rank 3 "
                r"for batch, sequence length, embedding, got rank [0-9]+"
            ),
        ):
            query_shape = [L, E]
            key_shape = [S, E]
            value_shape = [S, EV]

            query = np.random.rand(*query_shape)
            key = np.random.rand(*key_shape)
            value = np.random.rand(*value_shape)

            self._mb_eval_scaled_dot_product_attention(query, key, value)

        with pytest.raises(
            ValueError,
            match=(
                r"query, key, value must have a same batch dimension, got\n"
                r"\* query batch = \((?:\s*\d+\s*,)+\s*\d*\)\n"
                r"\* key batch = \((?:\s*\d+\s*,)+\s*\d*\)\n"
                r"\* value batch = \((?:\s*\d+\s*,)+\s*\d*\)"
            ),
        ):
            query_shape = [B + 1, L, E]
            key_shape = [B, S, E]
            value_shape = [B, S, EV]

            query = np.random.rand(*query_shape)
            key = np.random.rand(*key_shape)
            value = np.random.rand(*value_shape)

            self._mb_eval_scaled_dot_product_attention(query, key, value)

        with pytest.raises(
            ValueError,
            match=(
                r"query and key must have a same embedding dimension, got\n"
                r"\* query embedding = [0-9]+\n"
                r"\* key embedding = [0-9]+"
            ),
        ):
            query_shape = [B, L, E + 1]
            key_shape = [B, S, E]
            value_shape = [B, S, EV]

            query = np.random.rand(*query_shape)
            key = np.random.rand(*key_shape)
            value = np.random.rand(*value_shape)

            self._mb_eval_scaled_dot_product_attention(query, key, value)

        with pytest.raises(
            ValueError,
            match=(
                r"key and value must have a same sequence length, got\n"
                r"\* key sequence = [0-9]+\n"
                r"\* value sequence = [0-9]+"
            ),
        ):
            query_shape = [B, L, E]
            key_shape = [B, S + 1, E]
            value_shape = [B, S, EV]

            query = np.random.rand(*query_shape)
            key = np.random.rand(*key_shape)
            value = np.random.rand(*value_shape)

            self._mb_eval_scaled_dot_product_attention(query, key, value)

        with pytest.raises(
            ValueError,
            match=(
                r"key and mask must have a same sequence length, got\n"
                r"\* key sequence = [0-9]+\n"
                r"\* mask sequence = [0-9]+"
            ),
        ):
            query_shape = [B, L, E]
            key_shape = [B, S, E]
            value_shape = [B, S, EV]

            query = np.random.rand(*query_shape)
            key = np.random.rand(*key_shape)
            value = np.random.rand(*value_shape)

            mask = np.zeros(S + 1, dtype=bool)
            mask[-1] = True

            self._mb_eval_scaled_dot_product_attention(query, key, value, mask)

        with pytest.raises(
            ValueError,
            match=(
                r"Incompatible dim [0-9]+ in shapes "
                r"\((?:\s*\d+\s*,)+\s*\d*\) vs\. \((?:\s*\d+\s*,)+\s*\d*\)"
            ),
        ):
            query_shape = [B, L, E]
            key_shape = [B, S, E]
            value_shape = [B, S, EV]

            query = np.random.rand(*query_shape)
            key = np.random.rand(*key_shape)
            value = np.random.rand(*value_shape)

            mask = np.zeros((B + 1, L - 1, S), dtype=bool)
            mask[:, :, -1] = True

            self._mb_eval_scaled_dot_product_attention(query, key, value, mask)
