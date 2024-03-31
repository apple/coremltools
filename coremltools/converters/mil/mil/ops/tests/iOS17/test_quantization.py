#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
from typing import Tuple

import numpy as np
import pytest

from coremltools._deps import _HAS_TORCH, MSG_TORCH_NOT_FOUND
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.ops.tests.iOS17 import backends
from coremltools.converters.mil.mil.ops.tests.testing_utils import run_compare_builder
from coremltools.converters.mil.mil.types import builtin_to_string, numpy_type_to_builtin_type
from coremltools.converters.mil.testing_reqs import BackendConfig, compute_units
from coremltools.converters.mil.testing_utils import ssa_fn

if _HAS_TORCH:
    import torch

torch.manual_seed(1042)
np.random.seed(1042)


def _set_backend_precision(backend, precision):
    return BackendConfig(
        backend=backend.backend,
        precision=precision,
        opset_version=backend.opset_version,
    )

class TestQuantizationBase:
    @staticmethod
    def get_random_quantization_params(
        float_dtype: np.dtype,
        quant_dtype: np.dtype,
        input_rank: int,
        is_zp_present: bool = True,
        axis: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        return floating-point input, floating-point scale, integer zero point
        """

        x_shape = np.random.randint(low=1, high=5, size=(input_rank,))

        low, high = (-128, 128) if quant_dtype == np.int8 else (0, 256)

        # create quantized x
        x_q = np.random.randint(low=low, high=high, size=x_shape)

        # create scale and zero point, the dequantize x
        x_fp = None
        scale = None
        zp = None
        # quantize per tensor
        if axis is None:
            scale = np.array(np.random.rand())
            if is_zp_present:
                zp = np.array(np.random.randint(low=low, high=high))
                x_fp = (x_q - zp) * scale
            else:
                x_fp = x_q * scale
        # quantize per channel
        else:
            # prepare broadcast shape for latter dequantize
            broadcastable_shape = np.ones(input_rank, dtype=np.int32)
            broadcastable_shape[axis] = x_shape[axis]

            scale = np.random.rand(x_shape[axis])
            broadcasted_scale = np.reshape(scale, broadcastable_shape)

            if is_zp_present:
                zp = np.random.randint(low=low, high=high, size=x_shape[axis])
                broadcasted_zp = np.reshape(zp, broadcastable_shape)
                x_fp = (x_q - broadcasted_zp) * broadcasted_scale
            else:
                x_fp = x_q * broadcasted_scale

        x_fp = x_fp.astype(float_dtype)
        scale = scale.astype(float_dtype)
        zero_point = zp.astype(quant_dtype) if is_zp_present else None
        return x_fp, scale, zero_point

    @staticmethod
    def torch_quantize(
        x: np.ndarray,
        scale: np.ndarray,
        zero_point: np.ndarray,
        axis: int = None,
        quant_dtype: np.dtype = None,
    ) -> torch.Tensor:
        """
        return quantized x by pytorch
        """

        # quantization data type is either inferred from `zero_point`,
        # or explicitly provided
        if zero_point is not None:
            quant_dtype = zero_point.dtype
        assert quant_dtype is not None

        # if scale is scalar, then axis must be None
        # if scale is not scalar, then axis must have a value
        assert (len(scale.shape) == 0) != (axis is not None)

        x_torch = torch.from_numpy(x).to(torch.float32)
        s_torch = torch.from_numpy(scale).to(torch.float32)
        zp_torch = (
            torch.zeros(scale.shape, dtype=torch.int)
            if zero_point is None
            else torch.from_numpy(zero_point)
        )
        dtype_torch = torch.quint8 if quant_dtype == np.uint8 else torch.qint8

        output: np.ndarray
        if axis is None:
            output = torch.quantize_per_tensor(x_torch, s_torch, zp_torch, dtype_torch)
        else:
            if axis < 0:
                axis += len(x.shape)
            output = torch.quantize_per_channel(x_torch, s_torch, zp_torch, axis, dtype_torch)
        return output


class TestQuantize(TestQuantizationBase):
    @ssa_fn
    def test_builder_eval_scalar_params(self):
        v = mb.quantize(
            input=np.float32([[0, 2, 4], [0, 2, 4]]),
            zero_point=np.uint8(1),
            scale=np.float32(2),
            output_dtype="uint8",
        )
        np.testing.assert_allclose(np.array([[1, 2, 3], [1, 2, 3]]).astype(np.uint8), v.val)

    @ssa_fn
    def test_builder_eval_vector_params(self):
        v = mb.quantize(
            input=np.array([1, 2, 3, 4]).reshape(1, 1, 2, 2).astype(np.float32),
            zero_point=np.array([2, 4]).astype(np.int8),
            scale=np.array([1, 2]).astype(np.float32),
            axis=3,
            output_dtype="int8",
        )
        np.testing.assert_allclose(
            np.array([3, 5, 5, 6]).reshape(1, 1, 2, 2).astype(np.int8), v.val
        )

    @ssa_fn
    def test_builder_eval_vector_params_neg_axis(self):
        v = mb.quantize(
            input=np.array([1, 2, 3, 4]).reshape(1, 1, 2, 2).astype(np.float32),
            zero_point=np.array([2, 4]).astype(np.int8),
            scale=np.array([1, 2]).astype(np.float32),
            axis=-1,
            output_dtype="int8",
        )
        np.testing.assert_allclose(
            np.array([3, 5, 5, 6]).reshape(1, 1, 2, 2).astype(np.int8), v.val
        )

    @ssa_fn
    def test_builder_eval_no_zero_point(self):
        v = mb.quantize(
            input=np.float32([[0, 2, 4], [0, 2, 4]]),
            scale=np.float32(2),
            output_dtype="int8",
        )
        np.testing.assert_allclose(np.array([[0, 1, 2], [0, 1, 2]]).astype(np.int8), v.val)

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_smoke_builder_to_backend_quantize_per_tensor(self, compute_unit, backend):
        def build(x):
            x = mb.cast(x=x, dtype="fp16")
            quantized = mb.quantize(
                input=x,
                zero_point=np.int8(10),
                scale=np.float16(0.1),
                output_dtype="int8",
            )
            # TODO(rdar://107430678): Replace scale=1 zero_point=0 quantize/dequantize with cast
            dequantized = mb.dequantize(
                input=quantized,
                scale=np.float16(1),
            )
            return dequantized

        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float16)
        expected_output = np.array([0, 10, 20, 30], dtype=np.float16)
        expected_output_type = expected_output.shape + (
            numpy_type_to_builtin_type(expected_output.dtype),
        )
        run_compare_builder(
            build,
            input_placeholders={"x": mb.placeholder(shape=x.shape)},
            input_values={"x": x},
            expected_output_types=[expected_output_type],
            expected_outputs=[expected_output],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_smoke_builder_to_backend_quantize_per_channel(self, compute_unit, backend):
        def build(x):
            x = mb.cast(x=x, dtype="fp16")
            quantized = mb.quantize(
                input=x,
                zero_point=np.uint8([10, 0]),
                scale=np.float16([0.1, 0.01]),
                axis=0,
                output_dtype="uint8",
            )
            # TODO(rdar://107430678): Replace scale=1 zero_point=0 quantize/dequantize with cast
            dequantized = mb.dequantize(
                input=quantized,
                scale=np.float16(1),
            )
            return dequantized

        x = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float16)
        expected_output = np.array([[0, 10], [100, 200]], dtype=np.float16)
        expected_output_type = expected_output.shape + (
            numpy_type_to_builtin_type(expected_output.dtype),
        )
        run_compare_builder(
            build,
            input_placeholders={"x": mb.placeholder(shape=x.shape)},
            input_values={"x": x},
            expected_output_types=[expected_output_type],
            expected_outputs=[expected_output],
            compute_unit=compute_unit,
            backend=backend,
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        "compute_unit, backend, float_dtype, quant_dtype, compute_precision, input_rank, is_zp_present",
        itertools.product(
            compute_units,
            backends,
            (np.float32, np.float16),
            (np.int8, np.uint8),
            ("fp32", "fp16"),
            (1, 2, 3, 4, 5),
            (True, False),
        ),
    )
    def test_stress_builder_to_backend_quantize_all_possibilities(
        self,
        compute_unit,
        backend,
        float_dtype,
        quant_dtype,
        compute_precision,
        input_rank,
        is_zp_present,
    ):
        def build(x):
            x = mb.cast(x=x, dtype=builtin_to_string(numpy_type_to_builtin_type(float_dtype)))
            quantized = mb.quantize(
                input=x,
                zero_point=zero_point,
                scale=scale,
                axis=axis,
                output_dtype=builtin_to_string(numpy_type_to_builtin_type(quant_dtype)),
            )
            # TODO(rdar://107430678): Replace scale=1 zero_point=0 quantize/dequantize with cast
            dequantized = mb.dequantize(
                input=quantized,
                scale=float_dtype(1),
            )
            return dequantized

        for axis in [None] + [i for i in range(-input_rank, input_rank)]:
            x_fp, scale, zero_point = self.get_random_quantization_params(
                float_dtype, quant_dtype, input_rank, is_zp_present, axis
            )

            input_placeholders = {
                "x": mb.placeholder(
                    shape=x_fp.shape,
                    dtype=numpy_type_to_builtin_type(float_dtype),
                ),
            }
            input_values = {"x": x_fp}

            output_torch = self.torch_quantize(x_fp, scale, zero_point, axis, quant_dtype)
            output_torch_val = output_torch.int_repr().numpy()
            output_type = output_torch_val.shape + (numpy_type_to_builtin_type(np.float32),)
            expected_outputs = [output_torch_val]
            expected_output_types = [output_type]

            run_compare_builder(
                build,
                input_placeholders,
                input_values,
                expected_output_types,
                expected_outputs=expected_outputs,
                compute_unit=compute_unit,
                backend=_set_backend_precision(backend, compute_precision),
            )


class TestDequantize(TestQuantizationBase):
    @ssa_fn
    def test_builder_eval_scalar_params(self):
        v = mb.dequantize(
            input=np.array([[1, 2, 3], [1, 2, 3]]).astype(np.uint8),
            zero_point=np.uint8(1),
            scale=np.float32(2),
        )
        assert v.val is None
        np.testing.assert_allclose(
            np.float32([[0, 2, 4], [0, 2, 4]]),
            v.op.materialized_val_inference(),
        )

    @ssa_fn
    def test_builder_eval_vector_params(self):
        v = mb.dequantize(
            input=np.array([3, 5, 5, 6]).reshape(1, 1, 2, 2).astype(np.uint8),
            zero_point=np.array([2, 4]).astype(np.uint8),
            scale=np.array([1, 2]).astype(np.float32),
            axis=3,
        )
        assert v.val is None
        np.testing.assert_allclose(
            np.array([1, 2, 3, 4]).reshape(1, 1, 2, 2).astype(np.float32),
            v.op.materialized_val_inference(),
        )

    @ssa_fn
    def test_builder_eval_no_zero_point(self):
        v = mb.dequantize(
            input=np.array([[0, 1, 2], [0, 1, 2]]).astype(np.int8),
            scale=np.float32(2),
        )
        assert v.val is None
        np.testing.assert_allclose(
            np.float32([[0, 2, 4], [0, 2, 4]]),
            v.op.materialized_val_inference(),
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_smoke_builder_to_backend_dequantize_per_tensor(self, compute_unit, backend):
        def build(x):
            x = mb.cast(x=x, dtype="fp32")
            # TODO(rdar://107430678): Replace scale=1 zero_point=0 quantize/dequantize with cast
            quantized = mb.quantize(
                input=x,
                scale=np.float32(1),
                output_dtype="uint8",
            )
            dequantized = mb.dequantize(
                input=quantized,
                zero_point=np.uint8(5),
                scale=np.float32(0.2),
            )
            return dequantized

        x = np.array([5, 10, 15, 20], dtype=np.float32)
        expected_output = np.array([0, 1, 2, 3], dtype=np.float32)
        expected_output_type = expected_output.shape + (
            numpy_type_to_builtin_type(expected_output.dtype),
        )
        run_compare_builder(
            build,
            input_placeholders={"x": mb.placeholder(shape=x.shape)},
            input_values={"x": x},
            expected_output_types=[expected_output_type],
            expected_outputs=[expected_output],
            compute_unit=compute_unit,
            backend=_set_backend_precision(backend, "fp32"),
            atol=1e-3,
            rtol=1e-3,
        )

    @pytest.mark.parametrize("compute_unit, backend", itertools.product(compute_units, backends))
    def test_smoke_builder_to_backend_dequantize_per_channel(self, compute_unit, backend):
        def build(x):
            x = mb.cast(x=x, dtype="fp32")
            # TODO(rdar://107430678): Replace scale=1 zero_point=0 quantize/dequantize with cast
            quantized = mb.quantize(
                input=x,
                scale=np.float32(1),
                output_dtype="int8",
            )
            dequantized = mb.dequantize(
                input=quantized,
                zero_point=np.int8([-5, 5]),
                scale=np.float32([0.2, 0.3]),
                axis=1,
            )
            return dequantized

        x = np.array([[-10, -5], [0, 5]], dtype=np.float32)
        expected_output = np.array([[-1, -3], [1, 0]], dtype=np.float32)
        expected_output_type = expected_output.shape + (
            numpy_type_to_builtin_type(expected_output.dtype),
        )
        run_compare_builder(
            build,
            input_placeholders={"x": mb.placeholder(shape=x.shape)},
            input_values={"x": x},
            expected_output_types=[expected_output_type],
            expected_outputs=[expected_output],
            compute_unit=compute_unit,
            backend=_set_backend_precision(backend, "fp32"),
            atol=1e-3,
            rtol=1e-3,
        )

    @pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
    @pytest.mark.parametrize(
        "compute_unit, backend, float_dtype, quant_dtype, compute_precision, input_rank, is_zp_present",
        itertools.product(
            compute_units,
            backends,
            (np.float32, np.float16),
            (np.int8, np.uint8),
            ("fp32", "fp16"),
            (1, 2, 3, 4, 5),
            (True, False),
        ),
    )
    def test_stress_builder_to_backend_dequantize_all_possibilities(
        self,
        compute_unit,
        backend,
        float_dtype,
        quant_dtype,
        compute_precision,
        input_rank,
        is_zp_present,
    ):
        def build(x):
            x = mb.cast(x=x, dtype=builtin_to_string(numpy_type_to_builtin_type(float_dtype)))
            # TODO(rdar://107430678): Replace scale=1 zero_point=0 quantize/dequantize with cast
            quantized = mb.quantize(
                input=x,
                scale=float_dtype(1),
                output_dtype=builtin_to_string(numpy_type_to_builtin_type(quant_dtype)),
            )
            dequantized = mb.dequantize(
                input=quantized,
                zero_point=zero_point,
                scale=scale,
                axis=axis,
            )
            return dequantized

        for axis in [None] + [i for i in range(-input_rank, input_rank)]:
            x_fp, scale, zero_point = self.get_random_quantization_params(
                float_dtype, quant_dtype, input_rank, is_zp_present, axis
            )

            x_q = self.torch_quantize(x_fp, scale, zero_point, axis, quant_dtype)

            output_torch_val = torch.dequantize(x_q).numpy()
            output_type = output_torch_val.shape + (numpy_type_to_builtin_type(np.float32),)

            input_placeholders = {
                "x": mb.placeholder(
                    shape=x_fp.shape,
                    dtype=numpy_type_to_builtin_type(float_dtype),
                ),
            }
            input_values = {"x": x_q.int_repr().numpy()}

            expected_outputs = [output_torch_val]
            expected_output_types = [output_type]
            run_compare_builder(
                build,
                input_placeholders,
                input_values,
                expected_output_types,
                expected_outputs=expected_outputs,
                compute_unit=compute_unit,
                backend=_set_backend_precision(backend, compute_precision),
                rtol=1e-3,
            )
