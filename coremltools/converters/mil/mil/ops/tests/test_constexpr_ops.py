#  Copyright (c) 2022, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import itertools

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.ops.defs.iOS16 import constexpr_ops
from coremltools.converters.mil.mil.ops.tests.testing_utils import \
    run_compare_builder
from coremltools.converters.mil.testing_utils import (get_op_types_in_program,
                                                      ssa_fn)

backends = [("mlprogram", "fp32"), ("mlprogram", "fp16")]
compute_units = testing_reqs.compute_units


@pytest.mark.skipif(
    ct.utils._macos_version() < (13, 0),
    reason="ConstExpr ops available from macOS13 onwards.",
)
class TestConstexprAffineDequantize:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):

        t = np.array(range(4)).reshape(1, 1, 2, 2).astype(np.float32)
        decompressed_constant = (
            np.array([1, 2, 3, 4]).reshape(1, 1, 2, 2).astype(np.float32)
        )
        input_placeholders = {
            "x": mb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            quantized_data = np.array([3, 5, 5, 6]).reshape(1, 1, 2, 2).astype(np.uint8)
            scale = np.array([1, 2]).astype(np.float32)
            zero_point = np.array([2, 4]).astype(np.uint8)
            axis = 3
            y = mb.constexpr_affine_dequantize(
                quantized_data=quantized_data,
                zero_point=zero_point,
                scale=scale,
                axis=axis,
            )
            return mb.add(x=x, y=y)

        expected_output_types = (1, 1, 2, 2, types.fp32)
        expected_outputs = t + decompressed_constant.astype(np.float32)

        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            minimum_deployment_target=ct.target.iOS16,
        )

        # validate that the constexpr op is not removed by any graph pass
        prog = mlmodel._mil_program
        assert "constexpr_affine_dequantize" in get_op_types_in_program(prog)

    @ssa_fn
    def test_builder_eval(self):
        # scalar zero-point & scalar scale
        v = mb.constexpr_affine_dequantize(
            quantized_data=np.array([[1, 2, 3], [1, 2, 3]]).astype(np.uint8),
            zero_point=np.uint8(1),
            scale=np.float32(2),
            axis=0,
        )
        np.testing.assert_allclose(np.float32([[0, 2, 4], [0, 2, 4]]), v.val)

        # vector zero-point & scalar scale
        v = mb.constexpr_affine_dequantize(
            quantized_data=np.array([[1, 2, 3], [1, 2, 3]]).astype(np.int8),
            zero_point=np.array([1, 2]).astype(np.int8),
            scale=np.float32(2),
            axis=0,
        )
        np.testing.assert_allclose(np.float32([[0, 2, 4], [-2, 0, 2]]), v.val)

        # scalar zero-point & vector scale
        v = mb.constexpr_affine_dequantize(
            quantized_data=np.array([[1, 2, 3], [1, 2, 3]]).astype(np.uint8),
            zero_point=np.uint8(1),
            scale=np.array([2, 4]).astype(np.float32),
            axis=0,
        )
        np.testing.assert_allclose(np.float32([[0, 2, 4], [0, 4, 8]]), v.val)

        # vector zero-point & vector scale
        v = mb.constexpr_affine_dequantize(
            quantized_data=np.array([[1, 2, 3], [1, 2, 3]]).astype(np.int8),
            zero_point=np.array([1, 2]).astype(np.int8),
            scale=np.array([2, 4]).astype(np.float32),
            axis=0,
        )
        np.testing.assert_allclose(np.float32([[0, 2, 4], [-4, 0, 4]]), v.val)

    @staticmethod
    def affine_dequant_config_generator():
        np.random.seed(1984)

        for quant_dtype in [np.int8, np.uint8]:
            low = 0 if quant_dtype == np.uint8 else -128
            high = 255 if quant_dtype == np.uint8 else 127
            for rank in range(1, 6):
                shape = np.random.randint(low=2, high=5, size=rank)
                quantized_data = np.random.randint(
                    low=low, high=high, size=shape, dtype=quant_dtype
                )
                axis = np.random.choice(range(-rank, rank))
                scalar_zp = np.random.choice([True, False])
                scalar_sc = np.random.choice([True, False])
                zero_point = (
                    np.random.randint(
                        low=low,
                        high=high,
                        size=quantized_data.shape[axis],
                        dtype=quant_dtype,
                    )
                    if not scalar_zp
                    else np.random.choice(range(low, high)).astype(quant_dtype)
                )
                scale = (
                    np.random.rand(quantized_data.shape[axis]).astype(np.float32)
                    if not scalar_sc
                    else np.float32(np.random.rand())
                )  # fp16 is already covered under backends parameterization

                params = {
                    "quantized_data": quantized_data,
                    "zp": zero_point,
                    "sc": scale,
                    "axis": axis,
                }
                yield params

    @pytest.mark.parametrize(
        "compute_unit, backend, config",
        itertools.product(
            compute_units,
            backends,
            affine_dequant_config_generator.__func__()
        ),
    )
    def test_builder_stress(self, compute_unit, backend, config):

        quantized_data, zero_point, scale, axis = (
            config["quantized_data"],
            config["zp"],
            config["sc"],
            config["axis"],
        )

        def build(x):
            y = mb.constexpr_affine_dequantize(
                quantized_data=quantized_data,
                zero_point=zero_point,
                scale=scale,
                axis=axis,
            )
            return mb.add(x=x, y=y)

        expected_output_types = (
            *quantized_data.shape,
            types.numpy_type_to_builtin_type(scale.dtype),
        )

        t = np.random.rand(*quantized_data.shape).astype(scale.dtype)
        decompressed_constant = constexpr_ops.constexpr_affine_dequantize.decompress(
            quantized_data, zero_point, scale, axis
        )
        expected_outputs = t + decompressed_constant

        input_placeholders = {
            "x": mb.placeholder(shape=quantized_data.shape),
        }
        input_values = {"x": t}
        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            minimum_deployment_target=ct.target.iOS16,
        )

        # validate that the constexpr op is not removed by any graph pass
        prog = mlmodel._mil_program
        if "constexpr_affine_dequantize" not in get_op_types_in_program(prog):
            raise AssertionError("Invalidated: Test Failed")


@pytest.mark.skipif(
    ct.utils._macos_version() < (13, 0),
    reason="ConstExpr ops available from macOS13 onwards.",
)
class TestConstexprCast:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):

        t = np.array(range(4)).reshape(4, 1).astype(np.float32)
        decompressed_constant = np.array([1, 2, 3, 4]).reshape(4, 1).astype(np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            source_val = np.array([1, 2, 3, 4]).reshape(4, 1).astype(np.float16)
            y = mb.constexpr_cast(source_val=source_val, output_dtype="fp32")
            return mb.add(x=x, y=y)

        expected_output_types = (4, 1, types.fp32)
        expected_outputs = t + decompressed_constant.astype(np.float32)

        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            minimum_deployment_target=ct.target.iOS16,
        )

        # validate that the constexpr op is not removed by any graph pass
        prog = mlmodel._mil_program
        if "constexpr_cast" not in get_op_types_in_program(prog):
            raise AssertionError("Invalidated: Test Failed")

    @ssa_fn
    def test_builder_eval(self):
        v = mb.constexpr_cast(source_val=np.float16([1, 2]), output_dtype="fp32")
        np.testing.assert_allclose(np.float32([1, 2]), v.val)

    @staticmethod
    def cast_config_generator():
        np.random.seed(1984)

        for rank in range(1, 6):
            shape = np.random.randint(low=2, high=5, size=rank)
            source_val = np.random.rand(*shape).astype(np.float16)
            params = {
                "source_val": source_val,
                "output_dtype": "fp32",
            }
            yield params

    @pytest.mark.parametrize(
        "compute_unit, backend, config",
        itertools.product(
            compute_units,
            backends,
            cast_config_generator.__func__()
        ),
    )
    def test_builder_stress(self, compute_unit, backend, config):

        source_val, output_dtype = (
            config["source_val"],
            config["output_dtype"],
        )

        def build(x):
            y = mb.constexpr_cast(
                source_val=source_val,
                output_dtype=output_dtype,
            )
            return mb.add(x=x, y=y)

        expected_output_types = (
            *source_val.shape,
            types.string_to_builtin(output_dtype),
        )

        output_np_type = types.nptype_from_builtin(
            types.string_to_builtin(output_dtype)
        )
        t = np.random.rand(*source_val.shape).astype(output_np_type)
        decompressed_constant = source_val.astype(output_np_type)
        expected_outputs = t + decompressed_constant

        input_placeholders = {
            "x": mb.placeholder(shape=source_val.shape),
        }
        input_values = {"x": t}
        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            minimum_deployment_target=ct.target.iOS16,
        )

        # validate that the constexpr op is not removed by any graph pass
        prog = mlmodel._mil_program
        assert "constexpr_cast" in get_op_types_in_program(prog)


@pytest.mark.skipif(
    ct.utils._macos_version() < (13, 0),
    reason="ConstExpr ops available from macOS13 onwards.",
)
class TestConstexprLutToDense:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):

        t = np.array(range(4)).reshape(4, 1).astype(np.float32)
        decompressed_constant = np.array([1, 2, 3, 4]).reshape(4, 1).astype(np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            lut_data = np.array(
                [
                    -19.0,
                    4.0,
                    0.0,
                    -1.0,
                    1.0,
                    3.0,
                    5.0,
                    -8.0,
                    19,
                    13,
                    42,
                    4.5,
                    5.4,
                    2.0,
                    -6,
                    -7,
                ]
            ).astype(np.float32)
            indices = np.array([212, 21]).astype(np.uint8)
            shape = np.array([4, 1]).astype(np.uint32)
            y = mb.constexpr_lut_to_dense(lut=lut_data, indices=indices, shape=shape)
            return mb.add(x=x, y=y)

        expected_output_types = (4, 1, types.fp32)
        expected_outputs = t + decompressed_constant.astype(np.float32)

        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            minimum_deployment_target=ct.target.iOS16,
        )

        # validate that the constexpr op is not removed by any graph pass
        prog = mlmodel._mil_program
        assert "constexpr_lut_to_dense" in get_op_types_in_program(prog)

    @ssa_fn
    def test_builder_eval(self):
        v = mb.constexpr_lut_to_dense(
            lut=np.array([1.0, 2.0, 3.0, 4.0]),
            indices=np.array([10, 4]).astype(np.uint8),
            shape=np.array(
                [
                    5,
                ]
            ).astype(np.uint32),
        )
        np.testing.assert_allclose(
            np.float32([3, 3, 1, 1, 1]).astype(np.float32), v.val
        )

    @staticmethod
    def lut_config_generator():
        np.random.seed(1999)
        for lut_dtype in [np.float32]:  # [np.uint8, np.int8]:
            # float16 already covered under backends parameterization
            # Not possible to write 8-bit tests since no other op consumes uint8/int8 tensors
            for nbits in [1, 2, 4, 6, 8]:
                lut_size = 2**nbits
                if lut_dtype == np.uint8:
                    lut = np.random.randint(low=255, size=lut_size, dtype=np.uint8)
                elif lut_dtype == np.int8:
                    lut = np.random.randint(
                        low=-128, high=127, size=lut_size, dtype=np.int8
                    )
                else:
                    lut = np.random.rand(lut_size).astype(lut_dtype)
                for output_rank in range(1, 6):
                    output_shape = np.random.randint(low=2, high=5, size=output_rank)

                    indices = np.random.randint(
                        low=0, high=2**nbits, size=output_shape, dtype=np.uint8
                    )
                    indices_bitarray = np.unpackbits(
                        indices, bitorder="little"
                    ).reshape(-1, 8)
                    packed_indices = np.packbits(
                        indices_bitarray[:, :nbits], bitorder="little"
                    )

                    assert packed_indices.size == np.ceil(
                        nbits * np.prod(output_shape) / 8
                    ).astype(np.int32)
                    params = {
                        "indices": packed_indices,
                        "shape": output_shape,
                        "lut": lut,
                    }
                    yield params

    @pytest.mark.parametrize(
        "compute_unit, backend, config",
        itertools.product(
            compute_units,
            backends,
            lut_config_generator.__func__()
        ),
    )
    def test_builder_stress(self, compute_unit, backend, config):

        indices, lut, shape = (
            config["indices"],
            config["lut"],
            config["shape"],
        )

        def build(x):
            y = mb.constexpr_lut_to_dense(
                indices=indices,
                lut=lut,
                shape=shape.astype(np.uint32),
            )
            return mb.add(x=x, y=y)

        expected_output_types = (
            *shape,
            types.numpy_type_to_builtin_type(lut.dtype),
        )

        t = np.random.rand(*shape).astype(lut.dtype)
        decompressed_constant = constexpr_ops.constexpr_lut_to_dense.decompress(
            lut, indices, shape
        )
        expected_outputs = t + decompressed_constant

        input_placeholders = {
            "x": mb.placeholder(shape=shape),
        }
        input_values = {"x": t}
        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            minimum_deployment_target=ct.target.iOS16,
        )

        # validate that the constexpr op is not removed by any graph pass
        prog = mlmodel._mil_program
        if "constexpr_lut_to_dense" not in get_op_types_in_program(prog):
            raise AssertionError("Invalidated: Test Failed")


@pytest.mark.skipif(
    ct.utils._macos_version() < (13, 0),
    reason="ConstExpr ops available from macOS13 onwards.",
)
class TestConstexprSparseToDense:
    @pytest.mark.parametrize(
        "compute_unit, backend", itertools.product(compute_units, backends)
    )
    def test_builder_to_backend_smoke(self, compute_unit, backend):

        t = np.array(range(4)).reshape(4, 1).astype(np.float32)
        decompressed_constant = np.array([1, 2, 0, 4]).reshape(4, 1).astype(np.float32)
        input_placeholders = {
            "x": mb.placeholder(shape=t.shape),
        }
        input_values = {"x": t}

        def build(x):
            nonzero_data = np.array([1, 2, 4]).astype(np.float32)
            mask = np.array([11]).astype(np.uint8)
            shape = np.array([4, 1]).astype(np.uint32)
            y = mb.constexpr_sparse_to_dense(
                nonzero_data=nonzero_data, mask=mask, shape=shape
            )
            return mb.add(x=x, y=y)

        expected_output_types = (4, 1, types.fp32)
        expected_outputs = t + decompressed_constant.astype(np.float32)

        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            minimum_deployment_target=ct.target.iOS16,
        )

        # validate that the constexpr op is not removed by any graph pass
        prog = mlmodel._mil_program
        assert "constexpr_sparse_to_dense" in get_op_types_in_program(prog)

    @ssa_fn
    def test_builder_eval(self):
        v = mb.constexpr_sparse_to_dense(
            nonzero_data=np.array([1.0, 2.0, 4.0]),
            mask=np.array([11]).astype(np.uint8),
            shape=np.array(
                [
                    4,
                ]
            ).astype(np.uint32),
        )
        np.testing.assert_allclose(np.float32([1.0, 2.0, 0.0, 4.0]), v.val)

    @staticmethod
    def sparse_config_generator():
        np.random.seed(1999)

        for nonzero_data_dtype in [np.float32]:  # [np.uint8, np.int8]:
            # float16 already covered under backends parameterization
            # Not possible to write 8-bit tests since no other op consumes uint8/int8 tensors
            for output_rank in range(1, 6):
                output_shape = np.random.randint(low=2, high=5, size=output_rank)
                output_size = np.prod(output_shape)
                nBytes = np.ceil(output_size / 8).astype(np.int32)

                mask = np.random.randint(low=255, size=nBytes, dtype=np.uint8)
                bitarray = np.unpackbits(mask, bitorder="little")
                while any(bitarray[i] != 0 for i in range(output_size, len(bitarray))):
                    mask = np.random.randint(low=255, size=nBytes, dtype=np.uint8)
                    bitarray = np.unpackbits(mask, bitorder="little")

                nonzero_size = np.sum(
                    np.where(np.unpackbits(mask, bitorder="little") != 0, 1, 0)
                )

                if nonzero_data_dtype == np.uint8:
                    nonzero_data = np.random.randint(
                        low=255, size=nonzero_size, dtype=np.uint8
                    )
                elif nonzero_data_dtype == np.int8:
                    nonzero_data = np.random.randint(
                        low=-128, high=127, size=nonzero_size, dtype=np.int8
                    )
                else:
                    nonzero_data = np.random.rand(nonzero_size).astype(
                        nonzero_data_dtype
                    )

                params = {
                    "nonzero_data": nonzero_data,
                    "shape": output_shape,
                    "mask": mask,
                }
                yield params

    @pytest.mark.parametrize(
        "compute_unit, backend, config",
        itertools.product(
            compute_units,
            backends,
            sparse_config_generator.__func__()
        ),
    )
    def test_builder_stress(self, compute_unit, backend, config):

        nonzero_data, mask, shape = (
            config["nonzero_data"],
            config["mask"],
            config["shape"],
        )

        def build(x):
            y = mb.constexpr_sparse_to_dense(
                nonzero_data=nonzero_data,
                mask=mask,
                shape=shape.astype(np.uint32),
            )
            return mb.add(x=x, y=y)

        expected_output_types = (
            *shape,
            types.numpy_type_to_builtin_type(nonzero_data.dtype),
        )

        t = np.random.rand(*shape).astype(nonzero_data.dtype)
        decompressed_constant = constexpr_ops.constexpr_sparse_to_dense.decompress(
            nonzero_data, mask, shape
        )
        expected_outputs = t + decompressed_constant

        input_placeholders = {
            "x": mb.placeholder(shape=shape),
        }
        input_values = {"x": t}
        mlmodel = run_compare_builder(
            build,
            input_placeholders,
            input_values,
            expected_output_types,
            expected_outputs,
            compute_unit=compute_unit,
            backend=backend,
            minimum_deployment_target=ct.target.iOS16,
        )

        # validate that the constexpr op is not removed by any graph pass
        prog = mlmodel._mil_program
        if "constexpr_sparse_to_dense" not in get_op_types_in_program(prog):
            raise AssertionError("Invalidated: Test Failed")
