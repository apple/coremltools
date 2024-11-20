# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os
import shutil
import tempfile

import numpy as np
import pytest

import coremltools as ct
from coremltools import _SPECIFICATION_VERSION_IOS_18
from coremltools.converters.mil import mil
from coremltools.converters.mil.converter import mil_convert as _mil_convert
from coremltools.converters.mil.mil.builder import Builder as mb
from coremltools.libmilstoragepython import _BlobStorageReader as BlobReader
from coremltools.libmilstoragepython import _BlobStorageWriter as BlobWriter


class TestWeightBlob:
    @classmethod
    def setup_class(cls):
        cls.working_dir = tempfile.mkdtemp()

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls.working_dir):
            shutil.rmtree(cls.working_dir)

    def test_weight_blob_int4(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        # All values in input_arr should be within range of int4, although they are stored in int8.
        input_arr1 = np.array([-8, -2, 0, 2, 7], dtype=np.int8)
        offset1 = writer.write_int4_data(input_arr1)
        input_arr2 = np.array([3, -8, 5, 7, -6], dtype=np.int8)
        offset2 = writer.write_int4_data(input_arr2)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr1 = reader.read_int4_data(offset1)
        output_arr2 = reader.read_int4_data(offset2)
        np.testing.assert_equal(input_arr1, output_arr1)
        np.testing.assert_equal(input_arr2, output_arr2)

    def test_weight_blob_int4_invalid(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([-80, -2, 0, 2, 7], dtype=np.float32)
        with pytest.raises(
            ValueError, match="Value -80 is outside allowed subbyte datatype range \[-8, 7\]."
        ):
            writer.write_int4_data(input_arr)

    @pytest.mark.parametrize("nbits", (1, 2, 3, 4, 6))
    def test_weight_blob_unsigned_sub_byte(self, nbits):
        writer = BlobWriter(self.working_dir + "/net.wt")
        # All values in input_arr are within range of uint{nbits}, but stored in uint8.
        input_arr1 = np.random.randint(0, 2**nbits, (5,), dtype=np.uint8)
        write_method = getattr(writer, f"write_uint{nbits}_data")
        offset1 = write_method(input_arr1)
        input_arr2 = np.random.randint(0, 2**nbits, (5,), dtype=np.uint8)
        offset2 = write_method(input_arr2)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        read_method = getattr(reader, f"read_uint{nbits}_data")
        output_arr1 = read_method(offset1)
        output_arr2 = read_method(offset2)
        np.testing.assert_equal(input_arr1, output_arr1)
        np.testing.assert_equal(input_arr2, output_arr2)

    @pytest.mark.parametrize("nbits", (1, 2, 3, 4, 6))
    def test_weight_blob_unsigned_sub_byte_invalid(self, nbits):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([1, 80, 2, 0, 2])
        with pytest.raises(
            ValueError,
            match=f"Value 80 is outside allowed subbyte datatype range \[0, {2 ** nbits - 1}\].",
        ):
            getattr(writer, f"write_uint{nbits}_data")(input_arr)

    def test_weight_blob_int8(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([-5, -2, 0, 2, 5], dtype=np.int8)
        offset = writer.write_int8_data(input_arr)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr = reader.read_int8_data(offset)
        np.testing.assert_equal(input_arr, output_arr)

    def test_weight_blob_uint8(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        offset = writer.write_uint8_data(input_arr)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr = reader.read_uint8_data(offset)
        np.testing.assert_almost_equal(input_arr, output_arr)

    def test_weight_blob_int16(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([-5, -2, 0, 2, 5], dtype=np.int16)
        offset = writer.write_int16_data(input_arr)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr = reader.read_int16_data(offset)
        np.testing.assert_equal(input_arr, output_arr)

    def test_weight_blob_int32(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([-5, -2, 0, 2, 5], dtype=np.int32)
        offset = writer.write_int32_data(input_arr)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr = reader.read_int32_data(offset)
        np.testing.assert_equal(input_arr, output_arr)

    def test_weight_blob_uint16(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
        offset = writer.write_uint16_data(input_arr)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr = reader.read_uint16_data(offset)
        np.testing.assert_almost_equal(input_arr, output_arr)

    def test_weight_blob_uint32(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        offset = writer.write_uint32_data(input_arr)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr = reader.read_uint32_data(offset)
        np.testing.assert_almost_equal(input_arr, output_arr)

    def test_weight_blob_fp16(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([2.3, 4.6, 7.9], dtype=np.float16)
        input_arr_to_bytes_uint16 = np.frombuffer(input_arr.tobytes(), np.uint16)
        offset = writer.write_fp16_data(input_arr_to_bytes_uint16)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr_uint16 = reader.read_fp16_data(offset)
        output_arr = np.frombuffer(output_arr_uint16.tobytes(), np.float16)
        np.testing.assert_almost_equal(input_arr, output_arr)

    def test_weight_blob_fp32(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([1.0, 2.4, 3.9, -4.8, 5.2], dtype=np.float32)
        offset = writer.write_float_data(input_arr)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr = reader.read_float_data(offset)
        np.testing.assert_almost_equal(input_arr, output_arr)


@pytest.mark.skipif(ct.utils._macos_version() < (15, 0),
                    reason="Multi-function only supported on macOS 15+")
class TestWeightIDSharing:
    @staticmethod
    def test_single_function():
        @mb.program(
            input_specs=[mb.TensorSpec((500,))],
            opset_version=ct.target.iOS16,
        )
        def prog(x):
            val = np.random.rand(
                500,
            )
            const_1 = mb.const(val=val, name="const_1")
            const_2 = mb.const(val=val, name="const_2")
            const_3 = mb.const(val=val, name="const_3")

            # const 1 and 2 share the same weight id, so they should be serialized
            # as the same blob value
            const_1.op.weight_id = "0"
            const_2.op.weight_id = "0"

            x = mb.add(x=x, y=const_1)
            x = mb.add(x=x, y=const_2)
            x = mb.add(x=x, y=const_3)

            return x

        # skip all passes to avoid running the const_deduplicate pass
        prog.skip_all_passes = True
        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS16,
        )

        mil_file = open(os.path.join(mlmodel.get_compiled_model_path(), "model.mil"))
        mil_txt = mil_file.read()
        # In the above model, const_1 and const_2 are going to share the same blob file value.
        assert (
            'tensor<fp32, [500]> const_1 = const()[name = tensor<string, []>("const_1"), val = tensor<fp32, [500]>(BLOBFILE(path = tensor<string, []>("@model_path/weights/weight.bin"), offset = tensor<uint64, []>(64)))];'
            in mil_txt
        )
        assert (
            'tensor<fp32, [500]> const_2 = const()[name = tensor<string, []>("const_2"), val = tensor<fp32, [500]>(BLOBFILE(path = tensor<string, []>("@model_path/weights/weight.bin"), offset = tensor<uint64, []>(64)))];'
            in mil_txt
        )
        assert (
            'tensor<fp32, [500]> const_3 = const()[name = tensor<string, []>("const_3"), val = tensor<fp32, [500]>(BLOBFILE(path = tensor<string, []>("@model_path/weights/weight.bin"), offset = tensor<uint64, []>(2176)))];'
            in mil_txt
        )
        assert "add(x = x, y = const_1)" in mil_txt
        assert "add(x = add_0, y = const_2)" in mil_txt


    @staticmethod
    def test_multi_functions():

        val = np.random.rand(
            500,
        )

        @mb.function(
            input_specs=[mb.TensorSpec((500,))],
            opset_version=ct.target.iOS16,
        )
        def func(x):
            const_1 = mb.const(val=val, name="const_1")
            const_1.op.weight_id = "0"
            return mb.add(x=x, y=const_1)

        @mb.function(
            input_specs=[mb.TensorSpec((500,))],
            opset_version=ct.target.iOS16,
        )
        def func_1(x):
            const_2 = mb.const(val=val, name="const_2")
            const_3 = mb.const(val=val, name="const_3")
            # const_3 shared the same blob file value with const_1 in another function
            const_3.op.weight_id = "0"

            x = mb.add(x=x, y=const_2)
            return mb.add(x=x, y=const_3)

        prog = mil.Program()
        prog.add_function("main", func)
        prog.add_function("func_1", func_1)
        prog.export_as_multifunction = True

        # skip all passes to avoid running the const_deduplicate pass
        prog.skip_all_passes = True
        mlmodel = _mil_convert(
            prog,
            convert_to="mlprogram",
            convert_from="milinternal",
            specification_version=_SPECIFICATION_VERSION_IOS_18,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            export_multi_functions=True,
        )

        mil_file = open(os.path.join(mlmodel.get_compiled_model_path(), "model.mil"))
        mil_txt = mil_file.read()
        # In the above model, const_1 and const_3 are going to share the same blob file value.
        assert (
            'tensor<fp32, [500]> const_3 = const()[name = string("const_3"), val = tensor<fp32, [500]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];'
            in mil_txt
        )
        assert (
            'tensor<fp32, [500]> const_2 = const()[name = string("const_2"), val = tensor<fp32, [500]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(2176)))];'
            in mil_txt
        )
        assert (
            'tensor<fp32, [500]> const_1 = const()[name = string("const_1"), val = tensor<fp32, [500]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];'
            in mil_txt
        )
        assert "add(x = x, y = const_2)" in mil_txt
        assert "add(x = add_1, y = const_3)" in mil_txt
        assert "add(x = x, y = const_1)" in mil_txt
