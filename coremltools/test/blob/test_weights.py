# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os
import shutil
import tempfile
import unittest

import numpy as np

from coremltools.libmilstoragepython import _BlobStorageReader as BlobReader
from coremltools.libmilstoragepython import _BlobStorageWriter as BlobWriter


class WeightTest(unittest.TestCase):
    def setUp(self):
        self.working_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)

    def test_weight_blob_int8(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([-5, -2, 0, 2, 5], dtype=np.int8)
        offset = writer.write_int8_data(input_arr)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr = np.array(reader.read_int8_data(offset), np.int8)
        np.testing.assert_equal(input_arr, output_arr)

    def test_weight_blob_uint8(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        offset = writer.write_uint8_data(input_arr)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr = np.array(reader.read_uint8_data(offset), np.uint8)
        np.testing.assert_almost_equal(input_arr, output_arr)

    def test_weight_blob_fp16(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([2.3, 4.6, 7.9], dtype=np.float16)
        input_arr_to_bytes_uint16 = np.frombuffer(input_arr.tobytes(), np.uint16)
        offset = writer.write_fp16_data(input_arr_to_bytes_uint16)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr_uint16 = np.array(reader.read_fp16_data(offset), np.uint16)
        output_arr = np.frombuffer(output_arr_uint16.tobytes(), np.float16)
        np.testing.assert_almost_equal(input_arr, output_arr)

    def test_weight_blob_fp32(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([1.0, 2.4, 3.9, -4.8, 5.2], dtype=np.float32)
        offset = writer.write_float_data(input_arr)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr = np.array(reader.read_float_data(offset), np.float32)
        np.testing.assert_almost_equal(input_arr, output_arr)

if __name__ == "__main__":
    unittest.main()
