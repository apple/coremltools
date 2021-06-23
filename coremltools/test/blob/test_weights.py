import unittest
import tempfile
import os
import shutil
import numpy as np

from coremltools.libmilstoragepython import _BlobStorageWriter as BlobWriter
from coremltools.libmilstoragepython import _BlobStorageReader as BlobReader

class WeightTest(unittest.TestCase):
    def setUp(self):
        self.working_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)

    def test_weight_blob_uint8(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([1.0, 2, 3, 4, 5], dtype=np.uint8)
        offset = writer.write_uint8_data(input_arr)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr = np.array(reader.read_uint8_data(offset))
        np.testing.assert_almost_equal(input_arr, output_arr)

    def test_weight_blob_fp16(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([1.0, 2, 3, 4, 5], dtype=np.float16)
        offset = writer.write_fp16_data(input_arr)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr = np.array(reader.read_fp16_data(offset))
        np.testing.assert_almost_equal(input_arr, output_arr)

    def test_weight_blob_fp32(self):
        writer = BlobWriter(self.working_dir + "/net.wt")
        input_arr = np.array([1.0, 2, 3, 4, 5], dtype=np.float32)
        offset = writer.write_float_data(input_arr)
        writer = None

        reader = BlobReader(self.working_dir + "/net.wt")
        output_arr = np.array(reader.read_float_data(offset))
        np.testing.assert_almost_equal(input_arr, output_arr)

if __name__ == "__main__":
    unittest.main()
