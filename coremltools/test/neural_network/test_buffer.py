import unittest
import tempfile
import os
import shutil
import pytest
import numpy as np
from coremltools.libcoremlpython import _NeuralNetworkBuffer as NetBuffer

class WeightTest(unittest.TestCase):

    def setUp(self):
        self.working_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)

    def test_weight_read_write(self):
        nn_buffer = NetBuffer(self.working_dir + '/net.wt', NetBuffer.mode.write)
        input_arr = np.array([1.0, 2, 3, 4, 5])
        offset    = nn_buffer.add_buffer(input_arr)
        output_arr = np.array(nn_buffer.get_buffer(offset))
        np.testing.assert_almost_equal(input_arr, output_arr)

    def test_weight_read_write_large(self):
        nn_buffer = NetBuffer(self.working_dir + '/net.wt', NetBuffer.mode.write)
        input_arr = np.random.rand(100000,)
        offset    = nn_buffer.add_buffer(input_arr)
        output_arr = np.array(nn_buffer.get_buffer(offset))
        np.testing.assert_almost_equal(input_arr, output_arr)

    def test_weight_write_then_read(self):
        nn_buffer = NetBuffer(self.working_dir + '/net.wt', NetBuffer.mode.write)
        input_arr = np.random.rand(100000,)
        offset    = nn_buffer.add_buffer(input_arr)
        nn_buffer = NetBuffer(self.working_dir + '/net.wt', NetBuffer.mode.read)
        output_arr = np.array(nn_buffer.get_buffer(offset))
        np.testing.assert_almost_equal(input_arr, output_arr)

    def test_weight_write_then_append(self):
        nn_buffer = NetBuffer(self.working_dir + '/net.wt', NetBuffer.mode.write)
        input_arr = np.random.rand(100000,)
        offset    = nn_buffer.add_buffer(input_arr)
        nn_buffer = NetBuffer(self.working_dir + '/net.wt', NetBuffer.mode.append)
        input_arr = np.random.rand(100000,)
        offset    = nn_buffer.add_buffer(input_arr)
        output_arr = np.array(nn_buffer.get_buffer(offset))
        np.testing.assert_almost_equal(input_arr, output_arr)


if __name__ == '__main__':
    unittest.main()
