import warnings, os, logging

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import ssa_fn


def range_1d_old(start, end, step):
    os.environ["range_1d_flag"] = "0"
    return mb.range_1d(start=start, end=end, step=step)


def range_1d_new(start, end, step):
    os.environ["range_1d_flag"] = "1"
    return mb.range_1d(start=start, end=end, step=step)


# https://stackoverflow.com/a/47424657/3476782
class LevelRaiser(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.WARNING:
            raise Exception("mb.range_1d triggered a warning")
        return True


class TestRange1d:
    @ssa_fn
    def test_int32_arange_dtype(self):
        """Regression: without explicit dtype, np.arange defaults to int64
        which triggers a lossy downcast warning in MIL."""
        logging.getLogger("coremltools").addFilter(LevelRaiser())
        args = [
            (np.int32(0), np.int32(10), np.int32(1)),
            (np.float32(0), np.float32(10), np.float32(1)),
            (np.float16(0), np.float16(10), np.float16(1)),
            (0, 10, 2),
        ]
        val = [range_1d_new(*a) for a in args]
        val = [range_1d_old(*a) for a in args]
