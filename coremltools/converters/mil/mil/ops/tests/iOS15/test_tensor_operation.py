import logging

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.testing_utils import ssa_fn


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
        mb.range_1d(start=0, end=10, step=2)
