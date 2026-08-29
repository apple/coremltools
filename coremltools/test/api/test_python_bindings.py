# Copyright (c) 2026, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import subprocess
import sys

import pytest

import coremltools as ct
from coremltools.converters.mil import Builder as mb


_NUMPY_OWNER_RELEASE_SCRIPT = """
import ctypes
import sys
import time
import weakref

import coremltools as ct
import numpy as np

model = ct.models.MLModel(sys.argv[1], compute_units=ct.ComputeUnit.CPU_ONLY)
input_name = model.get_spec().description.input[0].name

check_gil = ctypes.pythonapi.PyGILState_Check
check_gil.restype = ctypes.c_int
gil_states = []

input_array = np.zeros((1, 4), dtype=np.float32)
input_ref = weakref.ref(
    input_array,
    lambda _: gil_states.append(bool(check_gil())),
)
model.predict({input_name: input_array})
del input_array

deadline = time.monotonic() + 10
while input_ref() is not None and time.monotonic() < deadline:
    time.sleep(0.05)

assert input_ref() is None, "Core ML did not release the NumPy input owner"
assert gil_states == [True], f"NumPy input owner release GIL states: {gil_states}"
"""


@pytest.mark.skipif(
    ct.utils._macos_version() < (12, 0),
    reason="ML Program prediction is available only on macOS 12+",
)
def test_numpy_input_owner_is_released_with_gil(tmp_path):
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 4))],
        opset_version=ct.target.iOS16,
    )
    def program(x):
        return mb.add(x=x, y=1.0)

    model = ct.convert(
        program,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )
    model_path = tmp_path / "add_one.mlpackage"
    model.save(str(model_path))

    # Core ML may release the NumPy owner asynchronously. Use a subprocess so
    # an unsafe release is reported as a test failure instead of terminating
    # the entire pytest process.
    result = subprocess.run(
        [sys.executable, "-c", _NUMPY_OWNER_RELEASE_SCRIPT, str(model_path)],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"Prediction subprocess exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
