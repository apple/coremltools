# Copyright (c) 2026, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import ctypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

# If the calling thread holds the GIL for the whole native call, another Python
# thread is blocked for (almost) all of it; if the GIL is released, the other
# thread's longest stall is a tiny fraction of it.
_BLOCKED_FRACTION_THRESHOLD = 0.5


def _longest_python_stall_fraction(call, attempts=3):
    """
    Runs ``call`` on a background thread while this thread executes Python in a
    loop, and returns the longest interval in which this thread made no progress,
    as a fraction of the duration of ``call``.

    Only the longest interval is used, so short GIL hand-offs before and after the
    native part of ``call`` do not affect the result. The smallest value of several
    attempts is returned, so that this thread being descheduled by a busy machine
    is not mistaken for a held GIL.
    """
    return min(_longest_stall_fraction_once(call) for _ in range(attempts))


def _longest_stall_fraction_once(call):
    started = threading.Event()
    done = threading.Event()
    window = []

    def run():
        started.set()
        window.append(time.perf_counter_ns())
        call()
        window.append(time.perf_counter_ns())
        done.set()

    # Record only intervals without progress that are longer than 1 ms, to keep
    # the loop cheap and its memory bounded.
    stalls = []
    previous = time.perf_counter_ns()
    thread = threading.Thread(target=run)
    thread.start()
    started.wait()
    while not done.is_set():
        now = time.perf_counter_ns()
        if now - previous > 1_000_000:
            stalls.append((previous, now))
        previous = now
    stalls.append((previous, time.perf_counter_ns()))
    thread.join()

    begin, end = window
    longest_stall = max(
        (min(b, end) - max(a, begin) for a, b in stalls if a < end and b > begin),
        default=0,
    )
    return longest_stall / (end - begin)


@pytest.fixture(scope="module")
def slow_model():
    # Small input and output so that Python-side conversion is negligible,
    # and a chain of large matmuls so that the native prediction is long.
    dim = 1024
    weight = np.random.default_rng(0).standard_normal((dim, dim)).astype(np.float32)
    weight /= np.sqrt(dim)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim), dtype=types.fp32)])
    def prog(x):
        w = mb.const(val=weight)
        h = mb.matmul(x=x, y=x, transpose_x=True)
        for _ in range(100):
            h = mb.tanh(x=mb.matmul(x=h, y=w))
        return mb.reduce_mean(x=h, axes=[0], keep_dims=True)

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,
        compute_precision=ct.precision.FLOAT32,
    )
    x = np.random.default_rng(1).standard_normal((1, dim)).astype(np.float32)
    model.predict({"x": x})  # warm up
    return model, x


@pytest.mark.skipif(
    ct.utils._macos_version() < (12, 0),
    reason="ML Program prediction is available only on macOS 12+",
)
class TestPredictThreading:
    @staticmethod
    @pytest.mark.parametrize("holds_gil", [True, False])
    def test_stall_measurement(slow_model, holds_gil):
        # Checks that the measurement tells a GIL-holding native call from a
        # GIL-releasing one: ctypes.PyDLL keeps the GIL during a foreign call,
        # ctypes.CDLL releases it.
        model, x = slow_model
        model.predict({"x": x})
        usec = min(max(model.last_predict_duration_in_nano_seconds // 1000, 50_000), 900_000)
        libc = ctypes.PyDLL(None) if holds_gil else ctypes.CDLL(None)
        libc.usleep.argtypes = [ctypes.c_uint32]
        fraction = _longest_python_stall_fraction(lambda: libc.usleep(usec))
        assert (fraction > _BLOCKED_FRACTION_THRESHOLD) == holds_gil, fraction

    @staticmethod
    def test_predict_releases_gil(slow_model):
        model, x = slow_model
        fraction = _longest_python_stall_fraction(lambda: model.predict({"x": x}))
        assert fraction < _BLOCKED_FRACTION_THRESHOLD, (
            f"Other Python threads were blocked for {fraction:.0%} of predict()"
        )

    @staticmethod
    def test_concurrent_predictions_match_serial():
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 64), dtype=types.fp32)])
        def prog(x):
            return mb.tanh(x=mb.matmul(x=x, y=x, transpose_y=True))

        model = ct.convert(
            prog,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
            compute_precision=ct.precision.FLOAT32,
        )
        output_name = model.get_spec().description.output[0].name
        rng = np.random.default_rng(2)
        inputs = [rng.standard_normal((4, 64)).astype(np.float32) for _ in range(32)]
        expected = [model.predict({"x": x})[output_name] for x in inputs]

        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda x: model.predict({"x": x})[output_name], inputs))

        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-7)
