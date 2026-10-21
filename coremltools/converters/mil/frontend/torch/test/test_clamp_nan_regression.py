# Copyright (c) 2026, Apple Inc. All rights reserved.
# Use of this source code is governed by a BSD-3-clause license.

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import coremltools as ct


class ClampModel(torch.nn.Module):
    def forward(self, x, lower, upper):
        y = torch.clamp(x, min=lower, max=upper)
        return y, y + 1, y * 2


def _inputs(layout, dtype, case="ordered"):
    x = np.array([[-3, 0, 5], [2, 4, -1]], dtype=dtype)
    shapes = {
        "full": ((2, 3), (2, 3)),
        "row": ((3,), (3,)),
        "scalar": ((1,), (1,)),
        "expand": ((2, 1), (1, 3)),
    }
    if layout == "expand":
        x = x[:1]
    lower_shape, upper_shape = shapes[layout]
    lower = np.full(lower_shape, -2, dtype=dtype)
    upper = np.full(upper_shape, 2, dtype=dtype)
    if case == "equal":
        lower.fill(2)
    elif case == "crossed":
        lower.fill(4)
    elif case == "mixed":
        lower.flat[0] = 4
    elif case.startswith("nan_"):
        x.flat[0] = 0
        bits = int(case[4:])
        for bit, value in zip((1, 2, 4), (x, lower, upper)):
            if bits & bit:
                value.flat[0] = np.nan
    elif case == "x_posinf":
        x.flat[0] = np.inf
    elif case == "x_neginf":
        x.flat[0] = -np.inf
    elif case == "unbounded":
        lower.fill(-np.inf)
        upper.fill(np.inf)
    elif case == "inf_crossed":
        lower.fill(np.inf)
        upper.fill(-np.inf)
    elif case == "inf_equal":
        lower.fill(np.inf)
        upper.fill(np.inf)
    return tuple(torch.from_numpy(v.copy()) for v in (x, lower, upper))


PARAMS = list(itertools.product(
    ("torchscript", "torchexport"),
    ("FLOAT32", "FLOAT16"),
    ("full", "row", "scalar", "expand"),
    ("float32", "int32"),
))


@pytest.fixture(scope="module", params=PARAMS)
def compiled_clamp(request):
    frontend, precision, layout, dtype_name = request.param
    if ct.utils._macos_version() < (14, 0):
        pytest.skip("iOS17 Core ML prediction requires macOS14 or newer")
    dtype = np.dtype(dtype_name)
    args = _inputs(layout, dtype)
    model = ClampModel().eval()
    if frontend == "torchscript":
        exported = torch.jit.trace(model, args)
    else:
        exported = torch.export.export(model, args).run_decompositions({})
    mlmodel = ct.convert(
        exported,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name=name, shape=tuple(value.shape), dtype=dtype.type)
            for name, value in zip(("x", "lower", "upper"), args)
        ],
        compute_precision=getattr(ct.precision, precision),
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS17,
    )
    assert len(mlmodel.get_spec().description.input) == 3
    return model, mlmodel, layout, dtype


def _check(compiled, case):
    model, mlmodel, layout, dtype = compiled
    args = _inputs(layout, dtype, case)
    expected = model(*args)
    actual = mlmodel.predict({
        name: value.numpy() for name, value in zip(("x", "lower", "upper"), args)
    })
    for output, reference in zip(mlmodel.get_spec().description.output, expected):
        a = actual[output.name]
        r = reference.detach().numpy()
        np.testing.assert_equal(a.shape, r.shape)
        np.testing.assert_allclose(a, r, rtol=1e-3, atol=1e-4, equal_nan=True)
        if np.isfinite(r).all():
            np.testing.assert_array_equal(a, r)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(r))
        np.testing.assert_array_equal(np.isposinf(a), np.isposinf(r))
        np.testing.assert_array_equal(np.isneginf(a), np.isneginf(r))


@pytest.mark.parametrize("case", ("ordered", "equal", "crossed", "mixed"))
def test_clamp_bound_order(compiled_clamp, case):
    _check(compiled_clamp, case)


@pytest.mark.parametrize("bits", range(1, 8))
def test_clamp_nan_inputs(compiled_clamp, bits):
    if compiled_clamp[3].kind != "f":
        pytest.skip("Integer inputs cannot contain NaN")
    _check(compiled_clamp, f"nan_{bits}")


@pytest.mark.parametrize(
    "case", ("x_posinf", "x_neginf", "unbounded", "inf_crossed", "inf_equal")
)
def test_clamp_infinite_inputs(compiled_clamp, case):
    if compiled_clamp[3].kind != "f":
        pytest.skip("Integer inputs cannot contain infinity")
    _check(compiled_clamp, case)
