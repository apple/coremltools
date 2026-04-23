# Copyright (c) 2026, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""Regression tests for MLModel temporary .mlpackage cleanup.

Historically `MLModel` constructed from a `Model_pb2` spec allocated a temp
`.mlpackage` that was only reclaimed at interpreter exit via `atexit`. This
caused tens of gigabytes of leaked temp directories when many short-lived
MLModel objects were created — for example during activation-statistics
collection in `linear_quantize_activations`. These tests pin the eager
cleanup contract (`MLModel.__del__` / `MLModel._close`).

Parallel-test safety: these tests never mutate ``$TMPDIR`` or
``tempfile.tempdir`` and only inspect ``model.package_path`` for the
specific MLModel instances they construct. Any other test running in
parallel against the same shared temp dir is therefore invisible to the
assertions here.
"""
import gc
import os

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct


def _build_spec_mlmodel():
    """Build a tiny mlprogram MLModel from a spec so that MLModel.__init__
    hits the `_create_mlpackage` path (i.e. the one that allocates tmp)."""

    class _Tiny(nn.Module):
        def forward(self, x):
            return x * 2.0

    traced = torch.jit.trace(_Tiny().eval(), torch.zeros(1, 4))
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, 4), dtype=np.float32)],
        outputs=[ct.TensorType(name="y", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )


class TestMLModelTempCleanup:
    def test_close_removes_temp_package(self):
        """``MLModel._close`` deletes this model's temp .mlpackage and is
        idempotent. We only inspect ``model.package_path`` — never the
        shared tmpdir — so this is safe under parallel test execution."""
        model = _build_spec_mlmodel()
        pkg = model.package_path
        assert pkg is not None
        assert os.path.exists(pkg), "expected temp mlpackage on disk"

        model._close()
        assert not os.path.exists(pkg), (
            f"_close should rmtree temp mlpackage, still present: {pkg}"
        )
        # Idempotent
        model._close()
        assert model.package_path is None

    def test_del_removes_temp_package(self):
        """Letting the MLModel go out of scope triggers cleanup via __del__.
        We captured ``pkg`` before ``del`` and only assert on that specific
        path, which makes the assertion independent of any other temp dirs
        that may exist in $TMPDIR from parallel tests."""
        model = _build_spec_mlmodel()
        pkg = model.package_path
        assert os.path.exists(pkg)

        del model
        gc.collect()
        assert not os.path.exists(pkg), (
            f"__del__ should rmtree temp mlpackage, still present: {pkg}"
        )

    def test_no_leak_across_many_mlmodel_constructions(self):
        """Simulates the activation-calibration workload that motivated the
        fix: constructing many temp MLModels in a row should not accumulate
        tmp dirs once each object has been released.

        We record each model's ``package_path`` before releasing it and then
        assert that every one of those specific paths is gone. This is
        strictly stronger than counting ``tmp*.mlpackage`` in $TMPDIR and
        doesn't race with other tests that may also be creating temp
        packages in the same directory."""
        pkgs = []
        for _ in range(5):
            m = _build_spec_mlmodel()
            pkgs.append(m.package_path)
            del m
            gc.collect()

        leaked = [p for p in pkgs if p is not None and os.path.exists(p)]
        assert not leaked, (
            f"{len(leaked)}/{len(pkgs)} temp mlpackage dirs survived __del__: "
            f"{leaked}"
        )

    def test_user_provided_package_path_is_not_deleted(self, tmp_path):
        """Loading a user-owned .mlpackage should NEVER trigger the cleanup
        path — only temp packages that MLModel itself created. ``tmp_path``
        here is the pytest-provided per-test unique directory, not a
        $TMPDIR override, so this remains parallel-safe."""
        # Build once via the spec path (creates temp), then save to a
        # user-owned location and load from disk.
        temp_model = _build_spec_mlmodel()
        user_pkg = str(tmp_path / "user_owned.mlpackage")
        temp_model.save(user_pkg)
        del temp_model
        gc.collect()

        assert os.path.exists(user_pkg)
        loaded = ct.models.MLModel(user_pkg)
        assert not getattr(loaded, "is_temp_package", False), (
            "Model loaded from a user path must not be flagged as temp."
        )
        del loaded
        gc.collect()
        assert os.path.exists(user_pkg), (
            "Cleanup must only touch temp packages, not user-provided paths."
        )
