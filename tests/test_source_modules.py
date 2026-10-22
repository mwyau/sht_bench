"""Unit coverage for the source-matrix conventions and execution harness."""
from __future__ import annotations

import json
import importlib
import subprocess
import types
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sht_bench.source_adapters import Adapter, Unsupported, Workload, capability, indices
from sht_bench.source_validation import acceptable, errors, nodes, reference_map, reference_points, validate


def workload(*, lmax=1, mmax=1, spin=0):
    return Workload("GL", 8, 12, lmax, mmax, spin=spin)


def coefficient(w, l, m, value=1, component=0):
    a = np.zeros((w.components, w.size), dtype=np.complex128)
    a[component, indices(w).index((l, m))] = value
    return a


def test_reference_scalar_analytical_modes():
    w = workload(lmax=2)
    theta = np.array([0.37, 0.91, 1.42])
    phi = np.array([0.0, np.pi / 3, np.pi / 2])

    y00 = reference_points(w, coefficient(w, 0, 0), theta, phi)[0]
    np.testing.assert_allclose(y00, np.full(3, 1 / np.sqrt(4 * np.pi)), atol=2e-15)

    y10 = reference_points(w, coefficient(w, 1, 0), theta, phi)[0]
    np.testing.assert_allclose(y10, np.sqrt(3 / (4 * np.pi)) * np.cos(theta), atol=2e-15)

    q11 = -np.sqrt(3 / (8 * np.pi)) * np.sin(theta)
    y11_real = reference_points(w, coefficient(w, 1, 1), theta, phi)[0]
    y11_imag = reference_points(w, coefficient(w, 1, 1, 1j), theta, phi)[0]
    np.testing.assert_allclose(y11_real, 2 * q11 * np.cos(phi), atol=2e-15)
    np.testing.assert_allclose(y11_imag, -2 * q11 * np.sin(phi), atol=2e-15)


def test_normalized_legendre_recurrence_matches_degree_two_forms():
    w = workload(lmax=2, mmax=2)
    theta = np.array([0.23, 0.74, 1.31, 2.7])
    phi = np.array([0.0, 0.3, 1.8, 2.5])
    x, s = np.cos(theta), np.sin(theta)
    expected = {
        (2, 0): np.sqrt(5 / (16 * np.pi)) * (3 * x * x - 1),
        (2, 1): -np.sqrt(15 / (8 * np.pi)) * s * x * 2 * np.cos(phi),
        (2, 2): np.sqrt(15 / (32 * np.pi)) * s * s * 2 * np.cos(2 * phi),
    }
    for (l, m), values in expected.items():
        actual = reference_points(w, coefficient(w, l, m), theta, phi)[0]
        np.testing.assert_allclose(actual, values, rtol=2e-14, atol=2e-15)


def test_reference_spin_one_analytical_modes_and_pole_limits():
    w = workload(lmax=1, spin=1)
    theta = np.array([0.2, 0.8, 2.1])
    phi = np.array([0.0, 0.7, 2.0])
    magnitude = np.sqrt(3 / (8 * np.pi)) * np.sin(theta)

    e10 = reference_points(w, coefficient(w, 1, 0), theta, phi)
    b10 = reference_points(w, coefficient(w, 1, 0, component=1), theta, phi)
    np.testing.assert_allclose(e10[0], magnitude, atol=2e-15)
    np.testing.assert_allclose(e10[1], 0, atol=2e-15)
    np.testing.assert_allclose(b10[0], 0, atol=2e-15)
    np.testing.assert_allclose(b10[1], magnitude, atol=2e-15)

    pole_theta = np.array([0.0, 0.0, np.pi, np.pi])
    pole_phi = np.array([0.0, np.pi / 2, 0.0, np.pi / 2])
    e11 = reference_points(w, coefficient(w, 1, 1), pole_theta, pole_phi)
    b11 = reference_points(w, coefficient(w, 1, 1, component=1), pole_theta, pole_phi)
    pole_scale = np.sqrt(3 / (4 * np.pi))
    pole_sign = np.cos(pole_theta)
    np.testing.assert_allclose(e11[0], pole_scale * pole_sign * np.cos(pole_phi), atol=2e-15)
    np.testing.assert_allclose(e11[1], -pole_scale * np.sin(pole_phi), atol=2e-15)
    np.testing.assert_allclose(b11[0], pole_scale * np.sin(pole_phi), atol=2e-15)
    np.testing.assert_allclose(b11[1], pole_scale * pole_sign * np.cos(pole_phi), atol=2e-15)


@pytest.mark.parametrize("grid", ["GL", "CC", "F1"])
@pytest.mark.parametrize("spin", [0, 1])
def test_reference_map_matches_independent_point_recurrence(grid, spin):
    w = Workload(grid, 8, 12, 3, 3, spin=spin, phi0=0.37)
    rng = np.random.default_rng(9123 + spin)
    a = (rng.normal(size=(w.components, w.size)) + 1j * rng.normal(size=(w.components, w.size))) / 4
    a[:, :w.lmax + 1] = a[:, :w.lmax + 1].real
    if spin:
        a[:, :1] = 0

    theta = np.repeat(nodes(w), w.nlon)
    phi = np.tile(w.phi0 + np.arange(w.nlon) * 2 * np.pi / w.nlon, w.nlat)
    points = reference_points(w, a, theta, phi).reshape(w.components, w.nlat, w.nlon)
    np.testing.assert_allclose(reference_map(w, a), points, rtol=2e-14, atol=3e-15)


def test_reference_map_has_analytic_scalar_normalization_and_longitude_phase():
    w = Workload("GL", 8, 12, 1, 1, phi0=0.37)
    theta = nodes(w)[:, None]
    phi = w.phi0 + np.arange(w.nlon)[None, :] * 2 * np.pi / w.nlon

    y00 = reference_map(w, coefficient(w, 0, 0))[0]
    np.testing.assert_allclose(y00, np.full((w.nlat, w.nlon), 1 / np.sqrt(4 * np.pi)), atol=2e-15)

    y11 = reference_map(w, coefficient(w, 1, 1))[0]
    expected = -2 * np.sqrt(3 / (8 * np.pi)) * np.sin(theta) * np.cos(phi)
    np.testing.assert_allclose(y11, expected, rtol=2e-14, atol=2e-15)


class _ReferenceAdapter:
    """Small exact real-harmonic adapter for exercising the validation gate."""

    def __init__(self, w, corrupt=False):
        self.w = w
        theta = np.repeat(nodes(w), w.nlon)
        phi = np.tile(np.arange(w.nlon) * 2 * np.pi / w.nlon, w.nlat)
        coeff = []
        for k, (l, m) in enumerate(indices(w)):
            for part in ((0, 1) if m else (0,)):
                a = np.zeros((1, w.size), dtype=np.complex128)
                a[0, k] = 1j if part else 1
                coeff.append(a)
        self._coeff = coeff
        self._matrix = np.column_stack([
            reference_points(w, a, theta, phi).ravel() for a in coeff
        ])
        self._pinv = np.linalg.pinv(self._matrix)
        self.corrupt = corrupt

    def encode(self, a):
        return np.asarray(a)

    def decode(self, a):
        return np.asarray(a)

    def synthesis(self, a):
        theta = np.repeat(nodes(self.w), self.w.nlon)
        phi = np.tile(np.arange(self.w.nlon) * 2 * np.pi / self.w.nlon, self.w.nlat)
        out = reference_points(self.w, a, theta, phi).reshape(1, self.w.nlat, self.w.nlon)
        return out + (0.125 if self.corrupt else 0)

    def analysis(self, m):
        solution = self._pinv @ np.asarray(m).ravel()
        a = np.zeros((1, self.w.size), dtype=np.complex128)
        cursor = 0
        for k, (_, order) in enumerate(indices(self.w)):
            if order:
                a[0, k] = solution[cursor] + 1j * solution[cursor + 1]
                cursor += 2
            else:
                a[0, k] = solution[cursor]
                cursor += 1
        return a


def test_validation_gate_accepts_reference_adapter_and_rejects_corruption():
    w = workload(lmax=1, mmax=1)
    good = validate(_ReferenceAdapter(w))
    assert good["status"] == "supported_and_validated"
    assert good["fixtures"]
    assert all(row["passed"] for row in good["fixtures"])

    bad = validate(_ReferenceAdapter(w, corrupt=True))
    assert bad["status"] == "numerical_validation_failed"
    assert any(not row["passed"] for row in bad["fixtures"])


def test_error_metrics_and_tolerance_gate():
    exact = errors(np.array([1.0, -2.0]), np.array([1.0, -2.0]))
    assert acceptable(exact, 1e-12)
    different = errors(np.array([1.0, -1.0]), np.array([1.0, -2.0]))
    assert not acceptable(different, 1e-4)


def test_workload_and_backend_capability_reasons_are_explicit():
    scalar = workload()
    assert capability("ducc", scalar) is None
    assert capability("pyspharm", replace(scalar, precision="float64")) == "Spherepack published API uses float32/complex64"
    assert capability("pyspharm", replace(scalar, spin=1, precision="float32")) is None
    assert capability("shtns", replace(scalar, nlat=16)) == "SHTns SIMD grid requires at least 32 latitude nodes"
    assert capability("shtns", replace(scalar, grid="F1", nlat=32)) is None
    assert capability("pyspharm", replace(scalar, grid="F1", precision="float32")) == "no native F1 path"
    with pytest.raises(ValueError, match="latitude count"):
        Workload("GL", 2, 3, 2, 1)
    with pytest.raises(ValueError, match="longitude count"):
        Workload("GL", 4, 2, 1, 1)
    with pytest.raises(Unsupported, match="Spherepack"):
        Adapter("pyspharm", replace(scalar, precision="float64"), 1)


def test_rust_gradient_zeros_constant_mode_in_private_copy(monkeypatch):
    from sht_bench import source_adapters

    calls = []

    class FakePlan:
        def __init__(self, grid, nlat, nlon, lmax, mmax, *, spin, phi0):
            self.spin = spin

        def synthesis_gradient(self, coeff):
            calls.append((self.spin, coeff.copy()))
            if coeff[0, 0] != 0:
                raise ValueError("spin-1 coefficient cannot contain the scalar constant mode")
            return coeff.copy()

    fake_module = types.SimpleNamespace(ShtPlan=FakePlan)
    monkeypatch.setattr(source_adapters.importlib, "import_module", lambda name: fake_module)
    w = Workload("GL", 5, 7, 2, 2, operation="gradient")
    adapter = Adapter("rustsht", w, 1)
    coefficients = np.ones((1, w.size), dtype=np.complex128)
    coefficients[0, 0] = 2.5
    original = coefficients.copy()
    result = adapter.operation(coefficients, np.zeros((1, w.nlat, w.nlon)), operation="gradient")()

    assert calls[0][0] == 1
    np.testing.assert_array_equal(calls[0][1][:, 0], 0)
    np.testing.assert_array_equal(calls[0][1][:, 1:], -original[:, 1:])
    np.testing.assert_array_equal(result, calls[0][1])
    np.testing.assert_array_equal(coefficients, original)


def _git(path: Path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, text=True, capture_output=True).stdout.strip()


def test_source_lock_latest_locked_and_dirty_checkout(tmp_path, monkeypatch):
    from sht_bench import sources

    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "--initial-branch=main")
    _git(origin, "config", "user.name", "Test")
    _git(origin, "config", "user.email", "test@example.invalid")
    (origin / "value.txt").write_text("immutable source\n")
    _git(origin, "add", "value.txt")
    _git(origin, "commit", "-m", "initial")
    sha = _git(origin, "rev-parse", "HEAD")
    monkeypatch.setattr(sources, "SOURCES", {"fake": (str(origin), "main")})

    lock = tmp_path / "source.lock.json"
    latest = sources.checkout(tmp_path / "latest", lock)
    assert latest["sources"]["fake"]["sha"] == sha
    assert json.loads(lock.read_text()) == latest

    reproduced = sources.checkout(tmp_path / "locked", lock, mode="locked")
    assert reproduced == latest
    checkout_path = tmp_path / "locked" / "fake"
    (checkout_path / "value.txt").write_text("local edit\n")
    with pytest.raises(ValueError, match="modified source checkout"):
        sources.checkout(tmp_path / "locked", lock, mode="locked")


def test_worker_keeps_setup_validation_and_first_calls_out_of_timing(tmp_path, monkeypatch):
    from bench_common import environment, timing
    from sht_bench import source_runner

    events = []
    timing_state = {"measuring": False}
    allowed = sorted(__import__("os").sched_getaffinity(0))

    class TimedAdapter:
        def __init__(self, backend, w, threads):
            events.append("setup")
            self.w = w
            self.module = types.SimpleNamespace(__file__="/tmp/source-backend.py")
            self.effective_threads = None
            self.thread_control = "test adapter"
            self.setup_policy = "test setup policy"
            self.setup_components_s = {"native_constructor": 0.001}
            self.native_setup_s = 0.001
            self.native_api = "test native API"
            self.native_conventions = {"conversion": "identity"}
            self.planning_policy = "test planning policy"

        def encode(self, a):
            return a

        def synthesis(self, a):
            events.append("transform_during_timing" if timing_state["measuring"] else "first_synthesis")
            return np.zeros((1, self.w.nlat, self.w.nlon))

        def analysis(self, m):
            events.append("first_analysis")
            return np.zeros((1, self.w.size), dtype=np.complex128)

        def canonical_synthesis_output(self, result):
            return np.asarray(result)

        def operation(self, a, m, out=None, *, operation=None):
            events.append("operation_prepare")
            selected = operation or self.w.operation
            if selected == "analysis":
                return lambda: self.analysis(m)
            return lambda: self.synthesis(a)

    monkeypatch.setattr(source_runner, "Adapter", TimedAdapter)
    monkeypatch.setattr(importlib, "import_module", lambda name: types.SimpleNamespace(__file__="/tmp/source-backend.py"))
    monkeypatch.setattr(source_runner, "physical_cores", lambda: (allowed[:1], {allowed[0]: "test-domain"}))
    monkeypatch.setattr(environment, "collect_environment", lambda **kwargs: {"mock": True})
    monkeypatch.setattr(source_runner, "validate", lambda adapter: events.append("validation") or {"status": "supported_and_validated", "tolerance": 1e-12})

    def measure(fn, **kwargs):
        events.append("measure_start")
        timing_state["measuring"] = True
        fn()
        timing_state["measuring"] = False
        events.append("measure_end")
        return 1, [0.001]

    monkeypatch.setattr(timing, "measure", measure)
    w = workload()
    task = {
        "id": "timing-boundary-test", "backend": "ducc", "workload": w.record(),
        "threads_requested": 1, "round": 0, "source_sha": "test", "build_mode": "test",
        "build_manifest": "test", "build_manifest_sha256": "test",
        "harness_sha256": source_runner.harness_hash(), "expected_import_path": "/tmp/source-backend.py",
        "repeat": 1, "min_time": 0.001,
    }
    task_path = tmp_path / "task.json"
    result_path = tmp_path / "result.json"
    task_path.write_text(json.dumps(task))
    original_affinity = set(__import__("os").sched_getaffinity(0))
    try:
        status = source_runner.worker(Namespace(task=task_path, result=result_path))
    finally:
        __import__("os").sched_setaffinity(0, original_affinity)

    assert status == 0
    result = json.loads(result_path.read_text())
    assert result["status"] == "timed_successfully"
    assert events.index("setup") < events.index("first_synthesis")
    assert events.index("first_synthesis") < events.index("first_analysis") < events.index("validation")
    assert events.index("validation") < events.index("measure_start") < events.index("transform_during_timing") < events.index("measure_end")
