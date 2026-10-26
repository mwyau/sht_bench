from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from sht_bench.source_adapters import Workload
from sht_bench.source_resolution import resolution_cases

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_fast_scaling.py"
SPEC = importlib.util.spec_from_file_location("run_fast_scaling", SCRIPT)
fast_scaling = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(fast_scaling)


def _build():
    return {
        "status": "built", "build_kind": "git-source", "sha": "abc123",
        "python_executable": "/python", "module_path": "/module",
        "binaries": [{"path": "/module", "sha256": "binary-sha"}],
    }


def test_scenarios_keep_shtns_float32_exclusion_compact():
    args = SimpleNamespace(
        backend_values=("shtns",), grid_values=("GL",), thread_values=(1,),
        precision_values=("float32", "float64"),
    )
    cases = fast_scaling.resolution_cases(("GL",), endpoints_only=True)
    groups, exclusions = fast_scaling._scenarios(
        args, cases, {"shtns": _build()}, "manifest-sha",
    )

    assert len(groups) == 2  # one group per spin; each group reuses its plan for both directions
    assert all(group["precision"] == "float64" for group in groups)
    assert len(exclusions) == 4
    assert all(item["status"] == "unsupported" for item in exclusions)
    assert {item["reason"] for item in exclusions} == {"Python API uses float64/complex128"}


def test_dense_profile_plans_exact_high_resolution_geometry():
    gl = resolution_cases(("GL",), profile="dense-to-1024")
    cc = resolution_cases(("CC",), profile="dense-to-1024")

    assert [case["lmax"] for case in gl] == [
        32, 45, 64, 90, 128, 160, 180, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
    ]
    assert [case["lmax"] for case in cc] == [
        36, 45, 60, 90, 120, 160, 180, 192, 224, 256, 320, 360, 384, 448, 512, 640, 768, 896, 1024,
    ]
    assert all((case["nlat"], case["nlon"]) == (2 * case["lmax"], 4 * case["lmax"]) for case in gl)
    assert all((case["nlat"], case["nlon"]) == (2 * case["lmax"] + 1, 4 * case["lmax"]) for case in cc)


def test_pyspharm_degree_limit_excludes_before_workload_construction(monkeypatch):
    cases = [
        case for case in resolution_cases(("GL",), profile="dense-to-1024")
        if case["lmax"] > 640
    ]
    args = SimpleNamespace(
        backend_values=("pyspharm",), grid_values=("GL",), thread_values=(1,),
        precision_values=("float32",),
    )

    def unexpected_workload(*args, **kwargs):
        raise AssertionError("a pyspharm workload above L640 must not be constructed")

    monkeypatch.setattr(fast_scaling, "Workload", unexpected_workload)
    groups, exclusions = fast_scaling._scenarios(args, cases, {"pyspharm": _build()}, "manifest-sha")

    assert groups == []
    assert len(exclusions) == len(cases) * 2 * 2
    assert {item["status"] for item in exclusions} == {"degree_limit"}
    assert {item["max_degree"] for item in exclusions} == {640}


def test_pyspharm_admission_requires_two_times_stored_plan_headroom(monkeypatch):
    monkeypatch.setattr(fast_scaling, "preflight", lambda *args, **kwargs: {
        "memory_budget_bytes": 20_000, "persistent_lower_bound_bytes": 12_000,
        "resource_limited": False,
    })
    result = fast_scaling._memory_admission(
        "pyspharm", Workload("GL", 16, 32, 8, 8, precision="float32"), None,
    )

    assert result["stored_plan_headroom_multiplier"] == 2
    assert result["stored_plan_headroom_bytes"] == 24_000
    assert result["required_memory_bytes"] == 24_000
    assert result["resource_limited"] is True


def test_principal_figure_order_starts_with_float64():
    names = [
        f"{grid}_{precision}_spin{spin}_{operation}.png"
        for grid, precision, spin, operation in fast_scaling._figure_order()
    ]

    assert names[:8] == [
        "GL_float64_spin0_analysis.png", "GL_float64_spin0_synthesis.png",
        "GL_float64_spin1_analysis.png", "GL_float64_spin1_synthesis.png",
        "CC_float64_spin0_analysis.png", "CC_float64_spin0_synthesis.png",
        "CC_float64_spin1_analysis.png", "CC_float64_spin1_synthesis.png",
    ]
    assert len(names) == len(set(names)) == 16


def test_csv_row_keeps_three_raw_samples_and_milliseconds():
    row = fast_scaling._record_row({
        "id": "cell", "backend": "ducc", "samples_s": [0.001, 0.002, 0.003],
        "median_s": 0.002, "output_shape": [1, 12], "status": "timed_successfully",
    })

    assert row["sample_1_s"] == 0.001
    assert row["sample_3_s"] == 0.003
    assert row["median_ms"] == 2.0
    assert row["output_shape"] == "[1, 12]"


def test_csv_output_uses_lf_line_endings(tmp_path):
    path = tmp_path / "timings.csv"
    fast_scaling._write_csv(path, [{
        "id": "cell", "backend": "ducc", "samples_s": [0.001, 0.002, 0.003],
        "median_s": 0.002, "status": "timed_successfully",
    }])

    assert b"\r" not in path.read_bytes()


def test_cli_configuration_is_full_ordered_thread_ladder():
    args = fast_scaling.parser().parse_args([])
    assert fast_scaling._parse_csv(args.threads, tuple(map(str, fast_scaling.THREADS)), "threads") == (
        "1", "2", "4", "8", "16",
    )
    assert args.profile == "standard"
    assert args.min_time == 0.005
