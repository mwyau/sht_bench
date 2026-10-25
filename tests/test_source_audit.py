"""Adversarial checks for source-matrix evidence auditing."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import pytest

from sht_bench.source_adapters import Workload
from sht_bench.source_audit import audit, expected_cases, required_base_workloads, required_fixtures, required_plan_workloads


def _write_run(root: Path, *, timed=False, timed_changes=None, row_changes=None, include_missing=False, extra=False,
               secondary=False, omit_base_scope=False, omit_secondary_scope=False,
               extra_plan=False, config_changes=None):
    root.mkdir(parents=True)
    build = {"builds": {"ducc": {"status": "built", "sha": "source-sha", "build_kind": "git-source", "module_path": "/isolated/ducc.py"}}}
    manifest_path = root / "build-manifest.json"
    manifest_path.write_text(json.dumps(build))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config = {
        "harness_sha256": "harness-sha", "build_manifest_sha256": manifest_sha,
        "degrees": "1", "threads": "1", "backends": "ducc", "rounds": 1,
        "repeat": 3, "min_time": 0.02, "timeout": 600, "smoke": True, "secondary": secondary,
    }
    plan_workloads = list(required_plan_workloads(config))
    if omit_base_scope:
        plan_workloads.remove(next(iter(required_base_workloads(config))))
    if omit_secondary_scope:
        plan_workloads.remove(next(w for w in plan_workloads if w.get("batch") == 4))
    if extra_plan:
        plan_workloads.append(Workload("GL", 8, 10, 2, 2, geometry_group="extra").record())
    plan = {"workloads": plan_workloads, "backends": ["ducc"], "threads": [1], "rounds": 1}
    if config_changes:
        config.update(config_changes)
    (root / "run-config.json").write_text(json.dumps(config))
    (root / "workload-plan.json").write_text(json.dumps(plan))
    cells = root / "cells"
    cells.mkdir()

    keys = sorted(expected_cases(plan))
    if include_missing:
        keys = keys[:-1]
    timed_written = False
    for index, key in enumerate(keys):
        backend, workload_json, threads, round_index = key
        row_workload = json.loads(workload_json)
        should_time = (timed and not timed_written and row_workload["grid"] == "GL"
                       and row_workload["spin"] == 0 and row_workload["precision"] == "float32"
                       and row_workload["allocation"] == "allocating" and row_workload["operation"] == "analysis")
        identity = {
            "backend": backend,
            "workload": row_workload,
            "threads_requested": threads,
            "round": round_index,
            "source_sha": "source-sha" + manifest_sha + "harness-sha",
        }
        cell_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        row = {
            "id": cell_id,
            "backend": backend,
            "workload": row_workload,
            "threads_requested": threads,
            "round": round_index,
            "source_sha": "source-sha",
            "build_mode": "git-source",
            "build_manifest": str(manifest_path),
            "build_manifest_sha256": manifest_sha,
            "harness_sha256": "harness-sha",
            "repeat": 3,
            "min_time": 0.02,
            "status": "unsupported",
            "reason": "fixture cell is intentionally unsupported",
        }
        if should_time:
            row.update(_timed_evidence(row["workload"]))
            timed_written = True
            if timed_changes:
                row.update(timed_changes)
        if row_changes and index == 0:
            row.update(row_changes)
        (cells / f"{cell_id}.json").write_text(json.dumps(row))

    if extra:
        row = {
            "backend": "ducc",
            "workload": Workload("GL", 8, 12, 2, 2).record(),
            "threads_requested": 1,
            "round": 0,
            "source_sha": "source-sha",
            "build_mode": "git-source",
            "build_manifest": str(manifest_path),
            "build_manifest_sha256": manifest_sha,
            "harness_sha256": "harness-sha",
            "repeat": 3,
            "min_time": 0.02,
            "status": "unsupported",
            "reason": "extra fixture cell",
        }
        identity = {"backend": row["backend"], "workload": row["workload"], "threads_requested": 1,
                    "round": 0, "source_sha": "source-sha" + manifest_sha + "harness-sha"}
        cell_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        row["id"] = cell_id
        (cells / f"{cell_id}.json").write_text(json.dumps(row))
    return root


def _metrics(value=0.0):
    return {"relative_l2": value, "max_absolute": value, "scaled_error": value}


def _timed_evidence(workload):
    w = Workload(**workload)
    tolerance = 5e-5 if w.precision == "float32" else 2e-10
    fixtures = []
    for name in sorted(required_fixtures(w)):
        fixture = {
            "name": name,
            "passed": True,
            "dtype_passed": True,
            "synthesis": _metrics(),
            "coefficient_recovery": _metrics(),
            "map_roundtrip": _metrics(),
        }
        if name == "random-triangular":
            fixture["full_grid_synthesis"] = _metrics()
            fixture["common_reference_analysis"] = _metrics()
        fixtures.append(fixture)
    samples = [0.01, 0.02, 0.03]
    return {
        "status": "timed_successfully",
        "import_path": "/isolated/ducc.py",
        "threads_effective": 1,
        "thread_control": "one physical core pinned",
        "validation": {"status": "supported_and_validated", "tolerance": tolerance, "fixtures": fixtures},
        "samples_s": samples,
        "median_s": statistics.median(samples),
        "p10_s": float(np.percentile(samples, 10)),
        "p90_s": float(np.percentile(samples, 90)),
        "std_s": float(np.std(samples)),
        "iterations": 1,
        "setup_s": 0.1,
        "first_synthesis_s": 0.1,
        "first_analysis_s": 0.1,
        "operation_prepare_s": 0.0,
        "peak_rss_bytes": 1000,
        "environment": {"process_affinity": [0], "physical_cpus": 16, "l3_domains": {"0": "l3-0"}},
    }


def test_audit_detects_missing_and_unrequested_cell_coverage(tmp_path):
    missing = audit(_write_run(tmp_path / "missing", include_missing=True))
    assert missing["coverage_complete"] is False
    assert missing["missing_cells"] == 1

    extra = audit(_write_run(tmp_path / "extra", extra=True))
    assert extra["coverage_complete"] is False
    assert extra["unexpected_cells"] == 1
    assert any("not requested" in issue["problem"] for issue in extra["evidence_issues"])


def test_audit_rejects_plan_that_omits_required_base_scope(tmp_path):
    result = audit(_write_run(tmp_path / "base-omission", omit_base_scope=True))
    assert result["coverage_complete"] is True
    assert result["missing_cells"] == 0
    assert result["evidence_consistent"] is False
    issue = next(issue for issue in result["evidence_issues"] if "omits required geometry" in issue["problem"])
    assert issue["missing_required_workloads"] == 1


def test_audit_rejects_plan_that_omits_required_secondary_scope(tmp_path):
    result = audit(_write_run(tmp_path / "secondary-omission", secondary=True, omit_secondary_scope=True))
    assert result["coverage_complete"] is True
    assert result["missing_cells"] == 0
    assert result["evidence_consistent"] is False
    issue = next(issue for issue in result["evidence_issues"] if "omits required geometry" in issue["problem"])
    assert issue["missing_required_workloads"] == 1


def test_audit_rejects_extra_unrequested_planned_workloads(tmp_path):
    result = audit(_write_run(tmp_path / "extra-plan", extra_plan=True))
    assert result["coverage_complete"] is True
    assert result["missing_cells"] == 0
    assert result["evidence_consistent"] is False
    issue = next(issue for issue in result["evidence_issues"] if "unrequested geometry" in issue["problem"])
    assert issue["unexpected_workloads"] == 1


def test_audit_normalizes_whitespace_in_selection_configuration(tmp_path):
    run = _write_run(tmp_path / "whitespace-config", config_changes={
        "grids": " GL,CC,F1 ",
        "precisions": " float64,float32 ",
        "spins": " 0,1 ",
        "orders": " full,quarter ",
        "geometry_groups": " minimum,quadrature,extended,simd,atmospheric,regular ",
        "threads": " 1 ",
        "backends": " ducc ",
        "allocations": " allocating,preallocated ",
    })
    result = audit(run)
    assert result["coverage_complete"] is True
    assert result["evidence_consistent"] is True


def test_audit_resolves_relative_build_manifest_from_result_directory(tmp_path):
    run = _write_run(tmp_path / "relative-manifest")
    for path in (run / "cells").glob("*.json"):
        row = json.loads(path.read_text())
        row["build_manifest"] = "build-manifest.json"
        path.write_text(json.dumps(row))
    result = audit(run)
    assert result["coverage_complete"] is True
    assert result["evidence_consistent"] is True


def test_explicit_relative_build_manifest_remains_relative_to_working_directory(tmp_path, monkeypatch):
    run = _write_run(tmp_path / "explicit-relative-manifest")
    monkeypatch.chdir(run)
    result = audit(run, build_manifest=Path("build-manifest.json"))
    assert result["coverage_complete"] is True
    assert result["evidence_consistent"] is True


def test_audit_rejects_forged_timed_status_with_failed_metrics(tmp_path):
    timed_changes = {"validation": {"status": "supported_and_validated", "tolerance": 5e-5, "fixtures": []}}
    run = _write_run(tmp_path / "forged", timed=True, timed_changes=timed_changes)
    result = audit(run)
    assert result["coverage_complete"] is True
    assert result["evidence_consistent"] is False
    assert any("missing or unexpected validation fixtures" in issue["problem"] for issue in result["evidence_issues"])


def test_audit_accepts_complete_valid_timing_evidence(tmp_path):
    run = _write_run(tmp_path / "valid", timed=True)
    result = audit(run)
    assert result["coverage_complete"] is True
    assert result["evidence_consistent"] is True


def test_filtered_plan_requires_selected_workloads_and_allocation_scope():
    config = {
        "degrees": "16,64,255",
        "smoke": True,
        "secondary": False,
        "grids": "GL,CC",
        "precisions": "float32",
        "spins": "0,1",
        "orders": "full",
        "geometry_groups": "minimum,quadrature,extended,regular",
        "batches": "1,4",
    }
    selected = list(required_base_workloads(config))
    assert selected
    assert all(item["grid"] in {"GL", "CC"} for item in selected)
    assert all(item["precision"] == "float32" for item in selected)
    assert all(item["mmax"] == item["lmax"] for item in selected)
    assert all(item["geometry_group"] in {"minimum", "quadrature", "extended", "regular"} for item in selected)
    assert {item["batch"] for item in selected} == {1, 4}

    plan = {
        "workloads": selected,
        "backends": ["ducc", "pyspharm"],
        "threads": [1],
        "rounds": 3,
        "allocations": ["allocating"],
    }
    assert len(set(expected_cases(plan))) == len(selected) * 2 * 2 * 3


def test_quarter_selection_has_no_distinct_l1_order():
    config = {
        "degrees": "1",
        "smoke": True,
        "secondary": False,
        "grids": "GL",
        "precisions": "float32",
        "spins": "0",
        "orders": "quarter",
        "geometry_groups": "minimum",
    }
    assert list(required_base_workloads(config)) == []


def test_audit_rejects_failed_numerical_metrics_even_when_fixture_claims_pass(tmp_path):
    evidence = _timed_evidence(Workload("GL", 8, 12, 1, 1).record())
    evidence["validation"]["fixtures"][0]["synthesis"]["relative_l2"] = 1.0
    run = _write_run(tmp_path / "bad-metrics", timed=True, timed_changes=evidence)
    result = audit(run)
    assert result["evidence_consistent"] is False
    assert any("invalid numerical metrics" in issue["problem"] for issue in result["evidence_issues"])


def test_audit_recomputes_raw_timing_median(tmp_path):
    evidence = _timed_evidence(Workload("GL", 8, 12, 1, 1).record())
    evidence["median_s"] = 100.0
    run = _write_run(tmp_path / "bad-median", timed=True, timed_changes=evidence)
    result = audit(run)
    assert result["evidence_consistent"] is False
    assert any("incorrect timing summary: median_s" in issue["problem"] for issue in result["evidence_issues"])


def test_audit_detects_mixed_harness_digest(tmp_path):
    run = _write_run(tmp_path / "mixed-harness", row_changes={"harness_sha256": "another-harness"})
    result = audit(run)
    assert result["evidence_consistent"] is False
    assert any("different harness or build manifest" in issue["problem"] for issue in result["evidence_issues"])
