"""Source-report exports retain workload identity and build provenance."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from sht_bench import source_audit, source_report
from sht_bench.colors import BACKEND_COLORS
from sht_bench.source_adapters import Workload


def _metrics():
    return {"relative_l2": 1e-15, "max_absolute": 1e-15, "scaled_error": 1e-15}


def _report_run(root: Path):
    run = root / "run"
    cells = run / "cells"
    cells.mkdir(parents=True)
    build = {
        "builds": {
            "ducc": {"status": "built", "sha": "ducc-source-sha", "build_kind": "git-source", "module_path": "/isolated/ducc.py"},
            "rustsht": {"status": "built", "requirement": "rustsht==1.2.3", "build_kind": "published-wheel", "module_path": "/isolated/rustsht.py"},
        }
    }
    manifest_path = run / "build-manifest.json"
    manifest_path.write_text(json.dumps(build, indent=2) + "\n")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    workload = Workload("GL", 8, 12, 1, 1, operation="synthesis").record()
    fixture = {"name": "random-triangular", "passed": True, "dtype_passed": True,
               "synthesis": _metrics(), "full_grid_synthesis": _metrics(),
               "common_reference_analysis": _metrics(), "coefficient_recovery": _metrics(),
               "map_roundtrip": _metrics()}
    rows = []
    for backend, source_sha, build_mode, import_path, median in (
        ("ducc", "ducc-source-sha", "git-source", "/isolated/ducc.py", 0.01),
        ("rustsht", "rustsht==1.2.3", "published-wheel", "/isolated/rustsht.py", 0.02),
    ):
        row = {
            "id": f"{backend}-cell", "backend": backend, "source_sha": source_sha,
            "build_mode": build_mode, "workload": workload, "threads_requested": 1,
            "round": 0, "status": "timed_successfully", "reason": "",
            "build_manifest": str(manifest_path), "build_manifest_sha256": manifest_sha,
            "harness_sha256": "harness-digest", "import_path": import_path,
            "median_s": median, "p10_s": median * 0.9, "p90_s": median * 1.1,
            "setup_s": 0.1, "setup_policy": "adapter setup measured",
            "native_setup_s": 0.05, "setup_components_s": {"native_constructor": 0.05},
            "native_api": "native API", "native_conventions": {"conversion": "identity"},
            "first_call_policy": "allocating native A/S calls",
            "planning_policy": "public native planning policy",
            "threads_effective": 1, "thread_control": "one pinned core",
            "operation_prepare_s": 0.001, "operation_prepare_policy": "buffers prepared before timing",
            "first_requested_operation_s": None, "first_synthesis_s": 0.002,
            "first_analysis_s": 0.003, "peak_rss_bytes": 10000,
            "validation": {"fixtures": [fixture]},
        }
        cell_path = cells / f"{row['id']}.json"
        cell_path.write_text(json.dumps(row, indent=2) + "\n")
        rows.append(row)
    return run, rows, build


def test_source_report_tables_only_cli_flag(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(source_report, "report", lambda directory, output, *, plots=True: calls.append((directory, output, plots)))
    assert source_report.main([str(tmp_path / "run"), "--output", str(tmp_path / "tables"), "--tables-only"]) == 0
    assert calls == [(tmp_path / "run", tmp_path / "tables", False)]


def test_source_report_exports_outcomes_timings_pairs_and_provenance(monkeypatch, tmp_path):
    run, _, build = _report_run(tmp_path)
    evidence = {"evidence_consistent": True, "expected_cells": 2, "missing_cells": 0,
                "coverage_complete": True, "status_counts": {"timed_successfully": 2}}
    monkeypatch.setattr(source_audit, "audit", lambda directory: evidence)
    output = tmp_path / "tables"
    manifest = source_report.report(run, output, plots=False)

    outcomes = list(csv.DictReader((output / "outcomes.csv").open()))
    timings = list(csv.DictReader((output / "timings.csv").open()))
    comparisons = list(csv.DictReader((output / "matched-ratios.csv").open()))
    paired = list(csv.DictReader((output / "paired-summary.csv").open()))
    scaling = list(csv.DictReader((output / "thread-scaling.csv").open()))
    accuracy = json.loads((output / "accuracy.json").read_text())

    assert len(outcomes) == 2
    assert len(timings) == 2
    assert len(comparisons) == 1
    assert len(paired) == 1
    assert len(scaling) == 2
    assert [row["backend"] for row in outcomes] == ["ducc", "rustsht"]
    assert all({"source_sha", "build_mode", "grid", "precision", "threads", "round", "operation"} <= row.keys() for row in outcomes)
    assert timings[1]["native_api"] == "native API"
    assert timings[1]["native_conventions"] == json.dumps({"conversion": "identity"}, sort_keys=True)
    assert timings[1]["setup_components_s"] == json.dumps({"native_constructor": 0.05}, sort_keys=True)
    assert comparisons[0]["build_mode"] == "published-wheel"
    assert comparisons[0]["reference_build_mode"] == "git-source"
    assert paired[0]["process_pairs"] == "1"
    assert scaling[1]["build_mode"] == "published-wheel"
    assert accuracy[0]["round"] == 0
    assert accuracy[0]["build_mode"] in {"git-source", "published-wheel"}
    assert (output / "build-provenance.json").read_text() == json.dumps(build, indent=2) + "\n"
    assert manifest["plots_requested"] is False
    assert manifest["command"].endswith("--tables-only")


def test_report_uses_fixed_backend_colors_and_separate_build_styles():
    assert source_report.COLORS == {name: BACKEND_COLORS[name] for name in ("ducc", "shtns", "rustsht", "pyspharm")}
    assert source_report.LINE_STYLES == {"git-source": "--", "published-wheel": "-"}
    assert source_report.MARKERS == {"git-source": "s", "published-wheel": "o"}
    sha = "abcdef0123456789abcdef0123456789abcdef01"
    assert source_report.revision_label(sha) == sha[:10]
    assert source_report.revision_label("rustsht==1.2.3") == "rustsht==1.2.3"
    assert source_report.revision_label("ABCDEF0123456789abcdef0123456789abcdef01") == "ABCDEF0123456789abcdef0123456789abcdef01"
