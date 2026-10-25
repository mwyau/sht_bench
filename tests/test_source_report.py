"""Source-report exports retain workload identity and build provenance."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from sht_bench import source_audit, source_report
from sht_bench.colors import BACKEND_COLORS
from sht_bench.source_adapters import Workload


def test_figure_gallery_small_and_empty_reports(tmp_path):
    stale=tmp_path/'figures-0001.md'
    stale.write_text('old generated page')
    assert source_report.figure_gallery(tmp_path,[])==([],[])
    assert not stale.exists()
    figures=[f'case-{i}.png' for i in range(48)]
    lines,pages=source_report.figure_gallery(tmp_path,figures)
    assert pages==[]
    assert len(lines)==48
    assert all(f'[![{name}]({name})]({name})' in line for name,line in zip(figures,lines))
    assert not list(tmp_path.glob('figures-*.md'))


def test_figure_gallery_pages_retain_every_figure_and_navigation(tmp_path):
    figures=[f'case-{i}.png' for i in range(97)]
    index,pages=source_report.figure_gallery(tmp_path,figures)
    assert pages==['figures-0001.md','figures-0002.md','figures-0003.md']
    contents=[(tmp_path/page).read_text() for page in pages]
    assert [text.count('[![') for text in contents]==[48,48,1]
    for name in figures:
        assert sum(text.count(f'[![{name}]({name})]({name})') for text in contents)==1
    assert all('[Results index](README.md)' in text for text in contents)
    assert '[Previous]' not in contents[0]
    assert '[Next](figures-0002.md)' in contents[0]
    assert '[Previous](figures-0001.md)' in contents[1]
    assert '[Next](figures-0003.md)' in contents[1]
    assert '[Next]' not in contents[2]
    assert all(any(f']({page})' in line for line in index) for page in pages)
    source_report.figure_gallery(tmp_path,figures[:48])
    assert not list(tmp_path.glob('figures-*.md'))


def _metrics():
    return {"relative_l2": 1e-15, "max_absolute": 1e-15, "scaled_error": 1e-15}


def _report_run(root: Path):
    run = root / "run"
    cells = run / "cells"
    cells.mkdir(parents=True)
    build = {
        "builds": {
            "ducc": {"status": "built", "sha": "ducc-source-sha", "build_kind": "local-git-source-snapshot", "module_path": "/isolated/ducc.py"},
            "rustsht": {"status": "built", "sha": "rustsht-source-sha", "build_kind": "local-git-source-snapshot", "module_path": "/isolated/rustsht.py"},
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
        ("ducc", "ducc-source-sha", "local-git-source-snapshot", "/isolated/ducc.py", 0.01),
        ("rustsht", "rustsht-source-sha", "local-git-source-snapshot", "/isolated/rustsht.py", 0.02),
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
    monkeypatch.setattr(source_audit, "audit", lambda directory, **kwargs: evidence)
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
    assert comparisons[0]["build_mode"] == "local-git-source-snapshot"
    assert comparisons[0]["reference_build_mode"] == "local-git-source-snapshot"
    assert paired[0]["process_pairs"] == "1"
    assert scaling[1]["build_mode"] == "local-git-source-snapshot"
    assert accuracy[0]["round"] == 0
    assert accuracy[0]["build_mode"] == "local-git-source-snapshot"
    assert (output / "build-provenance.json").read_text() == json.dumps(build, indent=2) + "\n"
    assert manifest["plots_requested"] is False
    assert manifest["command"].endswith("--tables-only")


def test_report_streams_compact_rows_and_reuses_audited_file_snapshot(monkeypatch, tmp_path):
    run, rows, _ = _report_run(tmp_path)
    output = tmp_path / "tables"
    snapshot = sorted((run / "cells").glob("*.json"))
    evidence = {"evidence_consistent": True, "expected_cells": 2, "missing_cells": 0,
                "coverage_complete": True, "status_counts": {"timed_successfully": 2}}

    def audit_and_add_late_file(directory, *, cell_paths):
        assert len(cell_paths) == 2
        late = dict(rows[0], id="late-cell")
        (directory / "cells" / "late-cell.json").write_text(json.dumps(late))
        return evidence

    monkeypatch.setattr(source_audit, "audit", audit_and_add_late_file)
    manifest = source_report.report(run, output, plots=False)
    outcomes = list(csv.DictReader((output / "outcomes.csv").open()))
    compact_output = tmp_path / "compact"
    compact_output.mkdir()
    compact, counts, recorded, _, _, _ = source_report.collect_for_report(
        run, compact_output, cell_paths=snapshot
    )

    assert manifest["raw_cells"] == recorded == 2
    assert len(outcomes) == 2
    assert counts["timed_successfully"] == 2
    assert len(compact) == 2
    assert compact[0]["maximum_validation_relative_l2"] == 1e-15
    assert compact[0].keys() == {
        "id", "backend", "source_sha", "build_mode", "workload", "threads_requested", "round",
        "median_s", "p10_s", "p90_s", "setup_s", "setup_policy", "native_setup_s",
        "setup_components_s", "native_api", "native_conventions", "first_call_policy",
        "planning_policy", "threads_effective", "thread_control", "operation_prepare_s",
        "operation_prepare_policy", "first_requested_operation_s", "first_synthesis_s",
        "first_analysis_s", "peak_rss_bytes", "maximum_validation_relative_l2",
    }
    assert not {"validation", "samples_s", "environment", "traceback"} & compact[0].keys()


def test_source_report_exports_requested_derivative_and_adjoint_validation(tmp_path):
    run = tmp_path / "run"
    cells = run / "cells"
    cells.mkdir(parents=True)
    output = tmp_path / "tables"
    output.mkdir()
    build_manifest = tmp_path / "build-manifest.json"
    build_manifest.write_text("{}\n")
    build_hash = hashlib.sha256(build_manifest.read_bytes()).hexdigest()

    cases = (
        ("gradient-valid", "gradient", "timed_successfully", {
            "passed": True,
            "tolerance": 2e-10,
            "independent_synthesis": {
                "relative_l2": 2e-12, "max_absolute": 3e-12, "scaled_error": 4e-12,
            },
        }),
        ("gradient-failed", "gradient", "numerical_validation_failed", {
            "passed": False,
            "tolerance": 2e-10,
            "independent_synthesis": {
                "relative_l2": 3e-5, "max_absolute": 2e-5, "scaled_error": 1e-5,
            },
        }),
        ("adjoint-valid", "adjoint_analysis", "timed_successfully", {
            "passed": True, "tolerance": 2e-10,
            "adjoint_lhs": 1.25, "adjoint_rhs": 1.25 + 1e-12,
            "scaled_residual": 4e-13,
        }),
        ("adjoint-failed", "adjoint_synthesis", "numerical_validation_failed", {
            "passed": False, "tolerance": 2e-10,
            "adjoint_lhs": 1.0, "adjoint_rhs": 1.5, "scaled_residual": 1/3,
        }),
    )
    for index, (cell_id, operation, status, operation_validation) in enumerate(cases):
        workload = Workload(
            "GL", 8, 12, 1, 1, operation=operation,
        ).record()
        workload["lmax"] = 1
        fixture = {
            "name": "random-triangular", "passed": True, "dtype_passed": True,
            "synthesis": {**_metrics(), "relative_l2": 1e-15},
            "full_grid_synthesis": {**_metrics(), "relative_l2": 8e-15},
            "common_reference_analysis": {**_metrics(), "relative_l2": 4e-15},
            "coefficient_recovery": {**_metrics(), "relative_l2": 2e-15},
            "map_roundtrip": {**_metrics(), "relative_l2": 6e-15},
        }
        row = {
            "id": cell_id, "backend": "rustsht", "source_sha": "rustsht-main-sha",
            "build_mode": "git-source", "workload": workload,
            "threads_requested": 1, "round": index % 3, "status": status,
            "build_manifest": str(build_manifest), "build_manifest_sha256": build_hash,
            "harness_sha256": "test-harness-sha", "validation": {"fixtures": [fixture]},
            "operation_validation": operation_validation,
            "median_s": 1e-4, "p10_s": 9e-5, "p90_s": 1.1e-4,
            "setup_s": 1e-3, "setup_policy": "setup", "planning_policy": "native plan",
            "operation_prepare_s": 1e-5, "first_synthesis_s": 1e-4,
            "first_analysis_s": 1e-4, "peak_rss_bytes": 1_000_000,
        }
        (cells / f"{cell_id}.json").write_text(json.dumps(row) + "\n")

    compact, counts, recorded, _, _, _ = source_report.collect_for_report(run, output)
    accuracy = json.loads((output / "accuracy.json").read_text())
    outcomes = {row["id"]: row for row in csv.DictReader((output / "outcomes.csv").open())}
    requested = {row["id"]: row for row in accuracy if row["validation_scope"] == "requested_operation"}
    base = [row for row in accuracy if row.get("validation_scope") == "base_analysis_synthesis"]
    compact_by_id = {row["id"]: row for row in compact}

    assert recorded == 4
    assert counts == {"timed_successfully": 2, "numerical_validation_failed": 2}
    assert len(base) == 4 * 5
    assert all(row["stage"] != "requested_adjoint_identity" for row in base)
    assert requested["gradient-valid"]["stage"] == "requested_derivative_synthesis"
    assert requested["gradient-valid"]["passed"] is True
    assert requested["gradient-valid"]["relative_l2"] == 2e-12
    assert requested["gradient-failed"]["stage"] == "requested_derivative_synthesis"
    assert requested["gradient-failed"]["passed"] is False
    assert requested["gradient-failed"]["relative_l2"] == 3e-5
    assert requested["adjoint-valid"]["stage"] == "requested_adjoint_identity"
    assert requested["adjoint-valid"]["passed"] is True
    assert requested["adjoint-valid"]["scaled_residual"] == 4e-13
    assert "relative_l2" not in requested["adjoint-valid"]
    assert requested["adjoint-failed"]["stage"] == "requested_adjoint_identity"
    assert requested["adjoint-failed"]["passed"] is False
    assert requested["adjoint-failed"]["scaled_residual"] == 1/3
    assert compact_by_id["gradient-valid"]["maximum_validation_relative_l2"] == 2e-12
    assert compact_by_id["adjoint-valid"]["maximum_validation_relative_l2"] == 8e-15
    assert outcomes["gradient-failed"]["status"] == "numerical_validation_failed"
    assert outcomes["adjoint-failed"]["status"] == "numerical_validation_failed"


def test_report_uses_fixed_backend_colors_and_separate_build_styles():
    assert source_report.COLORS == {name: BACKEND_COLORS[name] for name in ("ducc", "shtns", "rustsht", "pyspharm")}
    assert source_report.LINE_STYLES == {"local-git-source-snapshot": "--", "git-source": "--", "conda-forge": "-."}
    assert source_report.MARKERS == {"local-git-source-snapshot": "s", "git-source": "s", "conda-forge": "D"}
    assert source_report._series_label(
        "pyspharm", "conda-forge", "artifact-hash",
        {"version": "1.0.9", "build_string": "np2py314h2766e9b_1017"},
    ) == "pyspharm conda-forge 1.0.9-np2py314h2766e9b_1017"
    sha = "abcdef0123456789abcdef0123456789abcdef01"
    assert source_report.revision_label(sha) == sha[:10]
    assert source_report.revision_label("rustsht==1.2.3") == "rustsht==1.2.3"
    assert source_report.revision_label("ABCDEF0123456789abcdef0123456789abcdef01") == "ABCDEF0123456789abcdef0123456789abcdef01"


def test_matched_comparisons_include_pyspharm_only_on_exact_workload_identity():
    workload = {
        "grid": "GL", "nlat": 34, "nlon": 65, "lmax": 16, "mmax": 16,
        "spin": 0, "precision": "float32", "operation": "synthesis",
        "allocation": "allocating", "batch": 1, "phi0": 0.0,
    }
    reference = {
        "backend": "ducc", "source_sha": "ducc-sha", "build_mode": "git-source",
        "workload": workload, "threads_requested": 1, "round": 0, "median_s": 2e-4,
    }
    pyspharm = {
        "backend": "pyspharm", "source_sha": "conda-artifact", "build_mode": "conda-forge",
        "workload": workload, "threads_requested": 1, "round": 0, "median_s": 3e-4,
    }
    different_precision = {
        **pyspharm, "workload": {**workload, "precision": "float64"},
    }

    rows = source_report.matched_comparisons([reference, pyspharm, different_precision])

    assert len(rows) == 1
    assert rows[0]["backend"] == "pyspharm"
    assert rows[0]["build_mode"] == "conda-forge"
    assert math.isclose(rows[0]["ratio"], 1.5, rel_tol=1e-12)
