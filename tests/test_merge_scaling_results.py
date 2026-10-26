from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_scaling_results.py"
SPEC = importlib.util.spec_from_file_location("merge_scaling_results", SCRIPT)
merge = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(merge)


def _cell(backend: str, *, sha: str, build_hash: str, median_s: float) -> dict:
    return {
        "id": f"{backend}-cell-{sha}", "backend": backend, "grid": "GL",
        "case_id": "GL-N32", "resolution": "GL N=32", "nlat": 64, "nlon": 128,
        "lmax": 32, "mmax": 32, "precision": "float64", "spin": 0,
        "operation": "analysis", "threads_requested": 1, "process_round": 1,
        "status": "timed_successfully", "reason": None,
        "samples_s": [median_s * 0.9, median_s, median_s * 1.1],
        "median_s": median_s, "source_sha": sha,
        "build_manifest_sha256": build_hash,
    }


def _datasets():
    backends = ("ducc", "shtns", "rustsht", "pyspharm")
    baseline_builds = {
        name: {"ref": "main", "sha": f"old-{name}", "build_manifest_sha256": f"base-{name}"}
        for name in backends
    }
    baseline = {
        "schema": "sht_bench.fast_resolution_scaling.v1",
        "profile_schema": "sht_bench.source_resolution_profile.v1",
        "resolution_cases": [{"grid": "GL", "case_id": "GL-N32", "lmax": 32}],
        "configuration": {"resolution_profile": "dense-to-1024"},
        "runner_sha256": "runner", "build_manifest_sha256": "baseline-manifest",
        "builds": baseline_builds, "unsupported_and_skipped": [],
        "cells": [
            _cell("ducc", sha="ducc-sha", build_hash="base-ducc", median_s=0.002),
            _cell("shtns", sha="shtns-sha", build_hash="base-shtns", median_s=0.003),
            _cell("rustsht", sha="old-main", build_hash="base-rustsht", median_s=0.008),
            _cell("pyspharm", sha="pyspharm-sha", build_hash="base-pyspharm", median_s=0.004),
        ],
    }
    refresh_cell = _cell("rustsht", sha="new-dev", build_hash="dev-manifest", median_s=0.005)
    refresh = {
        "schema": baseline["schema"], "profile_schema": baseline["profile_schema"],
        "resolution_cases": baseline["resolution_cases"], "runner_sha256": "runner",
        "build_manifest_sha256": "dev-manifest",
        "builds": {"rustsht": {"ref": "dev", "sha": "dev-commit", "binary_sha256": "dev-binary"}},
        "unsupported_and_skipped": [], "cells": [refresh_cell],
    }
    return baseline, refresh


def _csv_row(cell: dict) -> dict[str, str]:
    return {key: str(value) for key, value in cell.items() if key not in ("samples_s",)} | {
        "sample_1_s": str(cell["samples_s"][0]),
        "sample_2_s": str(cell["samples_s"][1]),
        "sample_3_s": str(cell["samples_s"][2]),
    }


def test_merge_replaces_rustsht_only_and_keeps_other_rows_exact():
    baseline, refresh = _datasets()
    merged = merge.merge_datasets(baseline, refresh)

    assert merged["builds"]["rustsht"] == refresh["builds"]["rustsht"]
    for backend in ("ducc", "shtns", "pyspharm"):
        assert merged["builds"][backend] == baseline["builds"][backend]
        assert next(c for c in merged["cells"] if c["backend"] == backend) == next(
            c for c in baseline["cells"] if c["backend"] == backend
        )
    rust = next(c for c in merged["cells"] if c["backend"] == "rustsht")
    assert rust["source_sha"] == "new-dev"
    assert rust["median_s"] == 0.005
    assert merged["merge_provenance"]["replaced_rustsht_cell_count"] == 1

    fields = list(_csv_row(baseline["cells"][0]))
    base_rows = [_csv_row(cell) for cell in baseline["cells"]]
    refresh_rows = [_csv_row(refresh["cells"][0])]
    rows = merge.merge_csv_rows(fields, base_rows, fields, refresh_rows)
    original_nonrust = {row["id"]: row for row in base_rows if row["backend"] != "rustsht"}
    merged_by_id = {row["id"]: row for row in rows}
    assert all(merged_by_id[cell_id] == row for cell_id, row in original_nonrust.items())
    assert next(row for row in rows if row["backend"] == "rustsht")["source_sha"] == "new-dev"


def test_merge_rejects_incomplete_refresh_instead_of_falling_back():
    baseline, refresh = _datasets()
    refresh["cells"] = []
    with pytest.raises(ValueError, match="key coverage differs"):
        merge.merge_datasets(baseline, refresh)


def test_merged_saved_data_generates_exactly_sixteen_float64_first_figures(tmp_path):
    baseline, refresh = _datasets()
    merged = merge.merge_datasets(baseline, refresh)
    output = tmp_path / "merged"
    output.mkdir()
    (output / "timings.json").write_text(json.dumps(merged))

    merge._plot(output)

    figures = list((output / "plots").glob("*.png"))
    assert len(figures) == 16
    gallery = (output / "plots" / "README.md").read_text()
    names = re.findall(r"^!\[[^\]]*\]\(([^)]+)\)$", gallery, re.M)
    expected = [
        f"{grid}_{precision}_spin{spin}_{operation}.png"
        for precision in ("float64", "float32")
        for grid in ("GL", "CC")
        for spin in (0, 1)
        for operation in ("analysis", "synthesis")
    ]
    assert names == expected
    assert "RustSHT source: `dev` commit `dev-commit`." in gallery
