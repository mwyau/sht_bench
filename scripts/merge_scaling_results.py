#!/usr/bin/env python3
"""Replace one backend in a saved fast-scaling run and redraw the comparison.

The merge is keyed by the full workload identity. Measurements for all other
backends are copied from the baseline records without modification; failed
refresh records replace their baseline cells rather than falling back to old
timings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_fast_scaling import _plot  # noqa: E402

KEY_FIELDS = (
    "backend", "grid", "case_id", "lmax", "mmax", "precision", "spin",
    "operation", "threads_requested",
)
DEFINITION_FIELDS = (
    "grid", "case_id", "resolution", "nlat", "nlon", "lmax", "mmax",
    "precision", "spin", "operation", "threads_requested", "process_round",
)
INT_FIELDS = {"lmax", "mmax", "spin", "threads_requested", "process_round", "nlat", "nlon"}
PRESERVED_BACKENDS = ("ducc", "shtns", "pyspharm")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _key(record: dict[str, Any]) -> tuple[Any, ...]:
    try:
        return tuple(int(record[name]) if name in INT_FIELDS else record[name] for name in KEY_FIELDS)
    except KeyError as exc:
        raise ValueError(f"record is missing merge-key field {exc.args[0]!r}") from exc


def _index(cells: list[dict[str, Any]], label: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for cell in cells:
        key = _key(cell)
        if key in result:
            raise ValueError(f"duplicate {label} workload key: {key}")
        result[key] = cell
    return result


def _sorted_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(cells, key=lambda cell: (
        cell["grid"], int(cell["lmax"]), cell["backend"], cell["precision"],
        int(cell["spin"]), cell["operation"], int(cell["threads_requested"]),
    ))


def merge_datasets(
    baseline: dict[str, Any], refresh: dict[str, Any], *,
    baseline_label: str = "baseline", refresh_label: str = "refresh",
) -> dict[str, Any]:
    """Return a combined JSON dataset, replacing every old RustSHT key."""
    if baseline.get("schema") != refresh.get("schema"):
        raise ValueError("baseline and refresh use different result schemas")
    if baseline.get("profile_schema") != refresh.get("profile_schema"):
        raise ValueError("baseline and refresh use different resolution profile schemas")
    if baseline.get("resolution_cases") != refresh.get("resolution_cases"):
        raise ValueError("baseline and refresh resolution cases differ")
    if refresh.get("builds", {}).get("rustsht", {}).get("ref") != "dev":
        raise ValueError("RustSHT refresh build is not recorded as dev")
    if not refresh.get("builds", {}).get("rustsht", {}).get("sha"):
        raise ValueError("RustSHT refresh is missing its source SHA")
    if any(cell.get("backend") != "rustsht" for cell in refresh.get("cells", [])):
        raise ValueError("refresh dataset contains a backend other than rustsht")

    old_cells = baseline.get("cells", [])
    new_cells = refresh.get("cells", [])
    old_index = _index(old_cells, "baseline")
    new_index = _index(new_cells, "refresh")
    old_rust = {key: cell for key, cell in old_index.items() if cell["backend"] == "rustsht"}
    new_rust = {key: cell for key, cell in new_index.items() if cell["backend"] == "rustsht"}
    missing = old_rust.keys() - new_rust.keys()
    extra = new_rust.keys() - old_rust.keys()
    if missing or extra:
        raise ValueError(
            f"RustSHT refresh key coverage differs: {len(missing)} missing, {len(extra)} extra"
        )
    for key, old in old_rust.items():
        new = new_rust[key]
        if any(old.get(field) != new.get(field) for field in DEFINITION_FIELDS):
            raise ValueError(f"RustSHT workload definition changed for {key}")

    preserved = [cell for cell in old_cells if cell["backend"] != "rustsht"]
    combined_cells = _sorted_cells(preserved + list(new_rust.values()))
    combined_index = _index(combined_cells, "combined")
    if len(combined_index) != len(combined_cells):
        raise ValueError("combined data has duplicate workload keys")

    result = dict(baseline)
    result["cells"] = combined_cells
    result["builds"] = dict(baseline.get("builds", {}))
    result["builds"]["rustsht"] = refresh["builds"]["rustsht"]
    result["unsupported_and_skipped"] = [
        entry for entry in baseline.get("unsupported_and_skipped", [])
        if entry.get("backend") != "rustsht"
    ] + list(refresh.get("unsupported_and_skipped", []))
    result["status_counts"] = dict(sorted(Counter(cell["status"] for cell in combined_cells).items()))

    manifest_hashes: dict[str, list[str]] = {}
    for backend in sorted({cell["backend"] for cell in combined_cells}):
        hashes = sorted({
            str(cell["build_manifest_sha256"])
            for cell in combined_cells
            if cell["backend"] == backend and cell.get("build_manifest_sha256")
        })
        manifest_hashes[backend] = hashes
    result["build_manifest_sha256_by_backend"] = manifest_hashes

    result["merge_provenance"] = {
        "method": "replace RustSHT rows by full workload key; never fall back to baseline RustSHT timings",
        "key_fields": list(KEY_FIELDS),
        "baseline_run": {
            "label": baseline_label,
            "runner_sha256": baseline.get("runner_sha256"),
            "build_manifest_sha256": baseline.get("build_manifest_sha256"),
        },
        "rustsht_refresh_run": {
            "label": refresh_label,
            "source_ref": refresh["builds"]["rustsht"].get("ref"),
            "source_sha": refresh["builds"]["rustsht"].get("sha"),
            "runner_sha256": refresh.get("runner_sha256"),
            "build_manifest_sha256": refresh.get("build_manifest_sha256"),
        },
        "preserved_backends": list(PRESERVED_BACKENDS),
        "preserved_cell_counts": {
            backend: sum(cell["backend"] == backend for cell in preserved)
            for backend in PRESERVED_BACKENDS
        },
        "replaced_rustsht_cell_count": len(new_rust),
        "rustsht_refresh_status_counts": dict(sorted(Counter(
            cell["status"] for cell in new_rust.values()
        ).items())),
    }
    return result


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def merge_csv_rows(
    baseline_fields: list[str], baseline_rows: list[dict[str, str]],
    refresh_fields: list[str], refresh_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Replace RustSHT CSV rows while preserving every other row dictionary."""
    if baseline_fields != refresh_fields:
        raise ValueError("baseline and refresh CSV columns differ")
    old_index = _index(baseline_rows, "baseline CSV")
    new_index = _index(refresh_rows, "refresh CSV")
    old_rust = {key: row for key, row in old_index.items() if row["backend"] == "rustsht"}
    new_rust = {key: row for key, row in new_index.items() if row["backend"] == "rustsht"}
    if old_rust.keys() != new_rust.keys():
        raise ValueError(
            f"RustSHT refresh CSV key coverage differs: "
            f"{len(old_rust.keys() - new_rust.keys())} missing, "
            f"{len(new_rust.keys() - old_rust.keys())} extra"
        )
    for key, old in old_rust.items():
        new = new_rust[key]
        if any(old.get(field) != new.get(field) for field in DEFINITION_FIELDS):
            raise ValueError(f"RustSHT CSV workload definition changed for {key}")
    rows = [row for row in baseline_rows if row["backend"] != "rustsht"] + list(new_rust.values())
    return sorted(rows, key=lambda row: (
        0 if row["grid"] == "GL" else 1,
        int(row["lmax"]), row["backend"], row["precision"], int(row["spin"]),
        row["operation"], int(row["threads_requested"]),
    ))


def merge_runs(baseline_dir: Path, refresh_dir: Path, output_dir: Path) -> dict[str, Any]:
    baseline_dir = baseline_dir.expanduser().resolve()
    refresh_dir = refresh_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    for name in ("timings.json", "timings.csv", "statuses.json", "resolution-profile.json", "build-manifest.json", "run-config.json"):
        if not (baseline_dir / name).is_file():
            raise FileNotFoundError(f"baseline run lacks {name}: {baseline_dir}")
        if not (refresh_dir / name).is_file():
            raise FileNotFoundError(f"refresh run lacks {name}: {refresh_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    baseline = _read_json(baseline_dir / "timings.json")
    refresh = _read_json(refresh_dir / "timings.json")
    if _read_json(baseline_dir / "resolution-profile.json") != _read_json(refresh_dir / "resolution-profile.json"):
        raise ValueError("resolution-profile.json differs between runs")
    merged = merge_datasets(
        baseline, refresh, baseline_label=str(baseline_dir), refresh_label=str(refresh_dir),
    )

    base_fields, base_rows = _read_csv(baseline_dir / "timings.csv")
    refresh_fields, refresh_rows = _read_csv(refresh_dir / "timings.csv")
    merged_rows = merge_csv_rows(base_fields, base_rows, refresh_fields, refresh_rows)
    merged_json_ids = {cell["id"] for cell in merged["cells"]}
    if merged_json_ids != {row["id"] for row in merged_rows}:
        raise ValueError("merged JSON and CSV cell identities differ")
    base_nonrust_rows = {row["id"]: row for row in base_rows if row["backend"] != "rustsht"}
    merged_rows_by_id = {row["id"]: row for row in merged_rows}
    if any(merged_rows_by_id[cell_id] != row for cell_id, row in base_nonrust_rows.items()):
        raise ValueError("a non-RustSHT CSV row changed during merge")

    baseline_manifest_bytes = (baseline_dir / "build-manifest.json").read_bytes()
    refresh_manifest_bytes = (refresh_dir / "build-manifest.json").read_bytes()
    baseline_manifest = json.loads(baseline_manifest_bytes)
    refresh_manifest = json.loads(refresh_manifest_bytes)
    if "rustsht" not in refresh_manifest.get("builds", {}):
        raise ValueError("RustSHT refresh build manifest has no rustsht entry")
    combined_manifest = dict(baseline_manifest)
    combined_manifest["builds"] = dict(baseline_manifest.get("builds", {}))
    combined_manifest["builds"]["rustsht"] = refresh_manifest["builds"]["rustsht"]
    combined_manifest["source_build_manifests"] = {
        "baseline": {
            "path": str(baseline_dir / "build-manifest.json"),
            "sha256": _sha256(baseline_manifest_bytes),
        },
        "rustsht_refresh": {
            "path": str(refresh_dir / "build-manifest.json"),
            "sha256": _sha256(refresh_manifest_bytes),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_manifest_path = output_dir / "build-manifest.json"
    _write_json(combined_manifest_path, combined_manifest)
    merged["merge_provenance"]["combined_build_manifest_sha256"] = _sha256(combined_manifest_path.read_bytes())
    merged["merge_provenance"]["baseline_run_config"] = _read_json(baseline_dir / "run-config.json")
    merged["merge_provenance"]["rustsht_refresh_run_config"] = _read_json(refresh_dir / "run-config.json")

    shutil.copyfile(baseline_dir / "build-manifest.json", output_dir / "baseline-build-manifest.json")
    shutil.copyfile(refresh_dir / "build-manifest.json", output_dir / "rustsht-dev-build-manifest.json")
    shutil.copyfile(baseline_dir / "resolution-profile.json", output_dir / "resolution-profile.json")
    _write_json(output_dir / "timings.json", merged)
    with (output_dir / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=base_fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged_rows)

    failures = [cell for cell in merged["cells"] if cell["status"] != "timed_successfully"]
    status = {
        "schema": 1,
        "unsupported_and_skipped": merged["unsupported_and_skipped"],
        "execution_failures_and_resource_limits": [
            {key: cell.get(key) for key in (
                "id", "backend", "grid", "case_id", "precision", "spin",
                "operation", "threads_requested", "status", "reason",
            )}
            for cell in failures
        ],
    }
    _write_json(output_dir / "statuses.json", status)
    _write_json(output_dir / "merge-provenance.json", merged["merge_provenance"])
    _plot(output_dir)
    return merged


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True, help="complete baseline result directory")
    p.add_argument("--rustsht-refresh", type=Path, required=True, help="RustSHT-only dev result directory")
    p.add_argument("--output", type=Path, required=True, help="new merged result directory")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    merged = merge_runs(args.baseline, args.rustsht_refresh, args.output)
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "cells": len(merged["cells"]),
        "status_counts": merged["status_counts"],
        "rustsht_sha": merged["builds"]["rustsht"].get("sha"),
        "figures": 16,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
