#!/usr/bin/env python3
"""Run the lightweight, exploratory GL/CC resolution scaling benchmark.

Native backends are reused through :mod:`sht_bench.source_adapters`. This
runner performs only shape, dtype and finite-value smoke checks; it does not
generate a harmonic reference or run numerical audits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sht_bench.source_adapters import (  # noqa: E402
    MODULES,
    Adapter,
    Unsupported,
    Workload,
    capability,
    fixtures,
)
from sht_bench.source_resolution import (  # noqa: E402
    PROFILE_NAMES,
    PROFILE_SCHEMA,
    case_reference,
    resolution_cases,
    selected_threads,
)
from sht_bench.source_resources import (  # noqa: E402
    apply_address_space_limit,
    preflight,
)

BACKENDS = ("ducc", "shtns", "rustsht", "pyspharm")
THREADS = (1, 2, 4, 8, 16)
THREAD_EXECUTION_ORDER = (1, 8, 16, 2, 4)
PRECISIONS = ("float32", "float64")
OPERATIONS = ("analysis", "synthesis")
SCHEMA = "sht_bench.fast_resolution_scaling.v1"
DEFAULT_BUILD = ROOT / "results/gl-cc-resolution-scaling-build/build-manifest.json"
CSV_FIELDS = (
    "id", "backend", "grid", "case_id", "resolution", "nlat", "nlon",
    "lmax", "mmax", "precision", "spin", "operation", "threads_requested",
    "process_round", "status", "reason", "sample_1_s", "sample_2_s",
    "sample_3_s", "median_s", "median_ms", "warmup_s", "setup_s",
    "native_setup_s", "calibration_iterations", "output_shape", "output_dtype",
    "source_sha", "source_origin", "source_ref", "build_kind", "module_path",
    "binary_sha256", "build_manifest_sha256",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _append_progress(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _parse_csv(value: str, allowed: tuple, name: str) -> tuple:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)) or any(item not in allowed for item in values):
        raise ValueError(f"--{name} must be a non-empty, unique subset of {','.join(map(str, allowed))}")
    return values


def _load_builds(path: Path, backends: tuple[str, ...]) -> tuple[dict, str]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    records = manifest.get("builds", {})
    selected = {}
    for backend in backends:
        build = records.get(backend, {})
        entry = dict(build)
        entry["build_manifest"] = str(path.resolve())
        entry["build_manifest_sha256"] = _sha(raw)
        if build.get("status") == "built":
            python = Path(str(build.get("python_executable", "")))
            module = Path(str(build.get("module_path", "")))
            if not python.is_file():
                entry["status"] = "build_failed"
                entry["error"] = f"backend Python executable is missing: {python}"
            elif not module.exists():
                entry["status"] = "build_failed"
                entry["error"] = f"backend module path is missing: {module}"
            else:
                for binary in build.get("binaries", []):
                    binary_path = Path(binary["path"])
                    if not binary_path.is_file() or _sha(binary_path.read_bytes()) != binary.get("sha256"):
                        entry["status"] = "build_failed"
                        entry["error"] = f"installed binary is missing or changed: {binary_path}"
                        break
        selected[backend] = entry
    return selected, _sha(raw)


def _build_identity(entry: dict) -> dict:
    binaries = entry.get("binaries", [])
    return {
        key: entry.get(key)
        for key in (
            "status", "build_kind", "build_origin", "origin", "ref", "sha",
            "requirement", "python_executable", "python", "python_version", "numpy",
            "distribution_name", "distribution_version", "module_path", "native_api",
        )
    } | {
        "binary_sha256": [binary.get("sha256") for binary in binaries],
        "binary_paths": [binary.get("path") for binary in binaries],
        "resolved_conda_package": {
            key: entry.get("resolved_conda_package", {}).get(key)
            for key in ("name", "version", "build_string", "channel", "url", "sha256")
        } if entry.get("resolved_conda_package") else None,
        "conda_packages": [
            {key: package.get(key) for key in ("name", "version", "build", "channel", "sha256")}
            for package in entry.get("conda_packages", [])
        ],
    }


def _case_short(case: dict) -> dict:
    return case_reference(case)


def _case_ladder(case: dict) -> int:
    return int(case["lmax"])


def _working_floor(w: Workload) -> int:
    real_bytes = 4 if w.precision == "float32" else 8
    complex_bytes = 2 * real_bytes
    maps = w.components * w.nlat * w.nlon * real_bytes
    coefficients = w.components * w.size * complex_bytes
    # Four full map and coefficient buffers is a deliberately simple lower
    # admission estimate. The backend-specific preflight adds Spherepack's
    # stored-plan arrays when that implementation needs them.
    return 4 * (maps + coefficients)


def _memory_admission(backend: str, workload: Workload, cap_bytes: int | None) -> dict:
    observed = preflight(backend, workload, limit_bytes=cap_bytes)
    floor = _working_floor(workload)
    budget = observed.get("memory_budget_bytes")
    required = floor
    reason = "estimated transform buffers exceed observed available memory or explicit memory budget"
    if backend == "pyspharm" and observed.get("persistent_lower_bound_bytes") is not None:
        # Spherepack's stored plan holds four large arrays and the public API
        # exposes no plan-free or computed mode. Keep two times that measured
        # lower bound available, as well as the generic transform-buffer floor.
        plan_headroom = 2 * int(observed["persistent_lower_bound_bytes"])
        observed["stored_plan_headroom_multiplier"] = 2
        observed["stored_plan_headroom_bytes"] = plan_headroom
        required = max(required, plan_headroom)
        reason = "pyspharm stored-plan two-times headroom or estimated transform buffers exceed observed available memory or explicit memory budget"
    observed.update({
        "working_floor_bytes": floor,
        "required_memory_bytes": required,
        "resource_limited": bool(observed.get("resource_limited")),
    })
    if budget is not None and required > budget:
        observed["resource_limited"] = True
        observed["reason"] = reason
    return observed


def _cell_id(backend: str, case: dict, precision: str, spin: int, operation: str, threads: int, build_hash: str) -> str:
    return _sha(_canonical({
        "backend": backend, "case_id": case["case_id"], "precision": precision,
        "spin": spin, "operation": operation, "threads_requested": threads,
        "build_manifest_sha256": build_hash,
    }))[:20]


def _base_cell(
    *, backend: str, case: dict, precision: str, spin: int, operation: str,
    threads: int, build: dict, build_hash: str, status: str, reason: str | None = None,
) -> dict:
    w = Workload(
        grid=case["grid"], nlat=case["nlat"], nlon=case["nlon"],
        lmax=case["lmax"], mmax=case["lmax"], spin=spin,
        precision=precision, operation=operation,
    )
    binary = (build.get("binaries") or [{}])[0]
    record = {
        "id": _cell_id(backend, case, precision, spin, operation, threads, build_hash),
        "backend": backend, "grid": case["grid"], "case_id": case["case_id"],
        "resolution": case["resolution"], "nlat": w.nlat, "nlon": w.nlon,
        "lmax": w.lmax, "mmax": w.mmax, "precision": precision, "spin": spin,
        "operation": operation, "threads_requested": threads, "process_round": 1,
        "status": status, "reason": reason, "samples_s": [], "median_s": None,
        "source_sha": build.get("sha", build.get("requirement")),
        "build_kind": build.get("build_kind", build.get("build_origin")),
        "source_origin": build.get("origin", build.get("build_origin")),
        "source_ref": build.get("ref"), "module_path": build.get("module_path"),
        "binary_sha256": binary.get("sha256"),
        "build_manifest_sha256": build_hash,
    }
    return record


def _worker(task_path: Path, result_path: Path) -> int:
    """Execute both directions in one process to reuse the native grid plan."""
    import numpy as np

    task = json.loads(task_path.read_text())
    backend = task["backend"]
    case = task["case"]
    build = task["build"]
    q = task["threads_requested"]
    precision = task["precision"]
    spin = task["spin"]
    outcomes: list[dict] = []
    shared = {"setup_s": None, "native_setup_s": None, "import_s": None}

    def save() -> None:
        _write_json(result_path, {"task_id": task["task_id"], "shared": shared, "outcomes": outcomes})

    try:
        from sht_bench.source_runner import physical_cores

        cores, domains = physical_cores()
        if q > len(cores):
            raise Unsupported(f"requested {q} threads but only {len(cores)} physical CPUs are available")
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cores[:q])
        shared["cpu_affinity"] = cores[:q]
        shared["l3_domains"] = {str(cpu): domains[cpu] for cpu in cores[:q]}
        shared["thread_environment"] = {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES",
                "RAYON_NUM_THREADS", "DUCC0_NUM_THREADS",
            )
        }

        started = time.perf_counter()
        module = __import__(MODULES[backend])
        shared["import_s"] = time.perf_counter() - started
        if Path(module.__file__).resolve() != Path(build["module_path"]).resolve():
            raise RuntimeError(f"import path {module.__file__!r} differs from build manifest")

        memory_cap = task.get("memory_budget_bytes")
        shared["allocation_limit"] = apply_address_space_limit(memory_cap)
        w = Workload(
            grid=case["grid"], nlat=case["nlat"], nlon=case["nlon"],
            lmax=case["lmax"], mmax=case["lmax"], spin=spin,
            precision=precision, operation="synthesis",
        )
        started = time.perf_counter()
        adapter = Adapter(backend, w, q)
        shared["setup_s"] = time.perf_counter() - started
        shared["native_setup_s"] = adapter.native_setup_s
        shared["setup_components_s"] = adapter.setup_components_s
        shared["setup_policy"] = adapter.setup_policy
        shared["planning_policy"] = adapter.planning_policy
        shared["native_api"] = adapter.native_api
        shared["thread_control"] = adapter.thread_control
        shared["import_path"] = adapter.module.__file__

        canonical_coefficients = next(fixtures(w))[1]
        native_coefficients = adapter.encode(canonical_coefficients)
        dtype = np.dtype(precision)
        rng = np.random.default_rng(20261025 + w.lmax * 5 + spin)
        canonical_map = rng.standard_normal((w.components, w.nlat, w.nlon)).astype(dtype)

        for operation in task["operations"]:
            base = task["base_cells"][operation]
            try:
                fn = adapter.operation(native_coefficients, canonical_map, operation=operation)
                warm_start = time.perf_counter()
                native_output = fn()
                warmup_s = time.perf_counter() - warm_start
                if operation == "synthesis":
                    output = adapter.canonical_synthesis_output(native_output)
                    expected_shape = (w.components, w.nlat, w.nlon)
                    expected_dtype = dtype
                else:
                    output = adapter.decode(native_output)
                    output = np.asarray(output)
                    if w.spin == 0 and output.ndim == 1:
                        output = output[None, :]
                    expected_shape = (w.components, w.size)
                    expected_dtype = np.dtype(w.cdtype)
                output = np.asarray(output)
                if output.shape != expected_shape:
                    raise ValueError(f"output shape {output.shape} != expected {expected_shape}")
                if output.dtype != expected_dtype:
                    raise ValueError(f"output dtype {output.dtype} != expected {expected_dtype}")
                if not np.isfinite(output).all():
                    raise ValueError("output contains non-finite values")

                target_s = float(task["min_time"])
                iterations = 1
                while True:
                    start_ns = time.perf_counter_ns()
                    for _ in range(iterations):
                        fn()
                    elapsed_s = (time.perf_counter_ns() - start_ns) / 1e9
                    if elapsed_s >= target_s or iterations >= (1 << 16):
                        break
                    scale = max(2, min(16, math.ceil(target_s / max(elapsed_s, 1e-12))))
                    iterations *= scale
                samples = []
                for _ in range(3):
                    start_ns = time.perf_counter_ns()
                    for _ in range(iterations):
                        fn()
                    samples.append((time.perf_counter_ns() - start_ns) / 1e9 / iterations)
                outcome = {
                    **base, "status": "timed_successfully", "reason": None,
                    "samples_s": samples, "median_s": float(median(samples)),
                    "warmup_s": warmup_s, "setup_s": shared["setup_s"],
                    "native_setup_s": shared["native_setup_s"],
                    "calibration_iterations": iterations,
                    "output_shape": list(expected_shape), "output_dtype": str(expected_dtype),
                }
            except Unsupported as exc:
                outcome = {**base, "status": "unsupported", "reason": str(exc)}
            except MemoryError as exc:
                outcome = {**base, "status": "resource_limited", "reason": str(exc) or "worker memory allocation failed"}
            except Exception as exc:
                outcome = {**base, "status": "execution_failed", "reason": f"{type(exc).__name__}: {exc}"}
            outcomes.append(outcome)
            save()
    except Unsupported as exc:
        for operation in task["operations"]:
            base = task["base_cells"][operation]
            outcomes.append({**base, "status": "unsupported", "reason": str(exc)})
        save()
    except MemoryError as exc:
        for operation in task["operations"]:
            base = task["base_cells"][operation]
            outcomes.append({**base, "status": "resource_limited", "reason": str(exc) or "worker memory allocation failed"})
        save()
    except Exception as exc:
        for operation in task["operations"]:
            base = task["base_cells"][operation]
            outcomes.append({**base, "status": "execution_failed", "reason": f"{type(exc).__name__}: {exc}"})
        save()
    return 0


def _record_row(record: dict) -> dict:
    samples = record.get("samples_s", [])
    return {
        **{key: record.get(key) for key in CSV_FIELDS if key not in (
            "sample_1_s", "sample_2_s", "sample_3_s", "median_ms", "output_shape",
        )},
        "sample_1_s": samples[0] if len(samples) > 0 else None,
        "sample_2_s": samples[1] if len(samples) > 1 else None,
        "sample_3_s": samples[2] if len(samples) > 2 else None,
        "median_ms": record.get("median_s") * 1000 if record.get("median_s") is not None else None,
        "output_shape": json.dumps(record.get("output_shape")) if record.get("output_shape") is not None else None,
    }


def _write_csv(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for cell in cells:
            writer.writerow(_record_row(cell))


def _write_final(output: Path, config: dict, profile: dict, builds: dict, cells: list[dict], exclusions: list[dict], script_hash: str) -> None:
    result = {
        "schema": SCHEMA,
        "benchmark_label": "exploratory timing-only measurements; no numerical accuracy claim",
        "protocol": {
            "warmups": 1, "timed_samples": 3, "process_rounds": 1,
            "calibration_target_s": config["min_time_s"],
            "timed_setup_policy": "native setup and plans are constructed before the warm call and excluded from transform latency",
            "allocation": "native allocating transforms",
            "spectral_order": "full triangular, mmax=lmax",
        },
        "configuration": config,
        "profile_schema": PROFILE_SCHEMA,
        "resolution_cases": profile,
        "build_manifest_sha256": config["build_manifest_sha256"],
        "runner_sha256": script_hash,
        "builds": {name: _build_identity(entry) for name, entry in builds.items()},
        "cells": sorted(cells, key=lambda c: (c["grid"], c["lmax"], c["backend"], c["precision"], c["spin"], c["operation"], c["threads_requested"])),
        "unsupported_and_skipped": exclusions,
        "status_counts": dict(sorted(_count_statuses(cells).items())),
    }
    _write_json(output / "timings.json", result)
    _write_csv(output / "timings.csv", result["cells"])
    failed = [cell for cell in result["cells"] if cell["status"] != "timed_successfully"]
    status_record = {
        "schema": 1,
        "unsupported_and_skipped": exclusions,
        "execution_failures_and_resource_limits": [
            {key: cell.get(key) for key in (
                "id", "backend", "grid", "case_id", "precision", "spin",
                "operation", "threads_requested", "status", "reason",
            )}
            for cell in failed
        ],
    }
    _write_json(output / "statuses.json", status_record)


def _count_statuses(cells: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for cell in cells:
        counts[cell["status"]] += 1
    return counts


def _load_progress(progress_path: Path, final_path: Path, resume: bool) -> list[dict]:
    if final_path.exists():
        data = json.loads(final_path.read_text())
        return list(data.get("cells", []))
    if not progress_path.exists():
        return []
    records = []
    for line in progress_path.read_text().splitlines():
        if line.strip():
            records.extend(json.loads(line).get("outcomes", []))
    if not resume and records:
        raise ValueError("output already contains partial timings; pass --resume or choose another output directory")
    return records


def _scenarios(args, cases: list[dict], builds: dict, build_hash: str):
    grouped: dict[tuple, dict] = {}
    exclusions: dict[tuple, dict] = {}
    requested_threads = tuple(int(value) for value in args.thread_values)
    selected_precision = tuple(args.precision_values)
    for backend in args.backend_values:
        build = builds[backend]
        if build.get("status") != "built":
            exclusions[(backend, "build")] = {
                "backend": backend, "status": "build_failed", "reason": build.get("error", "backend build is unavailable"),
            }
            continue
        supported_threads = selected_threads(backend, requested_threads)
        if backend == "pyspharm":
            unsupported_q = [q for q in requested_threads if q != 1]
            if unsupported_q:
                exclusions[(backend, "thread-control", tuple(unsupported_q))] = {
                    "backend": backend, "status": "skipped", "threads": unsupported_q,
                    "reason": "conda-forge pyspharm public API has no native thread-count control; measured once at q1",
                }
        for case in cases:
            if backend == "pyspharm" and int(case["lmax"]) > 640:
                # Enforce the native degree ceiling before constructing a
                # Workload or starting a process that could allocate a plan.
                for precision in selected_precision:
                    for spin in (0, 1):
                        for operation in OPERATIONS:
                            key = (backend, case["case_id"], precision, spin, operation, "degree_limit")
                            exclusions[key] = {
                                "backend": backend, "status": "degree_limit",
                                "grid": case["grid"], "case_id": case["case_id"],
                                "resolution": case["resolution"], "lmax": case["lmax"],
                                "precision": precision, "spin": spin,
                                "operation": operation, "threads": [1],
                                "max_degree": 640,
                                "reason": "conda-forge pyspharm is capped at L640; no workload or native plan was constructed",
                            }
                continue
            for precision in selected_precision:
                for spin in (0, 1):
                    for q in supported_threads:
                        key = (backend, case["case_id"], precision, spin, q)
                        group = grouped.setdefault(key, {
                            "backend": backend, "case": case, "precision": precision,
                            "spin": spin, "threads_requested": q, "build": build,
                            "operations": [], "base_cells": {},
                        })
                        for operation in OPERATIONS:
                            w = Workload(
                                grid=case["grid"], nlat=case["nlat"], nlon=case["nlon"],
                                lmax=case["lmax"], mmax=case["lmax"], spin=spin,
                                precision=precision, operation=operation,
                            )
                            reason = capability(backend, w)
                            if reason:
                                exclusion_key = (backend, case["grid"], precision, spin, operation, reason)
                                entry = exclusions.setdefault(exclusion_key, {
                                    "backend": backend, "status": "unsupported", "grid": case["grid"],
                                    "precision": precision, "spin": spin, "operation": operation,
                                    "reason": reason, "case_ids": [], "lmax": [], "threads": list(supported_threads),
                                })
                                entry["case_ids"].append(case["case_id"])
                                entry["lmax"].append(case["lmax"])
                                continue
                            base = _base_cell(
                                backend=backend, case=case, precision=precision, spin=spin,
                                operation=operation, threads=q, build=build,
                                build_hash=build_hash, status="planned",
                            )
                            group["operations"].append(operation)
                            group["base_cells"][operation] = base
    for entry in exclusions.values():
        if "case_ids" in entry:
            entry["case_ids"] = sorted(set(entry["case_ids"]))
            entry["lmax"] = sorted(set(entry["lmax"]))
    grid_order = {grid: index for index, grid in enumerate(args.grid_values)}
    backend_order = {backend: index for index, backend in enumerate(args.backend_values)}
    precision_order = {precision: index for index, precision in enumerate(args.precision_values)}
    thread_order = {q: index for index, q in enumerate(THREAD_EXECUTION_ORDER)}
    groups = [group for group in grouped.values() if group["operations"]]
    groups.sort(key=lambda group: (
        thread_order.get(group["threads_requested"], len(thread_order)),
        grid_order[group["case"]["grid"]], group["case"]["lmax"],
        backend_order[group["backend"]], precision_order[group["precision"]], group["spin"],
    ))
    return groups, list(exclusions.values())


def _figure_order() -> list[tuple[str, str, int, str]]:
    return [
        (grid, precision, spin, operation)
        for precision in ("float64", "float32")
        for grid in ("GL", "CC")
        for spin in (0, 1)
        for operation in OPERATIONS
    ]


def _plot(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sht_bench.colors import BACKEND_COLORS

    result_path = output / "timings.json"
    result = json.loads(result_path.read_text())
    cells = result.get("cells", [])
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure_order = _figure_order()
    for grid, precision, spin, operation in figure_order:
        figure, axes = plt.subplots(1, 5, figsize=(18, 4.2), sharey=True)
        candidates = [
            row for row in cells
            if row.get("grid") == grid and row.get("precision") == precision
            and row.get("spin") == spin and row.get("operation") == operation
            and row.get("status") == "timed_successfully"
            and isinstance(row.get("median_s"), (int, float))
            and row["median_s"] > 0
        ]
        positive = [row["median_s"] for row in candidates]
        common_ylim = None
        if positive:
            lower = min(positive) / 1.6
            upper = max(positive) * 1.8
            common_ylim = (max(lower, 1e-7), upper)
        ladder = [case["lmax"] for case in result.get("resolution_cases", []) if case.get("grid") == grid]
        x_min, x_max = (min(ladder), max(ladder)) if ladder else (1, 2)
        x_pad = max(1.0, (x_max - x_min) * 0.03)
        for axis, q in zip(axes, THREADS):
            panel = [row for row in candidates if row["threads_requested"] == q]
            for backend in BACKENDS:
                by_degree = {
                    int(row["lmax"]): row
                    for row in panel if row["backend"] == backend
                }
                runs = []
                run = []
                for degree in ladder:
                    row = by_degree.get(degree)
                    if row is None:
                        if run:
                            runs.append(run)
                            run = []
                        continue
                    run.append(row)
                if run:
                    runs.append(run)
                if not runs:
                    continue
                for run_index, points in enumerate(runs):
                    axis.plot(
                        [row["lmax"] for row in points],
                        [row["median_s"] * 1000 for row in points],
                        marker="o", markersize=4, linewidth=1.5,
                        color=BACKEND_COLORS[backend],
                        label=backend if run_index == 0 else "_nolegend_",
                    )
            axis.set_title(f"q{q}")
            axis.set_yscale("log")
            axis.set_xlabel("lmax")
            axis.set_xlim(x_min - x_pad, x_max + x_pad)
            axis.grid(True, which="both", alpha=0.22)
            if common_ylim:
                axis.set_ylim(common_ylim[0] * 1000, common_ylim[1] * 1000)
            if not panel:
                axis.text(0.5, 0.5, "No successful timings", ha="center", va="center", transform=axis.transAxes, fontsize=9)
        axes[0].set_ylabel("Median warm-call latency (ms, log scale)")
        title = f"{grid} · {precision} · spin {spin} · {operation}"
        figure.suptitle(title + " — exploratory (3 samples, median)", fontsize=12)
        handles, labels = [], []
        for axis in axes:
            panel_handles, panel_labels = axis.get_legend_handles_labels()
            for handle, label in zip(panel_handles, panel_labels):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        if handles:
            figure.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
            figure.tight_layout(rect=(0, 0.13, 1, 0.92))
        else:
            figure.tight_layout(rect=(0, 0.03, 1, 0.92))
        name = f"{grid}_{precision}_spin{spin}_{operation}.png"
        figure.savefig(plot_dir / name, dpi=160, bbox_inches="tight")
        plt.close(figure)
    ordered_names = [f"{grid}_{precision}_spin{spin}_{operation}.png"
                     for grid, precision, spin, operation in figure_order]
    expected = set(ordered_names)
    figures = list(plot_dir.glob("*.png"))
    for stale in figures:
        if stale.name not in expected:
            stale.unlink()
    lines = [
        "# GL/CC resolution scaling figures", "",
        "Exploratory native allocating warm-call timings. Each figure separates grid, precision, spin and direction; the five panels show q1/q2/q4/q8/q16. Markers show measured points; line segments stop at missing or failed degrees. Float64 figures precede float32. SHTns is available at float64; conda-forge pyspharm is available at float32 and q1 only, with a hard L640 ceiling.", "",
    ]
    for name in ordered_names:
        label = name.removesuffix(".png").replace("_", " · ")
        lines.extend([f"## {label}", "", f"![{label}]({name})", ""])
    (plot_dir / "README.md").write_text("\n".join(lines))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-manifest", type=Path, default=DEFAULT_BUILD)
    p.add_argument("--output", type=Path, default=ROOT / "results/gl-cc-resolution-scaling")
    p.add_argument("--profile", choices=PROFILE_NAMES, default="standard", help="resolution ladder; dense-to-1024 includes 18 GL and 19 CC degrees")
    p.add_argument("--backend", default=",".join(BACKENDS), help="comma-separated: ducc,shtns,rustsht,pyspharm")
    p.add_argument("--grid", default="GL,CC", help="comma-separated GL,CC")
    p.add_argument("--threads", default=",".join(map(str, THREADS)), help="comma-separated 1,2,4,8,16")
    p.add_argument("--precision", default=",".join(PRECISIONS), help="comma-separated float32,float64")
    p.add_argument("--min-time", type=float, default=0.005, help="calibration target per timing sample, seconds")
    p.add_argument("--timeout", type=float, default=180, help="maximum seconds for one backend/resolution/thread worker")
    p.add_argument("--memory-limit-gib", type=float, help="optional memory admission cap")
    p.add_argument("--resume", action="store_true", help="resume compatible incremental work")
    p.add_argument("--smoke", action="store_true", help="only N32 GL and L36 CC (first profile entry)")
    p.add_argument("--replot", action="store_true", help="regenerate the 16 figures from timings.json")
    p.add_argument("--_worker", nargs=2, metavar=("TASK_JSON", "RESULT_JSON"), help=argparse.SUPPRESS)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args._worker:
        return _worker(Path(args._worker[0]), Path(args._worker[1]))
    try:
        args.backend_values = _parse_csv(args.backend, BACKENDS, "backend")
        args.grid_values = _parse_csv(args.grid, ("GL", "CC"), "grid")
        args.thread_values = tuple(int(value) for value in _parse_csv(args.threads, tuple(map(str, THREADS)), "threads"))
        args.precision_values = _parse_csv(args.precision, PRECISIONS, "precision")
    except ValueError as exc:
        parser().error(str(exc))
    if not math.isfinite(args.min_time) or args.min_time <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser().error("--min-time and --timeout must be finite and positive")
    if args.memory_limit_gib is not None and (not math.isfinite(args.memory_limit_gib) or args.memory_limit_gib <= 0):
        parser().error("--memory-limit-gib must be finite and positive")
    output = args.output.expanduser().resolve()
    final_path = output / "timings.json"
    if args.replot:
        if not final_path.exists():
            parser().error("--replot requires timings.json in --output")
        _plot(output)
        print(f"regenerated 16 figures under {output / 'plots'}")
        return 0
    if args.smoke:
        cases = resolution_cases(args.grid_values, profile=args.profile, endpoints_only=True)
    else:
        cases = resolution_cases(args.grid_values, profile=args.profile)
    builds, build_hash = _load_builds(args.build_manifest.expanduser().resolve(), args.backend_values)
    config = {
        "backend": list(args.backend_values), "grid": list(args.grid_values),
        "threads": list(args.thread_values), "precision": list(args.precision_values),
        "resolution_profile": args.profile,
        "thread_execution_order": [q for q in THREAD_EXECUTION_ORDER if q in args.thread_values],
        "thread_environment_policy": {
            "OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY"),
            "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND"),
            "OMP_PLACES": os.environ.get("OMP_PLACES"),
            "worker_affinity": "first q physical cores returned by source_runner.physical_cores",
            "per_worker_thread_variables": {
                "OMP_NUM_THREADS": "requested q", "RAYON_NUM_THREADS": "requested q",
                "DUCC0_NUM_THREADS": "requested q",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            },
        },
        "mmax": "full", "rounds": 1, "warmups": 1, "samples": 3,
        "min_time_s": args.min_time, "timeout_s": args.timeout,
        "memory_limit_bytes": int(args.memory_limit_gib * 2**30) if args.memory_limit_gib else None,
        "smoke": bool(args.smoke), "build_manifest": str(args.build_manifest.expanduser().resolve()),
        "build_manifest_sha256": build_hash,
    }
    config_path = output / "run-config.json"
    if args.resume:
        if not config_path.exists() or json.loads(config_path.read_text()) != config:
            parser().error("--resume requires an existing, compatible run-config.json")
    elif output.exists() and any(output.iterdir()):
        parser().error("output directory already contains data; choose a new directory or pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(config_path, config)
    manifest_copy = output / "build-manifest.json"
    selected_manifest_path = args.build_manifest.expanduser().resolve()
    if not manifest_copy.exists():
        shutil.copyfile(selected_manifest_path, manifest_copy)
    elif _sha(manifest_copy.read_bytes()) != build_hash:
        parser().error("output build-manifest.json differs from the selected immutable build manifest")
    profile = [_case_short(case) for case in cases]
    _write_json(output / "resolution-profile.json", {"schema": PROFILE_SCHEMA, "name": args.profile, "cases": profile})

    groups, exclusion_values = _scenarios(args, cases, builds, build_hash)
    exclusions = exclusion_values[:]
    if args.memory_limit_gib:
        cap_bytes = int(args.memory_limit_gib * 2**30)
    else:
        cap_bytes = None
    script_hash = _sha(Path(__file__).read_bytes())
    progress_path = output / ".progress.jsonl"
    cells = _load_progress(progress_path, final_path, args.resume)
    completed = {cell["id"] for cell in cells}
    task_hashes = set()
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            if line.strip():
                task_hashes.add(json.loads(line).get("task_id"))
    processes = 0
    for group in groups:
        missing = [op for op in group["operations"] if group["base_cells"][op]["id"] not in completed]
        if not missing:
            continue
        backend, case = group["backend"], group["case"]
        w = Workload(
            grid=case["grid"], nlat=case["nlat"], nlon=case["nlon"],
            lmax=case["lmax"], mmax=case["lmax"], spin=group["spin"],
            precision=group["precision"], operation="synthesis",
        )
        admission = _memory_admission(backend, w, cap_bytes)
        base_cells = {op: group["base_cells"][op] for op in missing}
        if admission.get("resource_limited"):
            new = [{**base_cells[op], "status": "resource_limited", "reason": admission.get("reason"), "samples_s": []} for op in missing]
            task_id = _sha(_canonical([cell["id"] for cell in new]))[:20]
            record = {"task_id": task_id, "outcomes": new}
            _append_progress(progress_path, record)
            cells.extend(new)
            completed.update(cell["id"] for cell in new)
            _write_csv(output / "timings.csv", cells)
            print(f"{backend} {case['case_id']} {group['precision']} s{group['spin']} q{group['threads_requested']} resource_limited", flush=True)
            continue
        task_id = _sha(_canonical({
            "cells": [base_cells[op]["id"] for op in missing],
            "runner_sha256": script_hash,
        }))[:20]
        if task_id in task_hashes:
            continue
        task = {
            "task_id": task_id, "backend": backend, "case": _case_short(case),
            "build": build_identity_for_worker(builds[backend]),
            "precision": group["precision"], "spin": group["spin"],
            "threads_requested": group["threads_requested"], "operations": missing,
            "base_cells": base_cells, "min_time": args.min_time,
            "memory_budget_bytes": admission.get("memory_budget_bytes"),
        }
        task_file = output / ".workers" / f"{task_id}.task.json"
        result_file = output / ".workers" / f"{task_id}.result.json"
        _write_json(task_file, task)
        env = os.environ.copy()
        q = group["threads_requested"]
        env.update({
            "OMP_NUM_THREADS": str(q), "RAYON_NUM_THREADS": str(q), "DUCC0_NUM_THREADS": str(q),
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        })
        try:
            process = subprocess.run(
                [builds[backend]["python_executable"], str(Path(__file__).resolve()), "--_worker", str(task_file), str(result_file)],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=args.timeout, check=False,
            )
            processes += 1
            if process.returncode != 0:
                reason = f"worker exited {process.returncode}: {process.stderr[-1000:]}"
            else:
                reason = None
        except subprocess.TimeoutExpired:
            processes += 1
            reason = f"worker exceeded timeout of {args.timeout:g} seconds"
        if result_file.exists():
            worker_result = json.loads(result_file.read_text())
            measured = worker_result.get("outcomes", [])
            shared = worker_result.get("shared", {})
        else:
            measured, shared = [], {}
        by_operation = {cell["operation"]: cell for cell in measured}
        final_outcomes = []
        for op in missing:
            cell = by_operation.get(op)
            if cell is None:
                status = "resource_limited" if reason and "timeout" in reason else "execution_failed"
                cell = {**base_cells[op], "status": status, "reason": reason or "worker did not record an outcome", "samples_s": []}
            cell["worker"] = {"import_s": shared.get("import_s"), "cpu_affinity": shared.get("cpu_affinity"),
                              "allocation_limit": shared.get("allocation_limit"), "native_api": shared.get("native_api"),
                              "thread_control": shared.get("thread_control"), "planning_policy": shared.get("planning_policy"),
                              "thread_environment": shared.get("thread_environment")}
            final_outcomes.append(cell)
        record = {"task_id": task_id, "shared": shared, "admission": admission, "outcomes": final_outcomes}
        _append_progress(progress_path, record)
        cells.extend(final_outcomes)
        completed.update(cell["id"] for cell in final_outcomes)
        task_hashes.add(task_id)
        _write_csv(output / "timings.csv", cells)
        successful = sum(cell["status"] == "timed_successfully" for cell in final_outcomes)
        failed = sum(cell["status"] != "timed_successfully" for cell in final_outcomes)
        print(f"{backend} {case['case_id']} {group['precision']} s{group['spin']} q{q}: {successful} timed, {failed} excluded", flush=True)
        if task_file.exists():
            task_file.unlink()
        if result_file.exists():
            result_file.unlink()
    # Include any explicit backend build failures as failed cells without trying
    # to start a process for the missing environment.
    if len(cells) != len({cell["id"] for cell in cells}):
        raise RuntimeError("duplicate result cell identity")
    _write_final(output, config, profile, builds, cells, exclusions, script_hash)
    if progress_path.exists():
        progress_path.unlink()
    _plot(output)
    print(json.dumps({"output": str(output), "cells": len(cells), "status_counts": _count_statuses(cells), "processes": processes, "figures": 16}, indent=2))
    return 0


def build_identity_for_worker(entry: dict) -> dict:
    return {
        key: entry.get(key) for key in (
            "status", "build_kind", "build_origin", "origin", "ref", "sha",
            "requirement", "python_executable", "python_version", "module_path",
        )
    } | {"binaries": entry.get("binaries", [])}


if __name__ == "__main__":
    raise SystemExit(main())
