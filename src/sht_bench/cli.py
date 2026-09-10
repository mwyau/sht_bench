from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from .grids import (
    ATMOSPHERIC_GL_GRIDS,
    DEFAULT_CC_LMAX,
    DEFAULT_GL_LMAX,
    cc_resolution,
    gl_grid_label,
    gl_grid_number,
)

if TYPE_CHECKING:
    from .backends import PreparedCase

BACKEND_NAMES = ("ducc", "shtns", "pyshtools", "pyspharm")
BACKEND_MODULES = {
    "ducc": "ducc0",
    "shtns": "shtns",
    "pyshtools": "pyshtools",
    "pyspharm": "spharm",
}
GRID_NAMES = ("gl", "cc")
OPERATION_NAMES = ("analysis", "synthesis")
DEFAULT_RUN_LMAX = [42, 85, 159, 319, 639]
UNSUPPORTED_BENCHMARK_CASES = {
    **{
        ("pyshtools", "gl", lmax): (
            "native SHTOOLS GLQ requires nlon=2*nlat-1 and cannot use the "
            "common atmospheric Gaussian grid"
        )
        for lmax in DEFAULT_GL_LMAX
    },
    ("pyspharm", "cc", 1023): (
        "pyspharm-syl 1.2.1 regular-grid setup fails at lmax=1023"
    ),
}
CELL_RESULT_RE = re.compile(r"^(installed|wheel|source)-(gl|cc)-t\d+\.json$")

BACKEND_COLORS = {
    "ducc": "#1f77b4",
    "shtns": "#ff7f0e",
    "pyshtools": "#2ca02c",
    "pyspharm": "#d62728",
}
BUILD_LINESTYLES = {
    "installed": "-",
    "wheel": "-",
    "source": "--",
}
_BACKEND_ORDER = {name: index for index, name in enumerate(BACKEND_NAMES)}
_BUILD_ORDER = {name: index for index, name in enumerate(BUILD_LINESTYLES)}


def _parse_int_sweep(value: str, *, minimum: int, name: str) -> list[int]:
    values: list[int] = []
    for token in (item.strip() for item in value.split(",")):
        if not token:
            continue
        if ":" not in token:
            try:
                values.append(int(token))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid {name} value: {token}") from exc
            continue
        try:
            parts = [int(part) for part in token.split(":")]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid {name} range: {token}") from exc
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError(
                f"{name} ranges must use start:stop or start:stop:step"
            )
        start, stop = parts[:2]
        step = parts[2] if len(parts) == 3 else 1
        if step == 0:
            raise argparse.ArgumentTypeError(f"{name} range step must not be zero")
        if (stop - start) * step < 0:
            raise argparse.ArgumentTypeError(f"{name} range step has the wrong sign")
        end = stop + (1 if step > 0 else -1)
        values.extend(range(start, end, step))
    if not values:
        raise argparse.ArgumentTypeError(f"expected at least one {name} value")
    if any(item < minimum for item in values):
        raise argparse.ArgumentTypeError(f"{name} values must be >= {minimum}")
    return sorted(set(values))


def _parse_lmax(value: str) -> list[int]:
    return _parse_int_sweep(value, minimum=2, name="lmax")


def _auto_threads() -> list[int]:
    count = os.cpu_count() or 1
    values = [1]
    value = 2
    while value <= count:
        values.append(value)
        value *= 2
    if values[-1] != count:
        values.append(count)
    return values


def _parse_threads(value: str) -> list[int]:
    if value.lower() == "auto":
        return _auto_threads()
    return _parse_int_sweep(value, minimum=1, name="threads")


def _parse_choices(value: str, choices: tuple[str, ...], name: str) -> list[str]:
    if value.lower() == "all":
        return list(choices)
    result = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(result) - set(choices))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown {name}(s): {', '.join(unknown)}")
    if not result:
        raise argparse.ArgumentTypeError(f"expected at least one {name}")
    return list(dict.fromkeys(result))


def _parse_backends(value: str) -> list[str]:
    return _parse_choices(value, BACKEND_NAMES, "backend")


def _parse_grids(value: str) -> list[str]:
    return _parse_choices(value, GRID_NAMES, "grid")


def _parse_operations(value: str) -> list[str]:
    return _parse_choices(value, OPERATION_NAMES, "operation")


def _set_thread_environment(threads: int) -> None:
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "DUCC0_NUM_THREADS",
    ):
        os.environ[name] = value


def _cpu_model() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _environment() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "logical_cpus": os.cpu_count(),
    }


def _time_block(fn: Callable[[], object], iterations: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    return (time.perf_counter_ns() - start) / 1e9


def measure(
    fn: Callable[[], object], *, warmup: int, repeat: int, min_time: float
) -> tuple[int, list[float]]:
    for _ in range(warmup):
        fn()
    iterations = 1
    while True:
        elapsed = _time_block(fn, iterations)
        if elapsed >= min_time or iterations >= 1 << 20:
            break
        scale = max(2, min(16, math.ceil(min_time / max(elapsed, 1e-12))))
        iterations *= scale
    samples: list[float] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeat):
            elapsed = _time_block(fn, iterations)
            samples.append(elapsed / iterations)
    finally:
        if was_enabled:
            gc.enable()
    return iterations, samples


def _grid_label(grid: str, lmax: int) -> str:
    if grid == "gl" and lmax in ATMOSPHERIC_GL_GRIDS:
        return gl_grid_label(lmax)
    if grid == "cc":
        return f"{cc_resolution(lmax):g}°"
    return f"L={lmax}"


def _record(
    case: PreparedCase,
    operation: str,
    iterations: int,
    samples: list[float],
    env: dict[str, Any],
    warmup: int,
    repeat: int,
    min_time: float,
) -> dict[str, Any]:
    return {
        **env,
        "backend": case.backend,
        "backend_version": case.backend_version,
        "operation": operation,
        "grid": case.grid,
        "grid_label": _grid_label(case.grid, case.lmax),
        "lmax": case.lmax,
        "nlat": case.nlat,
        "nlon": case.nlon,
        "grid_points": case.nlat * case.nlon,
        "threads_requested": case.threads_requested,
        "threads_actual": case.threads_actual,
        "spatial_dtype": case.spatial_dtype,
        "spectral_dtype": case.spectral_dtype,
        "warmup": warmup,
        "repeat": repeat,
        "min_time_s": min_time,
        "iterations_per_sample": iterations,
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples_s": samples,
        "notes": case.notes,
    }


def _write_results(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    fieldnames = [key for key in records[0] if key != "samples_s"] if records else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def _print_result(record: dict[str, Any]) -> None:
    print(
        f"{record['backend']:10s} {record['grid']:2s} {record['operation']:9s} "
        f"{record['grid_label']:12s} L={record['lmax']:4d} "
        f"t={record['threads_requested']:2d} {record['nlat']}x{record['nlon']} "
        f"median={record['median_s'] * 1e3:10.3f} ms"
    )


def run_benchmarks(args: argparse.Namespace) -> int:
    threads = args.threads[0] if isinstance(args.threads, list) else args.threads
    if isinstance(args.threads, list) and len(args.threads) != 1:
        raise SystemExit("sht-bench run accepts one thread count; use 'matrix' for sweeps")
    _set_thread_environment(threads)
    from .backends import PREPARERS, SUPPORTED_GRIDS

    env = _environment()
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    skipped: list[str] = []

    for backend in args.backend:
        prepare = PREPARERS[backend]
        for grid in args.grid:
            if grid not in SUPPORTED_GRIDS[backend]:
                skipped.append(f"{backend}/{grid}: unsupported combination")
                continue
            for lmax in args.lmax:
                case_key = (backend, grid, lmax)
                if case_key in UNSUPPORTED_BENCHMARK_CASES:
                    skipped.append(
                        f"{backend}/{grid} L={lmax}: {UNSUPPORTED_BENCHMARK_CASES[case_key]}"
                    )
                    continue
                try:
                    case = prepare(lmax, threads, args.seed + lmax, grid)
                except (ImportError, ModuleNotFoundError) as exc:
                    failures.append(f"{backend}: unavailable ({exc})")
                    break
                except Exception as exc:
                    failures.append(f"{backend}/{grid} L={lmax}: setup failed ({exc})")
                    if args.strict:
                        raise
                    break

                for operation in args.operation:
                    iterations, samples = measure(
                        getattr(case, operation),
                        warmup=args.warmup,
                        repeat=args.repeat,
                        min_time=args.min_time,
                    )
                    record = _record(
                        case,
                        operation,
                        iterations,
                        samples,
                        env,
                        args.warmup,
                        args.repeat,
                        args.min_time,
                    )
                    records.append(record)
                    _print_result(record)

    if records:
        _write_results(records, args.output)
        print(f"wrote {args.output} and {args.output.with_suffix('.csv')}")
    if skipped:
        print("\nUnsupported combinations skipped:", file=sys.stderr)
        for item in skipped:
            print(f"  - {item}", file=sys.stderr)
    if failures:
        print("\nUnavailable/failed combinations:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
    if args.strict and failures:
        return 1
    return 0 if records else 1


def _load_result_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"{path} does not contain a result-record list")
        for item in data:
            record = dict(item)
            record.setdefault("build_mode", "installed")
            record.setdefault(
                "grid_label", _grid_label(record["grid"], int(record["lmax"]))
            )
            records.append(record)
    if not records:
        raise ValueError("no benchmark records found")
    return records


def _effective_plot_build(record: dict[str, Any]) -> str:
    build = record.get("build_mode", "installed")
    if record["backend"] == "shtns" and build in {"wheel", "source"}:
        return "source"
    return build


def _prepare_plot_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    shtns_records: dict[tuple[Any, ...], tuple[int, dict[str, Any]]] = {}

    for record in records:
        build = record.get("build_mode", "installed")
        if record["backend"] != "shtns" or build not in {"wheel", "source"}:
            prepared.append(record)
            continue

        normalized = dict(record)
        normalized["build_mode"] = "source"
        key = (
            normalized["grid"],
            normalized["operation"],
            normalized["lmax"],
            normalized["threads_requested"],
        )
        priority = 0 if build == "source" else 1
        previous = shtns_records.get(key)
        if previous is None or priority < previous[0]:
            shtns_records[key] = (priority, normalized)

    prepared.extend(record for _, record in shtns_records.values())
    return prepared


def _series_label(record: dict[str, Any]) -> str:
    label = record["backend"]
    build = _effective_plot_build(record)
    if build != "installed":
        label += f" ({build})"
    return label


def _plot_style(record: dict[str, Any]) -> dict[str, str]:
    backend = record["backend"]
    build = _effective_plot_build(record)
    return {
        "color": BACKEND_COLORS[backend],
        "linestyle": BUILD_LINESTYLES.get(build, "-"),
    }


def _plot_series_sort_key(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    sample = rows[0]
    backend = sample["backend"]
    build = _effective_plot_build(sample)
    threads = int(sample.get("threads_requested", 0))
    return (
        _BACKEND_ORDER.get(backend, len(_BACKEND_ORDER)),
        _BUILD_ORDER.get(build, len(_BUILD_ORDER)),
        threads,
    )


def _cc_resolution(lmax: int) -> float:
    return cc_resolution(lmax)


def _format_resolution(value: float) -> str:
    return f"{value:g}"


def _is_atmospheric_gl_records(records: list[dict[str, Any]]) -> bool:
    if not records or {record["grid"] for record in records} != {"gl"}:
        return False
    for record in records:
        lmax = int(record["lmax"])
        case = ATMOSPHERIC_GL_GRIDS.get(lmax)
        if case is None or (int(record["nlat"]), int(record["nlon"])) != case[:2]:
            return False
    return True


def _plot_lines(
    records: list[dict[str, Any]],
    *,
    x_key: str,
    output: Path,
    title: str,
    stat: str,
    dpi: int,
    group_keys: tuple[str, ...],
    log_x: bool = True,
    log_y: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    grid_set = {record["grid"] for record in records}
    cc_resolution_axis = x_key == "lmax" and bool(records) and grid_set == {"cc"}
    gl_atmospheric_axis = x_key == "lmax" and _is_atmospheric_gl_records(records)

    def x_value(record: dict[str, Any]) -> float | int:
        if cc_resolution_axis:
            return cc_resolution(int(record["lmax"]))
        if gl_atmospheric_axis:
            grid_number = gl_grid_number(int(record["lmax"]))
            if grid_number is None:
                raise ValueError("missing Gaussian grid number for atmospheric GL case")
            return grid_number
        return record[x_key]

    fig, ax = plt.subplots()
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record.get(name, "installed") for name in group_keys)
        groups.setdefault(key, []).append(record)
    for _, rows in sorted(groups.items(), key=lambda item: _plot_series_sort_key(item[1])):
        rows.sort(key=x_value, reverse=cc_resolution_axis)
        sample = rows[0]
        label = _series_label(sample)
        if "threads_requested" in group_keys:
            label += f" t={sample['threads_requested']}"
        ax.plot(
            [x_value(row) for row in rows],
            [row[stat] * 1e3 for row in rows],
            marker="o",
            label=label,
            **_plot_style(sample),
        )
    if log_x:
        ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")

    if cc_resolution_axis:
        ax.set_xlabel("regular-grid spacing (degrees)")
        ticks = sorted(
            {cc_resolution(int(record["lmax"])) for record in records}, reverse=True
        )
        ax.set_xticks(ticks)
        ax.set_xticklabels([_format_resolution(value) for value in ticks])
        ax.invert_xaxis()
    elif gl_atmospheric_axis:
        pairs = sorted(
            {
                (
                    gl_grid_number(int(record["lmax"])),
                    gl_grid_label(int(record["lmax"])),
                )
                for record in records
            }
        )
        ax.set_xlabel("Gaussian grid / spectral truncation")
        ax.set_xticks([pair[0] for pair in pairs])
        ax.set_xticklabels([pair[1] for pair in pairs], rotation=45, ha="right")
    else:
        ax.set_xlabel(
            "maximum spherical-harmonic degree (lmax)"
            if x_key == "lmax"
            else "threads"
            if x_key == "threads_requested"
            else "grid points"
        )
        if x_key in {"lmax", "threads_requested"}:
            ticks = sorted({int(record[x_key]) for record in records})
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(value) for value in ticks])
            if x_key == "lmax" and len(ticks) > 8:
                ax.tick_params(axis="x", labelrotation=45)
                for tick_label in ax.get_xticklabels():
                    tick_label.set_horizontalalignment("right")

    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_ylabel(f"{stat.removesuffix('_s')} time per transform (ms; lower is better)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_results(args: argparse.Namespace) -> int:
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise SystemExit("plotting requires the 'plot' extra") from exc
    records = _prepare_plot_records(_load_result_files(args.input))
    args.output.mkdir(parents=True, exist_ok=True)
    for operation in sorted({record["operation"] for record in records}):
        op_records = [record for record in records if record["operation"] == operation]
        for grid in sorted({record["grid"] for record in op_records}):
            grid_records = [record for record in op_records if record["grid"] == grid]
            _plot_lines(
                grid_records,
                x_key=args.x,
                output=args.output / f"{grid}-{operation}-{args.x}.png",
                title=f"{grid.upper()} scalar {operation}",
                stat=args.stat,
                dpi=args.dpi,
                group_keys=("backend", "build_mode", "threads_requested"),
                log_x=args.log_x,
                log_y=args.log_y,
            )
    return 0


def _plot_index(
    created: list[str],
    *,
    grids: list[str],
    operations: list[str],
    threads: list[int],
    lmax_values: list[int],
) -> str:
    available = set(created)
    lines = [
        "# Benchmark plots",
        "",
        "Each thumbnail links to the full-size figure.",
        "Colors identify backends consistently across figures: DUCC is blue, SHTns orange, SHTOOLS/pyshtools green, and Spherepack/pyspharm red. Wheel and installed builds use solid lines; source builds use dashed lines. SHTns is source-built in both benchmark environments and is shown once as a dashed source series. Lower transform time is better.",
        "",
        "## Transform time versus spectral scale",
        "",
        "Each row fixes the requested thread count. Atmospheric GL figures use Gaussian-grid/spectral labels such as `N32/T42` and `N320/TL639`; CC figures use regular-grid spacing in degrees, from coarse to fine resolution.",
        "",
    ]

    for grid in grids:
        rows: list[str] = []
        for thread_count in threads:
            cells = [str(thread_count)]
            present = False
            for operation in operations:
                path = f"by-lmax/{grid}/{operation}/threads-{thread_count}.png"
                if path in available:
                    present = True
                    alt = f"{grid.upper()} {operation}, {thread_count} thread(s)"
                    cells.append(
                        f'<a href="{path}"><img src="{path}" alt="{alt}" width="420"></a>'
                    )
                else:
                    cells.append("—")
            if present:
                rows.append("| " + " | ".join(cells) + " |")
        if rows:
            lines.extend(
                [
                    f"### {grid.upper()}",
                    "",
                    "| Threads | " + " | ".join(op.title() for op in operations) + " |",
                    "| ---: | " + " | ".join("---" for _ in operations) + " |",
                    *rows,
                    "",
                ]
            )

    lines.extend(
        [
            "## Transform time versus threads",
            "",
            "Each row fixes the spectral scale; each figure compares backend/build series across requested thread counts.",
            "",
        ]
    )
    for grid in grids:
        rows = []
        for lmax in lmax_values:
            if grid == "cc":
                scale_label = f"{_format_resolution(cc_resolution(lmax))}° (L={lmax})"
            elif lmax in ATMOSPHERIC_GL_GRIDS:
                scale_label = gl_grid_label(lmax)
            else:
                scale_label = str(lmax)
            cells = [scale_label]
            present = False
            for operation in operations:
                path = f"by-threads/{grid}/{operation}/lmax-{lmax}.png"
                if path in available:
                    present = True
                    alt = f"{grid.upper()} {operation}, lmax={lmax}"
                    cells.append(
                        f'<a href="{path}"><img src="{path}" alt="{alt}" width="420"></a>'
                    )
                else:
                    cells.append("—")
            if present:
                rows.append("| " + " | ".join(cells) + " |")
        if rows:
            scale_header = (
                "Resolution"
                if grid == "cc"
                else "Gaussian grid / truncation"
                if grid == "gl"
                else "lmax"
            )
            lines.extend(
                [
                    f"### {grid.upper()}",
                    "",
                    f"| {scale_header} | "
                    + " | ".join(op.title() for op in operations)
                    + " |",
                    "| ---: | " + " | ".join("---" for _ in operations) + " |",
                    *rows,
                    "",
                ]
            )

    return "\n".join(lines) + "\n"


def plot_all(args: argparse.Namespace) -> int:
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise SystemExit("plotting requires the 'plot' extra") from exc
    records = _prepare_plot_records(_load_result_files(args.input))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    grids = sorted({record["grid"] for record in records})
    operations = sorted({record["operation"] for record in records})
    threads = sorted({int(record["threads_requested"]) for record in records})
    lmax_values = sorted({int(record["lmax"]) for record in records})

    for grid in grids:
        for operation in operations:
            for thread_count in threads:
                rows = [
                    record
                    for record in records
                    if record["grid"] == grid
                    and record["operation"] == operation
                    and int(record["threads_requested"]) == thread_count
                ]
                if not rows:
                    continue
                path = (
                    output
                    / "by-lmax"
                    / grid
                    / operation
                    / f"threads-{thread_count}.png"
                )
                _plot_lines(
                    rows,
                    x_key="lmax",
                    output=path,
                    title=f"{grid.upper()} {operation} — {thread_count} thread(s)",
                    stat=args.stat,
                    dpi=args.dpi,
                    group_keys=("backend", "build_mode"),
                )
                created.append(str(path.relative_to(output)))

    for grid in grids:
        for operation in operations:
            for lmax in lmax_values:
                rows = [
                    record
                    for record in records
                    if record["grid"] == grid
                    and record["operation"] == operation
                    and int(record["lmax"]) == lmax
                ]
                if not rows or len({record["threads_requested"] for record in rows}) < 2:
                    continue
                path = (
                    output
                    / "by-threads"
                    / grid
                    / operation
                    / f"lmax-{lmax}.png"
                )
                scale = _grid_label(grid, lmax)
                _plot_lines(
                    rows,
                    x_key="threads_requested",
                    output=path,
                    title=f"{grid.upper()} {operation} — {scale}",
                    stat=args.stat,
                    dpi=args.dpi,
                    group_keys=("backend", "build_mode"),
                )
                created.append(str(path.relative_to(output)))

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stat": args.stat,
        "input_files": [str(path) for path in args.input],
        "plots": created,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "README.md").write_text(
        _plot_index(
            created,
            grids=grids,
            operations=operations,
            threads=threads,
            lmax_values=lmax_values,
        )
    )
    print(f"wrote {len(created)} plots under {output}")
    return 0


def _matrix_output_name(build: str, grid: str, threads: int) -> str:
    return f"{build}-{grid}-t{threads}.json"


def _run_subprocess(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _backend_available(python: Path, backend: str) -> bool:
    module = BACKEND_MODULES[backend]
    code = f"import importlib.util; raise SystemExit(importlib.util.find_spec({module!r}) is None)"
    completed = subprocess.run(
        [str(python), "-c", code],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _build_matrix_env(mode: str, python_spec: str, reuse: bool) -> Path:
    root = Path(__file__).resolve().parents[2]
    env_dir = root / f".venv-bench-{mode}"
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if env_dir.exists() and reuse:
        return python
    script = root / "scripts" / "run_build_matrix.py"
    command = [
        sys.executable,
        str(script),
        "--build",
        mode,
        "--python",
        python_spec,
        "--setup-only",
    ]
    if reuse:
        command.append("--reuse")
    _run_subprocess(command)
    return python


def _matrix_lmax(args: argparse.Namespace, grid: str) -> list[int]:
    if args.lmax is not None:
        return args.lmax
    return DEFAULT_CC_LMAX if grid == "cc" else DEFAULT_GL_LMAX


def _existing_cell_results(output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in output_dir.glob("*.json")
        if CELL_RESULT_RE.fullmatch(path.name)
    )


def run_matrix(args: argparse.Namespace) -> int:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    build_modes = (
        ["installed"]
        if args.build == "installed"
        else (["wheel", "source"] if args.build == "both" else [args.build])
    )
    interpreters: dict[str, Path] = {"installed": Path(sys.executable)}
    for build in build_modes:
        if build != "installed":
            interpreters[build] = _build_matrix_env(build, args.python, args.reuse)

    updated_result_files: list[Path] = []
    failures: list[str] = []
    excluded: list[str] = []
    for build in build_modes:
        python = interpreters[build]
        selected_backends: list[str] = []
        for backend in args.backend:
            if build == "source" and backend == "pyspharm":
                excluded.append("source/pyspharm: source build is not benchmarked")
                continue
            if build != "installed" and not _backend_available(python, backend):
                excluded.append(
                    f"{build}/{backend}: package unavailable in the isolated "
                    f"environment on {platform.machine()}"
                )
                continue
            selected_backends.append(backend)
        if not selected_backends:
            continue

        for grid in args.grid:
            grid_lmax = _matrix_lmax(args, grid)
            for threads in args.threads:
                output = output_dir / _matrix_output_name(build, grid, threads)
                command = [
                    str(python),
                    "-m",
                    "sht_bench.cli",
                    "run",
                    "--backend",
                    ",".join(selected_backends),
                    "--grid",
                    grid,
                    "--lmax",
                    ",".join(map(str, grid_lmax)),
                    "--threads",
                    str(threads),
                    "--operation",
                    ",".join(args.operation),
                    "--warmup",
                    str(args.warmup),
                    "--repeat",
                    str(args.repeat),
                    "--min-time",
                    str(args.min_time),
                    "--seed",
                    str(args.seed),
                    "--output",
                    str(output),
                ]
                if args.strict:
                    command.append("--strict")
                try:
                    _run_subprocess(command)
                except subprocess.CalledProcessError as exc:
                    failures.append(f"{build}/{grid}/t{threads}: exit {exc.returncode}")
                    if args.strict:
                        raise
                    continue
                if output.exists():
                    data = json.loads(output.read_text())
                    for record in data:
                        record["build_mode"] = build
                    _write_results(data, output)
                    updated_result_files.append(output)

    result_files = (
        _existing_cell_results(output_dir) if args.merge_existing else updated_result_files
    )
    combined: list[dict[str, Any]] = []
    for path in result_files:
        combined.extend(json.loads(path.read_text()))
    combined_path = output_dir / "matrix.json"
    if combined:
        _write_results(combined, combined_path)

    default_lmax = {grid: _matrix_lmax(args, grid) for grid in args.grid}
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "build": args.build,
        "backends": args.backend,
        "grids": args.grid,
        "threads": args.threads,
        "lmax": args.lmax if args.lmax is not None else default_lmax,
        "operations": args.operation,
        "merge_existing": args.merge_existing,
        "updated_result_files": [str(path.name) for path in updated_result_files],
        "result_files": [str(path.name) for path in result_files],
        "excluded": excluded,
        "failures": failures,
    }
    (output_dir / "matrix-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"wrote combined matrix to {combined_path}")

    if args.plot and combined:
        plot_args = argparse.Namespace(
            input=[combined_path],
            output=output_dir / "plots",
            stat="median_s",
            dpi=180,
        )
        plot_all(plot_args)
    return 1 if args.strict and failures else (0 if combined else 1)


def _add_common_run_arguments(parser: argparse.ArgumentParser, *, matrix: bool) -> None:
    parser.add_argument(
        "--backend",
        type=_parse_backends,
        default=list(BACKEND_NAMES),
        help="comma-separated backends or 'all' (default: all)",
    )
    parser.add_argument(
        "--grid",
        type=_parse_grids,
        default=list(GRID_NAMES) if matrix else ["gl"],
        help="gl,cc,all (matrix default: all; run default: gl)",
    )
    parser.add_argument(
        "--lmax",
        type=_parse_lmax,
        default=None if matrix else DEFAULT_RUN_LMAX.copy(),
        help=(
            "comma-separated values and/or inclusive ranges; matrix defaults are "
            "grid-specific when omitted"
        ),
    )
    parser.add_argument(
        "--operation",
        type=_parse_operations,
        default=list(OPERATION_NAMES),
        help="analysis,synthesis,all (default: all)",
    )
    parser.add_argument(
        "--threads",
        type=_parse_threads,
        default=_auto_threads() if matrix else [1],
        help="thread counts/ranges or 'auto' (matrix default: auto)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--min-time", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of skipping unavailable/setup-failing combinations",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sht-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="run scalar SHT benchmarks in the current environment"
    )
    _add_common_run_arguments(run, matrix=False)
    run.add_argument(
        "--output", type=Path, default=Path("results/single-thread.json")
    )
    run.set_defaults(func=run_benchmarks)

    matrix = subparsers.add_parser(
        "matrix", help="run a Cartesian benchmark matrix in fresh processes"
    )
    _add_common_run_arguments(matrix, matrix=True)
    matrix.add_argument(
        "--build",
        choices=("installed", "wheel", "source", "both"),
        default="installed",
        help="package provenance dimension (default: installed)",
    )
    matrix.add_argument(
        "--python", default="3.13", help="Python used for wheel/source environments"
    )
    matrix.add_argument(
        "--reuse",
        action="store_true",
        help="reuse existing wheel/source environments",
    )
    matrix.add_argument(
        "--merge-existing",
        action="store_true",
        help="merge updated cells with existing matrix cell files in OUTPUT",
    )
    matrix.add_argument(
        "--output",
        type=Path,
        default=Path("results/matrix"),
        help="matrix result directory",
    )
    matrix.add_argument(
        "--plot",
        action="store_true",
        help="run plot-all into OUTPUT/plots after the matrix",
    )
    matrix.set_defaults(func=run_matrix)

    plot = subparsers.add_parser("plot", help="plot one or more result files")
    plot.add_argument("input", type=Path, nargs="+")
    plot.add_argument("--output", type=Path, default=Path("results/figures"))
    plot.add_argument("--x", choices=("lmax", "grid_points"), default="lmax")
    plot.add_argument(
        "--stat", choices=("median_s", "min_s", "mean_s"), default="median_s"
    )
    plot.add_argument("--linear-x", dest="log_x", action="store_false")
    plot.add_argument("--linear-y", dest="log_y", action="store_false")
    plot.add_argument("--dpi", type=int, default=180)
    plot.set_defaults(func=plot_results, log_x=True, log_y=True)

    plot_everything = subparsers.add_parser(
        "plot-all",
        help="render all spectral-scale and thread-scaling slices under one directory",
    )
    plot_everything.add_argument("input", type=Path, nargs="+")
    plot_everything.add_argument(
        "--output", type=Path, default=Path("results/plots")
    )
    plot_everything.add_argument(
        "--stat", choices=("median_s", "min_s", "mean_s"), default="median_s"
    )
    plot_everything.add_argument("--dpi", type=int, default=180)
    plot_everything.set_defaults(func=plot_all)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "warmup"):
        if args.warmup < 0 or args.repeat < 1 or args.min_time <= 0:
            parser.error("warmup/repeat/min-time values are invalid")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
