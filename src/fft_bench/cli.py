"""Command-line interface for the independent DUCC FFT benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench_common.environment import collect_environment
from bench_common.timing import measure

KIND_NAMES = ("r2c", "c2c")
PRECISION_NAMES = ("float32", "float64", "longdouble")
DEFAULT_SIZES = [256, 1024, 4096, 16384, 65536, 262144, 1048576]


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


def _parse_kinds(value: str) -> list[str]:
    return _parse_choices(value, KIND_NAMES, "FFT kind")


def _parse_dtypes(value: str) -> list[str]:
    return _parse_choices(value, PRECISION_NAMES, "dtype")


def _parse_sizes(value: str) -> list[int]:
    sizes: list[int] = []
    for token in (item.strip() for item in value.split(",")):
        if not token:
            continue
        try:
            size = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid FFT size: {token}") from exc
        if size < 1:
            raise argparse.ArgumentTypeError("FFT sizes must be >= 1")
        sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("expected at least one FFT size")
    return sorted(set(sizes))


def _auto_threads() -> list[int]:
    count = os.cpu_count() or 1
    return [value for value in (1, 2, 4) if value <= count] or [1]


def _parse_threads(value: str) -> list[int]:
    if value.lower() == "auto":
        return _auto_threads()
    values: list[int] = []
    for token in (item.strip() for item in value.split(",")):
        if not token:
            continue
        try:
            threads = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid threads value: {token}") from exc
        if threads < 1:
            raise argparse.ArgumentTypeError("threads values must be >= 1")
        values.append(threads)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one thread count")
    return sorted(set(values))


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


def _compiler_provenance() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for variable in ("CC", "CXX", "FC"):
        command = os.environ.get(variable)
        if not command:
            continue
        executable = shlex.split(command)[0]
        try:
            completed = subprocess.run(
                [executable, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            result[f"{variable.lower()}_version"] = None
            continue
        first_line = (completed.stdout or completed.stderr).splitlines()
        result[f"{variable.lower()}_version"] = first_line[0] if first_line else None
        result[f"{variable.lower()}_executable"] = executable
    return result


def _build_provenance() -> dict[str, Any]:
    return {
        "build_mode": os.environ.get("FFT_BENCH_BUILD_MODE", "installed"),
        "ducc_source": os.environ.get("DUCC_SOURCE", "unspecified"),
        "ducc_optimization": os.environ.get("DUCC0_OPTIMIZATION", "unspecified"),
        "ducc_use_nanobind": os.environ.get("DUCC0_USE_NANOBIND", "unset"),
        "cc": os.environ.get("CC"),
        "cxx": os.environ.get("CXX"),
        "cflags": os.environ.get("DUCC0_CFLAGS"),
        "flags": os.environ.get("DUCC0_FLAGS"),
        "lflags": os.environ.get("DUCC0_LFLAGS"),
        "compiler": _compiler_provenance(),
    }


def _record(
    case: Any,
    iterations: int,
    samples: list[float],
    environment: dict[str, Any],
    build: dict[str, Any],
    longdouble: dict[str, object],
    *,
    warmup: int,
    repeat: int,
    min_time: float,
) -> dict[str, Any]:
    import statistics

    return {
        **environment,
        "backend": case.backend,
        "backend_version": case.backend_version,
        "ducc_version": case.backend_version,
        "ducc_wrapper": case.ducc_wrapper,
        "build_provenance": build,
        "build_mode": build["build_mode"],
        "ducc_optimization": build["ducc_optimization"],
        "operation": case.kind,
        "kind": case.kind,
        "direction": "forward",
        "forward": True,
        "axes": [0],
        "normalization": "inorm=0",
        "requested_precision": case.requested_precision,
        "input_dtype": case.input_dtype.name,
        "output_dtype": case.output_dtype.name,
        "actual_input_dtype": case.input_dtype.str,
        "actual_output_dtype": case.output_dtype.str,
        "fft_size": case.size,
        "size": case.size,
        "threads_requested": case.threads_requested,
        "threads_actual": case.threads_actual,
        "output_preallocated": case.output_preallocated,
        "longdouble_classification": longdouble["classification"],
        "longdouble_itemsize": longdouble["itemsize"],
        "longdouble_bits": longdouble["bits"],
        "longdouble_nmant": longdouble["nmant"],
        "longdouble_eps": longdouble["eps"],
        "longdouble_eps_text": longdouble["eps_text"],
        "longdouble_dtype_name": longdouble["dtype_name"],
        "longdouble_significand_bits": longdouble["significand_bits"],
        "warmup": warmup,
        "repeat": repeat,
        "min_time_s": min_time,
        "iterations_per_sample": iterations,
        "iterations": iterations,
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples_s": samples,
        "notes": case.notes,
    }


def _skip_record(
    kind: str, precision: str, size: int, threads: int, reason: str
) -> dict[str, Any]:
    return {
        "status": "unsupported",
        "backend": "ducc",
        "kind": kind,
        "operation": kind,
        "requested_precision": precision,
        "fft_size": size,
        "threads_requested": threads,
        "reason": reason,
    }


def _write_payload(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


def _print_result(record: dict[str, Any]) -> None:
    print(
        f"{record['kind']:3s} {record['requested_precision']:10s} "
        f"N={record['fft_size']:7d} t={record['threads_requested']:2d} "
        f"median={record['median_s'] * 1e3:10.4f} ms"
    )


def run_benchmarks(args: argparse.Namespace) -> int:
    threads = args.threads[0] if isinstance(args.threads, list) else args.threads
    if isinstance(args.threads, list) and len(args.threads) != 1:
        raise SystemExit(
            "fft-bench run accepts one thread count; use 'matrix' for sweeps"
        )
    _set_thread_environment(threads)

    from .backends import UnsupportedFFTCase, longdouble_info, prepare_case

    environment = collect_environment(include_cpu_controls=True)
    import numpy as np

    environment["numpy_version"] = np.__version__
    build = _build_provenance()
    longdouble = longdouble_info()
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for kind in args.kind:
        for precision in args.dtype:
            for size in args.sizes:
                try:
                    case = prepare_case(
                        kind,
                        precision,
                        size,
                        threads,
                        seed=args.seed + size,
                    )
                except UnsupportedFFTCase as exc:
                    skipped.append(
                        _skip_record(kind, precision, size, threads, str(exc))
                    )
                    continue
                except (ImportError, ModuleNotFoundError) as exc:
                    failures.append(
                        {
                            "kind": kind,
                            "requested_precision": precision,
                            "fft_size": size,
                            "reason": f"DUCC unavailable: {exc}",
                        }
                    )
                    break
                except Exception as exc:
                    failure = {
                        "kind": kind,
                        "requested_precision": precision,
                        "fft_size": size,
                        "reason": f"setup failed: {exc}",
                    }
                    failures.append(failure)
                    if args.strict:
                        raise
                    continue

                iterations, samples = measure(
                    case.transform,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    min_time=args.min_time,
                )
                record = _record(
                    case,
                    iterations,
                    samples,
                    environment,
                    build,
                    longdouble,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    min_time=args.min_time,
                )
                records.append(record)
                _print_result(record)

    payload = {
        "schema_version": 1,
        "benchmark": "fft-bench",
        "generated_utc": datetime.now(UTC).isoformat(),
        "run": {
            "kind": args.kind,
            "dtype": args.dtype,
            "sizes": args.sizes,
            "threads": threads,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "min_time_s": args.min_time,
            "seed": args.seed,
        },
        "records": records,
        "skipped": skipped,
        "failures": failures,
    }
    _write_payload(payload, args.output)
    print(f"wrote {args.output}")
    if skipped:
        print("\nUnsupported combinations skipped:", file=sys.stderr)
        for item in skipped:
            print(
                f"  - {item['kind']}/{item['requested_precision']} "
                f"N={item['fft_size']} t={item['threads_requested']}: {item['reason']}",
                file=sys.stderr,
            )
    if failures:
        print("\nUnavailable/failed combinations:", file=sys.stderr)
        for item in failures:
            print(
                f"  - {item['kind']}/{item['requested_precision']} "
                f"N={item['fft_size']}: {item['reason']}",
                file=sys.stderr,
            )
    if args.strict and (skipped or failures):
        return 1
    return 0 if records else 1


def _load_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {"records": data, "skipped": [], "failures": []}
    if not isinstance(data, dict):
        raise TypeError(f"{path} does not contain an FFT result payload")
    data.setdefault("records", [])
    data.setdefault("skipped", [])
    data.setdefault("failures", [])
    return data


def _matrix_run_path(output_dir: Path, threads: int) -> Path:
    return output_dir / f"run-t{threads}.json"


def run_matrix(args: argparse.Namespace) -> int:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_files: list[str] = []

    for threads in args.threads:
        output = _matrix_run_path(output_dir, threads)
        command = [
            sys.executable,
            "-m",
            "fft_bench.cli",
            "run",
            "--kind",
            ",".join(args.kind),
            "--dtype",
            ",".join(args.dtype),
            "--sizes",
            ",".join(map(str, args.sizes)),
            "--threads",
            str(threads),
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
        child_environment = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "DUCC0_NUM_THREADS",
        ):
            child_environment[name] = str(threads)
        print("+", " ".join(shlex.quote(item) for item in command), flush=True)
        completed = subprocess.run(command, check=False, env=child_environment)
        if output.exists():
            payload = _load_payload(output)
            records.extend(payload.get("records", []))
            skipped.extend(payload.get("skipped", []))
            failures.extend(payload.get("failures", []))
            run_files.append(output.name)
        if completed.returncode != 0:
            failures.append(
                {
                    "threads_requested": threads,
                    "reason": f"child process exited with {completed.returncode}",
                }
            )

    combined = {
        "schema_version": 1,
        "benchmark": "fft-bench",
        "generated_utc": datetime.now(UTC).isoformat(),
        "run": {
            "kind": args.kind,
            "dtype": args.dtype,
            "sizes": args.sizes,
            "threads": args.threads,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "min_time_s": args.min_time,
            "seed": args.seed,
            "fresh_process_per_thread_count": True,
        },
        "run_files": run_files,
        "records": records,
        "skipped": skipped,
        "failures": failures,
    }
    combined_path = output_dir / "matrix.json"
    _write_payload(combined, combined_path)

    if args.plot:
        from .report import write_report

        report_dir = args.report if args.report is not None else output_dir / "report"
        write_report([combined], report_dir)
        print(f"wrote report to {report_dir}")

    if args.strict and (skipped or failures):
        return 1
    return 0 if records else 1


def make_report(args: argparse.Namespace) -> int:
    from .report import load_payload, write_report

    payloads = [load_payload(path) for path in args.input]
    write_report(payloads, args.output)
    print(f"wrote report to {args.output}")
    return 0


def _add_run_arguments(parser: argparse.ArgumentParser, *, matrix: bool) -> None:
    parser.add_argument(
        "--kind",
        type=_parse_kinds,
        default=list(KIND_NAMES),
        help="comma-separated transform kinds: r2c,c2c (default: both)",
    )
    parser.add_argument(
        "--dtype",
        type=_parse_dtypes,
        default=list(PRECISION_NAMES),
        help="comma-separated precision labels: float32,float64,longdouble",
    )
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SIZES.copy(),
        help="comma-separated positive 1-D FFT sizes",
    )
    parser.add_argument(
        "--threads",
        type=_parse_threads,
        default=[1],
        help="thread count(s), or 'auto' for a matrix",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--min-time", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if a requested case is unsupported or setup fails",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fft") if matrix else Path("results/fft/run.json"),
        help="JSON result path (run) or result directory (matrix)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fft-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run DUCC FFT benchmarks in this process")
    _add_run_arguments(run, matrix=False)
    run.set_defaults(func=run_benchmarks)

    matrix = subparsers.add_parser(
        "matrix", help="run each requested thread count in a fresh process"
    )
    _add_run_arguments(matrix, matrix=True)
    matrix.add_argument(
        "--plot",
        action="store_true",
        help="generate OUTPUT/report with Markdown and PNGs",
    )
    matrix.add_argument(
        "--report",
        type=Path,
        default=None,
        help="report directory (default: OUTPUT/report)",
    )
    matrix.set_defaults(func=run_matrix)

    report = subparsers.add_parser(
        "report", help="render a Markdown/PNG report from JSON result payloads"
    )
    report.add_argument("input", type=Path, nargs="+")
    report.add_argument("--output", type=Path, default=Path("results/fft/report"))
    report.set_defaults(func=make_report)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "warmup") and (
        args.warmup < 0 or args.repeat < 1 or args.min_time <= 0
    ):
        parser.error("warmup/repeat/min-time values are invalid")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
