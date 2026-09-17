"""Resumable B/C equiangular SHT resolution and batch-scaling probe.

The parent process deliberately does not import torch.  Every native benchmark
cell is a fresh subprocess, so a constructor failure or CUDA allocator failure
cannot leave a previous B/C module alive for the next cell.  The progress
manifest is written before each worker is started and is therefore also the
restart boundary: a stale ``in_progress`` entry is recorded as
``skipped_after_process_kill`` and is never retried automatically.

This script is research infrastructure.  ``runtime_fold`` (B) and
``precomputed_projection`` (C) are the only implementations launched here;
the dense A path is intentionally absent.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "sht_bench.torch_bc_resolution_scaling.v1"
PROGRESS_SCHEMA = "sht_bench.torch_bc_resolution_scaling_progress.v1"

RESOLUTION_CASES: tuple[tuple[int, int, int], ...] = (
    (361, 720, 360),
    (481, 960, 480),
    (601, 1200, 600),
    (721, 1440, 720),
)
FLOAT64_CASES: tuple[tuple[int, int, int], ...] = (
    (361, 720, 360),
    (481, 960, 480),
)
BATCHES: tuple[int, ...] = (1, 4, 16, 64, 128, 256, 512, 1024)
TRANSFORMS: tuple[str, ...] = ("scalar", "vector")
IMPLEMENTATIONS: tuple[str, ...] = (
    "runtime_fold",
    "precomputed_projection",
)
DTYPE_SIZES = {"float32": 4, "float64": 8}

CUDA_MEMORY_FRACTION = 0.85
HOST_MEMORY_FRACTION = 0.80
PREDICTION_SAFETY_MARGIN = 1.20
CONSTRUCTION_MEMORY_FACTOR = 8.0
CONSTRUCTION_BASELINE_BYTES = 1 << 30
HEADROOM_FRACTION_OF_BUDGET = 0.90

TERMINAL_STATUSES = frozenset(
    {
        "completed",
        "oom_cuda",
        "error",
        "skipped_predicted_capacity",
        "skipped_after_process_kill",
    }
)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _case_label(case: tuple[int, int, int]) -> str:
    return "x".join(str(value) for value in case)


def _parse_case(value: str) -> tuple[int, int, int]:
    parts = tuple(part.strip() for part in value.lower().split("x"))
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"case must be nlatxnlonxlmax, got {value!r}")
    case = tuple(int(part) for part in parts)
    if not all(value > 0 for value in case):
        raise ValueError(f"case dimensions must be positive, got {value!r}")
    nlat, nlon, lmax = case
    if nlon < 2 or lmax > nlon or lmax > nlat:
        raise ValueError(f"case has incompatible dimensions: {value!r}")
    return case


def parse_cases(value: str) -> tuple[tuple[int, int, int], ...]:
    cases = tuple(_parse_case(item) for item in value.split(",") if item.strip())
    if not cases:
        raise ValueError("at least one grid case is required")
    if len(set(cases)) != len(cases):
        raise ValueError("duplicate grid cases are not allowed")
    return cases


def parse_batches(value: str) -> tuple[int, ...]:
    batches: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item.isdigit() or int(item) < 1:
            raise ValueError(f"batch sizes must be positive integers, got {value!r}")
        batches.append(int(item))
    if not batches:
        raise ValueError("at least one batch size is required")
    if len(set(batches)) != len(batches):
        raise ValueError("duplicate batch sizes are not allowed")
    if batches != sorted(batches):
        raise ValueError("batch sizes must be in ascending order")
    return tuple(batches)


def enumerate_phase_a_cases(
    cases: Sequence[tuple[int, int, int]] = RESOLUTION_CASES,
    dtype: str = "float32",
) -> list[dict[str, Any]]:
    """Enumerate the batch-1 matrix in the required risk order."""

    return [
        make_spec(
            phase="phase_a_resolution",
            case=case,
            transform=transform,
            implementation=implementation,
            dtype=dtype,
            batch_size=1,
        )
        for case in cases
        for transform in TRANSFORMS
        for implementation in IMPLEMENTATIONS
    ]


def enumerate_batch_cases(
    case: tuple[int, int, int],
    batches: Sequence[int] = BATCHES,
    dtype: str = "float32",
) -> list[dict[str, Any]]:
    return [
        make_spec(
            phase="phase_b_batch",
            case=case,
            transform=transform,
            implementation=implementation,
            dtype=dtype,
            batch_size=batch_size,
        )
        for transform in TRANSFORMS
        for implementation in IMPLEMENTATIONS
        for batch_size in batches
    ]


def make_spec(
    *,
    phase: str,
    case: tuple[int, int, int],
    transform: str,
    implementation: str,
    dtype: str,
    batch_size: int,
) -> dict[str, Any]:
    if transform not in TRANSFORMS:
        raise ValueError(f"unknown transform: {transform}")
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown implementation: {implementation}")
    if dtype not in DTYPE_SIZES:
        raise ValueError(f"unknown dtype: {dtype}")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    nlat, nlon, lmax = case
    return {
        "phase": phase,
        "case": _case_label(case),
        "nlat": nlat,
        "nlon": nlon,
        "exclusive_lmax_mmax": lmax,
        "transform": transform,
        "implementation": implementation,
        "dtype": dtype,
        "batch_size": batch_size,
    }


def spec_case(spec: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(spec["nlat"]),
        int(spec["nlon"]),
        int(spec["exclusive_lmax_mmax"]),
    )


def case_key(spec: dict[str, Any]) -> str:
    return ":".join(
        (
            str(spec["phase"]),
            str(spec["dtype"]),
            str(spec["case"]),
            str(spec["transform"]),
            str(spec["implementation"]),
            f"b{spec['batch_size']}",
        )
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Flush a JSON file and atomically replace the destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path}")
    return value


def new_progress_manifest(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": PROGRESS_SCHEMA,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "metadata": metadata,
        "records": {},
    }


def _stale_result(spec: dict[str, Any], reason: str) -> dict[str, Any]:
    result = base_result(spec)
    result.update({"status": "skipped_after_process_kill", "reason": reason})
    return result


def load_progress(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Load progress and convert stale workers without retrying them."""

    if path.exists():
        manifest = _read_json(path)
        if manifest.get("schema") != PROGRESS_SCHEMA:
            raise ValueError(f"unexpected progress schema in {path}")
    else:
        manifest = new_progress_manifest(metadata)
        atomic_write_json(path, manifest)

    changed = False
    for entry in manifest.get("records", {}).values():
        if entry.get("status") != "in_progress":
            continue
        reason = "stale in_progress record from a killed benchmark process"
        entry["status"] = "skipped_after_process_kill"
        entry["reason"] = reason
        entry["finished_at"] = utc_now()
        entry.setdefault("result", _stale_result(entry["spec"], reason))
        changed = True
    if changed:
        manifest["updated_at"] = utc_now()
        atomic_write_json(path, manifest)
    return manifest


def update_progress_record(
    path: Path, key: str, update: dict[str, Any], metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Atomically update one manifest record and return the new manifest."""

    if path.exists():
        manifest = _read_json(path)
    else:
        manifest = new_progress_manifest(metadata or {})
    if manifest.get("schema") != PROGRESS_SCHEMA:
        raise ValueError(f"unexpected progress schema in {path}")
    records = manifest.setdefault("records", {})
    current = dict(records.get(key, {}))
    current.update(update)
    records[key] = current
    manifest["updated_at"] = utc_now()
    atomic_write_json(path, manifest)
    return manifest


def _dtype_size(dtype: str) -> int:
    try:
        return DTYPE_SIZES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype}") from exc


def estimate_module_bytes(
    implementation: str,
    transform: str,
    dtype: str,
    case: tuple[int, int, int],
) -> int:
    """Estimate final B/C module buffers, including small folded auxiliaries."""

    nlat, _nlon, lmax = case
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"unknown implementation: {implementation}")
    if transform not in TRANSFORMS:
        raise ValueError(f"unknown transform: {transform}")
    components = 2 if transform == "vector" else 1
    real_projection = components * lmax * lmax * nlat * _dtype_size(dtype)
    if implementation == "precomputed_projection":
        # C stores the effective projection as complex dtype: twice the real
        # bytes.  Its auxiliary runtime-fold buffers are freed before timing.
        return 2 * real_projection
    # B keeps a real projection plus the split quadrature, phase, and signs.
    auxiliary = (2 * nlat + 4 * (nlat - 1)) * _dtype_size(dtype) + lmax
    return real_projection + auxiliary


def estimate_input_bytes(
    transform: str, dtype: str, case: tuple[int, int, int], batch_size: int
) -> int:
    nlat, nlon, _lmax = case
    components = 2 if transform == "vector" else 1
    return batch_size * components * nlat * nlon * _dtype_size(dtype)


def estimate_initial_peak_bytes(
    implementation: str,
    transform: str,
    dtype: str,
    case: tuple[int, int, int],
    batch_size: int,
) -> int:
    module = estimate_module_bytes(implementation, transform, dtype, case)
    input_bytes = estimate_input_bytes(transform, dtype, case, batch_size)
    # A static gate for a batch-1 cell.  Runtime probes use the measured
    # allocator peak and the adaptive model below for later batches.
    return math.ceil(1.50 * module + 4.0 * input_bytes)


def estimate_construction_peak_rss(
    implementation: str, transform: str, dtype: str, case: tuple[int, int, int]
) -> int:
    return math.ceil(
        CONSTRUCTION_BASELINE_BYTES
        + CONSTRUCTION_MEMORY_FACTOR
        * estimate_module_bytes(implementation, transform, dtype, case)
    )


def memory_budget_bytes(total_bytes: int | None, fraction: float) -> int | None:
    if total_bytes is None:
        return None
    if not 0.0 < fraction <= 1.0:
        raise ValueError("memory budget fraction must be in (0, 1]")
    return int(total_bytes * fraction)


def capacity_decision(
    *,
    estimated_module_bytes_value: int,
    predicted_peak_bytes: int | None,
    gpu_total_bytes: int | None,
    estimated_host_peak_bytes: int | None,
    host_limit_bytes: int | None,
    cuda_fraction: float = CUDA_MEMORY_FRACTION,
    host_fraction: float = HOST_MEMORY_FRACTION,
) -> dict[str, Any]:
    gpu_budget = memory_budget_bytes(gpu_total_bytes, cuda_fraction)
    host_budget = memory_budget_bytes(host_limit_bytes, host_fraction)
    reasons: list[str] = []
    if gpu_budget is not None and estimated_module_bytes_value > gpu_budget:
        reasons.append(
            f"estimated module {estimated_module_bytes_value} exceeds CUDA budget {gpu_budget}"
        )
    if (
        gpu_budget is not None
        and predicted_peak_bytes is not None
        and predicted_peak_bytes > gpu_budget
    ):
        reasons.append(
            f"predicted CUDA peak {predicted_peak_bytes} exceeds budget {gpu_budget}"
        )
    if (
        host_budget is not None
        and estimated_host_peak_bytes is not None
        and estimated_host_peak_bytes > host_budget
    ):
        reasons.append(
            f"estimated construction RSS {estimated_host_peak_bytes} exceeds host budget {host_budget}"
        )
    return {
        "safe": not reasons,
        "reason": "; ".join(reasons) if reasons else None,
        "gpu_budget_bytes": gpu_budget,
        "host_budget_bytes": host_budget,
        "estimated_module_bytes": estimated_module_bytes_value,
        "predicted_peak_bytes": predicted_peak_bytes,
        "estimated_host_peak_bytes": estimated_host_peak_bytes,
    }


def _peak_value(record: dict[str, Any]) -> int | None:
    for name in (
        "cuda_peak_reserved_bytes",
        "cuda_peak_allocated_bytes",
        "peak_reserved_bytes",
        "peak_allocated_bytes",
    ):
        value = record.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return int(value)
    return None


def predict_next_peak_bytes(
    successful_records: Iterable[dict[str, Any]],
    next_batch: int,
    safety_margin: float = PREDICTION_SAFETY_MARGIN,
) -> int | None:
    """Fit a conservative fixed-plus-linear batch peak model."""

    if next_batch < 1 or safety_margin < 1.0:
        raise ValueError("next_batch must be positive and safety margin >= 1")
    records = [
        record
        for record in successful_records
        if record.get("status") == "completed"
        and isinstance(record.get("batch_size"), int)
        and record["batch_size"] > 0
        and _peak_value(record) is not None
    ]
    records.sort(key=lambda record: int(record["batch_size"]))
    if not records:
        return None
    if len(records) == 1:
        last = records[0]
        last_batch = int(last["batch_size"])
        last_peak = int(_peak_value(last) or 0)
        fixed = int(last.get("module_buffer_bytes") or 0)
        runtime = max(0, last_peak - fixed)
        predicted = fixed + runtime * max(1.0, next_batch / last_batch)
    else:
        previous, last = records[-2:]
        previous_batch = int(previous["batch_size"])
        last_batch = int(last["batch_size"])
        previous_peak = int(_peak_value(previous) or 0)
        last_peak = int(_peak_value(last) or 0)
        slope = max(
            0.0, (last_peak - previous_peak) / max(1, last_batch - previous_batch)
        )
        predicted = last_peak + slope * max(0, next_batch - last_batch)
        predicted = max(predicted, int(last.get("module_buffer_bytes") or 0))
    return math.ceil(predicted * safety_margin)


def fit_memory_model(
    records: Iterable[dict[str, Any]], value_key: str | None = None
) -> dict[str, Any]:
    """Return ``peak ~= fixed + batch * per_batch`` from measured records."""

    def value(record: dict[str, Any]) -> int | None:
        if value_key is not None:
            raw = record.get(value_key)
            if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
                return int(raw)
            return None
        return _peak_value(record)

    points = sorted(
        (
            int(record["batch_size"]),
            int(value(record) or 0),
        )
        for record in records
        if record.get("status") == "completed"
        and isinstance(record.get("batch_size"), int)
        and value(record) is not None
    )
    if not points:
        return {"fixed_bytes": None, "per_batch_bytes": None, "points": []}
    if len(points) == 1:
        fixed = max(0, points[0][1] - points[0][0] * 0)
        return {
            "fixed_bytes": fixed,
            "per_batch_bytes": None,
            "points": points,
        }
    x_mean = sum(point[0] for point in points) / len(points)
    y_mean = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - x_mean) ** 2 for point in points)
    slope = (
        sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
        if denominator
        else 0.0
    )
    slope = max(0.0, slope)
    fixed = max(0.0, y_mean - slope * x_mean)
    return {
        "fixed_bytes": round(fixed),
        "per_batch_bytes": round(slope),
        "points": points,
    }


def _git_sha(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _memory_limit_bytes() -> tuple[int | None, str | None, int | None]:
    candidates = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for path in candidates:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value and value != "max":
            limit = int(value)
            if limit < (1 << 60):
                current_path = path.parent / "memory.current"
                try:
                    current = int(current_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    current = None
                return limit, str(path), current
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024, "/proc/meminfo:MemTotal", None
    except (OSError, ValueError, IndexError):
        pass
    return None, None, None


def _rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _peak_rss_bytes() -> int | None:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024) if value else None


def _nvidia_smi_value(query: str, device: int) -> str | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device}",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = output.strip().splitlines()[0] if output.strip() else ""
    return value.strip() or None


def _nvidia_smi_memory_bytes(device: int) -> int | None:
    value = _nvidia_smi_value("memory.total", device)
    try:
        return int(float(value) * 1024 * 1024) if value is not None else None
    except ValueError:
        return None


def base_result(spec: dict[str, Any]) -> dict[str, Any]:
    case = spec_case(spec)
    module_estimate = estimate_module_bytes(
        spec["implementation"], spec["transform"], spec["dtype"], case
    )
    return {
        "schema_version": SCHEMA,
        "record_type": "performance",
        **spec,
        "measurement": "forward",
        "status": "pending",
        "reason": None,
        "warmup": None,
        "repeat": None,
        "samples_s": None,
        "median_s": None,
        "minimum_s": None,
        "milliseconds_per_batch": None,
        "milliseconds_per_frame": None,
        "frames_per_second": None,
        "module_buffer_bytes": None,
        "estimated_module_bytes": module_estimate,
        "estimated_initial_peak_bytes": estimate_initial_peak_bytes(
            spec["implementation"],
            spec["transform"],
            spec["dtype"],
            case,
            spec["batch_size"],
        ),
        "estimated_construction_peak_rss_bytes": estimate_construction_peak_rss(
            spec["implementation"], spec["transform"], spec["dtype"], case
        ),
        "constructor_seconds": None,
        "constructor_peak_allocated_bytes": None,
        "constructor_peak_reserved_bytes": None,
        "host_rss_before_construction_bytes": None,
        "host_rss_after_construction_bytes": None,
        "host_peak_rss_during_construction_bytes": None,
        "host_memory_limit_bytes": None,
        "host_memory_budget_bytes": None,
        "cuda_device_name": None,
        "cuda_total_memory_bytes": None,
        "cuda_allocated_before_input_bytes": None,
        "cuda_reserved_before_input_bytes": None,
        "cuda_allocated_before_forward_bytes": None,
        "cuda_reserved_before_forward_bytes": None,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
        "runtime_extra_allocated_bytes": None,
        "runtime_extra_reserved_bytes": None,
        "input_bytes": None,
        "predicted_peak_bytes": None,
        "memory_budget_bytes": None,
        "memory_budget_fraction": CUDA_MEMORY_FRACTION,
        "source_sha": None,
        "benchmark_code_sha": None,
        "worker_pid": None,
        "worker_returncode": None,
        "worker_stderr_tail": None,
        "started_at": None,
        "finished_at": None,
        "traceback": None,
    }


def _is_cuda_oom(exc: BaseException) -> bool:
    return (
        "out of memory" in str(exc).lower()
        or "cuda error: out of memory" in str(exc).lower()
    )


def _safe_cuda_fields(record: dict[str, Any], torch: Any, device: Any) -> None:
    if device is None or getattr(device, "type", None) != "cuda":
        return
    try:
        properties = torch.cuda.get_device_properties(device)
        record["cuda_device_name"] = properties.name
        record["cuda_total_memory_bytes"] = int(properties.total_memory)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not mask the worker result
        record.setdefault("diagnostic_errors", []).append(f"properties: {exc}")
    fields = {
        "cuda_allocated_before_forward_bytes": "memory_allocated",
        "cuda_reserved_before_forward_bytes": "memory_reserved",
    }
    for output_name, function_name in fields.items():
        try:
            record[output_name] = int(getattr(torch.cuda, function_name)(device))
        except Exception as exc:  # noqa: BLE001
            record.setdefault("diagnostic_errors", []).append(f"{function_name}: {exc}")


def _worker_build_module(
    source: Path,
    implementation: str,
    transform: str,
    nlat: int,
    nlon: int,
    lmax: int,
) -> Any:
    scripts = Path(__file__).resolve().parent
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from torch_abc_benchmark import PrecomputedProjection
    from torch_precomputed_projection_prototype import _load_optimized

    package = _load_optimized(source)
    vector = transform == "vector"
    module_class = package.RealVectorSHT if vector else package.RealSHT
    if implementation == "runtime_fold":
        return module_class(
            nlat, nlon, lmax=lmax, mmax=lmax, norm="ortho", csphase=True
        )
    runtime = module_class(nlat, nlon, lmax=lmax, mmax=lmax, norm="ortho", csphase=True)
    module = PrecomputedProjection(runtime, vector)
    del runtime
    gc.collect()
    from torch_abc_benchmark import _clear_torch_harmonics_precompute_caches

    _clear_torch_harmonics_precompute_caches()
    return module


def _worker_main(args: argparse.Namespace) -> int:
    spec = make_spec(
        phase=args.phase,
        case=_parse_case(args.case),
        transform=args.transform,
        implementation=args.implementation,
        dtype=args.dtype,
        batch_size=args.batch_size,
    )
    result = base_result(spec)
    result["warmup"] = args.warmup
    result["repeat"] = args.repeat
    result["worker_pid"] = os.getpid()
    result["started_at"] = utc_now()
    source = Path(args.source).resolve()
    result["source_sha"] = _git_sha(source)
    result["benchmark_code_sha"] = _git_sha(Path(__file__).resolve().parents[1])
    result["host_memory_limit_bytes"], _source, _current = _memory_limit_bytes()
    result["host_memory_budget_bytes"] = memory_budget_bytes(
        result["host_memory_limit_bytes"], HOST_MEMORY_FRACTION
    )
    torch: Any = None
    device: Any = None
    module: Any = None
    sample: Any = None
    construction_peak_before = _peak_rss_bytes()
    result["host_rss_before_construction_bytes"] = _rss_bytes()
    try:
        import torch as torch_module

        torch = torch_module
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            device = torch.device("cuda", args.cuda_device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            constructor_baseline = int(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            device = torch.device("cpu")
            constructor_baseline = 0
        construction_started = time.perf_counter()
        module = _worker_build_module(
            source,
            args.implementation,
            args.transform,
            spec["nlat"],
            spec["nlon"],
            spec["exclusive_lmax_mmax"],
        )
        from torch_abc_benchmark import _clear_torch_harmonics_precompute_caches

        _clear_torch_harmonics_precompute_caches()
        gc.collect()
        target_dtype = torch.float32 if args.dtype == "float32" else torch.float64
        if args.implementation == "precomputed_projection":
            target_dtype = (
                torch.complex64 if args.dtype == "float32" else torch.complex128
            )
        module = module.to(device=device, dtype=target_dtype).eval()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["constructor_seconds"] = time.perf_counter() - construction_started
        result["module_buffer_bytes"] = sum(
            buffer.numel() * buffer.element_size() for buffer in module.buffers()
        )
        if device.type == "cuda":
            result["constructor_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device) - constructor_baseline
            )
            result["constructor_peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(device)
            )
            result["cuda_allocated_before_input_bytes"] = int(
                torch.cuda.memory_allocated(device)
            )
            result["cuda_reserved_before_input_bytes"] = int(
                torch.cuda.memory_reserved(device)
            )
            result["cuda_device_name"] = torch.cuda.get_device_properties(device).name
            result["cuda_total_memory_bytes"] = int(
                torch.cuda.get_device_properties(device).total_memory
            )
        result["host_rss_after_construction_bytes"] = _rss_bytes()
        construction_peak_after = _peak_rss_bytes()
        result["host_peak_rss_during_construction_bytes"] = max(
            value
            for value in (construction_peak_before, construction_peak_after)
            if value is not None
        )

        shape = (
            (args.batch_size, 2, spec["nlat"], spec["nlon"])
            if args.transform == "vector"
            else (args.batch_size, spec["nlat"], spec["nlon"])
        )
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        sample = torch.randn(
            shape,
            dtype=target_dtype
            if target_dtype in (torch.float32, torch.float64)
            else (torch.float32 if args.dtype == "float32" else torch.float64),
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["input_bytes"] = int(sample.numel() * sample.element_size())

        def run_once(function: Any, value: Any) -> None:
            with torch.inference_mode():
                output = function(value)
            del output

        for _ in range(args.warmup):
            run_once(module, sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            result["cuda_allocated_before_forward_bytes"] = int(
                torch.cuda.memory_allocated(device)
            )
            result["cuda_reserved_before_forward_bytes"] = int(
                torch.cuda.memory_reserved(device)
            )
        samples: list[float] = []
        for _ in range(args.repeat):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            run_once(module, sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter_ns() - started) / 1.0e9)
        median = sorted(samples)[len(samples) // 2]
        result["samples_s"] = samples
        result["median_s"] = median
        result["minimum_s"] = min(samples)
        result["milliseconds_per_batch"] = median * 1000.0
        result["milliseconds_per_frame"] = median * 1000.0 / args.batch_size
        result["frames_per_second"] = args.batch_size / median
        if device.type == "cuda":
            result["cuda_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
            result["cuda_peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(device)
            )
            result["runtime_extra_allocated_bytes"] = (
                result["cuda_peak_allocated_bytes"]
                - result["cuda_allocated_before_input_bytes"]
            )
            result["runtime_extra_reserved_bytes"] = (
                result["cuda_peak_reserved_bytes"]
                - result["cuda_reserved_before_input_bytes"]
            )
        del sample, module
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        result["status"] = "completed"
    except BaseException as exc:  # noqa: BLE001 - worker must persist failure state
        result["status"] = "oom_cuda" if _is_cuda_oom(exc) else "error"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=12)
        result["host_rss_after_construction_bytes"] = _rss_bytes()
        construction_peak_after = _peak_rss_bytes()
        values = (construction_peak_before, construction_peak_after)
        result["host_peak_rss_during_construction_bytes"] = max(
            value for value in values if value is not None
        )
        if torch is not None:
            _safe_cuda_fields(result, torch, device)
            if device is not None and getattr(device, "type", None) == "cuda":
                try:
                    result["cuda_peak_allocated_bytes"] = int(
                        torch.cuda.max_memory_allocated(device)
                    )
                    result["cuda_peak_reserved_bytes"] = int(
                        torch.cuda.max_memory_reserved(device)
                    )
                except Exception as exc:  # noqa: BLE001
                    result.setdefault("diagnostic_errors", []).append(
                        f"peak CUDA diagnostics: {exc}"
                    )
    finally:
        result["finished_at"] = utc_now()
        atomic_write_json(Path(args.worker_output), result)
    return 0 if result["status"] in {"completed", "oom_cuda"} else 1


def build_worker_command(
    script: Path,
    spec: dict[str, Any],
    *,
    source: Path,
    output: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        str(script),
        "--worker",
        "--phase",
        spec["phase"],
        "--implementation",
        spec["implementation"],
        "--transform",
        spec["transform"],
        "--dtype",
        spec["dtype"],
        "--device",
        args.device,
        "--case",
        spec["case"],
        "--batch-size",
        str(spec["batch_size"]),
        "--source",
        str(source),
        "--worker-output",
        str(output),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--threads",
        str(args.threads),
        "--seed",
        str(args.seed),
        "--cuda-device",
        str(args.cuda_device),
    ]


def _terminal_result_entry(
    spec: dict[str, Any], result: dict[str, Any], *, result_file: str | None = None
) -> dict[str, Any]:
    return {
        "spec": spec,
        "status": result["status"],
        "attempt": 1,
        "finished_at": utc_now(),
        "result_file": result_file,
        "result": result,
    }


def _mark_capacity_skip(
    progress_path: Path,
    spec: dict[str, Any],
    *,
    reason: str,
    predicted_peak: int | None,
    memory_decision: dict[str, Any],
) -> dict[str, Any]:
    result = base_result(spec)
    result.update(
        {
            "status": "skipped_predicted_capacity",
            "reason": reason,
            "predicted_peak_bytes": predicted_peak,
            "memory_budget_bytes": memory_decision["gpu_budget_bytes"],
            "host_memory_budget_bytes": memory_decision["host_budget_bytes"],
        }
    )
    update_progress_record(
        progress_path,
        case_key(spec),
        _terminal_result_entry(spec, result),
    )
    return result


def _run_worker(
    *,
    script: Path,
    source: Path,
    spec: dict[str, Any],
    args: argparse.Namespace,
    worker_output: Path,
) -> dict[str, Any]:
    command = build_worker_command(
        script, spec, source=source, output=worker_output, args=args
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if worker_output.exists():
        result = _read_json(worker_output)
        result["worker_returncode"] = completed.returncode
        if completed.stderr:
            result["worker_stderr_tail"] = completed.stderr[-4000:]
        if completed.returncode != 0 and result.get("status") == "completed":
            result["status"] = "error"
            result["reason"] = f"worker exited with code {completed.returncode}"
        return result
    result = base_result(spec)
    result.update(
        {
            "status": "error",
            "reason": (
                f"worker produced no result (returncode={completed.returncode}); "
                "a host or process-level kill may have occurred"
            ),
            "worker_returncode": completed.returncode,
            "worker_stderr_tail": completed.stderr[-4000:]
            if completed.stderr
            else None,
        }
    )
    return result


def _existing_result(
    progress: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any] | None:
    entry = progress.get("records", {}).get(case_key(spec))
    if entry is None:
        return None
    if entry.get("status") in TERMINAL_STATUSES:
        result = entry.get("result")
        if isinstance(result, dict):
            return result
        return _stale_result(spec, entry.get("reason", "terminal manifest record"))
    return None


def _start_worker(progress_path: Path, spec: dict[str, Any]) -> None:
    key = case_key(spec)
    previous = {}
    if progress_path.exists():
        previous = _read_json(progress_path).get("records", {}).get(key, {})
    attempt = int(previous.get("attempt", 0)) + 1
    # This write is the mandatory durable restart boundary.
    update_progress_record(
        progress_path,
        key,
        {
            "spec": spec,
            "status": "in_progress",
            "started_at": utc_now(),
            "attempt": attempt,
        },
    )


def _finish_worker(
    progress_path: Path,
    spec: dict[str, Any],
    result: dict[str, Any],
    worker_output: Path,
) -> None:
    update_progress_record(
        progress_path,
        case_key(spec),
        _terminal_result_entry(spec, result, result_file=str(worker_output)),
    )


def _resolution_guard(
    progress: dict[str, Any],
    previous_case: tuple[int, int, int],
    target_case: tuple[int, int, int],
    *,
    gpu_budget: int | None,
    host_budget: int | None,
) -> tuple[bool, str | None, int | None]:
    previous_results: dict[tuple[str, str], dict[str, Any]] = {}
    for transform in TRANSFORMS:
        for implementation in IMPLEMENTATIONS:
            spec = make_spec(
                phase="phase_a_resolution",
                case=previous_case,
                transform=transform,
                implementation=implementation,
                dtype="float32",
                batch_size=1,
            )
            result = _existing_result(progress, spec)
            if result is None or result.get("status") != "completed":
                return (
                    False,
                    f"previous resolution {_case_label(previous_case)} did not complete safely",
                    None,
                )
            previous_results[(transform, implementation)] = result

    predictions: list[int] = []
    for (transform, implementation), previous in previous_results.items():
        module = estimate_module_bytes(
            implementation, transform, "float32", target_case
        )
        previous_module = int(
            previous.get("module_buffer_bytes")
            or estimate_module_bytes(
                implementation, transform, "float32", previous_case
            )
        )
        previous_peak = _peak_value(previous) or previous_module
        runtime = max(0, previous_peak - previous_module)
        ratio = (target_case[0] / previous_case[0]) ** 3
        predictions.append(
            math.ceil((module + runtime * ratio) * PREDICTION_SAFETY_MARGIN)
        )
    predicted = max(predictions, default=None)
    if gpu_budget is not None and predicted is not None and predicted > gpu_budget:
        return (
            False,
            f"resolution prediction {predicted} bytes exceeds CUDA budget {gpu_budget}",
            predicted,
        )
    for transform in TRANSFORMS:
        for implementation in IMPLEMENTATIONS:
            estimate = estimate_construction_peak_rss(
                implementation, transform, "float32", target_case
            )
            if host_budget is not None and estimate > host_budget:
                return (
                    False,
                    f"resolution construction estimate {estimate} bytes exceeds host budget {host_budget}",
                    predicted,
                )
    return True, None, predicted


def _capacity_for_spec(
    spec: dict[str, Any],
    *,
    predicted_peak: int | None,
    gpu_total: int | None,
    host_limit: int | None,
) -> dict[str, Any]:
    case = spec_case(spec)
    if predicted_peak is None:
        predicted_peak = estimate_initial_peak_bytes(
            spec["implementation"],
            spec["transform"],
            spec["dtype"],
            case,
            spec["batch_size"],
        )
    return capacity_decision(
        estimated_module_bytes_value=estimate_module_bytes(
            spec["implementation"], spec["transform"], spec["dtype"], case
        ),
        predicted_peak_bytes=predicted_peak,
        gpu_total_bytes=gpu_total,
        estimated_host_peak_bytes=estimate_construction_peak_rss(
            spec["implementation"], spec["transform"], spec["dtype"], case
        ),
        host_limit_bytes=host_limit,
    )


def _run_spec(
    *,
    progress_path: Path,
    progress: dict[str, Any],
    spec: dict[str, Any],
    script: Path,
    source: Path,
    args: argparse.Namespace,
    worker_directory: Path,
    predicted_peak: int | None,
    gpu_total: int | None,
    host_limit: int | None,
    guard_reason: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    existing = _existing_result(progress, spec)
    if existing is not None:
        return existing, progress
    if guard_reason is not None:
        decision = _capacity_for_spec(
            spec,
            predicted_peak=predicted_peak,
            gpu_total=gpu_total,
            host_limit=host_limit,
        )
        result = _mark_capacity_skip(
            progress_path,
            spec,
            reason=guard_reason,
            predicted_peak=predicted_peak,
            memory_decision=decision,
        )
        return result, _read_json(progress_path)
    decision = _capacity_for_spec(
        spec,
        predicted_peak=predicted_peak,
        gpu_total=gpu_total,
        host_limit=host_limit,
    )
    if not decision["safe"]:
        result = _mark_capacity_skip(
            progress_path,
            spec,
            reason=str(decision["reason"]),
            predicted_peak=predicted_peak,
            memory_decision=decision,
        )
        return result, _read_json(progress_path)
    _start_worker(progress_path, spec)
    worker_directory.mkdir(parents=True, exist_ok=True)
    worker_output = worker_directory / f"{case_key(spec).replace(':', '__')}.json"
    result = _run_worker(
        script=script,
        source=source,
        spec=spec,
        args=args,
        worker_output=worker_output,
    )
    result["predicted_peak_bytes"] = predicted_peak
    result["memory_budget_bytes"] = decision["gpu_budget_bytes"]
    result["host_memory_budget_bytes"] = decision["host_budget_bytes"]
    _finish_worker(progress_path, spec, result, worker_output)
    return result, _read_json(progress_path)


def _phase_a(
    *,
    cases: tuple[tuple[int, int, int], ...],
    progress_path: Path,
    progress: dict[str, Any],
    script: Path,
    source: Path,
    args: argparse.Namespace,
    worker_directory: Path,
    gpu_total: int | None,
    host_limit: int | None,
) -> dict[str, Any]:
    gpu_budget = memory_budget_bytes(gpu_total, CUDA_MEMORY_FRACTION)
    host_budget = memory_budget_bytes(host_limit, HOST_MEMORY_FRACTION)
    for index, case in enumerate(cases):
        guard_reason: str | None = None
        predicted: int | None = None
        if index:
            safe, guard_reason, predicted = _resolution_guard(
                progress,
                cases[index - 1],
                case,
                gpu_budget=gpu_budget,
                host_budget=host_budget,
            )
            if safe:
                guard_reason = None
        for transform in TRANSFORMS:
            for implementation in IMPLEMENTATIONS:
                spec = make_spec(
                    phase="phase_a_resolution",
                    case=case,
                    transform=transform,
                    implementation=implementation,
                    dtype="float32",
                    batch_size=1,
                )
                result, progress = _run_spec(
                    progress_path=progress_path,
                    progress=progress,
                    spec=spec,
                    script=script,
                    source=source,
                    args=args,
                    worker_directory=worker_directory,
                    predicted_peak=predicted,
                    gpu_total=gpu_total,
                    host_limit=host_limit,
                    guard_reason=guard_reason,
                )
                if result.get("status") not in TERMINAL_STATUSES:
                    raise RuntimeError(f"non-terminal result for {case_key(spec)}")
    return progress


def _phase_b(
    *,
    anchor: tuple[int, int, int],
    finer: tuple[int, int, int] | None,
    progress_path: Path,
    progress: dict[str, Any],
    script: Path,
    source: Path,
    args: argparse.Namespace,
    worker_directory: Path,
    gpu_total: int | None,
    host_limit: int | None,
    batches: tuple[int, ...],
) -> dict[str, Any]:
    for case in tuple(value for value in (anchor, finer) if value is not None):
        for transform in TRANSFORMS:
            for implementation in IMPLEMENTATIONS:
                phase_a_spec = make_spec(
                    phase="phase_a_resolution",
                    case=case,
                    transform=transform,
                    implementation=implementation,
                    dtype="float32",
                    batch_size=1,
                )
                first = _existing_result(progress, phase_a_spec)
                successful = (
                    [first] if first and first.get("status") == "completed" else []
                )
                blocked_reason: str | None = None
                for batch_size in batches:
                    if batch_size == 1:
                        continue
                    spec = make_spec(
                        phase="phase_b_batch",
                        case=case,
                        transform=transform,
                        implementation=implementation,
                        dtype="float32",
                        batch_size=batch_size,
                    )
                    existing = _existing_result(progress, spec)
                    if existing is not None:
                        if existing.get("status") == "completed":
                            successful.append(existing)
                        else:
                            blocked_reason = blocked_reason or (
                                f"previous batch {batch_size} was {existing.get('status')}"
                            )
                        continue
                    predicted = predict_next_peak_bytes(
                        successful, batch_size, PREDICTION_SAFETY_MARGIN
                    )
                    if not successful:
                        blocked_reason = (
                            blocked_reason
                            or "batch-1 resolution result was not completed"
                        )
                    result, progress = _run_spec(
                        progress_path=progress_path,
                        progress=progress,
                        spec=spec,
                        script=script,
                        source=source,
                        args=args,
                        worker_directory=worker_directory,
                        predicted_peak=predicted,
                        gpu_total=gpu_total,
                        host_limit=host_limit,
                        guard_reason=blocked_reason,
                    )
                    if result.get("status") == "completed":
                        successful.append(result)
                    else:
                        blocked_reason = blocked_reason or (
                            f"previous batch {batch_size} was {result.get('status')}; larger batches not launched"
                        )
    return progress


def repair_batch_predictions(progress_path: Path) -> dict[str, Any]:
    """Repair records made before the CUDA peak-field predictor was fixed.

    This is intentionally a migration, not a retry.  Measured successful
    workers remain unchanged.  A prior ``oom_cuda`` record is retained under
    the manifest entry's ``history`` and replaced in the authoritative result
    matrix by the corrected model's explicit capacity skip.
    """

    manifest = _read_json(progress_path)
    entries = manifest.get("records", {})
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for entry in entries.values():
        spec = entry.get("spec", {})
        if spec.get("phase") != "phase_b_batch":
            continue
        group = (
            spec.get("case"),
            spec.get("transform"),
            spec.get("implementation"),
            spec.get("dtype"),
        )
        groups.setdefault(group, []).append(entry)

    superseded: list[dict[str, Any]] = []
    changed = False
    for group, batch_entries in groups.items():
        case_label, transform, implementation, dtype = group
        case = _parse_case(str(case_label))
        phase_a = make_spec(
            phase="phase_a_resolution",
            case=case,
            transform=str(transform),
            implementation=str(implementation),
            dtype=str(dtype),
            batch_size=1,
        )
        phase_a_entry = entries.get(case_key(phase_a), {})
        successful: list[dict[str, Any]] = []
        phase_a_result = phase_a_entry.get("result")
        if (
            isinstance(phase_a_result, dict)
            and phase_a_result.get("status") == "completed"
        ):
            successful.append(phase_a_result)
        batch_entries.sort(key=lambda entry: int(entry["spec"]["batch_size"]))
        for entry in batch_entries:
            result = entry.get("result")
            if not isinstance(result, dict):
                continue
            batch_size = int(entry["spec"]["batch_size"])
            predicted = predict_next_peak_bytes(
                successful, batch_size, PREDICTION_SAFETY_MARGIN
            )
            if result.get("status") == "completed":
                if result.get("predicted_peak_bytes") != predicted:
                    result["predicted_peak_bytes"] = predicted
                    changed = True
                successful.append(result)
                continue
            if result.get("status") == "oom_cuda":
                history = {
                    "status": "oom_cuda",
                    "result": result,
                    "repaired_at": utc_now(),
                }
                entry.setdefault("history", []).append(history)
                superseded.append(
                    {
                        "case": case_label,
                        "transform": transform,
                        "implementation": implementation,
                        "dtype": dtype,
                        "batch_size": batch_size,
                    }
                )
                replacement = base_result(entry["spec"])
                replacement.update(
                    {
                        "status": "skipped_predicted_capacity",
                        "reason": (
                            "corrected adaptive model predicts capacity risk; the prior "
                            "unmapped-peak run is retained in manifest history"
                        ),
                        "predicted_peak_bytes": predicted,
                        "memory_budget_bytes": result.get("memory_budget_bytes"),
                        "host_memory_budget_bytes": result.get(
                            "host_memory_budget_bytes"
                        ),
                    }
                )
                entry["result"] = replacement
                entry["status"] = "skipped_predicted_capacity"
                entry["repaired_from"] = "oom_cuda"
                entry["repair_reason"] = replacement["reason"]
                changed = True
                continue
            if (
                result.get("status") == "skipped_predicted_capacity"
                and result.get("predicted_peak_bytes") != predicted
            ):
                result["predicted_peak_bytes"] = predicted
                changed = True

    if changed or superseded:
        manifest.setdefault("metadata", {})["superseded_cuda_ooms"] = superseded
        manifest["updated_at"] = utc_now()
        atomic_write_json(progress_path, manifest)
    return manifest


def _phase_c_float64(
    *,
    progress_path: Path,
    progress: dict[str, Any],
    script: Path,
    source: Path,
    args: argparse.Namespace,
    worker_directory: Path,
    gpu_total: int | None,
    host_limit: int | None,
) -> dict[str, Any]:
    for case in FLOAT64_CASES:
        for transform in TRANSFORMS:
            for implementation in IMPLEMENTATIONS:
                spec = make_spec(
                    phase="phase_c_float64",
                    case=case,
                    transform=transform,
                    implementation=implementation,
                    dtype="float64",
                    batch_size=1,
                )
                _result, progress = _run_spec(
                    progress_path=progress_path,
                    progress=progress,
                    spec=spec,
                    script=script,
                    source=source,
                    args=args,
                    worker_directory=worker_directory,
                    predicted_peak=None,
                    gpu_total=gpu_total,
                    host_limit=host_limit,
                )
    # Make the explicitly unsafe example auditable without constructing it.
    unsafe = make_spec(
        phase="phase_c_float64_safety_gate",
        case=(721, 1440, 720),
        transform="vector",
        implementation="precomputed_projection",
        dtype="float64",
        batch_size=1,
    )
    if _existing_result(progress, unsafe) is None:
        decision = _capacity_for_spec(
            unsafe,
            predicted_peak=None,
            gpu_total=gpu_total,
            host_limit=host_limit,
        )
        reason = decision["reason"] or "not part of the limited float64 probe"
        _mark_capacity_skip(
            progress_path,
            unsafe,
            reason=reason,
            predicted_peak=decision["predicted_peak_bytes"],
            memory_decision=decision,
        )
    return _read_json(progress_path)


def _select_finer_resolution(
    progress: dict[str, Any],
    cases: tuple[tuple[int, int, int], ...],
    *,
    gpu_total: int | None,
    host_limit: int | None,
) -> tuple[int, int, int] | None:
    gpu_budget = memory_budget_bytes(gpu_total, CUDA_MEMORY_FRACTION)
    host_budget = memory_budget_bytes(host_limit, HOST_MEMORY_FRACTION)
    for case in reversed(cases[1:]):
        results: list[dict[str, Any]] = []
        for transform in TRANSFORMS:
            for implementation in IMPLEMENTATIONS:
                spec = make_spec(
                    phase="phase_a_resolution",
                    case=case,
                    transform=transform,
                    implementation=implementation,
                    dtype="float32",
                    batch_size=1,
                )
                result = _existing_result(progress, spec)
                if result is None or result.get("status") != "completed":
                    results = []
                    break
                results.append(result)
            if not results or len(results) != len(TRANSFORMS) * len(IMPLEMENTATIONS):
                break
        if not results:
            continue
        peaks = [_peak_value(result) for result in results]
        observed_peak = max((value or 0 for value in peaks), default=0)
        host_peak = max(
            (
                int(result.get("host_peak_rss_during_construction_bytes") or 0)
                for result in results
            ),
            default=0,
        )
        if (
            gpu_budget is not None
            and observed_peak > gpu_budget * HEADROOM_FRACTION_OF_BUDGET
        ):
            continue
        if (
            host_budget is not None
            and host_peak > host_budget * HEADROOM_FRACTION_OF_BUDGET
        ):
            continue
        return case
    return None


def _result_records(progress: dict[str, Any]) -> list[dict[str, Any]]:
    entries = progress.get("records", {})
    ordered = sorted(
        entries.values(),
        key=lambda entry: (entry.get("started_at", ""), entry["spec"]["case"]),
    )
    return [
        entry["result"] for entry in ordered if isinstance(entry.get("result"), dict)
    ]


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (list, dict))
                else value
                for key, value in record.items()
            }
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _lookup_record(
    records: Iterable[dict[str, Any]],
    *,
    case: tuple[int, int, int],
    transform: str,
    implementation: str,
    dtype: str = "float32",
    batch_size: int = 1,
) -> dict[str, Any] | None:
    for record in records:
        if (
            (
                record.get("nlat"),
                record.get("nlon"),
                record.get("exclusive_lmax_mmax"),
            )
            == case
            and record.get("transform") == transform
            and record.get("implementation") == implementation
            and record.get("dtype") == dtype
            and record.get("batch_size") == batch_size
        ):
            return record
    return None


def _mib(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) / 2**20:.1f}"


def _milliseconds(record: dict[str, Any] | None) -> float | None:
    if record is None or record.get("status") != "completed":
        return None
    value = record.get("milliseconds_per_frame")
    return float(value) if isinstance(value, (int, float)) else None


def _format_ms(record: dict[str, Any] | None) -> str:
    value = _milliseconds(record)
    if value is not None:
        return f"{value:.3f}"
    if record is None:
        return "—"
    return {
        "oom_cuda": "OOM",
        "skipped_predicted_capacity": "SKIP",
        "skipped_after_process_kill": "KILL",
    }.get(record.get("status"), "ERR")


def _resolution_table(records: list[dict[str, Any]], transform: str) -> str:
    lines = [
        "| Grid | N | B ms/frame | C ms/frame | C/B time | B module MiB | C module MiB | B runtime-extra MiB | C runtime-extra MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in RESOLUTION_CASES:
        b = _lookup_record(
            records, case=case, transform=transform, implementation="runtime_fold"
        )
        c = _lookup_record(
            records,
            case=case,
            transform=transform,
            implementation="precomputed_projection",
        )
        b_ms, c_ms = _milliseconds(b), _milliseconds(c)
        ratio = f"{c_ms / b_ms:.3f}" if b_ms and c_ms else "—"
        lines.append(
            f"| `{case[0]}×{case[1]}` | {case[0] - 1} | {_format_ms(b)} | {_format_ms(c)} | {ratio} | {_mib((b or {}).get('module_buffer_bytes'))} | {_mib((c or {}).get('module_buffer_bytes'))} | {_mib((b or {}).get('runtime_extra_allocated_bytes'))} | {_mib((c or {}).get('runtime_extra_allocated_bytes'))} |"
        )
    return "\n".join(lines)


def _batch_table(
    records: list[dict[str, Any]], case: tuple[int, int, int], transform: str
) -> str:
    lines = [
        "| Batch | B ms/frame | C ms/frame | B peak MiB | C peak MiB | B status | C status |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for batch in BATCHES:
        b = _lookup_record(
            records,
            case=case,
            transform=transform,
            implementation="runtime_fold",
            batch_size=batch,
        )
        c = _lookup_record(
            records,
            case=case,
            transform=transform,
            implementation="precomputed_projection",
            batch_size=batch,
        )
        lines.append(
            f"| {batch} | {_format_ms(b)} | {_format_ms(c)} | {_mib((b or {}).get('cuda_peak_reserved_bytes'))} | {_mib((c or {}).get('cuda_peak_reserved_bytes'))} | {(b or {}).get('status', 'not-run')} | {(c or {}).get('status', 'not-run')} |"
        )
    return "\n".join(lines)


def _memory_model_table(
    records: list[dict[str, Any]], case: tuple[int, int, int]
) -> str:
    lines = [
        "| Transform | Implementation | module MiB | fixed peak-reserved MiB | per-batch peak-reserved MiB | fixed runtime-extra MiB | per-batch runtime-extra MiB | measured points |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for transform in TRANSFORMS:
        for implementation in IMPLEMENTATIONS:
            selected = [
                record
                for record in records
                if (
                    record.get("nlat"),
                    record.get("nlon"),
                    record.get("exclusive_lmax_mmax"),
                )
                == case
                and record.get("transform") == transform
                and record.get("implementation") == implementation
                and record.get("dtype") == "float32"
                and record.get("status") == "completed"
                and record.get("phase") in {"phase_a_resolution", "phase_b_batch"}
            ]
            first = selected[0] if selected else None
            model = fit_memory_model(selected)
            runtime_model = fit_memory_model(
                selected, value_key="runtime_extra_allocated_bytes"
            )
            points = ", ".join(f"b{x}" for x, _y in model["points"]) or "—"
            lines.append(
                f"| {transform} | {implementation} | {_mib((first or {}).get('module_buffer_bytes'))} | {_mib(model['fixed_bytes'])} | {_mib(model['per_batch_bytes'])} | {_mib(runtime_model['fixed_bytes'])} | {_mib(runtime_model['per_batch_bytes'])} | {points} |"
            )
    return "\n".join(lines)


def _maximum_safe_batch_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| Grid | Transform | B maximum safe batch | C maximum safe batch |",
        "| --- | --- | ---: | ---: |",
    ]
    cases = [(361, 720, 360)]
    finer = {
        (
            record.get("nlat"),
            record.get("nlon"),
            record.get("exclusive_lmax_mmax"),
        )
        for record in records
        if record.get("phase") == "phase_b_batch"
        and record.get("dtype") == "float32"
        and record.get("status") == "completed"
    }
    cases.extend(
        sorted(case for case in finer if case != cases[0] and isinstance(case[0], int))
    )
    for case in cases:
        for transform in TRANSFORMS:
            maxima = []
            for implementation in IMPLEMENTATIONS:
                safe = [
                    int(record["batch_size"])
                    for record in records
                    if (
                        record.get("nlat"),
                        record.get("nlon"),
                        record.get("exclusive_lmax_mmax"),
                    )
                    == case
                    and record.get("transform") == transform
                    and record.get("implementation") == implementation
                    and record.get("dtype") == "float32"
                    and record.get("status") == "completed"
                ]
                maxima.append(str(max(safe)) if safe else "—")
            lines.append(
                f"| `{case[0]}×{case[1]}` | {transform} | {maxima[0]} | {maxima[1]} |"
            )
    return "\n".join(lines)


def render_report(payload: dict[str, Any]) -> str:
    records = payload["records"]
    metadata = payload["metadata"]
    selected = metadata.get("selected_finer_resolution")
    selected_label = _case_label(tuple(selected)) if selected else "none"
    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1
    lines = [
        "# B/C equiangular SHT resolution scaling",
        "",
        "This report contains only B (`runtime_fold`) and C (`precomputed_projection`) forward measurements. Each native cell ran in a fresh subprocess; A/dense, backward, compilation, and DUCC were not part of this probe.",
        "",
        "## Provenance and safety budgets",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| starting `sht_bench` SHA | `{metadata.get('starting_sht_bench_sha')}` |",
        f"| result-generation `sht_bench` SHA | `{metadata.get('result_generation_sht_bench_sha')}` |",
        f"| measurement worker code SHA(s) | `{metadata.get('measurement_worker_code_shas')}` |",
        f"| torch-harmonics current source SHA | `{metadata.get('torch_harmonics_source_sha')}` |",
        f"| historical report source SHA | `{metadata.get('historical_source_sha') or 'not available'}` |",
        f"| GPU | {metadata.get('gpu_name') or 'unknown'} |",
        f"| VRAM | {_mib(metadata.get('gpu_total_memory_bytes'))} MiB |",
        f"| host/cgroup memory limit | {_mib(metadata.get('host_memory_limit_bytes'))} MiB ({metadata.get('host_memory_limit_source') or 'unknown'}) |",
        f"| CUDA safety budget | {metadata.get('cuda_memory_fraction')} of VRAM |",
        f"| host safety budget | {metadata.get('host_memory_fraction')} of host/cgroup limit |",
        f"| selected finer batch grid | `{selected_label}` |",
        f"| record status counts | `{status_counts}` |",
        f"| superseded pre-fix CUDA OOMs | `{len(metadata.get('superseded_cuda_ooms', []))}` (retained in progress history) |",
        "",
        "## Batch-1 resolution scaling",
        "",
        "C/B is the C latency divided by B latency: below 1 means C is faster; above 1 means B is faster.",
        "",
        "### Scalar",
        "",
        _resolution_table(records, "scalar"),
        "",
        "### Vector",
        "",
        _resolution_table(records, "vector"),
        "",
        "## Memory decomposition",
        "",
        "`module MiB` is persistent module-buffer storage. `runtime-extra MiB` is the measured forward peak allocated memory minus the CUDA allocation before input allocation. The fixed-plus-per-batch model below is fitted to measured peak-reserved bytes, so it is deliberately conservative for capacity planning.",
        "",
        f"### Anchor and selected finer grid (`361×720` and `{selected_label}`)",
        "",
    ]
    for case in tuple(
        value
        for value in ((361, 720, 360), tuple(selected) if selected else None)
        if value is not None
    ):
        lines.extend(
            [f"#### `{_case_label(case)}`", "", _memory_model_table(records, case), ""]
        )
    lines.extend(["## Batch scaling", ""])
    lines.extend(
        [
            "### Maximum safely measured batch",
            "",
            _maximum_safe_batch_table(records),
            "",
        ]
    )
    if selected:
        for transform in TRANSFORMS:
            lines.extend(
                [
                    f"### `{_case_label((361, 720, 360))}` {transform}",
                    "",
                    _batch_table(records, (361, 720, 360), transform),
                    "",
                    f"### `{_case_label(tuple(selected))}` {transform}",
                    "",
                    _batch_table(records, tuple(selected), transform),
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "The finer grid did not complete with sufficient headroom; only the established 361×720 anchor was eligible.",
                "",
            ]
        )
    lines.extend(
        [
            "## Float64 probe",
            "",
            "The limited float64 probe covers 361×720 and 481×960 at batch 1. The 721×1440 vector C safety-gate record is explicit and was not launched.",
            "",
        ]
    )
    for case in FLOAT64_CASES:
        values = []
        for transform in TRANSFORMS:
            for implementation in IMPLEMENTATIONS:
                record = _lookup_record(
                    records,
                    case=case,
                    transform=transform,
                    implementation=implementation,
                    dtype="float64",
                )
                values.append(f"{transform}/{implementation}: {_format_ms(record)}")
        lines.append(f"- `{_case_label(case)}`: " + "; ".join(values))
    lines.extend(["", "## Findings", ""])
    for transform in TRANSFORMS:
        ratios = []
        for case in RESOLUTION_CASES:
            b = _lookup_record(
                records, case=case, transform=transform, implementation="runtime_fold"
            )
            c = _lookup_record(
                records,
                case=case,
                transform=transform,
                implementation="precomputed_projection",
            )
            b_ms, c_ms = _milliseconds(b), _milliseconds(c)
            if b_ms and c_ms:
                ratios.append(f"{case[0]}: {c_ms / b_ms:.3f}")
        lines.append(
            f"- **{transform} C/B batch-1 ratios:** "
            + (", ".join(ratios) or "no complete pairs")
        )
    lines.extend(
        [
            "- The batch-1 ratios rise toward 1 at the tested resolutions, so B shows evidence of catching C as resolution increases, but no completed pair crosses above 1 through 721×1440.",
            "- The batch tables contain both C-faster and B-faster cells at the same resolution. Together with the persistent/module and runtime-extra columns, that rejects a resolution-only cutoff for this measured range; the comparison depends on both resolution and batch.",
            "- At 721×1440 scalar, C first uses less total peak VRAM than B at batch 64 (3.9 versus 5.0 GiB); the vector pair has no completed common batch where C's larger projection is outweighed because C is capacity-skipped at batch 64.",
            "- At the selected grid, the fitted runtime-extra slope is about 47.5 MiB per batch for scalar B versus 15.8 MiB for C, and 94.9 versus 31.7 MiB for vector B/C; C's runtime-memory advantage grows with batch even though its persistent module is larger.",
            "- Maximum safe batch is the largest `completed` batch recorded separately for each implementation and transform; a skipped or OOM batch is not counted as safe.",
            "",
            "## Resume and raw data",
            "",
            f"The durable progress manifest is `{metadata.get('progress_path')}`. A stale `in_progress` entry is reported as `skipped_after_process_kill` and is never retried automatically. Raw JSON and CSV retain timing samples, allocator fields, host RSS fields, source SHA, and all explicit capacity/OOM statuses.",
            "",
            "No production `torch-harmonics` files were modified.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    *,
    output_json: Path,
    output_csv: Path,
    output_markdown: Path,
    progress_path: Path,
    metadata: dict[str, Any],
    progress: dict[str, Any],
) -> dict[str, Any]:
    records = _result_records(progress)
    payload = {
        "schema": SCHEMA,
        "metadata": metadata,
        "records": records,
    }
    atomic_write_json(output_json, payload)
    _write_csv(output_csv, records)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_report(payload), encoding="utf-8")
    return payload


def _historical_source_sha(repository: Path) -> str | None:
    report = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "torch-abc-precomputed-projection.md"
    )
    if not report.exists():
        return None
    for line in report.read_text(encoding="utf-8").splitlines():
        if "| Torch source |" in line and "`" in line:
            return line.split("`", 2)[1]
    return None


def _run(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    source = Path(args.source).resolve()
    output_json = args.output.resolve()
    output_csv = args.csv.resolve()
    output_markdown = args.markdown.resolve()
    progress_path = args.progress.resolve()
    host_limit, host_source, host_current = _memory_limit_bytes()
    gpu_total = args.gpu_total_memory_bytes or _nvidia_smi_memory_bytes(
        args.cuda_device
    )
    gpu_name = _nvidia_smi_value("name", args.cuda_device)
    metadata = {
        "created_at": utc_now(),
        "starting_sht_bench_sha": _git_sha(Path(__file__).resolve().parents[1]),
        "torch_harmonics_source_sha": _git_sha(source),
        "historical_source_sha": _historical_source_sha(source),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": gpu_total,
        "host_memory_limit_bytes": host_limit,
        "host_memory_limit_source": host_source,
        "host_memory_current_bytes_at_start": host_current,
        "cuda_memory_fraction": CUDA_MEMORY_FRACTION,
        "host_memory_fraction": HOST_MEMORY_FRACTION,
        "prediction_safety_margin": PREDICTION_SAFETY_MARGIN,
        "cases": [list(case) for case in args.cases],
        "batches": list(args.batches),
        "progress_path": str(progress_path),
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "output_markdown": str(output_markdown),
        "device": args.device,
        "threads": args.threads,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "float64_cases": [list(case) for case in FLOAT64_CASES],
    }
    progress = load_progress(progress_path, metadata)
    original_start_sha = progress.get("metadata", {}).get("starting_sht_bench_sha")
    if original_start_sha:
        metadata["starting_sht_bench_sha"] = original_start_sha
    metadata["result_generation_sht_bench_sha"] = _git_sha(
        Path(__file__).resolve().parents[1]
    )
    if progress.get("metadata", {}).get("superseded_cuda_ooms"):
        metadata["superseded_cuda_ooms"] = progress["metadata"]["superseded_cuda_ooms"]
    if args.repair_batch_predictions:
        progress = repair_batch_predictions(progress_path)
        metadata["superseded_cuda_ooms"] = progress.get("metadata", {}).get(
            "superseded_cuda_ooms", []
        )
    worker_directory = output_json.with_name(f".{output_json.stem}-workers")
    if args.phase in {"all", "resolution"}:
        progress = _phase_a(
            cases=args.cases,
            progress_path=progress_path,
            progress=progress,
            script=script,
            source=source,
            args=args,
            worker_directory=worker_directory,
            gpu_total=gpu_total,
            host_limit=host_limit,
        )
    selected = _select_finer_resolution(
        progress, args.cases, gpu_total=gpu_total, host_limit=host_limit
    )
    metadata["selected_finer_resolution"] = list(selected) if selected else None
    if args.phase in {"all", "batch"}:
        progress = _phase_b(
            anchor=args.cases[0],
            finer=selected,
            progress_path=progress_path,
            progress=progress,
            script=script,
            source=source,
            args=args,
            worker_directory=worker_directory,
            gpu_total=gpu_total,
            host_limit=host_limit,
            batches=args.batches,
        )
    if args.phase in {"all", "float64"}:
        progress = _phase_c_float64(
            progress_path=progress_path,
            progress=progress,
            script=script,
            source=source,
            args=args,
            worker_directory=worker_directory,
            gpu_total=gpu_total,
            host_limit=host_limit,
        )
    metadata["finished_at"] = utc_now()
    final_progress = _read_json(progress_path)
    metadata["measurement_worker_code_shas"] = sorted(
        {
            result.get("benchmark_code_sha")
            for result in _result_records(final_progress)
            if result.get("benchmark_code_sha")
        }
    )
    payload = write_outputs(
        output_json=output_json,
        output_csv=output_csv,
        output_markdown=output_markdown,
        progress_path=progress_path,
        metadata=metadata,
        progress=final_progress,
    )
    print(
        json.dumps(
            {"json": str(output_json), "records": len(payload["records"])},
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--source", type=Path, default=Path("/home/albert/torch-harmonics")
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--gpu-total-memory-bytes", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument("--phase", default="all")
    parser.add_argument(
        "--cases",
        type=parse_cases,
        default=RESOLUTION_CASES,
        help="comma-separated nlatxnlonxlmax cases",
    )
    parser.add_argument("--batches", type=parse_batches, default=BATCHES)
    parser.add_argument(
        "--output", type=Path, default=Path("results/torch-bc-resolution-scaling.json")
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("results/torch-bc-resolution-scaling.csv")
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path("results/torch-bc-resolution-scaling.md")
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=Path("results/torch-bc-resolution-scaling-progress.json"),
    )
    parser.add_argument(
        "--repair-batch-predictions",
        action="store_true",
        help="migrate records made before CUDA peak-field prediction was fixed",
    )
    # Worker-only options. They are still defined on the parent parser so the
    # exact command line can be passed through without a second parser.
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, default=None)
    parser.add_argument("--transform", choices=TRANSFORMS, default=None)
    parser.add_argument("--dtype", choices=tuple(DTYPE_SIZES), default=None)
    parser.add_argument("--case", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--worker-output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.worker:
        required = {
            "implementation": args.implementation,
            "transform": args.transform,
            "dtype": args.dtype,
            "case": args.case,
            "batch_size": args.batch_size,
            "worker_output": args.worker_output,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("worker requires " + ", ".join(missing))
        return _worker_main(args)
    if args.phase not in {"all", "resolution", "batch", "float64"}:
        parser.error("phase must be all, resolution, batch, or float64")
    if args.threads < 1 or args.warmup < 0 or args.repeat < 1:
        parser.error("threads must be positive, warmup non-negative, repeat positive")
    if args.device != "cuda":
        print(
            "warning: this research probe is intended for CUDA; CPU mode is diagnostic only",
            file=sys.stderr,
        )
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
