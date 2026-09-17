"""Measure memory-safe GPU forward transforms for two equiangular grids.

Each cell runs in a separate process and constructs one compact Torch module.
The benchmark intentionally measures only inference forwards: no reference
module, inverse module, gradient graph, or construction-time expanded weights
are resident on the GPU.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class Case:
    name: str
    nlat: int
    nlon: int
    lmax: int


CASES = {
    "73x144": Case("73x144", 73, 144, 72),
    "721x1440": Case("721x1440", 721, 1440, 720),
}
TRANSFORMS = ("scalar", "vector")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _source_sha(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def _load_source(source: Path) -> tuple[Any, str]:
    source = source.resolve()
    sys.path.insert(0, str(source))
    import torch_harmonics

    module_path = Path(torch_harmonics.__file__).resolve()
    try:
        module_path.relative_to(source)
    except ValueError as exc:
        raise RuntimeError(
            f"Torch imported from {module_path}, outside requested source {source}"
        ) from exc
    return torch_harmonics, _source_sha(source)


def _synchronize() -> None:
    torch.cuda.synchronize()


def _buffer_bytes(module: Any) -> int:
    return sum(buffer.numel() * buffer.element_size() for buffer in module.buffers())


def _construct(cls: Any, case: Case, vector: bool) -> tuple[Any, float, int, int]:
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    module = (
        cls(
            case.nlat,
            case.nlon,
            lmax=case.lmax,
            mmax=case.lmax,
            grid="equiangular",
            norm="ortho",
            csphase=True,
        )
        .to(device="cuda", dtype=torch.float32)
        .eval()
    )
    _synchronize()
    setup_s = time.perf_counter() - start
    peak_constructor = torch.cuda.max_memory_allocated() - baseline
    return module, setup_s, _buffer_bytes(module), peak_constructor


def _elapsed(function: Any, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / 1000.0


def _time_forward(
    function: Any, warmup: int, repeat: int, min_time: float
) -> dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        _synchronize()
        iterations = 1
        while True:
            elapsed = _elapsed(function, iterations)
            if elapsed >= min_time or iterations >= 1 << 20:
                break
            scale = max(2, min(16, math.ceil(min_time / max(elapsed, 1.0e-12))))
            iterations *= scale
        samples = [_elapsed(function, iterations) / iterations for _ in range(repeat)]
    return {
        "iterations": iterations,
        "samples_s": samples,
        "minimum_s": min(samples),
        "median_s": sorted(samples)[len(samples) // 2],
        "timing_method": "cuda-event-compute",
    }


def _peak_forward(function: Any) -> int:
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = function()
        _synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del output
    torch.cuda.empty_cache()
    return peak


def _run_cell(
    torch_harmonics: Any,
    source_sha: str,
    case: Case,
    transform: str,
    warmup: int,
    repeat: int,
    min_time: float,
) -> dict[str, Any]:
    cls = (
        torch_harmonics.RealVectorSHT
        if transform == "vector"
        else torch_harmonics.RealSHT
    )
    module, setup_s, buffer_bytes, peak_constructor = _construct(
        cls, case, transform == "vector"
    )
    shape = (
        (1, 2, case.nlat, case.nlon)
        if transform == "vector"
        else (1, case.nlat, case.nlon)
    )
    sample = torch.randn(shape, device="cuda", dtype=torch.float32)
    function = functools.partial(module, sample)

    with torch.inference_mode():
        output = function()
        _synchronize()
        finite = bool(torch.isfinite(output.real).all().item())
        output_shape = list(output.shape)
    del output
    timing = _time_forward(function, warmup, repeat, min_time)
    peak_forward = _peak_forward(function)
    result = {
        "record_type": "performance",
        "backend": "torch-harmonics",
        "source_sha": source_sha,
        "case": case.name,
        "nlat": case.nlat,
        "nlon": case.nlon,
        "lmax_exclusive": case.lmax,
        "mmax_exclusive": case.lmax,
        "transform": transform,
        "dtype": "float32",
        "frames": 1,
        "setup_s": setup_s,
        "registered_buffer_bytes": buffer_bytes,
        "cuda_peak_constructor_delta_bytes": peak_constructor,
        "cuda_peak_forward_delta_bytes": peak_forward,
        "output_shape": output_shape,
        "finite_output": finite,
        **timing,
    }
    del module, sample, function
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _run_one_process(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.set_num_threads(1)
    torch_harmonics, source_sha = _load_source(args.source)
    case = CASES[args.case]
    record = _run_cell(
        torch_harmonics,
        source_sha,
        case,
        args.transform,
        args.warmup,
        args.repeat,
        args.min_time,
    )
    return {
        "schema": "sht_bench.torch_gpu_resampled.v1",
        "source": str(args.source.resolve()),
        "source_sha": source_sha,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "case": args.case,
        "transform": args.transform,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "min_time_s": args.min_time,
        "isolated_cell": True,
        "records": [record],
    }


def _run_isolated(
    args: argparse.Namespace, cases: list[str], transforms: list[str]
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    with __import__("tempfile").TemporaryDirectory(
        prefix="sht-bench-gpu-"
    ) as temp_name:
        temporary = Path(temp_name)
        for case in cases:
            for transform in transforms:
                output = temporary / f"{case}-{transform}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--source",
                    str(args.source),
                    "--case",
                    case,
                    "--transform",
                    transform,
                    "--warmup",
                    str(args.warmup),
                    "--repeat",
                    str(args.repeat),
                    "--min-time",
                    str(args.min_time),
                    "--output",
                    str(output),
                    "--cell",
                ]
                print("+", " ".join(command), flush=True)
                subprocess.run(command, check=True)
                payload = json.loads(output.read_text())
                metadata = metadata or {
                    key: payload[key] for key in payload if key != "records"
                }
                records.extend(payload["records"])
    assert metadata is not None
    metadata.update(
        {
            "cases": cases,
            "transforms": transforms,
            "isolated_cells": True,
            "records": records,
        }
    )
    return metadata


def _write(payload: dict[str, Any], output: Path) -> None:
    output = output if output.suffix == ".json" else output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    records = payload["records"]
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["output_shape"] = json.dumps(row["output_shape"])
            row["samples_s"] = json.dumps(row["samples_s"])
            writer.writerow(row)
    print(f"wrote {output} and {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/home/albert/torch-harmonics")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/torch-gpu-resampled")
    )
    parser.add_argument("--case", choices=tuple(CASES), default=None)
    parser.add_argument("--transform", choices=TRANSFORMS, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--min-time", type=float, default=0.2)
    parser.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1 or args.min_time <= 0:
        parser.error("warmup/repeat/min-time values are invalid")
    if args.cell and (args.case is None or args.transform is None):
        parser.error("--cell requires --case and --transform")
    if args.cell:
        payload = _run_one_process(args)
    else:
        cases = [args.case] if args.case else list(CASES)
        transforms = [args.transform] if args.transform else list(TRANSFORMS)
        payload = _run_isolated(args, cases, transforms)
    _write(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
