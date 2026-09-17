"""Compare CPU DUCC and resampled equiangular Torch analysis.

The two fixed Clenshaw--Curtis grids are deliberately paired with the largest
recoverable triangular degree range of the Torch implementation.  The Torch
source is loaded from a named checkout so the result remains reproducible.
"""

from __future__ import annotations

import argparse
import functools
import gc
import importlib.metadata
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench_common.timing import measure


@dataclass(frozen=True)
class Case:
    name: str
    nlat: int
    nlon: int
    degree: int

    @property
    def lmax(self) -> int:
        """Return the exclusive Torch degree limit."""

        return self.degree + 1

    @property
    def mmax(self) -> int:
        """Return the exclusive Torch order limit."""

        return self.degree + 1


CASES = (
    Case("cc-73x144-t71", 73, 144, 71),
    Case("cc-721x1440-t719", 721, 1440, 719),
)
DTYPES = {
    "float32": (np.float32, np.complex64),
    "float64": (np.float64, np.complex128),
}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _source_sha(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def _nalm(degree: int) -> int:
    return (degree + 1) * (degree + 2) // 2


def _packed_index(degree: int, order: int, ell: int) -> int:
    return order * (2 * degree + 1 - order) // 2 + ell


def _pack_ducc(coefficients: np.ndarray, degree: int) -> np.ndarray:
    packed = np.empty(
        coefficients.shape[:-2] + (_nalm(degree),), dtype=coefficients.dtype
    )
    for order in range(degree + 1):
        for ell in range(order, degree + 1):
            packed[..., _packed_index(degree, order, ell)] = coefficients[
                ..., ell, order
            ]
    return packed


def _unpack_ducc(packed: np.ndarray, degree: int) -> np.ndarray:
    coefficients = np.zeros(
        packed.shape[:-1] + (degree + 1, degree + 1), dtype=packed.dtype
    )
    for order in range(degree + 1):
        for ell in range(order, degree + 1):
            coefficients[..., ell, order] = packed[
                ..., _packed_index(degree, order, ell)
            ]
    return coefficients


def _coefficients(
    case: Case,
    real_dtype: np.dtype[Any],
    complex_dtype: np.dtype[Any],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((1, case.lmax, case.mmax), dtype=real_dtype)
    imag = rng.standard_normal((1, case.lmax, case.mmax), dtype=real_dtype)
    result = np.asarray(real + 1j * imag, dtype=complex_dtype)
    for ell in range(case.lmax):
        result[:, ell, ell + 1 :] = 0.0
    result[:, :, 0] = result[:, :, 0].real
    return result


def _metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = np.abs(actual - reference)
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    return {
        "relative_l2": float(np.linalg.norm(error.reshape(-1)) / reference_norm),
        "maximum_absolute": float(error.max()),
    }


def _buffer_bytes(module: Any) -> int:
    return sum(buffer.numel() * buffer.element_size() for buffer in module.buffers())


def _time_cpu(
    function: Callable[[], Any],
    *,
    torch: Any | None,
    warmup: int,
    repeat: int,
    min_time: float,
) -> dict[str, Any]:
    if torch is None:
        iterations, samples = measure(
            function, warmup=warmup, repeat=repeat, min_time=min_time
        )
    else:
        with torch.inference_mode():
            iterations, samples = measure(
                function, warmup=warmup, repeat=repeat, min_time=min_time
            )
    return {
        "iterations": iterations,
        "samples_s": samples,
        "minimum_s": min(samples),
        "median_s": float(np.median(samples)),
        "mean_s": float(np.mean(samples)),
        "timing_method": "perf_counter_ns-wall",
    }


def _load_torch(source: Path) -> tuple[Any, Any, str]:
    source = source.resolve()
    sys.path.insert(0, str(source))
    import torch
    import torch_harmonics

    module_path = Path(torch_harmonics.__file__).resolve()
    try:
        module_path.relative_to(source)
    except ValueError as exc:
        raise RuntimeError(
            f"Torch imported from {module_path}, outside requested source {source}"
        ) from exc
    return torch, torch_harmonics, _source_sha(source)


def _run_case(
    case: Case,
    dtype_name: str,
    *,
    torch: Any,
    torch_harmonics: Any,
    ducc: Any,
    threads: int,
    seed: int,
    warmup: int,
    repeat: int,
    min_time: float,
) -> list[dict[str, Any]]:
    real_dtype, complex_dtype = DTYPES[dtype_name]
    coefficients = _coefficients(case, real_dtype, complex_dtype, seed)
    packed = _pack_ducc(coefficients, case.degree)

    ducc_module = ducc.sht

    def ducc_synthesis() -> np.ndarray:
        return np.asarray(
            ducc_module.synthesis_2d(
                alm=packed,
                spin=0,
                lmax=case.degree,
                mmax=case.degree,
                geometry="CC",
                ntheta=case.nlat,
                nphi=case.nlon,
                nthreads=threads,
            )
        )

    ducc_map = ducc_synthesis()

    def ducc_analysis_raw() -> np.ndarray:
        return np.asarray(
            ducc_module.analysis_2d(
                map=ducc_map,
                spin=0,
                lmax=case.degree,
                mmax=case.degree,
                geometry="CC",
                nthreads=threads,
            )
        )

    torch_real_dtype = torch.float32 if dtype_name == "float32" else torch.float64
    start = time.perf_counter()
    analysis_module = (
        torch_harmonics.RealSHT(
            case.nlat,
            case.nlon,
            lmax=case.lmax,
            mmax=case.mmax,
            grid="equiangular",
            norm="ortho",
            csphase=True,
        )
        .to(device="cpu", dtype=torch_real_dtype)
        .eval()
    )
    synthesis_module = (
        torch_harmonics.InverseRealSHT(
            case.nlat,
            case.nlon,
            lmax=case.lmax,
            mmax=case.mmax,
            grid="equiangular",
            norm="ortho",
            csphase=True,
        )
        .to(device="cpu", dtype=torch_real_dtype)
        .eval()
    )
    setup_time = time.perf_counter() - start

    torch_map = torch.from_numpy(np.ascontiguousarray(ducc_map)).to(
        dtype=torch_real_dtype
    )
    torch_coefficients = torch.from_numpy(np.ascontiguousarray(coefficients)).to(
        dtype=torch.complex64 if dtype_name == "float32" else torch.complex128
    )

    with torch.inference_mode():
        torch_synthesis = synthesis_module(torch_coefficients)
        torch_analysis = analysis_module(torch_map)
    ducc_coefficients = _unpack_ducc(ducc_analysis_raw(), case.degree)
    correctness = {
        "synthesis": _metrics(torch_synthesis.detach().numpy(), ducc_map),
        "analysis": _metrics(torch_analysis.detach().numpy(), ducc_coefficients),
    }

    torch_analysis = functools.partial(analysis_module, torch_map)
    torch_synthesis = functools.partial(synthesis_module, torch_coefficients)

    common = {
        "case": case.name,
        "nlat": case.nlat,
        "nlon": case.nlon,
        "degree": case.degree,
        "torch_lmax_exclusive": case.lmax,
        "torch_mmax_exclusive": case.mmax,
        "dtype": dtype_name,
        "threads": threads,
        "frames": 1,
        "torch_analysis_buffer_bytes": _buffer_bytes(analysis_module),
        "torch_synthesis_buffer_bytes": _buffer_bytes(synthesis_module),
        "torch_setup_s": setup_time,
        "correctness": correctness,
    }
    records: list[dict[str, Any]] = []
    for backend, runtime, functions in (
        (
            "ducc0.sht",
            None,
            {"analysis": ducc_analysis_raw, "synthesis": ducc_synthesis},
        ),
        (
            "torch-harmonics",
            torch,
            {
                "analysis": torch_analysis,
                "synthesis": torch_synthesis,
            },
        ),
    ):
        for operation, function in functions.items():
            timing = _time_cpu(
                function,
                torch=runtime,
                warmup=warmup,
                repeat=repeat,
                min_time=min_time,
            )
            records.append(
                {
                    "record_type": "performance",
                    "backend": backend,
                    "operation": operation,
                    **common,
                    **timing,
                }
            )
    del analysis_module, synthesis_module, torch_map, torch_coefficients
    gc.collect()
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/home/albert/torch-harmonics")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/torch-ducc-resampled")
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--min-time", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--dtype",
        default="float32,float64",
        help="comma-separated real dtypes (default: float32,float64)",
    )
    args = parser.parse_args()
    dtype_names = tuple(item.strip() for item in args.dtype.split(",") if item.strip())
    unknown = set(dtype_names) - set(DTYPES)
    if not dtype_names or unknown:
        parser.error(
            f"--dtype must contain only float32/float64; got {sorted(unknown)}"
        )
    if args.threads < 1 or args.warmup < 0 or args.repeat < 1 or args.min_time <= 0:
        parser.error("threads/warmup/repeat/min-time values are invalid")

    import ducc0

    torch, torch_harmonics, source_sha = _load_torch(args.source)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    records: list[dict[str, Any]] = []
    for dtype_index, dtype_name in enumerate(dtype_names):
        for case in CASES:
            print(
                f"{dtype_name} {case.name} (CPU, {args.threads} thread(s))", flush=True
            )
            records.extend(
                _run_case(
                    case,
                    dtype_name,
                    torch=torch,
                    torch_harmonics=torch_harmonics,
                    ducc=ducc0,
                    threads=args.threads,
                    seed=args.seed + dtype_index * 10000 + case.degree,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    min_time=args.min_time,
                )
            )

    payload = {
        "schema": "sht_bench.torch_ducc_resampled.v1",
        "source": str(args.source.resolve()),
        "source_sha": source_sha,
        "ducc_version": _package_version("ducc0"),
        "torch_version": torch.__version__,
        "torch_harmonics_version": _package_version("torch-harmonics"),
        "device": "cpu",
        "threads": args.threads,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "min_time_s": args.min_time,
        "frames": 1,
        "cases": [case.__dict__ for case in CASES],
        "records": records,
    }
    output = (
        args.output
        if args.output.suffix == ".json"
        else args.output.with_suffix(".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_output = output.with_suffix(".csv")
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    import csv

    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["correctness"] = json.dumps(row["correctness"])
            row["samples_s"] = json.dumps(row["samples_s"])
            writer.writerow(row)
    print(f"wrote {output} and {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
