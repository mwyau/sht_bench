"""Compare CPU DUCC analysis with Torch A/B/C on CC grids.

This is a research-only scalar analysis benchmark.  DUCC is included as an
independent CPU reference, while the three Torch implementations are the
dense reference (A), runtime folding (B), and precomputed projection (C).
The input map is shared by all four implementations within each case.
"""

from __future__ import annotations

import argparse
import functools
import gc
import importlib.metadata
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_abc_benchmark import (
    BASE_SHA,
    DTYPES,
    _buffer_bytes,
    _build_module,
    _clear_torch_harmonics_precompute_caches,
    _move_module,
    _parse_cases,
    _timed,
)
from torch_precomputed_projection_prototype import (
    _load_dense_namespace,
    _load_optimized,
)

IMPLEMENTATIONS = (
    "ducc0",
    "dense",
    "runtime_fold",
    "precomputed_projection",
)
DEFAULT_CASES = "73x144x72,91x180x90,121x240x120,181x360x180,361x720x360"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _unpack_ducc(packed: np.ndarray, degree: int) -> np.ndarray:
    """Convert DUCC's leading-component m-major output to (batch, ell, m)."""

    result = np.zeros(packed.shape[:-1] + (degree + 1, degree + 1), dtype=packed.dtype)
    offset = 0
    for order in range(degree + 1):
        for ell in range(order, degree + 1):
            result[..., ell, order] = packed[..., offset]
            offset += 1
    return result


def _pack_ducc(coefficients: np.ndarray, degree: int) -> np.ndarray:
    """Convert (batch, ell, m) coefficients to DUCC's m-major layout."""

    packed = np.empty(
        coefficients.shape[:-2] + ((degree + 1) * (degree + 2) // 2,),
        dtype=coefficients.dtype,
    )
    offset = 0
    for order in range(degree + 1):
        for ell in range(order, degree + 1):
            packed[..., offset] = coefficients[..., ell, order]
            offset += 1
    return packed


def _band_limited_map(
    ducc: Any,
    *,
    nlat: int,
    nlon: int,
    degree: int,
    dtype: np.dtype[Any],
    threads: int,
    seed: int,
) -> np.ndarray:
    """Create one backend-neutral real map from a DUCC band-limited spectrum."""

    rng = np.random.default_rng(seed)
    coefficients = rng.standard_normal(
        (1, degree + 1, degree + 1), dtype=np.float64
    ).astype(dtype, copy=False)
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
    coefficients = coefficients.astype(complex_dtype, copy=False)
    coefficients[:, :, 0] = coefficients[:, :, 0].real
    upper = np.triu_indices(degree + 1, k=1)
    coefficients[:, upper[0], upper[1]] = 0.0
    return np.asarray(
        ducc.sht.synthesis_2d(
            alm=_pack_ducc(coefficients, degree),
            spin=0,
            lmax=degree,
            mmax=degree,
            geometry="CC",
            ntheta=nlat,
            nphi=nlon,
            nthreads=threads,
        )
    )


def _error(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.abs(actual - reference)
    denominator = max(float(np.linalg.norm(reference.reshape(-1))), 1.0e-30)
    return {
        "relative_l2": float(np.linalg.norm(difference.reshape(-1)) / denominator),
        "maximum_absolute": float(difference.max()),
    }


def _ducc_analysis(
    ducc: Any,
    sample: torch.Tensor,
    *,
    degree: int,
    threads: int,
) -> np.ndarray:
    maps = np.ascontiguousarray(sample.numpy())
    outputs = []
    for index in range(maps.shape[0]):
        result = ducc.sht.analysis_2d(
            map=maps[index : index + 1],
            spin=0,
            lmax=degree,
            mmax=degree,
            geometry="CC",
            nthreads=threads,
        )
        outputs.append(_unpack_ducc(np.asarray(result), degree))
    return np.concatenate(outputs, axis=0)


def _torch_analysis(module: torch.nn.Module, sample: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        return np.asarray(module(sample).numpy())


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import ducc0

    source = args.source.resolve()
    package = _load_optimized(source)
    dense = _load_dense_namespace(source, args.base)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        # The caller may already have initialized the inter-op pool.
        pass

    cases = _parse_cases(args.cases)
    records: list[dict[str, Any]] = []
    for case_index, (nlat, nlon, degree) in enumerate(cases):
        for dtype_name in args.dtypes.split(","):
            dtype_name = dtype_name.strip()
            if dtype_name not in DTYPES:
                raise ValueError(f"unsupported dtype {dtype_name!r}")
            dtype = DTYPES[dtype_name]
            if degree > nlat - 1:
                raise ValueError(
                    f"Torch exclusive limit {degree} exceeds grid limit {nlat - 1}"
                )
            ducc_degree = degree - 1
            if ducc_degree > nlat - 2:
                raise ValueError(
                    "DUCC CC analysis requires inclusive degree <= nlat - 2; "
                    f"got degree={ducc_degree}, nlat={nlat}"
                )
            map_array = _band_limited_map(
                ducc0,
                nlat=nlat,
                nlon=nlon,
                degree=ducc_degree,
                dtype=np.float32 if dtype is torch.float32 else np.float64,
                threads=args.threads,
                seed=8128 + case_index,
            )
            if args.batch > 1:
                map_array = np.repeat(map_array, args.batch, axis=0)
            sample = torch.from_numpy(np.ascontiguousarray(map_array)).to(dtype=dtype)

            ducc_output = _ducc_analysis(
                ducc0,
                sample,
                degree=ducc_degree,
                threads=args.threads,
            )
            torch_outputs: dict[str, np.ndarray] = {}
            torch_modules: dict[str, torch.nn.Module] = {}
            construction: dict[str, float] = {}
            for implementation in IMPLEMENTATIONS[1:]:
                started = time.perf_counter()
                module, diagnostics = _build_module(
                    implementation,
                    package,
                    dense,
                    vector=False,
                    nlat=nlat,
                    nlon=nlon,
                    lmax=degree,
                )
                module = _move_module(
                    module, implementation, dtype_name, torch.device("cpu")
                )
                construction[implementation] = time.perf_counter() - started
                torch_modules[implementation] = module
                torch_outputs[implementation] = _torch_analysis(module, sample)
                del diagnostics

            dense_output = torch_outputs["dense"]
            for implementation in IMPLEMENTATIONS:
                if implementation == "ducc0":
                    output = ducc_output
                    function = functools.partial(
                        _ducc_analysis,
                        ducc0,
                        sample,
                        degree=ducc_degree,
                        threads=args.threads,
                    )
                    setup_s = 0.0
                    buffer_bytes = 0
                    construction_s = 0.0
                else:
                    module = torch_modules[implementation]
                    output = torch_outputs[implementation]
                    function = functools.partial(module, sample)
                    setup_s = construction[implementation]
                    buffer_bytes = _buffer_bytes(module)
                    construction_s = construction[implementation]

                if implementation == "ducc0":
                    samples = _timed(
                        function, torch.device("cpu"), args.warmup, args.repeat
                    )
                else:
                    with torch.inference_mode():
                        samples = _timed(
                            function,
                            torch.device("cpu"),
                            args.warmup,
                            args.repeat,
                        )
                records.append(
                    {
                        "record_type": "performance",
                        "backend": implementation,
                        "operation": "analysis",
                        "transform": "scalar",
                        "dtype": dtype_name,
                        "device": "cpu",
                        "threads": args.threads,
                        "batch_size": args.batch,
                        "nlat": nlat,
                        "nlon": nlon,
                        "exclusive_lmax_mmax": degree,
                        "ducc_inclusive_lmax_mmax": ducc_degree,
                        "samples_s": samples,
                        "median_s": statistics.median(samples),
                        "milliseconds_per_frame": statistics.median(samples)
                        * 1000.0
                        / args.batch,
                        "module_buffer_bytes": buffer_bytes,
                        "construction_s": construction_s,
                        "setup_s": setup_s,
                        "output_vs_ducc": _error(output, ducc_output),
                        "output_vs_dense": _error(output, dense_output),
                    }
                )

            del torch_modules, torch_outputs, sample, ducc_output
            gc.collect()
            _clear_torch_harmonics_precompute_caches()

    return {
        "schema": "sht_bench.torch_ducc_abc_cpu.v1",
        "source": str(source),
        "source_sha": _git_sha(source),
        "dense_base_sha": args.base,
        "ducc0_version": _package_version("ducc0"),
        "device": "cpu",
        "threads": args.threads,
        "cases": cases,
        "implementations": IMPLEMENTATIONS,
        "method": "shared random real map; scalar CC analysis; inference-only timing",
        "records": records,
    }


def _git_sha(repository: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/home/albert/torch-harmonics")
    )
    parser.add_argument("--base", default=BASE_SHA)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--dtypes", default="float32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.batch < 1 or args.threads < 1:
        parser.error("batch and threads must be positive")
    result = _run(args)
    output = args.output or Path("results/torch-ducc-abc-cpu.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
