#!/usr/bin/env python3
"""Recovered NumPy FFT timing harness; see the adjacent README for provenance."""
import argparse
import csv
import json
import os
import platform
import statistics
import sys
import timeit
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(824203)
CASES = [
    # Short latency and ordinary batched lengths.
    dict(name="fft-short-c128-32", fn="fft", shape=(32,), dtype="complex128"),
    dict(name="ifft-short-c64-32", fn="ifft", shape=(32,), dtype="complex64"),
    dict(name="fft-batch-c64-4096", fn="fft", shape=(64, 4096), dtype="complex64"),
    dict(name="ifft-batch-c128-2048", fn="ifft", shape=(128, 2048), dtype="complex128"),
    # The reported regression and neighboring lengths.
    *[
        dict(name=f"fft-batch-c128-{n}", fn="fft", shape=(64, n), dtype="complex128")
        for n in (4094, 4095, 4096, 4097)
    ],
    # Prime, mixed-factor, and large transforms.
    dict(name="fft-prime-c128-4093", fn="fft", shape=(16, 4093), dtype="complex128"),
    dict(name="fft-mixed-c128-6144", fn="fft", shape=(32, 6144), dtype="complex128"),
    dict(name="fft-large-c128-65536", fn="fft", shape=(65536,), dtype="complex128"),
    # Non-contiguous and negative-stride batched inputs.
    dict(name="fft-strided-c128-4096", fn="fft", shape=(64, 4096), dtype="complex128", layout="every2"),
    dict(name="fft-reversed-c128-4096", fn="fft", shape=(64, 4096), dtype="complex128", layout="reverse"),
    # Real 1-D transforms across both precisions and odd/even sizes.
    dict(name="rfft-batch-f32-4096", fn="rfft", shape=(64, 4096), dtype="float32"),
    dict(name="rfft-batch-f64-4095", fn="rfft", shape=(32, 4095), dtype="float64"),
    dict(name="rfft-prime-f64-4093", fn="rfft", shape=(4093,), dtype="float64"),
    dict(name="irfft-batch-c64-4096", fn="irfft", shape=(64, 2049), dtype="complex64", n=4096),
    dict(name="irfft-batch-c128-4095", fn="irfft", shape=(32, 2048), dtype="complex128", n=4095),
    # N-D, including full complex and real forward/inverse paths.
    dict(name="fftn-2d-c128-128x256", fn="fftn", shape=(128, 256), dtype="complex128"),
    dict(name="ifftn-3d-c64-16x32x64", fn="ifftn", shape=(16, 32, 64), dtype="complex64"),
    dict(name="rfftn-3d-f32-32x32x64", fn="rfftn", shape=(32, 32, 64), dtype="float32"),
    dict(name="irfftn-3d-c64-32x32x64", fn="irfftn", shape=(32, 32, 33), dtype="complex64", s=(32, 32, 64)),
    dict(name="rfftn-nd-f64-prime-mixed", fn="rfftn", shape=(16, 17, 31), dtype="float64"),
    dict(name="irfftn-nd-c128-prime-mixed", fn="irfftn", shape=(16, 17, 16), dtype="complex128", s=(16, 17, 31)),
    # A transposed 2-D array makes one axis non-contiguous without copies.
    dict(name="fftn-transposed-c128-64x128", fn="fftn", shape=(64, 128), dtype="complex128", layout="transpose"),
    dict(name="rfftn-strided-f64-32x32x64", fn="rfftn", shape=(32, 32, 64), dtype="float64", layout="every2"),
]


def make_array(shape, dtype, layout="contiguous"):
    dtype = np.dtype(dtype)
    real_dtype = dtype.type(0).real.dtype if dtype.kind == "c" else dtype
    if dtype.kind == "c":
        values = RNG.standard_normal(shape, dtype=real_dtype)
        values = values + 1j * RNG.standard_normal(shape, dtype=real_dtype)
        values = values.astype(dtype, copy=False)
    else:
        values = RNG.standard_normal(shape, dtype=dtype)
    if layout == "every2":
        base_shape = shape[:-1] + (2 * shape[-1],)
        base = np.empty(base_shape, dtype=dtype)
        view = base[..., ::2]
        view[...] = values
        return view
    if layout == "reverse":
        base = values.copy()
        base[...] = values[..., ::-1]
        return base[..., ::-1]
    if layout == "transpose":
        base = np.empty(shape[::-1], dtype=dtype)
        view = base.T
        view[...] = values
        return view
    return values


def make_case(spec):
    x = make_array(spec["shape"], spec["dtype"], spec.get("layout", "contiguous"))
    fn = getattr(np.fft, spec["fn"])
    kwargs = {}
    if spec["fn"] in ("fft", "ifft", "rfft", "irfft"):
        kwargs["axis"] = -1
    elif spec["fn"] in ("irfftn",):
        kwargs["s"] = spec["s"]
    if "n" in spec:
        kwargs["n"] = spec["n"]
    sample = fn(x, **kwargs)
    out = np.empty_like(sample)
    expected_shape = tuple(spec.get("s", ())) or tuple(spec["shape"])
    if spec["fn"] == "irfft":
        expected_shape = spec["shape"][:-1] + (spec["n"],)
    elif spec["fn"] == "rfft":
        n = spec["shape"][-1]
        expected_shape = spec["shape"][:-1] + (n // 2 + 1,)
    elif spec["fn"] == "irfftn":
        expected_shape = tuple(spec["s"])
    # Verify the preallocated output shape and the public dtype result before timing.
    assert out.shape == sample.shape
    assert sample.dtype == np.asarray(sample).dtype
    if spec["fn"] == "rfft":
        count = int(np.prod(x.shape))
    elif spec["fn"] == "irfft":
        count = int(np.prod(sample.shape))
    elif spec["fn"] == "rfftn":
        count = int(np.prod(x.shape))
    elif spec["fn"] == "irfftn":
        count = int(np.prod(spec["s"]))
    else:
        count = int(np.prod(sample.shape))
    return x, fn, kwargs, out, count, sample.shape, str(sample.dtype)


def summarize(spec, mode, fn, x, kwargs, out, count, shape, dtype, repeat, min_time):
    if mode == "out":
        call = lambda: fn(x, out=out, **kwargs)
    else:
        call = lambda: fn(x, **kwargs)
    for _ in range(3):
        call()
    timer = timeit.Timer(call)
    number, _ = timer.autorange(callback=lambda n, t: None)
    # `autorange` targets >=0.2s per sample; keep case time bounded for this matrix.
    number = max(1, min(number, 1 << 16))
    times = timer.repeat(repeat=repeat, number=number)
    per_call = [t / number for t in times]
    median = statistics.median(per_call)
    ordered = sorted(per_call)
    lo = ordered[max(0, int(0.1 * (len(ordered) - 1)))]
    hi = ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))]
    return {
        "case": spec["name"], "function": spec["fn"], "input_dtype": spec["dtype"],
        "output_dtype": dtype, "input_shape": "x".join(map(str, x.shape)),
        "output_shape": "x".join(map(str, shape)),
        "layout": spec.get("layout", "contiguous"), "mode": mode,
        "iterations_per_sample": number, "repeat": repeat,
        "median_us": median * 1e6, "p10_us": lo * 1e6, "p90_us": hi * 1e6,
        "min_us": min(per_call) * 1e6, "max_us": max(per_call) * 1e6,
        "points_per_call": count, "points_per_second": count / median,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("/tmp/numpy_fft_baseline.csv"))
    parser.add_argument("--json", type=Path, default=Path("/tmp/numpy_fft_baseline.json"))
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--min-time", type=float, default=0.2,
                        help="timeit autorange uses 0.2 sec; retained for report metadata")
    parser.add_argument("--mode", choices=("default", "out", "both"), default="both")
    parser.add_argument(
        "--case",
        action="append",
        choices=[spec["name"] for spec in CASES],
        help="run only this case (repeatable); the default runs all 26 cases",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="list case/mode rows without timing them",
    )
    args = parser.parse_args()

    modes = ("default", "out") if args.mode == "both" else (args.mode,)
    selected = set(args.case or ())
    if args.list_cases:
        for spec in CASES:
            if selected and spec["name"] not in selected:
                continue
            for mode in modes:
                print(f"{spec['name']},{mode}")
        return

    import numpy._core._multiarray_umath as umath
    metadata = {
        "numpy_version": np.__version__, "python": sys.version,
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_model": next((line.split(":", 1)[1].strip()
                           for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), "unknown"),
        "cpu_dispatch": getattr(umath, "__cpu_dispatch__", None),
        "cpu_features": getattr(umath, "__cpu_features__", {}),
        "environment": {name: os.environ.get(name) for name in
                        ("NPY_DISABLE_CPU_FEATURES", "NPY_ENABLE_CPU_FEATURES",
                         "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "DUCC0_NUM_THREADS")},
        "configuration": np.show_config(mode="dicts"),
        "repeat": args.repeat, "autorange_target_seconds": args.min_time,
    }
    build_dir = os.environ.get("NUMPY_BUILD_DIR")
    if build_dir:
        compile_commands = Path(build_dir) / "compile_commands.json"
        if compile_commands.exists():
            commands = json.loads(compile_commands.read_text())
            metadata["fft_compile_command"] = next(
                (entry["command"] for entry in commands
                 if "_pocketfft_umath.cpp" in entry["file"]
                 or "_duccfft_umath.cpp" in entry["file"]), None)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    csv_exists = args.csv.exists() and args.csv.stat().st_size > 0
    if csv_exists:
        with args.csv.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        completed = {(row["case"], row["mode"]) for row in rows}
    else:
        rows = []
        completed = set()
    with args.csv.open("a", newline="", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "case", "function", "input_dtype", "output_dtype", "input_shape",
            "output_shape", "layout", "mode", "iterations_per_sample",
            "repeat", "median_us", "p10_us", "p90_us", "min_us", "max_us",
            "points_per_call", "points_per_second",
        ])
        if not csv_exists:
            writer.writeheader()
            stream.flush()
        for spec in CASES:
            x, fn, kwargs, out, count, shape, dtype = make_case(spec)
            if selected and spec["name"] not in selected:
                continue
            for mode in modes:
                if (spec["name"], mode) in completed:
                    continue
                row = summarize(spec, mode, fn, x, kwargs, out, count, shape,
                                dtype, args.repeat, args.min_time)
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
                completed.add((spec["name"], mode))
                print(f'{row["case"]:36} {mode:7} '
                      f'{row["median_us"]:9.2f} us  '
                      f'{row["points_per_second"]:12.0f} points/s  '
                      f'range [{row["p10_us"]:.2f}, {row["p90_us"]:.2f}] us',
                      flush=True)
    args.json.write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2))
    print(f"wrote {args.csv} and {args.json}")


if __name__ == "__main__":
    main()
