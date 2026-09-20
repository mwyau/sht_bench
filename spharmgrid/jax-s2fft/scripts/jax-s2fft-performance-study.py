#!/usr/bin/env python3
"""Measure compiled and warmed S2FFT JAX transform performance.

This development benchmark is separate from the spharmgrid test suite. It
records compilation time, warmed forward and inverse times, process peak RSS,
JAX device memory counters when available, array sizes, and errors against the
float64 NumPy S2FFT reference. The parent process runs one worker subprocess
per case so a timeout preserves an incomplete case in the result file.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import s2fft
from s2fft.recursions.price_mcewen import generate_precomputes_jax
from s2fft.sampling import s2_samples


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("jax", "jaxlib", "s2fft", "numpy")
    }


def _require_cuda(parser: argparse.ArgumentParser) -> None:
    """Require a CUDA-backed JAX device for a requested GPU run."""
    devices = jax.devices()
    cuda_devices = [
        device
        for device in devices
        if getattr(device, "platform", None) == "gpu"
        and (type(device).__name__ == "CudaDevice" or "CudaDevice" in repr(device))
    ]
    if not cuda_devices:
        parser.error(
            f"CUDA benchmark requires a JAX CudaDevice; JAX reported {devices!r}"
        )


def _device_memory_stats() -> list[dict[str, object]]:
    """Return JAX device memory counters when the backend exposes them."""
    snapshots: list[dict[str, object]] = []
    for device in jax.devices():
        snapshot: dict[str, object] = {"device": str(device)}
        try:
            values = device.memory_stats()
        except Exception as error:  # pragma: no cover - backend-specific
            snapshot["available"] = False
            snapshot["error"] = f"{type(error).__name__}: {error}"
        else:
            snapshot["available"] = values is not None
            if values is not None:
                snapshot.update({key: int(value) for key, value in values.items()})
        snapshots.append(snapshot)
    return snapshots


def _field(bandlimit: int, sampling: str, spin: int) -> np.ndarray:
    theta = s2_samples.thetas(bandlimit, sampling)
    latitude = np.pi / 2.0 - theta[:, None]
    nlon = s2_samples.nphi_equiang(bandlimit, sampling)
    longitude = np.arange(nlon)[None, :] * 2.0 * np.pi / nlon
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    if spin == 0:
        return (
            0.8
            + 0.35 * sine
            + 0.25 * (3.0 * sine**2 - 1.0) / 2.0
            + 0.20 * cosine * np.cos(longitude)
            + 0.17 * sine * cosine * np.sin(longitude)
            + 0.13 * cosine**2 * np.cos(2.0 * longitude)
            + 0.11 * sine * cosine**2 * np.sin(2.0 * longitude)
            + 0.07 * cosine**3 * np.cos(3.0 * longitude)
        )

    eastward = -1.4 * np.sin(longitude) - 1.2 * cosine * np.sin(2.0 * longitude)
    northward = -1.4 * sine * np.cos(longitude) - 1.2 * sine * cosine * np.cos(
        2.0 * longitude
    )
    return -northward + (1j if spin == 1 else -1j) * eastward


def _relative_max(values: np.ndarray, reference: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(reference))), 1.0e-30)
    return float(np.max(np.abs(values - reference)) / scale)


def _time_warm(
    function: Callable[[jax.Array], jax.Array],
    argument: jax.Array,
    repetitions: int,
) -> float:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        result = function(argument)
        result.block_until_ready()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _precomputes(
    bandlimit: int,
    sampling: str,
    spin: int,
    forward: bool,
) -> tuple[tuple[jax.Array, ...], float]:
    started = time.perf_counter()
    values = tuple(
        generate_precomputes_jax(
            bandlimit,
            spin=spin,
            sampling=sampling,
            forward=forward,
        )
    )
    for value in values:
        value.block_until_ready()
    return values, time.perf_counter() - started


def _precompute_record(values: tuple[jax.Array, ...]) -> dict[str, object]:
    return {
        "shapes": [list(value.shape) for value in values],
        "dtypes": [str(value.dtype) for value in values],
        "nbytes": int(sum(value.size * value.dtype.itemsize for value in values)),
        "devices": [str(value.device) for value in values],
    }


def _measure_compiled(
    function: Callable[[jax.Array], jax.Array],
    argument: jax.Array,
    repetitions: int,
) -> tuple[dict[str, float], jax.Array, Callable[[jax.Array], jax.Array]]:
    jitted = jax.jit(function)
    started = time.perf_counter()
    traced = jitted.trace(argument)
    trace_seconds = time.perf_counter() - started
    started = time.perf_counter()
    lowered = traced.lower()
    lower_seconds = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    output = executable(argument)
    output.block_until_ready()
    first_execution_seconds = time.perf_counter() - started
    return (
        {
            "trace_seconds": trace_seconds,
            "lower_seconds": lower_seconds,
            "compile_seconds": compile_seconds,
            "first_execution_seconds": first_execution_seconds,
            "warm_seconds_median": _time_warm(executable, argument, repetitions),
        },
        output,
        executable,
    )


def _benchmark_case(
    bandlimit: int,
    sampling: str,
    spin: int,
    requested_dtype: str,
    repetitions: int,
) -> dict[str, object]:
    reality = spin == 0
    reference_field = _field(bandlimit, sampling, spin)
    reference_field = np.asarray(
        reference_field,
        dtype=np.complex128 if spin else np.float64,
    )
    reference_coefficients = s2fft.forward_numpy(
        reference_field,
        bandlimit,
        spin=spin,
        sampling=sampling,
        reality=reality,
    )
    reference_inverse = s2fft.inverse_numpy(
        reference_coefficients,
        bandlimit,
        spin=spin,
        sampling=sampling,
        reality=reality,
    )
    input_dtype = (
        jnp.complex64
        if spin and requested_dtype == "float32"
        else jnp.complex128
        if spin
        else jnp.float32
        if requested_dtype == "float32"
        else jnp.float64
    )
    coefficient_dtype = (
        jnp.complex64 if requested_dtype == "float32" else jnp.complex128
    )
    field = jnp.asarray(reference_field, dtype=input_dtype)
    coefficients = jnp.asarray(reference_coefficients, dtype=coefficient_dtype)
    effective_input_dtype = str(field.dtype)
    effective_coefficient_dtype = str(coefficients.dtype)
    forward_precomps, precompute_forward_seconds = _precomputes(
        bandlimit, sampling, spin, True
    )
    inverse_precomps, precompute_inverse_seconds = _precomputes(
        bandlimit, sampling, spin, False
    )

    def forward(values: jax.Array) -> jax.Array:
        return s2fft.forward_jax(
            values,
            bandlimit,
            spin=spin,
            sampling=sampling,
            reality=reality,
            precomps=forward_precomps,
        )

    def inverse(values: jax.Array) -> jax.Array:
        return s2fft.inverse_jax(
            values,
            bandlimit,
            spin=spin,
            sampling=sampling,
            reality=reality,
            precomps=inverse_precomps,
        )

    gc.collect()
    rss_before = _peak_rss_bytes()
    device_memory_before = _device_memory_stats()
    forward_timing, forward_output, compiled_forward = _measure_compiled(
        forward, field, repetitions
    )
    rss_after_forward_compile = _peak_rss_bytes()
    device_memory_after_forward_compile = _device_memory_stats()

    inverse_timing, inverse_from_reference, compiled_inverse = _measure_compiled(
        inverse, coefficients, repetitions
    )
    rss_after_inverse_compile = _peak_rss_bytes()
    device_memory_after_inverse_compile = _device_memory_stats()

    forward_output = np.asarray(forward_output)
    inverse_from_reference = np.asarray(inverse_from_reference)
    round_trip = np.asarray(
        compiled_inverse(jnp.asarray(forward_output, dtype=coefficient_dtype))
    )
    return {
        "bandlimit": bandlimit,
        "sampling": sampling,
        "spin": spin,
        "requested_dtype": requested_dtype,
        "effective_input_dtype": effective_input_dtype,
        "effective_coefficient_dtype": effective_coefficient_dtype,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "reality": reality,
        "precompute_forward_seconds": precompute_forward_seconds,
        "precompute_inverse_seconds": precompute_inverse_seconds,
        "precompute_total_seconds": (
            precompute_forward_seconds + precompute_inverse_seconds
        ),
        "forward_precomps": _precompute_record(forward_precomps),
        "inverse_precomps": _precompute_record(inverse_precomps),
        "trace_forward_seconds": forward_timing["trace_seconds"],
        "lower_forward_seconds": forward_timing["lower_seconds"],
        "compile_forward_seconds": forward_timing["compile_seconds"],
        "first_forward_seconds": forward_timing["first_execution_seconds"],
        "warm_forward_seconds_median": forward_timing["warm_seconds_median"],
        "trace_inverse_seconds": inverse_timing["trace_seconds"],
        "lower_inverse_seconds": inverse_timing["lower_seconds"],
        "compile_inverse_seconds": inverse_timing["compile_seconds"],
        "first_inverse_seconds": inverse_timing["first_execution_seconds"],
        "warm_inverse_seconds_median": inverse_timing["warm_seconds_median"],
        "warm_repetitions": repetitions,
        "peak_rss_bytes_before_case": rss_before,
        "peak_rss_bytes_after_forward_compile": rss_after_forward_compile,
        "peak_rss_bytes_after_inverse_compile": rss_after_inverse_compile,
        "peak_rss_bytes_after_warm": _peak_rss_bytes(),
        "device_memory_before_case": device_memory_before,
        "device_memory_after_forward_compile": device_memory_after_forward_compile,
        "device_memory_after_inverse_compile": device_memory_after_inverse_compile,
        "device_memory_after_warm": _device_memory_stats(),
        "field_nbytes": int(field.size * field.dtype.itemsize),
        "coefficient_nbytes": int(coefficients.size * coefficients.dtype.itemsize),
        "forward_coefficient_relative_max": _relative_max(
            forward_output, reference_coefficients
        ),
        "inverse_spatial_relative_max": _relative_max(
            inverse_from_reference, reference_inverse
        ),
        "round_trip_relative_max": _relative_max(round_trip, reference_inverse),
    }


def _write_results(
    path: Path,
    results: list[dict[str, object]],
    *,
    case_timeout_seconds: float | None,
) -> None:
    device = jax.devices()[0]
    output = {
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _versions(),
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "backend": jax.default_backend(),
            "devices": [
                {
                    "platform": item.platform,
                    "device_kind": item.device_kind,
                    "id": item.id,
                    "type": type(item).__name__,
                    "repr": repr(item),
                }
                for item in jax.devices()
            ],
            "timing": (
                "precompute setup is outside transform timing; trace uses "
                "jitted_function.trace, lower uses traced.lower, compile uses "
                "lowered.compile, and warm is the median of blocking calls"
            ),
            "memory": (
                "Linux process peak RSS from resource.getrusage; cumulative "
                "within each worker process; JAX device memory_stats when available"
            ),
            "case_execution": "one subprocess per case",
            "case_timeout_seconds": case_timeout_seconds,
            "transform_path": (
                "ordinary S2FFT JAX transform with external O(L^2) "
                "Price–McEwen precomputations"
            ),
            "precompute_api": "s2fft.recursions.price_mcewen.generate_precomputes_jax",
            "precompute_memory": "O(L^2)",
            "reference": "s2fft.forward_numpy/inverse_numpy float64",
            "error": (
                "maximum absolute error divided by maximum absolute reference value"
            ),
            "primary_device": str(device),
        },
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")


def _tail(value: str | bytes | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[-limit:]


def _case_key(result: dict[str, object]) -> tuple[object, ...]:
    return (
        result.get("bandlimit"),
        result.get("sampling"),
        result.get("spin"),
        result.get("requested_dtype"),
    )


def _run_case_worker(
    args: argparse.Namespace,
    bandlimit: int,
    sampling: str,
    spin: int,
    requested_dtype: str,
) -> dict[str, object]:
    case = {
        "bandlimit": bandlimit,
        "sampling": sampling,
        "spin": spin,
        "requested_dtype": requested_dtype,
    }
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-output",
        "WORKER_OUTPUT",
        "--worker-bandlimit",
        str(bandlimit),
        "--worker-sampling",
        sampling,
        "--worker-spin",
        str(spin),
        "--worker-dtype",
        requested_dtype,
        "--repetitions",
        str(args.repetitions),
    ]
    if args.require_cuda:
        command.append("--require-cuda")

    with tempfile.TemporaryDirectory(prefix="spharmgrid-s2fft-case-") as directory:
        result_path = Path(directory) / "result.json"
        command[command.index("WORKER_OUTPUT")] = str(result_path)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=args.case_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return {
                **case,
                "status": "timeout",
                "timeout_seconds": args.case_timeout,
                "worker_wall_seconds": time.perf_counter() - started,
                "stdout_tail": _tail(error.stdout),
                "stderr_tail": _tail(error.stderr),
            }

        elapsed = time.perf_counter() - started
        if completed.returncode != 0 or not result_path.exists():
            return {
                **case,
                "status": "error",
                "returncode": completed.returncode,
                "worker_wall_seconds": elapsed,
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
            }

        result = json.loads(result_path.read_text())
        result["status"] = "complete"
        result["worker_wall_seconds"] = elapsed
        stderr_tail = _tail(completed.stderr)
        if stderr_tail:
            result["stderr_tail"] = stderr_tail
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bandlimits", type=int, nargs="+", default=[8, 32, 64, 128])
    parser.add_argument("--spins", type=int, nargs="+", default=[0, 1, -1])
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("float32", "float64"),
        default=["float32", "float64"],
    )
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=300.0,
        help="maximum wall time in seconds for each worker case",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an existing result file, retaining completed cases",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail unless JAX reports at least one CUDA-backed device",
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-bandlimit", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-sampling",
        choices=("gl", "mwss"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-spin", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-dtype",
        choices=("float32", "float64"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    if args.case_timeout <= 0.0:
        parser.error("--case-timeout must be positive")
    if not jax.devices():
        parser.error("JAX did not report an available device")
    if args.require_cuda:
        _require_cuda(parser)

    if args.worker_output is not None:
        if None in (
            args.worker_bandlimit,
            args.worker_sampling,
            args.worker_spin,
            args.worker_dtype,
        ):
            parser.error("worker case arguments are incomplete")
        result = _benchmark_case(
            args.worker_bandlimit,
            args.worker_sampling,
            args.worker_spin,
            args.worker_dtype,
            args.repetitions,
        )
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(json.dumps(result, indent=2) + "\n")
        return

    if args.output is None:
        parser.error("--output is required")

    results: list[dict[str, object]] = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text())
        results = list(existing.get("results", []))
    completed_cases = {
        _case_key(result)
        for result in results
        if result.get("status", "complete") == "complete"
    }
    _write_results(
        args.output,
        results,
        case_timeout_seconds=args.case_timeout,
    )
    for requested_dtype in args.dtypes:
        if requested_dtype == "float64" and not jax.config.read("jax_enable_x64"):
            continue
        for bandlimit in args.bandlimits:
            for sampling in ("gl", "mwss"):
                for spin in args.spins:
                    case = {
                        "bandlimit": bandlimit,
                        "sampling": sampling,
                        "spin": spin,
                        "requested_dtype": requested_dtype,
                    }
                    if args.resume and _case_key(case) in completed_cases:
                        continue
                    result = _run_case_worker(
                        args,
                        bandlimit,
                        sampling,
                        spin,
                        requested_dtype,
                    )
                    results.append(result)
                    print(json.dumps(result, sort_keys=True), flush=True)

                    _write_results(
                        args.output,
                        results,
                        case_timeout_seconds=args.case_timeout,
                    )


if __name__ == "__main__":
    main()
