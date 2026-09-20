#!/usr/bin/env python3
"""Measure S2FFT JAX precomputation and compilation stages.

This development study is separate from the spharmgrid test suite. It compares
S2FFT's ordinary JAX transform with and without externally generated
Price–McEwen recursion precomputations and measures setup, tracing, lowering,
compilation, and warmed execution separately. A parent process runs one worker
per case so a bounded attempt remains inspectable when compilation is slow.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import s2fft
from s2fft.sampling import s2_samples


def _versions() -> dict[str, str]:
    packages = ("jax", "jaxlib", "s2fft", "numpy")
    return {package: importlib.metadata.version(package) for package in packages}


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
        parser.error(f"CUDA study requires a JAX CudaDevice; JAX reported {devices!r}")


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


def _block(values: Sequence[jax.Array]) -> None:
    for value in values:
        value.block_until_ready()


def _array_record(values: Sequence[jax.Array]) -> dict[str, object]:
    return {
        "shapes": [list(value.shape) for value in values],
        "dtypes": [str(value.dtype) for value in values],
        "nbytes": int(sum(value.size * value.dtype.itemsize for value in values)),
        "devices": [str(value.device) for value in values],
    }


def _precomputes(
    bandlimit: int,
    sampling: str,
    spin: int,
    forward: bool,
) -> tuple[tuple[jax.Array, ...], float]:
    started = time.perf_counter()
    values = tuple(
        s2fft.generate_precomputes_jax(
            bandlimit,
            spin=spin,
            sampling=sampling,
            forward=forward,
        )
    )
    _block(values)
    return values, time.perf_counter() - started


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


def _measure_transform(
    function: Callable[[jax.Array], jax.Array],
    argument: jax.Array,
    repetitions: int,
) -> tuple[dict[str, object], jax.Array]:
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
    )


def _run_case(
    bandlimit: int,
    sampling: str,
    spin: int,
    path: str,
    reality: bool,
    repetitions: int,
) -> dict[str, object]:
    if reality and spin != 0:
        raise ValueError("reality=True is only valid for scalar cases")
    field = np.asarray(
        _field(bandlimit, sampling, spin),
        dtype=np.float64 if spin == 0 else np.complex128,
    )
    reference_coefficients = s2fft.forward_numpy(
        field,
        bandlimit,
        spin=spin,
        sampling=sampling,
        reality=reality,
    )
    reference_field = s2fft.inverse_numpy(
        reference_coefficients,
        bandlimit,
        spin=spin,
        sampling=sampling,
        reality=reality,
    )
    jax_field = jnp.asarray(field)
    jax_coefficients = jnp.asarray(reference_coefficients)

    forward_precomps: tuple[jax.Array, ...] = ()
    inverse_precomps: tuple[jax.Array, ...] = ()
    precompute_forward_seconds = 0.0
    precompute_inverse_seconds = 0.0
    if path == "precomps":
        forward_precomps, precompute_forward_seconds = _precomputes(
            bandlimit, sampling, spin, True
        )
        inverse_precomps, precompute_inverse_seconds = _precomputes(
            bandlimit, sampling, spin, False
        )
    elif path != "none":
        raise ValueError(f"unknown path: {path}")

    def forward(values: jax.Array) -> jax.Array:
        return s2fft.forward_jax(
            values,
            bandlimit,
            spin=spin,
            sampling=sampling,
            reality=reality,
            precomps=forward_precomps or None,
        )

    def inverse(values: jax.Array) -> jax.Array:
        return s2fft.inverse_jax(
            values,
            bandlimit,
            spin=spin,
            sampling=sampling,
            reality=reality,
            precomps=inverse_precomps or None,
        )

    forward_timing, forward_output = _measure_transform(forward, jax_field, repetitions)
    inverse_timing, inverse_output = _measure_transform(
        inverse, jax_coefficients, repetitions
    )
    round_trip = inverse(jnp.asarray(forward_output)).block_until_ready()
    forward_values = np.asarray(forward_output)
    inverse_values = np.asarray(inverse_output)
    round_trip_values = np.asarray(round_trip)
    result: dict[str, object] = {
        "bandlimit": bandlimit,
        "sampling": sampling,
        "spin": spin,
        "path": path,
        "reality": reality,
        "field_shape": list(field.shape),
        "coefficient_shape": list(reference_coefficients.shape),
        "field_nbytes": int(field.nbytes),
        "coefficient_nbytes": int(reference_coefficients.nbytes),
        "field_dtype": str(jax_field.dtype),
        "coefficient_dtype": str(jax_coefficients.dtype),
        "forward_output_dtype": str(forward_output.dtype),
        "inverse_output_dtype": str(inverse_output.dtype),
        "precompute_forward_seconds": precompute_forward_seconds,
        "precompute_inverse_seconds": precompute_inverse_seconds,
        "precompute_total_seconds": (
            precompute_forward_seconds + precompute_inverse_seconds
        ),
        "forward_timing": forward_timing,
        "inverse_timing": inverse_timing,
        "forward_coefficient_relative_max": _relative_max(
            forward_values, reference_coefficients
        ),
        "inverse_spatial_relative_max": _relative_max(inverse_values, reference_field),
        "round_trip_relative_max": _relative_max(round_trip_values, reference_field),
    }
    if path == "precomps":
        result["forward_precomps"] = _array_record(forward_precomps)
        result["inverse_precomps"] = _array_record(inverse_precomps)
    if sampling == "mwss":
        row_error = np.abs(inverse_values - reference_field)
        scale = max(float(np.max(np.abs(reference_field))), 1.0e-30)
        result.update(
            {
                "north_pole_relative_max": float(np.max(row_error[0]) / scale),
                "south_pole_relative_max": float(np.max(row_error[-1]) / scale),
                "interior_relative_max": float(np.max(row_error[1:-1]) / scale),
            }
        )
    return result


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
        result.get("path"),
        result.get("reality"),
    )


def _run_worker(
    args: argparse.Namespace,
    bandlimit: int,
    sampling: str,
    spin: int,
    path: str,
    reality: bool,
) -> dict[str, object]:
    case = {
        "bandlimit": bandlimit,
        "sampling": sampling,
        "spin": spin,
        "path": path,
        "reality": reality,
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
        "--worker-path",
        path,
        "--repetitions",
        str(args.repetitions),
    ]
    if reality:
        command.append("--worker-reality")
    if args.require_cuda:
        command.append("--require-cuda")

    with tempfile.TemporaryDirectory(prefix="spharmgrid-s2fft-precomp-") as directory:
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
    parser.add_argument("--bandlimits", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--spins", type=int, nargs="+", default=[0, 1, -1])
    parser.add_argument(
        "--paths",
        choices=("none", "precomps"),
        nargs="+",
        default=["none", "precomps"],
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--case-timeout", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-bandlimit", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-sampling", choices=("gl", "mwss"), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-spin", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-path", choices=("none", "precomps"), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-reality", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    if args.case_timeout <= 0.0:
        parser.error("--case-timeout must be positive")
    if not bool(jax.config.read("jax_enable_x64")):
        parser.error("the precomputation study requires JAX_ENABLE_X64=1")
    if args.require_cuda:
        _require_cuda(parser)
    if args.worker_output is not None:
        if None in (
            args.worker_bandlimit,
            args.worker_sampling,
            args.worker_spin,
            args.worker_path,
        ):
            parser.error("worker case arguments are incomplete")
        result = _run_case(
            args.worker_bandlimit,
            args.worker_sampling,
            args.worker_spin,
            args.worker_path,
            args.worker_reality,
            args.repetitions,
        )
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(json.dumps(result, indent=2) + "\n")
        return
    if args.output is None:
        parser.error("--output is required")

    results: list[dict[str, object]] = []
    if args.resume and args.output.exists():
        results = list(json.loads(args.output.read_text()).get("results", []))
    completed_cases = {
        _case_key(result)
        for result in results
        if result.get("status", "complete") == "complete"
    }
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _versions(),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "backend": jax.default_backend(),
        "devices": [
            {
                "platform": device.platform,
                "device_kind": device.device_kind,
                "id": device.id,
                "type": type(device).__name__,
                "repr": repr(device),
            }
            for device in jax.devices()
        ],
        "primary_device": str(jax.devices()[0]),
        "precompute_api": "s2fft.generate_precomputes_jax",
        "precompute_memory": "O(L^2)",
        "reference": "s2fft.forward_numpy/inverse_numpy float64",
        "timing": (
            "trace uses jitted_function.trace; lower uses traced.lower; compile "
            "uses lowered.compile; warm is median of blocking calls"
        ),
        "case_execution": "one subprocess per case",
        "case_timeout_seconds": args.case_timeout,
        "paths": {
            "none": "ordinary S2FFT JAX transform with precomps=None",
            "precomps": "ordinary S2FFT JAX transform with external O(L^2) precomps",
        },
    }
    output = {"metadata": metadata, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for path in args.paths:
        for bandlimit in args.bandlimits:
            for sampling in ("gl", "mwss"):
                for spin in args.spins:
                    realities = (False, True) if spin == 0 else (False,)
                    for reality in realities:
                        case = {
                            "bandlimit": bandlimit,
                            "sampling": sampling,
                            "spin": spin,
                            "path": path,
                            "reality": reality,
                        }
                        if args.resume and _case_key(case) in completed_cases:
                            continue
                        result = _run_worker(
                            args,
                            bandlimit,
                            sampling,
                            spin,
                            path,
                            reality,
                        )
                        results.append(result)
                        print(json.dumps(result, sort_keys=True), flush=True)
                        output["results"] = results
                        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
