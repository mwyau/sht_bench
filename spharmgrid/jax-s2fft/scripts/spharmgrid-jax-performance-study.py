#!/usr/bin/env python3
"""Measure representative spharmgrid JAX operation performance.

This development benchmark separates private S2FFT recursion setup from JAX
trace, lowering, compilation, warmed execution, and host/device transfer. It
also measures the corresponding direct S2FFT transform calls for the same
batched inputs. The script is not part of the spharmgrid test suite.
"""

from __future__ import annotations

import argparse
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

import spharmgrid as sg
import spharmgrid.jax as sgj
from spharmgrid.grids import grid_layout
from spharmgrid.jax import _backend
from spharmgrid.jax._backend import _native_bandlimit, _sampling


def _versions() -> dict[str, str]:
    packages = ("jax", "jaxlib", "s2fft", "numpy", "spharmgrid")
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
        parser.error(
            f"CUDA benchmark requires a JAX CudaDevice; JAX reported {devices!r}"
        )


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _device_memory_stats() -> list[dict[str, object]]:
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


def _grid(kind: str, bandlimit: int) -> sg.Grid:
    if kind == "gl":
        return sg.gaussian_grid(
            bandlimit,
            2 * bandlimit - 1,
            latitude_order="ascending",
        )
    if kind == "cc":
        return sg.clenshaw_curtis_grid(
            bandlimit + 1,
            2 * bandlimit,
            latitude_order="ascending",
        )
    raise ValueError(f"unknown grid kind: {kind}")


def _scalar_field(grid: sg.Grid) -> np.ndarray:
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    values = (
        0.8
        + 0.35 * sine
        + 0.25 * (3.0 * sine**2 - 1.0) / 2.0
        + 0.20 * cosine * np.cos(longitude)
        + 0.17 * sine * cosine * np.sin(longitude)
        + 0.13 * cosine**2 * np.cos(2.0 * longitude)
        + 0.11 * sine * cosine**2 * np.sin(2.0 * longitude)
        + 0.07 * cosine**3 * np.cos(3.0 * longitude)
        + 0.05 * (5.0 * sine**3 - 3.0 * sine) / 2.0
        + 0.03 * cosine**4 * np.sin(4.0 * longitude)
    )
    return np.broadcast_to(values, (grid.nlat, grid.nlon)).copy()


def _vector_field(grid: sg.Grid) -> tuple[np.ndarray, np.ndarray]:
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    eastward = -1.4 * np.sin(longitude) - 1.2 * cosine * np.sin(2.0 * longitude)
    northward = -1.4 * sine * np.cos(longitude) - 1.2 * sine * cosine * np.cos(
        2.0 * longitude
    )
    shape = (grid.nlat, grid.nlon)
    return (
        np.broadcast_to(eastward, shape).copy(),
        np.broadcast_to(northward, shape).copy(),
    )


def _block(value: object) -> None:
    jax.tree_util.tree_map(lambda item: item.block_until_ready(), value)


def _nbytes(value: object) -> int:
    return int(
        sum(
            item.size * item.dtype.itemsize for item in jax.tree_util.tree_leaves(value)
        )
    )


def _shapes(value: object) -> list[list[int]]:
    return [list(item.shape) for item in jax.tree_util.tree_leaves(value)]


def _measure(
    function: Callable[..., object],
    arguments: tuple[jax.Array, ...],
    repetitions: int,
) -> tuple[dict[str, float], object]:
    jitted = jax.jit(function)
    started = time.perf_counter()
    traced = jitted.trace(*arguments)
    trace_seconds = time.perf_counter() - started
    started = time.perf_counter()
    lowered = traced.lower()
    lower_seconds = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    output = executable(*arguments)
    _block(output)
    first_execution_seconds = time.perf_counter() - started
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        result = executable(*arguments)
        _block(result)
        samples.append(time.perf_counter() - started)
    return (
        {
            "trace_seconds": trace_seconds,
            "lower_seconds": lower_seconds,
            "compile_seconds": compile_seconds,
            "first_execution_seconds": first_execution_seconds,
            "warm_seconds_median": statistics.median(samples),
        },
        output,
    )


def _requirements(
    operation: str,
    source: sg.Grid,
    target: sg.Grid,
) -> list[tuple[int, str, int, bool]]:
    source_key = (_native_bandlimit(source), _sampling(source))
    target_key = (_native_bandlimit(target), _sampling(target))
    if operation == "filter":
        return [(*source_key, 0, True), (*source_key, 0, False)]
    if operation == "regrid":
        return [(*source_key, 0, True), (*target_key, 0, False)]
    if operation == "kinematics":
        return [
            (*source_key, 1, True),
            (*source_key, -1, True),
            (*source_key, 0, False),
        ]
    raise ValueError(f"unknown operation: {operation}")


def _prepare_precomputes(
    operation: str,
    source: sg.Grid,
    target: sg.Grid,
) -> tuple[float, int]:
    _backend._precomputes.cache_clear()
    started = time.perf_counter()
    values = []
    for bandlimit, sampling, spin, forward in _requirements(operation, source, target):
        values.extend(_backend._precomputes(bandlimit, sampling, spin, forward=forward))
    elapsed = time.perf_counter() - started
    nbytes = int(sum(value.size * value.dtype.itemsize for value in values))
    return elapsed, nbytes


def _canonical_numpy(field: np.ndarray, grid: sg.Grid) -> np.ndarray:
    layout = grid_layout(grid)
    result = np.take(field, layout.latitude.canonical_indices, axis=-2)
    return np.take(result, layout.longitude.canonical_indices, axis=-1)


def _case_grids(
    operation: str,
    bandlimit: int,
    kind: str,
) -> tuple[sg.Grid, sg.Grid]:
    if operation == "regrid" and kind == "gl_to_cc":
        source = _grid("gl", bandlimit)
        target = _grid("cc", bandlimit)
    elif operation == "regrid" and kind == "cc_to_gl":
        source = _grid("cc", bandlimit)
        target = _grid("gl", bandlimit)
    else:
        source = _grid(kind, bandlimit)
        target = source

    return source, target


def _host_arguments(operation: str, source: sg.Grid) -> tuple[np.ndarray, ...]:
    if operation in ("filter", "regrid"):
        field = _scalar_field(source)
        return (np.stack((field, 1.5 * field), axis=0),)
    eastward, northward = _vector_field(source)
    return (
        np.stack((eastward, 1.5 * eastward), axis=0),
        np.stack((northward, 1.5 * northward), axis=0),
    )


def _device_arguments(host_arguments: tuple[np.ndarray, ...]) -> tuple[jax.Array, ...]:
    return tuple(
        jax.device_put(jnp.asarray(value, dtype=jnp.float64))
        for value in host_arguments
    )


def _operation_function(
    operation: str,
    source: sg.Grid,
    target: sg.Grid,
) -> Callable[..., object]:
    if operation == "filter":
        return lambda field: sgj.filter(field, "T4", grid=source)
    if operation == "regrid":
        return lambda field: sgj.regrid(
            field,
            target,
            "T4",
            source_grid=source,
        )
    return lambda u, v: sgj.kinematics(u, v, grid=source)


def _direct_scalar_baseline(
    source: sg.Grid,
    target: sg.Grid,
    host_arguments: tuple[np.ndarray, ...],
    repetitions: int,
) -> dict[str, object]:
    source_bandlimit = _native_bandlimit(source)
    target_bandlimit = _native_bandlimit(target)
    source_sampling = _sampling(source)
    target_sampling = _sampling(target)
    source_precomps = _backend._precomputes(
        source_bandlimit, source_sampling, 0, forward=True
    )
    target_precomps = _backend._precomputes(
        target_bandlimit, target_sampling, 0, forward=False
    )
    canonical = np.stack(
        [_canonical_numpy(frame, source) for frame in host_arguments[0]], axis=0
    )
    field = jax.device_put(jnp.asarray(canonical, dtype=jnp.float64))

    def forward(values: jax.Array) -> object:
        return jax.vmap(
            lambda frame: s2fft.forward_jax(
                frame,
                source_bandlimit,
                sampling=source_sampling,
                reality=True,
                precomps=source_precomps,
            )
        )(values)

    def inverse(values: jax.Array) -> object:
        return jax.vmap(
            lambda frame: s2fft.inverse_jax(
                frame,
                target_bandlimit,
                sampling=target_sampling,
                reality=True,
                precomps=target_precomps,
            )
        )(values)

    forward_timing, coefficients = _measure(forward, (field,), repetitions)
    inverse_timing, _ = _measure(inverse, (coefficients,), repetitions)
    return {
        "forward": forward_timing,
        "inverse": inverse_timing,
        "warm_seconds_total": (
            forward_timing["warm_seconds_median"]
            + inverse_timing["warm_seconds_median"]
        ),
    }


def _direct_vector_baseline(
    source: sg.Grid,
    host_arguments: tuple[np.ndarray, ...],
    repetitions: int,
) -> dict[str, object]:
    bandlimit = _native_bandlimit(source)
    sampling = _sampling(source)
    plus_precomps = _backend._precomputes(bandlimit, sampling, 1, forward=True)
    minus_precomps = _backend._precomputes(bandlimit, sampling, -1, forward=True)
    inverse_precomps = _backend._precomputes(bandlimit, sampling, 0, forward=False)
    eastward = np.stack(
        [_canonical_numpy(frame, source) for frame in host_arguments[0]], axis=0
    )
    northward = np.stack(
        [_canonical_numpy(frame, source) for frame in host_arguments[1]], axis=0
    )
    unit = jnp.asarray(1j, dtype=jnp.complex128)
    q_plus = jax.device_put(-jnp.asarray(northward) + unit * jnp.asarray(eastward))
    q_minus = jax.device_put(-jnp.asarray(northward) - unit * jnp.asarray(eastward))

    def forward_plus(values: jax.Array) -> object:
        return jax.vmap(
            lambda frame: s2fft.forward_jax(
                frame,
                bandlimit,
                spin=1,
                sampling=sampling,
                reality=False,
                precomps=plus_precomps,
            )
        )(values)

    def forward_minus(values: jax.Array) -> object:
        return jax.vmap(
            lambda frame: s2fft.forward_jax(
                frame,
                bandlimit,
                spin=-1,
                sampling=sampling,
                reality=False,
                precomps=minus_precomps,
            )
        )(values)

    coefficients = jnp.zeros(
        (q_plus.shape[0], bandlimit, 2 * bandlimit - 1), dtype=jnp.complex128
    )

    def inverse(values: jax.Array) -> object:
        return jax.vmap(
            lambda frame: s2fft.inverse_jax(
                frame,
                bandlimit,
                sampling=sampling,
                reality=True,
                precomps=inverse_precomps,
            )
        )(values)

    plus_timing, _ = _measure(forward_plus, (q_plus,), repetitions)
    minus_timing, _ = _measure(forward_minus, (q_minus,), repetitions)
    inverse_timing, _ = _measure(inverse, (coefficients,), repetitions)
    return {
        "spin_plus_forward": plus_timing,
        "spin_minus_forward": minus_timing,
        "scalar_inverse": inverse_timing,
        "warm_seconds_total": sum(
            timing["warm_seconds_median"]
            for timing in (plus_timing, minus_timing, inverse_timing)
        ),
    }


def _run_case(
    operation: str,
    bandlimit: int,
    kind: str,
    repetitions: int,
) -> dict[str, object]:
    source, target = _case_grids(operation, bandlimit, kind)
    host_arguments = _host_arguments(operation, source)
    precompute_seconds, precompute_nbytes = _prepare_precomputes(
        operation, source, target
    )
    function = _operation_function(operation, source, target)
    host_to_device_started = time.perf_counter()
    arguments = _device_arguments(host_arguments)
    _block(arguments)
    host_to_device_seconds = time.perf_counter() - host_to_device_started
    timing, output = _measure(function, arguments, repetitions)
    device_to_host_started = time.perf_counter()
    host_output = jax.tree_util.tree_map(np.asarray, output)
    device_to_host_seconds = time.perf_counter() - device_to_host_started
    warm_seconds = timing["warm_seconds_median"]
    direct_baseline = (
        _direct_scalar_baseline(source, target, host_arguments, repetitions)
        if operation in ("filter", "regrid")
        else _direct_vector_baseline(source, host_arguments, repetitions)
    )
    batch_size = 2
    return {
        "operation": operation,
        "bandlimit": bandlimit,
        "grid": kind,
        "source_grid": source.kind,
        "target_grid": target.kind,
        "source_shape": [source.nlat, source.nlon],
        "target_shape": [target.nlat, target.nlon],
        "batch_size": batch_size,
        "input_nbytes": int(sum(value.nbytes for value in host_arguments)),
        "output_nbytes": _nbytes(host_output),
        "output_shapes": _shapes(host_output),
        "precompute_setup_seconds": precompute_seconds,
        "precompute_nbytes": precompute_nbytes,
        "host_to_device_seconds": host_to_device_seconds,
        "device_to_host_seconds": device_to_host_seconds,
        "trace_seconds": timing["trace_seconds"],
        "lower_seconds": timing["lower_seconds"],
        "compile_seconds": timing["compile_seconds"],
        "first_execution_seconds": timing["first_execution_seconds"],
        "warm_seconds_median": warm_seconds,
        "warm_seconds_per_frame": warm_seconds / batch_size,
        "direct_transform_baseline": direct_baseline,
        "operation_to_direct_warm_ratio": warm_seconds
        / float(direct_baseline["warm_seconds_total"]),
        "peak_rss_bytes": _peak_rss_bytes(),
        "device_memory_after_case": _device_memory_stats(),
    }


def _tail(value: str | bytes | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[-limit:]


def _case_key(result: dict[str, object]) -> tuple[object, ...]:
    return (result.get("operation"), result.get("bandlimit"), result.get("grid"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bandlimits", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=("filter", "regrid", "kinematics"),
        default=["filter", "regrid", "kinematics"],
    )
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--case-timeout", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-operation",
        choices=("filter", "regrid", "kinematics"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-bandlimit", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-grid", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    if args.case_timeout <= 0.0:
        parser.error("--case-timeout must be positive")
    if not bool(jax.config.read("jax_enable_x64")):
        parser.error("the spharmgrid operation study requires JAX_ENABLE_X64=1")
    if args.require_cuda:
        _require_cuda(parser)
    if args.worker_output is not None:
        if None in (args.worker_operation, args.worker_bandlimit, args.worker_grid):
            parser.error("worker case arguments are incomplete")
        result = _run_case(
            args.worker_operation,
            args.worker_bandlimit,
            args.worker_grid,
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
        "transform_path": "spharmgrid private adapter with S2FFT O(L^2) precomputes",
        "direct_comparison": (
            "operation-level timings are compared with direct S2FFT in the study "
            "method; the stored operation matrix records spharmgrid timing"
        ),
        "memory": (
            "Linux process peak RSS; JAX device memory_stats when available; "
            "array byte counts are explicit"
        ),
        "case_execution": "one subprocess per case",
        "case_timeout_seconds": args.case_timeout,
    }
    output = {"metadata": metadata, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for operation in args.operations:
        grids = ("gl", "cc") if operation != "regrid" else ("gl_to_cc", "cc_to_gl")
        for bandlimit in args.bandlimits:
            for kind in grids:
                case = {
                    "operation": operation,
                    "bandlimit": bandlimit,
                    "grid": kind,
                }
                if args.resume and _case_key(case) in completed_cases:
                    continue
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-output",
                    "WORKER_OUTPUT",
                    "--worker-operation",
                    operation,
                    "--worker-bandlimit",
                    str(bandlimit),
                    "--worker-grid",
                    kind,
                    "--repetitions",
                    str(args.repetitions),
                ]
                if args.require_cuda:
                    command.append("--require-cuda")
                with tempfile.TemporaryDirectory(
                    prefix="spharmgrid-jax-op-"
                ) as directory:
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
                        result = {
                            **case,
                            "status": "timeout",
                            "timeout_seconds": args.case_timeout,
                            "worker_wall_seconds": time.perf_counter() - started,
                            "stdout_tail": _tail(error.stdout),
                            "stderr_tail": _tail(error.stderr),
                        }
                    else:
                        elapsed = time.perf_counter() - started
                        if completed.returncode != 0 or not result_path.exists():
                            result = {
                                **case,
                                "status": "error",
                                "returncode": completed.returncode,
                                "worker_wall_seconds": elapsed,
                                "stdout_tail": _tail(completed.stdout),
                                "stderr_tail": _tail(completed.stderr),
                            }
                        else:
                            result = json.loads(result_path.read_text())
                            result["status"] = "complete"
                            result["worker_wall_seconds"] = elapsed
                            stderr_tail = _tail(completed.stderr)
                            if stderr_tail:
                                result["stderr_tail"] = stderr_tail
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
                output["results"] = results
                args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
