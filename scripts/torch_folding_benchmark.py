"""Compare dense and folded equiangular Torch SHT implementations.

The normal benchmark is a parent/orchestrator plus short-lived workers.  A
worker constructs exactly one implementation, one transform, one dtype, one
grid, one batch size, and one measurement mode.  The process boundary is the
CUDA allocator isolation boundary: the dense and folded modules are never live
on the GPU at the same time for a performance case.

The dense classes are loaded directly from the immutable ``--base`` commit.
The folded classes are imported from the read-only checkout named by
``--source``.  The script also retains a CPU correctness phase for the small
and medium cases, where keeping both reference modules in one process is
useful and harmless.

Example full run from the ``sht_bench`` checkout::

    PYTHONPATH=/home/albert/sht_bench/src \
      /home/albert/sht_bench/.venv/bin/python \
      scripts/torch_folding_benchmark.py \
      --device cuda \
      --output results/torch-folding-batched-cuda-v2.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any

BASE_SHA = "3278fb669483d04537aa60c87fb0754d989e2e70"
SCHEMA = "sht_bench.torch_folding.v2"
DEFAULT_CASES = ((73, 144, 72), (129, 256, 128), (257, 512, 256), (721, 1440, 720))
DEFAULT_CORRECTNESS_CASES = (
    (17, 32, 16),
    (73, 144, 72),
    (129, 256, 128),
    (257, 512, 256),
)
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_BACKWARD_BATCHES = (1, 16, 128)
NORMS = ("ortho", "four-pi", "schmidt", "unnorm")
DEFAULT_MEMORY_BUDGET_FRACTION = 0.85
DEFAULT_PREDICTION_SAFETY_MARGIN = 1.20


def _parse_case(value: str) -> tuple[int, int, int]:
    """Parse ``nlatxnlonxlmax`` and reject ambiguous case descriptions."""

    parts = tuple(part.strip() for part in value.lower().split("x"))
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"case must be nlatxnlonxlmax, got {value!r}")
    case = tuple(int(part) for part in parts)
    if not all(item > 0 for item in case):
        raise ValueError(f"case dimensions must be positive, got {value!r}")
    _nlat, nlon, lmax = case
    if nlon < 2 or lmax > nlon:
        raise ValueError(f"case has incompatible dimensions: {value!r}")
    return case


def _parse_cases(value: str) -> tuple[tuple[int, int, int], ...]:
    """Parse a comma-separated list of grid cases."""

    cases = tuple(_parse_case(item) for item in value.split(",") if item.strip())
    if not cases:
        raise ValueError("at least one grid case is required")
    if len(set(cases)) != len(cases):
        raise ValueError("duplicate grid cases are not allowed")
    return cases


def _parse_batch_list(value: str) -> tuple[int, ...]:
    """Parse and validate a comma-separated batch-size list."""

    batches: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item or not item.isdigit() or int(item) < 1:
            raise ValueError(f"batch sizes must be positive integers, got {value!r}")
        batches.append(int(item))
    if not batches:
        raise ValueError("at least one batch size is required")
    if len(set(batches)) != len(batches):
        raise ValueError("duplicate batch sizes are not allowed")
    if batches != sorted(batches):
        raise ValueError("batch sizes must be in ascending order")
    return tuple(batches)


def _parse_names(value: str, allowed: set[str], option: str) -> tuple[str, ...]:
    """Parse a comma-separated set of named options."""

    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names or set(names) - allowed or len(set(names)) != len(names):
        raise ValueError(f"{option} must contain unique values from {sorted(allowed)}")
    return names


def _case_label(case: tuple[int, int, int]) -> str:
    return "x".join(str(item) for item in case)


def _dtype_size(dtype_name: str) -> int:
    if dtype_name == "float32":
        return 4
    if dtype_name == "float64":
        return 8
    raise ValueError(f"unsupported dtype: {dtype_name}")


def _is_resampled_case(nlat: int, lmax: int) -> bool:
    return lmax > (nlat + 1) // 2


def estimate_projection_bytes(
    implementation: str,
    transform: str,
    dtype_name: str,
    nlat: int,
    lmax: int,
    mmax: int | None = None,
) -> int:
    """Estimate the large permanent Legendre projection buffer.

    The dense reference resamples to ``2*nlat-1`` latitude rings.  The current
    folded implementation keeps only the original ``nlat`` projection rings
    and applies the midpoint contribution at runtime.  The vector projection
    has two derivative components.  This function intentionally uses the
    source implementation's real storage dtype rather than counting a
    conceptual complex tensor.
    """

    if implementation not in {"dense", "folded"}:
        raise ValueError(f"unknown implementation: {implementation}")
    if transform not in {"scalar", "vector"}:
        raise ValueError(f"unknown transform: {transform}")
    if nlat < 1 or lmax < 1:
        raise ValueError("nlat and lmax must be positive")
    mmax = lmax if mmax is None else mmax
    if mmax < 1:
        raise ValueError("mmax must be positive")
    projection_nlat = nlat
    if implementation == "dense" and _is_resampled_case(nlat, lmax):
        projection_nlat = 2 * nlat - 1
    component_count = 2 if transform == "vector" else 1
    return component_count * mmax * lmax * projection_nlat * _dtype_size(dtype_name)


def estimate_module_storage_bytes(
    implementation: str,
    transform: str,
    dtype_name: str,
    nlat: int,
    lmax: int,
    mmax: int | None = None,
) -> int:
    """Estimate permanent module buffers for capacity planning.

    Auxiliary folded buffers are small relative to the projection but are
    included so the planner does not treat the projection estimate as an
    optimistic exact total.  The dense reference retains its parity signs on
    resampled cases.
    """

    mmax = lmax if mmax is None else mmax
    projection = estimate_projection_bytes(
        implementation, transform, dtype_name, nlat, lmax, mmax
    )
    extra = 0
    if _is_resampled_case(nlat, lmax):
        if implementation == "folded":
            size = _dtype_size(dtype_name)
            # even quadrature, midpoint quadrature, two real phase channels,
            # and one int8 parity sign per order.
            extra = (2 * nlat + 4 * (nlat - 1)) * size + mmax
        else:
            extra = mmax
    return projection + extra


def memory_budget_bytes(total_memory_bytes: int | None, fraction: float) -> int | None:
    """Return the configured steady-state memory budget."""

    if total_memory_bytes is None:
        return None
    if not 0.0 < fraction <= 1.0:
        raise ValueError("memory budget fraction must be in (0, 1]")
    return int(total_memory_bytes * fraction)


def capacity_decision(
    estimated_module_bytes: int,
    total_memory_bytes: int | None,
    memory_budget_fraction: float = DEFAULT_MEMORY_BUDGET_FRACTION,
) -> dict[str, Any]:
    """Decide whether permanent module storage can enter the CUDA budget."""

    budget = memory_budget_bytes(total_memory_bytes, memory_budget_fraction)
    if budget is None:
        return {
            "safe": True,
            "estimated_module_bytes": estimated_module_bytes,
            "memory_budget_bytes": None,
            "reason": None,
        }
    if estimated_module_bytes > budget:
        return {
            "safe": False,
            "estimated_module_bytes": estimated_module_bytes,
            "memory_budget_bytes": budget,
            "reason": (
                "permanent projection leaves insufficient VRAM headroom "
                f"({estimated_module_bytes} > {budget} bytes)"
            ),
        }
    return {
        "safe": True,
        "estimated_module_bytes": estimated_module_bytes,
        "memory_budget_bytes": budget,
        "reason": None,
    }


def _peak_for_prediction(record: dict[str, Any]) -> int | None:
    for key in ("peak_reserved_bytes", "peak_allocated_bytes"):
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return int(value)
    return None


def predict_next_peak_bytes(
    successful_records: Iterable[dict[str, Any]],
    next_batch: int,
    safety_margin: float = DEFAULT_PREDICTION_SAFETY_MARGIN,
) -> int | None:
    """Conservatively extrapolate the next native-batch peak.

    The last two observed peak-reserved values define a non-negative linear
    slope.  With one observation, the non-module portion is scaled by batch.
    A margin accounts for FFT workspace jumps and allocator rounding.
    """

    if next_batch < 1 or safety_margin < 1.0:
        raise ValueError("next_batch must be positive and safety margin >= 1")
    records = [
        record
        for record in successful_records
        if record.get("status") == "measured"
        and isinstance(record.get("batch_size"), int)
        and record.get("batch_size", 0) > 0
        and _peak_for_prediction(record) is not None
    ]
    records.sort(key=lambda record: int(record["batch_size"]))
    if not records:
        return None
    last = records[-1]
    last_batch = int(last["batch_size"])
    last_peak = int(_peak_for_prediction(last) or 0)
    module = int(last.get("module_buffer_bytes") or 0)
    if len(records) >= 2:
        previous = records[-2]
        previous_batch = int(previous["batch_size"])
        previous_peak = int(_peak_for_prediction(previous) or 0)
        delta_batch = max(1, last_batch - previous_batch)
        slope = max(0.0, (last_peak - previous_peak) / delta_batch)
        predicted = last_peak + slope * max(0, next_batch - last_batch)
    else:
        runtime = max(0, last_peak - module)
        predicted = module + runtime * max(1.0, next_batch / max(1, last_batch))
    return math.ceil(max(module, predicted) * safety_margin)


def _new_performance_record(
    *,
    implementation: str,
    transform: str,
    dtype_name: str,
    device: str,
    case: tuple[int, int, int],
    batch_size: int,
    measurement: str,
    warmup: int,
    repeat: int,
    source_sha: str | None,
    dense_base_sha: str,
    memory_budget: int | None,
    memory_budget_fraction: float,
    logical_batch_size: int | None = None,
    microbatch_size: int | None = None,
) -> dict[str, Any]:
    nlat, nlon, lmax = case
    projection = estimate_projection_bytes(
        implementation, transform, dtype_name, nlat, lmax
    )
    module_estimate = estimate_module_storage_bytes(
        implementation, transform, dtype_name, nlat, lmax
    )
    return {
        "schema_version": SCHEMA,
        "record_type": "performance",
        "implementation": implementation,
        "transform": transform,
        "dtype": dtype_name,
        "device": device,
        "nlat": nlat,
        "nlon": nlon,
        "exclusive_lmax_mmax": lmax,
        "exclusive_lmax": lmax,
        "exclusive_mmax": lmax,
        "batch_size": batch_size,
        "logical_batch_size": logical_batch_size,
        "microbatch_size": microbatch_size,
        "microbatch_count": (
            math.ceil(logical_batch_size / microbatch_size)
            if logical_batch_size and microbatch_size
            else None
        ),
        "measurement": measurement,
        "throughput_mode": None,
        "status": "pending",
        "reason": None,
        "warmup": warmup,
        "repeat": repeat,
        "samples_s": None,
        "median_s": None,
        "minimum_s": None,
        "seconds_per_batch": None,
        "milliseconds_per_frame": None,
        "frames_per_second": None,
        "module_buffer_bytes": None,
        "projection_bytes": projection,
        "estimated_module_bytes": module_estimate,
        "constructor_peak_bytes": None,
        "allocated_after_module_bytes": None,
        "reserved_after_module_bytes": None,
        "input_bytes": None,
        "allocated_before_timing_bytes": None,
        "reserved_before_timing_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "incremental_peak_allocated_bytes": None,
        "incremental_peak_reserved_bytes": None,
        "predicted_peak_bytes": None,
        "memory_budget_bytes": memory_budget,
        "memory_budget_fraction": memory_budget_fraction,
        "gpu_total_memory_bytes": None,
        "gpu_name": None,
        "output_l2_norm": None,
        "output_max_abs": None,
        "output_sample": None,
        "setup_s": None,
        "worker_pid": None,
        "worker_returncode": None,
        "worker_stderr_tail": None,
        "source_sha": source_sha,
        "dense_base_sha": dense_base_sha,
    }


def skipped_capacity_record(
    *,
    implementation: str,
    transform: str,
    dtype_name: str,
    device: str,
    case: tuple[int, int, int],
    batch_size: int,
    measurement: str,
    warmup: int,
    repeat: int,
    source_sha: str | None,
    dense_base_sha: str,
    memory_budget: int | None,
    memory_budget_fraction: float,
    reason: str,
    predicted_peak_bytes: int | None = None,
    logical_batch_size: int | None = None,
    microbatch_size: int | None = None,
) -> dict[str, Any]:
    """Build an explicit capacity/error record without launching a worker."""

    record = _new_performance_record(
        implementation=implementation,
        transform=transform,
        dtype_name=dtype_name,
        device=device,
        case=case,
        batch_size=batch_size,
        measurement=measurement,
        warmup=warmup,
        repeat=repeat,
        source_sha=source_sha,
        dense_base_sha=dense_base_sha,
        memory_budget=memory_budget,
        memory_budget_fraction=memory_budget_fraction,
        logical_batch_size=logical_batch_size,
        microbatch_size=microbatch_size,
    )
    record.update(
        {
            "status": "skipped_capacity",
            "reason": reason,
            "predicted_peak_bytes": predicted_peak_bytes,
        }
    )
    return record


def build_worker_command(
    script: Path,
    *,
    implementation: str,
    transform: str,
    dtype_name: str,
    device: str,
    case: tuple[int, int, int],
    batch_size: int,
    measurement: str,
    source: Path,
    dense_base_sha: str,
    output: Path,
    warmup: int,
    repeat: int,
    threads: int,
    seed: int,
    memory_budget_bytes_value: int | None,
    memory_budget_fraction: float,
    logical_batch_size: int | None = None,
    cuda_device: int = 0,
) -> list[str]:
    """Build the explicit child command used by the orchestrator."""

    command = [
        sys.executable,
        str(script),
        "--worker",
        "--implementation",
        implementation,
        "--transform",
        transform,
        "--dtype",
        dtype_name,
        "--device",
        device,
        "--case",
        _case_label(case),
        "--batch-size",
        str(batch_size),
        "--measurement",
        measurement,
        "--source",
        str(source),
        "--base",
        dense_base_sha,
        "--worker-output",
        str(output),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--threads",
        str(threads),
        "--seed",
        str(seed),
        "--memory-budget-fraction",
        str(memory_budget_fraction),
        "--cuda-device",
        str(cuda_device),
    ]
    if memory_budget_bytes_value is not None:
        command.extend(["--memory-budget-bytes", str(memory_budget_bytes_value)])
    if logical_batch_size is not None:
        command.extend(["--logical-batch-size", str(logical_batch_size)])
    return command


def aggregate_worker_record(
    record: dict[str, Any], returncode: int, stderr: str
) -> dict[str, Any]:
    """Attach parent-side process evidence without hiding worker status."""

    result = dict(record)
    result["worker_returncode"] = returncode
    if stderr:
        result["worker_stderr_tail"] = stderr[-4000:]
    if returncode != 0 and result.get("status") == "measured":
        result["status"] = "error"
        result["reason"] = f"worker exited with code {returncode}"
    return result


def logical_microbatch_count(logical_batch_size: int, microbatch_size: int) -> int:
    """Return the number of physical batches in a streamed logical workload."""

    if logical_batch_size < 1 or microbatch_size < 1:
        raise ValueError("logical and microbatch sizes must be positive")
    return math.ceil(logical_batch_size / microbatch_size)


def _load_dense_classes(base_sha: str, repository: Path) -> dict[str, Any]:
    source = subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{base_sha}:torch_harmonics/sht.py"],
        text=True,
    )
    namespace: dict[str, Any] = {
        "__name__": "torch_harmonics.sht_dense_reference",
        "__package__": "torch_harmonics",
    }
    exec(  # noqa: S102 - intentionally execute the source-pinned reference module
        compile(source, f"<torch-harmonics {base_sha[:8]} dense sht.py>", "exec"),
        namespace,
    )
    return namespace


def _load_optimized(source: Path) -> Any:
    source_string = str(source.resolve())
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    import importlib

    return importlib.import_module("torch_harmonics")


def _buffer_bytes(module: Any) -> int:
    return sum(buffer.numel() * buffer.element_size() for buffer in module.buffers())


def _projection_bytes(module: Any) -> int:
    return module.weights.numel() * module.weights.element_size()


def _error(actual: Any, reference: Any) -> dict[str, float]:
    difference = (actual - reference).abs()
    reference_norm = reference.norm().item()
    return {
        "relative_l2": (
            (difference.norm().item() / reference_norm) if reference_norm else 0.0
        ),
        "maximum_absolute": difference.max().item(),
    }


def _coefficients(
    torch: Any, batch: int, lmax: int, mmax: int, dtype: Any, *, zero_l0: bool = False
) -> Any:
    real = torch.randn(batch, lmax, mmax, dtype=dtype)
    imag = torch.randn(batch, lmax, mmax, dtype=dtype)
    result = torch.complex(real, imag)
    result[..., :, 0] = result[..., :, 0].real
    for degree in range(lmax):
        if degree + 1 < mmax:
            result[..., degree, degree + 1 :] = 0.0
    if zero_l0:
        result[..., 0, :] = 0.0
    return result


def _probe_modes(lmax: int) -> tuple[tuple[int, int], ...]:
    if lmax == 72:
        return (
            (70, 0),
            (70, 1),
            (70, 2),
            (70, 69),
            (70, 70),
            (71, 0),
            (71, 1),
            (71, 70),
            (71, 71),
        )
    degree = lmax - 1
    return tuple((degree, order) for order in (0, 1, max(1, degree - 1), degree))


def _single_mode(
    torch: Any,
    lmax: int,
    degree: int,
    order: int,
    dtype: Any,
    *,
    vector: bool,
    channel: int = 0,
) -> Any:
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    shape = (1, 2, lmax, lmax) if vector else (1, lmax, lmax)
    result = torch.zeros(shape, dtype=complex_dtype)
    index = (0, channel, degree, order) if vector else (0, degree, order)
    result[index] = 1.0 if order == 0 else 0.375 + 0.625j
    return result


def _synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _correctness_case(
    torch: Any,
    optimized: Any,
    dense: dict[str, Any],
    device: Any,
    dtype: Any,
    dtype_name: str,
    nlat: int,
    nlon: int,
    lmax: int,
    transforms: tuple[str, ...],
    source_sha: str,
) -> list[dict[str, Any]]:
    """Retain the detailed small/medium dense-vs-folded checks."""

    records: list[dict[str, Any]] = []
    torch.manual_seed(8128 + nlat + lmax)
    classes = tuple(
        item
        for item in (
            (
                "scalar",
                optimized.RealSHT,
                dense["RealSHT"],
                optimized.InverseRealSHT,
                dense["InverseRealSHT"],
                False,
            ),
            (
                "vector",
                optimized.RealVectorSHT,
                dense["RealVectorSHT"],
                optimized.InverseRealVectorSHT,
                dense["InverseRealVectorSHT"],
                True,
            ),
        )
        if item[0] in transforms
    )
    for (
        transform,
        optimized_class,
        dense_class,
        optimized_inverse,
        dense_inverse,
        vector,
    ) in classes:
        shape = (1, 2, nlat, nlon) if vector else (1, nlat, nlon)
        sample = torch.randn(*shape, device=device, dtype=dtype)
        opt = optimized_class(nlat, nlon, lmax=lmax, mmax=lmax).to(
            device=device, dtype=dtype
        )
        ref = dense_class(nlat, nlon, lmax=lmax, mmax=lmax).to(
            device=device, dtype=dtype
        )
        optimized_output = opt(sample)
        dense_output = ref(sample)
        record: dict[str, Any] = {
            "schema_version": SCHEMA,
            "record_type": "accuracy",
            "status": "measured",
            "implementation": "folded_vs_dense",
            "transform": transform,
            "dtype": dtype_name,
            "device": str(device),
            "nlat": nlat,
            "nlon": nlon,
            "exclusive_lmax_mmax": lmax,
            "source_sha": source_sha,
            "dense_base_sha": BASE_SHA,
            "spatial_input": _error(optimized_output, dense_output),
        }
        inverse = optimized_inverse(nlat, nlon, lmax=lmax, mmax=lmax).to(
            device=device, dtype=dtype
        )
        for degree, order in _probe_modes(lmax):
            for channel in range(2 if vector else 1):
                mode_coefficients = _single_mode(
                    torch,
                    lmax,
                    degree,
                    order,
                    dtype,
                    vector=vector,
                    channel=channel,
                ).to(device=device)
                mode_sample = inverse(mode_coefficients)
                mode_optimized = opt(mode_sample)
                mode_dense = ref(mode_sample)
                records.append(
                    {
                        "schema_version": SCHEMA,
                        "record_type": "accuracy",
                        "status": "measured",
                        "implementation": "folded_vs_dense",
                        "transform": transform,
                        "dtype": dtype_name,
                        "device": str(device),
                        "nlat": nlat,
                        "nlon": nlon,
                        "exclusive_lmax_mmax": lmax,
                        "source_sha": source_sha,
                        "dense_base_sha": BASE_SHA,
                        "spectrum_kind": "mode",
                        "mode_degree": degree,
                        "mode_order": order,
                        "vector_channel": channel if vector else None,
                        "spatial_input": _error(mode_optimized, mode_dense),
                        "roundtrip_optimized": _error(
                            mode_optimized, mode_coefficients
                        ),
                        "roundtrip_dense": _error(mode_dense, mode_coefficients),
                    }
                )
                del mode_coefficients, mode_sample, mode_optimized, mode_dense
        if vector:
            coefficients = _coefficients(torch, 1, lmax, lmax, dtype, zero_l0=True)
            coefficients = torch.stack((coefficients, coefficients), dim=1).to(
                device=device
            )
        else:
            coefficients = _coefficients(torch, 1, lmax, lmax, dtype).to(device=device)
        synthesized = inverse(coefficients)
        record["roundtrip_optimized"] = _error(opt(synthesized), coefficients)
        record["roundtrip_dense"] = _error(ref(synthesized), coefficients)
        records.append(record)
        del opt, ref, inverse, sample, synthesized, coefficients
        gc.collect()
    return records


def _convention_records(
    torch: Any,
    optimized: Any,
    dense: dict[str, Any],
    device: Any,
    dtype: Any,
    dtype_name: str,
    transforms: tuple[str, ...],
    source_sha: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    nlat, nlon, lmax = 17, 32, 16
    sample = torch.randn(1, nlat, nlon, device=device, dtype=dtype)
    vector_sample = torch.randn(1, 2, nlat, nlon, device=device, dtype=dtype)
    for norm in NORMS:
        for csphase in (True, False):
            common = {
                "schema_version": SCHEMA,
                "record_type": "convention",
                "status": "measured",
                "implementation": "folded_vs_dense",
                "dtype": dtype_name,
                "device": str(device),
                "nlat": nlat,
                "nlon": nlon,
                "exclusive_lmax_mmax": lmax,
                "norm": norm,
                "csphase": csphase,
                "source_sha": source_sha,
                "dense_base_sha": BASE_SHA,
            }
            if "scalar" in transforms:
                opt = optimized.RealSHT(
                    nlat, nlon, lmax=lmax, mmax=lmax, norm=norm, csphase=csphase
                ).to(device=device, dtype=dtype)
                ref = dense["RealSHT"](
                    nlat, nlon, lmax=lmax, mmax=lmax, norm=norm, csphase=csphase
                ).to(device=device, dtype=dtype)
                records.append(
                    {
                        **common,
                        "transform": "scalar",
                        "error": _error(opt(sample), ref(sample)),
                    }
                )
                del opt, ref
            if "vector" in transforms:
                opt_v = optimized.RealVectorSHT(
                    nlat, nlon, lmax=lmax, mmax=lmax, norm=norm, csphase=csphase
                ).to(device=device, dtype=dtype)
                ref_v = dense["RealVectorSHT"](
                    nlat, nlon, lmax=lmax, mmax=lmax, norm=norm, csphase=csphase
                ).to(device=device, dtype=dtype)
                records.append(
                    {
                        **common,
                        "transform": "vector",
                        "error": _error(opt_v(vector_sample), ref_v(vector_sample)),
                    }
                )
                del opt_v, ref_v
    del sample, vector_sample
    return records


def _dtype_from_name(torch: Any, dtype_name: str) -> Any:
    return {"float32": torch.float32, "float64": torch.float64}[dtype_name]


def _output_diagnostics(output: Any) -> dict[str, Any]:
    flat = output.reshape(-1)
    sample: list[dict[str, float]] = []
    for value in flat[: min(4, flat.numel())].detach().cpu().tolist():
        if isinstance(value, complex):
            sample.append({"real": float(value.real), "imag": float(value.imag)})
        else:
            sample.append({"real": float(value), "imag": 0.0})
    return {
        "output_l2_norm": float(output.norm().item()),
        "output_max_abs": float(output.abs().max().item()),
        "output_sample": sample,
    }


def _is_oom_exception(exc: BaseException) -> bool:
    return (
        "out of memory" in str(exc).lower()
        or "cuda error: out of memory" in str(exc).lower()
    )


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _git_sha(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def _configure_threads(torch: Any, threads: int) -> None:
    if threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A parent correctness phase may already have initialized the pool.
        pass


def _worker_measure_native(
    torch: Any,
    module: Any,
    device: Any,
    dtype: Any,
    transform: str,
    batch_size: int,
    nlat: int,
    nlon: int,
    measurement: str,
    warmup: int,
    repeat: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vector = transform == "vector"
    shape = (batch_size, 2, nlat, nlon) if vector else (batch_size, nlat, nlon)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sample = torch.randn(*shape, device=device, dtype=dtype)
    _synchronize(torch, device)

    def run_once(value: Any) -> None:
        if measurement == "forward":
            with torch.inference_mode():
                output = module(value)
            del output
            return
        input_value = value.detach().requires_grad_(True)
        output = module(input_value)
        output.abs().square().mean().backward()
        del input_value, output

    for _ in range(warmup):
        run_once(sample)
    _synchronize(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    allocated_before = (
        int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None
    )
    reserved_before = (
        int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else None
    )
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter_ns()
        run_once(sample)
        _synchronize(torch, device)
        samples.append((time.perf_counter_ns() - started) / 1.0e9)
    _synchronize(torch, device)
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    )
    with torch.inference_mode():
        diagnostic_output = module(sample)
    _synchronize(torch, device)
    diagnostics = _output_diagnostics(diagnostic_output)
    del diagnostic_output, sample
    timing = {
        "samples_s": samples,
        "median_s": sorted(samples)[len(samples) // 2],
        "minimum_s": min(samples),
        "seconds_per_batch": sorted(samples)[len(samples) // 2],
        "input_bytes": int(
            torch.zeros((), dtype=dtype).element_size() * math.prod(shape)
        ),
        "allocated_before_timing_bytes": allocated_before,
        "reserved_before_timing_bytes": reserved_before,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": (
            peak_allocated - allocated_before
            if peak_allocated is not None and allocated_before is not None
            else None
        ),
        "incremental_peak_reserved_bytes": (
            peak_reserved - reserved_before
            if peak_reserved is not None and reserved_before is not None
            else None
        ),
    }
    return timing, diagnostics


def _worker_measure_stream(
    torch: Any,
    module: Any,
    device: Any,
    dtype: Any,
    transform: str,
    logical_batch_size: int,
    microbatch_size: int,
    nlat: int,
    nlon: int,
    warmup: int,
    repeat: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vector = transform == "vector"
    max_shape = (
        (
            microbatch_size,
            2,
            nlat,
            nlon,
        )
        if vector
        else (microbatch_size, nlat, nlon)
    )
    element_size = torch.zeros((), dtype=dtype).element_size()

    def make_input(start: int, count: int) -> Any:
        shape = (count, 2, nlat, nlon) if vector else (count, nlat, nlon)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + start)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def run_logical() -> float:
        total = 0.0
        for start in range(0, logical_batch_size, microbatch_size):
            count = min(microbatch_size, logical_batch_size - start)
            sample = make_input(start, count)
            _synchronize(torch, device)
            started = time.perf_counter_ns()
            with torch.inference_mode():
                output = module(sample)
            _synchronize(torch, device)
            total += (time.perf_counter_ns() - started) / 1.0e9
            del output, sample
        return total

    for _ in range(warmup):
        run_logical()
    _synchronize(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    allocated_before = (
        int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None
    )
    reserved_before = (
        int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else None
    )
    samples = [run_logical() for _ in range(repeat)]
    _synchronize(torch, device)
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    )
    timing = {
        "samples_s": samples,
        "median_s": sorted(samples)[len(samples) // 2],
        "minimum_s": min(samples),
        "seconds_per_batch": sorted(samples)[len(samples) // 2],
        "input_bytes": int(element_size * math.prod(max_shape)),
        "allocated_before_timing_bytes": allocated_before,
        "reserved_before_timing_bytes": reserved_before,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": (
            peak_allocated - allocated_before
            if peak_allocated is not None and allocated_before is not None
            else None
        ),
        "incremental_peak_reserved_bytes": (
            peak_reserved - reserved_before
            if peak_reserved is not None and reserved_before is not None
            else None
        ),
    }
    return timing, {}


def _worker_execute(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    case = _parse_case(args.case)
    nlat, nlon, lmax = case
    source = Path(args.source).resolve()
    source_sha = _git_sha(source)
    memory_budget = args.memory_budget_bytes
    if memory_budget is None and args.device == "cuda" and torch.cuda.is_available():
        memory_budget = memory_budget_bytes(
            int(torch.cuda.get_device_properties(args.cuda_device).total_memory),
            args.memory_budget_fraction,
        )
    record = _new_performance_record(
        implementation=args.implementation,
        transform=args.transform,
        dtype_name=args.dtype,
        device=args.device,
        case=case,
        batch_size=args.batch_size,
        measurement=args.measurement,
        warmup=args.warmup,
        repeat=args.repeat,
        source_sha=source_sha,
        dense_base_sha=args.base,
        memory_budget=memory_budget,
        memory_budget_fraction=args.memory_budget_fraction,
        logical_batch_size=(
            args.logical_batch_size if args.measurement == "logical_stream" else None
        ),
        microbatch_size=(
            args.batch_size if args.measurement == "logical_stream" else None
        ),
    )
    record["worker_pid"] = os.getpid()
    _configure_threads(torch, args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        record.update({"status": "error", "reason": "CUDA requested but unavailable"})
        return record
    device = (
        torch.device(args.device, args.cuda_device)
        if args.device == "cuda"
        else torch.device("cpu")
    )
    dtype = _dtype_from_name(torch, args.dtype)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record["gpu_total_memory_bytes"] = int(properties.total_memory)
        record["gpu_name"] = properties.name

    if args.implementation == "folded":
        package = _load_optimized(source)
        module_class = (
            package.RealVectorSHT if args.transform == "vector" else package.RealSHT
        )
    else:
        dense = _load_dense_classes(args.base, source)
        module_class = (
            dense["RealVectorSHT"] if args.transform == "vector" else dense["RealSHT"]
        )

    started = time.perf_counter()
    if device.type == "cuda":
        construction_baseline = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        construction_baseline = None
    module = module_class(nlat, nlon, lmax=lmax, mmax=lmax)
    module = module.to(device=device, dtype=dtype).eval()
    _synchronize(torch, device)
    record["setup_s"] = time.perf_counter() - started
    record["module_buffer_bytes"] = _buffer_bytes(module)
    record["projection_bytes"] = _projection_bytes(module)
    record["allocated_after_module_bytes"] = (
        int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else None
    )
    record["reserved_after_module_bytes"] = (
        int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else None
    )
    record["constructor_peak_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) - (construction_baseline or 0)
        if device.type == "cuda"
        else None
    )

    if args.measurement == "logical_stream":
        timing, diagnostics = _worker_measure_stream(
            torch,
            module,
            device,
            dtype,
            args.transform,
            args.logical_batch_size,
            args.batch_size,
            nlat,
            nlon,
            args.warmup,
            args.repeat,
            args.seed,
        )
    else:
        timing, diagnostics = _worker_measure_native(
            torch,
            module,
            device,
            dtype,
            args.transform,
            args.batch_size,
            nlat,
            nlon,
            args.measurement,
            args.warmup,
            args.repeat,
            args.seed,
        )
    record.update(timing)
    record.update(diagnostics)
    record["status"] = "measured"
    record["seconds_per_batch"] = record["median_s"]
    record["milliseconds_per_frame"] = (
        record["median_s"]
        * 1000.0
        / (
            args.logical_batch_size
            if args.measurement == "logical_stream"
            else args.batch_size
        )
    )
    record["frames_per_second"] = (
        args.logical_batch_size
        if args.measurement == "logical_stream"
        else args.batch_size
    ) / record["median_s"]
    del module
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def _worker_main(args: argparse.Namespace) -> int:
    case = _parse_case(args.case)
    try:
        record = _worker_execute(args)
    except Exception as exc:  # noqa: BLE001 - worker must report and terminate cleanly
        status = "unexpected_oom" if _is_oom_exception(exc) else "error"
        record = _new_performance_record(
            implementation=args.implementation,
            transform=args.transform,
            dtype_name=args.dtype,
            device=args.device,
            case=case,
            batch_size=args.batch_size,
            measurement=args.measurement,
            warmup=args.warmup,
            repeat=args.repeat,
            source_sha=(
                _git_sha(Path(args.source).resolve())
                if Path(args.source).exists()
                else None
            ),
            dense_base_sha=args.base,
            memory_budget=args.memory_budget_bytes,
            memory_budget_fraction=args.memory_budget_fraction,
            logical_batch_size=(
                args.logical_batch_size
                if args.measurement == "logical_stream"
                else None
            ),
            microbatch_size=(
                args.batch_size if args.measurement == "logical_stream" else None
            ),
        )
        record.update(
            {
                "status": status,
                "reason": f"{type(exc).__name__}: {exc}",
                "worker_pid": os.getpid(),
                "traceback": traceback.format_exc(limit=8),
            }
        )
    output = Path(args.worker_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": SCHEMA, "record": record}, indent=2) + "\n")
    return 0


def _run_worker(
    command: list[str],
    worker_output: Path,
) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if worker_output.exists():
        payload = json.loads(worker_output.read_text())
        record = payload["record"]
    else:
        # This path is unusual (for example, a process-level crash before
        # Python can catch an OOM), but it must still retain the matrix key so
        # the parent can write a complete, inspectable error record.
        parsed = _worker_command_values(command)
        record = _new_performance_record(
            implementation=parsed["implementation"],
            transform=parsed["transform"],
            dtype_name=parsed["dtype"],
            device=parsed["device"],
            case=_parse_case(parsed["case"]),
            batch_size=int(parsed["batch_size"]),
            measurement=parsed["measurement"],
            warmup=int(parsed["warmup"]),
            repeat=int(parsed["repeat"]),
            source_sha=None,
            dense_base_sha=parsed["base"],
            memory_budget=(
                int(parsed["memory_budget_bytes"])
                if parsed.get("memory_budget_bytes") is not None
                else None
            ),
            memory_budget_fraction=float(parsed["memory_budget_fraction"]),
            logical_batch_size=(
                int(parsed["logical_batch_size"])
                if parsed.get("logical_batch_size") is not None
                else None
            ),
            microbatch_size=(
                int(parsed["batch_size"])
                if parsed["measurement"] == "logical_stream"
                else None
            ),
        )
        record.update(
            {
                "status": "error",
                "reason": "worker exited without a result file",
                "worker_pid": None,
            }
        )
    return aggregate_worker_record(record, completed.returncode, completed.stderr)


def _worker_command_values(command: list[str]) -> dict[str, str | None]:
    """Extract required matrix values from a generated child command."""

    values: dict[str, str | None] = {}
    option_names = {
        "implementation",
        "transform",
        "dtype",
        "device",
        "case",
        "batch_size",
        "measurement",
        "base",
        "warmup",
        "repeat",
        "memory_budget_fraction",
        "memory_budget_bytes",
        "logical_batch_size",
    }
    for index, token in enumerate(command[:-1]):
        if not token.startswith("--"):
            continue
        name = token[2:].replace("-", "_")
        if name in option_names:
            values[name] = command[index + 1]
    values.setdefault("memory_budget_bytes", None)
    values.setdefault("logical_batch_size", None)
    return values


def _run_adaptive_series(
    *,
    script: Path,
    source: Path,
    dense_base_sha: str,
    implementation: str,
    transform: str,
    dtype_name: str,
    device: str,
    case: tuple[int, int, int],
    batches: tuple[int, ...],
    measurement: str,
    warmup: int,
    repeat: int,
    threads: int,
    seed: int,
    total_memory: int | None,
    memory_fraction: float,
    prediction_margin: float,
    temp_dir: Path,
    worker_counter: list[int],
    logical_batch_size: int | None = None,
    cuda_device: int = 0,
) -> list[dict[str, Any]]:
    nlat, _nlon, lmax = case
    budget = memory_budget_bytes(total_memory, memory_fraction)
    estimate = estimate_module_storage_bytes(
        implementation, transform, dtype_name, nlat, lmax
    )
    decision = capacity_decision(estimate, total_memory, memory_fraction)
    records: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    blocked_reason: str | None = None
    for batch_size in batches:
        if not decision["safe"]:
            records.append(
                skipped_capacity_record(
                    implementation=implementation,
                    transform=transform,
                    dtype_name=dtype_name,
                    device=device,
                    case=case,
                    batch_size=batch_size,
                    measurement=measurement,
                    warmup=warmup,
                    repeat=repeat,
                    source_sha=_git_sha(source),
                    dense_base_sha=dense_base_sha,
                    memory_budget=budget,
                    memory_budget_fraction=memory_fraction,
                    reason=str(decision["reason"]),
                    logical_batch_size=logical_batch_size,
                    microbatch_size=(
                        batch_size if measurement == "logical_stream" else None
                    ),
                )
            )
            continue
        predicted = None
        if successful:
            predicted = predict_next_peak_bytes(
                successful, batch_size, prediction_margin
            )
        if blocked_reason is not None:
            records.append(
                skipped_capacity_record(
                    implementation=implementation,
                    transform=transform,
                    dtype_name=dtype_name,
                    device=device,
                    case=case,
                    batch_size=batch_size,
                    measurement=measurement,
                    warmup=warmup,
                    repeat=repeat,
                    source_sha=_git_sha(source),
                    dense_base_sha=dense_base_sha,
                    memory_budget=budget,
                    memory_budget_fraction=memory_fraction,
                    reason=blocked_reason,
                    predicted_peak_bytes=predicted,
                    logical_batch_size=logical_batch_size,
                    microbatch_size=(
                        batch_size if measurement == "logical_stream" else None
                    ),
                )
            )
            continue
        if predicted is not None and budget is not None and predicted > budget:
            blocked_reason = (
                "predicted peak exceeds memory budget; larger batches were not launched"
            )
            records.append(
                skipped_capacity_record(
                    implementation=implementation,
                    transform=transform,
                    dtype_name=dtype_name,
                    device=device,
                    case=case,
                    batch_size=batch_size,
                    measurement=measurement,
                    warmup=warmup,
                    repeat=repeat,
                    source_sha=_git_sha(source),
                    dense_base_sha=dense_base_sha,
                    memory_budget=budget,
                    memory_budget_fraction=memory_fraction,
                    reason=blocked_reason,
                    predicted_peak_bytes=predicted,
                    logical_batch_size=logical_batch_size,
                    microbatch_size=(
                        batch_size if measurement == "logical_stream" else None
                    ),
                )
            )
            continue
        worker_counter[0] += 1
        worker_output = temp_dir / f"worker-{worker_counter[0]:05d}.json"
        command = build_worker_command(
            script,
            implementation=implementation,
            transform=transform,
            dtype_name=dtype_name,
            device=device,
            case=case,
            batch_size=batch_size,
            measurement=measurement,
            source=source,
            dense_base_sha=dense_base_sha,
            output=worker_output,
            warmup=warmup,
            repeat=repeat,
            threads=threads,
            seed=seed,
            memory_budget_bytes_value=budget,
            memory_budget_fraction=memory_fraction,
            logical_batch_size=logical_batch_size,
            cuda_device=cuda_device,
        )
        record = _run_worker(command, worker_output)
        record["predicted_peak_bytes"] = predicted
        records.append(record)
        if record.get("status") == "measured":
            successful.append(record)
        elif record.get("status") == "unexpected_oom":
            blocked_reason = (
                "previous batch ended in unexpected OOM; larger batches skipped"
            )
        elif record.get("status") == "error":
            blocked_reason = (
                "previous batch ended in worker error; larger batches skipped"
            )
    return records


def _native_successes(
    records: list[dict[str, Any]],
    implementation: str,
    transform: str,
    dtype_name: str,
    case: tuple[int, int, int],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("record_type") == "performance"
        and record.get("measurement") == "forward"
        and record.get("implementation") == implementation
        and record.get("transform") == transform
        and record.get("dtype") == dtype_name
        and (record.get("nlat"), record.get("nlon"), record.get("exclusive_lmax_mmax"))
        == case
        and record.get("status") == "measured"
    ]


def _run_logical_stream_matrix(
    *,
    script: Path,
    source: Path,
    dense_base_sha: str,
    cases: tuple[tuple[int, int, int], ...],
    dtypes: tuple[str, ...],
    transforms: tuple[str, ...],
    native_records: list[dict[str, Any]],
    device: str,
    logical_batch_size: int,
    warmup: int,
    repeat: int,
    threads: int,
    seed: int,
    total_memory: int | None,
    memory_fraction: float,
    prediction_margin: float,
    temp_dir: Path,
    worker_counter: list[int],
    cuda_device: int = 0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        for dtype_name in dtypes:
            for transform in transforms:
                safe_batches: dict[str, int | None] = {}
                for implementation in ("dense", "folded"):
                    successes = _native_successes(
                        native_records, implementation, transform, dtype_name, case
                    )
                    safe_batches[implementation] = max(
                        (record["batch_size"] for record in successes), default=None
                    )
                common = min(
                    (value for value in safe_batches.values() if value is not None),
                    default=None,
                )
                for implementation in ("dense", "folded"):
                    best = safe_batches[implementation]
                    for mode, microbatch in (
                        ("best_safe", best),
                        ("common_safe", common),
                    ):
                        if microbatch is None:
                            records.append(
                                skipped_capacity_record(
                                    implementation=implementation,
                                    transform=transform,
                                    dtype_name=dtype_name,
                                    device=device,
                                    case=case,
                                    batch_size=0,
                                    measurement="logical_stream",
                                    warmup=warmup,
                                    repeat=repeat,
                                    source_sha=_git_sha(source),
                                    dense_base_sha=dense_base_sha,
                                    memory_budget=memory_budget_bytes(
                                        total_memory, memory_fraction
                                    ),
                                    memory_budget_fraction=memory_fraction,
                                    reason="no measured native batch was capacity-safe",
                                    logical_batch_size=logical_batch_size,
                                    microbatch_size=None,
                                )
                            )
                            records[-1]["throughput_mode"] = mode
                            continue
                        worker_counter[0] += 1
                        worker_output = (
                            temp_dir / f"worker-{worker_counter[0]:05d}.json"
                        )
                        command = build_worker_command(
                            script,
                            implementation=implementation,
                            transform=transform,
                            dtype_name=dtype_name,
                            device=device,
                            case=case,
                            batch_size=microbatch,
                            measurement="logical_stream",
                            source=source,
                            dense_base_sha=dense_base_sha,
                            output=worker_output,
                            warmup=warmup,
                            repeat=repeat,
                            threads=threads,
                            seed=seed,
                            memory_budget_bytes_value=memory_budget_bytes(
                                total_memory, memory_fraction
                            ),
                            memory_budget_fraction=memory_fraction,
                            logical_batch_size=logical_batch_size,
                            cuda_device=cuda_device,
                        )
                        stream_record = _run_worker(command, worker_output)
                        stream_record["throughput_mode"] = mode
                        stream_record["predicted_peak_bytes"] = None
                        records.append(stream_record)
    return records


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("record_type"),
        record.get("implementation"),
        record.get("transform"),
        record.get("dtype"),
        record.get("nlat"),
        record.get("nlon"),
        record.get("exclusive_lmax_mmax"),
        record.get("batch_size"),
        record.get("measurement"),
        record.get("throughput_mode"),
    )


def validate_result_payload(
    payload: dict[str, Any],
    *,
    expected_native_keys: set[tuple[Any, ...]] | None = None,
) -> list[str]:
    """Run structural checks before machine-readable results are checked in."""

    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append(f"unexpected schema: {payload.get('schema')!r}")
    seen: set[tuple[Any, ...]] = set()
    native_keys: set[tuple[Any, ...]] = set()
    for record in payload.get("records", []):
        if record.get("record_type") != "performance":
            continue
        key = _record_key(record)
        if key in seen:
            errors.append(f"duplicate performance key: {key}")
        seen.add(key)
        if record.get("measurement") == "forward":
            native_keys.add(key)
        if record.get("status") == "measured":
            median = record.get("median_s")
            if (
                not isinstance(median, (int, float))
                or not math.isfinite(float(median))
                or median < 0
            ):
                errors.append(f"invalid measured timing for {key}")
            for field in (
                "incremental_peak_allocated_bytes",
                "incremental_peak_reserved_bytes",
            ):
                value = record.get(field)
                if value is not None and value < 0:
                    errors.append(f"negative {field} for {key}")
        if record.get("measurement") == "logical_stream":
            logical = record.get("logical_batch_size")
            micro = record.get("microbatch_size")
            count = record.get("microbatch_count")
            if record.get("status") == "measured" and (
                logical is None
                or micro is None
                or count != logical_microbatch_count(logical, micro)
            ):
                errors.append(f"invalid logical stream dimensions for {key}")
    if expected_native_keys is not None:
        missing = expected_native_keys - native_keys
        if missing:
            errors.append(f"missing native records: {sorted(missing)!r}")
    return errors


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_result_files(
    payload: dict[str, Any], output: Path
) -> tuple[Path, Path, Path]:
    json_output = output if output.suffix == ".json" else output.with_suffix(".json")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_output = json_output.with_suffix(".csv")
    fieldnames: list[str] = []
    for record in payload.get("records", []):
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in payload.get("records", []):
            writer.writerow({key: _csv_value(record.get(key)) for key in fieldnames})
    markdown_output = json_output.with_suffix(".md")
    markdown_output.write_text(render_markdown_report(payload))
    return json_output, csv_output, markdown_output


def _case_name(record: dict[str, Any]) -> str:
    return f"{record['nlat']}x{record['nlon']}"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "SKIP"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "SKIP"
        return f"{value:.{digits}f}"
    return str(value)


def _mib(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "SKIP"
    return f"{value / (1024.0**2):.0f}"


def _status_timing(record: dict[str, Any] | None, field: str) -> str:
    if not record or record.get("status") != "measured":
        return "SKIP"
    return _fmt(record.get(field))


def _pair_records(
    records: list[dict[str, Any]],
    *,
    measurement: str,
    case: tuple[int, int, int],
    dtype_name: str,
    transform: str,
    batch_size: int | None = None,
    throughput_mode: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    selected = []
    for implementation in ("dense", "folded"):
        selected.append(
            next(
                (
                    record
                    for record in records
                    if record.get("implementation") == implementation
                    and record.get("measurement") == measurement
                    and record.get("nlat") == case[0]
                    and record.get("nlon") == case[1]
                    and record.get("exclusive_lmax_mmax") == case[2]
                    and record.get("dtype") == dtype_name
                    and record.get("transform") == transform
                    and (batch_size is None or record.get("batch_size") == batch_size)
                    and (
                        throughput_mode is None
                        or record.get("throughput_mode") == throughput_mode
                    )
                ),
                None,
            )
        )
    return selected[0], selected[1]


def _common_measured_batches(
    records: list[dict[str, Any]],
    case: tuple[int, int, int],
    dtype_name: str,
    transform: str,
) -> list[int]:
    batches: list[set[int]] = []
    for implementation in ("dense", "folded"):
        batches.append(
            {
                int(record["batch_size"])
                for record in records
                if record.get("record_type") == "performance"
                and record.get("measurement") == "forward"
                and record.get("implementation") == implementation
                and record.get("nlat") == case[0]
                and record.get("nlon") == case[1]
                and record.get("exclusive_lmax_mmax") == case[2]
                and record.get("dtype") == dtype_name
                and record.get("transform") == transform
                and record.get("status") == "measured"
            }
        )
    return sorted(batches[0] & batches[1])


def _max_accuracy(records: list[dict[str, Any]], field: str) -> float | None:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if isinstance(value, dict) and isinstance(
            value.get("relative_l2"), (int, float)
        ):
            values.append(float(value["relative_l2"]))
    return max(values) if values else None


def render_markdown_report(payload: dict[str, Any]) -> str:
    """Render the checked-in report directly from JSON records."""

    records = payload.get("records", [])
    provenance = payload.get("provenance", {})
    performance = [
        record for record in records if record.get("record_type") == "performance"
    ]
    forward = [
        record for record in performance if record.get("measurement") == "forward"
    ]
    logical = [
        record
        for record in performance
        if record.get("measurement") == "logical_stream"
    ]
    backward = [
        record
        for record in performance
        if record.get("measurement") == "forward_backward"
    ]
    lines = [
        "# Batched CUDA dense-vs-folded Torch SHT benchmark",
        "",
        "This report is generated from `torch-folding-batched-cuda-v2.json`; `SKIP` denotes an explicit capacity or worker-status record, not a fabricated timing.",
        "",
        "## Executive summary",
        "",
    ]
    unexpected_ooms = sum(
        record.get("status") == "unexpected_oom" for record in performance
    )
    measured_forward = sum(record.get("status") == "measured" for record in forward)
    lines.append(
        f"- The forward matrix contains {measured_forward} measured records and {unexpected_ooms} unexpected OOM records."
    )
    lines.append(
        "- Folded/dense ratios below 1.0 mean folded is faster; ratios are shown only when both implementations were measured at the same case, dtype, transform, and batch."
    )

    def summary_ratio(
        case: tuple[int, int, int],
        dtype_name: str,
        transform: str,
        batch: int | None = None,
    ) -> str:
        batches = _common_measured_batches(forward, case, dtype_name, transform)
        chosen = batch if batch in batches else (batches[-1] if batches else None)
        if chosen is None:
            return "not established"
        dense, folded = _pair_records(
            forward,
            measurement="forward",
            case=case,
            dtype_name=dtype_name,
            transform=transform,
            batch_size=chosen,
        )
        if (
            not dense
            or not folded
            or dense.get("status") != "measured"
            or folded.get("status") != "measured"
        ):
            return "not established"
        ratio = folded["milliseconds_per_frame"] / dense["milliseconds_per_frame"]
        return f"{ratio:.3f} at batch {chosen}"

    lines.append(
        f"- At 73x144, scalar float32 folded/dense is {summary_ratio((73, 144, 72), 'float32', 'scalar', 128)}; this is the direct low-resolution batching comparison."
    )
    lines.append(
        f"- At 257x512, scalar float32 folded/dense is {summary_ratio((257, 512, 256), 'float32', 'scalar')} at the largest common measured batch."
    )
    lines.append(
        f"- At 721x1440, scalar float32 is {summary_ratio((721, 1440, 720), 'float32', 'scalar', 1)} and scalar float64 is {summary_ratio((721, 1440, 720), 'float64', 'scalar', 1)} at batch 1 where available."
    )
    lines.append("")

    lines.extend(
        [
            "## Provenance",
            "",
            "| field | value |",
            "|---|---|",
            f"| run UTC | {provenance.get('created_at_utc', 'unknown')} |",
            f"| sht_bench HEAD | `{provenance.get('sht_bench_sha', 'unknown')}` |",
            f"| optimized source SHA | `{provenance.get('optimized_source_sha', 'unknown')}` |",
            f"| dense base SHA | `{provenance.get('dense_base_sha', 'unknown')}` |",
            f"| Python / PyTorch | {provenance.get('python_version', 'unknown')} / {provenance.get('torch_version', 'unknown')} |",
            f"| CUDA runtime | {provenance.get('cuda_runtime', 'unknown')} |",
            f"| GPU / total VRAM | {provenance.get('gpu_name', 'unknown')} / {_mib(provenance.get('gpu_total_memory_bytes'))} MiB |",
            f"| warmup / repeat | {provenance.get('warmup', 'unknown')} / {provenance.get('repeat', 'unknown')} |",
            f"| memory budget | {provenance.get('memory_budget_fraction', 'unknown'):.2f} of total VRAM |"
            if isinstance(provenance.get("memory_budget_fraction"), float)
            else "| memory budget | unknown |",
            f"| candidate native batches | `{','.join(str(x) for x in provenance.get('candidate_batch_sizes', []))}` |",
            "",
            "## Method",
            "",
            "The parent process launches one fresh worker per implementation × transform × dtype × grid × batch × measurement. Dense and folded modules therefore do not share a CUDA allocator state. Inputs are allocated before synchronized wall-clock timing; warmups and repeats are recorded per worker. The normal matrix never constructs the historical hoisted projection. Static projection estimates and conservative observed-peak extrapolation create explicit `skipped_capacity` records before unsafe batches are launched.",
            "",
            "## Native batch scaling",
            "",
            "| case | dtype | transform | batch | dense ms/frame | folded ms/frame | folded/dense | dense peak MiB | folded peak MiB |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    cases = sorted(
        {(r.get("nlat"), r.get("nlon"), r.get("exclusive_lmax_mmax")) for r in forward}
    )
    dtypes = sorted({r.get("dtype") for r in forward})
    transforms = sorted({r.get("transform") for r in forward})
    batches = sorted(
        {
            r.get("batch_size")
            for r in forward
            if isinstance(r.get("batch_size"), int) and r.get("batch_size") > 0
        }
    )
    for case in cases:
        case_tuple = (case[0], case[1], case[2])
        for dtype_name in dtypes:
            for transform in transforms:
                for batch in batches:
                    dense, folded = _pair_records(
                        forward,
                        measurement="forward",
                        case=case_tuple,
                        dtype_name=dtype_name,
                        transform=transform,
                        batch_size=batch,
                    )
                    ratio = None
                    if (
                        dense
                        and folded
                        and dense.get("status") == folded.get("status") == "measured"
                    ):
                        ratio = (
                            folded["milliseconds_per_frame"]
                            / dense["milliseconds_per_frame"]
                        )
                    lines.append(
                        f"| {_case_name(dense or folded or {'nlat': case[0], 'nlon': case[1]})} | {dtype_name} | {transform} | {batch} | {_status_timing(dense, 'milliseconds_per_frame')} | {_status_timing(folded, 'milliseconds_per_frame')} | {_fmt(ratio)} | {_mib(dense.get('peak_reserved_bytes') if dense else None)} | {_mib(folded.get('peak_reserved_bytes') if folded else None)} |"
                    )

    lines.extend(
        [
            "",
            "## Logical batch-128 streaming throughput",
            "",
            "The best-safe rows use each implementation's largest measured native microbatch. Common-safe rows use the same physical microbatch for both implementations.",
            "",
            "| case | dtype | transform | mode | dense microbatch | folded microbatch | dense ms/frame | folded ms/frame | dense frames/s | folded frames/s |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    logical_cases = sorted(
        {(r.get("nlat"), r.get("nlon"), r.get("exclusive_lmax_mmax")) for r in logical}
    )
    for case in logical_cases:
        case_tuple = (case[0], case[1], case[2])
        for dtype_name in sorted({r.get("dtype") for r in logical}):
            for transform in sorted({r.get("transform") for r in logical}):
                for mode in ("best_safe", "common_safe"):
                    dense, folded = _pair_records(
                        logical,
                        measurement="logical_stream",
                        case=case_tuple,
                        dtype_name=dtype_name,
                        transform=transform,
                        throughput_mode=mode,
                    )
                    lines.append(
                        f"| {_case_name(dense or folded or {'nlat': case[0], 'nlon': case[1]})} | {dtype_name} | {transform} | {mode} | {_fmt(dense.get('microbatch_size') if dense else None, 0)} | {_fmt(folded.get('microbatch_size') if folded else None, 0)} | {_status_timing(dense, 'milliseconds_per_frame')} | {_status_timing(folded, 'milliseconds_per_frame')} | {_status_timing(dense, 'frames_per_second')} | {_status_timing(folded, 'frames_per_second')} |"
                    )

    lines.extend(
        [
            "",
            "## Safe native batch maxima",
            "",
            "| case | dtype | transform | dense max batch | folded max batch |",
            "|---|---|---|---:|---:|",
        ]
    )
    for case in cases:
        case_tuple = (case[0], case[1], case[2])
        for dtype_name in dtypes:
            for transform in transforms:
                maxima = []
                for implementation in ("dense", "folded"):
                    measured = [
                        int(r["batch_size"])
                        for r in forward
                        if r.get("implementation") == implementation
                        and (r.get("nlat"), r.get("nlon"), r.get("exclusive_lmax_mmax"))
                        == case_tuple
                        and r.get("dtype") == dtype_name
                        and r.get("transform") == transform
                        and r.get("status") == "measured"
                    ]
                    maxima.append(str(max(measured)) if measured else "SKIP")
                lines.append(
                    f"| {case[0]}x{case[1]} | {dtype_name} | {transform} | {maxima[0]} | {maxima[1]} |"
                )

    lines.extend(
        [
            "",
            "## Forward+backward subset",
            "",
            "| case | dtype | transform | batch | dense ms/frame | folded ms/frame | folded/dense |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    backward_cases = sorted(
        {(r.get("nlat"), r.get("nlon"), r.get("exclusive_lmax_mmax")) for r in backward}
    )
    for case in backward_cases:
        case_tuple = (case[0], case[1], case[2])
        for dtype_name in sorted({r.get("dtype") for r in backward}):
            for transform in sorted({r.get("transform") for r in backward}):
                for batch in sorted(
                    {r.get("batch_size") for r in backward if r.get("batch_size")}
                ):
                    dense, folded = _pair_records(
                        backward,
                        measurement="forward_backward",
                        case=case_tuple,
                        dtype_name=dtype_name,
                        transform=transform,
                        batch_size=batch,
                    )
                    ratio = None
                    if (
                        dense
                        and folded
                        and dense.get("status") == folded.get("status") == "measured"
                    ):
                        ratio = (
                            folded["milliseconds_per_frame"]
                            / dense["milliseconds_per_frame"]
                        )
                    lines.append(
                        f"| {case[0]}x{case[1]} | {dtype_name} | {transform} | {batch} | {_status_timing(dense, 'milliseconds_per_frame')} | {_status_timing(folded, 'milliseconds_per_frame')} | {_fmt(ratio)} |"
                    )

    lines.extend(["", "## Correctness and conventions", ""])
    accuracy = [r for r in records if r.get("record_type") == "accuracy"]
    conventions = [r for r in records if r.get("record_type") == "convention"]
    if accuracy:
        lines.append(
            f"The CPU correctness phase produced {len(accuracy)} detailed records across scalar/vector, float32/float64, random round trips, representative high-degree/low-order modes, and the norm/Condon–Shortley convention checks. The largest recorded relative L2 errors by dtype were:"
        )
        lines.append("")
        lines.append(
            "| dtype | max spatial-input relative L2 | max optimized roundtrip relative L2 | max dense roundtrip relative L2 |"
        )
        lines.append("|---|---:|---:|---:|")
        for dtype_name in sorted({r.get("dtype") for r in accuracy}):
            subset = [r for r in accuracy if r.get("dtype") == dtype_name]
            lines.append(
                f"| {dtype_name} | {_fmt(_max_accuracy(subset, 'spatial_input'))} | {_fmt(_max_accuracy(subset, 'roundtrip_optimized'))} | {_fmt(_max_accuracy(subset, 'roundtrip_dense'))} |"
            )
    else:
        lines.append("No correctness records were included in this result family.")
    lines.append(f"Convention records: {len(conventions)}.")

    lines.extend(["", "## Findings", ""])

    low = (73, 144, 72)
    medium = (257, 512, 256)
    high = (721, 1440, 720)

    def finding_ratio(
        case: tuple[int, int, int], dtype_name: str, transform: str, batch: int
    ) -> str:
        dense, folded = _pair_records(
            forward,
            measurement="forward",
            case=case,
            dtype_name=dtype_name,
            transform=transform,
            batch_size=batch,
        )
        if (
            not dense
            or not folded
            or dense.get("status") != folded.get("status") == "measured"
        ):
            return "not measured for both implementations"
        return (
            f"{folded['milliseconds_per_frame'] / dense['milliseconds_per_frame']:.3f}"
        )

    lines.append(
        f"1. At 73x144, scalar float32 folded/dense is {finding_ratio(low, 'float32', 'scalar', 1)} at batch 1 and {summary_ratio(low, 'float32', 'scalar', 128)} at the largest requested low-resolution batch; batching therefore changes the observable gap only if those two measured ratios differ."
    )
    medium_crossover: dict[tuple[str, str], int | None] = {}
    for dtype_name in dtypes:
        for transform in transforms:
            crossover = None
            for batch in _common_measured_batches(
                forward, medium, dtype_name, transform
            ):
                dense, folded = _pair_records(
                    forward,
                    measurement="forward",
                    case=medium,
                    dtype_name=dtype_name,
                    transform=transform,
                    batch_size=batch,
                )
                if (
                    dense
                    and folded
                    and dense.get("status") == folded.get("status") == "measured"
                    and folded["milliseconds_per_frame"]
                    < dense["milliseconds_per_frame"]
                ):
                    crossover = batch
                    break
            medium_crossover[(dtype_name, transform)] = crossover
    lines.append(
        "2. At 257x512, the first measured scalar crossover is "
        + "; ".join(
            f"{dtype}: batch {medium_crossover[(dtype, 'scalar')] if medium_crossover[(dtype, 'scalar')] is not None else 'none'}"
            for dtype in dtypes
        )
        + "."
    )
    lines.append(
        f"3. At 721x1440, scalar float32 folded/dense is {summary_ratio(high, 'float32', 'scalar', 1)} and scalar float64 is {summary_ratio(high, 'float64', 'scalar', 1)} at batch 1; vector results are in the native table."
    )
    lines.append(
        "4. The safe native maxima are listed above; they are measured maxima, not claims that the next unlaunched batch would fit."
    )
    high_vector_f64_dense = [
        r
        for r in forward
        if (r.get("nlat"), r.get("nlon"), r.get("exclusive_lmax_mmax")) == high
        and r.get("dtype") == "float64"
        and r.get("transform") == "vector"
        and r.get("implementation") == "dense"
    ]
    high_vector_f64_folded = [
        r
        for r in forward
        if (r.get("nlat"), r.get("nlon"), r.get("exclusive_lmax_mmax")) == high
        and r.get("dtype") == "float64"
        and r.get("transform") == "vector"
        and r.get("implementation") == "folded"
    ]
    dense_skipped = any(
        r.get("status") == "skipped_capacity" for r in high_vector_f64_dense
    )
    folded_measured = any(r.get("status") == "measured" for r in high_vector_f64_folded)
    lines.append(
        f"5. For high-resolution float64 vector analysis, dense is {'capacity-skipped' if dense_skipped else 'not capacity-skipped'} while folded is {'measured at least once' if folded_measured else 'not measured'}; this is the direct evidence for whether folding enables an otherwise unsafe workload."
    )
    lines.append(
        "6. Dense can remain fast despite its larger projection: the same-batch ratios and peak-reserved columns show whether that tradeoff occurs for each resolution rather than assuming projection size determines runtime."
    )
    lines.append(
        "7. Runtime FFT/intermediate storage is visible in peak-reserved memory. Compare the approximately 2:1 dense/folded permanent projection columns in the provenance-independent records with the native peak columns before attributing the full advantage to folding."
    )
    logical_pairs = []
    for dtype_name in dtypes:
        for transform in transforms:
            dense, folded = _pair_records(
                logical,
                measurement="logical_stream",
                case=high,
                dtype_name=dtype_name,
                transform=transform,
                throughput_mode="best_safe",
            )
            if (
                dense
                and folded
                and dense.get("status") == folded.get("status") == "measured"
            ):
                logical_pairs.append(
                    (
                        dtype_name,
                        transform,
                        folded["frames_per_second"] / dense["frames_per_second"],
                    )
                )
    lines.append(
        "8. Logical batch-128 best-safe throughput ratios (folded/dense) at 721x1440 are "
        + (
            ", ".join(
                f"{dtype}/{transform} {ratio:.3f}"
                for dtype, transform, ratio in logical_pairs
            )
            if logical_pairs
            else "not established"
        )
        + "."
    )
    lines.append(
        "9. Float64 can change the crossover because both projection storage and runtime arithmetic double; the medium-resolution crossover line and float32/float64 native tables are the measured comparison."
    )
    lines.append(
        "10. Scalar/vector behavior differs when the vector rows show a different ratio or capacity maximum; vector projection storage is twice scalar storage, so the vector-specific rows are retained rather than averaged away."
    )

    lines.extend(
        [
            "",
            "## Capacity and errors",
            "",
            f"Measured forward records: {measured_forward}. Capacity skips: {sum(record.get('status') == 'skipped_capacity' for record in forward)}. Unexpected OOMs: {unexpected_ooms}. Other worker errors: {sum(record.get('status') == 'error' for record in performance)}.",
            "",
            "No hoisted projection was constructed by the standard matrix.",
            "",
        ]
    )
    return "\n".join(lines)


def _provenance(
    *,
    repository: Path,
    source: Path,
    dense_base_sha: str,
    torch: Any,
    device: str,
    warmup: int,
    repeat: int,
    memory_fraction: float,
    batches: tuple[int, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "sht_bench_sha": _git_sha(repository),
        "optimized_source_sha": _git_sha(source),
        "dense_base_sha": dense_base_sha,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cpu": _cpu_model(),
        "torch_threads": torch.get_num_threads(),
        "device": device,
        "warmup": warmup,
        "repeat": repeat,
        "memory_budget_fraction": memory_fraction,
        "candidate_batch_sizes": list(batches),
        "timing_method": "synchronized perf_counter_ns wall time; median of repeats",
        "memory_isolation": "one fresh subprocess per implementation/case/batch/measurement",
        "hoisted_projection": "disabled in standard matrix",
    }
    if device == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu_name": properties.name,
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "gpu_total_memory_bytes": int(properties.total_memory),
            }
        )
    else:
        result.update({"gpu_name": None, "gpu_total_memory_bytes": None})
    return result


def _expected_native_keys(
    cases: tuple[tuple[int, int, int], ...],
    dtypes: tuple[str, ...],
    transforms: tuple[str, ...],
    batches: tuple[int, ...],
) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for case in cases:
        for dtype_name in dtypes:
            for transform in transforms:
                for implementation in ("dense", "folded"):
                    for batch in batches:
                        keys.add(
                            (
                                "performance",
                                implementation,
                                transform,
                                dtype_name,
                                case[0],
                                case[1],
                                case[2],
                                batch,
                                "forward",
                                None,
                            )
                        )
    return keys


def _parent_main(args: argparse.Namespace) -> int:
    import torch

    source = Path(args.source).resolve()
    repository = Path.cwd().resolve()
    if not source.exists():
        raise SystemExit(f"source checkout does not exist: {source}")
    source_sha = _git_sha(source)
    cases = _parse_cases(args.cases)
    correctness_cases = _parse_cases(args.correctness_cases)
    batches = _parse_batch_list(args.batches)
    backward_batches = _parse_batch_list(args.backward_batches)
    dtypes = _parse_names(args.dtypes, {"float32", "float64"}, "--dtypes")
    transforms = _parse_names(args.transforms, {"scalar", "vector"}, "--transforms")
    if args.correctness_only and args.benchmark_only:
        raise SystemExit(
            "--correctness-only and --benchmark-only are mutually exclusive"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    _configure_threads(torch, args.threads)
    total_memory = (
        int(torch.cuda.get_device_properties(args.cuda_device).total_memory)
        if args.device == "cuda"
        else None
    )
    records: list[dict[str, Any]] = []

    if not args.benchmark_only:
        optimized = _load_optimized(source)
        dense = _load_dense_classes(args.base, source)
        correctness_device = torch.device("cpu")
        for dtype_name in dtypes:
            dtype = _dtype_from_name(torch, dtype_name)
            for case in correctness_cases:
                records.extend(
                    _correctness_case(
                        torch,
                        optimized,
                        dense,
                        correctness_device,
                        dtype,
                        dtype_name,
                        *case,
                        transforms,
                        source_sha,
                    )
                )
            records.extend(
                _convention_records(
                    torch,
                    optimized,
                    dense,
                    correctness_device,
                    dtype,
                    dtype_name,
                    transforms,
                    source_sha,
                )
            )
        del optimized, dense
        gc.collect()

    if not args.correctness_only:
        script = Path(__file__).resolve()
        worker_counter = [0]
        with tempfile.TemporaryDirectory(prefix="torch-folding-workers-") as temp_name:
            temp_dir = Path(temp_name)
            for case in cases:
                for dtype_name in dtypes:
                    for transform in transforms:
                        for implementation in ("dense", "folded"):
                            records.extend(
                                _run_adaptive_series(
                                    script=script,
                                    source=source,
                                    dense_base_sha=args.base,
                                    implementation=implementation,
                                    transform=transform,
                                    dtype_name=dtype_name,
                                    device=args.device,
                                    case=case,
                                    batches=batches,
                                    measurement="forward",
                                    warmup=args.warmup,
                                    repeat=args.repeat,
                                    threads=args.threads,
                                    seed=args.seed,
                                    total_memory=total_memory,
                                    memory_fraction=args.memory_budget_fraction,
                                    prediction_margin=args.prediction_safety_margin,
                                    temp_dir=temp_dir,
                                    worker_counter=worker_counter,
                                    cuda_device=args.cuda_device,
                                )
                            )
            if args.logical_batch_size:
                records.extend(
                    _run_logical_stream_matrix(
                        script=script,
                        source=source,
                        dense_base_sha=args.base,
                        cases=cases,
                        dtypes=dtypes,
                        transforms=transforms,
                        native_records=records,
                        device=args.device,
                        logical_batch_size=args.logical_batch_size,
                        warmup=args.stream_warmup,
                        repeat=args.stream_repeat,
                        threads=args.threads,
                        seed=args.seed,
                        total_memory=total_memory,
                        memory_fraction=args.memory_budget_fraction,
                        prediction_margin=args.prediction_safety_margin,
                        temp_dir=temp_dir,
                        worker_counter=worker_counter,
                        cuda_device=args.cuda_device,
                    )
                )
            if not args.skip_backward:
                for case in cases:
                    if case[0] not in {73, 257, 721}:
                        continue
                    candidate_batches = backward_batches if case[0] != 721 else batches
                    for dtype_name in dtypes:
                        for transform in transforms:
                            for implementation in ("dense", "folded"):
                                records.extend(
                                    _run_adaptive_series(
                                        script=script,
                                        source=source,
                                        dense_base_sha=args.base,
                                        implementation=implementation,
                                        transform=transform,
                                        dtype_name=dtype_name,
                                        device=args.device,
                                        case=case,
                                        batches=candidate_batches,
                                        measurement="forward_backward",
                                        warmup=args.backward_warmup,
                                        repeat=args.backward_repeat,
                                        threads=args.threads,
                                        seed=args.seed + 10000,
                                        total_memory=total_memory,
                                        memory_fraction=args.memory_budget_fraction,
                                        prediction_margin=args.prediction_safety_margin,
                                        temp_dir=temp_dir,
                                        worker_counter=worker_counter,
                                        cuda_device=args.cuda_device,
                                    )
                                )

    payload = {
        "schema": SCHEMA,
        "source": str(source),
        "source_sha": source_sha,
        "dense_base_sha": args.base,
        "device": args.device,
        "provenance": _provenance(
            repository=repository,
            source=source,
            dense_base_sha=args.base,
            torch=torch,
            device=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            memory_fraction=args.memory_budget_fraction,
            batches=batches,
        ),
        "records": records,
    }
    expected = (
        None
        if args.correctness_only
        else _expected_native_keys(cases, dtypes, transforms, batches)
    )
    validation_errors = validate_result_payload(payload, expected_native_keys=expected)
    if validation_errors:
        raise SystemExit(
            "result validation failed:\n" + "\n".join(validation_errors[:20])
        )
    json_output, csv_output, markdown_output = _write_result_files(
        payload, Path(args.output)
    )
    print(json_output)
    print(csv_output)
    print(markdown_output)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../torch-harmonics"))
    parser.add_argument("--base", default=BASE_SHA)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--backward-warmup", type=int, default=2)
    parser.add_argument("--backward-repeat", type=int, default=3)
    parser.add_argument("--stream-warmup", type=int, default=1)
    parser.add_argument("--stream-repeat", type=int, default=3)
    parser.add_argument(
        "--cases",
        default=",".join(_case_label(case) for case in DEFAULT_CASES),
    )
    parser.add_argument(
        "--correctness-cases",
        default=",".join(_case_label(case) for case in DEFAULT_CORRECTNESS_CASES),
    )
    parser.add_argument(
        "--batches", default=",".join(str(batch) for batch in DEFAULT_BATCHES)
    )
    parser.add_argument("--backward-batches", default="1,16,128")
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--transforms", default="scalar,vector")
    parser.add_argument("--logical-batch-size", type=int, default=128)
    parser.add_argument(
        "--memory-budget-fraction", type=float, default=DEFAULT_MEMORY_BUDGET_FRACTION
    )
    parser.add_argument(
        "--prediction-safety-margin",
        type=float,
        default=DEFAULT_PREDICTION_SAFETY_MARGIN,
    )
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/torch-folding-batched-cuda-v2.json"),
    )
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--skip-backward", action="store_true")

    # Private worker interface.  These options are intentionally still parsed
    # by the one script so tests and provenance use the same command builder.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--implementation", choices=("dense", "folded"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--transform", choices=("scalar", "vector"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), help=argparse.SUPPRESS
    )
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--measurement",
        choices=("forward", "forward_backward", "logical_stream"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--memory-budget-bytes", type=int, default=None, help=argparse.SUPPRESS
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.worker:
        required = (
            args.implementation,
            args.transform,
            args.dtype,
            args.case,
            args.batch_size,
            args.measurement,
            args.worker_output,
        )
        if any(value is None for value in required):
            parser.error(
                "worker mode requires implementation, transform, dtype, case, batch-size, measurement, and worker-output"
            )
        if args.batch_size < 1 or args.warmup < 0 or args.repeat < 1:
            parser.error(
                "worker batch-size must be positive, warmup non-negative, repeat positive"
            )
        if args.measurement == "logical_stream" and args.logical_batch_size < 1:
            parser.error("logical batch size must be positive")
        return _worker_main(args)
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be non-negative and repeat must be positive")
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
