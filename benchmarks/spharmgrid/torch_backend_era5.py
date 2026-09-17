"""Streamed CPU/CUDA benchmark for the optional spharmgrid Torch backend.

This benchmark is intentionally repository-only.  It reads the real ERA5
NetCDF field in bounded time batches, keeps all Torch work under inference mode,
and reports synchronized transform timings separately from batch loading.

The benchmark environment needs ``scipy`` for the classic NetCDF file,
``torch``, ``torch-harmonics``, and a checked-out spharmgrid source tree. Run
this command from the ``sht_bench`` repository root with that checkout on
``PYTHONPATH``. Example:

    PYTHONPATH=../spharmgrid/src python benchmarks/spharmgrid/torch_backend_era5.py \
        --dtype float32 --output-json /tmp/era5-torch-benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import spharmgrid as sg
import spharmgrid.torch as sgt
import spharmgrid.torch.nn as sgnn
import torch
import torch_harmonics
import xarray as xr
from numpy.typing import NDArray

Operation = Literal["filter", "regrid", "gradient"]
Backend = Literal[
    "ducc_cpu_1t",
    "ducc_cpu_16t",
    "torch_cpu_reusable",
    "torch_cuda_reusable",
]
TimingMode = Literal["compute", "pageable_h2d_compute"]
FloatArray = NDArray[Any]
IndexArray = NDArray[np.intp]
TorchOutput = torch.Tensor | tuple[torch.Tensor, torch.Tensor]
XarrayOutput = xr.DataArray | tuple[xr.DataArray, xr.DataArray]


def backend_label(backend: Backend, threads: int) -> str:
    """Return the report label for an internal backend identifier."""
    if backend in {"ducc_cpu_1t", "ducc_cpu_16t"}:
        return f"DUCC CPU {threads}T"
    return {
        "torch_cpu_reusable": "Torch CPU reusable",
        "torch_cuda_reusable": "Torch CUDA reusable",
    }[backend]


@dataclass(frozen=True, slots=True)
class Workload:
    """One timestamp selection and its bounded execution batch size."""

    name: str
    indices: IndexArray
    batch_size: int


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    """A host-resident batch ready for a backend call."""

    xarray: xr.DataArray
    values: FloatArray
    prepare_seconds: float


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""
    default_dataset = (
        Path(__file__).resolve().parents[3]
        / "PyStormTracker-Reference-Data"
        / "era5-2024"
        / "ERA5_mslp_6hr_2024_DET.nc"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--variable", default="msl")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--one-frame-repetitions", type=int, default=5)
    parser.add_argument("--month-repetitions", type=int, default=5)
    parser.add_argument("--year-repetitions", type=int, default=1)
    parser.add_argument("--ducc-threads", type=int, default=16)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="probe the requested CUDA batch sizes and skip the full benchmark",
    )
    parser.add_argument(
        "--probe-batches",
        default="32,64,128,256,384,512,576,640",
        help="comma-separated CUDA batch sizes used by --probe-only",
    )
    return parser.parse_args()


def dtype_config(name: str) -> tuple[torch.dtype, np.dtype]:
    """Return the matching Torch and NumPy dtypes for a benchmark run."""
    if name == "float32":
        return torch.float32, np.dtype(np.float32)
    if name == "float64":
        return torch.float64, np.dtype(np.float64)
    raise ValueError(f"unsupported benchmark dtype: {name}")


def configure_threads(threads: int) -> None:
    """Apply the explicit one-process CPU thread policy."""
    if threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def cpu_model() -> str:
    """Return the Linux CPU model or a portable fallback."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def device_metadata() -> dict[str, Any]:
    """Return Torch and accelerator metadata for the report."""
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_harmonics": torch_harmonics.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        metadata.update(
            {
                "gpu": properties.name,
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "gpu_total_memory_bytes": properties.total_memory,
            }
        )
    return metadata


def open_dataset(path: Path) -> xr.Dataset:
    """Open the classic ERA5 NetCDF without loading the data variable."""
    return xr.open_dataset(
        path,
        engine="scipy",
        decode_times=True,
        mask_and_scale=True,
    )


def workloads(ds: xr.Dataset, batch_size: int) -> list[Workload]:
    """Build the three required workloads from decoded timestamps."""
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    time_values = ds.time.values
    years = np.asarray(ds.time.dt.year)
    months = np.asarray(ds.time.dt.month)
    year_mask = years == 2024
    month_mask = year_mask & (months == 1)
    year_indices = np.flatnonzero(year_mask).astype(np.intp)
    month_indices = np.flatnonzero(month_mask).astype(np.intp)
    if year_indices.size == 0 or month_indices.size == 0:
        raise ValueError("dataset does not contain the required 2024 workloads")
    if time_values[0] != time_values[year_indices[0]]:
        raise ValueError("2024 does not begin at the first dataset timestamp")
    return [
        Workload("one frame", np.array([year_indices[0]], dtype=np.intp), 1),
        Workload("January 2024", month_indices, batch_size),
        Workload("full 2024", year_indices, batch_size),
    ]


def workload_repetitions(
    workload: Workload,
    backend: Backend,
    one_frame_repetitions: int,
    month_repetitions: int,
    year_repetitions: int,
) -> int:
    """Choose complete-workload repetitions for one benchmark case."""
    if workload.name == "one frame":
        return one_frame_repetitions
    if workload.name == "January 2024" and backend == "torch_cuda_reusable":
        return month_repetitions
    if workload.name == "full 2024":
        return year_repetitions
    return 1


def prepare_batch(
    field: xr.DataArray,
    indices: IndexArray,
    operation: Operation,
    target_grid: sg.Grid,
    threads: int,
    dtype: np.dtype,
) -> PreparedBatch:
    """Load one host batch and prepare the working grid when needed."""
    started = time.perf_counter()
    batch = field.isel(time=indices).load()
    if batch.dtype != dtype:
        batch = batch.astype(dtype)
        batch.load()
    if operation == "gradient":
        batch = sg.regrid(
            batch,
            target_grid,
            "T42",
            sht_threads=threads,
        ).astype(dtype)
        batch.load()
    values = np.asarray(batch.values, dtype=dtype)
    if values.ndim != 3:
        raise ValueError(f"expected a time batch, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("benchmark input contains non-finite values")
    return PreparedBatch(batch, values, time.perf_counter() - started)


def call_ducc(
    operation: Operation,
    field: xr.DataArray,
    source_grid: sg.Grid,
    target_grid: sg.Grid,
    threads: int,
) -> XarrayOutput:
    """Call the Xarray/DUCC API for one prepared batch."""
    del source_grid
    if operation == "filter":
        return sg.filter(field, "T42", sht_threads=threads)
    if operation == "regrid":
        return sg.regrid(field, target_grid, "T42", sht_threads=threads)
    result = sg.gradient(field, sht_threads=threads)
    return result["gradient_eastward"], result["gradient_northward"]


def call_torch_module(
    operation: Operation,
    module: torch.nn.Module,
    values: torch.Tensor,
    target_grid: sg.Grid,
) -> TorchOutput:
    """Call one reusable Torch module."""
    del target_grid
    if operation == "gradient":
        operators = cast(sgnn.SHTOperators, module)
        return operators.gradient(values)
    return cast(torch.Tensor, module(values))


def assert_cuda_io(
    input_tensor: torch.Tensor,
    output: TorchOutput,
    dtype: torch.dtype,
) -> None:
    """Assert CUDA placement and dtype preservation for one Torch call."""
    assert input_tensor.is_cuda
    assert input_tensor.dtype == dtype
    outputs = output if isinstance(output, tuple) else (output,)
    assert all(value.is_cuda for value in outputs)
    assert all(value.device == input_tensor.device for value in outputs)
    assert all(value.dtype == dtype for value in outputs)


def call_torch_functional(
    operation: Operation,
    values: torch.Tensor,
    source_grid: sg.Grid,
    target_grid: sg.Grid,
) -> TorchOutput:
    """Call one functional Torch API operation."""
    if operation == "filter":
        return sgt.filter(values, "T42", grid=source_grid)
    if operation == "regrid":
        return sgt.regrid(values, target_grid, "T42", source_grid=source_grid)
    return sgt.gradient(values, grid=target_grid)


def xarray_arrays(output: XarrayOutput) -> list[FloatArray]:
    """Materialize Xarray outputs without changing their dtype."""
    outputs = output if isinstance(output, tuple) else (output,)
    return [np.asarray(value.values) for value in outputs]


def torch_arrays(output: TorchOutput) -> list[FloatArray]:
    """Materialize Torch outputs on the host without changing their dtype."""
    outputs = output if isinstance(output, tuple) else (output,)
    return [value.detach().cpu().numpy() for value in outputs]


def consume_xarray(output: XarrayOutput) -> float:
    """Consume an eager Xarray output without retaining it."""
    return float(sum(float(array.reshape(-1)[0]) for array in xarray_arrays(output)))


def consume_torch(output: TorchOutput) -> float:
    """Consume a Torch output after synchronized execution."""
    outputs = output if isinstance(output, tuple) else (output,)
    return float(
        sum(float(value.detach().reshape(-1)[0].cpu().item()) for value in outputs)
    )


def call_timed_ducc(
    operation: Operation,
    batch: PreparedBatch,
    source_grid: sg.Grid,
    target_grid: sg.Grid,
    threads: int,
) -> tuple[float, float]:
    """Time one synchronized CPU DUCC call after host preparation."""
    started = time.perf_counter()
    output = call_ducc(operation, batch.xarray, source_grid, target_grid, threads)
    elapsed = time.perf_counter() - started
    checksum = consume_xarray(output)
    return elapsed, checksum


def call_timed_torch(
    operation: Operation,
    module: torch.nn.Module,
    values: FloatArray,
    target_grid: sg.Grid,
    device: torch.device,
    mode: TimingMode,
    dtype: torch.dtype,
) -> tuple[float, float]:
    """Time one reusable Torch call with synchronized CUDA boundaries."""
    host_tensor = torch.from_numpy(values)
    if device.type == "cuda" and mode == "compute":
        device_tensor = host_tensor.to(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = call_torch_module(operation, module, device_tensor, target_grid)
        assert_cuda_io(device_tensor, output, dtype)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    elif device.type == "cuda" and mode == "pageable_h2d_compute":
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        device_tensor = host_tensor.to(device)
        output = call_torch_module(operation, module, device_tensor, target_grid)
        assert_cuda_io(device_tensor, output, dtype)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    else:
        started = time.perf_counter()
        output = call_torch_module(operation, module, host_tensor, target_grid)
        elapsed = time.perf_counter() - started
    checksum = consume_torch(output)
    del output
    return elapsed, checksum


def warm_up(
    backend: Backend,
    operation: Operation,
    module: torch.nn.Module | None,
    batch: PreparedBatch,
    source_grid: sg.Grid,
    target_grid: sg.Grid,
    threads: int,
    dtype: torch.dtype,
) -> None:
    """Warm one backend on a representative first batch."""
    with torch.inference_mode():
        if backend in {"ducc_cpu_1t", "ducc_cpu_16t"}:
            output = call_ducc(
                operation,
                batch.xarray,
                source_grid,
                target_grid,
                threads,
            )
            consume_xarray(output)
            return
        if module is None:
            raise ValueError(f"missing module for {backend}")
        device = torch.device("cuda" if backend == "torch_cuda_reusable" else "cpu")
        values = torch.from_numpy(batch.values)
        if device.type == "cuda":
            values = values.to(device)
            output = call_torch_module(operation, module, values, target_grid)
            assert_cuda_io(values, output, dtype)
            torch.cuda.synchronize(device)
            consume_torch(output)
        else:
            output = call_torch_module(operation, module, values, target_grid)
            consume_torch(output)


def peak_memory_mib() -> tuple[float, float]:
    """Return current CUDA peak allocated and reserved memory in MiB."""
    factor = float(2**20)
    return (
        float(torch.cuda.max_memory_allocated()) / factor,
        float(torch.cuda.max_memory_reserved()) / factor,
    )


def run_case_backend(
    field: xr.DataArray,
    workload: Workload,
    operation: Operation,
    backend: Backend,
    module: torch.nn.Module | None,
    source_grid: sg.Grid,
    target_grid: sg.Grid,
    threads: int,
    repetitions: int,
    dtype: torch.dtype,
    numpy_dtype: np.dtype,
    mode: TimingMode = "compute",
) -> dict[str, Any]:
    """Run complete streamed workload repetitions for one backend."""
    if repetitions < 1:
        raise ValueError("workload repetitions must be positive")
    device = torch.device("cuda" if backend == "torch_cuda_reusable" else "cpu")
    if backend == "torch_cuda_reusable":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    repetition_totals: list[float] = []
    repetition_stream_seconds: list[float] = []
    repetition_prepare_seconds: list[float] = []
    all_checksums: list[float] = []
    stream_started = time.perf_counter()
    if workload.indices.size == 1:
        batches: Sequence[IndexArray] = [workload.indices]
    else:
        batches = [
            workload.indices[start : start + workload.batch_size]
            for start in range(0, workload.indices.size, workload.batch_size)
        ]
    with torch.inference_mode():
        for _ in range(repetitions):
            repetition_started = time.perf_counter()
            measured_seconds: list[float] = []
            checksums: list[float] = []
            prepare_seconds = 0.0
            for indices in batches:
                batch = prepare_batch(
                    field,
                    indices,
                    operation,
                    target_grid,
                    threads,
                    numpy_dtype,
                )
                prepare_seconds += batch.prepare_seconds
                if backend in {"ducc_cpu_1t", "ducc_cpu_16t"}:
                    elapsed, checksum = call_timed_ducc(
                        operation,
                        batch,
                        source_grid,
                        target_grid,
                        threads,
                    )
                else:
                    if module is None:
                        raise ValueError(f"missing module for {backend}")
                    elapsed, checksum = call_timed_torch(
                        operation,
                        module,
                        batch.values,
                        target_grid,
                        device,
                        mode,
                        dtype,
                    )
                measured_seconds.append(elapsed)
                checksums.append(checksum)
            repetition_totals.append(float(sum(measured_seconds)))
            repetition_prepare_seconds.append(prepare_seconds)
            repetition_stream_seconds.append(time.perf_counter() - repetition_started)
            all_checksums.extend(checksums)
    stream_seconds = time.perf_counter() - stream_started
    total_seconds = float(np.median(repetition_totals))
    batch_count = len(batches)
    last_batch_size = int(batches[-1].size)
    result: dict[str, Any] = {
        "operation": operation,
        "workload": workload.name,
        "frames": int(workload.indices.size),
        "batch_size": workload.batch_size,
        "last_batch_size": last_batch_size,
        "batches": batch_count,
        "backend": backend,
        "backend_label": backend_label(backend, threads),
        "mode": mode,
        "repetitions": repetitions,
        "total_seconds": total_seconds,
        "milliseconds_per_frame": total_seconds * 1000.0 / workload.indices.size,
        "frames_per_second": workload.indices.size / total_seconds,
        "stream_wall_seconds": stream_seconds,
        "median_stream_wall_seconds": float(np.median(repetition_stream_seconds)),
        "host_prepare_seconds": float(sum(repetition_prepare_seconds)),
        "median_host_prepare_seconds": float(np.median(repetition_prepare_seconds)),
        "checksum": float(sum(all_checksums) / len(all_checksums)),
        "repetition_total_seconds": repetition_totals,
        "minimum_total_seconds": min(repetition_totals),
        "maximum_total_seconds": max(repetition_totals),
    }
    if backend == "torch_cuda_reusable":
        allocated, reserved = peak_memory_mib()
        result["peak_gpu_allocated_mib"] = allocated
        result["peak_gpu_reserved_mib"] = reserved
    return result


def timed_functional(
    operation: Operation,
    values: FloatArray,
    source_grid: sg.Grid,
    target_grid: sg.Grid,
    repetitions: int,
    device: torch.device,
    mode: TimingMode,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Measure one-frame functional API latency including state construction."""
    host_tensor = torch.from_numpy(values)
    if device.type == "cuda" and mode == "compute":
        device_tensor = host_tensor.to(device)
        with torch.inference_mode():
            warmup_output = call_torch_functional(
                operation, device_tensor, source_grid, target_grid
            )
            assert_cuda_io(device_tensor, warmup_output, dtype)
            torch.cuda.synchronize(device)
            del warmup_output
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            samples: list[float] = []
            for _ in range(repetitions):
                started = time.perf_counter()
                output = call_torch_functional(
                    operation,
                    device_tensor,
                    source_grid,
                    target_grid,
                )
                assert_cuda_io(device_tensor, output, dtype)
                torch.cuda.synchronize(device)
                samples.append(time.perf_counter() - started)
                consume_torch(output)
    elif device.type == "cuda" and mode == "pageable_h2d_compute":
        with torch.inference_mode():
            warmup_tensor = host_tensor.to(device)
            warmup_output = call_torch_functional(
                operation,
                warmup_tensor,
                source_grid,
                target_grid,
            )
            assert_cuda_io(warmup_tensor, warmup_output, dtype)
            torch.cuda.synchronize(device)
            del warmup_tensor
            del warmup_output
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for _ in range(repetitions):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                device_tensor = host_tensor.to(device)
                output = call_torch_functional(
                    operation,
                    device_tensor,
                    source_grid,
                    target_grid,
                )
                assert_cuda_io(device_tensor, output, dtype)
                torch.cuda.synchronize(device)
                samples.append(time.perf_counter() - started)
                consume_torch(output)
                del device_tensor
    else:
        with torch.inference_mode():
            _ = call_torch_functional(operation, host_tensor, source_grid, target_grid)
            samples = []
            for _ in range(repetitions):
                started = time.perf_counter()
                output = call_torch_functional(
                    operation,
                    host_tensor,
                    source_grid,
                    target_grid,
                )
                samples.append(time.perf_counter() - started)
                consume_torch(output)
    total_seconds = float(np.median(samples))
    result: dict[str, Any] = {
        "operation": operation,
        "workload": "one frame",
        "frames": 1,
        "batch_size": 1,
        "last_batch_size": 1,
        "batches": 1,
        "backend": "torch_cuda_functional"
        if device.type == "cuda"
        else "torch_cpu_functional",
        "mode": mode,
        "repetitions": repetitions,
        "total_seconds": total_seconds,
        "milliseconds_per_frame": total_seconds * 1000.0,
        "frames_per_second": 1.0 / total_seconds,
    }
    if device.type == "cuda":
        allocated, reserved = peak_memory_mib()
        result["peak_gpu_allocated_mib"] = allocated
        result["peak_gpu_reserved_mib"] = reserved
    return result


def setup_module(
    operation: Operation,
    source_grid: sg.Grid,
    target_grid: sg.Grid,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, float]:
    """Construct one reusable module once and move it to its device."""
    started = time.perf_counter()
    if operation == "filter":
        module: torch.nn.Module = sgnn.SHTFilter(source_grid, "T42")
    elif operation == "regrid":
        module = sgnn.SHTRegrid(source_grid, target_grid, "T42")
    else:
        module = sgnn.SHTOperators(target_grid)
    module = module.to(device=device, dtype=dtype)
    return module, time.perf_counter() - started


def comparison_stats(reference: FloatArray, actual: FloatArray) -> dict[str, float]:
    """Return absolute, RMS, and relative RMS error statistics."""
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    reference64 = np.asarray(reference, dtype=np.float64)
    rms = float(np.sqrt(np.mean(difference**2)))
    reference_rms = float(np.sqrt(np.mean(reference64**2)))
    return {
        "max_absolute_error": float(np.max(np.abs(difference))),
        "rms_error": rms,
        "relative_rms_error": rms / reference_rms if reference_rms else float("nan"),
    }


def validate_operation(
    operation: Operation,
    frame: PreparedBatch,
    source_grid: sg.Grid,
    target_grid: sg.Grid,
    threads: int,
    cpu_module: torch.nn.Module,
    cuda_module: torch.nn.Module,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    """Compare DUCC, Torch CPU, and Torch CUDA on one representative frame."""
    with torch.inference_mode():
        reference = call_ducc(
            operation,
            frame.xarray,
            source_grid,
            target_grid,
            threads,
        )
        cpu_output = call_torch_module(
            operation,
            cpu_module,
            torch.from_numpy(frame.values),
            target_grid,
        )
        cpu_outputs = cpu_output if isinstance(cpu_output, tuple) else (cpu_output,)
        assert all(value.dtype == dtype for value in cpu_outputs)
        cuda_values = torch.from_numpy(frame.values).to("cuda")
        cuda_output = call_torch_module(
            operation,
            cuda_module,
            cuda_values,
            target_grid,
        )
        assert_cuda_io(cuda_values, cuda_output, dtype)
        torch.cuda.synchronize()
    reference_arrays = xarray_arrays(reference)
    cpu_arrays = torch_arrays(cpu_output)
    cuda_arrays = torch_arrays(cuda_output)
    labels = (
        ("field",)
        if operation != "gradient"
        else ("eastward gradient", "northward gradient")
    )
    results: list[dict[str, Any]] = []
    for label, ref, cpu, cuda in zip(
        labels,
        reference_arrays,
        cpu_arrays,
        cuda_arrays,
        strict=True,
    ):
        results.extend(
            [
                {
                    "operation": operation,
                    "output": label,
                    "comparison": "DUCC CPU vs Torch CPU",
                    **comparison_stats(ref, cpu),
                },
                {
                    "operation": operation,
                    "output": label,
                    "comparison": "DUCC CPU vs Torch CUDA",
                    **comparison_stats(ref, cuda),
                },
                {
                    "operation": operation,
                    "output": label,
                    "comparison": "Torch CPU vs Torch CUDA",
                    **comparison_stats(cpu, cuda),
                },
            ]
        )
    return results


def validate_ducc_threads(
    operation: Operation,
    frame: PreparedBatch,
    source_grid: sg.Grid,
    target_grid: sg.Grid,
    one_thread: int,
    many_threads: int,
) -> list[dict[str, Any]]:
    """Compare DUCC outputs from the controlled and multithreaded runs."""
    one_thread_output = call_ducc(
        operation,
        frame.xarray,
        source_grid,
        target_grid,
        one_thread,
    )
    many_thread_output = call_ducc(
        operation,
        frame.xarray,
        source_grid,
        target_grid,
        many_threads,
    )
    labels = (
        ("field",)
        if operation != "gradient"
        else ("eastward gradient", "northward gradient")
    )
    results: list[dict[str, Any]] = []
    for label, one, many in zip(
        labels,
        xarray_arrays(one_thread_output),
        xarray_arrays(many_thread_output),
        strict=True,
    ):
        results.append(
            {
                "operation": operation,
                "output": label,
                "comparison": (f"DUCC CPU {one_thread}T vs DUCC CPU {many_threads}T"),
                **comparison_stats(one, many),
            }
        )
    return results


def probe_batch_sizes(
    field: xr.DataArray,
    source_grid: sg.Grid,
    batch_sizes: Sequence[int],
    threads: int,
    dtype: torch.dtype,
    numpy_dtype: np.dtype,
) -> list[dict[str, Any]]:
    """Measure CUDA memory for candidate filter batch sizes."""
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("probe batch sizes must be positive")
    maximum = max(batch_sizes)
    indices = np.arange(maximum, dtype=np.intp)
    batch = prepare_batch(
        field,
        indices,
        "filter",
        source_grid,
        threads,
        numpy_dtype,
    )
    results: list[dict[str, Any]] = []
    for size in batch_sizes:
        module: torch.nn.Module | None = None
        values: torch.Tensor | None = None
        output: TorchOutput | None = None
        try:
            module, setup_seconds = setup_module(
                "filter",
                source_grid,
                source_grid,
                torch.device("cuda"),
                dtype,
            )
            values = torch.from_numpy(batch.values[:size]).to("cuda", dtype=dtype)
            with torch.inference_mode():
                warmup_output = call_torch_module("filter", module, values, source_grid)
                assert_cuda_io(values, warmup_output, dtype)
                torch.cuda.synchronize()
                del warmup_output
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                output = call_torch_module("filter", module, values, source_grid)
                assert_cuda_io(values, output, dtype)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                assert output is not None
                consume_torch(output)
            allocated, reserved = peak_memory_mib()
            results.append(
                {
                    "batch_size": size,
                    "status": "ok",
                    "seconds": elapsed,
                    "milliseconds_per_frame": elapsed * 1000.0 / size,
                    "peak_gpu_allocated_mib": allocated,
                    "peak_gpu_reserved_mib": reserved,
                    "module_setup_seconds": setup_seconds,
                }
            )
        except RuntimeError as error:
            is_oom = "out of memory" in str(error).lower()
            results.append(
                {
                    "batch_size": size,
                    "status": "oom" if is_oom else "error",
                    "error": str(error),
                }
            )
        finally:
            del output
            del values, module
            torch.cuda.empty_cache()
    return results


def parse_probe_batches(value: str) -> list[int]:
    """Parse comma-separated positive batch sizes."""
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    """Write machine-readable results when requested."""
    if path is not None:
        path.write_text(
            json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
        )


def main() -> None:
    """Run the selected real-data benchmark."""
    args = parse_args()
    torch_dtype, numpy_dtype = dtype_config(args.dtype)
    if args.ducc_threads < 1:
        raise ValueError("--ducc-threads must be positive")
    if any(
        repetitions < 1
        for repetitions in (
            args.one_frame_repetitions,
            args.month_repetitions,
            args.year_repetitions,
        )
    ):
        raise ValueError("repetition counts must be positive")
    configure_threads(args.threads)
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    if not torch.cuda.is_available():
        raise RuntimeError("the real-data benchmark requires an available CUDA device")
    metadata = device_metadata()
    tensor_contract = {
        "requested_dtype": args.dtype,
        "torch_input_dtype": str(torch_dtype).replace("torch.", "", 1),
        "torch_output_dtype": str(torch_dtype).replace("torch.", "", 1),
        "torch_cpu_device": "cpu",
        "torch_cuda_device": "cuda:0",
        "vector_outputs_checked": True,
    }
    with open_dataset(args.dataset) as ds:
        if args.variable not in ds.data_vars:
            raise ValueError(
                f"{args.variable!r} is not a data variable in {args.dataset}"
            )
        field = ds[args.variable]
        source_grid = sg.detect_grid(field.isel(time=0))
        target_grid = sg.gaussian_grid(
            43,
            86,
            latitude_order="descending",
            lon0=0.0,
        )
        if args.probe_only:
            payload = {
                "environment": metadata,
                "dataset": str(args.dataset),
                "variable": args.variable,
                "dtype": args.dtype,
                "tensor_contract": tensor_contract,
                "source_grid": {
                    "kind": source_grid.kind,
                    "nlat": source_grid.nlat,
                    "nlon": source_grid.nlon,
                },
                "probe": probe_batch_sizes(
                    field,
                    source_grid,
                    parse_probe_batches(args.probe_batches),
                    args.threads,
                    torch_dtype,
                    numpy_dtype,
                ),
            }
            write_json(args.output_json, payload)
            print(json.dumps(payload, indent=2, allow_nan=True))
            return

        workload_list = workloads(ds, args.batch_size)
        module_setup: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        validation: list[dict[str, Any]] = []
        for operation in ("filter", "regrid", "gradient"):
            cpu_module, cpu_setup = setup_module(
                operation,
                source_grid,
                target_grid,
                torch.device("cpu"),
                torch_dtype,
            )
            cuda_module, cuda_setup = setup_module(
                operation,
                source_grid,
                target_grid,
                torch.device("cuda"),
                torch_dtype,
            )
            module_setup.extend(
                [
                    {
                        "operation": operation,
                        "backend": "torch_cpu_reusable",
                        "setup_seconds": cpu_setup,
                    },
                    {
                        "operation": operation,
                        "backend": "torch_cuda_reusable",
                        "setup_seconds": cuda_setup,
                    },
                ]
            )
            representative = prepare_batch(
                field,
                np.array([workload_list[0].indices[0]], dtype=np.intp),
                operation,
                target_grid,
                args.threads,
                numpy_dtype,
            )
            validation.extend(
                validate_operation(
                    operation,
                    representative,
                    source_grid,
                    target_grid,
                    args.threads,
                    cpu_module,
                    cuda_module,
                    torch_dtype,
                )
            )
            validation.extend(
                validate_ducc_threads(
                    operation,
                    representative,
                    source_grid,
                    target_grid,
                    args.threads,
                    args.ducc_threads,
                )
            )
            print(f"validated {operation}", flush=True)
            for workload in workload_list:
                operation_typed = operation
                first_batch = prepare_batch(
                    field,
                    workload.indices[: min(workload.batch_size, workload.indices.size)],
                    operation_typed,
                    target_grid,
                    args.threads,
                    numpy_dtype,
                )
                backend_cases: tuple[
                    tuple[Backend, torch.nn.Module | None, int], ...
                ] = (
                    ("ducc_cpu_1t", None, args.threads),
                    ("ducc_cpu_16t", None, args.ducc_threads),
                    ("torch_cpu_reusable", cpu_module, args.threads),
                    ("torch_cuda_reusable", cuda_module, args.threads),
                )
                for backend, module, sht_threads in backend_cases:
                    warm_up(
                        backend,
                        operation_typed,
                        module,
                        first_batch,
                        source_grid,
                        target_grid,
                        sht_threads,
                        torch_dtype,
                    )
                    backend_typed = backend
                    repetitions = workload_repetitions(
                        workload,
                        backend_typed,
                        args.one_frame_repetitions,
                        args.month_repetitions,
                        args.year_repetitions,
                    )
                    result = run_case_backend(
                        field,
                        workload,
                        operation_typed,
                        backend_typed,
                        module,
                        source_grid,
                        target_grid,
                        sht_threads,
                        repetitions,
                        torch_dtype,
                        numpy_dtype,
                    )
                    results.append(result)
                    print(
                        f"{operation} / {workload.name} / "
                        f"{result['backend_label']}: "
                        f"{result['total_seconds']:.6f}s compute, "
                        f"{result['stream_wall_seconds']:.3f}s stream",
                        flush=True,
                    )
                transfer_result = run_case_backend(
                    field,
                    workload,
                    operation_typed,
                    "torch_cuda_reusable",
                    cuda_module,
                    source_grid,
                    target_grid,
                    args.threads,
                    workload_repetitions(
                        workload,
                        "torch_cuda_reusable",
                        args.one_frame_repetitions,
                        args.month_repetitions,
                        args.year_repetitions,
                    ),
                    torch_dtype,
                    numpy_dtype,
                    mode="pageable_h2d_compute",
                )
                results.append(transfer_result)
                print(
                    f"{operation} / {workload.name} / CUDA pageable H2D+compute: "
                    f"{transfer_result['total_seconds']:.6f}s",
                    flush=True,
                )
                if workload.name == "one frame":
                    one_frame = prepare_batch(
                        field,
                        workload.indices,
                        operation_typed,
                        target_grid,
                        args.threads,
                        numpy_dtype,
                    )
                    results.append(
                        timed_functional(
                            operation_typed,
                            one_frame.values,
                            source_grid,
                            target_grid,
                            args.one_frame_repetitions,
                            torch.device("cpu"),
                            "compute",
                            torch_dtype,
                        )
                    )
                    results.append(
                        timed_functional(
                            operation_typed,
                            one_frame.values,
                            source_grid,
                            target_grid,
                            args.one_frame_repetitions,
                            torch.device("cuda"),
                            "compute",
                            torch_dtype,
                        )
                    )
            del cpu_module, cuda_module

        times = np.asarray(ds.time.values)
        payload = {
            "environment": metadata,
            "dtype": args.dtype,
            "tensor_contract": tensor_contract,
            "dataset": {
                "path": str(args.dataset),
                "variable": args.variable,
                "dimensions": {str(name): int(size) for name, size in ds.sizes.items()},
                "time_first": str(times[0]),
                "time_last": str(times[-1]),
                "time_step_hours": 6,
                "source_grid": {
                    "kind": source_grid.kind,
                    "nlat": source_grid.nlat,
                    "nlon": source_grid.nlon,
                    "latitude_order": (
                        "descending"
                        if source_grid.latitude[0] > source_grid.latitude[-1]
                        else "ascending"
                    ),
                    "longitude_first": float(source_grid.longitude[0]),
                    "longitude_last": float(source_grid.longitude[-1]),
                },
                "target_grid": {
                    "kind": target_grid.kind,
                    "nlat": target_grid.nlat,
                    "nlon": target_grid.nlon,
                    "truncation": "T42",
                },
            },
            "thread_policy": {
                "ducc_sht_threads_1t": args.threads,
                "ducc_sht_threads_16t": args.ducc_threads,
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": 1,
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            },
            "repetition_policy": {
                "one_frame": args.one_frame_repetitions,
                "january_cuda": args.month_repetitions,
                "full_year": args.year_repetitions,
            },
            "workloads": [
                {
                    "name": workload.name,
                    "frames": int(workload.indices.size),
                    "first_timestamp": str(times[workload.indices[0]]),
                    "last_timestamp": str(times[workload.indices[-1]]),
                    "batch_size": workload.batch_size,
                }
                for workload in workload_list
            ],
            "module_setup": module_setup,
            "numerical_validation": validation,
            "results": results,
        }
    write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
