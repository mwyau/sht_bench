"""Compare dense equiangular analysis with the Appendix-A folded-ring path.

The dense classes are loaded directly from the immutable torch-harmonics
commit named by ``--base``. The optimized classes are imported from the source
checkout named by ``--source``. This keeps the numerical reference independent
of the optimized implementation and records the two source revisions in the
JSON result.

Run this script with a Python environment containing PyTorch and put this
checkout's ``src`` directory on ``PYTHONPATH`` when the result should be kept
with the sibling benchmark project, for example::

    PYTHONPATH=/home/albert/sht_bench/src \
      /home/albert/torch-harmonics/.venv/bin/python \
      scripts/torch_folding_benchmark.py --device cpu --output results/torch-folding
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

BASE_SHA = "3278fb669483d04537aa60c87fb0754d989e2e70"
DEFAULT_CASES = ((17, 32, 16), (73, 144, 72), (129, 256, 128), (257, 512, 256))
NORMS = ("ortho", "four-pi", "schmidt", "unnorm")


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
    sys.path.insert(0, str(source.resolve()))
    import torch_harmonics

    return torch_harmonics


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


def _timed(
    torch: Any, device: Any, function: Callable[[], Any], warmup: int, repeat: int
) -> dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        _synchronize(torch, device)
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            function()
            _synchronize(torch, device)
            samples.append(time.perf_counter() - start)
    return {
        "samples_s": samples,
        "median_s": sorted(samples)[len(samples) // 2],
        "minimum_s": min(samples),
    }


def _timed_backward(
    torch: Any, device: Any, module: Any, sample: Any, warmup: int, repeat: int
) -> dict[str, Any]:
    def function() -> None:
        value = sample.detach().requires_grad_(True)
        output = module(value)
        output.abs().square().mean().backward()

    for _ in range(warmup):
        function()
    _synchronize(torch, device)
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        _synchronize(torch, device)
        samples.append(time.perf_counter() - start)
    return {
        "samples_s": samples,
        "median_s": sorted(samples)[len(samples) // 2],
        "minimum_s": min(samples),
    }


def _peak_forward(torch: Any, device: Any, function: Callable[[], Any]) -> int | None:
    if device.type != "cuda":
        return None
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    function()
    _synchronize(torch, device)
    return torch.cuda.max_memory_allocated(device) - baseline


def _peak_backward(torch: Any, device: Any, module: Any, sample: Any) -> int | None:
    if device.type != "cuda":
        return None
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    value = sample.detach().requires_grad_(True)
    module(value).abs().square().mean().backward()
    _synchronize(torch, device)
    return torch.cuda.max_memory_allocated(device) - baseline


def _hoisted_projection(
    torch: Any, optimized: Any, module: Any, vector: bool
) -> tuple[Any, float]:
    """Build the construction-time equivalent of the runtime folded operator."""

    from torch_harmonics.sht import _fold_resampled_latitude

    phase_buffer = module._latitude_shift_phase
    phase = torch.complex(phase_buffer[0], phase_buffer[1])
    start = time.perf_counter()
    with torch.no_grad():
        projection = module.weights
        projection = projection.transpose(-2, -3)
        projection = _fold_resampled_latitude(
            projection,
            module._parity_signs,
            module._quadrature_weights,
            module._midpoint_weights,
            phase,
        )
        projection = projection.transpose(-2, -3).conj().contiguous()
    _synchronize(torch, module.weights.device)
    return projection, time.perf_counter() - start


def _hoisted_forward(
    torch: Any, optimized: Any, sample: Any, projection: Any, vector: bool
) -> Any:
    from torch_harmonics.fft import rfft

    transformed = rfft(
        sample, nmodes=projection.shape[-2], dim=-1, norm="forward"
    ).transpose(-1, -2)
    if not vector:
        return torch.einsum("...mk,mlk->...lm", transformed, projection)
    component_zero = torch.einsum(
        "...mk,mlk->...lm", transformed[..., 0, :, :], projection[0]
    )
    component_one = torch.einsum(
        "...mk,mlk->...lm", transformed[..., 1, :, :], projection[1]
    )
    spheroidal = component_zero + 1j * component_one
    component_zero = torch.einsum(
        "...mk,mlk->...lm", transformed[..., 0, :, :], projection[1]
    )
    component_one = torch.einsum(
        "...mk,mlk->...lm", transformed[..., 1, :, :], projection[0]
    )
    toroidal = 1j * component_zero - component_one
    return torch.stack((spheroidal, toroidal), dim=-3)


def _construct(
    torch: Any,
    device: Any,
    cls: Any,
    nlat: int,
    nlon: int,
    lmax: int,
    vector: bool,
    dtype: Any,
) -> tuple[Any, float, int, int, int | None]:
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    module = cls(nlat, nlon, lmax=lmax, mmax=lmax)
    module = module.to(device=device, dtype=dtype).eval()
    _synchronize(torch, device)
    setup_s = time.perf_counter() - start
    peak = (
        torch.cuda.max_memory_allocated(device) - baseline
        if device.type == "cuda"
        else None
    )
    return module, setup_s, _buffer_bytes(module), _projection_bytes(module), peak


def _benchmark_case(
    torch: Any,
    optimized: Any,
    dense: dict[str, Any],
    device: Any,
    dtype: Any,
    dtype_name: str,
    nlat: int,
    nlon: int,
    lmax: int,
    warmup: int,
    repeat: int,
    compile_graph: bool,
    transforms: tuple[str, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    classes = tuple(
        item
        for item in (
            ("scalar", optimized.RealSHT, dense["RealSHT"]),
            ("vector", optimized.RealVectorSHT, dense["RealVectorSHT"]),
        )
        if item[0] in transforms
    )
    for transform, optimized_class, dense_class in classes:
        optimized_module, opt_setup, opt_buffers, opt_projection, opt_peak = _construct(
            torch,
            device,
            optimized_class,
            nlat,
            nlon,
            lmax,
            transform == "vector",
            dtype,
        )
        dense_module, dense_setup, dense_buffers, dense_projection, dense_peak = (
            _construct(
                torch,
                device,
                dense_class,
                nlat,
                nlon,
                lmax,
                transform == "vector",
                dtype,
            )
        )
        sample_shape = (1, nlat, nlon) if transform == "scalar" else (1, 2, nlat, nlon)
        sample = torch.randn(*sample_shape, device=device, dtype=dtype)
        optimized_forward = _timed(
            torch,
            device,
            lambda module=optimized_module, value=sample: module(value),
            warmup,
            repeat,
        )
        dense_forward = _timed(
            torch,
            device,
            lambda module=dense_module, value=sample: module(value),
            warmup,
            repeat,
        )
        optimized_backward = _timed_backward(
            torch, device, optimized_module, sample, warmup, repeat
        )
        dense_backward = _timed_backward(
            torch, device, dense_module, sample, warmup, repeat
        )
        optimized_peak_forward = _peak_forward(
            torch,
            device,
            lambda module=optimized_module, value=sample: module(value),
        )
        dense_peak_forward = _peak_forward(
            torch,
            device,
            lambda module=dense_module, value=sample: module(value),
        )
        optimized_peak_backward = _peak_backward(
            torch, device, optimized_module, sample
        )
        dense_peak_backward = _peak_backward(torch, device, dense_module, sample)
        hoisted_projection, hoisted_setup = _hoisted_projection(
            torch, optimized, optimized_module, transform == "vector"
        )
        hoisted_forward = _timed(
            torch,
            device,
            lambda runtime=torch, package=optimized, value=sample, projection=hoisted_projection, is_vector=transform == "vector": _hoisted_forward(
                runtime, package, value, projection, is_vector
            ),
            warmup,
            repeat,
        )
        records.append(
            {
                "record_type": "performance",
                "transform": transform,
                "dtype": dtype_name,
                "nlat": nlat,
                "nlon": nlon,
                "exclusive_lmax_mmax": lmax,
                "optimized_setup_s": opt_setup,
                "dense_setup_s": dense_setup,
                "optimized_registered_buffer_bytes": opt_buffers,
                "dense_registered_buffer_bytes": dense_buffers,
                "optimized_projection_bytes": opt_projection,
                "dense_projection_bytes": dense_projection,
                "optimized_cuda_peak_constructor_bytes": opt_peak,
                "dense_cuda_peak_constructor_bytes": dense_peak,
                "optimized_cuda_peak_forward_bytes": optimized_peak_forward,
                "dense_cuda_peak_forward_bytes": dense_peak_forward,
                "optimized_cuda_peak_forward_backward_bytes": optimized_peak_backward,
                "dense_cuda_peak_forward_backward_bytes": dense_peak_backward,
                "hoisted_setup_s": hoisted_setup,
                "hoisted_projection_bytes": hoisted_projection.numel()
                * hoisted_projection.element_size(),
                "hoisted_effective_max_imag": hoisted_projection.imag.abs()
                .max()
                .item(),
                "hoisted_effective_relative_imag": hoisted_projection.imag.norm().item()
                / max(hoisted_projection.real.norm().item(), 1.0e-30),
                "hoisted_forward": hoisted_forward,
                "optimized_forward": optimized_forward,
                "dense_forward": dense_forward,
                "optimized_forward_backward": optimized_backward,
                "dense_forward_backward": dense_backward,
            }
        )

        if compile_graph:
            compiled = torch.compile(optimized_module, fullgraph=True, dynamic=False)
            compiled(sample)
            _synchronize(torch, device)
            compiled_timing = _timed(
                torch,
                device,
                lambda function=compiled, value=sample: function(value),
                warmup,
                repeat,
            )
            records[-1]["optimized_compiled_forward"] = compiled_timing
            del compiled
        del optimized_module, dense_module, sample
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records


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
) -> list[dict[str, Any]]:
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
            "record_type": "accuracy",
            "transform": transform,
            "dtype": dtype_name,
            "nlat": nlat,
            "nlon": nlon,
            "exclusive_lmax_mmax": lmax,
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
                        "record_type": "accuracy",
                        "transform": transform,
                        "dtype": dtype_name,
                        "nlat": nlat,
                        "nlon": nlon,
                        "exclusive_lmax_mmax": lmax,
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
    return records


def _convention_records(
    torch: Any,
    optimized: Any,
    dense: dict[str, Any],
    device: Any,
    dtype: Any,
    dtype_name: str,
    transforms: tuple[str, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    nlat, nlon, lmax = 17, 32, 16
    sample = torch.randn(1, nlat, nlon, device=device, dtype=dtype)
    vector_sample = torch.randn(1, 2, nlat, nlon, device=device, dtype=dtype)
    for norm in NORMS:
        for csphase in (True, False):
            if "scalar" in transforms:
                opt = optimized.RealSHT(
                    nlat, nlon, lmax=lmax, mmax=lmax, norm=norm, csphase=csphase
                ).to(device=device, dtype=dtype)
                ref = dense["RealSHT"](
                    nlat, nlon, lmax=lmax, mmax=lmax, norm=norm, csphase=csphase
                ).to(device=device, dtype=dtype)
                records.append(
                    {
                        "record_type": "convention",
                        "transform": "scalar",
                        "dtype": dtype_name,
                        "norm": norm,
                        "csphase": csphase,
                        "error": _error(opt(sample), ref(sample)),
                    }
                )
            if "vector" in transforms:
                opt_v = optimized.RealVectorSHT(
                    nlat, nlon, lmax=lmax, mmax=lmax, norm=norm, csphase=csphase
                ).to(device=device, dtype=dtype)
                ref_v = dense["RealVectorSHT"](
                    nlat, nlon, lmax=lmax, mmax=lmax, norm=norm, csphase=csphase
                ).to(device=device, dtype=dtype)
                records.append(
                    {
                        "record_type": "convention",
                        "transform": "vector",
                        "dtype": dtype_name,
                        "norm": norm,
                        "csphase": csphase,
                        "error": _error(opt_v(vector_sample), ref_v(vector_sample)),
                    }
                )
    return records


def _parse_cases(value: str) -> tuple[tuple[int, int, int], ...]:
    cases = []
    for item in value.split(","):
        nlat, nlon, lmax = (int(part) for part in item.split("x"))
        cases.append((nlat, nlon, lmax))
    return tuple(cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/home/albert/torch-harmonics")
    )
    parser.add_argument("--base", default=BASE_SHA)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--cases",
        default=",".join(f"{nlat}x{nlon}x{lmax}" for nlat, nlon, lmax in DEFAULT_CASES),
    )
    parser.add_argument("--output", type=Path, default=Path("results/torch-folding"))
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--dtypes", default=None, choices=("float32", "float64"), action="append"
    )
    parser.add_argument("--transforms", default="scalar,vector")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()

    optimized = _load_optimized(args.source)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dense = _load_dense_classes(args.base, args.source)
    cases = _parse_cases(args.cases)
    requested_dtypes = set(args.dtypes or ("float32", "float64"))
    dtypes = tuple(
        item
        for item in (("float32", torch.float32), ("float64", torch.float64))
        if item[0] in requested_dtypes
    )
    transforms = tuple(
        item.strip() for item in args.transforms.split(",") if item.strip()
    )
    if not transforms or set(transforms) - {"scalar", "vector"}:
        raise SystemExit("--transforms must contain scalar and/or vector")
    records: list[dict[str, Any]] = []
    if not args.benchmark_only:
        for dtype_name, dtype in dtypes:
            for nlat, nlon, lmax in cases:
                records.extend(
                    _correctness_case(
                        torch,
                        optimized,
                        dense,
                        device,
                        dtype,
                        dtype_name,
                        nlat,
                        nlon,
                        lmax,
                        transforms,
                    )
                )
            records.extend(
                _convention_records(
                    torch, optimized, dense, device, dtype, dtype_name, transforms
                )
            )
    if not args.correctness_only:
        for dtype_name, dtype in dtypes:
            for nlat, nlon, lmax in cases:
                records.extend(
                    _benchmark_case(
                        torch,
                        optimized,
                        dense,
                        device,
                        dtype,
                        dtype_name,
                        nlat,
                        nlon,
                        lmax,
                        args.warmup,
                        args.repeat,
                        args.compile,
                        transforms,
                    )
                )

    payload = {
        "schema": "sht_bench.torch_folding.v1",
        "source": str(args.source.resolve()),
        "source_sha": subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dense_base_sha": args.base,
        "device": str(device),
        "torch_version": torch.__version__,
        "threads": args.threads,
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
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {
                key: (
                    json.dumps(record.get(key))
                    if isinstance(record.get(key), (list, dict))
                    else record.get(key)
                )
                for key in fieldnames
            }
            writer.writerow(row)
    print(f"{output}\n{csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
