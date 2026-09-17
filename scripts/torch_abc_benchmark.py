"""Compare dense (A), runtime-folded (B), and precomputed (C) Torch SHTs.

This is research code.  ``precomputed_projection`` is intentionally defined
here rather than in torch-harmonics until its numerical, autograd, compile,
and performance trade-offs are established.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch_precomputed_projection_prototype import (
    _effective_from_runtime_fold,
    _load_dense_namespace,
    _load_optimized,
)

BASE_SHA = "3278fb669483d04537aa60c87fb0754d989e2e70"
IMPLEMENTATIONS = ("dense", "runtime_fold", "precomputed_projection")
TRANSFORMS = ("scalar", "vector")
DTYPES = {"float32": torch.float32, "float64": torch.float64}
DEFAULT_CASES = "73x144x72,91x180x90,121x240x120,181x360x180,361x720x360"


class PrecomputedProjection(torch.nn.Module):
    """Research-only C module with the exact complex effective contraction."""

    def __init__(self, runtime_module: torch.nn.Module, vector: bool):
        super().__init__()
        self.nlat = runtime_module.nlat
        self.nlon = runtime_module.nlon
        self.lmax = runtime_module.lmax
        self.mmax = runtime_module.mmax
        self.vector = vector
        effective = _effective_from_runtime_fold(runtime_module, vector)
        if not effective.is_complex():
            effective = torch.complex(effective, torch.zeros_like(effective))
        self.effective_max_imag = float(effective.imag.abs().max().item())
        self.effective_relative_imag = float(
            effective.imag.norm().item() / max(effective.real.norm().item(), 1.0e-30)
        )
        self.register_buffer("weights", effective.contiguous(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = torch.fft.rfft(x, dim=-1, norm="forward")[..., : self.mmax]
        transformed = transformed.transpose(-1, -2)
        if not self.vector:
            return torch.einsum("...mk,mlk->...lm", transformed, self.weights)

        w0, w1 = self.weights
        theta = transformed[..., 0, :, :]
        longitude = transformed[..., 1, :, :]
        spheroidal = torch.einsum("...mk,mlk->...lm", theta, w0) + 1j * torch.einsum(
            "...mk,mlk->...lm", longitude, w1
        )
        toroidal = 1j * torch.einsum("...mk,mlk->...lm", theta, w1) - torch.einsum(
            "...mk,mlk->...lm", longitude, w0
        )
        return torch.stack((spheroidal, toroidal), dim=-3)


def _parse_cases(value: str) -> tuple[tuple[int, int, int], ...]:
    cases = []
    for item in value.split(","):
        parts = item.strip().split("x")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError(f"case must be nlatxnlonxlmax, got {item!r}")
        cases.append(tuple(int(part) for part in parts))
    return tuple(cases)


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(","))
    if not result or any(item < 1 for item in result):
        raise ValueError("values must be positive integers")
    return result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _buffer_bytes(module: torch.nn.Module) -> int:
    return sum(buffer.numel() * buffer.element_size() for buffer in module.buffers())


def _build_module(
    implementation: str,
    package: Any,
    dense: dict[str, Any],
    *,
    vector: bool,
    nlat: int,
    nlon: int,
    lmax: int,
    norm: str = "ortho",
    csphase: bool = True,
) -> tuple[torch.nn.Module, dict[str, float]]:
    if vector:
        optimized_class = package.RealVectorSHT
        dense_class = dense["RealVectorSHT"]
    else:
        optimized_class = package.RealSHT
        dense_class = dense["RealSHT"]
    if implementation == "dense":
        module = dense_class(
            nlat, nlon, lmax=lmax, mmax=lmax, norm=norm, csphase=csphase
        )
        return module, {}
    runtime = optimized_class(
        nlat, nlon, lmax=lmax, mmax=lmax, norm=norm, csphase=csphase
    )
    if implementation == "runtime_fold":
        return runtime, {}
    if implementation != "precomputed_projection":
        raise ValueError(f"unknown implementation {implementation}")
    module = PrecomputedProjection(runtime, vector)
    diagnostics = {
        "effective_max_imag": module.effective_max_imag,
        "effective_relative_imag": module.effective_relative_imag,
    }
    del runtime
    return module, diagnostics


def _move_module(
    module: torch.nn.Module,
    implementation: str,
    dtype_name: str,
    device: torch.device,
) -> torch.nn.Module:
    if implementation != "precomputed_projection":
        target_dtype = DTYPES[dtype_name]
    else:
        target_dtype = torch.complex64 if dtype_name == "float32" else torch.complex128
    return module.to(device=device, dtype=target_dtype).eval()


def _timed(
    function: Callable[[], Any], device: torch.device, warmup: int, repeat: int
) -> list[float]:
    for _ in range(warmup):
        function()
    _synchronize(device)
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        _synchronize(device)
        samples.append(time.perf_counter() - started)
    return samples


def _error(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = (actual - reference).abs()
    return {
        "relative_l2": float(
            difference.norm().item() / max(reference.norm().item(), 1.0e-30)
        ),
        "maximum_absolute": float(difference.max().item()),
    }


def _sample(
    transform: str,
    batch: int,
    nlat: int,
    nlon: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    shape = (batch, 2, nlat, nlon) if transform == "vector" else (batch, nlat, nlon)
    return torch.randn(shape, dtype=dtype, device=device)


def _gradient(
    module: torch.nn.Module, sample: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    value = sample.detach().clone().requires_grad_(True)
    output = module(value)
    loss = output.abs().square().mean()
    loss.backward()
    return output.detach(), value.grad.detach()


def _compile_timing(
    module: torch.nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeat: int,
    backward: bool = False,
) -> tuple[list[float] | None, str | None]:
    if not hasattr(torch, "compile"):
        return None, "torch.compile unavailable"
    try:
        compiled = torch.compile(module, fullgraph=True, dynamic=False)
        if backward:
            _gradient(compiled, sample)
        else:
            compiled(sample)
        samples = _timed(
            (
                lambda function=compiled, value=sample: (
                    _gradient(function, value) if backward else function(value)
                )
            ),
            device,
            warmup,
            repeat,
        )
        del compiled
        return samples, None
    except Exception as exc:  # noqa: BLE001 - record compile limitations as data
        return None, f"{type(exc).__name__}: {exc}"


def _peak_delta(function: Callable[[], Any], device: torch.device) -> int | None:
    if device.type != "cuda":
        function()
        return None
    torch.cuda.synchronize(device)
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    result = function()
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    del result
    return peak - baseline


def _scalar_bmm(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    batch, mmax, _ = values.shape
    lmax = weights.shape[1]
    # The repeat is intentional: it accounts for the physical expansion needed
    # to map a distinct m projection matrix into torch.bmm's batch axis.
    matrices = weights.transpose(-1, -2).repeat(batch, 1, 1)
    result = torch.bmm(values.reshape(batch * mmax, 1, -1), matrices)
    return result.reshape(batch, mmax, lmax).transpose(-1, -2)


def _contraction_comparison(
    module: PrecomputedProjection,
    sample: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeat: int,
    compile_versions: bool = False,
) -> dict[str, Any]:
    transformed = torch.fft.rfft(sample, dim=-1, norm="forward")[..., : module.mmax]
    transformed = transformed.transpose(-1, -2)
    if module.vector:
        values = transformed[..., 0, :, :]
        weights = module.weights[0]
    else:
        values = transformed
        weights = module.weights
    einsum = lambda: torch.einsum("...mk,mlk->...lm", values, weights)
    bmm = lambda: _scalar_bmm(values, weights)
    einsum_samples = _timed(einsum, device, warmup, repeat)
    bmm_samples = _timed(bmm, device, warmup, repeat)
    result = {
        "einsum_samples_s": einsum_samples,
        "einsum_median_s": statistics.median(einsum_samples),
        "bmm_samples_s": bmm_samples,
        "bmm_median_s": statistics.median(bmm_samples),
        "bmm_relative_l2": _error(bmm(), einsum())["relative_l2"],
    }
    if not compile_versions:
        return result

    def einsum_function(
        input_values: torch.Tensor, input_weights: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("...mk,mlk->...lm", input_values, input_weights)

    def bmm_function(
        input_values: torch.Tensor, input_weights: torch.Tensor
    ) -> torch.Tensor:
        return _scalar_bmm(input_values, input_weights)

    for name, function in (("einsum", einsum_function), ("bmm", bmm_function)):
        try:
            compiled = torch.compile(function, fullgraph=True, dynamic=False)
            expected = function(values, weights)
            actual = compiled(values, weights)
            samples = _timed(
                lambda function=compiled: function(values, weights),
                device,
                warmup,
                repeat,
            )
            result[f"compiled_{name}_samples_s"] = samples
            result[f"compiled_{name}_median_s"] = statistics.median(samples)
            result[f"compiled_{name}_relative_l2"] = _error(actual, expected)[
                "relative_l2"
            ]
            result[f"compiled_{name}_error"] = None
            del compiled, expected, actual
        except Exception as exc:  # noqa: BLE001 - record compiler limits as data
            result[f"compiled_{name}_samples_s"] = None
            result[f"compiled_{name}_median_s"] = None
            result[f"compiled_{name}_relative_l2"] = None
            result[f"compiled_{name}_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    package = _load_optimized(source)
    dense = _load_dense_namespace(source, args.base)
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    records: list[dict[str, Any]] = []
    cases = _parse_cases(args.cases)
    for case_index, (nlat, nlon, lmax) in enumerate(cases):
        for dtype_name in args.dtypes.split(","):
            dtype_name = dtype_name.strip()
            dtype = DTYPES[dtype_name]
            for transform in args.transforms.split(","):
                transform = transform.strip()
                vector = transform == "vector"
                for implementation in IMPLEMENTATIONS:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize(device)
                        construction_baseline = int(torch.cuda.memory_allocated(device))
                        torch.cuda.reset_peak_memory_stats(device)
                    else:
                        construction_baseline = None
                    started = time.perf_counter()
                    module, construction_diagnostics = _build_module(
                        implementation,
                        package,
                        dense,
                        vector=vector,
                        nlat=nlat,
                        nlon=nlon,
                        lmax=lmax,
                    )
                    module = _move_module(module, implementation, dtype_name, device)
                    _synchronize(device)
                    construction_s = time.perf_counter() - started
                    construction_peak = (
                        int(torch.cuda.max_memory_allocated(device))
                        - (construction_baseline or 0)
                        if device.type == "cuda"
                        else None
                    )
                    sample = _sample(
                        transform,
                        max(args.batches),
                        nlat,
                        nlon,
                        dtype,
                        device,
                        8128 + case_index,
                    )
                    forward_samples_by_batch: dict[int, list[float]] = {}
                    for batch in args.batches:
                        batch_sample = sample[:batch]
                        forward_samples = _timed(
                            lambda function=module, value=batch_sample: function(value),
                            device,
                            args.warmup,
                            args.repeat,
                        )
                        forward_peak = _peak_delta(
                            lambda function=module, value=batch_sample: function(value),
                            device,
                        )
                        forward_samples_by_batch[batch] = forward_samples
                        records.append(
                            {
                                "record_type": "performance",
                                "implementation": implementation,
                                "transform": transform,
                                "dtype": dtype_name,
                                "device": args.device,
                                "nlat": nlat,
                                "nlon": nlon,
                                "exclusive_lmax_mmax": lmax,
                                "batch_size": batch,
                                "measurement": "forward",
                                "samples_s": forward_samples,
                                "median_s": statistics.median(forward_samples),
                                "milliseconds_per_frame": statistics.median(
                                    forward_samples
                                )
                                * 1000.0
                                / batch,
                                "module_buffer_bytes": _buffer_bytes(module),
                                "construction_s": construction_s,
                                "cuda_peak_constructor_delta_bytes": construction_peak,
                                "cuda_peak_forward_delta_bytes": forward_peak,
                                **construction_diagnostics,
                            }
                        )
                    if implementation == "precomputed_projection":
                        for contraction_batch in args.batches:
                            contraction = _contraction_comparison(
                                module,
                                sample[:contraction_batch],
                                device,
                                args.warmup,
                                args.repeat,
                                compile_versions=args.compile,
                            )
                            records.append(
                                {
                                    "record_type": "contraction",
                                    "implementation": implementation,
                                    "transform": transform,
                                    "dtype": dtype_name,
                                    "nlat": nlat,
                                    "nlon": nlon,
                                    "batch_size": contraction_batch,
                                    **contraction,
                                }
                            )
                    if args.backward:
                        value = sample[: args.backward_batch]
                        backward_samples = _timed(
                            lambda function=module, batch_value=value: _gradient(
                                function, batch_value
                            ),
                            device,
                            args.warmup,
                            args.repeat,
                        )
                        backward_peak = _peak_delta(
                            lambda function=module, batch_value=value: _gradient(
                                function, batch_value
                            ),
                            device,
                        )
                        records.append(
                            {
                                "record_type": "performance",
                                "implementation": implementation,
                                "transform": transform,
                                "dtype": dtype_name,
                                "device": args.device,
                                "nlat": nlat,
                                "nlon": nlon,
                                "exclusive_lmax_mmax": lmax,
                                "batch_size": args.backward_batch,
                                "measurement": "forward_backward",
                                "samples_s": backward_samples,
                                "median_s": statistics.median(backward_samples),
                                "milliseconds_per_frame": statistics.median(
                                    backward_samples
                                )
                                * 1000.0
                                / args.backward_batch,
                                "module_buffer_bytes": _buffer_bytes(module),
                                "construction_s": construction_s,
                                "cuda_peak_constructor_delta_bytes": construction_peak,
                                "cuda_peak_forward_backward_delta_bytes": backward_peak,
                                **construction_diagnostics,
                            }
                        )
                    if args.compile and args.batches[0] <= args.compile_max_batch:
                        compile_samples, compile_error = _compile_timing(
                            module,
                            sample[: args.batches[0]],
                            device,
                            args.warmup,
                            args.repeat,
                        )
                        records.append(
                            {
                                "record_type": "compiled",
                                "implementation": implementation,
                                "transform": transform,
                                "dtype": dtype_name,
                                "nlat": nlat,
                                "nlon": nlon,
                                "exclusive_lmax_mmax": lmax,
                                "batch_size": args.batches[0],
                                "measurement": "forward",
                                "samples_s": compile_samples,
                                "median_s": statistics.median(compile_samples)
                                if compile_samples
                                else None,
                                "error": compile_error,
                            }
                        )
                        if args.backward:
                            compile_backward_samples, compile_backward_error = (
                                _compile_timing(
                                    module,
                                    sample[: args.batches[0]],
                                    device,
                                    args.warmup,
                                    args.repeat,
                                    backward=True,
                                )
                            )
                            records.append(
                                {
                                    "record_type": "compiled",
                                    "implementation": implementation,
                                    "transform": transform,
                                    "dtype": dtype_name,
                                    "nlat": nlat,
                                    "nlon": nlon,
                                    "exclusive_lmax_mmax": lmax,
                                    "batch_size": args.batches[0],
                                    "measurement": "forward_backward",
                                    "samples_s": compile_backward_samples,
                                    "median_s": statistics.median(
                                        compile_backward_samples
                                    )
                                    if compile_backward_samples
                                    else None,
                                    "error": compile_backward_error,
                                }
                            )
                    del module, sample
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    # Separate float64 gradient comparisons provide correctness records for
    # both scalar and vector C autograd paths instead of treating a timing run
    # as validation.
    nlat, nlon, lmax = cases[0]
    for transform in args.transforms.split(","):
        transform = transform.strip()
        vector = transform == "vector"
        modules = {}
        for implementation in IMPLEMENTATIONS:
            module, _ = _build_module(
                implementation,
                package,
                dense,
                vector=vector,
                nlat=nlat,
                nlon=nlon,
                lmax=lmax,
            )
            modules[implementation] = _move_module(
                module, implementation, "float64", device
            )
        validation_sample = _sample(
            transform, 2, nlat, nlon, torch.float64, device, 2718
        )
        outputs = {
            implementation: module(validation_sample)
            for implementation, module in modules.items()
        }
        gradients = {
            implementation: _gradient(module, validation_sample)[1]
            for implementation, module in modules.items()
        }
        for implementation in IMPLEMENTATIONS[1:]:
            records.append(
                {
                    "record_type": "correctness",
                    "transform": transform,
                    "dtype": "float64",
                    "nlat": nlat,
                    "nlon": nlon,
                    "exclusive_lmax_mmax": lmax,
                    "implementation": implementation,
                    "output_vs_dense": _error(
                        outputs[implementation], outputs["dense"]
                    ),
                    "gradient_vs_dense": _error(
                        gradients[implementation], gradients["dense"]
                    ),
                    "output_vs_runtime_fold": _error(
                        outputs[implementation], outputs["runtime_fold"]
                    ),
                    "gradient_vs_runtime_fold": _error(
                        gradients[implementation], gradients["runtime_fold"]
                    ),
                }
            )
        del modules, validation_sample, outputs, gradients
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "schema": "sht_bench.torch_abc.v1",
        "source": str(source),
        "source_sha": subprocess_sha(source),
        "dense_base_sha": args.base,
        "device": args.device,
        "cases": cases,
        "records": records,
    }


def subprocess_sha(repository: Path) -> str:
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--dtypes", default="float32")
    parser.add_argument("--transforms", default="scalar,vector")
    parser.add_argument("--batches", default="1,4,8")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--backward-batch", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-max-batch", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.batches = _parse_positive_ints(args.batches)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    result = _run(args)
    output = args.output
    if output is None:
        output = Path("results/torch-abc.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
