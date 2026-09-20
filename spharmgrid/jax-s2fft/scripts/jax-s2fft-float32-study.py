#!/usr/bin/env python3
"""Measure ordinary S2FFT JAX precision for the spharmgrid sampling grids.

This development script is not part of the spharmgrid test suite.  It compares
single-precision JAX transforms with the float64 NumPy implementation supplied
by the same S2FFT release.  The optional precomputed path is included only as
diagnostic evidence and uses S2FFT's public precompute-transform functions.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from collections.abc import Callable
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
        parser.error(
            f"CUDA precision study requires a JAX CudaDevice; JAX reported {devices!r}"
        )


def _install_diagnostic_patches() -> list[str]:
    """Avoid two S2FFT 1.4.0 no-JIT helper issues during diagnostics."""
    from s2fft.utils import quadrature, quadrature_jax, resampling_jax

    patches = ["gl_quadrature_from_s2fft_numpy", "negative_spin_resampling_sign"]
    quadrature_dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    quadrature_jax.quad_weights_gl = lambda bandlimit: jnp.asarray(
        quadrature.quad_weights_gl(bandlimit), dtype=quadrature_dtype
    )
    original_periodic_extension = resampling_jax.periodic_extension_spatial_mwss

    def periodic_extension(field: jax.Array, bandlimit: int, spin: int) -> jax.Array:
        # S2FFT's no-JIT branch evaluates (-1) ** spin as an integer power for
        # negative scalar spins.  The intended extension factor is
        # (-1) ** abs(spin).
        return original_periodic_extension(
            field,
            bandlimit,
            np.asarray(abs(int(np.asarray(spin)))),
        )

    resampling_jax.periodic_extension_spatial_mwss = periodic_extension
    return patches


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


def _transform_functions(
    bandlimit: int,
    sampling: str,
    spin: int,
    path: str,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    if path == "ordinary_jax":

        def forward(field: jax.Array) -> jax.Array:
            return s2fft.forward_jax(
                field,
                bandlimit,
                spin=spin,
                sampling=sampling,
                reality=False,
            )

        def inverse(coefficients: jax.Array) -> jax.Array:
            return s2fft.inverse_jax(
                coefficients,
                bandlimit,
                spin=spin,
                sampling=sampling,
                reality=False,
            )

        return forward, inverse

    if path == "precomps_jax":
        forward_precomps = s2fft.generate_precomputes_jax(
            bandlimit,
            spin=spin,
            sampling=sampling,
            forward=True,
        )
        inverse_precomps = s2fft.generate_precomputes_jax(
            bandlimit,
            spin=spin,
            sampling=sampling,
            forward=False,
        )

        def forward(field: jax.Array) -> jax.Array:
            return s2fft.forward_jax(
                field,
                bandlimit,
                spin=spin,
                sampling=sampling,
                reality=False,
                precomps=forward_precomps,
            )

        def inverse(coefficients: jax.Array) -> jax.Array:
            return s2fft.inverse_jax(
                coefficients,
                bandlimit,
                spin=spin,
                sampling=sampling,
                reality=False,
                precomps=inverse_precomps,
            )

        return forward, inverse

    if path == "precomps_host":
        forward_precomps = [
            jnp.asarray(values)
            for values in s2fft.generate_precomputes(
                bandlimit,
                spin=spin,
                sampling=sampling,
                forward=True,
            )
        ]
        inverse_precomps = [
            jnp.asarray(values)
            for values in s2fft.generate_precomputes(
                bandlimit,
                spin=spin,
                sampling=sampling,
                forward=False,
            )
        ]

        def forward(field: jax.Array) -> jax.Array:
            return s2fft.forward_jax(
                field,
                bandlimit,
                spin=spin,
                sampling=sampling,
                reality=False,
                precomps=forward_precomps,
            )

        def inverse(coefficients: jax.Array) -> jax.Array:
            return s2fft.inverse_jax(
                coefficients,
                bandlimit,
                spin=spin,
                sampling=sampling,
                reality=False,
                precomps=inverse_precomps,
            )

        return forward, inverse

    if path != "precomputed_public":
        raise ValueError(f"unknown diagnostic path: {path}")

    from s2fft.precompute_transforms import construct, spherical

    forward_kernel = construct.spin_spherical_kernel(
        bandlimit,
        spin=spin,
        reality=False,
        sampling=sampling,
        forward=True,
        recursion="price-mcewen",
    ).astype(np.float32)
    inverse_kernel = construct.spin_spherical_kernel(
        bandlimit,
        spin=spin,
        reality=False,
        sampling=sampling,
        forward=False,
        recursion="price-mcewen",
    ).astype(np.float32)

    def forward(field: jax.Array) -> jax.Array:
        return spherical.forward(
            field,
            bandlimit,
            spin=spin,
            kernel=forward_kernel,
            sampling=sampling,
            reality=False,
            method="jax",
        )

    def inverse(coefficients: jax.Array) -> jax.Array:
        return spherical.inverse(
            coefficients,
            bandlimit,
            spin=spin,
            kernel=inverse_kernel,
            sampling=sampling,
            reality=False,
            method="jax",
        )

    return forward, inverse


def _run_case(
    bandlimit: int,
    sampling: str,
    spin: int,
    *,
    path: str,
    requested_dtype: str,
) -> dict[str, object]:
    field = _field(bandlimit, sampling, spin)
    field_dtype = np.complex128 if spin else np.float64
    field = np.asarray(field, dtype=field_dtype)
    reference_coefficients = s2fft.forward_numpy(
        field,
        bandlimit,
        spin=spin,
        sampling=sampling,
        reality=False,
    )
    reference_field = s2fft.inverse_numpy(
        reference_coefficients,
        bandlimit,
        spin=spin,
        sampling=sampling,
        reality=False,
    )
    if requested_dtype == "float64":
        real_input_dtype = jnp.complex128 if spin else jnp.float64
        inverse_dtype = jnp.complex128
    else:
        real_input_dtype = jnp.complex64 if spin else jnp.float32
        inverse_dtype = (
            jnp.complex128 if jax.config.read("jax_enable_x64") else jnp.complex64
        )
    jax_field = jnp.asarray(field, dtype=real_input_dtype)
    forward, inverse = _transform_functions(
        bandlimit,
        sampling,
        spin,
        path,
    )
    started = time.perf_counter()
    coefficients = np.asarray(forward(jax_field))
    inverse_from_reference = np.asarray(
        inverse(jnp.asarray(reference_coefficients, dtype=inverse_dtype))
    )
    round_trip = np.asarray(inverse(jnp.asarray(coefficients)))
    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "bandlimit": bandlimit,
        "sampling": sampling,
        "spin": spin,
        "path": path,
        "requested_dtype": requested_dtype,
        "effective_field_dtype": str(jax_field.dtype),
        "effective_coefficient_dtype": str(coefficients.dtype),
        "forward_coefficient_relative_max": _relative_max(
            coefficients, reference_coefficients
        ),
        "inverse_spatial_relative_max": _relative_max(
            inverse_from_reference, reference_field
        ),
        "round_trip_relative_max": _relative_max(round_trip, reference_field),
        "elapsed_seconds": elapsed,
    }
    if sampling == "mwss":
        row_error = np.abs(inverse_from_reference - reference_field)
        scale = max(float(np.max(np.abs(reference_field))), 1.0e-30)
        result.update(
            {
                "north_pole_relative_max": float(np.max(row_error[0]) / scale),
                "south_pole_relative_max": float(np.max(row_error[-1]) / scale),
                "interior_relative_max": float(np.max(row_error[1:-1]) / scale),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bandlimits",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
    )
    parser.add_argument("--precomputed", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="dtype of the input field and transform coefficients",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail unless JAX reports at least one CUDA-backed device",
    )
    parser.add_argument(
        "--precomps",
        choices=("jax", "host"),
        help="use S2FFT's O(L^2) JAX or host precomputations",
    )
    parser.add_argument("--no-jit-diagnostic-patches", action="store_true")
    args = parser.parse_args()

    patches: list[str] = []
    disable_jit = bool(jax.config.jax_disable_jit)
    if not args.no_jit_diagnostic_patches and disable_jit:
        patches = _install_diagnostic_patches()
    if args.precomputed and args.precomps:
        parser.error("--precomputed and --precomps are mutually exclusive")
    if args.dtype == "float64" and not jax.config.read("jax_enable_x64"):
        parser.error("--dtype float64 requires JAX_ENABLE_X64=1")
    if args.require_cuda:
        _require_cuda(parser)
    path = (
        "precomputed_public"
        if args.precomputed
        else f"precomps_{args.precomps}"
        if args.precomps
        else "ordinary_jax"
    )
    results = []
    for bandlimit in args.bandlimits:
        for sampling in ("gl", "mwss"):
            for spin in (0, 1, -1):
                result = _run_case(
                    bandlimit,
                    sampling,
                    spin,
                    path=path,
                    requested_dtype=args.dtype,
                )
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)

    output = {
        "metadata": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _versions(),
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "jax_disable_jit": disable_jit,
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
            "diagnostic_patches": patches,
            "reference": "s2fft.forward_numpy/inverse_numpy float64",
            "error": (
                "maximum absolute error divided by maximum absolute reference value"
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
