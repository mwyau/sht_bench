#!/usr/bin/env python3
"""Minimal ordinary S2FFT MWSS float32 reproducer for upstream reports."""

from __future__ import annotations

import importlib.metadata
import platform

import jax
import jax.numpy as jnp
import numpy as np
import s2fft
from s2fft.sampling import s2_samples


def relative_max(values: np.ndarray, reference: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(reference))), 1.0e-30)
    return float(np.max(np.abs(values - reference)) / scale)


def main() -> None:
    bandlimit = 8
    sampling = "mwss"
    theta = s2_samples.thetas(bandlimit, sampling)
    latitude = np.pi / 2.0 - theta[:, None]
    nlon = s2_samples.nphi_equiang(bandlimit, sampling)
    longitude = np.arange(nlon)[None, :] * 2.0 * np.pi / nlon
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    field = (
        0.8
        + 0.35 * sine
        + 0.25 * (3.0 * sine**2 - 1.0) / 2.0
        + 0.20 * cosine * np.cos(longitude)
        + 0.17 * sine * cosine * np.sin(longitude)
        + 0.13 * cosine**2 * np.cos(2.0 * longitude)
        + 0.11 * sine * cosine**2 * np.sin(2.0 * longitude)
        + 0.07 * cosine**3 * np.cos(3.0 * longitude)
    )
    reference_coefficients = s2fft.forward_numpy(
        field,
        bandlimit,
        sampling=sampling,
        reality=False,
    )
    reference_field = s2fft.inverse_numpy(
        reference_coefficients,
        bandlimit,
        sampling=sampling,
        reality=False,
    )
    jax_field = jnp.asarray(field, dtype=jnp.float32)
    coefficients = np.asarray(
        s2fft.forward_jax(
            jax_field,
            bandlimit,
            sampling=sampling,
            reality=False,
        )
    )
    inverse_field = np.asarray(
        s2fft.inverse_jax(
            jnp.asarray(reference_coefficients, dtype=jnp.complex64),
            bandlimit,
            sampling=sampling,
            reality=False,
        )
    )
    row_error = np.abs(inverse_field - reference_field)
    scale = max(float(np.max(np.abs(reference_field))), 1.0e-30)
    print("python", platform.python_version())
    print("jax", jax.__version__)
    print("jaxlib", importlib.metadata.version("jaxlib"))
    print("s2fft", importlib.metadata.version("s2fft"))
    print("jax_enable_x64", bool(jax.config.read("jax_enable_x64")))
    print(
        "forward_coefficient_relative_max",
        relative_max(coefficients, reference_coefficients),
    )
    print("inverse_spatial_relative_max", relative_max(inverse_field, reference_field))
    print("north_pole_relative_max", float(np.max(row_error[0]) / scale))
    print("south_pole_relative_max", float(np.max(row_error[-1]) / scale))
    print("interior_relative_max", float(np.max(row_error[1:-1]) / scale))


if __name__ == "__main__":
    main()
