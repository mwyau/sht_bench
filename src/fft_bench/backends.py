"""DUCC FFT preparation and runtime dtype checks.

This module intentionally contains an FFT-specific adapter rather than a
backend abstraction shared with ``sht_bench``.  DUCC's Python FFT interface is
small enough that the benchmark can make its call contract explicit here.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import numpy as np

KIND_NAMES = ("r2c", "c2c")
PRECISION_NAMES = ("float32", "float64", "longdouble")


class UnsupportedFFTCase(RuntimeError):
    """The installed DUCC binding cannot run the requested FFT case."""


@dataclass(frozen=True)
class DTypeSpec:
    """Actual NumPy input/output dtypes for one portable precision label."""

    kind: str
    requested_precision: str
    input_dtype: np.dtype
    output_dtype: np.dtype


@dataclass(frozen=True)
class PreparedFFT:
    """Prepared, validated transform whose setup is outside timed samples."""

    backend: str
    backend_version: str
    ducc_wrapper: str
    kind: str
    requested_precision: str
    input_dtype: np.dtype
    output_dtype: np.dtype
    size: int
    threads_requested: int
    threads_actual: int | None
    input_array: np.ndarray
    output_array: np.ndarray
    transform: Callable[[], object]
    output_preallocated: bool
    notes: str


def dtype_spec(kind: str, requested_precision: str) -> DTypeSpec:
    """Resolve a portable benchmark label to actual NumPy dtype objects."""

    if kind not in KIND_NAMES:
        raise ValueError(f"unsupported FFT kind: {kind!r}")
    if requested_precision not in PRECISION_NAMES:
        raise ValueError(f"unsupported FFT precision: {requested_precision!r}")

    if requested_precision == "float32":
        real_dtype = np.dtype(np.float32)
        complex_dtype = np.dtype(np.complex64)
    elif requested_precision == "float64":
        real_dtype = np.dtype(np.float64)
        complex_dtype = np.dtype(np.complex128)
    else:
        real_dtype = np.dtype(np.longdouble)
        complex_dtype = np.dtype(np.clongdouble)

    if kind == "r2c":
        return DTypeSpec(kind, requested_precision, real_dtype, complex_dtype)
    return DTypeSpec(kind, requested_precision, complex_dtype, complex_dtype)


def longdouble_info() -> dict[str, object]:
    """Characterize NumPy's platform-dependent ``longdouble`` ABI.

    NumPy's ``finfo.nmant`` is the mantissa-bit count reported for the stored
    significand; for the usual normalized binary formats, the effective
    significand precision includes the leading bit and is ``nmant + 1``.
    """

    dtype = np.dtype(np.longdouble)
    finfo = np.finfo(np.longdouble)
    double_info = np.finfo(np.float64)
    itemsize = int(dtype.itemsize)
    bits = int(finfo.bits)
    nmant = int(finfo.nmant)

    if itemsize == 16 and nmant == 112:
        classification = "IEEE binary128"
    elif itemsize == 16 and nmant == 63:
        classification = "x87 extended precision (64-bit significand, 16-byte storage)"
    elif nmant == int(double_info.nmant) and bits == int(double_info.bits):
        classification = "same precision as float64"
    else:
        classification = (
            f"platform-specific longdouble ({nmant + 1}-bit significand, "
            f"{itemsize}-byte storage)"
        )

    return {
        "dtype_name": dtype.name,
        "itemsize": itemsize,
        "bits": bits,
        "nmant": nmant,
        "eps": float(finfo.eps),
        "eps_text": str(finfo.eps),
        "significand_bits": nmant + 1,
        "nmant_convention": (
            "NumPy finfo.nmant mantissa-bit count; effective normalized binary "
            "significand precision is nmant + 1"
        ),
        "classification": classification,
        "wider_than_float64": bool(
            nmant > int(double_info.nmant) or finfo.eps < double_info.eps
        ),
    }


def classify_longdouble() -> str:
    """Return the human-readable runtime classification for ``np.longdouble``."""

    return str(longdouble_info()["classification"])


def _ducc_version() -> str:
    try:
        return version("ducc0")
    except PackageNotFoundError:
        return "unknown"


def _make_input(rng: np.random.Generator, spec: DTypeSpec, size: int) -> np.ndarray:
    if spec.kind == "r2c":
        values = rng.standard_normal(size)
    else:
        values = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    return np.ascontiguousarray(values, dtype=spec.input_dtype)


def _expected_output_shape(kind: str, size: int) -> tuple[int, ...]:
    return (size // 2 + 1,) if kind == "r2c" else (size,)


def _call_transform(
    fft: object,
    spec: DTypeSpec,
    input_array: np.ndarray,
    output_array: np.ndarray,
    threads: int,
) -> object:
    if spec.kind == "r2c":
        return fft.r2c(
            input_array,
            axes=(0,),
            forward=True,
            inorm=0,
            out=output_array,
            nthreads=threads,
        )
    return fft.c2c(
        input_array,
        axes=(0,),
        forward=True,
        inorm=0,
        out=output_array,
        nthreads=threads,
    )


def prepare_case(
    kind: str,
    requested_precision: str,
    size: int,
    threads: int,
    *,
    seed: int,
) -> PreparedFFT:
    """Import, allocate, validate, warm, and prepare one DUCC FFT case."""

    if size < 1:
        raise ValueError("FFT size must be >= 1")
    if threads < 1:
        raise ValueError("threads must be >= 1")
    spec = dtype_spec(kind, requested_precision)
    ducc0 = importlib.import_module("ducc0")
    try:
        fft = ducc0.fft
    except AttributeError as exc:
        raise UnsupportedFFTCase("installed ducc0 does not expose ducc0.fft") from exc

    wrapper = str(getattr(ducc0, "__wrapper__", "unknown"))
    ld_info = longdouble_info()
    if requested_precision == "longdouble":
        if not bool(ld_info["wider_than_float64"]):
            raise UnsupportedFFTCase(
                "NumPy longdouble is not wider than float64 on this platform; "
                "there is no distinct extended-precision case"
            )
        if wrapper != "pybind11":
            raise UnsupportedFFTCase(
                f"DUCC binding reports {wrapper!r}; genuine longdouble FFT support "
                "requires the pybind11 binding, not nanobind"
            )

    rng = np.random.default_rng(seed)
    input_array = _make_input(rng, spec, size)
    output_array = np.empty(_expected_output_shape(kind, size), dtype=spec.output_dtype)

    # This call is deliberately outside the timed region.  It validates the
    # actual installed binding, including the requested long-double dtype.
    try:
        returned = _call_transform(fft, spec, input_array, output_array, threads)
    except Exception as exc:
        if requested_precision == "longdouble":
            raise UnsupportedFFTCase(
                f"DUCC FFT rejected the requested NumPy longdouble dtype: {exc}"
            ) from exc
        raise

    result = np.asarray(returned)
    expected_shape = _expected_output_shape(kind, size)
    if result.shape != expected_shape:
        raise RuntimeError(
            f"DUCC {kind} returned shape {result.shape}, expected {expected_shape}"
        )
    if result.dtype != spec.output_dtype:
        raise RuntimeError(
            f"DUCC {kind} returned dtype {result.dtype}, expected {spec.output_dtype}"
        )
    if output_array.dtype != spec.output_dtype:
        raise RuntimeError(
            f"preallocated output has dtype {output_array.dtype}, "
            f"expected {spec.output_dtype}"
        )

    def transform() -> object:
        return _call_transform(fft, spec, input_array, output_array, threads)

    notes = (
        f"ducc0.fft.{kind}; forward=True; inorm=0; explicit nthreads={threads}; "
        "preallocated out=; setup and dtype validation excluded from timing"
    )
    if requested_precision == "longdouble":
        notes += f"; {ld_info['classification']}"

    return PreparedFFT(
        backend="ducc",
        backend_version=_ducc_version(),
        ducc_wrapper=wrapper,
        kind=kind,
        requested_precision=requested_precision,
        input_dtype=spec.input_dtype,
        output_dtype=spec.output_dtype,
        size=size,
        threads_requested=threads,
        threads_actual=None,
        input_array=input_array,
        output_array=output_array,
        transform=transform,
        output_preallocated=True,
        notes=notes,
    )
