"""Direct DUCC/Torch scalar SHT comparisons.

This module is intentionally separate from the historical matrix runner.  The
matrix uses its existing backend-specific random inputs and legacy grid
definitions; this focused path uses explicit cases, one canonical coefficient
array, and direct calls to ``ducc0.sht`` and a source checkout of
``torch-harmonics``.
"""

from __future__ import annotations

import csv
import gc
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bench_common.environment import collect_environment
from bench_common.timing import measure

from .grids import HIGH_BANDWIDTH_CC_CASES, SHTCase

REAL_DTYPE_NAMES = ("float32", "float64")
COMPARE_BACKEND_NAMES = ("ducc", "torch")
TORCH_ANALYSIS_METHODS = ("quadrature", "sampling-theorem")
DEFAULT_SEED = 20260910
RESULT_SCHEMA = "sht_bench.compare.v1"


class ComparisonUnavailable(RuntimeError):
    """Raised when a requested backend/source/device is unavailable."""


class ConventionError(RuntimeError):
    """Raised when the low-degree convention calibration does not pass."""


class AccuracyError(RuntimeError):
    """Raised when a valid comparison fails its declared tolerance."""


@dataclass(frozen=True, slots=True)
class DTypeSpec:
    name: str
    real: np.dtype[Any]
    complex: np.dtype[Any]


def dtype_spec(name: str | np.dtype[Any]) -> DTypeSpec:
    """Return the explicitly supported real/complex dtype pair."""

    normalized = str(name).lower()
    if normalized in {"f4", "<f4", "float32"}:
        return DTypeSpec("float32", np.dtype(np.float32), np.dtype(np.complex64))
    if normalized in {"f8", "<f8", "float64"}:
        return DTypeSpec("float64", np.dtype(np.float64), np.dtype(np.complex128))
    raise ValueError(f"unsupported comparison dtype {name!r}; use float32 or float64")


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _git_metadata(path: Path) -> dict[str, Any]:
    """Return read-only Git provenance for *path*, if it is a checkout."""

    path = path.resolve()
    try:
        root = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        sha = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(path), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {
            "path": str(path),
            "git_sha": None,
            "branch": None,
            "dirty": None,
            "git_root": None,
        }
    return {
        "path": str(path),
        "git_sha": sha,
        "branch": branch,
        "dirty": bool(status.strip()),
        "git_root": root,
    }


def _sht_bench_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return _git_metadata(root)


def default_spharmgrid_reference() -> Path | None:
    """Find the local read-only reference checkout without importing it."""

    configured = os.environ.get("SHT_BENCH_SPHARMGRID_REFERENCE")
    candidates = [
        Path(configured) if configured else None,
        Path("/home/albert/spharmgrid-private"),
        Path("/home/albert/spharmgrid"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / ".git").exists():
            return candidate.resolve()
    return None


def reference_provenance(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "git_sha": None,
            "branch": None,
            "dirty": None,
            "role": "reference-only; not a runtime dependency",
        }
    result = _git_metadata(path)
    result["role"] = "reference-only; not a runtime dependency"
    return result


def canonical_coefficients(
    lmax: int,
    dtype: str | np.dtype[Any],
    seed: int = DEFAULT_SEED,
) -> NDArray[np.complexfloating[Any, Any]]:
    """Generate deterministic canonical rectangular ``(ell, m)`` coefficients.

    The random values are generated in float64 once and then explicitly cast,
    so float32 and float64 are two precisions of the same backend-neutral field.
    Invalid ``m > ell`` entries are zero and the zonal column is purely real.
    """

    spec = dtype_spec(dtype)
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    rng = np.random.default_rng(seed)
    shape = (lmax + 1, lmax + 1)
    real = rng.standard_normal(shape, dtype=np.float64).astype(spec.real, copy=False)
    imag = rng.standard_normal(shape, dtype=np.float64).astype(spec.real, copy=False)
    result = np.asarray(real + 1j * imag, dtype=spec.complex)
    result[:, 0] = real[:, 0]
    result[np.triu_indices(lmax + 1, k=1)] = 0
    return result


def single_mode_coefficients(
    lmax: int,
    degree: int,
    order: int,
    dtype: str | np.dtype[Any],
) -> NDArray[np.complexfloating[Any, Any]]:
    """Return one deterministic coefficient mode in canonical rectangular form."""

    spec = dtype_spec(dtype)
    if not (0 <= order <= degree <= lmax):
        raise ValueError(f"invalid mode ({degree}, {order}) for lmax={lmax}")
    result = np.zeros((lmax + 1, lmax + 1), dtype=spec.complex)
    result[degree, order] = 1.0 if order == 0 else 0.375 + 0.625j
    return result


def required_single_modes(case: SHTCase) -> tuple[tuple[int, int], ...]:
    """Return mandatory low-order/high-degree probes valid for *case*."""

    lmax = case.lmax
    if lmax == 70:
        modes = ((70, 0), (70, 1), (70, 2), (70, 69), (70, 70))
    elif lmax == 71:
        modes = (
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
    else:
        modes = ((lmax, 0), (lmax, 1), (lmax, 2), (lmax, lmax - 1), (lmax, lmax))
    return tuple(mode for mode in modes if mode[1] <= mode[0] <= lmax)


def ducc_packed_size(lmax: int, mmax: int | None = None) -> int:
    """Return DUCC's inclusive m-major packed coefficient count."""

    if mmax is None:
        mmax = lmax
    if lmax < 0 or mmax < 0 or mmax > lmax:
        raise ValueError("DUCC inclusive mmax must satisfy 0 <= mmax <= lmax")
    return sum(lmax - m + 1 for m in range(mmax + 1))


def ducc_packed_index(lmax: int, m: int, ell: int) -> int:
    """Return the index of ``(ell, m)`` in DUCC's m-major layout."""

    if not (0 <= m <= ell <= lmax):
        raise ValueError(f"invalid DUCC coefficient index ell={ell}, m={m}")
    return m * (2 * lmax + 1 - m) // 2 + ell


def rectangular_to_ducc(
    coefficients: NDArray[Any],
    lmax: int | None = None,
    mmax: int | None = None,
) -> NDArray[Any]:
    """Pack canonical rectangular ``(..., ell, m)`` coefficients for DUCC."""

    array = np.asarray(coefficients)
    if array.ndim < 2:
        raise ValueError("rectangular coefficients must have at least two dimensions")
    inferred_lmax = array.shape[-2] - 1
    inferred_mmax = array.shape[-1] - 1
    if lmax is None:
        lmax = inferred_lmax
    if mmax is None:
        mmax = inferred_mmax
    if array.shape[-2:] != (lmax + 1, mmax + 1):
        raise ValueError(
            "rectangular coefficient shape does not match inclusive limits: "
            f"shape={array.shape[-2:]}, lmax={lmax}, mmax={mmax}"
        )
    if mmax > lmax:
        raise ValueError("DUCC triangular packing requires mmax <= lmax")
    result = np.empty(array.shape[:-2] + (ducc_packed_size(lmax, mmax),), dtype=array.dtype)
    for m in range(mmax + 1):
        for ell in range(m, lmax + 1):
            result[..., ducc_packed_index(lmax, m, ell)] = array[..., ell, m]
    return result


def ducc_to_rectangular(
    packed: NDArray[Any],
    lmax: int,
    mmax: int | None = None,
) -> NDArray[Any]:
    """Unpack DUCC's inclusive m-major storage to canonical rectangular form."""

    if mmax is None:
        mmax = lmax
    array = np.asarray(packed)
    expected = ducc_packed_size(lmax, mmax)
    if array.ndim < 1 or array.shape[-1] != expected:
        raise ValueError(
            f"packed coefficient shape must end in {expected}; got {array.shape}"
        )
    result = np.zeros(array.shape[:-1] + (lmax + 1, mmax + 1), dtype=array.dtype)
    for m in range(mmax + 1):
        for ell in range(m, lmax + 1):
            result[..., ell, m] = array[..., ducc_packed_index(lmax, m, ell)]
    return result


def comparison_metrics(
    actual: NDArray[Any],
    reference: NDArray[Any],
    dtype: str | np.dtype[Any],
) -> dict[str, Any]:
    """Calculate reproducible absolute, L2, and bounded-relative metrics.

    The denominator floor is ``eps * max(reference_max,
    reference_norm/sqrt(N), tiny)``.  Thus it scales with the reference and
    dtype rather than being a fixed arbitrary constant; zero-reference entries
    still receive a finite, precision-aware denominator.
    """

    left = np.asarray(actual)
    right = np.asarray(reference)
    if left.shape != right.shape:
        raise ValueError(f"comparison shapes differ: {left.shape} versus {right.shape}")
    spec = dtype_spec(dtype)
    error = np.abs(left - right)
    reference_abs = np.abs(right)
    count = int(right.size)
    reference_norm = float(np.linalg.norm(right.reshape(-1)))
    reference_max = float(reference_abs.max()) if count else 0.0
    reference_scale = max(
        reference_max,
        reference_norm / math.sqrt(count) if count else 0.0,
        float(np.finfo(spec.real).tiny),
    )
    floor = float(np.finfo(spec.real).eps * reference_scale)
    relative_l2 = float(np.linalg.norm((left - right).reshape(-1)) / reference_norm) if reference_norm else (
        0.0 if not error.any() else float("inf")
    )
    bounded = float(np.max(error / np.maximum(reference_abs, floor))) if count else 0.0
    return {
        "rel_l2": relative_l2,
        "max_abs": float(error.max()) if count else 0.0,
        "bounded_rel": bounded,
        "compared_values": count,
        "reference_norm": reference_norm,
        "reference_max": reference_max,
        "bounded_rel_floor": floor,
    }


def accuracy_limits(
    dtype: str | np.dtype[Any], quantity: str = "analysis"
) -> dict[str, float]:
    """Return measured starting limits for one cross-backend quantity.

    Analysis retains the requested initial targets.  Synthesis has a separate
    map-valued limit because direct high-bandwidth measurements establish a
    larger absolute round-off envelope as the grid grows; its relative limit
    remains equally strict.  These are acceptance limits for this source
    revision, not claims about another torch-harmonics build.
    """

    if quantity not in {"analysis", "synthesis"}:
        raise ValueError(f"unknown accuracy quantity {quantity!r}")
    if dtype_spec(dtype).name == "float32":
        return (
            {"rel_l2": 5.0e-5, "max_abs": 5.0e-6}
            if quantity == "analysis"
            else {"rel_l2": 5.0e-5, "max_abs": 1.5e-4}
        )
    return (
        {"rel_l2": 1.0e-10, "max_abs": 1.0e-10}
        if quantity == "analysis"
        else {"rel_l2": 1.0e-10, "max_abs": 2.0e-10}
    )


def _metrics_pass(
    metrics: dict[str, Any], dtype: str, quantity: str = "analysis"
) -> bool:
    limits = accuracy_limits(dtype, quantity)
    return (
        metrics["rel_l2"] <= limits["rel_l2"]
        and metrics["max_abs"] <= limits["max_abs"]
    )


@dataclass
class PreparedDUCC:
    case: SHTCase
    dtype: DTypeSpec
    threads: int
    module: Any
    setup_time_s: float
    coefficients: NDArray[Any] | None = None
    spatial: NDArray[Any] | None = None

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "backend": "ducc0.sht",
            "backend_version": _package_version("ducc0"),
            "backend_source": str(Path(self.module.__file__).resolve()),
            "backend_git_sha": None,
            "build_mode": "installed-package",
            "nthreads": self.threads,
        }

    def set_coefficients(self, coefficients: NDArray[Any]) -> None:
        array = np.asarray(coefficients, dtype=self.dtype.complex)
        if array.shape != (self.case.lmax + 1, self.case.lmax + 1):
            raise ValueError(f"unexpected canonical coefficient shape {array.shape}")
        self.coefficients = rectangular_to_ducc(
            array,
            lmax=self.case.lmax,
            mmax=self.case.mmax,
        )[None, ...]

    def set_spatial(self, spatial: NDArray[Any]) -> None:
        array = np.asarray(spatial, dtype=self.dtype.real)
        if array.shape == (self.case.nlat, self.case.nlon):
            array = array[None, ...]
        if array.shape != (1, self.case.nlat, self.case.nlon):
            raise ValueError(f"unexpected DUCC spatial shape {array.shape}")
        self.spatial = np.ascontiguousarray(array)

    def synthesis(self) -> NDArray[Any]:
        if self.coefficients is None:
            raise RuntimeError("DUCC coefficients were not prepared")
        result = self.module.synthesis_2d(
            alm=self.coefficients,
            spin=0,
            lmax=self.case.lmax,
            mmax=self.case.mmax,
            geometry="CC" if self.case.grid == "cc" else "GL",
            ntheta=self.case.nlat,
            nphi=self.case.nlon,
            nthreads=self.threads,
        )
        result = np.asarray(result)
        if result.shape != (1, self.case.nlat, self.case.nlon):
            raise RuntimeError(f"DUCC returned unexpected map shape {result.shape}")
        if result.dtype != self.dtype.real:
            raise RuntimeError(f"DUCC returned {result.dtype}, expected {self.dtype.real}")
        return result

    def analysis(self) -> NDArray[Any]:
        if self.spatial is None:
            raise RuntimeError("DUCC spatial input was not prepared")
        result = self.module.analysis_2d(
            map=self.spatial,
            spin=0,
            lmax=self.case.lmax,
            mmax=self.case.mmax,
            geometry="CC" if self.case.grid == "cc" else "GL",
            nthreads=self.threads,
        )
        result = np.asarray(result)
        expected = (1, ducc_packed_size(self.case.lmax, self.case.mmax))
        if result.shape != expected:
            raise RuntimeError(f"DUCC returned unexpected alm shape {result.shape}")
        if result.dtype != self.dtype.complex:
            raise RuntimeError(f"DUCC returned {result.dtype}, expected {self.dtype.complex}")
        return ducc_to_rectangular(
            result,
            self.case.lmax,
            self.case.mmax,
        )


def prepare_ducc(case: SHTCase, dtype: str, threads: int) -> PreparedDUCC:
    if case.grid not in {"cc", "gl"}:
        raise ComparisonUnavailable(f"DUCC comparison does not support grid {case.grid!r}")
    try:
        ducc0 = importlib.import_module("ducc0")
        module = ducc0.sht
    except (ImportError, AttributeError) as exc:
        raise ComparisonUnavailable(f"ducc0.sht is unavailable: {exc}") from exc
    spec = dtype_spec(dtype)
    start = time.perf_counter()
    state = PreparedDUCC(case, spec, threads, module, 0.0)
    state.setup_time_s = time.perf_counter() - start
    return state


@dataclass
class TorchRuntime:
    torch: Any
    module: Any
    source_path: Path
    module_path: Path
    git: dict[str, Any]
    distribution_version: str
    real_sht_signature: str
    cuda_runtime: str | None

    @property
    def supports_sampling_theorem(self) -> bool:
        return True

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "backend": "torch-harmonics",
            "backend_version": self.distribution_version,
            "backend_source": str(self.source_path),
            "backend_module_path": str(self.module_path),
            "backend_git_sha": self.git["git_sha"],
            "build_mode": "source-checkout",
            "torch_version": self.torch.__version__,
            "torch_harmonics_version": self.distribution_version,
            "torch_harmonics_git_sha": self.git["git_sha"],
            "torch_harmonics_branch": self.git["branch"],
            "torch_harmonics_real_sht_signature": self.real_sht_signature,
            "cuda_runtime": self.cuda_runtime,
        }


def _source_root(source_path: Path, module_path: Path) -> Path:
    candidate = source_path.resolve()
    if (candidate / "torch_harmonics").is_dir():
        root = candidate
    elif candidate.name == "torch_harmonics" and candidate.is_dir():
        root = candidate.parent
    else:
        root = candidate
    try:
        module_path.relative_to(root)
    except ValueError as exc:
        raise ComparisonUnavailable(
            f"torch-harmonics imported from {module_path}, outside requested source {root}"
        ) from exc
    return root


def load_torch_runtime(source_path: Path | None = None) -> TorchRuntime:
    """Load and verify a source checkout, never silently falling back to a wheel."""

    if source_path is not None:
        requested = source_path.resolve()
        if not requested.exists():
            raise ComparisonUnavailable(f"torch-harmonics source path does not exist: {requested}")
        sys.path.insert(0, str(requested))
    try:
        torch = importlib.import_module("torch")
        module = importlib.import_module("torch_harmonics")
    except ImportError as exc:
        raise ComparisonUnavailable(f"Torch/torch-harmonics is unavailable: {exc}") from exc
    module_path = Path(module.__file__).resolve()
    if source_path is None:
        try:
            root_text = subprocess.run(
                ["git", "-C", str(module_path.parent), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            requested = Path(root_text).resolve()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ComparisonUnavailable(
                "torch-harmonics must be imported from a Git source checkout; "
                "pass --torch-source explicitly"
            ) from exc
    root = _source_root(requested, module_path)
    git = _git_metadata(root)
    if git["git_sha"] is None:
        raise ComparisonUnavailable(f"torch-harmonics source is not a Git checkout: {root}")
    try:
        signature = str(inspect.signature(module.RealSHT))
    except (TypeError, ValueError) as exc:
        raise ComparisonUnavailable("cannot inspect torch_harmonics.RealSHT signature") from exc
    if "analysis" not in signature:
        raise ComparisonUnavailable(
            "installed torch-harmonics RealSHT has no analysis= parameter; "
            "the high-bandwidth source feature is required"
        )
    try:
        probe = module.RealSHT(
            3,
            4,
            lmax=2,
            mmax=2,
            grid="equiangular",
            norm="ortho",
            csphase=True,
            analysis="sampling-theorem",
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ComparisonUnavailable(
            "torch-harmonics RealSHT does not support analysis='sampling-theorem'"
        ) from exc
    del probe
    return TorchRuntime(
        torch=torch,
        module=module,
        source_path=root,
        module_path=module_path,
        git=git,
        distribution_version=_package_version("torch-harmonics"),
        real_sht_signature=signature,
        cuda_runtime=getattr(torch.version, "cuda", None),
    )


def _torch_real_dtype(runtime: TorchRuntime, dtype: DTypeSpec) -> Any:
    return runtime.torch.float32 if dtype.name == "float32" else runtime.torch.float64


def _torch_complex_dtype(runtime: TorchRuntime, dtype: DTypeSpec) -> Any:
    return runtime.torch.complex64 if dtype.name == "float32" else runtime.torch.complex128


@dataclass
class PreparedTorch:
    case: SHTCase
    dtype: DTypeSpec
    device: Any
    analysis_method: str
    runtime: TorchRuntime
    analysis_module: Any
    synthesis_module: Any
    setup_time_s: float
    validation_time_s: float
    map_input: Any | None = None
    coefficients_input: Any | None = None

    @property
    def identity(self) -> dict[str, Any]:
        result = dict(self.runtime.identity)
        result.update(
            {
                "device": str(self.device),
                "analysis_method": self.analysis_method,
            }
        )
        if str(self.device).startswith("cuda"):
            index = self.device.index
            if index is None:
                index = self.runtime.torch.cuda.current_device()
            result["cuda_device_index"] = int(index)
            result["cuda_device_name"] = self.runtime.torch.cuda.get_device_name(index)
            capability = self.runtime.torch.cuda.get_device_capability(index)
            result["cuda_compute_capability"] = f"{capability[0]}.{capability[1]}"
        else:
            result["cuda_device_index"] = None
            result["cuda_device_name"] = None
            result["cuda_compute_capability"] = None
        return result

    @property
    def actual_threads(self) -> int | None:
        return (
            int(self.runtime.torch.get_num_threads())
            if self.device.type == "cpu"
            else None
        )

    def set_coefficients(self, coefficients: NDArray[Any]) -> None:
        array = np.asarray(coefficients, dtype=self.dtype.complex)
        if array.shape != (self.case.lmax + 1, self.case.lmax + 1):
            raise ValueError(f"unexpected canonical coefficient shape {array.shape}")
        tensor = self.runtime.torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=_torch_complex_dtype(self.runtime, self.dtype))
        if tensor.dtype != _torch_complex_dtype(self.runtime, self.dtype) or tensor.device != self.device:
            raise RuntimeError("Torch coefficient input did not preserve requested dtype/device")
        self.coefficients_input = tensor

    def set_map(self, spatial: NDArray[Any]) -> None:
        array = np.asarray(spatial, dtype=self.dtype.real)
        if array.shape == (self.case.nlat, self.case.nlon):
            array = array[None, ...]
        if array.shape != (1, self.case.nlat, self.case.nlon):
            raise ValueError(f"unexpected Torch spatial shape {array.shape}")
        tensor = self.runtime.torch.from_numpy(np.ascontiguousarray(array))
        tensor = tensor.to(device=self.device, dtype=_torch_real_dtype(self.runtime, self.dtype))
        if tensor.dtype != _torch_real_dtype(self.runtime, self.dtype) or tensor.device != self.device:
            raise RuntimeError("Torch spatial input did not preserve requested dtype/device")
        self.map_input = tensor

    def analyze(self, spatial: Any | None = None) -> Any:
        tensor = self.map_input if spatial is None else spatial
        if tensor is None:
            raise RuntimeError("Torch spatial input was not prepared")
        return self.analysis_module(tensor)

    def synthesize(self, coefficients: Any | None = None) -> Any:
        tensor = self.coefficients_input if coefficients is None else coefficients
        if tensor is None:
            raise RuntimeError("Torch coefficient input was not prepared")
        return self.synthesis_module(tensor)

    def validate(self) -> None:
        if self.map_input is None or self.coefficients_input is None:
            raise RuntimeError("Torch validation inputs were not prepared")
        torch = self.runtime.torch
        expected_real = _torch_real_dtype(self.runtime, self.dtype)
        expected_complex = _torch_complex_dtype(self.runtime, self.dtype)
        if self.map_input.dtype != expected_real or self.map_input.device != self.device:
            raise RuntimeError("Torch map setup dtype/device assertion failed")
        if self.coefficients_input.dtype != expected_complex or self.coefficients_input.device != self.device:
            raise RuntimeError("Torch coefficient setup dtype/device assertion failed")
        with torch.inference_mode():
            coefficients = self.analyze(self.map_input)
            spatial = self.synthesize(self.coefficients_input)
        if coefficients.shape != (1, self.case.lmax + 1, self.case.mmax + 1):
            raise RuntimeError(f"Torch returned unexpected coefficient shape {coefficients.shape}")
        if spatial.shape != (1, self.case.nlat, self.case.nlon):
            raise RuntimeError(f"Torch returned unexpected map shape {spatial.shape}")
        if coefficients.dtype != expected_complex or coefficients.device != self.device:
            raise RuntimeError("Torch analysis output dtype/device assertion failed")
        if spatial.dtype != expected_real or spatial.device != self.device:
            raise RuntimeError("Torch synthesis output dtype/device assertion failed")


def prepare_torch(
    case: SHTCase,
    dtype: str,
    device: str,
    analysis_method: str,
    runtime: TorchRuntime,
    coefficients: NDArray[Any],
    spatial: NDArray[Any],
) -> PreparedTorch:
    if analysis_method not in TORCH_ANALYSIS_METHODS:
        raise ValueError(f"unknown Torch analysis method {analysis_method!r}")
    if device == "cuda" and not runtime.torch.cuda.is_available():
        raise ComparisonUnavailable("CUDA was requested but torch.cuda.is_available() is false")
    if device == "cuda":
        torch_device = runtime.torch.device("cuda", runtime.torch.cuda.current_device())
    else:
        torch_device = runtime.torch.device("cpu")
    spec = dtype_spec(dtype)
    grid_name = "equiangular" if case.grid == "cc" else "legendre-gauss"
    real_dtype = _torch_real_dtype(runtime, spec)
    start = time.perf_counter()
    analysis_module = runtime.module.RealSHT(
        case.nlat,
        case.nlon,
        lmax=case.lmax + 1,
        mmax=case.mmax + 1,
        grid=grid_name,
        norm="ortho",
        csphase=True,
        analysis=analysis_method,
    )
    synthesis_module = runtime.module.InverseRealSHT(
        case.nlat,
        case.nlon,
        lmax=case.lmax + 1,
        mmax=case.mmax + 1,
        grid=grid_name,
        norm="ortho",
        csphase=True,
    )
    analysis_module = analysis_module.to(device=torch_device, dtype=real_dtype).eval()
    synthesis_module = synthesis_module.to(device=torch_device, dtype=real_dtype).eval()
    setup_time_s = time.perf_counter() - start
    state = PreparedTorch(
        case,
        spec,
        torch_device,
        analysis_method,
        runtime,
        analysis_module,
        synthesis_module,
        setup_time_s,
        0.0,
    )
    state.set_coefficients(coefficients)
    state.set_map(spatial)
    validation_start = time.perf_counter()
    state.validate()
    state.validation_time_s = time.perf_counter() - validation_start
    return state


def _torch_cpu_or_cuda(device: str) -> str:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported Torch device {device!r}")
    return device


def _measure_cpu(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
    min_time: float,
    runtime: TorchRuntime | None = None,
) -> tuple[int, list[float], str]:
    if runtime is None:
        iterations, samples = measure(
            fn,
            warmup=warmup,
            repeat=repeat,
            min_time=min_time,
        )
    else:
        with runtime.torch.inference_mode():
            iterations, samples = measure(
                fn,
                warmup=warmup,
                repeat=repeat,
                min_time=min_time,
            )
    return iterations, samples, "perf_counter_ns-wall"


def _cuda_measure(
    fn: Callable[[], Any],
    runtime: TorchRuntime,
    device: Any,
    *,
    warmup: int,
    repeat: int,
    min_time: float,
) -> tuple[int, list[float], str]:
    torch = runtime.torch
    if device.type != "cuda":
        raise ValueError("CUDA event timing requires a CUDA device")

    def elapsed(iterations: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / 1000.0

    with torch.inference_mode(), torch.cuda.device(device):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        iterations = 1
        while True:
            elapsed_s = elapsed(iterations)
            if elapsed_s >= min_time or iterations >= 1 << 20:
                break
            scale = max(2, min(16, math.ceil(min_time / max(elapsed_s, 1.0e-12))))
            iterations *= scale
        samples: list[float] = []
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(repeat):
                samples.append(elapsed(iterations) / iterations)
        finally:
            if was_enabled:
                gc.enable()
    return iterations, samples, "cuda-event-compute"


def _timing_record_fields(
    iterations: int,
    samples: list[float],
    *,
    timing_method: str,
    warmup: int,
    repeat: int,
    min_time: float,
) -> dict[str, Any]:
    return {
        "timing_method": timing_method,
        "warmup": warmup,
        "repeat": repeat,
        "min_time_s": min_time,
        "iterations": iterations,
        "raw_timing_samples_s": samples,
        "samples_s": samples,
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _record_base(
    identity: dict[str, Any],
    case: SHTCase,
    dtype: str,
    device: str,
    threads: int,
    operation: str,
    *,
    record_type: str,
    analysis_method: str | None,
    process_isolated: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "record_type": record_type,
        **identity,
        "grid": case.grid,
        "case_name": case.name,
        "case": case.name,
        "nlat": case.nlat,
        "nlon": case.nlon,
        "grid_points": case.grid_points,
        "lmax": case.lmax,
        "mmax": case.mmax,
        "inclusive_lmax": case.lmax,
        "inclusive_mmax": case.mmax,
        "lmax_convention": "inclusive mathematical degree",
        "analysis_method": analysis_method,
        "analysis_variant": analysis_method,
        "dtype": dtype_spec(dtype).name,
        "spatial_dtype": dtype_spec(dtype).name,
        "spectral_dtype": "complex64"
        if dtype_spec(dtype).name == "float32"
        else "complex128",
        "device": device,
        "thread_cell_requested": threads,
        "threads_requested": threads if device == "cpu" else None,
        "actual_threads": None,
        "nthreads": identity.get("nthreads"),
        "torch_threads_requested": (
            threads
            if identity.get("backend") == "torch-harmonics" and device == "cpu"
            else None
        ),
        "torch_interop_threads": (
            1
            if identity.get("backend") == "torch-harmonics" and device == "cpu"
            else None
        ),
        "operation": operation,
        "process_isolated": process_isolated,
        "timing_method": None,
        "warmup": None,
        "repeat": None,
        "min_time_s": None,
        "iterations": None,
        "raw_timing_samples_s": None,
        "samples_s": None,
        "min_s": None,
        "median_s": None,
        "mean_s": None,
        "stdev_s": None,
        "setup_time_s": None,
        "setup_validation_s": None,
        "analysis_rel_l2": None,
        "analysis_max_abs": None,
        "analysis_bounded_rel": None,
        "analysis_compared_values": None,
        "analysis_reference_norm": None,
        "analysis_reference_max": None,
        "analysis_bounded_rel_floor": None,
        "synthesis_rel_l2": None,
        "synthesis_max_abs": None,
        "synthesis_bounded_rel": None,
        "synthesis_compared_values": None,
        "synthesis_reference_norm": None,
        "synthesis_reference_max": None,
        "synthesis_bounded_rel_floor": None,
        "roundtrip_rel_l2": None,
        "roundtrip_max_abs": None,
        "ducc_roundtrip_rel_l2": None,
        "torch_roundtrip_rel_l2": None,
        "accuracy_pass": None,
        "spectrum_kind": None,
        "mode_degree": None,
        "mode_order": None,
        "reference_backend": None,
        "notes": None,
    }
    return result


def _put_metrics(record: dict[str, Any], prefix: str, metrics: dict[str, Any]) -> None:
    mapping = {
        "rel_l2": f"{prefix}_rel_l2",
        "max_abs": f"{prefix}_max_abs",
        "bounded_rel": f"{prefix}_bounded_rel",
        "compared_values": f"{prefix}_compared_values",
        "reference_norm": f"{prefix}_reference_norm",
        "reference_max": f"{prefix}_reference_max",
        "bounded_rel_floor": f"{prefix}_bounded_rel_floor",
    }
    for source, target in mapping.items():
        record[target] = metrics[source]


def _accuracy_record(
    state: PreparedTorch,
    case: SHTCase,
    dtype: str,
    device: str,
    threads: int,
    process_isolated: bool,
    spectrum_kind: str,
    mode: tuple[int, int] | None,
    synthesis_metrics: dict[str, Any],
    analysis_metrics: dict[str, Any],
    torch_roundtrip: dict[str, Any],
    ducc_roundtrip: dict[str, Any],
    *,
    notes: str | None = None,
) -> dict[str, Any]:
    record = _record_base(
        state.identity,
        case,
        dtype,
        device,
        threads,
        "accuracy",
        record_type="accuracy",
        analysis_method=state.analysis_method,
        process_isolated=process_isolated,
    )
    _put_metrics(record, "analysis", analysis_metrics)
    _put_metrics(record, "synthesis", synthesis_metrics)
    record["roundtrip_rel_l2"] = torch_roundtrip["rel_l2"]
    record["roundtrip_max_abs"] = torch_roundtrip["max_abs"]
    record["ducc_roundtrip_rel_l2"] = ducc_roundtrip["rel_l2"]
    record["torch_roundtrip_rel_l2"] = torch_roundtrip["rel_l2"]
    record["spectrum_kind"] = spectrum_kind
    if mode is not None:
        record["mode_degree"], record["mode_order"] = mode
    record["reference_backend"] = "ducc0.sht"
    analysis_limits = accuracy_limits(dtype, "analysis")
    synthesis_limits = accuracy_limits(dtype, "synthesis")
    record["accuracy_pass"] = bool(
        analysis_metrics["rel_l2"] <= analysis_limits["rel_l2"]
        and analysis_metrics["max_abs"] <= analysis_limits["max_abs"]
        and synthesis_metrics["rel_l2"] <= synthesis_limits["rel_l2"]
        and synthesis_metrics["max_abs"] <= synthesis_limits["max_abs"]
    )
    record["setup_time_s"] = state.setup_time_s
    record["setup_validation_s"] = state.validation_time_s
    record["actual_threads"] = state.actual_threads
    record["notes"] = notes
    return record


def _calibration_modes(
    dtype: str,
) -> tuple[tuple[str, NDArray[Any], tuple[int, int]], ...]:
    return (
        ("Y00", single_mode_coefficients(1, 0, 0, dtype), (0, 0)),
        ("Y10", single_mode_coefficients(1, 1, 0, dtype), (1, 0)),
        ("Y11-real", single_mode_coefficients(1, 1, 1, dtype), (1, 1)),
        ("Y11-imag", _imaginary_mode(dtype), (1, 1)),
    )


def _imaginary_mode(dtype: str) -> NDArray[Any]:
    spec = dtype_spec(dtype)
    result = np.zeros((2, 2), dtype=spec.complex)
    result[1, 1] = 1.0j
    return result


def _make_calibration_record(
    state: PreparedTorch,
    cal_case: SHTCase,
    dtype: str,
    device: str,
    threads: int,
    name: str,
    mode: tuple[int, int],
    synth: dict[str, Any],
    analysis: dict[str, Any],
    process_isolated: bool,
) -> dict[str, Any]:
    record = _record_base(
        state.identity,
        cal_case,
        dtype,
        device,
        threads,
        "calibration",
        record_type="calibration",
        analysis_method=state.analysis_method,
        process_isolated=process_isolated,
    )
    _put_metrics(record, "analysis", analysis)
    _put_metrics(record, "synthesis", synth)
    record["spectrum_kind"] = name
    record["mode_degree"], record["mode_order"] = mode
    record["reference_backend"] = "ducc0.sht"
    record["setup_time_s"] = state.setup_time_s
    record["setup_validation_s"] = state.validation_time_s
    record["actual_threads"] = state.actual_threads
    record["accuracy_pass"] = _metrics_pass(analysis, dtype, "analysis") and _metrics_pass(
        synth, dtype, "synthesis"
    )
    record["notes"] = (
        "norm=ortho, csphase=True; direct Y00/Y10/Y11 convention calibration "
        "before random-spectrum comparison"
    )
    return record


def _run_calibration(
    case: SHTCase,
    dtype: str,
    device: str,
    threads: int,
    runtime: TorchRuntime,
    *,
    process_isolated: bool,
) -> list[dict[str, Any]]:
    cal_case = SHTCase(f"calibration-{case.nlat}x{case.nlon}", case.grid, case.nlat, case.nlon, 1)
    first = _calibration_modes(dtype)[0][1]
    ducc = prepare_ducc(cal_case, dtype, threads if device == "cpu" else 1)
    ducc.set_coefficients(first)
    initial_map = ducc.synthesis()
    torch_state = prepare_torch(
        cal_case,
        dtype,
        _torch_cpu_or_cuda(device),
        "sampling-theorem",
        runtime,
        first,
        initial_map,
    )
    records: list[dict[str, Any]] = []
    analysis_limits = accuracy_limits(dtype, "analysis")
    synthesis_limits = accuracy_limits(dtype, "synthesis")
    for name, coefficients, mode in _calibration_modes(dtype):
        ducc.set_coefficients(coefficients)
        reference_map = ducc.synthesis()
        ducc.set_spatial(reference_map)
        reference_coefficients = ducc.analysis()[0]
        torch_state.set_coefficients(coefficients)
        torch_state.set_map(reference_map)
        with runtime.torch.inference_mode():
            torch_map = torch_state.synthesize()
            torch_coefficients = torch_state.analyze()
        torch_map_np = torch_map.detach().cpu().numpy()
        torch_coeff_np = torch_coefficients.detach().cpu().numpy()
        synth = comparison_metrics(torch_map_np, reference_map, dtype)
        analysis = comparison_metrics(torch_coeff_np, ducc_to_batch(reference_coefficients), dtype)
        record = _make_calibration_record(
            torch_state,
            cal_case,
            dtype,
            device,
            threads,
            name,
            mode,
            synth,
            analysis,
            process_isolated,
        )
        records.append(record)
        if (
            synth["rel_l2"] > synthesis_limits["rel_l2"]
            or synth["max_abs"] > synthesis_limits["max_abs"]
            or analysis["rel_l2"] > analysis_limits["rel_l2"]
            or analysis["max_abs"] > analysis_limits["max_abs"]
        ):
            raise ConventionError(
                f"{case.name} {dtype} {device} {name} convention calibration failed: "
                f"analysis rel_l2={analysis['rel_l2']:.3e}, max_abs={analysis['max_abs']:.3e}; "
                f"synthesis rel_l2={synth['rel_l2']:.3e}, max_abs={synth['max_abs']:.3e}"
            )
    return records


def ducc_to_batch(coefficients: NDArray[Any]) -> NDArray[Any]:
    array = np.asarray(coefficients)
    return array if array.ndim == 3 else array[None, ...]


def _accuracy_for_spectrum(
    state: PreparedTorch,
    ducc: PreparedDUCC,
    coefficients: NDArray[Any],
    device: str,
    dtype: str,
    *,
    spectrum_kind: str,
    mode: tuple[int, int] | None,
    threads: int,
    process_isolated: bool,
) -> dict[str, Any]:
    ducc.set_coefficients(coefficients)
    reference_map = ducc.synthesis()
    ducc.set_spatial(reference_map)
    reference_coefficients = ducc.analysis()
    state.set_coefficients(coefficients)
    state.set_map(reference_map)
    with state.runtime.torch.inference_mode():
        torch_map = state.synthesize()
        torch_coefficients = state.analyze()
        torch_roundtrip = state.analyze(torch_map)
    torch_map_np = torch_map.detach().cpu().numpy()
    torch_coeff_np = torch_coefficients.detach().cpu().numpy()
    torch_roundtrip_np = torch_roundtrip.detach().cpu().numpy()
    synthesis = comparison_metrics(torch_map_np, reference_map, dtype)
    analysis = comparison_metrics(torch_coeff_np, reference_coefficients, dtype)
    torch_roundtrip_metrics = comparison_metrics(
        torch_roundtrip_np,
        coefficients[None, ...],
        dtype,
    )
    ducc_roundtrip_metrics = comparison_metrics(
        reference_coefficients,
        coefficients[None, ...],
        dtype,
    )
    notes = (
        "cross-backend synthesis: same canonical coefficients; cross-backend "
        "analysis: same DUCC-synthesized map; round trips retained as diagnostics"
    )
    return _accuracy_record(
        state,
        state.case,
        dtype,
        device,
        threads,
        process_isolated,
        spectrum_kind,
        mode,
        synthesis,
        analysis,
        torch_roundtrip_metrics,
        ducc_roundtrip_metrics,
        notes=notes,
    )


def _analysis_variants(case: SHTCase) -> tuple[str, ...]:
    if case.name == "cc-73x144-t36":
        return ("quadrature", "sampling-theorem")
    return ("sampling-theorem",)


def _performance_record(
    identity: dict[str, Any],
    case: SHTCase,
    dtype: str,
    device: str,
    threads: int,
    operation: str,
    analysis_method: str | None,
    process_isolated: bool,
    actual_threads: int | None,
    setup_time_s: float,
    setup_validation_s: float | None,
    timing: dict[str, Any],
    *,
    notes: str,
) -> dict[str, Any]:
    record = _record_base(
        identity,
        case,
        dtype,
        device,
        threads,
        operation,
        record_type="performance",
        analysis_method=analysis_method,
        process_isolated=process_isolated,
    )
    record.update(timing)
    record["actual_threads"] = actual_threads
    record["setup_time_s"] = setup_time_s
    record["setup_validation_s"] = setup_validation_s
    record["notes"] = notes
    return record


def _time_state(
    state: PreparedDUCC | PreparedTorch,
    operation: str,
    device: str,
    *,
    warmup: int,
    repeat: int,
    min_time: float,
) -> dict[str, Any]:
    method_name = {
        "analysis": "analyze" if isinstance(state, PreparedTorch) else "analysis",
        "synthesis": "synthesize" if isinstance(state, PreparedTorch) else "synthesis",
    }[operation]
    fn = getattr(state, method_name)
    if isinstance(state, PreparedTorch) and device == "cuda":
        iterations, samples, method = _cuda_measure(
            fn,
            state.runtime,
            state.device,
            warmup=warmup,
            repeat=repeat,
            min_time=min_time,
        )
    elif isinstance(state, PreparedTorch):
        iterations, samples, method = _measure_cpu(
            fn,
            warmup=warmup,
            repeat=repeat,
            min_time=min_time,
            runtime=state.runtime,
        )
    else:
        iterations, samples, method = _measure_cpu(
            fn,
            warmup=warmup,
            repeat=repeat,
            min_time=min_time,
        )
    return _timing_record_fields(
        iterations,
        samples,
        timing_method=method,
        warmup=warmup,
        repeat=repeat,
        min_time=min_time,
    )


def run_comparison_case(
    case: SHTCase,
    dtype: str,
    device: str,
    threads: int,
    backends: tuple[str, ...],
    runtime: TorchRuntime | None,
    *,
    seed: int,
    warmup: int,
    repeat: int,
    min_time: float,
    process_isolated: bool = True,
) -> list[dict[str, Any]]:
    """Run one isolated comparison cell and return rich records."""

    dtype = dtype_spec(dtype).name
    device = _torch_cpu_or_cuda(device)
    want_ducc = "ducc" in backends
    want_torch = "torch" in backends
    if device == "cuda" and want_ducc and not want_torch:
        raise ComparisonUnavailable(
            "DUCC has no CUDA comparison path; CUDA timing is Torch-only"
        )
    if want_torch and runtime is None:
        raise ComparisonUnavailable("Torch backend was selected without a verified source checkout")
    # When Torch is selected, DUCC is also needed as the cross-backend accuracy
    # oracle even though it has no CUDA timing path.
    ducc_for_reference = want_ducc or want_torch

    random_coefficients = canonical_coefficients(case.lmax, dtype, seed)
    mode_values = [
        (mode, single_mode_coefficients(case.lmax, mode[0], mode[1], dtype))
        for mode in required_single_modes(case)
    ]
    spectra: list[tuple[str, tuple[int, int] | None, NDArray[Any]]] = [
        ("mode", mode, coefficients) for mode, coefficients in mode_values
    ]
    spectra.append(("random", None, random_coefficients))
    first_coefficients = spectra[0][2] if spectra else random_coefficients

    ducc: PreparedDUCC | None = None
    states: dict[str, PreparedTorch] = {}
    records: list[dict[str, Any]] = []

    if ducc_for_reference:
        ducc = prepare_ducc(case, dtype, threads if device == "cpu" else 1)
        ducc.set_coefficients(first_coefficients)
        first_map = ducc.synthesis()
    else:
        rng = np.random.default_rng(seed + 1)
        first_map = rng.standard_normal(
            (1, case.nlat, case.nlon),
            dtype=dtype_spec(dtype).real,
        )

    if want_torch:
        assert runtime is not None
        if ducc is None:
            spatial_for_torch = first_map
        else:
            spatial_for_torch = first_map
        if want_ducc or want_torch:
            records.extend(
                _run_calibration(
                    case,
                    dtype,
                    device,
                    threads,
                    runtime,
                    process_isolated=process_isolated,
                )
            )
        for method in _analysis_variants(case):
            states[method] = prepare_torch(
                case,
                dtype,
                device,
                method,
                runtime,
                first_coefficients,
                spatial_for_torch,
            )

    if ducc is not None:
        ducc.set_coefficients(first_coefficients)
        current_map = ducc.synthesis()
    else:
        current_map = first_map

    if want_torch and ducc is not None:
        # Accuracy is performed completely before steady-state timing.  The
        # last spectrum is intentionally the common random input used below.
        for spectrum_kind, mode, coefficients in spectra:
            for state in states.values():
                records.append(
                    _accuracy_for_spectrum(
                        state,
                        ducc,
                        coefficients,
                        device,
                        dtype,
                        spectrum_kind=spectrum_kind,
                        mode=mode,
                        threads=threads,
                        process_isolated=process_isolated,
                    )
                )
            current_map = ducc.spatial
            assert current_map is not None
        # Every valid accuracy record is retained; do not turn a diagnostic
        # round trip into the cross-backend oracle.
        invalid = [
            record
            for record in records
            if record["record_type"] == "accuracy" and not record["accuracy_pass"]
        ]
        if invalid:
            first = invalid[0]
            raise AccuracyError(
                f"accuracy tolerance failed for {first['case_name']} {first['dtype']} "
                f"{first['device']} {first['analysis_method']} {first['spectrum_kind']}"
            )
    else:
        if ducc is not None:
            ducc.set_coefficients(random_coefficients)
            current_map = ducc.synthesis()
        for state in states.values():
            state.set_coefficients(random_coefficients)
            state.set_map(current_map)

    if ducc is not None and want_ducc and device == "cpu":
        ducc.set_coefficients(random_coefficients)
        ducc.set_spatial(current_map)
        for operation in ("analysis", "synthesis"):
            if operation not in {"analysis", "synthesis"}:
                continue
            timing = _time_state(
                ducc,
                operation,
                device,
                warmup=warmup,
                repeat=repeat,
                min_time=min_time,
            )
            records.append(
                _performance_record(
                    ducc.identity,
                    case,
                    dtype,
                    device,
                    threads,
                    operation,
                    None,
                    process_isolated,
                    ducc.threads,
                    ducc.setup_time_s,
                    None,
                    timing,
                    notes=(
                        "direct ducc0.sht scalar CC/GL transform; lmax/mmax are "
                        "inclusive; setup and coefficient packing excluded"
                    ),
                )
            )

    if want_torch:
        assert runtime is not None
        # The synthesis module is identical for all analysis variants.  Keep a
        # single inverse timing record per cell and identify analysis variants
        # only on analysis records.
        synthesis_state = states[_analysis_variants(case)[-1]]
        for method, state in states.items():
            if want_torch and (not want_ducc or ducc is not None):
                state.set_coefficients(random_coefficients)
                state.set_map(current_map)
            timing = _time_state(
                state,
                "analysis",
                device,
                warmup=warmup,
                repeat=repeat,
                min_time=min_time,
            )
            records.append(
                _performance_record(
                    state.identity,
                    case,
                    dtype,
                    device,
                    threads,
                    "analysis",
                    method,
                    process_isolated,
                    state.actual_threads,
                    state.setup_time_s,
                    state.validation_time_s,
                    timing,
                    notes=(
                        "direct torch-harmonics RealSHT; norm=ortho, csphase=True; "
                        f"analysis={method}; setup and NumPy/Torch conversion excluded"
                    ),
                )
            )
        synthesis_state.set_coefficients(random_coefficients)
        synthesis_state.set_map(current_map)
        timing = _time_state(
            synthesis_state,
            "synthesis",
            device,
            warmup=warmup,
            repeat=repeat,
            min_time=min_time,
        )
        records.append(
            _performance_record(
                synthesis_state.identity,
                case,
                dtype,
                device,
                threads,
                "synthesis",
                None,
                process_isolated,
                synthesis_state.actual_threads,
                synthesis_state.setup_time_s,
                synthesis_state.validation_time_s,
                timing,
                notes=(
                    "direct torch-harmonics InverseRealSHT; norm=ortho, csphase=True; "
                    "setup and NumPy/Torch conversion excluded"
                ),
            )
        )
    return records


def _json_output_path(path: Path) -> Path:
    return path if path.suffix.lower() == ".json" else path.with_suffix(".json")


def write_compare_results(payload: dict[str, Any], output: Path) -> tuple[Path, Path]:
    json_path = _json_output_path(output)
    csv_path = json_path.with_suffix(".csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    records = payload.get("records", [])
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {}
            for key in fieldnames:
                value = record.get(key)
                row[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
            writer.writerow(row)
    return json_path, csv_path


def _worker_payload(
    records: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    cases: list[str],
    dtypes: list[str],
    backends: list[str],
    device: str,
    threads: int,
    spharmgrid_reference: Path | None,
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "records": records,
        "skipped": skipped,
        "failures": failures,
        "worker": {
            "pid": os.getpid(),
            "cases": cases,
            "dtypes": dtypes,
            "backends": backends,
            "device": device,
            "thread_cell_requested": threads,
            "sht_bench": _sht_bench_provenance(),
            "spharmgrid_reference": reference_provenance(spharmgrid_reference),
            "environment": collect_environment(include_cpu_controls=True),
        },
    }


def run_compare_worker(args: Any) -> int:
    """Run one comparison cell; called only in a fresh process."""

    threads = int(args.threads[0] if isinstance(args.threads, list) else args.threads)
    _set_thread_environment_local(threads)
    device = args.torch_device[0] if isinstance(args.torch_device, list) else args.torch_device
    backends = tuple(args.backend)
    reference = Path(args.spharmgrid_reference).resolve() if args.spharmgrid_reference else default_spharmgrid_reference()
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    runtime: TorchRuntime | None = None
    effective_backends = backends
    fatal_convention_failure = False
    if "torch" in backends:
        try:
            if args.torch_source is not None:
                source = Path(args.torch_source)
            else:
                source = None
            runtime = load_torch_runtime(source)
            if device == "cuda" and not runtime.torch.cuda.is_available():
                raise ComparisonUnavailable("CUDA requested but no CUDA device is available")
            if device == "cpu":
                runtime.torch.set_num_threads(threads)
                try:
                    runtime.torch.set_num_interop_threads(1)
                except RuntimeError as exc:
                    raise ComparisonUnavailable(
                        f"could not set Torch inter-op threads in isolated worker: {exc}"
                    ) from exc
        except ComparisonUnavailable as exc:
            for case_name in args.case:
                for dtype in args.dtype:
                    skipped.append(
                        {
                            "backend": "torch-harmonics",
                            "case_name": case_name,
                            "dtype": dtype,
                            "device": device,
                            "thread_cell_requested": threads,
                            "reason": str(exc),
                        }
                    )
            runtime = None
            effective_backends = tuple(name for name in backends if name != "torch")

    if not effective_backends:
        payload = _worker_payload(
            records,
            skipped,
            failures,
            cases=args.case,
            dtypes=args.dtype,
            backends=list(backends),
            device=device,
            threads=threads,
            spharmgrid_reference=reference,
        )
        write_compare_results(payload, Path(args.output))
        return 0

    for case_name in args.case:
        case = next(case for case in HIGH_BANDWIDTH_CC_CASES if case.name == case_name)
        for dtype in args.dtype:
            try:
                cell_records = run_comparison_case(
                    case,
                    dtype,
                    device,
                    threads,
                    effective_backends,
                    runtime,
                    seed=args.seed,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    min_time=args.min_time,
                )
                records.extend(
                    record
                    for record in cell_records
                    if record["operation"] in args.operation
                    or record["record_type"] in {"calibration", "accuracy"}
                )
            except ComparisonUnavailable as exc:
                skipped.append(
                    {
                        "backend": ",".join(backends),
                        "case_name": case.name,
                        "dtype": dtype,
                        "device": device,
                        "thread_cell_requested": threads,
                        "reason": str(exc),
                    }
                )
            except (AccuracyError, ConventionError, RuntimeError, ValueError) as exc:
                failure = {
                    "backend": ",".join(backends),
                    "case_name": case.name,
                    "dtype": dtype,
                    "device": device,
                    "thread_cell_requested": threads,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                print(f"comparison failure: {failure}", file=sys.stderr)
                if isinstance(exc, ConventionError):
                    fatal_convention_failure = True
                    break
        if fatal_convention_failure:
            break

    payload = _worker_payload(
        records,
        skipped,
        failures,
        cases=args.case,
        dtypes=args.dtype,
        backends=list(backends),
        device=device,
        threads=threads,
        spharmgrid_reference=reference,
    )
    write_compare_results(payload, Path(args.output))
    return 1 if failures else 0


def _set_thread_environment_local(threads: int) -> None:
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "DUCC0_NUM_THREADS",
    ):
        os.environ[name] = value


def run_compare(args: Any) -> int:
    """Orchestrate fresh worker processes and serialize one rich comparison."""

    output = Path(args.output)
    devices = list(args.torch_device)
    thread_values = list(args.threads)
    jobs: list[tuple[str, int]] = []
    for device in devices:
        values = thread_values if device == "cpu" else thread_values[:1]
        jobs.extend((device, threads) for threads in values)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    worker_provenance: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="sht-bench-compare-") as temp_name:
        temporary = Path(temp_name)
        for index, (device, threads) in enumerate(jobs):
            fragment = temporary / f"worker-{index}.json"
            command = [
                sys.executable,
                "-m",
                "sht_bench.cli",
                "compare-worker",
                "--backend",
                ",".join(args.backend),
                "--case",
                ",".join(args.case),
                "--dtype",
                ",".join(args.dtype),
                "--operation",
                ",".join(args.operation),
                "--threads",
                str(threads),
                "--torch-device",
                device,
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
                "--min-time",
                str(args.min_time),
                "--seed",
                str(args.seed),
                "--output",
                str(fragment),
            ]
            if args.torch_source is not None:
                command.extend(["--torch-source", str(args.torch_source)])
            if args.spharmgrid_reference is not None:
                command.extend(["--spharmgrid-reference", str(args.spharmgrid_reference)])
            env = os.environ.copy()
            value = str(threads)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "DUCC0_NUM_THREADS",
            ):
                env[name] = value
            print("+", " ".join(command), flush=True)
            completed = subprocess.run(command, env=env, check=False)
            if not fragment.exists():
                failures.append(
                    {
                        "device": device,
                        "thread_cell_requested": threads,
                        "reason": f"worker exited {completed.returncode} without a result file",
                    }
                )
                continue
            payload = json.loads(fragment.read_text())
            records.extend(payload.get("records", []))
            skipped.extend(payload.get("skipped", []))
            failures.extend(payload.get("failures", []))
            if payload.get("worker"):
                worker_provenance.append(payload["worker"])
            if completed.returncode != 0 and not payload.get("failures"):
                failures.append(
                    {
                        "device": device,
                        "thread_cell_requested": threads,
                        "reason": f"worker exited {completed.returncode}",
                    }
                )

    reference = Path(args.spharmgrid_reference).resolve() if args.spharmgrid_reference else default_spharmgrid_reference()
    torch_source_metadata = (
        _git_metadata(Path(args.torch_source)) if args.torch_source else None
    )
    if torch_source_metadata is None:
        for record in records:
            if record.get("backend") == "torch-harmonics":
                torch_source_metadata = _git_metadata(Path(record["backend_source"]))
                break
    payload = {
        "schema": RESULT_SCHEMA,
        "command": "compare",
        "requested": {
            "backend": args.backend,
            "case": args.case,
            "dtype": args.dtype,
            "operation": args.operation,
            "threads": args.threads,
            "torch_device": args.torch_device,
            "torch_source": str(args.torch_source) if args.torch_source else None,
            "spharmgrid_reference": str(reference) if reference else None,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "min_time": args.min_time,
        },
        "sht_bench": _sht_bench_provenance(),
        "spharmgrid_reference": reference_provenance(reference),
        "torch_harmonics_source": torch_source_metadata,
        "workers": worker_provenance,
        "records": records,
        "skipped": skipped,
        "failures": failures,
    }
    json_path, csv_path = write_compare_results(payload, output)
    print(f"wrote {json_path} and {csv_path}")
    if skipped:
        print("\nUnsupported/unavailable combinations:", file=sys.stderr)
        for item in skipped:
            print(f"  - {item}", file=sys.stderr)
    if failures:
        print("\nComparison failures:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
    if args.strict and (skipped or failures):
        return 1
    return 0 if records else 1


__all__ = [
    "DEFAULT_SEED",
    "HIGH_BANDWIDTH_CC_CASES",
    "RESULT_SCHEMA",
    "AccuracyError",
    "ComparisonUnavailable",
    "ConventionError",
    "accuracy_limits",
    "canonical_coefficients",
    "comparison_metrics",
    "dtype_spec",
    "ducc_packed_index",
    "ducc_packed_size",
    "ducc_to_rectangular",
    "load_torch_runtime",
    "prepare_ducc",
    "prepare_torch",
    "rectangular_to_ducc",
    "required_single_modes",
    "run_compare",
    "run_compare_worker",
    "run_comparison_case",
    "write_compare_results",
]
