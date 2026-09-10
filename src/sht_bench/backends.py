from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .grids import ATMOSPHERIC_GL_GRIDS, grid_shape

Transform = Callable[[], object]
GRID_NAMES = ("gl", "cc")
SUPPORTED_GRIDS: dict[str, tuple[str, ...]] = {
    "ducc": ("gl", "cc"),
    "shtns": ("gl", "cc"),
    "pyshtools": ("gl",),
    "pyspharm": ("gl", "cc"),
}


@dataclass(frozen=True)
class PreparedCase:
    backend: str
    backend_version: str
    grid: str
    lmax: int
    nlat: int
    nlon: int
    threads_requested: int
    threads_actual: int | None
    spatial_dtype: str
    spectral_dtype: str
    analysis: Transform
    synthesis: Transform
    notes: str = ""


def nalm(lmax: int) -> int:
    return (lmax + 1) * (lmax + 2) // 2


def _require_grid(backend: str, grid: str) -> None:
    if grid not in SUPPORTED_GRIDS[backend]:
        raise ValueError(f"{backend} does not support benchmark grid {grid!r}")


def _package_version(dist_name: str) -> str:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "unknown"


def _random_real_alm(rng: np.random.Generator, lmax: int) -> NDArray[np.complex128]:
    alm = (
        rng.standard_normal(nalm(lmax))
        + 1j * rng.standard_normal(nalm(lmax))
    ).astype(np.complex128)
    alm[: lmax + 1] = alm[: lmax + 1].real
    return alm


def prepare_ducc(lmax: int, threads: int, seed: int, grid: str = "gl") -> PreparedCase:
    _require_grid("ducc", grid)
    ducc0 = import_module("ducc0")
    sht = ducc0.sht
    rng = np.random.default_rng(seed)
    nlat, nlon = grid_shape(lmax, grid)
    geometry = {"gl": "GL", "cc": "CC"}[grid]

    alm = _random_real_alm(rng, lmax)[None, :]
    spatial = rng.standard_normal((1, nlat, nlon), dtype=np.float64)

    def synthesis() -> object:
        return sht.synthesis_2d(
            alm=alm,
            spin=0,
            lmax=lmax,
            mmax=lmax,
            geometry=geometry,
            ntheta=nlat,
            nphi=nlon,
            nthreads=threads,
        )

    def analysis() -> object:
        return sht.analysis_2d(
            map=spatial,
            spin=0,
            lmax=lmax,
            mmax=lmax,
            geometry=geometry,
            nthreads=threads,
        )

    out_grid = np.asarray(synthesis())
    out_alm = np.asarray(analysis())
    if out_grid.shape != (1, nlat, nlon):
        raise RuntimeError(f"DUCC returned unexpected map shape {out_grid.shape}")
    if out_alm.shape != (1, nalm(lmax)):
        raise RuntimeError(f"DUCC returned unexpected alm shape {out_alm.shape}")

    return PreparedCase(
        backend="ducc",
        backend_version=_package_version("ducc0"),
        grid=grid,
        lmax=lmax,
        nlat=nlat,
        nlon=nlon,
        threads_requested=threads,
        threads_actual=threads,
        spatial_dtype=str(spatial.dtype),
        spectral_dtype=str(alm.dtype),
        analysis=analysis,
        synthesis=synthesis,
        notes=f"ducc0.sht {geometry} scalar transform; explicit nthreads",
    )


def prepare_shtns(lmax: int, threads: int, seed: int, grid: str = "gl") -> PreparedCase:
    _require_grid("shtns", grid)
    shtns = import_module("shtns")
    rng = np.random.default_rng(seed)
    nlat, nlon = grid_shape(lmax, grid)

    norm = shtns.sht_orthonormal | shtns.SHT_NO_CS_PHASE
    sh = shtns.sht(lmax, lmax, mres=1, norm=norm, nthreads=threads)
    grid_flag = shtns.sht_gauss if grid == "gl" else shtns.sht_reg_poles
    flags = grid_flag | shtns.SHT_PHI_CONTIGUOUS
    sh.set_grid(nlat=nlat, nphi=nlon, flags=flags, polar_opt=0.0)

    if (sh.nlat, sh.nphi) != (nlat, nlon):
        raise RuntimeError(
            f"SHTns changed requested grid {(nlat, nlon)} to {(sh.nlat, sh.nphi)}"
        )

    alm = _random_real_alm(rng, lmax)
    if alm.shape != (sh.nlm,):
        raise RuntimeError(f"SHTns nlm={sh.nlm}, expected {alm.size}")
    spatial = rng.standard_normal((nlat, nlon), dtype=np.float64)

    def synthesis() -> object:
        return sh.synth(alm)

    def analysis() -> object:
        return sh.analys(spatial)

    out_grid = np.asarray(synthesis())
    out_alm = np.asarray(analysis())
    if out_grid.shape != (nlat, nlon):
        raise RuntimeError(f"SHTns returned unexpected map shape {out_grid.shape}")
    if out_alm.shape != (nalm(lmax),):
        raise RuntimeError(f"SHTns returned unexpected alm shape {out_alm.shape}")

    grid_note = "Gauss" if grid == "gl" else "regular-poles/Clenshaw-Curtis"
    return PreparedCase(
        backend="shtns",
        backend_version=_package_version("shtns"),
        grid=grid,
        lmax=lmax,
        nlat=nlat,
        nlon=nlon,
        threads_requested=threads,
        threads_actual=None,
        spatial_dtype=str(spatial.dtype),
        spectral_dtype=str(alm.dtype),
        analysis=analysis,
        synthesis=synthesis,
        notes=(
            f"SHTns {grid_note}, phi-contiguous, orthonormal/no-CS, polar_opt=0; "
            "setup excluded; thread count requested through constructor"
        ),
    )


def prepare_pyshtools(lmax: int, threads: int, seed: int, grid: str = "gl") -> PreparedCase:
    _require_grid("pyshtools", grid)
    pysh = import_module("pyshtools")
    rng = np.random.default_rng(seed)
    nlat, nlon = grid_shape(lmax, grid)

    # Native SHTOOLS GLQ routines require nlon == 2*nlat - 1. Atmospheric
    # Gaussian grids use an even number of longitudes, so the default common
    # atmospheric GL cases are excluded by the benchmark driver.
    if lmax in ATMOSPHERIC_GL_GRIDS:
        raise ValueError(
            "native SHTOOLS GLQ requires nlon=2*nlat-1 and cannot use the "
            f"common atmospheric grid {nlat}x{nlon}"
        )

    zero, weights = pysh.expand.SHGLQ(lmax)
    spatial = rng.standard_normal((nlat, nlon), dtype=np.float64)
    cilm = rng.standard_normal((2, lmax + 1, lmax + 1), dtype=np.float64)
    for ell in range(lmax + 1):
        cilm[:, ell, ell + 1 :] = 0.0
    cilm[1, :, 0] = 0.0

    def synthesis() -> object:
        return pysh.expand.MakeGridGLQ(
            cilm,
            zero,
            lmax=lmax,
            norm=4,
            csphase=1,
            lmax_calc=lmax,
            extend=False,
        )

    def analysis() -> object:
        return pysh.expand.SHExpandGLQ(
            spatial,
            weights,
            zero,
            norm=4,
            csphase=1,
            lmax_calc=lmax,
        )

    out_grid = np.asarray(synthesis())
    out_cilm = np.asarray(analysis())
    if out_grid.shape != (nlat, nlon):
        raise RuntimeError(f"pyshtools returned unexpected map shape {out_grid.shape}")
    if out_cilm.shape != (2, lmax + 1, lmax + 1):
        raise RuntimeError(f"pyshtools returned unexpected coeff shape {out_cilm.shape}")

    return PreparedCase(
        backend="pyshtools",
        backend_version=_package_version("pyshtools"),
        grid=grid,
        lmax=lmax,
        nlat=nlat,
        nlon=nlon,
        threads_requested=threads,
        threads_actual=None,
        spatial_dtype=str(spatial.dtype),
        spectral_dtype=str(cilm.dtype),
        analysis=analysis,
        synthesis=synthesis,
        notes=(
            "native pyshtools/SHTOOLS GLQ low-level routines; orthonormal/no-CS; "
            "no explicit per-call thread control"
        ),
    )


def prepare_pyspharm(lmax: int, threads: int, seed: int, grid: str = "gl") -> PreparedCase:
    _require_grid("pyspharm", grid)
    spharm = import_module("spharm")
    rng = np.random.default_rng(seed)
    nlat, nlon = grid_shape(lmax, grid)

    gridtype = "gaussian" if grid == "gl" else "regular"
    sh = spharm.Spharmt(nlon, nlat, gridtype=gridtype, legfunc="stored")
    spatial = rng.standard_normal((nlat, nlon)).astype(np.float32)
    coeff = _random_real_alm(rng, lmax).astype(np.complex64)

    def synthesis() -> object:
        return sh.spectogrd(coeff)

    def analysis() -> object:
        return sh.grdtospec(spatial, ntrunc=lmax)

    out_grid = np.asarray(synthesis())
    out_coeff = np.asarray(analysis())
    if out_grid.shape != (nlat, nlon):
        raise RuntimeError(f"pyspharm returned unexpected map shape {out_grid.shape}")
    if out_coeff.shape != (nalm(lmax),):
        raise RuntimeError(f"pyspharm returned unexpected coeff shape {out_coeff.shape}")

    return PreparedCase(
        backend="pyspharm",
        backend_version=_package_version("pyspharm-syl"),
        grid=grid,
        lmax=lmax,
        nlat=nlat,
        nlon=nlon,
        threads_requested=threads,
        threads_actual=None,
        spatial_dtype=str(spatial.dtype),
        spectral_dtype=str(coeff.dtype),
        analysis=analysis,
        synthesis=synthesis,
        notes=(
            f"pyspharm-syl/Spherepack {gridtype} grid with stored Legendre functions; "
            "float32/complex64 API; no explicit thread control"
        ),
    )


PREPARERS = {
    "ducc": prepare_ducc,
    "shtns": prepare_shtns,
    "pyshtools": prepare_pyshtools,
    "pyspharm": prepare_pyspharm,
}
