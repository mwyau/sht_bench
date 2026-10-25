"""Versioned CC/GL profiles and cell planning for the source resolution sweep."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from typing import Iterator

import numpy as np

from .source_adapters import Workload, capability

PROFILE_SCHEMA = 'sht_bench.source_resolution_profile.v2'
BACKENDS = ('ducc', 'shtns', 'rustsht', 'pyspharm-cpp', 'pyspharm-syl')
OPERATIONS = ('analysis', 'synthesis')
SPINS = (0, 1)
PRECISIONS = ('float64', 'float32')
THREADS = (1, 2, 4, 8)

CC_PROFILE = (
    {'resolution': '2.5°', 'nlat': 73, 'nlon': 144, 'lmax': 36},
    {'resolution': '2.0°', 'nlat': 91, 'nlon': 180, 'lmax': 45},
    {'resolution': '1.5°', 'nlat': 121, 'nlon': 240, 'lmax': 60},
    {'resolution': '1.0°', 'nlat': 181, 'nlon': 360, 'lmax': 90},
    {'resolution': '0.75°', 'nlat': 241, 'nlon': 480, 'lmax': 120},
    {'resolution': '0.5°', 'nlat': 361, 'nlon': 720, 'lmax': 180},
    {'resolution': '0.25°', 'nlat': 721, 'nlon': 1440, 'lmax': 360},
)

GL_PROFILE = tuple(
    {
        'N': n,
        'nlat': 2 * n,
        'nlon': 4 * n,
        'lmax': n,
        'nominal_longitude_resolution_deg': 90.0 / n,
    }
    for n in (32, 45, 64, 90, 128, 180, 320)
)

BACKEND_PRECISIONS = {
    'ducc': PRECISIONS,
    'shtns': ('float64',),
    'rustsht': PRECISIONS,
    'pyspharm-cpp': PRECISIONS,
    'pyspharm-syl': ('float32',),
}


def colatitude_nodes(grid: str, nlat: int) -> np.ndarray:
    """Return the independent north-to-south colatitudes used by validation."""
    if grid == 'CC':
        return np.linspace(0.0, math.pi, nlat, dtype=np.float64)
    if grid == 'GL':
        mu, _ = np.polynomial.legendre.leggauss(nlat)
        return np.arccos(mu[::-1])
    raise ValueError(f'unsupported resolution grid {grid!r}')


def latitude_nodes(grid: str, nlat: int) -> np.ndarray:
    """Return geographic latitude in radians, north to south."""
    if grid == 'CC':
        return math.pi / 2 - colatitude_nodes(grid,nlat)
    mu, _ = np.polynomial.legendre.leggauss(nlat)
    return np.arcsin(mu[::-1])


def resolution_cases(grids: tuple[str, ...] = ('CC', 'GL'), *, endpoints_only: bool = False) -> list[dict]:
    """Build geometry records, including the exact latitude coordinate arrays."""
    cases = []
    for grid in grids:
        if grid == 'CC':
            samples = CC_PROFILE[:1] if endpoints_only else CC_PROFILE
            for item in samples:
                identity = f"CC-{item['resolution'].replace('°', 'deg')}"
                case = {
                    'case_id': identity,
                    'grid': 'CC',
                    **item,
                    'nominal_longitude_resolution_deg': 360.0 / item['nlon'],
                    'latitude_nodes': 'endpoint-inclusive Clenshaw-Curtis, north-to-south',
                }
                case['latitude_coordinates_radians'] = latitude_nodes('CC', item['nlat']).tolist()
                case['colatitude_coordinates_radians'] = colatitude_nodes('CC', item['nlat']).tolist()
                cases.append(case)
        elif grid == 'GL':
            samples = GL_PROFILE[:1] if endpoints_only else GL_PROFILE
            for item in samples:
                case = {
                    'case_id': f"GL-N{item['N']}",
                    'grid': 'GL',
                    **item,
                    'resolution': f"GL N={item['N']}",
                    'latitude_nodes': 'Gauss-Legendre, north-to-south',
                }
                case['latitude_coordinates_radians'] = latitude_nodes('GL', item['nlat']).tolist()
                case['colatitude_coordinates_radians'] = colatitude_nodes('GL', item['nlat']).tolist()
                cases.append(case)
        else:
            raise ValueError(f'unsupported resolution grid {grid!r}')
    return cases


def selected_threads(backend: str, requested: tuple[int, ...]) -> tuple[int, ...]:
    if backend in ('pyspharm', 'pyspharm-syl'):
        return (1,) if 1 in requested else ()
    return requested


def selected_precisions(backend: str, requested: tuple[str, ...]) -> tuple[str, ...]:
    allowed = BACKEND_PRECISIONS[backend]
    return tuple(p for p in requested if p in allowed)


def case_reference(case: dict) -> dict:
    coordinates = np.asarray(case['latitude_coordinates_radians'], dtype='<f8')
    colatitudes = np.asarray(case['colatitude_coordinates_radians'], dtype='<f8')
    return {
        key:value for key,value in case.items()
        if key not in ('latitude_coordinates_radians','colatitude_coordinates_radians')
    } | {
        'latitude_coordinate_count': int(coordinates.size),
        'latitude_coordinates_sha256': hashlib.sha256(coordinates.tobytes()).hexdigest(),
        'latitude_coordinates_artifact': 'resolution-profile.json:cases[].latitude_coordinates_radians',
        'colatitude_coordinates_sha256': hashlib.sha256(colatitudes.tobytes()).hexdigest(),
        'colatitude_coordinates_artifact': 'resolution-profile.json:cases[].colatitude_coordinates_radians',
    }


def workload_for(case: dict, lmax: int, mmax: int, spin: int, precision: str, operation: str) -> Workload:
    return Workload(
        grid=case['grid'], nlat=case['nlat'], nlon=case['nlon'],
        lmax=lmax, mmax=mmax, spin=spin, precision=precision,
        operation=operation, geometry_group=case['case_id'],
    )


def planned_cells(
    cases: list[dict], *, backends: tuple[str, ...] = BACKENDS,
    threads: tuple[int, ...] = THREADS, precisions: tuple[str, ...] = PRECISIONS,
    mmax_modes: tuple[str, ...] = ('full', 'half'), rounds: int = 3,
) -> Iterator[dict]:
    """Yield all requested operation cells, retaining native API exclusions."""
    for case in cases:
        lmax = case['lmax']
        for mode in mmax_modes:
            mmax = lmax if mode == 'full' else lmax // 2
            for spin in SPINS:
                for operation in OPERATIONS:
                    for backend in backends:
                        backend_threads = selected_threads(backend, threads)
                        for precision in selected_precisions(backend, precisions):
                            workload = workload_for(case, lmax, mmax, spin, precision, operation)
                            reason = capability(backend, workload)
                            for round_index in range(rounds):
                                for q in backend_threads:
                                    yield {
                                        'backend': backend,
                                        'workload': workload.record(),
                                        'threads_requested': q,
                                        'round': round_index,
                                        'mmax_mode': mode,
                                        'resolution_case': case_reference(case),
                                        'static_status': 'unsupported' if reason else 'planned',
                                        'static_reason': reason,
                                    }


def plan_counts(cells: Iterator[dict]) -> dict:
    counts = {'planned': 0, 'unsupported': 0, 'total': 0}
    by_backend = {}
    for cell in cells:
        status = cell['static_status']
        counts[status] += 1
        counts['total'] += 1
        entry = by_backend.setdefault(cell['backend'], {'planned': 0, 'unsupported': 0, 'total': 0})
        entry[status] += 1
        entry['total'] += 1
    counts['by_backend'] = by_backend
    return counts


def validation_workload(cell: dict) -> Workload:
    """Map either timed operation to the shared analysis+synthesis gate key."""
    return replace(Workload(**cell['workload']), operation='synthesis', allocation='allocating')
