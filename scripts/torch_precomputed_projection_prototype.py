"""Validate a precomposed equiangular latitude projection on tiny grids.

This is a research-only prototype for the PyTorch-specific precomputed
projection experiment.  It deliberately builds the small dense resampling
matrix ``R`` so that the construction can be checked against the exact
identity ``K = W @ R`` before production code is changed.  It is not a
production interpolation implementation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

BASE_SHA = "3278fb669483d04537aa60c87fb0754d989e2e70"
SOURCE = Path(__file__).resolve().parents[2] / "torch-harmonics"


def _load_dense_namespace(repository: Path, base_sha: str) -> dict[str, Any]:
    source = subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{base_sha}:torch_harmonics/sht.py"],
        text=True,
    )
    namespace: dict[str, Any] = {
        "__name__": "torch_harmonics.sht_dense_reference",
        "__package__": "torch_harmonics",
    }
    exec(  # noqa: S102 - this intentionally loads the source-pinned oracle
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


def _phase(module: Any) -> torch.Tensor:
    phase = module._latitude_shift_phase
    return torch.complex(phase[0], phase[1])


def _effective_from_runtime_fold(
    module: Any, vector: bool, row_chunk: int = 4
) -> torch.Tensor:
    """Apply B's latitude operator to the rows of its projection tensor.

    ``_fold_resampled_latitude`` applies ``U = Q_e + A* Q_o A`` to a column
    vector.  The projection needs ``P U``.  U is Hermitian by construction,
    so ``conj(U @ P.T).T`` is the desired matrix.  This uses the existing
    Fourier/folding structure without materializing a generic interpolation
    matrix.  The Legendre-degree rows are folded in bounded chunks to avoid
    a large CPU FFT workspace during construction.
    """

    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")

    from torch_harmonics.sht import _fold_resampled_latitude

    weights = module.weights
    if vector:
        # (derivative component, degree, order, latitude)
        rows = weights.permute(0, 2, 1, 3)
    else:
        # (degree, order, latitude)
        rows = weights.permute(1, 0, 2)
    chunks = []
    degree_axis = 1 if vector else 0
    for start in range(0, rows.shape[degree_axis], row_chunk):
        stop = min(start + row_chunk, rows.shape[degree_axis])
        rows_chunk = rows[:, start:stop] if vector else rows[start:stop]
        chunks.append(
            _fold_resampled_latitude(
                rows_chunk,
                module._parity_signs,
                module._quadrature_weights,
                module._midpoint_weights,
                _phase(module),
            )
        )
    folded = torch.cat(chunks, dim=degree_axis).conj()
    del chunks, rows
    if vector:
        return folded.permute(0, 2, 1, 3).contiguous()
    return folded.permute(1, 0, 2).contiguous()


def _dense_resampling_matrix(
    dense: dict[str, Any], nlat: int, mmax: int, vector: bool
) -> torch.Tensor:
    """Build R[m, dense_latitude, original_latitude] for a tiny grid."""

    # The leading basis axis lets the source helper operate on all original
    # latitude basis vectors in one call while retaining order as its second
    # axis.  This is intentionally only used for tiny validation cases.
    basis = torch.eye(nlat, dtype=torch.float64).unsqueeze(1).expand(nlat, mmax, nlat)
    signs = torch.ones(mmax, 1, dtype=torch.int8)
    if vector:
        signs[::2] = -1
    else:
        signs[1::2] = -1
    extended = dense["_periodic_latitude_extension"](basis, signs)
    upsampled = dense["_fourier_upsample_latitude"](
        extended, target_length=2 * extended.shape[-1]
    )
    dense_samples = upsampled[..., : 2 * nlat - 1]
    return dense_samples.permute(1, 2, 0).contiguous()


def _direct_effective(
    dense_module: Any, resampling: torch.Tensor, vector: bool
) -> torch.Tensor:
    """Compute K_direct = W_dense @ R for the tiny explicit matrix."""

    weights = dense_module.weights.to(resampling.dtype)
    if vector:
        return torch.einsum("dmlq,mqj->dmlj", weights, resampling).contiguous()
    return torch.einsum("mlq,mqj->mlj", weights, resampling).contiguous()


def _runtime_folded_projection(
    module: Any, values: torch.Tensor, vector: bool
) -> torch.Tensor:
    from torch_harmonics.sht import _fold_resampled_latitude

    return _fold_resampled_latitude(
        values,
        module._parity_signs,
        module._quadrature_weights,
        module._midpoint_weights,
        _phase(module),
    )


def _resample_forward(module: Any, values: torch.Tensor) -> torch.Tensor:
    from torch_harmonics.sht import (
        _fourier_shift_latitude,
        _periodic_latitude_extension,
    )

    extended = _periodic_latitude_extension(values, module._parity_signs)
    return _fourier_shift_latitude(extended, _phase(module))[
        ..., : values.shape[-1] - 1
    ]


def _resample_adjoint(module: Any, values: torch.Tensor) -> torch.Tensor:
    from torch_harmonics.sht import (
        _fourier_shift_latitude_adjoint,
        _periodic_latitude_extension_adjoint,
    )

    padded = torch.cat((values, torch.zeros_like(values)), dim=-1)
    shifted = _fourier_shift_latitude_adjoint(padded, _phase(module))
    return _periodic_latitude_extension_adjoint(shifted, module._parity_signs)


def _scalar_contract(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...mk,mlk->...lm", values, weights.to(values.dtype))


def _vector_contract(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w0, w1 = weights
    spheroidal = _scalar_contract(values[..., 0, :, :], w0) + 1j * _scalar_contract(
        values[..., 1, :, :], w1
    )
    toroidal = 1j * _scalar_contract(values[..., 0, :, :], w1) - _scalar_contract(
        values[..., 1, :, :], w0
    )
    return torch.stack((spheroidal, toroidal), dim=-3)


def _relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = reference.norm().item()
    return (actual - reference).norm().item() / denominator if denominator else 0.0


def _check_case(
    optimized: Any,
    dense: dict[str, Any],
    *,
    nlat: int,
    nlon: int,
    lmax: int,
    norm: str,
    csphase: bool,
) -> dict[str, Any]:
    mmax = lmax
    scalar = optimized.RealSHT(
        nlat, nlon, lmax=lmax, mmax=mmax, norm=norm, csphase=csphase
    ).to(dtype=torch.float64)
    dense_scalar = dense["RealSHT"](
        nlat, nlon, lmax=lmax, mmax=mmax, norm=norm, csphase=csphase
    ).to(dtype=torch.float64)
    vector = optimized.RealVectorSHT(
        nlat, nlon, lmax=lmax, mmax=mmax, norm=norm, csphase=csphase
    ).to(dtype=torch.float64)
    dense_vector = dense["RealVectorSHT"](
        nlat, nlon, lmax=lmax, mmax=mmax, norm=norm, csphase=csphase
    ).to(dtype=torch.float64)

    resampling_scalar = _dense_resampling_matrix(dense, nlat, mmax, False)
    resampling_vector = _dense_resampling_matrix(dense, nlat, mmax, True)
    direct_scalar = _direct_effective(dense_scalar, resampling_scalar, False)
    direct_vector = _direct_effective(dense_vector, resampling_vector, True)
    effective_scalar = _effective_from_runtime_fold(scalar, False)
    effective_vector = _effective_from_runtime_fold(vector, True)

    scalar_real_norm = effective_scalar.real.norm().item()
    vector_real_norm = effective_vector.real.norm().item()
    scalar_imag = effective_scalar.imag.abs().max().item()
    vector_imag = effective_vector.imag.abs().max().item()
    scalar_weight_error = _relative_error(effective_scalar, direct_scalar)
    vector_weight_error = _relative_error(effective_vector, direct_vector)

    generator = torch.Generator().manual_seed(8128 + nlat + lmax)
    scalar_values = torch.randn(3, mmax, nlat, generator=generator, dtype=torch.float64)
    scalar_values = torch.complex(
        scalar_values,
        torch.randn(3, mmax, nlat, generator=generator, dtype=torch.float64),
    )
    vector_values = torch.randn(
        3, 2, mmax, nlat, generator=generator, dtype=torch.float64
    )
    vector_values = torch.complex(
        vector_values,
        torch.randn(3, 2, mmax, nlat, generator=generator, dtype=torch.float64),
    )
    scalar_runtime = _scalar_contract(
        _runtime_folded_projection(scalar, scalar_values, False), scalar.weights
    )
    scalar_precomputed = _scalar_contract(scalar_values, effective_scalar)
    vector_runtime = _vector_contract(
        _runtime_folded_projection(vector, vector_values, True), vector.weights
    )
    vector_precomputed = _vector_contract(vector_values, effective_vector)

    adjoint_x = scalar_values
    adjoint_y = torch.complex(
        torch.randn(3, mmax, nlat - 1, generator=generator, dtype=torch.float64),
        torch.randn(3, mmax, nlat - 1, generator=generator, dtype=torch.float64),
    )
    lhs = (_resample_forward(scalar, adjoint_x).conj() * adjoint_y).sum()
    rhs = (adjoint_x.conj() * _resample_adjoint(scalar, adjoint_y)).sum()
    adjoint_error = (lhs - rhs).abs().item() / max(lhs.abs().item(), 1.0e-30)

    # A basis sweep checks every latitude column, including both poles and the
    # endpoint adjacent to the midpoint boundary.
    scalar_basis_error = 0.0
    vector_basis_error = 0.0
    for index in range(nlat):
        scalar_basis = torch.zeros(1, mmax, nlat, dtype=torch.complex128)
        scalar_basis[..., index] = 1.0 + 0.25j
        scalar_basis_error = max(
            scalar_basis_error,
            _relative_error(
                _scalar_contract(
                    _runtime_folded_projection(scalar, scalar_basis, False),
                    scalar.weights,
                ),
                _scalar_contract(scalar_basis, effective_scalar),
            ),
        )
        vector_basis = torch.zeros(1, 2, mmax, nlat, dtype=torch.complex128)
        vector_basis[..., 0, :, index] = 1.0 + 0.25j
        vector_basis[..., 1, :, index] = -0.5 + 0.75j
        vector_basis_error = max(
            vector_basis_error,
            _relative_error(
                _vector_contract(
                    _runtime_folded_projection(vector, vector_basis, True),
                    vector.weights,
                ),
                _vector_contract(vector_basis, effective_vector),
            ),
        )

    result = {
        "grid": f"{nlat}x{nlon}",
        "lmax_mmax": lmax,
        "norm": norm,
        "csphase": csphase,
        "scalar_effective_max_imag": scalar_imag,
        "vector_effective_max_imag": vector_imag,
        "scalar_effective_relative_imag_norm": (
            effective_scalar.imag.norm().item() / scalar_real_norm
        ),
        "vector_effective_relative_imag_norm": (
            effective_vector.imag.norm().item() / vector_real_norm
        ),
        "scalar_K_vs_W_R_relative_l2": scalar_weight_error,
        "vector_K_vs_W_R_relative_l2": vector_weight_error,
        "scalar_random_runtime_vs_precomputed_relative_l2": _relative_error(
            scalar_runtime, scalar_precomputed
        ),
        "vector_random_runtime_vs_precomputed_relative_l2": _relative_error(
            vector_runtime, vector_precomputed
        ),
        "resampling_adjoint_relative_error": adjoint_error,
        "scalar_basis_runtime_vs_precomputed_relative_l2": scalar_basis_error,
        "vector_basis_runtime_vs_precomputed_relative_l2": vector_basis_error,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--base", default=BASE_SHA)
    parser.add_argument("--tolerance", type=float, default=2.0e-12)
    args = parser.parse_args()
    optimized = _load_optimized(args.source)
    dense = _load_dense_namespace(args.source, args.base)

    results = []
    for nlat, nlon, lmax in ((5, 8, 4), (7, 12, 6), (9, 16, 8)):
        for norm in ("ortho", "four-pi", "schmidt", "unnorm"):
            for csphase in (True, False):
                results.append(
                    _check_case(
                        optimized,
                        dense,
                        nlat=nlat,
                        nlon=nlon,
                        lmax=lmax,
                        norm=norm,
                        csphase=csphase,
                    )
                )

    for result in results:
        print(json.dumps(result, sort_keys=True))
        for key, value in result.items():
            if key == "csphase" or key in {"grid", "lmax_mmax", "norm"}:
                continue
            if "effective_" in key and "imag" in key:
                continue
            if value > args.tolerance:
                raise SystemExit(
                    f"{key}={value:.3e} exceeds tolerance {args.tolerance:.3e}"
                )
    print(f"{len(results)} cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
