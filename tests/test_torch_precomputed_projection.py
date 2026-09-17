import importlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SOURCE = Path("/home/albert/torch-harmonics")


@pytest.fixture(scope="module")
def research_modules():
    sys.path.insert(0, str(SCRIPTS))
    prototype = importlib.import_module("torch_precomputed_projection_prototype")
    benchmark = importlib.import_module("torch_abc_benchmark")
    optimized = prototype._load_optimized(SOURCE)
    dense = prototype._load_dense_namespace(SOURCE, prototype.BASE_SHA)
    return prototype, benchmark, optimized, dense


@pytest.mark.parametrize("norm", ["ortho", "four-pi", "schmidt", "unnorm"])
@pytest.mark.parametrize("csphase", [True, False])
def test_tiny_effective_projection_matches_explicit_operator(
    research_modules, norm, csphase
):
    prototype, _, optimized, dense = research_modules
    result = prototype._check_case(
        optimized,
        dense,
        nlat=5,
        nlon=8,
        lmax=4,
        norm=norm,
        csphase=csphase,
    )
    assert result["scalar_K_vs_W_R_relative_l2"] < 2.0e-12
    assert result["vector_K_vs_W_R_relative_l2"] < 2.0e-12
    assert result["scalar_random_runtime_vs_precomputed_relative_l2"] < 2.0e-12
    assert result["vector_random_runtime_vs_precomputed_relative_l2"] < 2.0e-12
    assert result["scalar_basis_runtime_vs_precomputed_relative_l2"] < 2.0e-12
    assert result["vector_basis_runtime_vs_precomputed_relative_l2"] < 2.0e-12
    assert result["resampling_adjoint_relative_error"] < 2.0e-12
    assert result["scalar_effective_max_imag"] > 1.0e-8
    assert result["vector_effective_max_imag"] > 1.0e-8


def test_effective_projection_uses_original_latitude_dimension(research_modules):
    prototype, _, optimized, _ = research_modules
    runtime = optimized.RealSHT(73, 144, lmax=72, mmax=72)
    effective = prototype._effective_from_runtime_fold(runtime, vector=False)
    assert effective.shape == (72, 72, 73)
    assert effective.shape[-1] != 2 * 73 - 1
    assert torch.is_complex(effective)


def test_precomputed_projection_has_standard_backward(research_modules):
    _, benchmark, optimized, _ = research_modules
    runtime = optimized.RealSHT(5, 8, lmax=4, mmax=4)
    module = benchmark.PrecomputedProjection(runtime, vector=False).eval()
    values = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    output = module(values)
    output.abs().square().mean().backward()
    assert output.shape == (2, 4, 4)
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_precomputed_vector_projection_has_standard_backward(research_modules):
    _, benchmark, optimized, _ = research_modules
    runtime = optimized.RealVectorSHT(5, 8, lmax=4, mmax=4)
    module = benchmark.PrecomputedProjection(runtime, vector=True).eval()
    values = torch.randn(2, 2, 5, 8, dtype=torch.float64, requires_grad=True)
    output = module(values)
    output.abs().square().mean().backward()
    assert output.shape == (2, 2, 4, 4)
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_precomputed_projection_gradcheck(research_modules):
    _, benchmark, optimized, _ = research_modules
    runtime = optimized.RealSHT(5, 8, lmax=4, mmax=4)
    module = benchmark.PrecomputedProjection(runtime, vector=False).eval()
    values = torch.randn(1, 5, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        module,
        (values,),
        eps=1.0e-6,
        atol=1.0e-5,
        rtol=1.0e-4,
    )
