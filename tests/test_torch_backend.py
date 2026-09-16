import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sht_bench.comparison import (
    ComparisonUnavailable,
    _run_calibration,
    canonical_coefficients,
    load_torch_runtime,
    prepare_ducc,
    prepare_torch,
    required_single_modes,
    run_comparison_case,
)
from sht_bench.grids import HIGH_BANDWIDTH_CC_CASES


def _runtime():
    configured = os.environ.get("SHT_BENCH_TORCH_HARMONICS_SOURCE", "/home/albert/torch-harmonics")
    source = Path(configured)
    if not source.exists():
        pytest.skip(f"torch-harmonics source checkout is not present: {source}")
    try:
        return load_torch_runtime(source)
    except ComparisonUnavailable as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_y_mode_convention_calibration(dtype):
    runtime = _runtime()
    try:
        runtime.torch.set_num_threads(1)
        runtime.torch.set_num_interop_threads(1)
    except RuntimeError:
        # The command-line benchmark performs this in a fresh worker.  A test
        # process may already have initialized Torch's inter-op pool.
        pass
    records = _run_calibration(
        HIGH_BANDWIDTH_CC_CASES[1],
        dtype,
        "cpu",
        1,
        runtime,
        process_isolated=False,
    )
    assert [record["spectrum_kind"] for record in records] == [
        "Y00",
        "Y10",
        "Y11-real",
        "Y11-imag",
    ]
    assert all(record["accuracy_pass"] for record in records)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("case_index", [1, 2])
def test_high_bandwidth_cross_backend_accuracy(case_index, dtype):
    runtime = _runtime()
    case = HIGH_BANDWIDTH_CC_CASES[case_index]
    try:
        runtime.torch.set_num_threads(1)
        runtime.torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    records = run_comparison_case(
        case,
        dtype,
        "cpu",
        1,
        ("ducc", "torch"),
        runtime,
        seed=20260910,
        warmup=0,
        repeat=1,
        min_time=1.0e-4,
        process_isolated=False,
    )
    accuracy = [record for record in records if record["record_type"] == "accuracy"]
    assert accuracy
    assert all(record["analysis_method"] == "sampling-theorem" for record in accuracy)
    assert all(record["accuracy_pass"] for record in accuracy)
    modes = {(record["mode_degree"], record["mode_order"]) for record in accuracy if record["spectrum_kind"] == "mode"}
    assert modes == set(required_single_modes(case))
    assert any(record["spectrum_kind"] == "random" for record in accuracy)


def test_torch_cpu_dtype_and_device_are_preserved():
    runtime = _runtime()
    case = HIGH_BANDWIDTH_CC_CASES[0]
    records = run_comparison_case(
        case,
        "float32",
        "cpu",
        1,
        ("ducc", "torch"),
        runtime,
        seed=20260910,
        warmup=0,
        repeat=1,
        min_time=1.0e-4,
        process_isolated=False,
    )
    performance = [record for record in records if record["record_type"] == "performance"]
    assert {record["device"] for record in performance} == {"cpu"}
    assert {record["dtype"] for record in performance} == {"float32"}
    assert all(record["actual_threads"] == 1 for record in performance if record["backend"] == "torch-harmonics")


def test_torch_adapter_passes_exclusive_limits_and_conventions():
    runtime = _runtime()
    case = HIGH_BANDWIDTH_CC_CASES[2]
    coefficients = canonical_coefficients(case.lmax, "float64", seed=20260910)
    ducc = prepare_ducc(case, "float64", 1)
    ducc.set_coefficients(coefficients)
    spatial = ducc.synthesis()
    state = prepare_torch(
        case,
        "float64",
        "cpu",
        "sampling-theorem",
        runtime,
        coefficients,
        spatial,
    )
    assert state.analysis_module.lmax == case.lmax + 1
    assert state.analysis_module.mmax == case.mmax + 1
    assert state.analysis_module.grid == "equiangular"
    assert state.analysis_module.norm == "ortho"
    assert state.analysis_module.csphase is True
    assert state.analysis_module.analysis == "sampling-theorem"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_torch_cuda_dtype_and_device_are_preserved():
    runtime = _runtime()
    case = HIGH_BANDWIDTH_CC_CASES[0]
    records = run_comparison_case(
        case,
        "float64",
        "cuda",
        1,
        ("ducc", "torch"),
        runtime,
        seed=20260910,
        warmup=0,
        repeat=1,
        min_time=1.0e-4,
        process_isolated=False,
    )
    performance = [record for record in records if record["record_type"] == "performance"]
    torch_records = [record for record in performance if record["backend"] == "torch-harmonics"]
    assert torch_records
    assert {record["device"] for record in torch_records} == {"cuda"}
    assert {record["dtype"] for record in torch_records} == {"float64"}
    assert all(record["timing_method"] == "cuda-event-compute" for record in torch_records)
    assert all(record["process_pid"] > 0 for record in torch_records)
