import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "torch_bc_resolution_scaling.py"
)


@pytest.fixture(scope="module")
def benchmark_module():
    spec = importlib.util.spec_from_file_location("torch_bc_resolution_scaling", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase_a_enumeration_is_b_c_only_and_ordered(benchmark_module):
    cases = ((361, 720, 360), (481, 960, 480))
    specs = benchmark_module.enumerate_phase_a_cases(cases)
    assert [
        (item["case"], item["transform"], item["implementation"]) for item in specs
    ] == [
        ("361x720x360", "scalar", "runtime_fold"),
        ("361x720x360", "scalar", "precomputed_projection"),
        ("361x720x360", "vector", "runtime_fold"),
        ("361x720x360", "vector", "precomputed_projection"),
        ("481x960x480", "scalar", "runtime_fold"),
        ("481x960x480", "scalar", "precomputed_projection"),
        ("481x960x480", "vector", "runtime_fold"),
        ("481x960x480", "vector", "precomputed_projection"),
    ]
    assert all(item["batch_size"] == 1 for item in specs)


def test_atomic_manifest_update_and_stale_in_progress_is_non_retryable(
    benchmark_module, tmp_path
):
    path = tmp_path / "progress.json"
    benchmark_module.load_progress(path, {"run": "test"})
    spec = benchmark_module.make_spec(
        phase="phase_a_resolution",
        case=(361, 720, 360),
        transform="scalar",
        implementation="runtime_fold",
        dtype="float32",
        batch_size=1,
    )
    key = benchmark_module.case_key(spec)
    benchmark_module.update_progress_record(
        path,
        key,
        {"spec": spec, "status": "in_progress", "attempt": 1},
    )
    resumed = benchmark_module.load_progress(path, {"run": "test"})
    entry = resumed["records"][key]
    assert entry["status"] == "skipped_after_process_kill"
    assert entry["result"]["status"] == "skipped_after_process_kill"
    assert benchmark_module._existing_result(resumed, spec)["status"] == (
        "skipped_after_process_kill"
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_completed_case_is_skipped_and_aggregation_retains_result(
    benchmark_module, tmp_path
):
    path = tmp_path / "progress.json"
    benchmark_module.load_progress(path, {})
    spec = benchmark_module.make_spec(
        phase="phase_a_resolution",
        case=(361, 720, 360),
        transform="scalar",
        implementation="precomputed_projection",
        dtype="float32",
        batch_size=1,
    )
    result = benchmark_module.base_result(spec)
    result.update({"status": "completed", "median_s": 0.001})
    benchmark_module.update_progress_record(
        path,
        benchmark_module.case_key(spec),
        {
            "spec": spec,
            "status": "completed",
            "attempt": 1,
            "result": result,
        },
    )
    manifest = benchmark_module._read_json(path)
    assert benchmark_module._existing_result(manifest, spec) == result
    assert benchmark_module._result_records(manifest) == [result]


def test_capacity_skip_and_memory_model_are_conservative(benchmark_module):
    decision = benchmark_module.capacity_decision(
        estimated_module_bytes_value=900,
        predicted_peak_bytes=1_100,
        gpu_total_bytes=1_000,
        estimated_host_peak_bytes=100,
        host_limit_bytes=1_000,
    )
    assert decision["safe"] is False
    assert "CUDA budget" in decision["reason"]

    records = [
        {"status": "completed", "batch_size": 1, "peak_reserved_bytes": 200},
        {"status": "completed", "batch_size": 2, "peak_reserved_bytes": 300},
    ]
    model = benchmark_module.fit_memory_model(records)
    assert model["fixed_bytes"] == 100
    assert model["per_batch_bytes"] == 100
    assert benchmark_module.predict_next_peak_bytes(records, 4, 1.2) == 600


def test_module_estimate_keeps_c_complex_and_vector_double(benchmark_module):
    case = (721, 1440, 720)
    b_scalar = benchmark_module.estimate_module_bytes(
        "runtime_fold", "scalar", "float32", case
    )
    c_scalar = benchmark_module.estimate_module_bytes(
        "precomputed_projection", "scalar", "float32", case
    )
    c_vector = benchmark_module.estimate_module_bytes(
        "precomputed_projection", "vector", "float32", case
    )
    assert c_scalar / b_scalar > 1.99
    assert c_vector == 2 * c_scalar
