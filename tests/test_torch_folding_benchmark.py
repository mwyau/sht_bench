import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "torch_folding_benchmark.py"
SOURCE = Path("/home/albert/torch-harmonics")


@pytest.fixture(scope="module")
def benchmark_module():
    spec = importlib.util.spec_from_file_location("torch_folding_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_case_and_batch_parsing(benchmark_module):
    assert benchmark_module._parse_case("721x1440x720") == (721, 1440, 720)
    assert benchmark_module._parse_cases("73x144x72,129x256x128") == (
        (73, 144, 72),
        (129, 256, 128),
    )
    assert benchmark_module._parse_batch_list("1, 2, 4, 128") == (1, 2, 4, 128)
    with pytest.raises(ValueError):
        benchmark_module._parse_case("73x144")
    with pytest.raises(ValueError):
        benchmark_module._parse_batch_list("1,4,2")


def test_projection_estimation_matches_dense_resampling_ratio(benchmark_module):
    dense = benchmark_module.estimate_projection_bytes(
        "dense", "scalar", "float32", 721, 720
    )
    folded = benchmark_module.estimate_projection_bytes(
        "folded", "scalar", "float32", 721, 720
    )
    assert dense == 720 * 720 * (2 * 721 - 1) * 4
    assert dense / folded == pytest.approx((2 * 721 - 1) / 721)
    assert benchmark_module.estimate_projection_bytes(
        "dense", "vector", "float64", 721, 720
    ) == 2 * benchmark_module.estimate_projection_bytes(
        "dense", "scalar", "float64", 721, 720
    )


def test_capacity_budget_and_skip_record(benchmark_module):
    total = 12_000
    decision = benchmark_module.capacity_decision(10_500, total, 0.85)
    assert decision["safe"] is not True
    assert decision["memory_budget_bytes"] == 10_200
    record = benchmark_module.skipped_capacity_record(
        implementation="dense",
        transform="vector",
        dtype_name="float64",
        device="cuda",
        case=(721, 1440, 720),
        batch_size=1,
        measurement="forward",
        warmup=2,
        repeat=5,
        source_sha="optimized",
        dense_base_sha=benchmark_module.BASE_SHA,
        memory_budget=10_200,
        memory_budget_fraction=0.85,
        reason="permanent projection leaves insufficient VRAM headroom",
    )
    assert record["status"] == "skipped_capacity"
    assert record["projection_bytes"] > 0
    assert record["reason"]


def test_next_batch_prediction_is_conservative(benchmark_module):
    observed = [
        {
            "status": "measured",
            "batch_size": 1,
            "module_buffer_bytes": 100,
            "peak_reserved_bytes": 200,
        },
        {
            "status": "measured",
            "batch_size": 2,
            "module_buffer_bytes": 100,
            "peak_reserved_bytes": 300,
        },
    ]
    assert benchmark_module.predict_next_peak_bytes(observed, 4, 1.2) == 600
    assert benchmark_module.predict_next_peak_bytes([], 2) is None


def test_worker_command_and_aggregation(benchmark_module, tmp_path):
    output = tmp_path / "worker.json"
    command = benchmark_module.build_worker_command(
        SCRIPT,
        implementation="folded",
        transform="scalar",
        dtype_name="float32",
        device="cpu",
        case=(17, 32, 16),
        batch_size=1,
        measurement="forward",
        source=SOURCE,
        dense_base_sha=benchmark_module.BASE_SHA,
        output=output,
        warmup=0,
        repeat=1,
        threads=1,
        seed=8128,
        memory_budget_bytes_value=None,
        memory_budget_fraction=0.85,
    )
    assert "--worker" in command
    assert command[command.index("--implementation") + 1] == "folded"
    record = benchmark_module.aggregate_worker_record(
        {"status": "measured", "median_s": 0.1}, 0, "worker note"
    )
    assert record["worker_returncode"] == 0
    assert record["worker_stderr_tail"] == "worker note"


def test_logical_microbatch_calculation(benchmark_module):
    assert benchmark_module.logical_microbatch_count(128, 32) == 4
    assert benchmark_module.logical_microbatch_count(128, 48) == 3
    with pytest.raises(ValueError):
        benchmark_module.logical_microbatch_count(128, 0)


def test_result_schema_validation_and_missing_native_detection(benchmark_module):
    record = benchmark_module._new_performance_record(
        implementation="folded",
        transform="scalar",
        dtype_name="float32",
        device="cpu",
        case=(17, 32, 16),
        batch_size=1,
        measurement="forward",
        warmup=0,
        repeat=1,
        source_sha="optimized",
        dense_base_sha=benchmark_module.BASE_SHA,
        memory_budget=None,
        memory_budget_fraction=0.85,
    )
    record.update({"status": "measured", "median_s": 0.01})
    payload = {"schema": benchmark_module.SCHEMA, "records": [record]}
    expected = benchmark_module._record_key(record)
    assert (
        benchmark_module.validate_result_payload(
            payload, expected_native_keys={expected}
        )
        == []
    )
    missing = benchmark_module.validate_result_payload(
        payload,
        expected_native_keys={expected, ("missing",)},
    )
    assert any("missing native records" in error for error in missing)


@pytest.mark.skipif(
    not SOURCE.exists(), reason="optimized source checkout is unavailable"
)
def test_tiny_cpu_worker_smoke(tmp_path):
    output = tmp_path / "worker.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--worker",
        "--implementation",
        "folded",
        "--transform",
        "scalar",
        "--dtype",
        "float32",
        "--device",
        "cpu",
        "--case",
        "17x32x16",
        "--batch-size",
        "1",
        "--measurement",
        "forward",
        "--source",
        str(SOURCE),
        "--base",
        "3278fb669483d04537aa60c87fb0754d989e2e70",
        "--worker-output",
        str(output),
        "--warmup",
        "0",
        "--repeat",
        "1",
        "--threads",
        "1",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert payload["schema"] == "sht_bench.torch_folding.v2"
    assert payload["record"]["status"] == "measured"
    assert payload["record"]["worker_pid"] > 0
