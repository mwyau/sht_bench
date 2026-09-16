import json
from pathlib import Path

import numpy as np
import pytest

from sht_bench.comparison import (
    RESULT_SCHEMA,
    ComparisonUnavailable,
    canonical_coefficients,
    comparison_metrics,
    ducc_packed_size,
    ducc_to_rectangular,
    load_torch_runtime,
    rectangular_to_ducc,
    write_compare_results,
)
from sht_bench.grids import (
    HIGH_BANDWIDTH_CC_CASES,
    SHTCase,
    grid_shape,
)


def test_focused_case_registry_and_legacy_cc_geometry():
    assert [case.name for case in HIGH_BANDWIDTH_CC_CASES] == [
        "cc-73x144-t36",
        "cc-73x144-t70",
        "cc-73x144-t71",
        "cc-129x256-t127",
        "cc-257x512-t255",
    ]
    assert grid_shape(36, "cc") == (73, 144)
    assert grid_shape(72, "cc") == (145, 288)


@pytest.mark.parametrize(
    "nlat,nlon,lmax,allowed",
    [
        (73, 144, 70, True),
        (73, 144, 71, True),
        (73, 144, 72, False),
        (129, 256, 127, True),
        (257, 512, 255, True),
    ],
)
def test_triangular_high_bandwidth_cases(nlat, nlon, lmax, allowed):
    if allowed:
        case = SHTCase("test", "cc", nlat, nlon, lmax)
        assert case.mmax == lmax
    else:
        with pytest.raises(ValueError, match="recoverable bandwidth"):
            SHTCase("test", "cc", nlat, nlon, lmax)


@pytest.mark.parametrize("lmax", [0, 1, 2, 5, 11])
def test_ducc_rectangular_packing_is_exact(lmax):
    coefficients = canonical_coefficients(lmax, "float64", seed=8128 + lmax)
    packed = rectangular_to_ducc(coefficients)
    restored = ducc_to_rectangular(packed, lmax)
    assert packed.shape == (ducc_packed_size(lmax),)
    assert np.array_equal(restored, coefficients)


def test_ducc_packing_preserves_leading_batch_dimensions():
    coefficients = np.stack(
        [canonical_coefficients(4, "float32", seed=11), canonical_coefficients(4, "float32", seed=12)]
    )
    packed = rectangular_to_ducc(coefficients)
    restored = ducc_to_rectangular(packed, 4)
    assert restored.shape == coefficients.shape
    assert np.array_equal(restored, coefficients)


def test_bounded_relative_metric_records_its_dtype_scaled_floor():
    reference = np.array([1.0, 0.0, 1.0e-8], dtype=np.float32)
    actual = reference.copy()
    actual[1] = np.finfo(np.float32).eps
    metrics = comparison_metrics(actual, reference, "float32")
    assert metrics["compared_values"] == 3
    assert metrics["bounded_rel_floor"] > 0.0
    assert metrics["bounded_rel"] >= 1.0


def test_compare_result_json_is_rich_and_csv_is_flattened(tmp_path: Path):
    record = {
        "schema_version": RESULT_SCHEMA,
        "record_type": "performance",
        "backend": "ducc0.sht",
        "raw_timing_samples_s": [0.1, 0.2],
    }
    payload = {"schema": RESULT_SCHEMA, "records": [record], "skipped": [], "failures": []}
    json_path, csv_path = write_compare_results(payload, tmp_path / "comparison")
    assert json_path == tmp_path / "comparison.json"
    assert csv_path == tmp_path / "comparison.csv"
    assert json.loads(json_path.read_text())["records"][0]["raw_timing_samples_s"] == [
        0.1,
        0.2,
    ]
    assert "raw_timing_samples_s" in csv_path.read_text().splitlines()[0]


def test_torch_runtime_rejects_a_non_checkout_source(tmp_path: Path):
    with pytest.raises(ComparisonUnavailable, match="outside requested source"):
        load_torch_runtime(tmp_path)
