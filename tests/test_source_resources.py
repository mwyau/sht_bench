"""Resource preflight uses faithful, conservative native plan-size estimates."""
from __future__ import annotations

from sht_bench.source_adapters import Workload
from sht_bench import source_resources


def test_spherepack_stored_array_lower_bounds_match_integer_formulas():
    cc = source_resources.spherepack_stored_arrays(8, 16, "CC")
    assert cc["array_lengths"] == {"wshaes": 175, "wshses": 175, "wvhaes": 319, "wvhses": 319}
    assert cc["persistent_lower_bound_bytes"] == 3952
    assert cc["itemsize_bytes"] == 4

    gl = source_resources.spherepack_stored_arrays(8, 16, "GL")
    assert gl["array_lengths"] == {"wshags": 331, "wshsgs": 331, "wvhags": 355, "wvhsgs": 335}
    assert gl["persistent_lower_bound_bytes"] == 5408

    odd_cc = source_resources.spherepack_stored_arrays(9, 17, "CC")
    assert odd_cc["array_lengths"] == {"wshaes": 257, "wshses": 257, "wvhaes": 482, "wvhses": 482}
    assert odd_cc["persistent_lower_bound_bytes"] == 5912

    odd_gl = source_resources.spherepack_stored_arrays(9, 17, "GL")
    assert odd_gl["array_lengths"] == {"wshags": 464, "wshsgs": 464, "wvhags": 482, "wvhsgs": 500}
    assert odd_gl["persistent_lower_bound_bytes"] == 7640


def test_preflight_marks_only_plan_over_budget_as_resource_limited(monkeypatch):
    monkeypatch.setattr(source_resources, "available_memory", lambda: {
        "host_available_bytes": 100_000,
        "cgroup_limits": [{"path": "/mock", "limit_bytes": 100_000, "current_bytes": 90_000, "remaining_bytes": 10_000}],
        "available_bytes": 10_000,
    })
    small = Workload("GL", 8, 16, 4, 4, precision="float32")
    limited = source_resources.preflight("pyspharm", small)
    assert limited["memory_budget_bytes"] == 10_000
    assert limited["persistent_lower_bound_bytes"] == 5408
    assert limited["resource_limited"] is False

    syl_limited = source_resources.preflight("pyspharm-syl", small)
    assert syl_limited["persistent_lower_bound_bytes"] == 5408
    assert syl_limited["resource_limited"] is False

    constrained = source_resources.preflight("pyspharm", small, limit_bytes=100)
    assert constrained["memory_budget_bytes"] == 100
    assert constrained["resource_limited"] is True
    assert "persistent arrays alone exceed" in constrained["reason"]


def test_non_spherepack_backends_do_not_get_spherepack_limits(monkeypatch):
    monkeypatch.setattr(source_resources, "available_memory", lambda: {
        "host_available_bytes": 1, "cgroup_limits": [], "available_bytes": 1,
    })
    w = Workload("GL", 8, 16, 4, 4)
    result = source_resources.preflight("ducc", w)
    assert result["resource_limited"] is False
    assert result["persistent_lower_bound_bytes"] is None
