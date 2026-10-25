import pytest

from sht_bench.source_runner import _selection, select_batches, select_workloads, workloads


def test_source_workload_selectors_keep_only_requested_comparable_cases():
    planned = list(workloads([16], smoke=True))
    selected = select_workloads(
        planned,
        grids={"GL", "CC"},
        precisions={"float32"},
        spins={0, 1},
        orders={"full"},
        geometry_groups={"minimum", "quadrature", "extended"},
    )

    assert len(selected) == 8
    assert all(item.grid in {"GL", "CC"} for item in selected)
    assert all(item.precision == "float32" for item in selected)
    assert all(item.spin in {0, 1} for item in selected)
    assert all(item.mmax == item.lmax for item in selected)
    assert all(item.geometry_group in {"minimum", "quadrature", "extended"} for item in selected)


def test_workload_selection_parser_rejects_duplicates_and_unknown_values():
    with pytest.raises(ValueError):
        _selection("GL,GL", ("GL", "CC", "F1"), "--grids")
    with pytest.raises(ValueError):
        _selection("float32,float16", ("float32", "float64"), "--precisions")
    assert _selection(" GL, CC ", ("GL", "CC", "F1"), "--grids") == {"GL", "CC"}


def test_batch_selector_expands_only_requested_sizes():
    planned = list(workloads([64], smoke=True))[:2]
    selected = select_batches(planned, {1, 4})
    assert len(selected) == len(planned) * 2
    assert {item.batch for item in selected} == {1, 4}


def test_quarter_order_filter_excludes_degree_one_without_distinct_quarter_order():
    planned = list(workloads([1], smoke=True))
    assert planned
    assert all(item.mmax == item.lmax for item in planned)
    assert select_workloads(planned, orders={"quarter"}) == []
