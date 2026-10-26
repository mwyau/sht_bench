from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_fast_scaling.py"
SPEC = importlib.util.spec_from_file_location("run_fast_scaling", SCRIPT)
fast_scaling = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(fast_scaling)


def _build():
    return {
        "status": "built", "build_kind": "git-source", "sha": "abc123",
        "python_executable": "/python", "module_path": "/module",
        "binaries": [{"path": "/module", "sha256": "binary-sha"}],
    }


def test_scenarios_keep_shtns_float32_exclusion_compact():
    args = SimpleNamespace(
        backend_values=("shtns",), thread_values=(1,),
        precision_values=("float32", "float64"),
    )
    cases = fast_scaling.resolution_cases(("GL",), endpoints_only=True)
    groups, exclusions = fast_scaling._scenarios(
        args, cases, {"shtns": _build()}, "manifest-sha",
    )

    assert len(groups) == 2  # one group per spin; each group reuses its plan for both directions
    assert all(group["precision"] == "float64" for group in groups)
    assert len(exclusions) == 4
    assert all(item["status"] == "unsupported" for item in exclusions)
    assert {item["reason"] for item in exclusions} == {"Python API uses float64/complex128"}


def test_csv_row_keeps_three_raw_samples_and_milliseconds():
    row = fast_scaling._record_row({
        "id": "cell", "backend": "ducc", "samples_s": [0.001, 0.002, 0.003],
        "median_s": 0.002, "output_shape": [1, 12], "status": "timed_successfully",
    })

    assert row["sample_1_s"] == 0.001
    assert row["sample_3_s"] == 0.003
    assert row["median_ms"] == 2.0
    assert row["output_shape"] == "[1, 12]"


def test_cli_configuration_is_full_ordered_thread_ladder():
    args = fast_scaling.parser().parse_args([])
    assert fast_scaling._parse_csv(args.threads, tuple(map(str, fast_scaling.THREADS)), "threads") == (
        "1", "2", "4", "8", "16",
    )
    assert args.min_time == 0.005
