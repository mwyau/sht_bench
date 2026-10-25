import argparse
import csv
import json
from types import SimpleNamespace

import pytest

from sht_bench.cli import (
    BACKEND_COLORS,
    BACKEND_NAMES,
    BUILD_LINESTYLES,
    COMPARE_BACKEND_NAMES,
    DEFAULT_CC_LMAX,
    DEFAULT_GL_LMAX,
    UNSUPPORTED_BENCHMARK_CASES,
    _cc_resolution,
    _matrix_lmax,
    _parse_compare_backends,
    _parse_compare_cases,
    _parse_compare_dtypes,
    _parse_lmax,
    _parse_torch_devices,
    _parse_threads,
    _plot_index,
    _plot_lines,
    _plot_style,
    _write_results,
    _series_label,
    _prepare_plot_records,
    _pyspharm_cc_memory_limit,
    _build_conda_matrix_env,
    _matrix_output_name,
    _retain_current_pyspharm_results,
    run_matrix,
    CELL_RESULT_RE,
    build_parser,
)
from sht_bench.grids import ATMOSPHERIC_GL_GRIDS, HIGH_BANDWIDTH_CC_CASES, gl_grid_label


def test_parse_lmax_values():
    assert _parse_lmax("42,63,85") == [42, 63, 85]


def test_parse_lmax_inclusive_range():
    assert _parse_lmax("31:95:32") == [31, 63, 95]


def test_parse_lmax_mixed_and_deduplicated():
    assert _parse_lmax("31:95:32,63,127") == [31, 63, 95, 127]


def test_parse_threads():
    assert _parse_threads("1,2,4,8") == [1, 2, 4, 8]


def test_parse_focused_comparison_choices():
    assert _parse_compare_backends("ducc,torch") == list(COMPARE_BACKEND_NAMES)
    assert _parse_compare_cases("cc-73x144-t70,cc-73x144-t71") == [
        "cc-73x144-t70",
        "cc-73x144-t71",
    ]
    assert _parse_compare_dtypes("float32,float64") == ["float32", "float64"]
    assert _parse_torch_devices("cpu,cuda") == ["cpu", "cuda"]


def test_compare_parser_has_explicit_focused_defaults():
    args = build_parser().parse_args(["compare"])
    assert args.backend == ["ducc", "torch"]
    assert args.case == [case.name for case in HIGH_BANDWIDTH_CC_CASES]
    assert args.dtype == ["float32", "float64"]
    assert args.operation == ["analysis", "synthesis"]
    assert args.torch_device == ["cpu"]


def test_parse_lmax_rejects_zero_step():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_lmax("31:95:0")


def test_gl_default_atmospheric_grids():
    assert DEFAULT_GL_LMAX == [42, 63, 85, 159, 255, 319, 511, 639, 1023, 1279]
    assert [gl_grid_label(lmax) for lmax in DEFAULT_GL_LMAX] == [
        "F32",
        "F48",
        "F64",
        "F80",
        "F128",
        "F160",
        "F256",
        "F320",
        "F512",
        "F640",
    ]
    assert ATMOSPHERIC_GL_GRIDS[42][:2] == (64, 128)
    assert ATMOSPHERIC_GL_GRIDS[319][:2] == (320, 640)
    assert ATMOSPHERIC_GL_GRIDS[1279][:2] == (1280, 2560)


def test_cc_default_resolutions():
    assert DEFAULT_CC_LMAX == [36, 60, 72, 90, 120, 180, 360, 720]
    assert [_cc_resolution(lmax) for lmax in DEFAULT_CC_LMAX] == [
        2.5,
        1.5,
        1.25,
        1.0,
        0.75,
        0.5,
        0.25,
        0.125,
    ]
    args = SimpleNamespace(lmax=None)
    assert _matrix_lmax(args, "gl") == DEFAULT_GL_LMAX
    assert _matrix_lmax(args, "cc") == DEFAULT_CC_LMAX


def test_explicit_lmax_overrides_grid_defaults():
    args = SimpleNamespace(lmax=[36, 90])
    assert _matrix_lmax(args, "gl") == [36, 90]
    assert _matrix_lmax(args, "cc") == [36, 90]


def test_legacy_matrix_selects_conda_environment_for_pyspharm(monkeypatch):
    args = build_parser().parse_args(
        ["matrix", "--build", "source", "--backend", "pyspharm"]
    )
    assert args.backend == ["pyspharm"]
    assert args.build == "source"

    commands = []
    monkeypatch.setattr("sht_bench.cli._run_subprocess", lambda command, **_: commands.append(command))
    python = _build_conda_matrix_env("3.13", reuse=False)
    assert python.name == "python"
    command = commands[0]
    assert command[command.index("--build") + 1] == "wheel"
    assert command[command.index("--backend") + 1] == "pyspharm"
    assert "--setup-only" in command


def test_pyspharm_only_legacy_matrix_does_not_create_or_replace_uv_environment(
    tmp_path, monkeypatch
):
    args = build_parser().parse_args([
        "matrix", "--build", "wheel", "--backend", "pyspharm",
        "--grid", "gl", "--lmax", "42", "--threads", "1",
        "--output", str(tmp_path / "matrix"),
    ])
    conda_python = tmp_path / "conda-env" / "bin" / "python"
    monkeypatch.setattr("sht_bench.cli._build_matrix_env", lambda *a, **k: pytest.fail("uv env touched"))
    monkeypatch.setattr("sht_bench.cli._build_conda_matrix_env", lambda *a, **k: conda_python)
    monkeypatch.setattr("sht_bench.cli._backend_available", lambda *a, **k: True)
    commands = []
    monkeypatch.setattr("sht_bench.cli._run_subprocess", lambda command, **_: commands.append(command))

    assert run_matrix(args) == 1
    assert len(commands) == 1
    assert commands[0][0] == str(conda_python)


def test_matrix_replacement_removes_old_pyspharm_provenance_only():
    records = [
        {"backend": "ducc", "build_mode": "wheel"},
        {"backend": "pyspharm", "build_mode": "wheel"},
        {"backend": "pyspharm", "build_mode": "conda-forge"},
    ]
    assert _retain_current_pyspharm_results(records) == [records[0], records[2]]


def test_native_pyshtools_atmospheric_gl_is_excluded():
    assert ("pyshtools", "gl", 42) in UNSUPPORTED_BENCHMARK_CASES
    assert ("pyshtools", "gl", 1279) in UNSUPPORTED_BENCHMARK_CASES
    assert ("pyshtools", "gl", 31) not in UNSUPPORTED_BENCHMARK_CASES


def test_pyspharm_cc_1023_is_resource_limited_by_observed_memory_headroom():
    assert ("pyspharm", "cc", 1023) not in UNSUPPORTED_BENCHMARK_CASES
    assert ("pyspharm", "cc", 720) not in UNSUPPORTED_BENCHMARK_CASES
    assert "2x headroom" in _pyspharm_cc_memory_limit(1023, 70_930_000_000)
    assert _pyspharm_cc_memory_limit(1023, 110_000_000_000) is None
    assert _pyspharm_cc_memory_limit(720, 1) is None


def test_plot_style_is_stable_by_backend_and_build():
    assert set(BACKEND_NAMES) <= set(BACKEND_COLORS)
    assert len(set(BACKEND_COLORS.values())) == len(BACKEND_COLORS)
    assert BUILD_LINESTYLES == {
        "installed": "-",
        "wheel": "-",
        "source": "--",
        "conda-forge": "-.",
    }
    assert _plot_style({"backend": "pyspharm", "build_mode": "conda-forge"}) == {
        "color": BACKEND_COLORS["pyspharm"],
        "linestyle": "-.",
    }
    assert _series_label({
        "backend": "pyspharm", "build_mode": "conda-forge", "backend_version": "1.0.9",
        "build_string": "np2py313h854c110_1017",
    }) == "pyspharm (conda-forge 1.0.9-np2py313h854c110_1017)"
    assert _matrix_output_name("conda-forge", "cc", 1) == "conda-forge-cc-t1.json"
    assert CELL_RESULT_RE.fullmatch("conda-forge-cc-t1.json")

    for backend, color in BACKEND_COLORS.items():
        assert _plot_style({"backend": backend, "build_mode": "wheel"}) == {
            "color": color,
            "linestyle": "-" if backend != "shtns" else "--",
        }
        assert _plot_style({"backend": backend, "build_mode": "source"}) == {
            "color": color,
            "linestyle": "--",
        }


def test_shtns_plot_records_use_one_source_series():
    base = {
        "backend": "shtns",
        "grid": "gl",
        "operation": "analysis",
        "lmax": 42,
        "threads_requested": 1,
    }
    records = [
        {**base, "build_mode": "wheel", "median_s": 2.0},
        {**base, "build_mode": "source", "median_s": 1.0},
    ]
    prepared = _prepare_plot_records(records)
    assert prepared == [{**base, "build_mode": "source", "median_s": 1.0}]

    wheel_only = _prepare_plot_records([records[0]])
    assert wheel_only == [{**base, "build_mode": "source", "median_s": 2.0}]


def test_plot_index_inlines_clickable_comparisons():
    created = [
        "by-lmax/cc/analysis/threads-1.png",
        "by-lmax/cc/synthesis/threads-1.png",
        "by-threads/cc/analysis/lmax-36.png",
        "by-threads/cc/synthesis/lmax-36.png",
    ]

    index = _plot_index(
        created,
        grids=["cc"],
        operations=["analysis", "synthesis"],
        threads=[1],
        lmax_values=[36],
    )

    assert "## Transform time versus spectral scale" in index
    assert "CC figures use regular-grid spacing in degrees" in index
    assert "Installed builds and binary-wheel builds use solid lines, Git source builds use dashed lines, and conda-forge builds use dash-dot lines" in index
    assert "SHTns is source-built in both benchmark environments" in index
    assert "Lower transform time is better" in index
    assert "| Threads | Analysis | Synthesis |" in index
    assert "| Resolution | Analysis | Synthesis |" in index
    assert "2.5° (L=36)" in index

    conda_index = _plot_index(
        created,
        grids=["cc"],
        operations=["analysis", "synthesis"],
        threads=[1],
        lmax_values=[36],
        pyspharm_environment=(
            "The pyspharm series uses conda-forge 1.0.9 build `np2py313`, "
            "Python 3.13.15, and NumPy 2.5.4."
        ),
    )
    assert "conda-forge 1.0.9 build `np2py313`" in conda_index


def test_matrix_csv_keeps_union_of_build_provenance_fields(tmp_path):
    output = tmp_path / "matrix.json"
    _write_results([
        {"backend": "ducc", "build_mode": "wheel", "median_s": 0.1},
        {
            "backend": "pyspharm", "build_mode": "conda-forge", "median_s": 0.2,
            "build_identity": {"version": "1.0.9", "build_string": "np2py313"},
            "samples_s": [0.19, 0.2, 0.21],
        },
    ], output)

    rows = list(csv.DictReader(output.with_suffix(".csv").open()))
    assert json.loads(rows[1]["build_identity"]) == {
        "build_string": "np2py313", "version": "1.0.9",
    }
    assert "samples_s" not in rows[1]
    assert len(json.loads(output.read_text())) == 2


def test_plot_index_labels_atmospheric_gl_cases():
    created = [
        "by-lmax/gl/analysis/threads-1.png",
        "by-threads/gl/analysis/lmax-42.png",
    ]
    index = _plot_index(
        created,
        grids=["gl"],
        operations=["analysis"],
        threads=[1],
        lmax_values=[42],
    )
    assert "F32" in index
    assert "| Gaussian grid / truncation | Analysis |" in index


def test_conda_forge_series_renders_a_chart_with_package_identity(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "pyspharm.png"
    _plot_lines(
        [{
            "backend": "pyspharm", "build_mode": "conda-forge",
            "backend_version": "1.0.9", "build_string": "np2py313h854c110_1017",
            "grid": "gl", "operation": "analysis", "lmax": 42,
            "threads_requested": 1, "median_s": 0.001,
        }],
        x_key="lmax", output=output, title="GL scalar analysis",
        stat="median_s", dpi=72, group_keys=("backend", "build_mode"),
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 1000
