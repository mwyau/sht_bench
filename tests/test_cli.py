import argparse
from types import SimpleNamespace

import pytest

from sht_bench.cli import (
    BACKEND_COLORS,
    BACKEND_NAMES,
    BUILD_LINESTYLES,
    DEFAULT_CC_LMAX,
    DEFAULT_GL_LMAX,
    UNSUPPORTED_BENCHMARK_CASES,
    _cc_resolution,
    _matrix_lmax,
    _parse_lmax,
    _parse_threads,
    _plot_index,
    _plot_style,
    _prepare_plot_records,
)
from sht_bench.grids import ATMOSPHERIC_GL_GRIDS, gl_grid_label


def test_parse_lmax_values():
    assert _parse_lmax("42,63,85") == [42, 63, 85]


def test_parse_lmax_inclusive_range():
    assert _parse_lmax("31:95:32") == [31, 63, 95]


def test_parse_lmax_mixed_and_deduplicated():
    assert _parse_lmax("31:95:32,63,127") == [31, 63, 95, 127]


def test_parse_threads():
    assert _parse_threads("1,2,4,8") == [1, 2, 4, 8]


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


def test_native_pyshtools_atmospheric_gl_is_excluded():
    assert ("pyshtools", "gl", 42) in UNSUPPORTED_BENCHMARK_CASES
    assert ("pyshtools", "gl", 1279) in UNSUPPORTED_BENCHMARK_CASES
    assert ("pyshtools", "gl", 31) not in UNSUPPORTED_BENCHMARK_CASES


def test_pyspharm_cc_1023_is_excluded():
    assert ("pyspharm", "cc", 1023) in UNSUPPORTED_BENCHMARK_CASES
    assert ("pyspharm", "cc", 720) not in UNSUPPORTED_BENCHMARK_CASES


def test_plot_style_is_stable_by_backend_and_build():
    assert tuple(BACKEND_COLORS) == BACKEND_NAMES
    assert len(set(BACKEND_COLORS.values())) == len(BACKEND_NAMES)
    assert BUILD_LINESTYLES == {
        "installed": "-",
        "wheel": "-",
        "source": "--",
    }

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
    assert "Wheel and installed builds use solid lines; source builds use dashed lines" in index
    assert "SHTns is source-built in both benchmark environments" in index
    assert "Lower transform time is better" in index
    assert "| Threads | Analysis | Synthesis |" in index
    assert "| Resolution | Analysis | Synthesis |" in index
    assert "2.5° (L=36)" in index


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
