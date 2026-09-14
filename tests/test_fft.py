from __future__ import annotations

import argparse

import numpy as np
import pytest

from fft_bench.backends import (
    KIND_NAMES,
    PRECISION_NAMES,
    UnsupportedFFTCase,
    classify_longdouble,
    dtype_spec,
    longdouble_info,
    prepare_case,
)
from fft_bench.cli import (
    DEFAULT_SIZES,
    _parse_dtypes,
    _parse_kinds,
    _parse_sizes,
    _parse_threads,
)
from fft_bench.report import build_report, format_milliseconds, format_ratio


def test_longdouble_classification_reports_numpy_fields():
    info = longdouble_info()

    assert info["itemsize"] == np.dtype(np.longdouble).itemsize
    assert info["bits"] == np.finfo(np.longdouble).bits
    assert info["nmant"] == np.finfo(np.longdouble).nmant
    assert info["eps"] == float(np.finfo(np.longdouble).eps)
    assert info["eps_text"] == str(np.finfo(np.longdouble).eps)
    assert info["significand_bits"] == info["nmant"] + 1
    assert info["classification"] == classify_longdouble()
    assert "nmant" in info["nmant_convention"]


def test_longdouble_classification_does_not_use_dtype_name_alone():
    info = longdouble_info()
    if info["itemsize"] == 16 and info["nmant"] == 112:
        assert info["classification"] == "IEEE binary128"
    elif info["itemsize"] == 16 and info["nmant"] == 63:
        assert info["classification"].startswith("x87 extended precision")
    else:
        assert "float128" not in str(info["classification"])


def test_dtype_spec_uses_actual_numpy_dtypes():
    assert dtype_spec("r2c", "float32").input_dtype == np.dtype(np.float32)
    assert dtype_spec("r2c", "float32").output_dtype == np.dtype(np.complex64)
    assert dtype_spec("r2c", "float64").output_dtype == np.dtype(np.complex128)
    assert dtype_spec("c2c", "longdouble").input_dtype == np.dtype(np.clongdouble)
    assert dtype_spec("c2c", "longdouble").output_dtype == np.dtype(np.clongdouble)


def test_dtype_and_kind_parsers():
    assert _parse_kinds("r2c,c2c,r2c") == list(KIND_NAMES)
    assert _parse_dtypes("float32,float64,longdouble") == list(PRECISION_NAMES)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_dtypes("float128")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_kinds("fft")


def test_size_and_thread_parsers():
    assert DEFAULT_SIZES == [256, 1024, 4096, 16384, 65536, 262144, 1048576]
    assert _parse_sizes("1024,256,1024") == [256, 1024]
    assert _parse_threads("4,1,2") == [1, 2, 4]
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_sizes("0")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_threads("0")


def test_report_formatting_helpers():
    assert format_milliseconds(0.0000005) == "<0.001"
    assert format_milliseconds(0.01234) == "12.34"
    assert format_ratio(0.004, 0.002) == "2×"
    assert format_ratio(None, 0.002) == "—"

    records = []
    for kind in ("r2c", "c2c"):
        for precision, median in (
            ("float32", 0.001),
            ("float64", 0.002),
            ("longdouble", 0.004),
        ):
            records.append(
                {
                    "kind": kind,
                    "requested_precision": precision,
                    "fft_size": 256,
                    "threads_requested": 1,
                    "median_s": median,
                    "os": "Linux",
                    "architecture": "x86_64",
                    "cpu": "test CPU",
                    "python": "3.13.0",
                    "numpy_version": "2.5.0",
                    "backend_version": "0.41.0",
                    "ducc_wrapper": "pybind11",
                    "build_provenance": {
                        "build_mode": "source",
                        "ducc_optimization": "portable",
                    },
                    "longdouble_classification": "x87 extended precision (64-bit significand, 16-byte storage)",
                    "longdouble_itemsize": 16,
                    "longdouble_bits": 128,
                    "longdouble_nmant": 63,
                    "longdouble_eps": "1e-19",
                }
            )
    report = build_report(records, plot_files=["r2c-threads-1.png"])
    assert "## Runtime environment" in report
    assert "## Median transform time" in report
    assert "### r2c" in report and "### c2c" in report
    assert "x87 extended precision" in report
    assert "longdouble / float64" in report
    assert "2×" in report
    assert "r2c-threads-1.png" in report


def test_ducc_smoke_float_cases_if_installed():
    pytest.importorskip("ducc0")
    for kind, precision in (("r2c", "float32"), ("r2c", "float64"), ("c2c", "float64")):
        case = prepare_case(kind, precision, 16, 1, seed=7)
        result = np.asarray(case.transform())
        assert result.shape == case.output_array.shape
        assert result.dtype == case.output_dtype


def test_ducc_smoke_longdouble_only_for_genuine_support():
    ducc0 = pytest.importorskip("ducc0")
    info = longdouble_info()
    if not info["wider_than_float64"]:
        pytest.skip("NumPy longdouble is not wider than float64")
    if getattr(ducc0, "__wrapper__", None) != "pybind11":
        pytest.skip(
            "DUCC nanobind binding does not expose genuine longdouble FFT support"
        )
    try:
        case = prepare_case("r2c", "longdouble", 16, 1, seed=7)
    except UnsupportedFFTCase as exc:
        pytest.skip(str(exc))
    result = np.asarray(case.transform())
    assert result.dtype == np.dtype(np.clongdouble)
