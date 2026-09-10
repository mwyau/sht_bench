from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_build_matrix.py"
    spec = importlib.util.spec_from_file_location("run_build_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arm64_defaults_to_ducc_and_shtns():
    runner = _load_runner()
    assert runner.default_backends("aarch64") == ("ducc", "shtns")
    assert runner.default_backends("arm64") == ("ducc", "shtns")


def test_x86_64_keeps_all_backends():
    runner = _load_runner()
    assert runner.default_backends("x86_64") == runner.BACKEND_NAMES


def test_backend_parser_deduplicates_and_preserves_order():
    runner = _load_runner()
    assert runner.parse_backends("shtns,ducc,shtns") == ("shtns", "ducc")
