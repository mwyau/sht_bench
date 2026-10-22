from __future__ import annotations

import importlib.util
import copy
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


def test_source_and_wheel_labelled_shtns_use_git_checkout_without_real_build(tmp_path, monkeypatch):
    from sht_bench import sources

    runner = _load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "PROVENANCE", copy.deepcopy(runner.PROVENANCE))
    acquisitions = []
    install_calls = []
    commands = []
    locked_sources = {
        "ducc": {"origin": "https://example.invalid/ducc.git", "ref": "ducc0", "sha": "1" * 40},
        "shtns": {"origin": "https://example.invalid/shtns.git", "ref": "master", "sha": "2" * 40},
    }

    def checkout(source_root, lock_path):
        acquisitions.append((source_root, lock_path))
        return {"schema": 1, "sources": locked_sources}

    def run(command, *, env=None):
        commands.append(command)
        if command[:2] == ["uv", "venv"]:
            Path(command[-1]).mkdir(parents=True, exist_ok=True)

    def uv_install(python, requirement, **kwargs):
        install_calls.append((python, requirement, kwargs))

    monkeypatch.setattr(sources, "checkout", checkout)
    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(runner, "uv_install", uv_install)

    _, _, failures, unavailable = runner.create_environment(
        "source", "3.13", False, backends=("ducc", "shtns"), strict_builds=True
    )
    assert not failures and not unavailable
    assert len(acquisitions) == 1
    assert {call[1] for call in install_calls} == {
        str(tmp_path / ".sources" / "matrix" / "ducc"),
        str(tmp_path / ".sources" / "matrix" / "shtns"),
    }
    for backend in ("ducc", "shtns"):
        provenance = runner.PROVENANCE["source"][backend]
        assert provenance["build_kind"] == "git-source"
        assert provenance["build_sha"] == locked_sources[backend]["sha"]
        assert provenance["build_ref"] == locked_sources[backend]["ref"]
        assert provenance["source_lock"] == str(acquisitions[0][1])
    assert all("--no-binary" not in command and "--only-binary" not in command for command in commands)

    install_calls.clear()
    acquisitions.clear()
    commands.clear()
    _, _, failures, unavailable = runner.create_environment(
        "wheel", "3.13", False, backends=("shtns",), strict_builds=True
    )
    assert not failures and not unavailable
    assert len(acquisitions) == 1
    assert len(install_calls) == 1
    assert install_calls[0][1] == str(tmp_path / ".sources" / "matrix" / "shtns")
    provenance = runner.PROVENANCE["wheel"]["shtns"]
    assert provenance["build_kind"] == "git-source"
    assert provenance["build_sha"] == locked_sources["shtns"]["sha"]
    assert provenance["build_ref"] == locked_sources["shtns"]["ref"]
    assert provenance["source_lock"] == str(acquisitions[0][1])
