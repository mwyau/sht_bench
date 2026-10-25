from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from sht_bench import conda_environment as conda


def _records():
    return [
        {
            "name": "pyspharm",
            "version": "1.0.9",
            "build": "np2py314h2766e9b_1017",
            "build_number": 1017,
            "channel": "conda-forge",
            "url": "https://conda.anaconda.org/conda-forge/linux-64/pyspharm-1.0.9-np2py314h2766e9b_1017.conda",
            "sha256": "1edea5872986bd960a9d42f234960e36fdfd8cf6a110c37daaa53baf20b94b00",
            "depends": ["numpy >=2.5.3,<3", "python >=3.14.0rc2,<3.15.0a0"],
            "subdir": "linux-64",
        },
        {"name": "python", "version": "3.14.9", "build": "hcd007b5_100_cp314"},
        {"name": "numpy", "version": "2.5.4", "build": "py314hd3a7d6b_0"},
    ]


def _write_records(prefix: Path, records):
    metadata = prefix / "conda-meta"
    metadata.mkdir(parents=True)
    for index, record in enumerate(records):
        (metadata / f"{index}.json").write_text(json.dumps(record))


def test_find_conda_accepts_user_selected_conda_compatible_tool(tmp_path):
    executable = tmp_path / "micromamba"
    executable.write_text("tool")
    assert conda.find_conda({"MICROMAMBA_EXE": str(executable)}) == str(executable)


def test_create_command_uses_only_conda_forge_and_exact_runtime_versions(tmp_path):
    command = conda.create_command("/tools/conda", tmp_path / "env")
    assert command[:6] == [
        "/tools/conda",
        "create",
        "--yes",
        "--override-channels",
        "--channel",
        "conda-forge",
    ]
    assert "--strict-channel-priority" in command
    assert "python=3.14.9" in command
    assert "numpy=2.5.4" in command
    assert "pyspharm=1.0.9" in command
    assert not any("uv" in item or "pip" in item for item in command)


def test_verify_pyspharm_environment_returns_resolved_build_and_hash(tmp_path):
    _write_records(tmp_path, _records())
    identity = conda.verify_pyspharm_environment(tmp_path)
    assert identity["version"] == "1.0.9"
    assert identity["build_string"] == "np2py314h2766e9b_1017"
    assert identity["channel"] == "conda-forge"
    assert identity["sha256"] == _records()[0]["sha256"]
    assert identity["url"].endswith("pyspharm-1.0.9-np2py314h2766e9b_1017.conda")


def test_verify_pyspharm_environment_rejects_python_version_drift(tmp_path):
    records = _records()
    records[1]["version"] = "3.13.15"
    _write_records(tmp_path, records)
    with pytest.raises(ValueError, match="expected python 3.14.9"):
        conda.verify_pyspharm_environment(tmp_path)


def test_verify_pyspharm_environment_rejects_substitute_conda_package(tmp_path):
    records = _records() + [{"name": "pyspharm-alt", "version": "1.0.9"}]
    _write_records(tmp_path, records)
    with pytest.raises(ValueError, match="unexpected pyspharm distribution"):
        conda.verify_pyspharm_environment(tmp_path)


def test_verify_import_path_requires_module_inside_selected_prefix(tmp_path):
    module = tmp_path / "lib" / "python3.14" / "site-packages" / "spharm" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    assert conda.verify_import_path(tmp_path, module) == module
    with pytest.raises(ValueError, match="escaped the conda environment"):
        conda.verify_import_path(tmp_path, Path("/usr/lib/python3/spharm.py"))


def test_runtime_probe_rejects_precision_or_abi_drift(monkeypatch, tmp_path):
    prefix = tmp_path / "env"
    payload = {
        "module_path": str(prefix / "lib/python3.14/site-packages/spharm/__init__.py"),
        "native_module_path": str(prefix / "lib/python3.14/site-packages/_spherepack.so"),
        "native_sha256": "hash",
        "python": "3.14.9",
        "python_version": "3.14.9",
        "numpy": "2.5.4",
        "numpy_abi": 33554432,
        "analysis_dtype": "complex128",
        "synthesis_dtype": "float32",
        "linked_libraries": "libm.so.6",
    }
    completed = subprocess.CompletedProcess([], 0, json.dumps(payload) + "\n", "")
    monkeypatch.setattr(conda.subprocess, "run", lambda *args, **kwargs: completed)
    with pytest.raises(ValueError, match="did not preserve float32/complex64"):
        conda.runtime_probe(prefix)


def test_runtime_probe_rejects_python_version_drift(monkeypatch, tmp_path):
    prefix = tmp_path / "env"
    payload = {
        "module_path": str(prefix / "lib/python3.14/site-packages/spharm/__init__.py"),
        "native_module_path": str(prefix / "lib/python3.14/site-packages/_spherepack.so"),
        "native_sha256": "hash",
        "python": "3.13.15",
        "python_version": "3.13.15",
        "numpy": "2.5.4",
        "numpy_abi": 33554432,
        "analysis_dtype": "complex64",
        "synthesis_dtype": "float32",
        "linked_libraries": "libm.so.6",
    }
    completed = subprocess.CompletedProcess([], 0, json.dumps(payload) + "\n", "")
    monkeypatch.setattr(conda.subprocess, "run", lambda *args, **kwargs: completed)
    with pytest.raises(ValueError, match="expected 3.14.9 and 2.5.4"):
        conda.runtime_probe(prefix)


def test_installed_environment_exports_record_explicit_and_solver_views(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "export", "")

    monkeypatch.setattr(conda.subprocess, "run", run)
    exports = conda.installed_environment_exports("conda", tmp_path)
    assert exports == {"explicit": "export", "environment": "export"}
    assert calls[0][:3] == ["conda", "list", "--explicit"]
    assert calls[1][:3] == ["conda", "env", "export"]
