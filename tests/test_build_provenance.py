"""The conda binary stays separate from uv wheel provenance."""
import importlib.util
import json
from pathlib import Path
import sys


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_build_matrix.py"
_SPEC = importlib.util.spec_from_file_location("run_build_matrix", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
run_build_matrix = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = run_build_matrix
_SPEC.loader.exec_module(run_build_matrix)


def test_pyspharm_is_written_only_to_conda_forge_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(run_build_matrix, "ROOT", tmp_path)

    run_build_matrix.write_environment_files(
        python=None,
        mode="wheel",
        env={},
        build_failures={},
        unavailable_packages={},
        backends=("ducc", "pyspharm"),
        pyspharm_python=tmp_path / ".conda-bench-pyspharm" / "bin" / "python",
    )

    wheel = json.loads((tmp_path / "results" / "wheel-provenance.json").read_text())
    conda = json.loads((tmp_path / "results" / "conda-forge-provenance.json").read_text())
    assert wheel["requested_backends"] == ["ducc"]
    assert "pyspharm" not in wheel["packages"]
    assert "pyspharm_conda_python" not in wheel
    assert conda["requested_backends"] == ["pyspharm"]
    assert conda["package"]["build_kind"] == "conda-forge"
