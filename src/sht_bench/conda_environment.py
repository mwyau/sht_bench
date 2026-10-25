"""Helpers for the separately managed conda-forge pyspharm environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping


PYSPHARM_VERSION = "1.0.9"
PYTHON_VERSION = "3.13.15"
NUMPY_VERSION = "2.5.4"
CONDA_CHANNEL = "conda-forge"


def find_conda(environ: Mapping[str, str] | None = None) -> str:
    """Find an existing conda-compatible executable without touching base."""
    environment = os.environ if environ is None else environ
    for key in ("CONDA_EXE", "MAMBA_EXE", "MICROMAMBA_EXE"):
        candidate = environment.get(key)
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    for name in ("micromamba", "mamba", "conda"):
        candidate = shutil.which(name)
        if candidate:
            return str(Path(candidate).resolve())
    raise FileNotFoundError(
        "conda, mamba, or micromamba is required to provision pyspharm"
    )


def environment_python(prefix: Path) -> Path:
    """Return the interpreter path inside a conda prefix."""
    if os.name == "nt":
        return prefix / "python.exe"
    return prefix / "bin" / "python"


def create_command(
    conda: str,
    prefix: Path,
    *,
    python: str = PYTHON_VERSION,
    numpy: str = NUMPY_VERSION,
    pyspharm: str = PYSPHARM_VERSION,
) -> list[str]:
    """Build an explicit, single-channel environment creation command."""
    return [
        conda,
        "create",
        "--yes",
        "--override-channels",
        "--channel",
        CONDA_CHANNEL,
        "--strict-channel-priority",
        "--prefix",
        str(prefix),
        f"python={python}",
        f"numpy={numpy}",
        f"pyspharm={pyspharm}",
    ]


def conda_records(prefix: Path) -> list[dict]:
    """Read the installed package records directly from the target prefix."""
    metadata = prefix / "conda-meta"
    if not metadata.is_dir():
        raise FileNotFoundError(f"not a conda environment: {prefix}")
    return [json.loads(path.read_text()) for path in sorted(metadata.glob("*.json"))]


def verify_pyspharm_environment(
    prefix: Path,
    *,
    python: str = PYTHON_VERSION,
    numpy: str = NUMPY_VERSION,
    pyspharm: str = PYSPHARM_VERSION,
) -> dict:
    """Validate exact package identity and return its resolved artifact record."""
    records = conda_records(prefix)
    named = [str(record.get("name", "")).lower() for record in records]
    unexpected = sorted(
        name for name in named if name.startswith("pyspharm") and name != "pyspharm"
    )
    if unexpected:
        raise ValueError(f"unexpected pyspharm distribution(s) in conda prefix: {unexpected}")
    selected = [record for record in records if str(record.get("name", "")).lower() == "pyspharm"]
    if len(selected) != 1:
        raise ValueError(f"expected one conda pyspharm record, found {len(selected)}")
    package = selected[0]
    if str(package.get("version")) != pyspharm:
        raise ValueError(
            f"expected conda pyspharm {pyspharm}, found {package.get('version')}"
        )
    channel = str(package.get("channel", package.get("schannel", "")))
    url = str(package.get("url", ""))
    if CONDA_CHANNEL not in channel and CONDA_CHANNEL not in url:
        raise ValueError(f"pyspharm package did not come from {CONDA_CHANNEL}: {channel or url}")
    resolved = {
        "name": "pyspharm",
        "version": str(package.get("version")),
        "build_string": package.get("build", package.get("build_string")),
        "build_number": package.get("build_number"),
        "channel": channel or CONDA_CHANNEL,
        "url": package.get("url"),
        "sha256": package.get("sha256"),
        "md5": package.get("md5"),
        "depends": package.get("depends", []),
        "subdir": package.get("subdir"),
    }
    if not resolved["build_string"]:
        raise ValueError("conda pyspharm record has no resolved build string")
    for package_name, expected in (("python", python), ("numpy", numpy)):
        installed = [r for r in records if str(r.get("name", "")).lower() == package_name]
        if len(installed) != 1 or str(installed[0].get("version")) != expected:
            actual = [r.get("version") for r in installed]
            raise ValueError(f"expected {package_name} {expected}, found {actual}")
    return resolved


def verify_import_path(prefix: Path, module_path: str | Path) -> Path:
    """Require the imported Python module to come from the selected prefix."""
    root = prefix.resolve()
    path = Path(module_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"spharm import escaped the conda environment: {path}")
    return path


def runtime_probe(prefix: Path) -> dict:
    """Import spharm in the target prefix and report binary and dtype facts."""
    python = environment_python(prefix)
    code = r'''import hashlib,importlib,importlib.metadata as md,json,pathlib,subprocess,sys
import numpy as np
import spharm
native=importlib.import_module('_spherepack')
module_path=pathlib.Path(spharm.__file__).resolve()
native_path=pathlib.Path(native.__file__).resolve()
for path in (module_path,native_path):
    if not path.is_relative_to(pathlib.Path(sys.prefix).resolve()): raise RuntimeError('spharm import escaped target conda prefix')
for dist in md.distributions():
    name=(dist.metadata.get('Name') or '').lower().replace('_','-')
    if name.startswith('pyspharm') and name!='pyspharm': raise RuntimeError('unexpected Python distribution '+name)
plan=spharm.Spharmt(16,8,rsphere=1.0,gridtype='gaussian',legfunc='stored')
analysis=plan.grdtospec(np.zeros((8,16),dtype=np.float32),ntrunc=3)
synthesis=plan.spectogrd(np.zeros(10,dtype=np.complex64))
ldd=subprocess.run(['ldd',str(native_path)],capture_output=True,text=True).stdout if sys.platform.startswith('linux') else None
print(json.dumps({'module_path':str(module_path),'native_module_path':str(native_path),'native_sha256':hashlib.sha256(native_path.read_bytes()).hexdigest(),'python':sys.version,'python_version':'.'.join(map(str,sys.version_info[:3])),'numpy':np.__version__,'numpy_abi':np._core._multiarray_umath._get_ndarray_c_version(),'analysis_dtype':str(analysis.dtype),'synthesis_dtype':str(synthesis.dtype),'linked_libraries':ldd}))'''
    completed = subprocess.run(
        [str(python), "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    result = next(
        json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")
    )
    verify_import_path(prefix, result["module_path"])
    verify_import_path(prefix, result["native_module_path"])
    if result["analysis_dtype"] != "complex64" or result["synthesis_dtype"] != "float32":
        raise ValueError("pyspharm runtime did not preserve float32/complex64 precision")
    if result["python_version"] != PYTHON_VERSION or result["numpy"] != NUMPY_VERSION:
        raise ValueError(
            f"pyspharm runtime has Python {result['python_version']} and NumPy {result['numpy']}, "
            f"expected {PYTHON_VERSION} and {NUMPY_VERSION}"
        )
    return result


def installed_environment_exports(conda: str, prefix: Path) -> dict[str, str]:
    """Return reproducible explicit and human-readable environment exports."""
    explicit = subprocess.run(
        [conda, "list", "--explicit", "--prefix", str(prefix)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    environment = subprocess.run(
        [conda, "env", "export", "--prefix", str(prefix)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"explicit": explicit, "environment": environment}
