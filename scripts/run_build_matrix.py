from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
from sht_bench.conda_environment import (
    CONDA_CHANNEL,
    NUMPY_VERSION as PYSPHARM_NUMPY_VERSION,
    PYTHON_VERSION as PYSPHARM_PYTHON_VERSION,
    PYSPHARM_VERSION,
    create_command as conda_create_command,
    environment_python as conda_environment_python,
    find_conda,
    installed_environment_exports,
    runtime_probe,
    verify_pyspharm_environment,
)

BACKEND_NAMES = ("ducc", "shtns", "pyshtools", "pyspharm")
ARM64_MACHINES = {"aarch64", "arm64"}

VERSIONS = {
    "ducc": "0.41.0",
    "shtns": "3.7.5",
    "pyshtools": "4.14.1",
    "pyspharm": PYSPHARM_VERSION,
}

PROVENANCE = {
    "wheel": {
        "ducc": {
            "build_kind": "pypi-wheel",
            "build_source": f"ducc0=={VERSIONS['ducc']}",
        },
        "shtns": {
            "build_kind": "git-source",
            "build_source": "https://gricad-gitlab.univ-grenoble-alpes.fr/schaeffn/shtns.git@master",
            "build_note": (
                "SHTns is compiled from its Git checkout in both "
                "benchmark environments."
            ),
        },
        "pyshtools": {
            "build_kind": "pypi-wheel",
            "build_source": f"pyshtools=={VERSIONS['pyshtools']}",
        },
    },
    "source": {
        "ducc": {
            "build_kind": "git-source",
            "build_source": "https://github.com/mreineck/ducc.git@ducc0",
            "build_note": "DUCC source builds use the package's local build configuration.",
        },
        "shtns": {
            "build_kind": "git-source",
            "build_source": "https://gricad-gitlab.univ-grenoble-alpes.fr/schaeffn/shtns.git@master",
            "build_note": (
                "Same Git source policy in both environments; resolved SHA is recorded."
            ),
        },
        "pyshtools": {
            "build_kind": "pypi-sdist",
            "build_source": f"pyshtools=={VERSIONS['pyshtools']}",
        },
    },
    "conda-forge": {
        "pyspharm": {
            "build_kind": "conda-forge",
            "build_origin": CONDA_CHANNEL,
            "build_source": f"conda-forge::pyspharm={VERSIONS['pyspharm']}",
        },
    },
}


def default_backends(machine: str | None = None) -> tuple[str, ...]:
    """Return the default backend set for the host architecture."""
    host = (machine or platform.machine()).lower()
    if host in ARM64_MACHINES:
        return ("ducc", "shtns")
    return BACKEND_NAMES


def parse_backends(value: str) -> tuple[str, ...]:
    if value.lower() == "all":
        return BACKEND_NAMES
    backends = tuple(
        dict.fromkeys(
            item.strip().lower() for item in value.split(",") if item.strip()
        )
    )
    unknown = sorted(set(backends) - set(BACKEND_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown backend(s): {', '.join(unknown)}")
    if not backends:
        raise argparse.ArgumentTypeError("expected at least one backend")
    return backends


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def venv_python(path: Path) -> Path:
    if os.name == "nt":
        return path / "Scripts" / "python.exe"
    return path / "bin" / "python"


def uv_install(
    python: Path,
    requirement: str,
    *,
    binary: str | None = None,
    env: dict[str, str],
    no_build_isolation: bool = False,
) -> None:
    command = ["uv", "pip", "install", "--python", str(python)]
    if Path(requirement).is_dir():
        command.extend(["--reinstall", "--no-cache"])
    if no_build_isolation:
        command.append("--no-build-isolation")
    if binary is not None:
        package = requirement.split("==", 1)[0]
        flag = "--only-binary" if binary == "only" else "--no-binary"
        command.extend([flag, package])
    command.append(requirement)
    run(command, env=env)


def wheel_available(
    python: Path, requirement: str, *, env: dict[str, str]
) -> tuple[bool, str]:
    """Return whether uv can resolve a compatible wheel for *requirement*."""
    package = requirement.split("==", 1)[0]
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "--dry-run",
        "--only-binary",
        package,
        requirement,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    detail = (completed.stderr or completed.stdout).strip()
    return completed.returncode == 0, detail


def source_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CC", shutil.which("gcc") or shutil.which("cc") or "cc")
    env.setdefault("CXX", shutil.which("g++") or shutil.which("c++") or "c++")
    env.setdefault("FC", shutil.which("gfortran") or "gfortran")
    return env


def _attempt_build(
    backend: str,
    action: Callable[[], None],
    failures: dict[str, str],
    *,
    strict_builds: bool,
) -> None:
    try:
        action()
    except subprocess.CalledProcessError as exc:
        failures[backend] = (
            f"command exited with status {exc.returncode}: "
            + " ".join(map(str, exc.cmd))
        )
        print(
            f"warning: failed to install {backend}; continuing with remaining backends",
            file=sys.stderr,
        )
        if strict_builds:
            raise


def prepare_pyspharm_environment(*, reuse: bool) -> tuple[Path, dict]:
    """Create/select the dedicated conda-forge backend environment."""
    conda = find_conda()
    prefix = ROOT / ".conda-bench-pyspharm"
    if prefix.exists() and reuse:
        verify_pyspharm_environment(prefix)
    else:
        if prefix.exists():
            run([conda, "env", "remove", "--yes", "--prefix", str(prefix)])
        command = conda_create_command(
            conda,
            prefix,
            python=PYSPHARM_PYTHON_VERSION,
            numpy=PYSPHARM_NUMPY_VERSION,
            pyspharm=PYSPHARM_VERSION,
        )
        run(command)
    python = conda_environment_python(prefix)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "-e",
            ".[plot]",
            "pytest",
            "scipy",
        ]
    )
    package = verify_pyspharm_environment(prefix)
    runtime = runtime_probe(prefix)
    identity = {
        **package,
        **runtime,
        "build_kind": "conda-forge",
        "build_origin": CONDA_CHANNEL,
        "build_source": f"conda-forge::pyspharm={PYSPHARM_VERSION}",
        "prefix": str(prefix.resolve()),
        "python_executable": str(python.resolve()),
        "python_version": PYSPHARM_PYTHON_VERSION,
        "numpy_version": PYSPHARM_NUMPY_VERSION,
    }
    exports = installed_environment_exports(conda, prefix)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    explicit = results / "pyspharm-conda-forge-linux-64.explicit.txt"
    environment = results / "pyspharm-conda-forge-environment.yml"
    explicit.write_text(exports["explicit"])
    environment.write_text(exports["environment"])
    identity["explicit_export"] = str(explicit.resolve())
    identity["explicit_export_sha256"] = hashlib.sha256(explicit.read_bytes()).hexdigest()
    identity["environment_export"] = str(environment.resolve())
    PROVENANCE["conda-forge"]["pyspharm"].update(identity)
    return python, identity


def create_environment(
    mode: str,
    python_spec: str,
    reuse: bool,
    *,
    backends: Iterable[str] | None = None,
    strict_builds: bool = False,
) -> tuple[Path | None, dict[str, str], dict[str, str], dict[str, str], Path | None]:
    selected_backends = tuple(backends or default_backends())
    uv_backends = tuple(name for name in selected_backends if name != "pyspharm")
    env_dir = ROOT / f".venv-bench-{mode}"
    if uv_backends:
        if env_dir.exists() and not reuse:
            shutil.rmtree(env_dir)
        if not env_dir.exists():
            run(["uv", "venv", "--python", python_spec, str(env_dir)])
        python: Path | None = venv_python(env_dir)
    else:
        python = None
    build_env = source_environment() if mode == "source" else os.environ.copy()
    failures: dict[str, str] = {}
    unavailable: dict[str, str] = {}
    pyspharm_python: Path | None = None

    native_paths = {}
    required_native = set(uv_backends) & ({"ducc", "shtns"} if mode == "source" else {"shtns"})
    if required_native:
        from sht_bench.sources import checkout
        source_root = ROOT / ".sources" / "matrix"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        source_lock = ROOT / ".builds" / f"matrix-{stamp}.lock.json"
        acquired = checkout(source_root, source_lock)
        for name in required_native:
            entry = acquired["sources"][name]
            native_paths[name] = str(source_root / name)
            PROVENANCE[mode][name].update(build_source=entry["origin"],build_sha=entry["sha"],build_ref=entry["ref"],source_lock=str(source_lock))

    # Install the harness without pulling the compiled comparison backends.
    if python is not None:
        run(
            ["uv", "pip", "install", "--python", str(python), "-e", ".[plot]"],
            env=build_env,
        )

    if mode == "wheel":
        wheel_requirements = {
            "ducc": f"ducc0=={VERSIONS['ducc']}",
            "pyshtools": f"pyshtools=={VERSIONS['pyshtools']}",
        }
        installs: dict[str, Callable[[], None]] = {}
        for backend, requirement in wheel_requirements.items():
            if backend not in selected_backends:
                continue
            available, detail = wheel_available(python, requirement, env=build_env)
            if not available:
                unavailable[backend] = (
                    f"no compatible wheel resolved for {platform.system()} "
                    f"{platform.machine()}: {detail}"
                )
                print(
                    f"note: skipping {backend} wheel on {platform.machine()}; "
                    "no compatible binary distribution is available",
                    file=sys.stderr,
                )
                continue
            installs[backend] = lambda requirement=requirement: uv_install(
                python, requirement, binary="only", env=build_env
            )

        # SHTns publishes no wheel. It is compiled from Git in the
        # wheel-labelled environment and plotted as a source series.
        if "shtns" in selected_backends:
            installs["shtns"] = lambda: uv_install(
                python,
                native_paths["shtns"],
                env=build_env,
            )
    else:
        source_installs: dict[str, Callable[[], None]] = {
            "ducc": lambda: uv_install(
                python,
                native_paths.get("ducc", ""),
                env=build_env,
            ),
            "shtns": lambda: uv_install(
                python,
                native_paths.get("shtns", ""),
                env=build_env,
            ),
            "pyshtools": lambda: uv_install(
                python,
                f"pyshtools=={VERSIONS['pyshtools']}",
                binary="no",
                env=build_env,
            ),
        }
        installs = {
            backend: action
            for backend, action in source_installs.items()
            if backend in selected_backends
        }

    if "pyspharm" in selected_backends:
        try:
            pyspharm_python, _ = prepare_pyspharm_environment(reuse=reuse)
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            unavailable["pyspharm"] = f"conda-forge environment unavailable: {exc}"
            print(unavailable["pyspharm"], file=sys.stderr)
            if strict_builds:
                raise

    for backend, action in installs.items():
        _attempt_build(
            backend,
            action,
            failures,
            strict_builds=strict_builds,
        )

    return python, build_env, failures, unavailable, pyspharm_python


def _write_csv(records: list[dict[str, object]], path: Path) -> None:
    if not records:
        return
    fieldnames = [key for key in records[0] if key != "samples_s"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def annotate_results(output: Path, mode: str) -> None:
    records = json.loads(output.read_text())
    for record in records:
        backend = record["backend"]
        provenance_mode = "conda-forge" if backend == "pyspharm" else mode
        record["build_mode"] = provenance_mode
        record.update(PROVENANCE[provenance_mode][backend])

    output.write_text(json.dumps(records, indent=2) + "\n")
    _write_csv(records, output.with_suffix(".csv"))


def write_environment_files(
    python: Path | None,
    mode: str,
    env: dict[str, str],
    build_failures: dict[str, str],
    unavailable_packages: dict[str, str],
    backends: Iterable[str],
    pyspharm_python: Path | None = None,
) -> None:
    results = ROOT / "results"
    results.mkdir(exist_ok=True)

    if python is not None:
        completed = subprocess.run(
            ["uv", "pip", "freeze", "--python", str(python)],
            cwd=ROOT,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        (results / f"{mode}-environment.txt").write_text(completed.stdout)
    compiler_env = {
        key: env.get(key)
        for key in (
            "CC",
            "CXX",
            "FC",
            "CFLAGS",
            "CXXFLAGS",
            "FFLAGS",
            "LDFLAGS",
        )
        if env.get(key)
    }
    selected_backends = tuple(backends)
    uv_backends = tuple(name for name in selected_backends if name != "pyspharm")
    manifest = {
        "mode": mode,
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": str(python) if python is not None else None,
        "requested_backends": list(uv_backends),
        "packages": {
            backend: PROVENANCE[mode][backend]
            for backend in uv_backends
            if backend in PROVENANCE[mode]
        },
        "compiler_environment": compiler_env,
        "unavailable_packages": unavailable_packages,
        "build_failures": build_failures,
    }
    (results / f"{mode}-provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    if "pyspharm" in selected_backends:
        conda_manifest = {
            "mode": "conda-forge",
            "platform": platform.system(),
            "machine": platform.machine(),
            "requested_backends": ["pyspharm"],
            "python": str(pyspharm_python) if pyspharm_python else None,
            "package": PROVENANCE["conda-forge"]["pyspharm"],
            "unavailable": unavailable_packages.get("pyspharm"),
        }
        (results / "conda-forge-provenance.json").write_text(
            json.dumps(conda_manifest, indent=2) + "\n"
        )


def run_benchmark(
    python: Path | None,
    mode: str,
    bench_args: Iterable[str],
    build_env: dict[str, str],
    build_failures: dict[str, str],
    unavailable_packages: dict[str, str],
    backends: Iterable[str],
    pyspharm_python: Path | None = None,
) -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    selected = tuple(backends)
    uv_backends = tuple(name for name in selected if name != "pyspharm")
    psh_backends = ("pyspharm",) if "pyspharm" in selected else ()

    def run_group(interpreter: Path, names: tuple[str, ...], label: str) -> list[dict]:
        if not names:
            return []
        intermediate = results_dir / f"{mode}-{label}.json"
        command = [
            str(interpreter),
            "-m",
            "sht_bench.cli",
            "run",
            "--output",
            str(intermediate),
        ]
        forwarded = list(bench_args)
        cleaned: list[str] = []
        index = 0
        while index < len(forwarded):
            item = forwarded[index]
            if item == "--backend":
                index += 2
                continue
            if item.startswith("--backend="):
                index += 1
                continue
            cleaned.append(item)
            index += 1
        command.extend(["--backend", ",".join(names), *cleaned])
        run(command, env=build_env)
        return json.loads(intermediate.read_text())

    records = run_group(python, uv_backends, "uv") if python is not None else []
    if psh_backends and pyspharm_python is not None:
        records.extend(run_group(pyspharm_python, psh_backends, "conda-forge"))
    elif psh_backends:
        unavailable_packages["pyspharm"] = unavailable_packages.get(
            "pyspharm", "dedicated conda-forge environment is unavailable"
        )

    for record in records:
        backend = record["backend"]
        record["build_mode"] = "conda-forge" if backend == "pyspharm" else mode
        provenance_mode = "conda-forge" if backend == "pyspharm" else mode
        record.update(PROVENANCE[provenance_mode].get(backend, {}))
    if uv_backends and psh_backends:
        output_stem = f"{mode}-and-conda-forge"
    elif psh_backends:
        output_stem = "conda-forge"
    else:
        output_stem = mode
    output = results_dir / f"{output_stem}.json"
    output.write_text(json.dumps(records, indent=2) + "\n")
    _write_csv(records, output.with_suffix(".csv"))
    write_environment_files(
        python,
        mode,
        build_env,
        build_failures,
        unavailable_packages,
        backends,
        pyspharm_python,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create isolated uv and conda-forge environments and run the SHT benchmark."
        )
    )
    parser.add_argument(
        "--build",
        choices=("wheel", "source", "both"),
        default="both",
        help="package provenance to benchmark (default: both)",
    )
    parser.add_argument(
        "--backend",
        type=parse_backends,
        default=None,
        help=(
            "backends to provision: comma-separated names or 'all'; default is "
            "ducc,shtns on arm64/aarch64 and all backends elsewhere"
        ),
    )
    parser.add_argument(
        "--python",
        default="3.13",
        help="interpreter/version passed to 'uv venv' (default: 3.13)",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="reuse an existing .venv-bench-{wheel,source} environment",
    )
    parser.add_argument(
        "--strict-builds",
        action="store_true",
        help="abort if an available backend fails to install",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="create environments and provenance files without running a benchmark",
    )
    parser.add_argument(
        "bench_args",
        nargs=argparse.REMAINDER,
        help="arguments after '--' are forwarded to 'sht-bench run'",
    )
    return parser


def _has_forwarded_backend(bench_args: Iterable[str]) -> bool:
    return any(
        arg == "--backend" or arg.startswith("--backend=") for arg in bench_args
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backends = args.backend or default_backends()
    print(
        f"benchmark backends for {platform.machine()}: {','.join(backends)}",
        flush=True,
    )

    bench_args = args.bench_args
    if bench_args[:1] == ["--"]:
        bench_args = bench_args[1:]
    if not _has_forwarded_backend(bench_args):
        bench_args = ["--backend", ",".join(backends), *bench_args]

    modes = ("wheel", "source") if args.build == "both" else (args.build,)
    for mode in modes:
        mode_backends = backends
        if args.build == "both" and mode == "source":
            mode_backends = tuple(backend for backend in backends if backend != "pyspharm")
        python, build_env, build_failures, unavailable_packages, pyspharm_python = create_environment(
            mode,
            args.python,
            args.reuse,
            backends=mode_backends,
            strict_builds=args.strict_builds,
        )
        if args.setup_only:
            write_environment_files(
                python,
                mode,
                build_env,
                build_failures,
                unavailable_packages,
                mode_backends,
                pyspharm_python,
            )
            continue
        run_benchmark(
            python,
            mode,
            bench_args,
            build_env,
            build_failures,
            unavailable_packages,
            mode_backends,
            pyspharm_python,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
