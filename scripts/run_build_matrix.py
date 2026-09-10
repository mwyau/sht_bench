from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]

BACKEND_NAMES = ("ducc", "shtns", "pyshtools", "pyspharm")
ARM64_MACHINES = {"aarch64", "arm64"}

VERSIONS = {
    "ducc": "0.41.0",
    "shtns": "3.7.5",
    "pyshtools": "4.14.1",
    "pyspharm": "1.2.1",
}

PROVENANCE = {
    "wheel": {
        "ducc": {
            "build_kind": "pypi-wheel",
            "build_source": f"ducc0=={VERSIONS['ducc']}",
        },
        "shtns": {
            "build_kind": "pypi-sdist",
            "build_source": f"shtns=={VERSIONS['shtns']}",
            "build_note": (
                "SHTns 3.7.5 has no PyPI wheel; it is compiled locally in both "
                "benchmark environments."
            ),
        },
        "pyshtools": {
            "build_kind": "pypi-wheel",
            "build_source": f"pyshtools=={VERSIONS['pyshtools']}",
        },
        "pyspharm": {
            "build_kind": "pypi-wheel",
            "build_source": f"pyspharm-syl=={VERSIONS['pyspharm']}",
        },
    },
    "source": {
        "ducc": {
            "build_kind": "pypi-sdist",
            "build_source": f"ducc0=={VERSIONS['ducc']}",
            "build_note": "DUCC source builds use the package's local build configuration.",
        },
        "shtns": {
            "build_kind": "pypi-sdist",
            "build_source": f"shtns=={VERSIONS['shtns']}",
            "build_note": (
                "Same release artifact as in the wheel-labelled environment because "
                "SHTns publishes no wheel."
            ),
        },
        "pyshtools": {
            "build_kind": "pypi-sdist",
            "build_source": f"pyshtools=={VERSIONS['pyshtools']}",
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


def create_environment(
    mode: str,
    python_spec: str,
    reuse: bool,
    *,
    backends: Iterable[str] | None = None,
    strict_builds: bool = False,
) -> tuple[Path, dict[str, str], dict[str, str], dict[str, str]]:
    selected_backends = tuple(backends or default_backends())
    env_dir = ROOT / f".venv-bench-{mode}"
    if env_dir.exists() and not reuse:
        shutil.rmtree(env_dir)

    if not env_dir.exists():
        run(["uv", "venv", "--python", python_spec, str(env_dir)])

    python = venv_python(env_dir)
    build_env = source_environment() if mode == "source" else os.environ.copy()
    failures: dict[str, str] = {}
    unavailable: dict[str, str] = {}

    # Install the harness without pulling the compiled comparison backends.
    run(
        ["uv", "pip", "install", "--python", str(python), "-e", ".[plot]"],
        env=build_env,
    )

    if mode == "wheel":
        wheel_requirements = {
            "ducc": f"ducc0=={VERSIONS['ducc']}",
            "pyshtools": f"pyshtools=={VERSIONS['pyshtools']}",
            "pyspharm": f"pyspharm-syl=={VERSIONS['pyspharm']}",
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

        # SHTns publishes no wheel. It is compiled from its sdist in the
        # wheel-labelled environment and plotted as a source series.
        if "shtns" in selected_backends:
            installs["shtns"] = lambda: uv_install(
                python,
                f"shtns=={VERSIONS['shtns']}",
                binary="no",
                env=build_env,
            )
    else:
        source_installs: dict[str, Callable[[], None]] = {
            "ducc": lambda: uv_install(
                python,
                f"ducc0=={VERSIONS['ducc']}",
                binary="no",
                env=build_env,
            ),
            "shtns": lambda: uv_install(
                python,
                f"shtns=={VERSIONS['shtns']}",
                binary="no",
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

    for backend, action in installs.items():
        _attempt_build(
            backend,
            action,
            failures,
            strict_builds=strict_builds,
        )

    return python, build_env, failures, unavailable


def _write_csv(records: list[dict[str, object]], path: Path) -> None:
    if not records:
        return
    fieldnames = [key for key in records[0] if key != "samples_s"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def annotate_results(output: Path, mode: str) -> None:
    records = json.loads(output.read_text())
    for record in records:
        backend = record["backend"]
        record["build_mode"] = mode
        record.update(PROVENANCE[mode][backend])

    output.write_text(json.dumps(records, indent=2) + "\n")
    _write_csv(records, output.with_suffix(".csv"))


def write_environment_files(
    python: Path,
    mode: str,
    env: dict[str, str],
    build_failures: dict[str, str],
    unavailable_wheels: dict[str, str],
    backends: Iterable[str],
) -> None:
    results = ROOT / "results"
    results.mkdir(exist_ok=True)

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
    manifest = {
        "mode": mode,
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": str(python),
        "requested_backends": list(selected_backends),
        "packages": {
            backend: PROVENANCE[mode][backend]
            for backend in selected_backends
            if backend in PROVENANCE[mode]
        },
        "compiler_environment": compiler_env,
        "unavailable_wheels": unavailable_wheels,
        "build_failures": build_failures,
    }
    (results / f"{mode}-provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def run_benchmark(
    python: Path,
    mode: str,
    bench_args: Iterable[str],
    build_env: dict[str, str],
    build_failures: dict[str, str],
    unavailable_wheels: dict[str, str],
    backends: Iterable[str],
) -> None:
    output = ROOT / "results" / f"{mode}.json"
    command = [
        str(python),
        "-m",
        "sht_bench.cli",
        "run",
        "--output",
        str(output),
        *bench_args,
    ]
    run(command, env=build_env)
    annotate_results(output, mode)
    write_environment_files(
        python,
        mode,
        build_env,
        build_failures,
        unavailable_wheels,
        backends,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create isolated wheel/source environments and run the SHT benchmark."
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
        python, build_env, build_failures, unavailable_wheels = create_environment(
            mode,
            args.python,
            args.reuse,
            backends=backends,
            strict_builds=args.strict_builds,
        )
        if args.setup_only:
            write_environment_files(
                python,
                mode,
                build_env,
                build_failures,
                unavailable_wheels,
                backends,
            )
            continue
        run_benchmark(
            python,
            mode,
            bench_args,
            build_env,
            build_failures,
            unavailable_wheels,
            backends,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
