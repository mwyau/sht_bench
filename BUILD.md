# Build configuration

SHT timing depends on both the transform implementation and the binary used to execute it. `sht_bench` therefore separates binary-package and locally compiled environments and records the package source associated with each result.

## Package versions

| Backend | Binary-package environment | Source-build environment |
| --- | --- | --- |
| DUCC | PyPI `ducc0==0.41.0` wheel when available for the platform | PyPI `ducc0==0.41.0` source distribution |
| SHTns | PyPI `shtns==3.7.5` source distribution | same `shtns==3.7.5` source distribution |
| pyshtools / SHTOOLS | PyPI `pyshtools==4.14.1` wheel when available for the platform | PyPI `pyshtools==4.14.1` source distribution |
| pyspharm-syl / Spherepack | PyPI `pyspharm-syl==1.2.1` wheel when available for the platform | not benchmarked |

The binary-package environment requires an actual compatible wheel for backends designated as wheel builds. Before installation, the runner asks `uv` to resolve the pinned package with `--only-binary` in dry-run mode. If the current operating system, architecture, and Python version have no compatible wheel, that backend is recorded under `unavailable_wheels` and omitted from the wheel benchmark. It is not silently compiled from source.

SHTns 3.7.5 has no PyPI wheel, so both benchmark environments compile the same source distribution. Plotting therefore shows one SHTns source series rather than treating the two environments as a wheel-versus-source comparison.

`pyspharm-syl` is benchmarked from its published wheel only. The source-build environment contains DUCC, SHTns, and pyshtools/SHTOOLS on platforms where those backends are selected.

## Architecture-specific backend selection

The lower-level build runner is architecture-aware. On `aarch64` and `arm64`, its default backend set is limited to DUCC and SHTns. On other architectures the default remains all four backends. `--backend` can override the build-runner selection explicitly.

This keeps the ARM64 benchmark focused on the two implementations intended for direct x86-64/AArch64 comparison and avoids spending build time on pyshtools or pyspharm environments that are not part of that comparison. Source builds set GNU-compatible `CC`, `CXX`, and `FC` defaults when the caller has not already supplied them.

For an ARM64 run, use a separate result directory so measurements can later be compared with x86-64 results without overwriting cell files:

```bash
uv run --extra plot sht-bench matrix \
  --build both \
  --backend ducc,shtns \
  --threads auto \
  --output results/matrix-arm64 \
  --plot
```

If the platform has no compatible DUCC wheel, the wheel environment records that exclusion and still benchmarks the locally built DUCC source series. SHTns is source-built in either environment.

## Build modes

The `matrix` command accepts:

```text
installed  use packages in the active environment
wheel      create the binary-package environment
source     create the source-build environment
both       run wheel and source environments
```

For example:

```bash
uv run --extra plot sht-bench matrix \
  --build both \
  --threads 1,2,4,8 \
  --plot
```

The lower-level build runner remains available:

```bash
python scripts/run_build_matrix.py --build source -- --threads 1
python scripts/run_build_matrix.py --build wheel -- --threads 1
```

On ARM64, the equivalent direct invocation defaults to DUCC and SHTns. The backend set can also be stated explicitly:

```bash
python scripts/run_build_matrix.py \
  --build both \
  --backend ducc,shtns \
  -- --threads 1,2,4
```

`--setup-only` creates an isolated environment and writes its provenance without running a transform benchmark. The matrix command uses this mode internally so environment creation does not depend on any single backend being available.

The isolated environments are `.venv-bench-source` and `.venv-bench-wheel`.

## Source compilers

Caller-supplied `CC`, `CXX`, and `FC` values are preserved. The source environment otherwise selects available GNU compiler defaults for packages compiled locally.

A failed build of an otherwise available optional backend is recorded and the remaining backends continue. `--strict` changes benchmark setup failures into a failed benchmark run, while absence of a compatible wheel remains an explicit platform exclusion rather than a build failure.

## System dependencies

SHTns requires FFTW headers and libraries. On Debian or Ubuntu, including AArch64 Linux systems:

```bash
sudo apt-get install build-essential gfortran libfftw3-dev
```

## Recorded build information

Build runs store package source and revision information together with available compiler-related environment variables. Environment provenance records the operating system, machine architecture, requested backend set, unavailable wheels, and build failures. Matrix timing records contain `build_mode`, which keeps binary-package and source-build measurements separate during plotting.

The recorded build metadata do not yet include compiler version strings, linked FFT/BLAS/LAPACK libraries, CPU instruction-set selection, or the complete compiler and linker command lines. These omissions should be considered when comparing source-build results across systems.
