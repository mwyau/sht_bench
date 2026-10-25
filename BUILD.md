# Build configuration

SHT timing depends on both the transform implementation and the binary used to execute it. `sht_bench` therefore separates uv-managed package/source builds from the conda-forge Spherepack build and records the package source associated with each result.

## Package versions

| Backend | Binary-package environment | Source-build environment |
| --- | --- | --- |
| DUCC | PyPI `ducc0==0.41.0` wheel when available for the platform | Git `mreineck/ducc`, ref `ducc0`, resolved SHA recorded |
| SHTns | Git `schaeffn/shtns`, ref `master`, resolved SHA recorded | same Git source policy |
| pyshtools / SHTOOLS | PyPI `pyshtools==4.14.1` wheel when available for the platform | PyPI `pyshtools==4.14.1` source distribution |
| pyspharm / Spherepack | conda-forge `pyspharm=1.0.9`, in its own conda environment | not a Git source build |

The binary-package environment requires an actual compatible wheel for backends designated as wheel builds. Before installation, the runner asks `uv` to resolve the pinned package with `--only-binary` in dry-run mode. If the current operating system, architecture, and Python version have no compatible wheel, that backend is recorded under `unavailable_wheels` and omitted from the wheel benchmark. It is not silently compiled from source.

SHTns has no PyPI wheel, so both uv benchmark environments compile a Git checkout.
Each invocation records the resolved SHA. Plotting shows one SHTns source series.

Spherepack is imported as `spharm` from a separate conda-forge environment. The
Linux x86-64 comparison environment uses Python 3.13.15, NumPy 2.5.4, and
`pyspharm 1.0.9` build `np2py313h854c110_1017`. The environment specification
is [environments/pyspharm-conda-forge-linux-64.yml](environments/pyspharm-conda-forge-linux-64.yml);
the resolved artifact URL, package hash, native extension hash, linked
libraries, and full conda exports are recorded with each build. The solver
resolves the latest compatible package build for other host platforms.
`pyspharm` remains float32/complex64 and serial. Conda provenance is recorded
as `conda-forge`, with a distinct plot line style; it is not classified as a
PyPI wheel or Git source build.
The current numerical, performance, and capability results are in the
[pyspharm conda-forge report](docs/PYSPHARM_CONDA_RESULTS_2026-10-25.md),
with compact timing and validation data under `docs/data/`.

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
installed  use active packages for uv-managed backends; create/select the conda-forge pyspharm environment
wheel      create the binary-package environment for uv-managed backends
source     create the source-build environment
both       run wheel and source environments, plus the separate pyspharm environment
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

The isolated environments are `.venv-bench-source`, `.venv-bench-wheel`, and
`.conda-bench-pyspharm`. The conda environment is never installed into the
active uv or base conda environment.

## Source compilers

Caller-supplied `CC`, `CXX`, and `FC` values are preserved. The source environment otherwise selects available GNU compiler defaults for packages compiled locally.

A failed build of an otherwise available optional backend is recorded and the remaining backends continue. `--strict` changes benchmark setup failures into a failed benchmark run, while absence of a compatible wheel remains an explicit platform exclusion rather than a build failure.

## System dependencies

SHTns requires FFTW headers and libraries. On Debian or Ubuntu, including AArch64 Linux systems:

```bash
sudo apt-get install build-essential gfortran libfftw3-dev
```

## Recorded build information

Build runs store package source and revision information together with available compiler-related environment variables. Environment provenance records the operating system, machine architecture, requested backend set, unavailable packages, and build failures. Matrix timing records contain `build_mode`; `conda-forge` is kept separate from PyPI wheels and Git source builds during plotting.

The historical matrix metadata omit compiler version strings, linked FFT/BLAS/LAPACK libraries, CPU instruction-set selection, and complete compiler/linker command lines. The Git source workflow described below records compiler versions, native linkage, CPU configuration, and available generated build rules separately.

## Direct torch-harmonics comparison environment

The focused `sht-bench compare` command is separate from the historical matrix
runner and does not add Torch or the private `spharmgrid` checkout to normal
project dependencies. Install the exact local torch-harmonics checkout under
test into the active `uv` environment, or expose it with `PYTHONPATH`, and pass
the checkout explicitly:

```bash
uv pip install -e /path/to/torch-harmonics
uv run sht-bench compare \
  --backend ducc,torch \
  --case cc-73x144-t70,cc-73x144-t71 \
  --dtype float32,float64 \
  --torch-device cpu \
  --torch-source /path/to/torch-harmonics \
  --spharmgrid-reference /path/to/spharmgrid \
  --output results/torch-ducc
```

The source checkout used for the current feature work is
`feature/high-bandwidth-equiangular-sht` at
`fa1af8ddc2d9462fb69d2f9ffaf49be6284e9e7b`. The runner records the actual
checkout path, branch, dirty state, and SHA at runtime and verifies that
`torch_harmonics.RealSHT` accepts `analysis="sampling-theorem"` before any
high-bandwidth cell runs. A source checkout is required; a released wheel is
not substituted. The read-only spharmgrid checkout is used only to preserve
verified grid, normalization, phase, dtype/device, and reusable-module
conventions. It is provenance, not a runtime dependency or benchmark path.

The focused comparison keeps `sht_bench`'s inclusive mathematical `lmax` and
`mmax`; the Torch adapters pass the exclusive values `L+1`. DUCC receives the
inclusive values directly. Both float32/complex64 and float64/complex128 are
allocated explicitly. Module/grid/planning construction, coefficient and
NumPy/Torch conversion, transfers, validation, and checksums are outside
steady-state transform timing. CPU records use synchronized synchronous
`perf_counter_ns` wall timing. CPU thread-count cells run in fresh worker
processes with `torch.set_num_threads(requested)` and one inter-op thread. CUDA
records use preloaded device tensors and synchronized CUDA-event compute timing;
they are not combined with CPU wall-clock metrics. Raw samples, warmup,
repeat, iteration, setup, device, dtype, thread, and backend provenance are
retained in JSON.

The focused comparison currently accepts the measured starting tolerances of
relative L2 `5e-5` and maximum absolute `5e-6` for float32 analysis, and
`1e-10` for both float64 analysis metrics. Synthesis is reported separately;
its source-revision map envelope is `5e-5` relative L2 and `1.5e-4` maximum
absolute for float32, and `1e-10` relative L2 and `2e-10` maximum absolute for
float64. These are comparison acceptance limits for the recorded checkout,
not general claims about another torch-harmonics revision.

## FFT Actions build

The manually triggered FFT workflow is a separate build experiment. On both
Ubuntu runners it installs `ducc0==0.41.0` from its source distribution with
the normal pybind11 CMake branch. `DUCC0_USE_NANOBIND` is left unset,
`DUCC0_OPTIMIZATION=portable` is used on both architectures, and the run
verifies `ducc0.__wrapper__ == "pybind11"` before benchmarking. This is
required because DUCC 0.41.0's nanobind binding aliases its long-double FFT
type to double.

The workflow uses Python 3.13, records the binding, optimization, compiler
environment, and runtime NumPy long-double characterization in each raw
record, and runs thread counts no greater than the runner's reported CPU
count. It is manual-only because the measurements are performance-sensitive.
Before upload, it stages only the generated `README.md` and PNG figures;
JSON, logs, environments, wheels, extensions, and other working files remain
outside the uploaded artifact.

## Git source matrix

The Git source workflow uses separate environments for DUCC, SHTns, and
RustSHT, plus a separate conda-forge environment for pyspharm. Its source refs
are DUCC `ducc0`, SHTns `master`, and RustSHT `main`. Each Git build records
the resolved 40-character SHA; the conda build records the resolved package
version and build string.
The `matrix` workflow also acquires DUCC/SHTns Git sources, while retaining
release-wheel and optional SHTOOLS comparisons. Use the following commands for
the four-implementation comparison with numerical gates:

```bash
uv sync
uv run python scripts/build_sources.py --output results/source-run
uv run sht-bench source-matrix \
  --build-manifest results/source-run/build-manifest.json \
  --output results/source-run/smoke --smoke \
  --degrees 16,64,127 --threads 1 --rounds 3 --repeat 30
uv run --extra plot python -m sht_bench.source_report \
  results/source-run/smoke --output results/source-run/smoke-report
```

`build_sources.py` defaults to Python 3.13 for uv-managed source packages.
Native packages are installed from Git checkout paths through their current
build systems, including RustSHT's root Maturin release build. The pyspharm
environment is created from conda-forge with Python 3.13.15 and NumPy 2.5.4.
The compiler environment preserves caller settings;
defaults select GCC/G++/gfortran, `DUCC0_OPTIMIZATION=native`, and
`RUSTFLAGS='-C target-cpu=native'`. SHTns's Git build configures its FFTW/OpenMP
support; build logs, native linkage and FFTW pkg-config information record the
observed configuration. Conda package metadata, explicit package URLs and
hashes, the `spharm` import path, the `_spherepack` extension, runtime
precision, NumPy ABI, and dynamic linkage are recorded for pyspharm. Missing
packages and build failures have explicit outcomes.

Per-backend `--ducc-ref`, `--shtns-ref`, and `--rustsht-ref` override the refs.
To rebuild recorded sources into a new output directory:

```bash
uv run python scripts/build_sources.py --mode locked \
  --lock results/source-run/sources.lock.json \
  --output results/source-reproduction
```

Locked acquisition checks origin, ref overrides, clean tracked sources and
resolved SHAs. It reproduces source revisions; compiler and dependency versions
must also be compared using the two build manifests. It does not by itself
promise bitwise binary reproducibility. Checkouts in `.sources/`, isolated
environments, logs, native binaries and generated measurements are untracked.
`build-manifest.json` records compiler versions, flags, source origin/ref/SHA,
Python and NumPy versions, installed distribution versions, wheel tags and
installer metadata where applicable, verified import locations, binary
SHA256 hashes, linked libraries, dependency freezes, conda package metadata,
and available CPU/FFTW configuration.

Native build logs use verbose package-build output. Generated DUCC CMake/Ninja
rules, SHTns configure logs and Makefile flags, and RustSHT Cargo lock/profile
inputs and available RustSHT and Pulp release fingerprints are archived under
`build-metadata/` with SHA256 hashes. These files describe effective build
settings in addition to the caller's compiler environment. Dependency feature
flags and host CPU capabilities are recorded separately.

## CC/GL source resolution sweep

The focused resolution sweep is a separate experiment from the historical
`source-matrix` CLI. It builds each Git implementation in its own `uv`
environment from the local sibling checkouts, snapshots the selected commit
without modifying those checkouts, and creates a separate conda-forge
environment for pyspharm. RustSHT defaults to its local `main` ref. The pyspharm-cpp source
uses its recorded `third_party/ducc` Git submodule; the installed backend is
reported as DUCC-derived because it uses DUCC algorithms behind the
PySpharm-compatible `spharm_cpp.Spharmt` API. Its extension is built with the
checkout's scikit-build-core, CMake, and nanobind configuration and the
multiarch CMake profile. The build manifest records both the standalone DUCC
SHA and the nested DUCC submodule SHA.

The versioned profile contains seven endpoint-inclusive CC grids from
73×144/L36 through 721×1440/L360, and seven GL grids with `nlat=2*N`,
`nlon=4*N`, and `N=32,45,64,90,128,180,320`. Exact north-to-south latitude
and colatitude coordinate arrays are saved with the plan. Full and half order
means `mmax=L` and `mmax=L//2`, with unchanged `lmax=L`. pyspharm-cpp and
pyspharm expose only triangular truncation, so their half-order cells are
recorded unsupported; no coefficients are zeroed to simulate reduced order.
The conda-forge pyspharm matrix is float32/complex64 and serial. SHTns is
float64 only. GL spin-1 minimum-latitude cells require one additional latitude
row for the highest-degree modes to remain numerically observable. The
GL32×33/L16 Spherepack constructor failure is retained as a specific
unsupported workload. CC L1023 is resource-limited on this host: stored plan
arrays alone require 51,514,507,440 bytes, which does not leave the required
2x memory headroom.

Run the plan before building. The plan prints the exact API-supported and
unsupported outcome counts, and `--smoke` restricts it to CC 2.5° and GL N32:

```bash
uv run --extra plot python scripts/run_source_resolution_sweep.py --plan-only
uv run --extra plot python scripts/run_source_resolution_sweep.py \
  --output results/source-resolution-20261025 --build
uv run --extra plot python scripts/run_source_resolution_sweep.py \
  --output results/source-resolution-20261025 --resume
uv run --extra plot python scripts/run_source_resolution_sweep.py \
  --output results/source-resolution-20261025 --replot
```

The build command accepts `--ducc-source`, `--shtns-source`,
`--rustsht-source`, and `--pyspharm-cpp-source` path overrides, plus ref
overrides. Run outputs are never replaced. Resume checks the build-manifest,
profile, runner, harness, and workload identities. Every supported numerical
gate runs in a fresh process before its timing cells; thread-count cells also
run in fresh processes with affinity and backend thread controls set before
pool initialization. A gate is shared only by cells with the same backend
revision, geometry, `lmax`, `mmax`, precision, spin, and thread count. Raw
timing samples remain in per-cell JSON files under the ignored result
directory. Audited CSV summaries and plots can be recreated with `--replot`.

Backend timing comparisons require identical grid dimensions, `lmax`, `mmax`,
precision, spin, operation, and requested thread count. CC and GL results are
shown in separate facets; GL N320 is not presented as an identical workload
to CC 0.25°.
