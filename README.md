# sht_bench

`sht_bench` measures CPU performance of spherical harmonic transform (SHT) implementations exposed through Python. It times repeated scalar analysis and synthesis on atmospheric Gaussian grids and endpoint-including regular grids and records the numerical and execution metadata needed to interpret the measurements.

## Scope

The benchmark uses triangular truncation with `mmax = lmax` and includes four implementations:

| Backend | Python interface | Gauss–Legendre | Regular / Clenshaw–Curtis | Precision used |
| --- | --- | :---: | :---: | --- |
| DUCC | `ducc0.sht` | yes | yes | `float64` / `complex128` |
| SHTns | `shtns` | yes | yes | `float64` / `complex128` |
| SHTOOLS | native routines through `pyshtools` | compact GLQ only | no | `float64` real coefficient arrays |
| Spherepack | `pyspharm-syl` / `spharm` | yes | yes | `float32` / `complex64` |

The default GL matrix uses full atmospheric Gaussian grids. Native SHTOOLS GLQ routines require `nlon = 2*nlat - 1`, so SHTOOLS is not included in those common-grid GL cases; it remains available for custom compact GLQ cases. The released SHTOOLS routines used here also do not implement the endpoint-including regular grid used for the CC comparison. `pyspharm-syl` is benchmarked from its published wheel only.

## Grids

### Gauss–Legendre

The default GL matrix uses ten full atmospheric Gaussian grids. For a full regular Gaussian grid, `F` gives the number of Gaussian latitude circles between the equator and one pole, so `nlat = 2F` and `nlon = 4F`. Plots use `F` labels consistently for all ten cases; the corresponding spectral truncations are listed separately below. Gaussian latitudes are nonuniform, so the angular resolution is the nominal mean latitude spacing `90/F`, not a constant latitude increment.

| Plot label | Spectral truncation | `lmax` | Grid (`nlat × nlon`) | Nominal spacing |
| --- | --- | ---: | ---: | ---: |
| F32 | T42 | 42 | 64 × 128 | 2.8125° |
| F48 | T63 | 63 | 96 × 192 | 1.875° |
| F64 | T85 | 85 | 128 × 256 | 1.40625° |
| F80 | TL159 | 159 | 160 × 320 | 1.125° |
| F128 | TL255 | 255 | 256 × 512 | 0.703125° |
| F160 | TL319 | 319 | 320 × 640 | 0.5625° |
| F256 | TL511 | 511 | 512 × 1024 | 0.3515625° |
| F320 | TL639 | 639 | 640 × 1280 | 0.28125° |
| F512 | TL1023 | 1023 | 1024 × 2048 | 0.17578125° |
| F640 | TL1279 | 1279 | 1280 × 2560 | 0.140625° |

For a custom GL `lmax` that is not one of these default atmospheric cases, the compact quadrature grid is

```text
nlat = L + 1
nlon = 2*L + 1
```

This compact path is useful for backend-specific experiments but is not mixed into the default atmospheric GL comparison.

### Regular / Clenshaw–Curtis

The default CC matrix uses eight endpoint-including regular latitude–longitude resolutions. For maximum spherical-harmonic degree `L`,

```text
nlat = 2*L + 1
nlon = 4*L
delta_theta = delta_phi = 90 / L degrees
```

The grid contains both poles and omits the duplicate 360° longitude column.

| Resolution | `lmax` | Grid (`nlat × nlon`) |
| ---: | ---: | ---: |
| 2.5° | 36 | 73 × 144 |
| 1.5° | 60 | 121 × 240 |
| 1.25° | 72 | 145 × 288 |
| 1.0° | 90 | 181 × 360 |
| 0.75° | 120 | 241 × 480 |
| 0.5° | 180 | 361 × 720 |
| 0.25° | 360 | 721 × 1440 |
| 0.125° | 720 | 1441 × 2880 |

## Timed operations

For each backend, grid, spectral scale, and thread count, the benchmark measures:

- **analysis**: spatial grid to spherical-harmonic coefficients;
- **synthesis**: spherical-harmonic coefficients to spatial grid.

Backend construction, grid configuration, and reusable precomputation occur before timing. Each operation is warmed up, calibrated to a minimum timing interval, and then measured repeatedly with `time.perf_counter_ns`. Garbage collection is disabled during timed samples. JSON output retains every per-call timing sample together with minimum, median, mean, and standard deviation.

The measured quantity is Python-call transform latency. Output allocation is included when the backend interface does not expose an equivalent preallocated-output path.

## Installation

Python 3.13 supports the full package set used by the benchmark. SHTns requires FFTW development files for local compilation. On Debian or Ubuntu:

```bash
sudo apt-get install build-essential gfortran libfftw3-dev
uv sync --all-extras
```

A subset can be installed with selected extras, for example:

```bash
uv sync --extra ducc --extra pyshtools --extra plot
```

The `plot` extra installs Matplotlib and is required by `--plot`, `plot`, and `plot-all`.

Package versions and build inputs are listed in [BUILD.md](BUILD.md).

## Commands

### Run installed packages

`run` benchmarks packages in the active environment and accepts one thread count per invocation. Its default GL subset uses atmospheric cases from the table above.

```bash
uv run sht-bench run \
  --backend all \
  --grid gl \
  --lmax 42,85,159,319,639 \
  --threads 1 \
  --output results/single-thread.json
```

`--lmax` accepts comma-separated values and inclusive `start:stop[:step]` ranges.

### Run a matrix

`matrix` varies build mode, backend, grid, thread count, spectral scale, and operation. Each thread count runs in a fresh Python process so thread-related environment variables are set before numerical libraries are imported. When `--lmax` is omitted, GL and CC use the separate default sweeps listed above.

```bash
uv run --extra plot sht-bench matrix --plot
```

Binary-package and locally compiled builds can be compared with:

```bash
uv run --extra plot sht-bench matrix \
  --build both \
  --threads 1,2,4,8,16 \
  --plot
```

`pyspharm-syl` participates only in the wheel/installed portion of the matrix. SHTns has no wheel and is source-built in both isolated environments; plotting collapses those duplicate provenance labels to one dashed SHTns series.

See [docs/MATRIX.md](docs/MATRIX.md) for matrix dimensions, incremental reruns, and output files.

### Plot results

Plot selected result files:

```bash
uv run --extra plot sht-bench plot results/single-thread.json
```

Generate spectral-scale and thread-scaling figures from a matrix:

```bash
uv run --extra plot sht-bench plot-all \
  results/matrix/matrix.json \
  --output results/matrix/plots
```

GL spectral-scale figures use `F32`, `F48`, `F64`, `F80`, `F128`, `F160`, `F256`, `F320`, `F512`, and `F640`. CC spectral-scale figures use regular-grid spacing in degrees. Both axes retain logarithmic spacing while labeling every sampled case explicitly. `matrix --plot` runs the same plotting step after data collection.

## Thread control

DUCC receives `nthreads` explicitly. SHTns receives the requested thread count when the transform object is constructed. The pyshtools and pyspharm interfaces used here do not expose equivalent per-call thread controls; the benchmark sets OpenMP and common BLAS thread environment variables before importing those packages.

`--threads auto` selects powers of two up to the detected logical CPU count and also includes the exact logical CPU count when it is not a power of two. Controlled scaling measurements should specify thread counts explicitly and set CPU affinity separately.

## Result metadata

Each result record stores, where available:

- backend and package version;
- build mode;
- grid label and exact dimensions;
- operation and `lmax`;
- requested and reported thread counts;
- spatial and spectral dtypes;
- Python, operating system, machine, and CPU information;
- warm-up and calibration parameters;
- raw timing samples;
- minimum, median, mean, and standard deviation.

Build-specific metadata are described in [BUILD.md](BUILD.md).

## Interpretation

DUCC and SHTns are measured with double-precision data. Native SHTOOLS compact-GLQ runs also use double precision. The public `pyspharm-syl` interface uses `float32` spatial data and `complex64` coefficients, so comparisons with pyspharm are mixed-precision comparisons.

SHTns and pyspharm initialize reusable transform state before timing. The low-level SHTOOLS interface used here exposes a different setup model. The reported transform latency therefore excludes setup cost and persistent-memory cost.

SHTns uses `polar_opt=0`, which disables its optional polar approximation. Build configuration can also affect timing: DUCC source builds may use host-specific code generation, and SHTns performance depends on the FFTW installation against which it is compiled.

The benchmark records performance for the configured transform calls. Cross-backend numerical validation of coefficient normalization, phase conventions, and transform error is not part of the present timing protocol.
