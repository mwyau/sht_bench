# sht_bench

CPU performance benchmarks for spherical harmonic transform (SHT) implementations: **DUCC**, **SHTns**, **RustSHT**, and **conda-forge pyspharm** (Spherepack). The optional pyspharm-cpp compatibility backend uses DUCC algorithms and is labeled *DUCC-derived*.

## Current experiment: GL and CC scaling

The target is a **multi-resolution scaling curve for every supported backend** on an identical grid and spectral truncation. Separate scalar (spin 0) and vector (spin 1), analysis and synthesis, and float32 and float64. Request **1, 2, 4, 8 and 16 threads** where supported; pyspharm is measured at q1 only because its public API does not expose native thread control.

| Grid | Standard resolution ladder |
| --- | --- |
| **GL** (Gaussian) | N = 32, 45, 64, 90, 128, 180, 320; `nlat=2N`, `nlon=4N`, `lmax=N` |
| **CC** (regular latitude-longitude) | L = 36, 45, 60, 90, 120, 180, 360; `nlat=2L+1`, `nlon=4L`, `lmax=L` |

The requested deliverable is **16 main figures** (GL/CC × float32/float64 × spin0/spin1 × analysis/synthesis), each showing compatible backends against resolution with q1/q2/q4/q8/q16 panels. Exact timing records and failed/unsupported statuses accompany the plots.

### Preparation and existing command

```bash
uv sync --extra plot
uv run --extra plot python scripts/run_source_resolution_sweep.py \
  --plan-only --grid CC,GL \
  --backend ducc,shtns,rustsht,pyspharm \
  --threads 1,2,4,8,16 --precision float32,float64 \
  --mmax full --rounds 1 --repeat 3
```

The source runner also accepts `--build` to build isolated local backends, `--build-manifest PATH` to reuse a build, `--output results/resolution-scaling` to save results, `--min-time 0.005` to shorten calibration, and `--replot` to regenerate saved figures. See [build details](BUILD.md) and [matrix specification](docs/MATRIX.md).

**Implementation status:** The runner currently includes substantial numerical validation and auditing even with `--rounds 1 --repeat 3`; a lightweight execution path is still needed. The new four-backend, multi-degree scaling sweep **has not yet been published**. See [results status](docs/RESULTS.md). Do not interpret earlier L64-only controls or pyspharm-only degree sweeps as this deliverable.

Earlier SHT studies, their raw data and figures are in the [historical archive](archive/README.md). The independent [FFT benchmark](docs/FFT.md) and Torch experiments remain available without alteration.

**Precision:** DUCC and RustSHT have float32/float64 paths, SHTns Python bindings are float64/complex128, and the tested conda-forge pyspharm interface is float32/complex64. Do not make cross-precision speedup claims.
