# sht_bench

CPU performance benchmarks for spherical harmonic transform (SHT) implementations: **DUCC**, **SHTns**, **RustSHT**, and **conda-forge pyspharm** (Spherepack). The optional pyspharm-cpp compatibility backend uses DUCC algorithms and is labeled *DUCC-derived*.

## Current experiment: GL and CC scaling

The target is a **multi-resolution scaling curve for every supported backend** on an identical grid and spectral truncation. Separate scalar (spin 0) and vector (spin 1), analysis and synthesis, and float32 and float64. Request **1, 2, 4, 8 and 16 threads** where supported; pyspharm is measured at q1 only because its public API does not expose native thread control.

| Grid | Standard resolution ladder |
| --- | --- |
| **GL** (Gaussian) | N = 32, 45, 64, 90, 128, 180, 320; `nlat=2N`, `nlon=4N`, `lmax=N` |
| **CC** (regular latitude-longitude) | L = 36, 45, 60, 90, 120, 180, 360; `nlat=2L+1`, `nlon=4L`, `lmax=L` |

The requested deliverable is **16 main figures** (GL/CC × float32/float64 × spin0/spin1 × analysis/synthesis), each showing compatible backends against resolution with q1/q2/q4/q8/q16 panels. Exact timing records and failed/unsupported statuses accompany the plots.

### Run the lightweight sweep

```bash
uv sync --extra plot
uv run --extra plot python scripts/run_fast_scaling.py \
  --build-manifest results/gl-cc-resolution-scaling-build/build-manifest.json \
  --output results/gl-cc-resolution-scaling \
  --backend ducc,shtns,rustsht,pyspharm \
  --grid GL,CC --threads 1,2,4,8,16 \
  --precision float32,float64 --min-time 0.005 --timeout 180
```

The fast runner performs one warmup, three calibrated timing samples, and one process round per workload. Native setup stays outside transform timing. It checks only output shape, dtype and finite values, so the results are exploratory performance measurements, not numerical accuracy claims. Use this smoke command for GL N32 and CC L36, and pass `--resume` to continue compatible partial results:

```bash
uv run --extra plot python scripts/run_fast_scaling.py \
  --build-manifest results/gl-cc-resolution-scaling-build/build-manifest.json \
  --output results/.gl-cc-scaling-smoke \
  --backend ducc,shtns,rustsht,pyspharm --grid GL,CC \
  --threads 1 --precision float32,float64 --min-time 0.005 --timeout 180 --smoke
```

Regenerate figures from saved timings with:

```bash
uv run --extra plot python scripts/run_fast_scaling.py \
  --output results/gl-cc-resolution-scaling --replot
```

### Scaling results

The final figures are in the [16-figure gallery](results/gl-cc-resolution-scaling/plots/README.md). The compact [timing CSV](results/gl-cc-resolution-scaling/timings.csv), raw-sample [JSON results](results/gl-cc-resolution-scaling/timings.json), and build manifest accompany them. See [build details](BUILD.md), [matrix specification](docs/MATRIX.md), and [results notes](docs/RESULTS.md).

Earlier SHT studies, their raw data and figures are in the [historical archive](archive/README.md). The independent [FFT benchmark](docs/FFT.md) and Torch experiments remain available without alteration.

**Precision:** DUCC and RustSHT have float32/float64 paths, SHTns Python bindings are float64/complex128, and the tested conda-forge pyspharm interface is float32/complex64. Do not make cross-precision speedup claims.
