# sht_bench

CPU performance benchmarks for spherical harmonic transform (SHT) implementations: **DUCC**, **SHTns**, **RustSHT**, and conda-forge **pyspharm** (Spherepack). The optional pyspharm-cpp compatibility backend is derived from DUCC and is not an independent SHT implementation.

## Active benchmark: dense GL/CC scaling

The current result compares scalar (spin 0) and vector (spin 1) analysis and synthesis on full triangular order, at float64 and float32 where supported, with q1/q2/q4/q8/q16. GL uses `nlat=2L, nlon=4L`; regular CC uses `nlat=2L+1, nlon=4L`.

| Grid | Dense-to-1024 degrees |
| --- | --- |
| GL | 32, 45, 64, 90, 128, 160, 180, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024 |
| CC | 36, 45, 60, 90, 120, 160, 180, 192, 224, 256, 320, 360, 384, 448, 512, 640, 768, 896, 1024 |

Conda-forge pyspharm is float32 and q1 only, and the runner caps it at L640 before constructing any workload or native plan. The earlier seven-point sweep is preserved in `results/gl-cc-resolution-scaling/`; use `--profile standard` to reproduce that ladder.

## Results

The primary dataset contains **3,816 successful timings** and 8 pyspharm GL synthesis finite-output failures. There were no resource-limited cells. All measurements are exploratory timing-only results; output shape, dtype and finiteness were checked, with no numerical-accuracy claim. SHTns q8/q16 curves include intermittent ~12 ms plateaus and are flagged as suspect in the report.

### Float64 figures

- [GL spin 0 analysis](results/gl-cc-resolution-scaling-l1024/plots/GL_float64_spin0_analysis.png)
- [GL spin 0 synthesis](results/gl-cc-resolution-scaling-l1024/plots/GL_float64_spin0_synthesis.png)
- [GL spin 1 analysis](results/gl-cc-resolution-scaling-l1024/plots/GL_float64_spin1_analysis.png)
- [GL spin 1 synthesis](results/gl-cc-resolution-scaling-l1024/plots/GL_float64_spin1_synthesis.png)
- [CC spin 0 analysis](results/gl-cc-resolution-scaling-l1024/plots/CC_float64_spin0_analysis.png)
- [CC spin 0 synthesis](results/gl-cc-resolution-scaling-l1024/plots/CC_float64_spin0_synthesis.png)
- [CC spin 1 analysis](results/gl-cc-resolution-scaling-l1024/plots/CC_float64_spin1_analysis.png)
- [CC spin 1 synthesis](results/gl-cc-resolution-scaling-l1024/plots/CC_float64_spin1_synthesis.png)

### Float32 figures

- [GL spin 0 analysis](results/gl-cc-resolution-scaling-l1024/plots/GL_float32_spin0_analysis.png)
- [GL spin 0 synthesis](results/gl-cc-resolution-scaling-l1024/plots/GL_float32_spin0_synthesis.png)
- [GL spin 1 analysis](results/gl-cc-resolution-scaling-l1024/plots/GL_float32_spin1_analysis.png)
- [GL spin 1 synthesis](results/gl-cc-resolution-scaling-l1024/plots/GL_float32_spin1_synthesis.png)
- [CC spin 0 analysis](results/gl-cc-resolution-scaling-l1024/plots/CC_float32_spin0_analysis.png)
- [CC spin 0 synthesis](results/gl-cc-resolution-scaling-l1024/plots/CC_float32_spin0_synthesis.png)
- [CC spin 1 analysis](results/gl-cc-resolution-scaling-l1024/plots/CC_float32_spin1_analysis.png)
- [CC spin 1 synthesis](results/gl-cc-resolution-scaling-l1024/plots/CC_float32_spin1_synthesis.png)

The [ordered figure gallery](results/gl-cc-resolution-scaling-l1024/plots/README.md), raw-sample [CSV](results/gl-cc-resolution-scaling-l1024/timings.csv), [JSON results](results/gl-cc-resolution-scaling-l1024/timings.json), [status record](results/gl-cc-resolution-scaling-l1024/statuses.json), resolution profile and build manifest are in the result directory. See the concise [results report](docs/RESULTS.md).

## Reproduce

Run the tested GL32/CC36 smoke first, then the complete dense ladder:

```bash
uv run --extra plot python scripts/run_fast_scaling.py \
  --build-manifest results/gl-cc-resolution-scaling-build/build-manifest.json \
  --output /tmp/gl-cc-scaling-smoke --profile dense-to-1024 --smoke \
  --backend ducc,shtns,rustsht,pyspharm --grid GL,CC \
  --threads 1 --precision float64,float32 --min-time 0.005 --timeout 180

env -u OMP_WAIT_POLICY uv run --extra plot python scripts/run_fast_scaling.py \
  --build-manifest results/gl-cc-resolution-scaling-build/build-manifest.json \
  --output results/gl-cc-resolution-scaling-l1024-reproduction --profile dense-to-1024 \
  --backend ducc,shtns,rustsht,pyspharm --grid GL,CC \
  --threads 1,2,4,8,16 --precision float64,float32 --min-time 0.005 --timeout 180
```

The runner does one warmup and three short calibrated samples in one process round, excludes native setup from warm-call timings, and saves progress for compatible resumption. Recreate the figures without rerunning transforms with:

```bash
uv run --extra plot python scripts/run_fast_scaling.py \
  --output results/gl-cc-resolution-scaling-l1024 --replot
```

See [build details](BUILD.md), [matrix specification](docs/MATRIX.md), and the [historical archive](archive/README.md). The independent [FFT benchmark](docs/FFT.md) and Torch experiments remain available.
