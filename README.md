# sht_bench

CPU performance benchmarks for spherical harmonic transform (SHT) implementations: **DUCC**, **SHTns**, **RustSHT**, and conda-forge **pyspharm** (Spherepack). The optional pyspharm-cpp compatibility backend is derived from DUCC and is not an independent SHT implementation.

## Active results: dense GL/CC scaling with RustSHT dev

The current comparison replaces only the RustSHT rows in the original dense-to-L1024 dataset with a fresh run from RustSHT `dev`, commit `3194dc095c71343931b2ba0cfa0fc4f72dcab4cf`. DUCC, SHTns, and pyspharm measurements are preserved unchanged. The baseline RustSHT `main` measurements remain available for direct comparison.

The matrix covers scalar (spin 0) and vector (spin 1) analysis and synthesis, full triangular order, float64 and float32 where supported, and q1/q2/q4/q8/q16. GL uses `nlat=2L, nlon=4L`; regular CC uses `nlat=2L+1, nlon=4L`.

| Grid | Dense-to-1024 degrees |
| --- | --- |
| GL | 32, 45, 64, 90, 128, 160, 180, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024 |
| CC | 36, 45, 60, 90, 120, 160, 180, 192, 224, 256, 320, 360, 384, 448, 512, 640, 768, 896, 1024 |

The refreshed RustSHT run completed all 1,480 operation cells. The merged comparison contains 3,816 successful timings, 8 pyspharm GL synthesis finite-output failures, and no resource-limited cells. These are exploratory timing-only measurements with shape, dtype, and finite-output checks; no numerical-accuracy claim is made. The persistent SHTns q8/q16 plateaus remain flagged as suspect.

## Results

Float64 comparisons are presented first. At q1/L1024, RustSHT dev had the lowest scalar-analysis latency on both grids. It reduced float64 scalar-synthesis latency by 6.2–8.6× versus the old RustSHT main build at L320/L512/L1024, but DUCC remained faster at q1. RustSHT float32 did not improve overall: only 17 of 148 matched q1 cells were faster in the dev build.

### Float64 figures

- [GL spin 0 analysis](results/gl-cc-resolution-scaling-dev-refresh/plots/GL_float64_spin0_analysis.png)
- [GL spin 0 synthesis](results/gl-cc-resolution-scaling-dev-refresh/plots/GL_float64_spin0_synthesis.png)
- [GL spin 1 analysis](results/gl-cc-resolution-scaling-dev-refresh/plots/GL_float64_spin1_analysis.png)
- [GL spin 1 synthesis](results/gl-cc-resolution-scaling-dev-refresh/plots/GL_float64_spin1_synthesis.png)
- [CC spin 0 analysis](results/gl-cc-resolution-scaling-dev-refresh/plots/CC_float64_spin0_analysis.png)
- [CC spin 0 synthesis](results/gl-cc-resolution-scaling-dev-refresh/plots/CC_float64_spin0_synthesis.png)
- [CC spin 1 analysis](results/gl-cc-resolution-scaling-dev-refresh/plots/CC_float64_spin1_analysis.png)
- [CC spin 1 synthesis](results/gl-cc-resolution-scaling-dev-refresh/plots/CC_float64_spin1_synthesis.png)

### Float32 figures

- [GL spin 0 analysis](results/gl-cc-resolution-scaling-dev-refresh/plots/GL_float32_spin0_analysis.png)
- [GL spin 0 synthesis](results/gl-cc-resolution-scaling-dev-refresh/plots/GL_float32_spin0_synthesis.png)
- [GL spin 1 analysis](results/gl-cc-resolution-scaling-dev-refresh/plots/GL_float32_spin1_analysis.png)
- [GL spin 1 synthesis](results/gl-cc-resolution-scaling-dev-refresh/plots/GL_float32_spin1_synthesis.png)
- [CC spin 0 analysis](results/gl-cc-resolution-scaling-dev-refresh/plots/CC_float32_spin0_analysis.png)
- [CC spin 0 synthesis](results/gl-cc-resolution-scaling-dev-refresh/plots/CC_float32_spin0_synthesis.png)
- [CC spin 1 analysis](results/gl-cc-resolution-scaling-dev-refresh/plots/CC_float32_spin1_analysis.png)
- [CC spin 1 synthesis](results/gl-cc-resolution-scaling-dev-refresh/plots/CC_float32_spin1_synthesis.png)

The [ordered figure gallery](results/gl-cc-resolution-scaling-dev-refresh/plots/README.md), merged raw-sample [CSV](results/gl-cc-resolution-scaling-dev-refresh/timings.csv), [JSON results](results/gl-cc-resolution-scaling-dev-refresh/timings.json), [merge provenance](results/gl-cc-resolution-scaling-dev-refresh/merge-provenance.json), and build manifests are in the new result directory. The [RustSHT-only raw run](results/gl-cc-resolution-scaling-rustsht-dev-3194dc09/) records all 1,480 new cells. The untouched [baseline dataset](results/gl-cc-resolution-scaling-l1024/) retains the previous RustSHT main measurements and other backends.

See the [results report](docs/RESULTS.md) for matched latency ratios, thread scaling, exclusions, and caveats.

## Reproduce the refresh

Run RustSHT only, then key-merge its results with the saved baseline. This does not rerun DUCC, SHTns, or pyspharm:

```bash
env -u OMP_WAIT_POLICY uv run --extra plot python scripts/run_fast_scaling.py \
  --build-manifest results/gl-cc-resolution-scaling-rustsht-dev-build-3194dc09/build-manifest.json \
  --output results/gl-cc-resolution-scaling-rustsht-dev-3194dc09-reproduction \
  --profile dense-to-1024 --backend rustsht --grid GL,CC \
  --threads 1,2,4,8,16 --precision float64,float32 \
  --min-time 0.005 --timeout 180

uv run --extra plot python scripts/merge_scaling_results.py \
  --baseline results/gl-cc-resolution-scaling-l1024 \
  --rustsht-refresh results/gl-cc-resolution-scaling-rustsht-dev-3194dc09-reproduction \
  --output results/gl-cc-resolution-scaling-dev-refresh-reproduction
```

Regenerate the merged figures from saved timings without rerunning transforms:

```bash
uv run --extra plot python scripts/run_fast_scaling.py \
  --output results/gl-cc-resolution-scaling-dev-refresh-reproduction --replot
```

The runner uses one warmup, three calibrated timing samples, and one process round; native setup is excluded from warm-call timing. See [build details](BUILD.md), the [matrix specification](docs/MATRIX.md), and the [historical archive](archive/README.md).
