# GL/CC scaling results

The exploratory multi-resolution sweep is complete. The [gallery](../results/gl-cc-resolution-scaling/plots/README.md) contains exactly 16 figures. [timings.csv](../results/gl-cc-resolution-scaling/timings.csv) has one row per measured operation cell; [timings.json](../results/gl-cc-resolution-scaling/timings.json) retains the three raw samples, setup timing, workload identity and build metadata. The compact [status list](../results/gl-cc-resolution-scaling/statuses.json), [resolution profile](../results/gl-cc-resolution-scaling/resolution-profile.json), and exact [build manifest](../results/gl-cc-resolution-scaling/build-manifest.json) accompany the figures.

## Coverage

The sweep recorded **1,456 successful cells** across 728 isolated backend/resolution/precision/spin/thread processes, with analysis and synthesis sharing each initialized plan. Every cell has one warmup, three calibrated samples, one process round and full triangular truncation (`mmax=lmax`). There were no execution failures, timeouts or resource-limited cells.

| Backend | Completed workload coverage |
| --- | --- |
| DUCC | All seven GL and seven CC resolutions, float32/float64, spin 0/1, both directions, q1/q2/q4/q8/q16 (560 cells) |
| SHTns | All 14 resolutions, float64, spin 0/1, both directions, q1/q2/q4/q8/q16 (280 cells) |
| RustSHT | All 14 resolutions, float32/float64, spin 0/1, both directions, q1/q2/q4/q8/q16 (560 cells) |
| conda-forge pyspharm | All 14 resolutions, float32, spin 0/1, both directions, q1 only (56 cells) |

SHTns float32 is unsupported by its Python API. Pyspharm float64 is unsupported by the tested Spherepack runtime, and its public API has no native thread control, so q2–q16 are skipped. No pyspharm-cpp cells were run; that optional wrapper is DUCC-derived rather than an independent implementation.

The recorded source identities are DUCC `31d9fe9`, SHTns `4e69ceb`, RustSHT remote `main` `579c09e`, and conda-forge pyspharm 1.0.9 build `np2py314h2766e9b_1017` on Python 3.14.9 / NumPy 2.5.4. The build manifest also retains binary hashes and build details.

## Observed timing patterns

These are exploratory timing-only observations from three short samples. No numerical accuracy claim was tested.

- At GL N320, float64 scalar analysis, q1 medians were 4.51 ms for DUCC, 3.54 ms for SHTns, and 4.86 ms for RustSHT. At q16, DUCC and RustSHT fell to 0.64 ms and 0.81 ms, while SHTns rose to 11.8 ms.
- At the same GL N320 scalar analysis workload in float32, q1 medians were 4.43 ms for DUCC, 136.84 ms for RustSHT, and 304.79 ms for pyspharm. These values compare only the matching float32 workload; pyspharm has q1 data only.
- For CC spin-1 float32 synthesis at q1, the L36-to-L360 latency increase was about 166× for DUCC, 657× for RustSHT, and 962× for pyspharm. The scaling curves show the complete seven-point ladder for each backend.

Thread scaling varies with backend and workload. For example, at GL N320 float64 scalar synthesis, DUCC improved about 8× and RustSHT about 11× from q1 to q16; SHTns was about 3.5× slower at q16. Treat the values as exploratory: the runner checked only successful execution, output shape, dtype and finite values, not transform accuracy.

## Reproduction

```bash
uv run --extra plot python scripts/run_fast_scaling.py \
  --build-manifest results/gl-cc-resolution-scaling/build-manifest.json \
  --output results/gl-cc-resolution-scaling-reproduction \
  --backend ducc,shtns,rustsht,pyspharm \
  --grid GL,CC --threads 1,2,4,8,16 \
  --precision float32,float64 --min-time 0.005 --timeout 180
```

Regenerate the final figures without running transforms:

```bash
uv run --extra plot python scripts/run_fast_scaling.py \
  --output results/gl-cc-resolution-scaling --replot
```
