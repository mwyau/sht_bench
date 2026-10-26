# GL/CC resolution scaling results

The dense-to-1024 sweep recorded **3,816 successful warm-call timings**, 8 execution failures, and no resource-limited cells. These are exploratory timing-only results: every successful output passed shape, dtype and finite-value checks, but no numerical-accuracy claim was tested. The run used 18 GL degrees and 19 CC degrees, full triangular order, one warmup, three short calibrated samples, and one process round per supported backend/grid/precision/spin/thread group. Native plans and setup were excluded from the timed calls.

## Float64 scaling

The first eight gallery figures compare DUCC, RustSHT and SHTns. Each shows q1/q2/q4/q8/q16 panels; markers are measured points and line segments stop at missing or failed degrees.

### GL, float64

![GL float64 spin 0 analysis](../results/gl-cc-resolution-scaling-l1024/plots/GL_float64_spin0_analysis.png)

![GL float64 spin 0 synthesis](../results/gl-cc-resolution-scaling-l1024/plots/GL_float64_spin0_synthesis.png)

![GL float64 spin 1 analysis](../results/gl-cc-resolution-scaling-l1024/plots/GL_float64_spin1_analysis.png)

![GL float64 spin 1 synthesis](../results/gl-cc-resolution-scaling-l1024/plots/GL_float64_spin1_synthesis.png)

### CC, float64

![CC float64 spin 0 analysis](../results/gl-cc-resolution-scaling-l1024/plots/CC_float64_spin0_analysis.png)

![CC float64 spin 0 synthesis](../results/gl-cc-resolution-scaling-l1024/plots/CC_float64_spin0_synthesis.png)

![CC float64 spin 1 analysis](../results/gl-cc-resolution-scaling-l1024/plots/CC_float64_spin1_analysis.png)

![CC float64 spin 1 synthesis](../results/gl-cc-resolution-scaling-l1024/plots/CC_float64_spin1_synthesis.png)

At q1, spin-0 L1024 medians in milliseconds were:

| Grid | Direction | DUCC | RustSHT | SHTns |
| --- | --- | ---: | ---: | ---: |
| GL | Analysis | 93.3 | 82.5 | 104.1 |
| GL | Synthesis | 137.1 | 2,568.8 | 109.5 |
| CC | Analysis | 147.5 | 107.8 | 147.9 |
| CC | Synthesis | 191.6 | 2,545.5 | 255.2 |

All three float64 backends completed both grids through L1024. RustSHT had the lowest q1 scalar-analysis latency at the top degree, while its scalar-synthesis latency was much higher than DUCC and SHTns. The intermediate L160–L320 points are present in the curves, including the requested L180/L192/L224/L256 samples.

## Float32 scaling

The next eight figures compare DUCC and RustSHT at q1/q2/q4/q8/q16, with conda-forge pyspharm shown at q1 only.

### GL, float32

![GL float32 spin 0 analysis](../results/gl-cc-resolution-scaling-l1024/plots/GL_float32_spin0_analysis.png)

![GL float32 spin 0 synthesis](../results/gl-cc-resolution-scaling-l1024/plots/GL_float32_spin0_synthesis.png)

![GL float32 spin 1 analysis](../results/gl-cc-resolution-scaling-l1024/plots/GL_float32_spin1_analysis.png)

![GL float32 spin 1 synthesis](../results/gl-cc-resolution-scaling-l1024/plots/GL_float32_spin1_synthesis.png)

### CC, float32

![CC float32 spin 0 analysis](../results/gl-cc-resolution-scaling-l1024/plots/CC_float32_spin0_analysis.png)

![CC float32 spin 0 synthesis](../results/gl-cc-resolution-scaling-l1024/plots/CC_float32_spin0_synthesis.png)

![CC float32 spin 1 analysis](../results/gl-cc-resolution-scaling-l1024/plots/CC_float32_spin1_analysis.png)

![CC float32 spin 1 synthesis](../results/gl-cc-resolution-scaling-l1024/plots/CC_float32_spin1_synthesis.png)

For the matching q1, spin-0, L320 analysis cases, GL medians were 4.40 ms for DUCC, 131.00 ms for RustSHT and 268.80 ms for pyspharm. CC medians were 7.94 ms, 68.94 ms and 66.87 ms respectively. Pyspharm has no q2–q16 timing control.

## Thread scaling

For float64 spin-0 analysis at L1024, q1-to-q16 speedups were:

| Grid | DUCC | RustSHT | SHTns |
| --- | ---: | ---: | ---: |
| GL | 6.34× | 9.36× | 6.00× |
| CC | 7.15× | 5.33× | 1.38× |

Scaling depends on backend, grid, direction and spin. In particular, SHTns CC analysis gained little between q1 and q16 at the top degree. The figures show each thread count separately; no cross-precision speedups are inferred. Treat SHTns q8/q16 curves as suspect because many low and intermediate degrees contain intermittent ~12 ms plateaus, described below.

## Focused diagnostics

**RustSHT float32 scalar analysis.** In the new GL q1 run, L128 measured 8.52 ms in float32 versus 0.593 ms in float64; L320 measured 131.00 ms versus 4.779 ms. Source inspection of the built RustSHT `main` snapshot confirms that f64 enables ordered, paired and precomputed scalar paths and lowers the fused scalar-analysis threshold to L16. The f32 implementation inherits the generic false capability flags and the L255 threshold. This makes the scalar Legendre recurrence the plausible source of most of the gap; the FFT-versus-Legendre time split was not separately profiled, and no native code was changed.

**SHTns q8/q16.** The full run used `OMP_WAIT_POLICY` unset, `OMP_NUM_THREADS=q`, and affinity to the first q physical CPUs returned by the runner. The targeted full-run q16 GL spin-0 analysis medians were 0.0222 ms at L64 and 0.4936 ms at L320, but many other q8 and q16 cells show ~12 ms plateaus (for example GL q8 L192/L224 and q16 L192/L224). Standalone probes disagreed with those cells: q8 GL L320 analysis measured 0.611 ms under the default policy and 0.562 ms under PASSIVE, while the full sweep recorded 11.997 ms. PASSIVE also slowed q8 GL L64 analysis and did not improve q16 GL L320 consistently. No single policy fixed both sizes and thread counts, so the sweep kept the default and preserved the samples. Treat SHTns q8/q16 curves, especially small/intermediate degrees, as suspect.

## Coverage and exclusions

| Backend | Precision | GL maximum successful degree | CC maximum successful degree |
| --- | --- | --- | --- |
| DUCC | float64, float32 | 1024 | 1024 |
| RustSHT | float64, float32 | 1024 | 1024 |
| SHTns | float64 | 1024 | 1024 |
| conda-forge pyspharm | float32 | Analysis 640; synthesis 320 | Analysis and synthesis 640 |

There were 864 API-capability exclusions: SHTns float32 (740 requested cells) and pyspharm float64 through L640 (124 q1 cells). Another 48 q1 operation combinations above L640 were recorded as `degree_limit`; no pyspharm workload or native plan was constructed for GL/CC L768, L896 or L1024. Pyspharm GL float32 synthesis returned non-finite outputs at L384, L448, L512 and L640 for both spins, producing the 8 execution failures. Pyspharm CC analysis and synthesis remained finite through L640. No cell was resource-limited under the 2× stored-plan headroom policy; at L640 the Spherepack persistent lower bound was 11.76 GiB and the admission requirement was 23.53 GiB.

SHTns Python supports float64 only. Pyspharm is the conda-forge 1.0.9 build `np2py314h2766e9b_1017` on Python 3.14.9, uses float32/complex64, and has no public thread-count control. No pyspharm-cpp cells were run; that optional wrapper is DUCC-derived.

## Reproduction and source revisions

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

Regenerate the 16 PNGs from saved measurements without rerunning transforms:

```bash
uv run --extra plot python scripts/run_fast_scaling.py \
  --output results/gl-cc-resolution-scaling-l1024 --replot
```

The recorded source revisions are DUCC `31d9fe9`, SHTns `4e69ceb`, RustSHT `main` `579c09e`, and conda-forge pyspharm 1.0.9 build `np2py314h2766e9b_1017`. Exact build and binary metadata are in [build-manifest.json](../results/gl-cc-resolution-scaling-l1024/build-manifest.json); raw samples and worker settings are in [timings.json](../results/gl-cc-resolution-scaling-l1024/timings.json) and [timings.csv](../results/gl-cc-resolution-scaling-l1024/timings.csv).
