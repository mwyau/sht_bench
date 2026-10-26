# GL/CC scaling benchmark

## Standard workload

| Dimension | Values |
| --- | --- |
| GL | N = 32, 45, 64, 90, 128, 180, 320; `nlat=2N, nlon=4N, lmax=N` |
| CC | L = 36, 45, 60, 90, 120, 180, 360; `nlat=2L+1, nlon=4L, lmax=L` |
| Backend | DUCC, SHTns, RustSHT, conda-forge pyspharm; optional DUCC-derived pyspharm-cpp |
| Spin | 0 (scalar), 1 (vector) |
| Direction | Analysis, synthesis |
| Precision | float32 or float64 as actually supported |
| Requested threads | **1, 2, 4, 8, 16** where supported |
| Truncation | Full triangular, `mmax=lmax` (half-order optional later) |

Pyspharm has no public native thread control and is timed at **q1 only**. SHTns Python exposes float64 only. Do not replace one grid, backend, precision or native batch operation with another and call it a match.

## Dense high-resolution profile

The current comparison uses `--profile dense-to-1024`; `--profile standard` retains the seven-point ladder above. The comparison refreshes RustSHT from `dev` and reuses the saved DUCC, SHTns, and pyspharm measurements.

| Grid | Degrees |
| --- | --- |
| GL | 32, 45, 64, 90, 128, 160, 180, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024 |
| CC | 36, 45, 60, 90, 120, 160, 180, 192, 224, 256, 320, 360, 384, 448, 512, 640, 768, 896, 1024 |

The geometry remains `GL nlat=2L,nlon=4L` and `CC nlat=2L+1,nlon=4L`, with `mmax=lmax=L`. Conda-forge pyspharm is hard-capped at L640 before planning a workload; its stored-plan preflight requires two times the persistent lower bound. Pyspharm remains float32 and q1 only.

## Fast exploratory measurement protocol

**One warmup, three timed samples, one process round** per supported cell, keeping reusable construction outside warm-call timing. Save all three raw samples, the median latency in milliseconds, setup latency separately, CPU/thread request, exact workload, binary provenance and outcome status. Exclude unsupported, execution-failed and resource-limited cases from lines; leave holes visible.

Basic shape/dtype/finite checks are enough to establish that a timing call ran, **not** that its numerical algorithm is accurate. Label the resulting performance study exploratory. Avoid lengthy independent reference/round-trip audits and expensive repeated process rounds unless a specific anomaly requires investigation.

The primary timing runner is `scripts/run_fast_scaling.py`. It bypasses independent references, coefficient recovery and audit passes while retaining basic output shape, dtype and finite-value checks, build identity, resource admission, per-worker timeouts, incremental results and resumption. The comprehensive source runner remains separate for numerical-validation studies.

## Run configuration

```bash
env -u OMP_WAIT_POLICY uv run --extra plot python scripts/run_fast_scaling.py \
  --build-manifest results/gl-cc-resolution-scaling-rustsht-dev-build-3194dc09/build-manifest.json \
  --output results/gl-cc-resolution-scaling-rustsht-dev-3194dc09-reproduction \
  --profile dense-to-1024 --backend rustsht \
  --grid GL,CC --threads 1,2,4,8,16 \
  --precision float64,float32 --min-time 0.005 --timeout 180

uv run --extra plot python scripts/merge_scaling_results.py \
  --baseline results/gl-cc-resolution-scaling-l1024 \
  --rustsht-refresh results/gl-cc-resolution-scaling-rustsht-dev-3194dc09-reproduction \
  --output results/gl-cc-resolution-scaling-dev-refresh-reproduction
```

Run the small N32/L36 smoke first by adding `--smoke` and selecting a temporary output directory. The runner copies the exact build manifest into its output, writes partial data after each backend/resolution/thread workload, and supports compatible continuation with `--resume`. It runs q1 across the ladder first, then q8/q16, then q2/q4. Regenerate only the 16 primary figures using `--replot`.

## Figures and published results

Produce **16 scaling figures** for GL/CC × precision × spin × direction, with float64 first in the report and gallery. Each has **five panels** for q1/q2/q4/q8/q16, with an x-axis of increasing resolution and a logarithmic median-latency y-axis; each backend gets one curve based only on actually measured matching cases. Stop line segments at missing or failed degrees. Use the existing fixed backend colors. The q1-only pyspharm data may be shown only in the q1 panel, or as a clearly labeled *serial reference*, not a fake thread-scaling curve.

The merged comparison has one CSV, one JSON dataset with raw samples and per-backend build provenance, a status list, and exactly 16 primary PNGs under `results/gl-cc-resolution-scaling-dev-refresh/plots/`. The old main-build data remains under `results/gl-cc-resolution-scaling-l1024/`. The gallery is `plots/README.md`. See [current results](RESULTS.md) and the [historical archive](../archive/README.md).
