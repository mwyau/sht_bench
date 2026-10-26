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

## Fast exploratory measurement protocol

**One warmup, three timed samples, one process round** per supported cell, keeping reusable construction outside warm-call timing. Save all three raw samples, the median latency in milliseconds, setup latency separately, CPU/thread request, exact workload, binary provenance and outcome status. Exclude unsupported, execution-failed and resource-limited cases from lines; leave holes visible.

Basic shape/dtype/finite checks are enough to establish that a timing call ran, **not** that its numerical algorithm is accurate. Label the resulting performance study exploratory. Avoid lengthy independent reference/round-trip audits and expensive repeated process rounds unless a specific anomaly requires investigation.

**Runner status:** `scripts/run_source_resolution_sweep.py` currently performs heavy validation and audit work independent of `--repeat`. A lightweight runner path needs to be implemented before launching the requested low-overhead sweep.

## Run configuration

```bash
uv run --extra plot python scripts/run_source_resolution_sweep.py \
  --plan-only --backend ducc,shtns,rustsht,pyspharm \
  --grid CC,GL --threads 1,2,4,8,16 \
  --precision float32,float64 --mmax full --rounds 1 --repeat 3
```

For an existing build, supply `--build-manifest PATH`; for a new one, use `--build`. Save outputs under a new directory in `results/` and regenerate with `--replot`. Do not merge rows from differently configured runs without clear provenance.

## Figures and published results

Produce **16 scaling figures** for GL/CC × precision × spin × direction. Each has **five panels** for q1/q2/q4/q8/q16, with an x-axis of increasing resolution and a logarithmic median-latency y-axis; each backend gets one curve based only on actually measured matching cases. Use the existing fixed backend colors. The q1-only pyspharm data may be shown only in the q1 panel, or as a clearly labeled *serial reference*, not a fake thread-scaling curve.

Commit one tidy CSV/JSON dataset and a concise figure index; do not generate hundreds of optional setup/memory/validation images. The [current results status](RESULTS.md) and [historical archive](../archive/README.md) distinguish the planned sweep from earlier partial controls.
