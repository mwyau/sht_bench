# pyspharm conda-forge results — 2026-10-25

## Build and runtime

The active `pyspharm` backend uses conda-forge main, Linux x86-64,
`pyspharm` 1.0.9 build `np2py314h2766e9b_1017`. The separate conda environment
runs CPython 3.14.9 and NumPy 2.5.4 (NumPy ABI 33554432). The package artifact
is [`pyspharm-1.0.9-np2py314h2766e9b_1017.conda`](https://conda.anaconda.org/conda-forge/linux-64/pyspharm-1.0.9-np2py314h2766e9b_1017.conda),
SHA-256 `1edea5872986bd960a9d42f234960e36fdfd8cf6a110c37daaa53baf20b94b00`,
MD5 `eeff4953b1dc383de80803b5bfa9f2dc`.

The benchmark imports `spharm` from the conda prefix; its native module is
`_spherepack.cpython-314-x86_64-linux-gnu.so` (SHA-256
`94b3c6d2fa5b7083beef10be8e2bb3a9389f463cb2540909ffb2cd03861f5310`). Runtime
analysis returns `complex64` coefficients and synthesis returns `float32`
maps. The native extension links to `libm`, `libc`, and the system loader; it
has no direct FFT, BLAS, or LAPACK linkage. The imported module, extension,
package record, dependencies, and linked libraries are recorded in the
[environment provenance](../environments/pyspharm-conda-forge-linux-64.provenance.json).
The [explicit conda export](../environments/pyspharm-conda-forge-linux-64.explicit.txt)
and [reusable environment specification](../environments/pyspharm-conda-forge-linux-64.export.yml)
are tracked with the results.

New uv-managed environments default to Python 3.14.8 and NumPy 2.5.4. The
existing DUCC, SHTns, and RustSHT timing data still use Python 3.13.15; they
were not rerun for this focused migration and are not combined with these
pyspharm timings as matched ratios. Their source revisions are unchanged:
DUCC `31d9fe99fefa5ca29df17b7e6d2362b8c5c4a05f`, SHTns
`4e69ceb3fc5f19475d55b03ff0fdec210dc3075e`, and RustSHT
`75cf62ecfc4a324fc4f7d9e652c3d285dbf32cc4`.

## Source q1 rerun

The fresh three-round q1 run covered `lmax` 16, 64, and 127, one requested
thread, 30 warm repetitions, and a 0.02-second minimum timing window. It
accounted for all 1,824 requested outcomes: 126 timed successfully and passed
numerical validation; 1,698 were recorded as unsupported before timing. There
were no execution failures, numerical failures among timed cells, or pending
outcomes. All 4,086 validation-stage records passed the existing float32
threshold of `5e-5`.

| Maximum validation metric | Value | Limit |
| --- | ---: | ---: |
| Relative L2 error | `1.200734e-5` | `5e-5` |
| Maximum absolute error | `2.849102e-5` | `5e-5` |
| Scaled relative error | `5.603562e-6` | `5e-5` |

The unsupported outcomes preserve the actual capability restrictions:

| Reason | Outcomes |
| --- | ---: |
| Runtime provides float32/complex64, not float64/complex128 | 576 |
| No native F1 path | 576 |
| Public API supports full triangular truncation only | 288 |
| Public API has no output-buffer call | 144 |
| GL 32×33, L16 returns `vhsgsi ierror=3` | 96 |
| GL minimum spin-1 grid does not resolve highest-degree modes | 18 |

The [outcome ledger](data/pyspharm-conda-forge-2026-10-25/source-q1-outcomes.csv),
[timings](data/pyspharm-conda-forge-2026-10-25/source-q1-timings.csv),
[accuracy records](data/pyspharm-conda-forge-2026-10-25/source-q1-accuracy.csv),
[run configuration](data/pyspharm-conda-forge-2026-10-25/source-q1-run-config.json),
and [summary](data/pyspharm-conda-forge-2026-10-25/source-q1-summary.json)
are tracked. Each outcome's raw cell, including raw timing samples and
validation evidence, is in the compact
[q1 raw-cell archive](data/pyspharm-conda-forge-2026-10-25/source-q1-raw-cells.tar.gz).
The q1 build-manifest SHA-256 is
`bbd19df2ad7160655e7fb1f9b7c43ab1cf2d9e2b0e97ee935b067b5280a102fb`; the
harness SHA-256 is `b0ef7551bd462f6da1f3232e7b8eec12f31a8c91ecaa47669930b2555d486d0e`.

A separate non-timed check tested regular CC grids at L16 with odd and even
longitude lengths (33 and 34), scalar and spin-1 transforms, and batch sizes 1
and 4. All eight configurations passed 84 fixture checks at the same `5e-5`
tolerance. The maximum relative L2, absolute, and scaled errors were
`1.146489e-6`, `1.192093e-6`, and `6.669145e-7`. Batch shape and dtype evidence
is in the [focused CC/batch record](data/pyspharm-conda-forge-2026-10-25/focused-cc-batch-validation.json).
This is adapter validation; it does not establish separate public batch API
semantics or performance scaling.

## Legacy matrix rerun

The legacy command produced 180 timed records: 100 GL and 80 CC, each covering
analysis and synthesis at requested thread counts 1, 2, 4, 8, and 16. GL
bandwidths were 42, 63, 85, 159, 255, 319, 511, 639, 1023, and 1279; CC
bandwidths were 36, 60, 72, 90, 120, 180, 360, and 720. Every record retains
seven timing samples, three warmup calls, the 0.2-second minimum timing window,
and conda package provenance. No cell failed. Spherepack does not expose native
thread control; requested thread count remains part of the recorded workload
identity.

The [legacy matrix JSON](data/pyspharm-conda-forge-2026-10-25/legacy-matrix.json),
[CSV](data/pyspharm-conda-forge-2026-10-25/legacy-matrix.csv), and
[manifest](data/pyspharm-conda-forge-2026-10-25/legacy-matrix-manifest.json)
replace the prior published pyspharm rows. No old-version performance ratios
are reported.

## Numerical behavior and capability limits

The original GL minimum spin-1 issue persists when the capability filter is
bypassed: the L+1 grid loses high-degree spin-1 modes and fails the unchanged
validation gates. The active source run therefore records these 18 requests as
unsupported; spin-1 GL cases require at least `nlat = lmax + 2` for the tested
workloads.

The GL 32×33, L16 constructor still returns `vhsgsi ierror=3`; its 96 q1
variants remain explicitly unsupported. Direct CC L63 high-degree probes still
exceed the `5e-5` tolerance in coefficient recovery and map round trips, with
maximum relative L2 errors `9.380130e-5` for scalar and `9.247533e-5` for
spin-1. These diagnostics are recorded separately from timed q1 cells.

The CC L1023 package-specific exclusion has been removed. The new stored-plan
preflight estimates a 51,514,507,440-byte persistent lower bound and requires
103,029,014,880 bytes for the 2× headroom policy. The observed available memory
was 76,402,110,464 bytes, so the case was resource-limited before constructing
an adapter or native plan. The corrected dimensions are 2047×4092. The
[complete numerical probe record](data/pyspharm-conda-forge-2026-10-25/pyspharm-numerical-probes.json)
contains these calculations and the direct validation diagnostics.

## Regenerated figures

Every active pyspharm chart now comes from the conda-forge Python 3.14.9
results. Backend colors stay fixed; conda-forge is identified by line style in
the legacy figures. The source q1 charts show only pyspharm and publish no
cross-backend ratios. The figure index includes all q1 timing, setup, native
setup, memory, and error-versus-latency plots, the legacy GL/CC charts, and
setup/first-call summaries:

- [Complete figure index](figures/pyspharm-conda-forge-2026-10-25/README.md)
- [GL scalar synthesis](figures/pyspharm-conda-forge-2026-10-25/GL-scalar-synthesis.png)
- [CC spin-1 synthesis](figures/pyspharm-conda-forge-2026-10-25/CC-vector-synthesis.png)
- [CC spin-1 error versus latency](figures/pyspharm-conda-forge-2026-10-25/CC-vector-error-vs-latency.png)
- [Setup versus warm timing](figures/pyspharm-conda-forge-2026-10-25/q1-setup-vs-warm.png)
- [First-call versus warm timing](figures/pyspharm-conda-forge-2026-10-25/q1-first-call-vs-warm.png)

All displayed pyspharm times and errors are sourced from the tracked Python
3.14.9 CSV/JSON files. No other-backend figures were regenerated unnecessarily.
