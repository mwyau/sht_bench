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
historical DUCC, SHTns, and RustSHT matrix still uses Python 3.13.15. For a
direct comparison, a separate focused control reran all four backends with
Python 3.14.9 and NumPy 2.5.4; those measurements are reported below and kept
separate from the historical matrix. The source revisions are unchanged:
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

## Matched Python 3.14 source controls

The controls use the same build manifest for DUCC, SHTns, RustSHT, and
conda-forge pyspharm. All four runtime environments report CPython 3.14.9 and
NumPy 2.5.4. The pyspharm build is 1.0.9,
`np2py314h2766e9b_1017`, from conda-forge main. Source SHAs are DUCC
`31d9fe99fefa5ca29df17b7e6d2362b8c5c4a05f`, SHTns
`4e69ceb3fc5f19475d55b03ff0fdec210dc3075e`, and RustSHT
`75cf62ecfc4a324fc4f7d9e652c3d285dbf32cc4`.

Each control used one requested thread, three isolated process rounds, 30 warm
repetitions, and a 0.02-second minimum timing window. Setup and numerical
validation remain outside warm transform timing. Every timed cell passed its
configured precision gate; unsupported and numerical-failure cells are kept
in the outcome ledgers. Backend order alternated across rounds. The shared
build-manifest SHA-256 is
`bbd19df2ad7160655e7fb1f9b7c43ab1cf2d9e2b0e97ee935b067b5280a102fb`; the
selection-aware harness SHA-256 is
`cbdd6363ea4c8764e7055d598eea8fb2ceb5492060781d1bd4daa717c3fdae3d`.

The L64 control covered GL minimum and extended grids plus the minimum CC grid;
scalar and spin-1 analysis/synthesis; float32 and float64 where implemented; and
batch sizes 1 and 4 for allocating calls. It recorded all 576 requested
outcomes:

| Backend | Timed and validated | Unsupported | Numerical validation failure |
| --- | ---: | ---: | ---: |
| DUCC | 108 | 36 | 0 |
| pyspharm | 60 | 84 | 0 |
| RustSHT | 60 | 72 | 12 |
| SHTns | 24 | 108 | 12 |
| **Total** | **252** | **300** | **24** |

The 60 timed pyspharm cells all used float32 spatial data and complex64
coefficients. Their maximum relative L2 error was `1.200734e-5`, maximum
absolute error `2.849102e-5`, and maximum scaled error `5.603562e-6`, all within
the unchanged `5e-5` tolerance. Pyspharm's unsupported rows include float64,
batch analysis, and the minimum GL spin-1 cases that do not resolve the
highest-degree modes.

Only exact workload and precision matches enter pair ratios. Across 15 L64
workload groups with three process pairs each, pyspharm/DUCC median paired
ratios ranged from `0.210` to `0.918`. Across six groups, pyspharm/RustSHT
ratios ranged from `0.053` to `0.204`. No pyspharm/SHTns ratio is emitted:
SHTns uses float64/complex128, while pyspharm exposes float32/complex64. The
separate L64 float64 SHTns/DUCC comparisons remain available in the report.

A second control covered scalar float32/complex64 transforms on the regular CC
grid at L360 (`721×1440`). All 24 requested outcomes are accounted for:
DUCC and pyspharm each had six timed, validated cells; RustSHT's six cells
failed the numerical gate; and SHTns's six cells were unsupported at the
requested precision. Pyspharm's maximum relative L2 error was `3.990527e-6`,
maximum absolute error `1.049042e-5`, and maximum scaled error `1.351073e-6`.

| Operation | pyspharm median (ms) | DUCC median (ms) | Paired pyspharm/DUCC ratio |
| --- | ---: | ---: | ---: |
| Analysis | 96.094 | 14.278 | 6.651 |
| Synthesis | 57.891 | 6.736 | 8.595 |

The ratios are medians of three round-level paired ratios; the latency columns
are medians of the three cell medians, so dividing displayed latency values
does not necessarily reproduce the ratio exactly. These large-grid ratios
describe this single CC workload. They do not include RustSHT or SHTns because
neither produced a valid float32 comparison at this configuration.

The complete [L64 report and chart set](figures/pyspharm-conda-forge-2026-10-25/matched-q1-l64/README.md)
contains 147 exact matched cell pairs, outcome and accuracy tables, all timing
samples, the full raw cells, build provenance, and 100 figures for workload
groups containing pyspharm. The [L360 CC report](figures/pyspharm-conda-forge-2026-10-25/matched-cc-l360/README.md)
contains six matched pyspharm/DUCC pairs and 10 affected figures. Both audits
report complete coverage and no evidence issues.

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
are the current conda-forge pyspharm legacy matrix exports.

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
the legacy figures. The package-specific source q1 charts show only pyspharm;
the matched control reports add precision-matched DUCC and RustSHT comparisons
and retain SHTns in a separate float64 group. The figure index includes all
standalone q1, matched-control, and legacy GL/CC figures, plus setup summaries:

- [Complete figure index](figures/pyspharm-conda-forge-2026-10-25/README.md)
- [GL scalar synthesis](figures/pyspharm-conda-forge-2026-10-25/GL-scalar-synthesis.png)
- [CC spin-1 synthesis](figures/pyspharm-conda-forge-2026-10-25/CC-vector-synthesis.png)
- [CC spin-1 error versus latency](figures/pyspharm-conda-forge-2026-10-25/CC-vector-error-vs-latency.png)
- [Setup versus warm timing](figures/pyspharm-conda-forge-2026-10-25/q1-setup-vs-warm.png)
- [First-call versus warm timing](figures/pyspharm-conda-forge-2026-10-25/q1-first-call-vs-warm.png)

All displayed pyspharm times and errors are sourced from the tracked Python
3.14.9 CSV/JSON files. No other-backend figures were regenerated unnecessarily.
