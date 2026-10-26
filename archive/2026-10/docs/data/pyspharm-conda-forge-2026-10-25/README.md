# Conda-forge pyspharm results data — 2026-10-25

These tracked records are from `pyspharm` 1.0.9 build
`np2py314h2766e9b_1017`, Linux x86-64, CPython 3.14.9, and NumPy 2.5.4.
Package and environment hashes, exact conda exports, runtime dtypes, import
paths, extension hash, and linked libraries are recorded in the
[environment provenance](../../../../../environments/pyspharm-conda-forge-linux-64.provenance.json).

## Source q1

- `source-q1-outcomes.csv` records all 1,824 requested outcomes, including unsupported reasons. The `raw_cell` column identifies entries in `source-q1-raw-cells.tar.gz`.
- `source-q1-raw-cells.tar.gz` contains all 1,824 raw JSON cells, including timing samples and numerical evidence.
- `source-q1-timings.csv` contains the 126 validated timed outcomes and setup, first-call, and warm timing fields.
- `source-q1-accuracy.csv` and `source-q1-accuracy.json` contain all 4,086 validation-stage records.
- `source-q1-run-config.json` and `source-q1-summary.json` retain the run dimensions, build and harness hashes, coverage audit, validation maxima, and unsupported-reason counts.
- `source-q1-matched-ratios.csv` and `source-q1-paired-summary.csv` contain headers only because this package-specific run contains pyspharm alone.

## Same-environment matched controls

All four backends were also measured with Python 3.14.9 and NumPy 2.5.4.
Complete, self-contained reports include run configuration, exact workload
plans, audits, build provenance, raw cells, accuracy records, timing samples,
matched ratios, and regenerated charts:

- [L64 GL/CC q1 control](../../figures/pyspharm-conda-forge-2026-10-25/matched-q1-l64/README.md): 576 outcomes, 252 timed cells, 147 exact matched cell pairs, and 100 figures in pyspharm workload groups.
- [L360 regular CC control](../../figures/pyspharm-conda-forge-2026-10-25/matched-cc-l360/README.md): 24 outcomes, 12 timed cells, six pyspharm/DUCC pairs, and 10 figures.

## Legacy matrix

- `legacy-matrix.json` retains all seven timing samples for each of 180 GL/CC operation, bandwidth, and requested-thread cells.
- `legacy-matrix.csv`, `legacy-matrix-summary.json`, and `legacy-matrix-summary.json` provide the tabular export, run configuration, and compact counts/range.

## Numerical diagnostics

- `pyspharm-numerical-probes.json` records the direct GL/CC capability probes and CC L1023 memory preflight.
- `focused-cc-batch-validation.json` records the eight non-timed odd/even regular-grid, scalar/spin-1, batch-1/batch-4 validation cases.

All paths are repository-relative. Extensive output logs and work files are kept outside the tracked summary; the raw source q1 cell archive is tracked for timing and validation reproducibility.
