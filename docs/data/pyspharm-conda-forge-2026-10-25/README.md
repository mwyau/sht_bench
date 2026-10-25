# Conda-forge pyspharm benchmark data (2026-10-25)

This directory contains current results from the conda-forge pyspharm 1.0.9
build `np2py313h854c110_1017` (Linux x86-64, Python 3.13.15, NumPy 2.5.4).
Package and environment hashes, exact exports, runtime precision, extension
path, and linked libraries are recorded in
[the environment provenance](../../../environments/pyspharm-conda-forge-linux-64.provenance.json).

- `legacy-matrix.json` retains all seven timing samples for each of 180 GL/CC
  legacy matrix operation/thread/grid cells; its CSV omits only the sample list.
- `source-q1-outcomes.csv` records all 1,824 pyspharm outcomes, including
  unsupported cases and reasons. `source-q1-timings.csv`,
  `source-q1-matched-ratios.csv`, and `source-q1-paired-summary.csv` contain
  the supported timing cells and exact DUCC pairings.
- `source-q1-accuracy.csv` records every numerical validation stage for the
  source q1 run. `source-q1-summary.json` includes cell counts, tolerances,
  validation maxima, and run/build manifest hashes.
- `source-resolution-summary.json` records the audited limited resolution
  smoke and its setup, first-call, accuracy, and matched timing summaries.

The raw source-run cells remain in local ignored benchmark output directories;
these tracked exports omit local raw-cell links so every path here resolves in
the repository.
