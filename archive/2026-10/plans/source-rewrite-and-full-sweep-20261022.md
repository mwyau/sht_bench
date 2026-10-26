# Source matrix implementation and execution

Branch: `refactor/source-matrix-rustsht`. Keep all work local.

The requested RustSHT ref is `main` following the user's update on 22 October 2026. DUCC uses `ducc0`; SHTns uses `master`. Resolve these refs to immutable SHAs before builds. Required backend environments are independent; Spherepack uses a separate conda-forge pyspharm environment with its resolved package build recorded.

Parent owns `sources.py`, `source_adapters.py`, `source_validation.py`, `source_runner.py`, `source_report.py`, build/acquisition scripts, CLI and documentation. The single `benchmark_validator` Luna Max worker owns execution evidence, new tests and narrow validation harness scripts. Agent runtime names require underscores.

Milestones:

1. Acquire clean Git sources and write/check immutable source locks; compile and verify imports and native binaries.
2. Add explicit workload/capability records and reusable scalar/spin adapters.
3. Validate independent normalized-Legendre synthesis points, coefficient recovery and map round trips. Verify phase, vector components and pole limits before timing.
4. Run smoke L16, L64, L127, then all required degree/grid/order/precision/thread/allocation combinations. Preserve unsupported and failed outcomes. Run independent process rounds with interleaved backend order.
5. Run secondary workloads and adjoint/derivative gates, locked reproduction, full unit and FFT regression tests. Generate reports and locally commit coherent milestones.

The original source execution ended in partial runs and was superseded for the dependency migration by the bounded conda-forge q1 rerun. The current tracked result summary is in `docs/PYSPHARM_CONDA_RESULTS_2026-10-25.md`. Provisional debugging runs remain distinct from final timing sweeps; do not merge results produced by different adapter or validation implementations. Current machine-readable exports retain the run, build, and harness identities without links to local raw-cell files.

This earlier exhaustive execution plan was superseded by the user's focused dependency-migration scope. Do not restart its full workload ledger. Completion for the active migration requires the conda-forge package, numerical evidence, refreshed affected timings and charts, relevant regression tests, and a local commit. Successful source builds alone do not prove completion.
