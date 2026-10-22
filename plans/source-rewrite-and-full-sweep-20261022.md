# Source matrix implementation and execution

Branch: `refactor/source-matrix-rustsht`. Keep all work local.

The requested RustSHT ref is `main` following the user's update on 22 October 2026. DUCC uses `ducc0`; SHTns uses `master`. Resolve these refs to immutable SHAs before builds. Required backend environments are independent; Spherepack uses only the published pyspharm-syl 1.2.1 wheel.

Parent owns `sources.py`, `source_adapters.py`, `source_validation.py`, `source_runner.py`, `source_report.py`, build/acquisition scripts, CLI and documentation. The single `benchmark_validator` Luna Max worker owns execution evidence, new tests and narrow validation harness scripts. Agent runtime names require underscores.

Milestones:

1. Acquire clean Git sources and write/check immutable source locks; compile and verify imports and native binaries.
2. Add explicit workload/capability records and reusable scalar/spin adapters.
3. Validate independent normalized-Legendre synthesis points, coefficient recovery and map round trips. Verify phase, vector components and pole limits before timing.
4. Run smoke L16, L64, L127, then all required degree/grid/order/precision/thread/allocation combinations. Preserve unsupported and failed outcomes. Run independent process rounds with interleaved backend order.
5. Run secondary workloads and adjoint/derivative gates, locked reproduction, full unit and FFT regression tests. Generate reports and locally commit coherent milestones.

Execution evidence: `results/source-20261022/main/`. Provisional debugging runs are distinct from final timing sweeps. Do not merge results produced by different adapter or validation implementations. Raw records identify the harness hash and installed binary hashes through the build manifest.

Completion requires the original attachment's full workload ledger, actual numerical/benchmark evidence, locked reproduction, and reproducible plots. Successful source builds alone do not prove completion.
