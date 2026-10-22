# Completion evidence for the source refactor

The original attachment defines the scope. The user's later instruction selects RustSHT `main` and removes the opt branch. A complete run ledger is evidence for its own requested plan; it does not by itself prove the entire task is complete.

| Requirement | Evidence required | Current evidence or next action |
| --- | --- | --- |
| Local branch, no main modification or remote writes | Git history and local command record | `refactor/source-matrix-rustsht`, initial main `0d5462232c24ad926cc5a2e8bf653b131c93445f`, milestones `c38164e` and `3b77961`; no push/PR commands |
| Immutable intended sources | Lock plus clean origin/ref/SHA checks | `results/source-20261022/main/sources.lock.json`; RustSHT ref `main` |
| Genuine native Git builds and published Spherepack wheel | Build logs, import checks, binary hashes/linkage | `results/source-20261022/main/build-manifest.json`, `build-*.log`; preserve the smoke manifest digest |
| Effective build flags and native dependency settings | Generated compiler/linker rules, configure logs, Cargo inputs | Original metadata annex and locked build archives collected, including Pulp feature fingerprints |
| Independent scalar and spin conventions | Analytic harmonics, complete random reference maps, coefficient recovery and round trips | Source validation code, analytic probes, per-cell fixture metrics |
| Native full and quarter order | Exact `mmax` in workload records | Primary plan and per-cell outcomes; Spherepack quarter-order API exclusion explicit |
| Atmospheric GL and regular CC geometries | All original dimensions and established truncations present | Full plan running in session 9141; terminal exact-identity audit pending |
| Minimum high-bandwidth GL/CC/F1 | Minimum geometry outcomes, separate backend constraints | Smoke includes minimum plus identifiable extended/quadrature groups |
| All 21 degree candidates | Recorded supported/failed/unsupported/resource outcomes for each requested cell | All 21 requested in active full sweep; degree 2047 remains optional |
| Physical-core q1/q2/q4/q8/q16 | Affinity, physical core limit, L3 mapping and independent processes | q1 smoke complete; all five physical-core counts in active full sweep; q32 remains a distinct optional SMT experiment |
| Allocating and preallocated calls | Separate operations, output-buffer gate, raw timings | DUCC/RustSHT/SHTns adapters and per-cell records; explicit native API exclusions |
| Separate float64 and float32 groups | Dtypes, tolerances and separate reports | Native Rust32/Spherepack probes; preserve observed failures without relaxed tolerances |
| Native batches | Real batched API calls and full-batch numerical gate | DUCC native ring-synthesis probes passed GL/CC/F1, both precisions and spins; original Spherepack batch probes preserved |
| Adjoint transforms | Independently recomputed discrete identities | Secondary direct probes; full secondary outcome ledger pending |
| Gradient and DERIV1 | Normalization/sign fixtures and independent synthesis reference | All six native derivative gates passed per DUCC/RustSHT/SHTns; Rust constant-gradient projection regression passes; full ledger pending |
| Longitude controls | Independent parity/origin 2x2 cases on the same remaining dimensions | All 24 native parity/origin probes passed for each of DUCC and RustSHT; full ledger pending |
| Setup, first-call and memory measurements | Native setup versus reference checks separated, explicit unobservable state, peak RSS | Profiling split, native first-call boundaries and memory preflight applied; focused tests and four final native worker integrations pass |
| Timing robustness | Three interleaved process rounds, 30 raw batches or documented long-call policy, p10/median/p90/dispersion | Original smoke completed: 7,296 outcomes, 3,684 timed, 372 numerical failures, 3,228 unsupported, 12 execution failures; full sweep running |
| Locked representative reproduction | Rebuild using recorded SHAs, compare provenance, independently validate and time representative cases | Complete: 216/216 original/locked workers timed; all gates passed, worst relative L2 at most 3.46e-13 |
| Reports and figures | Provenance, capability/accuracy ledgers, scalar/spin/grid/precision/thread/setup/memory plots, matched ratios and ms differences | Audited smoke tables: 1,734 matched process pairs, 578 paired summaries; 896 figures rendered. Full-sweep report and pending gallery tests remain |
| Full unit suite and FFT runtime | Full pytest output plus relevant source/backend/FFT checks | Streaming/source/CLI suite: 47 passed; isolated DUCC FFT: 8 passed plus strict r2c/c2c float32/float64 CLI smoke; full compatible-source suite: 89 passed/11 skipped/13 existing external Torch API failures, detailed in compatible-torch-validation.json |
| No invalid timing comparison | Numerical/timing evidence audit before plotting | Original smoke audit passed: 7,296 expected/recorded, no missing/unexpected cells or evidence issues; audit adversarial tests pass. Streaming exports match every CSV row and all 135,876 accuracy entries; fixed-snapshot and compact collector tests pass |
| Final local commits and concise findings | Committed implementation/tests/docs and measured paired data | Implementation committed through f836333, status docs through 3b0fad4; gallery changes and full-sweep findings pending |

Do not treat native numerical failures or missing native APIs as adapter correctness without inspecting their evidence. Do not convert an execution failure to an unsupported capability until the native constraint is established.

## Active full execution

Session 9141 runs the combined primary and secondary plan under
`results/source-20261022/locked/full-secondary`: 1,790,880 requested outcomes,
three rounds, 30 timed samples and q1/q2/q4/q8/q16. The exact command, manifest
hash and live observations are recorded in
`results/source-20261022/locked/full-secondary-execution-status.json`.
The existing Luna validator monitors that same session. Freeze the timing harness
and locked environments until terminal; audit and render the full results
afterward. Implementation is committed through f836333. The smoke report has
896 rendered figures, with representative plots inspected.

Prioritize SHTns/DUCC versus RustSHT main in findings; retain Spherepack failures
without additional deep debugging. Full coverage and final findings remain pending.
