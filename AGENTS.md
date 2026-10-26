# Agent instructions

## Active goal

Deliver **real multi-resolution GL/CC scaling curves** for DUCC, SHTns, RustSHT (`main`) and conda-forge pyspharm. Optional pyspharm-cpp must be identified as **DUCC-derived**, not as an independent SHT implementation.

- Use the seven GL and seven CC resolutions in `src/sht_bench/source_resolution.py`.
- Cover scalar/spin-1 analysis and synthesis, float32/float64 where available, full triangular order, and **1, 2, 4, 8, 16 threads** where supported. Pyspharm has no native threading control: q1 only.
- Reuse existing working source environments; otherwise build locally. Pyspharm must be conda-forge Python 3.14, never the old PyPI `pyspharm-syl`.
- **Fast exploratory benchmark:** one warmup, three timing samples, one process round, short calibrated timing window. Keep native setup/planning separate from warm-call timing.
- Keep only basic shape, dtype and finite-output smoke checks plus known API/capability exclusions. Label timing-only measurements as exploratory; do not claim numerical accuracy without validation.
- Record raw samples, timings, backend/source identity, grid dimensions, degree/order, precision, spin, direction, thread count and failures in compact CSV/JSON. Generate **16 principal figures**: GL/CC × precision × spin × direction, with panels for q1/q2/q4/q8/q16 and backend scaling curves at actual measured resolutions.
- Preserve fixed backend colors. Do not mix precision groups, invent data, interpolate unsupported cases, or confuse DUCC-derived wrappers with independent algorithms.
- **Required code work:** `scripts/run_source_resolution_sweep.py` still performs expensive validation and audit passes. Add a genuinely lightweight mode or remove that overhead *before* starting the new scaling sweep; changing `--repeat` alone is insufficient.
- No giant combinatorial workloads, repeated full audits, hundreds of diagnostic plots, or unrelated refactors. Keep focused tests and frequent coherent commits.
- Current instructions are `README.md`, `BUILD.md` and `docs/MATRIX.md`. Historical reports live under `archive/` and are not current work orders.
- Check remote commits before merging local changes. Never force-push or overwrite work from another agent.
