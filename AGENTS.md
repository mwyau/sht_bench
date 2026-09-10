# AGENTS.md

## Repository rules

- Preserve the benchmark methodology when modifying code: grid definitions, normalization, precision, timing boundaries, thread handling, and build provenance must remain explicit and reproducible.
- Benchmark the named implementation and requested build provenance. Do not substitute another library, delegated implementation, or source build for a missing backend, grid path, or binary wheel; unsupported or unavailable combinations must remain explicit.
- Keep comparisons identifiable. Results must retain backend, grid, precision, build mode/source, `lmax`, thread count, and operation when those dimensions differ.
- Do not infer scientific validation or performance superiority from passing tests or successful timing runs. Accuracy and speed claims require the corresponding comparison data.
- Preserve timing boundaries. Keep backend construction, grid setup, planning/precomputation, and numerical validation outside transform timing unless a benchmark explicitly measures them. Retain raw timing samples, and keep thread-count runs process-isolated.
- Keep backend colors fixed across figures. Use line style or another independent visual encoding for build mode rather than changing backend color.
- Changes to pinned package versions, source revisions, compiler assumptions, or build recipes must update `BUILD.md` and recorded provenance.
- Use `uv` for the development environment and run the relevant tests before committing changes.
- User-facing documentation should describe implemented behavior. Keep unfinished methodological work in this file rather than in researcher-facing documentation.
- For scientific and technical prose, follow [mwyau/write-like-a-scientist](https://github.com/mwyau/write-like-a-scientist), especially `skills/write-like-a-scientist/SKILL.md`. Apply its research-software profile and atmospheric-science domain guidance when relevant.

## Remaining methodological work

- **Numerical validation:** generate a common real band-limited spectral field for each `lmax`, convert it to each backend's coefficient layout, normalization, and Condon–Shortley phase convention, compare synthesis against a common reference field, and measure synthesis–analysis round-trip error. Record relative L2 error, maximum absolute error, and a bounded relative-error metric with precision-appropriate tolerances.
- **CPU and thread controls:** record process affinity, physical core count, simultaneous multithreading status, effective backend thread count when queryable, CPU instruction-set features, and relevant power-management state. Distinguish physical-core scaling from simultaneous-multithreading scaling.
- **Native build metadata:** record compiler executable and version, compiler/linker flags, selected CPU microarchitecture or instruction set, SHTns FFTW version/configuration, and linked FFT, BLAS, and LAPACK libraries where applicable.
- **Timing robustness:** repeat complete benchmark cells or matrices in independent processes and randomize or interleave backend order to reduce thermal, frequency, and run-order bias.
- **Setup and memory:** measure constructor/grid-configuration time, transform-plan or Legendre-function precomputation time, persistent reusable state, and peak resident memory separately from steady-state transform latency.
- **Derived quantities:** after numerical validation is available, add transforms per second, grid points per second, speedup relative to a named reference implementation, parallel speedup/efficiency, and numerical error versus transform latency. Derived comparisons must identify backend, grid, precision, build mode, and thread count.
- **Vector transforms:** add vector and spin-1 benchmarks only after scalar normalization, sign, phase, and accuracy conventions are verified; include explicit convention tests before cross-library performance comparisons.
- **SHTOOLS regular-grid coverage:** add SHTOOLS to the endpoint-including regular-grid comparison only when an independent native Clenshaw–Curtis transform is available. Do not count a pyshtools path that delegates to DUCC as a separate implementation.
